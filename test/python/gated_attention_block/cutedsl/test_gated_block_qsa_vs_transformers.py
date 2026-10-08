# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPTIONAL cross-check of the re-derived QSA oracle against the HF transformers implementation of the layer
(``transformers.models.qwen4_exp``: ``Qwen4ExpTextAttention`` with its ``Qwen4ExpTextQSAIndexer``), on a 2-layer-sized
random-weight config.  Skipped, module-wide, whenever ``transformers`` or the ``qwen4_exp`` model is unavailable -- the
re-derived oracle (``gated_block_qsa_reference.py``) stays the acceptance gate; this is insurance against a
re-derivation slip, not a dependency.

Status: written against the HF source as it stood on 2026-10-07 (``modeling_qwen4_exp.py``: the indexer at 665-771,
the attention at 811-893; ``configuration_qwen4_exp.py``: the QSA validator at 187-229).  No environment of this
project carries ``transformers >= 5.16`` with the model yet, so this module has only been seen to SKIP; the first run
on an environment that has the package records the measured agreement in its log.

Mapping of the oracle's weights onto the HF module (fp32 end to end, so the comparison is fp32 accumulation order):
``q_proj.weight`` = the per-head interleave ``[q_h | gate_h]`` of the oracle's Q and GATE bands (``modeling:859-862``:
``view(..., -1, 2 * head_dim).chunk(2)``); ``k_proj`` / ``v_proj`` = the K / V bands; ``o_proj`` = ``w_o``; every
RMSNorm weight = the oracle's DIRECT multiplier minus one (HF applies ``1 + weight``, ``modeling:160-163``);
``indexer.index_qk_proj.weight`` = the INDEX band rows (q heads first, the raw-key head last, ``modeling:677-681``).
The RoPE tables are the oracle's own (HF's attention takes ``position_embeddings`` as given and rotates
``cos.shape[-1]`` dims).
"""

import dataclasses
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

transformers = pytest.importorskip("transformers", reason="the optional QSA cross-check needs transformers (>= 5.16 with the qwen4_exp model)")
try:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
except ImportError as exc:  # pragma: no cover - depends on the installed transformers
    pytest.skip(f"transformers {transformers.__version__} has no qwen4_exp model: {exc}", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import (  # noqa: E402
    RefQsaGeometry,
    RefQsaSpec,
    gated_attention_block_qsa_reference,
    make_qsa_inputs,
    qsa_indexer_reference,
    qsa_visible_mask,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# A shrunk Flash-Next: 4 Q heads over 2 KV heads, head dim 64 with 16 rotating dims, a 4 x 32 indexer under a budget of
# 8 tokens (top_k = 2 blocks of 4) so that S = 48 has rows past the identity bound (11 visible tokens).
SPEC = RefQsaSpec(block_size=4, top_k=2, index_band=True, index_heads=4, index_kv_heads=1, index_head_dim=32)
GEOM = RefQsaGeometry(d_model=128, h_q=4, h_kv=2, d_head=64, rope_dim=16, rope_base=1e6, qsa=SPEC)


def _hf_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=GEOM.d_model,
        num_hidden_layers=1,
        num_attention_heads=GEOM.h_q,
        num_key_value_heads=GEOM.h_kv,
        head_dim=GEOM.d_head,
        rms_norm_eps=GEOM.qk_norm_eps,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": GEOM.rope_base, "partial_rotary_factor": GEOM.rope_dim / GEOM.d_head},
        layer_types=["indexed_attention"],
        indexer_n_heads=SPEC.index_heads,
        indexer_kv_heads=SPEC.index_kv_heads,
        indexer_head_dim=SPEC.index_head_dim,
        indexer_budget=SPEC.token_budget,
        indexer_compress_ratio=SPEC.block_size,
    )


def _load_hf_attention(inp: dict, config) -> Qwen4ExpTextAttention:
    attn = Qwen4ExpTextAttention(config, layer_idx=0).to(device=DEV, dtype=torch.float32)
    g = GEOM
    d, hq = g.d_head, g.h_q
    o_q, o_g, o_k, o_v = g.offsets
    w = inp["w_qkvg"].float()
    wq, wg, wk, wv = w[o_q:o_g], w[o_g:o_k], w[o_k:o_v], w[o_v : g.n_qkvg_dense]
    w_i = w[g.n_qkvg_dense :]
    per_head = torch.cat([torch.cat([wq[h * d : (h + 1) * d], wg[h * d : (h + 1) * d]], dim=0) for h in range(hq)], dim=0)
    with torch.no_grad():
        attn.q_proj.weight.copy_(per_head)
        attn.k_proj.weight.copy_(wk)
        attn.v_proj.weight.copy_(wv)
        attn.o_proj.weight.copy_(inp["w_o"].float())
        attn.q_norm.weight.copy_(inp["w_q_norm"].float() - 1.0)
        attn.k_norm.weight.copy_(inp["w_k_norm"].float() - 1.0)
        attn.indexer.index_qk_proj.weight.copy_(w_i)
        attn.indexer.q_layernorm.weight.copy_(inp["w_iq_norm"].float() - 1.0)
        attn.indexer.k_layernorm.weight.copy_(inp["w_ik_norm"].float() - 1.0)
    return attn.eval()


def _additive_causal_mask(b: int, s: int, dtype=torch.float32) -> torch.Tensor:
    allowed = torch.tril(torch.ones(s, s, dtype=torch.bool, device=DEV))
    return (
        torch.where(allowed, torch.zeros((), dtype=dtype, device=DEV), torch.full((), torch.finfo(dtype).min, dtype=dtype, device=DEV))[None, None]
        .expand(b, 1, s, s)
        .contiguous()
    )


@pytest.mark.parametrize("S", [12, 48])
def test_oracle_matches_transformers_qwen4_exp(S):
    torch.manual_seed(0)
    inp = make_qsa_inputs(GEOM, 2, S, device=DEV, dtype=torch.float32, seed=100 + S, index_source="indexer")
    config = _hf_config()
    config._attn_implementation = "eager"
    attn = _load_hf_attention(inp, config)
    mask = _additive_causal_mask(2, S)
    with torch.no_grad():
        hf_out, _ = attn(inp["h"].float(), position_embeddings=(inp["cos"].float(), inp["sin"].float()), attention_mask=mask, past_key_values=None)
        hf_selected = attn.indexer(inp["h"].float(), (inp["cos"].float(), inp["sin"].float()), mask, None)  # additive: 0 = selected

    ref = gated_attention_block_qsa_reference(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        GEOM,
        block_ids=inp["block_ids"],
        block_lens=inp["block_lens"],
    )
    assert torch.isfinite(hf_out).all() and torch.isfinite(ref.out).all()

    # 1. the SELECTION: HF's selected-token mask ANDed with causal == the oracle's visible set, except on rows whose
    #    512th / 513th (here 2nd / 3rd) block scores tie within fp32 noise (top-k order is unspecified there).
    scores = inp["index_scores"]  # [B, S, n_blocks] fp32, -inf past the limit
    for b in range(2):
        vis = qsa_visible_mask(inp["block_ids"][b], inp["block_lens"][b], torch.arange(S, device=DEV), S, S, SPEC.block_size, top_k=SPEC.top_k)
        hf_vis = (hf_selected[b, 0] == 0) & (mask[b, 0] == 0)
        srt = torch.sort(scores[b], dim=-1, descending=True).values
        tied = (
            (srt.shape[-1] > SPEC.top_k) & ((srt[:, SPEC.top_k - 1] - srt[:, SPEC.top_k]).abs() < 1e-5)
            if srt.shape[-1] > SPEC.top_k
            else torch.zeros(S, dtype=torch.bool, device=DEV)
        )
        rows = ~tied
        assert torch.equal(vis[rows], hf_vis[rows]), f"batch {b}: the visible sets differ on an untied row"

    # 2. the OUTPUT: fp32 end to end on both sides -> accumulation order only.
    diff = (hf_out - ref.out.float()).abs().max().item()
    scale = ref.out.float().abs().max().item()
    cos = torch.nn.functional.cosine_similarity(hf_out.flatten(), ref.out.float().flatten(), dim=0).item()
    print(f"S={S}: max |HF - oracle| = {diff:.3e} (scale {scale:.3e}), cos = {cos:.8f}")
    torch.testing.assert_close(ref.out.float(), hf_out, rtol=1e-4, atol=1e-4 * max(scale, 1e-3))


def test_indexer_reference_matches_transformers_selection_scores():
    """The indexer's SCORES, not just the sets: HF exposes only the mask, so the scores are compared through a hook on
    ``torch.relu`` inside the indexer -- the same ``relu(q . kbar).sum(heads) / sqrt(d)`` arithmetic."""
    S = 24
    inp = make_qsa_inputs(GEOM, 1, S, device=DEV, dtype=torch.float32, seed=7, index_source="indexer")
    config = _hf_config()
    config._attn_implementation = "eager"
    attn = _load_hf_attention(inp, config)
    mask = _additive_causal_mask(1, S)
    seen = []
    orig_relu = torch.relu

    def spy(x):
        seen.append(x.detach().clone())
        return orig_relu(x)

    torch.relu = spy
    try:
        with torch.no_grad():
            attn.indexer(inp["h"].float(), (inp["cos"].float(), inp["sin"].float()), mask, None)
    finally:
        torch.relu = orig_relu
    # HF scores one query at a time (modeling:744-746): the spy saw one [n_blocks, heads] tensor per query with >= 1 block
    scores_ref = inp["index_scores"][0]  # [S, n_blocks_max]
    rows = [t for t in range(S) if (t + 1) // SPEC.block_size > 0]
    assert len(seen) == len(rows)
    for t, raw in zip(rows, seen):
        hf_scores = torch.relu(raw).sum(dim=-1) / (SPEC.index_head_dim**0.5)
        n = (t + 1) // SPEC.block_size
        torch.testing.assert_close(scores_ref[t, :n], hf_scores, rtol=1e-4, atol=1e-5)
