# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block END TO END under ``QsaSpec`` -- stage (4) is the index-list sparse core -- against the QSA oracle.

Every Rubin cell (``requires_rubin``) declares a block with ``GatedAttentionBlockGeometry(qsa=QsaSpec(...))``, checks, compiles
and runs it TWICE on sentinel-filled outputs, and compares ``out``, the gated O (the workspace slot stage (5) wrote in place)
and the LSE against ``gated_attention_block_qsa_reference`` run on the SAME inputs and the SAME block list: ``isfinite`` and
"no sentinel survived" BEFORE any cosine, ``cos >= 0.999`` on ``out`` (the end-to-end suite's bar), ``atol 2e-2`` on the gated
O and the LSE (the SDPA stage test's bar), a dead row (``seq_lens[b] == 0``, or an empty list with no open tail) EXACTLY
``out = 0`` / ``O = 0`` / ``LSE = -inf``, and the two launches bitwise.  Tolerances are the dense suites', none added.

The matrix: S in {512, 2051 (the last identity length), 2052 (the first length with a dropped block), 4096, 32768} x B in
{1, 2 (a different list per entry; ``seq_lens`` ``[S, S - 37]``; the dead entry ``[S, 0]``)} x geometries {24/2 d_model 2560
(TP 1), 12/1 (TP 2), 6/1 (TP 4: the LDG norm-kernel geometry), 32/2 d_model 4096, 8/2 d_model 512 for speed} x list sources
{full (the identity below the bound), the indexer reference, a random subset in random order} x ``fuse_norm_rope`` x
``return_lse`` x dtype {bf16, f16} x ``top_k`` {512, 256, 4} x the K / V layout {the slab stride (``inplace_qkv``), compact}
x the indexer band; the degenerate rows (a ``-1`` INSIDE the count is a VALUE -- that block absent; entries BEYOND the count
are never read -- bitwise invariant to garbage there and to the ``[B, S, top_k]`` spelling; ``block_lens`` out of range is
clamped on device; an empty list attends to its tail only; one block total at S = 4 / 5 / 7 with ``B x H_kv > 1``); the
ADVERSARIAL S = 2052 cell (a dominant key planted through ``h`` inside the block the 512-list omits for row 2051 -- within
budget vs the oracle ON the list, outside it vs the oracle on the FULL set by the plant's margin; both magnitudes reported);
the write-through twin (a ``QsaSpec`` block with ``paged_kv_page_size`` is bitwise the one without); the workspace of a
``QsaSpec`` block equal to the dense block's.  ``fuse_gate`` stays a typed decline here until the sparse epilogue-gate arm is
bound through the block (its cells invert then).

The IN-BLOCK INDEXER (``QsaSpec(index_source="indexer", index_band=True)``): the block derives the selection itself -- the
band's queries normed + rotated, the raw key compressed per block at the block start, the scorer's top-k -- and the cells run
the TWO-STAGE oracle: (1) the selection as a SET, (a) against the top-k of the scores recomputed in fp64 from the kernel's OWN
bf16 queries and compressed keys up to ties (the scorer's fp32 accumulate), (b) against the fp32-operand oracle under the
MARGIN RULE -- a block in exactly one of the two sets is accepted iff its oracle score is within ``tol`` of the row's k-th
oracle score, ``tol`` = the MEASURED perturbation of the scores by the bf16 rounding of the operands (never a tuned number)
-- and (c) the same against the HF-rounding oracle (the pooled key cast to bf16 before its norm); the kernel's queries and
compressed keys within one bf16 ulp of the once-rounded fp32 chain; the count rule on every row; (2) attention EXACT given
the kernel's selection (the block oracle on the kernel's list, the standard assertions).  Plus the identity below the bound
(no scorer launch, bitwise the caller-list block on the full list), the step-0 reuse pin (``block_ids_out`` / ``block_lens_out``
fed to a caller-list block reproduce the output bitwise; ``index_k_compressed`` rows written and the rest untouched), and the
write-through composition.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block import GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, index_k_raw_view  # noqa: E402
from cudnn.gated_attention_block.api import _cols, _SparseSdpa, _view  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import (  # noqa: E402
    RefQsaGeometry,
    RefQsaSpec,
    block_ids_contract_violations,
    full_block_ids,
    gated_attention_block_qsa_reference,
    make_qsa_inputs,
    qsa_indexer_reference,
    random_block_ids,
)
from gated_block_reference import RefGeometry, apply_partial_rope, gated_attention_block_reference, make_inputs  # noqa: E402

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

COS_OUT = 0.999  # the end-to-end suite's bar on `out` (test_block_end_to_end.py)
ATOL = 2e-2  # the SDPA stage test's bar on O / LSE (test_sdpa_stage_sm107.py) -- never widened here
BS = 4

# The Qwen3.8-Flash-Next QSA layer at TP 1 / TP 2 / TP 4, the 397B sibling, and a shrunk d256 geometry for speed.
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=256, rope_dim=64)
GEOM_TP2 = dict(d_model=2560, h_q=12, h_kv=1, d_head=256, rope_dim=64)
GEOM_TP4 = dict(d_model=2560, h_q=6, h_kv=1, d_head=256, rope_dim=64)  # the LDG norm-kernel geometry (no TMA tile fits 6 rows)
GEOM_397B = dict(d_model=4096, h_q=32, h_kv=2, d_head=256, rope_dim=64)  # the GQA group AT the record's cap (16)
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=256, rope_dim=64)

BF16, F16 = torch.bfloat16, torch.float16


def _sentinel(dtype):
    return 1.5e30 if dtype == torch.bfloat16 else 6.0e4  # f16 cannot hold 1.5e30; no block output reaches 6e4


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at each |x| (8 bits of mantissa), floored at the smallest normal's spacing."""
    e = torch.floor(torch.log2(x.float().abs().clamp_min(2.0**-126)))
    return torch.pow(2.0, e - 7)


@dataclass
class _Cell:
    """What one block run leaves behind: the per-launch (out, lse, gated O) triples, the oracle, the block, the inputs."""

    runs: list
    ref: object
    blk: object
    inp: dict
    ws: torch.Tensor
    geom: object
    ids: torch.Tensor
    lens: Optional[torch.Tensor]
    mags: dict = field(default_factory=dict)
    ix: Optional[dict] = None  # the in-block indexer's record: kernel_ids / kernel_lens, the qi / kbar slots, launches, the caller buffers


def _run(
    geom_kw,
    batch,
    seq_len,
    *,
    dtype=BF16,
    top_k=512,
    index_band=False,
    index_source="full",
    seq_lens=None,
    block_lens=True,
    return_lse=True,
    fuse_norm_rope=False,
    inplace_qkv=None,
    seed=0,
    launches=2,
    lists=None,
    inp=None,
    ref=None,
    paged_kv_page_size=0,
    pools=None,
    indexer=False,
    indexer_outputs=False,
) -> _Cell:
    """Declare, check, compile and run a QsaSpec block ``launches`` times on sentinel-filled outputs; the oracle on the same
    inputs and list.  ``lists`` overrides the generated ``(block_ids, block_lens)``; ``inp`` / ``ref`` reuse another cell's.
    ``indexer=True`` declares ``index_source="indexer"`` (the band implied): the block derives the selection and the oracle
    runs on the KERNEL's list (``ix`` records it); ``indexer_outputs`` hands the three optional caller buffers in."""
    band = index_band or indexer
    qsa = QsaSpec(top_k=top_k, index_band=band, index_source="indexer" if indexer else "caller")
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(top_k=top_k, index_band=band))
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=qsa)
    if inp is None:
        inp = make_qsa_inputs(ref_geom, batch, seq_len, dtype=dtype, seed=seed, index_source="indexer" if indexer else index_source, seq_lens=seq_lens)
    ids, lens = (inp["block_ids"], inp["block_lens"]) if lists is None else lists
    lens_arg = lens if block_lens else None
    if ref is None and not indexer:
        ref = gated_attention_block_qsa_reference(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            ref_geom,
            block_ids=ids,
            block_lens=lens_arg,
            seq_lens=seq_lens,
        )
    t = batch * seq_len
    out = torch.empty(batch, seq_len, geom.d_model, device="cuda", dtype=dtype)
    blk = GatedAttentionBlockFwd(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        out,
        geom,
        return_lse=return_lse,
        seq_lens_present=seq_lens is not None,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        paged_kv_page_size=paged_kv_page_size,
    )
    assert isinstance(blk._sdpa, _SparseSdpa)
    assert blk.check_support()
    blk.compile()
    ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device="cuda")
    lse = torch.empty(batch, geom.h_q, seq_len, device="cuda", dtype=torch.float32) if return_lse else None
    sent = _sentinel(dtype)
    lay = blk._layout()
    ix_kw, ix = {}, None
    if indexer:
        ix_kw = dict(w_iq_norm=inp["w_iq_norm"], w_ik_norm=inp["w_ik_norm"])
        if indexer_outputs:
            ix = dict(
                ids_out=torch.full((t, top_k), -7, dtype=torch.int32, device="cuda"),
                lens_out=torch.full((t,), -7, dtype=torch.int32, device="cuda"),
                kbar_out=torch.full((batch, seq_len // BS + 3, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device="cuda"),
            )
            ix_kw.update(block_ids_out=ix["ids_out"], block_lens_out=ix["lens_out"], index_k_compressed=ix["kbar_out"])
    runs = []
    for _ in range(launches):
        out.fill_(sent)
        if lse is not None:
            lse.fill_(sent)
        blk.execute(
            inp["h"],
            inp["w_qkvg"],
            inp["w_q_norm"],
            inp["w_k_norm"],
            inp["cos"],
            inp["sin"],
            inp["w_o"],
            out,
            ws,
            seq_lens=seq_lens,
            lse=lse,
            **({} if indexer else dict(block_ids=ids, block_lens=lens_arg)),
            **(pools or {}),
            **ix_kw,
        )
        torch.cuda.synchronize()
        o_gated = _view(ws, lay.o, (t, geom.h_q, geom.d_head), dtype).view(batch, seq_len, geom.h_q, geom.d_head).clone()
        runs.append((out.clone(), None if lse is None else lse.clone(), o_gated))
    if indexer:
        # The kernel's OWN list (the caller's buffer, the workspace slot, or the plan-time identity below the bound) and the
        # operands it scored: the oracle runs on THAT list (stage 2 of the two-stage oracle), never on a list of its own.
        st = blk._indexer
        v = st.views(ws, lay)
        k_ids = (ix["ids_out"] if ix is not None else (v["ids"] if v is not None else st._const["identity_ids"])).view(batch, seq_len, top_k).clone()
        k_lens = (ix["lens_out"] if ix is not None else st._const["counts"]).view(batch, seq_len).clone()
        ix = dict(
            ix or {},
            launches=st.launches,
            selects=st.selects,
            kernel_ids=k_ids,
            kernel_lens=k_lens,
            qi=None if v is None else v["qi"].clone(),
            kbar=None if v is None else v["kbar"].clone(),
        )
        ids, lens_arg = k_ids, k_lens
        if ref is None:
            ref = gated_attention_block_qsa_reference(
                inp["h"],
                inp["w_qkvg"],
                inp["w_q_norm"],
                inp["w_k_norm"],
                inp["cos"],
                inp["sin"],
                inp["w_o"],
                ref_geom,
                block_ids=k_ids,
                block_lens=k_lens,
                seq_lens=seq_lens,
            )
    return _Cell(runs=runs, ref=ref, blk=blk, inp=inp, ws=ws, geom=geom, ids=ids, lens=lens_arg, ix=ix)


def _check(c: _Cell, label: str) -> dict:
    """The standard assertions of every cell -- finite, no sentinel, dead rows exact, within the budget, launches bitwise --
    and the measured magnitudes, printed."""
    out, lse, o = c.runs[0]
    ref = c.ref
    sent = _sentinel(out.dtype)
    assert torch.isfinite(out.float()).all(), f"{label}: non-finite out"
    assert not (out.float() == sent).any(), f"{label}: a sentinel survived in out"
    assert torch.isfinite(o.float()).all(), f"{label}: non-finite gated O (a row the SDPA never wrote keeps the workspace fill)"
    for out2, lse2, o2 in c.runs[1:]:
        assert torch.equal(_bits(out), _bits(out2)) and torch.equal(_bits(o), _bits(o2)), f"{label}: two launches are not bitwise on out / O"
        if lse is not None:
            assert torch.equal(lse, lse2), f"{label}: two launches are not bitwise on LSE"
    dead = ref.n_visible == 0  # [B, S]
    cos = _cos(out, ref.out)
    d_out = float((out.float() - ref.out.float()).abs().max())
    d_o = float((o.float() - ref.o_gated.float()).abs().max())
    d_lse = 0.0
    if lse is not None:
        assert not (lse == sent).any(), f"{label}: a sentinel survived in LSE"
        dead_bhs = dead[:, None, :].expand_as(lse)
        if dead.any():
            assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{label}: a dead row's LSE must be -inf exactly"
        live = ~dead_bhs
        d_lse = float((lse[live] - ref.lse[live]).abs().max()) if live.any() else 0.0
        assert d_lse <= ATOL, f"{label}: LSE off the oracle by {d_lse} (budget {ATOL})"
    if dead.any():
        assert (o[dead] == 0).all() and (out[dead] == 0).all(), f"{label}: a dead row's O and out must be exactly 0"
    assert cos >= COS_OUT, f"{label}: cos(out) {cos} < {COS_OUT}"
    assert d_o <= ATOL, f"{label}: gated O off the oracle by {d_o} (budget {ATOL})"
    c.mags = dict(cos=cos, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()))
    print(f"\n{label}: cos(out) {cos:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}")
    return c.mags


# ============================================================================ the matrix
# (id, geometry, B, S, dtype, list source, top_k, seq_lens, block_lens given, return_lse, fuse_norm_rope, inplace_qkv, index_band)
_MATRIX = [
    # the two acceptance cells: the block end to end at the Flash-Next geometry, S = 512, the full list, bf16 AND f16
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, True, False, None, False, id="fn24-2-s512-bf16-full-lse"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, F16, "full", 512, None, True, True, False, None, False, id="fn24-2-s512-f16-full-lse"),
    # the dead entry
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "full", 512, (512, 0), True, True, False, None, False, id="fn24-2-s512-b2-dead-entry-bf16"),
    # S: the last identity length, top_k 256 past its bound, the indexer-reference lists, the long-S cell
    pytest.param(GEOM_FLASH_NEXT, 1, 2051, BF16, "full", 512, None, True, True, False, None, False, id="fn24-2-s2051-bf16-full-identity"),
    pytest.param(GEOM_FLASH_NEXT, 1, 2051, BF16, "synthetic", 256, None, True, True, False, None, False, id="fn24-2-s2051-bf16-synthetic-top256"),
    pytest.param(GEOM_FLASH_NEXT, 1, 4096, BF16, "indexer", 512, None, True, True, False, None, False, id="fn24-2-s4096-bf16-indexer"),
    pytest.param(GEOM_FLASH_NEXT, 1, 32768, BF16, "synthetic", 512, None, True, True, False, None, False, id="fn24-2-s32768-bf16-synthetic"),
    # B = 2: a different list per entry; a padded entry; block_lens absent (the kernel's derived count)
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "synthetic", 512, (512, 475), True, True, False, None, False, id="fn24-2-s512-b2-synthetic-lens-475"),
    pytest.param(GEOM_FLASH_NEXT, 2, 512, BF16, "synthetic", 512, None, False, True, False, None, False, id="fn24-2-s512-b2-synthetic-no-block-lens"),
    # the pipeline knobs: fused norm + RoPE (no band), no LSE, the compact K / V layout, the indexer band
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "synthetic", 512, None, True, True, True, None, False, id="fn24-2-s512-fuse-norm-rope"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, False, False, None, False, id="fn24-2-s512-no-lse"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "full", 512, None, True, True, False, False, False, id="fn24-2-s512-compact-kv"),
    pytest.param(GEOM_FLASH_NEXT, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, True, id="fn24-2-s512-index-band"),
    # the TP geometries (12/1, 6/1), the 397B sibling (GQA 16 = the record's cap), the shrunk geometry
    pytest.param(GEOM_TP2, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="tp2-12-1-s512-synthetic"),
    pytest.param(GEOM_TP2, 1, 512, F16, "full", 512, None, True, True, False, None, False, id="tp2-12-1-s512-f16-full"),
    pytest.param(GEOM_TP4, 2, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="tp4-6-1-s512-b2-synthetic-ldg-norm"),
    pytest.param(GEOM_397B, 1, 512, BF16, "synthetic", 512, None, True, True, False, None, False, id="g397b-32-2-s512-synthetic-gqa16"),
    pytest.param(GEOM_SMALL, 2, 512, BF16, "synthetic", 512, None, True, True, False, False, False, id="small8-2-s512-b2-synthetic-compact-kv"),
    pytest.param(GEOM_SMALL, 1, 300, BF16, "synthetic", 4, None, True, True, False, None, False, id="small8-2-s300-top4-synthetic"),
    pytest.param(GEOM_SMALL, 2, 300, F16, "full", 512, (300, 0), True, True, False, None, False, id="small8-2-s300-b2-f16-dead-entry"),
    pytest.param(GEOM_SMALL, 1, 2052, BF16, "synthetic", 512, None, True, True, False, None, False, id="small8-2-s2052-synthetic-dropped-block"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band", _MATRIX)
def test_qsa_block_matches_the_oracle(
    request, geom_kw, batch, seq_len, dtype, source, top_k, seq_lens, block_lens, return_lse, fuse_norm_rope, inplace_qkv, index_band
):
    """One cell of the matrix: the block vs the oracle on the same list, every standard assertion, the magnitudes printed."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    c = _run(
        geom_kw,
        batch,
        seq_len,
        dtype=dtype,
        top_k=top_k,
        index_band=index_band,
        index_source=source,
        seq_lens=lens_t,
        block_lens=block_lens,
        return_lse=return_lse,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        seed=seq_len + batch,
    )
    # the list is contract-clean (the test-side detector; the library never reads a list on the host)
    pos = torch.arange(seq_len, device="cuda").expand(batch, seq_len).reshape(-1)
    kv = torch.full((batch * seq_len,), seq_len, dtype=torch.long, device="cuda") if lens_t is None else lens_t.long().repeat_interleave(seq_len)
    assert (
        block_ids_contract_violations(
            c.ids.reshape(batch * seq_len, -1), pos, kv, block_size=BS, block_lens=c.lens.reshape(-1) if c.lens is not None else None, top_k=top_k
        )
        == []
    )
    m = _check(c, request.node.callspec.id)
    if seq_lens is not None and 0 in seq_lens:
        b = list(seq_lens).index(0)
        out, lse, o = c.runs[0]
        assert (out[b] == 0).all() and (o[b] == 0).all() and torch.isinf(lse[b]).all() and (lse[b] < 0).all(), "the dead entry must be exactly 0 / -inf"
        assert m["dead"] >= seq_len
    if source == "full" and seq_len <= QsaSpec(top_k=top_k).identity_bound and seq_lens is None:
        # the identity: with the full list the sparse block is the DENSE function within the budget (not bitwise: another summation order)
        dense_ref = gated_attention_block_reference(
            **{k: c.inp[k] for k in ("h", "w_qkvg", "w_q_norm", "w_k_norm", "cos", "sin", "w_o")}, geom=RefGeometry(**geom_kw)
        )
        cos_d = _cos(c.runs[0][0], dense_ref.out)
        d_lse = float((c.runs[0][1] - dense_ref.lse).abs().max()) if return_lse else 0.0
        print(f"identity vs the dense oracle: cos(out) {cos_d:.6f} max|d LSE| {d_lse:.2e}")
        assert cos_d >= COS_OUT and d_lse <= ATOL


# ============================================================================ the degenerate rows (the shrunk geometry)
@requires_rubin
def test_a_minus_one_inside_the_count_is_a_masked_block():
    """DESIGN 4.6: a ``-1`` at an index BELOW the count contributes no key -- a VALUE the oracle reproduces with that block
    absent (``qsa_visible_mask``); the distance from the full-list function is reported, never asserted."""
    geom_kw, B, S = GEOM_SMALL, 1, 600
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids, lens = full_block_ids(pos, 512, BS)
    holed = ids.clone()
    rows = torch.arange(S, device="cuda") >= 100
    cols = torch.arange(512, device="cuda") % 3 == 0
    hole = rows[None, :, None] & cols[None, None, :] & (holed >= 0)
    holed[hole] = -1
    assert int(hole.sum()) > 0 and block_ids_contract_violations(holed[0], pos[0], torch.full((S,), S, device="cuda"), block_size=BS, block_lens=lens[0]) != []
    c = _run(geom_kw, B, S, seed=11, lists=(holed.contiguous(), lens.contiguous()))
    m = _check(c, "small8-2 S=600 -1 inside the count (every 3rd entry of rows >= 100)")
    full = _run(geom_kw, B, S, seed=11, inp=c.inp, lists=(ids.contiguous(), lens.contiguous()))
    _check(full, "small8-2 S=600 the full list")
    teeth = float((c.runs[0][0][0, 100:].float() - full.runs[0][0][0, 100:].float()).abs().max())
    print(f"holed vs full list on rows >= 100: max|d out| {teeth:.4f} (the masked blocks move the function; reported)")
    assert m["d_o"] <= ATOL


@requires_rubin
def test_entries_beyond_the_count_are_never_read_and_both_list_spellings_agree():
    """DESIGN 4.6: entries at an index at or beyond the count are never read -- garbage there (0, 7, 1e6, -5) leaves out / O /
    LSE BITWISE the canonical run, with ``block_lens`` given or absent (the derived default count = the full list's), and the
    ``[B, S, top_k]`` spelling is bitwise the flat ``[T, top_k]`` one (an exact view, no copy)."""
    geom_kw, B, S = GEOM_SMALL, 2, 300
    pos = torch.arange(S, device="cuda").expand(B, S)
    gen = torch.Generator(device="cuda").manual_seed(5)
    ids, lens = random_block_ids(pos, 512, BS, generator=gen)
    canon = _run(geom_kw, B, S, seed=13, lists=(ids.contiguous(), lens.contiguous()))
    _check(canon, "small8-2 B=2 S=300 shuffled canonical")
    garbage = ids.clone()
    beyond = torch.arange(512, device="cuda")[None, None, :] >= lens.long()[:, :, None]
    fill = torch.tensor([0, 7, 1_000_000, -5], dtype=torch.int32, device="cuda")[torch.arange(512, device="cuda") % 4].expand(B, S, 512)
    garbage[beyond] = fill[beyond]
    assert block_ids_contract_violations(garbage.reshape(B * S, 512), pos.reshape(-1), torch.full((B * S,), S, device="cuda"), block_size=BS) != []
    o_c, l_c, g_c = canon.runs[0]
    for label, lists_, bl in (
        ("garbage beyond the count, block_lens given", (garbage.reshape(B * S, 512).contiguous(), lens.reshape(B * S).contiguous()), True),
        ("garbage beyond the count, block_lens absent", (garbage.contiguous(), lens.contiguous()), False),
        ("the canonical list, block_lens absent", (ids.reshape(B * S, 512).contiguous(), None), False),
    ):
        c = _run(geom_kw, B, S, seed=13, inp=canon.inp, ref=canon.ref, lists=lists_, block_lens=bl)
        _check(c, f"small8-2 B=2 S=300 {label}")
        o, l, g = c.runs[0]
        assert torch.equal(_bits(o), _bits(o_c)) and torch.equal(l, l_c) and torch.equal(_bits(g), _bits(g_c)), label


@requires_rubin
def test_block_lens_out_of_range_is_clamped_on_device_and_an_empty_list_attends_to_its_tail():
    """DESIGN 4.6: ``block_lens`` negative (-7 on odd rows) is clamped to 0 -> tail-only attention, DEAD (``out = 0``, ``LSE =
    -inf``) where there is no open tail (``(pos + 1) % 4 == 0``); above the derived count (1e5 on even rows) it is clamped to
    the default; an all-``-1`` list with ``block_lens = 0`` attends to its tail only on every row (dead where no tail)."""
    geom_kw, B, S = GEOM_SMALL, 1, 400
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids, lens = full_block_ids(pos, 512, BS)
    bad = torch.where(
        torch.arange(S, device="cuda") % 2 == 1,
        torch.full((S,), -7, dtype=torch.int32, device="cuda"),
        torch.full((S,), 100_000, dtype=torch.int32, device="cuda"),
    )[None]
    c = _run(geom_kw, B, S, seed=17, lists=(ids.contiguous(), bad.contiguous()))
    m = _check(c, "small8-2 S=400 block_lens -7 (odd rows) / 1e5 (even rows)")
    expect_dead = int(((torch.arange(S) % 2 == 1) & ((torch.arange(S) + 1) % 4 == 0)).sum())
    assert m["dead"] == expect_dead == 100
    empty = torch.full_like(ids, -1)
    zero = torch.zeros_like(lens)
    e = _run(geom_kw, B, S, seed=17, inp=c.inp, lists=(empty.contiguous(), zero.contiguous()))
    me = _check(e, "small8-2 S=400 empty lists (block_lens = 0): the tail only")
    assert me["dead"] == int(((torch.arange(S) + 1) % 4 == 0).sum()) == 100


@requires_rubin
@pytest.mark.parametrize("seq_len", [4, 5, 7])
def test_one_block_total_with_two_sequences(seq_len):
    """DESIGN 4.6: one block total (S = 4) and S = 5 / 7 (a 1- / 3-token tail), ``B x H_kv > 1`` with one block -- the kernel's
    ring at trip count 1 under the whole block (projection, norm + RoPE, the sparse SDPA, the gate, the out projection)."""
    c = _run(GEOM_SMALL, 2, seq_len, seed=seq_len)
    _check(c, f"small8-2 B=2 S={seq_len}")


# ============================================================================ the adversarial cell
def _inverse_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Undo ``apply_partial_rope`` at one position: rotate by the negative angle (the same table with ``-sin``)."""
    return apply_partial_rope(x[None, None, None, :], cos[None, None, :], -sin[None, None, :], rope_dim)[0, 0, 0]


@requires_rubin
def test_adversarial_s2052_planted_key_in_the_omitted_block():
    """DESIGN 4.3 at the block level.  At S = 2052 query 2051 has 513 complete blocks and the 512-wide full list omits block
    512 (tokens 2048..2051).  A dominant key is planted THROUGH ``h``: for one query head of each KV head, ``h[2049]`` /
    ``h[2050]`` is solved (least squares on the KV head's W_k rows) so that its post-norm key, rotated at its position,
    equals the row's post-norm, post-RoPE query -- the score beats every other key by >= 8 nats and the planted key holds
    > 99.9 % of the full-set mass (checked on the oracle's own post-RoPE Q / K).  The block is then within the budget vs the
    oracle ON the list everywhere, and at row 2051 for the planted heads OUTSIDE the budget vs the oracle on the FULL set
    (cos < 0.5 on the head's gated O, the diff >= 10 budgets, the LSE apart by more than the budget); every row below 2051 is
    within the budget vs the full set.  A block that ignored the list could not pass both.  Both magnitudes are reported."""
    geom_kw, B, S, row = GEOM_FLASH_NEXT, 1, 2052, 2051
    geom = GatedAttentionBlockGeometry(**geom_kw, qsa=QsaSpec())
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec())
    G = geom.gqa_ratio
    inp = make_qsa_inputs(ref_geom, B, S, dtype=BF16, seed=7, index_source="full")
    ids, lens = inp["block_ids"], inp["block_lens"]
    assert int(lens[0, row]) == 512 and int(ids[0, row].max()) == 511, "row 2051 lists blocks 0..511 and omits block 512"
    ref0 = gated_attention_block_qsa_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], ref_geom, block_ids=ids, block_lens=lens
    )
    plants = [(0, 2049), (G, 2050)]  # (query head, planted token inside block 512) -- one per KV head
    o_k = geom.qkvg_offsets[2]
    for hq, tok in plants:
        kv = hq // G
        q_rot = ref0.q[0, row, hq].float()  # post-norm, post-RoPE at position 2051 (unit RMS: the norm weight is 1)
        k_pre = _inverse_partial_rope(q_rot, inp["cos"][0, tok].float(), inp["sin"][0, tok].float(), geom.rope_dim)
        w_k = inp["w_qkvg"][o_k + kv * geom.d_head : o_k + (kv + 1) * geom.d_head].float()  # [D, d_model]: k_pre = W_k h
        h_new = torch.linalg.pinv(w_k) @ k_pre  # the least-norm h with W_k h = k_pre (RMSNorm is scale-free: the direction is what matters)
        inp["h"][0, tok] = h_new.to(BF16)
    ref = gated_attention_block_qsa_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], ref_geom, block_ids=ids, block_lens=lens
    )
    # the plant's margin and mass on the full set, from the oracle's own post-RoPE Q / K
    for hq, tok in plants:
        kv = hq // G
        sc = (ref.q[0, row, hq].float() @ ref.k[0, : row + 1, kv].float().t()) * geom.scale
        others = torch.cat([sc[:tok], sc[tok + 1 :]])
        margin = float(sc[tok] - others.max())
        mass = float(1.0 / (1.0 + torch.exp(others - sc[tok]).sum()))
        print(f"\nplant head {hq} token {tok}: score {float(sc[tok]):.2f}, margin {margin:.2f} nats, full-set mass {mass:.5f}")
        assert margin >= 8.0 and mass > 0.999, f"head {hq}: plant margin {margin:.2f} nats, mass {mass:.5f}"
    c = _run(geom_kw, B, S, seed=7, inp=inp, ref=ref, lists=(ids, lens))
    m = _check(c, "fn24-2 S=2052 adversarial: vs the oracle ON the list")
    # the FULL set: every one of the 513 complete blocks (the oracle at width 513 -- RefQsaSpec.top_k carries the width)
    pos = torch.arange(S, device="cuda").expand(B, S)
    ids_full, lens_full = full_block_ids(pos, 513, BS)
    assert int(ids_full[0, row].max()) == 512
    ref_full = gated_attention_block_qsa_reference(
        inp["h"],
        inp["w_qkvg"],
        inp["w_q_norm"],
        inp["w_k_norm"],
        inp["cos"],
        inp["sin"],
        inp["w_o"],
        ref_geom,
        block_ids=ids_full.contiguous(),
        block_lens=lens_full.contiguous(),
        spec=RefQsaSpec(top_k=513),
    )
    out, lse, o = c.runs[0]
    below = slice(0, row)
    cos_below = _cos(out[0, below], ref_full.out[0, below])
    d_o_below = float((o[0, below].float() - ref_full.o_gated[0, below].float()).abs().max())
    d_lse_below = float((lse[0, :, below] - ref_full.lse[0, :, below]).abs().max())
    print(f"rows < 2051 vs the FULL-set oracle: cos(out) {cos_below:.6f} max|d O_gated| {d_o_below:.4e} max|d LSE| {d_lse_below:.2e}")
    assert cos_below >= COS_OUT and d_o_below <= ATOL and d_lse_below <= ATOL
    report = []
    for hq, tok in plants:
        ok, full_h = o[0, row, hq].float(), ref_full.o_gated[0, row, hq].float()
        diff = float((ok - full_h).abs().max())
        cos_h = float(torch.nn.functional.cosine_similarity(ok, full_h, dim=0))
        d_lse = float(abs(lse[0, hq, row] - ref_full.lse[0, hq, row]))
        assert (
            cos_h < 0.5 and diff >= 10 * ATOL and d_lse > ATOL
        ), f"head {hq}: cos {cos_h:.3f} diff {diff:.4f} dLSE {d_lse:.3f} -- the omitted block was attended"
        report.append(f"head {hq}: gated-O diff {diff:.4f} cos {cos_h:.3f}, LSE apart {d_lse:.3f}")
    d_out_row = float((out[0, row].float() - ref_full.out[0, row].float()).abs().max())
    assert d_out_row > ATOL
    print(f"row 2051 vs the FULL-set oracle: {'; '.join(report)}; out row max|d| {d_out_row:.4f} (within the list's budget: {m['d_o']:.4e} / {m['d_lse']:.2e})")


# ============================================================================ serving twin + the workspace pin
def _nan_pool(n_pages, h, ps, d, dtype=BF16):
    return torch.full((n_pages, h, ps, d), float("nan"), device="cuda", dtype=dtype)


@requires_rubin
def test_a_qsa_block_with_write_through_is_bitwise_the_one_without():
    """The write-through acceptance on the SPARSE block (the dense half is test_block_cache_write.py's): a ``QsaSpec`` block
    with the indexer band declared with ``paged_kv_page_size=16`` writes the post-RoPE K / V and the raw indexer key through
    into the pools at ``slot_mapping`` and its ``out`` / LSE are BITWISE the same ``QsaSpec`` block without write-through; the
    pools are bitwise the slab bands attention consumed; padded slots write nothing; ``get_workspace_size`` is unchanged;
    the extra stage sits right before the sparse SDPA."""
    geom_kw, B, S, ps = GEOM_FLASH_NEXT, 1, 512, 16
    plain = _run(geom_kw, B, S, index_band=True, index_source="synthetic", seed=3)
    _check(plain, "fn24-2 band S=512 plain")
    t, geom = B * S, plain.geom
    n_pages = t // ps + 2
    gen = torch.Generator(device="cuda").manual_seed(0)
    slot = torch.randperm(n_pages * ps, generator=gen, device="cuda")[:t].to(torch.int32)
    slot[::13] = -1
    k_cache, v_cache = _nan_pool(n_pages, geom.h_kv, ps, geom.d_head), _nan_pool(n_pages, geom.h_kv, ps, geom.d_head)
    index_cache = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=BF16)
    wt = _run(
        geom_kw,
        B,
        S,
        index_band=True,
        index_source="synthetic",
        seed=3,
        inp=plain.inp,
        ref=plain.ref,
        lists=(plain.ids, plain.lens),
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot, index_k_raw=index_cache),
    )
    _check(wt, "fn24-2 band S=512 write-through")
    assert torch.equal(_bits(wt.runs[0][0]), _bits(plain.runs[0][0])) and torch.equal(
        wt.runs[0][1], plain.runs[0][1]
    ), "write-through changed the sparse block's output"
    assert torch.equal(_bits(wt.runs[0][2]), _bits(plain.runs[0][2]))
    assert wt.blk.get_workspace_size() == plain.blk.get_workspace_size()
    lay = wt.blk._layout()
    proj = _view(wt.ws, lay.proj, (t, geom.n_qkvg), BF16)
    k_rows, v_rows = _cols(proj, geom.qkvg_offsets[2], geom.h_kv, geom.d_head), _cols(proj, geom.qkvg_offsets[3], geom.h_kv, geom.d_head)
    live = slot >= 0
    s_ = slot[live].long()
    assert torch.equal(_bits(k_cache[s_ // ps, :, s_ % ps, :]), _bits(k_rows[live])) and torch.equal(
        _bits(v_cache[s_ // ps, :, s_ % ps, :]), _bits(v_rows[live])
    )
    raw = index_k_raw_view(proj, geom, B, S)
    assert torch.equal(_bits(index_cache[s_ // ps, s_ % ps, :]), _bits(raw[0, live, 0]))
    for pool in (k_cache, v_cache):
        assert (~torch.isnan(pool.float())).all(-1).sum().item() == int(live.sum()) * geom.h_kv
    assert wt.blk._stages.index(wt.blk._cache_write) == wt.blk._stages.index(wt.blk._sdpa) - 1


@requires_rubin
def test_a_qsa_block_workspace_equals_the_dense_blocks():
    """The sparse stage folds 0 bytes of scratch (the kernel has no GMEM workspace), so a ``QsaSpec`` block without the band
    needs exactly the dense block's workspace at the same geometry -- and its stage list has the same length."""
    c = _run(GEOM_SMALL, 2, 256, seed=1)
    _check(c, "small8-2 B=2 S=256 full")
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL)
    dense_inp = make_inputs(RefGeometry(**GEOM_SMALL), batch=2, seq_len=256, dtype=BF16)
    out = torch.empty(2, 256, geom.d_model, device="cuda", dtype=BF16)
    dense = GatedAttentionBlockFwd(
        dense_inp["h"],
        dense_inp["w_qkvg"],
        dense_inp["w_q_norm"],
        dense_inp["w_k_norm"],
        dense_inp["cos"],
        dense_inp["sin"],
        dense_inp["w_o"],
        out,
        geom,
        return_lse=True,
    )
    dense.check_support()
    dense.compile()
    assert c.blk._sdpa.scratch_workspace_bytes() == 0
    assert c.blk.get_workspace_size() == dense.get_workspace_size()
    assert len(c.blk._stages) == len(dense._stages)


# ============================================================================ the in-block indexer: the two-stage oracle
_TIE_TOL = 1e-4  # the scorer's fp32-accumulate budget at the k-th boundary (the standalone selector suite's)


def _set_vs_topk(in_set: torch.Tensor, scores: torch.Tensor, top_k: int, tol: float) -> tuple:
    """Per row: the SET ``in_set`` ``[R, N]`` against the top-k of ``scores`` ``[R, N]`` (fp64; ``-inf`` = not visible), every
    member of the symmetric difference accepted iff its score is within ``tol`` of the row's k-th score (the margin rule).
    Returns ``(flips, rows_with_flips, max_gap)``; asserts the rule."""
    rows, n = scores.shape
    k_eff = min(top_k, n)
    tk = torch.topk(scores, k_eff, dim=-1)
    exp_valid = torch.isfinite(tk.values)
    in_exp = torch.zeros(rows, n + 1, dtype=torch.bool, device=scores.device)
    in_exp.scatter_(1, torch.where(exp_valid, tk.indices, torch.full_like(tk.indices, n)), torch.ones_like(exp_valid))
    in_exp = in_exp[:, :n]
    kth = torch.where(exp_valid, tk.values, torch.full_like(tk.values, float("inf"))).min(dim=-1).values
    differs = in_set ^ in_exp
    flips = int(differs.sum())
    if flips == 0:
        return 0, 0, 0.0
    gap = (scores - kth[:, None]).abs()[differs]
    assert torch.isfinite(gap).all(), f"{int((~torch.isfinite(gap)).sum())} flipped ids lie outside the row's visible range"
    assert bool((gap <= tol).all()), f"{flips} ids differ from the top-k beyond the margin: max gap {float(gap.max()):.3e} > tol {tol:.3e}"
    return flips, int(differs.any(dim=-1).sum()), float(gap.max())


def _membership(ids: torch.Tensor, n: int) -> torch.Tensor:
    """``[R, top_k]`` int32 ids (``-1`` = none) -> ``[R, n]`` bool membership."""
    rows = ids.shape[0]
    m = torch.zeros(rows, n + 1, dtype=torch.bool, device=ids.device)
    m.scatter_(1, torch.where(ids >= 0, ids.long(), torch.full_like(ids, n).long()), torch.ones_like(ids, dtype=torch.bool))
    return m[:, :n]


def _indexer_two_stage(c: _Cell, label: str) -> dict:
    """Stage 1 of the two-stage oracle on an indexer cell (stage 2 -- attention exact given the kernel's list -- is ``_check``):
    the count rule and the ``-1`` prefix form on EVERY row; below the bound the identity; past it (a) the selection as a set
    against the top-k of the fp64 scores of the kernel's OWN bf16 operands up to ties, (b) against the fp32-operand oracle under
    the margin rule with ``tol`` = the MEASURED operand-rounding perturbation (+ the tie budget), (c) the same against the
    HF-rounding oracle, (d) the kernel's queries and compressed keys within one bf16 ulp of the once-rounded fp32 chain; padding
    rows (at or past the KV length) compared on their VISIBLE blocks, the ids they list past the range counted and reported."""
    geom, inp, ix = c.geom, c.inp, c.ix
    q = geom.qsa
    b, s = inp["h"].shape[:2]
    bs, top_k, d_i, h_i = q.block_size, q.top_k, q.index_head_dim, q.index_heads
    dev = inp["h"].device
    seq_lens = inp["seq_lens"]
    lengths = torch.full((b,), s, dtype=torch.long, device=dev) if seq_lens is None else seq_lens.to(dev).long()
    pos = torch.arange(s, device=dev)
    counts = torch.minimum(torch.div(pos + 1, bs, rounding_mode="floor"), torch.tensor(top_k, device=dev)).expand(b, s)
    k_ids, k_lens = ix["kernel_ids"], ix["kernel_lens"]
    valid = k_ids >= 0
    assert torch.equal(k_lens.long(), counts), f"{label}: the counts are not min(top_k, floor((pos + 1) / 4))"
    assert torch.equal(valid, torch.arange(top_k, device=dev)[None, None, :] < counts[..., None]), f"{label}: the valid ids are not a prefix of the count"
    assert int(k_ids.max()) < s // bs, f"{label}: an id past the last complete block"
    m = dict(identity=not ix["selects"], launches=ix["launches"])
    if not ix["selects"]:
        j = torch.arange(top_k, device=dev, dtype=torch.int32).expand(b, s, top_k)
        assert torch.equal(torch.where(valid, k_ids, torch.full_like(k_ids, -1)), torch.where(valid, j, torch.full_like(j, -1))), f"{label}: not the identity"
        assert ix["launches"] == (1 if "kbar_out" in ix else 0), f"{label}: {ix['launches']} indexer launches below the identity bound"
        print(f"\n{label}: the identity list (S <= {q.identity_bound}), {ix['launches']} indexer launch(es)")
        return m
    assert ix["launches"] == 4 + (1 if "kbar_out" in ix else 0), f"{label}: {ix['launches']} launches"
    ref_geom = RefQsaGeometry(**{k: getattr(geom, k) for k in ("d_model", "h_q", "h_kv", "d_head", "rope_dim")}, qsa=RefQsaSpec(top_k=top_k, index_band=True))
    spec = ref_geom.qsa
    oracle = {}
    for hf in (False, True):
        oracle[hf] = qsa_indexer_reference(
            inp["h"], inp["w_i"], inp["w_iq_norm"], inp["w_ik_norm"], None, None, inp["cos"], inp["sin"], spec, ref_geom, seq_lens=seq_lens, hf_rounding=hf
        )
    scores_o, _, lens_o, _, k_c_o = oracle[False]
    n_blocks = k_c_o.shape[1]
    live = pos[None, :] < lengths[:, None]  # [B, S]: the rows below the KV length, where the oracle and the kernel see the same blocks
    assert torch.equal(lens_o[live].long(), counts[live]), f"{label}: the oracle's counts disagree on live rows"
    # (d) the operands: the kernel's qi / kbar vs the once-rounded fp32 chain, one bf16 ulp
    qi_k = ix["qi"].view(b, s, h_i, d_i)
    kbar_k = ix["kbar"]
    x = c.ref.index_q_raw.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + float(q.index_norm_eps)) * inp["w_iq_norm"].float()
    q_o = apply_partial_rope(y, inp["cos"].float(), inp["sin"].float(), geom.rope_dim)
    d_q = (qi_k.float() - q_o).abs() / _bf16_ulp(q_o)
    assert float(d_q.max()) <= 1.0, f"{label}: an indexer query is {float(d_q.max()):.2f} bf16 ulps off the fp32 chain"
    nb_live = torch.div(lengths, bs, rounding_mode="floor")  # complete blocks per sequence
    blk_live = torch.arange(n_blocks, device=dev)[None, :] < nb_live[:, None]  # [B, NB]
    d_k = ((kbar_k.float() - k_c_o.float()).abs() / _bf16_ulp(k_c_o))[blk_live]
    assert float(d_k.max()) <= 1.0, f"{label}: a compressed key is {float(d_k.max()):.2f} bf16 ulps off the fp32 chain"
    m.update(ulp_q=float(d_q.max()), ulp_k=float(d_k.max()))
    # the kernel's own scores (fp64 from its bf16 operands) under the position rule the scorer applied
    s_k = torch.relu(torch.einsum("bshd,bjd->bshj", qi_k.double(), kbar_k.double())).sum(2) / math.sqrt(d_i)
    limit = torch.arange(n_blocks, device=dev)[None, None, :] < torch.div(pos + 1, bs, rounding_mode="floor")[None, :, None]
    s_k = s_k.masked_fill(~limit, float("-inf"))
    in_k = _membership(k_ids.reshape(b * s, top_k), n_blocks).view(b, s, n_blocks)
    # (a) the set == top-k of the kernel-operand scores up to ties (live rows: the sets the kernel and the oracle both define)
    fl_a, rows_a, gap_a = _set_vs_topk(in_k[live], s_k[live], top_k, _TIE_TOL)
    # (b) the margin rule vs the fp32-operand oracle: tol = the measured perturbation of the scores by the operand rounding
    vis_live = torch.isfinite(scores_o) & live[..., None]
    tol_o = float((scores_o.double() - s_k).abs()[vis_live].max()) + _TIE_TOL
    fl_b, rows_b, gap_b = _set_vs_topk(in_k[live], scores_o[live].double(), top_k, tol_o)
    # (c) the HF-rounding arm (the pooled key cast to bf16 before its norm, both norms returning bf16)
    scores_hf = oracle[True][0]
    tol_hf = float((scores_hf.double() - s_k).abs()[vis_live].max()) + _TIE_TOL
    fl_c, rows_c, gap_c = _set_vs_topk(in_k[live], scores_hf[live].double(), top_k, tol_hf)
    # padding rows: the kernel's VISIBLE subset against the oracle's set; the ids it lists past the visible range are masked by the core
    pad = ~live
    extra = 0
    if bool(pad.any()):
        vis_rows = torch.arange(n_blocks, device=dev)[None, None, :] < nb_live[:, None, None]
        extra = int((in_k & ~vis_rows)[pad].sum())
        _set_vs_topk((in_k & vis_rows)[pad], scores_o[pad].double(), top_k, tol_o)
    n_live = int(live.sum())
    m.update(
        rows_live=n_live,
        flips_vs_own=fl_a,
        rows_vs_own=rows_a,
        gap_vs_own=gap_a,
        tol_oracle=tol_o,
        flips_vs_oracle=fl_b,
        rows_vs_oracle=rows_b,
        gap_vs_oracle=gap_b,
        tol_hf=tol_hf,
        flips_vs_hf=fl_c,
        rows_vs_hf=rows_c,
        gap_vs_hf=gap_c,
        padding_rows=int(pad.sum()),
        padding_extra_ids=extra,
        agree_oracle=100.0 * (1.0 - rows_b / max(1, n_live)),
    )
    print(
        f"\n{label}: {n_live} live rows; vs the kernel's own operands {fl_a} flips ({rows_a} rows, gap {gap_a:.2e} <= {_TIE_TOL}); "
        f"vs the fp32-operand oracle {fl_b} flips ({rows_b} rows, {m['agree_oracle']:.3f} % rows exact, gap {gap_b:.3e} <= tol {tol_o:.3e}); "
        f"vs the HF-rounding oracle {fl_c} flips ({rows_c} rows, gap {gap_c:.3e} <= tol {tol_hf:.3e}); qi {m['ulp_q']:.2f} / kbar {m['ulp_k']:.2f} ulp; "
        f"{int(pad.sum())} padding rows, {extra} ids past their range"
    )
    return m


# (id, geometry, B, S, seq_lens, caller outputs)
_INDEXER_MATRIX = [
    pytest.param(GEOM_SMALL, 1, 2560, None, False, id="small8-2-s2560-indexer"),
    pytest.param(GEOM_SMALL, 1, 2560, None, True, id="small8-2-s2560-indexer-outputs"),
    pytest.param(GEOM_SMALL, 2, 2300, (2300, 2100), False, id="small8-2-s2300-b2-indexer-lens-2100"),
    pytest.param(GEOM_SMALL, 1, 2052, None, False, id="small8-2-s2052-indexer-first-dropped-block"),
    pytest.param(GEOM_FLASH_NEXT, 1, 4096, None, True, id="fn24-2-s4096-indexer-outputs"),
    pytest.param(GEOM_SMALL, 2, 2051, None, False, id="small8-2-s2051-b2-indexer-identity"),
    pytest.param(GEOM_SMALL, 1, 300, None, True, id="small8-2-s300-indexer-identity-outputs"),
]


@requires_rubin
@pytest.mark.parametrize("geom_kw, batch, seq_len, seq_lens, outputs", _INDEXER_MATRIX)
def test_indexer_block_matches_the_two_stage_oracle(request, geom_kw, batch, seq_len, seq_lens, outputs):
    """One indexer cell: stage 1 (the selection, ``_indexer_two_stage``) and stage 2 (attention exact given the kernel's list,
    ``_check``); below the identity bound the block is BITWISE the caller-list block on the full list (the same core, the same
    list, the same operands) and launches no scorer."""
    lens_t = None if seq_lens is None else _i32(list(seq_lens))
    c = _run(geom_kw, batch, seq_len, indexer=True, indexer_outputs=outputs, seq_lens=lens_t, seed=seq_len + 7 * batch)
    label = request.node.callspec.id
    m1 = _indexer_two_stage(c, label)
    pos = torch.arange(seq_len, device="cuda").expand(batch, seq_len).reshape(-1)
    kv = torch.full((batch * seq_len,), seq_len, dtype=torch.long, device="cuda") if lens_t is None else lens_t.long().repeat_interleave(seq_len)
    bad = block_ids_contract_violations(c.ids.reshape(batch * seq_len, -1), pos, kv, block_size=BS, block_lens=c.lens.reshape(-1), top_k=c.geom.qsa.top_k)
    if lens_t is None:
        assert bad == [], bad  # the kernel's list is contract-clean on every row; a padded row may list blocks past its range (masked)
    m2 = _check(c, label)
    if m1["identity"]:
        caller = _run(
            geom_kw,
            batch,
            seq_len,
            index_band=True,
            seq_lens=lens_t,
            inp=c.inp,
            ref=c.ref,
            lists=(c.ix["kernel_ids"].contiguous(), c.ix["kernel_lens"].contiguous()),
        )
        assert (
            torch.equal(_bits(c.runs[0][0]), _bits(caller.runs[0][0]))
            and torch.equal(c.runs[0][1], caller.runs[0][1])
            and torch.equal(_bits(c.runs[0][2]), _bits(caller.runs[0][2]))
        ), f"{label}: the identity path is not bitwise the caller-list block"
        assert c.blk.get_workspace_size() == caller.blk.get_workspace_size()
    if outputs:
        ix, nb = c.ix, seq_len // BS
        assert torch.equal(ix["ids_out"].view(batch, seq_len, -1), ix["kernel_ids"]) and torch.equal(ix["lens_out"].view(batch, seq_len), ix["kernel_lens"])
        kb = ix["kbar_out"]
        if ix["selects"]:
            assert torch.equal(_bits(kb[:, :nb]), _bits(ix["kbar"])), f"{label}: the caller's compressed keys differ from the slot's"
        assert torch.isnan(kb[:, nb:].float()).all(), f"{label}: a row past the prompt's complete blocks was written"
        if lens_t is not None:
            for bi, ln in enumerate(seq_lens):
                nb_b = ln // BS
                assert (
                    torch.isnan(kb[bi, nb_b:nb].float()).all() and not torch.isnan(kb[bi, :nb_b].float()).any()
                ), f"{label}: entry {bi}: rows past its {nb_b} blocks"
    print(f"{label}: {m1} | {m2}")


@requires_rubin
def test_indexer_outputs_feed_a_caller_list_block_bitwise():
    """The step-0 reuse pin: the list the indexer block hands back (``block_ids_out`` / ``block_lens_out``) fed to a caller-list
    block on the same inputs reproduces ``out`` / the gated O / the LSE BITWISE (one core, one list, one set of operands); the
    list equals the one a second indexer block leaves in its workspace (as sets per row), and ``index_k_compressed``'s rows past
    the prompt's complete blocks stay the caller's."""
    geom_kw, B, S = GEOM_SMALL, 1, 2560
    c = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=23)
    _indexer_two_stage(c, "small8-2 S=2560 indexer (outputs)")
    _check(c, "small8-2 S=2560 indexer (outputs)")
    caller = _run(geom_kw, B, S, index_band=True, inp=c.inp, ref=c.ref, lists=(c.ix["ids_out"].clone(), c.ix["lens_out"].clone()))
    _check(caller, "small8-2 S=2560 caller list = the indexer's output")
    for a, b_ in zip(c.runs[0], caller.runs[0]):
        assert torch.equal(_bits(a) if a.dtype != torch.float32 else a, _bits(b_) if b_.dtype != torch.float32 else b_)
    again = _run(geom_kw, B, S, indexer=True, indexer_outputs=False, inp=c.inp, ref=c.ref)
    assert torch.equal(torch.sort(again.ix["kernel_ids"], dim=-1).values, torch.sort(c.ix["kernel_ids"], dim=-1).values)
    assert torch.equal(_bits(again.ix["kbar"]), _bits(c.ix["kbar_out"][:, : S // BS]))


@requires_rubin
def test_indexer_identity_below_the_bound_launches_only_a_requested_cache_compress():
    """At ``S <= identity_bound`` the indexer block launches nothing of its own (the list is a plan-time constant, the
    workspace the caller-list block's); with ``index_k_compressed`` given exactly ONE launch runs -- the key compress into the
    caller's cache, within one bf16 ulp of the oracle's compressed keys, rows past the prompt untouched -- and the identity
    list / counts are copied into ``block_ids_out`` / ``block_lens_out``."""
    geom_kw, B, S = GEOM_SMALL, 2, 2051
    lens_t = _i32([2051, 1500])
    plain = _run(geom_kw, B, S, indexer=True, seq_lens=lens_t, seed=5)
    m = _indexer_two_stage(plain, "small8-2 S=2051 B=2 indexer identity (no outputs)")
    assert m["identity"] and m["launches"] == 0
    _check(plain, "small8-2 S=2051 B=2 indexer identity")
    caller = _run(
        geom_kw,
        B,
        S,
        index_band=True,
        seq_lens=lens_t,
        inp=plain.inp,
        ref=plain.ref,
        lists=(plain.ix["kernel_ids"].contiguous(), plain.ix["kernel_lens"].contiguous()),
    )
    assert torch.equal(_bits(plain.runs[0][0]), _bits(caller.runs[0][0])) and plain.blk.get_workspace_size() == caller.blk.get_workspace_size()
    with_cache = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seq_lens=lens_t, inp=plain.inp, ref=plain.ref)
    assert with_cache.ix["launches"] == 1
    ref_geom = RefQsaGeometry(**geom_kw, qsa=RefQsaSpec(index_band=True))
    _, _, _, _, k_c = qsa_indexer_reference(
        plain.inp["h"],
        plain.inp["w_i"],
        plain.inp["w_iq_norm"],
        plain.inp["w_ik_norm"],
        None,
        None,
        plain.inp["cos"],
        plain.inp["sin"],
        ref_geom.qsa,
        ref_geom,
        seq_lens=lens_t,
    )
    kb = with_cache.ix["kbar_out"]
    for bi, ln in enumerate((2051, 1500)):
        nb = ln // BS
        d = ((kb[bi, :nb].float() - k_c[bi, :nb].float()).abs() / _bf16_ulp(k_c[bi, :nb])).max().item()
        assert d <= 1.0, f"entry {bi}: {d:.2f} ulps"
        assert torch.isnan(kb[bi, nb:].float()).all()
    assert torch.equal(with_cache.ix["ids_out"].view(B, S, -1), plain.ix["kernel_ids"]) and torch.equal(
        with_cache.ix["lens_out"].view(B, S), plain.ix["kernel_lens"]
    )
    assert torch.equal(_bits(with_cache.runs[0][0]), _bits(plain.runs[0][0]))


@requires_rubin
def test_indexer_block_with_write_through_is_bitwise_the_one_without():
    """The indexer composes with the paged write-through: the same selection, ``out`` / LSE / gated O bitwise, the raw indexer
    key's pool written at the slots (the stage order: indexer, cache write, sparse SDPA)."""
    geom_kw, B, S, ps = GEOM_SMALL, 1, 2560, 16
    plain = _run(geom_kw, B, S, indexer=True, indexer_outputs=True, seed=9)
    _indexer_two_stage(plain, "small8-2 S=2560 indexer plain")
    _check(plain, "small8-2 S=2560 indexer plain")
    t, geom = B * S, plain.geom
    n_pages = t // ps + 2
    gen = torch.Generator(device="cuda").manual_seed(1)
    slot = torch.randperm(n_pages * ps, generator=gen, device="cuda")[:t].to(torch.int32)
    slot[::11] = -1
    k_cache, v_cache = _nan_pool(n_pages, geom.h_kv, ps, geom.d_head), _nan_pool(n_pages, geom.h_kv, ps, geom.d_head)
    index_cache = torch.full((n_pages, ps, geom.qsa.index_head_dim), float("nan"), device="cuda", dtype=BF16)
    wt = _run(
        geom_kw,
        B,
        S,
        indexer=True,
        indexer_outputs=True,
        seed=9,
        inp=plain.inp,
        ref=plain.ref,
        paged_kv_page_size=ps,
        pools=dict(k_cache=k_cache, v_cache=v_cache, slot_mapping=slot, index_k_raw=index_cache),
    )
    _check(wt, "small8-2 S=2560 indexer write-through")
    assert (
        torch.equal(wt.ix["ids_out"], plain.ix["ids_out"])
        and torch.equal(_bits(wt.runs[0][0]), _bits(plain.runs[0][0]))
        and torch.equal(wt.runs[0][1], plain.runs[0][1])
    )
    st = wt.blk._stages
    assert st.index(wt.blk._indexer) == st.index(wt.blk._cache_write) - 1 == st.index(wt.blk._sdpa) - 2
    lay = wt.blk._layout()
    raw = index_k_raw_view(_view(wt.ws, lay.proj, (t, geom.n_qkvg), BF16), geom, B, S)
    live = slot >= 0
    s_ = slot[live].long()
    assert torch.equal(_bits(index_cache[s_ // ps, s_ % ps, :]), _bits(raw[0, live, 0]))


# ============================================================================ host: what the matrix cannot run yet
def test_fuse_gate_under_qsa_is_still_the_typed_decline():
    """The ``fuse_gate`` axis of the matrix waits for the sparse epilogue-gate arm to be bound through the block; until then
    the declaration declines it naming the feature (this test INVERTS when the arm lands: the matrix gains its gated cells)."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=dev)  # noqa: E731
    with pytest.raises(NotImplementedError, match="fuse_gate=True"):
        GatedAttentionBlockFwd(
            z(1, 8, geom.d_model),
            z(geom.n_qkvg, geom.d_model),
            z(geom.d_head),
            z(geom.d_head),
            z(1, 8, 64),
            z(1, 8, 64),
            z(geom.d_model, geom.h_q * geom.d_head),
            z(1, 8, geom.d_model),
            geom,
            fuse_gate=True,
        )
