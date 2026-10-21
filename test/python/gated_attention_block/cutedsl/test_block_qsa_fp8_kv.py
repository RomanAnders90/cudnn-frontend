# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gated attention block's e4m3 KV cache (``kv_cache_dtype=torch.float8_e4m3fn``): the serving stacks' ``--kv-cache-dtype fp8``
recipe -- the pools hold e4m3 codes under per-tensor fp32 DEscales ``execute(k_scale=, v_scale=)`` (value = code x scale), a bf16 Q
attends over the dequantized rows.  Two forms, one attribute: (a) the WRITE quantizes the post-RoPE K / V on the way into the pools --
the two-stage write-through (two per-tensor casts + the bit copy through the pools' bf16 views) or, under ``fuse_prepare=True``, the
prepare launch's e4m3 pool arm (bitwise the two-stage form) -- on every form that writes through; (b) the BLOCK-SPARSE DECODE mode
(``kv_source="paged"`` on a dense ``[B, S_q]`` QsaSpec declaration, caller lists or the in-block indexer) reads the pools through the
sparse core's IN-KERNEL dequant arm: the gather warps land the listed e4m3 rows and expand them to the bf16 operand tile in SMEM,
``k_scale`` folded into the softmax scale and ``v_scale`` into the normaliser -- every list form, ``S_q``, split, LSE and gate placement
of the bf16 mode, no scratch (the 2026-10-14 gathered cast + decode tile form is retired).  bf16 Q over an e4m3 pool only (no fp8-Q form).

Host cells (any device; the declaration cells even on the CPU): the declaration FACTS (the sparse stage reading e4m3 pools, the
write-through stage or the e4m3 prepare, the bf16 mode's rules for the gate / the split / the layout, the LSE servable, the carve =
the bf16 twin's + the three e4m3 slots at the END and no scratch, the stage order, the write-through-only declaration, the in-block
indexer over e4m3 pools); the typed declines that STAY (e5m2, another dtype, the attribute without the write-through, ``quant``, f16
activations, the dense DECODE form over e4m3 pools -- the gap --, the EXTEND form, pools of the wrong dtype both ways) and the seven
limits of the retired form now SERVED (``S_q > 4``, ``return_lse``, a split, ``fuse_norm_rope``, an explicit ``inplace_qkv``, the
per-row list, the indexer); the execute-time form checks on an UNCOMPILED block under ``set_sync_debug_mode("error")`` (the scales
required under the attribute, refused without it, their form); the envelope's ``kv_cache_dtypes`` under caller lists AND the indexer.

Rubin cells (``requires_rubin``; one compiled block per cell): the decode ladder against the fp32 oracle ON THE DEQUANTIZED POOL --
the dense block reference over the WHOLE sequence gives the post-norm / post-RoPE K / V; the PREFIX rows are quantized by torch with
the per-tensor scales (``scale = amax / 448``) into NaN-filled e4m3 pools, the STEP rows are the block's own quantizing write; the
oracle attends in fp32 over ``bf16(fp32(code) x scale)`` of every row (the step rows' codes computed the way the block's cast computes
them: ``sat_e4m3(row x (1 / scale))``) with the bf16 post-RoPE Q over ``qsa_visible_mask(..., pos0=)`` (the shared list) or per row,
then the gate and the out projection -- ``cos(out) > 0.999``, the gated O (the workspace slot) and the LSE within ``atol 2e-2`` (the
sparse module's bars, never widened), a dead row (``kv_lens = 0``; a row at a negative position) EXACTLY 0 / ``-inf``, the pools' step
rows BITWISE torch's quantization of the block's own K rows and V band, the prefix rows and the NULL block untouched, two launches
bitwise; REPORTED without a gate: the distance to the PRE-quantization bf16 oracle (the serving recipe's own error).  The ladder:
``S_q`` 1 .. 4 shared and 4 / 8 per row, B 1 / 2 / 3 / 4, ``kv_lens`` up to 20000 incl. 2052, a dead entry and a sequence shorter than
the step, pages 16 / 32 / 64 on HND / unified / NHD pools, ``block_lens`` given / absent, the gate in the core's epilogue, in the split
combine and separate, splits 1 / 4 / 16 with ``return_lse``, ``fuse_norm_rope``, an explicit ``inplace_qkv``, ``fuse_prepare`` (the
prepare's e4m3 pool arm), the indexer band's raw-key write-through (bf16), the 24 / 2 Flash-Next geometry; the MTP residues on a
short sequence; the POWER-OF-TWO pin: at ``k_scale`` / ``v_scale`` in {1, 0.5, 2} the e4m3 block's ``out``, gated O and LSE are
BITWISE the bf16 block fed pools holding ``bf16(code x scale)`` (the fold's exactness: the kernel's own pin lifted to the block, the
gate in the combine included); the in-block INDEXER over e4m3 pools (the two-stage oracle: the selection the fp64 top-k of the block's
own operands up to ties, the attention within budget of the fp32 oracle fed the kernel's list over the dequantized pool; ``fuse_prepare``
on == off BITWISE on ``out``, the list and the pools, the census 8 vs 13 launches by kernel name; the first execute captured and replayed
bitwise); the write-through-only PREFILL form in both write forms (``out`` BITWISE the bf16-pool twin's, the pools bitwise the torch cast
and each other); ``get_workspace_size()`` honest (a sentinel past the carve untouched; no scratch slot); CUDA-graph capture of the FIRST
execute on a NEW stream with no warm-up under ``set_sync_debug_mode("error")``, the replay bitwise eager, and the replay over NEW scales /
lengths / tables / lists / slots / inputs bitwise the eager run over them.
"""

import contextlib
import os
import sys
import types

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block import KV_SOURCE_PAGED, GatedAttentionBlockFwd, GatedAttentionBlockGeometry, QsaSpec, index_k_raw_view
from cudnn.gated_attention_block.api import KV_CACHE_E4M3, _cols, _plan_workspace, _QsaPrepare, _QuantizedCacheWrite, _SparseSdpa, _view

requirement_error = cutedsl_requirement_error("the gated attention block's e4m3 KV-cache tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_block_qsa_decode as _qsa  # noqa: E402  (the bf16 decode cases: the indexer case this module turns into its e4m3 twin)
from gated_block_qsa_reference import full_block_ids, qsa_visible_mask, random_block_ids  # noqa: E402
from gated_block_reference import RefGeometry, build_rope_tables, gated_attention_block_reference  # noqa: E402

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16, F16, E4M3 = torch.bfloat16, torch.float16, torch.float8_e4m3fn
D = 256
ROPE = 64
BS = 4  # tokens per selectable block
COS_OUT = 0.999  # the end-to-end suite's bar on `out`
ATOL = 2e-2  # the sparse module's bar on O / LSE -- never widened here
E4M3_MAX = 448.0
GEOM_SMALL = dict(d_model=512, h_q=8, h_kv=2, d_head=D, rope_dim=ROPE)  # group 4, cheap projections
GEOM_FLASH_NEXT = dict(d_model=2560, h_q=24, h_kv=2, d_head=D, rope_dim=ROPE)  # Qwen3.8-Flash-Next TP 1 (group 12)


@contextlib.contextmanager
def _no_host_sync():
    """Every declaration-time and form check stays on the host without a device sync (Rule 3) -- enforced, not read."""
    if not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev)


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of a bf16 / f16 / e4m3 tensor: a NaN-safe bitwise comparison."""
    return x.contiguous().view(torch.int8 if x.element_size() == 1 else torch.int16)


def _sentinel(dtype):
    return float(torch.finfo(dtype).max) / 4


def _pools(n_pages, h, page, layout, dtype, device=_DEV, fill=float("nan")):
    """Two pools as ``[P, H, page, D]`` views: ``hnd`` (compact), ``nhd`` (by strides), ``unified`` (one ``[P, H, page, 2D]`` content
    row per token, K at ``+0``, V at ``+D``).  NaN-filled (an e4m3 NaN is the code 0x7f): a row the core should never read poisons O."""
    mk = lambda *shape: torch.full(shape, fill, device=device, dtype=torch.float32).to(dtype)  # noqa: E731
    if layout == "hnd":
        return tuple(mk(n_pages, h, page, D) for _ in range(2))
    if layout == "nhd":
        return tuple(mk(n_pages, page, h, D).permute(0, 2, 1, 3) for _ in range(2))
    assert layout == "unified"
    row = mk(n_pages, h, page, 2 * D)
    return row[..., :D], row[..., D:]


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The block's quantizing write as torch computes it: ``sat_e4m3(fp32(row) x fp32(1 / scale))`` -- one fp32 product, one rounding
    (torch's cast rounds to nearest even like the kernel's ``cvt.rn``; the clamp is the ``satfinite`` the kernel applies)."""
    inv = torch.reciprocal(scale.float())
    return (x.float() * inv).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)


def _dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The dequantized value a bf16 pool would hold: ``bf16(fp32(code) x scale)`` -- one fp32 rounding (exact at a power-of-two scale)."""
    return (codes.float() * scale.float()).to(BF16)


def _scale_of(x: torch.Tensor) -> torch.Tensor:
    """A per-tensor DEscale calibrated on ``x`` (the serving stacks' ``amax / 448``), a 1-element fp32 CUDA tensor."""
    amax = x.float().abs().amax().clamp_min(1e-12)
    return (amax / E4M3_MAX).reshape(1).to(torch.float32).contiguous()


def _scale_const(v: float) -> torch.Tensor:
    return torch.tensor([float(v)], dtype=torch.float32, device=_DEV)


def _geom(geom_kw, top_k=512, index_band=False, index_list_form=None, **over):
    kw = dict(geom_kw, causal_bottom_right=True)
    kw.update(over)
    return GatedAttentionBlockGeometry(**kw, qsa=QsaSpec(top_k=top_k, index_band=index_band, index_list_form=index_list_form)), RefGeometry(**kw)


def _declare(
    geom_kw,
    batch,
    s_q,
    *,
    page=16,
    layout="hnd",
    n_pages=8,
    dtype=BF16,
    pool_dtype=E4M3,
    top_k=512,
    index_band=False,
    index_list_form=None,
    geom_over=None,
    qsa=True,
    **blk_kw,
):
    """A declared (not compiled) block on EMPTY tensors: the declaration cells (any device).  ``kv_cache_dtype`` rides ``blk_kw``."""
    kw0 = dict(geom_kw)
    kw0.update(geom_over or {})
    if qsa:
        geom, _ = _geom(geom_kw, top_k=top_k, index_band=index_band, index_list_form=index_list_form, **(geom_over or {}))
    else:
        geom = GatedAttentionBlockGeometry(**kw0)
    mk = lambda *shape: torch.empty(*shape, dtype=dtype, device=_DEV)  # noqa: E731
    kc, vc = _pools(n_pages, geom.h_kv, page, layout, pool_dtype)
    bt = torch.zeros(batch, n_pages, dtype=torch.int32, device=_DEV)
    kw = dict(paged_kv_page_size=page, kv_source=KV_SOURCE_PAGED, sample_k_cache=kc, sample_v_cache=vc, sample_block_table=bt, kv_cache_dtype=E4M3)
    kw.update(blk_kw)
    blk = GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=dtype, device=_DEV),
        torch.ones(D, dtype=dtype, device=_DEV),
        mk(batch, s_q, ROPE),
        mk(batch, s_q, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )
    return blk, dict(k_cache=kc, v_cache=vc, block_table=bt, geom=geom)


def _indexer_samples(batch, n_pages=8, page=16, cap=8):
    """The decode indexer's three side-cache samples (their FORM): the paged compressed-key cache, the ring, the ring block table."""
    return dict(
        sample_index_kv_cache=torch.empty(n_pages, page // BS, 1, 128, dtype=BF16, device=_DEV),
        sample_index_state_cache=torch.empty(batch + 1, cap, 1, 128, dtype=BF16, device=_DEV),
        sample_index_state_block_table=torch.zeros(batch, 1, dtype=torch.int32, device=_DEV),
    )


def _declare_indexer(batch, s_q, **blk_kw):
    """A declared (not compiled) decode-INDEXER block over e4m3 pools on EMPTY tensors."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    kc, vc = _pools(8, geom.h_kv, 16, "hnd", E4M3)
    kw = dict(
        paged_kv_page_size=16,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(batch, 8, dtype=torch.int32, device=_DEV),
        kv_cache_dtype=E4M3,
        **_indexer_samples(batch),
    )
    kw.update(blk_kw)
    return GatedAttentionBlockFwd(
        mk(batch, s_q, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=BF16, device=_DEV),
        torch.ones(D, dtype=BF16, device=_DEV),
        mk(batch, s_q, ROPE),
        mk(batch, s_q, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(batch, s_q, geom.d_model),
        geom,
        **kw,
    )


# ---------------------------------------------------------------------------
# Host: the declaration facts, the typed declines, the execute form checks, the envelope, the carve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 8])
def test_declaration_facts_of_the_e4m3_mode(s_q):
    """Over e4m3 pools the block-sparse decode declaration builds the quantizing write-through and the SPARSE stage reading e4m3 pools
    (the in-kernel dequant arm: ``_SparseSdpa.kv_e4m3``, the pool descriptors in e4m3, the two scale operands), and follows the bf16
    mode's rules -- in place, the gate where the bf16 twin puts it, the fill-rule split, the list form of ``S_q``, the LSE servable --;
    the carve is the twin's plus ``kc8`` / ``vc8`` / ``kv_inv`` at the END and NO scratch (the twin without the attribute is
    byte-identical to before); ``fuse_prepare=True`` swaps the write-through for the prepare's e4m3 pool arm."""
    B = 3
    with _no_host_sync():
        blk, env = _declare(GEOM_SMALL, B, s_q, fuse_gate=True)
        twin, _ = _declare(GEOM_SMALL, B, s_q, fuse_gate=True, pool_dtype=BF16, kv_cache_dtype=None)
        lse, _ = _declare(GEOM_SMALL, B, s_q, return_lse=True)
        fused, _ = _declare(GEOM_SMALL, B, s_q, fuse_prepare=True, fuse_gate=True)
    assert blk.kv_cache_dtype == E4M3 and blk.kv_cache_e4m3 and blk.sparse_decode and blk.qsa_shared_list is (s_q <= 4) and not hasattr(blk, "upcast_decode")
    st = blk._sdpa
    assert isinstance(st, _SparseSdpa) and st.kv_e4m3 and st.paged and st.kv_cache_dtype == E4M3 and isinstance(blk._cache_write, _QuantizedCacheWrite)
    ops = st._operands()
    assert ops["k"].dtype == E4M3 and ops["v"].dtype == E4M3 and ops["q"].dtype == BF16
    assert ops["k_scale"].shape == (1,) and ops["k_scale"].dtype == torch.float32 and ops["v_scale"].shape == (1,)
    assert twin._sdpa._operands()["k"].dtype == BF16 and twin._sdpa._operands()["k_scale"] is None and not twin._sdpa.kv_e4m3
    # the bf16 mode's rules, read off the twin
    assert blk.inplace_qkv is True and blk.gate_fused is twin.gate_fused and blk.decode_split_kv == twin.decode_split_kv and blk.seq_lens_present
    assert blk.qsa_list_form == twin.qsa_list_form and st.list_per_sequence is twin._sdpa.list_per_sequence and st.split_kv == twin._sdpa.split_kv
    gate_stage = [] if blk.gate_fused else ["sigmoid_gate"]
    assert [x.name for x in blk._stages] == ["qkv_gate_proj", "qk_norm_rope", "cache_write_e4m3", "sdpa_sparse"] + gate_stage + ["out_proj"]
    assert lse._sdpa.want_lse and lse._sdpa.kv_e4m3 and lse.return_lse
    assert fused.fuse_prepare is True and isinstance(fused._prepare, _QsaPrepare) and fused._prepare.kv_e4m3 and fused._cache_write is None
    assert [x.name for x in fused._stages] == ["qkv_gate_proj", "qsa_prepare", "sdpa_sparse"] + ([] if fused.gate_fused else ["sigmoid_gate"]) + ["out_proj"]
    # the carve: the twin's slots, then kc8 / vc8 / kv_inv; no gathered-cast scratch
    lay, lay0 = blk._layout(), twin._layout()
    assert (lay.proj, lay.q, lay.k, lay.v, lay.o) == (lay0.proj, lay0.q, lay0.k, lay0.v, lay0.o) and lay.q == -1 and lay.proj == 0
    assert lay.kc8 == lay0.total_bytes and lay.vc8 > lay.kc8 and lay.kv_inv > lay.vc8 and lay.total_bytes > lay.kv_inv
    assert lay0.kc8 == lay0.vc8 == lay0.kv_inv == -1
    assert not hasattr(lay, "kv_gathered") and not hasattr(lay, "kv_len_t")
    assert (
        lay.total_bytes - lay0.total_bytes == lay.total_bytes - lay.kc8
    )  # the three e4m3 slots are the whole difference (the Rubin sentinel cell reads get_workspace_size)
    # the write-through-only declaration (the prefill form over the block's own K / V): the e4m3 write, the sparse stage over bf16 operands
    with _no_host_sync():
        pre, _ = _declare(
            GEOM_SMALL, 2, 8, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, geom_over=dict(causal_bottom_right=False)
        )
    assert pre.kv_cache_e4m3 and isinstance(pre._cache_write, _QuantizedCacheWrite) and pre.inplace_qkv is True
    assert isinstance(pre._sdpa, _SparseSdpa) and not pre._sdpa.kv_e4m3 and pre._sdpa.kv_cache_dtype is None and not pre._sdpa.paged
    lp = pre._layout()
    assert lp.kc8 >= 0 and lp.vc8 >= 0 and lp.kv_inv >= 0
    # the dense (no QsaSpec) prefill with the e4m3 write-through is served too
    with _no_host_sync():
        dense, _ = _declare(GEOM_SMALL, 2, 8, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, qsa=False)
    assert dense.kv_cache_e4m3 and isinstance(dense._cache_write, _QuantizedCacheWrite) and dense.qsa is None


def test_declaration_facts_of_the_indexer_over_e4m3_pools():
    """``QsaSpec(index_source="indexer")`` over e4m3 pools CONSTRUCTS (served since 2026-10-21): the sparse stage reads e4m3 pools, the
    indexer's decode step is built as over bf16 pools (it reads the compressed-key cache, never the pools), ``fuse_prepare`` on gives the
    prepare's e4m3 pool arm and off the quantizing write-through, the carve is the bf16 declaration's plus the three e4m3 slots."""
    with _no_host_sync():
        on = _declare_indexer(2, 4, fuse_prepare=True, fuse_gate=True, return_lse=True)
        off = _declare_indexer(2, 4, fuse_prepare=False, fuse_gate=True, return_lse=True)
        twin = _declare_indexer(
            2,
            4,
            fuse_prepare=False,
            fuse_gate=True,
            return_lse=True,
            kv_cache_dtype=None,
            sample_k_cache=_pools(8, 2, 16, "hnd", BF16)[0],
            sample_v_cache=_pools(8, 2, 16, "hnd", BF16)[1],
        )
    for b in (on, off):
        assert b.indexer_decode and b.kv_cache_e4m3 and b._sdpa.kv_e4m3 and b._indexer_decode is not None and b.qsa_list_form == "per_row"
        assert b.return_lse and b._sdpa.want_lse and b.decode_split_kv == twin.decode_split_kv and b.gate_fused is twin.gate_fused
    assert on._prepare is not None and on._prepare.kv_e4m3 and on._prepare.decode_arms and on._cache_write is None
    assert off._prepare is None and isinstance(off._cache_write, _QuantizedCacheWrite)
    assert [x.name for x in on._stages] == ["qkv_gate_proj", "qsa_prepare", "qsa_indexer", "sdpa_sparse"] + ([] if on.gate_fused else ["sigmoid_gate"]) + [
        "out_proj"
    ]
    assert [x.name for x in off._stages] == ["qkv_gate_proj", "qk_norm_rope", "qsa_indexer", "cache_write_e4m3", "sdpa_sparse"] + (
        [] if off.gate_fused else ["sigmoid_gate"]
    ) + ["out_proj"]
    lay, lay0 = off._layout(), twin._layout()
    assert lay.kc8 >= 0 and lay.vc8 > lay.kc8 and lay.kv_inv > lay.vc8 and lay0.kc8 == -1 and not hasattr(lay, "kv_gathered")
    # the e4m3 slots sit before the decode indexer's five (which stay at the very END): the indexer slots shift by exactly the e4m3 slots' size
    assert (lay.proj, lay.o) == (lay0.proj, lay0.o) and lay.ix_q > lay.kv_inv and lay.ix_q - lay0.ix_q == lay.total_bytes - lay0.total_bytes


def test_the_carve_without_the_attribute_is_byte_identical_and_the_e4m3_slots_append():
    g, _ = _geom(GEOM_SMALL)
    base = _plan_workspace(g, 2, 4, BF16, False, False, True)
    assert base == _plan_workspace(g, 2, 4, BF16, False, False, True, kv_cache_e4m3=False) and base.kc8 == -1
    e4 = _plan_workspace(g, 2, 4, BF16, False, False, True, kv_cache_e4m3=True)
    assert (e4.proj, e4.q, e4.k, e4.v, e4.o) == (base.proj, base.q, base.k, base.v, base.o) and e4.kc8 == base.total_bytes
    assert e4.vc8 - e4.kc8 >= 2 * 4 * g.h_kv * D and e4.kv_inv - e4.vc8 >= 2 * 4 * g.h_kv * D and e4.total_bytes >= e4.kv_inv + 8
    assert not hasattr(e4, "kv_gathered") and not hasattr(e4, "kv_len_t")  # no gathered-cast scratch: the core reads the pools in-kernel
    with pytest.raises(ValueError, match="form of the bf16 pipeline"):
        _plan_workspace(g, 2, 4, BF16, False, False, False, fp8=True, kv_cache_e4m3=True)


def test_typed_declines_of_the_e4m3_mode():
    """Every decline by name, device-free (the declaration never touches the device), the mirror images included -- and the seven limits of
    the retired gathered-cast form now SERVED through the in-kernel arm (each a constructing declaration whose sparse stage reads e4m3)."""
    dec = _declare
    # the attribute's own rows
    with pytest.raises(NotImplementedError, match="float8_e5m2.*e4m3 KV cache is the one served"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=torch.float8_e5m2)
    with pytest.raises(ValueError, match="kv_cache_dtype must be None, the activation dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=torch.int8)
    with pytest.raises(TypeError, match="kv_cache_dtype must be a torch.dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype="e4m3")
    with pytest.raises(ValueError, match="differs from the activation dtype"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=F16, pool_dtype=BF16)
    with pytest.raises(ValueError, match="belongs to the paged KV-cache write-through"):
        dec(GEOM_SMALL, 2, 8, paged_kv_page_size=0, kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, qsa=False)
    with pytest.raises(NotImplementedError, match="e4m3 arm expands the pools to bf16 operand tiles.*f16 activations over an e4m3 cache are a follow-up"):
        dec(GEOM_SMALL, 2, 1, dtype=F16)
    # the pools' dtype both ways (the declaration samples)
    with pytest.raises(ValueError, match="sample_k_cache must be torch.float8_e4m3fn"):
        dec(GEOM_SMALL, 2, 1, pool_dtype=BF16)
    with pytest.raises(ValueError, match="sample_k_cache must be torch.bfloat16"):
        dec(GEOM_SMALL, 2, 1, kv_cache_dtype=None)
    # quant / training: the write-through's own rows speak
    from cudnn.gated_attention_block import QuantSpec

    g, _ = _geom(GEOM_SMALL)
    mk8 = lambda *shape: torch.empty(*shape, dtype=E4M3, device=_DEV)  # noqa: E731
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with quant=QuantSpec"):
        GatedAttentionBlockFwd(
            mk8(2, 8, 512),
            mk8(g.n_qkvg, 512),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.empty(2, 8, ROPE, dtype=BF16, device=_DEV),
            torch.empty(2, 8, ROPE, dtype=BF16, device=_DEV),
            mk8(512, 8 * D),
            torch.empty(2, 8, 512, dtype=BF16, device=_DEV),
            GatedAttentionBlockGeometry(**GEOM_SMALL),
            quant=QuantSpec(descale_h=1.0, descale_w_qkvg=1.0, descale_w_o=1.0, scale_q=1.0, scale_k=1.0, scale_v=1.0, scale_o=1.0),
            paged_kv_page_size=16,
            kv_cache_dtype=E4M3,
        )
    with pytest.raises(NotImplementedError, match="paged_kv_page_size with save_for_backward=True"):
        dec(
            GEOM_SMALL,
            2,
            8,
            kv_source="self",
            sample_k_cache=None,
            sample_v_cache=None,
            sample_block_table=None,
            qsa=False,
            save_for_backward=True,
            inplace_qkv=False,
        )
    # over the pools: the two declines that stay -- the dense DECODE form (the gap) and the EXTEND form
    with pytest.raises(NotImplementedError, match="dense geometry \\(the DECODE form\\).*in-kernel dequant arm"):
        dec(GEOM_SMALL, 2, 1, qsa=False, geom_over=dict(is_causal=True, causal_bottom_right=True))
    with pytest.raises(NotImplementedError, match="EXTEND form over e4m3 pools"):
        _declare_packed_extend()
    # the seven limits of the retired form, SERVED since 2026-10-21 (the in-kernel arm): each constructs with the sparse stage reading e4m3
    served = [
        ("S_q 5 (the per-row list past the shared cap)", 5, {}),
        ("return_lse", 2, dict(return_lse=True)),
        ("a split", 2, dict(decode_split_kv=4)),
        ("fuse_norm_rope", 2, dict(fuse_norm_rope=True)),
        ("inplace_qkv=True explicit", 2, dict(inplace_qkv=True)),
        ("inplace_qkv=False", 2, dict(inplace_qkv=False)),
        ("the declared per-row list at S_q 4", 4, dict(index_list_form="per_row")),
        ("fuse_prepare", 2, dict(fuse_prepare=True)),
    ]
    with _no_host_sync():
        for label, s_q, kw in served:
            blk, _ = dec(GEOM_SMALL, 2, s_q, **kw)
            assert isinstance(blk._sdpa, _SparseSdpa) and blk._sdpa.kv_e4m3 and blk.kv_cache_e4m3 and blk.sparse_decode, label
        assert dec(GEOM_SMALL, 2, 2, decode_split_kv=4)[0].decode_split_kv == 4 and dec(GEOM_SMALL, 2, 2, return_lse=True)[0]._sdpa.want_lse
        assert (
            dec(GEOM_SMALL, 2, 2, inplace_qkv=False)[0].inplace_qkv is False and dec(GEOM_SMALL, 2, 4, index_list_form="per_row")[0].qsa_list_form == "per_row"
        )
        assert dec(GEOM_SMALL, 2, 5)[0].qsa_list_form == "per_row" and dec(GEOM_SMALL, 2, 2, fuse_norm_rope=True)[0].fuse_norm_rope_effective
        # the in-block indexer over e4m3 pools: served (the former `indexer_e4m3_pool` row is gone from the envelope)
        ix = _declare_indexer(1, 1)
        assert ix.indexer_decode and ix._sdpa.kv_e4m3
    env = GatedAttentionBlockFwd.serving_envelope(ix.geom, BF16, _DEV, device_cc=(10, 7)).decode
    assert "indexer_e4m3_pool" not in env.declines and env.kv_cache_dtypes == (BF16, E4M3)


def _declare_packed_extend():
    """A packed (thd=True) EXTEND declaration on a DENSE geometry over e4m3 pools: the table's rows pass (bottom-right, causal, d256)
    and the attribute's EXTEND row declines."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True)
    T = 32
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731
    kc, vc = _pools(8, geom.h_kv, 16, "hnd", E4M3)
    return GatedAttentionBlockFwd(
        mk(T, geom.d_model),
        mk(geom.n_qkvg, geom.d_model),
        torch.ones(D, dtype=BF16, device=_DEV),
        torch.ones(D, dtype=BF16, device=_DEV),
        mk(T, ROPE),
        mk(T, ROPE),
        mk(geom.d_model, geom.h_q * D),
        mk(T, geom.d_model),
        geom,
        thd=True,
        num_sequences=2,
        max_seq_len=T,
        paged_kv_page_size=16,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=kc,
        sample_v_cache=vc,
        sample_block_table=torch.zeros(2, 8, dtype=torch.int32, device=_DEV),
        kv_cache_dtype=E4M3,
    )


def test_execute_form_checks_of_the_e4m3_mode():
    """The scales are a DECLARATION-vs-argument contract checked before any plan exists: required (1-element fp32 CUDA on h's device)
    under the attribute, refused on a bf16-pool block; the pools in the pool dtype at execute -- every refusal on an UNCOMPILED block
    without a device sync (Rule 3)."""
    if not torch.cuda.is_available():
        pytest.skip("the execute form checks need CUDA tensors (the device is never touched)")
    B, s_q = 2, 2
    blk, env = _declare(GEOM_SMALL, B, s_q)
    twin, env2 = _declare(GEOM_SMALL, B, s_q, pool_dtype=BF16, kv_cache_dtype=None)
    g = env["geom"]
    mk = lambda *shape, dt=BF16: torch.zeros(*shape, dtype=dt, device=_DEV)  # noqa: E731
    one = torch.ones(1, dtype=torch.float32, device=_DEV)
    ids = torch.full((B, 512), -1, dtype=torch.int32, device=_DEV)
    common = dict(
        w_qkvg=mk(g.n_qkvg, g.d_model),
        w_q_norm=torch.ones(D, dtype=BF16, device=_DEV),
        w_k_norm=torch.ones(D, dtype=BF16, device=_DEV),
        cos=mk(B, s_q, ROPE),
        sin=mk(B, s_q, ROPE),
        w_o=mk(g.d_model, g.h_q * D),
        out=mk(B, s_q, g.d_model),
        workspace=mk(16, dt=torch.uint8),
        block_ids=ids,
        slot_mapping=torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV),
        kv_lens=torch.zeros(B, dtype=torch.int32, device=_DEV),
    )

    def run(b, e, **over):
        kw = dict(common, k_cache=e["k_cache"], v_cache=e["v_cache"], block_table=e["block_table"])
        kw.update(over)
        b.execute(mk(B, s_q, g.d_model), **kw)

    with _no_host_sync():
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, v_scale=one)
        with pytest.raises(ValueError, match="needs v_scale at execute"):
            run(blk, env, k_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=one.to(BF16), v_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=torch.ones(2, dtype=torch.float32, device=_DEV), v_scale=one)
        with pytest.raises(ValueError, match="needs k_scale at execute"):
            run(blk, env, k_scale=1.0, v_scale=one)
        with pytest.raises(ValueError, match="k_scale, v_scale: the per-tensor scales of an e4m3 KV cache"):
            run(twin, env2, k_scale=one, v_scale=one)
        with pytest.raises(ValueError, match="k_cache must be torch.float8_e4m3fn"):
            run(blk, env, k_cache=env2["k_cache"], v_cache=env2["v_cache"], k_scale=one, v_scale=one)
        with pytest.raises(ValueError, match="k_cache must be torch.bfloat16"):
            run(twin, env2, k_cache=env["k_cache"], v_cache=env["v_cache"])
        # past the form checks the uncompiled block stops at its own plan guard -- nothing above touched the device
        with pytest.raises(RuntimeError, match="call compile\\(\\) before execute\\(\\)"):
            run(blk, env, k_scale=one, v_scale=one)
        # the scales on a block without the write-through at all
        plain, _ = _declare(
            GEOM_SMALL,
            B,
            s_q,
            paged_kv_page_size=0,
            kv_source="self",
            sample_k_cache=None,
            sample_v_cache=None,
            sample_block_table=None,
            kv_cache_dtype=None,
            geom_over=dict(causal_bottom_right=False),
        )
        per_token_ids = torch.full((B * s_q, 512), -1, dtype=torch.int32, device=_DEV)  # the prefill form's list (the index check runs first)
        with pytest.raises(ValueError, match="k_scale, v_scale: the paged KV-cache inputs belong to a block declared with paged_kv_page_size > 0"):
            plain.execute(mk(B, s_q, g.d_model), **dict(common, block_ids=per_token_ids, slot_mapping=None, kv_lens=None, k_scale=one, v_scale=one))
        # the sparse stage alone holds the scales to its declaration both ways (the block never hands them to a stage over bf16 operands)
        with pytest.raises(ValueError, match="k_scale / v_scale are the e4m3 pools' per-tensor DEscales"):
            plain._sdpa._impl = plain._sdpa._compiled = True  # past the plan guard without a plan: the form checks run first
            plain._sdpa.execute(
                mk(B, s_q, g.h_q, D), mk(B, s_q, g.h_kv, D), mk(B, s_q, g.h_kv, D), mk(B, s_q, g.h_q, D), block_ids=per_token_ids, k_scale=one, v_scale=one
            )


def test_the_envelope_names_the_pool_dtypes():
    """``decode.kv_cache_dtypes``: the activation dtype always; e4m3 on a QsaSpec geometry with bf16 activations -- caller lists AND the
    in-block indexer (since 2026-10-21); never on a dense geometry or under f16."""
    g, _ = _geom(GEOM_SMALL)
    ix = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True, qsa=QsaSpec(index_source="indexer", index_band=True))
    dense = GatedAttentionBlockGeometry(**GEOM_SMALL, causal_bottom_right=True)
    for cc in ((10, 7), (10, 0)):
        env = GatedAttentionBlockFwd.serving_envelope(g, BF16, _DEV, device_cc=cc)
        assert env.decode.kv_cache_dtypes == (BF16, E4M3) and env.decode.sparse_core
        assert GatedAttentionBlockFwd.serving_envelope(ix, BF16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (BF16, E4M3)
        assert GatedAttentionBlockFwd.serving_envelope(g, F16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (F16,)
        assert GatedAttentionBlockFwd.serving_envelope(dense, BF16, _DEV, device_cc=cc).decode.kv_cache_dtypes == (BF16,)
    assert any("kv_cache_dtype" in r and "in-kernel" in r.lower() for r in env.decode.requires)


# ---------------------------------------------------------------------------
# Rubin: the oracle on the dequantized pool
# ---------------------------------------------------------------------------


def _rope_tables(n_pos, dtype, ref_geom):
    cos, sin = build_rope_tables(n_pos, ROPE, base=ref_geom.rope_base, batch=1, device=_DEV, dtype=dtype)
    return cos, sin  # [1, n_pos, ROPE]


def _dequantized_ref(ref, k_scale, v_scale):
    """The dense reference with its K / V replaced by the dequantized e4m3 codes (what the kernel sees after the fold at a power-of-two
    scale, and up to the fold's fp32 roundings otherwise): ``bf16(fp32(sat_e4m3(row x (1 / scale))) x scale)``."""
    return types.SimpleNamespace(q=ref.q, gate=ref.gate, k=_dequantize(_quantize(ref.k, k_scale), k_scale), v=_dequantize(_quantize(ref.v, v_scale), v_scale))


class _Fp8Case:
    """One e4m3-cache sparse DECODE step over a batch of sequences with lengths ``kv_lens`` (0 = a dead entry; a length below ``s_q``
    leaves the rows with a negative position dead): the whole-sequence bf16 reference per entry, per-tensor DEscales calibrated on
    the reference K / V of every live entry (or the given constants: the power-of-two pin), the NaN-filled e4m3 pools holding the
    torch-quantized PREFIX rows, the block table (page 0 = the NULL block), the slot mapping of the step's tokens, ONE list per sequence
    at the step-0 position (the shared form) or one per row, and two fp32 oracles of the step's gated O, LSE and ``out``: on the
    DEQUANTIZED rows (the gate) and on the pre-quantization bf16 rows (reported)."""

    def __init__(
        self,
        geom_kw,
        s_q,
        kv_lens,
        *,
        page=16,
        layout="hnd",
        seed=0,
        top_k=512,
        block_lens=True,
        fuse_gate=False,
        index_band=False,
        n_pages=None,
        index_list_form=None,
        split=None,
        return_lse=False,
        fuse_norm_rope=False,
        inplace_qkv=None,
        fuse_prepare=None,
        scales=None,
    ):
        self.s_q, self.kv_lens, self.page, self.layout, self.top_k = int(s_q), [int(x) for x in kv_lens], page, layout, top_k
        self.block_lens_given, self.fuse_gate, self.index_band = bool(block_lens), bool(fuse_gate), bool(index_band)
        self.return_lse, self.split, self.fuse_norm_rope, self.inplace_qkv, self.fuse_prepare = (
            bool(return_lse),
            split,
            bool(fuse_norm_rope),
            inplace_qkv,
            fuse_prepare,
        )
        self.index_list_form = index_list_form
        self.shared = (self.s_q <= 4) if index_list_form is None else (index_list_form == "shared")
        self.geom, self.ref_geom = _geom(geom_kw, top_k=top_k, index_band=index_band, index_list_form=index_list_form)
        geom, ref_geom, dtype = self.geom, self.ref_geom, BF16
        self.dtype = dtype
        h_q, h_kv, d_model = geom.h_q, geom.h_kv, geom.d_model
        G = h_q // h_kv
        B = len(self.kv_lens)
        g = torch.Generator(device=_DEV).manual_seed(seed)
        self.gen = g

        def randn(*shape, std=0.02):
            return (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(dtype)

        n_dense = sum(geom.qkvg_block_widths[:4])
        w_qkvg = randn(geom.n_qkvg, d_model)
        self.w = dict(
            w_qkvg=w_qkvg, w_q_norm=torch.ones(D, dtype=dtype, device=_DEV), w_k_norm=torch.ones(D, dtype=dtype, device=_DEV), w_o=randn(d_model, h_q * D)
        )
        L_max = max(max(self.kv_lens), s_q)
        cos_t, sin_t = _rope_tables(L_max, dtype, ref_geom)
        scale = geom.scale
        self.h = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.cos = cos_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.sin = sin_t[0, :1].expand(s_q, ROPE).clone()[None].expand(B, s_q, ROPE).clone()
        self.live_rows = torch.zeros(B, s_q, dtype=torch.bool, device=_DEV)
        refs, ids_rows, lens_rows = {}, [], []
        for b, L in enumerate(self.kv_lens):
            pos = torch.arange(s_q, device=_DEV) + (L - s_q)
            live = pos >= 0
            self.live_rows[b] = live
            pos0 = torch.tensor([L - s_q], device=_DEV)
            if self.shared:
                ids_b, lens_b = random_block_ids(pos0, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV), shuffle=True)
            else:
                ids_b, lens_b = random_block_ids(pos, top_k, BS, generator=g, kv_lens=torch.tensor([L], device=_DEV).expand(s_q), shuffle=True)
            ids_rows.append(ids_b)
            lens_rows.append(lens_b)
            if L == 0:
                continue  # a dead entry: no sequence, no reference, block 0, slots -1
            h_full = randn(1, L, d_model, std=1.0)
            ref = gated_attention_block_reference(
                h_full, w_qkvg[:n_dense], self.w["w_q_norm"], self.w["w_k_norm"], cos_t[:, :L], sin_t[:, :L], self.w["w_o"], ref_geom
            )
            refs[b] = (ref, pos, live, ids_b, lens_b)
            p_live = pos[live]
            self.h[b, live] = h_full[0, p_live]
            self.cos[b, live] = cos_t[0, p_live]
            self.sin[b, live] = sin_t[0, p_live]
        # the per-tensor DEscales: calibrated on every live entry's whole-sequence K / V (a static serving calibration), or the constants given
        if scales is None:
            all_k = torch.cat([r[0].k[0].reshape(-1) for r in refs.values()]) if refs else torch.ones(1, device=_DEV)
            all_v = torch.cat([r[0].v[0].reshape(-1) for r in refs.values()]) if refs else torch.ones(1, device=_DEV)
            self.k_scale, self.v_scale = _scale_of(all_k), _scale_of(all_v)
        else:
            self.k_scale, self.v_scale = _scale_const(scales[0]), _scale_const(scales[1])
        # the two oracles per entry: on the DEQUANTIZED rows (prefix AND step rows through the e4m3 codes) and on the bf16 rows
        self.ref_out = torch.zeros(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.ref_o_gated = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_o_gated_bf16 = torch.zeros(B, s_q, h_q, D, dtype=dtype, device=_DEV)
        self.ref_lse = torch.full((B, h_q, s_q), float("-inf"), dtype=torch.float32, device=_DEV)
        self.ref_codes = {}  # b -> (K codes [L, h_kv, D], V codes): the prefix rows go into the pools; the step rows pin the write-through
        for b, (ref, pos, live, ids_b, lens_b) in refs.items():
            L = self.kv_lens[b]
            k_codes, v_codes = _quantize(ref.k[0], self.k_scale), _quantize(ref.v[0], self.v_scale)
            self.ref_codes[b] = (k_codes, v_codes)
            kd, vd = _dequantize(k_codes, self.k_scale), _dequantize(v_codes, self.v_scale)
            p_live = pos[live]
            rows = ids_b.expand(s_q, top_k) if self.shared else ids_b
            lens_arg = (lens_b.expand(s_q) if self.shared else lens_b) if block_lens else None
            allowed = qsa_visible_mask(rows, lens_arg, pos, L, L, BS, top_k=top_k, pos0=(L - s_q) if self.shared else None)[live]
            q_rows = ref.q[0][p_live].float()
            gate_rows = ref.gate[0][p_live].float()
            for which, (kk_all, vv_all, o_dst) in (("deq", (kd, vd, self.ref_o_gated)), ("bf16", (ref.k[0], ref.v[0], self.ref_o_gated_bf16))):
                o = torch.zeros(int(live.sum()), h_q, D, dtype=torch.float32, device=_DEV)
                lse = torch.full((h_q, int(live.sum())), float("-inf"), dtype=torch.float32, device=_DEV)
                for hh in range(h_q):
                    kk, vv = kk_all[:, hh // G].float(), vv_all[:, hh // G].float()
                    sc = (q_rows[:, hh] @ kk.t()) * scale
                    sc = sc.masked_fill(~allowed, float("-inf"))
                    rmax = sc.amax(dim=-1)
                    dead = torch.isinf(rmax) & (rmax < 0)
                    safe = torch.where(dead, torch.zeros_like(rmax), rmax)
                    p = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
                    den = p.sum(dim=-1)
                    den_safe = torch.where(dead, torch.ones_like(den), den)
                    oc = (p @ vv) / den_safe[:, None]
                    o[:, hh] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
                    lse[hh] = torch.where(dead, torch.full_like(den, float("-inf")), safe + torch.log(den))
                o16 = o.to(dtype)  # the oracle's single rounding of O, then the gate in fp32, then one rounding
                og = (o16.float() * torch.sigmoid(gate_rows)).to(dtype)
                o_dst[b, live] = og
                if which == "deq":
                    self.ref_out[b, live] = (og.reshape(-1, h_q * D).float() @ self.w["w_o"].float().t()).to(dtype)
                    self.ref_lse[b, :, live] = lse
        self.block_ids = torch.stack([x.reshape(top_k) if self.shared else x.reshape(s_q, top_k) for x in ids_rows]).contiguous()
        self.block_lens = torch.stack([x.reshape(()) if self.shared else x.reshape(s_q) for x in lens_rows]).contiguous()
        assert self.block_ids.dtype == torch.int32 and self.block_lens.dtype == torch.int32
        # pages: block 0 reserved (NULL); each live entry gets ceil(L / P) pages from a permutation of 1 .. n_pages - 1; -1 past them
        need = sum(-(-L // page) for L in self.kv_lens)
        self.n_pages = n_pages if n_pages is not None else need + 1 + 2
        assert self.n_pages >= need + 1, (self.n_pages, need)
        perm = (torch.randperm(self.n_pages - 1, generator=g, device=_DEV) + 1).tolist()
        self.max_pages = max(-(-L_max // page), 1)
        self.block_table = torch.full((B, self.max_pages), -1, dtype=torch.int32, device=_DEV)
        it = iter(perm)
        slots = torch.full((B * s_q,), -1, dtype=torch.int32, device=_DEV)
        self.k_cache, self.v_cache = _pools(self.n_pages, h_kv, page, layout, E4M3)
        self.index_cache = torch.full((self.n_pages, page, geom.qsa.index_head_dim), float("nan"), dtype=dtype, device=_DEV) if index_band else None
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                self.block_table[b] = 0  # the dead entry: the NULL block, slots -1
                continue
            pages = [next(it) for _ in range(-(-L // page))]
            self.block_table[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device=_DEV)
            k_codes, v_codes = self.ref_codes[b]
            for p in range(max(L - s_q, 0)):  # the PREFIX: the torch-quantized codes into the pools
                pg, off = pages[p // page], p % page
                self.k_cache[pg, :, off] = k_codes[p]
                self.v_cache[pg, :, off] = v_codes[p]
            for j in range(s_q):  # the step's live tokens: written through (and quantized) by the block
                p = L - s_q + j
                if p >= 0:
                    slots[b * s_q + j] = pages[p // page] * page + p % page
        self.slot_mapping = slots
        self.kv_lens_t = torch.tensor(self.kv_lens, dtype=torch.int32, device=_DEV)
        self.prefix_bits = (_bits(self.k_cache).clone(), _bits(self.v_cache).clone())
        self.blk_kw = dict(
            paged_kv_page_size=page,
            kv_source=KV_SOURCE_PAGED,
            sample_k_cache=self.k_cache,
            sample_v_cache=self.v_cache,
            sample_block_table=self.block_table,
            fuse_gate=fuse_gate,
            kv_cache_dtype=E4M3,
            return_lse=self.return_lse,
            decode_split_kv=split,
            fuse_norm_rope=self.fuse_norm_rope,
        )
        if inplace_qkv is not None:
            self.blk_kw["inplace_qkv"] = inplace_qkv
        if fuse_prepare is not None:
            self.blk_kw["fuse_prepare"] = fuse_prepare
        self.out = torch.empty(B, s_q, d_model, dtype=dtype, device=_DEV)
        self.lse = torch.empty(B, h_q, s_q, dtype=torch.float32, device=_DEV) if self.return_lse else None
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], self.cos, self.sin, self.w["w_o"], self.out, self.geom, **self.blk_kw
        )
        assert isinstance(self.blk._sdpa, _SparseSdpa) and self.blk._sdpa.kv_e4m3 and self.blk.sparse_decode and self.blk.qsa_shared_list is self.shared
        self.ws = None

    def compile(self):
        assert self.blk.check_support()
        self.blk.compile()
        self.ws = torch.full((self.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        return self

    def execute(
        self,
        *,
        out=None,
        ws=None,
        lse=None,
        h=None,
        cos=None,
        sin=None,
        kv_lens=None,
        block_table=None,
        slot_mapping=None,
        ids=None,
        lens=None,
        k_scale=None,
        v_scale=None,
        stream=None,
    ):
        out = self.out if out is None else out
        self.blk.execute(
            self.h if h is None else h,
            self.w["w_qkvg"],
            self.w["w_q_norm"],
            self.w["w_k_norm"],
            self.cos if cos is None else cos,
            self.sin if sin is None else sin,
            self.w["w_o"],
            out,
            self.ws if ws is None else ws,
            lse=(self.lse if lse is None else lse) if self.return_lse else None,
            block_ids=self.block_ids if ids is None else ids,
            block_lens=(self.block_lens if lens is None else lens) if self.block_lens_given else None,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_table=self.block_table if block_table is None else block_table,
            kv_lens=self.kv_lens_t if kv_lens is None else kv_lens,
            slot_mapping=self.slot_mapping if slot_mapping is None else slot_mapping,
            index_k_raw=self.index_cache,
            current_stream=stream,
            k_scale=self.k_scale if k_scale is None else k_scale,
            v_scale=self.v_scale if v_scale is None else v_scale,
        )
        return out

    def run(self):
        if self.ws is None:
            self.compile()
        self.out.fill_(_sentinel(self.dtype))
        if self.lse is not None:
            self.lse.fill_(_sentinel(self.dtype))
        self.execute()
        torch.cuda.synchronize()
        return self.out

    def views(self, ws=None):
        """The post-norm-RoPE Q and K (the slab bands in place, the compact slots out of place), the slab's GATE and V bands, the O slot, the
        slab -- what the e4m3 mode reads and writes."""
        ws = self.ws if ws is None else ws
        g, lay = self.geom, self.blk._ws
        B, S = self.blk.batch, self.blk.seq_len
        t = B * S
        proj = _view(ws, lay.proj, (t, g.n_qkvg), self.dtype)
        o_q, o_g, o_k, o_v = g.qkvg_offsets[:4]
        q = (_cols(proj, o_q, g.h_q, g.d_head) if lay.q < 0 else _view(ws, lay.q, (t, g.h_q, g.d_head), self.dtype)).view(B, S, g.h_q, g.d_head)
        k = (_cols(proj, o_k, g.h_kv, g.d_head) if lay.k < 0 else _view(ws, lay.k, (t, g.h_kv, g.d_head), self.dtype)).view(B, S, g.h_kv, g.d_head)
        gate = _cols(proj, o_g, g.h_q, g.d_head).view(B, S, g.h_q, g.d_head)
        v = _cols(proj, o_v, g.h_kv, g.d_head).view(B, S, g.h_kv, g.d_head)
        o = _view(ws, lay.o, (t, g.h_q, g.d_head), self.dtype).view(B, S, g.h_q, g.d_head)
        return q, gate, k, v, o, proj

    def label(self):
        form = f"{'shared' if self.shared else 'per-row'}{'' if self.index_list_form is None else ' (declared)'}"
        return (
            f"[{self.geom.h_q}/{self.geom.h_kv} S_q={self.s_q} {form} e4m3 p{self.page} {self.layout} split={self.blk.decode_split_kv} "
            f"gate={'fused' if self.blk.gate_fused else 'launch'} lse={self.return_lse} fuse_prepare={self.blk.fuse_prepare} lens={self.kv_lens}]"
        )

    def check(self, out=None, ws=None, lse=None, *, what="out"):
        """The standard assertions: finite, no sentinel, dead rows exact, within the budget vs the oracle on the DEQUANTIZED pool (out, the
        gated O slot, the LSE), the pools' step rows BITWISE torch's quantization of the block's own K rows and V band, the prefix rows and
        the NULL block untouched; the distance to the bf16 oracle printed (no gate)."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        lse = self.lse if lse is None else lse
        sent = _sentinel(self.dtype)
        live = self.live_rows
        dead = ~live
        q, gate, k, v, o, proj = self.views(ws)
        assert torch.isfinite(out.float()).all(), f"{what}: non-finite out"
        assert not (out.float() == sent).any(), f"{what}: a sentinel survived in out"
        assert torch.isfinite(o.float()).all(), f"{what}: non-finite gated O (a row the sparse core never wrote keeps the workspace fill)"
        if dead.any():
            assert (out[dead] == 0).all() and (o[dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[live], self.ref_out[live])
        d_out = float((out[live].float() - self.ref_out[live].float()).abs().max())
        d_o = float((o[live].float() - self.ref_o_gated[live].float()).abs().max())
        d_o_bf16 = float((o[live].float() - self.ref_o_gated_bf16[live].float()).abs().max())
        d_lse = 0.0
        if self.return_lse:
            assert lse is not None and not (lse == sent).any(), f"{what}: a sentinel survived in LSE"
            dead_bhs = dead[:, None, :].expand_as(lse)
            if dead.any():
                assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
            d_lse = float((lse[~dead_bhs] - self.ref_lse[~dead_bhs]).abs().max()) if live.any() else 0.0
            assert d_lse <= ATOL, f"{what}: LSE off the oracle on the dequantized pool by {d_lse} (budget {ATOL})"
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        assert d_o <= ATOL, f"{what}: gated O off the oracle on the dequantized pool by {d_o} (budget {ATOL})"
        # the quantizing write: the step's live rows are torch's e4m3 codes of the block's own normed K rows / V band, bitwise
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        kf, vf = k.reshape(-1, self.geom.h_kv, D), v.reshape(-1, self.geom.h_kv, D)
        assert torch.equal(
            _bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(kf[lv], self.k_scale))
        ), f"{what}: the K pool rows are not the quantized K rows"
        assert torch.equal(
            _bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(vf[lv], self.v_scale))
        ), f"{what}: the V pool rows are not the quantized V rows"
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            now = _bits(pool)
            touched = (now != before).any(dim=-1).any(dim=1)  # [n_pages, page] rows whose bits moved
            rows_moved = touched.nonzero()
            written = set((int(p) for p in (s_ // self.page * self.page + s_ % self.page).tolist()))
            assert all(int(pg) * self.page + int(off) in written for pg, off in rows_moved), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        if self.index_cache is not None:
            raw = index_k_raw_view(proj, self.geom, self.blk.batch, self.blk.seq_len)  # [B, S, 1, index_head_dim]
            assert torch.equal(_bits(self.index_cache[s_ // self.page, s_ % self.page, :]), _bits(raw.reshape(-1, raw.shape[-1])[lv]))
            assert (~torch.isnan(self.index_cache.float())).all(-1).sum().item() == int(lv.sum())
        print(
            f"\n{self.label()} {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| vs the DEQUANTIZED oracle {d_o:.4e} (budget {ATOL}) max|d LSE| {d_lse:.2e}; "
            f"vs the PRE-quantization bf16 oracle {d_o_bf16:.4e} (reported, no gate); dead rows {int(dead.sum())}; k_scale {float(self.k_scale):.4e} v_scale {float(self.v_scale):.4e}"
        )
        return dict(cos=c, d_out=d_out, d_o=d_o, d_lse=d_lse, d_o_bf16=d_o_bf16, dead=int(dead.sum()))


def _two_launches(case: _Fp8Case, what: str):
    a = case.run().clone()
    o_a = case.views()[4].clone()
    lse_a = None if case.lse is None else case.lse.clone()
    b = case.run().clone()
    o_b = case.views()[4].clone()
    assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(o_a), _bits(o_b)), f"{what}: two launches are not bitwise"
    if lse_a is not None:
        assert torch.equal(lse_a, case.lse), f"{what}: two launches are not bitwise on LSE"
    return case.check(what=what)


# (geometry, S_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed, index_list_form, split, return_lse, fuse_norm_rope, inplace_qkv, fuse_prepare)
_LADDER = [
    pytest.param(
        GEOM_SMALL, 1, (2052, 8192, 0, 4097), 16, "hnd", False, True, False, 1, None, None, False, False, None, None, id="Sq1-B4-p16-hnd-dead-entry-gate-launch"
    ),
    pytest.param(
        GEOM_SMALL,
        4,
        (300, 0, 1000, 77),
        64,
        "nhd",
        True,
        False,
        False,
        2,
        None,
        1,
        False,
        False,
        None,
        None,
        id="Sq4-B4-p64-nhd-no-block-lens-fuse-gate-epilogue",
    ),
    pytest.param(
        GEOM_SMALL,
        3,
        (2, 20000, 7),
        32,
        "unified",
        True,
        True,
        False,
        3,
        None,
        None,
        False,
        False,
        None,
        None,
        id="Sq3-B3-p32-unified-shorter-than-the-step-20000",
    ),  # a 2-token sequence under a 3-row step: one row at a negative position (a 1-token sequence would hand the fp32 reference a K = 1 matmul, which torch 2.10 routes to a Triton kernel this node cannot build for sm_107a)
    pytest.param(GEOM_FLASH_NEXT, 1, (4096,), 16, "hnd", True, True, False, 4, None, None, False, False, None, None, id="Sq1-B1-flash-next-24-2-p16"),
    pytest.param(
        GEOM_SMALL, 3, (8192, 2052), 16, "hnd", False, True, True, 5, None, None, False, False, None, None, id="Sq3-B2-p16-index-band-raw-key-write-through"
    ),
    # the limits of the retired gathered-cast form, SERVED through the in-kernel arm (2026-10-21)
    pytest.param(
        GEOM_SMALL,
        4,
        (300, 0, 2052, 77),
        16,
        "hnd",
        True,
        True,
        False,
        6,
        "per_row",
        4,
        True,
        False,
        None,
        None,
        id="Sq4-B4-per-row-declared-split4-lse-gate-combine",
    ),
    pytest.param(
        GEOM_SMALL, 8, (300, 100), 16, "hnd", False, True, False, 7, None, 1, True, False, None, None, id="Sq8-B2-per-token-extend-with-prefix-unsplit-lse"
    ),
    pytest.param(
        GEOM_SMALL,
        4,
        (8191, 8190, 8189, 8188),
        16,
        "hnd",
        True,
        True,
        False,
        8,
        None,
        16,
        True,
        True,
        None,
        None,
        id="Sq4-B4-every-residue-split16-lse-fuse-norm-rope",
    ),
    pytest.param(
        GEOM_SMALL,
        2,
        (8192, 2052, 0, 77),
        64,
        "nhd",
        True,
        True,
        False,
        9,
        None,
        None,
        False,
        False,
        True,
        True,
        id="Sq2-B4-p64-nhd-inplace-explicit-fuse-prepare",
    ),
    pytest.param(
        GEOM_SMALL, 2, (4097, 300), 16, "hnd", False, False, False, 10, None, 2, True, False, False, None, id="Sq2-B2-out-of-place-split2-lse-no-block-lens"
    ),
]


@requires_rubin
@pytest.mark.parametrize(
    "geom_kw, s_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed, index_list_form, split, return_lse, fuse_norm_rope, inplace_qkv, fuse_prepare",
    _LADDER,
)
def test_e4m3_decode_matches_the_oracle_on_the_dequantized_pool(
    geom_kw, s_q, kv_lens, page, layout, fuse_gate, block_lens, index_band, seed, index_list_form, split, return_lse, fuse_norm_rope, inplace_qkv, fuse_prepare
):
    """The e4m3 decode ladder (module docstring): every cell within the budget vs the step-0-anchored (shared) or per-row oracle on the
    DEQUANTIZED pool, the LSE too, dead rows exact, the write rows the torch cast bitwise, two launches bitwise; the bf16-oracle distance
    reported.  The sparse core's in-kernel arm serves every row (the kernel template is the sparse one, never the decode tile)."""
    case = _Fp8Case(
        geom_kw,
        s_q,
        kv_lens,
        page=page,
        layout=layout,
        seed=seed,
        fuse_gate=fuse_gate,
        block_lens=block_lens,
        index_band=index_band,
        index_list_form=index_list_form,
        split=split,
        return_lse=return_lse,
        fuse_norm_rope=fuse_norm_rope,
        inplace_qkv=inplace_qkv,
        fuse_prepare=fuse_prepare,
    )
    case.compile()
    tp = case.blk._sdpa._impl.template_params()
    assert tp.kv_dtype == 0 and tp.paged_kv and tp.bottom_right and case.blk._sdpa._impl.kv_e4m3
    if split is not None:
        assert case.blk.decode_split_kv == split
    if fuse_prepare:
        assert case.blk._prepare is not None and case.blk._prepare.kv_e4m3 and case.blk._cache_write is None
    _two_launches(case, "ladder")


@requires_rubin
@pytest.mark.parametrize("r", [1, 3], ids=lambda r: f"pos0-mod4-{r}")
def test_mtp_rows_over_e4m3_pools_share_the_step0_list(r):
    """The speculative rows of a SHORT sequence at a residue of the step-0 position where a block completes after step 0 (``r = 1``)
    and where none does (``r = 3``): the e4m3 mode's rows within the budget of the step-0-anchored oracle on the dequantized pool."""
    s_q = 4
    L = 56 + r + s_q
    case = _Fp8Case(GEOM_SMALL, s_q, (L, L + 4), page=16, seed=10 + r, fuse_gate=True)
    case.compile()
    _two_launches(case, f"mtp r={r}")


# (S_q, index_list_form, split, return_lse, fuse_gate, scales)
_POW2 = [
    pytest.param(1, None, 4, True, True, (1.0, 1.0), id="Sq1-shared-split4-lse-gate-combine-scales-1-1"),
    pytest.param(1, None, 4, True, True, (0.5, 2.0), id="Sq1-shared-split4-lse-gate-combine-scales-0.5-2"),
    pytest.param(4, "per_row", 1, True, False, (0.5, 2.0), id="Sq4-per-row-unsplit-lse-gate-launch-scales-0.5-2"),
    pytest.param(2, None, 1, False, True, (2.0, 0.5), id="Sq2-shared-unsplit-gate-epilogue-scales-2-0.5"),
]


@requires_rubin
@pytest.mark.parametrize("s_q, index_list_form, split, return_lse, fuse_gate, scales", _POW2)
def test_e4m3_at_a_power_of_two_scale_is_bitwise_the_bf16_block_fed_the_dequantized_pool(s_q, index_list_form, split, return_lse, fuse_gate, scales):
    """The fold's exactness lifted from the kernel to the block: at ``k_scale`` / ``v_scale`` powers of two the e4m3 block's ``out``, gated O
    and LSE are BITWISE the bf16 block's over pools holding ``bf16(code x scale)`` -- EVERY row the e4m3 run read: the torch-quantized
    prefix and the step rows the e4m3 block itself wrote (copied dequantized into the bf16 twin's pools, which then writes nothing:
    ``slot_mapping`` all ``-1``).  The gate in the split combine, in the epilogue and as its own launch; the shared and the per-row list."""
    case = _Fp8Case(
        GEOM_SMALL,
        s_q,
        (2052, 300, 4097),
        page=16,
        seed=20 + s_q,
        scales=scales,
        split=split,
        return_lse=return_lse,
        fuse_gate=fuse_gate,
        index_list_form=index_list_form,
    )
    case.compile()
    out8 = case.run().clone()
    o8 = case.views()[4].clone()
    lse8 = None if case.lse is None else case.lse.clone()
    case.check(what="e4m3 at a power-of-two scale")
    geom, h_kv = case.geom, case.geom.h_kv
    k16, v16 = _pools(case.n_pages, h_kv, case.page, case.layout, BF16)
    k16.copy_(_dequantize(case.k_cache, case.k_scale))  # exact: a code times a power of two is a bf16 value; NaN rows stay NaN
    v16.copy_(_dequantize(case.v_cache, case.v_scale))
    out16 = torch.empty_like(case.out)
    lse16 = torch.empty_like(case.lse) if case.lse is not None else None
    twin = GatedAttentionBlockFwd(
        case.h,
        case.w["w_qkvg"],
        case.w["w_q_norm"],
        case.w["w_k_norm"],
        case.cos,
        case.sin,
        case.w["w_o"],
        out16,
        geom,
        paged_kv_page_size=case.page,
        kv_source=KV_SOURCE_PAGED,
        sample_k_cache=k16,
        sample_v_cache=v16,
        sample_block_table=case.block_table,
        fuse_gate=fuse_gate,
        return_lse=return_lse,
        decode_split_kv=split,
    )
    assert not twin.kv_cache_e4m3 and not twin._sdpa.kv_e4m3 and twin.decode_split_kv == case.blk.decode_split_kv and twin.gate_fused is case.blk.gate_fused
    assert twin.check_support()
    twin.compile()
    ws16 = torch.full((twin.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    out16.fill_(_sentinel(BF16))
    twin.execute(
        case.h,
        case.w["w_qkvg"],
        case.w["w_q_norm"],
        case.w["w_k_norm"],
        case.cos,
        case.sin,
        case.w["w_o"],
        out16,
        ws16,
        lse=lse16,
        block_ids=case.block_ids,
        block_lens=case.block_lens,
        k_cache=k16,
        v_cache=v16,
        block_table=case.block_table,
        kv_lens=case.kv_lens_t,
        slot_mapping=torch.full_like(case.slot_mapping, -1),  # the twin writes nothing: its pools already hold the dequantized step rows
    )
    torch.cuda.synchronize()
    lay16 = twin._ws
    o16 = _view(ws16, lay16.o, (case.blk.batch * s_q, geom.h_q, D), BF16).view_as(o8)
    assert torch.equal(_bits(out16), _bits(out8)), "out: the e4m3 block differs from the bf16 block fed the dequantized pool at a power-of-two scale"
    assert torch.equal(_bits(o16), _bits(o8)), "the gated O differs"
    if lse8 is not None:
        assert torch.equal(lse16, lse8), "the LSE differs"
    print(f"\n{case.label()}: out / O / LSE BITWISE the bf16 block over the dequantized pool at k_scale {scales[0]} v_scale {scales[1]}")


@requires_rubin
def test_get_workspace_size_is_honest_and_holds_no_scratch():
    """The e4m3 staging rows, the inverse scales and (under a split) the sparse core's fp32 partials sit inside ``get_workspace_size()``;
    nothing past the size is written: a sentinel tail survives an execute; there is no gathered-cast scratch any more (the carve of an
    e4m3 declaration is the bf16 twin's plus the three e4m3 slots)."""
    case = _Fp8Case(GEOM_SMALL, 2, (4096, 512, 0), page=16, seed=31, fuse_gate=True, split=4, return_lse=True)
    case.compile()
    blk = case.blk
    lay = blk._layout()
    assert blk._sdpa.scratch_workspace_bytes() > 0 and blk.get_workspace_size() >= lay.total_bytes + blk._sdpa.scratch_workspace_bytes()
    assert not hasattr(lay, "kv_gathered") and not hasattr(lay, "kv_len_t") and lay.kv_inv > lay.vc8 > lay.kc8 >= 0
    twin, _ = _declare(GEOM_SMALL, 3, 2, pool_dtype=BF16, kv_cache_dtype=None, fuse_gate=True, decode_split_kv=4, return_lse=True)
    assert lay.total_bytes - twin._layout().total_bytes == lay.total_bytes - lay.kc8  # the three e4m3 slots are the whole difference of the carves
    need = blk.get_workspace_size()
    ws = torch.empty(need + 65536, dtype=torch.uint8, device=_DEV)
    ws[need:].fill_(0xA5)
    ws[:need].fill_(0x7F)
    out = torch.empty_like(case.out)
    lse = torch.empty_like(case.lse)
    case.execute(out=out, ws=ws, lse=lse)
    torch.cuda.synchronize()
    assert (ws[need:] == 0xA5).all(), "bytes past get_workspace_size() were written"
    case.check(out, ws, lse, what="sentinel-tailed workspace")
    inv = _view(ws, lay.kv_inv, (2,), torch.float32)
    assert torch.equal(inv, torch.reciprocal(torch.cat([case.k_scale, case.v_scale])))


@requires_rubin
def test_the_quantizing_write_on_the_prefill_form_matches_the_torch_cast_in_both_write_forms_and_leaves_out_unchanged():
    """The write-only form (the sparse block over its OWN K / V with e4m3 pools, ``kv_source="self"``), in BOTH write forms -- the two-stage
    write-through and the fused prepare's e4m3 pool arm (``fuse_prepare=True``): the pools' rows are torch's e4m3 codes of the block's
    post-RoPE K / V BITWISE (and equal to each other), ``out`` is BITWISE the bf16-pool twin's (the write never changes attention);
    padded slots (-1) write nothing, every unwritten slot still NaN."""
    B, S, page = 2, 64, 16
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL, qsa=QsaSpec(top_k=512))
    g = torch.Generator(device=_DEV).manual_seed(77)
    randn = lambda *shape, std=0.02: (torch.randn(*shape, generator=g, device=_DEV, dtype=torch.float32) * std).to(BF16)  # noqa: E731
    cos, sin = build_rope_tables(S, ROPE, base=10000.0, batch=B, device=_DEV, dtype=BF16)
    h, w_qkvg, w_o = randn(B, S, geom.d_model, std=1.0), randn(geom.n_qkvg, geom.d_model), randn(geom.d_model, geom.h_q * D)
    ones = torch.ones(D, dtype=BF16, device=_DEV)
    ids_b, lens_b = full_block_ids(torch.arange(S, device=_DEV), 512, BS)
    ids = ids_b[None].expand(B, S, 512).contiguous()
    lens = lens_b[None].expand(B, S).contiguous()
    slots = torch.arange(B * S, dtype=torch.int32, device=_DEV)
    slots[::9] = -1  # a padded token every ninth row writes nothing
    n_pages = -(-(B * S) // page) + 2
    k_scale, v_scale = torch.tensor([0.0123], device=_DEV), torch.tensor([0.0456], device=_DEV)
    runs = {}
    for tag, pool_dtype, attr, scales, fused in (
        ("e4m3", E4M3, E4M3, dict(k_scale=k_scale, v_scale=v_scale), False),
        ("e4m3-prepare", E4M3, E4M3, dict(k_scale=k_scale, v_scale=v_scale), True),
        ("bf16", BF16, None, {}, False),
    ):
        kc, vc = _pools(n_pages, geom.h_kv, page, "hnd", pool_dtype)
        out = torch.empty(B, S, geom.d_model, dtype=BF16, device=_DEV)
        blk = GatedAttentionBlockFwd(h, w_qkvg, ones, ones, cos, sin, w_o, out, geom, paged_kv_page_size=page, kv_cache_dtype=attr, fuse_prepare=fused)
        assert blk.check_support()
        blk.compile()
        ws = torch.full((blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
        blk.execute(h, w_qkvg, ones, ones, cos, sin, w_o, out, ws, block_ids=ids, block_lens=lens, k_cache=kc, v_cache=vc, slot_mapping=slots, **scales)
        torch.cuda.synchronize()
        lay = blk._ws
        proj = _view(ws, lay.proj, (B * S, geom.n_qkvg), BF16)
        _o_q, _o_g, o_k, o_v = geom.qkvg_offsets[:4]
        runs[tag] = (out.clone(), kc, vc, _cols(proj, o_k, geom.h_kv, D).clone(), _cols(proj, o_v, geom.h_kv, D).clone(), blk)
    out8, kc8, vc8, k_rows, v_rows, blk8 = runs["e4m3"]
    out8p, kc8p, vc8p, k_rows_p, v_rows_p, blk8p = runs["e4m3-prepare"]
    out16, kc16, vc16, k_rows16, v_rows16, _ = runs["bf16"]
    assert isinstance(blk8._cache_write, _QuantizedCacheWrite) and blk8.inplace_qkv is True and not blk8._sdpa.kv_e4m3
    assert blk8p._prepare is not None and blk8p._prepare.kv_e4m3 and blk8p._cache_write is None and not blk8p._sdpa.kv_e4m3
    assert torch.equal(_bits(out8), _bits(out16)) and torch.equal(_bits(out8p), _bits(out16)), "the e4m3 write changed the block's own attention"
    assert torch.equal(_bits(k_rows), _bits(k_rows16)) and torch.equal(_bits(v_rows), _bits(v_rows16))
    assert torch.equal(_bits(k_rows_p), _bits(k_rows16)) and torch.equal(_bits(v_rows_p), _bits(v_rows16)), "the prepare's slab rows differ"
    lv = slots >= 0
    s_ = slots[lv].long()
    assert torch.equal(_bits(kc8[s_ // page, :, s_ % page, :]), _bits(_quantize(k_rows[lv], k_scale)))
    assert torch.equal(_bits(vc8[s_ // page, :, s_ % page, :]), _bits(_quantize(v_rows[lv], v_scale)))
    assert torch.equal(_bits(kc8p), _bits(kc8)) and torch.equal(_bits(vc8p), _bits(vc8)), "the prepare's e4m3 pools differ from the two-stage write-through's"
    assert torch.equal(_bits(kc16[s_ // page, :, s_ % page, :]), _bits(k_rows[lv]))
    for pool in (kc8, vc8, kc8p, vc8p):
        assert int((~torch.isnan(pool.float())).all(-1).sum()) == int(lv.sum()) * geom.h_kv, "an unwritten or padded slot moved"
    print(f"\nprefill write e4m3: {int(lv.sum())} of {B * S} tokens written, out bitwise the bf16-pool twin, pools bitwise the torch cast in both write forms")


def _written(pool):
    return int((~torch.isnan(pool.float())).all(-1).sum())


@requires_rubin
def test_capture_of_the_first_execute_replays_bitwise_and_takes_new_scales_lengths_lists_slots_and_inputs():
    """The serving stack's defaults: the FIRST execute of a fresh declaration captured on a NEW stream with no eager warm-up under
    ``set_sync_debug_mode("error")`` (no host sync in the path: scales, lengths, tables, lists and slots are device data; the inverse
    scales are a device op on the capture stream); the capture launches nothing, the replay is BITWISE a fresh eager run; new
    ``k_scale`` / ``v_scale`` / ``kv_lens`` / ``block_table`` / ``block_ids`` / ``block_lens`` / ``slot_mapping`` / ``h`` / RoPE rows
    through the captured tensors replay bitwise the eager run over them -- the scales are execute inputs read in-kernel, never a plan constant."""
    case = _Fp8Case(GEOM_SMALL, 2, (300, 0, 1000, 77), page=16, seed=41, fuse_gate=True, split=4, return_lse=True)  # fresh: never executed
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    k_before = _written(case.k_cache)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()  # the FIRST execute of this declaration, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all() and _written(case.k_cache) == k_before, "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed, lse_replayed = case.out.clone(), case.lse.clone()
        case.check(replayed, lse=lse_replayed, what="replay")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(
            case.lse, lse_replayed
        ), "the replay of the captured first execute differs from the eager run"
        # a NEW step through the captured tensors: new scales (a recalibration), new lengths / table / lists / slots / h / RoPE rows -- the
        # oracle of the new step through a sibling case over the same shapes
        sib = _Fp8Case(GEOM_SMALL, 2, (301, 0, 1000, 77), page=16, seed=43, fuse_gate=True, split=4, return_lse=True, n_pages=case.n_pages)
        assert sib.block_table.shape == case.block_table.shape and sib.k_cache.shape == case.k_cache.shape
        assert not torch.equal(sib.k_scale, case.k_scale)
        case.k_cache.copy_(sib.k_cache)
        case.v_cache.copy_(sib.v_cache)
        case.block_table.copy_(sib.block_table)
        case.kv_lens_t.copy_(sib.kv_lens_t)
        case.slot_mapping.copy_(sib.slot_mapping)
        case.block_ids.copy_(sib.block_ids)
        case.block_lens.copy_(sib.block_lens)
        case.k_scale.copy_(sib.k_scale)
        case.v_scale.copy_(sib.v_scale)
        case.h.copy_(sib.h)
        case.cos.copy_(sib.cos)
        case.sin.copy_(sib.sin)
        case.w["w_qkvg"].copy_(sib.w["w_qkvg"])
        case.w["w_o"].copy_(sib.w["w_o"])
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.ref_out, case.ref_o_gated, case.ref_o_gated_bf16, case.ref_lse, case.live_rows, case.kv_lens = (
            sib.ref_out,
            sib.ref_o_gated,
            sib.ref_o_gated_bf16,
            sib.ref_lse,
            sib.live_rows,
            sib.kv_lens,
        )
        case.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2, lse_replayed2 = case.out.clone(), case.lse.clone()
        case.check(replayed2, lse=lse_replayed2, what="replay over new scales / lengths / table / lists / slots / inputs")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(
            case.lse, lse_replayed2
        ), "the replay over new scales / lengths / tables / lists / slots differs from the eager run over them"
    finally:
        graph.reset()


# ---------------------------------------------------------------------------
# Rubin: the in-block indexer over e4m3 pools (the bf16 indexer case turned into its e4m3 twin)
# ---------------------------------------------------------------------------


class _IndexerE4m3Case(_qsa._IndexerCase):
    """The decode-INDEXER case of ``test_block_qsa_decode`` over e4m3 pools: the same inputs, references, side caches, table and slots as
    the bf16 case; the pools' prefix rows quantized by torch with per-tensor scales calibrated on the whole-sequence K / V, the oracle's
    K / V the DEQUANTIZED codes (the indexer's selection oracle is unchanged: it reads the compressed-key cache), the block declared with
    ``kv_cache_dtype=torch.float8_e4m3fn`` over the e4m3 samples; ``execute`` hands the scales over; ``check`` pins the pools' step
    rows as the torch cast."""

    def __init__(self, *args, scales=None, **kw):
        super().__init__(*args, **kw)
        h_kv = self.geom.h_kv
        if scales is None:
            all_k = torch.cat([r.k[0].reshape(-1) for r in self.refs.values()]) if self.refs else torch.ones(1, device=_DEV)
            all_v = torch.cat([r.v[0].reshape(-1) for r in self.refs.values()]) if self.refs else torch.ones(1, device=_DEV)
            self.k_scale, self.v_scale = _scale_of(all_k), _scale_of(all_v)
        else:
            self.k_scale, self.v_scale = _scale_const(scales[0]), _scale_const(scales[1])
        k8, v8 = _pools(self.n_pages, h_kv, self.page, self.layout, E4M3)
        k8.copy_(_quantize(self.k_cache, self.k_scale))  # the bf16 prefix rows -> codes; NaN rows -> the NaN code
        v8.copy_(_quantize(self.v_cache, self.v_scale))
        self.k_cache, self.v_cache = k8, v8
        self.prefix_bits = (_bits(k8).clone(), _bits(v8).clone())
        self.refs = {b: _dequantized_ref(r, self.k_scale, self.v_scale) for b, r in self.refs.items()}
        self.blk_kw = dict(self.blk_kw, sample_k_cache=k8, sample_v_cache=v8, kv_cache_dtype=E4M3)
        tables = (None, None) if self.rope_tables_only else (self.cos, self.sin)
        self.blk = GatedAttentionBlockFwd(
            self.h, self.w["w_qkvg"], self.w["w_q_norm"], self.w["w_k_norm"], *tables, self.w["w_o"], self.out, self.geom, **self.blk_kw
        )
        assert self.blk.kv_cache_e4m3 and self.blk._sdpa.kv_e4m3 and self.blk.indexer_decode and self.blk._indexer_decode is not None

    def execute(self, **kw):
        kw.setdefault("k_scale", self.k_scale)
        kw.setdefault("v_scale", self.v_scale)
        return super().execute(**kw)

    def check(self, out=None, ws=None, lse=None, *, what="out"):
        """``_Case.check`` with the pools' step rows pinned as torch's e4m3 codes of the slab's K / V rows."""
        out = self.out if out is None else out
        ws = self.ws if ws is None else ws
        lse = self.lse if lse is None else lse
        sent = _sentinel(self.dtype)
        live = self.live_rows
        dead = ~live
        q, gate, k, v, o, proj = self.slab_views(ws)
        assert torch.isfinite(out.float()).all() and not (out.float() == sent).any(), f"{what}: out non-finite or a sentinel survived"
        assert torch.isfinite(o.float()).all(), f"{what}: non-finite gated O"
        if dead.any():
            assert (out[dead] == 0).all() and (o[dead] == 0).all(), f"{what}: a dead row's out and O must be exactly 0"
        c = _cos(out[live], self.ref_out[live])
        d_out = float((out[live].float() - self.ref_out[live].float()).abs().max())
        d_o = float((o[live].float() - self.ref_o_gated[live].float()).abs().max())
        d_lse = 0.0
        if self.return_lse:
            assert lse is not None and not (lse == sent).any(), f"{what}: a sentinel survived in LSE"
            dead_bhs = dead[:, None, :].expand_as(lse)
            if dead.any():
                assert torch.isinf(lse[dead_bhs]).all() and (lse[dead_bhs] < 0).all(), f"{what}: a dead row's LSE must be -inf exactly"
            d_lse = float((lse[~dead_bhs] - self.ref_lse[~dead_bhs]).abs().max()) if live.any() else 0.0
            assert d_lse <= ATOL, f"{what}: LSE off the oracle on the dequantized pool by {d_lse} (budget {ATOL})"
        assert c > COS_OUT, f"{what}: cos(out) {c} <= {COS_OUT}"
        assert d_o <= ATOL, f"{what}: gated O off the oracle on the dequantized pool by {d_o} (budget {ATOL})"
        slot = self.slot_mapping
        lv = slot >= 0
        s_ = slot[lv].long()
        kf, vf = k.reshape(-1, self.geom.h_kv, D), v.reshape(-1, self.geom.h_kv, D)
        assert torch.equal(_bits(self.k_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(kf[lv], self.k_scale))), f"{what}: K pool rows"
        assert torch.equal(_bits(self.v_cache[s_ // self.page, :, s_ % self.page, :]), _bits(_quantize(vf[lv], self.v_scale))), f"{what}: V pool rows"
        for pool, before in zip((self.k_cache, self.v_cache), self.prefix_bits):
            now = _bits(pool)
            touched = (now != before).any(dim=-1).any(dim=1)
            written = set((int(p) for p in (s_ // self.page * self.page + s_ % self.page).tolist()))
            assert all(int(pg) * self.page + int(off) in written for pg, off in touched.nonzero()), f"{what}: a pool row outside the step's slots moved"
            assert not touched[0].any(), f"{what}: the NULL block moved"
        print(
            f"\n{self.label()} e4m3 {what}: cos(out) {c:.6f} max|d out| {d_out:.4e} max|d O_gated| {d_o:.4e} max|d LSE| {d_lse:.2e} dead rows {int(dead.sum())}"
        )
        return dict(cos=c, d_out=d_out, d_o=d_o, d_lse=d_lse, dead=int(dead.sum()))


def _launch_names(run):
    """The CUDA kernel launches of one callable under ``torch.profiler`` (memset / memcpy listed apart), or ``None`` when the profiler
    records no CUDA activity (CUPTI absent -- the ONLY failure swallowed; the caller skips explicitly)."""
    from torch.profiler import ProfilerActivity, profile

    prof = profile(activities=[ProfilerActivity.CUDA])
    try:
        prof.start()
    except Exception:  # noqa: BLE001 -- CUPTI unavailable on this box: the bitwise pins stand on their own
        return None
    try:
        run()
        torch.cuda.synchronize()
    finally:
        prof.stop()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if not names:
        return None
    kern = [n for n in names if "memset" not in n.lower() and "memcpy" not in n.lower()]
    return kern, [n for n in names if n not in kern]


# (S_q, kv_lens, split, fuse_prepare, seed)
_INDEXER_E4M3 = [
    pytest.param(1, (2052, 8192), 4, True, 201, id="Sq1-B2-split4-fused-prepare"),
    pytest.param(4, (2052, 8192), None, False, 202, id="Sq4-B2-default-split-unfused"),
    pytest.param(4, (300, 0, 8191, 77), 4, True, 203, id="Sq4-B4-dead-entry-shorter-than-the-step-split4-fused"),
]


@requires_rubin
@pytest.mark.parametrize("s_q, kv_lens, split, fuse_prepare, seed", _INDEXER_E4M3)
def test_indexer_decode_over_e4m3_pools_matches_the_two_stage_oracle(s_q, kv_lens, split, fuse_prepare, seed):
    """The in-block indexer's decode step over e4m3 pools: the selection the fp64 top-k of the block's own operands up to ties (the first
    stage, unchanged: the indexer reads the compressed-key cache), the attention within budget of the fp32 oracle fed the kernel's list
    over the DEQUANTIZED pool (the second stage), the ring / cache rows bitwise, the pools' step rows the torch cast, two launches bitwise,
    both knob readings (the prepare's e4m3 pool arm and the quantizing write-through)."""
    case = _IndexerE4m3Case(GEOM_SMALL, s_q, kv_lens, page=16, seed=seed, split=split, fuse_gate=True, return_lse=True, fuse_prepare=fuse_prepare)
    case.compile()
    assert case.blk.fuse_prepare is fuse_prepare and (case.blk._prepare is not None) is fuse_prepare
    if fuse_prepare:
        assert case.blk._prepare.kv_e4m3 and case.blk._cache_write is None
    else:
        assert isinstance(case.blk._cache_write, _QuantizedCacheWrite)
    _qsa._two_launches_indexer(case, f"indexer e4m3 fuse_prepare={fuse_prepare}")


@requires_rubin
def test_fused_prepare_over_e4m3_pools_is_bitwise_the_two_stage_write_through():
    """The indexer form over e4m3 pools with ``fuse_prepare`` ON vs OFF on the same inputs: ``out``, the LSE, the list AND the pools BITWISE
    (the prepare's e4m3 pool arm computes the two-stage write-through's codes op for op); the census by kernel name under
    ``set_sync_debug_mode("error")``: the fused chain is SEVEN launches shorter -- the prepare launch replaces the norm + RoPE pass, the
    pool-1 query norm, the ring + compress step, the two reciprocal writes of the scales (torch kernels on the launch stream), the two
    per-tensor casts and the bit copy (eight launches) -- 7 vs 14 at a split with the gate in the combine; no memset / memcpy on either path."""
    on = _IndexerE4m3Case(GEOM_SMALL, 1, (2052, 8192), page=16, seed=211, split=4, fuse_gate=True, return_lse=True, fuse_prepare=True)
    off = _IndexerE4m3Case(GEOM_SMALL, 1, (2052, 8192), page=16, seed=211, split=4, fuse_gate=True, return_lse=True, fuse_prepare=False)
    on.compile()
    off.compile()
    on.run()
    off.run()
    on.check_selection("fused")
    off.check_selection("unfused")
    on.check(what="fused")
    off.check(what="unfused")
    assert torch.equal(_bits(on.out), _bits(off.out)) and torch.equal(on.lse, off.lse), "fused vs unfused differ on out / LSE"
    assert torch.equal(on.kernel_ids(), off.kernel_ids()), "the lists differ"
    assert torch.equal(_bits(on.k_cache), _bits(off.k_cache)) and torch.equal(_bits(on.v_cache), _bits(off.v_cache)), "the e4m3 pools differ"
    assert torch.equal(_bits(on.index_kv_cache), _bits(off.index_kv_cache)) and torch.equal(_bits(on.index_state_cache), _bits(off.index_state_cache))
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        census_on, census_off = _launch_names(on.execute), _launch_names(off.execute)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    if census_on is None or census_off is None:
        pytest.skip("the CUDA profiler recorded no kernel activity on this box (CUPTI unavailable); the bitwise pins stand")
    kern_on, other_on = census_on
    kern_off, other_off = census_off
    print(
        f"\ncensus fused {len(kern_on)}: {[n.split('_tensorptr')[0][:60] for n in kern_on]}\ncensus unfused {len(kern_off)}: {[n.split('_tensorptr')[0][:60] for n in kern_off]}"
    )
    assert not other_on and not other_off, f"memset / memcpy on the execute path: {other_on} {other_off}"
    assert len(kern_off) - len(kern_on) == 7 and sum("qsa_prepare" in n for n in kern_on) == 1 and not any("qsa_prepare" in n for n in kern_off)
    assert (len(kern_on), len(kern_off)) == (7, 14), (len(kern_on), len(kern_off))
    assert sum("elementwise" in n for n in kern_off) == 2 and not any(
        "elementwise" in n for n in kern_on
    ), "the two reciprocal writes belong to the unfused e4m3 chain only"


@requires_rubin
def test_indexer_e4m3_first_execute_captures_and_replays_bitwise_over_new_scales():
    """The indexer form over e4m3 pools captured on its FIRST execute (no warm-up; ``set_sync_debug_mode("error")``), the replay bitwise a
    fresh eager run; new scales through the captured tensors (a recalibration) replay bitwise the eager run over them -- the scales are
    read in-kernel, never a plan constant."""
    # the 24 / 2 band geometry: its slab (N 13952) rides the small-M route's tile at these row counts, capture-safe from the first execute; the
    # 8 / 2 band slab (N 5760) takes the GEMM graph route, whose handle is created eagerly (the decode module's indexer capture cell, the same choice)
    case = _IndexerE4m3Case(GEOM_FLASH_NEXT, 1, (2052, 300), page=16, seed=221, split=4, fuse_gate=True, return_lse=True, fuse_prepare=True)
    case.prime_prefix()
    case.blk.check_support()
    case.blk.compile()
    case.ws = torch.full((case.blk.get_workspace_size(),), 0x7F, dtype=torch.uint8, device=_DEV)
    case.out.fill_(float("nan"))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                case.execute()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(case.out).all(), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        case.oracle_from_list()
        replayed, ids_r, lse_r = case.out.clone(), case.kernel_ids().clone(), case.lse.clone()
        case.check_selection("replay")
        case.check(what="replay")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed)) and torch.equal(case.kernel_ids(), ids_r) and torch.equal(case.lse, lse_r)
        # new scales (a recalibration by 2x): the pools' prefix rows re-quantized by torch to the new scales, the oracle refreshed
        k_new, v_new = case.k_scale * 2.0, case.v_scale * 0.5
        k8, v8 = _pools(case.n_pages, case.geom.h_kv, case.page, case.layout, E4M3)
        k8.copy_(_quantize(_dequantize(case.k_cache, case.k_scale), k_new))
        v8.copy_(_quantize(_dequantize(case.v_cache, case.v_scale), v_new))
        case.k_cache.copy_(k8)
        case.v_cache.copy_(v8)
        case.k_scale.copy_(k_new)
        case.v_scale.copy_(v_new)
        case.prefix_bits = (_bits(case.k_cache).clone(), _bits(case.v_cache).clone())
        case.refs = {b: _dequantized_ref(types.SimpleNamespace(q=r.q, gate=r.gate, k=r.k, v=r.v), k_new, v_new) for b, r in case.refs.items()}
        case.out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        case.oracle_from_list()
        replayed2, lse2 = case.out.clone(), case.lse.clone()
        case.check(what="replay over new scales")
        case.out.fill_(float("nan"))
        case.execute()
        torch.cuda.synchronize()
        assert torch.equal(_bits(case.out), _bits(replayed2)) and torch.equal(case.lse, lse2), "the replay over new scales differs from the eager run"
    finally:
        graph.reset()


def test_the_write_through_fold_never_serves_the_e4m3_cache():
    """``fuse_cache_write`` (the paged write-through FOLDED into the fused norm + RoPE projection's epilogue; None = AUTO) composes with the
    e4m3 cache by staying OFF: the epilogue writes 16-bit rows, the e4m3 write-through QUANTIZES.  AUTO on a ``fuse_norm_rope=True``
    e4m3 declaration keeps the separate quantizing launch (the fork's pool arm unarmed); an explicit ``True`` is a typed decline naming
    the attribute; the same declaration over 16-bit pools folds, so the pin reads the composition and not a disabled fold."""
    kw = dict(
        kv_source="self", sample_k_cache=None, sample_v_cache=None, sample_block_table=None, geom_over=dict(causal_bottom_right=False), fuse_norm_rope=True
    )
    with _no_host_sync():
        e4, _ = _declare(GEOM_SMALL, 2, 8, **kw)
    assert e4.kv_cache_e4m3 and e4.fuse_norm_rope and e4.fuse_cache_write is None
    assert e4.cache_write_fused is False and isinstance(e4._cache_write, _QuantizedCacheWrite) and e4._proj.kv_pool_page_size == 0
    with _no_host_sync():
        b16, _ = _declare(GEOM_SMALL, 2, 8, pool_dtype=BF16, kv_cache_dtype=None, **kw)
    assert not b16.kv_cache_e4m3 and b16.cache_write_fused is True and b16._cache_write is None and b16._proj.kv_pool_page_size == 16
    with pytest.raises(NotImplementedError, match="fuse_cache_write=True with kv_cache_dtype=torch.float8_e4m3fn"):
        with _no_host_sync():
            _declare(GEOM_SMALL, 2, 8, fuse_cache_write=True, **kw)


def test_the_packed_prefill_over_e4m3_pools_is_declined_by_name():
    """The packed (``thd=True``) prefill's quantizing write-through is a TYPED DECLINE until its accept cell lands (the doc page and the
    tracker say so; a review found the constructor accepting it): a packed ``kv_source="self"`` declaration with the attribute raises
    ``NotImplementedError`` naming ``thd`` and the attribute, before any stage is built; the same packing over bf16 pools is served
    (the bf16 write-through addresses tokens), so the pin reads the e4m3 row and not a disabled packing."""
    geom = GatedAttentionBlockGeometry(**GEOM_SMALL)
    T = 32
    mk = lambda *shape: torch.empty(*shape, dtype=BF16, device=_DEV)  # noqa: E731

    def declare(**kw):
        return GatedAttentionBlockFwd(
            mk(T, geom.d_model),
            mk(geom.n_qkvg, geom.d_model),
            torch.ones(D, dtype=BF16, device=_DEV),
            torch.ones(D, dtype=BF16, device=_DEV),
            mk(T, ROPE),
            mk(T, ROPE),
            mk(geom.d_model, geom.h_q * D),
            mk(T, geom.d_model),
            geom,
            thd=True,
            num_sequences=2,
            max_seq_len=T,
            paged_kv_page_size=16,
            **kw,
        )

    with pytest.raises(NotImplementedError, match="kv_cache_dtype=torch.float8_e4m3fn with thd=True"):
        with _no_host_sync():
            declare(kv_cache_dtype=E4M3)
    with _no_host_sync():
        b16 = declare()
    assert b16.thd and not b16.kv_cache_e4m3 and b16.paged_kv_page_size == 16 and b16._cache_write is not None
    assert not isinstance(b16._cache_write, _QuantizedCacheWrite)
