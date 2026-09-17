# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-local pointwise stages of the MQA sparse-attention block as ONE FROST kernel.

One recipe serves the four in-place row passes of the block, each a chain of
optional steps on a ``[T * heads, d]`` row space of the io dtype (bf16):

    RMSNorm(w, eps)  ->  interleaved RoPE on the LAST ``rope_dim`` dims  ->  block-32 ue8m0 fake-quant
                         (+ an optional per-row LSE fold on the side)

* ``q_norm``   -- ``d=1280, heads=1, apply_norm``                      (rows of the low-rank Q)
* ``q_rope``   -- ``d=512, heads=64, rope_dim=64`` (nothing else)      (tail-only lane map, below)
* ``kv_chain`` -- ``d=512, heads=1, apply_norm, rope_dim=64, fake_quant`` in place on ``kv_all[:, :S]``
* ``o_unrope`` -- ``d=512, heads=64, rope_dim=64, rope_inverse`` (+ ``lse_fold``)

**Lane maps.**  Every lane moves 16 B (8 bf16) per access.

* *Full-row map* (a norm or a fake-quant reads every element): ``lanes = min(32, d // 8)`` lanes per row,
  ``chunks = d // (lanes * 8)`` accesses per lane; element ``e = (c * lanes + lane) * 8 + i``.  A row is a
  contiguous 512 B request per chunk.  The RoPE tail ``[d - rope_dim, d)`` lies inside the LAST chunk on lanes
  ``lane >= lanes - rope_dim // 8``; the interleaved partner of ``x[2k]`` is ``x[2k + 1]``, the NEXT element of
  the SAME 16 B access, so the rotation needs no shuffle at all (the gated block's rotate_half map needed a
  ``shfl.bfly`` per element).  A 32-element fake-quant block is 4 consecutive lanes of one chunk: the block
  amax is an in-lane ternary tree + a 2-step ``shfl.bfly`` (mask 1, 2).
* *Tail-only map* (RoPE and nothing else touches the data): ``rope_dim // 8`` lanes per row read and write
  ONLY the 128 B tail.  The untouched ``[0, d - rope_dim)`` never leave memory -- passthrough is a free
  ``torch.equal``, and the pass moves ``rope_dim / d`` of the row (1/8 at 64/512).

The row space is ``[B, R, d]`` with ``R = tokens_per_batch * heads``: rows compact within a batch, the
batch stride free (``kv_all[:, :S]`` is a strided view of ``[B, S + N_c, 512]``; plan 2 D4).  The grid is
``(ceil(R / rows_per_cta), B)`` so no lane divides by a runtime batch extent.  Tail rows clamp their loads
to the last row and skip every store.

**Math -- the model's THREE bf16 rounding points (default arm, ``three_roundings=True``):**

1. ``var = sum(x^2) / d`` (fp32, a true divide), ``rstd = rsqrt(var + eps)``, ``y = (x * rstd) * w``; round 1
   = a bf16 round trip when a later step follows (the model materialises the RMSNorm output in bf16).
2. RoPE as the fp32 complex multiply ``(x0 + i x1)(cos + i sin)`` per adjacent pair, ``sin`` negated for the
   inverse (``freqs_cis.conj()``); round 2 = a bf16 round trip when the fake-quant follows.
3. Per 32-block: ``amax = max(max|y|, 1e-4)``; ``byte = cvt.rp.ue8m0(amax * fp32(1/448))`` (= the model's
   ``fast_log2_ceil`` bit for bit on normal inputs), ``s = 2^(byte - 127)``, ``q = e4m3_rne_sat(y * 2^-byte')``,
   ``y = fp8x2_to_f32(q) * s``; round 3 = the bf16 store (exact: an e4m3 code times a power of two IS a bf16).

``three_roundings=False`` keeps steps 1-3 in fp32 and rounds once at the store (strictly more accurate; not
the model).  The LSE fold is the H64 -> block convention, on the row's lane 0:
``lse' = select(lse == +inf, sink, select(sink == +inf, +inf, logaddexp(lse, sink)))`` -- a bare
``logaddexp`` NaNs at ``sink = +inf`` on a live row; ``sink = -inf`` yields ``lse`` exactly.

**Barrier table: NONE.**  No mbarrier, no named barrier, no SMEM: the only cross-lane traffic is
``shfl.sync`` with a full mask (the sum-of-squares butterfly, the 4-lane amax butterfly), issued outside
every divergent branch (the RoPE table load under ``if in_rope`` and the LSE fold under ``if lane == 0``
contain no shuffle).

**SMEM buffer table: EMPTY by design.**  Every warp access is a contiguous 512 B (full map) or 4 x 128 B
(tail map) request with no reuse and no MMA consumer, so staging through shared memory would be a
round trip for nothing (the gated block measured the same kernel shape register-bound, not
request-starved: ``gated_attention_block/kernels/qk_norm_rope.py`` module docstring).

**Byte model** (:func:`moved_bytes`; the denominator of a GB/s figure -- cos/sin and ``w`` are excluded,
they are ``[S, 32]`` / ``[d]`` and L2-resident across the heads of a token): full map ``2 * B * R * d * 2``,
tail map ``2 * B * R * rope_dim * 2``, plus ``8 * B * R`` for an LSE fold.  Ideal at 32K tokens on Rubin
(12846 GB/s): q_norm 0.013 ms, q_rope 0.042, kv_chain 0.006, o_unrope 0.043 -- vs the torch chain's
0.57 / 1.63 / 0.67 / 1.63 (plan 2.5).

Public surface (append-only): :class:`NormRopeRecipe`, :func:`validate_shape`, :func:`build_norm_rope`
-> :class:`NormRopeKernel` (``.run`` in place), :func:`norm_rope_reference` (the torch mirror),
:func:`moved_bytes`.  The block imports this module LAZILY (``pointwise_impl="frost"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.device import compute_capability, resolve_device
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, e8m0_from_amax, f16x2_to_f32, fmax_f32, fp32_to_fp16, fp32_to_fp8x2, fp8x2_to_f32, lane_group_sum
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

ELEMS_PER_ACCESS = 8  # bf16 / f16 per 16-byte access
ACCESS_BYTES = 16
WARP = 32
QUANT_BLOCK = 32  # elements per ue8m0 scale (K:41-95 ``block_size``, M:27 ``fp8_block_size``)
AMAX_FLOOR = 1e-4  # K:76 ``amax = max(amax, 1e-4)``
AMAX_FLOOR_BITS = 0x38D1B717  # fp32(1e-4): a REGISTER operand for ``max.f32`` (a folded float immediate ICEs libNVVM)
E4M3_MAX = 448.0
DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_COMPILE_OPTIONS = "--enable-tvm-ffi"
_IO_DTYPES = (torch.bfloat16, torch.float16)
_MIN_CC_FAKE_QUANT = (10, 0)  # ``cvt.rp.satfinite.ue8m0x2.f32`` is sm_100+; the e4m3 cvts are sm_89+

_FAKE_STREAM = None
compiled_cache = {}


# ---------------------------------------------------------------------------
# Recipe + shape algebra (no DSL, no device)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormRopeRecipe:
    """Build-time facts of one row pass.  Everything here is derivable from the DECLARATION, so it is a legal
    compile key; the token count is NOT in it (it rides in as a symbolic extent + runtime row counts).

    ``d``: elements per row.  ``heads``: rows per token (the RoPE table row is ``row // heads``; the LSE fold
    reads ``sink[row % heads]``).  ``apply_norm``: RMSNorm with a ``[d]`` weight first.  ``rope_dim``: rotate
    the LAST ``rope_dim`` dims as adjacent pairs (0 = none); ``rope_inverse`` negates ``sin``.  ``fake_quant``:
    the block-32 ue8m0 epilogue.  ``lse_fold``: fold ``sink`` into a per-row LSE on the side.
    ``three_roundings``: the model's intermediate bf16 round trips (default) or one rounding at the store.
    """

    d: int
    heads: int
    apply_norm: bool
    rope_dim: int
    rope_inverse: bool = False
    fake_quant: bool = False
    lse_fold: bool = False
    three_roundings: bool = True
    io_dtype: torch.dtype = torch.bfloat16

    @property
    def tail_only(self) -> bool:
        """RoPE and nothing else touches the data: only the 128 B tail of each row is moved."""
        return bool(self.rope_dim) and not self.apply_norm and not self.fake_quant

    @property
    def lanes_per_row(self) -> int:
        return self.rope_dim // ELEMS_PER_ACCESS if self.tail_only else min(WARP, self.d // ELEMS_PER_ACCESS)

    @property
    def chunks(self) -> int:
        """16-byte accesses each lane makes per row."""
        return 1 if self.tail_only else self.d // (self.lanes_per_row * ELEMS_PER_ACCESS)

    @property
    def touches_data(self) -> bool:
        return bool(self.apply_norm or self.rope_dim or self.fake_quant)

    @property
    def rope_pairs(self) -> int:
        return self.rope_dim // 2


def validate_shape(recipe: NormRopeRecipe, threads_per_cta: int = DEFAULT_THREADS_PER_CTA) -> None:
    """Raise ``ValueError`` on any geometry this kernel cannot address -- never an ``assert``: these come
    from user-facing geometry, so they must survive ``python -O`` and name themselves."""
    d, rope_dim, heads = recipe.d, recipe.rope_dim, recipe.heads
    if not isinstance(d, int) or isinstance(d, bool) or d <= 0 or d % ELEMS_PER_ACCESS != 0:
        raise ValueError(f"d must be a positive multiple of {ELEMS_PER_ACCESS} (one 16-byte access per lane), got {d!r}")
    if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0:
        raise ValueError(f"heads must be a positive int, got {heads!r}")
    if not isinstance(rope_dim, int) or isinstance(rope_dim, bool) or rope_dim < 0:
        raise ValueError(f"rope_dim must be a non-negative int, got {rope_dim!r}")
    if recipe.io_dtype not in _IO_DTYPES:
        raise ValueError(f"io_dtype must be bf16 or f16, got {recipe.io_dtype}")
    if not recipe.touches_data and not recipe.lse_fold:
        raise ValueError("recipe is an identity (no norm, no RoPE, no fake_quant, no lse_fold); drop the stage instead of launching it")
    if recipe.rope_inverse and rope_dim == 0:
        raise ValueError("rope_inverse=True with rope_dim=0 rotates nothing; set rope_dim or drop rope_inverse")
    if rope_dim:
        if rope_dim % (2 * ELEMS_PER_ACCESS) != 0:
            raise ValueError(f"rope_dim must be a multiple of {2 * ELEMS_PER_ACCESS} (whole 16-byte accesses of adjacent pairs), got {rope_dim}")
        if rope_dim > d:
            raise ValueError(f"rope_dim={rope_dim} exceeds d={d}")
    if recipe.fake_quant:
        if d % QUANT_BLOCK != 0:
            raise ValueError(f"fake_quant needs d % {QUANT_BLOCK} == 0 (whole ue8m0 blocks per row), got d={d}")
        if recipe.io_dtype is not torch.bfloat16:
            raise ValueError("fake_quant is defined on bf16 rows (the dequantised e4m3 x 2^k grid is bf16-exact), got io_dtype=" f"{recipe.io_dtype}")
    lanes = recipe.lanes_per_row
    if recipe.tail_only:
        if WARP % lanes != 0:
            raise ValueError(f"tail map: rope_dim={rope_dim} gives {lanes} lanes/row, which must divide a warp (rope_dim in 16, 32, 64, 128, 256)")
        if d % rope_dim != 0:
            raise ValueError(f"tail map needs d % rope_dim == 0 so every row's tail is 16-byte aligned, got d={d}, rope_dim={rope_dim}")
    else:
        if WARP % lanes != 0:
            raise ValueError(f"d={d} gives {lanes} lanes/row, which must divide a warp so the reductions stay in-row")
        if d != lanes * ELEMS_PER_ACCESS * recipe.chunks:
            raise ValueError(
                f"d={d} is not covered exactly by {lanes} lanes x {recipe.chunks} accesses x {ELEMS_PER_ACCESS}; use a multiple of {lanes * ELEMS_PER_ACCESS}"
            )
        if rope_dim > lanes * ELEMS_PER_ACCESS:
            raise ValueError(
                f"rope_dim={rope_dim} must lie inside the LAST access chunk ({lanes * ELEMS_PER_ACCESS} elements) so no lane rotates across chunks"
            )
    if threads_per_cta % lanes != 0 or threads_per_cta % WARP != 0 or threads_per_cta <= 0:
        raise ValueError(f"threads_per_cta={threads_per_cta} must be a positive multiple of a warp and of the {lanes} lanes per row")


# ---------------------------------------------------------------------------
# Device helpers (trace-time Python over traced values)
# ---------------------------------------------------------------------------


def _opaque_f32_bits(bits: int):
    """An fp32 constant as a REGISTER the optimizer cannot fold into an asm immediate (frost-tile-dsl.md 7)."""
    return inline_ptx(f"mov.b32 $0, 0x{bits:08X};", write_only_types=[cutlass.Float32])


def _f32_bits(bits: int):
    return cutlass.Int32(bits).bitcast(cutlass.Float32)


def _round_trip(vals, io_ty):
    """fp32 -> io dtype -> fp32 on a list of fp32 (even length): the model's intermediate materialisation."""
    out = []
    for i in range(0, len(vals), 2):
        lo, hi = f16x2_to_f32(fp32_to_fp16(vals[i], vals[i + 1], dtype=io_ty), dtype=io_ty)
        out.append(lo)
        out.append(hi)
    return out


def _load_row_chunk(addr, io_ty):
    """One 16-byte access -> 8 fp32."""
    vals = []
    for w in ld_global_v4(addr, cutlass.Int32):
        lo, hi = f16x2_to_f32(w, dtype=io_ty)
        vals.append(lo)
        vals.append(hi)
    return vals


def _store_row_chunk(addr, vals, io_ty):
    packed = [fp32_to_fp16(vals[i], vals[i + 1], dtype=io_ty) for i in range(0, ELEMS_PER_ACCESS, 2)]
    st_global_v4(addr, packed, cutlass.Int32)


def _load_f32x4(addr):
    return list(ld_global_v4(addr, cutlass.Float32))


def _rotate_pairs(vals, cos_v, sin_v):
    """``(x0 + i x1) * (cos + i sin)`` on 4 adjacent pairs -- the interleaved RoPE of M:392-406 in fp32.

    Spelled as ``fma(x0, cos, -(x1 * sin))`` / ``fma(x0, sin, x1 * cos)`` on purpose: that is the contraction
    nvcc applies to ``c10::complex<float>::operator*`` (``a*c - b*d`` / ``a*d + b*c``), so the kernel's rotation
    is BIT-IDENTICAL to torch's CUDA complex multiply (probed 2026-09-17 against ``torch.complex(a, b) *
    torch.complex(c, d)`` over 2^20 random operands: this order matches every element, the other
    ``fma(-x1, sin, x0*cos)`` order and the un-fused spelling do not).  The inverse (``sin -> -sin``) keeps the
    same operations, negation being exact, so it matches ``x * freqs_cis.conj()`` bit for bit too."""
    out = []
    for p in range(ELEMS_PER_ACCESS // 2):
        x0, x1 = vals[2 * p], vals[2 * p + 1]
        out.append(cute.math.fma(x0, cos_v[p], -(x1 * sin_v[p])))
        out.append(cute.math.fma(x0, sin_v[p], x1 * cos_v[p]))
    return out


def _fake_quant_block(vals, floor_reg):
    """The K:41-95 epilogue on ONE lane's 8 elements of a 32-element block shared with lanes ``lane ^ 1``, ``lane ^ 2``.

    Every lane of the warp must reach the two ``shfl.bfly`` (call outside divergent branches)."""
    amax = abs_max_tree(vals)
    amax = fmax_f32(amax, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(1), 31, kind=nvvm.Shfl.BFLY)))
    amax = fmax_f32(amax, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, cutlass.Int32(2), 31, kind=nvvm.Shfl.BFLY)))
    amax = fmax_f32(amax, floor_reg)  # K:76
    # ``byte = ceil_pow2(amax * fp32(1/448))`` as an E8M0 exponent -- bit for bit the model's fast_log2_ceil on
    # normal inputs (the floor keeps the product >= 2^-23, far above fp32 subnormals); ``rcp = 2^-k``, ``s = 2^k``.
    rcp, byte = e8m0_from_amax(amax)
    scale = (byte << 23).bitcast(cutlass.Float32)
    out = []
    for p in range(ELEMS_PER_ACCESS // 2):
        q = fp32_to_fp8x2(vals[2 * p] * rcp, vals[2 * p + 1] * rcp)  # RNE, saturating: K:85 ``T.Cast(FP8, clamp(x / s))``
        lo, hi = fp8x2_to_f32(q)
        out.append(lo * scale)
        out.append(hi * scale)
    return out


@cute.jit
def _lse_fold(lse_v: cutlass.Float32, sink_v: cutlass.Float32) -> cutlass.Float32:
    """``select(lse == +inf, sink, select(sink == +inf, +inf, logaddexp(lse, sink)))`` in fp32 (plan 2 D1).

    A ``@cute.jit`` so the ternaries below stage to ``arith.select`` (in a plain helper they are Python
    control flow on a traced Boolean -- ``PHASE_DYNAMIC_TO_STATIC_BOOL``)."""
    pos_inf = _f32_bits(0x7F800000)
    neg_inf = _f32_bits(0xFF800000)
    m = cute.math.max(lse_v, sink_v)
    t = cute.math.exp(-cute.math.abs(lse_v - sink_v))  # 0 when exactly one operand is -inf; NaN only when both are
    r = m + cute.math.log1p(t)
    r = neg_inf if m == neg_inf else r  # logaddexp(-inf, -inf) = -inf, as torch
    r = pos_inf if sink_v == pos_inf else r
    return sink_v if lse_v == pos_inf else r


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@cute.kernel
def frost_norm_rope(
    mX: cute.Tensor,  # [B, R, d] io dtype; rows compact, batch stride symbolic; in place
    mW: Optional[cute.Tensor],  # [d] io dtype, or None (no RMSNorm traced)
    mCos: Optional[cute.Tensor],  # [B, Tok, rope_dim // 2] fp32, batch stride symbolic (0 = one table for every batch)
    mSin: Optional[cute.Tensor],
    mLseIn: Optional[cute.Tensor],  # [B, R] fp32 (row-major; batch stride symbolic)
    mSink: Optional[cute.Tensor],  # [heads] fp32
    mLseOut: Optional[cute.Tensor],  # [B, heads, Tok] fp32, strides (sym, sym, 1)
    n_rows: cutlass.Int32,  # R = Tok * heads, rows per batch
    eps: cutlass.Float32,
    d: cutlass.Constexpr[int],
    heads: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    rope_inverse: cutlass.Constexpr[bool],
    fake_quant: cutlass.Constexpr[bool],
    three_roundings: cutlass.Constexpr[bool],
    tail_only: cutlass.Constexpr[bool],
    lanes: cutlass.Constexpr[int],
    chunks: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
) -> None:
    """``rows_per_group`` consecutive rows per lane group: every load first, then reduce / rotate / quantise /
    store.  ``apply_norm`` and ``lse_fold`` are decided at TRACE time from the presence of ``mW`` and
    ``mLseOut`` (the same presence switch the gated kernel uses for its weights and rstd)."""
    io_ty = mX.element_type
    bpe = cutlass.const_expr(io_ty.width // 8)
    apply_norm = cutlass.const_expr(mW is not None)
    lse_fold = cutlass.const_expr(mLseOut is not None)
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    rope_lane0 = cutlass.const_expr(lanes - rope_lanes)  # first rope lane of the LAST chunk (0 on the tail map)
    last_chunk = cutlass.const_expr(chunks - 1)
    row_bytes = cutlass.const_expr(d * bpe)
    tail_bytes = cutlass.const_expr((d - rope_dim) * bpe if tail_only else 0)
    round_after_norm = cutlass.const_expr(three_roundings and apply_norm and (rope_dim > 0 or fake_quant))
    round_after_rope = cutlass.const_expr(three_roundings and rope_dim > 0 and fake_quant)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    batch = cutlass.Int32(cute.arch.block_idx()[1])
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    x_batch = mX.iterator.toint() + batch.to(cutlass.Int64) * cutlass.Int64(mX.stride[0]) * cutlass.Int64(bpe)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    in_rope = (lane >= cutlass.Int32(rope_lane0)) if cutlass.const_expr(rope_dim > 0 and not tail_only) else cutlass.Boolean(True)
    # The rope table row is 16-byte-aligned per lane: pair index 4 * (lane - rope_lane0).
    rope_tab_off = ((lane - cutlass.Int32(rope_lane0)) if cutlass.const_expr(not tail_only) else lane).to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    floor_reg = _opaque_f32_bits(AMAX_FLOOR_BITS) if cutlass.const_expr(fake_quant) else cutlass.Float32(0.0)

    # --- PASS 1: every load this thread makes -----------------------------------------------------------
    rows = []
    row_addrs = []
    toks = []
    heads_of = []
    xs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        addr = x_batch + row_r.to(cutlass.Int64) * cutlass.Int64(row_bytes) + cutlass.Int64(tail_bytes) + lane_off
        row_x = []
        for c in cutlass.range_constexpr(chunks):
            row_x.append(_load_row_chunk(addr + cutlass.Int64(c * lanes * ACCESS_BYTES), io_ty))
        rows.append(row)
        row_addrs.append(addr)
        toks.append(row_r // cutlass.Int32(heads))
        heads_of.append(row_r % cutlass.Int32(heads))
        xs.append(row_x)

    # --- PASS 2: reduce, normalise, rotate, quantise, store ----------------------------------------------
    for r in cutlass.range_constexpr(rows_per_group):
        ys = xs[r]
        if cutlass.const_expr(apply_norm):
            acc = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(chunks):
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + ys[c][i] * ys[c][i]
            var = lane_group_sum(acc, lanes) / cutlass.Float32(d)  # M:291 ``x.square().mean(-1)``: a divide
            rstd = cute.math.rsqrt(var + eps, fastmath=True)
            # The weight is consumed HERE, after the reduction (register relief; it is an L1/L2 hit by construction).
            w_base = mW.iterator.toint() + lane_off
            normed = []
            for c in cutlass.range_constexpr(chunks):
                w_c = _load_row_chunk(w_base + cutlass.Int64(c * lanes * ACCESS_BYTES), mW.element_type)
                y_c = [(ys[c][i] * rstd) * w_c[i] for i in range(ELEMS_PER_ACCESS)]  # M:292-293 ``(weight * (x * rsqrt))``
                normed.append(_round_trip(y_c, io_ty) if cutlass.const_expr(round_after_norm) else y_c)
            ys = normed

        if cutlass.const_expr(rope_dim > 0):
            tok = toks[r]
            cos_v = [cutlass.Float32(0.0)] * (ELEMS_PER_ACCESS // 2)
            sin_v = [cutlass.Float32(0.0)] * (ELEMS_PER_ACCESS // 2)
            if in_rope:
                tab = batch.to(cutlass.Int64) * cutlass.Int64(mCos.stride[0]) + tok.to(cutlass.Int64) * cutlass.Int64(mCos.stride[1])
                cos_v = _load_f32x4(mCos.iterator.toint() + tab * cutlass.Int64(4) + rope_tab_off)
                tab_s = batch.to(cutlass.Int64) * cutlass.Int64(mSin.stride[0]) + tok.to(cutlass.Int64) * cutlass.Int64(mSin.stride[1])
                sin_v = _load_f32x4(mSin.iterator.toint() + tab_s * cutlass.Int64(4) + rope_tab_off)
            if cutlass.const_expr(rope_inverse):
                sin_v = [-s for s in sin_v]  # M:399 ``freqs_cis.conj()``
            rotated = _rotate_pairs(ys[last_chunk], cos_v, sin_v)
            if cutlass.const_expr(round_after_rope):
                rotated = _round_trip(rotated, io_ty)
            if cutlass.const_expr(tail_only):
                ys = [rotated]
            else:
                # Only the rope lanes keep the rotation; the select is per element (statement-level so the
                # preprocessor stages it), never a multiply by a 0/1 mask.
                ys = [list(c_vals) for c_vals in ys]
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    ys[last_chunk][i] = rotated[i] if in_rope else ys[last_chunk][i]

        if cutlass.const_expr(fake_quant):
            ys = [_fake_quant_block(ys[c], floor_reg) for c in range(chunks)]

        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(chunks):
                _store_row_chunk(row_addrs[r] + cutlass.Int64(c * lanes * ACCESS_BYTES), ys[c], io_ty)

        if cutlass.const_expr(lse_fold):
            if lane == cutlass.Int32(0):
                if rows[r] < n_rows:
                    lse_addr = mLseIn.iterator.toint() + (
                        batch.to(cutlass.Int64) * cutlass.Int64(mLseIn.stride[0]) + rows[r].to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    sink_addr = mSink.iterator.toint() + heads_of[r].to(cutlass.Int64) * cutlass.Int64(4)
                    folded = _lse_fold(ld_global(lse_addr, cutlass.Float32), ld_global(sink_addr, cutlass.Float32))
                    out_idx = (
                        batch.to(cutlass.Int64) * cutlass.Int64(mLseOut.stride[0])
                        + heads_of[r].to(cutlass.Int64) * cutlass.Int64(mLseOut.stride[1])
                        + toks[r].to(cutlass.Int64)
                    )
                    st_global(mLseOut.iterator.toint() + out_idx * cutlass.Int64(4), folded, cutlass.Float32)


@cute.jit
def norm_rope_launch(
    x: cute.Tensor,
    w: Optional[cute.Tensor],
    cos: Optional[cute.Tensor],
    sin: Optional[cute.Tensor],
    lse_in: Optional[cute.Tensor],
    sink: Optional[cute.Tensor],
    lse_out: Optional[cute.Tensor],
    n_rows: cutlass.Int32,
    eps: cutlass.Float32,
    n_blocks_x: cutlass.Int32,
    n_batch: cutlass.Int32,
    d: cutlass.Constexpr[int],
    heads: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    rope_inverse: cutlass.Constexpr[bool],
    fake_quant: cutlass.Constexpr[bool],
    three_roundings: cutlass.Constexpr[bool],
    tail_only: cutlass.Constexpr[bool],
    lanes: cutlass.Constexpr[int],
    chunks: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_norm_rope(
        x,
        w,
        cos,
        sin,
        lse_in,
        sink,
        lse_out,
        n_rows,
        eps,
        d,
        heads,
        rope_dim,
        rope_inverse,
        fake_quant,
        three_roundings,
        tail_only,
        lanes,
        chunks,
        threads_per_cta,
        rows_per_group,
    ).launch(grid=(n_blocks_x, n_batch, 1), block=(threads_per_cta, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host: compile (cached), operand normalisation, launch
# ---------------------------------------------------------------------------


def _fake_batched_rows(dtype, ncols: int, *, assumed_align: int = 16):
    """``[B, R, ncols]`` with a SYMBOLIC batch stride and compact rows: one artifact serves ``kv_all[:, :S]``
    (batch stride ``(S + N_c) * d``) and every compact ``[T, heads, d]`` / ``[T, d]`` view alike."""
    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(cute.sym_int(), cute.sym_int(), ncols),
        stride=(cute.sym_int(), ncols, 1),
        assumed_align=assumed_align,
    )


@dataclass(frozen=True)
class NormRopeKernel:
    """A compiled recipe bound to a device.  ``run`` is the lowered launch: shape checks, no allocation."""

    recipe: NormRopeRecipe
    compiled: object
    device: int
    n_rows_max: int
    eps: float
    threads_per_cta: int
    rows_per_group: int

    @property
    def rows_per_cta(self) -> int:
        return (self.threads_per_cta // self.recipe.lanes_per_row) * self.rows_per_group

    def run(self, x, w, cos, sin, *, lse_in=None, sink=None, lse_out=None, stream) -> None:
        """In place on ``x``.

        ``x``: ``[T, d]`` (``heads == 1``), ``[B, S, d]`` (``heads == 1``; batch stride free -- ``kv_all[:, :S]``),
        ``[T, heads, d]`` (compact) or ``[B, S, heads, d]`` (compact within a batch).  ``w``: ``[d]`` iff the recipe
        applies the norm.  ``cos``/``sin``: fp32 ``[Tok, rope_dim // 2]`` (one table for every batch) or
        ``[B, Tok, rope_dim // 2]``, where ``Tok`` is ``x``'s per-batch token extent; ``None`` iff ``rope_dim == 0``.
        LSE fold operands (iff ``recipe.lse_fold``): ``lse_in`` fp32 ``[T, heads]`` / ``[B, Tok, heads]``,
        ``sink`` fp32 ``[heads]``, ``lse_out`` fp32 ``[B, heads, Tok]`` (last dim contiguous).
        ``stream``: a raw ``CUstream`` int (or ``cuda.CUstream``).
        """
        ops = check_operands(self.recipe, x, w, cos, sin, lse_in=lse_in, sink=sink, lse_out=lse_out, n_rows_max=self.n_rows_max, device=self.device)
        if ops.n_batch == 0 or ops.n_rows == 0:
            return  # nothing to launch; a zero grid is a launch error, not an empty pass
        n_blocks_x = (ops.n_rows + self.rows_per_cta - 1) // self.rows_per_cta
        self.compiled(
            ops.x,
            ops.w,
            ops.cos,
            ops.sin,
            ops.lse_in,
            ops.sink,
            ops.lse_out,
            cutlass.Int32(ops.n_rows),
            cutlass.Float32(self.eps),
            cutlass.Int32(n_blocks_x),
            cutlass.Int32(ops.n_batch),
            stream if isinstance(stream, cuda.CUstream) else cuda.CUstream(int(stream)),
        )


@dataclass(frozen=True)
class _Operands:
    """The kernel-shaped views ``check_operands`` hands to the artifact."""

    x: torch.Tensor  # [B, R, d]
    w: Optional[torch.Tensor]
    cos: Optional[torch.Tensor]  # [B, Tok, P]
    sin: Optional[torch.Tensor]
    lse_in: Optional[torch.Tensor]  # [B, R]
    sink: Optional[torch.Tensor]
    lse_out: Optional[torch.Tensor]  # [B, heads, Tok]
    n_batch: int
    n_rows: int  # per batch
    n_tok: int  # per batch


def _row_space(recipe: NormRopeRecipe, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """``x`` -> a ``[B, R, d]`` view (compact rows, free batch stride) + ``(B, R)``; typed rejects."""
    d, heads = recipe.d, recipe.heads
    if x.dtype != recipe.io_dtype:
        raise ValueError(f"x must be {recipe.io_dtype} (the recipe's io_dtype), got {x.dtype}")
    if x.ndim == 2:
        if heads != 1:
            raise ValueError(f"a 2-D x [T, d] needs heads == 1, recipe has heads={heads}; pass [T, heads, d]")
        t, dd = x.shape
        if dd != d or x.stride(1) != 1 or (t > 1 and x.stride(0) != d):
            raise ValueError(f"x [T, d] must be compact with d={d}, got shape {tuple(x.shape)} strides {tuple(x.stride())}")
        return x.as_strided((1, t, d), (max(t, 1) * d, d, 1)), 1, t
    if x.ndim == 3:
        if heads == 1:  # [B, S, d]: batch stride free (kv_all[:, :S])
            b, s, dd = x.shape
            if dd != d or x.stride(2) != 1 or (s > 1 and x.stride(1) != d):
                raise ValueError(f"x [B, S, d] must have compact rows with d={d}, got shape {tuple(x.shape)} strides {tuple(x.stride())}")
            return x.as_strided((b, s, d), (x.stride(0) if b > 1 else max(s, 1) * d, d, 1)), b, s
        t, h, dd = x.shape
        if h != heads or dd != d or x.stride(2) != 1 or (h > 1 and x.stride(1) != d) or (t > 1 and x.stride(0) != heads * d):
            raise ValueError(f"x [T, heads, d] must be compact with heads={heads}, d={d}, got shape {tuple(x.shape)} strides {tuple(x.stride())}")
        return x.as_strided((1, t * heads, d), (max(t, 1) * heads * d, d, 1)), 1, t * heads
    if x.ndim == 4:
        b, s, h, dd = x.shape
        if h != heads or dd != d or x.stride(3) != 1 or (h > 1 and x.stride(2) != d) or (s > 1 and x.stride(1) != heads * d):
            raise ValueError(
                f"x [B, S, heads, d] must be compact within a batch with heads={heads}, d={d}, got shape {tuple(x.shape)} strides {tuple(x.stride())}"
            )
        return x.as_strided((b, s * heads, d), (x.stride(0) if b > 1 else max(s, 1) * heads * d, d, 1)), b, s * heads
    raise ValueError(f"x must be [T, d], [B, S, d], [T, heads, d] or [B, S, heads, d], got {x.ndim}-D")


def _rope_table(name: str, t: torch.Tensor, n_batch: int, n_tok: int, pairs: int) -> torch.Tensor:
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be fp32, got {t.dtype}")
    if t.ndim == 2:
        if tuple(t.shape) != (n_tok, pairs) or t.stride(1) != 1 or (n_tok > 1 and t.stride(0) != pairs):
            raise ValueError(f"{name} must be a compact [Tok={n_tok}, rope_dim//2={pairs}] table, got shape {tuple(t.shape)} strides {tuple(t.stride())}")
        return t.as_strided((n_batch, n_tok, pairs), (0, pairs, 1))
    if t.ndim == 3:
        if tuple(t.shape) != (n_batch, n_tok, pairs) or t.stride(2) != 1 or (n_tok > 1 and t.stride(1) != pairs):
            raise ValueError(
                f"{name} must be [B={n_batch}, Tok={n_tok}, rope_dim//2={pairs}] with compact rows, got shape {tuple(t.shape)} strides {tuple(t.stride())}"
            )
        return t.as_strided((n_batch, n_tok, pairs), (t.stride(0) if n_batch > 1 else max(n_tok, 1) * pairs, pairs, 1))
    raise ValueError(f"{name} must be [Tok, rope_dim//2] or [B, Tok, rope_dim//2], got {t.ndim}-D")


def check_operands(
    recipe: NormRopeRecipe, x, w, cos, sin, *, lse_in=None, sink=None, lse_out=None, n_rows_max: Optional[int] = None, device: Optional[int] = None
) -> _Operands:
    """Every operand against the recipe, BOTH directions, typed -- the host half of ``run`` (no device work).

    A norm-on artifact bound to ``None`` weights dereferences a null pointer; a norm-off one handed a weight
    would silently ignore it.  The same holds for the RoPE tables and the three LSE-fold operands.  Every bound
    operand must also live on ``x``'s device: a host (or other-GPU) weight passes every dtype / shape / stride /
    alignment gate and the launch then dereferences a foreign pointer -- an asynchronous illegal-address fault
    on a later sync instead of the ValueError promised here.
    """
    x3, n_batch, n_rows = _row_space(recipe, x)
    n_tok = n_rows // recipe.heads
    if x.data_ptr() % ACCESS_BYTES != 0 or (n_batch > 1 and (x3.stride(0) * x.element_size()) % ACCESS_BYTES != 0):
        raise ValueError("x must be 16-byte aligned (base pointer and batch stride) for the vectorised accesses")
    if device is not None and (x.device.type != "cuda" or (x.device.index if x.device.index is not None else torch.cuda.current_device()) != device):
        raise ValueError(f"x lives on {x.device}; this kernel was built for cuda:{device}")
    for name, t in (("w", w), ("cos", cos), ("sin", sin), ("lse_in", lse_in), ("sink", sink), ("lse_out", lse_out)):
        if t is not None and t.device != x.device:
            raise ValueError(f"{name} lives on {t.device}, x on {x.device}; every operand must share x's device")
    if n_rows_max is not None and n_batch * n_rows > n_rows_max:
        raise ValueError(f"x has {n_batch * n_rows} rows (B * Tok * heads), above the n_rows_max={n_rows_max} this kernel was built for")

    if recipe.apply_norm:
        if w is None:
            raise ValueError("this recipe applies the RMSNorm (apply_norm=True); the [d] weight must be bound")
        if w.dtype != recipe.io_dtype or tuple(w.shape) != (recipe.d,) or not w.is_contiguous():
            raise ValueError(f"w must be a contiguous {recipe.io_dtype} [d={recipe.d}] vector, got {w.dtype} {tuple(w.shape)}")
        if w.data_ptr() % ACCESS_BYTES != 0:
            raise ValueError("w must be 16-byte aligned")
    elif w is not None:
        raise ValueError("this recipe applies no RMSNorm (apply_norm=False); pass w=None -- a weight here would be silently ignored")

    if recipe.rope_dim:
        if cos is None or sin is None:
            raise ValueError(f"this recipe rotates rope_dim={recipe.rope_dim}; cos and sin must be bound")
        cos3 = _rope_table("cos", cos, n_batch, n_tok, recipe.rope_pairs)
        sin3 = _rope_table("sin", sin, n_batch, n_tok, recipe.rope_pairs)
        if cos.data_ptr() % ACCESS_BYTES != 0 or sin.data_ptr() % ACCESS_BYTES != 0:
            raise ValueError("cos/sin must be 16-byte aligned")
    else:
        if cos is not None or sin is not None:
            raise ValueError("this recipe has rope_dim=0; pass cos=sin=None -- tables here would be silently ignored")
        cos3 = sin3 = None

    if recipe.lse_fold:
        if lse_in is None or sink is None or lse_out is None:
            raise ValueError("this recipe folds the sink into the LSE (lse_fold=True); lse_in, sink and lse_out must all be bound")
        for name, t in (("lse_in", lse_in), ("sink", sink), ("lse_out", lse_out)):
            if t.dtype != torch.float32:
                raise ValueError(f"{name} must be fp32, got {t.dtype}")
        heads = recipe.heads
        if lse_in.ndim == 2:
            if tuple(lse_in.shape) != (n_batch * n_tok, heads) or not lse_in.is_contiguous():
                raise ValueError(
                    f"lse_in must be a contiguous [T={n_batch * n_tok}, heads={heads}] (or [B, Tok, heads]), got {tuple(lse_in.shape)} strides {tuple(lse_in.stride())}"
                )
            lse2 = lse_in.as_strided((n_batch, n_rows), (max(n_rows, 1), 1))
        elif lse_in.ndim == 3:
            if tuple(lse_in.shape) != (n_batch, n_tok, heads) or lse_in.stride(2) != 1 or (n_tok > 1 and lse_in.stride(1) != heads):
                raise ValueError(
                    f"lse_in must be [B={n_batch}, Tok={n_tok}, heads={heads}] with compact rows, got {tuple(lse_in.shape)} strides {tuple(lse_in.stride())}"
                )
            lse2 = lse_in.as_strided((n_batch, n_rows), (lse_in.stride(0) if n_batch > 1 else max(n_rows, 1), 1))
        else:
            raise ValueError(f"lse_in must be [T, heads] or [B, Tok, heads], got {lse_in.ndim}-D")
        if tuple(sink.shape) != (heads,) or not sink.is_contiguous():
            raise ValueError(f"sink must be a contiguous fp32 [heads={heads}], got {tuple(sink.shape)}")
        if lse_out.ndim != 3 or tuple(lse_out.shape) != (n_batch, heads, n_tok) or lse_out.stride(2) != 1:
            raise ValueError(
                f"lse_out must be [B={n_batch}, heads={heads}, Tok={n_tok}] with a contiguous last dim, got {tuple(lse_out.shape)} strides {tuple(lse_out.stride())}"
            )
        lse_out3 = lse_out.as_strided(
            (n_batch, heads, n_tok), (lse_out.stride(0) if n_batch > 1 else heads * max(n_tok, 1), lse_out.stride(1) if heads > 1 else max(n_tok, 1), 1)
        )
        for name, t in (("lse_in", lse_in), ("sink", sink), ("lse_out", lse_out)):
            if t.data_ptr() % 4 != 0:
                raise ValueError(f"{name} must be 4-byte aligned")
    else:
        if lse_in is not None or sink is not None or lse_out is not None:
            raise ValueError("this recipe has lse_fold=False; pass lse_in=sink=lse_out=None -- they would be silently ignored")
        lse2 = lse_out3 = None
    return _Operands(x3, w, cos3, sin3, lse2, sink, lse_out3, n_batch, n_rows, n_tok)


def build_norm_rope(
    recipe: NormRopeRecipe,
    *,
    n_rows_max: int,
    device,
    eps: float = 1e-20,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    compile_options: str = DEFAULT_COMPILE_OPTIONS,
) -> NormRopeKernel:
    """Compile (cached) from the RECIPE alone -- no allocation, no launch -- and bind the result to ``device``.

    ``n_rows_max`` is the block's declared bound on ``B * Tok * heads`` per launch (``run`` rejects more with a
    ``ValueError``); the artifact itself is shape-agnostic (symbolic extents + runtime row counts).  ``eps`` is
    the RMSNorm epsilon (``M:641`` / ``M:644``: ``1e-20``).  ``fake_quant`` needs the sm_100+ ``ue8m0`` cvt and
    is a typed ``NotImplementedError`` below that, unless ``compile_options`` names a ``--gpu-arch`` (the
    trace-compile-on-any-box dev knob: ``"--enable-tvm-ffi --gpu-arch sm_107a"``).
    """
    validate_shape(recipe, threads_per_cta)
    if not isinstance(n_rows_max, int) or isinstance(n_rows_max, bool) or n_rows_max <= 0:
        raise ValueError(f"n_rows_max must be a positive int, got {n_rows_max!r}")
    if not isinstance(rows_per_group, int) or rows_per_group <= 0:
        raise ValueError(f"rows_per_group must be a positive int, got {rows_per_group!r}")
    if not (eps > 0.0):
        raise ValueError(f"eps must be > 0, got {eps}")
    dev = resolve_device(device)
    if recipe.fake_quant and "--gpu-arch" not in compile_options and compute_capability(dev) < _MIN_CC_FAKE_QUANT:
        raise NotImplementedError(
            f"fake_quant needs the ue8m0 / e4m3 conversions of sm_100+ (cvt.rp.satfinite.ue8m0x2.f32); cuda:{dev} is cc {compute_capability(dev)}"
        )
    global _FAKE_STREAM
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (recipe, int(threads_per_cta), int(rows_per_group), compile_options, dev)
    if key not in compiled_cache:
        f32 = cutlass.Float32
        x = _fake_batched_rows(recipe.io_dtype, recipe.d)
        w = (
            cute.runtime.make_fake_compact_tensor(dtype=_convert_to_cutlass_data_type(recipe.io_dtype), shape=(recipe.d,), assumed_align=16)
            if recipe.apply_norm
            else None
        )
        tables = [_fake_batched_rows(torch.float32, recipe.rope_pairs) if recipe.rope_dim else None for _ in range(2)]
        if recipe.lse_fold:
            lse_in = cute.runtime.make_fake_tensor(dtype=f32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(), 1), assumed_align=4)
            sink = cute.runtime.make_fake_compact_tensor(dtype=f32, shape=(recipe.heads,), assumed_align=4)
            lse_out = cute.runtime.make_fake_tensor(
                dtype=f32, shape=(cute.sym_int(), cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(), cute.sym_int(), 1), assumed_align=4
            )
        else:
            lse_in = sink = lse_out = None
        compiled_cache[key] = cute.compile(
            norm_rope_launch,
            x,
            w,
            *tables,
            lse_in,
            sink,
            lse_out,
            cutlass.Int32(0),  # n_rows      ) runtime; the zeros only pin the TYPE at trace time
            cutlass.Float32(eps),  # eps
            cutlass.Int32(0),  # n_blocks_x  )
            cutlass.Int32(0),  # n_batch     )
            int(recipe.d),
            int(recipe.heads),
            int(recipe.rope_dim),
            bool(recipe.rope_inverse),
            bool(recipe.fake_quant),
            bool(recipe.three_roundings),
            bool(recipe.tail_only),
            int(recipe.lanes_per_row),
            int(recipe.chunks),
            int(threads_per_cta),
            int(rows_per_group),
            _FAKE_STREAM,
            options=compile_options,
        )
    return NormRopeKernel(
        recipe=recipe,
        compiled=compiled_cache[key],
        device=dev,
        n_rows_max=int(n_rows_max),
        eps=float(eps),
        threads_per_cta=int(threads_per_cta),
        rows_per_group=int(rows_per_group),
    )


def moved_bytes(recipe: NormRopeRecipe, n_tokens: int, n_batch: int = 1) -> int:
    """HBM traffic of one launch -- the denominator for a GB/s figure.  Rows are read once and written once
    (only the tail on the tail map); an LSE fold reads and writes one fp32 per row.  cos/sin and ``w`` are
    excluded: ``[Tok, rope_dim//2]`` and ``[d]``, L2-resident across the heads of a token."""
    rows = n_batch * n_tokens * recipe.heads
    width = recipe.rope_dim if recipe.tail_only else recipe.d
    bpe = torch.tensor([], dtype=recipe.io_dtype).element_size()
    return 2 * rows * width * bpe + (8 * rows if recipe.lse_fold else 0)


# ---------------------------------------------------------------------------
# The torch mirror
# ---------------------------------------------------------------------------

_FP8_MAX_INV_F32 = torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32)


def _ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2 x)`` for positive fp32 -- K:22-37 (``fast_log2_ceil`` + ``fast_pow2``) without exp2/log2."""
    m, e = torch.frexp(x)
    k = e - (m == 0.5).to(e.dtype)
    return torch.ldexp(torch.ones_like(x), k)


def fake_quant_reference(y: torch.Tensor, block: int = QUANT_BLOCK) -> torch.Tensor:
    """K:41-124 in torch on the LAST dim (fp32 in, ``y.dtype`` out): ``amax = max(max|x|, 1e-4)``,
    ``s = ceil_pow2(amax * fp32(1/448))``, ``e4m3_rne_sat(x / s) * s``."""
    yb = y.float().unflatten(-1, (-1, block))
    amax = yb.abs().amax(-1, keepdim=True).clamp_min(AMAX_FLOOR)
    s = _ceil_pow2(amax * _FP8_MAX_INV_F32.to(amax.device))
    q = (yb / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).to(y.dtype).flatten(-2)


def lse_fold_reference(lse: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """``select(lse == +inf, sink, select(sink == +inf, +inf, logaddexp(lse, sink)))`` (plan 2 D1), broadcasting."""
    inf = torch.tensor(float("inf"), dtype=torch.float32, device=lse.device)
    live = torch.where(sink == inf, inf, torch.logaddexp(lse.float(), sink.float()))
    return torch.where(lse == inf, sink.float().expand_as(live), live)


def norm_rope_reference(recipe: NormRopeRecipe, x, w, cos, sin, *, eps: float = 1e-20, lse_in=None, sink=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-torch twin of :meth:`NormRopeKernel.run`, OUT OF PLACE: ``(y, lse_out)`` with ``y`` shaped like ``x``
    and ``lse_out`` fp32 ``[B, heads, Tok]`` (``None`` unless ``recipe.lse_fold``).

    The model's op order with its three bf16 materialisations (``three_roundings``): RMSNorm as M:289-293
    (``x.float()``, ``mean``, ``rsqrt``, ``weight * x``, ``.to(dtype)``), RoPE as M:392-406 (``view_as_complex``
    on adjacent pairs, ``freqs_cis.conj()`` for the inverse, one rounding on ``copy_``), fake-quant as
    K:41-124.  ``three_roundings=False`` keeps the chain in fp32 and rounds once.  Accepts exactly the operand
    forms ``run`` accepts.
    """
    x3, n_batch, n_rows = _row_space(recipe, x)
    n_tok = n_rows // recipe.heads
    # The same operand checks as ``run`` (both directions); the fold's output is allocated here.
    lse_out = torch.empty(n_batch, recipe.heads, n_tok, dtype=torch.float32, device=x.device) if recipe.lse_fold else None
    ops = check_operands(recipe, x, w, cos, sin, lse_in=lse_in, sink=sink, lse_out=lse_out)
    io = recipe.io_dtype
    y = ops.x.float()  # [B, R, d]
    if recipe.apply_norm:
        var = y.square().mean(-1, keepdim=True)  # M:291
        y = ops.w.float() * (y * torch.rsqrt(var + eps))  # M:292-293
        if recipe.three_roundings and (recipe.rope_dim or recipe.fake_quant):
            y = y.to(io).float()
    if recipe.rope_dim:
        fc = torch.complex(ops.cos.float(), ops.sin.float())  # [B, Tok, P]
        if recipe.rope_inverse:
            fc = fc.conj()  # M:399
        fc = fc.unsqueeze(2).expand(n_batch, n_tok, recipe.heads, recipe.rope_pairs).reshape(n_batch, n_rows, recipe.rope_pairs)
        rot = torch.view_as_complex(y[..., -recipe.rope_dim :].unflatten(-1, (-1, 2)).contiguous())  # M:397 adjacent pairs
        y = y.clone()
        y[..., -recipe.rope_dim :] = torch.view_as_real(rot * fc).flatten(-2)
        if recipe.three_roundings and recipe.fake_quant:
            y = y.to(io).float()
    if recipe.fake_quant:
        y = fake_quant_reference(y).float()
    out = y.to(io).reshape(x.shape) if recipe.touches_data else x.clone()
    if recipe.lse_fold:
        lse2 = ops.lse_in.reshape(n_batch, n_tok, recipe.heads).float()  # [B, Tok, heads]
        lse_out = lse_fold_reference(lse2, ops.sink.float().view(1, 1, recipe.heads)).permute(0, 2, 1).contiguous()  # [B, heads, Tok]
    return out, lse_out


frost_norm_rope.set_name_prefix("cudnn", remove_cutlass_symbol=True)
