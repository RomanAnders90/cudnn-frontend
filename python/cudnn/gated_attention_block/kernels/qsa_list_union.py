# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_list_union``: the per-row block lists of ONE sequence merged into ONE canonical UNION list with a per-row membership
nibble per block -- the input contract of the block-sparse d256 forward's UNION rendering (one work item per (sequence, KV head,
split chunk) that gathers the union of a sequence's ``S_q`` lists once and computes every row's columns against it).

Contract (the host entry ``gated_attention_block/qsa_list_union.py`` checks the FORM; this module documents the function):

* in: ``block_ids [B, S_q, top_k]`` int32 row-contiguous (viewed ``[B, S_q x top_k]``; ANY order inside a row; ``-1`` = no entry),
  ``block_lens [B, S_q]`` int32 or None (an entry of row ``j`` at index ``>= clamp(block_lens[b, j], 0, top_k)`` is ignored; None =
  every non-negative entry counts), ``S_q <= UNION_MAX_ROWS = 4``, ``top_k <= 512``, ``S_q x top_k <= UNION_LIST_WIDTH = 2048``, every
  id ``< 2^UNION_ID_BITS`` (the key ``(id << 2) | j`` and the word ``(id << 4) | mask`` must fit an int32; the block's declaration bounds the
  ids by its table, the host entry declines a wider ``n_blocks_max`` by name);
* out: ``union_ids [B, 2048]`` int32 WORDS ``(blk << 4) | mask`` in STRICTLY DESCENDING block order over ``[0, union_lens[b])``, ``-1``
  past it (every one of the 2048 slots is written: a word is never left to a previous launch); ``union_lens [B]`` int32 = the number
  of DISTINCT listed blocks (``<= S_q x top_k``); ``mask`` bit ``j`` is set iff row ``j`` lists ``blk``.  A word's mask is never 0.
* two launches are bitwise (no atomic decides a word; the sort is a fixed network, the ranks come from a scan); duplicates WITHIN a
  row collapse to one entry (the per-row tile would attend such a block twice -- a list contract violation the test tree's
  ``block_ids_contract_violations`` detector rejects; here the union holds the block once with the row's bit set once).

Algorithm (ONE ``THREADS`` = 1024-thread CTA per sequence, grid ``(B, 1, 1)``; every site a ``bar.sync 0`` reached by EVERY thread of
the CTA, ``SUM(issuing lanes) == THREADS`` at every row -- the decode top-k kernel's barrier discipline, no mbarrier):

1. thread ``t`` loads entries ``e = t`` and ``e = t + THREADS`` of the row block's ``S_q x top_k`` flat entries (``j = e // top_k``, ``i = e %
   top_k``; an ``e`` past ``S_q x top_k`` is a dummy) and forms the KEY ``(id << 2) | j`` -- a dummy (``-1``, or ``i`` at or past the row's
   count) takes ``INT32_MIN``, which sorts LAST in the descending order below -- into ``s_keys[e]`` (8 KiB);
2. a bitonic sort of the 2048 keys in SMEM, DESCENDING: 66 compare-exchange stages (``k`` = 2 .. 2048, ``jj`` = k/2 .. 1), one pair per
   thread per stage (``i = (t // jj) x 2 jj + t % jj``, ``l = i + jj``), one barrier per stage;
3. a run of equal ids is adjacent (the key sorts by id, then by ``j``): ``head(p) = valid(p) & (p == 0 | id(p - 1) != id(p))``; the head
   ORs ``1 << j`` over its run (``while`` the next position holds the same id: at most ``S_q - 1`` successors on a contract-clean list, the
   row's duplicates on a violating one -- a dummy never matches a valid id);
4. an exclusive scan of the head flags in POSITION order (the two halves ``[0, 1024)`` and ``[1024, 2048)`` scanned separately -- a
   warp inclusive scan + the cross-warp prefix through ``s_wsum`` -- the upper half offset by the lower half's total) gives every head its
   rank ``r``; the head writes ``(id << 4) | mask`` at ``union_ids[b, r]``, every thread writes ``-1`` at the slots ``t`` / ``t + THREADS`` at or
   past ``|U|``, thread 0 writes ``union_lens[b] = |U|``.

Barrier table (``bar.sync 0``, every thread, once per visit):

| site | when | what it orders (writes before -> reads after) |
|---|---|---|
| B0 | after the loads | ``s_keys[e]`` (two per thread) -> the sort's first stage |
| S_1 .. S_66 | after each compare-exchange stage | the stage's stores -> the next stage's loads (the last one -> the run detection's reads of ``s_keys[p - 1]`` / ``s_keys[p + r]``) |
| B2 | after the warp sums | ``s_wsum[0 .. 31]`` / ``s_wsum[32 .. 63]`` (lane 31 of every warp) -> the cross-warp prefix every thread reads |

SMEM table (dynamic shared memory through ``SmemAllocator``, ``SMEM_BYTES``; no descriptor reads any buffer, so each row states the per-lane stride that keeps the banks
spread): ``s_keys`` int32 x 2048 -- the loads / the final writes at ``t`` and ``t + 1024`` (4 B between lanes: conflict-free), the sort
stages at ``i`` / ``i + jj`` (``jj >= 32``: consecutive lanes consecutive words; ``jj < 32``: lanes ``2 jj`` apart inside a 64-word span:
a 2-way conflict at ``jj = 16``, 4-way at 8 ... the standard bitonic-in-SMEM cost, 15 of the 66 stages); ``s_wsum`` int32 x 64 -- one word
per warp per half (lane 31 writes, every lane reads slot ``lane``: conflict-free).

Degenerate inputs, each with its handling: every entry ``-1`` or every count 0 -> no head, ``|U| = 0``, every slot ``-1``; ``S_q x top_k <
2048`` -> the entries past it are dummies (sorted last, never heads); disjoint lists at ``S_q x top_k = 2048`` -> ``|U| = 2048``, no
padding; identical lists -> ``|U| = top_k`` with every row's bit; a duplicate inside a row -> one head, the bit set once; ``block_lens``
out of range -> clamped to ``[0, top_k]`` on device; a count below the row's valid prefix -> the entries at or past it ignored (dummies);
``B = 0`` -> the host entry returns before any launch.

Cost: ~2-4 us at ``B`` 32 (the 66 barrier stages of one 1024-thread CTA dominate; the CTAs run in parallel) -- the launch the union
route spends to gather a sequence's KV once instead of once per row.  Rule 7: this module imports the CuTe DSL at module scope and is
imported lazily by the host entry once the DSL floor is met.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn._cutlass_compat import SmemAllocator
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.device import compute_capability, current_device

__all__ = [
    "UNION_LIST_WIDTH",
    "UNION_MAX_ROWS",
    "UNION_ID_BITS",
    "UNION_TOP_K_MAX",
    "THREADS",
    "QsaListUnionRecipe",
    "compile_qsa_list_union",
    "run_qsa_list_union",
    "qsa_list_union",
]

UNION_LIST_WIDTH = 2048
"""The union list's width in words = the CTA's sort capacity: ``S_q x top_k <= 2048`` (4 rows x 512)."""
UNION_MAX_ROWS = 4
"""The rows per sequence the mask nibble carries (bits 0..3 of a word)."""
UNION_ID_BITS = 27
"""Block ids below ``2^27`` keep the key ``(id << 2) | j`` and the word ``(id << 4) | mask`` inside an int32."""
UNION_TOP_K_MAX = 512
THREADS = 1024
SMEM_BYTES = UNION_LIST_WIDTH * 4 + 2 * (THREADS // 32) * 4 + 128
"""Dynamic shared memory of one CTA: the 2048 keys, the two per-half warp-sum rows, the allocator's slack."""
_FULL_MASK = 0xFFFFFFFF
_INT32_MIN = -(1 << 31)
_FAKE_STREAM = None
compiled_cache = {}
# Imported normally (not through the template loader), so the module names its own source identity for the persistent compiled-plan cache.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def validate_union_geometry(s_q: int, top_k: int) -> None:
    """Raise ``ValueError`` on a geometry the kernel cannot hold (never an ``assert``: user-facing)."""
    if not 1 <= int(s_q) <= UNION_MAX_ROWS:
        raise ValueError(f"qsa_list_union: s_q must be in [1, {UNION_MAX_ROWS}] (the mask nibble carries four rows), got {s_q}")
    if not 1 <= int(top_k) <= UNION_TOP_K_MAX:
        raise ValueError(f"qsa_list_union: top_k must be in [1, {UNION_TOP_K_MAX}], got {top_k}")
    if int(s_q) * int(top_k) > UNION_LIST_WIDTH:
        raise ValueError(f"qsa_list_union: s_q x top_k = {int(s_q) * int(top_k)} exceeds the union capacity {UNION_LIST_WIDTH}")


@cute.jit
def _warp_scan_incl(v, lane):
    """Branch-free inclusive warp scan of ``v`` (Int32) in lane order: shuffles up by 1, 2, 4, 8, 16 with arithmetic selects."""
    incl = v
    for s in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(incl, 1 << s, mask_and_clamp=0)
        incl = incl + (other if lane >= cutlass.Int32(1 << s) else cutlass.Int32(0))
    return incl


@cute.jit
def _cross_warp_excl(s_wsum, base: cutlass.Constexpr[int], lane, warp, nwarps: cutlass.Constexpr[int]):
    """After a barrier that published ``s_wsum[base + w]`` for every warp ``w < nwarps``: this warp's EXCLUSIVE prefix over the warp
    sums (every thread reads slot ``lane``, scans in-warp, takes slot ``warp`` with one index shuffle) and the total over every warp."""
    idx = lane if lane < cutlass.Int32(nwarps) else cutlass.Int32(0)
    v = s_wsum[cutlass.Int32(base) + idx]
    v = v if lane < cutlass.Int32(nwarps) else cutlass.Int32(0)
    incl = _warp_scan_incl(v, lane)
    excl = incl - v
    mine = cute.arch.shuffle_sync(excl, warp, mask_and_clamp=31)
    total = cute.arch.shuffle_sync(incl, cutlass.Int32(nwarps - 1), mask_and_clamp=31)
    return mine, total


@cute.kernel
def frost_qsa_list_union(
    mIds: cute.Tensor,  # [B, S_q x top_k] int32; row stride symbolic, unit column stride
    mLens: Optional[cute.Tensor],  # [B, S_q] int32 contiguous, or None (every non-negative entry counts)
    mOut: cute.Tensor,  # [B, UNION_LIST_WIDTH] int32 contiguous
    mOutLens: cute.Tensor,  # [B] int32 contiguous
    s_q: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
) -> None:
    """Grid ``(B, 1, 1)``: ``block_idx.x`` is the sequence; one 1024-thread CTA per sequence (the module docstring's algorithm)."""
    n_entries = cutlass.const_expr(s_q * top_k)
    nwarps = cutlass.const_expr(THREADS // 32)
    width = cutlass.const_expr(UNION_LIST_WIDTH)
    smem = SmemAllocator()
    s_keys = smem.allocate_tensor(cutlass.Int32, cute.make_layout((width,)), byte_alignment=16)
    s_wsum = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2 * nwarps,)), byte_alignment=16)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = cutlass.Int32(cute.arch.lane_idx())
    warp = cutlass.Int32(cute.arch.make_warp_uniform(tidx // cutlass.Int32(32)))
    b = cutlass.Int32(cute.arch.block_idx()[0])
    zero = cutlass.Int32(0)
    one = cutlass.Int32(1)
    minus_one = cutlass.Int32(-1)
    int_min = cutlass.Int32(_INT32_MIN)
    ids = cutlass.make_array_view(mIds)
    out = cutlass.make_array_view(mOut)
    out_lens = cutlass.make_array_view(mOutLens)

    # --- (1) the keys: two entries per thread; a dummy takes INT32_MIN (sorts last in the descending order)
    for half in cutlass.range_constexpr(2):
        e = tidx + cutlass.Int32(half * THREADS)
        key = int_min
        if cutlass.const_expr(half * THREADS < n_entries):
            if e < cutlass.Int32(n_entries):
                j = e // cutlass.Int32(top_k)
                i = e - j * cutlass.Int32(top_k)
                ident = cutlass.Int32(ids[b, e])
                valid = ident >= zero
                if cutlass.const_expr(mLens is not None):
                    lens = cutlass.make_array_view(mLens)
                    cnt = cutlass.Int32(lens[b, j])
                    cnt = cute.math.min(cute.math.max(cnt, zero), cutlass.Int32(top_k))
                    valid = valid & (i < cnt)
                key = ((ident << cutlass.Int32(2)) | j) if valid else int_min
        s_keys[e] = key
    cute.arch.barrier()  # B0

    # --- (2) the bitonic sort, descending: 66 stages, one compare-exchange per thread per stage
    for k_log in cutlass.range_constexpr(1, 12):
        k = 1 << k_log
        for step in cutlass.range_constexpr(k_log):
            jj = 1 << (k_log - 1 - step)
            i = (tidx // cutlass.Int32(jj)) * cutlass.Int32(2 * jj) + (tidx % cutlass.Int32(jj))
            l = i + cutlass.Int32(jj)
            a = s_keys[i]
            c = s_keys[l]
            desc = (i & cutlass.Int32(k)) == zero
            swap = (a < c) if desc else (a > c)
            lo = c if swap else a
            hi = a if swap else c
            s_keys[i] = lo
            s_keys[l] = hi
            cute.arch.barrier()  # S_k,jj

    # --- (3) the run heads and their masks, (4) the ranks: the two halves scanned in POSITION order
    heads = []
    words = []
    for half in cutlass.range_constexpr(2):
        p = tidx + cutlass.Int32(half * THREADS)
        key = s_keys[p]
        valid = key != int_min
        ident = key >> cutlass.Int32(2)
        prev_same = cutlass.Boolean(False)
        if p > zero:
            prev_same = (s_keys[p - one] >> cutlass.Int32(2)) == ident
        head = valid & (~prev_same)
        mask = cutlass.Int32(1) << (key & cutlass.Int32(3))
        r = one
        if head:
            # the successors of the same id (at most s_q - 1 on a contract-clean list; a row's duplicates on a violating one); a dummy's id
            # (INT32_MIN >> 2) never equals a valid one
            while (p + r < cutlass.Int32(width)) and ((s_keys[p + r] >> cutlass.Int32(2)) == ident):
                mask = mask | (cutlass.Int32(1) << (s_keys[p + r] & cutlass.Int32(3)))
                r = r + one
        heads.append(head)
        words.append((ident << cutlass.Int32(4)) | mask)
    flag_lo = one if heads[0] else zero
    flag_hi = one if heads[1] else zero
    incl_lo = _warp_scan_incl(flag_lo, lane)
    incl_hi = _warp_scan_incl(flag_hi, lane)
    if lane == cutlass.Int32(31):
        s_wsum[warp] = incl_lo
        s_wsum[cutlass.Int32(nwarps) + warp] = incl_hi
    cute.arch.barrier()  # B2
    carry_lo, total_lo = _cross_warp_excl(s_wsum, 0, lane, warp, nwarps)
    carry_hi, total_hi = _cross_warp_excl(s_wsum, nwarps, lane, warp, nwarps)
    total = total_lo + total_hi
    rank_lo = carry_lo + incl_lo - flag_lo
    rank_hi = total_lo + carry_hi + incl_hi - flag_hi
    if heads[0]:
        out[b, rank_lo] = words[0]
    if heads[1]:
        out[b, rank_hi] = words[1]
    # the padding: every slot at or past |U| takes -1 (slot t by thread t, slot t + 1024 by thread t)
    for half in cutlass.range_constexpr(2):
        slot = tidx + cutlass.Int32(half * THREADS)
        if slot >= total:
            out[b, slot] = minus_one
    if tidx == zero:
        out_lens[b] = total


frost_qsa_list_union.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_list_union_launch(
    ids: cute.Tensor,
    lens: Optional[cute.Tensor],
    out: cute.Tensor,
    out_lens: cute.Tensor,
    rows: cutlass.Int32,
    s_q: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    kernel = frost_qsa_list_union(ids, lens, out, out_lens, s_q, top_k)
    kernel.launch(grid=(rows, 1, 1), block=(THREADS, 1, 1), smem=SMEM_BYTES, stream=stream)


class QsaListUnionRecipe(NamedTuple):
    """Build-time facts of one artifact: ``s_q`` / ``top_k`` and the ``block_lens`` presence are compile-time constants of the kernel;
    the sequence count and the row stride enter as runtime tensor shapes."""

    compiled: object
    s_q: int
    top_k: int
    has_lens: bool
    device: int


def _fake_rows(dtype):
    """``[rows, cols]`` with a symbolic row stride and a unit column stride, 16-B aligned base."""
    return cute.runtime.make_fake_tensor(dtype=dtype, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(), 1), assumed_align=16)


def compile_qsa_list_union(*, s_q: int, top_k: int, has_lens: bool) -> QsaListUnionRecipe:
    """Build the artifact from the geometry ALONE -- no device allocation, no launch.  Order of the gates (Rule 7 first): the CuTe DSL
    floor, the geometry, the DSL's target for the current device.  ``NotImplementedError`` names the version or the target;
    ``ValueError`` names the malformed geometry."""
    global _FAKE_STREAM
    too_old = cutedsl_requirement_error("qsa_list_union")
    if too_old is not None:
        raise NotImplementedError(too_old)
    s_q, top_k, has_lens = int(s_q), int(top_k), bool(has_lens)
    validate_union_geometry(s_q, top_k)
    device = current_device()
    cc = compute_capability(device)
    arch_problem = cutedsl_arch_requirement_error(cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (s_q, top_k, has_lens, device)
    if key not in compiled_cache:
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_list_union")
        compiled_cache[key] = compile_cached(
            qsa_list_union_launch,
            _fake_rows(cutlass.Int32),
            (
                cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=4)
                if has_lens
                else None
            ),
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=16),
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4),
            cutlass.Int32(1),  # rows: runtime, the literal pins the type
            s_q,
            top_k,
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_qsa_list_union",
        )
    return QsaListUnionRecipe(compiled_cache[key], s_q, top_k, has_lens, device)


def run_qsa_list_union(
    r: QsaListUnionRecipe, block_ids: torch.Tensor, block_lens: Optional[torch.Tensor], out_ids: torch.Tensor, out_lens: torch.Tensor, *, stream
) -> None:
    """The lowered launch: form checks and the launch -- no conversion, no allocation, no torch op (Rules 1 / 3 / 5).  ``block_ids [B,
    S_q, top_k]`` (or ``[B, S_q x top_k]``) int32 with contiguous rows, ``block_lens [B, S_q]`` int32 contiguous iff the recipe has them,
    ``out_ids [B, 2048]`` / ``out_lens [B]`` int32 contiguous, all on the recipe's device; ``stream`` is the raw handle (``None`` = torch's
    current stream of that device)."""
    if not isinstance(block_ids, torch.Tensor) or block_ids.dtype != torch.int32 or not block_ids.is_cuda:
        raise ValueError("block_ids must be an int32 CUDA tensor")
    rows = int(block_ids.shape[0])
    flat = block_ids.view(rows, -1) if block_ids.dim() == 3 else block_ids
    if flat.dim() != 2 or int(flat.shape[1]) != r.s_q * r.top_k or flat.stride(1) != 1:
        raise ValueError(
            f"block_ids must be [B, {r.s_q}, {r.top_k}] (or [B, {r.s_q * r.top_k}]) with contiguous rows for this artifact, got {tuple(block_ids.shape)}"
        )
    if flat.data_ptr() % 16 != 0 or (flat.stride(0) * 4) % 16 != 0:
        raise ValueError("block_ids must be 16-B aligned with a 16-B-aligned row stride")
    device = block_ids.device
    if r.has_lens:
        if (
            not isinstance(block_lens, torch.Tensor)
            or block_lens.dtype != torch.int32
            or tuple(block_lens.shape) != (rows, r.s_q)
            or not block_lens.is_contiguous()
        ):
            raise ValueError(
                f"block_lens must be a contiguous int32 [{rows}, {r.s_q}] tensor for this artifact, got {None if block_lens is None else (tuple(block_lens.shape), block_lens.dtype)}"
            )
        if block_lens.device != device:
            raise ValueError(f"block_lens must live on {device}, got {block_lens.device}")
    elif block_lens is not None:
        raise ValueError("this artifact was compiled without block_lens; compile with has_lens=True to pass them")
    for name, t, shape in (("out_ids", out_ids, (rows, UNION_LIST_WIDTH)), ("out_lens", out_lens, (rows,))):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous int32 {shape} tensor, got {None if not isinstance(t, torch.Tensor) else (tuple(t.shape), t.dtype)}")
        if t.device != device:
            raise ValueError(f"{name} must live on {device}, got {t.device}")
    if out_ids.data_ptr() % 16 != 0:
        raise ValueError("out_ids must be 16-B aligned")
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    if dev_index != r.device:
        raise ValueError(f"the tensors live on device {dev_index} but the artifact was compiled for device {r.device} (compile per device)")
    if rows == 0:
        return
    handle = int(stream) if stream is not None else torch.cuda.current_stream(device).cuda_stream
    r.compiled(ids=flat, lens=block_lens if r.has_lens else None, out=out_ids, out_lens=out_lens, rows=cutlass.Int32(rows), stream=cuda.CUstream(handle))


def qsa_list_union(block_ids: torch.Tensor, block_lens: Optional[torch.Tensor], out_ids: torch.Tensor, out_lens: torch.Tensor, *, stream=None) -> torch.Tensor:
    """Compile (cached) and run once -- the convenience form for the host entry, the tests and the block.  ``block_ids [B, S_q, top_k]``
    int32, ``block_lens [B, S_q]`` int32 or None, ``out_ids [B, 2048]`` / ``out_lens [B]`` int32 -- the CALLER's buffers (nothing is
    allocated here); ``out_ids`` is returned for convenience."""
    too_old = cutedsl_requirement_error("qsa_list_union")
    if too_old is not None:
        raise NotImplementedError(too_old)
    if not isinstance(block_ids, torch.Tensor) or block_ids.dim() != 3:
        raise ValueError("block_ids must be a [B, S_q, top_k] int32 CUDA tensor")
    r = compile_qsa_list_union(s_q=int(block_ids.shape[1]), top_k=int(block_ids.shape[2]), has_lens=block_lens is not None)
    run_qsa_list_union(r, block_ids, block_lens, out_ids, out_lens, stream=stream)
    return out_ids
