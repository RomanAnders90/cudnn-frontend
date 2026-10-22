# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The KERNEL half of the fused decode indexer (``qsa_indexer_fused``): the prepare, the paged scorer and the decode top-k as ONE launch.

The host half -- the dispatch table, the counter / barrier / SMEM tables, the compile keys, the served entries ``compile_qsa_indexer_fused``
/ ``run_qsa_indexer_fused`` and their form checks -- lives in ``qsa_indexer_fused.py``, which imports no kernel module and gates the DSL
floor and the device's target BEFORE importing this one (Rule 7).  This module imports the CuTe DSL and the three role modules at import
time like its neighbours; nothing here is a public entry.

The body, role by role (the tables it implements are the front door's docstring; the dispatch arithmetic is :func:`dispatch_table`'s,
decoded on device from the three kernel scalars ``n_prep`` / ``n_chunks`` / ``s_q``):

* R0 (``blockIdx.x < n_prep``): ``prepare_role`` with the GLOBAL warp unit ``wid = blockIdx.x x (threads / 32) + warp`` and the query arm
  folded out (its three operands are ``None``: nobody reads ``ix_q`` under the fused form -- the scorer derives the rows itself); the P17
  pair sits AROUND the role call (the prepare kernel's own placement).  A warp unit past the last arm's count computes on clamped rows and
  stores nothing: the role's rule, no idle warp in the ledger sense.
* R1, the ``n_chunks`` CTAs of a sequence: the CTA's ``units`` 4-warp scorer units run ``score_role(chunk_id = k x units + unit, b,
  unit, ..., fused=True, derive=True, units)`` over ONE SMEM carve (the units' rings at ``unit x ring_elems``, one query tile, one ``sNew``
  row); the warps ``>= 4 x units`` take NO role (they skip the branch before any PDL site and before any barrier, so the named barriers of
  ids 1 and ``2 + units`` count ``128 x units`` exactly).  A DEAD unit -- the last CTA of a sequence whose ``chunk_id`` is at or past the
  standalone chunk count ``ceil(n_blocks_max / blocks_per_cta)`` (the chunk count is not a multiple of ``units``) -- scores nothing (its
  first key is at or past ``n_blocks_max >= n_vis_max``: a dead warp of the role), still joins the CTA-wide barriers its ledger counts
  (``128 x units`` threads), writes no counts (chunk 0's unit alone does) and the release stays ONE warp of the CTA.
* R2, the ``s_q`` CTAs behind them: ``topk_role(row = b x s_q + s, fused=True)`` with the row's counter word and the target ``n_chunks``;
  the acquire + reset seam (and its test-only ``check_arrivals`` / deadline arms) is the role's own.
* The two test-only mis-orders render only with the deadline (the front door refuses them without it): ``consumers_first`` puts a
  sequence's top-k CTAs ahead of its scorer CTAs inside the sequence's range; ``consumers_global`` puts EVERY sequence's top-k CTAs ahead
  of EVERY scorer CTA (``[R2 x T][R1 x B x n_chunks]``, decoded from ``n_tok``) -- the table that strands every resident slot on a
  consumer under in-order dispatch (the per-sequence twin never does: at most one sequence's consumers wait on undispatched producers),
  so it is the trap test's detector input.

The SMEM carve: the scorer's three buffers are declared INSIDE the R1 branch and the top-k's allocator runs inside the R2 branch; the launch
passes the plan's union (``dynamic_smem_bytes``) as ``smem`` and ``smem_merge_branch_allocs`` so the two mutually exclusive branches alias
the same bytes (a CTA runs ONE role).  No static SMEM, no mbarrier, no TMA, no cluster: the one new cross-CTA primitive is the per-row
counter class (``gsync``).  Every role branch is a ``Constexpr`` arm (the test-only ``roles`` key renders a role-isolated twin with the
same ABI and the same block / SMEM for the per-role register census).
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.QSA.selection_paged import BLOCK_SIZE, THREADS as SCORER_UNIT_THREADS

from .cache_write import _fake_pool, _fake_slot
from .qk_norm_rope import _fake, fake_rowmajor_dynamic_token_stride
from .qsa_compress import _fake_compact, _fake_i32_table, _fake_paged_state, _fake_strided
from .qsa_prepare import QsaPrepareIndexArms, prepare_role
from .qsa_score_paged import SNEW_ELEMS, _fake_f32_2d, _fake_i32_1d, _fake_kbar, score_role, score_smem_elems
from .qsa_topk_decode import _fake_rows, topk_role

__all__ = [
    "FROST_SOURCE_DIGEST",
    "FusedKey",
    "frost_qsa_indexer_fused",
    "qsa_indexer_fused_launch",
    "fused_fake_args",
    "compile_fused",
    "launch_fused",
]

_WARP = 32
_FAKE_STREAM = None
compiled_cache = {}
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins it with the
# compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers every other ``.py`` the
# kernel is traced from (the three role modules among them).
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------- the kernel


@cute.kernel
def frost_qsa_indexer_fused(
    # ---- the prepare role's operands (prepare_role's, in its order; the query arm's three slots are not taken -- the derive owns them)
    mQ: cute.Tensor,  # [T, H_q,  D] the slab's Q band (rewritten in place)
    mK: cute.Tensor,  # [T, H_kv, D] the K band (rewritten in place)
    mV: cute.Tensor,  # [T, H_kv, D] the V band
    mQo: cute.Tensor,  # [T, H_q,  D] out (aliases mQ in the served form)
    mKo: cute.Tensor,  # [T, H_kv, D] out (aliases mK)
    mWq: Optional[cute.Tensor],  # [D] or None (RoPE-only Q / K)
    mWk: Optional[cute.Tensor],
    mCos: Optional[cute.Tensor],  # [T, rope_dim] per-token rows, or None: the RoPE-table arm
    mSin: Optional[cute.Tensor],
    mKPool: cute.Tensor,  # [P, H_kv, page_size, D]
    mVPool: cute.Tensor,
    mSlot: cute.Tensor,  # [T] int32 / int64
    mIKsrc: cute.Tensor,  # [T, 1, D_i] the band's raw indexer key (the prepare's ring arm AND the derive's completing row read it)
    mIKpool: Optional[cute.Tensor],  # [P, 1, page_size, D_i] the raw key's pool, or None
    mRing: cute.Tensor,  # [blocks, cap, 1, D_i] the paged raw-key ring
    mComp: cute.Tensor,  # [num_pages, P / pool, 1, D_i] the compressed-key cache (the prepare writes the completing block)
    mWik: cute.Tensor,  # [D_i]
    mCosPos: cute.Tensor,  # [1 or B, S_pos, rope_dim]
    mSinPos: cute.Tensor,
    mNCommit: Optional[cute.Tensor],  # [B] int32 or None
    mCSlot: cute.Tensor,  # [B, S_q] int32 out: the prepare's slot report
    mKvLens: cute.Tensor,  # [B] int32: the KV length AFTER the step
    mBlockTable: cute.Tensor,  # [B, max_pages] int32: the MAIN block table (the prepare's cache rows, the scorer's pages, the derive's -2 predicate)
    mStateTable: cute.Tensor,  # [B, 1] int32: the ring's block table
    mRingSlots: Optional[cute.Tensor],  # [T] the stack's ring slots, or None (the derived rows)
    # ---- the derive's two operands the prepare role does not take under the fused form
    mIQsrc: cute.Tensor,  # [T, H_i, D_i] the band's indexer-query heads (token / head strides symbolic)
    mWiq: cute.Tensor,  # [D_i]
    # ---- the scorer role's operands
    mKbar: cute.Tensor,  # [num_pages, R, D_i] the compressed-key cache as the scorer reads it (a view of mComp)
    mLogits: cute.Tensor,  # [T, n_blocks_max] fp32
    mCounts: cute.Tensor,  # [T] int32
    mBlockLens: Optional[cute.Tensor],  # [T] int32 or None
    # ---- the top-k role's operands
    mLogitsBits: cute.Tensor,  # [T, n_blocks_max] the same logits as their int32 bits
    mOut: cute.Tensor,  # [T, top_k] int32
    # ---- runtime scalars
    n_tok: cutlass.Int32,  # T
    n_slots: cutlass.Int64,
    n_islots: cutlass.Int64,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    h_i: cutlass.Int32,
    eps: cutlass.Float32,
    eps_i: cutlass.Float32,
    w_offset_i: cutlass.Float32,
    scale: cutlass.Float32,  # the scorer's scale
    n_prep: cutlass.Int32,  # the dispatch table's R0 CTA count
    n_chunks: cutlass.Int32,  # the dispatch table's R1 CTAs per sequence = the counter target
    cnt_row_base: cutlass.Int64,  # the plan-owned counter words: cnt_row[t] at cnt_row_base + 8 t
    deadline_ns: cutlass.Int64,  # the test-only trap rendering's deadline (unread without `with_deadline`)
    diag_addr: cutlass.Int64,  # its diagnostic words (unread without `with_deadline`)
    # ---- compile keys: the prepare's
    d: cutlass.Constexpr[int],
    d_i: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    h_i_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    # ---- the scorer's
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    scorer_mma_batch: cutlass.Constexpr[int],  # the scorer's n-tiles per MMA batch under the fused register budget (0 = the standalone rule)
    scorer_l2_hint: cutlass.Constexpr[bool],  # the scorer's key-copy form (the hinted copy where the DSL's compiler serves it; the front door resolves it)
    # ---- the top-k's
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    # ---- the fused kernel's own
    units: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    consumers_first: cutlass.Constexpr[bool],  # test-only: the mis-ordered dispatch table (a sequence's top-k CTAs ahead of its scorer CTAs)
    consumers_global: cutlass.Constexpr[bool],  # test-only: EVERY sequence's top-k CTAs ahead of EVERY scorer CTA (the trap test's stranding table)
    check_arrivals: cutlass.Constexpr[bool],  # test-only: the top-k's exact-count trap before its reset
    with_deadline: cutlass.Constexpr[bool],  # test-only: the bounded poll + trap
    with_prepare: cutlass.Constexpr[bool],  # test-only role isolation (the served rendering has all three)
    with_scorer: cutlass.Constexpr[bool],
    with_topk: cutlass.Constexpr[bool],
) -> None:
    """One CTA = one ROLE decoded from ``blockIdx.x`` against the dispatch scalars (module docstring; the tables in ``qsa_indexer_fused.py``)."""
    warps_per_cta = cutlass.const_expr(threads // _WARP)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = cutlass.Int32(cute.arch.make_warp_uniform(tidx >> cutlass.Int32(5)))
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    zero_i = cutlass.Int32(0)
    s_q_i = cutlass.Int32(s_q)
    eight = cutlass.Int64(8)
    if bidx < n_prep:
        # ---------------- R0: the prepare (the attention rows + the raw-key / compress arm; no query arm) ----------------
        if cutlass.const_expr(with_prepare):
            if cutlass.const_expr(use_pdl):
                wait_on_dependent_grids()
            wid = bidx * cutlass.Int32(warps_per_cta) + warp
            prepare_role(
                mQ,
                mK,
                mV,
                mQo,
                mKo,
                mWq,
                mWk,
                mCos,
                mSin,
                mKPool,
                mVPool,
                mSlot,
                mIKsrc,
                mIKpool,
                None,  # mIQsrc: the query arm is the derive's (nobody reads ix_q under the fused form)
                None,  # mIQout
                None,  # mWiq
                mRing,
                mComp,
                mWik,
                mCosPos,
                mSinPos,
                None,  # mPos0: the paged form derives the positions from kv_lens
                mNCommit,
                mCSlot,
                mKvLens,
                mBlockTable,
                mStateTable,
                mRingSlots,
                n_tok,
                s_q_i,
                n_slots,
                n_islots,
                h_q,
                h_kv,
                h_i,
                eps,
                eps_i,
                w_offset_i,
                d,
                d_i,
                rope_dim,
                pool,
                page_size,
                slot_i64,
                h_q_ct,
                h_kv_ct,
                h_i_ct,
                const_head_counts,
                wid,
                lane,
                tidx,
            )
            if cutlass.const_expr(use_pdl):
                launch_dependent_grids()
    else:
        r = bidx - n_prep
        if cutlass.const_expr(consumers_global):
            # the test-only stranding table: [R2 x T][R1 x B x n_chunks] (every sequence's top-k CTAs ahead of every scorer CTA)
            is_scorer = r >= n_tok
            r_s = r - n_tok
            b_scorer = r_s // n_chunks
            chunk_k = r_s - b_scorer * n_chunks
            b_topk = r // s_q_i
            row_k = r - b_topk * s_q_i
        else:
            per_seq = n_chunks + s_q_i
            b_scorer = r // per_seq
            b_topk = b_scorer
            k = r - b_scorer * per_seq
            if cutlass.const_expr(consumers_first):
                is_scorer = k >= s_q_i
                chunk_k = k - s_q_i
                row_k = k
            else:
                is_scorer = k < n_chunks
                chunk_k = k
                row_k = k - n_chunks
        if is_scorer:
            # ---------------- R1: `units` scorer units over one SMEM carve; the warps past them take no role ----------------
            if cutlass.const_expr(with_scorer):
                ring_elems_u, tile_elems = score_smem_elems(h_i_ct, s_q, units_per_warp, ring_depth)
                sK = cutlass.Array(cutlass.Int32, units * ring_elems_u, alignment=128, space=cutlass.AddressSpace.smem)
                sA = cutlass.Array(cutlass.Int32, tile_elems, alignment=128, space=cutlass.AddressSpace.smem)
                sNew = cutlass.Array(cutlass.Int32, SNEW_ELEMS, alignment=16, space=cutlass.AddressSpace.smem)
                unit = tidx >> cutlass.Int32(7)
                tidx_in_unit = tidx & cutlass.Int32(SCORER_UNIT_THREADS - 1)
                warp_in_unit = tidx_in_unit >> cutlass.Int32(5)
                if unit < cutlass.Int32(units):
                    chunk_id = chunk_k * cutlass.Int32(units) + unit
                    score_role(
                        None,  # mQ: the query tile is derived
                        mKbar,
                        mBlockTable,
                        mKvLens,
                        mLogits,
                        mCounts,
                        mBlockLens,
                        sK,
                        sA,
                        cutlass.Int32(top_k),
                        scale,
                        chunk_id,
                        b_scorer,
                        unit,
                        warp_in_unit,
                        tidx_in_unit,
                        unit * cutlass.Int32(ring_elems_u),
                        h_i_ct,
                        s_q,
                        rows_per_page,
                        units_per_warp,
                        ring_depth,
                        use_pdl,
                        scorer_l2_hint,
                        True,  # fused
                        True,  # derive
                        cnt_row_base + (b_scorer * s_q_i).to(cutlass.Int64) * eight,  # the sequence's S_q consecutive counter words
                        units,
                        mIQsrc,
                        mIKsrc,
                        mWiq,
                        mWik,
                        mCos,
                        mSin,
                        mCosPos,
                        mSinPos,
                        mRing,
                        mStateTable,
                        mNCommit,
                        sNew,
                        eps_i,
                        w_offset_i,
                        rope_dim,
                        pool,
                        scorer_mma_batch,
                    )
        else:
            # ---------------- R2: the top-k of one row behind its sequence's scorer CTAs ----------------
            if cutlass.const_expr(with_topk):
                row = b_topk * s_q_i + row_k
                topk_role(
                    mLogitsBits,
                    mCounts,
                    mOut,
                    row,
                    cols_cap,
                    top_k,
                    threads,
                    use_pdl,
                    True,  # fused
                    cnt_row_base + row.to(cutlass.Int64) * eight,
                    n_chunks,
                    check_arrivals,
                    deadline_ns if cutlass.const_expr(with_deadline) else None,
                    diag_addr if cutlass.const_expr(with_deadline) else None,
                )


frost_qsa_indexer_fused.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_indexer_fused_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mQo: cute.Tensor,
    mKo: cute.Tensor,
    mWq: Optional[cute.Tensor],
    mWk: Optional[cute.Tensor],
    mCos: Optional[cute.Tensor],
    mSin: Optional[cute.Tensor],
    mKPool: cute.Tensor,
    mVPool: cute.Tensor,
    mSlot: cute.Tensor,
    mIKsrc: cute.Tensor,
    mIKpool: Optional[cute.Tensor],
    mRing: cute.Tensor,
    mComp: cute.Tensor,
    mWik: cute.Tensor,
    mCosPos: cute.Tensor,
    mSinPos: cute.Tensor,
    mNCommit: Optional[cute.Tensor],
    mCSlot: cute.Tensor,
    mKvLens: cute.Tensor,
    mBlockTable: cute.Tensor,
    mStateTable: cute.Tensor,
    mRingSlots: Optional[cute.Tensor],
    mIQsrc: cute.Tensor,
    mWiq: cute.Tensor,
    mKbar: cute.Tensor,
    mLogits: cute.Tensor,
    mCounts: cute.Tensor,
    mBlockLens: Optional[cute.Tensor],
    mLogitsBits: cute.Tensor,
    mOut: cute.Tensor,
    n_tok: cutlass.Int32,
    n_slots: cutlass.Int64,
    n_islots: cutlass.Int64,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    h_i: cutlass.Int32,
    eps: cutlass.Float32,
    eps_i: cutlass.Float32,
    w_offset_i: cutlass.Float32,
    scale: cutlass.Float32,
    n_prep: cutlass.Int32,
    n_chunks: cutlass.Int32,
    n_ctas: cutlass.Int32,
    cnt_row_base: cutlass.Int64,
    deadline_ns: cutlass.Int64,
    diag_addr: cutlass.Int64,
    d: cutlass.Constexpr[int],
    d_i: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    h_i_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    scorer_mma_batch: cutlass.Constexpr[int],
    scorer_l2_hint: cutlass.Constexpr[bool],
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    units: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    consumers_first: cutlass.Constexpr[bool],
    consumers_global: cutlass.Constexpr[bool],
    check_arrivals: cutlass.Constexpr[bool],
    with_deadline: cutlass.Constexpr[bool],
    with_prepare: cutlass.Constexpr[bool],
    with_scorer: cutlass.Constexpr[bool],
    with_topk: cutlass.Constexpr[bool],
    smem_bytes: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_qsa_indexer_fused(
        mQ,
        mK,
        mV,
        mQo,
        mKo,
        mWq,
        mWk,
        mCos,
        mSin,
        mKPool,
        mVPool,
        mSlot,
        mIKsrc,
        mIKpool,
        mRing,
        mComp,
        mWik,
        mCosPos,
        mSinPos,
        mNCommit,
        mCSlot,
        mKvLens,
        mBlockTable,
        mStateTable,
        mRingSlots,
        mIQsrc,
        mWiq,
        mKbar,
        mLogits,
        mCounts,
        mBlockLens,
        mLogitsBits,
        mOut,
        n_tok,
        n_slots,
        n_islots,
        h_q,
        h_kv,
        h_i,
        eps,
        eps_i,
        w_offset_i,
        scale,
        n_prep,
        n_chunks,
        cnt_row_base,
        deadline_ns,
        diag_addr,
        d,
        d_i,
        rope_dim,
        pool,
        page_size,
        slot_i64,
        h_q_ct,
        h_kv_ct,
        h_i_ct,
        const_head_counts,
        s_q,
        rows_per_page,
        units_per_warp,
        ring_depth,
        scorer_mma_batch,
        scorer_l2_hint,
        cols_cap,
        top_k,
        units,
        threads,
        use_pdl,
        consumers_first,
        consumers_global,
        check_arrivals,
        with_deadline,
        with_prepare,
        with_scorer,
        with_topk,
    ).launch(grid=(n_ctas, 1, 1), block=(threads, 1, 1), smem=smem_bytes, smem_merge_branch_allocs=True, stream=stream, use_pdl=use_pdl)


# --------------------------------------------------------------------------------------------- compile


class FusedKey(NamedTuple):
    """Every compile key of one fused rendering, in one place (the front door builds it; the cache key is its plain values)."""

    dtype: str
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    page_size: int
    slot_dtype: str
    apply_norm: bool
    index: tuple  # tuple(QsaPrepareIndexArms)
    s_q: int
    units: int
    threads: int
    units_per_warp: int
    ring_depth: int
    scorer_mma_batch: int
    has_block_lens: bool
    cols_cap: int
    top_k: int
    use_pdl: bool
    const_head_counts: bool
    smem_bytes: int
    check_arrivals: bool
    with_deadline: bool
    consumers_first: bool
    roles: tuple
    device: int
    consumers_global: bool = False  # appended (the key is append-only): the global mis-order of the trap test
    scorer_l2_hint: bool = True  # appended: the scorer's key-copy arm as BUILT (the front door resolves the DSL compiler's floor like the standalone scorer)


def fused_fake_args(key: FusedKey, *, stream=None) -> list:
    """The positional arguments ``cute.compile(qsa_indexer_fused_launch, *args, ...)`` takes for ``key``: the three role modules' own
    fake tensors (every shape symbolic but the compile keys), the runtime scalars as type-pinning literals, the constants.  Public so a
    host trace-compile (the register census, the PTX census) renders exactly the served kernel with ``--gpu-arch sm_107a`` and no device."""
    global _FAKE_STREAM
    if stream is None:
        if _FAKE_STREAM is None:
            _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
        stream = _FAKE_STREAM
    ix = QsaPrepareIndexArms(*key.index)
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}[key.dtype]
    slot_dtype = {"torch.int32": torch.int32, "torch.int64": torch.int64}[key.slot_dtype]
    h_i, d_i, pool = int(ix.heads), int(ix.head_dim), int(ix.pool)
    tok = cute.sym_int()
    batch = cute.sym_int()
    table_dim = key.rope_dim if key.rope_dim else 1
    bands = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, key.d) for h in (key.h_q, key.h_kv, key.h_kv, key.h_q, key.h_kv)]  # q, k, v, q_out, k_out
    weights = [_fake(dtype, (key.d,), (0,)) for _ in range(2)] if key.apply_norm else [None, None]
    tables = [None, None] if ix.rope_table else [_fake(dtype, (tok, table_dim), (1, 0)) for _ in range(2)]
    pools = [_fake_pool(dtype, key.h_kv, key.page_size, key.d) for _ in range(2)]
    slot = _fake_slot(slot_dtype)
    ik_src = fake_rowmajor_dynamic_token_stride(dtype, tok, 1, d_i)
    ik_pool = _fake_pool(dtype, 1, key.page_size, d_i) if ix.raw_pool else None
    ring = _fake_paged_state(dtype, d_i)
    comp = _fake_paged_state(dtype, d_i)
    w_ik = _fake(dtype, (d_i,), (0,))
    cos_pos, sin_pos = [_fake_strided(dtype, (cute.sym_int(), cute.sym_int(), table_dim), (cute.sym_int(), cute.sym_int(), 1)) for _ in range(2)]
    n_commit = _fake_compact(torch.int32, (batch,), (0,)) if ix.n_commit else None
    cslot = _fake_compact(torch.int32, (batch, cute.sym_int()), (1, 0))
    kv_lens = _fake_compact(torch.int32, (batch,), (0,))
    block_table = _fake_i32_table((batch, cute.sym_int()), (cute.sym_int(), 1))
    state_table = _fake_i32_table((batch, 1), (cute.sym_int(), 1))
    ring_slots = _fake_slot(slot_dtype) if ix.ring_slots else None
    iq_src = fake_rowmajor_dynamic_token_stride(dtype, tok, h_i, d_i)
    w_iq = _fake(dtype, (d_i,), (0,))
    roles = tuple(key.roles)
    return [
        *bands,
        *weights,
        *tables,
        *pools,
        slot,
        ik_src,
        ik_pool,
        ring,
        comp,
        w_ik,
        cos_pos,
        sin_pos,
        n_commit,
        cslot,
        kv_lens,
        block_table,
        state_table,
        ring_slots,
        iq_src,
        w_iq,
        _fake_kbar(key.page_size // BLOCK_SIZE),  # the scorer's view of the compressed-key cache
        _fake_f32_2d(),  # logits
        _fake_i32_1d(),  # counts
        _fake_i32_1d() if key.has_block_lens else None,
        _fake_rows(cutlass.Int32),  # the logits as int32 bits (the top-k's view)
        _fake_rows(cutlass.Int32),  # block_ids
        cutlass.Int32(0),  # n_tok      ) runtime values: the literals pin
        cutlass.Int64(0),  # n_slots    ) their TYPE at trace time only
        cutlass.Int64(0),  # n_islots   )
        cutlass.Int32(key.h_q),
        cutlass.Int32(key.h_kv),
        cutlass.Int32(h_i),
        cutlass.Float32(key.eps),
        cutlass.Float32(1e-6),  # eps_i
        cutlass.Float32(0.0),  # w_offset_i
        cutlass.Float32(0.0),  # scale
        cutlass.Int32(0),  # n_prep
        cutlass.Int32(0),  # n_chunks
        cutlass.Int32(0),  # n_ctas
        cutlass.Int64(0),  # cnt_row_base
        cutlass.Int64(0),  # deadline_ns
        cutlass.Int64(0),  # diag_addr
        int(key.d),
        int(d_i),
        int(key.rope_dim),
        int(pool),
        int(key.page_size),
        slot_dtype == torch.int64,
        int(key.h_q),
        int(key.h_kv),
        int(h_i),
        bool(key.const_head_counts),
        int(key.s_q),
        int(key.page_size // BLOCK_SIZE),
        int(key.units_per_warp),
        int(key.ring_depth),
        int(key.scorer_mma_batch),
        bool(key.scorer_l2_hint),
        int(key.cols_cap),
        int(key.top_k),
        int(key.units),
        int(key.threads),
        bool(key.use_pdl),
        bool(key.consumers_first),
        bool(key.consumers_global),
        bool(key.check_arrivals),
        bool(key.with_deadline),
        "prepare" in roles,
        "scorer" in roles,
        "topk" in roles,
        int(key.smem_bytes),
        stream,
    ]


def compile_fused(key: FusedKey):
    """The one ``compile_cached`` call behind ``compile_qsa_indexer_fused`` (the gates ran in the front door): the in-process memo first,
    the persistent compiled-plan cache behind it (keyed by this file's digest + the key's plain values -- the role modules are in the
    environment manifest), the compile ON the plan's device whichever device is current."""
    if key not in compiled_cache:
        cache_key = template_key(globals(), {"key": tuple(key)}, "compile_qsa_indexer_fused")
        with torch.cuda.device(key.device):
            compiled_cache[key] = compile_cached(
                qsa_indexer_fused_launch, *fused_fake_args(key), options="--enable-tvm-ffi", cache_key=cache_key, symbol="cudnn_qsa_indexer_fused"
            )
    return compiled_cache[key]


def launch_fused(compiled, args: list, *, device: int) -> None:
    """The lowered launch: ``args`` is the kernel's positional operand list (built by the front door's form checks); the launch runs in the
    artifact's context whichever device is current at the call."""
    with torch.cuda.device(device):
        compiled(*args)
