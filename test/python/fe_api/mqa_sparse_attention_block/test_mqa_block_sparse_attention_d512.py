# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""W3a micro tests for the gathered-list d512 sparse-attention kernel's building blocks.

The kernel itself (a fork of ``sdpa/fwd/kernels/sm107/prefill_d512_f16.py``) lands in wave B; this
module pins the four primitives it is built from, each against an independent reference:

* ``tile_dsl.tma.tma_gather4`` -- a ROUNDTRIP micro kernel gathers ``R`` rows of a ``[n, 512]`` bf16
  view through gather4 into the d512 kernel's 128-B-swizzled K/V ``SmemTile`` layout (8 sub-boxes of
  ``R`` rows x 128 B), TMA-stores the tile back through a plain TILED s128b descriptor and compares
  bitwise with ``kv[ids]`` -- so the "gather4 lands the tiled layout" claim (plan 1(c)) is proven by the
  store path, not asserted.  ``-1`` and past-the-end ids must come back as zero rows with their bytes
  still counted (the ``expect_tx`` is the FULL tile).  Both ``cta_group=1`` (local mbar) and
  ``cta_group=2`` under a ``(2,1,1)`` cluster with LEADER-ONLY ``expect_tx`` (the routing verdict Q-d:
  the non-leader's local mbar must NOT flip, the leader's must -- a mis-route hangs, exit 124).
* M3 (Q-b): the same primitive timed from ONE issuing warp, 128 issues per 64-KiB stage, sorted vs
  random 1-KiB rows of an L2-resident buffer, ring depth 1 and 2 -- GB/s per SM, ns per issue and
  ``clock64`` cycles per issue are PRINTED, never asserted (a rate is a measurement, not a contract).
* ``tile_dsl.mask.apply_membership_chunk`` vs a torch ``where`` on random bit words, including all-zero
  and all-one words and the 32-wide form -- runs on ANY CUDA device (pure ALU).
* ``mqa_sparse_attention_block.kernels.union_lists.build_union_lists`` vs a brute-force CPU mirror
  written here (``union_lists_reference``: dict-of-multiplicities, the multiset rule spelled with
  loops), over duplicates within and across lists, ``tok >= S``, out-of-range ids, and the all-``-1``
  quad (``n_tiles == 1``); plus a pin against the oracle's ``multiplicity_mask`` on
  ``window_idxs`` + ``compressed_idxs_synthetic`` lists.
* an ``sm_107a`` trace-compile of the micro kernels on ANY box whose nvdisasm decodes it: the
  ``UTMALDG.2D.GATHER4`` count, the ``.2CTA`` form under ``cta_group=2``, ``STL/LDL == 0``.

Accept tests that launch gather4 are ``requires_rubin``; the CPU / any-GPU tests run everywhere.

W3b (below the micro tests): the FORK itself, ``kernels/sparse_attention_d512.py``, driven through its stated interface
(``compile(b, s, n_kv_rows, n_clusters, has_lse)`` -> ``fn(q, kv2d, o, lse|None, sinks, union_ids, union_bits,
union_ntiles, scale_log2, stream)``) with the block-owned ``build_union_lists`` pre-pass in front -- every accept asserts
against ``mqa_block_reference.sparse_attention_reference`` under the bf16 budget (rtol ``2**-7``, atol ``2**-8 max|kv|``:
NEVER bitwise, never widened) and the LSE within ``1e-4`` of the fp64 arm, with sentinel-filled O, keyless rows exact,
``set_sync_debug_mode("error")`` around the launch, and two launches bitwise.  Those tests skip typed until the fork module
exists and are ``requires_rubin``; the DESC_VERSION twins, the validator rejects and the sm_107a trace-compile run on any box
with the DSL.  The block-side wiring (adapter, workspace slots, runners) is covered under ``attention="d512"`` in
``test_mqa_block_sparse_attention_stage.py`` and ``test_mqa_block_end_to_end.py``.
"""

from __future__ import annotations

import functools
import glob
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

# Rootdir-qualified (a bare `import mqa_block_reference` collides in one `pytest fe_api` session).
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.mqa_sparse_attention_block.kernels import sparse_attention as SA
from cudnn.mqa_sparse_attention_block.kernels.union_lists import (
    TILE_ROWS,
    TOKENS_PER_CLUSTER,
    WORDS_PER_SLOT,
    build_union_lists,
    n_clusters_for,
    u_max_tiles_for,
    validate_union_lists_shapes,
)

pytestmark = pytest.mark.L0

_SM107 = (10, 7)
_DSL_ERROR = cutedsl_requirement_error("MQA sparse-attention block d512 micro tests")
requires_dsl = pytest.mark.skipif(_DSL_ERROR is not None, reason=str(_DSL_ERROR))


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"gather4 through the FROST tensor map targets SM107; found {_cc()}")

D = 512  # KV row width in bf16 elements = 1 KiB, the d512 kernel's TILE_K == TILE_O
COLS = 64  # one gather box-row = 64 elements = 128 B (the s128b swizzle atom)
N_SUB = D // COLS  # 8 sub-boxes per K/V tile, `tma_subtile_stride_elems = R * COLS` (D512:583-606)

if _DSL_ERROR is None:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from cutlass.experimental import primitives as nvvm
    from cutlass.experimental.cuda import tensor_map as tmap

    from cudnn.frost.tile_dsl.barrier import arrive_expect_tx, wait
    from cudnn.frost.tile_dsl.handles import GmemTileTma, SmemTile
    from cudnn.frost.tile_dsl.mask import apply_membership_chunk
    from cudnn.frost.tile_dsl.tma import ldg_int32x4, tma_gather4, tma_store_commit, tma_store_tile, tma_store_wait

    # ------------------------------------------------------------------ roundtrip micro kernel
    # Barrier table (one CTA): `mb` -- TMA_LOAD, init 1; producer = ONE elected lane's arrive.expect_tx
    # (cta_group=1: every CTA arms its own R*1024 B; cta_group=2: only the pair LEADER arms 2*R*1024 B,
    # the non-leader's gathers credit the leader's copy through the bit-24 clear); bytes = R/4 quads x 8
    # sub-boxes x 512 B per CTA; consumer = the issuing warp (cta_group=1) / the leader's warp (=2), then a
    # cluster barrier publishes "landed" to the non-leader before its TMA store reads the tile.
    # SMEM table: `sBuf` bf16 R*512 (64/128 KiB), 1024-aligned, written by gather4 (async proxy), read by
    # TMA-STG (async proxy) -> no fence_proxy; swizzle s128b because BOTH descriptors read it.
    @cute.kernel
    def _roundtrip_kernel(
        ids: cute.Tensor,  # int32 [n_cta * R]: flat row ids into the kv view (-1 / >= n = zero row)
        din: cutlass.GridConstant[tmap.TensorMap],
        dout: cutlass.GridConstant[tmap.TensorMap],
        probe: cute.Tensor,  # int32 [n_cta * 2]: [local mbar phase-0 complete?, gather4 issues]
        R: cutlass.Constexpr[int],
        cta_group: cutlass.Constexpr[int],
    ):
        sBuf = cutlass.Array(cutlass.BFloat16, R * D, alignment=1024, space=cutlass.AddressSpace.smem)
        mb = cutlass.Array(cutlass.Int64, 1, alignment=8, space=cutlass.AddressSpace.smem)
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        cta_rank = cute.arch.block_idx_in_cluster() if cutlass.const_expr(cta_group == 2) else cutlass.Int32(0)
        is_leader = cta_rank == cutlass.Int32(0)
        if tidx < cutlass.Int32(32):
            if nvvm.elect_sync():  # ONE warp, ONE lane (P4)
                nvvm.mbarrier_init(mb.subview(0), 1)
        nvvm.fence_mbarrier_init()
        nvvm.barrier_cta_sync()
        if cutlass.const_expr(cta_group == 2):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        row0 = bidx * cutlass.Int32(R)
        n_quads = cutlass.const_expr(R // 4)
        if tidx < cutlass.Int32(32):
            # lanes 0..n_quads-1 each fetch their quad's four ids (16 B, read-only) BEFORE the arm; the
            # elected lane issues with the ids shuffled to it -- shuffles OUTSIDE the elect branch.
            q_lane = tidx if tidx < cutlass.Int32(n_quads) else cutlass.Int32(0)
            addr = ids.iterator.toint() + (row0 + q_lane * cutlass.Int32(4)).to(cutlass.Int64) * cutlass.Int64(4)
            i0, i1, i2, i3 = ldg_int32x4(addr)
            tx = R * D * 2 * cta_group  # the FULL tile, OOB rows included
            if cutlass.const_expr(cta_group == 1):
                arrive_expect_tx(mb.subview(0), tx, pred=nvvm.elect_sync())
            else:
                arrive_expect_tx(mb.subview(0), tx, pred=is_leader & nvvm.elect_sync())
            for g in cutlass.range_constexpr(n_quads):
                r0 = cute.arch.shuffle_sync(i0, g)
                r1 = cute.arch.shuffle_sync(i1, g)
                r2 = cute.arch.shuffle_sync(i2, g)
                r3 = cute.arch.shuffle_sync(i3, g)
                if nvvm.elect_sync():
                    for i in cutlass.range_constexpr(N_SUB):
                        # sub-box i (columns 64i..64i+63) at i*R*COLS; quad g at 4g rows x 128 B inside it
                        tma_gather4(din, sBuf.subview(i * R * COLS + g * 4 * COLS), mb.subview(0), cutlass.Int32(i * COLS), r0, r1, r2, r3, cta_group=cta_group)
            if cutlass.const_expr(cta_group == 1):
                wait(mb.subview(0), cutlass.Int32(0))
            else:
                if is_leader:
                    wait(mb.subview(0), cutlass.Int32(0))
        if cutlass.const_expr(cta_group == 2):
            cute.arch.cluster_arrive()  # every thread of both CTAs: the leader's arrives after its wait
            cute.arch.cluster_wait()
        nvvm.barrier_cta_sync()
        if tidx == cutlass.Int32(0):
            flipped = nvvm.mbarrier_try_wait_parity(mb.subview(0), cutlass.Int32(0))
            probe[bidx * 2] = cutlass.Int32(1) if flipped else cutlass.Int32(0)
            probe[bidx * 2 + 1] = cutlass.Int32(n_quads * N_SUB)
            handle = SmemTile(
                base=sBuf.subview(0),
                elems_per_stage=R * D,
                leading_byte_offset=0,
                stride_byte_offset=0,
                layout=0,
                tma_loads_per_tile=N_SUB,
                tma_granu_elems=COLS,
                tma_subtile_stride_elems=R * COLS,
            )
            tma_store_tile(handle, GmemTileTma(dout)(cutlass.Int32(0), row0))
            tma_store_commit()
            tma_store_wait(0)

    @cute.jit
    def _roundtrip_host(kv, out, ids, probe, n_cta, R: cutlass.Constexpr[int], cta_group: cutlass.Constexpr[int], stream: cuda.CUstream):
        # ONE 2-D descriptor over the [n, 512] view, box = 1 row x 64 elements (128 B), s128b -- the fork's
        # `tma_kv_desc` (plan 1(c)); the store side is an ordinary tiled (R, 64) s128b box.
        din = tmap.create_tensor_map_tiled_from_view(
            kv, box_dims=(1, COLS), swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        dout = tmap.create_tensor_map_tiled_from_view(
            out, box_dims=(R, COLS), swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        _roundtrip_kernel(ids, din, dout, probe, R, cta_group).launch(grid=(n_cta, 1, 1), block=(128, 1, 1), cluster=(cta_group, 1, 1), stream=stream)

    # ------------------------------------------------------------------ M3 timing kernel
    # One warp per CTA, a DEPTH-deep ring of 64-KiB stages: per stage 16 quads x 8 sub-boxes = 128 gather4
    # from the elected lane; stage s is re-armed only after ITS previous fill landed (the `_full` wait
    # stands in for the kernel's `_empty` ring -- the consumer here is nobody).  `mb[s]` TMA_LOAD init 1.
    @cute.kernel
    def _timing_kernel(
        ids: cute.Tensor,  # int32 [n_cta * n_stages * R]
        din: cutlass.GridConstant[tmap.TensorMap],
        cycles: cute.Tensor,  # int64 [n_cta]: clock64 delta over the whole stage loop
        n_stages: cutlass.Int32,
        R: cutlass.Constexpr[int],
        DEPTH: cutlass.Constexpr[int],
    ):
        sBuf = cutlass.Array(cutlass.BFloat16, DEPTH * R * D, alignment=1024, space=cutlass.AddressSpace.smem)
        mb = cutlass.Array(cutlass.Int64, DEPTH, alignment=8, space=cutlass.AddressSpace.smem)
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        bidx = cutlass.Int32(cute.arch.block_idx()[0])
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(DEPTH):
                nvvm.mbarrier_init(mb.subview(s), 1)
        nvvm.fence_mbarrier_init()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        n_quads = cutlass.const_expr(R // 4)
        q_lane = tidx if tidx < cutlass.Int32(n_quads) else cutlass.Int32(0)
        tx = R * D * 2
        t0 = cute.arch.clock64()
        for it in cutlass.range(0, n_stages, 1, unroll=1):
            s = it % cutlass.Int32(DEPTH)
            rnd = it // cutlass.Int32(DEPTH)
            stage_row0 = (bidx * n_stages + it) * cutlass.Int32(R)
            addr = ids.iterator.toint() + (stage_row0 + q_lane * cutlass.Int32(4)).to(cutlass.Int64) * cutlass.Int64(4)
            i0, i1, i2, i3 = ldg_int32x4(addr)
            if it >= cutlass.Int32(DEPTH):
                wait(mb.subview(s), (rnd - cutlass.Int32(1)) & cutlass.Int32(1))
            arrive_expect_tx(mb.subview(s), tx, pred=nvvm.elect_sync())
            for g in cutlass.range_constexpr(n_quads):
                r0 = cute.arch.shuffle_sync(i0, g)
                r1 = cute.arch.shuffle_sync(i1, g)
                r2 = cute.arch.shuffle_sync(i2, g)
                r3 = cute.arch.shuffle_sync(i3, g)
                if nvvm.elect_sync():
                    for i in cutlass.range_constexpr(N_SUB):
                        tma_gather4(
                            din, sBuf.subview(s * R * D + i * R * COLS + g * 4 * COLS), mb.subview(s), cutlass.Int32(i * COLS), r0, r1, r2, r3, cta_group=1
                        )
        for d in cutlass.range_constexpr(DEPTH):  # drain the last DEPTH stages
            it = n_stages - cutlass.Int32(DEPTH) + cutlass.Int32(d)
            if it >= cutlass.Int32(0):
                wait(mb.subview(it % cutlass.Int32(DEPTH)), (it // cutlass.Int32(DEPTH)) & cutlass.Int32(1))
        t1 = cute.arch.clock64()
        if tidx == cutlass.Int32(0):
            cycles[bidx] = t1 - t0

    @cute.jit
    def _timing_host(kv, ids, cycles, n_cta, n_stages, R: cutlass.Constexpr[int], DEPTH: cutlass.Constexpr[int], stream: cuda.CUstream):
        din = tmap.create_tensor_map_tiled_from_view(
            kv, box_dims=(1, COLS), swizzle=tmap.TensorMapSwizzle.s128b, l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        _timing_kernel(ids, din, cycles, n_stages, R, DEPTH).launch(grid=(n_cta, 1, 1), block=(32, 1, 1), stream=stream)

    # ------------------------------------------------------------------ membership probe kernel
    @cute.kernel
    def _membership_kernel(scores: cute.Tensor, words: cute.Tensor, out: cute.Tensor, N: cutlass.Constexpr[int]):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])
        row = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(128) + tidx
        elems = []
        for i in cutlass.range_constexpr(N):
            elems.append(scores[row * N + i])
        vec = cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)
        res = apply_membership_chunk(vec, words[row * 2], words[row * 2 + 1], n=N)
        for i in cutlass.range_constexpr(N):
            out[row * N + i] = res[i]

    @cute.jit
    def _membership_host(scores, words, out, n_blocks, N: cutlass.Constexpr[int], stream: cuda.CUstream):
        _membership_kernel(scores, words, out, N).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)

    _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    def _fake_1d(dtype, align=16):
        return make_fake_tensor(dtype, (cute.sym_int(),), (1,), assumed_align=align)

    def _fake_kv():
        return make_fake_tensor(cutlass.BFloat16, (cute.sym_int(), D), (D, 1), assumed_align=16)

    def compile_roundtrip(R: int, cta_group: int, options: str = "--enable-tvm-ffi"):
        return cute.compile(
            _roundtrip_host,
            _fake_kv(),
            _fake_kv(),
            _fake_1d(cutlass.Int32),
            _fake_1d(cutlass.Int32),
            cutlass.Int32(0),
            R,
            cta_group,
            _FAKE_STREAM,
            options=options,
        )

    def compile_timing(R: int, depth: int, options: str = "--enable-tvm-ffi"):
        return cute.compile(
            _timing_host,
            _fake_kv(),
            _fake_1d(cutlass.Int32),
            _fake_1d(cutlass.Int64),
            cutlass.Int32(0),
            cutlass.Int32(0),
            R,
            depth,
            _FAKE_STREAM,
            options=options,
        )

    def compile_membership(N: int, options: str = "--enable-tvm-ffi"):
        return cute.compile(
            _membership_host, _fake_1d(cutlass.Float32), _fake_1d(cutlass.Int32), _fake_1d(cutlass.Float32), cutlass.Int32(0), N, _FAKE_STREAM, options=options
        )

    def _stream():
        return cuda.CUstream(int(torch.cuda.current_stream().cuda_stream))


# ============================================================================ brute-force mirror
def union_lists_reference(topk_idxs: torch.Tensor, *, seq_len: int, n_kv_rows: int, u_max_tiles: int):
    """Loop-spelled twin of ``build_union_lists`` (plan 1(b) multiset rule), CPU, for the tests only.

    Per (batch, cluster): ``mult[x][j]`` = how many slots of token ``4c + j`` (``< seq_len``) name the
    in-range row ``x``; the union lists ``(x, k)`` for ``k < max_j mult[x][j]`` ascending; entry ``(x, k)``
    carries bit ``j`` iff ``mult[x][j] > k``.  Returns ``(ids, bits, ntiles)`` in the kernel's layouts.
    """
    b, s, _ = topk_idxs.shape
    nc = n_clusters_for(seq_len)
    u_max = TILE_ROWS * u_max_tiles
    ids = torch.full((b, nc, u_max), -1, dtype=torch.int32)
    bits = torch.zeros((b, nc, u_max_tiles, TOKENS_PER_CLUSTER, WORDS_PER_SLOT), dtype=torch.int64)
    ntiles = torch.ones((b, nc), dtype=torch.int32)
    lists = topk_idxs.cpu().tolist()
    for bb in range(b):
        for c in range(nc):
            mult = {}
            for j in range(TOKENS_PER_CLUSTER):
                tok = TOKENS_PER_CLUSTER * c + j
                if tok >= seq_len:
                    continue
                for x in lists[bb][tok]:
                    if 0 <= x < n_kv_rows:
                        mult.setdefault(x, [0] * TOKENS_PER_CLUSTER)[j] += 1
            entries = []
            for x in sorted(mult):
                for k in range(max(mult[x])):
                    entries.append((x, [j for j in range(TOKENS_PER_CLUSTER) if mult[x][j] > k]))
            assert len(entries) <= u_max
            for pos, (x, js) in enumerate(entries):
                ids[bb, c, pos] = bb * n_kv_rows + x
                for j in js:
                    bits[bb, c, pos // TILE_ROWS, j, (pos % TILE_ROWS) // 32] |= 1 << (pos % 32)
            ntiles[bb, c] = max(1, -(-len(entries) // TILE_ROWS))
    bits = torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
    return ids, bits, ntiles


def _run_union_lists(topk, *, seq_len, n_kv_rows, u_max_tiles):
    b = topk.shape[0]
    nc = n_clusters_for(seq_len)
    dev = topk.device
    ids = torch.full((b, nc, TILE_ROWS * u_max_tiles), 7, dtype=torch.int32, device=dev)  # poison
    bits = torch.full((b, nc, u_max_tiles, TOKENS_PER_CLUSTER, WORDS_PER_SLOT), -1, dtype=torch.int32, device=dev)
    nt = torch.zeros((b, nc), dtype=torch.int32, device=dev)
    build_union_lists(topk, seq_len=seq_len, n_kv_rows=n_kv_rows, u_max_tiles=u_max_tiles, out_ids=ids, out_bits=bits, out_ntiles=nt)
    return ids, bits, nt


_UNION_CASES = [
    # (batch, seq_len, topk, n_kv_rows, seed)
    pytest.param(1, 7, 5, 20, 0, id="tail-cluster-tok>=S"),
    pytest.param(2, 9, 12, 40, 1, id="B2-oob-ids"),
    pytest.param(1, 4, 3, 4, 2, id="tiny-heavy-duplicates"),
    pytest.param(2, 13, 64, 300, 3, id="two-tiles"),
    pytest.param(1, 8, 640, 700, 4, id="K640-many-tiles"),
    pytest.param(1, 5, 40, 30, 5, id="n_kv<K-dense-dupes"),
]


@pytest.mark.parametrize("b, s, k, n_kv, seed", _UNION_CASES)
def test_union_lists_matches_brute_force(b, s, k, n_kv, seed):
    """Random lists with negatives, -1, out-of-range ids (>= n_kv), duplicates within and across lists."""
    g = torch.Generator().manual_seed(seed)
    topk = torch.randint(-3, n_kv + 4, (b, s, k), generator=g).to(torch.int32)
    topk[0, 0] = -1  # one token with no key
    topk[0, 1, :3] = n_kv - 1  # a valid id THREE times in token 1 and once in token 2 (cluster 0):
    topk[0, 2, 0] = n_kv - 1  #   3 union copies, copy 0 carries bits {1, 2}, copies 1-2 carry bit {1} only
    if s >= 8:
        topk[0, 4:8] = -1  # a whole cluster (cluster 1) with no key -- leaves cluster 0 live for the s < 8 cases
    ut = u_max_tiles_for(k)
    ids, bits, nt = _run_union_lists(topk, seq_len=s, n_kv_rows=n_kv, u_max_tiles=ut)
    ref_ids, ref_bits, ref_nt = union_lists_reference(topk, seq_len=s, n_kv_rows=n_kv, u_max_tiles=ut)
    assert torch.equal(ids.cpu(), ref_ids), "union_ids"
    assert torch.equal(bits.cpu(), ref_bits), "union_bits"
    assert torch.equal(nt.cpu(), ref_nt), "union_ntiles"
    if s >= 8:  # the all -1 quad: one tile, no ids, no bits
        assert int(nt[0, 1]) == 1 and bool((ids[0, 1] == -1).all()) and int(bits[0, 1].abs().sum()) == 0
    assert int(nt.min()) >= 1
    # every set bit names a real column; every real column is named by >= 1 slot
    for bb in range(b):
        for c in range(n_clusters_for(s)):
            n_valid = int((ids[bb, c] != -1).sum())
            words = torch.zeros(ut * WORDS_PER_SLOT, dtype=torch.int64)
            for j in range(TOKENS_PER_CLUSTER):
                words |= bits[bb, c, :, j, :].reshape(-1).to(torch.int64) & 0xFFFFFFFF
            named = sum(bin(int(w)).count("1") for w in words)
            assert named == n_valid, (bb, c, named, n_valid)


def test_union_lists_bits_reproduce_the_oracle_multiplicity():
    """Reconstruct each token's per-row multiplicity from (ids, bits) and compare with ``R.multiplicity_mask``
    on the oracle's own list builders (window + synthetic compressed, ids already offset by S)."""
    b, s, ratio, window, topk = 2, 37, 4, 8, 6
    g = torch.Generator().manual_seed(11)
    idxs = torch.cat([R.window_idxs(b, s, window), R.compressed_idxs_synthetic(b, s, ratio, topk, g)], dim=-1)
    idxs[1, 5, :3] = idxs[1, 5, 3]  # duplicates within one list (counted twice by the oracle: M:415)
    n_kv = s + s // ratio
    ut = u_max_tiles_for(idxs.shape[-1])
    ids, bits, nt = _run_union_lists(idxs, seq_len=s, n_kv_rows=n_kv, u_max_tiles=ut)
    want = R.multiplicity_mask(idxs, n_kv)  # [B, S, n_kv] fp32 counts
    got = torch.zeros_like(want)
    for bb in range(b):
        for c in range(n_clusters_for(s)):
            for j in range(TOKENS_PER_CLUSTER):
                tok = TOKENS_PER_CLUSTER * c + j
                if tok >= s:
                    continue
                for col in range(TILE_ROWS * int(nt[bb, c])):
                    w = int(bits[bb, c, col // TILE_ROWS, j, (col % TILE_ROWS) // 32]) & 0xFFFFFFFF
                    if (w >> (col % 32)) & 1:
                        flat = int(ids[bb, c, col])
                        assert flat != -1, "a set bit on a pad column"
                        assert flat // n_kv == bb
                        got[bb, tok, flat % n_kv] += 1
    assert torch.equal(got, want)
    # ids are ascending within the valid prefix, -1 after it
    for bb in range(b):
        for c in range(n_clusters_for(s)):
            row = ids[bb, c].tolist()
            n_valid = sum(1 for x in row if x != -1)
            assert all(x != -1 for x in row[:n_valid]) and all(x == -1 for x in row[n_valid:])
            assert row[:n_valid] == sorted(row[:n_valid])
            assert int(nt[bb, c]) == max(1, -(-n_valid // TILE_ROWS))


def test_union_lists_empty_list_and_zero_seq_len():
    ids, bits, nt = _run_union_lists(torch.empty(1, 5, 0, dtype=torch.int32), seq_len=5, n_kv_rows=5, u_max_tiles=1)
    assert bool((ids == -1).all()) and int(bits.abs().sum()) == 0 and bool((nt == 1).all())
    ids, bits, nt = _run_union_lists(torch.empty(1, 0, 4, dtype=torch.int32), seq_len=0, n_kv_rows=5, u_max_tiles=1)
    assert ids.shape == (1, 0, 128) and nt.shape == (1, 0)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t: t.to(torch.int64), "int32"),
        (lambda t: t[0], r"\[B, S, K\]"),
        (lambda t: t[:, :-1], "must equal seq_len"),
    ],
)
def test_union_lists_rejects_bad_topk(mutate, match):
    topk = torch.zeros(1, 8, 4, dtype=torch.int32)
    ut = u_max_tiles_for(4)
    nc = n_clusters_for(8)
    ids = torch.empty(1, nc, 128 * ut, dtype=torch.int32)
    bits = torch.empty(1, nc, ut, 4, 4, dtype=torch.int32)
    nt = torch.empty(1, nc, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        validate_union_lists_shapes(mutate(topk), seq_len=8, n_kv_rows=8, u_max_tiles=ut, out_ids=ids, out_bits=bits, out_ntiles=nt)


def test_union_lists_rejects_bad_workspace():
    topk = torch.zeros(1, 8, 100, dtype=torch.int32)
    nc = n_clusters_for(8)
    ok = dict(seq_len=8, n_kv_rows=8)
    mk = lambda ut: (torch.empty(1, nc, 128 * ut, dtype=torch.int32), torch.empty(1, nc, ut, 4, 4, dtype=torch.int32), torch.empty(1, nc, dtype=torch.int32))
    ids, bits, nt = mk(4)
    assert u_max_tiles_for(100) == 4
    with pytest.raises(ValueError, match="cannot hold 4 x K"):
        validate_union_lists_shapes(topk, u_max_tiles=3, out_ids=ids, out_bits=bits, out_ntiles=nt, **ok)
    with pytest.raises(ValueError, match="union_bits must have shape"):
        validate_union_lists_shapes(topk, u_max_tiles=4, out_ids=ids, out_bits=bits[:, :, :3], out_ntiles=nt, **ok)
    with pytest.raises(ValueError, match="union_ids must be int32"):
        validate_union_lists_shapes(topk, u_max_tiles=4, out_ids=ids.to(torch.int64), out_bits=bits, out_ntiles=nt, **ok)
    with pytest.raises(ValueError, match="below 2\\*\\*15"):
        ids2, bits2, nt2 = mk(256)
        validate_union_lists_shapes(topk, u_max_tiles=256, out_ids=ids2, out_bits=bits2, out_ntiles=nt2, **ok)
    with pytest.raises(ValueError, match="topk must be >= 0"):
        u_max_tiles_for(-1)


# ============================================================================ tile_dsl primitives: trace-time contracts
@requires_dsl
def test_tma_gather4_rejects_a_cta_group_outside_1_2():
    with pytest.raises(ValueError, match="cta_group must be 1 or 2"):
        tma_gather4(None, None, None, 0, 0, 0, 0, 0, cta_group=3)


@requires_dsl
def test_apply_membership_chunk_rejects_a_chunk_wider_than_two_words():
    with pytest.raises(ValueError, match="n must be in 1..64"):
        apply_membership_chunk(None, None, None, n=65)


# ============================================================================ membership numerics (any CUDA device)
@requires_dsl
@requires_cuda
@pytest.mark.parametrize("N", [64, 32])
def test_apply_membership_chunk_matches_torch(N):
    """Cell i keeps S[i] iff bit i of (word_lo, word_hi) is set, else -inf; includes all-zero and all-one words."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    n_blocks, rows = 3, 3 * 128
    scores = torch.randn(rows, N, device=dev, dtype=torch.float32)
    words = torch.randint(-(2**31), 2**31 - 1, (rows, 2), device=dev, dtype=torch.int64).to(torch.int32)
    words[0] = 0  # nothing kept
    words[1] = -1  # everything kept (all ones)
    words[2, 0], words[2, 1] = 1, 0  # only cell 0
    words[3, 0], words[3, 1] = 0, -(2**31)  # only cell 63 (bit 31 of word_hi)
    out = torch.full_like(scores, 12345.0)
    art = compile_membership(N)
    art(scores.reshape(-1), words.reshape(-1), out.reshape(-1), cutlass.Int32(n_blocks), _stream())
    torch.cuda.synchronize()
    bits = torch.arange(N, device=dev)
    w = torch.where(bits < 32, words[:, :1].to(torch.int64) & 0xFFFFFFFF, words[:, 1:].to(torch.int64) & 0xFFFFFFFF)  # [rows, N]
    keep = ((w >> (bits % 32)) & 1) == 1
    want = torch.where(keep, scores, torch.full_like(scores, float("-inf")))
    assert torch.equal(out, want)
    assert bool(torch.isinf(out[0]).all()) and torch.equal(out[1], scores[1])
    assert bool(torch.isinf(out[2, 1:]).all()) and out[2, 0] == scores[2, 0]
    if N == 64:
        assert out[3, 63] == scores[3, 63] and bool(torch.isinf(out[3, :63]).all())


# ============================================================================ gather4 roundtrip (Rubin)
def _make_ids(n_rows: int, n_ids: int, *, g: torch.Generator, order: str, n_kv: int, oob_every: int = 0) -> torch.Tensor:
    if order == "sorted":
        ids = (torch.arange(n_ids) * 3) % n_kv
    else:
        ids = torch.randint(0, n_kv, (n_ids,), generator=g)
    ids = ids.to(torch.int32)
    if oob_every:
        ids[::oob_every] = -1  # no key -> zero row, bytes counted
        ids[oob_every // 2 :: oob_every * 3] = n_kv  # past the end -> also TMA-OOB, zero row
    return ids


@requires_dsl
@requires_rubin
@pytest.mark.parametrize("R, cta_group, n_cta", [(64, 1, 3), (128, 1, 2), (64, 2, 4)], ids=["K-like-cga1", "V-like-cga1", "K-like-cga2-leader-mbar"])
def test_gather4_roundtrip_lands_the_tiled_swizzled_layout(R, cta_group, n_cta):
    """gather4 -> Swz128B SmemTile (the d512 K/V layout) -> tiled TMA store == kv[ids] bitwise; -1 rows zero.

    Under cta_group=2 ((2,1,1) cluster) ONLY the leader arms expect_tx = 2 x 64 KiB and waits; a PASS is the
    routing verdict Q-d (bytes credited to the pair leader's mbar through the bit-24 clear), a HANG (exit 124)
    means the P12 fallback.  The probe also pins that the non-leader's LOCAL mbar never completed phase 0."""
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(R + cta_group)
    n_kv = 4096
    kv = torch.randn(n_kv, D, generator=g).to(torch.bfloat16).to(dev)
    ids = _make_ids(n_kv, n_cta * R, g=g, order="random", n_kv=n_kv, oob_every=7).to(dev)
    out = torch.full((n_cta * R, D), 3.0, dtype=torch.bfloat16, device=dev)
    probe = torch.full((n_cta * 2,), -9, dtype=torch.int32, device=dev)
    art = compile_roundtrip(R, cta_group)
    art(kv, out, ids, probe, cutlass.Int32(n_cta), _stream())
    torch.cuda.synchronize()
    idl = ids.to(torch.int64)
    want = torch.where(((idl >= 0) & (idl < n_kv)).unsqueeze(1), kv[idl.clamp(0, n_kv - 1)], torch.zeros(1, D, dtype=torch.bfloat16, device=dev))
    bad_rows = (out != want).any(dim=1).nonzero().flatten().tolist()
    assert not bad_rows, f"{len(bad_rows)} rows differ, first {bad_rows[:8]} (ids {ids[bad_rows[:8]].tolist()})"
    assert bool(((idl < 0) | (idl >= n_kv)).any()), "the case must exercise OOB rows"
    pr = probe.cpu().view(n_cta, 2)
    assert bool((pr[:, 1] == R // 4 * N_SUB).all()), pr
    if cta_group == 1:
        assert bool((pr[:, 0] == 1).all()), f"every CTA's local mbar completes at cta_group=1: {pr}"
    else:
        leaders = torch.arange(n_cta) % 2 == 0
        assert bool((pr[leaders, 0] == 1).all()), f"the pair leaders' mbars complete: {pr}"
        assert bool((pr[~leaders, 0] == 0).all()), f"routing verdict: the non-leaders' LOCAL mbars must NOT complete (bytes went to the leader): {pr}"
    # second launch bitwise
    out2 = torch.zeros_like(out)
    art(kv, out2, ids, probe, cutlass.Int32(n_cta), _stream())
    torch.cuda.synchronize()
    assert torch.equal(out, out2)


# ============================================================================ M3: gather4 issue/delivery rate from one warp (Rubin)
@requires_dsl
@requires_rubin
@pytest.mark.parametrize("depth", [1, 2])
def test_m3_gather4_rate_from_one_warp(depth, capsys):
    """PRINTS GB/s per SM, ns per issue and clock64 cycles per issue -- 128 gather4 per 64-KiB stage from ONE
    warp, sorted vs random 1-KiB rows of an 8-MiB (L2-resident) buffer, grid = 1 SM and grid = every SM.
    Nothing about the rate is asserted (Q-b decides the fork's issuing-warp count from these numbers)."""
    dev = torch.device("cuda")
    R, n_stages = 64, 256
    n_kv = 8192  # 8 MiB bf16 x 512: L2-resident after the first pass
    kv = torch.randn(n_kv, D).to(torch.bfloat16).to(dev)
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    art = compile_timing(R, depth)
    g = torch.Generator().manual_seed(3)
    lines = [f"M3 gather4 R={R} depth={depth}: {n_stages} stages x 128 issues x 512 B per CTA; kv {n_kv} rows x 1 KiB; SMs={n_sm}"]
    for grid in (1, n_sm):
        for order in ("sorted", "random"):
            ids = _make_ids(n_kv, grid * n_stages * R, g=g, order=order, n_kv=n_kv).to(dev)
            cycles = torch.zeros(grid, dtype=torch.int64, device=dev)
            art(kv, ids, cycles, cutlass.Int32(grid), cutlass.Int32(n_stages), _stream())  # warm (JIT + L2)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                art(kv, ids, cycles, cutlass.Int32(grid), cutlass.Int32(n_stages), _stream())
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e-3)
            t = sorted(ts)[len(ts) // 2]
            bytes_per_cta = n_stages * R * D * 2
            issues_per_cta = n_stages * (R // 4) * N_SUB
            cyc = cycles.double().median().item()
            lines.append(
                f"  grid={grid:>3} {order:>6}: {t*1e3:8.3f} ms  {bytes_per_cta / t / 1e9:8.1f} GB/s per SM ({bytes_per_cta * grid / t / 1e12:6.2f} TB/s aggregate)  "
                f"{t * 1e9 / issues_per_cta:7.1f} ns/issue  clock64 {cyc / issues_per_cta:7.1f} cycles/issue"
            )
            assert cyc > 0
    with capsys.disabled():
        print("\n" + "\n".join(lines))


# ============================================================================ sm_107a trace-compile + SASS (any box with a decoding nvdisasm)
def _nvdisasm_candidates():
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm107a_known_to_the_dsl() -> bool:
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


_SASS_PROBE = textwrap.dedent("""
    import glob, importlib.util, os, subprocess, sys
    test_path, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(test_path))))  # test/python -> `fe_api.` imports
    spec = importlib.util.spec_from_file_location("mqa_d512_micro", test_path)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    opts = "--enable-tvm-ffi --gpu-arch sm_107a --keep-cubin"  # --keep-cubin, NOT --keep-sass (the wheel nvdisasm ICEs on sm_107a)
    M.compile_timing(64, 2, options=opts)
    M.compile_roundtrip(64, 2, options=opts)
    cubins = sorted(glob.glob(os.path.join(dump, "*.sm_107a.cubin")))
    if len(cubins) != 2:
        print("FAIL expected 2 sm_107a cubins in", dump, "got", cubins); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes sm_107a"); sys.exit(0)
    for cub in cubins:
        sass = subprocess.run([nvd, "-c", cub], capture_output=True, text=True, check=True).stdout.splitlines()
        tag = "TIMING" if "_timing_host" in os.path.basename(cub) else "ROUNDTRIP2"
        print(tag, "GATHER4", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4" in ln))
        print(tag, "GATHER4_2CTA", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4.2CTA" in ln))
        print(tag, "R2UR", sum(1 for ln in sass if "R2UR" in ln))
        print(tag, "SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
        print(tag, "SHFL", sum(1 for ln in sass if "SHFL" in ln))
        print(tag, "LINES", len(sass))
    """)


@requires_dsl
def test_sm107a_trace_compile_gather4_sass(tmp_path):
    """Compile the M3 kernel (depth 2) and the cga2 roundtrip for Rubin HERE (no device match needed) and read the
    SASS: 128 ``UTMALDG.2D.GATHER4`` per stage body (one per issue, fully unrolled), the ``.2CTA`` form under
    cta_group=2, and ``STL/LDL == 0`` on the issuing warp (the fork's TMA-LDG warp runs at 40 registers).
    SKIPS when the DSL predates sm_107a or no candidate nvdisasm decodes it -- never fails on the toolkit."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "gather4_sm107a"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, os.path.abspath(__file__), str(dump), *cands], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {}
    for ln in out:
        parts = ln.split()
        if len(parts) == 3 and parts[0] in ("TIMING", "ROUNDTRIP2"):
            stats[(parts[0], parts[1])] = int(parts[2])
    print(f"\nsm_107a SASS: {stats} via {[ln for ln in out if ln.startswith('NVDISASM')]}")
    assert stats[("TIMING", "GATHER4")] == 128, "one UTMALDG.2D.GATHER4 per issue: 16 quads x 8 sub-boxes"
    assert stats[("TIMING", "GATHER4_2CTA")] == 0
    assert stats[("ROUNDTRIP2", "GATHER4_2CTA")] == 128, "cta_group=2 emits the .2CTA form"
    assert stats[("TIMING", "SPILL")] == 0 and stats[("ROUNDTRIP2", "SPILL")] == 0
    assert stats[("TIMING", "R2UR")] > 0, "row coordinates reach the uniform datapath (R2UR) -- the elected-lane issue model"


# ============================================================================ W3b: the fork -- kernels/sparse_attention_d512.py
# Everything below drives the FORK MODULE directly through its stated interface (plan 1(h)):
#   compile(b, s, n_kv_rows, n_clusters, has_lse) -> fn(q, kv2d, o, lse|None, sinks, union_ids, union_bits, union_ntiles, scale_log2, stream)
# fed by the block-owned pre-pass ``build_union_lists`` (the SAME three int32 views the block carves).  ``sinks`` are the
# block's natural-log ``attn_sink`` (the fork folds them as ``max(final_max_nat, sink_logit)``, plan 1(e)); ``scale_log2`` is
# ``scale * log2(e)`` (``SA.d512_scale_log2``).  The module is loaded through the FROST template loader with the SAME
# ``(path, SparseAttentionD512Params)`` key the block's adapter uses, so a mixed run shares one module instance per
# specialization; ``u_max_tiles = ceil(4K / 128)`` (the adapter's choice) is the union-column pitch of every view.
requires_fork = pytest.mark.skipif(not SA.d512_fork_available(), reason=f"the d512 fork module {SA.D512_FORK_MODULE} has not landed (wave W3b, F1)")

H = 64  # heads per token = the fork's work item (4 tokens x 64 heads per 256-row cluster); TP=2 (32 heads) is refused
_SENTINEL = 1.5e30  # a bf16-representable magnitude no attention output reaches: a surviving cell was never stored
_EXACT = dict(p_dtype=torch.float32, out_dtype=torch.float32)  # the oracle's fp64 arm (LSE bar 1e-4)
_FRESH_PROCESS_ENV = "MQA_D512_FRESH_PROCESS_LAUNCHES"  # the Validate agent sets it to the launch count (12); unset = skip typed
# Per child: import + one JIT + one launch + the two oracle arms (the fp64 exact arm dominates at 32K: 256 chunks x 2 fp64
# einsums over 640 gathered rows); a kill here reads as 124 = HANG.  Keyed by S so the 32K arm is not mis-classified as a hang.
_FRESH_PROCESS_TIMEOUT_S = {2048: 600, 32768: 900}
_FRESH_PROCESS_S = [2048, 32768]  # plan 4 row 4: "12 fresh-process launches classified 0/124/other at S = 2048 AND 32768"
_S_32K = 32768  # the released prompt length: 8192 clusters over 53 resident clusters (212 SMs) = ~155 tiles per CTA (risk 11)


def _assert_bf16_budget(o_a, o_b, kv, label=""):
    """The oracle's documented bar (``test_mqa_block_reference_selfcheck.py:63-67``, plan 3.4): a CORRECT kernel rounds bf16 P
    against its RUNNING max and rescales by alpha per tile, so its O differs from the batch oracle on 10-27 % of the elements
    by one bf16 ulp.  rtol ``2**-7``, atol ``2**-8 * max|kv|`` -- NEVER bitwise, never widened."""
    torch.testing.assert_close(o_a.float(), o_b.float(), rtol=2**-7, atol=2**-8 * float(kv.float().abs().max()), msg=lambda m: f"{label}: {m}")


def _fork_path() -> str:
    return importlib.util.find_spec(SA.D512_FORK_MODULE).origin


def _code_lines(src: str) -> str:
    """The source without comment lines (a pin on a substring must not be satisfiable by prose)."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


@functools.lru_cache(maxsize=None)
def _fork_module(u_max_tiles: int, has_sink: bool = True):
    """The fork specialized by ``SparseAttentionD512Params`` through ``cudnn.frost.template_loader`` (plan 1(h)).  The params
    class lives in the fork, so the plain import (``SA.d512_fork_module``, the default specialization) supplies it; the
    specialized instance is a separate module object, cached by ``(path, params)`` like the block's adapter."""
    from cudnn.frost.template_loader import load_template

    fork = SA.d512_fork_module()
    params = fork.SparseAttentionD512Params(has_sink=bool(has_sink), heads_per_tile=H, u_max_tiles=int(u_max_tiles))
    return load_template(_fork_path(), params, tag=getattr(SA, "D512_TEMPLATE_TAG", "mqa_sparse_attention_d512"))


@functools.lru_cache(maxsize=None)
def _fork_compiled(u_max_tiles: int, has_sink: bool, b: int, s: int, n_kv: int, nc: int, has_lse: bool):
    """One artifact per ``(specialization, B, S, N, NC, has_lse)`` -- every distinct shape is a JIT (~20-40 s), so the tests
    below share shapes where the contract allows (the forced-``n_tiles`` cases all compile ONE kernel)."""
    return _fork_module(u_max_tiles, has_sink).compile(int(b), int(s), int(n_kv), int(nc), bool(has_lse))


def _case_fork(*, batch, seq_len, ratio, window=16, topk=8, seed=0, sink=None, device="cuda"):
    """``q [B,S,64,512]`` / ``kv_all [B, S + N_c, 512]`` bf16, ``sink [64]`` fp32, ``topk_idxs [B,S,K]`` int32 in the block's layouts:
    the window list ++ the synthetic compressed list (ids offset by ``S``).  ``ratio 0`` = a window-only layer (``N = S``);
    ``N_c = S // ratio`` -- ZERO when ``S < ratio`` (the legal degenerate: the list is the window alone).  Small lists by default
    (``K <= 24`` -> one union tile) so the degenerate sweep's cost is the per-shape JIT, not the oracle."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq_len, H, D, generator=g, device=device).to(torch.bfloat16)
    kv = torch.randn(batch, seq_len, D, generator=g, device=device).to(torch.bfloat16)
    idxs = R.window_idxs(batch, seq_len, window, device=device)
    n_c = seq_len // ratio if ratio > 0 else 0
    if n_c > 0:
        kv = torch.cat([kv, torch.randn(batch, n_c, D, generator=g, device=device).to(torch.bfloat16)], dim=1)
        idxs = torch.cat([idxs, R.compressed_idxs_synthetic(batch, seq_len, ratio, topk, g, device=device)], dim=-1)
    if sink is None:
        sink = torch.randn(H, generator=g, device=device, dtype=torch.float32)
    else:
        sink = torch.full((H,), float(sink), device=device, dtype=torch.float32)
    return q, kv.contiguous(), sink, idxs.contiguous(), float(D) ** -0.5


def _launch_fork(q, kv, sink, idxs, scale, *, has_sink=True, want_lse=True, u_max_tiles=None):
    """ONE launch of the fork over zero-copy views, exactly as the adapter binds them: the pre-pass into fresh union views,
    sentinel-filled O, NaN-filled LSE, the kernel call under ``set_sync_debug_mode("error")`` (no D2H on the execute path)
    with ``torch.cuda.memory_allocated`` pinned across it (one call, no copy).  Returns ``(o, lse | None, union_ntiles)``."""
    b, s, h, d = (int(v) for v in q.shape)
    assert (h, d) == (H, D) and q.is_contiguous() and kv.is_contiguous(), "the fork's work item is 64 heads x 512, BSHD compact"
    n_kv, nc = int(kv.shape[1]), n_clusters_for(s)
    ut = u_max_tiles_for(int(idxs.shape[2])) if u_max_tiles is None else int(u_max_tiles)
    fn = _fork_compiled(ut, has_sink, b, s, n_kv, nc, want_lse)
    kv2d = kv.view(b * n_kv, d)
    assert kv2d.data_ptr() == kv.data_ptr(), "the [B*N, 512] KV view must be zero-copy (one buffer, one descriptor)"
    sinks = sink if has_sink else torch.zeros(h, dtype=torch.float32, device=q.device)
    o = torch.full_like(q, _SENTINEL)
    lse = torch.full((b, h, s), math.nan, dtype=torch.float32, device=q.device) if want_lse else None
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ids, bits, nt = _run_union_lists(idxs, seq_len=s, n_kv_rows=n_kv, u_max_tiles=ut)  # FLAGGED pre-pass: allocates, must not sync
        before = torch.cuda.memory_allocated()
        fn(q, kv2d, o, lse, sinks, ids, bits, nt, SA.d512_scale_log2(scale), stream)
        after = torch.cuda.memory_allocated()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert after == before, f"the fork launch allocated {after - before} bytes (it must be one call over zero-copy views)"
    return o, lse, nt


def _dead_tokens(idxs, n_kv):
    """``[B, S]`` bool: tokens whose list names NO in-range row (``-1`` everywhere, or only sanitised ids)."""
    return ~((idxs >= 0) & (idxs < n_kv)).any(dim=-1)


def _check_fork(o, lse, q, kv, sink, idxs, scale, *, label, has_sink=True):
    """Every accept bar at once (plan 4 row 4, sdpa-invariants sections 1-2):
    * no sentinel survivor on any row ``< S`` (dead rows are STORED as zeros, never skipped);
    * O within the bf16 budget of the gathered oracle;
    * LSE within ``1e-4`` of the fp64 arm on finite cells, the non-finite cells EXACT, no NaN anywhere;
    * every keyless token: ``O == 0`` exactly (a SELECT, never ``residue * 0``) and ``LSE == sink`` exactly with a sink /
      ``-inf`` without one.
    ``idxs`` must already be the oracle's vocabulary (``-1`` = no key, everything else in ``[0, N)``)."""
    n_kv = int(kv.shape[1])
    sent = torch.full((), _SENTINEL, dtype=o.dtype, device=o.device)
    survivors = int((o == sent).sum())
    assert survivors == 0, f"{label}: {survivors} sentinel cells survived (rows never stored) at {(o == sent).any(-1).nonzero()[:8].tolist()}"
    assert torch.isfinite(o.float()).all(), f"{label}: non-finite O"
    ref_sink = sink if has_sink else torch.full_like(sink, -math.inf)
    ref_o, _ = R.sparse_attention_reference(q, kv, ref_sink, idxs, scale)
    _assert_bf16_budget(o, ref_o, kv, f"{label} O")
    dead = _dead_tokens(idxs, n_kv).nonzero().tolist()
    for b_, tok in dead:
        assert torch.equal(
            o[b_, tok], torch.zeros_like(o[b_, tok])
        ), f"{label}: keyless token ({b_}, {tok}) has O != 0 (max |O| {o[b_, tok].float().abs().max().item()})"
    if lse is not None:
        _, ref_lse = R.sparse_attention_reference(q, kv, ref_sink, idxs, scale, **_EXACT)
        assert not torch.isnan(lse).any(), f"{label}: NaN / unwritten LSE cells at {torch.isnan(lse).nonzero()[:8].tolist()}"
        fin = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), fin), f"{label}: the finite / non-finite LSE pattern differs from the oracle"
        torch.testing.assert_close(lse[fin], ref_lse[fin], rtol=0, atol=1e-4, msg=lambda m: f"{label} LSE: {m}")
        assert torch.equal(lse[~fin], ref_lse[~fin]), f"{label}: non-finite LSE cells differ from the oracle"
        for b_, tok in dead:
            assert torch.equal(
                lse[b_, :, tok], ref_sink
            ), f"{label}: keyless token ({b_}, {tok}) LSE != {'sink' if has_sink else '-inf'}: {lse[b_, :4, tok].tolist()}"
    return ref_o


# ---------------------------------------------------------------------------- any box with the DSL: module-level pins (no launch)
@requires_dsl
@requires_fork
def test_fork_desc_version_is_one_and_wired_into_every_smem_tile():
    """The two DESC_VERSION pins of ``test_sdpa_fwd_dsl_sm107.py``, ported because the fork lives outside ``sm107/``: the
    d512 slabs put ``sP_xfer`` at exactly 262144, past the 14-bit version-0 descriptor window (the accumulator comes out
    EXACTLY zero, no crash -- mma-tma-matrix.md section 6), so the module constant must be 1 -- and it is only meaningful if
    EVERY ``SmemTile(`` takes ``desc_version=DESC_VERSION`` (the MXFP8 sibling shipped NaN from one re-literalled tile)."""
    mod = _fork_module(1)
    assert mod.DESC_VERSION == 1, f"DESC_VERSION={mod.DESC_VERSION}: the fork inherits the P-xfer ring at 262144 and needs the 15-bit descriptor root"
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    n_tiles = len(re.findall(r"\bSmemTile\($", code, re.M))
    assert n_tiles > 0
    n_wired = code.count("desc_version=DESC_VERSION")
    assert n_wired == n_tiles, f"{n_tiles} SmemTile(s) but {n_wired} wired to DESC_VERSION"
    assert not re.search(r"desc_version=[01]\b", code), "a re-literalled desc_version bypasses DESC_VERSION"


@requires_dsl
@requires_fork
def test_fork_validator_pins_read_tile_arrivers_to_the_issuing_warp_ledger():
    """Every warp that consumes the tile id to issue gathers MUST ``read_tile_id_arrive`` (the scheduler ring re-arms a slot
    after its credits; a non-crediting reader lags a wrap and gathers the wrong rows, silent).  The M3 verdict -- THREE
    dedicated gather-issuing warps per CTA (warps 8-10; warp 11 is the spare that completes their warpgroup) -- turns the dense
    d512 ledger of 25 (config_sm107 :1141-1156, validated against the 8-warp parent) into ``25 + 3 x 4 CTAs = 37``.  The fork
    exposes the ledger (``READ_TILE_ARRIVERS`` / ``N_GATHER_WARPS`` / ``_BASE_CFG``) and its validator re-derives it from the
    body; this pins the stamped ``CFG`` value to the module constant, to the derivation AND to the literal 37, so a silent
    change to EITHER input (the parent's ledger, the issuer count) is a finding rather than a pass."""
    mod = _fork_module(1)
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    assert "_validate_sparse_cfg" in code and re.search(
        r"READ_TILE_ARRIVERS\s*[=!]=", code
    ), "the validator must compare READ_TILE_ARRIVERS against the derived ledger"
    cga = int(mod.CFG.CGA_M) * int(mod.CFG.CGA_N)
    assert (
        cga == 4 and int(mod._BASE_CFG.READ_TILE_ARRIVERS) == 25
    ), f"the parent is the cga4x1 body with a validated ledger of 25 (got cga={cga}, {mod._BASE_CFG.READ_TILE_ARRIVERS})"
    assert int(mod.N_GATHER_WARPS) == 3, f"N_GATHER_WARPS={mod.N_GATHER_WARPS}: M3 sized the gather issue at THREE warps per CTA (2.3x -> 3)"
    ledger = int(mod._BASE_CFG.READ_TILE_ARRIVERS) + int(mod.N_GATHER_WARPS) * cga
    assert (
        int(mod.CFG.READ_TILE_ARRIVERS) == int(mod.READ_TILE_ARRIVERS) == ledger == 37
    ), f"READ_TILE_ARRIVERS: CFG={mod.CFG.READ_TILE_ARRIVERS} module={mod.READ_TILE_ARRIVERS} derived={ledger} (the landed ledger is 37)"
    assert mod.CFG.SCHEDULER_POLICY == 0 and mod.CFG.MASK_FLAGS == 0 and mod.CFG.STAGES_KV == 2, "the fork's config invariants (plan 1(h))"
    assert mod.CFG.PACK_GQA == 1 and mod.CFG.QH_PER_KH == H and mod.CFG.HAS_SINK == 1, "PackGQA rows: 64 heads per token, sink per head"


@requires_dsl
@requires_fork
@pytest.mark.parametrize(
    "kw, match",
    [
        pytest.param(dict(heads_per_tile=32), r"64|heads_per_tile|QH_PER_KH|TP", id="tp2-32-heads-refused"),
        # The validator's PREFIX, not a predicate's wording: predicate order is the fork's, and for fp8 the shared make_cfg_d512
        # picks the FP8 ring depths first (STAGES_KV 3 / XFER_STAGES 4), so that predicate fires before the dtype one.
        pytest.param(dict(dtype_qkv=0), r"^sparse_attention_d512: ", id="fp8-not-a-f16-d512-body"),
        pytest.param(dict(u_max_tiles=256), r"2\*\*15|32768|u_max_tiles|column", id="union-column-index-past-15-bits"),
    ],
)
def test_fork_validator_rejects_bad_params(kw, match):
    """``_validate_sparse_cfg`` raises ``ValueError`` at module load (plan time, never at execute) for a specialization the
    body cannot serve -- surfaced through ``load_template`` exactly as the adapter would see it.  The contract under test is
    "the fork's validator declines, typed, at load"; which predicate names the refusal first is the validator's business."""
    from cudnn.frost.template_loader import load_template

    fork = SA.d512_fork_module()
    params = fork.SparseAttentionD512Params(**kw)
    with pytest.raises(ValueError, match=match):
        load_template(_fork_path(), params, tag="mqa_sparse_attention_d512_reject")


_FORK_SASS_PROBE = textwrap.dedent("""
    import glob, importlib.util, os, subprocess, sys
    dump, cands = sys.argv[1], sys.argv[2:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump          # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "cubin"            # the wheel nvdisasm cannot decode sm_107a: keep the cubin, disassemble it ourselves
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")  # trace + ptxas for Rubin on ANY box (no launch)
    from cudnn.mqa_sparse_attention_block.kernels import sparse_attention as SA
    from cudnn.frost.template_loader import load_template
    fork = SA.d512_fork_module()
    path = importlib.util.find_spec(SA.D512_FORK_MODULE).origin
    mod = load_template(path, fork.SparseAttentionD512Params(u_max_tiles=1), tag="mqa_sparse_attention_d512_sass")
    mod.compile(1, 16, 32, 4, True)  # (b, s, n_kv_rows, n_clusters, has_lse)
    cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")))
    if not cubins:
        print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
    nvd = None
    for c in cands:
        try:
            proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            print("REJECT", c, "->", repr(exc)); continue
        if proc.returncode == 0 and proc.stdout.strip():
            nvd = c; print("NVDISASM", c); break
        print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if nvd is None:
        print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
    sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
    print("FORK CUBIN", os.path.basename(cubins[-1]))
    print("FORK GATHER4", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4" in ln))
    print("FORK GATHER4_2CTA", sum(1 for ln in sass if "UTMALDG" in ln and "GATHER4.2CTA" in ln))
    print("FORK UTMALDG", sum(1 for ln in sass if "UTMALDG" in ln))
    print("FORK SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("FORK USETMAXREG", sum(1 for ln in sass if "USETMAXREG" in ln))
    print("FORK LINES", len(sass))
    """)


@requires_dsl
@requires_fork
def test_fork_sm107a_trace_compile_sass(tmp_path):
    """Trace + compile the FORK for Rubin on THIS box (no device match needed: ``CUTE_DSL_ARCH=sm_107a``) and read the SASS:
    the gather4 producer is present (``UTMALDG.2D.GATHER4``), the register split is real (``USETMAXREG``), and the spill count
    is PRINTED (the ``STL/LDL == 0`` gate on the 40-register TMA-LDG / TMA-STG warps is the fork author's, per warp -- a flat
    count cannot attribute it).  SKIPS when the DSL predates sm_107a or no candidate nvdisasm decodes it; a compile failure
    is a FAIL (the fork must build for sm_107a here -- that is the A100's job in this workflow)."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / "fork_sm107a"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _FORK_SASS_PROBE, str(dump), *cands], capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, f"fork trace-compile for sm_107a failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("FORK ") and len(ln.split()) == 3 and ln.split()[2].isdigit()}
    print(f"\nfork sm_107a SASS: {stats}")
    assert stats["GATHER4"] > 0, "the fork's K/V producer is gather4 (UTMALDG.2D.GATHER4)"
    assert stats["USETMAXREG"] > 0, "the per-role register split must reach SASS (USETMAXREG.*), not just PTX"


# ---------------------------------------------------------------------------- Rubin: PackGQA-port sites
@requires_dsl
@requires_rubin
@requires_fork
def test_fork_packgqa_rows_sink_per_head_and_head_permutation():
    """The nine row-space -> (token, head) sites (plan 1(i) risk 8) pinned from the outside: (1) the sink enters the LSE per
    HEAD -- ``lse[b, h, tok] == logaddexp(lse_keys[b, tok], sink[h])`` with a wide per-head spread (a per-tile sink would
    give one value per token); (2) rows are (token, head): permuting the heads of ``q`` and ``sink`` permutes O and LSE rows
    BITWISE, because every row's arithmetic (its dot products, its softmax lane, its TMEM columns) is its own."""
    q, kv, _, idxs, scale = _case_fork(batch=2, seq_len=12, ratio=1, seed=7)
    g = torch.Generator(device="cuda").manual_seed(70)
    sink = (torch.randn(H, generator=g, device="cuda") * 3.0).float().contiguous()
    o, lse, _ = _launch_fork(q, kv, sink, idxs, scale)
    _check_fork(o, lse, q, kv, sink, idxs, scale, label="per-head sink")
    _, lse_keys = R.sparse_attention_reference(q, kv, torch.full_like(sink, -math.inf), idxs, scale, **_EXACT)
    want = torch.logaddexp(lse_keys, sink.view(1, H, 1))
    torch.testing.assert_close(lse, want, rtol=0, atol=1e-4, msg=lambda m: f"LSE is not logaddexp(keys, sink[h]) per head: {m}")
    assert float(lse[0, :, 0].max() - lse[0, :, 0].min()) > 1.0, "the per-head sink must spread the LSE across the 64 heads of one token"
    perm = torch.randperm(H, generator=g, device="cuda")
    o_p, lse_p, _ = _launch_fork(q[:, :, perm].contiguous(), kv, sink[perm].contiguous(), idxs, scale)
    assert torch.equal(o_p, o[:, :, perm]), "permuting the heads must permute the O rows bitwise (row = token_local * 64 + head)"
    assert torch.equal(lse_p, lse[:, perm]), "permuting the heads must permute the LSE rows bitwise (lse[b, row_head, tok])"


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_membership_isolates_each_token_slot_from_its_cluster_neighbours():
    """Four adjacent tokens with DISJOINT lists share one union tile: token ``4c + j`` must attend ONLY its own 8 rows.  The
    oracle comparison proves the positive; the negative control -- the same rows attended over the whole cluster union, what a
    missing / all-ones membership mask would compute -- must NOT match on any token."""
    S, B, K, N = 8, 1, 8, 64
    g = torch.Generator(device="cuda").manual_seed(8)
    q = torch.randn(B, S, H, D, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(B, N, D, generator=g, device="cuda").to(torch.bfloat16)
    sink = torch.randn(H, generator=g, device="cuda", dtype=torch.float32)
    tok = torch.arange(S, device="cuda")
    idxs = ((tok // 4) * 32 + (tok % 4) * K).view(B, S, 1) + torch.arange(K, device="cuda").view(1, 1, K)  # disjoint 8-row lists per token
    idxs = idxs.int().contiguous()
    scale = float(D) ** -0.5
    o, lse, nt = _launch_fork(q, kv, sink, idxs, scale)
    assert bool((nt == 1).all()), nt.tolist()
    _check_fork(o, lse, q, kv, sink, idxs, scale, label="disjoint neighbour lists")
    union_idx = idxs.view(B, S // 4, 4 * K).repeat_interleave(4, dim=1).contiguous()  # every token lists the cluster's 32 rows
    o_union, lse_union = R.sparse_attention_reference(q, kv, sink, union_idx, scale)
    per_tok = (o.float() - o_union.float()).abs().amax(dim=(2, 3))  # [B, S]
    assert bool((per_tok > 2**-5).all()), f"a token that saw its neighbours' keys would match the union oracle: {per_tok.tolist()}"
    assert bool(((lse - lse_union).abs().amax(dim=1) > 1e-2).all()), "the LSE must see only the token's own keys"


# ---------------------------------------------------------------------------- Rubin: the degenerate sweep (sdpa-invariants sections 1 / 9)
_SWEEP_S = [1, 2, 3, 4, 5, 127, 128, 129, 300, 1000, 2048]
_SWEEP_KINDS = [("win", 0), ("r1", 1), ("r2", 2)]  # window-only (the has_compressed_kv=False shape), ratio 1, ratio 2 (= S < ratio at S=1)


@requires_dsl
@requires_rubin
@requires_fork
@pytest.mark.parametrize("kind, ratio", _SWEEP_KINDS, ids=[k for k, _ in _SWEEP_KINDS])
@pytest.mark.parametrize("batch", [1, 2], ids=["B1", "B2"])
@pytest.mark.parametrize("seq_len", _SWEEP_S, ids=[f"S{s}_" for s in _SWEEP_S])
def test_fork_degenerate_sweep(seq_len, batch, kind, ratio):
    """``S in {1,2,3,4,5,127,128,129,300,1000,2048} x B in {1,2} x {window-only, ratio 1, ratio 2}``: one cluster, ``S % 4 != 0``
    (padded tokens in the tail cluster get no rows and no keys), ``S < ratio`` (``N_c == 0``, the window list alone), ``B > 1``
    with a single cluster, the scheduler ring wrapping (S = 2048 -> 512 clusters), tail tiles with ``-1`` padding.  Every case is
    one JIT (``compile`` pins the shape), so run it in ``--k-any`` groups by ``S<n>_``."""
    q, kv, sink, idxs, scale = _case_fork(batch=batch, seq_len=seq_len, ratio=ratio, seed=seq_len * 7 + batch)
    if ratio > seq_len:  # S < ratio: no compressed rows -- the list is the window alone
        assert int(kv.shape[1]) == seq_len and int(idxs.shape[2]) == min(seq_len, 16)
    o, lse, nt = _launch_fork(q, kv, sink, idxs, scale)
    assert nt.shape == (batch, n_clusters_for(seq_len)) and int(nt.min()) >= 1
    _check_fork(o, lse, q, kv, sink, idxs, scale, label=f"S={seq_len} B={batch} {kind}")


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_keyless_token_and_all_minus_one_quad():
    """One token with no key inside a live cluster (its 64 rows: ``O == 0`` by SELECT, ``LSE == sink`` exactly, its neighbours
    within budget) and two hand-built all-``-1`` quads (one mid-sequence, one the LAST cluster): ``n_tiles == 1`` (never 0),
    one zero-MMA tile, every row dead, no NaN anywhere."""
    q, kv, sink, idxs, scale = _case_fork(batch=2, seq_len=40, ratio=1, seed=3)
    idxs[0, 5] = -1
    idxs[1, 8:12] = -1
    idxs[0, 36:40] = -1
    o, lse, nt = _launch_fork(q, kv, sink, idxs, scale)
    assert int(nt[1, 2]) == 1 and int(nt[0, 9]) == 1, nt.tolist()
    dead = _dead_tokens(idxs, int(kv.shape[1]))
    assert bool(dead[0, 5]) and bool(dead[1, 8:12].all()) and bool(dead[0, 36:40].all()) and int(dead.sum()) == 9
    _check_fork(o, lse, q, kv, sink, idxs, scale, label="keyless token + all -1 quads")


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_duplicates_within_and_across_lists():
    """Duplicates are distinct slots and count TWICE (the multiset union carries copies): within one list (a slot repeated
    2x and 3x), across the lists of one cluster (token 10 names token 9's id: ONE union copy carrying both bits), a 4x
    repeat in another cluster.  Within budget of the oracle; the duplicated rows CHANGE vs the plain list; clusters whose
    lists are untouched reproduce the plain run bitwise (identical gathers, identical arithmetic)."""
    q, kv, sink, idxs, scale = _case_fork(batch=1, seq_len=16, ratio=1, seed=5)
    dup = idxs.clone()
    dup[0, 9, 1] = dup[0, 9, 0]
    dup[0, 9, 2] = dup[0, 9, 0]
    dup[0, 10, 0] = dup[0, 9, 0]
    dup[0, 12, :4] = dup[0, 12, 0]
    o_dup, lse_dup, nt_dup = _launch_fork(q, kv, sink, dup, scale)
    _check_fork(o_dup, lse_dup, q, kv, sink, dup, scale, label="duplicates")
    o_one, lse_one, _ = _launch_fork(q, kv, sink, idxs, scale)
    assert not torch.equal(o_dup[0, 9], o_one[0, 9]) and not torch.equal(o_dup[0, 12], o_one[0, 12]), "a duplicated slot must change its row"
    assert torch.equal(o_dup[0, :8], o_one[0, :8]) and torch.equal(lse_dup[0, :, :8], lse_one[0, :, :8]), "clusters 0-1 are untouched: bitwise"


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_ids_outside_the_kv_range_are_sanitised_to_no_key():
    """Ids ``>= N`` (incl. exactly ``N`` and ``2**30``) and ``< -1`` are "no key": dropped by the pre-pass with their bits
    cleared (a zero-filled row with a set bit would be a real logit of 0).  At ``B = 2`` the flat row space is ``b * N + id``,
    so an unmasked ``id >= N`` of batch 0 would gather batch 1's rows -- the oracle on the SANITISED list is the truth."""
    q, kv, sink, idxs, scale = _case_fork(batch=2, seq_len=12, ratio=2, seed=6)
    n = int(kv.shape[1])
    bad = idxs.clone()
    bad[0, 3, :4] = torch.tensor([n, n + 7, -5, n - 1], device="cuda", dtype=torch.int32)  # n - 1 is the largest VALID id
    bad[1, 10, 0] = 2**30
    o, lse, _ = _launch_fork(q, kv, sink, bad, scale)
    clean = torch.where((bad >= 0) & (bad < n), bad, torch.full_like(bad, -1))
    _check_fork(o, lse, q, kv, sink, clean, scale, label="sanitised ids")
    o_plain, _, _ = _launch_fork(q, kv, sink, idxs, scale)
    assert not torch.equal(o[0, 3], o_plain[0, 3]), "the planted slots replaced live ids, so row (0, 3) must have changed"


@requires_dsl
@requires_rubin
@requires_fork
@pytest.mark.parametrize("n_rows", [128, 200, 256, 384, 700, 768, 1152, 2304], ids=lambda n: f"rows{n}")
def test_fork_forced_n_tiles(n_rows):
    """Lists built to realise ``n_tiles = ceil(n_rows / 128) in {1, 2, 3, 6, 9, 18}`` per cluster (``n_rows`` distinct KV rows
    spread round-robin over the four tokens, ``K = 640`` slots each, ``-1``-padded; 200 and 700 leave the last tile partial):
    the K/V ring (``STAGES_KV = 2``) wraps from 3 tiles on, the eight ``n_tiles`` bounds sites (P14) see every count, and a
    partial last tile mixes live columns with ``-1`` = OOB zero rows.  ONE shape, so one JIT for all eight cases."""
    S, B, K, N = 16, 1, 640, 2560
    want = -(-n_rows // TILE_ROWS)
    assert 4 * K >= n_rows and u_max_tiles_for(K) >= want
    g = torch.Generator(device="cuda").manual_seed(n_rows)
    q = torch.randn(B, S, H, D, generator=g, device="cuda").to(torch.bfloat16)
    kv = torch.randn(B, N, D, generator=g, device="cuda").to(torch.bfloat16)
    sink = torch.randn(H, generator=g, device="cuda", dtype=torch.float32)
    idxs = torch.full((B, S, K), -1, dtype=torch.int32, device="cuda")
    for c in range(n_clusters_for(S)):
        rows = torch.randperm(N, generator=g, device="cuda")[:n_rows].sort().values
        for j in range(TOKENS_PER_CLUSTER):
            mine = rows[j::TOKENS_PER_CLUSTER]
            idxs[0, TOKENS_PER_CLUSTER * c + j, : mine.numel()] = mine.int()
    scale = float(D) ** -0.5
    o, lse, nt = _launch_fork(q, kv, sink, idxs, scale)
    assert bool((nt == want).all()), f"n_tiles {nt.tolist()} != {want}"
    _check_fork(o, lse, q, kv, sink, idxs, scale, label=f"n_tiles={want} (rows {n_rows})")


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_sink_corners_on_live_and_keyless_rows():
    """``sink = -inf``: live rows == the no-sink math, a keyless row gives ``new_sum = 0 -> beta = +inf`` inside the fold and
    is correct ONLY through the ``_row_empty`` selects (``O == 0``, ``LSE == -inf``; plan 1(e)) -- and it must agree with the
    ``has_sink=False`` specialization within the budget.  ``sink = +inf``: ``O == 0`` and ``LSE == +inf`` on EVERY row, no NaN
    (the ``sink_term`` select keeps ``exp(inf - inf)`` from forming)."""
    q, kv, _, idxs, scale = _case_fork(batch=1, seq_len=16, ratio=1, seed=9)
    idxs[0, 6] = -1  # a keyless token among live ones
    ninf = torch.full((H,), -math.inf, device="cuda", dtype=torch.float32)
    o, lse, _ = _launch_fork(q, kv, ninf, idxs, scale)
    _check_fork(o, lse, q, kv, ninf, idxs, scale, label="sink=-inf")
    assert torch.equal(o[0, 6], torch.zeros_like(o[0, 6])) and bool(torch.isneginf(lse[0, :, 6]).all())
    o_ns, lse_ns, _ = _launch_fork(q, kv, ninf, idxs, scale, has_sink=False)  # the has_sink=False specialization (zeros handed as sinks)
    _check_fork(o_ns, lse_ns, q, kv, ninf, idxs, scale, label="has_sink=False", has_sink=False)
    _assert_bf16_budget(o, o_ns, kv, "sink=-inf vs has_sink=False")
    torch.testing.assert_close(lse, lse_ns, rtol=0, atol=1e-4)
    pinf = torch.full((H,), math.inf, device="cuda", dtype=torch.float32)
    o, lse, _ = _launch_fork(q, kv, pinf, idxs, scale)
    _check_fork(o, lse, q, kv, pinf, idxs, scale, label="sink=+inf")
    assert torch.equal(o, torch.zeros_like(o)), "sink=+inf must give O == 0 on every row"
    assert bool((lse == math.inf).all()), "sink=+inf must give LSE == +inf on every row, never NaN"


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_nan_in_never_referenced_kv_rows_does_not_leak():
    """The block's contract: ``kv_all`` must be finite on every row ANY token of a cluster names (a member-for-A-not-B row is
    multiplied into B's O with ``P = 0`` and ``0 * NaN = NaN``); rows NO list names are never gathered (``-1`` pads are
    TMA-OOB zeros, not reads), so NaN there must not reach O or LSE.  Oracle + budget on the CLEAN kv."""
    q, kv, sink, idxs, scale = _case_fork(batch=1, seq_len=32, ratio=1, seed=11)
    n = int(kv.shape[1])
    referenced = torch.zeros(n, dtype=torch.bool, device="cuda")
    referenced[idxs[(idxs >= 0) & (idxs < n)].long()] = True
    assert not bool(referenced.all()), "the case must leave some rows unreferenced"
    kv_nan = kv.clone()
    kv_nan[0, ~referenced] = math.nan
    o, lse, _ = _launch_fork(q, kv_nan, sink, idxs, scale)
    _check_fork(o, lse, q, kv, sink, idxs, scale, label="NaN in never-referenced rows")


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_released_list_geometry_two_launches_bitwise_and_zero_copy(capsys):
    """The released list geometry (window 128 ++ top-k 512 at ratio 2: ``K = 640``, ``u_max_tiles = 20``) at ``S = 1024``
    (``S % 4 == 0``, 256 clusters -> every CTA runs a 2nd work item on a 204-SM part, realistic ``n_tiles`` ~ 8-12): two
    launches bitwise; the handover is zero-copy (``kv2d`` shares ``kv_all``'s pointer, O lands in the caller's sentinel-filled
    buffer, no allocation across the call -- all asserted inside ``_launch_fork``); the union tile statistics are PRINTED for
    the perf table's ``n_tiles`` column.  ``S`` must be >= 1024: the synthetic Indexer tail caps the compressed list at
    ``S // ratio`` ids (``compressed_idxs_synthetic``), so ``S = 1000`` yields ``K = 628`` and never reaches the released pitch."""
    q, kv, sink, idxs, scale = _case_fork(batch=1, seq_len=1024, ratio=2, window=128, topk=512, seed=13)
    assert int(idxs.shape[2]) == 640 and u_max_tiles_for(640) == 20
    o1, lse1, nt = _launch_fork(q, kv, sink, idxs, scale)
    o2, lse2, _ = _launch_fork(q, kv, sink, idxs, scale)
    assert torch.equal(o1, o2) and torch.equal(lse1, lse2), "two launches on identical inputs must be bitwise identical"
    _check_fork(o1, lse1, q, kv, sink, idxs, scale, label="S=1024 K=640")
    with capsys.disabled():
        print(f"\nfork S=1024 K=640: n_tiles mean {nt.float().mean().item():.2f} min {int(nt.min())} max {int(nt.max())} over {nt.numel()} clusters")


@requires_dsl
@requires_rubin
@requires_fork
def test_fork_s32768_released_geometry_wraps_the_scheduler_ring(capsys):
    """The released prompt length (plan risk 11): ``S = 32768``, ``K = 640`` (``u_max_tiles = 20``), ``B = 1`` -> 8192 clusters
    over ~53 resident clusters on 212 SMs = ~155 tiles per CTA.  A scheduler-ring over- / under-arrival compounds PER WRAP
    (mbarrier-patterns P3: a silent one-tile-early advance below ~7 wraps, a wedge past ~12+), so every case at S <= 2048 (512
    clusters, ~2.4 wraps) is blind to it; this is the one in-process shape that is not.  Same bars as every accept case
    (sentinel survivors 0, the bf16 O budget, LSE 1e-4 / non-finite exact, no allocation across the launch, no sync); the
    union tile statistics are printed.  ~4 GiB of q + O plus a 2 GiB [S, S/2] score matrix in the list generator: run it alone
    on its own GPU (``--k-any=test_fork_s32768``), under the runner's timeout (exit 124 = HANG -> the ring, not the limit)."""
    q, kv, sink, idxs, scale = _case_fork(batch=1, seq_len=_S_32K, ratio=2, window=128, topk=512, seed=17)
    assert int(idxs.shape[2]) == 640 and u_max_tiles_for(640) == 20 and n_clusters_for(_S_32K) == 8192
    o, lse, nt = _launch_fork(q, kv, sink, idxs, scale)
    _check_fork(o, lse, q, kv, sink, idxs, scale, label=f"S={_S_32K} K=640")
    with capsys.disabled():
        print(f"\nfork S={_S_32K} K=640: n_tiles mean {nt.float().mean().item():.2f} min {int(nt.min())} max {int(nt.max())} over {nt.numel()} clusters")


# ---------------------------------------------------------------------------- Rubin: fresh-process launch loop (opt-in, slow)
_FRESH_PROCESS_CHILD = textwrap.dedent("""
    import math, sys, torch
    test_python, S, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sys.path.insert(0, test_python)  # the oracle, rootdir-qualified as the suite imports it
    from fe_api.mqa_sparse_attention_block import mqa_block_reference as R
    from cudnn.mqa_sparse_attention_block.kernels import sparse_attention as SA
    from cudnn.mqa_sparse_attention_block.kernels.union_lists import TILE_ROWS, TOKENS_PER_CLUSTER, WORDS_PER_SLOT, build_union_lists, n_clusters_for, u_max_tiles_for
    from cudnn.frost.template_loader import load_template
    import importlib.util
    B, H, D, WINDOW, TOPK, RATIO = 1, 64, 512, 128, 512, 2
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=g, device=dev).to(torch.bfloat16)
    n_c = S // RATIO
    kv = torch.randn(B, S + n_c, D, generator=g, device=dev).to(torch.bfloat16)
    idxs = torch.cat([R.window_idxs(B, S, WINDOW, device=dev), R.compressed_idxs_synthetic(B, S, RATIO, TOPK, g, device=dev)], dim=-1).contiguous()
    sink = torch.randn(H, generator=g, device=dev, dtype=torch.float32)
    scale = float(D) ** -0.5
    K = int(idxs.shape[2]); n_kv = S + n_c; nc = n_clusters_for(S); ut = u_max_tiles_for(K)
    ids = torch.empty(B, nc, TILE_ROWS * ut, dtype=torch.int32, device=dev)
    bits = torch.empty(B, nc, ut, TOKENS_PER_CLUSTER, WORDS_PER_SLOT, dtype=torch.int32, device=dev)
    nt = torch.empty(B, nc, dtype=torch.int32, device=dev)
    build_union_lists(idxs, seq_len=S, n_kv_rows=n_kv, u_max_tiles=ut, out_ids=ids, out_bits=bits, out_ntiles=nt)
    fork = SA.d512_fork_module()
    mod = load_template(importlib.util.find_spec(SA.D512_FORK_MODULE).origin, fork.SparseAttentionD512Params(u_max_tiles=ut), tag="mqa_sparse_attention_d512_fresh")
    fn = mod.compile(B, S, n_kv, nc, True)
    o = torch.full_like(q, 1.5e30)
    lse = torch.full((B, H, S), math.nan, dtype=torch.float32, device=dev)
    fn(q, kv.view(B * n_kv, D), o, lse, sink, ids, bits, nt, SA.d512_scale_log2(scale), int(torch.cuda.current_stream(dev).cuda_stream))
    torch.cuda.synchronize()  # ONE launch per process: a hang parks here and the parent's timeout classifies it as 124
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    _, ref_lse = R.sparse_attention_reference(q, kv, sink, idxs, scale, p_dtype=torch.float32, out_dtype=torch.float32)
    ok_o = bool(((o.float() - ref_o.float()).abs() <= 2**-7 * ref_o.float().abs() + 2**-8 * kv.float().abs().max()).all())
    ok_lse = bool(((lse - ref_lse).abs() <= 1e-4).all())
    survivors = int((o == torch.full((), 1.5e30, dtype=o.dtype, device=dev)).sum())
    print(f"CHILD S={S} seed={seed} ok_o={ok_o} ok_lse={ok_lse} sentinel_survivors={survivors} n_tiles_mean={nt.float().mean().item():.2f}", flush=True)
    sys.exit(0 if (ok_o and ok_lse and survivors == 0) else 5)
    """)


@requires_dsl
@requires_rubin
@requires_fork
@pytest.mark.parametrize("seq_len", _FRESH_PROCESS_S, ids=[f"fresh_S{s}" for s in _FRESH_PROCESS_S])  # bracket-free ids: `-k fresh_S32768`
def test_fork_fresh_process_launches(seq_len, capsys):
    """``N`` fresh processes (``MQA_D512_FRESH_PROCESS_LAUNCHES=N``, the Validate agent's marker; unset = skip typed), each ONE
    JIT + ONE launch of the released list geometry at ``S`` -- 2048 (512 clusters -> the scheduler ring wraps ~2.4x on 212 SMs)
    AND 32768 (8192 clusters -> ~155 tiles per CTA: the P3 ring-drift class advances a tile early below ~7 wraps and wedges past
    ~12+, so nothing under 32K can show it) -- classified by exit code: ``0`` ok / ``124`` hang (killed at the per-S child
    timeout) / other (a numerics miss exits 5, a launch failure raises).  The undefined-behaviour hazards -- an un-elected init,
    an over-arrival, an undrained ring -- fail at a per-LAUNCH rate with correct numerics whenever they complete, so a single
    passing run proves nothing and a second launch in one process reuses a warm artifact (frost-gotchas.md, "count over >= 12
    FRESH PROCESSES").  The three counts are printed; the assertion is ``hang == other == 0``.  Run each S arm in its own runner
    call (``--k-any=fresh_S2048`` / ``--k-any=fresh_S32768``): 12 x ~60 s at 2K, 12 x ~3 min at 32K."""
    marker = os.environ.get(_FRESH_PROCESS_ENV, "")
    if not marker.strip():
        pytest.skip(f"set {_FRESH_PROCESS_ENV}=<launch count> (the Validate agent's marker) to run the per-process launch loop")
    n = int(marker)
    child_timeout = _FRESH_PROCESS_TIMEOUT_S[seq_len]
    test_python = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    counts = {"ok": 0, "hang": 0, "other": 0}
    lines = []
    for i in range(n):
        cmd = [sys.executable, "-c", _FRESH_PROCESS_CHILD, test_python, str(seq_len), str(i)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=child_timeout, env=dict(os.environ))
            rc, tail = proc.returncode, (proc.stdout.strip().splitlines()[-1:] + proc.stderr.strip().splitlines()[-3:])
        except subprocess.TimeoutExpired as exc:
            rc, tail = 124, [f"killed after {child_timeout} s", str(exc.stdout or "")[-300:]]
        cls = "ok" if rc == 0 else ("hang" if rc == 124 else "other")
        counts[cls] += 1
        lines.append(f"  launch {i:2d}: exit {rc:3d} {cls:5s} {' | '.join(t for t in tail if t)}")
    with capsys.disabled():
        print(f"\nfresh-process launches S={seq_len} x {n}: ok={counts['ok']} hang(124)={counts['hang']} other={counts['other']}")
        print("\n".join(lines))
    assert counts["hang"] == 0 and counts["other"] == 0, f"fresh-process launches: {counts} (see the per-launch lines above)"


# ---------------------------------------------------------------------------- the block-side declines that belong to the fork (any device)
@requires_cuda
@pytest.mark.skipif(_cc() == _SM107, reason="this IS the target arch")
def test_d512_adapter_declines_a_non_rubin_device_typed():
    """Off Rubin the adapter's decline is typed either way: the fork absent -> "has not landed"; present -> the arch."""
    ad = SA.D512SparseAttention(batch=1, seq_len=8, n_kv_rows=8, n_heads=H, head_dim=D, topk=8, scale=float(D) ** -0.5, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError) as ei:
        ad.check_support()
    msg = str(ei.value)
    assert ("Rubin" in msg or "SM107" in msg) if SA.d512_fork_available() else ("has not landed" in msg), msg


def test_d512_adapter_declines_a_head_geometry_other_than_64x512_before_touching_a_device():
    """``(n_heads, head_dim) != (64, 512)`` is a device-independent typed decline (TP=2's 32 heads included), ahead of the
    fork-present and arch checks -- so it reads the same on the A100 and on Rubin, with or without the fork."""
    for h, d in ((32, 512), (128, 512), (64, 256)):
        ad = SA.D512SparseAttention(batch=1, seq_len=8, n_kv_rows=8, n_heads=h, head_dim=d, topk=8, scale=float(d) ** -0.5, dtype=torch.bfloat16, device="cpu")
        with pytest.raises(NotImplementedError, match="64"):
            ad.check_support()


def test_d512_adapter_declines_a_union_column_index_past_15_bits_and_a_non_half_dtype():
    """``K`` whose worst-case union (``4K`` rows) needs a column index of 15+ bits is a ``ValueError`` at plan time; a non-bf16/f16
    dtype is a ``NotImplementedError`` -- both device-independent, both ahead of the fork-present check."""
    kw = dict(batch=1, seq_len=8, n_kv_rows=8, n_heads=H, head_dim=D, scale=float(D) ** -0.5, device="cpu")
    with pytest.raises(NotImplementedError, match="bf16"):
        SA.D512SparseAttention(topk=8, dtype=torch.float32, **kw).check_support()
    if SA.d512_fork_available():  # the fork-present check precedes the K check; the K check is reachable only once it exists
        with pytest.raises(ValueError, match="column"):
            SA.D512SparseAttention(topk=8192, dtype=torch.bfloat16, **kw).check_support()
