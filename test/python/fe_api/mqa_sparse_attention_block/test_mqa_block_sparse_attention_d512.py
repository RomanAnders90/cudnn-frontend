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
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

# Rootdir-qualified (a bare `import mqa_block_reference` collides in one `pytest fe_api` session).
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R

from cudnn.frost.buffers import cutedsl_requirement_error
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
