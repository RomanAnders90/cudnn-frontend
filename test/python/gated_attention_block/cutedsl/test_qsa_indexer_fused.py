# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_indexer_fused``: the fused decode indexer's contract -- the host half (the dispatch table and the pre-development tables) and
the kernel-level tiers against the public surface the tables and the compile / run entries define.

HOST (no GPU, no trace-compile): the forward-progress placement is pinned over every served cell (``producers_before_consumers``: the
linear index of every scorer CTA of a sequence below every top-k CTA of that sequence); beside it the dispatch table's CTA counts are
pinned against the three standalone kernels' own grid arithmetic (the prepare's warp units, the scorer's ``(chunks, B)`` grid, the top-k's
``(1, rows)`` grid -- computed by THEIR functions and compared), the module's mirrored constants against the kernels' own, the docstring's
dispatch rows against the function, the device decode against the host encoding, the mis-ordered twin against the pin (RED on purpose),
the docstring's tables for presence, and the dead units of the last scorer CTA (a chunk count that is not a multiple of ``units``).  The
cells that import a kernel module skip below the CuTe DSL floor (Rule 7); the dispatch-table cells import no kernel module (the host
facts come through the scorer's front door).

KERNEL LEVEL (``qsa_indexer_fused_step.py`` is the harness: the step's operands, the three-launch reference = the standalone ``qsa_prepare``
with the decode arms -> ``qsa_score_paged`` -> ``qsa_topk_decode`` on the same primed caches, the fused launch behind ONE adapter, the
child-process entries): the fused launch is BITWISE the three-launch form (the list, ``counts``, ``block_lens``, the slot report, the Q / K
bands, the K / V pool rows, the ring rows, the compressed-key cache rows, the logits buffer) over the dispatch rows' cells at the route's
units on the default block size and on the 1024-thread knob, at units 2 / 4 / 8 with ``check_arrivals`` armed; the completing row is
derived bitwise at every alignment; the degenerate matrix
(a padding sequence, ``kv_lens < S_q``, a dead page under a visible block and under the completing block, a completing block past the
table's width, the ring at its minimum cap, the identity region, the dead units of the last scorer CTA); the counter lifecycle (zero
after every launch in fresh processes, a captured FIRST execute replaying bitwise with zero counters, two steps back to back on one
plan); the trap rendering (a grid larger than the resident capacity terminates under an armed deadline; a placement that strands the
consumers traps instead of hanging); the host PTX census of the fused rendering and the per-role register pin.  Every kernel-level
cell compiles the fused plan FIRST (against the pre-body module that was the stubs' ``NotImplementedError``: the test commit's RED run;
against the body a compile-key or operand spelling the front door does not take fails in the harness's one adapter), a Rule 7 decline
by name (the DSL floor / target, the device family, the SMEM ceiling) is a skip, and a device the three-launch reference cannot run on
skips by the scorer's own reason.  The census cells trace-compile the served body for sm_107a on the host under the target's device
facts.  The kernel timeout is never raised: a child that does not return within its budget FAILS naming the hang.
"""

import importlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block.kernels import qsa_indexer_fused as F
from cudnn.QSA import selection_paged as SP

pytestmark = [pytest.mark.L0]

_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("qsa_indexer_fused")  # None at or above the floor
needs_dsl = pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed for the step's operands")
requires_rubin = pytest.mark.requires_rubin  # the suite's registered marker (conftest.py): skipped off SM107
_HERE = Path(__file__).resolve().parent
_HARNESS = _HERE / "qsa_indexer_fused_step.py"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import cubin_pin_support as cps  # noqa: E402  (the pins' skip-or-fail contract: the DSL string + the bundled assembler the register figures were taken under)

# the Qwen 3.8 Flash-Next block's geometry (api.py's QsaSpec defaults: index_heads 4, index_head_dim 128; d_head 256)
H_Q, H_KV, D, INDEX_HEADS, INDEX_HEAD_DIM, TOP_K = 24, 2, 256, 4, 128, 512
BATCHES = (1, 4, 32)
S_QS = (1, 4)
N_BLOCKS_MAX = (520, 8192, 32768)  # B 4 x 2K (the identity cell), 32K and 128K KV tokens at 4 tokens per block
PAGES = (16, 32, 64)
UNITS = (1, 2, 4, 8)
THREADS = (1024, 512)
SM_COUNT = 212


def _served(units: int, threads: int) -> bool:
    return units * F.SCORER_UNIT_THREADS <= threads


def _table(b, s_q, nbm, units, threads, **kw):
    return F.dispatch_table(b * s_q, b, s_q, H_Q, H_KV, INDEX_HEADS, nbm, units, threads, d=D, index_head_dim=INDEX_HEAD_DIM, **kw)


CELLS = list(itertools.product(BATCHES, S_QS, N_BLOCKS_MAX))
KNOBS = list(itertools.product(UNITS, THREADS))


@pytest.mark.parametrize("units,threads", KNOBS, ids=[f"u{u}_t{t}" for u, t in KNOBS])
@pytest.mark.parametrize("b,s_q,nbm", CELLS, ids=[f"B{b}_S{s}_N{n}" for b, s, n in CELLS])
def test_dispatch_table_places_every_producer_before_its_consumers(b, s_q, nbm, units, threads):
    """Over every served (B, S_q, n_blocks_max, page, units, threads) the served order puts every R1(b) below every R2(b); the
    page size never enters the table (the same table at every page); a unit count the block size cannot hold is a typed refusal."""
    if not _served(units, threads):
        with pytest.raises(ValueError, match="scorer threads exceed the block size"):
            _table(b, s_q, nbm, units, threads)
        return
    tables = [_table(b, s_q, nbm, units, threads) for _ in PAGES]  # the page is a kernel-side fact (the table words), not a dispatch one
    assert all(tb == tables[0] for tb in tables)
    tb = tables[0]
    assert tb.producers_before_consumers(), tb
    assert tb.first_violation() is None
    # the ledger of C1: every row's word has exactly n_chunks producers (one elected lane each) placed below its one consumer
    for seq in range(b):
        producers = [tb.linear_index(F.ROLE_SCORER, seq, k) for k in range(tb.n_chunks)]
        assert len(producers) == tb.n_chunks
        for s in range(s_q):
            consumer = tb.linear_index(F.ROLE_TOPK, seq, s)
            assert max(producers) < consumer < tb.n_ctas
    # R0 waits on nothing and sits first
    assert all(tb.role_of(i)[0] == F.ROLE_PREPARE for i in range(tb.n_prep_ctas))
    assert tb.n_prep_ctas >= 1 and tb.n_chunks >= 1


@pytest.mark.parametrize("order", [o for o in F.DISPATCH_ORDERS if o != "sequence-major"])
@pytest.mark.parametrize("b,s_q,nbm", CELLS, ids=[f"B{b}_S{s}_N{n}" for b, s, n in CELLS])
def test_the_mis_ordered_twin_fails_the_pin(b, s_q, nbm, order):
    """The trap rendering's inputs: both mis-orders are REFUSED by the same pin -- the detector of the forward-progress assumption is
    live, not vacuous.  Only the GLOBAL twin places every sequence's top-k CTAs ahead of the first scorer CTA (the per-sequence twin keeps
    forward progress under in-order dispatch: at most one sequence's consumers wait on undispatched producers)."""
    tb = _table(b, s_q, nbm, 1, 1024, order=order)
    assert not tb.producers_before_consumers()
    v = tb.first_violation()
    assert v is not None and v[2] > v[3], v
    assert tb.n_ctas == _table(b, s_q, nbm, 1, 1024).n_ctas  # the same grid, only the order differs
    first_producer = min(tb.linear_index(F.ROLE_SCORER, bb, k) for bb in range(b) for k in range(tb.n_chunks))
    ahead = sum(1 for bb in range(b) for s in range(s_q) if tb.linear_index(F.ROLE_TOPK, bb, s) < first_producer)
    assert ahead == (b * s_q if order == "consumers-first-global" else s_q), (order, ahead)


@pytest.mark.parametrize("units,threads", [k for k in KNOBS if _served(*k)], ids=[f"u{u}_t{t}" for u, t in KNOBS if _served(u, t)])
@pytest.mark.parametrize("b,s_q,nbm", CELLS, ids=[f"B{b}_S{s}_N{n}" for b, s, n in CELLS])
@needs_dsl
def test_dispatch_table_cta_counts_match_the_three_kernels_grids(b, s_q, nbm, units, threads):
    """The host pin of the CTA counts: the prepare's warp units from ``qsa_prepare.prepare_warp_counts`` (the query arm dropped), the
    scorer's chunk count from ``run_qsa_score_paged``'s arithmetic (``ceil(n_blocks_max / blocks_per_cta)``) folded ``units`` per CTA, the
    top-k's ``(1, rows)`` grid at a capacity ``cols_cap_for`` accepts -- computed by the three kernels' own functions and compared."""
    from cudnn.gated_attention_block.kernels import qsa_prepare, qsa_topk_decode

    t = b * s_q
    tb = _table(b, s_q, nbm, units, threads)
    # R0 -- the prepare: its warp units with the indexer-query arm OFF and the raw-key / compress arm ON
    attn, qi, raw = qsa_prepare.prepare_warp_counts(
        t, h_q=H_Q, h_kv=H_KV, d=D, index_heads=INDEX_HEADS, index_head_dim=INDEX_HEAD_DIM, index_q=False, index_raw=True
    )
    assert qi == 0 and (attn, raw) == F.prepare_role_warps(t, h_q=H_Q, h_kv=H_KV, d=D, index_head_dim=INDEX_HEAD_DIM)
    assert tb.n_prep_ctas == -(-(attn + raw) // (threads // 32))
    # the standalone prepare's grid at its 128 threads carries the query arm too: the fused role drops exactly those warps
    attn_s, qi_s, raw_s = qsa_prepare.prepare_warp_counts(
        t, h_q=H_Q, h_kv=H_KV, d=D, index_heads=INDEX_HEADS, index_head_dim=INDEX_HEAD_DIM, index_q=True, index_raw=True
    )
    assert (attn_s, raw_s) == (attn, raw) and qi_s == -(-(t * INDEX_HEADS) // 2)
    standalone_prep_ctas = -(-(attn_s + qi_s + raw_s) // (qsa_prepare.DEFAULT_THREADS_PER_CTA // 32))
    assert standalone_prep_ctas >= 1
    # R1 -- the scorer: the standalone grid is (ceil(n_blocks_max / blocks_per_cta(1)), B); the fused CTA holds `units` of those chunks
    standalone_chunks = -(-nbm // SP.blocks_per_cta(1))
    assert tb.n_chunks == -(-standalone_chunks // units)
    assert tb.n_chunks * units >= standalone_chunks > (tb.n_chunks - 1) * units
    assert tb.n_scorer_ctas == b * tb.n_chunks
    # R2 -- the top-k: one CTA per row = the standalone (1, rows) grid; the capacity and the geometry are the kernel's own
    assert tb.n_topk_ctas == t
    cap = qsa_topk_decode.cols_cap_for(nbm, 1, threads)
    assert cap == F.topk_cols_cap(nbm, threads)
    qsa_topk_decode.validate_topk_geometry(cap, TOP_K, 1, threads)  # raises on a geometry the single-CTA form cannot address
    assert tb.n_ctas == tb.n_prep_ctas + b * (tb.n_chunks + s_q)


@needs_dsl
def test_mirrored_constants_equal_the_kernels_own():
    from cudnn.gated_attention_block.kernels import qk_norm_rope, qsa_topk_decode

    assert F.TOPK_MAX_EPT == qsa_topk_decode.MAX_EPT
    assert F.TOPK_MAX_ROWS == qsa_topk_decode.MAX_ROWS
    assert set(F.FUSED_THREAD_COUNTS) == set(qsa_topk_decode.THREAD_COUNTS) and F.DEFAULT_THREADS in F.FUSED_THREAD_COUNTS
    assert F.DEFAULT_THREADS == 512  # the register census's verdict: the 64-register scorer spills at 1024 (the module docstring of DEFAULT_THREADS)
    assert F.scorer_mma_batch_for(1024) == F.SCORER_MMA_BATCH_AT_64_REGS == 1 and F.scorer_mma_batch_for(512) == 0
    assert F._ELEMS_PER_ACCESS == qk_norm_rope.ELEMS_PER_ACCESS
    for d in (64, 128, 256, 512):
        assert F._lanes_per_row(d) == qk_norm_rope.lanes_per_row(d)
    assert F._SCORE_ROW_ALIGN == SP._SCORE_ROW_ALIGN
    # the fused CTA's named-barrier id space is spelled ONCE (the front door) and read by both roles; the ceiling holds at the widest served units
    from cudnn.frost.tile_dsl import gsync
    from cudnn.gated_attention_block.kernels import qsa_score_paged

    assert F.COUNTER_BAR_ID == gsync._BAR_ID_MIN == 1 and F.SCORER_BAR_ID_BASE == 2 and F._BAR_ID_MAX == gsync._BAR_ID_MAX == 15
    assert qsa_score_paged.COUNTER_BAR_ID == qsa_topk_decode.COUNTER_BAR_ID == F.COUNTER_BAR_ID and qsa_score_paged.SCORER_BAR_ID_BASE == F.SCORER_BAR_ID_BASE
    assert F.SCORER_BAR_ID_BASE + max(F.SCORER_UNIT_CHOICES) <= F._BAR_ID_MAX
    for n in (1, 8, 520, 1024, 2048, 8192, 32768, 65536):
        for thr in F.FUSED_THREAD_COUNTS:
            assert F.topk_cols_cap(n, thr) == qsa_topk_decode.cols_cap_for(n, 1, thr)
    # the SMEM table's top-k figure (113.4 KiB at 1024 threads) is the kernel's own
    assert qsa_topk_decode.smem_bytes_for(32768, 1, 1024, TOP_K) == 116_144
    assert qsa_topk_decode.smem_bytes_for(8192, 1, 1024, TOP_K) == 116_144
    doc_union = F.dynamic_smem_bytes(F.scorer_smem_bytes(2, 1, INDEX_HEADS), qsa_topk_decode.smem_bytes_for(32768, 1, 1024, TOP_K))
    assert doc_union == 116_144  # B <= 4 plans (1-2 units): the top-k's footprint is the union -> the standard carveout


def test_scorer_smem_follows_the_standalone_per_unit_and_the_union():
    one_ring = 4 * 4 * 2048  # 4 warps x 4 slots x 2 KiB at ring_depth 4 = the standalone scorer's sK
    tile = 16 * 256  # one m-tile of the padded query tile
    assert F.scorer_smem_bytes(1, 1, INDEX_HEADS) == one_ring + tile + 272
    assert F.scorer_smem_bytes(8, 1, INDEX_HEADS) == 8 * one_ring + tile + 272  # 260 KiB + the derive's row: the oversized carveout
    assert F.scorer_smem_bytes(8, 1, INDEX_HEADS) > 227 * 1024 > F.scorer_smem_bytes(8, 1, INDEX_HEADS, ring_depth=2) == 8 * (4 * 3 * 2048) + tile + 272
    assert F.scorer_smem_bytes(8, 4, INDEX_HEADS) == F.scorer_smem_bytes(8, 1, INDEX_HEADS)  # S_q 4 x 4 heads = 16 rows = still one m-tile
    assert F.dynamic_smem_bytes(F.scorer_smem_bytes(1, 1, INDEX_HEADS), 116_144) == 116_144
    assert F.dynamic_smem_bytes(F.scorer_smem_bytes(8, 1, INDEX_HEADS), 116_144) == F.scorer_smem_bytes(8, 1, INDEX_HEADS)


@pytest.mark.parametrize(
    "b,s_q,kv_tokens,units,threads,expect,waves",
    [
        # the DEFAULT 512-thread form (W = 16 warps per CTA; units <= 4): the served routes
        (1, 1, 8192, 1, 512, (2, 16, 1, 19), 1),
        (1, 1, 32768, 1, 512, (2, 64, 1, 67), 1),
        (1, 1, 131072, 2, 512, (2, 128, 1, 131), 1),
        (4, 1, 8192, 1, 512, (8, 16, 1, 76), 1),
        (4, 1, 32768, 2, 512, (8, 32, 1, 140), 1),
        (4, 1, 131072, 4, 512, (8, 64, 1, 268), 2),
        (1, 4, 8192, 1, 512, (8, 16, 4, 28), 1),
        (4, 4, 8192, 1, 512, (29, 16, 4, 109), 1),
        (32, 1, 8192, 4, 512, (57, 4, 1, 217), 2),
        (32, 1, 32768, 4, 512, (57, 16, 1, 601), 3),
        (32, 1, 131072, 4, 512, (57, 64, 1, 2137), 11),
        (32, 4, 8192, 4, 512, (228, 4, 4, 484), 3),
        (32, 4, 32768, 4, 512, (228, 16, 4, 868), 5),
        (32, 4, 131072, 4, 512, (228, 64, 4, 2404), 12),
        # the 1024-thread knob (W = 32: R0 halves; 8 units per R1 CTA at B 32)
        (1, 1, 8192, 1, 1024, (1, 16, 1, 18), 1),
        (1, 1, 32768, 1, 1024, (1, 64, 1, 66), 1),
        (1, 1, 131072, 2, 1024, (1, 128, 1, 130), 1),
        (4, 1, 8192, 1, 1024, (4, 16, 1, 72), 1),
        (4, 1, 32768, 2, 1024, (4, 32, 1, 136), 1),
        (4, 1, 131072, 4, 1024, (4, 64, 1, 264), 2),
        (1, 4, 8192, 1, 1024, (4, 16, 4, 24), 1),
        (4, 4, 8192, 1, 1024, (15, 16, 4, 95), 1),
        (32, 1, 8192, 8, 1024, (29, 2, 1, 125), 1),
        (32, 1, 32768, 8, 1024, (29, 8, 1, 317), 2),
        (32, 1, 131072, 8, 1024, (29, 32, 1, 1085), 6),
        (32, 4, 8192, 8, 1024, (114, 2, 4, 306), 2),
        (32, 4, 32768, 8, 1024, (114, 8, 4, 498), 3),
        (32, 4, 131072, 8, 1024, (114, 32, 4, 1266), 6),
    ],
    ids=lambda v: str(v) if not isinstance(v, tuple) else "x".join(str(x) for x in v),
)
def test_dispatch_rows_come_out_of_the_function(b, s_q, kv_tokens, units, threads, expect, waves):
    """The docstring's dispatch rows at the DEFAULT 512 threads (W = 16) and at the 1024-thread knob (W = 32): R0 CTAs, R1 per sequence,
    R2 per sequence, the total, the waves at 212 SMs -- the rows the C4 route author and the gate read."""
    assert F.DEFAULT_THREADS == 512, "the table's default rows are the 512-thread form"
    tb = _table(b, s_q, kv_tokens // SP.BLOCK_SIZE, units, threads)
    assert (tb.n_prep_ctas, tb.n_chunks, tb.s_q, tb.n_ctas) == expect, tb
    assert tb.waves(SM_COUNT) == waves


@pytest.mark.parametrize("order", F.DISPATCH_ORDERS)
@pytest.mark.parametrize("b,s_q,nbm,units,threads", [(1, 1, 2048, 1, 1024), (4, 4, 8192, 2, 1024), (32, 1, 32768, 4, 512), (3, 2, 520, 1, 512)])
def test_the_device_decode_is_the_inverse_of_the_host_encoding(b, s_q, nbm, units, threads, order):
    tb = _table(b, s_q, nbm, units, threads, order=order)
    seen = set()
    for bidx in range(tb.n_ctas):
        role, seq, k = tb.role_of(bidx)
        assert tb.linear_index(role, seq, k) == bidx
        seen.add((role, seq, k))
    assert len(seen) == tb.n_ctas
    assert sum(1 for r, _, _ in seen if r == F.ROLE_PREPARE) == tb.n_prep_ctas
    assert sum(1 for r, _, _ in seen if r == F.ROLE_SCORER) == tb.n_scorer_ctas
    assert sum(1 for r, _, _ in seen if r == F.ROLE_TOPK) == tb.n_topk_ctas == tb.n_rows
    # the ranges tile the grid exactly, in linear order, R0 first
    ranges = tb.ranges()
    assert ranges[0][0] == F.ROLE_PREPARE and ranges[0][2] == 0
    assert all(ranges[i][3] == ranges[i + 1][2] for i in range(len(ranges) - 1)) and ranges[-1][3] == tb.n_ctas
    with pytest.raises(ValueError, match="outside the grid"):
        tb.role_of(tb.n_ctas)
    with pytest.raises(ValueError, match="outside"):
        tb.linear_index(F.ROLE_SCORER, 0, tb.n_chunks)


def test_geometry_refusals_name_the_rule():
    ok = dict(h_q=H_Q, h_kv=H_KV, index_heads=INDEX_HEADS, n_blocks_max=2048, units=1, threads=1024)

    def call(t=1, b=1, s_q=1, **over):
        kw = dict(ok, **over)
        return F.dispatch_table(
            t,
            b,
            s_q,
            kw["h_q"],
            kw["h_kv"],
            kw["index_heads"],
            kw["n_blocks_max"],
            kw["units"],
            kw["threads"],
            **{k: v for k, v in over.items() if k in ("d", "index_head_dim", "units_per_warp", "order")},
        )

    assert call().n_ctas == 18
    with pytest.raises(ValueError, match="t must equal b x s_q"):
        call(t=3, b=1, s_q=1)
    with pytest.raises(ValueError, match="t must equal b x s_q"):
        call(t=0, b=0, s_q=1)
    with pytest.raises(ValueError, match="multiple of 8"):
        call(n_blocks_max=2052)
    with pytest.raises(ValueError, match="n_blocks_max must be a positive"):
        call(n_blocks_max=0)
    with pytest.raises(ValueError, match="units must be one of"):
        call(units=3)
    with pytest.raises(ValueError, match="threads must be one of"):
        call(threads=256)
    with pytest.raises(ValueError, match="scorer threads exceed the block size"):
        call(units=8, threads=512)
    with pytest.raises(ValueError, match="past the fused kernel's reach"):
        call(n_blocks_max=65536 + 8)  # cols_cap 131072 / 1024 = 128 keys per thread > 64: the chain with the cluster top-k serves it
    assert call(n_blocks_max=65536).n_chunks == 512  # exactly the reach at 1024 threads
    with pytest.raises(ValueError, match="past the fused kernel's reach"):
        call(n_blocks_max=65536, threads=512)
    with pytest.raises(ValueError, match="query envelope"):
        call(t=17, b=1, s_q=17)  # 17 x 4 = 68 rows > 64
    with pytest.raises(ValueError, match="index_head_dim = 128 only"):
        call(index_head_dim=64)
    with pytest.raises(ValueError, match="multiple of 8"):
        call(d=100)
    with pytest.raises(ValueError, match="units_per_warp must be one of"):
        call(units_per_warp=3)
    with pytest.raises(ValueError, match="order must be one of"):
        call(order="random")
    with pytest.raises(ValueError, match="at most 65535 rows"):
        call(t=65536, b=65536, s_q=1)
    with pytest.raises(ValueError, match="role must be one of"):
        call().linear_index(3, 0, 0)


def test_units_per_warp_folds_into_the_chunk_count():
    # the scorer's own knob: a warp streaming 2 units covers 256 rows per 4-warp unit, halving the chunks of a sequence
    assert F.scorer_chunks(8192, 1, 1) == 64 and F.scorer_chunks(8192, 1, 2) == 32 and F.scorer_chunks(8192, 4, 2) == 8
    assert _table(1, 1, 8192, 1, 1024, units_per_warp=2).n_chunks == 32
    assert F.scorer_smem_bytes(1, 1, INDEX_HEADS, units_per_warp=2) == 4 * 5 * 2048 + 16 * 256 + 272  # ring_slots = min(5, 8) = 5


def test_module_docstring_carries_the_pre_development_tables_and_the_contract():
    doc = " ".join(F.__doc__.split())  # whitespace-normalized: the tables wrap
    for heading in (
        "Dispatch table",
        "Counter table",
        "Barrier table of the fused CTA",
        "SMEM table",
        "Forward progress",
        "PDL sites per role",
        "Degenerate inputs",
        "Oracle statement",
    ):
        assert heading in doc, heading
    # the counter ledger and the barrier ledger, as the house rule spells them
    assert "1 lane x n_chunks" in doc and "SUM == target" in doc and "SUM(issuing lanes) == count" in doc
    assert "'bar.sync 2 + unit," in doc and "'bar.sync 2 + units, 128 x units'" in doc and "bar_id = 1" in doc
    assert "UNCONDITIONAL" in doc, "the release must be unconditional per R1 CTA (a dead sequence arrives too)"
    assert "counter_store_u64" in doc and "release_add" in doc and "acquire_geq" in doc
    # the release gate names ONE warp of the CTA (a per-unit warp index would arrive once per unit), the reset site and its detector
    assert "warp_gate = (unit == 0) & (warp_in_unit == 0)" in doc and "``warp_in_unit == 0`` alone holds on one warp of EVERY unit" in doc
    assert "if warp == 0: if nvvm.elect_sync(): counter_store_u64(addr, cutlass.Int64(0))" in doc and "check_arrivals" in doc
    assert "units 2 and 8" in doc
    # the idle warps of an R1 CTA take no role; the derive's row store is lane-matched; one barrier at the publish site under derive
    assert "take NO role" in doc and "count 128 x units EXACTLY" in doc
    assert "LANE-MATCHED" in doc and "no second warp sync" in doc
    assert "rendered AT THE SAME SITE in place of the" in doc and "'bar.sync 0' at this site (fused=False)" in doc
    # the PTX census of the counter sites and of the PDL pairs
    assert "red.relaxed.gpu.global.add.u64 == S_q per R1 rendering" in doc and "no red.release anywhere" in doc and "``griddepcontrol.wait`` == 3" in doc
    assert (
        "fence.acq_rel.gpu == 1" in doc and "ONE elected lane: fence.acq_rel.gpu once" in doc
    ), "the release form: the barrier, then the elected lane's fence + relaxed reds"
    assert "``DEFAULT_THREADS`` = 512" in doc and "the 1024-thread knob" in doc, "the dispatch header names the default block size and the knob"
    assert "after ``prepare_role`` returns" in doc and "every node of the programmatic chain" in doc
    # the SMEM union and the two derive sites
    assert "sNew" in doc and "sA" in doc and "260.3 KiB" in doc and "113.4 KiB" in doc and "196 KiB" in doc and "OVERSIZED" in doc
    assert "73.4 KiB" in doc and "132.3 KiB" in doc and "512 x 128 = the file" in doc, "the union row names the default form's figures"
    # the assumption, its detectors and its denial
    assert "non-decreasing linear ``blockIdx`` order" in doc and "NOT: a CUDA programming-model guarantee" in doc
    assert "deadline_ns" in doc and "consumers-first" in doc and "consumers-first-global" in doc
    # P17: the wait is the first instruction of every role; the dependent's obligation is stated
    assert "FIRST instruction of every role" in doc and "MUST issue its own ``griddepcontrol.wait``" in doc
    # the degenerate matrix names every row the plan lists
    for row in ("kv_lens[b] == 0", "kv_lens[b] < S_q", "a -1 (or out-of-range) page", "the -2 reports", "the identity region", "the ring wrap (cap = S_q + 3"):
        assert row in doc, row
    # the oracle and the tie clause; no perf claim
    assert "BITWISE the three-launch form" in doc and "fp64 top-k" in doc and "ties may differ" in doc and "never widened" in doc
    assert "No performance number is claimed" in doc


def test_served_entries_gate_before_the_kernel_module(monkeypatch):
    """The two served entries keep Rule 7's order: the DSL floor is read FIRST (a modeled sub-floor DSL is a typed decline naming it), the
    form refusals of the fused declaration (the query arm, a step wider than the compress ratio, a compress ratio other than the scorer's
    block size, ``units_per_warp`` > 1 at the 1024-thread knob, the mis-ordered table without the trap's
    deadline) are typed and raised before the kernel module is imported; the compile keys stay in the docstrings (``cols_cap, top_k``, the
    test-only ``deadline_ns`` / ``check_arrivals``), the counter contract in the run entry's; the counter-word checks are DSL-free."""
    import sys
    from typing import NamedTuple

    class _Arms(NamedTuple):  # the prepare's QsaPrepareIndexArms record, field for field (that class lives in the kernel module: not imported here)
        heads: int
        head_dim: int
        raw_pool: bool = True
        query: bool = False
        compress: bool = False
        n_commit: bool = False
        pool: int = 4
        paged: bool = False
        ring_slots: bool = False
        rope_table: bool = False

    kernel_module = "cudnn.gated_attention_block.kernels.qsa_indexer_fused_kernel"
    before = kernel_module in sys.modules
    arms = _Arms(heads=INDEX_HEADS, head_dim=INDEX_HEAD_DIM, raw_pool=False, query=False, compress=True, pool=4, paged=True)
    base = dict(dtype=torch.bfloat16, h_q=H_Q, h_kv=H_KV, d=D, rope_dim=64, eps=1e-6, page_size=16, index=arms, s_q=1, n_blocks_max=2048, top_k=TOP_K, units=1)
    monkeypatch.setattr(F, "cutedsl_requirement_error", lambda what: f"{what} requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2")
    with pytest.raises(NotImplementedError, match="qsa_indexer_fused requires nvidia-cutlass-dsl >= 4.7.0; found 4.6.2"):
        F.compile_qsa_indexer_fused(**base)
    monkeypatch.setattr(F, "cutedsl_requirement_error", lambda what: None)
    with pytest.raises(ValueError, match="index.query must be False"):
        F.compile_qsa_indexer_fused(**dict(base, index=arms._replace(query=True)))
    with pytest.raises(ValueError, match="compress=True and paged=True"):
        F.compile_qsa_indexer_fused(**dict(base, index=arms._replace(paged=False)))
    with pytest.raises(NotImplementedError, match="exceed the compress ratio pool"):
        F.compile_qsa_indexer_fused(**dict(base, s_q=5))
    # the same reach in the DSL-free dispatch table (what the block's route reads at declaration): a step wider than the block size is not
    # routed, a step of block_size rows is
    with pytest.raises(ValueError, match="rows per sequence exceed the fused step's block size 4"):
        F.dispatch_table(5, 1, 5, H_Q, H_KV, INDEX_HEADS, 2048, 1, 512)
    assert F.dispatch_table(4, 1, 4, H_Q, H_KV, INDEX_HEADS, 2048, 1, 512).n_topk_ctas == 4
    # the pool is pinned to the scorer's block size: pool 8 at S_q 1 passes the ratio gate alone (the review's case), pool 2 fails both
    with pytest.raises(NotImplementedError, match="index.pool must be 4, got 8"):
        F.compile_qsa_indexer_fused(**dict(base, index=arms._replace(pool=8), s_q=1))
    with pytest.raises(NotImplementedError, match="index.pool must be 4, got 2"):
        F.compile_qsa_indexer_fused(**dict(base, index=arms._replace(pool=2), s_q=1))
    # the 1024-thread knob serves units_per_warp 1 alone (the register census: the scorer spills at 64 registers in every measured batch)
    with pytest.raises(NotImplementedError, match="units_per_warp > 1 is served at 512 threads"):
        F.compile_qsa_indexer_fused(**dict(base, threads=1024, units_per_warp=2))
    assert F.scorer_mma_batch_for(1024, 1) == F.scorer_mma_batch_for(1024, 2) == F.SCORER_MMA_BATCH_AT_64_REGS and F.scorer_mma_batch_for(512, 2) == 0
    with pytest.raises(NotImplementedError, match="bf16 activations"):
        F.compile_qsa_indexer_fused(**dict(base, dtype=torch.float16))
    with pytest.raises(ValueError, match="served only with deadline_ns"):
        F.compile_qsa_indexer_fused(**dict(base, order="consumers-first"))
    with pytest.raises(ValueError, match="served only with deadline_ns"):
        F.compile_qsa_indexer_fused(**dict(base, order="consumers-first-global"))
    with pytest.raises(ValueError, match="order must be one of"):
        F.compile_qsa_indexer_fused(**dict(base, order="random", deadline_ns=1))
    with pytest.raises(ValueError, match="roles must be a non-empty subset"):
        F.compile_qsa_indexer_fused(**dict(base, roles=("tile",)))
    with pytest.raises(ValueError, match="units must be one of"):
        F.compile_qsa_indexer_fused(**dict(base, units=3))
    assert (kernel_module in sys.modules) == before, "a typed decline imported the kernel module"
    assert "cols_cap, top_k" in F.compile_qsa_indexer_fused.__doc__ and "deadline_ns" in F.compile_qsa_indexer_fused.__doc__
    assert "check_arrivals" in F.compile_qsa_indexer_fused.__doc__
    assert "cnt_row" in F.run_qsa_indexer_fused.__doc__
    # the plan-owned counter words: int64, 1-D, contiguous, CUDA, >= T words -- each a typed ValueError (no device needed for the refusals)
    with pytest.raises(ValueError, match="int64"):
        F._check_counter_words(torch.zeros(4, dtype=torch.int32), 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="contiguous 1-D CUDA tensor"):
        F._check_counter_words(torch.zeros(4, dtype=torch.int64), 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="contiguous 1-D CUDA tensor"):
        F._check_counter_words(torch.zeros(2, 2, dtype=torch.int64), 4, torch.device("cpu"))


# ===========================================================================================================================================
# The host pins of the dispatch table's dead units and of the forward-progress wording (the re-audit's notes)
# ===========================================================================================================================================


@pytest.mark.parametrize("nbm,units,dead", [(520, 2, 1), (520, 8, 3), (1032, 2, 1), (1032, 8, 7), (8192, 8, 0), (2048, 2, 0)])
def test_the_dead_units_of_the_last_scorer_cta_are_in_the_table(nbm, units, dead):
    """``n_chunks = ceil(ceil(n_blocks_max / 128) / units)``: when the standalone chunk count is not a multiple of ``units`` the last R1
    CTA holds ``dead`` units whose ``chunk_id >= ceil(n_blocks_max / 128)`` -- they take no scorer work, still join the CTA-wide barriers
    their ledger counts, and the release stays one warp per CTA.  The dispatch table names that row."""
    tb = _table(4, 1, nbm, units, 1024)
    standalone = -(-nbm // SP.blocks_per_cta(1))
    assert tb.n_chunks * units - standalone == dead
    for k in range(tb.n_chunks):
        for u in range(units):
            chunk = k * units + u
            assert (chunk >= standalone) == (k == tb.n_chunks - 1 and u >= units - dead), (k, u)
    doc = " ".join(F.__doc__.split())
    assert (
        "dead unit" in doc.lower() and "chunk_id >=" in doc
    ), "the dispatch table names no dead-unit row (a unit whose chunk_id >= the standalone chunk count)"


def test_the_placement_docstring_names_this_grid_only():
    """Under ``use_pdl`` every role waits on the PREVIOUS grid, so the host pin's docstring says 'nothing of this grid', never 'nothing'."""
    doc = " ".join(F.DispatchTable.producers_before_consumers.__doc__.split())
    assert "nothing of this grid" in doc, doc


# ===========================================================================================================================================
# The kernel-level tiers: the harness, the gates, the cells
# ===========================================================================================================================================


def _H():
    """The step harness (it imports the three kernel modules, which import the DSL): loaded on demand below the Rule 7 gate."""
    if _CUTEDSL_REQUIREMENT is not None:
        pytest.skip(str(_CUTEDSL_REQUIREMENT))
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    return importlib.import_module("qsa_indexer_fused_step")


def _case(b, s_q, nbm, page=16, *, kv=None, seed=1, geom=None, **kw):
    H = _H()
    return H.StepCase(
        H.FLASH_NEXT if geom is None else geom, batch=b, s_q=s_q, n_blocks_max=nbm, page=page, kv_lens=kv or H.kv_pattern(b, s_q, nbm), seed=seed, **kw
    )


def _fused(case, **kw):
    """``FusedPlan(case, **kw)``: a Rule 7 decline of the fused compile skips by name; the pre-body stub's ``NotImplementedError`` and a missing
    compile key (``TypeError``) are this module's RED and propagate."""
    H = _H()
    try:
        return H.FusedPlan(case, **kw)
    except NotImplementedError as e:
        why = H.decline_reason(e)
        if why is not None:
            pytest.skip(f"the fused indexer declines here by name: {why}")
        raise


def _skip_unless_reference():
    why = _H().reference_decline_reason()
    if why is not None:
        pytest.skip(why)


def _step(case, *, units, check_arrivals=None, **kw):
    """The fused plan FIRST (RED on the stub), then the three-launch reference and the fused launch on fresh copies of the same primed
    buffers.  ``check_arrivals`` is armed at every ``units > 1`` (the over-arrival of a per-unit release gate shows only there)."""
    H = _H()
    plan = _fused(case, units=units, check_arrivals=(units > 1) if check_arrivals is None else check_arrivals, **kw)
    _skip_unless_reference()
    _, ref_x = H.run_reference(case, threads=plan.keys["threads"], has_block_lens=plan.keys["has_block_lens"], use_pdl=plan.keys["use_pdl"])
    fused_x, cnt = H.run_fused_once(case, plan)
    return plan, ref_x, fused_x, cnt


def _child(mode, args, *, timeout):
    """One child interpreter of the harness (``mode`` ``launches`` / ``trap``); its ONE ``RESULT`` line as a dict.  No RESULT = the child
    died (the pre-body stub's traceback among the causes): a failure naming the output; a child past its budget is a HANG, named, never a
    bigger timeout."""
    env = dict(os.environ)
    env.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(_HARNESS), mode, json.dumps(args)], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace")[-2000:] if isinstance(e.stdout, bytes) else str(e.stdout or "")[-2000:]
        pytest.fail(f"HANG: the child did not return within {timeout} s (re-audit the counter / barrier tables; the timeout stays): {mode} {args}\n{out}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        pytest.fail(
            f"no RESULT line from the child (rc {proc.returncode}, {time.time() - t0:.1f} s): {mode} {args}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    res = json.loads(lines[-1][len("RESULT ") :])
    if "declined" in res:
        pytest.skip(f"the fused indexer declines here by name: {res['declined']}")
    print(f"\n{mode} {args}: {res}")
    return proc, res


# ---------------------------------------------------------------------------------------------------------------------------------------
# the harness's own contract (no fused kernel involved): the three-launch reference on real hardware
# ---------------------------------------------------------------------------------------------------------------------------------------


def _assert_canonical_fp64_set(logits, counts, ids, top_k):
    """The decode top-k's contract on the reference's own outputs: per row ``min(top_k, counts)`` ids strictly DESCENDING then -1, every id
    below the count, the selected VALUES exactly the fp64 top-k's values, every id differing from ``torch.topk`` an exact tie."""
    for t in range(logits.shape[0]):
        n = int(min(max(int(counts[t]), 0), logits.shape[1]))
        k_eff = min(top_k, n)
        row = ids[t]
        assert bool((row[k_eff:] == -1).all()), f"row {t}: the padding is not -1"
        sel = row[:k_eff].long()
        if k_eff == 0:
            continue
        assert bool((sel[1:] < sel[:-1]).all()) and bool((sel < n).all()) and bool((sel >= 0).all()), f"row {t}: not a descending id list below the count"
        ref = torch.topk(logits[t, :n].double(), k_eff)
        assert torch.equal(torch.sort(logits[t, sel].double(), descending=True).values, ref.values), f"row {t}: the selected values are not the fp64 top-k's"
        kth = float(ref.values[-1])
        for i in set(sel.tolist()) ^ set(ref.indices.tolist()):
            assert float(logits[t, i]) == kth, f"row {t}: id {i} differs from torch.topk beyond an exact tie"


@needs_cuda
@pytest.mark.parametrize(
    "b,s_q,nbm,page,kw", [(2, 4, 2048, 16, {}), (3, 1, 520, 32, {}), (2, 4, 2048, 16, dict(rope_table=True, ring_slots=True, slot_dtype=torch.int64))]
)
def test_the_three_launch_reference_is_self_consistent(b, s_q, nbm, page, kw):
    """The oracle's own contract, independent of the fused kernel: the three-launch form run twice on fresh copies is bitwise; ``counts``
    follow the count rule ``clamp(floor((pos + 1) / 4), 0, n_blocks_max)``; ``block_lens = min(top_k, counts)``; the list is canonical and its
    set is the fp64 top-k of the written logits up to exact ties; the slot report names the block every completing row completes and that
    block's cache row was written (it differs from the prime); the columns at or past the count keep the sentinel."""
    H = _H()
    _skip_unless_reference()
    case = _case(b, s_q, nbm, page, seed=7, **kw)
    _, x1 = H.run_reference(case)
    _, x2 = H.run_reference(case)
    a, c = case.outputs(x1, fused=False), case.outputs(x2, fused=False)
    assert all(H.same(a[k], c[k]) for k in a), [k for k in a if not H.same(a[k], c[k])]
    n_vis = case.n_vis().reshape(-1)
    assert torch.equal(x1["counts"].long(), n_vis), "counts differ from the count rule"
    assert torch.equal(x1["lens"].long(), n_vis.clamp(max=case.geom.top_k)), "block_lens differs from min(top_k, counts)"
    for t in range(case.T):
        n = int(n_vis[t])
        assert bool((x1["logits"][t, :n] != H.LOGIT_SENTINEL).all()) and bool(
            (x1["logits"][t, n:] == H.LOGIT_SENTINEL).all()
        ), f"row {t}: the written columns are not exactly [0, {n})"
    _assert_canonical_fp64_set(x1["logits"], x1["counts"], x1["ids"], case.geom.top_k)
    for seq, hit in enumerate(case.completing()):
        if hit is None:
            assert bool((x1["cslot"][seq] == -1).all()), f"sequence {seq}: a slot report without a completing row"
        else:
            j, blk = hit
            assert int(x1["cslot"][seq, j]) == blk and bool((x1["cslot"][seq, :j] == -1).all()) and bool((x1["cslot"][seq, j + 1 :] == -1).all()), x1["cslot"][
                seq
            ]
            assert not H.same(case.kbar_row(x1, seq, blk), case.kbar_row({"comp": case.comp0}, seq, blk)), f"sequence {seq}: block {blk} was not written"
    assert bool(torch.isnan(x1["qi"].float()).logical_not().all()), "the reference's indexer queries were not all written"


# ---------------------------------------------------------------------------------------------------------------------------------------
# T2 -- the fused launch is bitwise the three-launch form
# ---------------------------------------------------------------------------------------------------------------------------------------


def _t2_cells():
    """``(b, s_q, n_blocks_max, page, units, extra)``: the dispatch rows' cells at the route's units on the DEFAULT block size (512 by the
    register pin: units 1 at B <= 4 for 8K / 32K tokens, 2 at 128K, 4 at B 32 -- 8 units fit the 1024-thread knob alone), the knob at
    units 1 / 2 / 8, the units 2 / 4 / 8 variants with ``check_arrivals`` (three of them with a chunk count that is not a multiple of
    ``units``: dead units in the last scorer CTA), the optional ``block_lens`` arm, the plan's other arms (the RoPE-table arm, the stack's
    ring slots with int64 slots), the PDL pair.  The small grid first; the B 32 and the 128K cells carry a ``big_`` id."""
    small, big = [], []
    for b in (1, 4):
        for s_q in (1, 4):
            for n in (2048, 8192):
                for page in (16, 32, 64):
                    small.append((b, s_q, n, page, 1, {}))
    small += [
        (1, 1, 2048, 16, 2, {}),
        (4, 1, 8192, 16, 4, {}),
        (1, 1, 2048, 16, 8, dict(threads=1024)),
        (4, 4, 8192, 16, 8, dict(threads=1024)),
        (4, 1, 1032, 16, 2, {}),
        (4, 1, 1032, 16, 8, dict(threads=1024)),
        (1, 4, 520, 32, 8, dict(threads=1024)),
        (1, 1, 2048, 16, 1, dict(has_block_lens=False)),
        (4, 4, 2048, 16, 1, dict(rope_table=True)),
        (4, 1, 8192, 16, 1, dict(ring_slots=True, slot_dtype=torch.int64)),
        (4, 4, 8192, 16, 2, dict(use_pdl=True)),
        (1, 1, 2048, 16, 1, dict(threads=1024)),
        (4, 1, 8192, 64, 2, dict(threads=1024)),
        (4, 4, 8192, 16, 1, dict(threads=1024, rope_table=True)),
    ]
    for b in (1, 4):
        for s_q in (1, 4):
            for page in (16, 32, 64):
                big.append((b, s_q, 32768, page, 2, {}))
    for s_q in (1, 4):
        for n in (2048, 8192, 32768):
            for page in (16, 32, 64):
                big.append((32, s_q, n, page, 4, {}))
    for s_q in (1, 4):
        for page in (16, 64):
            big.append((32, s_q, 8192, page, 8, dict(threads=1024)))
    return small, big


def _t2_id(cell, big=False):
    b, s_q, n, page, units, extra = cell
    tag = "".join(f"_{k}{int(v) if isinstance(v, bool) else str(v).replace('torch.', '')}" for k, v in extra.items())
    return f"{'big_' if big else ''}B{b}_S{s_q}_N{n}_p{page}_u{units}{tag}"


_T2_SMALL, _T2_BIG = _t2_cells()
_T2 = _T2_SMALL + _T2_BIG
_T2_IDS = [_t2_id(c) for c in _T2_SMALL] + [_t2_id(c, big=True) for c in _T2_BIG]
_CASE_KW = ("rope_table", "ring_slots", "slot_dtype")  # the extras that shape the step; the rest are plan knobs


@needs_cuda
@pytest.mark.parametrize("cell", _T2, ids=_T2_IDS)
def test_fused_indexer_is_bitwise_the_three_launch_form(cell):
    """The oracle: ``out`` (the list), ``counts``, ``block_lens``, the slot report, the Q / K bands, the K / V pool rows, the ring rows, the
    compressed-key cache rows and the logits buffer all ``torch.equal`` the three-launch form's on the same primed caches; the counters read
    zero after the launch; the plan's grid is the dispatch table's.  ``check_arrivals`` is armed at units > 1 (an over-arriving release gate
    traps there instead of hiding behind the reset)."""
    H = _H()
    b, s_q, n, page, units, extra = cell
    case = _case(b, s_q, n, page, seed=1000 + _T2.index(cell), **{k: v for k, v in extra.items() if k in _CASE_KW})
    plan, ref_x, fused_x, cnt = _step(case, units=units, **{k: v for k, v in extra.items() if k not in _CASE_KW})
    label = _t2_id(cell)
    H.assert_bitwise(case, ref_x, fused_x, label)
    H.assert_counters_zero(cnt, label)
    standalone = -(-n // SP.blocks_per_cta(1))
    assert plan.table.n_chunks == -(-standalone // units) and plan.table.n_ctas == plan.table.n_prep_ctas + b * (plan.table.n_chunks + s_q)
    assert bool((fused_x["ids"] != H.IDS_SENTINEL).all()), "a list slot was never written"


# ---------------------------------------------------------------------------------------------------------------------------------------
# T3 -- the completing row is derived bitwise at every alignment
# ---------------------------------------------------------------------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("s_q,r", [(4, 0), (4, 1), (4, 2), (4, 3), (1, 0), (1, 1), (1, 2), (1, 3)], ids=lambda v: f"{v}")
def test_the_completing_row_is_derived_bitwise_at_every_alignment(s_q, r):
    """S_q 4 with ``pos0 % 4 == r`` (exactly one step row completes a block: row ``3 - r``) and S_q 1 with ``(pos0 + 1) % 4 == r`` (a
    completing block iff ``r == 0``): the derived ``blk_new`` row -- substituted into the scorer role's ring slot while the prepare role
    writes the cache row -- shows as the logits column ``blk_new`` of every row that sees it, bitwise the three-launch form's (which scored
    the row the prepare had written), and the cache row written by the fused launch is bitwise the prepare's; the rows before the
    completing one do not see the column; a step without a completing row writes no cache row of its sequence."""
    H = _H()
    nbm = 2048
    base = [4 * 600, 4 * 637, 4 * (nbm - 2)]
    pos0 = [k + (r if s_q == 4 else (r - 1) % 4) for k in base]
    kv = [p + s_q for p in pos0]
    case = _case(3, s_q, nbm, 16, kv=kv, seed=200 + 10 * s_q + r)
    _, ref_x, fused_x, cnt = _step(case, units=1)
    label = f"alignment S_q {s_q} r {r}"
    H.assert_bitwise(case, ref_x, fused_x, label)
    H.assert_counters_zero(cnt, label)
    for seq, hit in enumerate(case.completing()):
        if s_q == 4:
            assert hit is not None and hit[0] == 3 - r, (seq, hit)
        else:
            assert (hit is not None) == (r == 0), (seq, hit, r)
        rows = range(seq * s_q, (seq + 1) * s_q)
        if hit is None:
            assert bool((fused_x["cslot"][seq] == -1).all())
            pages = case.block_table[seq].long()
            assert H.same(fused_x["comp"][pages], case.comp0[pages]), f"sequence {seq}: a cache row written without a completing row"
            continue
        j, blk = hit
        assert int(fused_x["cslot"][seq, j]) == blk and int(ref_x["cslot"][seq, j]) == blk
        assert H.same(case.kbar_row(fused_x, seq, blk), case.kbar_row(ref_x, seq, blk)), f"sequence {seq}: the cache row of block {blk} differs"
        assert not H.same(case.kbar_row(fused_x, seq, blk), case.kbar_row({"comp": case.comp0}, seq, blk)), f"sequence {seq}: block {blk} was not written"
        for s, t in enumerate(rows):
            # the sentinel is compared in fp32 (a tensor comparison): the Python double 1.5e30 is not an fp32 value, float(tensor) would never equal it
            col_f, col_r = fused_x["logits"][t, blk], ref_x["logits"][t, blk]
            if s >= j:
                assert bool(col_f != H.LOGIT_SENTINEL) and H.same(col_f, col_r), f"row {t}: the derived column {blk} differs or was not scored"
            else:
                assert bool(col_f == H.LOGIT_SENTINEL) and bool(
                    col_r == H.LOGIT_SENTINEL
                ), f"row {t}: column {blk} is not visible to a row before the completing one"


# ---------------------------------------------------------------------------------------------------------------------------------------
# T4 -- the degenerate matrix
# ---------------------------------------------------------------------------------------------------------------------------------------


def _ids_identity(n, top_k, device):
    out = torch.full((top_k,), -1, dtype=torch.int32, device=device)
    out[:n] = torch.arange(n - 1, -1, -1, device=device, dtype=torch.int32)
    return out


_T4_ROWS = [
    "kv_lens_zero_padding_sequence",
    "kv_lens_below_s_q",
    "dead_page_under_a_visible_block",
    "dead_page_at_the_completing_block",
    "completing_block_past_the_table_width",
    "ring_at_its_minimum_cap_wraps_S4",
    "ring_at_its_minimum_cap_wraps_S1",
    "identity_region",
    "dead_units_u2_1032",
    "dead_units_u8_1032",
    "dead_units_u8_520",
]


@needs_cuda
@pytest.mark.parametrize("row", _T4_ROWS)
def test_degenerate_inputs_follow_the_matrix(row):
    """The module docstring's matrix, every row bitwise the three-launch form with the counters zero after the launch (every R1 CTA
    arrives unconditionally: a dead sequence too), plus the row's own observable: every slot -1 / the identity list / no ring, cache or pool
    row for a padding sequence; a zero logit under a dead page; the ``-2`` report with no cache row; the ring wrap at ``cap = S_q + 3``; the
    dead units of the last scorer CTA."""
    H = _H()
    units, kw = 1, {}
    if row == "kv_lens_zero_padding_sequence":
        case = _case(3, 4, 2048, 16, kv=[4 * 600 + 2, 0, 4 * 33 + 1], padding=(1,), seed=301)
    elif row == "kv_lens_below_s_q":
        case = _case(3, 4, 2048, 16, kv=[2, 4 * 600 + 1, 3], seed=302)
    elif row == "dead_page_under_a_visible_block":
        case = _case(2, 1, 2048, 16, kv=[4 * 2048, 4 * 700], seed=303)
        case.block_table[0, 10] = -1  # blocks 40..43 of sequence 0: a -1 page under visible blocks
        case.block_table[1, 3] = case.n_kbar_pages + 5  # blocks 12..15 of sequence 1: an out-of-range page under visible blocks
    elif row == "dead_page_at_the_completing_block":
        case = _case(2, 4, 2048, 16, kv=[4 * 600, 4 * 601 + 2], seed=304)  # sequence 0: row 3 completes block 599 (page 149, row 3)
        case.block_table[0, 599 // case.rpp] = -1
    elif row == "completing_block_past_the_table_width":
        case = _case(2, 4, 40, 16, kv=[164, 4 * 20 + 1], seed=305, max_pages=10)  # block 40 = page 10, past the 10-column table: -2
    elif row == "ring_at_its_minimum_cap_wraps_S4":
        # pos0 = 1191 / 295 (both 3 mod 4 and 1 mod 7): row 0 completes a block and pools ring positions pos0 - 3 .. pos0 - 1 = ring rows 5, 6, 0
        # (the wrap), while the step writes rows 1 .. 4 -- disjoint, the host rule cap >= S_q + 3 at equality
        case = _case(2, 4, 2048, 16, kv=[1191 + 4, 295 + 4], seed=306, ring_cap=7)
    elif row == "ring_at_its_minimum_cap_wraps_S1":
        case = _case(
            2, 1, 2048, 16, kv=[4 * 300, 4 * 71], seed=307, ring_cap=4
        )  # pos0 = 1199 / 283 (3 mod 4): the completing row pools ring rows 0, 1, 2 at the minimum cap
    elif row == "identity_region":
        case = _case(4, 1, 520, 16, kv=[4 * 500, 4 * 513, 4 * 520, 4 * 100], seed=308)  # counts 500 / 513 (one past the identity) / 520 / 100
    elif row.startswith("dead_units"):
        units, nbm = (2, 1032) if row == "dead_units_u2_1032" else (8, 1032) if row == "dead_units_u8_1032" else (8, 520)
        kw = dict(threads=1024) if units == 8 else {}  # 8 units fit the 1024-thread knob alone
        case = _case(2, 1, nbm, 16, kv=[4 * nbm, 4 * (nbm - 130) + 3], seed=309)
    else:  # pragma: no cover
        raise AssertionError(row)
    plan, ref_x, fused_x, cnt = _step(case, units=units, **kw)
    H.assert_bitwise(case, ref_x, fused_x, row)
    H.assert_counters_zero(cnt, row)
    ids, counts, lens, cslot = fused_x["ids"], fused_x["counts"], fused_x["lens"], fused_x["cslot"]
    if row == "kv_lens_zero_padding_sequence":
        rows = slice(4, 8)
        assert bool((ids[rows] == -1).all()) and counts[rows].tolist() == [0] * 4 and lens[rows].tolist() == [0] * 4 and bool((cslot[1] == -1).all())
        assert H.same(fused_x["ring"][0], case.ring0[0]) and H.same(fused_x["comp"][0], case.comp0[0]), "the null ring block / page 0 were written"
        written = int((~torch.isnan(fused_x["k_pool"].float())).all(-1).sum())
        assert written == int(case.live.sum()) * case.geom.h_kv, f"pool rows written {written} != live slots x heads (a padding sequence's token owns no slot)"
    elif row == "kv_lens_below_s_q":
        for seq in (0, 2):  # rows at positions -2 / -1 / 0 / 1 (resp. -1 / 0 / 1 / 2): no visible block, no completing row
            rows = slice(4 * seq, 4 * seq + 4)
            assert bool((ids[rows] == -1).all()) and counts[rows].tolist() == [0] * 4 and bool((cslot[seq] == -1).all()), (seq, counts[rows].tolist())
    elif row == "dead_page_under_a_visible_block":
        assert bool((fused_x["logits"][0, 40:44] == 0.0).all()) and bool(
            (fused_x["logits"][1, 12:16] == 0.0).all()
        ), "a dead page under visible blocks did not read as zero keys"
        assert bool((fused_x["logits"][0, :40] != H.LOGIT_SENTINEL).all()) and int(counts[0]) == 2048 and int(counts[1]) == 700
    elif row == "dead_page_at_the_completing_block":
        assert case.completing()[0] == (3, 599) and int(cslot[0, 3]) == -2, cslot.tolist()
        assert float(fused_x["logits"][3, 599]) == 0.0 == float(ref_x["logits"][3, 599]), "the completing block on a dead page must score as a zero key"
        # sequence 0's live pages and the null page 0 stay the prime's (no row for the -2 report, nothing redirected to page 0); sequence 1
        # completes block 600 on a live page and writes it in both forms (the bitwise check above already holds the two caches equal)
        pages0 = case.block_table[0][case.block_table[0] >= 0].long()
        assert H.same(fused_x["comp"][pages0], case.comp0[pages0]) and H.same(fused_x["comp"][0], case.comp0[0]), "a cache row was written under a dead page"
        j1, blk1 = case.completing()[1]
        assert (
            (j1, blk1) == (1, 600) and int(cslot[1, j1]) == blk1 and not H.same(case.kbar_row(fused_x, 1, blk1), case.kbar_row({"comp": case.comp0}, 1, blk1))
        )
    elif row == "completing_block_past_the_table_width":
        assert case.completing()[0] == (3, 40) and int(cslot[0, 3]) == -2 and counts[:4].tolist() == [40] * 4, (cslot.tolist(), counts.tolist())
        pages0 = case.block_table[0].long()
        assert H.same(fused_x["comp"][pages0], case.comp0[pages0]), "a cache row was written for a block past the table's width"
        j1, blk1 = case.completing()[1]  # sequence 1 completes block 19 inside the table: written in both forms
        assert int(cslot[1, j1]) == blk1 and not H.same(case.kbar_row(fused_x, 1, blk1), case.kbar_row({"comp": case.comp0}, 1, blk1))
    elif row.startswith("ring_at_its_minimum_cap"):
        assert case.ring_n == case.S_q + 3 and all(hit is not None for hit in case.completing()), case.completing()
        for seq, (j, blk) in enumerate(case.completing()):
            assert int(cslot[seq, j]) == blk
    elif row == "identity_region":
        for t, n in enumerate(counts.tolist()):
            if n <= case.geom.top_k:
                assert torch.equal(ids[t], _ids_identity(n, case.geom.top_k, ids.device)), f"row {t}: not the identity list at {n} visible blocks"
            else:
                assert bool((ids[t] >= 0).all())
        assert counts.tolist() == [500, 513, 520, 100]
    else:
        standalone = -(-case.n_blocks_max // SP.blocks_per_cta(1))
        assert standalone % units and plan.table.n_chunks == -(-standalone // units), (standalone, units, plan.table)


# ---------------------------------------------------------------------------------------------------------------------------------------
# T5 -- the counter lifecycle
# ---------------------------------------------------------------------------------------------------------------------------------------

_PROCESSES = int(os.environ.get("QSA_FUSED_PROCESSES", "12"))
_LAUNCHES = int(os.environ.get("QSA_FUSED_LAUNCHES", "100"))


@needs_cuda
def test_counters_are_zero_after_every_launch_in_fresh_processes():
    """The fresh-process screen: 12 interpreters x 100 launches of one plan (half at units 2 on the default block size, half at units 8 on
    the 1024-thread knob, ``check_arrivals`` armed in every one): the counters read zero after every launch and every launch is bitwise the
    first and the three-launch reference."""
    case = _case(4, 1, 2048, 16, seed=5)
    _fused(case, units=2, check_arrivals=True)  # the decline, named here before any child is spawned
    _skip_unless_reference()
    for i in range(_PROCESSES):
        units, threads = (2, F.DEFAULT_THREADS) if i % 2 == 0 else (8, 1024)
        proc, res = _child(
            "launches",
            dict(batch=4, s_q=1, n_blocks_max=2048, page=16, units=units, threads=threads, check_arrivals=True, launches=_LAUNCHES, seed=100 + i),
            timeout=1500,
        )
        assert proc.returncode == 0 and res.get("ok"), f"process {i} (units {units}): {res}"


@needs_cuda
def test_a_captured_first_execute_replays_bitwise_with_zero_counters_after_every_replay():
    """The serving stacks' default: the FIRST execute of a fresh plan captured on a NEW stream with no warm-up under
    ``set_sync_debug_mode("error")`` (no host sync, no allocation in the path); the capture launches nothing; three replays are each
    bitwise the eager three-launch reference and leave the counters at zero (the in-kernel reset is captured with the kernel: no memset
    node)."""
    H = _H()
    case = _case(2, 4, 2048, 16, seed=11)
    plan = _fused(case, units=1)
    _skip_unless_reference()
    _, ref_x = H.run_reference(case)
    cnt = H.new_counters(case)
    X = case.fresh()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan.run(X, cnt, stream=stream.cuda_stream)  # the FIRST execute of this plan, captured
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert bool((X["ids"] == H.IDS_SENTINEL).all()) and int(cnt.abs().sum()) == 0, "the capture launched work"
        for i in range(3):
            case.restore_slab(X)  # the step rewrites Q / K in place: a replay is the same step over the pristine slab
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            H.assert_bitwise(case, ref_x, X, f"replay {i}")
            H.assert_counters_zero(cnt, f"replay {i}")
    finally:
        graph.reset()


@needs_cuda
def test_two_steps_back_to_back_on_one_plan_follow_the_reference_step_by_step():
    """Step 1 at ``kv_lens`` L, step 2 at L + S_q on the SAME ring / cache / pools through ONE plan (the second step derives its completing
    row from ring rows the first step wrote): each step bitwise the three-launch form's own two steps, the counters zero after each."""
    H = _H()
    b, s_q, nbm = 3, 4, 2048
    kv1 = [4 * 300 + 2, 4 * 17 + 1, 4 * 2040]
    case1 = _case(b, s_q, nbm, 16, kv=kv1, seed=21)
    plan = _fused(case1, units=1)
    _skip_unless_reference()
    case2 = _case(b, s_q, nbm, 16, kv=[x + s_q for x in kv1], seed=21, slab_seed=22)  # the same caches and tables, the next step's slab
    assert torch.equal(case2.block_table, case1.block_table) and H.same(case2.comp0, case1.comp0) and H.same(case2.ring0, case1.ring0)
    ref = H.ThreeLaunch(case1)
    stream = H.current_stream_handle()
    cnt = H.new_counters(case1)
    rx, fx = case1.fresh(), case1.fresh()
    ref.run(rx, stream=stream)
    plan.run(fx, cnt, stream=stream)
    H.assert_bitwise(case1, rx, fx, "step 1")
    H.assert_counters_zero(cnt, "step 1")
    rx2, fx2 = case2.fresh_from(rx), case2.fresh_from(fx)
    ref2 = H.ThreeLaunch(case2)
    ref2.run(rx2, stream=stream)
    plan2 = _fused(case2, units=1)
    assert plan2.plan is plan.plan or plan2.keys == plan.keys, "step 2 must reuse step 1's plan (the compile keys are the declaration's)"
    plan2.run(fx2, cnt, stream=stream)
    H.assert_bitwise(case2, rx2, fx2, "step 2")
    H.assert_counters_zero(cnt, "step 2")
    for seq, hit in enumerate(case2.completing()):
        assert hit is not None and int(fx2["cslot"][seq, hit[0]]) == hit[1], (seq, hit, fx2["cslot"][seq].tolist())


# ---------------------------------------------------------------------------------------------------------------------------------------
# T6 -- the trap rendering: forward progress under an armed deadline, and the detector of the placement assumption
# ---------------------------------------------------------------------------------------------------------------------------------------


def _sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@requires_rubin
@needs_cuda
@pytest.mark.parametrize("b,s_q,nbm,units,ctas", [(32, 1, 8192, 8, 317), (32, 1, 32768, 1, 8253)], ids=["B32_32K_u8_317ctas", "B32_128K_u1_forced_8253ctas"])
def test_fused_indexer_grid_larger_than_the_resident_capacity_terminates(b, s_q, nbm, units, ctas):
    """A grid of more CTAs than the device holds resident (several waves of spinning consumers) completes under the 2-s deadline trap --
    every acquire returns because every producer of a consumer is dispatched before it -- bitwise the reference, the counters zero;
    a trap or a hang here is the forward-progress assumption failing on this part + driver."""
    tb = _table(b, s_q, nbm, units, 1024)
    assert tb.n_ctas == ctas and tb.n_ctas > _sm_count(), (tb, _sm_count())
    _, res = _child("trap", dict(batch=b, s_q=s_q, n_blocks_max=nbm, page=16, units=units, threads=1024, deadline_s=2.0, seed=31), timeout=1500)
    assert res["completed"] and not res["trapped"], res
    assert res["counters_zero"] and res["differs_from_reference"] == [], res


def _stranding_order(b, s_q, nbm, units, sms):
    """A served dispatch order that violates the placement pin AND puts at least ``sms`` top-k CTAs ahead of the first scorer CTA -- the
    only placement that strands every resident slot on a consumer whose producers cannot be dispatched (a per-sequence interleave keeps
    forward progress under in-order dispatch: at most one sequence's consumers ever wait on undispatched producers)."""
    seen = {}
    for order in F.DISPATCH_ORDERS:
        tb = _table(b, s_q, nbm, units, 1024, order=order)
        if tb.producers_before_consumers():
            continue
        first_producer = min(tb.linear_index(F.ROLE_SCORER, bb, k) for bb in range(b) for k in range(tb.n_chunks))
        ahead = sum(1 for bb in range(b) for s in range(s_q) if tb.linear_index(F.ROLE_TOPK, bb, s) < first_producer)
        seen[order] = ahead
        if ahead >= sms:
            return order
    pytest.fail(
        f"no dispatch order strands the machine: the trap test needs an order placing >= {sms} top-k CTAs (the SM count) ahead of the first "
        f"scorer CTA, found {seen} -- a per-sequence consumers-first interleave keeps forward progress under in-order dispatch (every resident "
        "consumer but the last sequence's has its producers dispatched), so the detector needs a GLOBAL mis-order (every sequence's top-k "
        "CTAs before any scorer CTA) and a row count past the SM count"
    )


@needs_cuda
def test_consumer_before_producer_order_traps():
    """The detector of the forward-progress assumption, live: a plan whose dispatch table strands every resident slot on a consumer
    (every top-k CTA of every sequence ahead of every scorer CTA, more rows than SMs) TRAPS under the 2-s deadline -- a launch failure
    within seconds, never a hang (the child's budget is the hang detector, never raised)."""
    sms = _sm_count()
    b, s_q, nbm, units = sms + 8, 1, 520, 1
    order = _stranding_order(b, s_q, nbm, units, sms)
    _, res = _child("trap", dict(batch=b, s_q=s_q, n_blocks_max=nbm, page=16, units=units, order=order, deadline_s=2.0, compare=False, seed=37), timeout=900)
    assert res["trapped"] and not res["completed"], res
    assert res["elapsed_s"] < 60.0, f"the deadline must end the launch within seconds, not the outer budget: {res}"


# ---------------------------------------------------------------------------------------------------------------------------------------
# T8 -- the host PTX census of the fused rendering and the per-role local-memory pin (sm_107a trace-compiles, no GPU)
# ---------------------------------------------------------------------------------------------------------------------------------------

_RENDER_PROBE = textwrap.dedent("""
    import glob, json, os, sys
    dump, arch, harness_dir, kw = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
    os.environ["CUTE_DSL_DUMP_DIR"] = dump                      # read once, at the first cutlass import
    os.environ["CUTE_DSL_KEEP"] = "cubin,ptx"                   # the one token list of the keep switch
    os.environ["CUTE_DSL_ARCH"] = arch                          # unconditional: an inherited value would pin another target
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"   # a compiled-plan cache HIT skips ptxas and dumps nothing
    for k in ("CUTE_DSL_LINEINFO", "CUTE_DSL_LOC_TRACEBACKS"):
        os.environ.pop(k, None)
    sys.path.insert(0, harness_dir)
    import torch
    import qsa_indexer_fused_step as H
    import cudnn.frost.device as DEV
    from cudnn.frost.device import build_device
    from cudnn.gated_attention_block.kernels import qsa_indexer_fused as F
    # the host has no target device: the front door reads the TARGET's device facts here (cc 10.7; the part's opt-in and oversized
    # SMEM ceilings, 227 KiB / 327 KiB), so the compile takes the target's limits under the arch override, as the launch on the node does
    DEV.compute_capability = lambda d: (10, 7)
    DEV.shared_memory_per_block_optin = lambda d: 232448
    DEV.oversized_shared_memory_per_block = lambda d: 334848
    keys = H.fused_compile_keys(H.FLASH_NEXT, s_q=kw["s_q"], page=kw["page"], n_blocks_max=kw["n_blocks_max"], units=kw["units"], threads=kw["threads"],
                                use_pdl=kw.get("use_pdl", False), check_arrivals=kw.get("check_arrivals", False), deadline_ns=kw.get("deadline_ns"),
                                roles=kw.get("roles"), order=kw.get("order"), units_per_warp=kw.get("units_per_warp", 1), device=0)
    with build_device(0):  # the compile key's device index, pinned: the trace needs no CUDA device (CUTE_DSL_ARCH names the target)
        try:
            F.compile_qsa_indexer_fused(**keys)
        except NotImplementedError as e:
            if H.decline_reason(e) is not None:
                print("DECLINED " + str(e).replace(chr(10), " ")); sys.exit(0)
            raise
    ptx = sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)
    cub = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
    print("PTX " + (ptx[-1] if ptx else "")); print("CUBIN " + (cub[-1] if cub else ""))
    """)

_PTX_KEYS = (
    "red.relaxed.gpu.global.add.u64",
    "red.release",
    "ld.acquire.gpu.global.u64",
    "nanosleep",
    "st.relaxed.gpu.global.u64",
    "fence.acq_rel.gpu",
    "griddepcontrol.wait",
    "griddepcontrol.launch_dependents",
    "bar.sync",
    "barrier.cluster",
    "mbarrier",
    "cp.async.bulk",
    ".cluster",
    "mma.sync",
    "cp.async.cg",
    "st.shared",
    "ld.shared",
    ".local",
    "%globaltimer",
    "trap;",
    "fence.sc.sys",
)


def _cuda_tool(name):
    """A CUDA binary (``nvdisasm`` / ``cuobjdump`` / ``ptxas``) from ``CUDNN_TEST_CUDA_BIN``, the toolkit roots, then PATH; None when absent."""
    dirs = [os.environ.get("CUDNN_TEST_CUDA_BIN")] + [
        os.path.join(os.environ[v], "bin") for v in ("CUDA_HOME", "CUDAToolkit_ROOT", "CUDA_PATH") if os.environ.get(v)
    ]
    for d in [x for x in dirs if x] + ["/usr/local/cuda/bin"]:
        p = os.path.join(d, name)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(name)


def _render(tmp_path, tag, kw, timeout=1800):
    """Trace-compile ONE fused rendering for sm_107a in a child (the compile is a STAGE with its own budget, never read as a hang) and
    return ``(ptx text, cubin path)``; a Rule 7 decline of the host rendering skips naming it."""
    _H()  # the DSL floor (the harness imports the kernel modules)
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
    except Exception:  # noqa: BLE001
        pytest.skip("this CuTe DSL has no sm_107a target")
    dump = tmp_path / f"render_{tag}"
    dump.mkdir()
    probe = tmp_path / f"render_{tag}.py"
    probe.write_text(_RENDER_PROBE)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(probe), str(dump), "sm_107a", str(_HERE), json.dumps(kw)], capture_output=True, text=True, timeout=timeout)
    print(f"\nSTAGE compile {tag} {time.time() - t0:.1f} s rc {proc.returncode}")
    assert proc.returncode == 0, f"the sm_107a trace-compile of {tag} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}"
    declined = next((ln for ln in proc.stdout.splitlines() if ln.startswith("DECLINED ")), None)
    if declined:
        pytest.skip(f"the host rendering {tag} declines by name: {declined[9:]}")
    ptx = next((ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("PTX ")), "")
    cubin = next((ln[6:] for ln in proc.stdout.splitlines() if ln.startswith("CUBIN ")), "")
    assert ptx and os.path.isfile(ptx), f"no PTX landed in {dump}: {sorted(os.listdir(dump))}\n{proc.stdout[-2000:]}"
    return Path(ptx).read_text(), (cubin if cubin and os.path.isfile(cubin) else None)


def _census(ptx: str):
    return {k: ptx.count(k) for k in _PTX_KEYS}


def _sass_counts(cubin):
    """``(reg, stl, ldl)`` of a cubin through ``cuobjdump -res-usage`` + ``nvdisasm -c`` when a toolkit that decodes the target is found."""
    nvdisasm, cuobjdump = _cuda_tool("nvdisasm"), _cuda_tool("cuobjdump")
    if nvdisasm is None or cuobjdump is None:
        pytest.skip("no nvdisasm / cuobjdump on this host (CUDNN_TEST_CUDA_BIN names a toolkit that decodes sm_107a)")
    usage = subprocess.run([cuobjdump, "-res-usage", cubin], capture_output=True, text=True, timeout=300)
    dis = subprocess.run([nvdisasm, "-c", cubin], capture_output=True, text=True, timeout=600)
    if usage.returncode != 0 or dis.returncode != 0:
        pytest.skip(f"this toolkit cannot decode the sm_107a cubin: {usage.stderr[-300:]} {dis.stderr[-300:]}")
    reg = int(m.group(1)) if (m := re.search(r"REG:(\d+)", usage.stdout)) else -1
    stl = len(re.findall(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P[0-9T]\s+)?STL", dis.stdout, re.M))
    ldl = len(re.findall(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P[0-9T]\s+)?LDL", dis.stdout, re.M))
    return reg, stl, ldl


_CENSUS_CELLS = [
    ("u1_t1024_s1_cap32768", dict(s_q=1, page=16, n_blocks_max=32768, units=1, threads=1024)),
    ("u2_t1024_s4_cap8192_pdl", dict(s_q=4, page=16, n_blocks_max=8192, units=2, threads=1024, use_pdl=True)),
    ("u8_t1024_s1_cap32768", dict(s_q=1, page=16, n_blocks_max=32768, units=8, threads=1024)),
    ("u1_t512_s1_cap32768", dict(s_q=1, page=16, n_blocks_max=32768, units=1, threads=512)),
    ("u2_t1024_s1_cap8192_check_arrivals", dict(s_q=1, page=16, n_blocks_max=8192, units=2, threads=1024, check_arrivals=True)),
    ("u1_t1024_s1_cap8192_deadline", dict(s_q=1, page=16, n_blocks_max=8192, units=1, threads=1024, deadline_ns=2_000_000_000)),
    (
        "u1_t512_s1_cap8192_deadline_global",
        dict(s_q=1, page=16, n_blocks_max=8192, units=1, threads=512, deadline_ns=2_000_000_000, order="consumers-first-global"),
    ),
    ("u4_t512_s4_cap32768", dict(s_q=4, page=16, n_blocks_max=32768, units=4, threads=512)),
]


@pytest.mark.parametrize("tag,kw", _CENSUS_CELLS, ids=[c[0] for c in _CENSUS_CELLS])
def test_ptx_census_of_the_fused_rendering(tmp_path, tag, kw):
    """The counter sites of the tables in the sm_107a PTX: ``red.relaxed.gpu.global.add.u64`` x S_q (the arrival count of the ledger, one
    site per row of the sequence) behind exactly ONE ``fence.acq_rel.gpu`` (the elected lane's: gsync's release form -- no ``red.release``
    anywhere), ``ld.acquire.gpu.global.u64`` == 2 (+1 under ``check_arrivals``) with ``nanosleep`` beside it,
    ``st.relaxed.gpu.global.u64`` == 1 (the reset); no ``.cluster``, no mbarrier, no bulk copy; no global atom and no
    global red outside the gsync sites (the top-k's shared histogram atomics are counted separately); ``griddepcontrol`` 3 + 3 under the
    PDL key and 0 + 0 without; the deadline's ``%globaltimer`` / ``trap`` / ``fence.sc.sys`` only when armed; a named barrier over
    ``128 x units`` threads in the scorer role; no local memory at 8192 columns (the 32-key tile at 32768 x 1024 is allowed its few)."""
    ptx, _ = _render(tmp_path, tag, kw)
    c = _census(ptx)
    print(f"\n[{tag}] {c}")
    s_q, units = kw["s_q"], kw["units"]
    assert c["red.relaxed.gpu.global.add.u64"] == s_q and c["red.release"] == 0, c
    assert c["ld.acquire.gpu.global.u64"] == 2 + (1 if kw.get("check_arrivals") else 0), c
    assert c["nanosleep"] >= 1 and c["st.relaxed.gpu.global.u64"] == 1 and c["fence.acq_rel.gpu"] == 1, c  # ONE fence: the R1 elected lane's
    assert c["barrier.cluster"] == 0 and c[".cluster"] == 0 and c["mbarrier"] == 0 and c["cp.async.bulk"] == 0, c
    global_atoms = re.findall(r"\b(?:atom|red)\.[\w.]*global[\w.]*", ptx)
    assert sorted(set(global_atoms)) == ["red.relaxed.gpu.global.add.u64"] and len(global_atoms) == s_q, global_atoms
    pdl = 3 if kw.get("use_pdl") else 0
    assert c["griddepcontrol.wait"] == pdl and c["griddepcontrol.launch_dependents"] == pdl, c
    armed = kw.get("deadline_ns") is not None
    assert (c["%globaltimer"] > 0) == armed and (c["fence.sc.sys"] > 0) == armed, c
    assert (c["trap;"] > 0) == (armed or bool(kw.get("check_arrivals"))), c  # the deadline's trap, or the exact-count trap of check_arrivals
    assert c["mma.sync"] > 0 and c["cp.async.cg"] > 0 and c["st.shared"] > 0 and c["bar.sync"] >= 8, c
    if units in (1, 2, 4):
        assert re.search(
            rf"bar\.sync\s+[^;]*,\s*{128 * units}\s*;", ptx
        ), f"no named barrier over 128 x {units} threads (the scorer role's publish / release barrier)"
    if kw["n_blocks_max"] <= 8192:
        assert c[".local"] == 0, c


_ROLE_PIN_DSL = ("nvidia-cutlass-dsl 4.8.0", "ptxas 13.4.59")  # the DSL build (the distribution string, the bundled assembler) the figures were taken under
_ROLE_PINS = [
    # (role, units, threads, key overrides, (STL, LDL)) of the role-isolated sm_107a rendering -- S_q 1 and 8192 columns unless the overrides
    # say otherwise: the register census's figures, EXACT
    ("prepare", 1, 512, {}, (0, 0)),
    ("scorer", 1, 512, {}, (0, 0)),
    ("scorer", 2, 512, {}, (0, 0)),
    ("scorer", 4, 512, {}, (0, 0)),
    ("topk", 1, 512, {}, (0, 0)),
    ("prepare", 1, 1024, {}, (0, 0)),
    ("scorer", 1, 1024, {}, (0, 0)),
    ("scorer", 2, 1024, {}, (0, 0)),
    ("scorer", 8, 1024, {}, (1, 1)),  # the knob's recorded pair at 64 registers (the scorer at MMA batch 1): the figure that made 512 the default
    ("topk", 1, 1024, {}, (0, 0)),
    # the review's wider scope: the S_q 4 scorer at the default route (B 32: 4 units) and at the knob, the served units_per_warp 2 point
    # (512 threads: 99 registers, spill-free; the knob refuses it), the default route's 128K top-k (32768 columns = the 64-key tile at
    # 512 threads: its standalone 512 form's own pair) and the knob's 32768-column top-k
    ("scorer", 1, 512, dict(s_q=4), (0, 0)),
    ("scorer", 4, 512, dict(s_q=4), (0, 0)),
    ("scorer", 1, 1024, dict(s_q=4), (1, 1)),
    ("scorer", 1, 512, dict(units_per_warp=2), (0, 0)),
    ("topk", 1, 512, dict(n_blocks_max=32768), (27, 34)),
    ("topk", 1, 1024, dict(n_blocks_max=32768), (14, 14)),
]


@pytest.mark.parametrize(
    "role,units,threads,over,expect",
    _ROLE_PINS,
    ids=[f"{r}_u{u}_t{t}" + "".join(f"_{k}{v}" for k, v in o.items()) for r, u, t, o, _ in _ROLE_PINS],
)
def test_per_role_local_memory_matches_the_register_census(tmp_path, role, units, threads, over, expect):
    """The register pin BEFORE any GPU time: a role-isolated sm_107a rendering (the ``roles`` compile key traces the other branches empty at
    the SAME block size, SMEM and ABI).  At the DEFAULT 512 threads (128 registers) every role is spill-free at every served units; at the
    1024-thread knob (64 registers flat) the prepare, the top-k at 8192 columns and the scorer at 1 / 2 units are spill-free and the scorer
    at 8 units (and at S_q 4) holds exactly the recorded ``STL`` / ``LDL`` pair -- the figure that made 512 the default, never a silent one;
    the 32768-column top-k ships with its standalone form's own pair at either block size.  EXACT pins (a change in either direction is a
    change to record), valid for the DSL build they were taken under: another build skips naming both sides, or fails under
    ``CUDNN_FROST_REQUIRE_CUBIN_PINS=1`` (the cubin pins' contract); ``REG`` reads at most the file over the block."""
    cps.check_arch()
    cps.check_dsl(_ROLE_PIN_DSL[0])
    kw = {**dict(s_q=1, page=16, n_blocks_max=8192, units=units, threads=threads, roles=[role]), **over}
    ptx, cubin = _render(tmp_path, f"only_{role}_u{units}_t{threads}" + "".join(f"_{k}{v}" for k, v in over.items()), kw)
    if expect == (0, 0):
        assert ptx.count(".local") == 0, f"{role}: local memory in the PTX of a rendering recorded spill-free"
    # (a recorded pair is ptxas's: the PTX of the 8-unit scorer carries no .local and still spills one pair in SASS; the 32768-column top-k
    # carries its register tile's .local -- the SASS STL / LDL count below is the figure of record either way)
    assert cubin is not None, "no cubin was kept"
    cps.check_compiler(_ROLE_PIN_DSL[1], cps.compiler_of_cubin(cubin))
    reg, stl, ldl = _sass_counts(cubin)
    print(f"\n[{role} u{units} t{threads}] REG {reg} STL {stl} LDL {ldl}")
    assert 0 < reg <= 65536 // threads, reg
    assert (stl, ldl) == expect, f"{role} at units {units} / {threads} threads: STL {stl} / LDL {ldl}; the census recorded {expect}"
