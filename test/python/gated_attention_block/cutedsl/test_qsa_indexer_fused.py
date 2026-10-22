# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_indexer_fused``: the host half of the fused decode indexer -- the dispatch table and the pre-development tables.

The tables precede the body: no kernel body yet, so every cell here is a HOST cell (no GPU, no trace-compile).  The forward-progress
placement is pinned over every served cell (``producers_before_consumers``: the linear index of every scorer CTA of a sequence below every
top-k CTA of that sequence); beside it the dispatch table's CTA counts are pinned against the three standalone kernels' own grid
arithmetic (the prepare's warp units, the scorer's ``(chunks, B)`` grid, the top-k's ``(1, rows)`` grid -- computed by THEIR functions and
compared), the module's mirrored constants against the kernels' own, the docstring's dispatch rows against the function, the device
decode against the host encoding, the mis-ordered twin against the pin (RED on purpose), and the docstring's tables for presence.  The
cells that import a kernel module skip below the CuTe DSL floor (Rule 7); the dispatch-table cells import no kernel module (the host
facts come through the scorer's front door).
"""

import itertools

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error
from cudnn.gated_attention_block.kernels import qsa_indexer_fused as F
from cudnn.QSA import selection_paged as SP

pytestmark = [pytest.mark.L0]

_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("qsa_indexer_fused")  # None at or above the floor
needs_dsl = pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))

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


@pytest.mark.parametrize("b,s_q,nbm", CELLS, ids=[f"B{b}_S{s}_N{n}" for b, s, n in CELLS])
def test_the_mis_ordered_twin_fails_the_pin(b, s_q, nbm):
    """The trap rendering's input: the consumers-first order is REFUSED by the same pin -- the detector of the forward-progress
    assumption is live, not vacuous."""
    tb = _table(b, s_q, nbm, 1, 1024, order="consumers-first")
    assert not tb.producers_before_consumers()
    v = tb.first_violation()
    assert v is not None and v[2] > v[3], v
    assert tb.n_ctas == _table(b, s_q, nbm, 1, 1024).n_ctas  # the same grid, only the order differs


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
    "b,s_q,kv_tokens,units,expect,waves",
    [
        (1, 1, 8192, 1, (1, 16, 1, 18), 1),
        (1, 1, 32768, 1, (1, 64, 1, 66), 1),
        (1, 1, 131072, 2, (1, 128, 1, 130), 1),
        (4, 1, 8192, 1, (4, 16, 1, 72), 1),
        (4, 1, 32768, 2, (4, 32, 1, 136), 1),
        (4, 1, 131072, 4, (4, 64, 1, 264), 2),
        (1, 4, 8192, 1, (4, 16, 4, 24), 1),
        (4, 4, 8192, 1, (15, 16, 4, 95), 1),
        (32, 1, 8192, 8, (29, 2, 1, 125), 1),
        (32, 1, 32768, 8, (29, 8, 1, 317), 2),
        (32, 1, 131072, 8, (29, 32, 1, 1085), 6),
        (32, 4, 8192, 8, (114, 2, 4, 306), 2),
        (32, 4, 32768, 8, (114, 8, 4, 498), 3),
        (32, 4, 131072, 8, (114, 32, 4, 1266), 6),
    ],
)
def test_dispatch_rows_come_out_of_the_function(b, s_q, kv_tokens, units, expect, waves):
    """The docstring's dispatch rows: R0 CTAs, R1 per sequence, R2 per sequence, the total, the waves at 212 SMs."""
    tb = _table(b, s_q, kv_tokens // SP.BLOCK_SIZE, units, 1024)
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
    assert "red.release.gpu.global.add.u64 == S_q per R1 rendering" in doc and "``griddepcontrol.wait`` == 3" in doc
    assert "after ``prepare_role`` returns" in doc and "every node of the programmatic chain" in doc
    # the SMEM union and the two derive sites
    assert "sNew" in doc and "sA" in doc and "260.3 KiB" in doc and "113.4 KiB" in doc and "196 KiB" in doc and "OVERSIZED" in doc
    # the assumption, its detectors and its denial
    assert "non-decreasing linear ``blockIdx`` order" in doc and "NOT: a CUDA programming-model guarantee" in doc
    assert "deadline_ns" in doc and "consumers-first" in doc
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
    form refusals of the fused declaration (the query arm, a step wider than the compress ratio, the mis-ordered table without the trap's
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
    with pytest.raises(NotImplementedError, match="bf16 activations"):
        F.compile_qsa_indexer_fused(**dict(base, dtype=torch.float16))
    with pytest.raises(ValueError, match="served only with deadline_ns"):
        F.compile_qsa_indexer_fused(**dict(base, order="consumers-first"))
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
