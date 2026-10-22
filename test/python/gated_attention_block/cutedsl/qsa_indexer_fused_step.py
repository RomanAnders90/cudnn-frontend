# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused decode indexer's step harness: one block-sparse DECODE step's operands, the THREE-LAUNCH reference, the fused launch
behind ONE adapter, the bitwise comparison and the child-process entries (``test_qsa_indexer_fused.py`` drives it).

The reference is the serving step's own form, built from the three standalone kernels in the same process on the same primed caches
(the block's decode stage calls them in exactly this order and with exactly these operands): ``qsa_prepare`` with the decode arms
``QsaPrepareIndexArms(raw_pool=False, query=True, compress=True, paged=True, ring_slots, rope_table)`` -- the Q / K norm + RoPE in place,
the K / V pool rows, the indexer queries into the compact ``qi`` buffer, the raw key into the ring and the block a row completes into the
paged compressed-key cache --, then ``qsa_score_paged`` over ``qi`` and the cache through the main block table (``scale = 1 / sqrt(128)``,
``block_lens`` written), then ``qsa_topk_decode`` (the single-CTA form, 1024 threads) into the ``[T, top_k]`` list.  The fused launch runs
the SAME step on an identical set of buffers with the prepare's query arm OFF (the scorer role derives the query rows and the completing
compressed-key row itself), so every output but ``qi`` is compared: the list, ``counts``, ``block_lens``, the slot report, the Q / K bands
of the slab, the K / V pools, the ring, the compressed-key cache and the logits buffer -- all bitwise (``torch.equal`` on the bit patterns:
a NaN-filled pool row compares as its bits).

THE ADAPTER (:class:`FusedPlan`) is the one place that spells the fused kernel's host surface -- the compile keys of
``compile_qsa_indexer_fused`` and the operands of ``run_qsa_indexer_fused`` as the kernel module's docstrings and tables name them: the
prepare's arms + ``d / rope_dim / page_size / slot_dtype / apply_norm / eps`` (``index_head_dim`` and ``pool`` ride in the arms record), the
scorer's ``(s_q, units_per_warp, ring_depth, has_block_lens)`` (``h_i`` is the arms' head count), the top-k's ``(cols_cap, top_k)``,
``units`` / ``threads`` / ``use_pdl`` / ``device``, the test-only ``deadline_ns`` / ``check_arrivals`` / ``roles`` / ``order``; the run binds the
slab bands, the pools, the ring, the cache, the tables, the workspace views, the counters and the stream by KEYWORD.  Against the
pre-body module both entries raise ``NotImplementedError("... not served yet")`` and every cell below is RED by design.

Child entries (``python qsa_indexer_fused_step.py <mode> '<json>'``; a trap leaves a sticky context, a fresh-process screen needs fresh
processes): ``launches`` -- N launches of one plan in one process, the counters read back zero after every one and the outputs bitwise the
first launch's and the three-launch reference's; ``trap`` -- one launch of a plan compiled with a deadline (and an optional dispatch
order), reporting whether it completed within the deadline or trapped.  Each prints ONE ``RESULT <json>`` line; a Rule 7 decline of the
fused compile is reported as ``{"declined": <reason>}`` (the parent skips), the pre-body ``NotImplementedError`` propagates (RED).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from cudnn.gated_attention_block.kernels import qsa_indexer_fused as F
from cudnn.gated_attention_block.kernels.qsa_compress import min_ring_rows
from cudnn.gated_attention_block.kernels.qsa_prepare import QsaPrepareIndexArms, compile_qsa_prepare, run_qsa_prepare
from cudnn.gated_attention_block.kernels.qsa_score_paged import compile_qsa_score_paged, dsl_decline_reason, run_qsa_score_paged
from cudnn.gated_attention_block.kernels.qsa_topk_decode import cols_cap_for, compile_qsa_topk_decode, run_qsa_topk_decode
from cudnn.QSA.selection_paged import BLOCK_SIZE, INDEX_HEAD_DIM, QSA_TOP_K

BS = BLOCK_SIZE  # tokens per compressed block (the pool of the compress arm)
D_I = INDEX_HEAD_DIM
LOGIT_SENTINEL = 1.5e30  # a finite magnitude no logit reaches: a column still holding it was never written (the scorer suite's value)
IDS_SENTINEL = -2  # never produced by the top-k (ids >= 0, the padding -1): a slot still holding it was never written
COUNT_SENTINEL = -7
CSLOT_SENTINEL = -9
NOT_SERVED = "not served yet"  # the pre-body stubs' text: the one NotImplementedError that must stay RED
# Rule 7 declines of the fused compile (the DSL floor / target, the device family, PDL below cc 9.0, the SMEM opt-in limit): a skip by name
DECLINE_RE = re.compile(r"nvidia-cutlass-dsl|sm_107a|compute capability|\bcc\s*\d+\.\d+|needs cc|shared memory|the device allows|opt-in", re.IGNORECASE)


class Geometry(NamedTuple):
    """The block's geometry the step is built at (``FLASH_NEXT`` = Qwen 3.8 Flash-Next TP 1: 24 / 2 heads of 256, RoPE 64, 4 indexer heads
    of 128, top-k 512; ``SMALL`` = 8 query heads: cheaper projections, the same indexer band)."""

    h_q: int = 24
    h_kv: int = 2
    d: int = 256
    rope: int = 64
    h_i: int = 4
    d_i: int = D_I
    top_k: int = QSA_TOP_K
    eps: float = 1e-6
    index_eps: float = 1e-6


FLASH_NEXT = Geometry()
SMALL = Geometry(h_q=8, h_kv=2)


def is_not_served(e: BaseException) -> bool:
    return isinstance(e, NotImplementedError) and NOT_SERVED in str(e)


def decline_reason(e: BaseException) -> Optional[str]:
    """The Rule 7 decline text of a fused compile that refused by name (None for anything else, the pre-body stub included)."""
    if isinstance(e, NotImplementedError) and not is_not_served(e) and DECLINE_RE.search(str(e)):
        return str(e)
    return None


def reference_decline_reason(device: Optional[int] = None) -> Optional[str]:
    """Why the three-launch reference cannot run here (the scorer's own gate; the prepare and the single-CTA top-k run wherever it does)."""
    if not torch.cuda.is_available():
        return "the three-launch reference needs a CUDA device"
    return dsl_decline_reason(device)


def bits(x: torch.Tensor) -> torch.Tensor:
    """The bit patterns of a tensor (NaN-safe equality: a NaN-filled pool row compares as its bits)."""
    x = x.contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    return x


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(bits(a), bits(b)))


def kv_pattern(batch: int, s_q: int, n_blocks_max: int) -> List[int]:
    """Deterministic live KV lengths AFTER the step for ``batch`` sequences: sequence 0 at the cache's ceiling (every block visible), the
    rest spread over the range with every residue modulo the block size represented -- all ``>= s_q`` (the degenerate rows are the
    degenerate cells' own)."""
    cap = BS * n_blocks_max
    out = [cap]
    for b in range(1, batch):
        out.append(s_q + (b * 6133 + 97 * (b % 4) + b % 4) % max(cap - s_q, 1))
    return out


class StepCase:
    """One decode step's operands at ``geom``: the ``[T, N]`` projection slab (Q | GATE | K | V | IQ | IK bands; ``T = batch x s_q``), the
    norm weights, the per-token and the position-indexed RoPE tables, the K / V pools and the slot mapping, the raw-key ring (block 0 the
    null block, blocks 1..B the sequences') and the paged compressed-key cache (page 0 unused by a live sequence, a PERMUTED block table),
    the state table, the KV lengths AFTER the step (``kv_lens``; a padding sequence has 0, sits on ring block 0 / page 0 and owns no pool
    slot).  Every mutable buffer is cloned per form by :meth:`fresh`, so the reference and the fused launch start from identical bytes."""

    def __init__(
        self,
        geom: Geometry,
        *,
        batch: int,
        s_q: int,
        n_blocks_max: int,
        page: int,
        kv_lens: Sequence[int],
        seed: int,
        device=None,
        dtype=torch.bfloat16,
        slot_dtype=torch.int32,
        ring_cap: Optional[int] = None,
        rope_table: bool = False,
        ring_slots: bool = False,
        padding: Sequence[int] = (),
        max_pages: Optional[int] = None,
        n_kbar_pages: Optional[int] = None,
        slab_seed: Optional[int] = None,
    ):
        dev = torch.device("cuda") if device is None else torch.device(device)
        self.dev, self.geom, self.dtype, self.slot_dtype = dev, geom, dtype, slot_dtype
        self.B, self.S_q, self.T = int(batch), int(s_q), int(batch) * int(s_q)
        self.n_blocks_max, self.page, self.rpp = int(n_blocks_max), int(page), int(page) // BS
        self.rope_table, self.ring_slots, self.padding = bool(rope_table), bool(ring_slots), tuple(int(b) for b in padding)
        self.max_pages = -(-self.n_blocks_max // self.rpp) if max_pages is None else int(max_pages)
        if self.max_pages * self.rpp < self.n_blocks_max:
            raise ValueError(f"max_pages {self.max_pages} x {self.rpp} rows cannot address n_blocks_max {self.n_blocks_max} (the scorer refuses it)")
        if len(kv_lens) != self.B:
            raise ValueError(f"kv_lens must name {self.B} sequences, got {len(kv_lens)}")
        h_q, h_kv, d, h_i, d_i, T = geom.h_q, geom.h_kv, geom.d, geom.h_i, geom.d_i, self.T
        self.n = (2 * h_q + 2 * h_kv) * d + (h_i + 1) * d_i
        self.o_q, self.o_g, self.o_k, self.o_v = 0, h_q * d, 2 * h_q * d, 2 * h_q * d + h_kv * d
        self.o_iq, self.o_ik = (2 * h_q + 2 * h_kv) * d, (2 * h_q + 2 * h_kv) * d + h_i * d_i
        gen = torch.Generator(device=dev).manual_seed(int(seed))

        def rnd(*shape, std=1.0):
            return (torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32) * std).to(dtype)

        self.slab0 = rnd(T, self.n)
        if slab_seed is not None:  # the next step's projections: the caches, the tables and the slots stay the seed's, the slab is new
            g2 = torch.Generator(device=dev).manual_seed(int(slab_seed))
            self.slab0 = torch.randn(T, self.n, generator=g2, device=dev, dtype=torch.float32).to(dtype)
        self.w_q, self.w_k = (1.0 + 0.05 * rnd(d).float()).to(dtype), (1.0 - 0.05 * rnd(d).float()).to(dtype)
        self.w_iq, self.w_ik = (1.0 + 0.03 * rnd(d_i).float()).to(dtype), (1.0 + 0.02 * rnd(d_i).float()).to(dtype)
        kv = [0 if b in self.padding else int(x) for b, x in enumerate(kv_lens)]
        self.kv_lens_list = kv
        self.kv_lens = torch.tensor(kv, dtype=torch.int32, device=dev)
        # the position tables cover every position the main table can address (the RoPE-table arm's host bound) plus a margin of one
        # block table column, so a step row past the table's reach meets exactly ONE -2 predicate (the table's width, never the position table)
        self.s_pos = (self.max_pages + 1) * self.page
        inv = 1.0 / (1e7 ** (torch.arange(0, geom.rope, 2, device=dev, dtype=torch.float32) / geom.rope))
        emb = torch.cat([torch.arange(self.s_pos, device=dev, dtype=torch.float32)[:, None] * inv] * 2, -1)
        self.cos_pos, self.sin_pos = emb.cos().to(dtype)[None].contiguous(), emb.sin().to(dtype)[None].contiguous()  # [1, S_pos, rope]
        pos0 = self.kv_lens.long() - self.S_q
        tok_pos = (pos0[:, None] + torch.arange(self.S_q, device=dev)[None, :]).reshape(T).clamp_(0, self.s_pos - 1)
        self.cos_t, self.sin_t = self.cos_pos[0, tok_pos].contiguous(), self.sin_pos[0, tok_pos].contiguous()  # [T, rope] per token
        # the K / V pools and the slot mapping: a random slot per token, a padding sequence's tokens own none
        self.n_pages = T // self.page + 3
        self.slot = torch.randperm(self.n_pages * self.page, generator=gen, device=dev)[:T].to(slot_dtype)
        for b in self.padding:
            self.slot[b * self.S_q : (b + 1) * self.S_q] = -1
        self.live = self.slot >= 0
        # the ring: blocks 0..B of cap rows (block 0 null), random raw keys of earlier steps everywhere
        self.ring_n = max(8, min_ring_rows(self.S_q, BS)) if ring_cap is None else int(ring_cap)
        if self.ring_n < min_ring_rows(self.S_q, BS):
            raise ValueError(f"ring cap {self.ring_n} < min_ring_rows({self.S_q}) = {min_ring_rows(self.S_q, BS)}")
        self.ring0 = rnd(self.B + 1, self.ring_n, 1, d_i)
        # the compressed-key cache: a plausible key in every row of every page (earlier blocks are read by the scorer), a permuted table
        self.n_kbar_pages = 1 + self.B * self.max_pages if n_kbar_pages is None else int(n_kbar_pages)
        if self.n_kbar_pages < 1 + self.B * self.max_pages:
            raise ValueError("n_kbar_pages must hold a page per (sequence, table column) plus the null page 0")
        self.comp0 = rnd(self.n_kbar_pages, self.rpp, 1, d_i)
        perm = torch.randperm(self.n_kbar_pages - 1, generator=gen, device=dev)[: self.B * self.max_pages] + 1
        self.block_table = perm.view(self.B, self.max_pages).to(torch.int32).contiguous()
        self.state_table = torch.arange(1, self.B + 1, device=dev, dtype=torch.int32).view(self.B, 1).contiguous()
        for b in self.padding:
            self.block_table[b] = 0
            self.state_table[b] = 0
        pos = pos0[:, None] + torch.arange(self.S_q, device=dev)[None, :]
        live_rows = (pos >= 0) & (self.state_table.long() > 0)
        self.ring_slots_t = (
            torch.where(live_rows, self.state_table.long() * self.ring_n + pos % self.ring_n, torch.full_like(pos, -1)).reshape(T).to(slot_dtype)
            if ring_slots
            else None
        )

    # -- host truths ---------------------------------------------------------------------------------------------------------------
    def positions(self) -> torch.Tensor:
        """``[B, S_q]`` int64: the position of every step row (negative = a dead row)."""
        return (self.kv_lens.long() - self.S_q)[:, None] + torch.arange(self.S_q, device=self.dev)[None, :]

    def n_vis(self) -> torch.Tensor:
        """``[B, S_q]`` int64: the visible complete-block count of every row, ``clamp(floor((pos + 1) / 4), 0, n_blocks_max)``."""
        return torch.div(self.positions() + 1, BS, rounding_mode="floor").clamp(0, self.n_blocks_max)

    def completing(self) -> List[Optional[Tuple[int, int]]]:
        """Per sequence ``(j, blk_new)`` for the step row that completes a block (``(pos + 1) % 4 == 0`` at a position ``>= 0``), else None."""
        out: List[Optional[Tuple[int, int]]] = []
        pos = self.positions().tolist()
        for b in range(self.B):
            hit = None
            for j, p in enumerate(pos[b]):
                if p >= 0 and (p + 1) % BS == 0:
                    hit = (j, (p + 1) // BS - 1)
            out.append(hit)
        return out

    def kbar_row(self, X: dict, b: int, blk: int) -> torch.Tensor:
        """Block ``blk`` of sequence ``b`` in copy ``X``'s compressed-key cache, through the block table (``[D_i]``)."""
        return X["comp"][int(self.block_table[b, blk // self.rpp]), blk % self.rpp, 0]

    # -- the buffers of one form -----------------------------------------------------------------------------------------------------
    def fresh(self) -> dict:
        g, dev, dt = self.geom, self.dev, self.dtype
        pool = lambda h, d: torch.full((self.n_pages, h, self.page, d), float("nan"), device=dev, dtype=dt)  # noqa: E731
        return dict(
            slab=self.slab0.clone(),
            k_pool=pool(g.h_kv, g.d),
            v_pool=pool(g.h_kv, g.d),
            ring=self.ring0.clone(),
            comp=self.comp0.clone(),
            cslot=torch.full((self.B, self.S_q), CSLOT_SENTINEL, dtype=torch.int32, device=dev),
            qi=torch.full((self.T, g.h_i, g.d_i), float("nan"), device=dev, dtype=dt),  # the reference's compact indexer queries (fused: unwritten)
            logits=torch.full((self.T, self.n_blocks_max), LOGIT_SENTINEL, dtype=torch.float32, device=dev),
            counts=torch.full((self.T,), COUNT_SENTINEL, dtype=torch.int32, device=dev),
            lens=torch.full((self.T,), COUNT_SENTINEL, dtype=torch.int32, device=dev),
            ids=torch.full((self.T, g.top_k), IDS_SENTINEL, dtype=torch.int32, device=dev),
        )

    def fresh_from(self, prev: dict) -> dict:
        """The next step's buffers over the PREVIOUS step's caches: the pools, the ring and the compressed-key cache carried over (the
        same tensors), the slab and every output fresh."""
        X = self.fresh()
        for k in ("k_pool", "v_pool", "ring", "comp"):
            X[k] = prev[k]
        return X

    def restore_slab(self, X: dict) -> None:
        """The step rewrites Q / K in place (norm + RoPE): a second launch over the same slab is a different step -- restore first."""
        X["slab"].copy_(self.slab0)

    def cols(self, slab: torch.Tensor, off: int, h: int, d: int) -> torch.Tensor:
        return torch.as_strided(slab, (self.T, h, d), (self.n, d, 1), slab.storage_offset() + off)

    def bands(self, slab: torch.Tensor) -> dict:
        g = self.geom
        return dict(
            q=self.cols(slab, self.o_q, g.h_q, g.d),
            k=self.cols(slab, self.o_k, g.h_kv, g.d),
            v=self.cols(slab, self.o_v, g.h_kv, g.d),
            iq=self.cols(slab, self.o_iq, g.h_i, g.d_i),
            ik=self.cols(slab, self.o_ik, 1, g.d_i),
        )

    def arms(self, *, query: bool) -> QsaPrepareIndexArms:
        """The decode step's prepare arms: the three-launch form carries the query arm, the fused form drops it (the scorer role derives
        the query rows); the raw-key POOL arm is off at decode (the ring keeps the raw key)."""
        g = self.geom
        return QsaPrepareIndexArms(
            heads=g.h_i,
            head_dim=g.d_i,
            raw_pool=False,
            query=bool(query),
            compress=True,
            n_commit=False,
            pool=BS,
            paged=True,
            ring_slots=self.ring_slots,
            rope_table=self.rope_table,
        )

    def paged_kwargs(self) -> dict:
        kw = dict(kv_lens=self.kv_lens, block_table=self.block_table, index_state_block_table=self.state_table)
        if self.ring_slots:
            kw["index_state_slot_mapping"] = self.ring_slots_t
        return kw

    def outputs(self, X: dict, *, fused: bool) -> Dict[str, torch.Tensor]:
        """Every output of the step this form writes (the reference's ``qi`` excluded from the fused form: nothing writes it there)."""
        b = self.bands(X["slab"])
        out = dict(
            ids=X["ids"],
            counts=X["counts"],
            block_lens=X["lens"],
            cslot=X["cslot"],
            q_band=b["q"],
            k_band=b["k"],
            slab=X["slab"],
            k_pool=X["k_pool"],
            v_pool=X["v_pool"],
            ring=X["ring"],
            comp=X["comp"],
            logits=X["logits"],
        )
        if not fused:
            out["qi"] = X["qi"]
        return out


def assert_bitwise(case: StepCase, ref: dict, fused: dict, label: str) -> None:
    """The oracle: every output of the fused launch ``torch.equal`` (on the bits) the three-launch form's, on the same primed caches."""
    torch.cuda.synchronize()
    a, b = case.outputs(ref, fused=False), case.outputs(fused, fused=True)
    bad = [nm for nm in b if not same(a[nm], b[nm])]
    assert not bad, f"{label}: the fused launch differs from the three-launch form on {bad}"
    # the step's untouched bands stay untouched in both forms
    bands = case.bands(fused["slab"])
    orig = case.bands(case.slab0)
    for nm in ("v", "iq", "ik"):
        assert same(bands[nm], orig[nm]), f"{label}: the fused launch rewrote the {nm} band"


def assert_counters_zero(cnt_row: torch.Tensor, label: str) -> None:
    torch.cuda.synchronize()
    nz = cnt_row.nonzero().flatten().tolist()
    assert not nz, f"{label}: cnt_row is not zero after the launch at rows {nz[:8]} (values {cnt_row[nz[:8]].tolist()})"


def new_counters(case: StepCase) -> torch.Tensor:
    """The plan-owned counter words: int64 ``[T]``, zero at build on the ambient stream, outside any capture."""
    return torch.zeros(case.T, dtype=torch.int64, device=case.dev)


class ThreeLaunch:
    """The reference: the three standalone kernels compiled for ``case`` (the prepare with the query + paged compress arms, the scorer, the
    single-CTA top-k at ``threads``) and run in stream order on one set of buffers."""

    def __init__(
        self, case: StepCase, *, units_per_warp: int = 1, ring_depth: int = 4, has_block_lens: bool = True, use_pdl: bool = False, threads: int = 1024
    ):
        g = case.geom
        self.case, self.has_block_lens = case, bool(has_block_lens)
        self.r_prep = compile_qsa_prepare(
            dtype=case.dtype,
            h_q=g.h_q,
            h_kv=g.h_kv,
            d=g.d,
            rope_dim=g.rope,
            eps=g.eps,
            page_size=case.page,
            slot_dtype=case.slot_dtype,
            apply_norm=True,
            index=case.arms(query=True),
            use_pdl=use_pdl,
        )
        self.r_score = compile_qsa_score_paged(
            h_i=g.h_i, s_q=case.S_q, page_size=case.page, units_per_warp=units_per_warp, has_block_lens=has_block_lens, use_pdl=use_pdl, ring_depth=ring_depth
        )
        self.cols_cap = cols_cap_for(case.n_blocks_max, 1, threads)
        self.r_topk = compile_qsa_topk_decode(cols_cap=self.cols_cap, top_k=g.top_k, cluster_ctas=1, threads_per_cta=threads, use_pdl=use_pdl)

    def run(self, X: dict, *, stream) -> None:
        c, g = self.case, self.case.geom
        b = c.bands(X["slab"])
        cos_t, sin_t = (None, None) if c.rope_table else (c.cos_t, c.sin_t)
        run_qsa_prepare(
            self.r_prep,
            b["q"],
            b["k"],
            b["v"],
            None,
            None,
            c.w_q,
            c.w_k,
            cos_t,
            sin_t,
            X["k_pool"],
            X["v_pool"],
            c.slot,
            index_k_raw_src=b["ik"],
            index_q_src=b["iq"],
            index_q_out=X["qi"],
            w_iq_norm=c.w_iq,
            ring=X["ring"],
            index_k_compressed=X["comp"],
            w_ik_norm=c.w_ik,
            cos_pos=c.cos_pos,
            sin_pos=c.sin_pos,
            pos0=None,
            n_commit=None,
            compressed_slot=X["cslot"],
            seq_q=c.S_q,
            index_eps=g.index_eps,
            index_norm_weight_offset=0.0,
            stream=stream,
            **c.paged_kwargs(),
        )
        run_qsa_score_paged(
            self.r_score,
            X["qi"],
            X["comp"],
            c.block_table,
            c.kv_lens,
            X["logits"],
            X["counts"],
            X["lens"] if self.has_block_lens else None,
            top_k=g.top_k,
            scale=float(g.d_i) ** -0.5,
            stream=stream,
        )
        run_qsa_topk_decode(self.r_topk, X["logits"], X["counts"], X["ids"], stream=stream)


def fused_compile_keys(
    geom: Geometry,
    *,
    s_q: int,
    page: int,
    n_blocks_max: int,
    units: int,
    threads: int = 1024,
    slot_dtype=torch.int32,
    dtype=torch.bfloat16,
    rope_table: bool = False,
    ring_slots: bool = False,
    units_per_warp: int = 1,
    ring_depth: int = 4,
    has_block_lens: bool = True,
    use_pdl: bool = False,
    device: Optional[int] = None,
    deadline_ns: Optional[int] = None,
    check_arrivals: bool = False,
    roles: Optional[Sequence[str]] = None,
    order: Optional[str] = None,
) -> dict:
    """The compile keys of ``compile_qsa_indexer_fused`` from the step's shapes alone (no tensor, no device: the host renderings use it
    under ``build_device``).  The test-only keys (``deadline_ns``, ``check_arrivals``, ``roles``, ``order``) are passed only when armed."""
    keys = dict(
        dtype=dtype,
        h_q=geom.h_q,
        h_kv=geom.h_kv,
        d=geom.d,
        rope_dim=geom.rope,
        eps=geom.eps,
        page_size=int(page),
        slot_dtype=slot_dtype,
        apply_norm=True,
        index=QsaPrepareIndexArms(
            heads=geom.h_i,
            head_dim=geom.d_i,
            raw_pool=False,
            query=False,
            compress=True,
            n_commit=False,
            pool=BS,
            paged=True,
            ring_slots=bool(ring_slots),
            rope_table=bool(rope_table),
        ),
        s_q=int(s_q),
        units_per_warp=int(units_per_warp),
        ring_depth=int(ring_depth),
        has_block_lens=bool(has_block_lens),
        cols_cap=cols_cap_for(int(n_blocks_max), 1, int(threads)),
        top_k=geom.top_k,
        units=int(units),
        threads=int(threads),
        use_pdl=bool(use_pdl),
        device=device,
    )
    if deadline_ns is not None:
        keys["deadline_ns"] = int(deadline_ns)
    if check_arrivals:
        keys["check_arrivals"] = True
    if roles is not None:
        keys["roles"] = tuple(roles)
    if order is not None:
        keys["order"] = str(order)
    return keys


class FusedPlan:
    """The fused launch behind the adapter: ``compile_qsa_indexer_fused(**keys)`` once, ``run_qsa_indexer_fused`` per step (keyword operands)."""

    def __init__(self, case: StepCase, *, units: int, threads: int = 1024, **extra):
        self.case = case
        self.keys = fused_compile_keys(
            case.geom,
            s_q=case.S_q,
            page=case.page,
            n_blocks_max=case.n_blocks_max,
            units=units,
            threads=threads,
            slot_dtype=case.slot_dtype,
            dtype=case.dtype,
            rope_table=case.rope_table,
            ring_slots=case.ring_slots,
            **extra,
        )
        self.table = F.dispatch_table(
            case.T,
            case.B,
            case.S_q,
            case.geom.h_q,
            case.geom.h_kv,
            case.geom.h_i,
            case.n_blocks_max,
            units,
            threads,
            d=case.geom.d,
            index_head_dim=case.geom.d_i,
            units_per_warp=self.keys["units_per_warp"],
            order=self.keys.get("order", "sequence-major"),
        )
        self.plan = F.compile_qsa_indexer_fused(**self.keys)

    def run(self, X: dict, cnt_row: torch.Tensor, *, stream) -> None:
        c, g = self.case, self.case.geom
        b = c.bands(X["slab"])
        cos_t, sin_t = (None, None) if c.rope_table else (c.cos_t, c.sin_t)
        F.run_qsa_indexer_fused(
            self.plan,
            q=b["q"],
            k=b["k"],
            v=b["v"],
            w_q=c.w_q,
            w_k=c.w_k,
            cos=cos_t,
            sin=sin_t,
            k_cache=X["k_pool"],
            v_cache=X["v_pool"],
            slot_mapping=c.slot,
            index_k_raw_src=b["ik"],
            index_q_src=b["iq"],
            w_iq_norm=c.w_iq,
            ring=X["ring"],
            index_k_compressed=X["comp"],
            w_ik_norm=c.w_ik,
            cos_pos=c.cos_pos,
            sin_pos=c.sin_pos,
            compressed_slot=X["cslot"],
            seq_q=c.S_q,
            index_eps=g.index_eps,
            index_norm_weight_offset=0.0,
            ix_logits=X["logits"],
            ix_counts=X["counts"],
            ix_block_lens=X["lens"] if self.keys["has_block_lens"] else None,
            ix_ids=X["ids"],
            top_k=g.top_k,
            scale=float(g.d_i) ** -0.5,
            cnt_row=cnt_row,
            stream=stream,
            **c.paged_kwargs(),
        )


def current_stream_handle() -> int:
    return torch.cuda.current_stream().cuda_stream


def run_reference(case: StepCase, *, threads: int = 1024, has_block_lens: bool = True, use_pdl: bool = False) -> Tuple[ThreeLaunch, dict]:
    ref = ThreeLaunch(case, threads=threads, has_block_lens=has_block_lens, use_pdl=use_pdl)
    X = case.fresh()
    ref.run(X, stream=current_stream_handle())
    torch.cuda.synchronize()
    return ref, X


def run_fused_once(case: StepCase, plan: FusedPlan, cnt_row: Optional[torch.Tensor] = None) -> Tuple[dict, torch.Tensor]:
    cnt_row = new_counters(case) if cnt_row is None else cnt_row
    X = case.fresh()
    plan.run(X, cnt_row, stream=current_stream_handle())
    torch.cuda.synchronize()
    return X, cnt_row


# ------------------------------------------------------------------------------------------------------------------ the child entries


def _case_from(args: dict) -> StepCase:
    geom = FLASH_NEXT if args.get("geom", "flash_next") == "flash_next" else SMALL
    kv = args.get("kv_lens") or kv_pattern(int(args["batch"]), int(args["s_q"]), int(args["n_blocks_max"]))
    return StepCase(
        geom,
        batch=int(args["batch"]),
        s_q=int(args["s_q"]),
        n_blocks_max=int(args["n_blocks_max"]),
        page=int(args.get("page", 16)),
        kv_lens=kv,
        seed=int(args.get("seed", 1)),
        rope_table=bool(args.get("rope_table", False)),
        ring_slots=bool(args.get("ring_slots", False)),
        padding=tuple(args.get("padding", ())),
    )


def _plan_or_decline(case: StepCase, args: dict, **extra):
    try:
        return FusedPlan(case, units=int(args.get("units", 1)), threads=int(args.get("threads", 1024)), **extra), None
    except NotImplementedError as e:
        why = decline_reason(e)
        if why is None:
            raise
        return None, why


def _result(d: dict) -> int:
    print("RESULT " + json.dumps(d), flush=True)
    return 0


def child_launches(args: dict) -> int:
    """N launches of one plan in one process: the counters zero after every launch, every launch bitwise the first and the reference."""
    case = _case_from(args)
    plan, why = _plan_or_decline(case, args, check_arrivals=bool(args.get("check_arrivals", False)))
    if plan is None:
        return _result({"declined": why})
    _, ref_x = run_reference(case)
    n = int(args.get("launches", 100))
    cnt_row = new_counters(case)
    X = case.fresh()
    first = None
    nonzero_after: List[int] = []
    differs: List[int] = []
    t0 = time.time()
    for i in range(n):
        case.restore_slab(X)
        plan.run(X, cnt_row, stream=current_stream_handle())
        torch.cuda.synchronize()
        if int(cnt_row.abs().sum().item()) != 0:
            nonzero_after.append(i)
        outs = case.outputs(X, fused=True)
        if first is None:
            first = {k: v.clone() for k, v in outs.items()}
            ref = case.outputs(ref_x, fused=False)
            bad = [k for k in outs if not same(outs[k], ref[k])]
            if bad:
                return _result({"ok": False, "launches": n, "first_differs_from_reference": bad, "elapsed_s": round(time.time() - t0, 3)})
        elif any(not same(outs[k], first[k]) for k in outs):
            differs.append(i)
    return _result(
        {
            "ok": not nonzero_after and not differs,
            "launches": n,
            "nonzero_counters_after": nonzero_after[:16],
            "launches_differing_from_the_first": differs[:16],
            "grid": plan.table.n_ctas,
            "elapsed_s": round(time.time() - t0, 3),
        }
    )


def child_trap(args: dict) -> int:
    """One launch of a plan compiled with a deadline (``deadline_s``) and an optional dispatch ``order``: completed (and bitwise the
    reference) or trapped (a sticky launch failure within the deadline) -- never a hang."""
    case = _case_from(args)
    deadline_ns = int(float(args.get("deadline_s", 2.0)) * 1e9)
    extra = dict(deadline_ns=deadline_ns)
    if args.get("order"):
        extra["order"] = str(args["order"])
    plan, why = _plan_or_decline(case, args, **extra)
    if plan is None:
        return _result({"declined": why})
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    res = {"grid": plan.table.n_ctas, "sms": int(props.multi_processor_count), "order": plan.table.order, "deadline_s": args.get("deadline_s", 2.0)}
    cnt_row = new_counters(case)
    X = case.fresh()
    t0 = time.time()
    try:
        plan.run(X, cnt_row, stream=current_stream_handle())
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 -- a device trap is a sticky launch failure; the elapsed time says it was the deadline, not a hang
        res.update(
            {"completed": False, "trapped": True, "elapsed_s": round(time.time() - t0, 3), "error": type(e).__name__ + ": " + str(e).splitlines()[0][:200]}
        )
        return _result(res)
    res.update({"completed": True, "trapped": False, "elapsed_s": round(time.time() - t0, 3), "counters_zero": int(cnt_row.abs().sum().item()) == 0})
    if args.get("compare", True):
        _, ref_x = run_reference(case)
        a, b = case.outputs(ref_x, fused=False), case.outputs(X, fused=True)
        res["differs_from_reference"] = [k for k in b if not same(a[k], b[k])]
    return _result(res)


def main(argv: List[str]) -> int:
    os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    mode, args = argv[0], json.loads(argv[1]) if len(argv) > 1 else {}
    if mode == "launches":
        return child_launches(args)
    if mode == "trap":
        return child_trap(args)
    print(f"unknown mode {mode!r}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
