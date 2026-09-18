# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin SM107 gathered-list sparse attention, d_qk = d_v = 512, BF16/FP16 -- a FORK of the dense
``cudnn/sdpa/fwd/kernels/sm107/prefill_d512_f16.py`` (the cga4x1 role-split pipeline).

What is the same (BYTE-IDENTICAL to the parent, by design): the barrier inventory and every init count,
the SMEM buffers (sizes, offsets, swizzles, the ``DESC_VERSION = 1`` descriptors), the TMEM layout, the MMA
descriptors / idescs, the sg1 correction warp, the O epilogue store loop, the P / alpha / stats DSMEM
ships and every end-of-kernel drain.  The pipeline is still

    sg0 (CTAs 0, 1):  TMA-LDG Q + K_ring; BMM1 = Q.K^T -> S_acc; streaming softmax; ship alpha + P + stats
    sg1 (CTAs 2, 3):  TMA-LDG V_ring; correction; BMM2 = P.V -> O; normalise + cast + TMA-STG O + LSE

Five deltas (plan ``dsv41_frost_unfused_PLAN.md`` section 1, (a)..(e)):

(1) ROWS = (token, head).  ``HEADS_PER_TILE = 64`` heads of ``TOKENS_PER_TILE = 2`` adjacent tokens fill
    one 128-row tile (the SM100 twin's PackGQA row mapping): lane ``tid`` <-> row ``tid`` <-> token
    ``tid // 64``, head ``tid % 64``.  The collective M = 256 of the CTA pair = 4 adjacent tokens x 64 heads =
    ONE CLUSTER ``c``; the pair leader holds tokens ``4c, 4c+1`` and the peer ``4c+2, 4c+3``.  Q and O move
    through PACKED TMA boxes ``(1, 2, 64, 64)`` on the ``[B, S, 64, 512]`` BSHD tensors; grid
    ``(NC * 4, 1, B)`` with ``NC = ceil(S / 4)``; the NATURAL decode is inlined (two lines).  Tokens
    ``>= S`` (``S % 4 != 0``) get zero Q rows (TMA OOB, bytes still counted), zero membership bits, O := 0 by
    SELECT, an OOB-clipped O store and no LSE write (``tok < S`` guard).  STATED ASSUMPTION: when
    ``S % 4 in {1, 2}`` the tail cluster's PEER Q box (tokens ``4c+2, 4c+3``) is FULLY OOB in the seq dim, and
    ``mb_tma_q_full``'s 262144 B balance rests on a fully-OOB TILED box still crediting its full
    ``complete_tx`` -- the same TMA property the ``-1`` gather rows rely on, validated for ``gather4`` by the
    W3a roundtrip micro but UNVALIDATED for the tiled 4-D box.  Pinned by the ``S % 4`` sentinel test at
    ``S in {5, 6}``, ``B > 1`` (a hang there names this row: S = 4, 7, 8 pass, S = 5, 6 exit 124).  Fallback
    if it ever fails: clamp ``q_row_base`` to ``S - TOKENS_PER_TILE`` on the peer so the box is at most
    partially OOB -- the clamped rows' membership bits are zero, so they are dead by construction (O := 0 by
    select, LSE guarded by ``tok < S``, the O box itself still fully OOB and clipped).
(2) K / V TILES COME FROM A UNION LIST.  The block's pre-pass (``kernels/union_lists.py``) hands every
    cluster an ascending multiset union of its four tokens' key lists as FLAT rows of the
    ``[B * N, 512]`` KV view (``-1``-padded), a per-slot membership bit table and a tile count
    ``n_tiles >= 1``.  K == V is one tensor and ONE 2-D tensor map (box ``(1, 64)``, s128b); every K / V
    tile is assembled by ``tma_gather4`` (four rows x one 128-B box-row per issue, 512 B) into the UNCHANGED
    ``SmemTile`` layouts -- the 128-B swizzle is a function of the SMEM address, and a quad of consecutive
    rows lands at a 512-B-aligned offset inside a 1024-B-aligned sub-box, so the tiled load's layout is
    reproduced exactly (validated by the W3a roundtrip micro).  A ``-1`` / sanitised id is a TMA-OOB row:
    ZERO-FILLED, bytes counted, so the transaction counts are the parent's.
(3) MASK = MEMBERSHIP.  Each softmax lane loads its 16-B bit slot ``union_bits[b, c, t, j, 0:4]`` at the top
    of every KV iteration and ``apply_membership_chunk`` writes a TRUE ``-inf`` into every union column that
    is not one of its own token's keys.  ONE select after the raw row max keeps the finite ``-3.4e38``
    marker the online softmax is written against: a dead (fully masked) tile-row contributes P = 0 with
    alpha = 1 for a live row and alpha = 0 on a still-dead one, exactly as the parent's first iteration.
(4) COUNT-BASED EMPTINESS.  ``kv_left = 0``, ``kv_right = union_ntiles[b, c]`` (never 0) at every bounds
    site (P14: one helper, the same ``(b, c)`` in every role); ``_row_empty = final_ell == 0`` is now EXACT
    (masked P are exactly 0) and the parent's geometric empty-range block is gone.
(5) PER-ROW SINK + LSE.  ``sink_logit = sinks[row_head]``; the fold gains one select so ``sink = +inf``
    cannot form ``exp(inf - inf)``; ``lse[b, row_head, tok]`` (natural log, SINK INCLUDED, ``= sink`` on a
    keyless row with a sink, ``-inf`` without one).

------------------------------------------------------------------------------------------------------------
Warp map (12 warps = 384 threads; the parent has 8 = 256) and the ISSUING-WARP LEDGER
------------------------------------------------------------------------------------------------------------
M3 (c09, locked 2376 MHz, 2026-09-17): ONE issuing warp sustains one gather4 per 36.7 cycles = 15.7 ns
(sorted == random rows, grid 1 == 212, depth 1 == 2 -> pure ISSUE serialisation: R2UR + UTMALDG per issue).
A stage is 128 issues per CTA (64 KiB) and the per-tile MMA floor is 0.862 us = 16 cycles per issue, so one
issuer is 2.3x too slow and THREE are needed.  Every warp that consumes the tile id to issue gathers must
credit the scheduler ring (``read_tile_id_arrive``), so the arriver count moves with the issuer count.

    warp  0..3   compute   sg0 softmax / sg1 correction+epilogue   SOFTMAX_REGS = 240    [unchanged]
    warp  4      MMA       sg0 BMM1 / sg1 BMM2 (leader), sg1 P12 forwarder (non-leader)  AUX_REGS = 64
    warp  5      TMA-LDG   sg0: Q packed box + K_ring expect_tx ARMS;  sg1: V_ring ARMS  64  [no longer
                           issues K / V bytes -- see "arm vs issue" below]
    warp  6      TMA-STG   sg1 O store + mb_tma_o_empty; sg0 scheduler spin only         64  [unchanged]
    warp  7      scheduler CLC try_cancel loop                                            64  [unchanged]
                           (the parent's config says 40 for these four; under the 384-thread launch cap the
                           split is REAL and the MMA body spills into its k-loop at 40 -- see AUX_REGS)
    warp  8..10  GATHER    issuer w = warp - 8: its quad range of every K (sg0) / V (sg1) stage,
                           ``read_tile_id_arrive`` per tile, ring drain at exit           40  [NEW]
    warp  11     spare     ``setmaxnreg.dec`` only, exits after init.  Exists so warps 8..11 form ONE
                           COMPLETE warpgroup: ``setmaxnreg.sync.aligned`` is warpgroup-collective, and the
                           parent's "MMA == TMALDG == TMASTG == SCHED regs" equality is that same rule for
                           warps 4..7.                                          GATHER_REGS = 40  [NEW]

Why extra warps and not the parent's idle slots: the sg0 TMA-STG warp is idle but the sg1 one stores O at
every tile end (its share of the NEXT tile's first stage would wait behind the store); the MMA warp is quiet
only on the NON-leader CTA of each pair (the leaders run BMM1 / BMM2) -- neither gives THREE issuers on EVERY
CTA.  Three dedicated gather warps do, identically on all four CTAs, and leave the O-store and the P12
forwarder paths untouched.  The DSL emits ``nvvm.reqntid`` from the static block size, so ptxas caps the
kernel at 65536 / 384 = 170 -> 168 registers/thread before ``setmaxnreg`` (12 x 168 x 32 = 64512 <= 65536;
warps 4..7 release 4 x 104 x 32, warps 8..11 release 4 x 128 x 32, the softmax group takes 4 x 72 x 32 =
9216).  ``cuobjdump -res-usage`` on the sm_107a cubin is the check (REG = 168, USETMAXREG present).

Issue split per stage per CTA (``ISSUERS_PER_CTA = 3``; quads = 4 consecutive union rows; 512 B per issue):

    K (sg0 CTA p): union rows [64p, 64p+64) of tile t x d [0, 512) = 16 quads x 8 boxes = 128 issues
        issuer 0: quads [0, 5)   -> 40 issues;  issuer 1: [5, 10) -> 40;  issuer 2: [10, 16) -> 48
        ids at union_ids[b, c, 128 t + 64 p + 4 g .. +4];  dst = sK[s] + i * 4096 + 256 g elems;  col = 64 i
    V (sg1 CTA p): union rows [0, 128) x d [256p, 256p+256) = 32 quads x 4 boxes = 128 issues
        issuer 0: quads [0, 10) -> 40 issues;  issuer 1: [10, 21) -> 44;  issuer 2: [21, 32) -> 44
        ids at union_ids[b, c, 128 t + 4 g .. +4];  dst = sV[s] + i * 8192 + 256 g elems;  col = 256 p + 64 i

Per issuing warp per stage: lanes ``0 .. n_quads-1`` each ``ld.global.nc.v4.s32`` their quad's ids BEFORE
the ``_empty`` wait (latency hidden under back-pressure); after the wait the four ids are shuffled to the
elected lane quad by quad (shuffles OUTSIDE the elect branch) and that lane issues the quad's boxes.

ARM vs ISSUE.  The leader's TMA-LDG lane arms ``expect_tx`` (131072 B, unchanged) after its OWN ``_empty``
wait; the gather warps issue after THEIR ``_empty`` waits.  The arm may land after some bytes: an mbarrier's
tx-count may run negative (PTX: valid range +-(2^20 - 1)) and the phase cannot complete before the single
pending arrival -- which is the arm itself.  The parent already relies on this (the non-leader CTA's tiled
TMA issues with no arm of its own, bytes routed to the leader by the bit-24 clear).  MEASURED SAFE on Rubin
(fractal-ts2-128, 2026-09-17): one issuing warp per CTA -- every work-item boundary lands ALL 131072 B before
the arm -- ran 205 second-or-later work items at ``n_tiles = 5`` bitwise-repeatable and exact.

S_acc READ COMPLETION (the defect that failed the first validation; symptom -> cause -> fix so nobody re-bisects
it).  Symptom: on ~60 % of the 2nd+ work items at ``n_tiles = 5`` (7 % at 3, none at ``n_tiles <= 2``, first
work items ALWAYS exact, two launches never bitwise) O and LSE were off on rows of every token, LSE
consistently LOW, and attribution lists (one token per issuer's quad range) showed the mis-read columns were
EXACTLY union columns 64..127 -- the PEER CTA's gathered rows -- on both CTAs' tokens.  Every K-path suspect
was then eliminated by variants on the node: per-issuer ``expect_tx`` arms (49/256 bad), a ``.shared::cluster``
gather destination (121), LOCAL mbars + P12 forward (108), no L2 hint (118), tiled 8-KiB boxes from three warps
(66) and from ONE warp -- the parent's exact K load inside this body (53); peer-side ``nanosleep`` 0.25 / 0.5 /
1 us (~110) vs 4 us (clean, because the pipeline turned gather-bound), one issuing warp (clean, gather-bound).
The SASS named it: in the softmax the ``mb_s_acc_empty`` arrive (``USYNCS.ARRIVE`` on the mapa'd leader
address) sat BETWEEN the two ``LDTM.x64`` of S_acc -- chunk 0 (columns 0..63) before it, chunk 1 (columns
64..127) AFTER it.  ``tcgen05.ld`` is asynchronous and this arm (inherited from the parent's MASKED softmax arm)
never issued the PTX-mandated ``tcgen05.wait::ld``; the arrive is relaxed and has no data dependency on the
loads, so the compiler was free to sink the second load past it.  Whenever the MMA is the bottleneck it is
parked on ``s_acc_empty`` and fires ``MMA(u+2)`` the instant the arrive lands, overwriting ``S_acc[parity]``
under the still-pending read of chunk 1: "peer rows" were never the peer's data, they were the second half
of S.  Fix: ``nvvm.tcgen05_wait(kind=LOAD)`` right after the two loads (``_sg0_softmax_kv_iter``); both
``LDTM.x64`` now precede the arrive in SASS and every probe is bitwise clean (frost-kernels.md section 3 is
the rule; the SM100 twin ``sm100/prefill_d512_f16.py:757`` has the wait, the three ``sm107/prefill_d512_*``
MASKED arms do not -- their dense default path uses the fused inline-asm ``tcgen05.ld.red`` whose side-effect
ordering keeps the loads ahead by accident).  Detector without a GPU: ``nvdisasm`` the cubin and check that
every ``LDTM`` of a TMEM slot precedes the ``USYNCS.ARRIVE`` that publishes the slot.

------------------------------------------------------------------------------------------------------------
Barrier-table DELTA (everything not listed: init, producer kind, scope, bootstrap and drain UNCHANGED --
the full table is the ``Bars`` comment block below, copied from the parent)
------------------------------------------------------------------------------------------------------------
| barrier               | init      | change                                     | lane arithmetic after the fork                      |
|-----------------------|-----------|--------------------------------------------|-----------------------------------------------------|
| mb_read_tile_id[2]    | 25 -> 37  | + 3 gather warps x 4 CTAs                   | compute 4x4=16 + TMALDG 1x4=4 + GATHER 3x4=12 +     |
|                       |           |                                            | sg0 MMA leader 1 + sg1 MMA 2 + sg1 TMASTG 2 = 37;   |
|                       |           |                                            | read_tile_id_arrive = ONE lane per target CTA copy  |
| mb_tma_q_full         | 1         | Q box (1, 2, 64, 64), coords (0, 0, 2 q, b) | 1 arrive (leader elect) + 262144 B (2 CTAs x 8 x 16 KiB); |
|                       |           |                                            | the peer's box is FULLY OOB on the tail cluster at   |
|                       |           |                                            | S % 4 in {1, 2}: full credit ASSUMED (delta (1))     |
| mb_tma_k_full[2]      | 1         | PRODUCER INSTRUCTION: 2 CTAs x 3 warps x    | 1 arrive (TMA-LDG leader elect, expect_tx 131072) + |
|                       |           | {40,40,48} gather4 x 512 B, bit-24 routed  | 2 x 128 x 512 = 131072 B on the leader's copy       |
| mb_tma_v_full[2]      | 1         | same, {40,44,44} issues                    | same                                                |
| mb_tma_k/v_empty[2]   | 1         | WAITERS: TMA-LDG + 3 gather warps per CTA  | producer unchanged (1 multicast commit, pred=elect) |
|                       |           | (waits do not count); each drains at exit  |                                                     |
| mb_empty_mainloop     | 2         | unreachable (n_tiles >= 1, CLAMPED in      | --                                                  |
|                       |           | _load_n_tiles); init kept                  |                                                     |
| every other barrier   |           | none                                       | as the parent                                       |

SMEM-table DELTA: NONE.  Only the WRITER of ``sKV_raw`` changes (tiled TMA -> gather4 into the same 1024-
aligned sub-boxes); sizes, offsets, swizzles, budgets, ``DESC_VERSION = 1`` (``sP_xfer_raw`` at 262144) as
the parent.  Index / membership data goes GMEM -> registers: no new buffer, no new mbarrier.  TMEM unchanged.

Invariant for the barrier inspector: ``union_ntiles[b, c]`` is read by ONE helper (``_load_n_tiles``) at the
tile-start and payload sites of every role that loops over KV -- compute, MMA leader, sg1 MMA forwarder,
TMA-LDG, the three gather warps -- with the same ``(b, c)``; TMA-STG never computes bounds.  The helper
carries the two SELECT guards (index forced to 0 on the terminal payload whose ctaid words are unspecified;
``n_tiles`` clamped to >= 1), so every role still advances identically on every path (P14).

Loaded by the block through ``cudnn.frost.template_loader.load_template(path, SparseAttentionD512Params(...),
tag=...)``; the plain-import default keeps the module usable standalone.  Outside ``engines.py`` /
``graph.plans`` for v1.
"""

import dataclasses as _dc
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass.experimental.primitives import VoteSync, vote_sync
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    Scope,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) -- still used for sched.mb_* (Sched not in Bars).
    wait,
)
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import apply_membership_chunk
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.pointwise import row_max_reduction, row_reduction_pair, vec_scale_pair
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.scheduler import (
    SCHED_NATURAL,
    Sched,
    read_clc_payload,
    read_tile_id_arrive,
    scheduler_warp_loop,
)
from cudnn.frost.tile_dsl.tma import (
    TMA_L2_EVICT_LAST,
    cp_async_bulk_shared_cluster_shared_cta,
    ld_global,
    ldg_int32x4,
    opaque_i64,
    tma_gather4,
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
)
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.mqa_sparse_attention_block.kernels.union_lists import (
    TILE_ROWS as _UL_TILE_ROWS,
    U_MAX_COLS_LIMIT as _UL_U_MAX_COLS_LIMIT,
    TOKENS_PER_CLUSTER as _UL_TOKENS_PER_CLUSTER,
    WORDS_PER_SLOT as _UL_WORDS_PER_SLOT,
)
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d512


# ----------------------------------------------------------------------------
# Template parameters -- the loader injects FROST_TEMPLATE_PARAMS; the default
# keeps a plain `import` usable as a standalone driver.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SparseAttentionD512Params:
    """Compile-time record of this fork (frozen + hashable: the template_loader cache key).

    ``dtype_qkv``      2 = BF16, 3 = FP16 (the parent's vocabulary; O inherits it).
    ``has_sink``       fold ``sinks[row_head]`` into the denominator (and the LSE).
    ``heads_per_tile`` heads packed per 128-row tile; 64 only (TP = 2 -> 32 heads -> 8 tokens per cluster
                       would need a different union geometry: refused, documented in the plan).
    ``u_max_tiles``    the pre-pass's tile capacity (``U_MAX = 128 * u_max_tiles`` ids per cluster).
    """

    dtype_qkv: int = 2
    has_sink: bool = True
    heads_per_tile: int = 64
    u_max_tiles: int = 20


PARAMS: SparseAttentionD512Params = globals().get("FROST_TEMPLATE_PARAMS", SparseAttentionD512Params())

# The shared d512 config with the PackGQA row mapping stamped; every other knob is the parent's default.
_BASE_CFG, _TMA = make_cfg_d512(
    TemplateParams(
        dtype_qkv=PARAMS.dtype_qkv,
        has_sink=PARAMS.has_sink,
        pack_gqa=True,
        qh_per_kh=PARAMS.heads_per_tile,
        cta_mma=2,
        sched_policy=SCHED_NATURAL,
    )
)

# ----------------------------------------------------------------------------
# Warp population DELTA: three dedicated gather-issuing warps + one spare that
# completes their warpgroup (see the module docstring's ledger).
# ----------------------------------------------------------------------------
N_GATHER_WARPS = 3  # issuers per CTA beyond the arming TMA-LDG warp -- sized by M3 (2.3x -> 3)
GATHER_WARP_BASE = _BASE_CFG.TOTAL_WARPS  # 8: the parent's 8 warps keep their ids
# setmaxnreg.sync.aligned is WARPGROUP-collective (4 aligned warps, same count): pad warps 8.. to a full group.
N_SPARE_WARPS = (-N_GATHER_WARPS) % 4
TOTAL_WARPS = GATHER_WARP_BASE + N_GATHER_WARPS + N_SPARE_WARPS  # 12
# Every warp that consumes the tile id to issue gathers credits the scheduler ring (scheduler.py refill gate):
# the parent's 25 (validated against its 8-warp body in config_sm107) + one per gather warp per CTA.
READ_TILE_ARRIVERS = _BASE_CFG.READ_TILE_ARRIVERS + N_GATHER_WARPS * _BASE_CFG.CGA_M * _BASE_CFG.CGA_N  # 25 + 12 = 37

# Register split.  The DSL emits nvvm.reqntid from the static block size, so ptxas caps the kernel at
# 65536 / 384 = 170 -> 168 registers/thread and the setmaxnreg split becomes REAL (the 256-thread parent
# compiles at REG=203 with NO USETMAXREG: ptxas drops the split when the natural allocation already fits,
# so its "40-register MMA warp" never actually ran at 40).  At a real 40 the MMA body -- four 64-bit
# operand descriptors, two idescs, eight pipeline states -- spills 43 STL/LDL INTO THE k-LOOP (LDL.64
# between UTCHMMA issues, sm_107a cubin 2026-09-17); 64 clears it.  setmaxnreg is warpgroup-collective, so
# warps 4..7 (MMA, TMA-LDG, TMA-STG, scheduler) share ONE count and warps 8..11 (gather + spare) another.
AUX_REGS = 64  # warps 4..7 (was the shared config's OTHER_REGS = 40)
GATHER_REGS = 40  # warps 8..11: the gather issue loop fits (0 STL/LDL at 40)
LAUNCH_REGS = ((65536 // (TOTAL_WARPS * 32)) // 8) * 8  # 168: what every warp holds at launch, before the split

CFG = _dc.replace(
    _BASE_CFG,
    TOTAL_WARPS=TOTAL_WARPS,
    THREADS_PER_CTA=TOTAL_WARPS * 32,
    READ_TILE_ARRIVERS=READ_TILE_ARRIVERS,
    OTHER_REGS=AUX_REGS,
    MMA_REGS=AUX_REGS,
)
Cfg = type(CFG)

# === Row mapping (the SM100 twin's PackGQA arm) ===
HEADS_PER_TILE = CFG.QH_PER_KH  # 64
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE  # 2
TOKENS_PER_CLUSTER = TOKENS_PER_TILE * CFG.CTA_MMA  # 4 -- the pre-pass's cluster
U_MAX_TILES = PARAMS.u_max_tiles
U_MAX = _UL_TILE_ROWS * U_MAX_TILES  # ids per cluster list
# Membership bit table: one tile = TOKENS_PER_CLUSTER slots x WORDS_PER_SLOT words = one 64-B line.
MEMBERSHIP_SLOT_BYTES = _UL_WORDS_PER_SLOT * 4  # 16
MEMBERSHIP_TILE_BYTES = TOKENS_PER_CLUSTER * MEMBERSHIP_SLOT_BYTES  # 64


def _validate_sparse_cfg(cfg, params: SparseAttentionD512Params) -> None:
    """The fork's structural contract, raised as ``ValueError`` at template load (never a module assert).

    Every predicate is a body fact: the config the shared ``make_cfg_d512`` produced must agree with the
    geometry this file hard-wires (row mapping, gather split, membership words, the arriver count).
    """
    cga = cfg.CGA_M * cfg.CGA_N
    derived_arrivers = (
        (cfg.SOFTMAX_WG_WARPS + 1 + N_GATHER_WARPS) * cga  # compute + TMA-LDG + gather warps on every CTA
        + cga // (2 * cfg.CTA_MMA)  # sg0 MMA leader
        + cga // 2  # sg1 MMA (leader + P12 forwarder)
        + cfg.CTA_MMA * cfg.CGA_N  # sg1 TMA-STG
    )
    heads = params.heads_per_tile
    checks = [
        (cfg.PACK_GQA == 1, "PACK_GQA must be 1 (rows are (token, head))"),
        (heads == 64 and cfg.QH_PER_KH == heads, f"heads_per_tile must be 64 (TP=2 -> 32 heads is refused), got {heads} / QH_PER_KH={cfg.QH_PER_KH}"),
        (cfg.TILE_M % heads == 0 and cfg.TILE_M // heads == 2, f"TILE_M={cfg.TILE_M} must pack exactly 2 tokens of {heads} heads"),
        (cfg.TILE_M // heads * cfg.CTA_MMA == _UL_TOKENS_PER_CLUSTER, "cluster tokens must equal union_lists.TOKENS_PER_CLUSTER (4)"),
        (cfg.MASK_FLAGS == 0, "band / padding masks are not part of this kernel (membership bits are the mask)"),
        (cfg.SCHEDULER_POLICY == SCHED_NATURAL, "SCHEDULER_POLICY must be NATURAL (the decode is inlined)"),
        (cfg.STAGES_KV == 2 and cfg.XFER_STAGES == 2, f"STAGES_KV / XFER_STAGES must be 2 / 2 (got {cfg.STAGES_KV} / {cfg.XFER_STAGES})"),
        (cfg.TILE_N == 128 and cfg.TILE_N == _UL_TILE_ROWS, f"TILE_N must be 128 = union_lists.TILE_ROWS (got {cfg.TILE_N})"),
        (cfg.TILE_K == 512 and cfg.TILE_O == 512, f"TILE_K / TILE_O must be 512 (got {cfg.TILE_K} / {cfg.TILE_O})"),
        (cfg.DTYPE_QKV in (2, 3), f"dtype_qkv must be 2=BF16 or 3=FP16 (got {cfg.DTYPE_QKV})"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "O inherits the QKV dtype"),
        (cfg.THD_VARLEN == 0 and cfg.SEQ_KV_LENS_PRESENT == 0 and cfg.SEQ_Q_LENS_PRESENT == 0, "THD / padded-KV / padded-Q arms are deleted in this fork"),
        (cfg.HAS_SINK == int(params.has_sink), "HAS_SINK must follow params.has_sink"),
        (cfg.STATS_LOG2 == 0, "LSE is natural-log only"),
        (
            1 <= params.u_max_tiles and params.u_max_tiles * _UL_TILE_ROWS < _UL_U_MAX_COLS_LIMIT,
            f"u_max_tiles * 128 must stay below {_UL_U_MAX_COLS_LIMIT} (got {params.u_max_tiles})",
        ),
        (cfg.CGA_M == 4 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2 and cfg.TILES_Q == 1, "the role-split geometry is cga4x1 / CTA_MMA=2 / TILES_Q=1"),
        (cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == cfg.V_SWZ_BYTES == cfg.O_SWZ_BYTES == 128, "K and V share ONE s128b tensor map; every SMEM tile is Swz128B"),
        (cfg.TOTAL_WARPS == TOTAL_WARPS and cfg.THREADS_PER_CTA == TOTAL_WARPS * 32, "warp population must be the parent's 8 + gather + spare"),
        # The role dispatch's `else` arm IS the gather warpgroup: every warp id below GATHER_WARP_BASE must be
        # claimed by a named role, or a drifted role id runs the gather group's warpgroup-collective
        # setmaxnreg.dec and never fires its read_tile_id_arrive (init 37 unreachable -> hang at every shape).
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and sorted((cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID)) == list(range(cfg.SOFTMAX_WG_WARPS, GATHER_WARP_BASE)),
            "role warps must occupy 0..GATHER_WARP_BASE-1 (softmax 0..3, then MMA / TMA-LDG / TMA-STG / scheduler) so the else-branch dispatch is exactly the gather warpgroup",
        ),
        (N_GATHER_WARPS >= 1 and (N_GATHER_WARPS + N_SPARE_WARPS) % 4 == 0, "gather + spare warps must complete a warpgroup (setmaxnreg.sync.aligned)"),
        (
            cfg.READ_TILE_ARRIVERS == derived_arrivers == READ_TILE_ARRIVERS,
            f"READ_TILE_ARRIVERS must be {derived_arrivers} for this body (got {cfg.READ_TILE_ARRIVERS})",
        ),
        (_UL_WORDS_PER_SLOT * 32 == cfg.TILE_N, "one membership slot (4 words) must cover one 128-column tile"),
        # Register split (frost-kernels.md section 2, generalised to three warpgroups): every count 8-aligned in
        # 24..256; the launch allocation fits the file; the split fits it too; INCREASE / DECREASE go the right way.
        (all(r % 8 == 0 and 24 <= r <= 256 for r in (cfg.SOFTMAX_REGS, cfg.OTHER_REGS, GATHER_REGS)), "register counts must be 8-aligned in 24..256"),
        (cfg.MMA_REGS == cfg.OTHER_REGS == AUX_REGS, "warps 4..7 form one warpgroup: MMA / TMA-LDG / TMA-STG / scheduler share one register count"),
        (TOTAL_WARPS * LAUNCH_REGS * 32 <= 65536, f"launch allocation {TOTAL_WARPS} x {LAUNCH_REGS} x 32 exceeds the 64K register file"),
        (
            cfg.SOFTMAX_WG_WARPS * cfg.SOFTMAX_REGS
            + (GATHER_WARP_BASE - cfg.SOFTMAX_WG_WARPS) * cfg.OTHER_REGS
            + (N_GATHER_WARPS + N_SPARE_WARPS) * GATHER_REGS
            <= 65536 // 32,
            "the post-split register budget exceeds the 64K register file",
        ),
        (cfg.SOFTMAX_REGS >= LAUNCH_REGS >= max(cfg.OTHER_REGS, GATHER_REGS), "softmax INCREASEs from and the other roles DECREASE from the launch count"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"sparse_attention_d512: {msg}")


_validate_sparse_cfg(CFG, PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point (the parent's
# rationale, verbatim in spirit): a version-0 descriptor addresses 14 bits = 256 KiB, this flavor's slabs
# put the P transfer ring at EXACTLY 262144, and a wrapped descriptor multiplies the bottom of SMEM with no
# crash (accumulator exactly zero).  Never re-literal at a call site.
DESC_VERSION: int = 1
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O TMA box / store params follow O's swizzle (independent of V under cga2).
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

# ----------------------------------------------------------------------------
# Storage dtype dispatch -- folded at trace time on CFG.DTYPE_QKV (BF16 / FP16 only).
# ----------------------------------------------------------------------------
if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    P_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    P_STORAGE_DTYPE = cutlass.Float16
else:  # unreachable past _validate_sparse_cfg; kept so the dispatch cannot fall through silently
    raise ValueError(f"sparse_attention_d512: DTYPE_QKV={CFG.DTYPE_QKV} not supported (2=BF16, 3=FP16)")
MMA_KIND = nvvm.Tcgen05MMAKind.F16
OUT_STORAGE_DTYPE = STORAGE_DTYPE


# ----------------------------------------------------------------------------
# Derived constants -- the parent's, unchanged.
# ----------------------------------------------------------------------------
CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

# Per-CTA buffer element counts + collective TMA transaction byte counts.
qBufferElems = CFG.TILE_M * CFG.TILE_K  # 65536  -> 128 KiB @ BPE=2
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA  # 32768  ->  64 KiB
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA  # 32768  ->  64 KiB
oBufferElems = CFG.TILE_M * CFG.TILE_O  # 65536  -> 128 KiB @ BPE_O=2
pXferElems = CFG.TILE_M * CFG.TILE_N  # 16384  ->  32 KiB / xfer stage

# Per-stage xfer SMEM sizes (DSMEM peer-to-peer payloads).
pXferBytes = pXferElems * CFG.BPE  # 32 KiB / stage
alphaXferBytes = CFG.TILE_M * 4  # 512 B  / stage (FP32)
statsXferBytes = 2 * CFG.TILE_M * 4  # 1 KiB total (ell + max)

# Collective TMA transaction bytes (leader expect_tx under cga2) -- UNCHANGED: gather4 delivers the same
# bytes per stage as the tiled load (OOB rows are zero-filled AND counted).
qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

# Per-call BMM2 V SMEM advance (between the BMM2_LOOP_N_BLOCKS=2 BMM2 calls).
BMM2_V_NBLOCK_ADVANCE = CFG.TILE_N * (CFG.BMM2_N_PER_CALL // CFG.CTA_MMA) * CFG.BPE

# O store chunks (TMA-STG arrive granularity -- 128 B per chunk).
N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128  # 8 chunks @ d=512 f16

# ----------------------------------------------------------------------------
# Gather geometry (delta 2).  One issue = 4 consecutive union rows x one 128-B box-row.
# ----------------------------------------------------------------------------
GATHER_ROWS_PER_ISSUE = 4
GATHER_COLS_PER_ISSUE = TMA_QK_GRANU_ELEMS  # 64 elems = 128 B = one swizzle row; the KV tensor map's box is (1, 64)
GATHER_BYTES_PER_ISSUE = GATHER_ROWS_PER_ISSUE * GATHER_COLS_PER_ISSUE * CFG.BPE  # 512
ISSUERS_PER_CTA = N_GATHER_WARPS
# K (sg0 CTA p): union rows [64p, 64p+64) x d [0, 512) into sK[s]'s 8 sub-boxes of 64 rows x 128 B.
K_ROWS_PER_CTA = CFG.TILE_N // CFG.CTA_MMA  # 64
K_ROW_QUADS = K_ROWS_PER_CTA // GATHER_ROWS_PER_ISSUE  # 16
K_BOXES = CFG.TILE_K // GATHER_COLS_PER_ISSUE  # 8
K_SUBBOX_ELEMS = K_ROWS_PER_CTA * TMA_QK_GRANU_ELEMS  # 4096 == sK.tma_subtile_stride_elems
# V (sg1 CTA p): union rows [0, 128) x d [256p, 256p+256) into sV[s]'s 4 sub-boxes of 128 rows x 128 B.
V_ROW_QUADS = CFG.TILE_N // GATHER_ROWS_PER_ISSUE  # 32
V_BOXES = (CFG.TILE_O // CFG.CTA_MMA) // GATHER_COLS_PER_ISSUE  # 4
V_SUBBOX_ELEMS = CFG.TILE_N * TMA_VO_GRANU_ELEMS  # 8192 == sV.tma_subtile_stride_elems


def _quad_range(n_quads: int, issuer: int) -> tuple:
    """Issuer ``issuer``'s half-open quad range: the ``n_quads`` quads of a stage split as evenly as possible."""
    return (n_quads * issuer) // ISSUERS_PER_CTA, (n_quads * (issuer + 1)) // ISSUERS_PER_CTA


K_QUAD_RANGES = tuple(_quad_range(K_ROW_QUADS, w) for w in range(ISSUERS_PER_CTA))  # ((0,5),(5,10),(10,16))
V_QUAD_RANGES = tuple(_quad_range(V_ROW_QUADS, w) for w in range(ISSUERS_PER_CTA))  # ((0,10),(10,21),(21,32))
assert K_ROW_QUADS * K_BOXES * GATHER_BYTES_PER_ISSUE * CFG.CTA_MMA == kTmaTransactionBytes, "K gather issues must deliver exactly the collective K bytes"
assert V_ROW_QUADS * V_BOXES * GATHER_BYTES_PER_ISSUE * CFG.CTA_MMA == vTmaTransactionBytes, "V gather issues must deliver exactly the collective V bytes"
assert max(hi - lo for lo, hi in K_QUAD_RANGES + V_QUAD_RANGES) <= 32, "one lane per quad: an issuer's quad share must fit a warp"
assert TMA_QK_GRANU_ELEMS == TMA_VO_GRANU_ELEMS == GATHER_COLS_PER_ISSUE, "K and V share one (1, 64) box"

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

# ----------------------------------------------------------------------------
# Softmax-body constants -- module-level so the top-level @cute.jit helper
# `_sg0_softmax_kv_iter` can read them without a closure capture.
# ----------------------------------------------------------------------------
P_TMA_ITERS = (CFG.TILE_N * CFG.BPE) // CFG.Q_SWZ_BYTES  # 2
P_D_BLOCK = CFG.TILE_N // P_TMA_ITERS  # 64 elements / chunk
P_BLOCK_BYTES = CFG.TILE_M * P_D_BLOCK  # 8192 elements / TMA chunk
SOFTMAX_CHUNK = 64
SOFTMAX_N_CHUNKS_LOAD = CFG.TILE_N // SOFTMAX_CHUNK  # 2 @ TILE_N=128: chunk c masks with words 2c, 2c+1
assert SOFTMAX_N_CHUNKS_LOAD * 2 == _UL_WORDS_PER_SLOT, "two 32-bit membership words per 64-column chunk"
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
# Streaming-softmax constants.  NEG_INF_F32 is the FINITE "no max yet" marker the online softmax is
# written against; the membership mask writes a TRUE -inf and ONE select maps a dead tile-row onto it.
NEG_INF_F32 = cutlass.Float32(-3.4028235e38)
RESCALE_THRESHOLD_F32 = cutlass.Float32(CFG.RESCALE_THRESHOLD)


# ----------------------------------------------------------------------------
# Named arrival-count constants (P3: every init(...) references a name).
# ----------------------------------------------------------------------------
COMPUTE_LANES = CFG.SOFTMAX_WG_WARPS * 32  # 4 warps * 32 lanes = 128
SM_LANES_TOTAL = 2 * COMPUTE_LANES  # 256 -- both sg0 CTAs * 128 sm threads
TWO_LANES_TOTAL = 2  # both sg1 CTAs * 1 elect-one to leader
KV_EMPTY_ARRIVERS = CFG.CGA_N  # = 1 -- cga2 commit multicast counts as 1


# ----------------------------------------------------------------------------
# Compile-time budget checks -- Rubin SM107 caps (the parent's, unchanged).
# ----------------------------------------------------------------------------
_SG0_SMEM_DATA_KIB = (
    qBufferElems * CFG.BPE  # Q (sg0)
    + CFG.STAGES_KV * kBufferElems * CFG.BPE  # K_ring (sg0)
    + CFG.XFER_STAGES * pXferBytes  # P xfer (source side staging)
    + CFG.XFER_STAGES * alphaXferBytes  # alpha xfer (source side staging)
    + statsXferBytes  # stats xfer
) / 1024
_SG1_SMEM_DATA_KIB = (
    CFG.STAGES_KV * vBufferElems * CFG.BPE  # V_ring (sg1)
    + oBufferElems * CFG.BPE_O  # O (sg1)
    + CFG.XFER_STAGES * pXferBytes  # P xfer (sink side staging)
    + CFG.XFER_STAGES * alphaXferBytes  # alpha xfer (sink side staging)
    + statsXferBytes  # stats xfer
    + CFG.TILE_M * 4  # LSE staging (sg1)
) / 1024
assert _SG0_SMEM_DATA_KIB <= 323, f"sg0 SMEM data {_SG0_SMEM_DATA_KIB:.1f} KiB exceeds 323 KiB headroom under 327 KiB cap"
assert _SG1_SMEM_DATA_KIB <= 323, f"sg1 SMEM data {_SG1_SMEM_DATA_KIB:.1f} KiB exceeds 323 KiB headroom under 327 KiB cap"

_TMEM_SG0_COLS = CFG.XFER_STAGES * CFG.TILE_N  # 2 * 128 = 256 cols for S_acc parities
_TMEM_SG1_COLS = CFG.TILE_O  # 512 cols for O
assert _TMEM_SG0_COLS <= 576, f"sg0 S_acc TMEM overflow: {_TMEM_SG0_COLS} > 576"
assert _TMEM_SG1_COLS <= 576, f"sg1 O TMEM overflow: {_TMEM_SG1_COLS} > 576"


# ----------------------------------------------------------------------------
# Bars -- the parent's role-split inventory, UNCHANGED (fork per the per-pipeline
# fork pattern).  Full table; the DELTA column lives in the module docstring.
# ----------------------------------------------------------------------------
# Barrier table (cga4x1 / CTA_MMA=2 -- role-split, SM107)
# ============================================================================
# | Name                     | Stages           | Init                       | Producer / arrive site                          | Consumer / wait site                  | Scope / bootstrap                                   |
# |--------------------------|------------------|----------------------------|-------------------------------------------------|---------------------------------------|-----------------------------------------------------|
# | mb_tma_q_full            | 1                | ONE_LANE                   | sg0 TMA-LDG : arrive_expect_tx (Q bytes)        | sg0 MMA (BMM1)                        | cga2 leader-only                                    |
# | mb_tma_q_empty           | 1                | ONE_LANE                   | sg0 MMA : arrive_mma after last-iter BMM1 commit| sg0 TMA-LDG (next-tile Q load)        | leader-multicast; pre-arm via q_empty_state=1       |
# | mb_tma_k_full[s]         | STAGES_KV (=2)   | ONE_LANE                   | sg0 TMA-LDG : arrive_expect_tx (K bytes);       | sg0 MMA (BMM1)                        | cga2 leader-only                                    |
# |                          |                  |                            |   bytes: 3 gather warps x 2 CTAs, gather4       |                                       |                                                     |
# | mb_tma_k_empty[s]        | STAGES_KV        | KV_EMPTY_ARRIVERS (=1)     | sg0 MMA : arrive_mma after BMM1 commit          | sg0 TMA-LDG + 3 gather warps          | leader-multicast; pre-arm via PipelineState(0,1)    |
# | mb_tma_v_full[s]         | STAGES_KV        | ONE_LANE                   | sg1 TMA-LDG : arrive_expect_tx (V bytes);       | sg1 MMA (BMM2)                        | cga2 leader-only                                    |
# |                          |                  |                            |   bytes: 3 gather warps x 2 CTAs, gather4       |                                       |                                                     |
# | mb_tma_v_empty[s]        | STAGES_KV        | KV_EMPTY_ARRIVERS (=1)     | sg1 MMA : arrive_mma after BMM2 commit          | sg1 TMA-LDG + 3 gather warps          | leader-multicast; pre-arm via PipelineState(0,1)    |
# | mb_bmm1_done[p]          | XFER_STAGES (=2) | ONE_LANE                   | sg0 MMA : arrive_mma after BMM1 commit          | sg0 softmax                           | local; not bootstrapped                             |
# | mb_bmm2_done[p]          | XFER_STAGES      | ONE_LANE                   | sg1 MMA : arrive_mma after BMM2 commit          | sg1 corr (epilogue gate)              | local; not bootstrapped                             |
# | mb_bmm2_ready[p*2+c]     | XFER_STAGES*2(=4)| SM_LANES_TOTAL (=256)      | sg1 corr (all-thread) : arrive_on_peer->leader  | sg1 MMA leader (per-half-N-block)     | cga2 leader-waited                                  |
# | mb_s_acc_empty[p]        | XFER_STAGES      | SM_LANES_TOTAL (=256)      | sg0 softmax (all-thread) : arrive_on_peer->leader| sg0 MMA leader (S_acc TMEM free)     | cga2 leader-waited; pre-arm via PipelineState(0,1); |
# |                          |                  |                            |   AFTER tcgen05.wait::ld of BOTH S chunks        |                                       |   the arrive must follow the wait::ld (see doc)     |
# | mb_p_xfer_full[p]        | XFER_STAGES      | leader: CTA_MMA(=2)        | sg0 softmax bulk_copy->peer; sg1 peer->leader DSMEM| sg1 MMA leader (BMM2 P operand)    | P12; leader=2 / non-leader=1                        |
# |                          |                  | non-leader: ONE_LANE(=1)   |                                                  |                                       |                                                     |
# | mb_p_xfer_empty[p]       | XFER_STAGES      | ONE_LANE                   | sg1 MMA leader : arrive_mma multicast->sg0      | sg0 softmax (next P slot reuse)       | P13 MMA-commit multicast; pre-arm via PipelineState(0,1) |
# | mb_alpha_xfer_full[p]    | XFER_STAGES      | ONE_LANE                   | sg1 corr (local) : arrive_expect_tx (alpha)     | sg1 corr (apply alpha)                | local; not bootstrapped                             |
# | mb_alpha_xfer_empty[p]   | XFER_STAGES      | COMPUTE_LANES (=128)       | sg1 corr (all-thread) : arrive_on_peer->sg0     | sg0 softmax (next alpha slot reuse)   | local; pre-arm via PipelineState(0,1)               |
# | mb_stats_xfer_full       | 1                | ONE_LANE                   | sg1 corr (local) : arrive_expect_tx (ell+max)   | sg1 corr (final stats consume)        | local; not bootstrapped                             |
# | mb_stats_xfer_empty      | 1                | ONE_LANE                   | sg1 corr (elect) : arrive_on_peer->sg0          | sg0 softmax (next tile stats slot)    | local; pre-arm via stats_xfer_empty_phase=1         |
# | mb_tma_o_full[c]         | N_O_CHUNKS (=8)  | COMPUTE_LANES (=128)       | sg1 corr (all-thread) : arrive per chunk        | sg1 TMA-STG (per-chunk TMA)           | local; not bootstrapped                             |
# | mb_tma_o_empty           | 1                | ONE_WARP (=32)             | sg1 TMA-STG warp : arrive after STG drain       | sg1 corr (next-tile O staging reuse)  | local; pre-arm via PipelineState(0,1)               |
# | mb_empty_mainloop        | 1                | TWO_LANES_TOTAL (=2)       | UNREACHABLE in this fork (n_tiles >= 1); init kept| sg1 MMA leader (empty-kv branch)    | cga2 leader-waited                                  |
# | mb_tmem_dealloc          | 1                | ONE_LANE                   | both sgs compute (elect) : arrive_on_peer->partner| both sgs MMA (alloc/dealloc fence)   | local; not bootstrapped                             |
# | mb_read_tile_id[s]       | SCHEDULER_STAGES | READ_TILE_ARRIVERS (=37)   | one lane per target CTA from every tile-id READER| scheduler warp (slot refill gate)    | Sched (not in Bars); see the module docstring       |
# ============================================================================
class Bars(NamedTuple):
    """Role-split mbarrier inventory (Rubin SM107) -- the parent's, unchanged."""

    # ---- TMA Q/K/V handshakes (sg0: Q + K; sg1: V) ----------------------
    mb_tma_q_full: object  # [1]            sg0  TMA-LDG -> MMA
    mb_tma_q_empty: object  # [1]            sg0  MMA -> TMA-LDG (next-tile Q)
    mb_tma_k_full: object  # [STAGES_KV]    sg0  TMA-LDG arm + gather bytes -> MMA
    mb_tma_k_empty: object  # [STAGES_KV]    sg0  MMA -> TMA-LDG + gather warps (back-pressure)
    mb_tma_v_full: object  # [STAGES_KV]    sg1  TMA-LDG arm + gather bytes -> MMA
    mb_tma_v_empty: object  # [STAGES_KV]    sg1  MMA -> TMA-LDG + gather warps

    # ---- MMA -> softmax / corr handshakes -------------------------------
    mb_bmm1_done: object  # [XFER_STAGES]      sg0 MMA -> softmax
    mb_bmm2_done: object  # [XFER_STAGES]      sg1 MMA -> corr (epilogue gate)
    mb_bmm2_ready: object  # [XFER_STAGES * 2]  sg1 corr -> MMA (per half-N-block)
    mb_s_acc_empty: object  # [XFER_STAGES]      sg0 softmax -> MMA (S_acc TMEM free)

    # ---- DSMEM xfer rings (sg0 -> sg1) ----------------------------------
    mb_p_xfer_full: object  # [XFER_STAGES]   sg0 -> sg1 (P12 leader=2 / non-leader=1)
    mb_p_xfer_empty: object  # [XFER_STAGES]   sg1 leader MMA -> sg0 (P13 multicast)
    mb_alpha_xfer_full: object  # [XFER_STAGES]   sg0 softmax (local expect_tx) -> sg1 corr
    mb_alpha_xfer_empty: object  # [XFER_STAGES]   sg1 corr (all-thread DSMEM) -> sg0
    mb_stats_xfer_full: object  # [1]             sg1 corr (local expect_tx) one-shot per tile
    mb_stats_xfer_empty: object  # [1]             sg0 softmax (elect DSMEM) -> sg1

    # ---- sg1 TMA-STG handshakes -----------------------------------------
    mb_tma_o_full: object  # [N_O_CHUNKS]   sg1 corr -> TMA-STG (per-128B chunk)
    mb_tma_o_empty: object  # [1]            sg1 TMA-STG warp -> corr

    # ---- End-of-kernel teardown -----------------------------------------
    mb_empty_mainloop: object  # [1]            sg1 corr -> sg1 MMA leader (empty-kv branch; unreachable here)
    mb_tmem_dealloc: object  # [1]            both sgs compute -> cga2 partner MMA


# ----------------------------------------------------------------------------
# KernelTmemLayout -- unchanged.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelTmemLayout:
    """SM107 TMEM carves: sg0 = S_acc(XFER_STAGES parities); sg1 = O.  576-col cap, is_exclusive=True."""

    TOTAL_COLS: int = 576
    S_ACC_COLS: int = 128  # = CFG.TILE_N -- one parity slot
    S_ACC_PARITY0_OFF: int = 0
    S_ACC_PARITY1_OFF: int = 128
    O_OFF: int = 0
    O_COLS: int = 512  # = CFG.TILE_O


LAYOUT = KernelTmemLayout()
assert CFG.XFER_STAGES * LAYOUT.S_ACC_COLS <= LAYOUT.TOTAL_COLS, f"sg0 TMEM overflow: {CFG.XFER_STAGES * LAYOUT.S_ACC_COLS} > {LAYOUT.TOTAL_COLS}"
assert LAYOUT.O_OFF + LAYOUT.O_COLS <= LAYOUT.TOTAL_COLS, f"sg1 TMEM overflow: {LAYOUT.O_OFF + LAYOUT.O_COLS} > {LAYOUT.TOTAL_COLS}"


# ----------------------------------------------------------------------------
# Bars factory -- unchanged (mb_p_xfer_full keeps its RUNTIME P12 init count via
# override_count in the kernel's init loop).
# ----------------------------------------------------------------------------
def _make_bars(CFG, N_O_CHUNKS: int):
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        # TMA Q/K/V
        mb_tma_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_q_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tma_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tma_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        # MMA -> softmax / corr
        mb_bmm1_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS),
            stages=CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS,
            init_count=SM_LANES_TOTAL,
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_s_acc_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=SM_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        # DSMEM xfer rings -- mb_p_xfer_full has a RUNTIME init count (P12): leader = CTA_MMA, non-leader = 1.
        mb_p_xfer_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.CTA_MMA, producer=Producer.TMA_LOAD),
        mb_p_xfer_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_alpha_xfer_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_alpha_xfer_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_stats_xfer_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_stats_xfer_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        # sg1 TMA-STG
        mb_tma_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_tma_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        # Teardown
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=TWO_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
    )


# ----------------------------------------------------------------------------
# SMEM swizzle / layout enums -- unchanged (every tile is Swz128B at d=512 f16).
# ----------------------------------------------------------------------------
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q
SMEM_LAYOUT_P = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

# BMM2 manual k-step iters (derived, never a literal).
NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2


# ----------------------------------------------------------------------------
# Tile decode -- the NATURAL / SPLIT_PIPELINE=1 arm of _common_blackwell.make_sdpa_helpers,
# inlined (two lines; the factory's bare call is what made LPT write nothing on the d512 line,
# and this kernel pins NATURAL).  Cluster c = q_super_idx // CTA_MMA is the same on all 4 CTAs.
# ----------------------------------------------------------------------------
@cute.jit
def _decode_initial(bidx, bidy, bidz, cta_in_pair):
    q_super_idx = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
    return q_super_idx, bidy, bidz


@cute.jit
def _decode_payload(t0, t1, cta_in_pair):
    q_super_idx = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(CFG.CTA_MMA) + cta_in_pair
    head = t1 & cutlass.Int32(0xFFFF)
    batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
    return q_super_idx, head, batch


@cute.jit
def _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid):
    """``union_ntiles[b, c]`` -- THE bounds read of every KV-looping role (P14: one helper, one address per
    ``(b, c)``, so every role runs the same trip count).  Two guards, both SELECTs (branchless, warp-uniform,
    no phase / ring bookkeeping touched):

    (a) ``is_valid``: on the TERMINAL scheduler payload (``is_valid_tile == 0``) the ctaid words are
        UNSPECIFIED -- the PTX / DSL contract defines ``clusterlaunchcontrol`` payload fields only when the
        cancel succeeded (``cute/arch/clc.py``), and FROST's ``read_clc_payload`` hands back the raw words.
        The dense parent's ``_bounds_for_tile`` was arithmetic-only, so it never dereferenced them; this
        helper does, so the index is forced to 0 (always inside ``union_ntiles``) and the dead result is
        discarded by the exiting ``while``.  Without it a garbage ``t0`` is a wild global read up to
        2**31 x 4 B past the tensor -> ``cudaErrorIllegalAddress`` on an otherwise-correct launch.
    (b) ``n_tiles < 1``: a ``0`` (pre-pass contract violation, or a hand-built list) would run the sg0 softmax
        and sg1 correction for zero iterations while the sg1 compute path still waits ``mb_bmm2_done`` /
        ``mb_stats_xfer_full`` and TMA-STG waits ``mb_tma_o_full`` with no MMA to fire them -> exit 124 for
        the whole launch.  Clamped to ONE zero-MMA tile: tile 0's ids are ``-1``-padded (TMA-OOB, zero-filled,
        bytes counted) and the slot bits are zero, so every row is dead -> O := 0 by select, LSE = sink or
        ``-inf`` -- exactly the all ``-1`` cluster of delta (1).  The pre-pass contract stays the primary guard.
    """
    cluster_idx = q_super_idx // cutlass.Int32(CFG.CTA_MMA)
    idx = batch_idx * n_clusters + cluster_idx
    idx = cutlass.Int32(arith.select((is_valid != cutlass.Int32(0)).ir_value(), idx.ir_value(), cutlass.Int32(0).ir_value()))
    n_tiles = ld_global(ntiles_base + idx.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
    n_tiles = cutlass.Int32(arith.select((n_tiles < cutlass.Int32(1)).ir_value(), cutlass.Int32(1).ir_value(), n_tiles.ir_value()))
    return cute.arch.make_warp_uniform(n_tiles)


# ============================================================================
# Kernel entry.
# ============================================================================
@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_kv_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    union_ids: cute.Tensor,
    union_bits: cute.Tensor,
    union_ntiles: cute.Tensor,
    seqlen_q: cutlass.Int32,
    n_clusters: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # ------------------------------------------------------------------
    # SMEM allocations -- the parent's, unchanged (Q (sg0) | O (sg1) and
    # K_ring (sg0) | V_ring (sg1) alias at the same offsets; each physical CTA
    # runs ONE role).
    # ------------------------------------------------------------------
    _QO_ELEMS = qBufferElems if qBufferElems >= oBufferElems else oBufferElems
    _KV_ELEMS = CFG.STAGES_KV * kBufferElems if kBufferElems >= vBufferElems else CFG.STAGES_KV * vBufferElems
    sQO_raw = cutlass.Array(STORAGE_DTYPE, _QO_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, _KV_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ_raw = sQO_raw  # sg0 view
    sO_raw = sQO_raw  # sg1 view
    sK_raw = sKV_raw  # sg0 view
    sV_raw = sKV_raw  # sg1 view

    # P xfer (sg0 -> sg1) SMEM -- XFER_STAGES rings, kept resident.  Sits at EXACTLY 262144: DESC_VERSION.
    sP_xfer_raw = cutlass.Array(P_STORAGE_DTYPE, CFG.XFER_STAGES * pXferElems, alignment=128, space=cutlass.AddressSpace.smem)
    # alpha xfer (sg0 -> sg1) SMEM.
    sAlpha_xfer_raw = cutlass.Array(cutlass.Float32, CFG.XFER_STAGES * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    # stats xfer (sg0 -> sg1) SMEM -- 1-shot per tile, holds ell + max (2 * TILE_M floats).
    sStats_xfer_raw = cutlass.Array(cutlass.Float32, 2 * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    # LSE staging on sg1 (never written / read; kept so the SMEM table stays the parent's).
    sLSE_raw = cutlass.Array(cutlass.Float32, CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)

    # Typed SmemTile handles.  EVERY tile takes the module-level DESC_VERSION (= 1 here).
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=K_SUBBOX_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=V_SUBBOX_ELEMS,
        desc_version=DESC_VERSION,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
        desc_version=DESC_VERSION,
    )

    # ------------------------------------------------------------------
    # Bars allocation -- per the barrier table above.
    # ------------------------------------------------------------------
    bars = _make_bars(CFG, N_O_CHUNKS)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # tile_id_smem stride 8 Int32/stage (32 B per stage; 16 B payload + 16 B padding).
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    # ------------------------------------------------------------------
    # cga2 / role-split identity -- unchanged.
    # ------------------------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    # sg0_mcast_mask = 0x3 -- both sg0 CTAs (used by sg1 leader's P13 multicast on mb_p_xfer_empty).
    sg0_mcast_mask = cutlass.Int32(0x3)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Sub-group identity (sg0 = CTAs 0, 1; sg1 = CTAs 2, 3).
    sg_id = cta_id_x // cutlass.Int32(CFG.CTA_MMA)
    is_sg0 = sg_id == cutlass.Int32(0)
    is_sg1 = sg_id == cutlass.Int32(1)
    # Cross-sg peer = this CTA XOR CTA_MMA (CTAs 0<->2, 1<->3).
    cross_sg_peer = cta_id_x ^ cutlass.Int32(CFG.CTA_MMA)

    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # ------------------------------------------------------------------
    # Mbar init -- Phase 1 of cluster init (P4): ONE warp, ONE lane.  Unchanged
    # except the scheduler ring's arriver count (READ_TILE_ARRIVERS = 37).
    # ------------------------------------------------------------------
    if warp_idx == 0:
        if nvvm.elect_sync():
            # ---- TMA Q/K/V handshakes ----
            bars.mb_tma_q_full.init()
            bars.mb_tma_q_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_full[ks].init()
                bars.mb_tma_k_empty[ks].init()
                bars.mb_tma_v_full[ks].init()
                bars.mb_tma_v_empty[ks].init()

            # ---- MMA -> softmax / corr handshakes ----
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                bars.mb_s_acc_empty[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()

                # ---- DSMEM xfer rings (sg0 -> sg1) ----
                # P12: leader = CTA_MMA arrives; non-leader = 1 (runtime override_count).
                p_full_init = cutlass.Int32(
                    arith.select(
                        is_leader.ir_value(),
                        cutlass.Int32(CFG.CTA_MMA).ir_value(),
                        cutlass.Int32(CFG.ONE_LANE).ir_value(),
                    )
                )
                bars.mb_p_xfer_full[p].init(override_count=p_full_init)
                bars.mb_p_xfer_empty[p].init()
                bars.mb_alpha_xfer_full[p].init()
                bars.mb_alpha_xfer_empty[p].init()

            # ---- one-shot stats ring ----
            bars.mb_stats_xfer_full.init()
            bars.mb_stats_xfer_empty.init()

            # ---- sg1 TMA-STG handshakes ----
            for c in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[c].init()
            bars.mb_tma_o_empty.init()

            # ---- end-of-kernel teardown ----
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            # ---- scheduler ring (NOT in Bars; still raw nvvm.mbarrier_init) ----
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4 cluster fence (cga2-aware): cross-CTA arrive_on_peer fires below.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    # ------------------------------------------------------------------
    # Per-warp role dispatch (see the module docstring's warp map).
    # ------------------------------------------------------------------
    if warp_idx >= cutlass.Int32(CFG.SOFTMAX_WG0_BASE) and warp_idx < cutlass.Int32(CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp_group(
            is_sg0=is_sg0,
            is_sg1=is_sg1,
            seqlen_q=seqlen_q,
            n_clusters=n_clusters,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            sO=sO,
            sP_xfer_raw=sP_xfer_raw,
            sAlpha_xfer_raw=sAlpha_xfer_raw,
            sStats_xfer_raw=sStats_xfer_raw,
            sLSE_raw=sLSE_raw,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            union_bits=union_bits,
            union_ntiles=union_ntiles,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            cross_sg_peer=cross_sg_peer,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # cga2 non-leader paths fork -- sg0 non-leader is quiet, sg1 non-leader
        # forwards mb_p_xfer_full to leader (P12) in a persistent loop.
        if is_leader:
            _mma_warp_group(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                n_clusters=n_clusters,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sP_xfer_raw=sP_xfer_raw,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                union_ntiles=union_ntiles,
                mcast_mask=mcast_mask,
                sg0_mcast_mask=sg0_mcast_mask,
                cta_in_pair=cta_in_pair,
            )
        else:
            _mma_warp_non_leader(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                n_clusters=n_clusters,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                union_ntiles=union_ntiles,
                cta_in_pair=cta_in_pair,
                leader_cta_id=leader_cta_id,
            )

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        _tmaldg_warp_group(
            is_sg0=is_sg0,
            tma_q_desc=tma_q_desc,
            sQ=sQ,
            bars=bars,
            sched=sched,
            union_ntiles=union_ntiles,
            n_clusters=n_clusters,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            is_sg1=is_sg1,
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            cta_in_pair=cta_in_pair,
        )

    elif warp_idx == cutlass.Int32(CFG.SCHED_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)

    else:
        # Warps GATHER_WARP_BASE .. TOTAL_WARPS-1: one complete warpgroup, every warp executes the
        # SAME setmaxnreg.dec (warpgroup-collective) -- the issuers then run their tile loop, the
        # spare warp(s) exit.  One traced copy of the issuer body per quad range (Constexpr issuer).
        nvvm.setmaxregister(GATHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        for w in cutlass.range_constexpr(N_GATHER_WARPS):  # trace-time unroll: a plain range() here would stage `w`
            if warp_idx == cutlass.Int32(GATHER_WARP_BASE + w):
                nvvm.prefetch_tensormap(tma_kv_desc.get_ptr())
                _gather_warp_group(
                    w,
                    is_sg0=is_sg0,
                    tma_kv_desc=tma_kv_desc,
                    union_ids=union_ids,
                    union_ntiles=union_ntiles,
                    sK=sK,
                    sV=sV,
                    bars=bars,
                    sched=sched,
                    n_clusters=n_clusters,
                    cta_in_pair=cta_in_pair,
                )


# ============================================================================
# sg0 softmax-iter helper -- ONE path: tcgen05.ld + tcgen05.wait::ld + membership
# mask + software row-max (the HW-fused max cannot observe the -inf written after
# the load).  Byte-identical to the parent from the s_acc_empty arrive down; the
# wait::ld is the one addition (the parent's masked arm lacks it -- see the module
# docstring, "S_acc READ COMPLETION").
# ============================================================================
@cute.jit
def _sg0_softmax_kv_iter(
    kv_loop,
    # State (threaded, returned updated):
    sg0_xfer_state,
    bmm1_done_state,
    total_max,
    total_sum_vec,
    # TMEM / SMEM bases:
    tmem_base_addr,
    bars,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    # Per-tile / per-lane:
    bits_base,
    scale_log2,
    tid_in_wg,
    is_lead_warp,
    leader_cta_id,
    cross_sg_peer,
):
    # Membership bits for THIS lane's token slot in tile kv_loop: 16 B = 4 words = 128 union columns,
    # loaded before the waits so the latency hides under the BMM1 handshake.  `j` is warp-uniform (64
    # lanes per token) -> one L1 line per warp per tile.
    bits_addr = bits_base + kv_loop.to(cutlass.Int64) * cutlass.Int64(MEMBERSHIP_TILE_BYTES)
    w0, w1, w2, w3 = ldg_int32x4(bits_addr)

    cur_parity_S = sg0_xfer_state.idx
    cur_phase_S = sg0_xfer_state.phase
    bars.mb_alpha_xfer_empty[cur_parity_S].wait(cur_phase_S)
    bars.mb_p_xfer_empty[cur_parity_S].wait(cur_phase_S)
    bars.mb_bmm1_done[bmm1_done_state.idx].wait(bmm1_done_state.phase)
    sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
    bmm1_done_state = advance(bmm1_done_state, CFG.XFER_STAGES)

    s_addr_base = tmem_base_addr + cur_parity_S * cutlass.Int32(LAYOUT.S_ACC_COLS)

    raw_chunks = [
        nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * SOFTMAX_CHUNK), cutlass.Float32),
            num=SOFTMAX_CHUNK,
        )
        for c in range(SOFTMAX_N_CHUNKS_LOAD)
    ]
    # tcgen05.ld is ASYNCHRONOUS: complete BOTH S_acc reads before anything below can publish the parity slot.  Without
    # this wait the mb_s_acc_empty arrive (relaxed, cluster scope, no data dependency on the loads) was scheduled BETWEEN
    # the two LDTM.x64 -- the MMA then overwrote S_acc[parity] columns 64..127 while chunk 1 was still being read
    # (module docstring, "S_acc READ COMPLETION"; frost-kernels.md section 3).  Never remove; never move below the arrive.
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    # Chunk c (union columns 64c .. 64c+63 of this tile) masks with words 2c, 2c+1: a ZERO bit is
    # "not one of this token's keys" -> a TRUE -inf (never the finite marker).
    words = ((w0, w1), (w2, w3))
    chunks_S = [apply_membership_chunk(raw_chunks[c], words[c][0], words[c][1], n=SOFTMAX_CHUNK) for c in range(SOFTMAX_N_CHUNKS_LOAD)]
    chunks_max = [row_max_reduction(chunks_S[c]) for c in range(SOFTMAX_N_CHUNKS_LOAD)]
    reg_S_vec = vec_concat(chunks_S)
    current_max_raw = chunks_max[0]
    for m in chunks_max[1:]:
        current_max_raw = cute.math.max(current_max_raw, m)
    reg_S_tile = RegTile(reg_S_vec, size=CFG.TILE_N)
    # THE one select of delta (3): a tile-row with no member column has raw max -inf; map it onto the
    # finite marker so `-inf - (-inf)` never forms below.  Live row / dead tile: is_first false, the
    # marker minus m clears no threshold -> alpha = exp2(0) = 1, P = exp2(-inf - m) = 0.  Dead row /
    # dead tile: is_first, alpha = exp2(marker) = 0 on a zero sum, total_max stays the marker.
    _dead_tile = current_max_raw == cutlass.Float32(float("-inf"))
    current_max = cutlass.Float32(arith.select(_dead_tile.ir_value(), NEG_INF_F32.ir_value(), (current_max_raw * scale_log2).ir_value()))

    # All-thread DSMEM arrive on sg0 leader's mb_s_acc_empty (P11) --
    # doubles as the tmem_load_fence cross-warp sync.
    bars.mb_s_acc_empty[cur_parity_S].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    # Online softmax (RESCALE_THRESHOLD skip).
    old_total_max = total_max
    is_first = total_max == NEG_INF_F32
    update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD_F32)
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    exp_input = cutlass.Float32(
        arith.select(
            is_first.ir_value(),
            NEG_INF_F32.ir_value(),
            (old_total_max - total_max).ir_value(),
        )
    )
    alpha = cute.math.exp2(exp_input, fastmath=True)

    # reg_S = reg_S * scale_log2 - total_max; then exp2 (a -inf cell -> exactly 0).
    reg_S_scaled = reg_S_tile.vec * scale_log2 - total_max
    reg_P_fp32 = cute.math.exp2(reg_S_scaled, fastmath=True)
    reg_P_half_vec = reg_P_fp32.to(P_STORAGE_DTYPE)
    reg_P_half = RegTile(reg_P_half_vec, size=CFG.TILE_N)

    # ---- Write P[tid, :] to SMEM xfer ring slot[parity] ----
    p_xfer_slot = sP_xfer_raw.subview(cur_parity_S * cutlass.Int32(pXferElems))
    for chunk in cutlass.range_constexpr(P_TMA_ITERS):
        smem_off = cutlass.Int32(chunk * P_BLOCK_BYTES) + tid_in_wg * cutlass.Int32(P_D_BLOCK)
        smem_ptr = p_xfer_slot.subview(smem_off).data_ptr()
        chunk_P = reg_P_half[chunk * P_D_BLOCK : (chunk + 1) * P_D_BLOCK].vec
        smem_ptr.store_swizzled(
            chunk_P,
            alignment=64,
            swizzle=P_SMEM_SWIZZLE,
        )

    # ---- Write alpha[tid] (FP32) to alpha xfer slot[parity] ----
    alpha_slot = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
    alpha_slot.store(alpha)

    # Update total_sum = total_sum * alpha + row_reduction(reg_P).
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    iter_sum_pair = row_reduction_pair(reg_P_fp32)
    total_sum_vec = total_sum_vec * alpha_pair + iter_sum_pair

    # Fence SMEM->async; sync the 4 compute warps so all 128 rows of
    # alpha + P are written before bulk_copy issues.
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

    # DSMEM bulk_copy alpha + P -> cross-sg sg1 peer's SMEM (predicated ship, P16).
    ship_pred = is_lead_warp & nvvm.elect_sync()
    local_alpha_src = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M))
    peer_alpha_dst = nvvm.mapa(local_alpha_src, cross_sg_peer, addrspace=7)
    peer_alpha_full_mbar = nvvm.mapa(bars.mb_alpha_xfer_full[cur_parity_S].smem_ptr, cross_sg_peer, addrspace=7)
    local_p_src = sP_xfer_raw.subview(cur_parity_S * cutlass.Int32(pXferElems))
    peer_p_dst = nvvm.mapa(local_p_src, cross_sg_peer, addrspace=7)
    peer_p_full_mbar = nvvm.mapa(bars.mb_p_xfer_full[cur_parity_S].smem_ptr, cross_sg_peer, addrspace=7)
    cp_async_bulk_shared_cluster_shared_cta(
        peer_alpha_dst,
        local_alpha_src,
        peer_alpha_full_mbar,
        alphaXferBytes,
        pred=ship_pred,
    )
    cp_async_bulk_shared_cluster_shared_cta(
        peer_p_dst,
        local_p_src,
        peer_p_full_mbar,
        pXferBytes,
        pred=ship_pred,
    )

    return (sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec)


# ============================================================================
# Compute warp group -- sg-conditional softmax (sg0) or correction (sg1).
# 4 warps x 32 lanes = 128 threads.  sg1 correction and the O store loop are the
# parent's; deltas (1), (3), (4), (5) sit in the prologue, the sg0 loop and the
# epilogue's beta / LSE fold.
# ============================================================================
@cute.jit
def _compute_warp_group(
    is_sg0,
    is_sg1,
    seqlen_q,
    n_clusters,
    scale_log2,
    tmem_ptr_i32,
    sQ,
    sO,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    sStats_xfer_raw,
    sLSE_raw,
    bars,
    sched,
    lse_tensor,
    sinks_tensor,
    union_bits,
    union_ntiles,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    cross_sg_peer,
):
    # Wait MMA's TMEM-alloc publish (named barrier 1, count 32 * (SOFTMAX_WG_WARPS + 1) = 160).
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base_addr = tmem_ptr_i32.load()

    # Per-thread / per-warp identifiers -- compute warps occupy CTA tid_x in
    # [0, 128); tid_in_wg == tid_x == the tile ROW: token tid // 64, head tid % 64.
    tid_in_wg = cute.arch.thread_idx()[0]
    wid_in_wg = tid_in_wg // cutlass.Int32(32)
    is_lead_warp = wid_in_wg == cutlass.Int32(0)
    token_local = tid_in_wg // cutlass.Int32(HEADS_PER_TILE)
    head_local = tid_in_wg % cutlass.Int32(HEADS_PER_TILE)
    # Membership slot j = 2 * cta_in_pair + token_local (the pair leader holds tokens 4c, 4c+1).
    slot_j = cta_in_pair * cutlass.Int32(TOKENS_PER_TILE) + token_local

    bits_base_all = union_bits.iterator.toint()
    ntiles_base = union_ntiles.iterator.toint()

    # Common state -- persistent-tile scheduler decode.
    q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # kv_loop bounds -- count-based (delta 4): [0, n_tiles[b, c]), n_tiles >= 1.
    kv_left = cutlass.Int32(0)
    kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, cutlass.Int32(1))

    # ---- Per-tensor swizzle for SMEM stores (same Swz128B as Q/O at d=512 f16) ----
    _O_EPI_SWIZZLE = cutlass.Swizzle(3, 4, 3)

    # Epilogue tile params (O SMEM stride).  TILE_O=512, BPE_O=2 -> 8 chunks.
    O_EPI_BLOCK_SIZE = 64 // CFG.BPE_O  # 32 fp16
    O_TMA_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES  # 8 chunks
    O_D_BLOCK = CFG.TILE_O // O_TMA_ITERS  # 64 elements / chunk
    O_TMA_GRANU_ELEMS = CFG.TILE_M * O_D_BLOCK  # 8192 elements / TMA chunk
    O_BLOCKS_PER_SUB = 128 // O_EPI_BLOCK_SIZE  # 4 (f16)
    O_CHUNK_ELEMS = 128 // CFG.BPE_O  # 64 (f16)

    # ---- sg0 state -- streaming softmax + DSMEM ship ----
    bmm1_done_state = PipelineState.start(phase=0)
    # PipelineState(0, 1) -> both mb_p_xfer_empty and mb_alpha_xfer_empty are pre-armed.
    sg0_xfer_state = PipelineState.start(phase=1)
    stats_xfer_empty_phase = cutlass.Int32(1)

    # ---- sg1 state -- recv alpha + correction + epilogue ----
    alpha_full_state = PipelineState.start(phase=0)
    sg1_bmm2_done_state = PipelineState.start(phase=0)
    stats_xfer_full_phase = cutlass.Int32(0)
    epilogue_state = cutlass.Int32(0)

    # ---- streaming-softmax accumulators (sg0) -- Vector[Float32,2] for packed FMUL2/FADD2 lowering.
    total_max = NEG_INF_F32
    total_sum_vec = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            # ============================================================
            # sg0 -- streaming softmax + ship alpha + P + stats.
            # ============================================================
            total_max = NEG_INF_F32
            total_sum_vec = cutlass.Vector.from_elements(
                (cutlass.Float32(0.0), cutlass.Float32(0.0)),
                cutlass.Float32,
            )

            # This lane's membership slot for the cluster: union_bits[b, c, 0, j, 0] as a byte address;
            # tile t sits MEMBERSHIP_TILE_BYTES further.
            cluster_idx = q_super_idx // cutlass.Int32(CFG.CTA_MMA)
            list_line = ((batch_idx * n_clusters + cluster_idx) * cutlass.Int32(U_MAX_TILES)).to(cutlass.Int64) * cutlass.Int64(TOKENS_PER_CLUSTER) + slot_j.to(
                cutlass.Int64
            )
            bits_base = bits_base_all + list_line * cutlass.Int64(MEMBERSHIP_SLOT_BYTES)

            # ONE loop, every tile through the membership path (n_tiles >= 1, so it always runs).
            for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec = _sg0_softmax_kv_iter(
                    _kv,
                    sg0_xfer_state,
                    bmm1_done_state,
                    total_max,
                    total_sum_vec,
                    tmem_base_addr,
                    bars,
                    sP_xfer_raw,
                    sAlpha_xfer_raw,
                    bits_base,
                    scale_log2,
                    tid_in_wg,
                    is_lead_warp,
                    leader_cta_id,
                    cross_sg_peer,
                )

            # ---- End-of-tile: ship final stats (max, ell) ----
            bars.mb_stats_xfer_empty.wait(stats_xfer_empty_phase)
            stats_xfer_empty_phase = stats_xfer_empty_phase ^ cutlass.Int32(1)

            final_sum = total_sum_vec[0] + total_sum_vec[1]
            # Stats layout in SMEM: ell[TILE_M] then max[TILE_M].
            stats_sum_slot = sStats_xfer_raw.subview(tid_in_wg)
            stats_max_slot = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            stats_sum_slot.store(final_sum)
            stats_max_slot.store(total_max)

            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

            # Predicated stats ship (P16).
            ship_pred = is_lead_warp & nvvm.elect_sync()
            peer_stats_dst = nvvm.mapa(sStats_xfer_raw, cross_sg_peer, addrspace=7)
            peer_stats_full_mbar = nvvm.mapa(bars.mb_stats_xfer_full.smem_ptr, cross_sg_peer, addrspace=7)
            cp_async_bulk_shared_cluster_shared_cta(
                peer_stats_dst,
                sStats_xfer_raw,
                peer_stats_full_mbar,
                statsXferBytes,
                pred=ship_pred,
            )
        else:
            # ============================================================
            # sg1 -- correction (apply alpha) + epilogue (normalize, cast,
            # store O, LSE).  The parent's, with n_tiles >= 1 (no empty arm).
            # ============================================================
            tmem_O_base = tmem_base_addr + cutlass.Int32(LAYOUT.O_OFF)

            # --- iter 0: BMM2(0) doesn't depend on alpha; fire both bmm2_ready half slots immediately.
            cur_parity_0 = alpha_full_state.idx
            bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # Arm + wait alpha_xfer_full[parity_0].
            bars.mb_alpha_xfer_full[cur_parity_0].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
            bars.mb_alpha_xfer_full[cur_parity_0].wait(alpha_full_state.phase)
            alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

            # Notify sg0 (cross-sg peer) alpha slot is empty.
            bars.mb_alpha_xfer_empty[cur_parity_0].arrive_on_peer(cross_sg_peer)

            # --- iter 1..n_kv-1: apply alpha to O before BMM2(kv) ---
            CORR_BLOCK_SIZE = 16
            CORR_BLOCKS_TOTAL = CFG.TILE_O // CORR_BLOCK_SIZE  # 32
            CORR_BLOCKS_PER_HALF = CORR_BLOCKS_TOTAL // 2  # 16

            for _kv in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                cur_parity = alpha_full_state.idx

                bars.mb_alpha_xfer_full[cur_parity].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
                bars.mb_alpha_xfer_full[cur_parity].wait(alpha_full_state.phase)
                alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

                # Wait prior BMM2 done so O is committed.
                bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
                sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

                # Read alpha[tid] from xfer_in[cur_parity].
                alpha_addr = sAlpha_xfer_raw.subview(cur_parity * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
                alpha_corr = alpha_addr.load()

                # all_alpha_one ballot: skip rescale entirely if every lane's alpha == 1.0
                # (a dead tile for every live row of the warp lands here).
                alpha_is_one = alpha_corr == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                # ---- Half-1: cols [0..TILE_O/2) ----
                if ~all_alpha_one:
                    for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                        o_off = tmem_O_base + cutlass.Int32(block * CORR_BLOCK_SIZE)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                            num=CORR_BLOCK_SIZE,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                            o_scaled,
                        )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                # Notify sg0 alpha slot empty.
                bars.mb_alpha_xfer_empty[cur_parity].arrive_on_peer(cross_sg_peer)
                # Half-1 ready -- sg1 leader BMM2 sub-tile 0 unblocked.
                bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                # ---- Half-2: cols [TILE_O/2..TILE_O) ----
                if ~all_alpha_one:
                    for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                        block_idx = CORR_BLOCKS_PER_HALF + block
                        o_off = tmem_O_base + cutlass.Int32(block_idx * CORR_BLOCK_SIZE)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                            num=CORR_BLOCK_SIZE,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                            o_scaled,
                        )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                # Half-2 ready -- sg1 leader BMM2 sub-tile 1 unblocked.
                bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # ---- Final BMM2-done wait ----
            bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
            sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

            # ---- Epilogue ----
            epilogue_state = epilogue_state ^ cutlass.Int32(1)
            bars.mb_tma_o_empty.wait(epilogue_state)

            # Arm stats_xfer_full (sg0 delivers via DSMEM bulk_copy).
            bars.mb_stats_xfer_full.arrive(n_bytes=statsXferBytes, pred=is_lead_warp & nvvm.elect_sync())
            bars.mb_stats_xfer_full.wait(stats_xfer_full_phase)
            stats_xfer_full_phase = stats_xfer_full_phase ^ cutlass.Int32(1)

            # Per-thread lds_32 of final_ell / final_max.
            ell_addr = sStats_xfer_raw.subview(tid_in_wg)
            max_addr = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            final_ell = ell_addr.load()
            final_max = max_addr.load()

            # Stats consumed -> notify sg0 stats_xfer.out free to overwrite.
            bars.mb_stats_xfer_empty.arrive_on_peer(cross_sg_peer, pred=is_lead_warp & nvvm.elect_sync())

            # Row identity (delta 1 / 5): token and head of THIS lane's row.
            q_row_global = q_super_idx * cutlass.Int32(TOKENS_PER_TILE) + token_local
            row_head = head_idx * cutlass.Int32(HEADS_PER_TILE) + head_local

            # Compute beta, lse -- HAS_SINK fold with the per-row sink and ONE extra select (delta 5).
            LN2 = cutlass.Float32(0.6931471805599453)
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = cutlass.Float32(sinks_arr[row_head])
                final_max_nat = final_max * LN2
                new_max_nat = cute.math.max(final_max_nat, sink_logit)
                scale_sink = cute.math.exp(final_max_nat - new_max_nat, fastmath=True)
                # sink == new_max -> exactly 1.0, so sink = +inf gives (scale_sink 0, new_sum 1, beta 0,
                # O 0, lse +inf) instead of exp(inf - inf) = NaN on every live row; sink = -inf gives
                # sink_term 0 = the no-sink fold.
                _sink_is_max = sink_logit == new_max_nat
                sink_term = cutlass.Float32(
                    arith.select(_sink_is_max.ir_value(), cutlass.Float32(1.0).ir_value(), cute.math.exp(sink_logit - new_max_nat, fastmath=True).ir_value())
                )
                new_sum = final_ell * scale_sink + sink_term
                beta = scale_sink / new_sum
                lse = new_max_nat + cute.math.log(new_sum, fastmath=True)
            else:
                final_ell_safe = cute.math.max(final_ell, cutlass.Float32(1e-30))
                beta = cutlass.Float32(1.0) / final_ell_safe
                lse = final_max * LN2 + cute.math.log(final_ell_safe, fastmath=True)

            # --- row with no member column in any of its tiles (an all -1 / zero-bit slot, or a padded
            # token >= S).  Masked P are EXACTLY 0 and a live row's sum is >= 2^-8 under the lazy rescale,
            # so `final_ell == 0` is exact (delta 4) -- no geometric test, no loop bounds needed here.
            # Without it the 1e-30 floor leaks (lse = marker*LN2 + log(1e-30), beta = 1e30 x residue) and
            # a sink = -inf dead row has new_sum = 0 -> beta = +inf (0 * inf = NaN): both selected away.
            _row_empty = final_ell == cutlass.Float32(0.0)
            if cutlass.const_expr(CFG.HAS_SINK):
                # A keyless row with a sink holds the sink's mass alone: LSE = sink_logit, SELECTED.
                lse = cutlass.Float32(arith.select(_row_empty.ir_value(), sink_logit.ir_value(), lse.ir_value()))
            else:
                lse = cutlass.Float32(arith.select(_row_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse.ir_value()))

            # Cast O fp32 -> bf16/fp16 with bit-permuted TMEM block index (the parent's loop, verbatim).
            #   TMEM[ 0:128] = d_v[  0:128]   (call 0, leader half)
            #   TMEM[128:256] = d_v[256:384]   (call 0, peer half)
            #   TMEM[256:384] = d_v[128:256]   (call 1, leader half)
            #   TMEM[384:512] = d_v[384:512]   (call 1, peer half)
            sO_base = sO[0].base

            for b in cutlass.range_constexpr(CFG.TILE_O // O_EPI_BLOCK_SIZE):
                b_intra = b & (O_BLOCKS_PER_SUB - 1)
                b_sub = b // O_BLOCKS_PER_SUB
                tmem_sub = ((b_sub & 1) << 1) | ((b_sub & 2) >> 1)
                tmem_block = tmem_sub * O_BLOCKS_PER_SUB + b_intra

                o_addr = tmem_O_base + cutlass.Int32(tmem_block * O_EPI_BLOCK_SIZE)
                o_fp32 = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_EPI_BLOCK_SIZE,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_fp32 * beta
                # SELECT the zero (never `* 0`): the residue can be a NaN bit pattern, and NaN * 0 is NaN.
                _o_elems = []
                for _i in cutlass.range_constexpr(O_EPI_BLOCK_SIZE):
                    _o_elems.append(cutlass.Float32(arith.select(_row_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value())))
                o_half = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)

                # SMEM store -- TMA-O grain layout (O_D_BLOCK elements / chunk).
                col_offset_const = (b * O_EPI_BLOCK_SIZE) % O_D_BLOCK
                block_idx_const = (b * O_EPI_BLOCK_SIZE) // O_D_BLOCK
                block_offset_const = block_idx_const * O_TMA_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(O_D_BLOCK)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()
                smem_ptr.store_swizzled(
                    o_half,
                    alignment=64,
                    swizzle=_O_EPI_SWIZZLE,
                )

                # Per-128B-chunk arrive on TMA-STG.
                if ((b + 1) * O_EPI_BLOCK_SIZE) % O_CHUNK_ELEMS == 0:
                    chunk = (b * O_EPI_BLOCK_SIZE) // O_CHUNK_ELEMS
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_tma_o_full[chunk].arrive()

            # Write LSE [B, 64, S] (natural log, sink included) -- one row per lane, padded tokens skipped.
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_arr[batch_idx, row_head, q_row_global] = lse

        # End-of-tile: advance scheduler.
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid_tile)

    # End-of-kernel: trailing-arrive drain (sg0 only) -- unchanged.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            if is_lead_warp:
                if nvvm.elect_sync():
                    for _p in cutlass.range_constexpr(CFG.XFER_STAGES):
                        bars.mb_p_xfer_empty[sg0_xfer_state.idx].wait(sg0_xfer_state.phase)
                        bars.mb_alpha_xfer_empty[sg0_xfer_state.idx].wait(sg0_xfer_state.phase)
                        sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
                    bars.mb_stats_xfer_empty.wait(stats_xfer_empty_phase)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        # Each CTA's compute warp fires 1 elect-arrive_on_peer to its cga2 partner (init = ONE_LANE).
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta, pred=is_lead_warp & nvvm.elect_sync())


# ============================================================================
# MMA warp group -- sg-conditional BMM1 (sg0 leader) / BMM2 (sg1 leader).
# The parent's; only the bounds source changed (count-based, delta 4).
# ============================================================================
@cute.jit
def _mma_warp_group(
    is_sg0,
    is_sg1,
    n_clusters,
    sQ,
    sK,
    sV,
    sP_xfer_raw,
    tmem_ptr_i32,
    bars,
    sched,
    union_ntiles,
    mcast_mask,
    sg0_mcast_mask,
    cta_in_pair,
):
    # SM107 cap = 576 cols; is_exclusive=True enforced inside tmem_alloc wrapper.
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    # Publish to compute warpgroup (waits on barrier_id=1 with count 160).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    # Do column arithmetic on raw Int8 ptr; retype at use site.
    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    # idesc M is COLLECTIVE (per-CTA M * CTA_MMA) under cga2.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
    )
    # BMM2 idesc -- N per call = 256 (NOT TILE_O); 2 calls per BMM2.
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2 with N per call = 256, B-transpose, manual k-step driven by kernel.
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.BMM2_N_PER_CALL,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    ntiles_base = union_ntiles.iterator.toint()

    # Persistent-tile scheduler state.
    q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    kv_left = cutlass.Int32(0)
    kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, cutlass.Int32(1))

    # sg0 leader state (BMM1).
    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    # pre-armed: PipelineState(0, 1) so iter-0/1 wait passes on fresh barriers.
    s_acc_empty_state = PipelineState.start(phase=1)

    # sg1 leader state (BMM2).
    kv_state_V = PipelineState.start(phase=0)
    sg1_mma_state = PipelineState.start(phase=0)
    bmm2_ready_state = PipelineState.start(phase=0)
    bmm2_done_prod_state = PipelineState.start(phase=0)

    # P xfer ring SmemTile (sg1 leader): TILE_M x TILE_N, Q's swizzle, DESC_VERSION (sits at 262144).
    sP_xfer = SmemTile(
        base=sP_xfer_raw,
        elems_per_stage=pXferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_P,
        desc_version=DESC_VERSION,
    )

    # BMM1 leader: desc_Q is tile-static (Q stays resident under d=512 / no Q-union-K alias).
    desc_Q = sQ[0].desc()
    # BMM2 leader: V N-block advance in ELEMENTS (BMM2_V_NBLOCK_ADVANCE is bytes).
    V_NBLOCK_ADVANCE_ELEMS = BMM2_V_NBLOCK_ADVANCE // CFG.BPE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            # ============================================================
            # sg0 leader -- BMM1: Q x K^T -> S_acc[parity]
            # ============================================================
            bars.mb_tma_q_full.wait(q_full_phase)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                cur_parity_K = s_acc_empty_state.idx
                bars.mb_s_acc_empty[cur_parity_K].wait(s_acc_empty_state.phase)
                s_acc_empty_state = advance(s_acc_empty_state, CFG.XFER_STAGES)

                bars.mb_tma_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                # S_acc TMEM offset = parity * S_ACC_COLS (TmemTile layout).
                s_acc_off = cur_parity_K * cutlass.Int32(LAYOUT.S_ACC_COLS)
                mma_ss(bmm1_desc, desc_Q, desc_K, tmem_raw.subview(s_acc_off), accumulate=False)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[cur_parity_K].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_tma_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            # After last BMM1: multicast Q-empty so both peers' TMA warps advance (P13: fires after drain).
            bars.mb_tma_q_empty.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        else:
            # ============================================================
            # sg1 leader -- BMM2: P x V -> O via mma_ss, 2 N-block calls
            # ============================================================
            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                cur_parity = sg1_mma_state.idx

                # Leader's own arrive on mb_p_xfer_full[parity] (P12: 1 of CTA_MMA).
                bars.mb_p_xfer_full[cur_parity].arrive(n_bytes=pXferBytes, pred=nvvm.elect_sync())
                bars.mb_p_xfer_full[cur_parity].wait(sg1_mma_state.phase)
                sg1_mma_state = advance(sg1_mma_state, CFG.XFER_STAGES)

                bars.mb_tma_v_full[kv_state_V.idx].wait(kv_state_V.phase)

                accum_b2 = kv_loop > kv_left

                # Rebuild SmemDescs for this iter's P parity and V stage.
                desc_P = sP_xfer[cur_parity].desc()
                desc_V_n0 = sV[kv_state_V.idx].desc()
                desc_V_n1 = sV[kv_state_V.idx].shifted(V_NBLOCK_ADVANCE_ELEMS).desc()

                # N-block 0: writes O TMEM cols [0..BMM2_N_PER_CALL).
                bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                mma_ss(bmm2_desc, desc_P, desc_V_n0, tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF)), accumulate=accum_b2)

                # N-block 1: writes O TMEM cols [BMM2_N_PER_CALL..TILE_O).
                bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                mma_ss(bmm2_desc, desc_P, desc_V_n1, tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + CFG.BMM2_N_PER_CALL)), accumulate=accum_b2)

                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[bmm2_done_prod_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_tma_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                # P13 multicast on mb_p_xfer_empty -- sg0_mcast_mask (= 0x3) targets BOTH sg0 CTAs.
                bars.mb_p_xfer_empty[cur_parity].arrive(mcast_mask=sg0_mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
                bmm2_done_prod_state = advance(bmm2_done_prod_state, CFG.XFER_STAGES)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid_tile)

    # End-of-warp tmem_dealloc -- wait fan-in from both compute warps then free.
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# ============================================================================
# MMA warp non-leader paths -- sg0 minimal quiet vs sg1 P12 forwarder (the parent's).
# ============================================================================
@cute.jit
def _mma_warp_non_leader(
    is_sg0,
    is_sg1,
    n_clusters,
    tmem_ptr_i32,
    bars,
    sched,
    union_ntiles,
    cta_in_pair,
    leader_cta_id,
):
    # All-lanes warp-collective tmem_alloc (no elect_sync gating).
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    # Match leader's named-barrier arrive count (lead is +1 on barrier_id=1).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    # sg0 non-leader is quiet (just wait dealloc); sg1 non-leader runs the persistent P12 forwarder loop.
    if is_sg1:
        ntiles_base = union_ntiles.iterator.toint()
        q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
        is_valid_tile = cutlass.Int32(1)
        sched_state = PipelineState.start()

        kv_left = cutlass.Int32(0)
        kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, cutlass.Int32(1))

        nlmma_state = PipelineState.start(phase=0)

        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                # Arm own expect_tx for P-bytes delivered by the cross-sg sg0 partner; wait the local mbar.
                bars.mb_p_xfer_full[nlmma_state.idx].arrive(n_bytes=pXferBytes, pred=nvvm.elect_sync())
                bars.mb_p_xfer_full[nlmma_state.idx].wait(nlmma_state.phase)
                cur_parity = nlmma_state.idx
                nlmma_state = advance(nlmma_state, CFG.XFER_STAGES)
                # DSMEM-arrive on leader's mb_p_xfer_full[parity] (P12: the second of CTA_MMA arrives).
                bars.mb_p_xfer_full[cur_parity].arrive_on_peer(leader_cta_id, pred=nvvm.elect_sync())

            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

            wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid_tile)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# ============================================================================
# TMA-LDG warp -- sg0: Q (packed box, one-shot per tile) + the K_ring expect_tx
# ARMS; sg1: the V_ring ARMS.  The K / V BYTES are issued by the gather warps
# (see the module docstring, "ARM vs ISSUE").  Ring states, drains as the parent.
# ============================================================================
@cute.jit
def _tmaldg_warp_group(
    is_sg0,
    tma_q_desc,
    sQ,
    bars,
    sched,
    union_ntiles,
    n_clusters,
    is_leader,
    cta_in_pair,
):
    q_empty_phase = cutlass.Int32(1)  # pre-armed (iter-0 wait passes)
    kv_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    ntiles_base = union_ntiles.iterator.toint()

    q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    # PackGQA coords: head coord = the packed head's first Q head (0 here), seq coord in TOKENS.
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(TOKENS_PER_TILE))

    kv_left = cutlass.Int32(0)
    kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, cutlass.Int32(1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            # ---- sg0: Q (one-shot per tile) + K_ring arms ----------------
            bars.mb_tma_q_empty.wait(q_empty_phase)
            q_empty_phase = q_empty_phase ^ cutlass.Int32(1)
            bars.mb_tma_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            # On the tail cluster at S % 4 in {1, 2} the PEER's box is fully OOB (q_row_base >= S): the
            # 262144-B arm above assumes the tiled load still credits its full box (module docstring, delta (1)).
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_head_idx, q_row_base, batch_idx),
                bars.mb_tma_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=None,
            )

            for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_tma_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                kv_state = advance(kv_state, CFG.STAGES_KV)
        else:
            # ---- sg1: V_ring arms -----------------------------------------
            for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_tma_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(TOKENS_PER_TILE))
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid_tile)

    # cga2: drain trailing K/V/Q empty arrives so SMEM isn't torn down while
    # leader's multicast commits are still in-flight (P15).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
            bars.mb_tma_q_empty.wait(q_empty_phase)
        else:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# ============================================================================
# Gather issue -- one stage's share of quads from one warp (delta 2).
# ============================================================================
@cute.jit
def _issue_quads(
    tma_kv_desc,
    dst_stage_base,
    mbar_ptr,
    col_base,
    i0,
    i1,
    i2,
    i3,
    l2_hint,
    q_lo: cutlass.Constexpr[int],
    q_hi: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    subbox_elems: cutlass.Constexpr[int],
):
    """Quads ``q_lo .. q_hi-1`` of the stage: lane ``g`` holds quad ``q_lo + g``'s four row ids; per quad
    the ids are shuffled to the elected lane (shuffles OUTSIDE the elect branch -- a shuffle inside a
    divergent branch is undefined) and that lane issues the quad's ``n_boxes`` gather4s.  Quad ``q`` lands
    at rows ``4q .. 4q+3`` of every sub-box: ``dst = base + i * subbox_elems + 4 q * 64`` elements = a 512-B-
    aligned offset inside the 1024-B-aligned sub-box, which is what makes the layout the tiled load's.
    Bytes route to the 2-SM pair leader's mbar (``cta_group=2`` clears bit 24 of the local address)."""
    for g in cutlass.range_constexpr(q_hi - q_lo):
        r0 = cute.arch.shuffle_sync(i0, g)
        r1 = cute.arch.shuffle_sync(i1, g)
        r2 = cute.arch.shuffle_sync(i2, g)
        r3 = cute.arch.shuffle_sync(i3, g)
        if nvvm.elect_sync():
            for i in cutlass.range_constexpr(n_boxes):
                tma_gather4(
                    tma_kv_desc,
                    dst_stage_base.subview(i * subbox_elems + (q_lo + g) * GATHER_ROWS_PER_ISSUE * GATHER_COLS_PER_ISSUE),
                    mbar_ptr,
                    col_base + cutlass.Int32(i * GATHER_COLS_PER_ISSUE),
                    r0,
                    r1,
                    r2,
                    r3,
                    cta_group=CFG.CTA_MMA,
                    l2_hint=l2_hint,
                )


# ============================================================================
# Gather warps -- issuer `issuer` of ISSUERS_PER_CTA: its quad share of every K
# (sg0) / V (sg1) stage of every tile.  Reads the tile id (-> read_tile_id_arrive),
# waits the ring's `_empty` like the TMA-LDG warp, drains it at exit (P15).
# ============================================================================
@cute.jit
def _gather_warp_group(
    issuer: cutlass.Constexpr[int],
    is_sg0,
    tma_kv_desc,
    union_ids,
    union_ntiles,
    sK,
    sV,
    bars,
    sched,
    n_clusters,
    cta_in_pair,
):
    kv_state = PipelineState.start(phase=1)  # the `_empty` rings are pre-armed (parent's TMA-LDG state)
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    # The TMA L2 cache-policy operand must be a REGISTER: one opaque mov per warp, reused by every issue.
    l2_hint = opaque_i64(TMA_L2_EVICT_LAST)

    ids_base = union_ids.iterator.toint()
    ntiles_base = union_ntiles.iterator.toint()

    k_q_lo, k_q_hi = K_QUAD_RANGES[issuer]
    v_q_lo, v_q_hi = V_QUAD_RANGES[issuer]
    # Lane g < n_quads fetches quad (q_lo + g); the other lanes re-read quad q_lo (a valid, unused address).
    k_lane = cutlass.Int32(arith.select((lane < cutlass.Int32(k_q_hi - k_q_lo)).ir_value(), lane.ir_value(), cutlass.Int32(0).ir_value()))
    v_lane = cutlass.Int32(arith.select((lane < cutlass.Int32(v_q_hi - v_q_lo)).ir_value(), lane.ir_value(), cutlass.Int32(0).ir_value()))
    # Per-peer K row offset (sg0: rows [64p, 64p+64) of each tile) / V d_v offset (sg1: cols [256p, +256)).
    k_row_off = cta_in_pair * cutlass.Int32(K_ROWS_PER_CTA)
    v_col_base = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    cluster_idx = q_super_idx // cutlass.Int32(CFG.CTA_MMA)
    list_base = (batch_idx * n_clusters + cluster_idx).to(cutlass.Int64) * cutlass.Int64(U_MAX)  # element index of union_ids[b, c, 0]
    kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, cutlass.Int32(1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            for _kv in cutlass.range(0, kv_right, 1, unroll=1):
                # ids of this lane's quad in tile _kv (16 B, read-only) -- BEFORE the ring wait.
                row_idx = list_base + (
                    _kv * cutlass.Int32(CFG.TILE_N) + k_row_off + cutlass.Int32(k_q_lo * GATHER_ROWS_PER_ISSUE) + k_lane * cutlass.Int32(GATHER_ROWS_PER_ISSUE)
                ).to(cutlass.Int64)
                i0, i1, i2, i3 = ldg_int32x4(ids_base + row_idx * cutlass.Int64(4))
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                _issue_quads(
                    tma_kv_desc,
                    sK[kv_state.idx].base,
                    bars.mb_tma_k_full[kv_state.idx].smem_ptr,
                    cutlass.Int32(0),
                    i0,
                    i1,
                    i2,
                    i3,
                    l2_hint,
                    k_q_lo,
                    k_q_hi,
                    K_BOXES,
                    K_SUBBOX_ELEMS,
                )
                kv_state = advance(kv_state, CFG.STAGES_KV)
        else:
            for _kv in cutlass.range(0, kv_right, 1, unroll=1):
                row_idx = list_base + (
                    _kv * cutlass.Int32(CFG.TILE_N) + cutlass.Int32(v_q_lo * GATHER_ROWS_PER_ISSUE) + v_lane * cutlass.Int32(GATHER_ROWS_PER_ISSUE)
                ).to(cutlass.Int64)
                i0, i1, i2, i3 = ldg_int32x4(ids_base + row_idx * cutlass.Int64(4))
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                _issue_quads(
                    tma_kv_desc,
                    sV[kv_state.idx].base,
                    bars.mb_tma_v_full[kv_state.idx].smem_ptr,
                    v_col_base,
                    i0,
                    i1,
                    i2,
                    i3,
                    l2_hint,
                    v_q_lo,
                    v_q_hi,
                    V_BOXES,
                    V_SUBBOX_ELEMS,
                )
                kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        cluster_idx = q_super_idx // cutlass.Int32(CFG.CTA_MMA)
        list_base = (batch_idx * n_clusters + cluster_idx).to(cutlass.Int64) * cutlass.Int64(U_MAX)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        kv_right = _load_n_tiles(ntiles_base, batch_idx, q_super_idx, n_clusters, is_valid_tile)

    # P15 drain: the LAST stages' multicast `_empty` commits may still be in flight; stay resident until
    # they land (the TMA-LDG warp drains the same ring -- a second waiter is harmless and keeps this
    # warp's PipelineState honest).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
        else:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# ============================================================================
# TMA-STG warp -- sg1 only: O TMA store (packed box); the parent's, one coord line changed.
# ============================================================================
@cute.jit
def _tmastg_warp_group(
    is_sg1,
    tma_o_desc,
    sO,
    bars,
    sched,
    cta_in_pair,
):
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx = _decode_initial(sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_in_pair)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        # sg0's TMA-STG slot is idle but spins the scheduler so persistent tile claims advance in lockstep
        # with sg1.  sg0 does NOT call read_tile_id_arrive (not in the READ_TILE_ARRIVERS derivation).
        if is_sg1:
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            # Wait every chunk of O ready.
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[chunk].wait(o_full_phase)

            # Both sg1 peers store their own 2 tokens x 64 heads through the packed box; rows past S are
            # clipped by the descriptor's seq extent (a partial or fully-OOB box writes nothing there).
            q_row_coord = q_super_idx * cutlass.Int32(TOKENS_PER_TILE)
            tma_store_tile(
                sO[0],
                tma_o(cutlass.Int32(0), head_idx * cutlass.Int32(HEADS_PER_TILE), q_row_coord, batch_idx),
            )
            tma_store_commit()
            tma_store_wait(0)

            bars.mb_tma_o_empty.arrive()
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

            o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _decode_payload(nxt_q, nxt_hb, cta_in_pair)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# ============================================================================
# Host launcher.
# ============================================================================
@cute.jit
def _host(
    q_tensor: cute.Tensor,
    kv_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    union_ids: cute.Tensor,
    union_bits: cute.Tensor,
    union_ntiles: cute.Tensor,
    scale_softmax_log2: cutlass.Float32,
    stream: _cuda_driver.CUstream = None,
) -> None:
    # Static extents from the fakes: q / o [B, S, 64, 512] BSHD compact, kv [B * N, 512], ids [B, NC, U_MAX],
    # bits [B, NC, U_MAX_TILES, 4, 4], ntiles [B, NC].
    B = q_tensor.shape[0]
    SQ = q_tensor.shape[1]
    NC = union_ntiles.shape[1]

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    # Packed Q / O boxes: (1 batch, 2 tokens, 64 heads, 64 d-elems); d fastest, then head, then token, so
    # the 128 SMEM rows are token_local * 64 + head == the compute lane index.
    qo_box_q = (1, TOKENS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qo_box_o = (1, TOKENS_PER_TILE, HEADS_PER_TILE, TMA_O_GRANU_ELEMS_HOST)
    stride_order = (3, 2, 1, 0)

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qo_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # ONE 2-D map over the [B * N, 512] KV view: box = 1 row x 64 elems = 128 B = one swizzle row; both
    # the K-major BMM1 B operand and the MN-major BMM2 B operand are Swz128B, so one map serves K and V.
    tma_kv_desc = tmap.create_tensor_map_tiled_from_view(
        kv_tensor,
        box_dims=(1, GATHER_COLS_PER_ISSUE),
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=qo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    # One cluster (CGA_M = 4 CTAs) per 4-token group: grid_x = NC * CGA_M; y = 1 packed head; z = batch.
    _kernel(
        tma_q_desc,
        tma_kv_desc,
        tma_o_desc,
        lse_tensor,
        sinks_tensor,
        union_ids,
        union_bits,
        union_ntiles,
        cutlass.Int32(SQ),
        cutlass.Int32(NC),
        scale_softmax_log2,
    ).launch(
        grid=(NC * CFG.CGA_M, 1, B),
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


def n_clusters_for(s: int) -> int:
    """Clusters of TOKENS_PER_CLUSTER (4) adjacent tokens; the tail cluster is partial when ``s % 4 != 0``."""
    return -(-int(s) // TOKENS_PER_CLUSTER)


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int,
    s: int,
    n_kv_rows: int,
    n_clusters: int,
    has_lse: bool,
) -> Callable:
    """Compile the fork with every extent concrete (TMA descriptor strides are pinned at compile time).

    Returns ``run(q, kv2d, o, lse_or_None, sinks, union_ids, union_bits, union_ntiles, scale_log2, stream)``:
    ``q`` / ``o`` ``[B, S, 64, 512]`` BSHD compact in the template dtype; ``kv2d = kv_all.view(B * N, 512)``
    (the CALLER asserts ``kv2d.data_ptr() == kv_all.data_ptr()``: one buffer, one descriptor); ``lse``
    ``[B, 64, S]`` fp32 or None (``has_lse``); ``sinks (64,)`` fp32 (read only when the template has a sink);
    ``union_ids [B, NC, U_MAX]`` / ``union_bits [B, NC, U_MAX_TILES, 4, 4]`` / ``union_ntiles [B, NC]`` int32
    from ``kernels/union_lists.build_union_lists``; ``scale_log2 = scale * log2(e) > 0``; ``stream`` an int
    handle or a ``CUstream``.  ``run.compiled`` is the raw artifact.  Every extent is checked at the
    tvm-ffi boundary; ``S == 0`` is the caller's no-launch case.
    """
    _cache_key = _template_key(globals(), locals(), "compile")
    if b < 1 or s < 1:
        raise ValueError(f"sparse_attention_d512: need B >= 1 and S >= 1 (got B={b}, S={s}); S == 0 is a no-launch at the caller")
    if n_kv_rows < 1 or b * n_kv_rows >= 2**31:
        raise ValueError(f"sparse_attention_d512: the flat KV row space B * N = {b * n_kv_rows} must be >= 1 and fit int32")
    if n_clusters != n_clusters_for(s):
        raise ValueError(f"sparse_attention_d512: n_clusters must be ceil(S / {TOKENS_PER_CLUSTER}) = {n_clusters_for(s)} for S={s} (got {n_clusters})")

    def _fake_bshd(dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, (b, s, HEADS_PER_TILE, CFG.TILE_K), stride_order=(3, 2, 1, 0), assumed_align=16)

    fake_q = _fake_bshd(STORAGE_DTYPE)
    fake_kv = cute.runtime.make_fake_compact_tensor(STORAGE_DTYPE, (b * n_kv_rows, CFG.TILE_K), stride_order=(1, 0), assumed_align=16)
    fake_o = _fake_bshd(OUT_STORAGE_DTYPE)
    # has_lse=False: the LSE argument is None-specialized and the store is compiled out entirely.
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, HEADS_PER_TILE, s), stride_order=(2, 1, 0), assumed_align=16) if has_lse else None
    fake_sinks = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (HEADS_PER_TILE,), stride_order=(0,), assumed_align=16)
    fake_ids = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b, n_clusters, U_MAX), stride_order=(2, 1, 0), assumed_align=16)
    fake_bits = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (b, n_clusters, U_MAX_TILES, TOKENS_PER_CLUSTER, _UL_WORDS_PER_SLOT), stride_order=(4, 3, 2, 1, 0), assumed_align=16
    )
    fake_ntiles = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b, n_clusters), stride_order=(1, 0), assumed_align=16)

    compiled = _compile_cached(
        _host,
        fake_q,
        fake_kv,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_ids,
        fake_bits,
        fake_ntiles,
        cutlass.Float32(0.0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_mqa_sparse_attention_d512",
    )

    def run(q, kv2d, o, lse, sinks, union_ids, union_bits, union_ntiles, scale_log2, stream):
        if not float(scale_log2) > 0.0:
            raise ValueError(f"sparse_attention_d512: scale_log2 must be > 0 (got {scale_log2}); the dead-tile select assumes a positive scale")
        if (lse is None) == has_lse:
            raise ValueError(f"sparse_attention_d512: compiled with has_lse={has_lse}; pass {'an' if has_lse else 'no'} LSE tensor")
        cu_stream = stream if isinstance(stream, _cuda_driver.CUstream) else _cuda_driver.CUstream(int(stream))
        compiled(q, kv2d, o, lse, sinks, union_ids, union_bits, union_ntiles, cutlass.Float32(float(scale_log2)), cu_stream)

    run.compiled = compiled
    return run
