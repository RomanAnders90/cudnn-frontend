# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused INDEXER launch of a block-sparse DECODE step: the prepare, the paged scorer and the decode top-k as ONE kernel.

Today the step's indexer is three launches -- ``qsa_prepare`` (the Q / K norm + RoPE in place, the K / V pool rows, the raw indexer key
into the ring and the block it completes into the compressed-key cache), ``qsa_score_paged`` (fp32 ReLU-summed head scores per visible
block + the per-row visible counts) and ``qsa_topk_decode`` (the per-row list in canonical order).  This kernel is the three as ONE
launch whose CTAs decode a ROLE from ``blockIdx.x`` against a host-built dispatch table (one block size, one dynamic SMEM size, one
register budget = the MAX over the roles); the three bodies stay the three kernels' own ``@cute.jit`` role functions (the standalone
kernels keep calling them: their cubins stay byte-identical -- a pin), and the one new cross-CTA primitive is a per-row GMEM counter
released by the scorer CTAs and acquired by the top-k CTA (``tile_dsl/gsync.py``: ``release_add`` / ``acquire_geq`` /
``counter_store_u64``).  The scorer role DERIVES the indexer rows it needs from this step's prepare itself (the ``S_q x 4`` indexer-query
rows and the one compressed-key row this step completes), so the prepare role is OFF the indexer's critical path: it waits on nothing
and is waited on by nothing; the kernel's completion orders its pool / ring / cache writes before the sparse tile and before the next
step.  Every output is produced by the same fp32 operations in the same order as the three launches: the fused step is BITWISE the
three-launch form -- a PERFORMANCE knob of the block (``indexer_megakernel``), never a numerics choice.

**This module holds the pre-development tables and the host half (it imports no kernel module at import time).**  The kernel body --
the role functions' fused caller ``frost_qsa_indexer_fused``, its launch wrapper and the fake-argument builder of the host renderings --
lives in ``qsa_indexer_fused_kernel.py`` (the derive prologue ``_derive_indexer_rows`` beside the scorer role it serves, in
``qsa_score_paged.py``); ``compile_qsa_indexer_fused`` / ``run_qsa_indexer_fused`` below gate the DSL floor and the device's target BEFORE
importing it (Rule 7), then compile one rendering per compile key and launch it once per step.  Served arms: the prepare's
``QsaPrepareIndexArms(query=False, compress=True, paged=True, ...)`` (``ring_slots`` / ``rope_table`` / ``n_commit`` as the plan has them; the
query arm is REFUSED -- the scorer derives the rows), bf16 activations (the scorer packs bf16 keys), ``S_q <= pool`` (the derive publishes
the ONE compressed-key row a step of at most ``pool`` rows can complete; a wider step is a typed decline), ``rope_dim > 0`` (the compress
arm's position tables are the derive's).  No performance number is claimed here: the model behind the fusion lives outside this file, and
every row of it is replaced by the gate's measurement.

Dispatch table (sequence-major: every consumer right behind its producers; ``n_chunks = ceil(n_blocks_max / (128 x units))``;
``THREADS`` = 1024 (the top-k's measured best form; 512 the fallback), ``W = THREADS / 32`` warps per CTA; ``T = B x S_q`` rows)::

    role  blockIdx.x range (linear)                                           CTAs                       body
    R0    [0, n_prep)                                                         ceil((attn + raw) / W)     prepare_role(wid = bidx x W + warp):
          attn = ceil(T x (H_q + 2 H_kv) / (32 / lanes(D))),                                              the attention rows + the raw-key /
          raw  = ceil(T / (32 / lanes(D_i)))  (the query arm DROPPED:                                     compress arm, QsaPrepareIndexArms(
          nobody reads ix_q under the fused form)                                                         query=False, compress=True, paged=True)
    R1    [n_prep + b x (n_chunks + S_q), + n_chunks)      for b in [0, B)   B x n_chunks               score_role(chunk_id = k x units + unit, b,
                                                                                                          unit, fused=True, derive=True, units)
    R2    [n_prep + b x (n_chunks + S_q) + n_chunks, + S_q) for b in [0, B)  B x S_q = T                topk_role(row = b x S_q + s, fused=True)

    idle warps: the warps >= 4 x units of an R1 CTA (THREADS 1024 at units < 8; 512 at units < 4) take NO role -- they return at the
    role branch, before any PDL site and before any barrier, so the ledgers of ids 1 and 2 + units below count 128 x units EXACTLY
    (an exited warp arrives on no counted named barrier; a warp that reached score_role with unit >= units would index ring slices
    past the SMEM carve).  R0 has no idle warps in the ledger sense: a warp unit past the last arm's warp count computes on clamped
    rows and stores nothing (the prepare role's own rule), and the prepare has no barrier.  R2 uses every warp of the CTA.
    dead units: n_chunks = ceil(ceil(n_blocks_max / 128) / units), so when the standalone chunk count is not a multiple of units the
    LAST R1 CTA of every sequence holds units whose chunk_id >= ceil(n_blocks_max / 128) -- a dead unit: its first key is at or past
    n_blocks_max >= n_vis_max, so every warp of it is a dead warp of the role (no key copy, no logits store, no counts write: chunk
    0's unit alone writes them), it still runs the role and joins the CTA-wide barriers its ledger counts (ids 1 and 2 + units stay
    128 x units), and the release is still exactly one warp per CTA (unit 0, which is never dead: k x units < the standalone count).
    The dispatch table never shrinks the CTA for it (one block size per launch); the bitwise cells cover a chunk count that is not a
    multiple of units.

    order: [R0 x n_prep] [R1(b=0) x n_chunks] [R2(b=0) x S_q] [R1(b=1) x n_chunks] [R2(b=1) x S_q] ...
    decode on device (warp-uniform; kernel scalars n_prep, n_chunks, s_q): bidx < n_prep -> R0;  r = bidx - n_prep;
    b = r / (n_chunks + s_q); k = r % (n_chunks + s_q); k < n_chunks -> R1 chunk k of b; else R2 row b x s_q + (k - n_chunks).
    the test-only twins (served with ``deadline_ns`` alone, never by a plan): consumers-first = [R0] [R2(b=0) x S_q] [R1(b=0) x n_chunks]
    ... (k < s_q -> R2 row b x s_q + k; else R1 chunk k - s_q of b) -- it violates the pin but keeps forward progress under in-order
    dispatch (at most one sequence's consumers wait on undispatched producers), so it can never make the trap fire;
    consumers-first-global = [R0] [R2 x T] [R1 x B x n_chunks] (r < T -> R2 row r; else r' = r - T: R1 chunk r' % n_chunks of
    b = r' / n_chunks) -- EVERY sequence's top-k CTAs ahead of EVERY scorer CTA: with more rows than resident slots every slot holds a
    consumer whose producers cannot be dispatched, the one table that strands the machine = the trap test's detector input.

    cell (B x KV tokens x S_q; units)   R0   R1 per seq   R2 per seq   total   waves at 1 CTA / SM (212 SMs)
    B 1 x 8K x 1 (1)                     1    16           1            18      1
    B 1 x 32K x 1 (1)                    1    64           1            66      1
    B 1 x 128K x 1 (2)                   1    128          1            130     1
    B 4 x 8K / 32K / 128K x 1 (1/2/4)    4    16 / 32 / 64 1            72 / 136 / 264   1 / 1 / 2
    B 1 x 8K x 4 (1) / B 4 x 8K x 4 (1)  4 / 15  16        4            24 / 95          1
    B 32 x 8K / 32K / 128K x 1 (8)       29   2 / 8 / 32   1            125 / 317 / 1085 1 / 2 / 6
    B 32 x 8K / 32K / 128K x 4 (8)       114  2 / 8 / 32   4            306 / 498 / 1266 2 / 3 / 6

(:func:`dispatch_table` computes every row from the three kernels' own grid arithmetic; the test module's placement cell compares it
against them.)

Counter table (``SUM(issuing lanes) == target`` per word; the lifecycle: PLAN-OWNED words, zero at build, consumer-reset)::

    #   name / words                      producer (issuing lanes)                                   SUM == target          consumer / wait                                    reset                       start
    C1  cnt_row[t], t in [0, T): T x u64  every R1 CTA of sequence b, AFTER its last logits /      1 lane x n_chunks      R2 CTA of row t: acquire_geq(cnt_row[t], n_chunks,  ONE lane of the R2 CTA      0 at plan build
        on ceil(T / 16) 128-B lines       counts / block_lens store: release_add_rows(cnt_row[b S_q],  CTAs == n_chunks       warp_gate = warp == 0 (warp = the CTA-level warp   right after acquire_geq     (torch.zeros on
        (counter_word_offset)             S_q, 1, warp_gate = (unit == 0) & (warp_in_unit == 0),     per cnt_row[t]         index of the top-k role), bar_id = 1, bar_threads   returns (the site below;    the ambient
                                          bar_id = 1, bar_threads = 128 x units): ONE call per CTA                           = THREADS) BEFORE its first counts / logits load   every arrival observed;     stream, outside
                                          -- one fence + one named barrier, then the S_q words'                              (the identity path included)                       no other reader in this     any capture;
                                          red.release by ONE elected lane (S_q sites) -- the gate                                                                               launch)                     never a per-
                                          names ONE warp of the whole CTA -- UNCONDITIONAL (a dead                                                                                                          execute fill)
                                          sequence -- kv_lens 0, counts 0 -- arrives too: the
                                          count is the protocol's, never the data's)

    The gate, exactly: inside score_role ``warp`` is ``warp_in_unit`` (0..3 inside the unit), so ``warp_in_unit == 0`` alone holds on one
    warp of EVERY unit -- ``units`` elected lanes per CTA, units x n_chunks arrivals per word, the ledger violated at every units > 1 plan
    (cnt_row[t] reaches n_chunks after ceil(n_chunks / units) CTAs and the top-k reads columns still in flight; the reset then hides the
    over-arrival, so the zero-after-launch cells stay green).  The gate is therefore ``(unit == 0) & (warp_in_unit == 0)``: one warp of
    the CTA, warp-uniform (gsync's convention: a role that runs several copies per CTA derives its gate from the CTA-global warp
    identity, never from a per-copy warp index; gsync's trace-time guard cannot see a traced predicate).  The reset site, exactly:
    ``if warp == 0: if nvvm.elect_sync(): counter_store_u64(addr, cutlass.Int64(0))`` right after ``acquire_geq`` returns (its elect is
    internal and it exposes no acquiring lane; the literal is an ``Int64``, never a Python ``0`` into inline PTX).  Under the test-only
    compile key ``check_arrivals`` that same lane re-reads the word (``ld.acquire.gpu.global.u64``) before the reset and TRAPS unless it equals
    n_chunks exactly -- the over-arrival detector the reset otherwise erases; the bitwise cells run it at units 2 and 8.

    release = every storing thread's stores -> fence.acq_rel.gpu by every thread -> the named barrier (id 1) over the CTA's units ->
    ONE elected lane's red.release.gpu.global.add.u64 per word of the sequence (``gsync.release_add_rows``: the fence and the barrier
    ONCE, then S_q reds -- every red is a release by the elected lane the one fence + barrier made cumulative, so S_q separate ``release_add``
    calls would order the same stores at S_q times the fence + barrier cost); acquire = one lane's ld.acquire.gpu.global.u64 poll (nanosleep.u32 128 between
    tests; the other lanes and warps park in the named barrier), then the barrier publishes it.  No async-proxy arm: R2 reads the logits /
    counts through generic loads (the fence.proxy.async.global arm is the TILE role's, not built here).  The PTX census of the fused
    rendering pins the sites: red.release.gpu.global.add.u64 == S_q per R1 rendering, ld.acquire.gpu.global.u64 == 2 per R2 rendering
    (the poll's first test and its loop; one more under ``check_arrivals``), st.relaxed.gpu.global.u64 == 1 per R2 rendering (the reset),
    no atom.*.global / red.*.global anywhere else (the top-k's red.shared / atom.shared histogram is counted separately), no .cluster.
    The words are never read above a PDL wait that a previous launch of this kernel could race: the slab GEMM sits between two launches
    in stream order, and under ``use_pdl`` that holds only while EVERY programmatic node between them (the tile, the combine, the
    out-projection, the reduce, the slab GEMM) executes its own griddepcontrol.wait before completing -- the serving PDL test's census
    of the two griddepcontrol sites on every node of the programmatic chain is the detector.  Two concurrent executes of ONE plan on two
    streams are undefined (as the workspace already is); a launch that does not complete leaves a sticky context in which no later
    launch of the plan runs, so no residue is ever read.

Barrier table of the fused CTA (one role per CTA; the three roles' INTRA-role tables are the three kernels' own and unchanged --
``qsa_prepare.py`` (none: no barrier, no SMEM), ``qsa_score_paged.py`` ("Barrier table"), ``qsa_topk_decode.py`` ("Barrier table");
named-barrier ids are allocated disjointly across the roles so the audit reads one ledger)::

    id          role  site                                             guard / issuing lanes                                 SUM(issuing lanes) == count
    0           R2    bar.sync 0 at the top-k's sites B0 / B1_1 / B2 /  every thread of the CTA, once per visit (the identity   THREADS == THREADS (the top-k's own table,
                      B3 / C1 / R1 / F2                                 path performs no barrier)                              at THREADS = 1024 or 512)
    1           R1    gsync release_add_rows(..., S_q, 1, bar_id = 1,   every thread of the CTA's units (the fence by all; the  128 x units == 128 x units; ONE visit (the
                      bar_threads = 128 x units), one call after the    S_q reds by ONE lane: warp_gate = (unit == 0) &        idle warps >= 4 x units returned before the
                      last store                                        (warp_in_unit == 0) + elect = one warp of the CTA)     site and count nowhere; a dead unit visits)
    1           R2    gsync acquire_geq(..., bar_id = 1, bar_threads =  every thread of the R2 CTA (one lane polls, the rest   THREADS == THREADS; one visit
                      THREADS) before the first counts / logits load    park in the barrier)
    2 + unit    R1    the unit's query-tile barrier of the fused form   all 128 threads of unit ``unit`` (4 warps x 32), once,  128 == 128 per unit ('bar.sync 2 + unit,
    (unit <           WITHOUT derive (fused=True, derive=False):        after every thread's cp.async.wait_group of its       128'); the standalone kernel renders
     units)           'bar.sync 2 + unit, 128' at the site of the       prologue group (a live warp or not)                   'bar.sync 0' at this site (fused=False)
                      standalone body's bar.sync 0
    2 + units   R1    the derive-publish barrier (fused=True,           all 128 x units threads of the R1 CTA, once, after     128 x units == 128 x units; one visit
                      derive=True): 'bar.sync 2 + units, 128 x units'   the derive warps' st.shared (generic stores ->
                      rendered AT THE SAME SITE in place of the         generic ld.shared: no proxy fence) and before any
                      per-unit barrier -- under derive the per-unit     unit's A-fragment reads / the owning warp's sNew read
                      barrier would order nothing (the query-tile
                      cp.async group is empty, the key rings are
                      per-warp), so ONE barrier serves both jobs
    --          R0    none (the prepare has no barrier and no SMEM)     --                                                    --
    warp        R1    bar.warp.sync per n-tile (the ring's own)          all 32 lanes of the warp, inside the warp-uniform      32 == the warp
                                                                        live branch only
    P17         all   griddepcontrol.wait at the top of R0 / R1 / R2;   under the ``use_pdl`` compile key only                 --
                      launch_dependents once per CTA (sites below)

    No mbarrier, no TMA, no tcgen05, no cluster in any role (the three kernels have none today); the one new cross-CTA primitive is C1.
    The derive warps are warps 0-2 of the R1 CTA (unit 0); the 'owning warp' is the one warp (in whichever unit) whose n-tile range
    holds ``blk_new``.  Its overwrite of the slot row is a warp-local step, LANE-MATCHED to the copy map: the 16 granules of row ``r``
    of a slot were copied by the 16 lanes ``16 x (r & 1) + c`` of that warp (``_copy_ntile``: lane ``16 x (r & 1) + c`` copies granule
    ``c`` of row ``r`` to position ``c ^ (4 x (r & 1))``) and are read by the 4 lanes of quad ``g = r`` (``_read_key_frags``), so each of
    those 16 lanes overwrites exactly the 16 B its own cp.async wrote, between the warp's cp.async.wait_group(i) of that n-tile (its own
    copies complete) and the bar.warp.sync that follows it (which publishes the stores to the reader lanes) -- no second warp sync,
    never a CTA barrier.  A store by any other lane, or one placed after the bar.warp.sync, races a copy still in flight or a reader
    lane (the alignment cells of the test plan -- the derived row against the prepare's cache row bitwise at every alignment -- are the
    detector).

SMEM table (per role; the dynamic allocation = the MAX over the plan's roles, carved per role at trace time -- a CTA runs ONE role, so
the roles' buffers alias trivially; no descriptor reads any buffer here, so the swizzle's one job is the banks)::

    buffer (role)         bytes                                     writer / how                                      reader / how                         lifetime / swizzle
    sK (R1)               units x 4 warps x ring_slots x 2 KiB      cp.async.cg, 16 B per lane (today's copies);      ld.shared.v4 B fragments by the     the unit's mainloop; the kernel's
                          = 32 KiB per unit at ring_depth 4         the derive: ONE st.shared row of 256 B into       slot's warp                          c ^ (4 x (r & 1)) granule layout (one
                          (ring_slots = min(ring_depth + 1,         the slot holding blk_new -- the owning warp's                                          wavefront per quarter-warp); the
                          4 x units_per_warp))                      16 copying lanes of that row, lane-matched                                             derive's row store follows the same XOR
                                                                    (the note above), between wait_group(i) and
                                                                    the bar.warp.sync of that n-tile
    sA (R1)               m_tiles x 4 KiB (one m-tile at S_q <= 4:  the derive warps' st.shared (bf16 rows, 16-B       every warp's ld.shared.v4 A          written before 'bar.sync 2 + units',
                          16 rows x 256 B), ONE per CTA shared by   granules) in place of today's cp.async from       fragments (8 per m-tile per lane)    read after it, for the role's lifetime;
                          every unit                                ix_q; the padding rows zero                                                            the same layout as today's copy
    sNew (R1)             256 B + 16 B (the completing row + its    the derive warps' st.shared (bf16) + one word       the owning warp (one warp of one      written before 'bar.sync 2 + units',
                          column / presence word), ONE per CTA      (blk_new or -1)                                   unit): 16-B ld.shared per lane       read once after it; 16 lanes x 16 B =
                                                                                                                                                           one 256-B row, conflict-free
    s_hist / s_pool /     113.4 KiB (116,144 B) at 1024 threads     the top-k's own                                   the top-k's own                      the top-k's own table
    s_rank_* / small (R2) (the 2 x 4096-bin histogram 32 KiB + the
                          staging pool 80 KiB + the rank list 1 KiB)
    R0                    0                                         --                                                --                                   --
    union                 max(R2, R1, 0) = 113.4 KiB at 1-2 units (B <= 4 plans: the standard carveout) / 260.3 KiB at 8 units (B 32 plans:
                          8 x 32 KiB + 4 KiB + 272 B -> the OVERSIZED carveout, L1 -> 8 kB: the scorer's cp.async.cg bypasses L1, the
                          prepare's and the top-k's loads are streamed once, nothing relies on L1 hits) / 196 KiB at 8 units x ring_depth
                          2 (the standard carveout; a knob the gate sets).  CTAs per SM = 1 everywhere (registers: 1024 x 64 = the file).

Forward progress (the assumption, stated exactly).  This kernel relies on ONE property: the hardware dispatches the CTAs of a grid in
non-decreasing linear ``blockIdx`` order (a CTA with a lower linear index is never dispatched after one with a higher index).  Under
it, placing every PRODUCER of a counter at a lower index than every CONSUMER of that counter makes every wait terminate: when an R2
CTA is resident, every R1 CTA of its sequence has been dispatched -- each is finished or resident, and no R1 CTA waits on anything of
this grid (R0 and R1 wait on nothing of this grid -- under ``use_pdl`` every role waits on the PREVIOUS grid, once, at its top; the
roles are acyclic) -- so they complete and the acquire returns, even when the grid exceeds the resident capacity (B 32 x 32K: 317 CTAs
over 212 SMs) and even when every resident slot holds a spinning consumer.  What the property is NOT: a CUDA programming-model
guarantee (thread blocks must be executable in any order); it is an observed property of this part + driver pinned by test (the gsync
``chain`` arm, 424 CTAs = two waves, 1000 / 1000 on cc 10.7 and 10.0).  A driver that broke it would surface as a HANG (exit 124 under
the kernel timeout, or the deadline trap in tests: ``deadline_ns``, a test-only compile key -- ``trap`` + a diagnostic word, never a
bigger timeout), never as a wrong answer.  Other kernels on the GPU (another stream, MPS) only shrink the resident capacity: a slower
step, never a deadlock.  No cluster launch in the fused grid (the top-k's cluster form is kept OUT of it: a cluster launch would cluster
every role's CTAs and the argument would move to cluster granularity).  :meth:`DispatchTable.producers_before_consumers` is the host
pin of the placement; the mis-ordered twins (``order = "consumers-first"`` per sequence, ``"consumers-first-global"`` across the grid --
the latter the trap rendering's input, since only a GLOBAL mis-order strands the resident capacity) are RED on purpose.

PDL sites per role (P17; under the ``use_pdl`` compile key only, the launch attribute set with it; AUTO stays off until the chain's
table): ``griddepcontrol.wait`` is the FIRST instruction of every role -- ahead of R0's slab-band / ``kv_lens`` / table reads, of R1's
``kv_lens`` / table / slab-band / ring / cache reads (the derive's inputs are the slab GEMM's output: the predecessor), of R2's counter
acquire and ``counts`` / ``logits`` reads; no global write sits above a wait.  ``griddepcontrol.launch_dependents`` once per CTA: R0
after ``prepare_role`` returns (today's site: the prepare kernel's own pair sits AROUND the role call -- the role carries no PDL site, so
the fused caller issues R0's wait before the call and its release after it, after every store of the role; the tile is a CLC grid whose
early launch costs nothing, so no earlier site is wanted), R1 after its prologue copies (today's site, inside the role), R2 right after
the ``counts`` read (today's site, inside the role; the identity path fires it too).  The PTX census of the fused rendering under
``use_pdl`` pins the pairs: ``griddepcontrol.wait`` == 3 and ``griddepcontrol.launch_dependents`` == 3 (one pair per role).  The
dependent is the sparse tile: launched with the programmatic-stream-serialization attribute it may start while this grid runs and MUST
issue its own ``griddepcontrol.wait`` before it touches the list, the pools, the ring or the cache -- its wait returns only after this
grid has completed and flushed, so the early release is a performance choice, never a correctness surface.  No TMEM is allocated by any
role (the P17 ``tmem_alloc`` exception does not arise).

Degenerate inputs, each with its handling per role (every one a cell of the fused module's tests; the counters read 0 after the launch
in every row)::

    input                              R0 prepare                                  R1 scorer (+ the derive)                                R2 top-k
    kv_lens[b] == 0 (a padding         every pos < 0: dead rows -- no ring row,    n_vis = 0: no logits column, counts = block_lens =     counts 0 -> the identity path: every
    sequence on the null ring block /  no block, slot -1; the null ring block      0, no key copy; the derive computes on the clamped      slot -1; the acquire + reset as on a
    page 0)                            and page 0 are never written                (position 0) band rows that nothing consumes;          live row
                                                                                   the S_q releases FIRE (unconditional)
    kv_lens[b] < S_q (a contract       the rows at a negative position are dead    the rows at a negative position get n_vis = 0           as above per row
    violation)                         as above                                    likewise; the releases fire
    a -1 (or out-of-range) page under  a completing block on a negative page /     the part's copy is ZERO-FILLED (src-size 0; the         a 0 logit ranks like any other
    a visible block                    past the cache reports -2, writes no block  address is page 0, a real row): logit exactly 0,
                                                                                   still written; the derive does NOT substitute a row
                                                                                   the prepare reports -2 for (the same predicate)
    a completing block past the        reports -2, writes no block (no ring row    the derive sees the same -2 predicate: no sNew row,    --
    table's width (the -2 reports)     either when the table is the ceiling)       presence word -1; the slot keeps the copied bytes
    the identity region                --                                          block_lens = n_vis                                     the identity list: every visible id, -1
    (counts <= top_k)                                                                                                                      past them; no histogram, no bar.sync 0
    the ring wrap (cap = S_q + 3 =     this step writes ring positions pos0 ..     the derive reads ring positions pos0 - 3 .. pos0 - 1    --
    min_ring_rows(S_q, pool 4))        pos0 + S_q - 1                               (earlier steps' rows): disjoint from R0's writes of
                                                                                   this launch whenever cap >= S_q + 3 (the host rule)
    no completing block this step      no cache row                                 sNew presence -1: no substitution anywhere             --
    (S_q 1: three steps in four)

Oracle statement.  The fused launch is BITWISE the three-launch form built from the three standalone kernels in the same process on the
same primed caches: ``out`` / the list, ``counts``, ``block_lens``, the K / V pool rows, the ring rows and the cache rows written this step
all ``torch.equal`` -- at units 1, 2 and 8 (the over-arrival of a per-unit release gate shows only at units > 1), with ``check_arrivals``
armed and the counters read back zero after every launch; the derived ``blk_new`` row equals the prepare's cache row bitwise at every
alignment -- the two derivations use the SAME ``@cute.jit`` helpers (``rmsnorm_rstd``, ``rope_rotate_half``, the ``qsa_compress`` pool-1
arithmetic) in the SAME lane geometry (16 lanes per 128-wide row, the butterfly shuffle's order) as the prepare's arms.  The suite's gate
underneath stays the fp64 one: the selection equals the fp64 top-k of the same logits up to exact ties at the k-th value.  The tie
clause: a ptxas contraction difference in the derive would show as a 1-ulp logit and, on a tie row only, as a list difference --
reported against the fp64-set gate and declared ("within one fp32 rounding, ties may differ"), never widened; the knob's AUTO default
needs the bitwise pin or an explicit opt-in.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import torch

from cudnn.frost.buffers import cutedsl_requirement_error

# the two sibling kernels' host-side facts (Rule 7: a declaration reads them without importing a kernel module)
from cudnn.QSA.selection_paged import (
    BLOCK_SIZE,
    INDEX_HEAD_DIM,
    KEYS_PER_NTILE,
    MAX_QUERY_ROWS,
    MMA_M,
    NTILES_IN_FLIGHT,
    NTILES_PER_UNIT,
    RING_DEPTH_CHOICES,
    THREADS as SCORER_UNIT_THREADS,
    UNITS_PER_WARP_CHOICES,
    _check_i32_1d,
    _kbar_rows_view,
    blocks_per_cta,
    query_row_map,
    validate_params,
)

__all__ = [
    "ROLE_PREPARE",
    "ROLE_SCORER",
    "ROLE_TOPK",
    "ROLE_NAMES",
    "ROLES",
    "DISPATCH_ORDERS",
    "DEFAULT_THREADS",
    "FUSED_THREAD_COUNTS",
    "SCORER_UNIT_CHOICES",
    "SCORER_UNIT_THREADS",
    "TOPK_MAX_EPT",
    "TOPK_MAX_ROWS",
    "DispatchTable",
    "QsaIndexerFusedRecipe",
    "dispatch_table",
    "prepare_role_warps",
    "scorer_chunks",
    "scorer_smem_bytes",
    "topk_cols_cap",
    "dynamic_smem_bytes",
    "compile_qsa_indexer_fused",
    "run_qsa_indexer_fused",
]

ROLE_PREPARE = 0
ROLE_SCORER = 1
ROLE_TOPK = 2
ROLE_NAMES = ("prepare", "scorer", "topk")

DISPATCH_ORDERS = ("sequence-major", "consumers-first", "consumers-first-global")
"""``sequence-major`` is the served order (every R2(b) right behind every R1(b)); ``consumers-first`` puts a sequence's top-k CTAs
AHEAD of its scorer CTAs (the deliberately mis-ordered table of the host pin: it violates the placement, yet keeps forward progress under
in-order dispatch); ``consumers-first-global`` puts EVERY sequence's top-k CTAs ahead of EVERY scorer CTA -- the table that strands the
machine, the trap test's detector input.  Neither twin is ever launched by a plan (the compile serves them with ``deadline_ns`` alone);
the tuple is append-only."""

DEFAULT_THREADS = 512
"""The fused block size the register census of the host rendering chose (the go / no-go before any GPU time): at 1024 threads every role
runs flat at 64 registers and the SCORER role spills there -- 156 B at its standalone MMA batch, 28-36 B at batch 2, and still 4 B (one
STL / LDL pair) at batch 1 in the 8-unit and the S_q 4 renderings -- while at 512 threads (128 registers) it is spill-free at its own
batch; the spills left at 512 are the top-k's 64-key tile at 32768 columns (its standalone 512 form's own).  The 1024 form (the top-k's
measured best: its 512 form costs +0.6 / +1.7 us at 8192 / 32768 columns) stays served as an explicit knob with the scorer at batch 1;
at 512 threads at most 4 scorer units fit a CTA (the dispatch table refuses 8)."""
SCORER_MMA_BATCH_AT_64_REGS = 1
"""The scorer role's n-tiles per MMA batch under the 1024-thread (64-register) budget: 1 -- the register census's smallest-spill point
(0 / 0 at one unit and S_q 1, one STL / LDL pair at 8 units or S_q 4; batch 2 spills 28-36 B, the standalone batch 4 156 B).  0 would be
the standalone rule (4 at ``units_per_warp`` 1)."""


def scorer_mma_batch_for(threads: int, units_per_warp: int = 1) -> int:
    """The scorer's MMA batch at the fused block size: the standalone rule at 512 threads (128 registers: the role keeps its form), the
    64-register figure at 1024."""
    return SCORER_MMA_BATCH_AT_64_REGS if int(threads) == 1024 and int(units_per_warp) == 1 else 0


FUSED_THREAD_COUNTS = (1024, 512)  # the top-k's THREAD_COUNTS, largest first: the served block sizes of the fused kernel
SCORER_UNIT_CHOICES = (1, 2, 4, 8)  # 4-warp scorer units per R1 CTA (1-2 at B <= 4, 8 at B 32; a typed per-plan constant)
TOPK_MAX_EPT = 64  # the top-k's register tile per thread (its MAX_EPT; pinned equal in the test module)
TOPK_MAX_ROWS = 65535  # the top-k's grid.y bound (its MAX_ROWS; pinned equal in the test module)
_WARP = 32
_ELEMS_PER_ACCESS = 8  # bf16 / fp16 elements per 16-byte access (qk_norm_rope.ELEMS_PER_ACCESS; pinned equal in the test module)
_SCORE_ROW_ALIGN = 8  # n_blocks_max is a multiple of 8 fp32 (the scorer's row alignment; pinned equal in the test module)
_ROW_BYTES = INDEX_HEAD_DIM * 2  # one compressed row, bf16
_SLOT_BYTES = KEYS_PER_NTILE * _ROW_BYTES  # one n-tile of 8 rows in the ring: 2 KiB
_A_TILE_BYTES = MMA_M * _ROW_BYTES  # one m-tile of the padded query tile: 16 rows x 256 B
_SNEW_BYTES = _ROW_BYTES + 16  # the derive's completing row + its presence / column word (16-B aligned; qsa_score_paged.SNEW_ELEMS x 4)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _lanes_per_row(d: int) -> int:
    """``qk_norm_rope.lanes_per_row``: enough lanes for one 16-byte access each, capped at a warp (re-derived here so this module
    imports no kernel module; the test module pins the two equal)."""
    return min(_WARP, d // _ELEMS_PER_ACCESS)


def prepare_role_warps(t: int, *, h_q: int, h_kv: int, d: int, index_head_dim: int = INDEX_HEAD_DIM) -> Tuple[int, int]:
    """The prepare role's warp units for ``t`` tokens, ``(attention warps, raw-key warps)`` -- ``qsa_prepare.prepare_warp_counts`` with the
    indexer-QUERY arm dropped (``index_q=False``: the scorer role derives the query rows itself) and the raw-key / compress arm on
    (``index_raw=True``).  The same arithmetic the prepare repeats on device from its runtime token count."""
    rpw_a = _WARP // _lanes_per_row(d)
    rpw_i = _WARP // _lanes_per_row(index_head_dim)
    return _ceil_div(t * (h_q + 2 * h_kv), rpw_a), _ceil_div(t, rpw_i)


def scorer_chunks(n_blocks_max: int, units: int, units_per_warp: int = 1) -> int:
    """R1 CTAs per sequence: ``ceil(n_blocks_max / (blocks_per_cta(units_per_warp) x units))`` -- the standalone scorer's chunk count
    folded ``units`` chunks per CTA (unit ``u`` of CTA ``k`` runs the standalone body for chunk ``k x units + u``)."""
    return _ceil_div(int(n_blocks_max), blocks_per_cta(units_per_warp) * int(units))


def topk_cols_cap(n_blocks_max: int, threads: int) -> int:
    """``qsa_topk_decode.cols_cap_for(n_blocks_max, 1, threads)`` re-derived here (no kernel module imported): the smallest power of two serving
    ``n_blocks_max`` columns, at least one element per thread (pinned equal in the test module)."""
    if n_blocks_max < 1:
        raise ValueError(f"n_blocks_max must be >= 1, got {n_blocks_max}")
    cap = 1
    while cap < n_blocks_max:
        cap *= 2
    return max(cap, int(threads))


def scorer_smem_bytes(units: int, s_q: int, h_i: int, *, ring_depth: int = NTILES_IN_FLIGHT, units_per_warp: int = 1) -> int:
    """The R1 CTA's SMEM: ``units`` rings of ``4 warps x ring_slots x 2 KiB`` (the standalone scorer's ``sK`` per unit) + ONE query tile
    ``m_tiles x 4 KiB`` shared by the units + the derive's ``sNew`` row (256 B + 16 B).  ``ring_slots = min(ring_depth + 1, n_tiles)``
    with ``n_tiles = 4 x units_per_warp`` (the scorer's rule)."""
    n_tiles = NTILES_PER_UNIT * int(units_per_warp)
    ring_slots = min(int(ring_depth) + 1, n_tiles)
    return int(units) * (SCORER_UNIT_THREADS // _WARP) * ring_slots * _SLOT_BYTES + query_row_map(s_q, h_i).m_tiles * _A_TILE_BYTES + _SNEW_BYTES


def dynamic_smem_bytes(scorer_bytes: int, topk_bytes: int) -> int:
    """The fused kernel's ONE dynamic allocation: the MAX over the compiled roles (the prepare needs 0).  ``topk_bytes`` is
    ``qsa_topk_decode.smem_bytes_for(cols_cap, 1, threads, top_k)`` (a kernel-module helper: the caller hands it in so this module
    imports no kernel module)."""
    return max(int(scorer_bytes), int(topk_bytes), 0)


class DispatchTable(NamedTuple):
    """The CTA ranges of one fused launch (plan constants handed to the kernel as scalars).

    ``n_prep_ctas`` R0 CTAs first, then per sequence ``n_chunks`` R1 CTAs followed by ``s_q`` R2 CTAs (``order = "sequence-major"``, the
    served order), or -- the host pin's twin -- the ``s_q`` R2 CTAs AHEAD of the ``n_chunks`` R1 CTAs inside every sequence's range
    (``"consumers-first"``), or -- the trap test's twin -- ALL ``T`` R2 CTAs ahead of ALL ``B x n_chunks`` R1 CTAs
    (``"consumers-first-global"``).  ``units`` is the scorer units per R1 CTA, ``threads`` the block size of every CTA."""

    n_prep_ctas: int
    n_chunks: int
    units: int
    s_q: int
    batch: int
    threads: int
    order: str = "sequence-major"

    @property
    def n_rows(self) -> int:
        return self.batch * self.s_q

    @property
    def per_sequence(self) -> int:
        """CTAs per sequence: its scorer chunks + its top-k rows."""
        return self.n_chunks + self.s_q

    @property
    def n_scorer_ctas(self) -> int:
        return self.batch * self.n_chunks

    @property
    def n_topk_ctas(self) -> int:
        return self.n_rows

    @property
    def n_ctas(self) -> int:
        """The grid: ``(n_ctas, 1, 1)``."""
        return self.n_prep_ctas + self.batch * self.per_sequence

    def waves(self, sm_count: int) -> int:
        """Waves at ONE CTA per SM (registers: ``threads x 64`` = the file at 1024 threads)."""
        return _ceil_div(self.n_ctas, int(sm_count))

    def linear_index(self, role: int, b: int, k: int) -> int:
        """The linear ``blockIdx.x`` of CTA ``k`` of role ``role`` (R0: ``k`` in ``[0, n_prep_ctas)``, ``b`` ignored; R1: chunk ``k`` of
        sequence ``b``; R2: row ``s = k`` of sequence ``b``, the top-k row ``b x s_q + k``)."""
        if role == ROLE_PREPARE:
            if not 0 <= k < self.n_prep_ctas:
                raise ValueError(f"prepare CTA {k} outside [0, {self.n_prep_ctas})")
            return k
        if not 0 <= b < self.batch:
            raise ValueError(f"sequence {b} outside [0, {self.batch})")
        base = self.n_prep_ctas + b * self.per_sequence
        if role == ROLE_SCORER:
            if not 0 <= k < self.n_chunks:
                raise ValueError(f"scorer chunk {k} outside [0, {self.n_chunks})")
            if self.order == "consumers-first-global":
                return self.n_prep_ctas + self.n_rows + b * self.n_chunks + k
            return base + (k if self.order == "sequence-major" else self.s_q + k)
        if role == ROLE_TOPK:
            if not 0 <= k < self.s_q:
                raise ValueError(f"top-k row {k} outside [0, {self.s_q})")
            if self.order == "consumers-first-global":
                return self.n_prep_ctas + b * self.s_q + k
            return base + (self.n_chunks + k if self.order == "sequence-major" else k)
        raise ValueError(f"role must be one of {ROLE_NAMES} (0 / 1 / 2), got {role}")

    def role_of(self, bidx: int) -> Tuple[int, int, int]:
        """The device decode: ``(role, b, k)`` of linear index ``bidx`` (the inverse of :meth:`linear_index`)."""
        if not 0 <= bidx < self.n_ctas:
            raise ValueError(f"blockIdx.x {bidx} outside the grid [0, {self.n_ctas})")
        if bidx < self.n_prep_ctas:
            return ROLE_PREPARE, 0, bidx
        r = bidx - self.n_prep_ctas
        if self.order == "consumers-first-global":
            if r < self.n_rows:
                return (ROLE_TOPK,) + divmod(r, self.s_q)
            return (ROLE_SCORER,) + divmod(r - self.n_rows, self.n_chunks)
        b, k = divmod(r, self.per_sequence)
        if self.order == "sequence-major":
            return (ROLE_SCORER, b, k) if k < self.n_chunks else (ROLE_TOPK, b, k - self.n_chunks)
        return (ROLE_TOPK, b, k) if k < self.s_q else (ROLE_SCORER, b, k - self.s_q)

    def ranges(self) -> List[Tuple[int, int, int, int]]:
        """``(role, b, start, stop)`` per contiguous range, in linear order."""
        out = [(ROLE_PREPARE, 0, 0, self.n_prep_ctas)]
        if self.order == "consumers-first-global":
            out += [(ROLE_TOPK, b, self.n_prep_ctas + b * self.s_q, self.n_prep_ctas + (b + 1) * self.s_q) for b in range(self.batch)]
            first = self.n_prep_ctas + self.n_rows
            out += [(ROLE_SCORER, b, first + b * self.n_chunks, first + (b + 1) * self.n_chunks) for b in range(self.batch)]
            return out
        for b in range(self.batch):
            base = self.n_prep_ctas + b * self.per_sequence
            if self.order == "sequence-major":
                out.append((ROLE_SCORER, b, base, base + self.n_chunks))
                out.append((ROLE_TOPK, b, base + self.n_chunks, base + self.per_sequence))
            else:
                out.append((ROLE_TOPK, b, base, base + self.s_q))
                out.append((ROLE_SCORER, b, base + self.s_q, base + self.per_sequence))
        return out

    def producers_before_consumers(self) -> bool:
        """The forward-progress placement: for every sequence ``b`` and every row ``s`` of it, EVERY producer of ``cnt_row[b s_q + s]``
        (the ``n_chunks`` R1 CTAs of ``b``) has a lower linear index than its consumer (the R2 CTA of the row); R0 and R1 wait on
        nothing of this grid (under ``use_pdl`` every role waits on the PREVIOUS grid, once, at its top).  True for the served order,
        False for the mis-ordered twin."""
        for b in range(self.batch):
            last_producer = max(self.linear_index(ROLE_SCORER, b, k) for k in range(self.n_chunks))
            for s in range(self.s_q):
                if not last_producer < self.linear_index(ROLE_TOPK, b, s):
                    return False
        return True

    def first_violation(self) -> Optional[Tuple[int, int, int, int]]:
        """``(b, s, producer_index, consumer_index)`` of the first consumer placed at or before one of its producers, or None."""
        for b in range(self.batch):
            for s in range(self.s_q):
                consumer = self.linear_index(ROLE_TOPK, b, s)
                for k in range(self.n_chunks):
                    producer = self.linear_index(ROLE_SCORER, b, k)
                    if producer >= consumer:
                        return b, s, producer, consumer
        return None


def dispatch_table(
    t: int,
    b: int,
    s_q: int,
    h_q: int,
    h_kv: int,
    index_heads: int,
    n_blocks_max: int,
    units: int,
    threads: int,
    *,
    d: int = 256,
    index_head_dim: int = INDEX_HEAD_DIM,
    units_per_warp: int = 1,
    order: str = "sequence-major",
) -> DispatchTable:
    """The dispatch table of one fused launch from the three standalone kernels' grid arithmetic (every refusal a ``ValueError``
    naming the rule, never an ``assert``: the block's ``check_support`` reads them).

    * R0: ``ceil((attn + raw) / (threads / 32))`` CTAs with :func:`prepare_role_warps` (the prepare's warp units at ``threads`` per CTA
      instead of its standalone 128);
    * R1: :func:`scorer_chunks` CTAs per sequence (the scorer's ``ceil(n_blocks_max / blocks_per_cta)`` chunks, ``units`` per CTA);
    * R2: ``s_q`` CTAs per sequence = the top-k's ``(1, T)`` grid, one CTA per row, at ``threads`` (the single-CTA form only).
    """
    t, b, s_q, h_q, h_kv, index_heads, n_blocks_max, units, threads = (int(x) for x in (t, b, s_q, h_q, h_kv, index_heads, n_blocks_max, units, threads))
    if b < 1 or s_q < 1 or t != b * s_q:
        raise ValueError(f"t must equal b x s_q with b, s_q >= 1 (the fused step is B x S_q rows), got t = {t}, b = {b}, s_q = {s_q}")
    if h_q < 1 or h_kv < 1:
        raise ValueError(f"h_q and h_kv must be >= 1, got {h_q} / {h_kv}")
    if d < _ELEMS_PER_ACCESS or d % _ELEMS_PER_ACCESS:
        raise ValueError(f"d must be a positive multiple of {_ELEMS_PER_ACCESS} (one 16-byte access per lane), got {d}")
    if index_head_dim != INDEX_HEAD_DIM:
        raise ValueError(f"the paged scorer addresses index_head_dim = {INDEX_HEAD_DIM} only, got {index_head_dim}")
    if index_heads < 1 or s_q * query_row_map(s_q, index_heads).padded > MAX_QUERY_ROWS:
        raise ValueError(
            f"s_q x next_pow2(index_heads) = {s_q} x {query_row_map(s_q, max(index_heads, 1)).padded} exceeds the scorer's {MAX_QUERY_ROWS}-row query envelope"
        )
    if n_blocks_max < 1 or n_blocks_max % _SCORE_ROW_ALIGN:
        raise ValueError(f"n_blocks_max must be a positive multiple of {_SCORE_ROW_ALIGN} (the scorer's fp32 row alignment), got {n_blocks_max}")
    if units not in SCORER_UNIT_CHOICES:
        raise ValueError(f"units must be one of {SCORER_UNIT_CHOICES} (4-warp scorer units per CTA), got {units}")
    if units_per_warp not in UNITS_PER_WARP_CHOICES:
        raise ValueError(f"units_per_warp must be one of {UNITS_PER_WARP_CHOICES}, got {units_per_warp}")
    if threads not in FUSED_THREAD_COUNTS:
        raise ValueError(f"threads must be one of {FUSED_THREAD_COUNTS} (the top-k's single-CTA forms), got {threads}")
    if units * SCORER_UNIT_THREADS > threads:
        raise ValueError(
            f"units x {SCORER_UNIT_THREADS} = {units * SCORER_UNIT_THREADS} scorer threads exceed the block size {threads}: at most {threads // SCORER_UNIT_THREADS} units at {threads} threads"
        )
    cap = topk_cols_cap(n_blocks_max, threads)
    if cap // threads > TOPK_MAX_EPT:
        raise ValueError(
            f"the top-k's single-CTA form holds at most {TOPK_MAX_EPT} keys per thread: cols_cap {cap} / {threads} threads = {cap // threads} -- "
            f"n_blocks_max {n_blocks_max} is past the fused kernel's reach at {threads} threads (the chain with the cluster top-k serves it)"
        )
    if t > TOPK_MAX_ROWS:
        raise ValueError(f"the top-k addresses at most {TOPK_MAX_ROWS} rows per launch, got T = {t}")
    if order not in DISPATCH_ORDERS:
        raise ValueError(f"order must be one of {DISPATCH_ORDERS}, got {order!r}")
    attn, raw = prepare_role_warps(t, h_q=h_q, h_kv=h_kv, d=d, index_head_dim=index_head_dim)
    n_prep = _ceil_div(attn + raw, threads // _WARP)
    return DispatchTable(n_prep, scorer_chunks(n_blocks_max, units, units_per_warp), units, s_q, b, threads, order)


ROLES = ("prepare", "scorer", "topk")
"""The served rendering carries all three roles; a subset is the test-only role-isolated rendering (the same ABI, block and SMEM)."""
_STREAM_NONE = object()


class QsaIndexerFusedRecipe(NamedTuple):
    """Build-time facts of one fused artifact: everything that is a compile key (the three kernels' own keys + the fused kernel's), the
    dynamic SMEM figure, the device.  The batch, the KV lengths, the table widths, the page counts and every stride ride in at runtime: one
    artifact serves every step of its declaration at one ``(units, threads)``."""

    compiled: object
    key: tuple  # the kernel module's FusedKey (the compile key, plain values)
    dtype: object
    slot_dtype: object
    apply_norm: bool
    index: object  # QsaPrepareIndexArms
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    page_size: int
    s_q: int
    units: int
    threads: int
    units_per_warp: int
    ring_depth: int
    has_block_lens: bool
    cols_cap: int
    top_k: int
    use_pdl: bool
    device: int
    smem_bytes: int
    check_arrivals: bool
    with_deadline: bool
    roles: tuple
    order: str


def _kernel_module():
    """The kernel half, imported lazily (it imports the CuTe DSL): every caller gated the floor and the target first (Rule 7)."""
    from . import qsa_indexer_fused_kernel as km

    return km


def compile_qsa_indexer_fused(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    page_size: int,
    index,
    s_q: int,
    n_blocks_max: int,
    top_k: int,
    units: int,
    threads: int = DEFAULT_THREADS,
    slot_dtype=torch.int32,
    apply_norm: bool = True,
    units_per_warp: int = 1,
    ring_depth: int = NTILES_IN_FLIGHT,
    has_block_lens: bool = True,
    use_pdl: bool = False,
    device: Optional[int] = None,
    const_head_counts: bool = True,
    index_cache_dtype=None,
    deadline_ns: Optional[int] = None,
    check_arrivals: bool = False,
    roles: Tuple[str, ...] = ROLES,
    order: str = "sequence-major",
    scorer_mma_batch: Optional[int] = None,
) -> QsaIndexerFusedRecipe:
    """Build the fused artifact from SHAPES ALONE (no allocation, no launch): traces ``frost_qsa_indexer_fused`` for one plan.

    Compile keys (every one a trace-time constant): the prepare's ``QsaPrepareIndexArms`` (``query=False, compress=True, paged=True``,
    ``ring_slots`` / ``rope_table`` / ``n_commit`` as the plan has them) + ``d, rope_dim, page, index_head_dim, pool``, ``dtype`` /
    ``slot_dtype`` / ``apply_norm`` / ``eps``; the scorer's ``(h_i, s_q, page, units_per_warp, ring_depth, has_block_lens x 2)``; the top-k's
    ``(cols_cap, top_k)`` with ``cols_cap = topk_cols_cap(n_blocks_max, threads)`` (one artifact serves every ``n_blocks_max`` up to it);
    ``units``, ``threads`` (1024 / 512), ``use_pdl``, the device; the test-only ``deadline_ns`` (the trap rendering: a ``%globaltimer`` deadline
    on the top-k's acquire, the launch then takes the deadline and a diagnostic buffer), ``check_arrivals`` (the top-k role's exact-count
    trap before its counter reset), ``roles`` (a role-isolated rendering for the per-role register census) and ``order``
    (``"consumers-first"`` / ``"consumers-first-global"``: the mis-ordered dispatch tables, served only with the deadline).  The dynamic SMEM = :func:`dynamic_smem_bytes`
    over the compiled roles, checked against the device's ceiling -- the standard opt-in limit, or the oversized carveout where the part
    has one (a figure above the standard carveout names it; past both the declaration is declined by name).  Rule 7's facade gates (the
    DSL floor, the ``sm_107a`` target on cc 10.7 -- the public wheel carries it from 4.8.0 --, cc 9.0 for PDL) decline by name before any
    import of the kernel modules; the geometry refusals are :func:`dispatch_table`'s ``ValueError`` s."""
    too_old = cutedsl_requirement_error("qsa_indexer_fused")
    if too_old is not None:
        raise NotImplementedError(too_old)
    from cudnn.frost.device import compute_capability, current_device, oversized_shared_memory_per_block, shared_memory_per_block_optin
    from cudnn.frost.buffers import cutedsl_arch_requirement_error

    h_q, h_kv, d, rope_dim, page_size, s_q, n_blocks_max, top_k, units, threads = (
        int(x) for x in (h_q, h_kv, d, rope_dim, page_size, s_q, n_blocks_max, top_k, units, threads)
    )
    if index is None or not getattr(index, "compress", False) or not getattr(index, "paged", False):
        raise ValueError(
            "qsa_indexer_fused: index must be a QsaPrepareIndexArms with compress=True and paged=True (the decode step over the stack's paged ring and compressed-key cache)"
        )
    if getattr(index, "query", False):
        raise ValueError(
            "qsa_indexer_fused: index.query must be False -- the scorer role derives the indexer-query rows itself (nobody reads ix_q under the fused form)"
        )
    if dtype != torch.bfloat16:
        raise NotImplementedError(f"qsa_indexer_fused serves bf16 activations (the scorer packs bf16 queries and compressed keys), got {dtype}")
    if index_cache_dtype is not None and index_cache_dtype != dtype:
        raise NotImplementedError(f"qsa_indexer_fused: the compressed-key cache is the activation dtype ({dtype}); an {index_cache_dtype} cache is a follow-up")
    if slot_dtype not in (torch.int32, torch.int64):
        raise ValueError(f"slot_mapping must be int32 or int64, got {slot_dtype}")
    if rope_dim < 1:
        raise NotImplementedError(
            "qsa_indexer_fused: rope_dim must be > 0 (the derive rotates the indexer rows at their positions from the compress arm's tables)"
        )
    if s_q > int(index.pool):
        raise NotImplementedError(
            f"qsa_indexer_fused: S_q = {s_q} rows per sequence exceed the compress ratio pool = {index.pool}: the derive publishes the ONE compressed-key row "
            "a step of at most `pool` rows can complete (a wider step completes several); the three-launch chain serves it"
        )
    if not isinstance(roles, (tuple, list)) or not roles or any(r not in ROLES for r in roles):
        raise ValueError(f"roles must be a non-empty subset of {ROLES}, got {roles!r}")
    roles = tuple(r for r in ROLES if r in roles)
    if order not in DISPATCH_ORDERS:
        raise ValueError(f"order must be one of {DISPATCH_ORDERS}, got {order!r}")
    if order != "sequence-major" and deadline_ns is None:
        raise ValueError(
            f"order={order!r} is a trap rendering's mis-ordered dispatch table: it is served only with deadline_ns (a lost dependency must trap, never hang)"
        )
    if deadline_ns is not None and int(deadline_ns) < 1:
        raise ValueError(f"deadline_ns must be a positive nanosecond count, got {deadline_ns!r}")
    tb = dispatch_table(
        s_q,
        1,
        s_q,
        h_q,
        h_kv,
        int(index.heads),
        n_blocks_max,
        units,
        threads,
        d=d,
        index_head_dim=int(index.head_dim),
        units_per_warp=units_per_warp,
        order=order,
    )
    validate_params(h_i=int(index.heads), s_q=s_q, page_size=page_size, units_per_warp=units_per_warp, ring_depth=ring_depth)
    dev = current_device() if device is None else int(device)
    cc = compute_capability(dev)
    arch_problem = cutedsl_arch_requirement_error(cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if cc < (9, 0):
        raise NotImplementedError(
            f"qsa_indexer_fused needs a cc 9.0+ device: its per-row counters elect their signalling lane with elect.sync (and PDL uses griddepcontrol); "
            f"found cc {cc[0]}.{cc[1]} -- the three-launch chain serves this device"
        )
    km = _kernel_module()
    from .qsa_prepare import validate_prepare_shape
    from .qsa_score_paged import l2_hint_available
    from .qsa_topk_decode import smem_bytes_for, validate_topk_geometry

    validate_prepare_shape(d, rope_dim, page_size, threads, int(index.head_dim), int(index.pool))
    cols_cap = topk_cols_cap(n_blocks_max, threads)
    validate_topk_geometry(cols_cap, top_k, 1, threads)
    topk_bytes = int(smem_bytes_for(cols_cap, 1, threads, top_k)) if "topk" in roles else 0
    scorer_bytes = scorer_smem_bytes(units, s_q, int(index.heads), ring_depth=ring_depth, units_per_warp=units_per_warp) if "scorer" in roles else 0
    smem_bytes = dynamic_smem_bytes(scorer_bytes, topk_bytes)
    optin = int(shared_memory_per_block_optin(dev))
    oversized = int(oversized_shared_memory_per_block(dev))
    if smem_bytes > max(optin, oversized):
        raise NotImplementedError(
            f"qsa_indexer_fused at units = {units}, ring_depth = {ring_depth}, cols_cap = {cols_cap} needs {smem_bytes} B of dynamic shared memory per CTA; "
            f"the device allows {optin} B (opt-in){f' / {oversized} B (the oversized carveout)' if oversized else ''} -- fewer units or ring_depth 2"
        )
    key = km.FusedKey(
        dtype=str(dtype),
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        rope_dim=rope_dim,
        eps=float(eps),
        page_size=page_size,
        slot_dtype=str(slot_dtype),
        apply_norm=bool(apply_norm),
        index=tuple(index),
        s_q=s_q,
        units=units,
        threads=threads,
        units_per_warp=int(units_per_warp),
        ring_depth=int(ring_depth),
        scorer_mma_batch=scorer_mma_batch_for(threads, units_per_warp) if scorer_mma_batch is None else int(scorer_mma_batch),
        has_block_lens=bool(has_block_lens),
        cols_cap=int(cols_cap),
        top_k=top_k,
        use_pdl=bool(use_pdl),
        const_head_counts=bool(const_head_counts),
        smem_bytes=int(smem_bytes),
        check_arrivals=bool(check_arrivals),
        with_deadline=deadline_ns is not None,
        consumers_first=order == "consumers-first",
        roles=roles,
        device=dev,
        consumers_global=order == "consumers-first-global",
        scorer_l2_hint=l2_hint_available(),  # the scorer's key copies: the hinted form where the DSL's compiler serves it (the standalone scorer's default arm)
    )
    compiled = km.compile_fused(key)
    return QsaIndexerFusedRecipe(
        compiled=compiled,
        key=tuple(key),
        dtype=dtype,
        slot_dtype=slot_dtype,
        apply_norm=bool(apply_norm),
        index=index,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        rope_dim=rope_dim,
        eps=float(eps),
        page_size=page_size,
        s_q=s_q,
        units=units,
        threads=threads,
        units_per_warp=int(units_per_warp),
        ring_depth=int(ring_depth),
        has_block_lens=bool(has_block_lens),
        cols_cap=int(cols_cap),
        top_k=top_k,
        use_pdl=bool(use_pdl),
        device=dev,
        smem_bytes=int(smem_bytes),
        check_arrivals=bool(check_arrivals),
        with_deadline=deadline_ns is not None,
        roles=roles,
        order=order,
    )


def _check_counter_words(cnt_row, t: int, device) -> None:
    """The plan-owned counter words: int64, 1-D, contiguous, on the artifact's device, at least ``T`` words, 8-B aligned."""
    if not isinstance(cnt_row, torch.Tensor) or cnt_row.dtype != torch.int64:
        raise ValueError(f"cnt_row must be an int64 tensor (one u64 counter word per query row), got {getattr(cnt_row, 'dtype', type(cnt_row).__name__)}")
    if cnt_row.dim() != 1 or not cnt_row.is_contiguous() or not cnt_row.is_cuda or cnt_row.device != device:
        raise ValueError(
            f"cnt_row must be a contiguous 1-D CUDA tensor on {device}, got shape {tuple(cnt_row.shape)} strides {tuple(cnt_row.stride())} on {cnt_row.device}"
        )
    if int(cnt_row.numel()) < t:
        raise ValueError(f"cnt_row holds {int(cnt_row.numel())} words; the step has T = {t} rows (one word per row)")
    if cnt_row.data_ptr() % 8:
        raise ValueError("cnt_row must be 8-byte aligned")


def run_qsa_indexer_fused(
    r: QsaIndexerFusedRecipe,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: Optional[torch.Tensor],
    w_k: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_k_raw_src: torch.Tensor,
    index_q_src: torch.Tensor,
    w_iq_norm: torch.Tensor,
    ring: torch.Tensor,
    index_k_compressed: torch.Tensor,
    w_ik_norm: torch.Tensor,
    cos_pos: torch.Tensor,
    sin_pos: torch.Tensor,
    compressed_slot: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    index_state_block_table: torch.Tensor,
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_lens: Optional[torch.Tensor],
    block_ids: torch.Tensor,
    cnt_row: torch.Tensor,
    scale: float,
    stream,
    index_k_raw_pool: Optional[torch.Tensor] = None,
    index_state_slot_mapping: Optional[torch.Tensor] = None,
    n_commit: Optional[torch.Tensor] = None,
    index_eps: float = 1e-6,
    index_norm_weight_offset: float = 0.0,
    deadline_ns: int = 0,
    diag: Optional[torch.Tensor] = None,
) -> None:
    """ONE launch binding the slab band (``q`` / ``k`` / ``v`` rewritten in place, ``index_k_raw_src`` the raw indexer key, ``index_q_src`` the
    indexer-query heads), the pools (``k_cache`` / ``v_cache`` at ``slot_mapping``; ``index_k_raw_pool`` iff the arms trace it), the ring and the
    compressed-key cache (``ring`` / ``index_k_compressed``), the tables (``cos`` / ``sin`` per-token rows, or ``None`` under the RoPE-table
    arm; ``cos_pos`` / ``sin_pos`` position-indexed; ``w_q`` / ``w_k`` / ``w_iq_norm`` / ``w_ik_norm``), the workspace views (``logits``
    ``[T, n_blocks_max]`` fp32, ``counts`` ``[T]``, ``block_lens`` ``[T]`` iff traced, ``block_ids`` ``[T, top_k]`` -- the caller's list --,
    ``compressed_slot`` ``[B, S_q]``), ``kv_lens``, ``block_table`` / ``index_state_block_table`` / ``index_state_slot_mapping`` (iff traced),
    ``n_commit`` (iff traced), the plan-owned counters (``cnt_row``: int64 ``[T]`` zero before the first launch and zero again after every
    complete one; a short or wrong-dtype buffer is a typed ``ValueError``), the scorer's ``scale`` and the stream (a raw CUDA stream handle);
    the grid is ``(DispatchTable.n_ctas, 1, 1)`` at ``threads``.  The trap rendering (``deadline_ns`` at compile) takes ``deadline_ns``
    (nanoseconds, > 0) and ``diag`` (int64 ``[2]``, CUDA or pinned host memory) at every launch; a production recipe refuses both.  Every
    operand's FORM is checked here with the three kernels' own checks (no device read, no conversion, no allocation)."""
    import cuda.bindings.driver as cuda

    from .qk_norm_rope import check_norm_weights_match_recipe
    from .qsa_compress import _check_int32_vector, _check_row_tensor, check_paged_state, check_paged_step_operands, min_ring_rows
    from .qsa_prepare import _check_band
    from .qsa_topk_decode import _check_rows
    from ..cache_write_contract import check_pool, check_slot_mapping

    km = _kernel_module()
    ix = r.index
    d_i, h_i, pool = int(ix.head_dim), int(ix.heads), int(ix.pool)
    if not isinstance(q, torch.Tensor) or q.dim() != 3 or int(q.shape[0]) < 1:
        raise ValueError("q must be a [T >= 1, H_q, D] band view")
    t = int(q.shape[0])
    if t % r.s_q:
        raise ValueError(f"the step holds T = {t} rows, not a multiple of the artifact's S_q = {r.s_q}")
    batch = t // r.s_q
    device = q.device
    if device.index is not None and device.index != r.device:
        raise ValueError(f"this artifact was compiled for CUDA device {r.device}; the operands live on {device} (compile one per device)")
    _check_band("q", q, t=t, h=r.h_q, d=r.d, dtype=r.dtype)
    _check_band("k", k, t=t, h=r.h_kv, d=r.d, dtype=r.dtype)
    _check_band("v", v, t=t, h=r.h_kv, d=r.d, dtype=r.dtype)
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.apply_norm:
        for nm, w in (("w_q", w_q), ("w_k", w_k)):
            if w.dim() != 1 or int(w.shape[0]) != r.d or not w.is_contiguous():
                raise ValueError(f"{nm} must be a contiguous [{r.d}] norm weight, got shape {tuple(w.shape)}")
            _check_row_tensor(nm, w, dtype=r.dtype, last_dim=r.d)
    if ix.rope_table:
        if cos is not None or sin is not None:
            raise ValueError("cos / sin are refused on a recipe that traced the RoPE-table arm (index.rope_table=True): pass cos=None, sin=None")
    else:
        if cos is None or sin is None:
            raise ValueError(f"cos / sin [T={t}, {r.rope_dim}] per-token tables are required (this recipe traced no RoPE-table arm)")
        for nm, tb_ in (("cos", cos), ("sin", sin)):
            if tb_.dim() != 2 or int(tb_.shape[0]) != t or not tb_.is_contiguous():
                raise ValueError(f"{nm} must be a contiguous [T={t}, {r.rope_dim}] per-token table, got shape {tuple(tb_.shape)} strides {tuple(tb_.stride())}")
            _check_row_tensor(nm, tb_, dtype=r.dtype, last_dim=r.rope_dim)
    check_pool(k_cache, "k_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=r.dtype)
    check_pool(v_cache, "v_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=r.dtype)
    check_slot_mapping(slot_mapping, "slot_mapping", t=t)
    if slot_mapping.dtype != r.slot_dtype:
        raise ValueError(f"slot_mapping is {slot_mapping.dtype} but this artifact was compiled for {r.slot_dtype}; the slot dtype is fixed per artifact")
    n_slots = min(int(k_cache.shape[0]), int(v_cache.shape[0])) * r.page_size
    _check_band("index_k_raw_src", index_k_raw_src, t=t, h=1, d=d_i, dtype=r.dtype)
    _check_band("index_q_src", index_q_src, t=t, h=h_i, d=d_i, dtype=r.dtype)
    if (index_k_raw_pool is not None) != bool(ix.raw_pool):
        raise ValueError(
            f"this artifact was compiled {'with' if ix.raw_pool else 'without'} the raw key's pool arm; index_k_raw_pool must match (no silent fallback)"
        )
    n_islots = 0
    if index_k_raw_pool is not None:
        if index_k_raw_pool.dim() != 3:
            raise ValueError(f"index_k_raw_pool must be a 3-D [num_pages, {r.page_size}, {d_i}] pool, got shape {tuple(index_k_raw_pool.shape)}")
        check_pool(index_k_raw_pool.unsqueeze(1), "index_k_raw_pool", h=1, page_size=r.page_size, d=d_i, dtype=r.dtype)
        n_islots = int(index_k_raw_pool.shape[0]) * r.page_size
    for nm, w in (("w_iq_norm", w_iq_norm), ("w_ik_norm", w_ik_norm)):
        if not isinstance(w, torch.Tensor) or w.dim() != 1 or int(w.shape[0]) != d_i or not w.is_contiguous():
            raise ValueError(f"{nm} must be a contiguous [{d_i}] norm weight, got {getattr(w, 'shape', type(w).__name__)}")
        _check_row_tensor(nm, w, dtype=r.dtype, last_dim=d_i)
    need = min_ring_rows(r.s_q, pool)
    check_paged_state(
        "ring",
        ring,
        dtype=r.dtype,
        d=d_i,
        min_rows=need,
        what=f"raw-key ring: a step of {r.s_q} rows needs rows + pool - 1 = {r.s_q} + {pool} - 1 so this launch's ring writes never alias its ring reads (the derive's included)",
    )
    check_paged_state(
        "index_k_compressed", index_k_compressed, dtype=r.dtype, d=d_i, min_rows=1, what="compressed-key cache, P / pool rows per attention page", cache=True
    )
    for nm, tb_ in (("cos_pos", cos_pos), ("sin_pos", sin_pos)):
        if not isinstance(tb_, torch.Tensor) or tb_.dim() != 3 or int(tb_.shape[0]) not in (1, batch) or int(tb_.shape[1]) < 1:
            raise ValueError(f"{nm} must be [1 or B={batch}, S_pos >= 1, {r.rope_dim}] (position-indexed), got {getattr(tb_, 'shape', type(tb_).__name__)}")
        _check_row_tensor(nm, tb_, dtype=r.dtype, last_dim=r.rope_dim)
    if tuple(cos_pos.shape) != tuple(sin_pos.shape):
        raise ValueError(f"cos_pos and sin_pos must share a shape, got {tuple(cos_pos.shape)} and {tuple(sin_pos.shape)}")
    check_paged_step_operands(
        paged=True,
        ring_slots=bool(ix.ring_slots),
        ring_slot_dtype=r.slot_dtype,
        batch=batch,
        rows=r.s_q,
        pos0=None,
        kv_lens=kv_lens,
        block_table=block_table,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
        ring=ring,
    )
    if ix.rope_table:
        n_pos = int(block_table.shape[1]) * int(index_k_compressed.shape[1]) * pool
        if int(cos_pos.shape[1]) < n_pos:
            raise ValueError(
                f"cos_pos / sin_pos hold {int(cos_pos.shape[1])} positions, but the RoPE-table arm rotates every row of the step at its position and the main "
                f"block table addresses {int(block_table.shape[1])} pages x {int(index_k_compressed.shape[1]) * pool} tokens: hand a table of at least {n_pos} rows"
            )
    _check_int32_vector("compressed_slot", compressed_slot, (batch, r.s_q))
    if (n_commit is not None) != bool(ix.n_commit):
        raise ValueError(
            f"this artifact was compiled {'with' if ix.n_commit else 'without'} the accepted-prefix load; n_commit must match (no silent fallback)"
        )
    if n_commit is not None:
        _check_int32_vector("n_commit", n_commit, (batch,))
    # the scorer's operands
    rows_per_page = r.page_size // BLOCK_SIZE
    kbar_view = _kbar_rows_view(index_k_compressed, rows_per_page, device)
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.dtype != torch.float32:
        raise ValueError(
            f"logits must be an fp32 [T, n_blocks_max] tensor, got {getattr(logits, 'shape', type(logits).__name__)} {getattr(logits, 'dtype', '')}"
        )
    lt, n_blocks_max = (int(x) for x in logits.shape)
    if lt != t or n_blocks_max < 1 or n_blocks_max > r.cols_cap:
        raise ValueError(f"logits must be [T={t}, 1 <= n_blocks_max <= cols_cap={r.cols_cap}], got {tuple(logits.shape)}")
    ls0, ls1 = (int(x) for x in logits.stride())
    if ls1 != 1 or (ls0 * 4) % 16 or logits.data_ptr() % 16 or logits.device != device:
        raise ValueError(
            f"logits needs a unit column stride, 16-byte-aligned rows and base, on {device}; got strides {tuple(logits.stride())}, {logits.device}"
        )
    if int(block_table.shape[1]) * rows_per_page < n_blocks_max:
        raise ValueError(
            f"block_table maps max_pages x R = {int(block_table.shape[1])} x {rows_per_page} compressed rows, fewer than the n_blocks_max = {n_blocks_max} columns of logits"
        )
    _check_i32_1d(counts, "counts", t, device)
    if r.has_block_lens != (block_lens is not None):
        raise ValueError(f"this artifact was compiled {'with' if r.has_block_lens else 'without'} block_lens; the call must match (Rule 1: no silent fallback)")
    if block_lens is not None:
        _check_i32_1d(block_lens, "block_lens", t, device)
    _check_rows("block_ids", block_ids, dtype=torch.int32, rows=t, device=device)
    if int(block_ids.shape[1]) != r.top_k:
        raise ValueError(f"block_ids must be [T, top_k = {r.top_k}] for this artifact, got shape {tuple(block_ids.shape)}")
    if not float(scale) > 0.0:
        raise ValueError(f"scale must be positive (the ordering is scale-free), got {scale}")
    _check_counter_words(cnt_row, t, device)
    if r.with_deadline:
        if int(deadline_ns) < 1 or not isinstance(diag, torch.Tensor) or diag.dtype != torch.int64 or int(diag.numel()) < 2 or diag.data_ptr() % 8:
            raise ValueError(
                "the trap rendering takes deadline_ns > 0 and diag (an int64 tensor of at least two words, CUDA or pinned host memory) at every launch"
            )
        diag_addr = int(diag.data_ptr())
    else:
        if int(deadline_ns) or diag is not None:
            raise ValueError("deadline_ns / diag belong to the trap rendering (compile with deadline_ns); a production recipe takes neither")
        diag_addr = 0
    tb = dispatch_table(
        t, batch, r.s_q, r.h_q, r.h_kv, h_i, n_blocks_max, r.units, r.threads, d=r.d, index_head_dim=d_i, units_per_warp=r.units_per_warp, order=r.order
    )
    if r.order == "sequence-major" and not tb.producers_before_consumers():
        raise RuntimeError("qsa_indexer_fused: the dispatch table placed a consumer at or before one of its producers (the forward-progress pin)")
    args = [
        q,
        k,
        v,
        q,
        k,
        w_q,
        w_k,
        cos,
        sin,
        k_cache,
        v_cache,
        slot_mapping,
        index_k_raw_src,
        None if index_k_raw_pool is None else index_k_raw_pool.unsqueeze(1),
        ring,
        index_k_compressed,
        w_ik_norm,
        cos_pos,
        sin_pos,
        n_commit,
        compressed_slot,
        kv_lens,
        block_table,
        index_state_block_table,
        index_state_slot_mapping,
        index_q_src,
        w_iq_norm,
        kbar_view,
        logits,
        counts,
        block_lens,
        logits.view(torch.int32),
        block_ids,
        cutlass_int32(t),
        cutlass_int64(n_slots),
        cutlass_int64(n_islots),
        cutlass_int32(r.h_q),
        cutlass_int32(r.h_kv),
        cutlass_int32(h_i),
        cutlass_float32(r.eps),
        cutlass_float32(index_eps),
        cutlass_float32(index_norm_weight_offset),
        cutlass_float32(scale),
        cutlass_int32(tb.n_prep_ctas),
        cutlass_int32(tb.n_chunks),
        cutlass_int32(tb.n_ctas),
        cutlass_int64(int(cnt_row.data_ptr())),
        cutlass_int64(int(deadline_ns)),
        cutlass_int64(diag_addr),
        cuda.CUstream(int(stream)),
    ]
    km.launch_fused(r.compiled, args, device=r.device)


def cutlass_int32(x: int):
    import cutlass

    return cutlass.Int32(int(x))


def cutlass_int64(x: int):
    import cutlass

    return cutlass.Int64(int(x))


def cutlass_float32(x: float):
    import cutlass

    return cutlass.Float32(float(x))
