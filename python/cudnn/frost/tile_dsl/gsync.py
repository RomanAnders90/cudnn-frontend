# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Grid-scope dependency counters: release / acquire on 64-bit words in global memory, and the last-arriver election.

An mbarrier orders work inside one CTA (or one cluster).  A hand-over between CTAs that share no cluster -- the split
partials of one attention unit and the CTA that merges them, the split-K planes of one m-tile and the CTA that reduces
them, the tiles of one phase and the items of the next inside a persistent kernel -- is ordered through a COUNTER in
global memory, and every such site in a FROST kernel takes exactly one of the three forms below.  The forms are the
standard release / acquire pattern of the PTX memory model at ``.gpu`` scope (the same shape as CUTLASS's stream-K
``Barrier``: all threads publish, one lane signals, one lane polls, a CTA barrier spreads the result), with the fence
recipe of the FROST kernels baked in so no call site spells it by hand.

The three entries (every one is called by EVERY thread of the role -- all ``bar_threads`` of them):

``release_add(addr, n, *, warp_gate, bar_id, bar_threads, async_proxy=False)``
    RELEASE: [``fence.proxy.async.global`` by every thread, ``async_proxy`` only] -> ``fence.acq_rel.gpu`` by every thread
    (the writer's own generic stores become cumulative through the barrier below) -> ``bar.sync bar_id, bar_threads``
    (the role's warps) -> ONE lane: ``red.release.gpu.global.add.u64 [addr], n``.
``acquire_geq(addr, target, *, warp_gate, bar_id, bar_threads, backoff_ns=128, async_proxy=False, deadline_ns=None, ...)``
    ACQUIRE: ONE lane polls ``ld.acquire.gpu.global.u64 [addr]`` until the value is ``>= target`` (one tight test, then a
    TIMER ``nanosleep.u32 backoff_ns`` between tests -- the ``wait_poll`` shape of ``barrier.py``: the polling lane's warp
    siblings reconverge at the end of the gate region and the role's other warps park in the named barrier, so a poll
    never starves the SMSP siblings) -> [``fence.proxy.async.global`` by that warp, ``async_proxy`` only] ->
    ``bar.sync bar_id, bar_threads`` publishes the acquire to the role's other warps.
``last_arriver(addr, count, *, warp_gate, flag_smem, bar_id, bar_threads, reset=True, async_proxy=False) -> (is_last, old)``
    LAST ARRIVER: the RELEASE prefix above, then ONE lane: ``atom.acq_rel.gpu.global.add.u64 old, [addr], 1``; the
    lane stores ``old`` into ``flag_smem`` (one 8-byte SMEM word the caller owns), ``bar.sync`` broadcasts it, every
    thread derives the CTA-UNIFORM ``is_last``; the last arriver's threads execute ``fence.acq_rel.gpu`` before
    returning, so they may read every other arriver's payload through generic loads right away.  ``reset=True`` is the
    TARGET form: ``is_last = old == count - 1`` and the electing lane stores 0 back (``st.relaxed.gpu``) -- the plain
    store races with nothing because the atom's return value proves every one of the ``count`` arrivals on that word
    has happened; the next launch (stream-ordered, or behind ``griddepcontrol.wait`` under PDL) starts from zero.
    ``reset=False`` is the MODULUS form: the counter is never reset and ``is_last = (old + 1) % count == 0``; it
    needs no zeroing between launches at all and ``old // count`` numbers the launch.

SCOPES.  Every atom, load and fence is ``.gpu``; nothing here is ``.cluster`` scoped (a ``.release.cluster`` makes
ptxas drain the whole kernel at every site -- ``barrier.py`` ``arrive_on_leader_release``) and nothing is ``.sys``
except the ``fence.sc.sys`` that precedes a deadline trap (so a diagnostic written to mapped pinned host memory is
readable after the context is gone).  The async-proxy arms are inline PTX because the DSL's ``nvvm.fence_proxy``
wrapper carries the ``shared`` form only; they are needed exactly where the CONSUMER reads the payload through the
async proxy (a TMA load of a slab, of O, of a page another CTA wrote with generic stores): writer side in
``release_add`` / ``last_arriver`` (before the gpu fence), reader side in ``acquire_geq`` (after the acquire, on the
polling warp; a TMA-issuing warp other than the polling one fences itself after the named barrier --
:func:`fence_proxy_async_global`).  Generic-proxy consumers (the merge's ``ld.global`` of the partials, the reduce's
loads of the planes) need neither arm.

THE LANE LEDGER (write it in the kernel's barrier table, one row per counter class, exactly like an mbarrier's init
count): ``issuing lanes = ONE lane per CTA (the elected lane of the one warp whose threads pass warp_gate) x the
group's CTA count == the target`` -- e.g. ``cnt_unit[b, g]: 1 lane x S split CTAs == S``; ``cnt_p3[m]: 1 lane x
SLICES == SLICES``; a grid-wide rendezvous ``1 x gridDim.x == gridDim.x``.  ``warp_gate`` is a REQUIRED keyword argument
on every entry and must be a per-thread predicate that holds on the lanes of ONE warp (``warp_idx == 0``); the elect
is internal.  A call without the gate does not issue (a ``TypeError`` at trace time), a Python constant for the gate is
refused (a bare ``True`` across a multi-warp role is the P3 over-arrival of ``mbarrier-patterns.md``), so the
over-arrival that ``if nvvm.elect_sync():`` alone writes on an ``init(1)`` mbarrier cannot be written here.  The
named barrier (``bar_id`` 1-15, ``bar_threads`` a multiple of 32 = the role's thread count) is the caller's resource:
list it in the barrier table beside the kernel's other named barriers, and call the entry from every thread it counts.

LIFECYCLE.  The counter words live in the plan's workspace, one 128-byte line per counter class (a release atom must
never share a line with a polled word of another class; :func:`counter_word_offset` lays them out), are written to zero
on the launch stream by the plan's one-shot init at the first execute, and from then on the KERNEL leaves them at
zero: the target form resets by the last arriver, a persistent kernel's last-alive CTA stores 0 to every class it
owns at exit (:func:`counter_store_u64`).  A kernel that does not complete leaves a sticky context in which no later
launch of the plan runs, so no launch ever reads a residue.  Capture is unaffected: a captured first execute captures
the zero fill, every replay re-applies it, every complete launch leaves zeros.  Under PDL the first access to a counter
sits behind the role's ``griddepcontrol.wait`` -- that is the caller's placement, not this module's.

NO CO-RESIDENCY IS NEEDED by the last-arriver form and by any acquire whose producers never wait on the acquirer: an
ordinary launch of more CTAs than fit on the device completes it wave by wave (every arriver leaves; the last one does
the extra work).  Only a rendezvous in which every CTA acquires the others' releases needs all of them resident (a
cooperative launch, or a grid no larger than the occupancy probe allows); its poll must then carry the deadline.
Evidence (the probe of ``test_tile_dsl_gsync.py``): ORDINARY launches of 212 and 424 CTAs with a 209 KiB dynamic-SMEM
footprint (one CTA per SM, so 424 CTAs on a 204-SM cc 10.7 part are 2.08 waves) ran 1000 consecutive launches on ONE
counter buffer with every launch's ``old`` set a permutation of ``0 .. grid - 1`` and the counter back at zero after
each (the self-reset holds launch after launch), and the chained release / acquire form delivered every producer's
payload to its acquirer over the same 2 x 1000 launches; the test's device tier repeats those arms, the CUDA-graph
capture + replay, the co-resident rendezvous and the deadline trap at 100 launches per cell (green on a cc 10.0 part).

THE DEADLINE.  ``acquire_geq(..., deadline_ns=<Int64>, diag_addr=<Int64 or None>, code=<int>)`` arms a
``%globaltimer`` deadline on the poll: past it the lane stores ``code`` and the last value seen to ``diag_addr`` (two
u64 words; mapped pinned host memory keeps them readable after the context dies), executes ``fence.sc.sys`` and
``trap`` -- a lost dependency becomes a launch failure that NAMES its seam within the deadline instead of a hang.
``deadline_ns=None`` (the default) renders no timer, no diagnostic and no trap: production kernels of the ordinary,
co-residency-free forms stay on the default; debug builds and every rendezvous form arm it.  The deadline bounds the
spin, not the kernel: the outer run timeout keeps its own meaning (a kernel that does not return is still a hang).

Pinned by ``test_tile_dsl_gsync.py``: the emitted PTX carries ``red.release.gpu.global.add.u64``,
``ld.acquire.gpu.global.u64``, ``atom.acq_rel.gpu.global.add.u64``, ``fence.acq_rel.gpu``, the
``fence.proxy.async.global`` arm only when asked, ``%globaltimer`` / ``trap`` only with a deadline, NO ``.release.cluster``
and NO ``.acquire.cluster``; the gate is keyword-only without a default; a constant gate is refused.
"""

import inspect

from cutlass.base_dsl.typing import Pointer  # noqa: F401  (the type of ``flag_smem``; re-exported for callers)
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .barrier import POLL_SLEEP_NS, named_barrier_fence

__all__ = [
    "COUNTER_LINE_BYTES",
    "COUNTER_WORD_BYTES",
    "COUNTER_WORDS_PER_LINE",
    "acquire_geq",
    "counter_store_u64",
    "counter_word_offset",
    "fence_acq_rel_gpu",
    "fence_proxy_async_global",
    "last_arriver",
    "ld_acquire_gpu_u64",
    "release_add",
]

# One 128-byte line per counter class (review A9 of the decode design: a release atom must never share a line with a polled
# word of another class); 16 eight-byte words per line.
COUNTER_LINE_BYTES: int = 128
COUNTER_WORD_BYTES: int = 8
COUNTER_WORDS_PER_LINE: int = COUNTER_LINE_BYTES // COUNTER_WORD_BYTES

# The named-barrier domain of ``bar.sync id, n``: ids 0-15 (0 is the CTA-wide ``__syncthreads`` barrier, kept out of the
# primitives so a role barrier never aliases it), ``n`` a positive multiple of the warp size.
_BAR_ID_MIN, _BAR_ID_MAX = 1, 15
_WARP = 32


def counter_word_offset(line: int, word: int = 0) -> int:
    """Byte offset of ``word`` of counter class ``line`` from the base of a counter slab laid out one 128-B line per
    class.  A class with more than 16 words (20 m-tiles of a split-K reduce) spans consecutive lines: pass the FIRST
    line of the class and the word index, and keep the classes' first lines ``ceil(words / 16)`` apart so no two
    classes share a line."""
    line, word = int(line), int(word)
    if line < 0 or word < 0:
        raise ValueError(f"gsync.counter_word_offset: line and word must be >= 0, got line={line} word={word}")
    return line * COUNTER_LINE_BYTES + word * COUNTER_WORD_BYTES


def _require_gate(warp_gate, entry: str) -> None:
    """Trace-time guard: the warp gate must be a per-thread PREDICATE (a ``cutlass.Boolean`` value), never a Python
    constant -- ``True`` across a multi-warp role is the over-arrival these entries exist to make unwritable."""
    if warp_gate is None or isinstance(warp_gate, (bool, int)):
        raise TypeError(
            f"gsync.{entry}: warp_gate must be a per-thread predicate that holds on the lanes of ONE warp "
            f"(e.g. ``warp_idx == 0``), got the Python constant {warp_gate!r}; the lane ledger is one elected lane of "
            "that warp per CTA -- a constant gate on a multi-warp role is the P3 over-arrival"
        )


def _require_barrier(bar_id, bar_threads, entry: str) -> None:
    """Trace-time guard on the named barrier: a Python-int id in 1..15 and a positive multiple-of-32 thread count."""
    if not isinstance(bar_id, int) or isinstance(bar_id, bool) or not (_BAR_ID_MIN <= bar_id <= _BAR_ID_MAX):
        raise TypeError(f"gsync.{entry}: bar_id must be a Python int in {_BAR_ID_MIN}..{_BAR_ID_MAX} (0 is the CTA barrier), got {bar_id!r}")
    if not isinstance(bar_threads, int) or isinstance(bar_threads, bool) or bar_threads < _WARP or bar_threads % _WARP:
        raise TypeError(f"gsync.{entry}: bar_threads must be a positive multiple of {_WARP} (the role's thread count), got {bar_threads!r}")


def gate_parameter(entry) -> inspect.Parameter:
    """The ``warp_gate`` parameter of one of the entries (for the host pin: keyword-only, no default)."""
    return inspect.signature(entry).parameters["warp_gate"]


# ----------------------------------------------------------------------------------------------------------------------------
# The atoms (one instruction each; the gate and the elect are the entries' below -- these issue on EVERY calling lane)
# ----------------------------------------------------------------------------------------------------------------------------


@cute.jit
def fence_acq_rel_gpu():
    """``fence.acq_rel.gpu`` -- the gpu-scope fence of the release and the last-arriver recipes."""
    inline_ptx("fence.acq_rel.gpu;", write_only_types=[], read_only_args=[])


@cute.jit
def fence_proxy_async_global():
    """``fence.proxy.async.global`` -- orders this thread's generic-proxy global accesses against the async proxy (TMA)."""
    inline_ptx("fence.proxy.async.global;", write_only_types=[], read_only_args=[])


@cute.jit
def _fence_sc_sys():
    """``fence.sc.sys`` -- before a deadline trap, so a diagnostic in mapped pinned host memory is visible to the host."""
    inline_ptx("fence.sc.sys;", write_only_types=[], read_only_args=[])


@cute.jit
def _trap():
    inline_ptx("trap;", write_only_types=[], read_only_args=[])


@cute.jit
def ld_acquire_gpu_u64(addr):
    """``ld.acquire.gpu.global.u64`` of the 8-byte word at generic address ``addr`` (an ``Int64``)."""
    return inline_ptx("ld.acquire.gpu.global.u64 {$w0}, [{$r0}];", write_only_types=[cutlass.Int64], read_only_args=[addr])


@cute.jit
def _red_release_gpu_add_u64(addr, n):
    inline_ptx("red.release.gpu.global.add.u64 [{$r0}], {$r1};", write_only_types=[], read_only_args=[addr, n])


@cute.jit
def _atom_acq_rel_gpu_add_u64(addr, n):
    return inline_ptx("atom.acq_rel.gpu.global.add.u64 {$w0}, [{$r0}], {$r1};", write_only_types=[cutlass.Int64], read_only_args=[addr, n])


@cute.jit
def counter_store_u64(addr, value):
    """``st.relaxed.gpu.global.u64 [addr], value`` -- the kernel-side reset of a counter word by the thread that has
    proved itself the last user of it in this launch (the target form's last arriver; a persistent kernel's last-alive
    CTA at exit).  Relaxed at gpu scope: it must not be reordered ahead of the proof (the atom that returned the
    final value), and the next launch reads it after a kernel boundary."""
    inline_ptx("st.relaxed.gpu.global.u64 [{$r0}], {$r1};", write_only_types=[], read_only_args=[addr, value])


@cute.jit
def _st_global_u64(addr, value):
    inline_ptx("st.global.u64 [{$r0}], {$r1};", write_only_types=[], read_only_args=[addr, value])


# ----------------------------------------------------------------------------------------------------------------------------
# The entries
# ----------------------------------------------------------------------------------------------------------------------------


@cute.jit
def release_add(
    addr,
    n,
    *,
    warp_gate,
    bar_id: cutlass.Constexpr[int],
    bar_threads: cutlass.Constexpr[int],
    async_proxy: cutlass.Constexpr[bool] = False,
):
    """RELEASE ``n`` arrivals on the counter word at ``addr`` (the module docstring's first form).

    Called by every thread of the role.  ``addr``: the word's generic address (``Int64``); ``n``: the increment (1 for
    one CTA's arrival).  ``warp_gate``: the per-thread predicate of the ONE warp that signals (required, keyword-only;
    the elect is internal).  ``bar_id`` / ``bar_threads``: the role's named barrier.  ``async_proxy=True`` adds the
    writer-side ``fence.proxy.async.global`` for a payload the consumer TMA-loads.
    Ledger row: ``1 lane x <arriving CTAs> == target``."""
    _require_gate(warp_gate, "release_add")
    _require_barrier(bar_id, bar_threads, "release_add")
    if cutlass.const_expr(async_proxy):
        fence_proxy_async_global()
    fence_acq_rel_gpu()
    named_barrier_fence(bar_id, bar_threads)
    if warp_gate:
        if nvvm.elect_sync():
            _red_release_gpu_add_u64(addr, cutlass.Int64(n))


@cute.jit
def acquire_geq(
    addr,
    target,
    *,
    warp_gate,
    bar_id: cutlass.Constexpr[int],
    bar_threads: cutlass.Constexpr[int],
    backoff_ns: cutlass.Constexpr[int] = POLL_SLEEP_NS,
    async_proxy: cutlass.Constexpr[bool] = False,
    deadline_ns=None,
    diag_addr=None,
    code: cutlass.Constexpr[int] = 0,
):
    """ACQUIRE: block the role until the counter word at ``addr`` reads ``>= target`` (the module docstring's second
    form).

    Called by every thread of the role.  One elected lane of the ``warp_gate`` warp polls (``ld.acquire.gpu`` with a
    ``nanosleep.u32 backoff_ns`` between tests); the named barrier publishes the acquire to the other warps.
    ``async_proxy=True`` adds the reader-side ``fence.proxy.async.global`` on the polling warp (for a payload this
    CTA then TMA-loads).  ``deadline_ns`` (an ``Int64``, nanoseconds of ``%globaltimer``) arms the bounded spin: past
    it the lane writes ``code`` and the last counter value to ``diag_addr`` (two u64 words, when given), fences at
    system scope and traps.  ``None`` renders no deadline (production: no timer reads in the poll)."""
    _require_gate(warp_gate, "acquire_geq")
    _require_barrier(bar_id, bar_threads, "acquire_geq")
    if cutlass.const_expr(int(backoff_ns) < 0):
        raise TypeError(f"gsync.acquire_geq: backoff_ns must be >= 0, got {backoff_ns!r}")
    target64 = cutlass.Int64(target)
    if warp_gate:
        if nvvm.elect_sync():
            if cutlass.const_expr(deadline_ns is not None):
                t0 = cute.arch.globaltimer()
            v = ld_acquire_gpu_u64(addr)
            while v < target64:
                if cutlass.const_expr(int(backoff_ns) > 0):
                    nvvm.nanosleep(backoff_ns)
                v = ld_acquire_gpu_u64(addr)
                if cutlass.const_expr(deadline_ns is not None):
                    if cute.arch.globaltimer() - t0 > cutlass.Int64(deadline_ns):
                        if cutlass.const_expr(diag_addr is not None):
                            _st_global_u64(diag_addr, cutlass.Int64(code))
                            _st_global_u64(diag_addr + cutlass.Int64(COUNTER_WORD_BYTES), v)
                            _fence_sc_sys()
                        _trap()
        if cutlass.const_expr(async_proxy):
            fence_proxy_async_global()
    named_barrier_fence(bar_id, bar_threads)


@cute.jit
def last_arriver(
    addr,
    count,
    *,
    warp_gate,
    flag_smem,
    bar_id: cutlass.Constexpr[int],
    bar_threads: cutlass.Constexpr[int],
    reset: cutlass.Constexpr[bool] = True,
    async_proxy: cutlass.Constexpr[bool] = False,
):
    """Arrive on the counter word at ``addr`` and learn, CTA-uniformly, whether this CTA is the LAST of ``count``
    arrivers (the module docstring's third form).  Returns ``(is_last, old)``: ``is_last`` a ``Boolean`` that is the
    same on every thread of the role, ``old`` the ``Int64`` value the atom returned (this CTA's arrival rank in the
    target form; ``old // count`` the launch number in the modulus form).

    Called by every thread of the role.  ``flag_smem``: a ``Pointer`` to one 8-byte SMEM word owned by the caller for
    the broadcast (``cutlass.Array(Int64, 1, alignment=8, space=smem).data_ptr()`` wrapped as ``Pointer(..., dtype=Int64)``;
    it is written once per call by the electing lane and read by every thread after the named barrier).  ``reset=True``
    = the target form (``is_last = old == count - 1``, the counter stored back to 0 by the electing lane);
    ``reset=False`` = the modulus form (``is_last = (old + 1) % count == 0``, the counter never reset).  The last
    arriver's threads return behind a ``fence.acq_rel.gpu`` and may read the other arrivers' payload at once.
    Ledger row: ``1 lane x count == count``."""
    _require_gate(warp_gate, "last_arriver")
    _require_barrier(bar_id, bar_threads, "last_arriver")
    count64 = cutlass.Int64(count)
    if cutlass.const_expr(async_proxy):
        fence_proxy_async_global()
    fence_acq_rel_gpu()
    named_barrier_fence(bar_id, bar_threads)
    if warp_gate:
        if nvvm.elect_sync():
            old_lane = _atom_acq_rel_gpu_add_u64(addr, cutlass.Int64(1))
            if cutlass.const_expr(reset):
                if old_lane == count64 - cutlass.Int64(1):
                    counter_store_u64(addr, cutlass.Int64(0))
            flag_smem.store(old_lane)
    named_barrier_fence(bar_id, bar_threads)
    old = flag_smem.load()
    if cutlass.const_expr(reset):
        is_last = old == count64 - cutlass.Int64(1)
    else:
        is_last = (old + cutlass.Int64(1)) % count64 == cutlass.Int64(0)
    if is_last:
        fence_acq_rel_gpu()
    return is_last, old
