# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tile_dsl.gsync`` -- the grid-scope counter primitives (``release_add`` / ``acquire_geq`` / ``last_arriver``).

Tiers, one probe kernel (``_PROBE_SRC``, written to ``tmp_path`` and run in a subprocess so a trap or a sticky context
cannot poison the pytest session; the same source is the device smoke of the development runner):

* tier 0 (pure host): the three entries take ``warp_gate`` keyword-only without a default; a Python-constant gate and a
  missing gate are refused at trace time; the counter-line helper.
* tier 1 (any box whose cutlass-dsl knows the arch): the sm_107a and sm_100a trace-compiles of the RENDEZVOUS probe
  (every entry in one kernel) carry ``red.release.gpu.global.add.u64`` / ``ld.acquire.gpu.global.u64`` /
  ``atom.acq_rel.gpu.global.add.u64`` / ``fence.acq_rel.gpu``, the ``fence.proxy.async.global`` arm only when asked,
  ``%globaltimer`` + ``trap`` only with a deadline, ``nanosleep.u32`` in the poll, the named barriers, NO
  ``.release.cluster`` and NO ``.acquire.cluster``; the modulus form carries no reset store.
* tier 2 (an SM100-line GPU): ORDINARY launches of 212 and 424 CTAs with a 209 KiB dynamic-SMEM footprint (one CTA per
  SM, so 424 CTAs are two or more waves on every part of the line): the last arriver (target and modulus forms) and the
  chained release / acquire complete N back-to-back launches on ONE buffer with every CTA's rank a permutation of
  ``0 .. grid - 1`` per launch (the self-reset holds), the last arriver reading every peer's payload; a CUDA-graph
  capture + replay of both; a co-resident rendezvous at the SM count; and the deadline that traps (exit 4 with the
  seam code in the pinned diagnostic) instead of hanging.
"""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

from frost_test_utils import arch_known_to_the_dsl, requires_blackwell, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_PROBE_SRC = textwrap.dedent('''
    """gsync probe: one kernel, five device arms + two trace-time refusal arms + the PTX pins (module docstring of the test).

    Barrier table of the probe (named barrier 1 = the whole 128-thread CTA = the role; every entry is called by all 128 threads):
      bar.sync 1, 128  inside release_add / acquire_geq / last_arriver, CTA-uniform call sites only.
    Counter ledger (one 128-B line each; the plan-side zero is the host's torch.zeros before the FIRST launch only):
      cnt      (line 0): LA arms 1 lane x grid == grid (target form resets; modulus form monotonic); CHAIN: 1 lane x (grid - 1)
               releases, the provably-last CTA stores 0; RENDEZVOUS: 1 lane x grid == grid, stored 0 by the last finisher.
      epoch    (line 1): written by exactly one thread per launch after every CTA has read it (the last arriver / the chain's last CTA).
      cnt_done (line 2): RENDEZVOUS only, 1 lane x grid == grid (target form, self-reset).
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.typing import Pointer
    from cutlass.cute.arch.nvvm_wrappers import inline_ptx
    from cutlass.experimental import primitives as nvvm

    from cudnn.frost.tile_dsl.gsync import COUNTER_WORDS_PER_LINE, acquire_geq, counter_store_u64, counter_word_offset, last_arriver, release_add

    I32, I64 = cutlass.Int32, cutlass.Int64
    THREADS = 128
    BAR_ROLE = 1
    MODE_LA_TARGET, MODE_LA_MODULUS, MODE_CHAIN, MODE_RENDEZVOUS, MODE_TRAP, MODE_BAD_GATE, MODE_MISSING_GATE = range(7)
    MODES = {"la_target": MODE_LA_TARGET, "la_modulus": MODE_LA_MODULUS, "chain": MODE_CHAIN, "rendezvous": MODE_RENDEZVOUS, "trap": MODE_TRAP,
             "bad_gate": MODE_BAD_GATE, "missing_gate": MODE_MISSING_GATE}
    OFF_CNT, OFF_EPOCH, OFF_DONE = counter_word_offset(0), counter_word_offset(1), counter_word_offset(2)
    COUNTER_WORDS = 3 * COUNTER_WORDS_PER_LINE
    PAT_MUL = 1000003
    TRAP_CODE, CHAIN_CODE, RDV_CODE = 7, 2, 3


    @cute.jit
    def ld_u64(addr):
        return inline_ptx("ld.global.u64 {$w0}, [{$r0}];", write_only_types=[I64], read_only_args=[addr])


    @cute.jit
    def st_u64(addr, v):
        inline_ptx("st.global.u64 [{$r0}], {$r1};", write_only_types=[], read_only_args=[addr, v])


    @cute.jit
    def st_shared_u8(smem_addr, v):
        inline_ptx("st.shared.u8 [{$r0}], {$r1};", write_only_types=[], read_only_args=[smem_addr, v])


    @cute.kernel
    def probe(cnt_base: I64, payload: I64, olds: I64, chk: I64, diag: I64, grid: I32, deadline_ns: I64,
              mode: cutlass.Constexpr[int], smem_bytes: cutlass.Constexpr[int], async_proxy: cutlass.Constexpr[bool], with_deadline: cutlass.Constexpr[bool]):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = I32(cute.arch.thread_idx()[0])
        bid = I32(cute.arch.block_idx()[0])
        gate = warp == I32(0)
        s_flag = cutlass.Array(I64, 1, alignment=8, space=cutlass.AddressSpace.smem)
        flag = Pointer(s_flag.data_ptr(), dtype=I64)
        if cutlass.const_expr(smem_bytes > 0):
            # the dynamic footprint is REAL (one CTA per SM at 209 KiB): first and last byte written
            smem_base = cute.arch.get_dyn_smem(cutlass.Int8, alignment=16).toint()
            if tidx == I32(0):
                st_shared_u8(smem_base, bid)
                st_shared_u8(smem_base + I32(smem_bytes - 1), bid)
        cnt = cnt_base + I64(OFF_CNT)
        epoch_addr = cnt_base + I64(OFF_EPOCH)
        cnt_done = cnt_base + I64(OFF_DONE)
        epoch = ld_u64(epoch_addr)
        pat = epoch * I64(PAT_MUL) + I64(bid) * I64(7) + I64(1)
        grid64 = I64(grid)
        if cutlass.const_expr(mode in (MODE_LA_TARGET, MODE_LA_MODULUS)):
            if tidx == I32(0):
                st_u64(payload + I64(bid) * I64(8), pat)
            is_last, old = last_arriver(cnt, grid, warp_gate=gate, flag_smem=flag, bar_id=BAR_ROLE, bar_threads=THREADS,
                                        reset=(mode == MODE_LA_TARGET), async_proxy=async_proxy)
            if tidx == I32(0):
                st_u64(olds + (epoch * grid64 + I64(bid)) * I64(8), old)
            if is_last:
                # every peer's payload of this epoch, read after the acq_rel atom + the broadcast: thread t sums slots t, t + 128, ...
                acc = I64(0)
                for j in cutlass.range(tidx, grid, THREADS):
                    acc = acc + ld_u64(payload + I64(j) * I64(8))
                st_u64(chk + (epoch * I64(THREADS) + I64(tidx)) * I64(8), acc)
                if tidx == I32(0):
                    st_u64(epoch_addr, epoch + I64(1))
        elif cutlass.const_expr(mode == MODE_CHAIN):
            # CTA c > 0 acquires the c releases of CTAs 0 .. c - 1 (each released only after its own acquire: a chain in CTA order,
            # deadlock-free without co-residency because a CTA waits only on lower-numbered CTAs that wait on nothing above them)
            if bid > I32(0):
                if cutlass.const_expr(with_deadline):
                    acquire_geq(cnt, bid, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy,
                                deadline_ns=deadline_ns, diag_addr=diag, code=CHAIN_CODE)
                else:
                    acquire_geq(cnt, bid, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy)
                if tidx == I32(0):
                    st_u64(chk + (epoch * grid64 + I64(bid)) * I64(8), ld_u64(payload + I64(bid - I32(1)) * I64(8)))
            if tidx == I32(0):
                st_u64(payload + I64(bid) * I64(8), pat)
                st_u64(olds + (epoch * grid64 + I64(bid)) * I64(8), I64(bid))
            if bid < grid - I32(1):
                release_add(cnt, 1, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy)
            else:
                # the chain's last CTA: every other CTA is past its acquire (it released before we could acquire), nothing touches cnt again
                if tidx == I32(0):
                    counter_store_u64(cnt, I64(0))
                    st_u64(epoch_addr, epoch + I64(1))
        elif cutlass.const_expr(mode == MODE_RENDEZVOUS):
            if tidx == I32(0):
                st_u64(payload + I64(bid) * I64(8), pat)
            release_add(cnt, 1, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy)
            if cutlass.const_expr(with_deadline):
                acquire_geq(cnt, grid, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy,
                            deadline_ns=deadline_ns, diag_addr=diag, code=RDV_CODE)
            else:
                acquire_geq(cnt, grid, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, async_proxy=async_proxy)
            if tidx == I32(0):
                nxt = (bid + I32(1)) % grid
                st_u64(chk + (epoch * grid64 + I64(bid)) * I64(8), ld_u64(payload + I64(nxt) * I64(8)))
            is_last, old = last_arriver(cnt_done, grid, warp_gate=gate, flag_smem=flag, bar_id=BAR_ROLE, bar_threads=THREADS)
            if tidx == I32(0):
                st_u64(olds + (epoch * grid64 + I64(bid)) * I64(8), old)
            if is_last:
                if tidx == I32(0):
                    counter_store_u64(cnt, I64(0))
                    st_u64(epoch_addr, epoch + I64(1))
        elif cutlass.const_expr(mode == MODE_TRAP):
            # a target no release ever reaches: the deadline must end the launch with TRAP_CODE in the pinned diag
            acquire_geq(cnt, I64(1) << 40, warp_gate=gate, bar_id=BAR_ROLE, bar_threads=THREADS, deadline_ns=deadline_ns, diag_addr=diag, code=TRAP_CODE)
        elif cutlass.const_expr(mode == MODE_BAD_GATE):
            release_add(cnt, 1, warp_gate=True, bar_id=BAR_ROLE, bar_threads=THREADS)  # refused at trace time: a constant gate
        elif cutlass.const_expr(mode == MODE_MISSING_GATE):
            release_add(cnt, 1, bar_id=BAR_ROLE, bar_threads=THREADS)  # refused at trace time: no gate at all


    probe.set_name_prefix("cudnn", remove_cutlass_symbol=True)


    @cute.jit
    def launch(cnt_base: I64, payload: I64, olds: I64, chk: I64, diag: I64, grid: I32, deadline_ns: I64,
               mode: cutlass.Constexpr[int], smem_bytes: cutlass.Constexpr[int], async_proxy: cutlass.Constexpr[bool], with_deadline: cutlass.Constexpr[bool],
               stream: cuda.CUstream):
        probe(cnt_base, payload, olds, chk, diag, grid, deadline_ns, mode, smem_bytes, async_proxy, with_deadline).launch(
            grid=(grid, 1, 1), block=(THREADS, 1, 1), smem=smem_bytes, stream=stream)


    # ------------------------------------------------------------------ host side ------------------------------------------------------------------
    PTX_PINS = (("RED_RELEASE_GPU", "red.release.gpu.global.add.u64"), ("LD_ACQUIRE_GPU", "ld.acquire.gpu.global.u64"),
                ("ATOM_ACQ_REL_GPU", "atom.acq_rel.gpu.global.add.u64"), ("FENCE_ACQ_REL_GPU", "fence.acq_rel.gpu"),
                ("FENCE_PROXY_ASYNC_GLOBAL", "fence.proxy.async.global"), ("FENCE_SC_SYS", "fence.sc.sys"), ("RELEASE_CLUSTER", ".release.cluster"),
                ("ACQUIRE_CLUSTER", ".acquire.cluster"), ("NANOSLEEP", "nanosleep.u32"), ("GLOBALTIMER", "%globaltimer"), ("TRAP", "trap;"),
                ("BAR_SYNC_ROLE", f"bar.sync {BAR_ROLE}, {THREADS}"), ("ELECT_SYNC", "elect.sync"), ("RESET_STORE", "st.relaxed.gpu.global.u64"),
                ("REM_S64", "rem.s64"))


    def compile_host(arch, mode, async_proxy, with_deadline, dump):
        """Trace-compile for ``arch`` (no device needed) and print one ``PTX <key> <count>`` line per pin."""
        import glob
        from cutlass.cute.runtime import make_fake_stream

        cute.compile(launch, I64(0), I64(0), I64(0), I64(0), I64(0), I32(1), I64(0), int(mode), 0, bool(async_proxy), bool(with_deadline),
                     make_fake_stream(use_tvm_ffi_env_stream=False), options=f"--enable-tvm-ffi --gpu-arch {arch} --keep-ptx")
        print("COMPILED", arch, mode, flush=True)
        ptxs = glob.glob(os.path.join(dump, f"*.{arch}.ptx"))
        if not ptxs:
            print("FAIL no ptx landed in", dump, os.listdir(dump))
            return 3
        ptx = open(ptxs[0]).read()
        for key, pat in PTX_PINS:
            print("PTX", key, ptx.count(pat))
        return 0


    class PinnedDiag:
        """Two u64 words in MAPPED PINNED host memory: readable after a trap without a CUDA call."""

        def __init__(self):
            import torch

            self.host = torch.zeros(4, dtype=torch.int64).pin_memory()
            err, dptr = cuda.cuMemHostGetDevicePointer(self.host.data_ptr(), 0)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise SystemExit(f"cuMemHostGetDevicePointer failed ({err})")
            self.dev_ptr = int(dptr)


    def run_arm(mode_name, grid, launches, *, smem_kib=209, deadline_s=None, async_proxy=False, graph=False, device_index=0):
        """Run one device arm; returns the verdict dict (``ok`` plus the evidence).  A trap surfaces as ``trapped`` with the diag code."""
        import time
        import torch

        mode = MODES[mode_name]
        dev = torch.device("cuda", device_index)
        torch.cuda.set_device(dev)
        props = torch.cuda.get_device_properties(dev)
        with_deadline = deadline_s is not None
        smem_bytes = int(smem_kib) * 1024
        counters = torch.zeros(COUNTER_WORDS, dtype=torch.int64, device=dev)  # the plan-side zero (once, before the first launch)
        payload = torch.zeros(max(grid, 1), dtype=torch.int64, device=dev)
        olds = torch.full((launches, max(grid, 1)), -1, dtype=torch.int64, device=dev)
        # the LA arms record 128 per-thread partials per launch (row stride THREADS); the chain / rendezvous arms one word per CTA (row stride grid)
        chk = torch.full((launches, THREADS if mode in (MODE_LA_TARGET, MODE_LA_MODULUS) else max(grid, 1)), -1, dtype=torch.int64, device=dev)
        diag = PinnedDiag()
        stream = torch.cuda.Stream(dev) if graph else torch.cuda.current_stream(dev)
        cu = cuda.CUstream(stream.cuda_stream)
        t0 = time.time()
        fn = cute.compile(launch, I64(0), I64(0), I64(0), I64(0), I64(0), I32(1), I64(0), int(mode), int(smem_bytes), bool(async_proxy), bool(with_deadline), cu)
        compile_s = time.time() - t0
        dl = I64(int(deadline_s * 1e9)) if with_deadline else I64(0)
        args = (I64(counters.data_ptr()), I64(payload.data_ptr()), I64(olds.data_ptr()), I64(chk.data_ptr()), I64(diag.dev_ptr), I32(grid), dl)
        res = {"mode": mode_name, "grid": int(grid), "launches": int(launches), "smem_kib": int(smem_kib), "graph": bool(graph), "async_proxy": bool(async_proxy),
               "deadline_s": deadline_s, "device": props.name, "cc": f"{props.major}.{props.minor}", "sms": int(props.multi_processor_count), "compile_s": round(compile_s, 2)}
        t0 = time.time()
        try:
            with torch.cuda.stream(stream):
                if graph:
                    fn(*args, cu)  # module load outside the capture
                    torch.cuda.synchronize()
                    counters.zero_(); olds.fill_(-1); chk.fill_(-1)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=stream):
                        fn(*args, cu)
                    counters.zero_(); olds.fill_(-1); chk.fill_(-1)
                    torch.cuda.synchronize()
                    for _ in range(launches):
                        g.replay()
                else:
                    for _ in range(launches):
                        fn(*args, cu)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001  -- a device trap is a sticky launch failure; the pinned diag names the seam
            d = diag.host.tolist()
            res.update({"ok": False, "trapped": True, "diag_code": int(d[0]), "diag_value": int(d[1]), "elapsed_s": round(time.time() - t0, 3),
                        "error": type(e).__name__ + ": " + str(e).splitlines()[0][:200] if str(e) else type(e).__name__})
            return res
        res["elapsed_s"] = round(time.time() - t0, 3)
        c = counters.cpu()
        o = olds.cpu()
        k = chk.cpu()
        ar = torch.arange(grid, dtype=torch.int64)
        bad = []
        if mode in (MODE_LA_TARGET, MODE_LA_MODULUS):
            for e in range(launches):
                want = ar if mode == MODE_LA_TARGET else ar + e * grid
                if not torch.equal(torch.sort(o[e]).values, want):
                    bad.append(f"launch {e}: olds not the permutation {int(want[0])}..{int(want[-1])}: {o[e][:8].tolist()}...")
                exp = grid * (e * PAT_MUL + 1) + 7 * grid * (grid - 1) // 2
                got = int(k[e, :THREADS].sum())
                if got != exp:
                    bad.append(f"launch {e}: the last arriver's payload sum {got} != {exp}")
            cnt_want = 0 if mode == MODE_LA_TARGET else launches * grid
            res.update({"cnt_after": int(c[OFF_CNT // 8]), "cnt_want": cnt_want, "epoch_after": int(c[OFF_EPOCH // 8]), "last_arrivers_per_launch": "1" if not bad else "see bad"})
            if int(c[OFF_CNT // 8]) != cnt_want or int(c[OFF_EPOCH // 8]) != launches:
                bad.append(f"counters after: cnt {int(c[OFF_CNT // 8])} (want {cnt_want}) epoch {int(c[OFF_EPOCH // 8])} (want {launches})")
        elif mode == MODE_CHAIN:
            for e in range(launches):
                if not torch.equal(o[e], ar):
                    bad.append(f"launch {e}: not every CTA recorded itself: {o[e][:8].tolist()}...")
                want = e * PAT_MUL + 7 * ar[:-1] + 1  # payload of CTA c - 1 as read by CTA c (c >= 1)
                if not torch.equal(k[e, 1:grid], want):
                    bad.append(f"launch {e}: acquired payload mismatch at {(k[e, 1:grid] != want).nonzero().flatten()[:8].tolist()}")
            res.update({"cnt_after": int(c[OFF_CNT // 8]), "cnt_want": 0, "epoch_after": int(c[OFF_EPOCH // 8])})
            if int(c[OFF_CNT // 8]) != 0 or int(c[OFF_EPOCH // 8]) != launches:
                bad.append(f"counters after: cnt {int(c[OFF_CNT // 8])} epoch {int(c[OFF_EPOCH // 8])} (want 0 / {launches})")
        elif mode == MODE_RENDEZVOUS:
            for e in range(launches):
                if not torch.equal(torch.sort(o[e]).values, ar):
                    bad.append(f"launch {e}: cnt_done ranks not a permutation: {o[e][:8].tolist()}...")
                want = e * PAT_MUL + 7 * ((ar + 1) % grid) + 1
                if not torch.equal(k[e, :grid], want):
                    bad.append(f"launch {e}: peer payload mismatch at {(k[e, :grid] != want).nonzero().flatten()[:8].tolist()}")
            res.update({"cnt_after": int(c[OFF_CNT // 8]), "cnt_done_after": int(c[OFF_DONE // 8]), "epoch_after": int(c[OFF_EPOCH // 8])})
            if int(c[OFF_CNT // 8]) != 0 or int(c[OFF_DONE // 8]) != 0 or int(c[OFF_EPOCH // 8]) != launches:
                bad.append(f"counters after: cnt {int(c[OFF_CNT // 8])} cnt_done {int(c[OFF_DONE // 8])} epoch {int(c[OFF_EPOCH // 8])}")
        elif mode == MODE_TRAP:
            bad.append("the deadline did not trap: the launch completed")
        res.update({"ok": not bad, "bad": bad[:12], "n_bad": len(bad), "trapped": False})
        return res


    if __name__ == "__main__":
        import argparse

        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd", required=True)
        p = sub.add_parser("ptx")
        p.add_argument("arch"); p.add_argument("mode"); p.add_argument("dump"); p.add_argument("--async-proxy", action="store_true"); p.add_argument("--deadline", action="store_true")
        r = sub.add_parser("run")
        r.add_argument("mode"); r.add_argument("--grid", type=int, required=True); r.add_argument("--launches", type=int, required=True)
        r.add_argument("--smem-kib", type=int, default=209); r.add_argument("--deadline-s", type=float, default=None); r.add_argument("--async-proxy", action="store_true")
        r.add_argument("--graph", action="store_true"); r.add_argument("--device", type=int, default=0)
        a = ap.parse_args()
        if a.cmd == "ptx":
            sys.exit(compile_host(a.arch, MODES[a.mode], a.async_proxy, a.deadline, a.dump))
        res = run_arm(a.mode, a.grid, a.launches, smem_kib=a.smem_kib, deadline_s=a.deadline_s, async_proxy=a.async_proxy, graph=a.graph, device_index=a.device)
        print("RESULT", json.dumps(res), flush=True)
        sys.exit(0 if res.get("ok") else (4 if res.get("trapped") else 1))
    ''')

ARCHES = ["sm_107a", "sm_100a"]
PROBE_IMPORTS = "import json, os, sys\n"


def write_probe(tmp_path):
    probe_py = tmp_path / "gsync_probe.py"
    probe_py.write_text(PROBE_IMPORTS + _PROBE_SRC)
    return probe_py


def _run(argv, *, env=None, timeout=900):
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=timeout, env=env)


def _ptx(tmp_path, arch, mode, *, async_proxy=False, deadline=False):
    dump = tmp_path / f"dump_{arch}_{mode}_{int(async_proxy)}{int(deadline)}"
    dump.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx")
    argv = [str(write_probe(tmp_path)), "ptx", arch, mode, str(dump)] + (["--async-proxy"] if async_proxy else []) + (["--deadline"] if deadline else [])
    proc = _run(argv, env=env)
    counts = {ln.split()[1]: int(ln.split()[2]) for ln in proc.stdout.splitlines() if ln.startswith("PTX ") and len(ln.split()) == 3}
    return proc, counts


# ---------------------------------------------------------------------------
# Tier 0: the API shape (pure host)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["release_add", "acquire_geq", "last_arriver"])
def test_warp_gate_is_a_required_keyword_argument(entry):
    """The over-arrival is unwritable at the signature: ``warp_gate`` is keyword-only and has no default on every entry."""
    from cudnn.frost.tile_dsl import gsync

    p = gsync.gate_parameter(getattr(gsync, entry))
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, f"{entry}: warp_gate must be keyword-only, is {p.kind}"
    assert p.default is inspect.Parameter.empty, f"{entry}: warp_gate must have no default, has {p.default!r}"


@pytest.mark.parametrize("bad", [True, False, 1, None])
def test_python_constant_gate_is_refused(bad):
    from cudnn.frost.tile_dsl import gsync

    with pytest.raises(TypeError, match="warp_gate must be a per-thread predicate"):
        gsync._require_gate(bad, "release_add")


@pytest.mark.parametrize("bar_id,bar_threads", [(0, 128), (16, 128), (1, 100), (1, 0), (1.0, 128), (True, 128)])
def test_named_barrier_domain_is_checked(bar_id, bar_threads):
    from cudnn.frost.tile_dsl import gsync

    with pytest.raises(TypeError, match="bar_(id|threads)"):
        gsync._require_barrier(bar_id, bar_threads, "release_add")


def test_counter_word_offset_keeps_one_line_per_class():
    from cudnn.frost.tile_dsl import gsync

    assert gsync.COUNTER_LINE_BYTES == 128 and gsync.COUNTER_WORDS_PER_LINE == 16
    assert gsync.counter_word_offset(0) == 0 and gsync.counter_word_offset(1) == 128 and gsync.counter_word_offset(3, 4) == 3 * 128 + 32
    assert gsync.counter_word_offset(1, 0) - gsync.counter_word_offset(0, 15) == 8  # adjacent lines touch, never overlap
    with pytest.raises(ValueError):
        gsync.counter_word_offset(-1)


# ---------------------------------------------------------------------------
# Tier 1: trace-time refusals and the PTX pins (no device)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bad_gate", "missing_gate"])
def test_over_arrival_cannot_be_written(tmp_path, mode):
    """A kernel that calls ``release_add`` with a constant gate, or with no gate, does not trace-compile; the error names
    ``warp_gate`` (the P3 over-arrival of a bare elect across warps has no spelling through this module)."""
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, _ = _ptx(tmp_path, "sm_107a", mode)
    assert proc.returncode != 0, f"the {mode} probe must NOT compile:\n{proc.stdout[-1500:]}"
    assert "warp_gate" in proc.stderr + proc.stdout, f"the refusal must name warp_gate:\n{proc.stderr[-3000:]}"
    assert "COMPILED" not in proc.stdout


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("async_proxy,deadline", [(True, True), (False, False)], ids=["async+deadline", "production"])
def test_ptx_forms_of_the_three_entries(tmp_path, arch, async_proxy, deadline):
    """The RENDEZVOUS probe holds every entry: release_add -> ``red.release.gpu``; acquire_geq -> ``ld.acquire.gpu`` +
    ``nanosleep``; last_arriver -> ``atom.acq_rel.gpu`` + the reset store; ``fence.acq_rel.gpu`` on both recipes; the
    ``fence.proxy.async.global`` arm and the ``%globaltimer`` / ``trap`` / ``fence.sc.sys`` deadline only when asked; the
    named barrier; NO ``.release.cluster`` / ``.acquire.cluster`` anywhere."""
    if not arch_known_to_the_dsl(arch):
        pytest.skip(f"this cutlass-dsl has no {arch}")
    proc, c = _ptx(tmp_path, arch, "rendezvous", async_proxy=async_proxy, deadline=deadline)
    assert proc.returncode == 0 and f"COMPILED {arch}" in proc.stdout, f"{arch} trace-compile failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    print(f"\n[{arch} async={async_proxy} deadline={deadline}] {c}")
    assert c["RED_RELEASE_GPU"] >= 1 and c["LD_ACQUIRE_GPU"] >= 1 and c["ATOM_ACQ_REL_GPU"] >= 1, c
    assert c["FENCE_ACQ_REL_GPU"] >= 2, c
    assert c["RELEASE_CLUSTER"] == 0 and c["ACQUIRE_CLUSTER"] == 0, f"a cluster-scope release / acquire is a kernel-wide drain: {c}"
    assert c["NANOSLEEP"] >= 1 and c["ELECT_SYNC"] >= 1 and c["BAR_SYNC_ROLE"] >= 4, c
    assert c["RESET_STORE"] >= 1, f"the target-form last arriver must store the counter back to zero: {c}"
    if async_proxy:
        assert c["FENCE_PROXY_ASYNC_GLOBAL"] >= 2, c  # the writer side (release_add) and the reader side (acquire_geq)
    else:
        assert c["FENCE_PROXY_ASYNC_GLOBAL"] == 0, f"no proxy fence unless asked: {c}"
    if deadline:
        assert c["GLOBALTIMER"] >= 2 and c["TRAP"] >= 1 and c["FENCE_SC_SYS"] >= 1, c
    else:
        assert c["GLOBALTIMER"] == 0 and c["TRAP"] == 0 and c["FENCE_SC_SYS"] == 0, f"a production render carries no deadline: {c}"


def test_ptx_modulus_form_never_resets(tmp_path):
    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    proc, c = _ptx(tmp_path, "sm_107a", "la_modulus")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert c["ATOM_ACQ_REL_GPU"] == 1 and c["RESET_STORE"] == 0 and c["REM_S64"] >= 1, c
    assert c["RELEASE_CLUSTER"] == 0 and c["ACQUIRE_CLUSTER"] == 0, c


# ---------------------------------------------------------------------------
# Tier 2: the device arms (an SM100-line GPU; each arm its own process)
# ---------------------------------------------------------------------------

DEVICE_LAUNCHES = int(os.environ.get("GSYNC_DEVICE_LAUNCHES", "100"))


def _device_run(tmp_path, mode, grid, launches, *extra, timeout=900):
    argv = [str(write_probe(tmp_path)), "run", mode, "--grid", str(grid), "--launches", str(launches), *extra]
    proc = _run(argv, timeout=timeout)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, f"no RESULT line (rc {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    res = json.loads(line[-1][len("RESULT ") :])
    print(f"\n{res}")
    return proc, res


def _sm_count():
    import torch

    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@requires_blackwell
@pytest.mark.parametrize("grid", [212, 424])
@pytest.mark.parametrize("mode", ["la_target", "la_modulus", "chain"])
def test_ordinary_launch_needs_no_coresidency(tmp_path, mode, grid):
    """ORDINARY launches of 212 / 424 CTAs at 209 KiB dynamic SMEM (one CTA per SM: more CTAs than SMs on every SM100-line
    part, 424 = two or more waves) complete ``DEVICE_LAUNCHES`` back-to-back launches on ONE counter buffer: the ranks of
    every launch are a permutation (the self-reset holds), the last arriver reads every peer's payload, the chain's acquirer
    reads its predecessor's payload; the counters read their expected final values."""
    proc, res = _device_run(tmp_path, mode, grid, DEVICE_LAUNCHES)
    assert proc.returncode == 0 and res["ok"], f"{mode} grid {grid}: {res.get('bad') or res}"
    assert res["sms"] < 424, f"the 424-CTA cell must be more than one wave to make the point (SMs {res['sms']})"


@requires_blackwell
@pytest.mark.parametrize("mode", ["la_target", "chain"])
def test_graph_capture_and_replay(tmp_path, mode):
    """One capture, ``DEVICE_LAUNCHES`` replays of the same node: every replay's records are correct and the counters self-reset."""
    proc, res = _device_run(tmp_path, mode, 212, DEVICE_LAUNCHES, "--graph")
    assert proc.returncode == 0 and res["ok"], f"{mode} graph replay: {res.get('bad') or res}"


@requires_blackwell
def test_coresident_rendezvous_at_the_sm_count(tmp_path):
    """Every CTA releases, then acquires ``grid`` arrivals (co-residency needed: grid = SM count, one CTA per SM), reads its
    successor's payload; the last finisher resets the rendezvous counter.  The poll carries the 2 s deadline."""
    proc, res = _device_run(tmp_path, "rendezvous", _sm_count(), min(DEVICE_LAUNCHES, 100), "--deadline-s", "2.0")
    assert proc.returncode == 0 and res["ok"], f"rendezvous: {res.get('bad') or res}"


@requires_blackwell
def test_deadline_traps_instead_of_hanging(tmp_path):
    """An acquire whose target is never released ends the launch within the deadline: exit 4, the seam code in the pinned
    diagnostic, elapsed well under the process budget (a hang would be the outer timeout, exit 124)."""
    proc, res = _device_run(tmp_path, "trap", 4, 1, "--deadline-s", "0.3", "--smem-kib", "0", timeout=600)
    assert proc.returncode == 4 and res.get("trapped") and res.get("diag_code") == 7, f"the deadline must trap with code 7: rc {proc.returncode} {res}"
    assert res["elapsed_s"] < 30.0, res
