# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The paged decode scorer's served renderings and their ``sm_107a`` cubin record (``renderings/md5_qsa_score_paged_sm_107a.txt``).

A rendering of ``kernels/qsa_score_paged.py`` is one compile key ``(h_i, s_q, page_size, units_per_warp, ring_depth, has_block_lens,
use_pdl, l2_hint, cache dtype)``; :data:`SERVED_KEYS` lists every key ``test_qsa_score_paged.py`` compiles (the fp64-reference cells, the
row-map branches, the knob cells, the optional ``block_lens`` arm, the PDL arm, the selector's knob cells, both arms of the ``l2_hint``
knob and the e4m3 read arm's cells).  :func:`render_md5s` trace-compiles each of
them for ``sm_107a`` in ONE fresh interpreter -- no GPU: ``CUTE_DSL_ARCH`` needs no matching device -- with the DSL asked to keep the
cubin, and returns the md5 of each rendering's cubin.  The committed record pins those md5s: a change of the kernel's source that is
meant to leave the standalone kernel alone (a body moved into a role function, an argument that became ``Optional``) proves it by a
byte-identical cubin per key (the house rule for every seam that must leave a served kernel alone).

Regenerate the record (a deliberate change of the kernel; the diff names every key that moved)::

    cd test/python/gated_attention_block/cutedsl && python qsa_score_paged_renderings.py > renderings/md5_qsa_score_paged_sm_107a.txt

A record stays valid for the DSL build it names -- ``dsl=<distribution> <version>`` and ``compiler=ptxas <version>``, the assembler the
cubins carry (two wheels that share a version string but assemble differently would otherwise collide): the test skips, naming both
sides, under another DSL build, or FAILS under ``CUDNN_FROST_REQUIRE_CUBIN_PINS=1`` (``cubin_pin_support``).  This module imports no DSL
at import time (the test module gates the floor before the kernel import; the probe runs in a child interpreter).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

RECORD = Path(__file__).resolve().parent / "renderings" / "md5_qsa_score_paged_sm_107a.txt"
ARCH = "sm_107a"
KERNEL = "qsa_score_paged"


class RenderKey(NamedTuple):
    h_i: int
    s_q: int
    page_size: int
    units_per_warp: int = 1
    ring_depth: int = 4
    has_block_lens: bool = True
    use_pdl: bool = False
    l2_hint: bool = True  # the key copies' arm as BUILT (the launch's Constexpr): the hinted copy, the resolved default on a 13.4+ compiler
    e4m3: bool = False  # the cache dtype: bf16, or the e4m3 read arm (float8_e4m3fn queries and rows)

    @property
    def tag(self) -> str:
        return (
            f"h{self.h_i}_s{self.s_q}_p{self.page_size}_u{self.units_per_warp}_d{self.ring_depth}_l{int(self.has_block_lens)}_pdl{int(self.use_pdl)}"
            f"_hint{int(self.l2_hint)}_e4m3{int(self.e4m3)}"
        )


def _served_keys() -> Tuple[RenderKey, ...]:
    keys: List[RenderKey] = []
    for s_q in (1, 2, 3, 4):  # the fp64-reference cells and the selector cells: H_i 4, the knobs at their defaults
        for page in (16, 32, 64):
            keys.append(RenderKey(4, s_q, page))
    keys.append(RenderKey(3, 1, 8, units_per_warp=4))  # the fresh variant of the compiled-plan-cache cell
    for h_i, s_q, page in ((1, 20, 16), (2, 3, 32), (3, 2, 16), (8, 2, 16), (16, 3, 64), (17, 2, 16), (4, 5, 8)):  # the row-map branches
        keys.append(RenderKey(h_i, s_q, page))
    for page in (16, 64):  # the units knob
        for units in (1, 2, 4, 8):
            keys.append(RenderKey(4, 2, page, units_per_warp=units))
    for depth in (2, 4, 8):  # the ring-depth knob
        keys.append(RenderKey(4, 3, 16, units_per_warp=2, ring_depth=depth))
    keys.append(RenderKey(4, 2, 32, has_block_lens=False))  # the optional block_lens arm
    keys.append(RenderKey(4, 4, 16, use_pdl=True))  # the PDL arm
    keys.append(RenderKey(4, 3, 16, units_per_warp=4))  # the selector's knob cells
    keys.append(RenderKey(4, 3, 16, use_pdl=True))
    keys.append(RenderKey(4, 2, 16, l2_hint=False))  # the l2_hint knob cell: the plain copies (the hinted arm is the default key above)
    for s_q in (1, 2, 3, 4):  # the e4m3 read arm's fp64-reference cells (pages 8 / 16 / 32 / 64)
        for page in (8, 16, 32, 64):
            keys.append(RenderKey(4, s_q, page, e4m3=True))
    for h_i, s_q, page in ((1, 20, 16), (2, 3, 32), (3, 2, 16), (8, 2, 8), (16, 3, 64), (17, 2, 16), (4, 5, 8)):  # the e4m3 row-map branches
        keys.append(RenderKey(h_i, s_q, page, e4m3=True))
    for page in (8, 16, 64):  # the e4m3 knobs cell: units 1 / 2 / 4 / 8 at depth 4, units 2 at depth 2 / 8 -- and their bf16 twins
        for units in (1, 2, 4, 8):
            keys.append(RenderKey(4, 2, page, units_per_warp=units, e4m3=True))
            keys.append(RenderKey(4, 2, page, units_per_warp=units))
        for depth in (2, 8):
            keys.append(RenderKey(4, 2, page, units_per_warp=2, ring_depth=depth, e4m3=True))
            keys.append(RenderKey(4, 2, page, units_per_warp=2, ring_depth=depth))
    for units in (1, 2):  # the e4m3 SMEM cell: both arms at units 1 / 2
        keys.append(RenderKey(4, 1, 16, units_per_warp=units, e4m3=True))
        keys.append(RenderKey(4, 1, 16, units_per_warp=units))
    out: List[RenderKey] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return tuple(out)


SERVED_KEYS: Tuple[RenderKey, ...] = _served_keys()

# The child interpreter: the DSL's dump directory, the kept artefact and the target are read ONCE at the first ``import cutlass``
# (frost-harness-realities: "CUTE_DSL_KEEP artefacts land in CUTE_DSL_DUMP_DIR at import cutlass"), the compiled-plan cache is off (a
# hit skips ptxas and dumps no cubin); every key compiles the kernel's launch entry with the SAME fake operands the compile site builds,
# the dump directory is emptied before each compile so the ONE cubin it leaves is the key's.
_PROBE = textwrap.dedent("""
    import glob, hashlib, json, os, re, sys
    dump, arch, keys = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    os.environ["CUTE_DSL_DUMP_DIR"] = dump
    os.environ["CUTE_DSL_KEEP"] = "cubin"
    os.environ["CUTE_DSL_ARCH"] = arch
    os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
    import cutlass, cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import make_fake_stream
    from cudnn.frost.buffers import cutedsl_state
    from cudnn.gated_attention_block.kernels import qsa_score_paged as sp
    print("DSL", " ".join(cutedsl_state()[1] or ("?", "?")))
    print("DIGEST", sp.FROST_SOURCE_DIGEST)
    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    for tag, h_i, s_q, page_size, units, depth, has_lens, use_pdl, l2_hint, e4m3 in keys:
        for f in glob.glob(os.path.join(dump, "*")):
            os.remove(f)
        rows_per_page = page_size // sp.BLOCK_SIZE
        dt = torch.float8_e4m3fn if e4m3 else torch.bfloat16
        cute.compile(
            sp.qsa_score_paged_launch, sp._fake_q(h_i, dt), sp._fake_kbar(rows_per_page, dt), sp._fake_i32_2d(), sp._fake_i32_1d(), sp._fake_f32_2d(),
            sp._fake_i32_1d(), sp._fake_i32_1d() if has_lens else None, cutlass.Int32(0), cutlass.Float32(0.0), cutlass.Int32(0), cutlass.Int32(0),
            int(h_i), int(s_q), int(rows_per_page), int(units), int(depth), bool(use_pdl), bool(l2_hint), stream,
            options=f"--enable-tvm-ffi --gpu-arch {arch}",
        )
        cubins = glob.glob(os.path.join(dump, "*.cubin"))
        if len(cubins) != 1:
            print("FAIL", tag, "expected one cubin in the dump, found", sorted(os.path.basename(c) for c in cubins)); sys.exit(3)
        data = open(cubins[0], "rb").read()
        m = re.search(rb"Cuda compilation tools, release (\\d+\\.\\d+), V(\\d+\\.\\d+\\.\\d+)", data)  # the assembler's own string in the cubin
        print("COMPILER", tag, "ptxas " + m.group(2).decode() if m else "unknown")
        print("MD5", tag, hashlib.md5(data).hexdigest(), flush=True)
    """)


class Rendered(NamedTuple):
    dsl: str  # "<distribution> <version>" of the DSL that rendered
    digest: str  # the kernel module's FROST_SOURCE_DIGEST in the child
    md5s: Dict[str, str]  # key tag -> cubin md5
    compiler: str = "unknown"  # "ptxas <version>": the assembler the rendered cubins carry (one for every key)


def render_md5s(dump_dir, keys=SERVED_KEYS, *, timeout: int = 2400) -> Rendered:
    """Trace-compile ``keys`` for :data:`ARCH` in one child interpreter (``dump_dir`` a fresh, writable directory) and return the md5 of
    every rendering's cubin.  A compile failure raises with the child's tail."""
    dump = Path(dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-c", _PROBE, str(dump), ARCH, json.dumps([[k.tag, *k] for k in keys])]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
    if proc.returncode != 0:
        raise RuntimeError(f"{ARCH} trace-compile of {KERNEL} failed (rc {proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-6000:]}")
    md5s = {m.group(1): m.group(2) for m in re.finditer(r"^MD5 (\S+) ([0-9a-f]{32})$", proc.stdout, re.M)}
    dsl = next((ln.split(" ", 1)[1] for ln in proc.stdout.splitlines() if ln.startswith("DSL ")), "?")
    digest = next((ln.split(" ", 1)[1] for ln in proc.stdout.splitlines() if ln.startswith("DIGEST ")), "?")
    compilers = {m.group(2) for m in re.finditer(r"^COMPILER (\S+) (.+)$", proc.stdout, re.M)}
    if len(compilers) != 1:
        raise RuntimeError(f"the renderings name {len(compilers)} assemblers ({sorted(compilers)}); one DSL build assembles every key:\n{proc.stdout[-4000:]}")
    missing = [k.tag for k in keys if k.tag not in md5s]
    if missing:
        raise RuntimeError(f"the probe printed no md5 for {missing}:\n{proc.stdout[-4000:]}")
    return Rendered(dsl, digest, md5s, compilers.pop())


def read_record(path=RECORD) -> Tuple[Optional[str], Dict[str, str]]:
    """``(dsl, {tag: md5})`` of the committed record (``dsl`` None when the file names none)."""
    dsl, want = None, {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("dsl="):
            dsl = ln[4:].strip()
        m = re.match(rf"{KERNEL} {ARCH} (\S+) rc=0 cubin_md5=([0-9a-f]{{32}})$", ln)
        if m:
            want[m.group(1)] = m.group(2)
    return dsl, want


def read_record_meta(path=RECORD) -> Dict[str, str]:
    """The record's ``key=value`` header lines (``dsl``, ``compiler``) -- the DSL build the md5s are a function of."""
    meta: Dict[str, str] = {}
    for ln in Path(path).read_text().splitlines():
        m = re.match(r"^([a-z_]+)=(.*)$", ln)
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return meta


def arch_known(arch: str = ARCH) -> bool:
    """Whether the installed DSL has the target (the public 4.8.0 wheel is the first with ``sm_107a``)."""
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(arch)
        return True
    except Exception:  # noqa: BLE001 -- an older DSL raises KeyError / ValueError / ImportError; all mean "no such target"
        return False


def main(argv=None) -> int:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", default=None, help="the child's dump directory (a fresh temporary directory by default)")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="qsa_score_paged_renderings_") as tmp:
        rendered = render_md5s(args.dump or tmp)
    print(f"# The paged decode scorer's served renderings: the {ARCH} cubin md5 per compile key (host trace-compile; the probe and the key list are")
    print("# qsa_score_paged_renderings.py's).  Lines: dsl=<distribution> <version>; compiler=ptxas <version> (the assembler the cubins carry);")
    print(f"# {KERNEL} {ARCH} <key tag> rc=0 cubin_md5=<md5>.  A record stays valid for the DSL build (version + assembler) it names.")
    print("# The key tag spells (h_i, s_q, page_size, units_per_warp, ring_depth, has_block_lens, use_pdl, l2_hint, e4m3 cache dtype).")
    print(f"# kernel source digest at the record: {rendered.digest}")
    print(f"dsl={rendered.dsl}")
    print(f"compiler={rendered.compiler}")
    for k in SERVED_KEYS:
        print(f"{KERNEL} {ARCH} {k.tag} rc=0 cubin_md5={rendered.md5s[k.tag]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
