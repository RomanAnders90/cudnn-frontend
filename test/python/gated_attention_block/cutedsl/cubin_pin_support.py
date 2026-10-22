# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared support of the ``sm_107a`` cubin pins (``test_qsa_prepare.py``, ``test_qsa_score_paged.py``, ``test_qsa_topk_decode.py``).

A pin holds a served kernel's rendering byte-identical to a committed md5 record.  The cubin is a function of the DSL BUILD -- its version
string AND the PTX assembler it bundles (two wheels that share a version string but assemble differently would otherwise collide) -- so a
record names both, and a pin compares both before it trusts an md5:

* the installed DSL (``cutedsl_state``, no import) against the record's ``dsl`` line;
* the assembler that produced the rendered cubin (:func:`compiler_of_cubin`: the ``Cuda compilation tools, release R, V<version>`` string
  the assembler leaves in every cubin) against the record's ``compiler`` line.

On a mismatch (or a DSL without the ``sm_107a`` target) a pin SKIPS by default, naming both sides -- but a skipped pin is uncovered, not
passing, so the pin-owning CI job sets ``CUDNN_FROST_REQUIRE_CUBIN_PINS=1`` and every such skip becomes a FAILURE that names both
sides (:func:`skip_or_fail`); a DSL bump re-renders the records in the same change.  This module imports no DSL at import time.
"""

import os
import re
from pathlib import Path
from typing import Optional

import pytest

from cudnn.frost.buffers import cutedsl_state

REQUIRE_ENV = "CUDNN_FROST_REQUIRE_CUBIN_PINS"
COMPILER_RE = re.compile(rb"Cuda compilation tools, release (\d+\.\d+), V(\d+\.\d+\.\d+)")
COMPILER_UNKNOWN = "unknown"


def pins_required() -> bool:
    """``CUDNN_FROST_REQUIRE_CUBIN_PINS=1``: a pin that cannot run is a failure, never a skip."""
    return os.environ.get(REQUIRE_ENV, "") == "1"


def compiler_of_cubin(path) -> str:
    """``ptxas <V>`` from the assembler's version string inside the cubin at ``path`` (``unknown`` when the cubin carries none)."""
    m = COMPILER_RE.search(Path(path).read_bytes())
    return f"ptxas {m.group(2).decode()}" if m else COMPILER_UNKNOWN


def compiler_of_bytes(data: bytes) -> str:
    """:func:`compiler_of_cubin` on the cubin's bytes (for a probe that already holds them)."""
    m = COMPILER_RE.search(data)
    return f"ptxas {m.group(2).decode()}" if m else COMPILER_UNKNOWN


def installed_dsl() -> Optional[str]:
    """``<distribution> <version>`` of the installed DSL, as the records spell it (None when none is installed)."""
    _installed, version = cutedsl_state()
    return " ".join(version) if version else None


def mismatch_reason(kind: str, want, have) -> str:
    return f"the record was rendered with the {kind} {want!r}; this environment has {have!r}: the cubin is a function of the DSL build -- re-render the record under this {kind} to pin it"


def skip_or_fail(reason: str) -> None:
    """Skip the pin for ``reason`` -- or, under ``CUDNN_FROST_REQUIRE_CUBIN_PINS=1``, FAIL it with the same reason."""
    if pins_required():
        pytest.fail(f"{REQUIRE_ENV}=1: the cubin pin must run here -- {reason}")
    pytest.skip(reason)


def arch_known(arch: str = "sm_107a") -> bool:
    """Whether the installed DSL has the target (the public 4.8.0 wheel is the first with ``sm_107a``)."""
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string(arch)
        return True
    except Exception:  # noqa: BLE001 -- an older DSL raises KeyError / ValueError / ImportError; all mean "no such target"
        return False


def check_arch(arch: str = "sm_107a") -> None:
    if not arch_known(arch):
        skip_or_fail(f"this CuTe DSL has no {arch} target")


def check_dsl(record_dsl: Optional[str]) -> None:
    """The installed DSL must be the record's; otherwise skip-or-fail naming both."""
    have = installed_dsl()
    if record_dsl is None or have != record_dsl:
        skip_or_fail(mismatch_reason("DSL", record_dsl, have))


def check_compiler(record_compiler: Optional[str], have: str) -> None:
    """The assembler that produced the rendered cubin must be the record's; otherwise skip-or-fail naming both.  A record without a
    ``compiler`` line (none is committed any more) pins the DSL string alone."""
    if record_compiler is not None and have != record_compiler:
        skip_or_fail(mismatch_reason("assembler", record_compiler, have))
