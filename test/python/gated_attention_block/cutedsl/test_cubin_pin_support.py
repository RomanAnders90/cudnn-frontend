# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cubin_pin_support``: the strictness contract of the ``sm_107a`` cubin pins (host; no DSL, no GPU).

By default a pin that cannot run -- another DSL build, another bundled assembler, a DSL without the target -- SKIPS naming both sides;
under ``CUDNN_FROST_REQUIRE_CUBIN_PINS=1`` the same condition FAILS naming both sides (a skipped pin is uncovered, not passing).  The
assembler is read out of the cubin's own version string.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cubin_pin_support as cps  # noqa: E402

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("required", [False, True], ids=["default_skips", "opt_in_fails"])
def test_a_dsl_mismatch_skips_by_default_and_fails_under_the_opt_in(monkeypatch, required):
    monkeypatch.setenv(cps.REQUIRE_ENV, "1" if required else "")
    monkeypatch.setattr(cps, "installed_dsl", lambda: "nvidia-cutlass-dsl 9.9.9")
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(outcome) as ei:
        cps.check_dsl("nvidia-cutlass-dsl 4.8.0")
    msg = str(ei.value)
    assert "nvidia-cutlass-dsl 4.8.0" in msg and "nvidia-cutlass-dsl 9.9.9" in msg, msg
    assert ("must run here" in msg) is required
    # the same DSL string passes without an outcome
    monkeypatch.setattr(cps, "installed_dsl", lambda: "nvidia-cutlass-dsl 4.8.0")
    cps.check_dsl("nvidia-cutlass-dsl 4.8.0")


@pytest.mark.parametrize("required", [False, True], ids=["default_skips", "opt_in_fails"])
def test_an_assembler_mismatch_skips_by_default_and_fails_under_the_opt_in(monkeypatch, required):
    monkeypatch.setenv(cps.REQUIRE_ENV, "1" if required else "")
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(outcome) as ei:
        cps.check_compiler("ptxas 13.4.59", "ptxas 13.3.27")
    msg = str(ei.value)
    assert "ptxas 13.4.59" in msg and "ptxas 13.3.27" in msg, msg
    cps.check_compiler("ptxas 13.4.59", "ptxas 13.4.59")
    cps.check_compiler(None, "ptxas 13.4.59")  # a record without the line pins the DSL string alone


@pytest.mark.parametrize("required", [False, True], ids=["default_skips", "opt_in_fails"])
def test_a_missing_target_skips_by_default_and_fails_under_the_opt_in(monkeypatch, required):
    monkeypatch.setenv(cps.REQUIRE_ENV, "1" if required else "")
    monkeypatch.setattr(cps, "arch_known", lambda arch="sm_107a": False)
    with pytest.raises(pytest.fail.Exception if required else pytest.skip.Exception, match="no sm_107a target"):
        cps.check_arch("sm_107a")


def test_the_assembler_is_read_out_of_the_cubin(tmp_path):
    cubin = tmp_path / "probe.cubin"
    cubin.write_bytes(b"\x7fELF" + b"\x00" * 64 + b"Cuda compilation tools, release 13.4, V13.4.59\x00" + b"\x00" * 16)
    assert cps.compiler_of_cubin(cubin) == "ptxas 13.4.59"
    assert cps.compiler_of_bytes(b"release 13.3, V13.3.27") == "unknown"  # the whole string, never a fragment
    assert cps.compiler_of_bytes(b"Cuda compilation tools, release 13.3, V13.3.27") == "ptxas 13.3.27"
    cubin.write_bytes(b"\x7fELF" + b"\x00" * 64)
    assert cps.compiler_of_cubin(cubin) == cps.COMPILER_UNKNOWN


def test_the_opt_in_is_the_exact_value_one(monkeypatch):
    for value, expect in (("", False), ("0", False), ("1", True), ("yes", False)):
        monkeypatch.setenv(cps.REQUIRE_ENV, value)
        assert cps.pins_required() is expect, value
    monkeypatch.delenv(cps.REQUIRE_ENV, raising=False)
    assert cps.pins_required() is False
