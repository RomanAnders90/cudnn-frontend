# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strided / batched FROST GEMM plans the MQA block's projections stand on.

``gated_attention_block.kernels.proj_gemm.build_proj_gemm`` grew six append-only
knobs (``a_row_stride, c_row_stride, batch, a_batch_stride, b_batch_stride,
c_batch_stride``) so that

* stage 9 (``o_a_proj``) runs EIGHT launches of ONE plan on column slices of the
  ``[T, 32768]`` attention output into column slices of the ``[T, 8192]`` LoRA
  slab -- declared row strides, no repack (D3);
* stage 5 (``kv_proj``) writes the ``[:, :S]`` PREFIX of the caller's
  ``[B, S + N_c, 512]`` KV buffer directly -- a batched C at batch stride
  ``(S + N_c) * 512`` (D4);
* the one-launch batched ``wo_a`` (batch stride SMALLER than the row stride) is
  the unmeasured M6 form, gated ``xfail(strict=False)`` until the dev node says.

Accept tests need Rubin (the block's target); the declaration / bind-check
rejects run on any device.
"""

import pytest
import torch
import torch.nn.functional as F

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("MQA sparse-attention block GEMM-plan tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

from cudnn.gated_attention_block.kernels.proj_gemm import (  # noqa: E402
    ProjGemmPlan,
    _NoFrostPlan,
    _why_no_frost_plan,
    build_proj_gemm,
    handle_for_stream,
    run_proj_gemm,
)

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

_BF16 = torch.bfloat16


def _launch(plan, a, w, out, stream=None):
    ws = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    run_proj_gemm(plan, a, w, out, ws, stream=stream)


def _assert_gemm_close(out, ref, label=""):
    """bf16 GEMM output vs the fp32 product of the same bf16 operands: the two differ
    by accumulation order and ONE bf16 rounding, so 2**-6 relative plus a small
    absolute term; and a cosine floor so a zero / garbage output cannot slip past
    the relative bar on tiny values."""
    out_f, ref_f = out.float(), ref.float()
    torch.testing.assert_close(out_f, ref_f, rtol=2**-6, atol=2**-6 * float(ref_f.abs().max()), msg=lambda m: f"{label}: {m}")
    cos = F.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0).item()
    assert cos > 0.999, f"{label}: cos {cos}"
    assert torch.isfinite(out_f).all(), f"{label}: non-finite output"


# ---------------------------------------------------------------------------
# Accept -- Rubin
# ---------------------------------------------------------------------------

_T, _GROUPS, _GW, _R = 256, 8, 4096, 1024  # o_a_proj at the released geometry (M = 256 tokens)


def _wo_a_operands(seed=0):
    torch.manual_seed(seed)
    o = (torch.randn(_T, _GROUPS * _GW, device="cuda") * 0.05).to(_BF16)
    w = (torch.randn(_GROUPS, _R, _GW, device="cuda") * 0.02).to(_BF16)  # [G, o_lora_rank, group_width] -- the grouped view
    ref = torch.einsum("tgd,grd->tgr", o.view(_T, _GROUPS, _GW).float(), w.float()).reshape(_T, _GROUPS * _R)
    return o, w, ref


@requires_rubin
def test_wo_a_eight_launches_with_declared_row_strides_match_einsum_and_the_compact_launches():
    """D3: one plan declared at ``a_row_stride = 8*4096``, ``c_row_stride = 8*1024``,
    launched eight times on column slices.  Numerics against the einsum, and BIT-EQUAL
    to the packed plan on compact copies (same kernel, same tile, same accumulation
    order -- only the descriptors' strides differ)."""
    o, w, ref = _wo_a_operands()
    o_lora = torch.zeros(_T, _GROUPS * _R, device="cuda", dtype=_BF16)
    plan = build_proj_gemm(m=_T, k=_GW, n=_R, dtype=_BF16, label="wo_a_strided", a_row_stride=_GROUPS * _GW, c_row_stride=_GROUPS * _R)
    assert plan.a_strides == (_T * _GROUPS * _GW, _GROUPS * _GW, 1)
    assert plan.c_strides == (_T * _GROUPS * _R, _GROUPS * _R, 1)
    assert plan.b_strides == (_GW * _R, _GW, 1) and plan.b_batch == 1 and plan.batch == 1
    for g in range(_GROUPS):
        _launch(plan, o[:, g * _GW : (g + 1) * _GW], w[g], o_lora[:, g * _R : (g + 1) * _R])
    torch.cuda.synchronize()
    _assert_gemm_close(o_lora, ref, "wo_a 8-launch")

    compact = build_proj_gemm(m=_T, k=_GW, n=_R, dtype=_BF16, label="wo_a_compact")
    assert compact.tile_config_name == plan.tile_config_name, (compact.tile_config_name, plan.tile_config_name)
    for g in range(_GROUPS):
        out_g = torch.empty(_T, _R, device="cuda", dtype=_BF16)
        _launch(compact, o[:, g * _GW : (g + 1) * _GW].contiguous(), w[g].contiguous(), out_g)
        torch.cuda.synchronize()
        assert torch.equal(out_g, o_lora[:, g * _R : (g + 1) * _R]), f"group {g}: the strided launch is not bit-equal to the compact one"


@requires_rubin
@pytest.mark.parametrize("batch", [1, 2])
def test_kv_proj_writes_the_prefix_of_a_wider_kv_buffer(batch):
    """D4: ``kv_all[:, :S]`` is the GEMM's C at batch stride ``(S + N_c) * 512``; the
    compressed rows ``[:, S:]`` (a sentinel here) are untouched, and every batch
    entry is bit-equal to the compact single-batch plan."""
    s, n_c, k, n = 256, 96, 512, 512
    torch.manual_seed(1)
    x = (torch.randn(batch, s, k, device="cuda") * 0.1).to(_BF16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(_BF16)
    kv_all = torch.full((batch, s + n_c, n), 7.0, device="cuda", dtype=_BF16)
    plan = build_proj_gemm(m=s, k=k, n=n, dtype=_BF16, label="kv_prefix", batch=batch, a_batch_stride=s * k, c_row_stride=n, c_batch_stride=(s + n_c) * n)
    assert plan.batch == batch and plan.b_batch == 1 and plan.c_strides == ((s + n_c) * n, n, 1)
    _launch(plan, x, w, kv_all[:, :s, :])
    torch.cuda.synchronize()
    _assert_gemm_close(kv_all[:, :s], F.linear(x.float(), w.float()), f"kv prefix B={batch}")
    assert torch.all(kv_all[:, s:] == 7.0), "the GEMM wrote past the [:, :S] prefix"

    compact = build_proj_gemm(m=s, k=k, n=n, dtype=_BF16, label="kv_compact")
    for b in range(batch):
        out_b = torch.empty(s, n, device="cuda", dtype=_BF16)
        _launch(compact, x[b], w, out_b)
        torch.cuda.synchronize()
        assert torch.equal(out_b, kv_all[b, :s]), f"batch {b}: the prefix launch is not bit-equal to the compact one"


@requires_rubin
@pytest.mark.xfail(
    strict=False,
    reason="M6: a batched frost_gemm whose batch stride (4096) is SMALLER than its row stride (32768) is unmeasured; a PASS flips this and enables wo_a_batched",
)
def test_M6_wo_a_one_batched_launch_with_an_interleaved_batch_stride():
    o, w, ref = _wo_a_operands()
    o_lora = torch.zeros(_T, _GROUPS * _R, device="cuda", dtype=_BF16)
    plan = build_proj_gemm(
        m=_T,
        k=_GW,
        n=_R,
        dtype=_BF16,
        label="wo_a_batched",
        batch=_GROUPS,
        a_row_stride=_GROUPS * _GW,
        a_batch_stride=_GW,
        b_batch_stride=_R * _GW,
        c_row_stride=_GROUPS * _R,
        c_batch_stride=_R,
    )
    assert plan.b_batch == _GROUPS
    a_b = o.view(_T, _GROUPS, _GW).permute(1, 0, 2)  # [G, T, gw], strides (gw, G*gw, 1)
    c_b = o_lora.view(_T, _GROUPS, _R).permute(1, 0, 2)  # [G, T, r],  strides (r, G*r, 1)
    _launch(plan, a_b, w, c_b)
    torch.cuda.synchronize()
    _assert_gemm_close(o_lora, ref, "wo_a batched (M6)")


@requires_rubin
def test_strided_output_is_not_silently_zero():
    """Twin of the gated block's guard for the >256 KiB tcgen05 SMEM-descriptor
    hazard, on the strided declaration: a version-0 descriptor whose operand
    starts past 262144 wraps and the accumulator comes out EXACTLY ZERO with no
    error -- a cosine against a zero tensor is NaN rather than a failure, so
    assert non-zero explicitly.  Also that the OTHER half of the wide C slab is
    untouched."""
    m, k, n = 1024, 4096, 1024
    a_wide = torch.full((m, 2 * k), 0.05, device="cuda", dtype=_BF16)
    w = torch.full((n, k), 0.05, device="cuda", dtype=_BF16)
    out_wide = torch.zeros(m, 2 * n, device="cuda", dtype=_BF16)
    plan = build_proj_gemm(m=m, k=k, n=n, dtype=_BF16, label="nonzero_strided", a_row_stride=2 * k, c_row_stride=2 * n)
    _launch(plan, a_wide[:, :k], w, out_wide[:, :n])
    torch.cuda.synchronize()
    written = out_wide[:, :n]
    nonzero = (written != 0).float().mean().item()
    assert nonzero > 0.99, f"only {100 * nonzero:.1f}% of the output is non-zero -- suspect the >256 KiB SMEM descriptor version"
    torch.testing.assert_close(written.float().mean().item(), 0.05 * 0.05 * k, rtol=2e-2, atol=0)
    assert torch.all(out_wide[:, n:] == 0), "the strided store leaked into the other half of the slab"


# ---------------------------------------------------------------------------
# Reject -- any device
# ---------------------------------------------------------------------------


def test_build_rejects_overlapping_rows():
    with pytest.raises(ValueError, match="rows may not overlap"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", a_row_stride=63)
    with pytest.raises(ValueError, match="rows may not overlap"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", c_row_stride=16)


def test_build_rejects_strides_that_are_not_16_byte_multiples():
    with pytest.raises(ValueError, match="16 bytes"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", c_row_stride=33)  # 66 B
    with pytest.raises(ValueError, match="16 bytes"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", batch=2, a_batch_stride=8 * 64 + 4)


def test_build_rejects_a_non_positive_batch_and_batch_strides():
    with pytest.raises(ValueError, match="batch must be >= 1"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", batch=0)
    with pytest.raises(ValueError, match="batch strides must be >= 1"):
        build_proj_gemm(m=8, k=64, n=32, dtype=_BF16, label="p", batch=2, c_batch_stride=0)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="this torch has no float8_e4m3fn")
def test_build_rejects_the_stride_knobs_on_the_block_scale_path():
    with pytest.raises(ValueError, match="packed operands only"):
        build_proj_gemm(m=128, k=128, n=128, dtype=torch.float8_e4m3fn, label="p", block_scale=True, batch=2)


def test_run_refuses_a_bind_that_disagrees_with_the_declaration():
    """The strided plan RECORDS its declaration and ``run_proj_gemm`` compares the
    bound rank-3 views against it BEFORE any route -- on the JIT route the kernel
    reads the runtime strides and would otherwise run silently on the wrong
    layout.  A plan object suffices; nothing is launched."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    plan = ProjGemmPlan(
        graph=None,
        a=object(),
        b=object(),
        c=object(),
        m=8,
        k=64,
        n=32,
        label="probe",
        batch=1,
        b_batch=1,
        a_strides=(8 * 128, 128, 1),
        b_strides=(64 * 32, 64, 1),
        c_strides=(8 * 32, 32, 1),
    )
    plan.route = "jit-only"  # with no artifact: the runner raises its own RuntimeError AFTER the layout check passes
    w = torch.empty(32, 64, dtype=_BF16, device=dev)
    out = torch.empty(8, 32, dtype=_BF16, device=dev)
    packed_a = torch.empty(8, 64, dtype=_BF16, device=dev)  # row stride 64 != the declared 128
    with pytest.raises(ValueError, match="declared shape"):
        run_proj_gemm(plan, packed_a, w, out, None, stream=0)
    strided_a = torch.empty(8, 128, dtype=_BF16, device=dev)[:, :64]  # the declared view
    with pytest.raises(RuntimeError, match="jit-only"):
        run_proj_gemm(plan, strided_a, w, out, None, stream=0)
    wrong_out = torch.empty(8, 64, dtype=_BF16, device=dev)[:, :32]  # out row stride 64 != the declared 32
    with pytest.raises(ValueError, match="out is bound with shape"):
        run_proj_gemm(plan, strided_a, w, wrong_out, None, stream=0)


def test_a_legacy_plan_carries_no_declaration_and_is_not_checked():
    """A plan built without the knobs keeps ``a_strides is None`` -- byte-identical
    behaviour to before the extension (the gated block's callers pass packed or
    padded views and are not second-guessed)."""
    plan = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=1, k=1, n=1, label="legacy")
    assert plan.batch == 1 and plan.b_batch == 1 and plan.a_strides is None and plan.b_strides is None and plan.c_strides is None
    assert plan.flops() == 2


def test_missing_frost_plan_is_a_distinct_typed_failure():
    """Twin of the gated block's guard: an unpinned graph would silently run a
    cuDNN backend plan and every number measured off it would be that kernel's."""
    assert issubclass(_NoFrostPlan, RuntimeError)
    why = _why_no_frost_plan()
    assert "Refusing to fall back" in why


@requires_cuda
def test_run_proj_gemm_refuses_a_handle_bound_to_another_stream():
    """Given BOTH a handle and a stream they must agree (a handle bound elsewhere
    would run the GEMM off the block's launch stream): a typed ``ValueError``
    before any route."""
    import cudnn

    side = torch.cuda.Stream()
    h = cudnn.create_handle()
    cudnn.set_stream(handle=h, stream=side.cuda_stream)
    plan = ProjGemmPlan(graph=None, a=None, b=None, c=None, m=1, k=1, n=1, label="probe")
    with pytest.raises(ValueError, match="bound to stream"):
        run_proj_gemm(plan, None, None, None, None, h, stream=torch.cuda.default_stream().cuda_stream)


@requires_cuda
def test_graph_route_handles_are_cached_per_device_and_stream():
    import cudnn

    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    h1 = handle_for_stream(torch.device("cuda"), s1.cuda_stream)
    assert handle_for_stream(torch.device("cuda", torch.cuda.current_device()), s1.cuda_stream) is h1
    assert cudnn.get_stream(h1) == s1.cuda_stream
    h2 = handle_for_stream(torch.device("cuda"), s2.cuda_stream)
    assert h2 is not h1 and cudnn.get_stream(h2) == s2.cuda_stream
