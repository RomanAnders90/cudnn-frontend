# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row-local pointwise kernel of the MQA sparse-attention block (``kernels/norm_rope.py``): RMSNorm,
interleaved RoPE on the last dims, the block-32 ue8m0 fake-quant epilogue, the LSE fold.

The kernel is vectorized LDG/STG plus warp shuffles -- no tcgen05, no TMA -- so every arm WITHOUT the
fake-quant runs on any CUDA device (``requires_cuda``), and the tests say so rather than gating on Rubin.
The fake-quant arm needs the sm_100+ ``cvt.rp.satfinite.ue8m0x2.f32`` (``requires_quant_arch``); below that
``build_norm_rope`` declines with a typed ``NotImplementedError`` (asserted here on such a box).

Bars, as in plan section 4 row 2, with two spellings made precise:

* "within 1 bf16 ulp" for a ROTATION is measured at the pair's magnitude (``max(|x0|, |x1|)``): a rotation's
  natural error scale is its operand norm, and a result that cancels to ~0 has no meaningful relative ulp.
  The FORWARD RoPE turns out BITWISE with torch's complex multiply (the kernel spells nvcc's fma order), so
  that bar is ``torch.equal``; the inverse round trip is the two-rounding bar at the pair scale.
* the full ``kv_chain`` bar (ii): for every 32-block where the kernel and the mirror differ, EITHER every
  delta is within one step of the block's DEQUANTIZED e4m3 grid (a 1-bf16-ulp pre-quant difference that
  straddles an e4m3 rounding midpoint moves the output by one e4m3 CODE, i.e. 16 bf16 ulps for a normal
  code -- no correct kernel can do better, see ``_full_chain_bar``), OR the block amax sits within 1 bf16
  ulp of a ``448 * 2^k`` boundary and the block equals the mirror re-quantized with the neighbouring
  power-of-two scale; (iii) the output is idempotent under ``act_quant_mirror``.  The literal
  "1 bf16 ulp of the output" count is REPORTED next to it.  The un-quantized chain is separately held to
  1 bf16 ulp, and the quant epilogue to ``torch.equal`` on the kernel's own pre-quant values.
"""

import functools
import math

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("MQA sparse-attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.mqa_sparse_attention_block.kernels.norm_rope import (  # noqa: E402
    AMAX_FLOOR,
    E4M3_MAX,
    NormRopeRecipe,
    build_norm_rope,
    check_operands,
    compiled_cache,
    fake_quant_reference,
    lse_fold_reference,
    moved_bytes,
    norm_rope_reference,
    validate_shape,
)

# Rootdir-qualified (pytest runs from test/python): a bare `import mqa_block_reference` would collide across fe_api packages.
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R  # noqa: E402

pytestmark = pytest.mark.L0

_EPS = 1e-20  # M:641 / M:644
_THETA = 10000.0


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_quant_arch = pytest.mark.skipif(_cc() is None or _cc() < (10, 0), reason=f"fake_quant needs the sm_100+ ue8m0 cvt; found cc {_cc()}")
requires_pre_quant_arch = pytest.mark.skipif(_cc() is None or _cc() >= (10, 0), reason="the decline is only observable below sm_100")

Q_NORM = NormRopeRecipe(d=1280, heads=1, apply_norm=True, rope_dim=0)
Q_ROPE = NormRopeRecipe(d=512, heads=64, apply_norm=False, rope_dim=64)
O_UNROPE = NormRopeRecipe(d=512, heads=64, apply_norm=False, rope_dim=64, rope_inverse=True)
O_UNROPE_LSE = NormRopeRecipe(d=512, heads=64, apply_norm=False, rope_dim=64, rope_inverse=True, lse_fold=True)
KV_CHAIN = NormRopeRecipe(d=512, heads=1, apply_norm=True, rope_dim=64, fake_quant=True)
KV_CHAIN_ONE_ROUNDING = NormRopeRecipe(d=512, heads=1, apply_norm=True, rope_dim=64, fake_quant=True, three_roundings=False)
KV_PREQUANT = NormRopeRecipe(d=512, heads=1, apply_norm=True, rope_dim=64)
QUANT_ONLY = NormRopeRecipe(d=512, heads=1, apply_norm=False, rope_dim=0, fake_quant=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _gen(seed=0):
    return torch.Generator(device="cuda").manual_seed(seed)


def _ulp_bf16(scale: torch.Tensor) -> torch.Tensor:
    """bf16 spacing at ``|scale|`` (8 significant bits: ``2**(e - 8)`` with ``frexp``'s ``e``); the smallest
    subnormal at zero."""
    _, e = torch.frexp(scale.float().abs())
    ulp = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), e - 8)
    return torch.where(scale.float() == 0, torch.full_like(ulp, 2.0**-133), ulp)


def _within_one_ulp(got, want, scale=None):
    """``(ok, n_diff)``: every ``|got - want| <= ulp_bf16(scale or want)``, and the differing-element count."""
    err = (got.float() - want.float()).abs()
    ulp = _ulp_bf16(want if scale is None else scale)
    return bool((err <= ulp).all()), int((err > 0).sum())


def _pair_scale(x_tail: torch.Tensor) -> torch.Tensor:
    """The rotation's error scale per element: the larger magnitude of its adjacent pair."""
    p = x_tail.float().unflatten(-1, (-1, 2))
    return p.abs().amax(-1, keepdim=True).expand_as(p).flatten(-2)


def _tables(seq_len: int, device="cuda"):
    return R.build_rope_tables(seq_len, 64, _THETA, device=device)


@functools.lru_cache(maxsize=None)
def _torch_complex_mul_is_the_kernel_contraction(n: int = 1 << 16) -> bool:
    """Does THIS torch build's ``complex * complex`` on THIS device contract as ``fma(a, c, -(b*d))`` /
    ``fma(a, d, b*c)`` -- the kernel's spelling, which is nvcc's for the sm_80 SASS torch ships?

    On an arch the wheel carries no SASS for (Rubin under torch 2.13+cu130) the driver JIT-compiles torch's PTX and
    ptxas may fuse the products differently: kernel and torch then differ by one fp32 ulp on ~30 % of the elements,
    which the bf16 store hides except where the fp32 value straddles a bf16 rounding boundary (8 of 1.05M tail
    elements at T=257 on the Rubin dev node, 2026-09-17, every one within 1 bf16 ulp at the pair scale).  So the
    bitwise bar is asserted only where torch's contraction IS the kernel's; elsewhere the 1-bf16-ulp-at-pair-scale bar
    is the contract and the flip count is printed.  ``frost_dev/probe_torch_complex_mul_contraction.py`` prints the
    per-spelling match counts for a device."""
    g = torch.Generator(device="cuda").manual_seed(0)
    a, b, c, d = (torch.randn(n, generator=g, device="cuda") for _ in range(4))
    ref = torch.view_as_real(torch.complex(a, b) * torch.complex(c, d))

    def fma(x, y, z):  # the fp32 fma result: the exact product-sum in fp64 (48 + 24 bits fit), rounded once
        return (x.double() * y.double() + z.double()).float()

    return bool((fma(a, c, -(b * d)) == ref[:, 0]).all() and (fma(a, d, b * c) == ref[:, 1]).all())


def _e4m3_step(q: torch.Tensor) -> torch.Tensor:
    """Spacing of the e4m3 grid at ``|q|`` (``q`` an e4m3 value): ``2**(floor(log2 q) - 3)`` for normals
    (``q >= 2**-6``), ``2**-9`` for subnormals."""
    _, e = torch.frexp(q)
    step = torch.ldexp(torch.ones_like(q), e - 4)  # floor(log2 q) = e - 1
    return torch.where(q >= 2.0**-6, step, torch.full_like(q, 2.0**-9))


def _requant(z_blocks: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """K:81-86 with a GIVEN scale ``s`` (``[..., 1]``) on ``[..., 32]`` fp32 blocks -> bf16."""
    q = (z_blocks / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * s).to(torch.bfloat16)


def _full_chain_bar(got: torch.Tensor, z_m: torch.Tensor, want: torch.Tensor, s: torch.Tensor, block: int = 32) -> dict:
    """Plan 4 row 2 bar (ii) per differing 32-block; returns the counts and asserts no block fails both classes.

    ``got``: kernel bf16 ``[..., d]``; ``z_m``: the mirror's pre-quant bf16; ``want = act_quant_mirror(z_m)``;
    ``s``: its fp32 scales ``[..., d // 32]``.
    """
    gb = got.float().unflatten(-1, (-1, block))
    wb = want.float().unflatten(-1, (-1, block))
    zb = z_m.float().unflatten(-1, (-1, block))
    sb = s.unsqueeze(-1)
    delta = (gb - wb).abs()
    diff_blk = (delta > 0).any(-1)
    # class 1: every element within ONE STEP of the block's dequantized e4m3 grid (evaluated at the mirror value)
    noise_ok = (delta <= _e4m3_step(wb.abs() / sb) * sb).all(-1)
    literal_ok = (delta <= _ulp_bf16(wb)).all(-1)  # the plan's literal spelling, reported
    # class 2: amax within 1 bf16 ulp of a 448 * 2^k boundary -> the kernel took the neighbouring scale
    amax = zb.abs().amax(-1).clamp_min(AMAX_FLOOR)
    top, bottom = E4M3_MAX * s, E4M3_MAX * s / 2
    near_top = (amax - top).abs() <= _ulp_bf16(top)
    near_bottom = (amax - bottom).abs() <= _ulp_bf16(bottom)
    s_nb = torch.where(near_top, s * 2, torch.where(near_bottom, s / 2, s)).unsqueeze(-1)
    step_ok = (near_top | near_bottom) & (gb == _requant(zb, s_nb).float()).all(-1)
    bad = diff_blk & ~noise_ok & ~step_ok
    counts = dict(
        elements=int(delta.numel()),
        flipped_elements=int((delta > 0).sum()),
        differing_blocks=int(diff_blk.sum()),
        literal_one_bf16_ulp_blocks=int((diff_blk & literal_ok).sum()),
        one_e4m3_step_blocks=int((diff_blk & noise_ok).sum()),
        scale_step_blocks=int((diff_blk & step_ok & ~noise_ok).sum()),
        bad_blocks=int(bad.sum()),
    )
    print("full-chain bar:", counts)
    assert (
        counts["bad_blocks"] == 0
    ), f"blocks failing both classes of bar (ii): {counts}; worst |delta|/step = {float((delta / (_e4m3_step(wb.abs() / sb) * sb)).max())}"
    return counts


def _kv_case(batch, seq_len, n_c, seed):
    """``kv_all [B, S + N_c, 512]`` bf16 with ``x = kv_all[:, :S]`` (a strided batch view, as the block passes it)."""
    g = _gen(seed)
    kv_all = (
        torch.randn(batch, seq_len + n_c, 512, generator=g, device="cuda") * torch.logspace(-2, 2, batch * (seq_len + n_c), device="cuda").view(batch, -1, 1)
    ).to(torch.bfloat16)
    w = (torch.randn(512, generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)
    cos, sin = _tables(seq_len)
    return kv_all, w, cos, sin


def _kv_mirror(x, w, cos, sin):
    """``act_quant_mirror(rope(rmsnorm(x)))`` from the oracle's own twins: the three-rounding model chain."""
    z = R.apply_rope_interleaved(R.rmsnorm(x, w, _EPS), cos, sin)
    y, s = R.act_quant_mirror(z, 32, return_scales=True)
    return z, y, s


# ---------------------------------------------------------------------------
# Shape algebra, operand contract, byte model -- no GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe, lanes, chunks, tail",
    [
        (Q_NORM, 32, 5, False),
        (Q_ROPE, 8, 1, True),
        (O_UNROPE_LSE, 8, 1, True),
        (KV_CHAIN, 32, 2, False),
        (QUANT_ONLY, 32, 2, False),
        (NormRopeRecipe(64, 1, False, 64), 8, 1, True),
    ],
)
def test_lane_maps(recipe, lanes, chunks, tail):
    validate_shape(recipe)
    assert (recipe.lanes_per_row, recipe.chunks, recipe.tail_only) == (lanes, chunks, tail)
    width = recipe.rope_dim if tail else recipe.d
    assert lanes * chunks * 8 == width


@pytest.mark.parametrize(
    "recipe, threads, match",
    [
        (NormRopeRecipe(250, 1, True, 0), 128, "multiple of 8"),
        (NormRopeRecipe(576, 1, True, 0), 128, "not covered exactly"),
        (NormRopeRecipe(512, 1, True, 24), 128, "multiple of 16"),
        (NormRopeRecipe(256, 1, False, 512), 128, "exceeds d"),
        (NormRopeRecipe(1280, 1, True, 320), 128, "LAST access chunk"),
        (NormRopeRecipe(520, 1, True, 0, fake_quant=True), 128, "fake_quant needs d % 32"),
        (NormRopeRecipe(512, 1, False, 0, fake_quant=True, io_dtype=torch.float16), 128, "defined on bf16"),
        (NormRopeRecipe(96, 1, False, 48), 128, "must divide a warp"),
        (NormRopeRecipe(200, 1, False, 64), 128, "d % rope_dim == 0"),
        (NormRopeRecipe(512, 1, False, 0), 128, "identity"),
        (NormRopeRecipe(512, 1, True, 0, rope_inverse=True), 128, "rotates nothing"),
        (NormRopeRecipe(512, 1, True, 0, io_dtype=torch.float32), 128, "bf16 or f16"),
        (NormRopeRecipe(512, 0, True, 0), 128, "heads must be"),
        (NormRopeRecipe(512, 1, True, 0), 100, "multiple of a warp"),
    ],
)
def test_validate_shape_rejects(recipe, threads, match):
    with pytest.raises(ValueError, match=match):
        validate_shape(recipe, threads)


def test_moved_bytes_is_the_row_traffic():
    t = 32768
    assert moved_bytes(Q_NORM, t) == 2 * t * 1280 * 2
    assert moved_bytes(Q_ROPE, t) == 2 * t * 64 * 64 * 2  # tail map: only 128 B of every 1 KiB row
    assert moved_bytes(KV_CHAIN, t, 2) == 2 * 2 * t * 512 * 2
    assert moved_bytes(O_UNROPE_LSE, t) == 2 * t * 64 * 64 * 2 + 8 * t * 64


def test_check_operands_rejects_both_directions():
    """Weights / tables / fold operands must agree with the recipe in BOTH directions -- typed, before any launch."""
    x = torch.empty(4, 512, dtype=torch.bfloat16)
    x64 = torch.empty(4, 64, 512, dtype=torch.bfloat16)
    w = torch.ones(512, dtype=torch.bfloat16)
    cos = torch.ones(4, 32)
    lse = torch.zeros(4, 64)
    sink = torch.zeros(64)
    lse_out = torch.zeros(1, 64, 4)
    kv_prequant = NormRopeRecipe(512, 1, True, 64)
    with pytest.raises(ValueError, match="weight must be bound"):
        check_operands(kv_prequant, x, None, cos, cos)
    with pytest.raises(ValueError, match="pass w=None"):
        check_operands(QUANT_ONLY, x, w, None, None)
    with pytest.raises(ValueError, match="cos and sin must be bound"):
        check_operands(kv_prequant, x, w, None, None)
    with pytest.raises(ValueError, match="pass cos=sin=None"):
        check_operands(QUANT_ONLY, x, None, cos, cos)
    with pytest.raises(ValueError, match="lse_in, sink and lse_out must all be bound"):
        check_operands(O_UNROPE_LSE, x64, None, cos, cos)
    with pytest.raises(ValueError, match="lse_fold=False"):
        check_operands(O_UNROPE, x64, None, cos, cos, lse_in=lse, sink=sink, lse_out=lse_out)
    with pytest.raises(ValueError, match="io_dtype"):
        check_operands(QUANT_ONLY, x.float(), None, None, None)
    with pytest.raises(ValueError, match="needs heads == 1"):
        check_operands(Q_ROPE, x, None, cos, cos)
    with pytest.raises(ValueError, match="compact \\[Tok=4, rope_dim//2=32\\]"):
        check_operands(kv_prequant, x, w, torch.ones(4, 16), cos)
    with pytest.raises(ValueError, match="lse_out must be"):
        check_operands(O_UNROPE_LSE, x64, None, cos, cos, lse_in=lse, sink=sink, lse_out=torch.zeros(1, 4, 64))
    with pytest.raises(ValueError, match="sink must be"):
        check_operands(O_UNROPE_LSE, x64, None, cos, cos, lse_in=lse, sink=torch.zeros(63), lse_out=lse_out)
    with pytest.raises(ValueError, match="n_rows_max"):
        check_operands(QUANT_ONLY, x, None, None, None, n_rows_max=3)
    with pytest.raises(ValueError, match="compact rows"):
        check_operands(QUANT_ONLY, torch.empty(2, 4, 1024, dtype=torch.bfloat16)[..., :512], None, None, None)
    # every bound operand shares x's device (a meta x stands in for a GPU one here: data_ptr() == 0 passes alignment)
    x_meta = torch.empty(4, 512, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="w lives on cpu, x on meta"):
        check_operands(kv_prequant, x_meta, w, cos.to("meta"), cos.to("meta"))
    with pytest.raises(ValueError, match="sin lives on cpu, x on meta"):
        check_operands(kv_prequant, x_meta, w.to("meta"), cos.to("meta"), cos)
    with pytest.raises(ValueError, match="lse_out lives on cpu, x on meta"):
        check_operands(O_UNROPE_LSE, x64.to("meta"), None, cos.to("meta"), cos.to("meta"), lse_in=lse.to("meta"), sink=sink.to("meta"), lse_out=lse_out)
    # the accepted forms normalise to [B, R, d]
    ops = check_operands(O_UNROPE_LSE, x64, None, cos, cos, lse_in=lse, sink=sink, lse_out=lse_out)
    assert tuple(ops.x.shape) == (1, 256, 512) and tuple(ops.cos.shape) == (1, 4, 32) and ops.cos.stride(0) == 0
    kv_all = torch.empty(2, 6, 512, dtype=torch.bfloat16)
    ops = check_operands(KV_CHAIN, kv_all[:, :4], w, cos, cos)
    assert tuple(ops.x.shape) == (2, 4, 512) and ops.x.stride(0) == 6 * 512 and (ops.n_batch, ops.n_rows, ops.n_tok) == (2, 4, 4)


def test_reference_is_the_oracle_composition_cpu():
    """``norm_rope_reference`` is bit-identical to the oracle's own twins composed (CPU, no kernel)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 13, 512, generator=g).to(torch.bfloat16)
    w = (torch.randn(512, generator=g) * 0.5 + 1).to(torch.bfloat16)
    cos, sin = _tables(13, device="cpu")
    z, y, _ = _kv_mirror(x, w, cos, sin)
    got, none = norm_rope_reference(KV_CHAIN, x, w, cos, sin, eps=_EPS)
    assert none is None and torch.equal(got, y)
    assert torch.equal(norm_rope_reference(KV_PREQUANT, x, w, cos, sin, eps=_EPS)[0], z)
    assert torch.equal(norm_rope_reference(QUANT_ONLY, x, None, None, None)[0], R.act_quant_mirror(x, 32))
    assert torch.equal(fake_quant_reference(x.float()).to(torch.bfloat16), R.act_quant_mirror(x, 32))
    xq = torch.randn(2, 13, 64, 512, generator=g).to(torch.bfloat16)
    assert torch.equal(norm_rope_reference(Q_ROPE, xq, None, cos, sin)[0], R.apply_rope_interleaved(xq, cos, sin))
    assert torch.equal(norm_rope_reference(O_UNROPE, xq, None, cos, sin)[0], R.apply_rope_interleaved(xq, cos, sin, inverse=True))
    xn = torch.randn(13, 1280, generator=g).to(torch.bfloat16)
    wn = torch.randn(1280, generator=g).to(torch.bfloat16)
    assert torch.equal(norm_rope_reference(Q_NORM, xn, wn, None, None, eps=_EPS)[0], R.rmsnorm(xn, wn, _EPS))
    # the LSE fold twin, corner by corner
    inf = float("inf")
    lse = torch.tensor([[1.0, inf, -inf, 2.0, inf, -inf]]).T.expand(6, 6).contiguous()
    sink = torch.tensor([0.5, inf, -inf, -inf, -inf, inf])
    got = lse_fold_reference(lse, sink)
    for r in range(6):
        for h in range(6):
            a, s = float(lse[r, h]), float(sink[h])
            want = s if a == inf else (inf if s == inf else float(torch.logaddexp(torch.tensor(a), torch.tensor(s))))
            assert float(got[r, h]) == want, (r, h, a, s, float(got[r, h]), want)


# ---------------------------------------------------------------------------
# Any CUDA device: q_norm, q_rope, o_unrope (+ LSE fold)
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("t", [1, 3, 13, 257, 1024])
def test_q_norm_within_one_bf16_ulp(t):
    """d=1280 RMSNorm vs ``R.rmsnorm``: the only legal differences are fp32 summation order and the rsqrt
    (a few fp32 ulps), i.e. at most 1 bf16 ulp after the store.  Ragged CTAs at every t here (8 rows per CTA)."""
    g = _gen(t)
    x = torch.randn(t, 1280, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(1280, generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)
    k = build_norm_rope(Q_NORM, n_rows_max=1 << 20, device=x.device, eps=_EPS)
    want = R.rmsnorm(x, w, _EPS)
    assert torch.equal(norm_rope_reference(Q_NORM, x, w, None, None, eps=_EPS)[0], want)
    y = x.clone()
    k.run(y, w, None, None, stream=_stream())
    torch.cuda.synchronize()
    ok, n_diff = _within_one_ulp(y, want)
    print(f"q_norm t={t}: {n_diff}/{y.numel()} elements differ from the oracle (all within 1 bf16 ulp: {ok})")
    assert ok, f"q_norm t={t}: a delta above 1 bf16 ulp; {n_diff} differing elements"
    y2 = x.clone()
    k.run(y2, w, None, None, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(y, y2), "two launches are not bit-identical"


@requires_cuda
@pytest.mark.parametrize("t", [1, 3, 13, 257])
def test_q_rope_is_bitwise_with_the_model_rotation_and_passes_through(t):
    """Tail-only map on ``[T, 64, 512]``: the untouched dims never leave memory (``torch.equal``), the rotated
    tail is within 1 bf16 ulp at the pair scale of ``apply_rotary_emb`` -- and BIT-IDENTICAL to it wherever torch's
    complex multiply uses the kernel's fma contraction (``_torch_complex_mul_is_the_kernel_contraction``) -- and a
    second launch is bit-identical."""
    g = _gen(100 + t)
    x = torch.randn(t, 64, 512, generator=g, device="cuda").to(torch.bfloat16)
    cos, sin = _tables(t)
    k = build_norm_rope(Q_ROPE, n_rows_max=1 << 22, device=x.device)
    y = x.clone()
    k.run(y, None, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    want = R.apply_rope_interleaved(x[None], cos, sin)[0]
    assert torch.equal(y[..., :448], x[..., :448]), "passthrough dims are not a bit-exact copy"
    if t > 1:
        assert not torch.equal(y[1:, :, 448:], x[1:, :, 448:]), "the tail did not rotate"
    assert torch.equal(y[0], x[0]), "position 0 is the identity rotation"
    ok, n_diff = _within_one_ulp(y[..., 448:], want[..., 448:], scale=_pair_scale(x[..., 448:]))
    assert ok, f"q_rope t={t}: {n_diff} tail elements differ from torch and at least one by more than 1 bf16 ulp at the pair scale"
    if _torch_complex_mul_is_the_kernel_contraction():
        assert torch.equal(y, want), f"RoPE differs from torch's complex multiply on {n_diff} elements although torch contracts as the kernel does"
    else:
        print(
            f"q_rope t={t}: torch's complex multiply is not the kernel's fma contraction on this device; {n_diff}/{y[..., 448:].numel()} tail elements differ, all within 1 bf16 ulp at the pair scale"
        )
    y2 = x.clone()
    k.run(y2, None, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(y, y2)


@requires_cuda
@pytest.mark.parametrize("t", [1, 3, 13, 257])
def test_tail_map_ragged_cta_rows(t):
    """``heads=1, d=64, rope_dim=64``: 32 rows per CTA on the tail map, so every t here leaves a ragged last
    CTA whose clamped loads must not corrupt the rows that exist (a sentinel guards the tail of a bigger buffer)."""
    g = _gen(200 + t)
    buf = torch.full((t + 40, 64), 1.5e3, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(t, 64, generator=g, device="cuda").to(torch.bfloat16)
    buf[:t] = x
    cos, sin = _tables(t)
    r = NormRopeRecipe(64, 1, False, 64)
    k = build_norm_rope(r, n_rows_max=1 << 20, device=x.device)
    k.run(buf[:t], None, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(buf[:t], R.apply_rope_interleaved(x[None], cos, sin)[0])
    assert torch.all(buf[t:] == 1.5e3), "a tail CTA wrote past the last row"


@requires_cuda
def test_inverse_rope_is_the_conjugate_and_round_trips():
    """``o_unrope``: within 1 bf16 ulp at the pair scale of ``apply_rotary_emb(..., inverse=True)`` (bitwise where
    torch's complex multiply is the kernel's contraction, see ``_torch_complex_mul_is_the_kernel_contraction``);
    ``inverse(rope(x))`` is within the two bf16 roundings (1 ulp at the pair scale) of ``x``."""
    t = 257
    g = _gen(7)
    x = torch.randn(t, 64, 512, generator=g, device="cuda").to(torch.bfloat16)
    cos, sin = _tables(t)
    kf = build_norm_rope(Q_ROPE, n_rows_max=1 << 22, device=x.device)
    ki = build_norm_rope(O_UNROPE, n_rows_max=1 << 22, device=x.device)
    assert kf.compiled is not ki.compiled
    y = x.clone()
    kf.run(y, None, cos, sin, stream=_stream())
    z = y.clone()
    ki.run(z, None, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    want_z = R.apply_rope_interleaved(y[None], cos, sin, inverse=True)[0]
    assert torch.equal(z[..., :448], want_z[..., :448])
    ok_z, n_z = _within_one_ulp(z[..., 448:], want_z[..., 448:], scale=_pair_scale(y[..., 448:]))
    assert ok_z, f"inverse RoPE: {n_z} tail elements differ from torch's conjugate multiply and at least one by more than 1 bf16 ulp at the pair scale"
    if _torch_complex_mul_is_the_kernel_contraction():
        assert torch.equal(z, want_z), f"inverse RoPE differs from torch on {n_z} elements although torch contracts as the kernel does"
    else:
        print(
            f"o_unrope: torch's complex multiply is not the kernel's fma contraction on this device; {n_z}/{z[..., 448:].numel()} tail elements differ, all within 1 bf16 ulp at the pair scale"
        )
    assert torch.equal(z[..., :448], x[..., :448])
    ok, n_diff = _within_one_ulp(z[..., 448:], x[..., 448:], scale=_pair_scale(x[..., 448:]))
    print(f"inverse(rope(x)) round trip: {n_diff}/{z[..., 448:].numel()} tail elements differ from x, all within 1 bf16 ulp of the pair scale: {ok}")
    assert ok
    # inverse == forward with -sin (M:399 conj)
    z2 = y.clone()
    kf.run(z2, None, cos, (-sin).contiguous(), stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(z2, z)


@requires_cuda
def test_lse_fold_corners_and_x_forms_agree():
    """``o_unrope`` with the fold on ``[B, S, 64, 512]`` + ``[B, S, 32]`` tables (B=2): every corner of
    ``select(lse == +inf, sink, select(sink == +inf, +inf, logaddexp))`` exact, finite rows against
    ``torch.logaddexp``; the flat ``[T, 64, 512]`` + ``[T, 32]`` form gives the same bytes."""
    b, s = 2, 13
    g = _gen(9)
    x = torch.randn(b, s, 64, 512, generator=g, device="cuda").to(torch.bfloat16)
    cos, sin = _tables(s)
    cos_b, sin_b = cos.unsqueeze(0).expand(b, s, 32).contiguous(), sin.unsqueeze(0).expand(b, s, 32).contiguous()
    inf = float("inf")
    lse_in = torch.randn(b * s, 64, generator=g, device="cuda") * 3
    lse_in[0, :8] = inf  # keyless rows (the H64 convention)
    lse_in[3, 2] = -inf
    lse_in[3, 5] = -inf
    sink = torch.randn(64, generator=g, device="cuda")
    sink[1] = inf
    sink[2] = -inf
    sink[3] = inf
    sink[5] = -inf
    k = build_norm_rope(O_UNROPE_LSE, n_rows_max=1 << 22, device=x.device)
    y = x.clone()
    lse_out = torch.full((b, 64, s), 7.0, device="cuda")
    k.run(y, None, cos_b, sin_b, lse_in=lse_in, sink=sink, lse_out=lse_out, stream=_stream())
    torch.cuda.synchronize()
    want_y, want_lse = norm_rope_reference(O_UNROPE_LSE, x, None, cos_b, sin_b, lse_in=lse_in, sink=sink)
    assert torch.equal(y, want_y)
    assert torch.equal(y, R.apply_rope_interleaved(x, cos_b, sin_b, inverse=True))
    lse_th = lse_in.view(b, s, 64).permute(0, 2, 1)  # [B, heads, S]
    sink_b = sink.view(1, 64, 1)
    # corners, asserted on the OUTPUT directly
    assert torch.equal(lse_out[:, :, 0][lse_th[:, :, 0] == inf], sink[lse_th[0, :, 0] == inf]), "lse == +inf must yield sink"
    assert torch.all(lse_out[:, 1] == inf) and torch.all(lse_out[:, 3] == inf), "sink == +inf must yield +inf on live rows"
    live = torch.isfinite(lse_th)
    assert torch.equal(lse_out[:, 2][live[:, 2]], lse_th[:, 2][live[:, 2]]), "sink == -inf must yield lse exactly"
    assert lse_out[0, 2, 3] == -inf and lse_out[0, 5, 3] == -inf, "logaddexp(-inf, -inf) is -inf"
    # live rows with a finite sink: the fp32 logaddexp itself (exp / log1p lowering vs torch's expf / log1pf)
    live_finite = live & torch.isfinite(sink_b).expand_as(live)
    torch.testing.assert_close(lse_out[live_finite], torch.logaddexp(lse_th, sink_b)[live_finite], rtol=1e-6, atol=2e-6)
    # every other cell is a corner and must be EXACT against the select spelling
    assert torch.equal(lse_out[~live_finite], want_lse[~live_finite])
    torch.testing.assert_close(lse_out, want_lse, rtol=1e-6, atol=2e-6, equal_nan=False)
    assert not torch.any(lse_out == 7.0), "an LSE cell was never written"
    # the flat form
    y3 = x.view(b * s, 64, 512).clone()
    lse_out3 = torch.empty(1, 64, b * s, device="cuda")
    k.run(y3, None, cos_b.view(b * s, 32), sin_b.view(b * s, 32), lse_in=lse_in, sink=sink, lse_out=lse_out3, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(y3.view(b, s, 64, 512), y)
    assert torch.equal(lse_out3.view(64, b, s).permute(1, 0, 2), lse_out)


@requires_cuda
def test_full_map_norm_rope_on_a_strided_batch_view():
    """The kv_chain WITHOUT the quant (runs anywhere): the full-row map's in-lane RoPE tail on ``kv_all[:, :S]``
    (batch stride ``(S + N_c) * 512``), within 1 bf16 ulp of ``rope(rmsnorm(x))``; the compressed rows untouched."""
    for b, s in ((1, 13), (2, 257), (2, 1000)):
        kv_all, w, cos, sin = _kv_case(b, s, s // 2, seed=s)
        sentinel = kv_all[:, s:].clone()
        x = kv_all[:, :s]
        want = R.apply_rope_interleaved(R.rmsnorm(x, w, _EPS), cos, sin)
        k = build_norm_rope(KV_PREQUANT, n_rows_max=1 << 22, device=x.device, eps=_EPS)
        k.run(x, w, cos, sin, stream=_stream())
        torch.cuda.synchronize()
        ok, n_diff = _within_one_ulp(x, want)
        print(f"kv norm+rope B={b} S={s}: {n_diff}/{x.numel()} differ from the oracle, all within 1 bf16 ulp: {ok}")
        assert ok
        assert torch.equal(kv_all[:, s:], sentinel), "the kernel wrote into the compressed rows"


@requires_cuda
def test_cache_key_and_bounds():
    """Same recipe -> the same artifact; any recipe field -> a distinct one (a norm-on artifact bound to no
    weight dereferences null).  ``n_rows_max`` and the bound device are enforced before any launch."""
    n0 = len(compiled_cache)
    a = build_norm_rope(KV_PREQUANT, n_rows_max=64, device=0, eps=_EPS)
    b = build_norm_rope(KV_PREQUANT, n_rows_max=64, device=0, eps=_EPS)
    c = build_norm_rope(NormRopeRecipe(512, 1, True, 64, rope_inverse=True), n_rows_max=64, device=0, eps=_EPS)
    assert a.compiled is b.compiled and a.compiled is not c.compiled and len(compiled_cache) >= n0 + 1
    x = torch.zeros(65, 512, dtype=torch.bfloat16, device="cuda")
    cos = torch.zeros(65, 32, device="cuda")
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="n_rows_max=64"):
        a.run(x, w, cos, cos, stream=_stream())
    with pytest.raises(ValueError, match="built for cuda:0"):
        a.run(x[:64].cpu(), w.cpu(), cos[:64].cpu(), cos[:64].cpu(), stream=_stream())
    with pytest.raises(ValueError, match="w lives on cpu, x on cuda:0"):  # a host weight passes every other gate
        a.run(x[:64], w.cpu(), cos[:64], cos[:64], stream=_stream())
    with pytest.raises(ValueError, match="cos lives on cpu, x on cuda:0"):
        a.run(x[:64], w, cos[:64].cpu(), cos[:64], stream=_stream())
    with pytest.raises(ValueError, match="n_rows_max must be"):
        build_norm_rope(KV_PREQUANT, n_rows_max=0, device=0)
    a.run(x[:0], w, cos[:0], cos[:0], stream=_stream())  # zero rows: a validated no-op, not a zero grid


@requires_pre_quant_arch
def test_fake_quant_declines_below_sm100():
    with pytest.raises(NotImplementedError, match="sm_100"):
        build_norm_rope(KV_CHAIN, n_rows_max=64, device=0)


# ---------------------------------------------------------------------------
# sm_100+ (Rubin dev node): the fake-quant epilogue
# ---------------------------------------------------------------------------


@requires_quant_arch
def test_quant_only_is_bitwise_with_act_quant_mirror():
    """Bar (i): the epilogue alone, ``torch.equal`` against the oracle, on every corner the model's ``act_quant``
    has -- the all-zero block (floor scale ``2**-22``), a block straddling the ``1e-4`` amax floor, an amax
    exactly ``448 * 2**k``, one bf16 ulp above it (a scale step), sub-floor values, values already on the grid."""
    g = _gen(11)
    rows = 64
    x = (torch.randn(rows, 512, generator=g, device="cuda") * torch.logspace(-6, 4, rows, device="cuda")[:, None]).to(torch.bfloat16)
    x[0].zero_()  # every block all-zero
    x[1, :32].zero_()
    x[1, 40] = 1.0  # a zero block next to a live one
    x[2].zero_()
    x[2, 3] = 448.0  # amax * fp32(1/448) == 1.0 exactly -> scale 1
    x[2, 5] = -224.0
    x[3].zero_()
    x[3, 3] = 450.0  # one bf16 ulp above 448 -> scale 2
    x[4].zero_()
    x[4, 33] = 1e-4  # bf16(1e-4) is just below the floor
    x[4, 65] = 1.0001e-4  # just above
    x[4, 97] = 3e-5  # sub-floor: quantised against 2**-22, not its own amax
    x[5] = R.act_quant_mirror(x[5][None], 32)[0]  # already on the grid: fixed point
    x[6] = torch.tensor(448.0 * 2**-7, device="cuda").to(torch.bfloat16)
    # The FINITE bf16 max (255 * 2**120): amax * fp32(1/448) -> scale 2**120, e4m3(-255) = -256, -256 * 2**120 overflows
    # to -inf in fp32 on both sides.  NOT ``-3.4e38``: that literal rounds to -inf in bf16 (bf16 max is 3.3895e38), and an
    # inf amax is out of contract for both the kernel (``e8m0_from_amax``) and the mirror (``frexp(inf)`` -> scale 1 -> -448,
    # where the model's ``fast_log2_ceil`` gives NaN) -- the first cut compared two undefined answers (Rubin, 2026-09-17).
    x[7, :] = -torch.finfo(torch.bfloat16).max
    want = R.act_quant_mirror(x, 32)
    k = build_norm_rope(QUANT_ONLY, n_rows_max=1 << 20, device=x.device)
    y = x.clone()
    k.run(y, None, None, None, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(
        y, want
    ), f"quant-only differs from act_quant_mirror on {int((y != want).sum())} elements; first rows {torch.nonzero(y != want)[:8].tolist()}"
    assert torch.equal(y[0], x[0]) and torch.equal(y[5], x[5])
    assert bool((y[7] == -math.inf).all()), "the bf16-max row must overflow to -inf in the dequant"
    # Idempotence on the FINITE rows: an inf row is outside the mirror's contract (``frexp(inf)`` -> scale 1 -> -448,
    # where the model's ``fast_log2_ceil`` gives NaN), so re-quantising row 7 is not a fixed point on either side.
    finite = torch.isfinite(y.float()).all(-1)
    assert int(finite.sum()) == rows - 1
    assert torch.equal(R.act_quant_mirror(y[finite], 32), y[finite]), "not idempotent"
    y2 = x.clone()
    k.run(y2, None, None, None, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(y, y2)


@requires_quant_arch
@pytest.mark.parametrize("b, s", [(1, 13), (2, 257), (2, 1000)])
def test_kv_chain_full_bar(b, s):
    """Bars (ii) + (iii) on ``kv_all[:, :S]`` (the block's strided view) against
    ``act_quant_mirror(rope(rmsnorm(x)))``; flip rate and scale-step count printed; two launches bitwise."""
    kv_all, w, cos, sin = _kv_case(b, s, s // 2, seed=1000 + s)
    sentinel = kv_all[:, s:].clone()
    x = kv_all[:, :s]
    x0 = x.clone()
    z_m, want, scales = _kv_mirror(x0, w, cos, sin)
    k = build_norm_rope(KV_CHAIN, n_rows_max=1 << 22, device=x.device, eps=_EPS)
    k.run(x, w, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    _full_chain_bar(x, z_m, want, scales)
    assert torch.equal(R.act_quant_mirror(x, 32), x), "bar (iii): the output is not a fixed point of act_quant_mirror"
    assert torch.equal(kv_all[:, s:], sentinel)
    y2 = x0.clone()
    k.run(y2, w, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    assert torch.equal(y2, x)


@requires_quant_arch
@pytest.mark.parametrize("b, s", [(2, 257), (2, 1000)])
def test_kv_chain_quant_epilogue_is_bitwise_on_the_kernels_own_prequant(b, s):
    """The strict decomposition of bar (ii): the pre-quant chain (norm + RoPE, three roundings) is within 1 bf16
    ulp of the oracle -- the element's OWN ulp on the 448 normed-only dims, the PAIR-scale ulp on the 64 rotated
    dims (a rotated element near cancellation inherits its partner's magnitude in the error: ``[1,694,451]`` at
    S=1000 is 3e-8 from a 0.023 pair, 2 own-ulps and 0.000 pair-ulps off, Rubin 2026-09-17, where torch's
    imaginary-part fma contraction differs from the kernel's, see ``_torch_complex_mul_is_the_kernel_contraction``)
    -- and the quant epilogue applied to the KERNEL's pre-quant values is ``torch.equal`` to ``act_quant_mirror`` of
    them -- so every output difference is a pre-quant rounding event, never the quant."""
    kv_all, w, cos, sin = _kv_case(b, s, 0, seed=2000 + s)
    x0 = kv_all.clone()
    z_k = kv_all.clone()
    build_norm_rope(KV_PREQUANT, n_rows_max=1 << 22, device=x0.device, eps=_EPS).run(z_k, w, cos, sin, stream=_stream())
    out = x0.clone()
    build_norm_rope(KV_CHAIN, n_rows_max=1 << 22, device=x0.device, eps=_EPS).run(out, w, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    normed = R.rmsnorm(x0, w, _EPS)
    want_pre = R.apply_rope_interleaved(normed, cos, sin)
    ok_head, n_head = _within_one_ulp(z_k[..., :448], want_pre[..., :448])
    ok_tail, n_tail = _within_one_ulp(z_k[..., 448:], want_pre[..., 448:], scale=_pair_scale(normed[..., 448:]))
    print(
        f"kv pre-quant chain B={b} S={s}: {n_head}/{z_k[..., :448].numel()} normed dims differ (all within 1 own bf16 ulp: {ok_head}); "
        f"{n_tail}/{z_k[..., 448:].numel()} rotated dims differ (all within 1 bf16 ulp at the pair scale: {ok_tail}); "
        f"torch complex-mul is the kernel's contraction here: {_torch_complex_mul_is_the_kernel_contraction()}"
    )
    assert ok_head and ok_tail
    assert torch.equal(
        out, R.act_quant_mirror(z_k, 32)
    ), f"quant epilogue differs from act_quant_mirror on the kernel's own pre-quant values: {int((out != R.act_quant_mirror(z_k, 32)).sum())} elements"


@requires_quant_arch
def test_one_rounding_arm():
    """``three_roundings=False`` keeps norm and RoPE in fp32: it is a DIFFERENT function from the model's chain
    (they must disagree somewhere at this size) and matches its own torch mirror under bar (ii)."""
    b, s = 2, 257
    kv_all, w, cos, sin = _kv_case(b, s, 0, seed=3000)
    x0 = kv_all.clone()
    want, _ = norm_rope_reference(KV_CHAIN_ONE_ROUNDING, x0, w, cos, sin, eps=_EPS)
    z_m = norm_rope_reference(NormRopeRecipe(512, 1, True, 64, three_roundings=False), x0, w, cos, sin, eps=_EPS)[0]
    _, scales = R.act_quant_mirror(z_m, 32, return_scales=True)
    out = x0.clone()
    build_norm_rope(KV_CHAIN_ONE_ROUNDING, n_rows_max=1 << 22, device=x0.device, eps=_EPS).run(out, w, cos, sin, stream=_stream())
    three = x0.clone()
    build_norm_rope(KV_CHAIN, n_rows_max=1 << 22, device=x0.device, eps=_EPS).run(three, w, cos, sin, stream=_stream())
    torch.cuda.synchronize()
    # the one-rounding mirror rounds z_m once to bf16 only at the quant; bar (ii) against that mirror's grid
    _full_chain_bar(out, z_m, want, scales)
    assert torch.equal(R.act_quant_mirror(out, 32), out)
    assert not torch.equal(out, three), "the rounding knob changed nothing"
