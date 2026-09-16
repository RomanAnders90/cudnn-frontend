# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial review of ``mqa_block_reference.py`` -- counterexamples and source-level pins.

Two kinds of test live here:

* COUNTEREXAMPLES (``test_dense_masked_keyless_row_with_neg_inf_sink_agrees_with_gathered``,
  ``test_exact_math_cross_check_is_tf32_immune``): written RED against the
  reviewed ``mqa_block_reference.py`` (2026-09-16) and green since the two findings they
  document were fixed (the dense oracle floors its max and SELECTs the empty
  denominator; the exact-math arm computes in fp64). They stay as regression
  pins. Do not widen their tolerances.
* PINS against the SOURCE (``kernel.py`` ``K:`` / ``model.py`` ``M:`` of the
  V4.1-Flash release), spelled independently of ``mqa_block_reference.py``: a sequential
  emulation of the ``sparse_attn_kernel`` loop (K:350-387), the bit-level
  ``fast_log2_ceil`` / ``fast_pow2`` twin (K:22-37), an independent e2m1
  round-half-even, and the compress_kv RoPE position slice (M:754). These pass
  today and guard the oracle against drift.
"""

import math

import pytest
import torch

# Rootdir-qualified: a bare `import reference` collides with fe_api/gated_attention_block/mqa_block_reference.py in one
# pytest session (`pytest fe_api/`).
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R

pytestmark = pytest.mark.L0

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_EXACT = dict(p_dtype=torch.float32, out_dtype=torch.float32)


def _case(device, *, batch, seq_len, ratio, h=4, d=64, window=128, topk=512, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq_len, h, d, generator=g, device=device).to(torch.bfloat16)
    kv = torch.randn(batch, seq_len, d, generator=g, device=device).to(torch.bfloat16)
    idxs = R.window_idxs(batch, seq_len, window, device=device)
    if ratio > 0 and seq_len // ratio > 0:
        kv = torch.cat([kv, torch.randn(batch, seq_len // ratio, d, generator=g, device=device).to(torch.bfloat16)], dim=1)
        idxs = torch.cat([idxs, R.compressed_idxs_synthetic(batch, seq_len, ratio, topk, g, device=device)], dim=-1)
    sink = torch.randn(h, generator=g, device=device, dtype=torch.float32)
    return q, kv, sink, idxs, float(d) ** -0.5


# ---------------------------------------------------------------------------
# COUNTEREXAMPLE 1: the two oracles disagree on a keyless row when sink = -inf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_dense_masked_keyless_row_with_neg_inf_sink_agrees_with_gathered(device):
    """``dense_masked_reference`` says its row max is "finite: the sink column is always
    there" -- false when ``sink = -inf``: an all-``-1`` row then has every column at
    ``-inf``, ``m = -inf``, ``exp(-inf - -inf) = NaN``. The gathered oracle SELECTs
    ``O = 0``, ``LSE = -inf`` there (its documented FROST convention). The two oracles
    must agree on every input they both accept."""
    q, kv, sink, idxs, scale = _case(device, batch=1, seq_len=8, ratio=0, window=4)
    idxs = idxs.clone()
    idxs[:, 3, :] = -1
    no_sink = torch.full_like(sink, -torch.inf)
    o_g, lse_g = R.sparse_attention_reference(q, kv, no_sink, idxs, scale)
    o_d, lse_d = R.dense_masked_reference(q, kv, no_sink, idxs, scale)
    assert torch.equal(o_g[:, 3], torch.zeros_like(o_g[:, 3])) and torch.isneginf(lse_g[:, :, 3]).all()  # the gathered convention
    assert torch.isfinite(o_d[:, 3].float()).all(), "dense oracle: NaN on a keyless row with sink = -inf"
    assert torch.equal(o_d[:, 3], o_g[:, 3])
    assert torch.equal(lse_d[:, :, 3], lse_g[:, :, 3])


# ---------------------------------------------------------------------------
# COUNTEREXAMPLE 2: the exact-math variant is a TF32 reference in the CI containers
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is a CUDA matmul mode")
def test_exact_math_cross_check_is_tf32_immune():
    """The DLFW CI containers force ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` (see
    ``.claude/rules/frost-gotchas.md``, "A hand-rolled fp32 reference is a TF32
    reference in the CI containers"). The ``p_dtype=float32`` "exact-math" variant
    feeds a full-precision fp32 ``P`` into an einsum, which TF32 rounds to 10 mantissa
    bits, so ``test_gathered_matches_dense_masked[..-cuda]`` and
    ``test_duplicate_ids_count_twice[cuda]`` fail there (12 of 21 CUDA arms on an A100,
    2026-09-16). An exact-math variant must not depend on a process-wide flag: compute
    it in float64 (``.double()`` the operands) or pin the matmul precision inside the
    oracle. The bf16-``P`` default is bf16-valued on both GEMM inputs and is immune
    (pinned below)."""
    q, kv, sink, idxs, scale = _case("cuda", batch=2, seq_len=300, ratio=1)
    prev_flag, prev_prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        o_g, lse_g = R.sparse_attention_reference(q, kv, sink, idxs, scale, **_EXACT)
        o_d, lse_d = R.dense_masked_reference(q, kv, sink, idxs, scale, **_EXACT)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_flag
        torch.set_float32_matmul_precision(prev_prec)
    torch.testing.assert_close(o_g, o_d, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse_g, lse_d, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is a CUDA matmul mode")
def test_default_bf16_p_oracles_stay_within_budget_under_tf32():
    """The kernel-mirror default (bf16 ``P``) has bf16-valued operands on every GEMM
    input, so TF32 (10 mantissa bits) reproduces its PRODUCTS exactly; what moves is
    only the fp32 accumulation order of the tensor-core kernel -- a few 1-ulp bf16
    flips of ``O`` (measured: the dense spelling flips some elements at S=300 ratio=1,
    the gathered one none), never the 3e-4 shift the fp32-``P`` arm shows above."""
    q, kv, sink, idxs, scale = _case("cuda", batch=2, seq_len=300, ratio=1)
    prev_flag, prev_prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        o_g0, lse_g0 = R.sparse_attention_reference(q, kv, sink, idxs, scale)
        o_d0, lse_d0 = R.dense_masked_reference(q, kv, sink, idxs, scale)
        torch.backends.cuda.matmul.allow_tf32 = True
        o_g1, lse_g1 = R.sparse_attention_reference(q, kv, sink, idxs, scale)
        o_d1, lse_d1 = R.dense_masked_reference(q, kv, sink, idxs, scale)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_flag
        torch.set_float32_matmul_precision(prev_prec)
    atol = 2**-8 * float(kv.float().abs().max())
    for o0, o1 in ((o_g0, o_g1), (o_d0, o_d1)):
        diff = (o0.float() - o1.float()).abs()
        assert (diff <= 2**-7 * o0.float().abs() + atol).all(), float(diff.max())
        assert (diff > 0).float().mean() < 0.01, "more than 1 % of O flipped: not accumulation-order noise"
    torch.testing.assert_close(lse_g0, lse_g1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_d0, lse_d1, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# PIN: the gathered oracle vs a sequential emulation of sparse_attn_kernel (K:350-387)
# ---------------------------------------------------------------------------


def _sparse_attn_kernel_emulation(q, kv, sink, idxs, scale, block=64):
    """K:350-387 spelled step by step: 64-slot blocks, running max from the finite floor,
    ``alpha`` rescale, fp32 row-sum, ``P`` cast to bf16 against the RUNNING max, the sink
    folded into the denominator once at the end. Returns ``(o bf16, lse_frost)`` where
    ``lse_frost = log(sum_exp) + scores_max`` is the sink-included LSE the loop implies."""
    b, s, h, d = q.shape
    topk = idxs.shape[-1]
    pad = (-topk) % block
    idx = idxs.long()
    if pad:
        idx = torch.cat([idx, torch.full((b, s, pad), -1, dtype=torch.long, device=q.device)], dim=-1)  # K:360
    qf = q.float()
    acc_o = torch.zeros(b, s, h, d, device=q.device)  # K:350
    sum_exp = torch.zeros(b, s, h, device=q.device)  # K:351
    m = torch.full((b, s, h), -1e30, device=q.device)  # K:355
    bidx = torch.arange(b, device=q.device)[:, None, None]
    for t0 in range(0, idx.shape[-1], block):
        idb = idx[..., t0 : t0 + block]
        valid = idb != -1
        kvb = kv.float()[bidx, idb.clamp_min(0)] * valid.unsqueeze(-1)  # K:362 zero row
        acc_s = torch.where(valid.unsqueeze(2), 0.0, -math.inf) + torch.einsum("bshd,bskd->bshk", qf, kvb)  # K:364-365
        acc_s = acc_s * scale  # K:366-367 AFTER the GEMM
        m_prev = m.clone()  # K:368
        m = torch.maximum(m, acc_s.amax(-1))  # K:369 running max
        alpha = torch.exp(m_prev - m)  # K:371
        p = torch.exp(acc_s - m.unsqueeze(-1))  # K:373
        sum_exp = sum_exp * alpha + p.sum(-1)  # K:374-376 fp32 P
        p16 = p.to(torch.bfloat16).float()  # K:377
        acc_o = acc_o * alpha.unsqueeze(-1) + torch.einsum("bshk,bskd->bshd", p16, kvb)  # K:379-380
    sum_exp = sum_exp + torch.exp(sink.view(1, 1, h) - m)  # K:383 sink once, no value row
    o = (acc_o / sum_exp.unsqueeze(-1)).to(torch.bfloat16)  # K:385-387
    return o, (torch.log(sum_exp) + m).permute(0, 2, 1)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("seq_len, ratio, h, d", [(300, 0, 4, 64), (300, 1, 4, 64), (1024, 2, 4, 64), (256, 1, 8, 512)])
def test_gathered_oracle_matches_sequential_kernel_loop(device, seq_len, ratio, h, d):
    q, kv, sink, idxs, scale = _case(device, batch=2, seq_len=seq_len, ratio=ratio, h=h, d=d)
    o_k, lse_k = _sparse_attn_kernel_emulation(q, kv, sink, idxs, scale)
    o_r, lse_r = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    # O: the documented bf16-P budget (P rounded against a running vs the final max)
    diff = (o_r.float() - o_k.float()).abs()
    assert (diff <= 2**-7 * o_k.float().abs() + 2**-8 * kv.float().abs().max()).all(), float(diff.max())
    # ... and the disagreement is real (the batch spelling cannot be bit-exact, docstring claim)
    assert not torch.equal(o_r, o_k)
    # LSE: P-independent, so the two agree to fp32 accumulation noise
    torch.testing.assert_close(lse_r, lse_k, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_kernel_loop_corner_rows(device):
    """First 64-slot block all -1 with keys later (alpha = exp(-1e30 - m) = 0), a duplicated
    slot, a keyless row: O agrees within budget / exactly 0; the kernel's implied LSE on
    a keyless row is +inf (log(inf) - 1e30) where the oracle publishes ``sink``."""
    q, kv, sink, idxs, scale = _case(device, batch=1, seq_len=200, ratio=1, window=16, topk=200)
    idxs = idxs.clone()
    idxs[0, 5, :] = -1
    idxs[0, 150, :64] = -1
    idxs[0, 160, 0] = idxs[0, 160, 1]
    o_k, lse_k = _sparse_attn_kernel_emulation(q, kv, sink, idxs, scale)
    o_r, lse_r = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    assert torch.equal(o_k[0, 5], torch.zeros_like(o_k[0, 5])) and torch.equal(o_r[0, 5], torch.zeros_like(o_r[0, 5]))
    assert torch.isposinf(lse_k[0, :, 5]).all() and torch.equal(lse_r[0, :, 5], sink)
    rows = [150, 160]
    diff = (o_r[0, rows].float() - o_k[0, rows].float()).abs()
    assert (diff <= 2**-7 * o_k[0, rows].float().abs() + 2**-8 * kv.float().abs().max()).all()
    torch.testing.assert_close(lse_r[0, :, rows], lse_k[0, :, rows], rtol=1e-5, atol=1e-5)
    # sink = -inf AND no key: the kernel loop divides 0 / 0 (NaN); the oracle selects 0 / -inf
    o_k2, _ = _sparse_attn_kernel_emulation(q, kv, torch.full_like(sink, -torch.inf), idxs, scale)
    o_r2, lse_r2 = R.sparse_attention_reference(q, kv, torch.full_like(sink, -torch.inf), idxs, scale)
    assert torch.isnan(o_k2[0, 5].float()).all()
    assert torch.equal(o_r2[0, 5], torch.zeros_like(o_r2[0, 5])) and torch.isneginf(lse_r2[0, :, 5]).all()


# ---------------------------------------------------------------------------
# PIN: act_quant mirror vs the bit-level K:22-37 twin (no frexp / ldexp)
# ---------------------------------------------------------------------------


def _fast_log2_ceil(x):
    bits = x.contiguous().view(torch.int32)  # K:24 reinterpret
    exp_x = (bits >> 23) & 0xFF  # K:25
    man = bits & ((1 << 23) - 1)  # K:26
    return exp_x - 127 + (man != 0).to(torch.int32)  # K:27


def _fast_pow2(k):
    return ((k + 127) << 23).view(torch.float32)  # K:32-33


def _act_quant_bit_twin(x, block=32):
    xb = x.unflatten(-1, (-1, block)).float()
    amax = torch.maximum(xb.abs().amax(-1, keepdim=True), torch.tensor(1e-4, dtype=torch.float32, device=x.device))  # K:74-76
    s = _fast_pow2(_fast_log2_ceil(amax * torch.tensor(1 / 448, dtype=torch.float32, device=x.device)))  # K:36-37, K:46, K:78
    q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()  # K:85
    return (q * s).to(x.dtype).flatten(-2), s.squeeze(-1)


@pytest.mark.parametrize("device", _DEVICES)
def test_act_quant_mirror_matches_bit_level_fast_log2_ceil_twin(device):
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        x = (torch.randn(64, 512, generator=g, device=device) * torch.logspace(-6, 4, 64, device=device)[:, None]).to(torch.bfloat16)
        x[0].zero_()
        x[0, 5] = 448.0  # amax * fp32(1/448) == 1.0 exactly: ceil(log2) must be 0, not 1
        x[1].zero_()
        x[1, 7] = 448.0 * 2**-7
        x[2].zero_()
        x[2, 9] = 1e-4  # at the floor
        x[3].zero_()
        x[3, 11] = 1.5e-4
        y_r, s_r = R.act_quant_mirror(x, 32, return_scales=True)
        y_t, s_t = _act_quant_bit_twin(x, 32)
        assert torch.equal(s_r, s_t) and torch.equal(y_r, y_t)


# ---------------------------------------------------------------------------
# PIN: e2m1 round-to-nearest-even vs torch.round (half-to-even) on the e2m1 step grid
# ---------------------------------------------------------------------------


def test_e2m1_rne_matches_round_half_even_spelling():
    v = torch.cat([torch.linspace(-6, 6, 4801), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 6.0, -6.0])])
    a = v.abs()
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))  # e2m1 spacing: 0.5 below 2, 1 in [2,4), 2 in [4,6]
    want = torch.copysign(torch.round(a / step) * step, v)  # torch.round is half-to-even
    assert torch.equal(R.e2m1_round_to_nearest_even(v), want)


# ---------------------------------------------------------------------------
# PIN: make_inputs RoPEs the compressed latent at position j * ratio (M:754, M:758)
# ---------------------------------------------------------------------------


def _precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """M:368-389 verbatim, with the int-typed arguments the JSON delivers."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x, freqs_cis, inverse=False):
    """M:392-406 verbatim (in place on the slice)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@pytest.mark.parametrize("seq_len, ratio", [(301, 2), (64, 1), (7, 2)])
def test_compress_kv_is_roped_at_group_start_positions(monkeypatch, seq_len, ratio):
    geom = R.GEOMETRY_TINY
    calls = []
    orig = R.apply_rope_interleaved

    def spy(x, cos, sin, inverse=False):
        out = orig(x, cos, sin, inverse)
        calls.append((x.clone(), out.clone()))
        return out

    monkeypatch.setattr(R, "apply_rope_interleaved", spy)
    inputs = R.make_inputs(geom, 2, seq_len, ratio, seed=7)
    pre, post = calls[-1]  # the compressed latent is the last RoPE make_inputs applies
    assert pre.shape == (2, seq_len // ratio, geom.head_dim)
    fc = _precompute_freqs_cis(geom.rope_dim, seq_len, 65536, 160000, 16, 32, 1)  # compressed-layer table (M:680-696)
    want = pre.clone()
    _apply_rotary_emb(want[..., -geom.rope_dim :], fc[: seq_len - seq_len % ratio : ratio])  # M:754, M:758
    assert torch.equal(want, post)
    if ratio > 1:
        naive = pre.clone()
        _apply_rotary_emb(naive[..., -geom.rope_dim :], fc[: seq_len // ratio])  # position j: the wrong table
        assert not torch.equal(naive, post)
    # the stored compress_kv is the fp4 / e4m3 block-16 image of that rotated latent
    assert torch.equal(inputs["compress_kv"], R.fp4_e4m3_block16_mirror(post, 16))
    # and the block's own q / kv use the full table at positions 0..S-1 (M:767)
    x = torch.randn(2, seq_len, 3, geom.head_dim).to(torch.bfloat16)
    y_model = x.clone()
    _apply_rotary_emb(y_model[..., -geom.rope_dim :], fc)
    assert torch.equal(orig(x, inputs["cos"], inputs["sin"]), y_model)
