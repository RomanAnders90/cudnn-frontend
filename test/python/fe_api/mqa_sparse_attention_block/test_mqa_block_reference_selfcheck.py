# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-checks of the pure-torch oracles in ``mqa_block_reference.py`` -- CPU-runnable, no cudnn.

The two attention spellings (gathered vs dense-masked) must agree; every
elementwise twin is pinned to a hand-derived property of its source function
(``model.py`` ``M:`` / ``kernel.py`` ``K:`` of the V4.1-Flash release). Nothing
here touches a FROST kernel: a failure means the ORACLE is wrong.
"""

import math

import pytest
import torch

# Rootdir-qualified (as fe_api/causal_conv1d_bulk/test_causal_conv1d_bulk_sm100.py does): a bare `from reference
# import` collides with fe_api/gated_attention_block/mqa_block_reference.py in one pytest session (`pytest fe_api/`).
from fe_api.mqa_sparse_attention_block.mqa_block_reference import (
    GEOMETRY_FLASH_41,
    GEOMETRY_TINY,
    RefGeometry,
    act_quant_mirror,
    apply_rope_interleaved,
    block_baseline,
    block_reference,
    build_rope_tables,
    compressed_idxs_synthetic,
    dense_masked_reference,
    e2m1_round_to_nearest_even,
    fp4_e4m3_block16_mirror,
    freqs_cis_complex,
    make_inputs,
    multiplicity_mask,
    sparse_attention_reference,
    window_idxs,
    yarn_band,
)

pytestmark = pytest.mark.L0

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_EXACT = dict(p_dtype=torch.float32, out_dtype=torch.float32)  # exact-math variant: no bf16 P / O rounding

# Flash-4.1 YaRN parameters (inference_config.json) as ``build_rope_tables`` takes them.
_FLASH_YARN = dict(factor=16.0, beta_fast=32.0, beta_slow=1.0, original_seq_len=65536)


def _attention_case(device, *, batch, seq_len, ratio, h=4, d=64, window=128, topk=512, seed=0):
    """Random q / kv_all / sink / topk_idxs for the attention oracles at a small head geometry."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq_len, h, d, generator=g, device=device).to(torch.bfloat16)
    kv = torch.randn(batch, seq_len, d, generator=g, device=device).to(torch.bfloat16)
    idxs = window_idxs(batch, seq_len, window, device=device)
    if ratio > 0 and seq_len // ratio > 0:
        n_c = seq_len // ratio
        kv = torch.cat([kv, torch.randn(batch, n_c, d, generator=g, device=device).to(torch.bfloat16)], dim=1)
        idxs = torch.cat([idxs, compressed_idxs_synthetic(batch, seq_len, ratio, topk, g, device=device)], dim=-1)
    sink = torch.randn(h, generator=g, device=device, dtype=torch.float32)
    return q, kv, sink, idxs, float(d) ** -0.5


def _assert_bf16_budget(o_a, o_b, kv):
    """Two spellings that both round P to bf16 from fp32 inputs differing by accumulation
    order / max choice: a P entry may flip by one bf16 ulp (2**-8 relative), and the
    row-normalized PV sum then moves by at most 2**-8 * max|kv|; plus one bf16 ulp of O."""
    torch.testing.assert_close(o_a.float(), o_b.float(), rtol=2**-7, atol=2**-8 * float(kv.float().abs().max()))


# ---------------------------------------------------------------------------
# (1) gathered oracle == dense-masked oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [96, 128, 300, 1024])
def test_gathered_matches_dense_masked(device, seq_len, ratio):
    q, kv, sink, idxs, scale = _attention_case(device, batch=2, seq_len=seq_len, ratio=ratio)
    # exact-math mode: the only differences are fp32 accumulation order and exp/log rounding
    o_g, lse_g = sparse_attention_reference(q, kv, sink, idxs, scale, **_EXACT)
    o_d, lse_d = dense_masked_reference(q, kv, sink, idxs, scale, **_EXACT)
    assert o_g.shape == q.shape and lse_g.shape == (2, q.shape[2], seq_len)
    torch.testing.assert_close(o_g, o_d, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse_g, lse_d, rtol=1e-5, atol=1e-5)
    # kernel-mirror mode (bf16 P, bf16 O): agree within the bf16-P budget; LSE is P-independent
    o_g16, lse_g16 = sparse_attention_reference(q, kv, sink, idxs, scale)
    o_d16, lse_d16 = dense_masked_reference(q, kv, sink, idxs, scale)
    assert o_g16.dtype == torch.bfloat16 and o_d16.dtype == torch.bfloat16
    _assert_bf16_budget(o_g16, o_d16, kv)
    torch.testing.assert_close(lse_g16, lse_g, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse_d16, lse_d, rtol=1e-5, atol=1e-5)
    # and the mirror mode is the exact-math value rounded, up to the same budget
    _assert_bf16_budget(o_g16, o_g, kv)


# ---------------------------------------------------------------------------
# (2) all -1 rows -> O exactly 0 and LSE == sink   (K:355 floor; FROST LSE convention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_all_minus_one_rows_give_zero_output_and_sink_lse(device):
    q, kv, sink, idxs, scale = _attention_case(device, batch=2, seq_len=64, ratio=1, window=16, topk=8)
    dead = [0, 7, 40]
    idxs = idxs.clone()
    idxs[:, dead, :] = -1
    for fn in (sparse_attention_reference, dense_masked_reference):
        o, lse = fn(q, kv, sink, idxs, scale)
        # O: exactly zero (kernel: exp(sink + 1e30) = +inf in the denominator, 0 / inf = 0)
        assert torch.equal(o[:, dead], torch.zeros_like(o[:, dead])), fn.__name__
        # LSE: exactly the sink logit -- the sink is the only mass; the kernel itself emits no LSE
        for b in range(2):
            for r in dead:
                assert torch.equal(lse[b, :, r], sink), (fn.__name__, b, r)
        live = [i for i in range(64) if i not in dead]
        assert torch.isfinite(o[:, live].float()).all() and torch.isfinite(lse[:, :, live]).all()
    # sink = -inf AND no key: the TileLang kernel would divide 0 / 0; BOTH oracles SELECT 0 and LSE = -inf
    for fn in (sparse_attention_reference, dense_masked_reference):
        o, lse = fn(q, kv, torch.full_like(sink, -torch.inf), idxs, scale)
        assert torch.equal(o[:, dead], torch.zeros_like(o[:, dead])), fn.__name__
        assert torch.isneginf(lse[:, :, dead]).all(), fn.__name__
        assert torch.isfinite(o[:, live].float()).all() and torch.isfinite(lse[:, :, live]).all(), fn.__name__


# ---------------------------------------------------------------------------
# (3) sink = -inf equals the no-sink softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_neg_inf_sink_is_plain_softmax(device):
    q, kv, sink, idxs, scale = _attention_case(device, batch=2, seq_len=64, ratio=2, window=16, topk=8)
    no_sink = torch.full_like(sink, -torch.inf)
    o, lse = sparse_attention_reference(q, kv, no_sink, idxs, scale, **_EXACT)
    o_d, lse_d = dense_masked_reference(q, kv, no_sink, idxs, scale, **_EXACT)
    # explicit no-sink softmax, spelled a third way (torch.softmax / logsumexp over the gathered scores)
    b, s, h, d = q.shape
    idx = idxs.long()
    valid = idx != -1
    kv_g = kv.float()[torch.arange(b, device=device)[:, None, None], idx.clamp_min(0)] * valid.unsqueeze(-1)
    sc = torch.einsum("bshd,bskd->bshk", q.float(), kv_g) * scale
    sc = sc.masked_fill(~valid.unsqueeze(2), -torch.inf)
    p = torch.softmax(sc, dim=-1)
    o_ref = torch.einsum("bshk,bskd->bshd", p, kv_g)
    lse_ref = torch.logsumexp(sc, dim=-1).permute(0, 2, 1)
    for o_x, lse_x in ((o, lse), (o_d, lse_d)):
        torch.testing.assert_close(o_x, o_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(lse_x, lse_ref, rtol=1e-5, atol=1e-5)
    # and a finite sink really changes the answer (the sink is not a no-op)
    o_s, lse_s = sparse_attention_reference(q, kv, torch.zeros_like(sink), idxs, scale, **_EXACT)
    assert (lse_s - lse).abs().max() > 1e-3 and (o_s - o).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# (4) duplicates count twice (M:415 "handles every slot independently")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_duplicate_ids_count_twice(device):
    q, kv, sink, idxs, scale = _attention_case(device, batch=1, seq_len=64, ratio=0, window=16, h=2, d=32)
    dup = idxs.clone()
    dup[:, 16:, 0] = dup[:, 16:, 1]  # rows with a full window: slot 0 repeats slot 1's id
    count = multiplicity_mask(dup, kv.shape[1])
    assert (count[:, 16:].max(-1).values == 2).all() and (count[:, :16].max(-1).values == 1).all()
    o_g, lse_g = sparse_attention_reference(q, kv, sink, dup, scale, **_EXACT)
    o_d, lse_d = dense_masked_reference(q, kv, sink, dup, scale, **_EXACT)  # count == 2 -> + log 2
    torch.testing.assert_close(o_g, o_d, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(lse_g, lse_d, rtol=1e-5, atol=1e-5)
    # a deduplicated list (the repeated slot dropped) is a DIFFERENT answer -- duplicates are not merged
    dedup = dup.clone()
    dedup[:, 16:, 0] = -1
    o_1, lse_1 = sparse_attention_reference(q, kv, sink, dedup, scale, **_EXACT)
    assert (lse_g[:, :, 16:] - lse_1[:, :, 16:]).abs().max() > 1e-3
    assert (o_g[:, 16:] - o_1[:, 16:]).abs().max() > 1e-3
    torch.testing.assert_close(o_g[:, :16], o_1[:, :16])  # untouched rows unchanged


# ---------------------------------------------------------------------------
# (5) act_quant mirror (K:41-124 as called at M:707)
# ---------------------------------------------------------------------------


def _random_e4m3_grid(rows, cols, block, device, g):
    """bf16 values exactly on the e4m3 x 2**k grid, one k per block."""
    codes = (torch.randn(rows, cols, generator=g, device=device) * 100).to(torch.float8_e4m3fn).float()
    k = torch.randint(-10, 10, (rows, cols // block, 1), generator=g, device=device)
    scale = torch.ldexp(torch.ones(rows, cols // block, 1, device=device), k)
    return (codes.unflatten(-1, (cols // block, block)) * scale).flatten(-2).to(torch.bfloat16)


@pytest.mark.parametrize("device", _DEVICES)
def test_act_quant_mirror_properties(device):
    g = torch.Generator(device=device).manual_seed(1)
    x = (torch.randn(8, 512, generator=g, device=device) * torch.logspace(-3, 3, 8, device=device)[:, None]).to(torch.bfloat16)
    y, s = act_quant_mirror(x, 32, return_scales=True)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape and s.shape == (8, 16)
    # idempotent: the output is exactly the dequantized e4m3 x 2**k grid
    assert torch.equal(act_quant_mirror(y, 32), y)
    # scales are powers of two >= 2**-22 and cover amax: amax / 448 <= s < 2 * amax / 448
    m, e = torch.frexp(s)
    assert (m == 0.5).all() and (s >= 2.0**-22).all()
    amax = x.float().unflatten(-1, (16, 32)).abs().amax(-1).clamp_min(1e-4)
    assert (s >= amax / 448).all() and (s < 2 * amax / 448).all()
    # a random block changes by at most half an e4m3 ulp (2**-4 relative for normals, 2**-10 * s absolute for subnormals)
    err = (y.float() - x.float()).abs()
    bound = 2.0**-4 * x.float().abs() + 2.0**-10 * s.repeat_interleave(32, dim=-1)
    assert (err <= bound).all(), float((err - bound).max())
    # all-zero block: floor amax 1e-4 -> scale 2**-22 (K:76 + K:36-37), output 0
    z = torch.zeros(3, 64, dtype=torch.bfloat16, device=device)
    z[1, 40] = 1.0  # one non-zero block next to zero ones
    yz, sz = act_quant_mirror(z, 32, return_scales=True)
    assert torch.equal(sz[0], torch.full((2,), 2.0**-22, device=device)) and sz[1, 0] == 2.0**-22 and sz[2, 0] == 2.0**-22
    assert torch.equal(yz[0], z[0]) and torch.equal(yz[2], z[2]) and yz[1, 40] == 1.0
    # tiny values (below the 1e-4 floor) quantize against 2**-22, not against their own amax
    t = torch.full((1, 32), 3e-5, dtype=torch.bfloat16, device=device)
    yt, st = act_quant_mirror(t, 32, return_scales=True)
    assert st.item() == 2.0**-22 and (yt.float() - 3e-5).abs().max() < 2.0**-4 * 3e-5 + 2.0**-10 * 2.0**-22
    # values already on an e4m3 x 2**k grid pass through unchanged
    grid = _random_e4m3_grid(6, 256, 32, device, g)
    assert torch.equal(act_quant_mirror(grid, 32), grid)
    # boundary: amax exactly 448 * 2**k gives s = 2**k (448 * fp32(1/448) == 1.0 exactly)
    edge = torch.zeros(1, 32, dtype=torch.bfloat16, device=device)
    edge[0, 3] = 448.0
    edge[0, 5] = -224.0
    ye, se = act_quant_mirror(edge, 32, return_scales=True)
    assert se.item() == 1.0 and torch.equal(ye, edge)
    with pytest.raises(ValueError, match="multiple of the block size"):
        act_quant_mirror(torch.zeros(2, 40, dtype=torch.bfloat16, device=device), 32)


# ---------------------------------------------------------------------------
# (6) fp4 / e4m3 block-16 mirror (K:127-203 with scale_dtype=e4m3, as called at M:760)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_fp4_e4m3_mirror_properties(device):
    g = torch.Generator(device=device).manual_seed(2)
    x = (torch.randn(8, 512, generator=g, device=device) * torch.logspace(-3, 2, 8, device=device)[:, None]).to(torch.bfloat16)
    y, s = fp4_e4m3_block16_mirror(x, 16, return_scales=True)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape and s.shape == (8, 32)
    # scales are e4m3 values >= 2**-9 (the floor 6 * 2**-9 / 6, K:161); every value is an e2m1 code times its scale
    assert torch.equal(s.to(torch.float8_e4m3fn).float(), s) and (s >= 2.0**-9).all()
    codes = y.float().unflatten(-1, (32, 16)) / s.unsqueeze(-1)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    assert (codes.abs().unsqueeze(-1) == values).any(-1).all()
    # three regimes of the SOURCE math (the logspace rows exercise all of them):
    amax = x.float().unflatten(-1, (32, 16)).abs().amax(-1)
    floored = amax < 6 * 2.0**-9  # s is the floor scale
    normal = amax >= 6 * 2.0**-6  # s is a normal e4m3: the max element is code 6
    band = ~floored & ~normal  # s is a subnormal e4m3 (k * 2**-9): the max element is code 4 or 6
    assert floored.any() and band.any() and normal.any()
    assert (s[floored] == 2.0**-9).all()
    assert (codes.abs().amax(-1) == 6.0)[normal].all()
    assert (codes.abs().amax(-1) >= 4.0)[band].all() and (codes.abs().amax(-1) <= 6.0).all()
    # idempotent wherever the scale is recoverable: normal-range and floored blocks reproduce bit for bit
    y2, s2 = fp4_e4m3_block16_mirror(y, 16, return_scales=True)
    stable = (floored | normal).repeat_interleave(16, dim=-1)
    assert torch.equal(y2[stable], y[stable]) and torch.equal(s2[floored | normal], s[floored | normal])
    # ... and the subnormal-scale band is where the source's own recipe is NOT idempotent, deterministically:
    # amax 0.018 -> amax/6 = 1.54 * 2**-9 -> s = 2**-8 -> 0.018 / s = 4.6 -> code 4 -> y = 2**-6;
    # second pass: 2**-6 / 6 = 1.33 * 2**-9 -> s = 2**-9 -> 2**-6 / s = 8 -> clamp 6 -> y2 = 6 * 2**-9 != y
    w = torch.zeros(1, 16, dtype=torch.bfloat16, device=device)
    w[0, 0] = 0.018
    yw, sw = fp4_e4m3_block16_mirror(w, 16, return_scales=True)
    yw2, sw2 = fp4_e4m3_block16_mirror(yw, 16, return_scales=True)
    assert sw.item() == 2.0**-8 and yw[0, 0].item() == 2.0**-6
    assert sw2.item() == 2.0**-9 and yw2[0, 0].item() == 6 * 2.0**-9
    # all-zero block: floor scale e4m3(2**-9) = 2**-9, output 0
    z = torch.zeros(2, 32, dtype=torch.bfloat16, device=device)
    yz, sz = fp4_e4m3_block16_mirror(z, 16, return_scales=True)
    assert (sz == 2.0**-9).all() and torch.equal(yz, z)
    # e2m1 RNE ties go to the even code (0, 1, 2, 4), non-ties to the nearest value, sign preserved
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
    assert torch.equal(e2m1_round_to_nearest_even(ties), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], device=device))
    assert torch.equal(e2m1_round_to_nearest_even(-ties), -torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], device=device))
    off = torch.tensor([0.0, 0.2, 0.3, 0.7, 0.8, 1.3, 1.7, 2.4, 2.6, 3.4, 3.6, 4.9, 5.1, 6.0], device=device)
    want = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.5, 1.5, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0], device=device)
    assert torch.equal(e2m1_round_to_nearest_even(off), want)
    assert torch.equal(e2m1_round_to_nearest_even(-off), -want)


# ---------------------------------------------------------------------------
# (7) RoPE tables vs torch.polar of the M:368-389 formula, recomputed here
# ---------------------------------------------------------------------------


def _polar_table_from_formula(seq_len, dim, base, yarn):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if yarn is not None and yarn["original_seq_len"] > 0:

        def corrected_dim(rotations):
            return dim * math.log(yarn["original_seq_len"] / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(yarn["beta_fast"])), 0)
        high = min(math.ceil(corrected_dim(yarn["beta_slow"])), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / yarn["factor"] * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seq_len), freqs)
    return torch.polar(torch.ones_like(angles), angles), freqs


@pytest.mark.parametrize("yarn", [None, _FLASH_YARN], ids=["plain", "yarn"])
def test_build_rope_tables_matches_torch_polar(yarn):
    seq_len, dim = 2048, 64
    base = 160000.0 if yarn else 10000.0
    fc_ref, freqs_ref = _polar_table_from_formula(seq_len, dim, base, yarn)
    fc = freqs_cis_complex(seq_len, dim, base, yarn=yarn)
    assert fc.dtype == torch.complex64 and fc.shape == (seq_len, dim // 2)
    torch.testing.assert_close(fc.real, fc_ref.real, rtol=0, atol=1e-6)
    torch.testing.assert_close(fc.imag, fc_ref.imag, rtol=0, atol=1e-6)
    cos, sin = build_rope_tables(seq_len, dim, base, yarn=yarn)
    assert cos.dtype == torch.float32 and cos.shape == (seq_len, dim // 2)
    torch.testing.assert_close(cos, fc_ref.real, rtol=0, atol=1e-6)
    torch.testing.assert_close(sin, fc_ref.imag, rtol=0, atol=1e-6)
    assert torch.equal(torch.complex(cos, sin), fc)  # the table IS the complex table, bit for bit
    if yarn is None:
        # plain arm: frequency k is exactly base**(-2k/dim), position 0 is the identity rotation
        torch.testing.assert_close(freqs_ref, torch.tensor([base ** (-2 * k / dim) for k in range(dim // 2)], dtype=torch.float32))
        assert torch.equal(cos[0], torch.ones(dim // 2)) and torch.equal(sin[0], torch.zeros(dim // 2))
    else:
        # the YaRN band for the Flash values is low = 15, high = 25: indices 0-15 unscaled, 25-31 divided by 16
        assert yarn_band(dim, base, yarn) == (15, 25)
        cd_fast = dim * math.log(yarn["original_seq_len"] / (yarn["beta_fast"] * 2 * math.pi)) / (2 * math.log(base))
        cd_slow = dim * math.log(yarn["original_seq_len"] / (yarn["beta_slow"] * 2 * math.pi)) / (2 * math.log(base))
        assert (math.floor(cd_fast), math.ceil(cd_slow)) == (15, 25)
        plain = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        torch.testing.assert_close(freqs_ref[:16], plain[:16])
        torch.testing.assert_close(freqs_ref[25:], plain[25:] / 16)
        assert ((freqs_ref[16:25] > plain[16:25] / 16) & (freqs_ref[16:25] < plain[16:25])).all()
    # original_seq_len = 0 disables YaRN (M:377 ``if original_seq_len > 0``)
    off = dict(_FLASH_YARN, original_seq_len=0)
    assert torch.equal(freqs_cis_complex(64, dim, base, yarn=off), freqs_cis_complex(64, dim, base, yarn=None))
    with pytest.raises(ValueError, match="yarn needs keys"):
        build_rope_tables(8, dim, base, yarn={"factor": 16.0})


# ---------------------------------------------------------------------------
# (8) apply_rope_interleaved (M:392-406)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_rope_interleaved_roundtrip_and_passthrough(device):
    g = torch.Generator(device=device).manual_seed(3)
    seq_len, rope_dim = 300, 16
    cos, sin = build_rope_tables(seq_len, rope_dim, 160000.0, yarn=_FLASH_YARN, device=device)
    for shape in ((2, seq_len, 64), (2, seq_len, 3, 64)):
        x = torch.randn(*shape, generator=g, device=device).to(torch.bfloat16)
        y = apply_rope_interleaved(x, cos, sin)
        z = apply_rope_interleaved(y, cos, sin, inverse=True)
        assert y.dtype == x.dtype and y.shape == x.shape
        assert torch.equal(y[..., :-rope_dim], x[..., :-rope_dim]) and torch.equal(z[..., :-rope_dim], x[..., :-rope_dim])
        assert not torch.equal(y[..., -rope_dim:], x[..., -rope_dim:])  # the rotation happened (position 0 excepted)
        assert torch.equal(y[:, 0], x[:, 0])  # position 0 is the identity rotation: bf16 -> fp32 -> bf16 is exact
        # two bf16 roundings on a norm-preserving rotation
        torch.testing.assert_close(z[..., -rope_dim:].float(), x[..., -rope_dim:].float(), rtol=2**-6, atol=2**-6)
        # a [B, S, P] table (the block's input form) gives the same answer as the [S, P] one
        cos_b, sin_b = cos.unsqueeze(0).expand(2, -1, -1).contiguous(), sin.unsqueeze(0).expand(2, -1, -1).contiguous()
        assert torch.equal(apply_rope_interleaved(x, cos_b, sin_b), y)
    # inverse == conjugate: rotating with (cos, -sin) is the inverse
    x = torch.randn(1, seq_len, 2, 32, generator=g, device=device)
    assert torch.equal(apply_rope_interleaved(x, cos, sin, inverse=True), apply_rope_interleaved(x, cos, -sin))
    with pytest.raises(ValueError, match="sequence extent"):
        apply_rope_interleaved(x[:, :10], cos, sin)
    with pytest.raises(ValueError, match="exceeds x last dim"):
        apply_rope_interleaved(torch.zeros(1, seq_len, 8, device=device), cos, sin)


@pytest.mark.parametrize("device", _DEVICES)
def test_apply_rope_interleaved_pairs_are_adjacent(device):
    """Hand-computed rotation of two adjacent pairs on the LAST four dims, in fp32 (no rounding)."""
    seq_len = 5
    g = torch.Generator(device=device).manual_seed(4)
    x = torch.randn(1, seq_len, 3, 8, generator=g, device=device)  # [B, S, H, D]; rope_dim 4 = 2 pairs
    ang = torch.randn(seq_len, 2, generator=g, device=device)
    cos, sin = torch.cos(ang), torch.sin(ang)
    y = apply_rope_interleaved(x, cos, sin)
    assert torch.equal(y[..., :4], x[..., :4])
    c, s = cos[None, :, None, :], sin[None, :, None, :]  # [1, S, 1, 2]
    a0, b0, a1, b1 = x[..., 4], x[..., 5], x[..., 6], x[..., 7]
    # pair 0 = (x[-4], x[-3]) with frequency 0, pair 1 = (x[-2], x[-1]) with frequency 1
    torch.testing.assert_close(y[..., 4], a0 * c[..., 0] - b0 * s[..., 0])
    torch.testing.assert_close(y[..., 5], a0 * s[..., 0] + b0 * c[..., 0])
    torch.testing.assert_close(y[..., 6], a1 * c[..., 1] - b1 * s[..., 1])
    torch.testing.assert_close(y[..., 7], a1 * s[..., 1] + b1 * c[..., 1])
    y_inv = apply_rope_interleaved(x, cos, sin, inverse=True)
    torch.testing.assert_close(y_inv[..., 4], a0 * c[..., 0] + b0 * s[..., 0])
    torch.testing.assert_close(y_inv[..., 5], -a0 * s[..., 0] + b0 * c[..., 0])
    # [B, S, D] form takes the same table
    y3 = apply_rope_interleaved(x[:, :, 0, :], cos, sin)
    torch.testing.assert_close(y3, y[:, :, 0, :])


# ---------------------------------------------------------------------------
# (9) window list (M:417-420)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [5, 128, 300])
def test_window_idxs_matches_hand_loop(seq_len):
    window = 128
    idxs = window_idxs(2, seq_len, window)
    width = min(seq_len, window)
    assert idxs.dtype == torch.int32 and idxs.shape == (2, seq_len, width)
    want = torch.empty(seq_len, width, dtype=torch.int32)
    for i in range(seq_len):
        for k in range(width):
            pos = max(0, i - window + 1) + k
            want[i, k] = pos if pos <= i else -1  # row i attends max(0, i-127) .. i inclusive
    assert torch.equal(idxs[0], want) and torch.equal(idxs[1], want)
    assert (idxs == torch.arange(seq_len)[None, :, None]).any(-1).all()  # self is always in the list
    assert idxs.max() == seq_len - 1 and ((idxs != -1).sum(-1)[0] == torch.arange(1, seq_len + 1).clamp_max(window)).all()


# ---------------------------------------------------------------------------
# (10) synthetic compressed list (M:564-580)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len, ratio, topk", [(300, 1, 512), (300, 2, 512), (1024, 2, 512), (1000, 1, 8), (5, 2, 512), (1, 2, 512)])
def test_compressed_idxs_synthetic_contract(seq_len, ratio, topk):
    b = 2
    n_c = seq_len // ratio
    width = min(topk, n_c)
    g = torch.Generator().manual_seed(5)
    idxs = compressed_idxs_synthetic(b, seq_len, ratio, topk, g)
    assert idxs.dtype == torch.int32 and idxs.shape == (b, seq_len, width)
    if width == 0:
        return
    valid = idxs != -1
    # -1 is a TAIL: never a valid id after a -1
    assert (valid[..., 1:].int() <= valid[..., :-1].int()).all()
    # offset by S: every real id names a row of the compressed part [S, S + N_c)
    assert (idxs[valid] >= seq_len).all() and (idxs[valid] < seq_len + n_c).all()
    # sorted strictly ascending among the real ids
    diff = idxs[..., 1:] - idxs[..., :-1]
    assert (diff[valid[..., 1:]] > 0).all()
    # reachable: group j is visible to query i iff j * ratio + ratio - 1 <= i   (M:564-565)
    i = torch.arange(seq_len)[None, :, None]
    j = (idxs - seq_len).long()
    assert ((j * ratio + ratio - 1 <= i) | ~valid).all()
    # width of the real prefix is min(topk, (i + 1) // ratio)   (M:578-580)
    want_n = ((torch.arange(seq_len) + 1) // ratio).clamp_max(topk)
    assert torch.equal(valid.sum(-1)[0], want_n) and torch.equal(valid.sum(-1)[1], want_n)
    # relative=True is the same draw without the offset
    g2 = torch.Generator().manual_seed(5)
    rel = compressed_idxs_synthetic(b, seq_len, ratio, topk, g2, relative=True)
    assert torch.equal(torch.where(valid, rel + seq_len, -1), idxs)
    # a second batch entry is an independent draw (when there is anything to choose)
    if width < n_c:
        assert not torch.equal(idxs[0], idxs[1])
    with pytest.raises(ValueError, match="ratio must be >= 1"):
        compressed_idxs_synthetic(b, seq_len, 0, topk, g)


# ---------------------------------------------------------------------------
# (11) the block oracle end to end
# ---------------------------------------------------------------------------


def _check_block_outputs(geom, out, b, s, n_c):
    h, d = geom.n_heads, geom.head_dim
    assert out.qr.shape == (b, s, geom.q_lora_rank) and out.qr.dtype == torch.bfloat16
    assert out.q.shape == (b, s, h, d) and out.q.dtype == torch.bfloat16
    assert out.window_kv.shape == (b, s, d) and out.window_kv.dtype == torch.bfloat16
    assert out.kv_all.shape == (b, s + n_c, d)
    assert out.o.shape == (b, s, h, d) and out.o.dtype == torch.bfloat16
    assert out.o_unrot.shape == (b, s, h, d)
    assert out.o_lora.shape == (b, s, geom.o_groups, geom.o_lora_rank) and out.o_lora.dtype == torch.bfloat16
    assert out.out.shape == (b, s, geom.d_model) and out.out.dtype == torch.bfloat16
    assert out.lse.shape == (b, h, s) and out.lse.dtype == torch.float32
    for name in ("qr", "q", "window_kv", "o", "o_unrot", "o_lora", "out", "lse"):
        assert torch.isfinite(getattr(out, name).float()).all(), name
    # the window rows are on the e4m3 x 2**k grid (op 6) -- the fake-quant is idempotent on them
    assert torch.equal(act_quant_mirror(out.window_kv, 32), out.window_kv)
    assert torch.equal(out.kv_all[:, :s], out.window_kv)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("ratio", [0, 2])
def test_block_reference_tiny_geometry(device, ratio):
    geom = GEOMETRY_TINY
    b, s = 2, 300
    inputs = make_inputs(geom, b, s, ratio, seed=7, device=device)
    n_c = s // ratio if ratio else 0
    assert inputs["cos"].shape == (b, s, geom.rope_dim // 2) and inputs["cos"].dtype == torch.float32
    assert inputs["attn_sink"].shape == (geom.n_heads,) and inputs["attn_sink"].dtype == torch.float32
    if ratio:
        assert inputs["compress_kv"].shape == (b, n_c, geom.head_dim) and inputs["compress_kv"].dtype == torch.bfloat16
        assert inputs["topk_idxs"].shape == (b, s, min(s, geom.window) + min(geom.index_topk, n_c))
        # the compressed rows are on the fp4 x e4m3 grid (plan 1.5): the fp4 mirror is idempotent on them
        assert torch.equal(fp4_e4m3_block16_mirror(inputs["compress_kv"], 16), inputs["compress_kv"])
    else:
        assert inputs["compress_kv"] is None and inputs["topk_idxs"].shape == (b, s, min(s, geom.window))
    out = block_reference(geom, inputs)
    _check_block_outputs(geom, out, b, s, n_c)
    # the op-9 output agrees with the dense-masked spelling on the same q / kv_all / list
    o_d, lse_d = dense_masked_reference(out.q, out.kv_all, inputs["attn_sink"], inputs["topk_idxs"], geom.scale)
    _assert_bf16_budget(out.o, o_d, out.kv_all)
    torch.testing.assert_close(out.lse, lse_d, rtol=1e-5, atol=1e-5)
    # deterministic
    out2 = block_reference(geom, inputs)
    assert torch.equal(out2.out, out.out) and torch.equal(out2.lse, out.lse)
    # the framework-shaped baseline lands near the oracle (attribution only; bf16 GEMMs differ in accumulation)
    base = block_baseline(geom, inputs)
    cos_sim = torch.nn.functional.cosine_similarity(base.out.float().flatten(), out.out.float().flatten(), dim=0)
    assert cos_sim > 0.99, float(cos_sim)


def test_block_reference_head_dim_512_cpu():
    """(n_heads=8, head_dim=512) with the Flash d_model / ranks at S=256, ratio 1, on CPU."""
    geom = RefGeometry(n_heads=8, head_dim=512)
    assert geom.group_width == 512 and geom.n_o_lora == 8192
    b, s = 1, 256
    inputs = make_inputs(geom, b, s, 1, seed=11, device="cpu")
    assert inputs["topk_idxs"].shape == (b, s, 128 + 256)
    out = block_reference(geom, inputs)
    _check_block_outputs(geom, out, b, s, s)
    # the op-9 output agrees with the dense-masked spelling on the same q / kv_all / list (the only
    # head_dim=512 run in the suite; `lse > -80` would be tautological since lse >= sink by construction)
    o_d, lse_d = dense_masked_reference(out.q, out.kv_all, inputs["attn_sink"], inputs["topk_idxs"], geom.scale)
    _assert_bf16_budget(out.o, o_d, out.kv_all)
    torch.testing.assert_close(out.lse, lse_d, rtol=1e-5, atol=1e-5)


def test_ref_geometry_flash_numbers_and_rejects():
    g = GEOMETRY_FLASH_41
    assert (g.group_width, g.n_o_lora, g.n_kv_slots_max) == (4096, 8192, 640)
    assert g.scale == 512**-0.5
    assert RefGeometry(has_compressed_kv=False).n_kv_slots_max == 128
    assert yarn_band(g.rope_dim, g.compress_rope_theta, g.yarn) == (15, 25)
    with pytest.raises(ValueError, match="multiple of 32"):
        RefGeometry(head_dim=100)
    with pytest.raises(ValueError, match="rope_dim must be even"):
        RefGeometry(rope_dim=63)
    with pytest.raises(ValueError, match="rope_dim must be even"):
        RefGeometry(head_dim=64, rope_dim=128)
    with pytest.raises(ValueError, match="divisible by o_groups"):
        RefGeometry(n_heads=5, o_groups=3)
    with pytest.raises(ValueError, match="norm_eps"):
        RefGeometry(norm_eps=0.0)
    with pytest.raises(ValueError, match="kv_fake_quant"):
        RefGeometry(kv_fake_quant="fp8")
    with pytest.raises(ValueError, match="has_compressed_kv"):
        make_inputs(RefGeometry(has_compressed_kv=False), 1, 8, 2)
