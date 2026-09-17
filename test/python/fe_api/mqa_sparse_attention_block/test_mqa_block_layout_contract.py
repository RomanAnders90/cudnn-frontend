# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layout / geometry contract for the MQA sparse-attention block. CPU-only, no GPU,
no FROST opt-in needed (nothing here builds a plan).

These pin the things that are silently wrong rather than loudly failing: a
group width that stops dividing, a RoPE table in the wrong pairing, a YaRN band
that quietly moved, a weight "view" that copies. Every number is derived in the
test from the model formula, never read back from the code under test.
"""

import math
import subprocess
import sys

import pytest
import torch

from cudnn.mqa_sparse_attention_block import (
    KV_FAKE_QUANT_BLOCK,
    KV_FAKE_QUANT_MODES,
    MqaSparseAttentionBlockGeometry,
    YarnParams,
    build_grouped_wo_a,
    build_rope_tables,
)

pytestmark = pytest.mark.L0


# The dataclass defaults ARE the released shape: DeepSeek-V4.1-Flash at TP=1
# (frost_dev/dsv4_ref/v41_flash/inference_config.json); layers 0-1 are the window-only variant.
RELEASED = MqaSparseAttentionBlockGeometry()
RELEASED_WINDOW_ONLY = MqaSparseAttentionBlockGeometry(has_compressed_kv=False)
# A small geometry every rule still holds on, for the reject tests.
SMALL = dict(d_model=256, n_heads=8, head_dim=64, rope_dim=16, q_lora_rank=64, o_lora_rank=32, o_groups=4, window=16, index_topk=8)


# ---------------------------------------------------------------------------
# The released shape (the defaults), pinned field by field
# ---------------------------------------------------------------------------


def test_defaults_are_the_released_shape():
    g = RELEASED
    assert (g.d_model, g.n_heads, g.head_dim, g.rope_dim) == (5120, 64, 512, 64)
    assert (g.q_lora_rank, g.o_lora_rank, g.o_groups) == (1280, 1024, 8)
    assert (g.window, g.index_topk, g.has_compressed_kv) == (128, 512, True)
    assert g.norm_eps == 1e-20
    assert g.attn_scale is None
    assert g.kv_fake_quant == "fp8_block32_ue8m0"
    g.validate()
    RELEASED_WINDOW_ONLY.validate()
    assert RELEASED_WINDOW_ONLY == MqaSparseAttentionBlockGeometry(**{**RELEASED.__dict__, "has_compressed_kv": False})


def test_released_shape_derived_numbers():
    g = RELEASED
    assert g.scale == pytest.approx(512**-0.5)
    assert g.scale == pytest.approx(1.0 / math.sqrt(512))
    assert g.nope_dim == 448
    assert g.heads_per_group == 8
    assert g.group_width == 4096  # 64 * 512 // 8, K of each grouped W_o_a GEMM
    assert g.n_o_lora == 8192  # 8 * 1024, K of W_o_b
    assert g.n_kv_slots_max == 640  # 128 window + 512 compressed
    assert RELEASED_WINDOW_ONLY.n_kv_slots_max == 128
    assert RELEASED_WINDOW_ONLY.group_width == 4096 and RELEASED_WINDOW_ONLY.n_o_lora == 8192


def test_attn_scale_overrides_the_derived_scale():
    g = MqaSparseAttentionBlockGeometry(attn_scale=0.25)
    g.validate()
    assert g.scale == 0.25


def test_module_constants():
    assert KV_FAKE_QUANT_BLOCK == 32
    assert KV_FAKE_QUANT_MODES == ("fp8_block32_ue8m0", "none")
    assert RELEASED.head_dim % KV_FAKE_QUANT_BLOCK == 0


# ---------------------------------------------------------------------------
# validate() -- one reject per rule
# ---------------------------------------------------------------------------


def test_validate_rejects_head_dim_not_multiple_of_32_for_fake_quant_blocks():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "head_dim": 80})  # 80 % 32 != 0, 80 % 16 == 0
    with pytest.raises(ValueError, match=r"kv_fake_quant='fp8_block32_ue8m0' needs head_dim % 32 == 0"):
        g.validate()


def test_validate_rejects_head_dim_not_multiple_of_32_without_fake_quant():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "head_dim": 80, "kv_fake_quant": "none"})
    with pytest.raises(ValueError, match=r"head_dim must be a multiple of 32"):
        g.validate()


def test_validate_rejects_odd_rope_dim():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "rope_dim": 15})
    with pytest.raises(ValueError, match=r"rope_dim must be even"):
        g.validate()


def test_validate_rejects_rope_dim_zero():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "rope_dim": 0})
    with pytest.raises(ValueError, match=r"rope_dim must be in \(0, head_dim=64\]"):
        g.validate()


def test_validate_rejects_rope_dim_above_head_dim():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "rope_dim": 66})
    with pytest.raises(ValueError, match=r"rope_dim must be in \(0, head_dim=64\]"):
        g.validate()


def test_validate_rejects_heads_times_dim_not_divisible_by_groups():
    # 8 heads * 64 = 512; o_groups=3 does not divide it.
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "o_groups": 3})
    with pytest.raises(ValueError, match=r"n_heads \* head_dim \(512\) must be divisible by o_groups \(3\)"):
        g.validate()


def test_validate_rejects_groups_that_straddle_a_head():
    # 6 heads * 64 = 384 IS divisible by 4, so only the whole-heads rule fires: group_width 96 = 1.5 heads,
    # and the documented per-head map (heads_per_group, build_grouped_wo_a) would be false.
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "n_heads": 6})
    assert (g.n_heads * g.head_dim) % g.o_groups == 0 and g.heads_per_group == 1 and g.group_width == 96
    with pytest.raises(ValueError, match=r"n_heads \(6\) must be divisible by o_groups \(4\)"):
        g.validate()


@pytest.mark.parametrize("field", ["d_model", "n_heads", "head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "window", "index_topk"])
@pytest.mark.parametrize("bad", [0, -1])
def test_validate_rejects_non_positive_ints(field, bad):
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, field: bad})
    with pytest.raises(ValueError, match=rf"{field} must be a positive int"):
        g.validate()


def test_validate_rejects_non_positive_norm_eps():
    for eps in (0.0, -1e-20):
        g = MqaSparseAttentionBlockGeometry(**{**SMALL, "norm_eps": eps})
        with pytest.raises(ValueError, match=r"norm_eps must be > 0"):
            g.validate()


def test_validate_rejects_non_positive_attn_scale():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "attn_scale": 0.0})
    with pytest.raises(ValueError, match=r"attn_scale must be > 0"):
        g.validate()


def test_validate_rejects_unknown_kv_fake_quant():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "kv_fake_quant": "fp8_block64"})
    with pytest.raises(ValueError, match=r"kv_fake_quant must be one of \('fp8_block32_ue8m0', 'none'\)"):
        g.validate()


def test_validate_accepts_the_small_geometry_in_both_fake_quant_modes():
    MqaSparseAttentionBlockGeometry(**SMALL).validate()
    MqaSparseAttentionBlockGeometry(**{**SMALL, "kv_fake_quant": "none"}).validate()
    MqaSparseAttentionBlockGeometry(**{**SMALL, "has_compressed_kv": False}).validate()
    # the inclusive boundary: every head dim rotates
    MqaSparseAttentionBlockGeometry(**{**SMALL, "rope_dim": 64}).validate()


def test_index_topk_is_dead_and_may_be_zero_on_a_window_only_layer():
    g = MqaSparseAttentionBlockGeometry(**{**SMALL, "has_compressed_kv": False, "index_topk": 0})
    g.validate()
    assert g.n_kv_slots_max == SMALL["window"]
    with pytest.raises(ValueError, match=r"index_topk must be a non-negative int \(unused when has_compressed_kv=False\)"):
        MqaSparseAttentionBlockGeometry(**{**SMALL, "has_compressed_kv": False, "index_topk": -1}).validate()
    # ... and stays a positive int when the layer has compressed KV (the parametrized reject above covers 0 / -1)
    with pytest.raises(ValueError, match=r"index_topk must be a positive int"):
        MqaSparseAttentionBlockGeometry(**{**SMALL, "index_topk": 0}).validate()


# ---------------------------------------------------------------------------
# build_grouped_wo_a -- a view of the checkpoint, never a copy
# ---------------------------------------------------------------------------


def _weight_like(geom, layout: str) -> torch.Tensor:
    """A deterministic, NaN-free [n_o_lora, group_width] bf16 weight in one of three memory layouts
    (torch.empty leaves bytes that can decode as NaN, and torch.equal is false on NaN even for identical storage)."""
    rows, cols = geom.n_o_lora, geom.group_width
    g = torch.Generator().manual_seed(0)
    if layout == "contiguous":
        return torch.randn(rows, cols, generator=g, dtype=torch.bfloat16)  # bf16 directly: half the fill of an fp32 randn
    if layout == "row_padded":  # a column slice of a wider matrix: row stride cols + 8, unit column stride
        return torch.randn(rows, cols + 8, generator=g, dtype=torch.bfloat16)[:, :cols]
    if layout == "transposed":  # unit ROW stride: the transpose of a [cols, rows] matrix
        return torch.randn(cols + 8, rows, generator=g, dtype=torch.bfloat16)[:cols].t()
    raise ValueError(layout)


@pytest.mark.parametrize("layout", ["contiguous", "row_padded", "transposed"])
@pytest.mark.parametrize("geom", [MqaSparseAttentionBlockGeometry(**SMALL)], ids=["small"])
def test_grouped_wo_a_is_a_view_with_the_model_layout(geom, layout):
    """The docstring promises a VIEW 'for ANY row stride' -- pin it for a contiguous, a row-padded and a
    transposed checkpoint tensor (only the leading dim is split, so every one is view-compatible)."""
    w = _weight_like(geom, layout)
    assert w.shape == (geom.n_o_lora, geom.group_width)
    assert (layout == "contiguous") == w.is_contiguous()
    grouped = build_grouped_wo_a(w, geom)
    assert grouped.shape == (geom.o_groups, geom.o_lora_rank, geom.group_width)
    assert grouped.dtype == w.dtype
    assert grouped.data_ptr() == w.data_ptr()
    assert grouped.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
    assert grouped.stride()[1:] == w.stride() and grouped.stride()[0] == geom.o_lora_rank * w.stride()[0]
    # Same bytes as the model's `self.wo_a.weight.view(n_groups, o_lora_rank, -1)` (M:786).
    assert torch.equal(grouped, w.view(geom.o_groups, geom.o_lora_rank, -1))
    assert torch.equal(grouped.reshape(geom.n_o_lora, geom.group_width), w)
    # Writing through the view lands in the checkpoint tensor.
    grouped[1, 2, 3] = 7.0
    assert w[geom.o_lora_rank + 2, 3].item() == 7.0


def test_grouped_wo_a_is_a_view_at_the_released_shape():
    geom = RELEASED
    w = _weight_like(geom, "contiguous")  # 8192 x 4096 bf16 = 64 MiB
    grouped = build_grouped_wo_a(w, geom)
    assert grouped.shape == (8, 1024, 4096) and grouped.data_ptr() == w.data_ptr()
    assert torch.equal(grouped[3], w[3 * 1024 : 4 * 1024])


def test_grouped_wo_a_row_g_is_the_checkpoint_rows_of_group_g():
    geom = MqaSparseAttentionBlockGeometry(**SMALL)
    w = torch.arange(geom.n_o_lora * geom.group_width, dtype=torch.float32).view(geom.n_o_lora, geom.group_width)
    grouped = build_grouped_wo_a(w, geom)
    for g in range(geom.o_groups):
        assert torch.equal(grouped[g], w[g * geom.o_lora_rank : (g + 1) * geom.o_lora_rank])


def test_grouped_wo_a_rejects_wrong_shapes():
    geom = MqaSparseAttentionBlockGeometry(**SMALL)
    with pytest.raises(ValueError, match=r"w_o_a must have 128 rows \(o_groups \* o_lora_rank = 4 \* 32\), got 127"):
        build_grouped_wo_a(torch.empty(geom.n_o_lora - 1, geom.group_width), geom)
    with pytest.raises(ValueError, match=r"w_o_a must have 128 columns \(group_width = n_heads \* head_dim // o_groups = 8 \* 64 // 4\), got 129"):
        build_grouped_wo_a(torch.empty(geom.n_o_lora, geom.group_width + 1), geom)
    with pytest.raises(ValueError, match=r"w_o_a must be 2-D"):
        build_grouped_wo_a(torch.empty(geom.o_groups, geom.o_lora_rank, geom.group_width), geom)
    # An invalid geometry is refused before any shape check.
    with pytest.raises(ValueError, match=r"rope_dim must be even"):
        build_grouped_wo_a(torch.empty(geom.n_o_lora, geom.group_width), MqaSparseAttentionBlockGeometry(**{**SMALL, "rope_dim": 15}))


# ---------------------------------------------------------------------------
# build_rope_tables -- interleaved-pair fp32 tables, twin of M:368-389
# ---------------------------------------------------------------------------


def _base_freqs(rope_dim: int, theta: float) -> torch.Tensor:
    """M:376 spelled independently: one frequency per adjacent pair."""
    return torch.tensor([theta ** (-(2.0 * k) / rope_dim) for k in range(rope_dim // 2)], dtype=torch.float64)


def test_rope_tables_shapes_dtypes_and_identity_at_position_zero():
    cos, sin = build_rope_tables(64, 64, 10000.0)
    assert cos.shape == sin.shape == (64, 32)
    assert cos.dtype == sin.dtype == torch.float32
    assert cos.is_contiguous() and sin.is_contiguous()
    assert torch.equal(cos[0], torch.ones(32)) and torch.equal(sin[0], torch.zeros(32))
    torch.testing.assert_close(cos * cos + sin * sin, torch.ones_like(cos), rtol=0, atol=2e-6)


def test_rope_tables_without_yarn_match_the_formula():
    seq_len, rope_dim, theta = 300, 64, 10000.0
    cos, sin = build_rope_tables(seq_len, rope_dim, theta)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float64), _base_freqs(rope_dim, theta))
    torch.testing.assert_close(cos.double(), torch.cos(angles), rtol=0, atol=1e-5)
    torch.testing.assert_close(sin.double(), torch.sin(angles), rtol=0, atol=1e-5)
    # Column k rotates pair (2k, 2k+1): the highest-frequency pair is column 0 (angle t at t=1).
    assert sin[1, 0].item() == pytest.approx(math.sin(1.0), rel=1e-6)


def test_rope_tables_yarn_band_for_the_released_compressed_layers():
    """theta 160000, YaRN (16, 32, 1, 65536) -> low 15 / high 25 from M:379-383.

    Frequency index 15 is unscaled, 25..31 are divided by 16, 16..24 blend
    linearly in between -- computed here from the formula, not read back.
    """
    rope_dim, theta = 64, 160000.0
    factor, beta_fast, beta_slow, original = 16.0, 32.0, 1.0, 65536

    def corrected_dim(rotations):
        return rope_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(corrected_dim(beta_fast)), 0)
    high = min(math.ceil(corrected_dim(beta_slow)), rope_dim - 1)
    assert (low, high) == (15, 25)

    base = _base_freqs(rope_dim, theta)
    ramp = torch.clamp((torch.arange(rope_dim // 2, dtype=torch.float64) - low) / (high - low), 0.0, 1.0)
    expected = base * (1.0 - ramp) + (base / factor) * ramp
    # The band, spelled out.
    assert torch.equal(expected[: low + 1], base[: low + 1])  # 0..15 unscaled
    assert torch.equal(expected[high:], base[high:] / factor)  # 25..31 / 16
    inner = (expected[low + 1 : high] / base[low + 1 : high]).tolist()
    assert all(1.0 / factor < r < 1.0 for r in inner) and inner == sorted(inner, reverse=True)  # 16..24 blended, monotone

    cos, sin = build_rope_tables(4, rope_dim, theta, yarn=YarnParams(factor, beta_fast, beta_slow, original))
    recovered = torch.atan2(sin[1].double(), cos[1].double())  # angle at t=1 == frequency (all < pi)
    torch.testing.assert_close(recovered, expected, rtol=2e-6, atol=0)
    ratio = recovered / base
    torch.testing.assert_close(ratio[: low + 1], torch.ones(low + 1, dtype=torch.float64), rtol=2e-6, atol=0)
    torch.testing.assert_close(ratio[high:], torch.full((rope_dim // 2 - high,), 1.0 / factor, dtype=torch.float64), rtol=2e-6, atol=0)

    # The tuple spelling is the same table.
    cos_t, sin_t = build_rope_tables(4, rope_dim, theta, yarn=(factor, beta_fast, beta_slow, original))
    assert torch.equal(cos_t, cos) and torch.equal(sin_t, sin)
    # ... and YaRN actually changed something relative to the plain table.
    cos_plain, _ = build_rope_tables(4, rope_dim, theta)
    assert not torch.equal(cos_plain, cos)


def _precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """model.py M:369-389 VERBATIM (minus the lru_cache), with the int-typed arguments the JSON delivers."""
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


@pytest.mark.parametrize(
    "theta, yarn, model_args",
    [
        (10000.0, None, dict(original_seq_len=0, base=10000, factor=16, beta_fast=32, beta_slow=1)),  # window-only layers (M:690-698)
        (160000.0, (16.0, 32.0, 1.0, 65536), dict(original_seq_len=65536, base=160000, factor=16, beta_fast=32, beta_slow=1)),  # compressed layers
    ],
    ids=["plain_theta_10000", "yarn_theta_160000"],
)
def test_rope_tables_are_a_bit_exact_twin_of_the_model_function(theta, yarn, model_args):
    """``build_rope_tables`` sells itself as a twin of ``precompute_freqs_cis`` (M:368-389): pin that with
    ``torch.equal`` against a verbatim copy, not an atol -- a float64 'improvement' or a ``torch.cos/sin``
    spelling would pass the 1e-5 / 2e-6 tests above and silently break the twin property. Both tables are
    built on the CPU: ``polar()`` differs between CPU and CUDA by ~1 ulp, so the claim is per device."""
    seq_len, rope_dim = 4096, 64
    ref = _precompute_freqs_cis(rope_dim, seq_len, **model_args)
    cos, sin = build_rope_tables(seq_len, rope_dim, theta, yarn=yarn)
    assert ref.dtype == torch.complex64 and ref.shape == (seq_len, rope_dim // 2)
    assert torch.equal(torch.view_as_real(ref)[..., 0], cos)
    assert torch.equal(torch.view_as_real(ref)[..., 1], sin)
    if yarn is not None:  # the YarnParams spelling is the same table
        cos_p, sin_p = build_rope_tables(seq_len, rope_dim, theta, yarn=YarnParams(*yarn))
        assert torch.equal(cos_p, cos) and torch.equal(sin_p, sin)


def test_rope_tables_inverse_is_the_conjugate():
    """M:399 `freqs_cis.conj()`: the inverse rotation is the same table with -sin."""
    cos, sin = build_rope_tables(8, 16, 10000.0)
    x = torch.randn(8, 16, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
    xc = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    table = torch.complex(cos, sin)
    rotated = torch.view_as_real(xc * table).flatten(-2)
    back = torch.view_as_real(torch.view_as_complex(rotated.unflatten(-1, (-1, 2))) * torch.complex(cos, -sin)).flatten(-2)
    torch.testing.assert_close(back, x, rtol=0, atol=1e-5)
    # Interleaved pairing: pair k of the flat vector is (2k, 2k+1).
    x2k, x2k1 = x[:, 0::2], x[:, 1::2]
    torch.testing.assert_close(rotated[:, 0::2], x2k * cos - x2k1 * sin, rtol=0, atol=1e-6)
    torch.testing.assert_close(rotated[:, 1::2], x2k * sin + x2k1 * cos, rtol=0, atol=1e-6)


def test_rope_tables_reject_bad_arguments():
    with pytest.raises(ValueError, match=r"seq_len must be a positive int"):
        build_rope_tables(0, 64, 10000.0)
    with pytest.raises(ValueError, match=r"rope_dim must be a positive even int"):
        build_rope_tables(4, 63, 10000.0)
    with pytest.raises(ValueError, match=r"rope_dim must be a positive even int"):
        build_rope_tables(4, 0, 10000.0)
    with pytest.raises(ValueError, match=r"theta must be > 0"):
        build_rope_tables(4, 64, 0.0)
    with pytest.raises(ValueError, match=r"original_seq_len must be > 0"):
        build_rope_tables(4, 64, 160000.0, yarn=YarnParams(16.0, 32.0, 1.0, 0))
    with pytest.raises(ValueError, match=r"YarnParams.factor must be > 0"):
        build_rope_tables(4, 64, 160000.0, yarn=(0.0, 32.0, 1.0, 65536))
    with pytest.raises(ValueError, match=r"beta_fast / beta_slow must be > 0"):
        build_rope_tables(4, 64, 160000.0, yarn=(16.0, 0.0, 1.0, 65536))
    with pytest.raises(ValueError, match=r"beta_fast / beta_slow must be > 0"):
        build_rope_tables(4, 64, 160000.0, yarn=YarnParams(16.0, 32.0, -1.0, 65536))
    with pytest.raises(ValueError, match=r"yarn must be YarnParams or"):
        build_rope_tables(4, 64, 160000.0, yarn=(16.0, 32.0))


# ---------------------------------------------------------------------------
# import-time contract: no torch at `import cudnn.mqa_sparse_attention_block`
# ---------------------------------------------------------------------------


def test_package_import_does_not_import_torch():
    """``import cudnn.mqa_sparse_attention_block`` stays torch-free; the forward API
    (``api.py`` imports torch and cuda.bindings) is resolved LAZILY on first
    attribute access, and only then does torch load."""
    code = (
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch already loaded before the import'\n"
        "import cudnn.mqa_sparse_attention_block as m\n"
        "print('torch' in sys.modules, m.MqaSparseAttentionBlockGeometry().group_width)\n"
        "assert 'MqaSparseAttentionBlockFwd' in m.__all__ and 'MqaSparseAttentionBlockFwd' in dir(m)\n"
        "assert 'cudnn.mqa_sparse_attention_block.api' not in sys.modules, 'api.py was imported eagerly'\n"
        "F = m.MqaSparseAttentionBlockFwd\n"
        "print('torch' in sys.modules, F.__name__, m.ATTENTION_IMPLS, m.POINTWISE_IMPLS)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "False 4096", out.stdout
    assert lines[1] == "True MqaSparseAttentionBlockFwd ('torch', 'dsa', 'd512') ('torch', 'frost')", out.stdout
