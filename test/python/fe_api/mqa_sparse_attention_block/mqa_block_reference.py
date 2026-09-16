# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch oracles for the MQA sparse-attention block (DeepSeek-V4.1-Flash prefill attention).

This module imports NOTHING from ``cudnn`` on purpose: a reference that imports the
thing it validates can agree with a bug. Every function below is a twin of a
function in the reference implementation (``model.py`` / ``kernel.py`` of the
V4.1-Flash release, cited as ``M:`` / ``K:`` line ranges), spelled in batch torch
with the model's bf16 rounding points kept where the model has them.

Two attention oracles, for two different jobs:

``sparse_attention_reference``
    The GATHERED oracle: the batch form of ``K:311-389`` ``sparse_attn_kernel``.
    ``-1`` slots are a zero KV row with a ``-inf`` score, the row max is floored
    at ``-1e30`` (an all-``-1`` row yields exactly 0), the sink enters the
    denominator ONCE with no value row, the fp32 row-sum is taken BEFORE ``P`` is
    rounded to bf16 for the PV product. It cannot be bit-exact with the kernel
    (see the function docstring, which records the measured gap); compare with
    the bf16 budget. The exact-math arm (``p_dtype=float32``) computes in fp64.

``dense_masked_reference``
    An INDEPENDENT spelling: a dense ``[S, S + N_c]`` score matrix, a multiplicity
    mask built by ``scatter_add`` (duplicates count twice), the sink as an extra
    column with a zero value row. Exists to catch gather bugs in the first oracle.

``block_reference`` runs the whole block (ops 1-12 of the forward, ``M:765-789``);
``block_baseline`` is the framework-shaped bf16 chain for attribution only.

The RoPE here is INTERLEAVED (adjacent pairs, GPT-J style, ``M:392-406``) on the
LAST ``rope_dim`` dims -- not the gated block's rotate_half/NeoX table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Geometry (an independent mirror of the block's dataclass; Flash-4.1 defaults)
# ---------------------------------------------------------------------------

KV_FAKE_QUANT_MODES = ("fp8_block32_ue8m0", "none")


@dataclass(frozen=True)
class RefGeometry:
    """Mirror of ``MqaSparseAttentionBlockGeometry``, kept independent on purpose.

    The first block of fields mirrors the block's geometry (Flash-4.1 values from
    ``inference_config.json``: dim 5120, 64 heads x 512, rope 64, q_lora 1280,
    o_lora 1024 x 8 groups, window 128, index_topk 512, eps 1e-20). The trailing
    RoPE-table parameters are REFERENCE-ONLY: the block takes the table as an
    input and never learns about YaRN; ``make_inputs`` needs them to build it
    (theta 10000 without YaRN on window-only layers, theta 160000 with YaRN
    factor 16 over 65536 on compressed layers -- ``M:680-698``).
    """

    d_model: int = 5120
    n_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window: int = 128
    index_topk: int = 512
    has_compressed_kv: bool = True
    norm_eps: float = 1e-20
    attn_scale: Optional[float] = None
    kv_fake_quant: str = "fp8_block32_ue8m0"
    # --- reference-only RoPE table parameters (not block geometry) ---
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    yarn_factor: float = 16.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_original_seq_len: int = 65536

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` (never assert) on a geometry the block cannot serve."""
        if self.head_dim <= 0 or self.head_dim % 32 != 0:
            raise ValueError(f"head_dim must be a positive multiple of 32 (the fake-quant block), got {self.head_dim}")
        if self.rope_dim <= 0 or self.rope_dim % 2 != 0 or self.rope_dim > self.head_dim:
            raise ValueError(f"rope_dim must be even and in (0, head_dim={self.head_dim}], got {self.rope_dim}")
        if (self.n_heads * self.head_dim) % self.o_groups != 0:
            raise ValueError(f"n_heads*head_dim ({self.n_heads * self.head_dim}) must be divisible by o_groups ({self.o_groups})")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be > 0, got {self.norm_eps}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.index_topk < 0:
            raise ValueError(f"index_topk must be >= 0, got {self.index_topk}")
        if self.kv_fake_quant not in KV_FAKE_QUANT_MODES:
            raise ValueError(f"kv_fake_quant must be one of {KV_FAKE_QUANT_MODES}, got {self.kv_fake_quant!r}")
        for name in ("d_model", "n_heads", "q_lora_rank", "o_lora_rank", "o_groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    @property
    def scale(self) -> float:
        # M:651 ``softmax_scale = head_dim**-0.5``; no mscale / temperature anywhere.
        return self.attn_scale if self.attn_scale is not None else float(self.head_dim) ** -0.5

    @property
    def group_width(self) -> int:
        return self.n_heads * self.head_dim // self.o_groups

    @property
    def n_o_lora(self) -> int:
        return self.o_groups * self.o_lora_rank

    @property
    def n_kv_slots_max(self) -> int:
        return self.window + (self.index_topk if self.has_compressed_kv else 0)

    @property
    def yarn(self) -> dict:
        """The ``yarn=`` argument of ``build_rope_tables`` for a compressed layer."""
        return dict(
            factor=self.yarn_factor,
            beta_fast=self.yarn_beta_fast,
            beta_slow=self.yarn_beta_slow,
            original_seq_len=self.yarn_original_seq_len,
        )


# Flash-4.1 at TP=1 (``inference_config.json``).
GEOMETRY_FLASH_41 = RefGeometry()

# A shrunk shape with the same structure, for oracle-speed correctness runs.
GEOMETRY_TINY = RefGeometry(d_model=64, n_heads=4, head_dim=64, rope_dim=16, q_lora_rank=32, o_lora_rank=16, o_groups=2)


# ---------------------------------------------------------------------------
# RoPE tables -- twin of ``precompute_freqs_cis`` (M:368-389)
# ---------------------------------------------------------------------------

_YARN_KEYS = ("factor", "beta_fast", "beta_slow", "original_seq_len")


def _check_yarn(yarn: Optional[Mapping]) -> Optional[Mapping]:
    if yarn is None:
        return None
    missing = [k for k in _YARN_KEYS if k not in yarn]
    if missing:
        raise ValueError(f"yarn needs keys {_YARN_KEYS}, missing {missing}")
    return yarn


def yarn_band(rope_dim: int, theta: float, yarn: Mapping) -> Tuple[int, int]:
    """``(low, high)`` frequency-index band of the YaRN ramp (M:377-383).

    ``corrected_dim(r) = dim * ln(original / (r * 2 pi)) / (2 ln base)``;
    ``low = max(floor(cd(beta_fast)), 0)``, ``high = min(ceil(cd(beta_slow)), dim - 1)``.
    Flash-4.1 (dim 64, base 160000, original 65536, 32 / 1) gives ``(15, 25)``.
    """
    yarn = _check_yarn(yarn)
    original = float(yarn["original_seq_len"])

    def corrected_dim(rotations: float) -> float:
        return rope_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(corrected_dim(float(yarn["beta_fast"]))), 0)
    high = min(math.ceil(corrected_dim(float(yarn["beta_slow"]))), rope_dim - 1)
    return low, high


def rope_frequencies(rope_dim: int, theta: float, *, yarn: Optional[Mapping] = None) -> torch.Tensor:
    """fp32 ``[rope_dim // 2]`` per-pair angular frequencies, YaRN-blended when asked.

    Twin of the first half of ``precompute_freqs_cis`` (M:376-386). As in the
    source, YaRN is applied only when ``original_seq_len > 0``.
    """
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be a positive even number, got {rope_dim}")
    yarn = _check_yarn(yarn)
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if yarn is not None and yarn["original_seq_len"] > 0:
        low, high = yarn_band(rope_dim, theta, yarn)
        ramp = ((torch.arange(rope_dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / float(yarn["factor"]) * (1 - smooth) + freqs * smooth
    return freqs


def freqs_cis_complex(seq_len: int, rope_dim: int, theta: float, *, yarn: Optional[Mapping] = None) -> torch.Tensor:
    """complex64 ``[seq_len, rope_dim // 2]`` -- byte-for-byte ``precompute_freqs_cis`` (M:388-389).

    ``torch.polar(ones, outer(arange(seq_len), freqs))``: the positional index is
    the int64 ``arange`` promoted to fp32 by ``outer``, exactly as in the source.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    freqs = rope_frequencies(rope_dim, theta, yarn=yarn)
    angles = torch.outer(torch.arange(seq_len), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def build_rope_tables(
    seq_len: int,
    rope_dim: int,
    theta: float,
    *,
    yarn: Optional[Mapping] = None,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` fp32 ``[seq_len, rope_dim // 2]`` for INTERLEAVED pairs.

    ``cos[t, k] + i sin[t, k]`` is exactly ``freqs_cis_complex(...)[t, k]``, so a
    table built here rotates bit-identically to the model's complex multiply
    (``apply_rope_interleaved`` reassembles the complex number from it).
    ``yarn=None`` is the window-only layer (plain theta); a dict with
    ``factor, beta_fast, beta_slow, original_seq_len`` is the compressed layer.
    """
    fc = freqs_cis_complex(seq_len, rope_dim, theta, yarn=yarn)
    return fc.real.contiguous().to(device), fc.imag.contiguous().to(device)


# ---------------------------------------------------------------------------
# Elementwise twins: RoPE (M:392-406), RMSNorm (M:280-293)
# ---------------------------------------------------------------------------


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Exact twin of ``apply_rotary_emb`` (M:392-406), returning a rotated COPY.

    Rotates the LAST ``2 * cos.shape[-1]`` dims of ``x``'s last axis as ADJACENT
    complex pairs ``(x[2k], x[2k+1])`` by ``cos + i sin`` (``inverse`` conjugates,
    i.e. ``theta -> -theta``). fp32 math (the source does ``x.float()`` and the
    complex multiply), a SINGLE rounding back to ``x.dtype`` on write
    (``y.copy_(x)``), passthrough dims returned bit-identical.

    ``x`` is ``[B, S, D]`` or ``[B, S, H, D]`` (sequence axis 1, as in the
    source's two ``view`` arms); ``cos``/``sin`` are ``[S, P]`` or ``[B, S, P]``.
    The rotation is spelled with ``view_as_complex`` exactly as the source is,
    so it is the model's op sequence rather than a re-derivation of it.
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"x must be [B, S, D] or [B, S, H, D], got shape {tuple(x.shape)}")
    if cos.shape != sin.shape or cos.ndim not in (2, 3):
        raise ValueError(f"cos/sin must be [S, P] or [B, S, P] with equal shapes, got {tuple(cos.shape)} / {tuple(sin.shape)}")
    pairs = cos.shape[-1]
    rope_dim = 2 * pairs
    seq_len = x.shape[1]
    if cos.shape[-2] != seq_len:
        raise ValueError(f"cos/sin sequence extent {cos.shape[-2]} != x sequence extent {seq_len}")
    if rope_dim > x.shape[-1]:
        raise ValueError(f"rope_dim {rope_dim} (2 * cos.shape[-1]) exceeds x last dim {x.shape[-1]}")
    freqs_cis = torch.complex(cos.float(), sin.float())
    if inverse:
        freqs_cis = freqs_cis.conj()
    batch = cos.shape[0] if cos.ndim == 3 else 1
    if x.ndim == 3:
        freqs_cis = freqs_cis.reshape(batch, seq_len, pairs)
    else:
        freqs_cis = freqs_cis.reshape(batch, seq_len, 1, pairs)
    rot = x[..., -rope_dim:]
    rot_c = torch.view_as_complex(rot.float().unflatten(-1, (-1, 2)).contiguous())
    rot = torch.view_as_real(rot_c * freqs_cis).flatten(-2)
    out = x.clone()
    out[..., -rope_dim:] = rot.to(x.dtype)
    return out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Twin of ``RMSNorm.forward`` (M:287-293): fp32 variance, weight promoted, cast back to ``x.dtype``."""
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


# ---------------------------------------------------------------------------
# Fake-quant twins: act_quant (K:41-124), fp4_act_quant with E4M3 scales (K:127-203)
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP8_MAX_INV_F32 = torch.tensor(1.0 / _FP8_MAX, dtype=torch.float32)  # K:44-46 ``fp8_max_inv = 1 / fp8_max`` as an fp32 constant
_FP8_AMAX_FLOOR = 1e-4  # K:76
_FP4_MAX = 6.0
_FP4_E4M3_AMAX_FLOOR = 6 * (2**-9)  # K:161


def _ceil_log2_pow2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(k, 2**k)`` with ``k = ceil(log2(x))`` for positive fp32 ``x`` -- K:22-37 without exp2/log2.

    ``fast_log2_ceil``: exponent - 127 + (mantissa != 0). With ``frexp`` giving
    ``x = m * 2**e``, ``m in [0.5, 1)``: ``x`` is a power of two iff ``m == 0.5``,
    so ``ceil(log2 x) = e - (m == 0.5)``. ``fast_pow2`` is ``ldexp(1, k)``.
    (``torch.exp2`` / ``torch.log2`` are jiterator ops that fail on cc 10.7.)
    """
    m, e = torch.frexp(x)
    k = e - (m == 0.5).to(e.dtype)
    return k, torch.ldexp(torch.ones_like(x), k)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.shape[-1]
    if block <= 0 or n % block != 0:
        raise ValueError(f"last dim {n} must be a positive multiple of the block size {block}")
    return x.unflatten(-1, (n // block, block))


def act_quant_scales(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """The fp32 ``[..., N // block]`` power-of-two scales ``act_quant`` computes (K:74-78) and, in-place, discards."""
    xb = _blocks(x, block).float()
    amax = xb.abs().amax(-1).clamp_min(_FP8_AMAX_FLOOR)
    _, s = _ceil_log2_pow2(amax * _FP8_MAX_INV_F32.to(amax.device))
    return s


def act_quant_mirror(x: torch.Tensor, block: int = 32, *, return_scales: bool = False):
    """Twin of ``act_quant(x, block, "ue8m0", e8m0, inplace=True)`` (K:41-124) as called at M:707.

    Per contiguous ``block`` along the last dim, fp32 internally:
    ``amax = max(max|x|, 1e-4)``; ``s = 2**ceil(log2(amax * fp32(1/448)))`` (floor
    scale ``2**-22``); ``y = x.dtype( fp32( e4m3( clamp(x / s, -448, 448) ) ) * s )``.
    The clamp precedes the e4m3 cast exactly as at K:85 (``T.Cast(FP8,
    T.clamp(...))``); the CUDA ``__nv_fp8_e4m3`` constructor that cast lowers to
    is saturating (satfinite), and torch >= 2.13 saturates too, so the clamp is a
    no-op made explicit. The result is exactly on the e4m3 x ``2**k`` grid, so it is
    idempotent and, for bf16 input, bf16-exact. Returns ``x.dtype`` (the kernel's
    ``out_dtype = in_dtype`` when in place: bf16 for the model).

    ``T.Cast(FP8, ...)`` (K:85) and ``.to(torch.float8_e4m3fn)`` both round to
    nearest-even; plan Q2's only remaining open item is the FP4 cast's rounding.
    """
    xb = _blocks(x, block).float()
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(_FP8_AMAX_FLOOR)
    _, s = _ceil_log2_pow2(amax * _FP8_MAX_INV_F32.to(amax.device))
    q = (xb / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float()
    y = (q * s).to(x.dtype).flatten(-2)
    if return_scales:
        return y, s.squeeze(-1)
    return y


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e2m1_round_to_nearest_even(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 ``v`` (already inside ``[-6, 6]``) to the e2m1 value set with RNE ties, explicitly.

    Codes in order: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (subnormal 0.5, then 1.m x 2^e).
    A tie at a midpoint goes to the EVEN code index: 0.25 -> 0, 0.75 -> 1,
    1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4. No ``torch.float4`` casts.
    The sign is carried by ``copysign`` (``-0.0`` stays ``-0.0``, like a sign-bit code).
    """
    a = v.abs().float()
    mids = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=v.device)
    idx_down = torch.bucketize(a, mids, right=False)  # a tie lands on the LOWER code
    idx_up = torch.bucketize(a, mids, right=True)  # a tie lands on the UPPER code
    idx = torch.where(idx_down % 2 == 0, idx_down, idx_up)  # off a tie both agree; on a tie take the even code
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=v.device)
    return torch.copysign(values[idx], v.float())


def fp4_e4m3_block16_mirror(x: torch.Tensor, block: int = 16, *, return_scales: bool = False):
    """Twin of ``fp4_act_quant(x, 16, inplace=True, scale_dtype=float8_e4m3fn)`` (K:127-203) as called at M:760.

    Per contiguous ``block`` along the last dim, fp32 internally:
    ``amax = max(max|x|, 6 * 2**-9)``; ``s = fp32(e4m3(amax / 6))``;
    ``y = x.dtype( fp32( e2m1_rne( clamp(x / s, -6, 6) ) ) * s )``.
    Idempotency (a property of the SOURCE math, mirrored faithfully) holds in
    two regimes and fails in the band between them:
    * ``amax >= 6 * 2**-6`` (``s`` a NORMAL e4m3, 3 mantissa bits): ``amax / s``
      lies in ``[6 / (1 + 2**-4), 6 * (1 + 2**-4)]``, so the max element is code
      6, ``amax' = 6 s`` exactly and the second pass recovers ``s``;
    * ``amax < 6 * 2**-9`` (floored): ``s = 2**-9`` on both passes;
    * ``6 * 2**-9 <= amax < 6 * 2**-6``: ``s`` is a SUBNORMAL e4m3 (``k * 2**-9``,
      up to 33 % relative error), the max element may quantise to code 4 (e.g.
      ``amax = 0.018 -> s = 2**-8, code 4``) and a second pass re-scales to
      ``2**-9`` and clamps -- NOT idempotent. Unit-RMS latents never sit there.

    ``amax / 6`` is clamped to 448 before the e4m3 cast: that mirrors the
    saturating (satfinite) fp32 -> e4m3 cast of the CUDA path ``T.Cast(FP8, .)``
    lowers to; torch >= 2.13 saturates too, so the clamp is a no-op made explicit
    (unreachable anyway for ``|x| <= 2688``). Plan Q2's remaining open item is
    the ``T.Cast(FP4, .)`` rounding mode, spelled here as RNE.
    """
    xb = _blocks(x, block).float()
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(_FP4_E4M3_AMAX_FLOOR)
    s = (amax / _FP4_MAX).clamp_max(_FP8_MAX).to(torch.float8_e4m3fn).float()
    q = e2m1_round_to_nearest_even((xb / s).clamp(-_FP4_MAX, _FP4_MAX))
    y = (q * s).to(x.dtype).flatten(-2)
    if return_scales:
        return y, s.squeeze(-1)
    return y


# ---------------------------------------------------------------------------
# Index lists: window (M:417-420, start_pos 0) and synthetic compressed (M:564-580)
# ---------------------------------------------------------------------------


def window_idxs(batch: int, seq_len: int, window: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Twin of ``get_window_topk_idxs(window, batch, seq_len, start_pos=0)`` (M:417-420, M:426).

    int32 ``[B, S, min(S, window)]``: row ``i`` lists positions ``max(0, i - window + 1) .. i``
    inclusive (``window`` keys including self), ``-1`` for slots before the sequence started.
    """
    if seq_len < 0 or window < 1 or batch < 1:
        raise ValueError(f"need seq_len >= 0, window >= 1, batch >= 1; got {seq_len}, {window}, {batch}")
    end = torch.arange(seq_len, device=device).unsqueeze(1)
    idxs = (end - window + 1).clamp(0) + torch.arange(min(seq_len, window), device=device)
    idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    return idxs.int().unsqueeze(0).expand(batch, -1, -1).contiguous()


def compressed_idxs_synthetic(
    batch: int,
    seq_len: int,
    ratio: int,
    topk: int,
    generator: torch.Generator,
    *,
    relative: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Random SORTED reachable compressed ids per query row, in the Indexer's output convention.

    Spelled as the Indexer's tail (M:564-565, M:578-580) over RANDOM scores:
    ``compress_lens[i] = (i + 1) // ratio`` (group ``j`` is visible to query ``i``
    iff ``j * ratio + ratio - 1 <= i``); unreachable scores are ``-inf``;
    ``topk = min(topk, S // ratio)``; ``idxs = topk(sorted=False).indices.sort()``;
    ``where(idxs < compress_lens, idxs + offset, -1)`` with ``offset = S`` (rows
    ``[S, S + N_c)`` of the concatenated KV, M:776-777) -- or ``0`` when
    ``relative=True``. Result: ascending real ids first, a ``-1`` tail, width
    ``min(topk, S // ratio)``; ``[B, S, 0]`` when ``S < ratio`` (M:729-730).
    """
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1 for a compressed list, got {ratio}")
    if topk < 0 or seq_len < 0 or batch < 1:
        raise ValueError(f"need topk >= 0, seq_len >= 0, batch >= 1; got {topk}, {seq_len}, {batch}")
    n_c = seq_len // ratio
    topk = min(topk, n_c)
    if topk == 0:
        return torch.empty(batch, seq_len, 0, dtype=torch.int32, device=device)
    offset = 0 if relative else seq_len
    score = torch.rand(batch, seq_len, n_c, generator=generator, device=generator.device).to(device)
    compress_lens = (torch.arange(1, seq_len + 1, device=device) // ratio).unsqueeze(-1)
    score.masked_fill_(torch.arange(n_c, device=device) >= compress_lens, -torch.inf)
    idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_inputs(
    geom: RefGeometry,
    batch: int,
    seq_len: int,
    ratio: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Seeded inputs in the block's declared layouts.

    ``ratio`` is the layer's ``compress_ratio``: ``0`` = window-only layer (no
    compressed part, plain theta table); ``1`` / ``2`` = compressed layer (YaRN
    table with ``compress_rope_theta``, a synthetic ``compress_kv`` and index list).

    Weights are ``nn.Linear`` shaped ``[out, in]`` bf16 with ``std = in**-0.5`` so
    every bf16 GEMM output is O(1); RMSNorm weights are bf16 near 1 (not exactly
    1, so an ignored weight is visible). ``attn_sink`` is fp32 ``[n_heads]``
    (M:639). ``cos`` / ``sin`` are fp32 ``[B, S, rope_dim // 2]`` interleaved-pair
    tables (the block's input form; ``[S, P]`` is one ``[0]`` away).

    ``compress_kv`` ``[B, S // ratio, head_dim]`` bf16 is built per plan 1.5:
    random latent -> RMSNorm (M:485) -> RoPE at position ``j * ratio`` with the
    layer table (M:753-758) -> fp4 / e4m3 block-16 fake-quant (M:760). It is
    ``None`` when ``ratio == 0`` or ``S < ratio``. ``topk_idxs`` ``[B, S, K]``
    int32 is the window list ++ the compressed list (``K = min(S, window) +
    min(index_topk, S // ratio)``), compressed ids offset by ``S`` (M:580).
    """
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, got {ratio}")
    if ratio > 0 and not geom.has_compressed_kv:
        raise ValueError("ratio > 0 needs geom.has_compressed_kv=True")
    if batch < 1 or seq_len < 1:
        raise ValueError(f"need batch >= 1 and seq_len >= 1, got {batch}, {seq_len}")
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, std: float = 1.0, out_dtype: torch.dtype = dtype) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * std).to(out_dtype)

    def norm_weight(n: int) -> torch.Tensor:
        return (1.0 + 0.05 * torch.randn(n, generator=g, device=device, dtype=torch.float32)).to(dtype)

    if ratio > 0:
        cos, sin = build_rope_tables(seq_len, geom.rope_dim, geom.compress_rope_theta, yarn=geom.yarn, device=device)
    else:
        cos, sin = build_rope_tables(seq_len, geom.rope_dim, geom.rope_theta, yarn=None, device=device)

    inputs = {
        "x": randn(batch, seq_len, geom.d_model, std=1.0),
        "w_q_a": randn(geom.q_lora_rank, geom.d_model, std=geom.d_model**-0.5),
        "w_q_norm": norm_weight(geom.q_lora_rank),
        "w_q_b": randn(geom.n_heads * geom.head_dim, geom.q_lora_rank, std=geom.q_lora_rank**-0.5),
        "w_kv": randn(geom.head_dim, geom.d_model, std=geom.d_model**-0.5),
        "w_kv_norm": norm_weight(geom.head_dim),
        "w_o_a": randn(geom.o_groups, geom.o_lora_rank, geom.group_width, std=geom.group_width**-0.5),
        "w_o_b": randn(geom.d_model, geom.n_o_lora, std=geom.n_o_lora**-0.5),
        "attn_sink": torch.randn(geom.n_heads, generator=g, device=device, dtype=torch.float32),
        "cos": cos.unsqueeze(0).expand(batch, -1, -1).contiguous(),
        "sin": sin.unsqueeze(0).expand(batch, -1, -1).contiguous(),
        "compress_kv": None,
        "topk_idxs": None,
        "ratio": ratio,
    }

    win = window_idxs(batch, seq_len, geom.window, device=device)
    n_c = seq_len // ratio if ratio > 0 else 0
    if n_c > 0:
        latent = randn(batch, n_c, geom.head_dim, std=1.0)
        latent = rmsnorm(latent, norm_weight(geom.head_dim), geom.norm_eps)
        # a latent stands for the first token of its group, so group j takes position j * ratio (M:753-758)
        cos_c = cos[: seq_len - seq_len % ratio : ratio]
        sin_c = sin[: seq_len - seq_len % ratio : ratio]
        latent = apply_rope_interleaved(latent, cos_c, sin_c)
        inputs["compress_kv"] = fp4_e4m3_block16_mirror(latent, 16)
        comp = compressed_idxs_synthetic(batch, seq_len, ratio, geom.index_topk, g, device=device)
        inputs["topk_idxs"] = torch.cat([win, comp], dim=-1)
    else:
        inputs["topk_idxs"] = win
    return inputs


# ---------------------------------------------------------------------------
# Attention oracles
# ---------------------------------------------------------------------------

_MAX_FLOOR = -1e30  # K:355: finite so an all -1 row gives O = 0, not NaN


def _work_dtype(p_dtype: torch.dtype) -> torch.dtype:
    """Compute dtype of the two attention oracles: fp32 for the kernel-mirror (bf16 ``P``)
    arm, FLOAT64 for the exact-math (fp32 ``P``) arm so it cannot become a TF32 reference
    under ``torch.backends.cuda.matmul.allow_tf32`` (the CI containers force that flag)."""
    return torch.float64 if p_dtype == torch.float32 else torch.float32


def _check_attention_args(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, topk_idxs: torch.Tensor) -> None:
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError(f"need q [B,S,H,D], kv [B,N,D], topk_idxs [B,S,K]; got {tuple(q.shape)}, {tuple(kv.shape)}, {tuple(topk_idxs.shape)}")
    b, s, h, d = q.shape
    if kv.shape[0] != b or kv.shape[2] != d:
        raise ValueError(f"kv must be [{b}, N, {d}], got {tuple(kv.shape)}")
    if topk_idxs.shape[:2] != (b, s):
        raise ValueError(f"topk_idxs must be [{b}, {s}, K], got {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32 (K:333), got {topk_idxs.dtype}")
    if sink.shape != (h,) or sink.dtype != torch.float32:
        raise ValueError(f"attn_sink must be fp32 [{h}] (M:639), got {tuple(sink.shape)} {sink.dtype}")
    n = kv.shape[1]
    if topk_idxs.numel() and (int(topk_idxs.min()) < -1 or int(topk_idxs.max()) >= n):
        raise ValueError(f"topk_idxs must lie in [-1, {n}), got [{int(topk_idxs.min())}, {int(topk_idxs.max())}]")


def _finish_row(num: torch.Tensor, sum_p: torch.Tensor, m: torch.Tensor, m_raw: torch.Tensor, sink_b: torch.Tensor, out_dtype: torch.dtype):
    """Shared epilogue: ``o = num / (sum_p + exp(sink - m))`` (K:382-387) and the FROST-convention LSE.

    ``m`` is the ``-1e30``-floored running max the kernel divides against; ``m_raw``
    is the true row max (``-inf`` on an all-``-1`` row). ``lse = logaddexp(m_raw +
    log(sum_p), sink)``: sink INCLUDED (the D512 convention), exactly ``sink`` on
    a keyless row, the plain log-sum-exp when ``sink = -inf``. The kernel emits
    no LSE at all. ``o`` on a keyless row: ``exp(sink + 1e30) = +inf`` and
    ``0 / inf = 0`` exactly, as in the kernel; with ``sink = -inf`` AND no key the
    kernel would give ``0 / 0 = NaN`` -- we SELECT 0 there (FROST convention).
    """
    sum_exp = sum_p + torch.exp(sink_b.to(m.dtype) - m)
    o = torch.where((sum_exp > 0).unsqueeze(-1), num / sum_exp.unsqueeze(-1), torch.zeros_like(num)).to(out_dtype)
    lse_keys = m_raw + torch.log(sum_p)  # -inf on a keyless row (log 0)
    lse = torch.logaddexp(lse_keys, sink_b.to(m.dtype))
    lse = torch.where(torch.isnan(lse), torch.full_like(lse, -torch.inf), lse)  # both -inf: no key, no sink
    return o, lse.to(torch.float32)


def sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
    *,
    p_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype = torch.bfloat16,
    q_chunk: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GATHERED oracle of ``sparse_attn_kernel`` (K:311-389) in batch form.

    ``q [B, S, H, D]``, ``kv [B, N, D]`` (ONE tensor for both GEMMs, K == V),
    ``sink [H]`` fp32, ``topk_idxs [B, S, K]`` int32 with ``-1`` = no key.
    Returns ``(o [B, S, H, D] out_dtype, lse [B, H, S] fp32)``.

    Per row: gather ``kv[b, idx]`` (a ZERO row for ``-1``); fp32 scores from the
    bf16 values (exact products, fp32 accumulation); ``-inf`` where ``idx == -1``;
    ``scale`` applied AFTER the product (K:366); row max floored at ``-1e30``
    (K:355); ``p = exp(s - m)``; fp32 denominator ``sum p + exp(sink - m)`` (sink
    ONCE, per head, no value row, K:374 + K:382-383); numerator with ``P`` cast to
    ``p_dtype`` (bf16, K:339/K:377) before the PV product; ``o = num / den``.

    Bitwise equality with the kernel is IMPOSSIBLE in a batch form: the kernel
    rounds ``P`` to bf16 against the RUNNING max of its 64-slot blocks and
    rescales the fp32 accumulators by ``alpha`` per block, so its bf16 ``P``
    values and its fp32 accumulation order differ from any single-pass spelling.
    Compare with a tolerance -- and set that tolerance from measurement, never
    widen it silently. Measured against a sequential emulation of the K:350-387
    loop (``test_mqa_block_reference_adversarial.py::_sparse_attn_kernel_emulation``,
    2026-09-16): the bf16 ``O`` of this oracle differs from the emulated kernel's
    in 10.8 % (S=300, K=128) to 27.1 % (S=640, K=640, d=512) of elements, every
    difference EXACTLY one bf16 ulp (max |dO| 3.9e-3 .. 7.8e-3 at |O| ~ 1), and
    all of it inside the ``_assert_bf16_budget`` bar (rtol ``2**-7``, atol
    ``2**-8 * max|kv|``); LSE agrees to <= 9.5e-7. So a kernel-vs-oracle bar for
    the D512 stage must be that bf16 budget -- a bitwise or 1e-4 bar is red on
    10-27 % of the elements of a CORRECT kernel. Corner rows: keyless row ``O = 0``
    in both (the kernel's implied LSE is +inf where this oracle publishes
    ``sink``); ``sink = -inf`` AND no key: kernel ``0 / 0 = NaN``, oracle selected
    ``0`` / ``-inf``.

    ``p_dtype=torch.float32`` / ``out_dtype=torch.float32`` give the exact-math
    variant used to cross-check the two oracles tightly. That arm runs its
    scores / PV einsums and exp / log in FLOAT64 (cast back to fp32 at the end):
    an fp32 einsum with a full-precision fp32 ``P`` is a TF32 einsum whenever
    ``torch.backends.cuda.matmul.allow_tf32`` is set -- the DLFW CI containers
    force it via ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`` -- and TF32 never applies
    to fp64. The default bf16-``P`` arm keeps fp32: its GEMM operands are
    bf16-valued, so TF32 reproduces their products exactly.
    """
    _check_attention_args(q, kv, sink, topk_idxs)
    b, s, h, d = q.shape
    dev = q.device
    work = _work_dtype(p_dtype)
    kvf = kv.to(work)
    sink_b = sink.to(dev).to(work).view(1, 1, h)
    o = torch.empty(b, s, h, d, dtype=out_dtype, device=dev)
    lse = torch.empty(b, s, h, dtype=torch.float32, device=dev)
    bidx = torch.arange(b, device=dev)[:, None, None]
    for lo in range(0, s, max(1, q_chunk)):
        hi = min(s, lo + q_chunk)
        idx = topk_idxs[:, lo:hi].long()  # [B, C, K]
        valid = idx != -1
        kv_g = kvf[bidx, idx.clamp_min(0)]  # [B, C, K, D]
        kv_g = kv_g.masked_fill(~valid.unsqueeze(-1), 0.0)  # K:362: -1 -> zero row
        scores = torch.einsum("bchd,bckd->bchk", q[:, lo:hi].to(work), kv_g)
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)  # K:364: -1 -> -inf
        scores = scores * scale  # K:366-367: AFTER the GEMM
        m_raw = scores.amax(-1)  # -inf on an all -1 row
        m = m_raw.clamp_min(_MAX_FLOOR)  # K:355
        p = torch.exp(scores - m.unsqueeze(-1))  # 0 on -1 slots (and everywhere on a keyless row)
        sum_p = p.sum(-1)  # K:374-376: fp32 P
        num = torch.einsum("bchk,bckd->bchd", p.to(p_dtype).to(work), kv_g)  # K:377-380: bf16 P
        o[:, lo:hi], lse[:, lo:hi] = _finish_row(num, sum_p, m, m_raw, sink_b, out_dtype)
    return o, lse.permute(0, 2, 1).contiguous()


def multiplicity_mask(topk_idxs: torch.Tensor, n_kv: int) -> torch.Tensor:
    """fp32 ``[B, S, n_kv]`` count of how many list slots name each KV row (duplicates count twice)."""
    b, s, _ = topk_idxs.shape
    idx = topk_idxs.long()
    valid = idx != -1
    count = torch.zeros(b, s, n_kv, dtype=torch.float32, device=topk_idxs.device)
    count.scatter_add_(-1, idx.clamp_min(0), valid.float())
    return count


def dense_masked_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
    *,
    p_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype = torch.bfloat16,
    q_chunk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """INDEPENDENT spelling of ``sparse_attention_reference`` over a dense ``[S, N]`` score matrix.

    A multiplicity ``count[b, s, n]`` (``scatter_add`` of ones over the valid list
    slots) replaces the gather: ``score = q . kv * scale + log(count)`` where
    ``count > 0`` (``log 2`` = the slot repeated), ``-inf`` where ``count == 0``.
    The sink is an EXTRA COLUMN with a zero value row, so the row max includes it:
    a keyless row with a FINITE sink has ``m = sink``, ``p_sink = 1``, ``o = 0``,
    ``lse = sink`` and the ``-1e30`` floor is never active. With ``sink = -inf``
    AND no key every column is ``-inf``, so the max is floored at ``-1e30`` (as
    K:355 does) and the empty denominator is SELECTed to ``o = 0``, ``lse = -inf``
    -- the same FROST convention as the gathered oracle, so the two spellings
    agree on every input both accept. Same compute dtype rule as the gathered
    oracle (fp32 for the bf16-``P`` arm, fp64 for the exact arm), the same
    ``p_dtype`` rounding point on ``P`` before the PV product, same signature and
    return convention.

    Because ``exp(s + log 2) != 2 exp(s)`` in fp32 and the max may differ, the bf16
    ``P`` of the two spellings can differ by one bf16 ulp in a few entries; the
    exact-math variant (``p_dtype=out_dtype=float32``) agrees to fp32 tolerance.
    """
    _check_attention_args(q, kv, sink, topk_idxs)
    b, s, h, d = q.shape
    n = kv.shape[1]
    dev = q.device
    work = _work_dtype(p_dtype)
    kvf = kv.to(work)
    kv_ext = torch.cat([kvf, torch.zeros(b, 1, d, dtype=work, device=dev)], dim=1)  # sink value row = 0
    sink_col = sink.to(dev).to(work).view(1, 1, h, 1)
    o = torch.empty(b, s, h, d, dtype=out_dtype, device=dev)
    lse = torch.empty(b, s, h, dtype=torch.float32, device=dev)
    for lo in range(0, s, max(1, q_chunk)):
        hi = min(s, lo + q_chunk)
        count = multiplicity_mask(topk_idxs[:, lo:hi], n).to(work)  # [B, C, N]
        logc = torch.where(count > 0, torch.log(count), torch.full_like(count, -torch.inf))
        scores = torch.einsum("bchd,bnd->bchn", q[:, lo:hi].to(work), kvf) * scale + logc.unsqueeze(2)
        scores = torch.cat([scores, sink_col.expand(b, hi - lo, h, 1)], dim=-1)  # [B, C, H, N + 1]
        # finite for a finite sink (the sink column is always there); with sink = -inf AND no key every
        # column is -inf, so floor the max as K:355 does and select the empty denominator below
        m = scores.amax(-1, keepdim=True).clamp_min(_MAX_FLOOR)
        p = torch.exp(scores - m)
        den = p.sum(-1)  # 0 only when no key AND sink = -inf
        num = torch.einsum("bchn,bnd->bchd", p.to(p_dtype).to(work), kv_ext)
        o[:, lo:hi] = torch.where((den > 0).unsqueeze(-1), num / den.unsqueeze(-1), torch.zeros_like(num)).to(out_dtype)
        lse[:, lo:hi] = torch.where(den > 0, m.squeeze(-1) + torch.log(den), torch.full_like(den, -torch.inf)).to(torch.float32)
    return o, lse.permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# The block (M:765-789), ops 1-12
# ---------------------------------------------------------------------------


@dataclass
class RefOutputs:
    """Everything the oracle computed, so a mismatch localizes to a stage.

    ``o`` is the sparse-attention output (op 9, what the attention kernel must
    match) BEFORE the inverse RoPE; ``o_unrot`` (appended) is after it.
    """

    qr: torch.Tensor  # [B, S, q_lora_rank]        bf16, op 1
    q: torch.Tensor  # [B, S, H, D]                 bf16, ops 2-3 (post-RoPE)
    window_kv: torch.Tensor  # [B, S, D]             bf16, ops 4-6 (post fake-quant; M:716 `window_kv = kv`): the block writes these rows of kv_all
    o: torch.Tensor  # [B, S, H, D]                 bf16, op 9 (pre inverse-RoPE)
    o_lora: torch.Tensor  # [B, S, G, o_lora_rank]   bf16, op 11
    out: torch.Tensor  # [B, S, d_model]             bf16, op 12
    lse: torch.Tensor  # [B, H, S]                   fp32, FROST convention (sink included)
    o_unrot: Optional[torch.Tensor] = None  # [B, S, H, D] bf16, op 10 (post inverse-RoPE), appended
    kv_all: Optional[torch.Tensor] = None  # [B, S + N_c, D] bf16, the attention KV, appended


def _linear_bf16_out(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``F.linear`` of bf16 operands with fp32 accumulation and ONE rounding to ``x.dtype``.

    The model's bf16 mode is a device bf16 GEMM (fp32 accumulate, bf16 out). The
    operands are bf16-valued, so their products are exact in fp32 (and in TF32);
    only the accumulation order differs from the device GEMM.
    """
    return F.linear(x.float(), w.float()).to(x.dtype)


def _check_block_inputs(geom: RefGeometry, inputs: dict) -> None:
    x = inputs["x"]
    if x.ndim != 3 or x.shape[-1] != geom.d_model:
        raise ValueError(f"x must be [B, S, {geom.d_model}], got {tuple(x.shape)}")
    expect = {
        "w_q_a": (geom.q_lora_rank, geom.d_model),
        "w_q_norm": (geom.q_lora_rank,),
        "w_q_b": (geom.n_heads * geom.head_dim, geom.q_lora_rank),
        "w_kv": (geom.head_dim, geom.d_model),
        "w_kv_norm": (geom.head_dim,),
        "w_o_a": (geom.o_groups, geom.o_lora_rank, geom.group_width),
        "w_o_b": (geom.d_model, geom.n_o_lora),
        "attn_sink": (geom.n_heads,),
    }
    for name, shape in expect.items():
        if tuple(inputs[name].shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(inputs[name].shape)}")
    if inputs["compress_kv"] is not None and not geom.has_compressed_kv:
        raise ValueError("compress_kv given but geom.has_compressed_kv is False")


def block_reference(geom: RefGeometry, inputs: dict) -> RefOutputs:
    """The forward of ``Attention`` (M:765-789) with the model's bf16 rounding points.

    1 ``qr = q_norm(wq_a(x))``; 2 ``q = wq_b(qr).unflatten(-1, (H, D))``; 3 RoPE on
    the last ``rope_dim`` dims; 4 ``kv = kv_norm(wkv(x))``; 5 RoPE; 6 block-32 ue8m0
    fake-quant of the whole vector (skipped under ``kv_fake_quant="none"``);
    7-8 ``kv_all = cat(kv, compress_kv)``, ``topk_idxs`` as given; 9 sparse
    attention with the sink; 10 inverse RoPE on ``o``; 11 grouped ``wo_a`` einsum
    (fp32 accumulate -> bf16); 12 ``wo_b``. Every GEMM output, norm, RoPE,
    fake-quant and the attention output round to bf16 exactly where the model does.
    """
    _check_block_inputs(geom, inputs)
    x = inputs["x"]
    b, s, _ = x.shape
    cos, sin = inputs["cos"], inputs["sin"]
    dtype = x.dtype

    qr = rmsnorm(_linear_bf16_out(x, inputs["w_q_a"]), inputs["w_q_norm"], geom.norm_eps)  # M:770
    q = _linear_bf16_out(qr, inputs["w_q_b"]).unflatten(-1, (geom.n_heads, geom.head_dim))  # M:771
    q = apply_rope_interleaved(q, cos, sin)  # M:772

    kv = rmsnorm(_linear_bf16_out(x, inputs["w_kv"]), inputs["w_kv_norm"], geom.norm_eps)  # M:705
    kv = apply_rope_interleaved(kv, cos, sin)  # M:706
    if geom.kv_fake_quant == "fp8_block32_ue8m0":
        kv = act_quant_mirror(kv, 32)  # M:707: whole post-RoPE vector, RoPE tail included

    compress_kv = inputs["compress_kv"]
    kv_all = kv if compress_kv is None else torch.cat([kv, compress_kv.to(dtype)], dim=1)  # M:777
    topk_idxs = inputs["topk_idxs"]

    o, lse = sparse_attention_reference(q, kv_all, inputs["attn_sink"], topk_idxs, geom.scale)  # M:780
    o_unrot = apply_rope_interleaved(o, cos, sin, inverse=True)  # M:781

    o_g = o_unrot.reshape(b, s, geom.o_groups, geom.group_width)  # M:785: group g = heads 8g..8g+7 contiguous
    o_lora = torch.einsum("bsgd,grd->bsgr", o_g.float(), inputs["w_o_a"].float()).to(dtype)  # M:786-787
    out = _linear_bf16_out(o_lora.flatten(2), inputs["w_o_b"])  # M:788
    return RefOutputs(qr=qr, q=q, window_kv=kv, o=o, o_lora=o_lora, out=out, lse=lse, o_unrot=o_unrot, kv_all=kv_all)


def block_baseline(geom: RefGeometry, inputs: dict) -> RefOutputs:
    """Framework-shaped bf16 chain for attribution only: device bf16 ``F.linear`` /
    ``einsum`` and the DENSE masked attention (O3 spelling). Same rounding points
    as ``block_reference`` otherwise; not an oracle."""
    _check_block_inputs(geom, inputs)
    x = inputs["x"]
    b, s, _ = x.shape
    cos, sin = inputs["cos"], inputs["sin"]
    dtype = x.dtype
    qr = rmsnorm(F.linear(x, inputs["w_q_a"].to(dtype)), inputs["w_q_norm"], geom.norm_eps)
    q = apply_rope_interleaved(F.linear(qr, inputs["w_q_b"].to(dtype)).unflatten(-1, (geom.n_heads, geom.head_dim)), cos, sin)
    kv = apply_rope_interleaved(rmsnorm(F.linear(x, inputs["w_kv"].to(dtype)), inputs["w_kv_norm"], geom.norm_eps), cos, sin)
    if geom.kv_fake_quant == "fp8_block32_ue8m0":
        kv = act_quant_mirror(kv, 32)
    compress_kv = inputs["compress_kv"]
    kv_all = kv if compress_kv is None else torch.cat([kv, compress_kv.to(dtype)], dim=1)
    o, lse = dense_masked_reference(q, kv_all, inputs["attn_sink"], inputs["topk_idxs"], geom.scale)
    o_unrot = apply_rope_interleaved(o, cos, sin, inverse=True)
    o_lora = torch.einsum("bsgd,grd->bsgr", o_unrot.reshape(b, s, geom.o_groups, geom.group_width), inputs["w_o_a"].to(dtype))
    out = F.linear(o_lora.flatten(2), inputs["w_o_b"].to(dtype))
    return RefOutputs(qr=qr, q=q, window_kv=kv, o=o, o_lora=o_lora, out=out, lse=lse, o_unrot=o_unrot, kv_all=kv_all)
