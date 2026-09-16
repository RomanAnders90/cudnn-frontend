# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry of the MQA sparse-attention block: what the stages specialize on.

Plan-time only, never runtime data. Everything here is a plain Python value the
caller derives from a model declaration; nothing here allocates, and ``torch``
is imported lazily inside the two load-time helpers so ``import cudnn`` stays
cheap (``python/cudnn/AGENTS.md``, import-time rule).

The block is ONE attention layer of the form (op geometry, not a model name --
DeepSeek-V4.1-Flash is the shipped instance and supplies the dataclass defaults):

    qr  = RMSNorm(x @ W_q_a^T)                       [B,S,q_lora_rank]
    q   = (qr @ W_q_b^T).view(B,S,n_heads,head_dim); RoPE on the LAST rope_dim dims
    kv  = fake_quant(RoPE_last(RMSNorm(x @ W_kv^T))) [B,S,head_dim]   ONE vector per token
    o   = sparse_attn(q, kv_all, sink, topk_idxs)    [B,S,n_heads,head_dim]
    o   = inverse-RoPE on the LAST rope_dim dims
    out = concat_g( o[:, :, g] @ W_o_a[g]^T ) @ W_o_b^T                [B,S,d_model]

Conventions the whole package and every reference/test depend on (each one is a
fact of the reference implementation, cited as ``model.py`` ``M:<line>`` and
``kernel.py`` ``K:<line>`` of the verified source tree):

**K == V.** There is a single KV projection of width ``head_dim``
(``self.wkv = Linear(self.dim, self.head_dim)``, ``M:643``; ``kv_norm =
RMSNorm(head_dim)``, ``M:644``). The attention kernel feeds the SAME gathered
row to both GEMMs -- ``T.gemm(q_shared, kv_shared, ...)`` at ``K:365`` and
``T.gemm(acc_s_cast, kv_shared, ...)`` at ``K:380``. There is no separate V,
no KV head dimension (MQA, one KV shared by all ``n_heads`` query heads), and
no way to pass a distinct V into this block.

**Sink.** ``attn_sink`` is an fp32 parameter of shape ``[n_heads]``
(``M:639``). It enters the softmax DENOMINATOR only, ONCE, after every KV
block has been folded in: ``sum_exp[i] += exp(attn_sink[i] - scores_max[i])``
(``K:383``); there is no value row for it and the reference emits no LSE. If
this block ever returns an LSE, it is the FROST convention -- sink INCLUDED.

**-1 slot.** An index of ``-1`` in ``topk_idxs`` means "no key": the gather
substitutes a ZERO row (``K:362``) and the score is forced to ``-inf`` BEFORE
the QK GEMM (``K:364``). The running max starts at the FINITE floor ``-1e30``
(``K:355``) so a row whose every slot is ``-1`` yields ``O = 0``, not NaN.
Duplicated indices are distinct slots and are counted twice (``M:415``,
"handles every slot independently"); nothing dedupes.

**Index offset.** ``topk_idxs`` is the CONCATENATION of the sliding-window list
and the compressed list, and it indexes ONE tensor ``kv_all = cat([kv,
compress_kv], dim=1)`` (``M:777``). Window ids are absolute token positions
``max(0, i - window + 1) .. i`` with ``-1`` where the window predates the
sequence (``M:417-420``). Compressed ids are ALREADY OFFSET by ``S`` (the
window row count): ``where(idxs < compress_lens, idxs + offset, -1)`` with
``offset = kv.size(1)`` (``M:580``, ``M:776``), sorted ascending with the
``-1`` run at the tail. A model dump is therefore a valid block input with no
re-indexing.

**RoPE pairing is INTERLEAVED (GPT-J): element ``2k`` pairs with ``2k+1``**
(``view_as_complex(x.float().unflatten(-1, (-1, 2)))``, ``M:397``), applied
to the LAST ``rope_dim`` dims of every head (``q[..., -rd:]``, ``M:772``;
``kv[..., -rd:]``, ``M:706``), and the attention OUTPUT gets the conjugate
rotation at the query position (``inverse=True``, ``M:781``). Not the
rotate_half / NeoX layout of ``cudnn.gated_attention_block``.

**Grouped output projection.** ``W_o_a`` is block-diagonal over
``o_groups``: group ``g`` sees heads ``g*n_heads/o_groups ..`` contiguous
(``o.view(bsz, seqlen, n_groups, -1)``, ``M:785``) and its weight is a
zero-copy VIEW of the checkpoint matrix (``self.wo_a.weight.view(n_groups,
o_lora_rank, -1)``, ``M:786``) -- see :func:`build_grouped_wo_a`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover -- typing only; torch is imported lazily below
    import torch


# The window KV rows are fake-quantised in contiguous blocks of this many
# elements: ``act_quant(kv, fp8_block_size=32, "ue8m0", e8m0, inplace=True)``
# at ``M:707`` with ``fp8_block_size = 32`` (``M:27``) and
# ``scale_fmt = "ue8m0"`` (``M:29``). ``act_quant`` requires
# ``head_dim % block_size == 0`` (``K:108``).
KV_FAKE_QUANT_BLOCK = 32

# The two spellings ``kv_fake_quant`` accepts. ``"fp8_block32_ue8m0"`` mirrors
# the model: per block, ``amax = max(|x|, 1e-4)`` (``K:76``), ``s =
# 2^ceil(log2(amax / 448))`` (``K:78``), ``y = bf16(fp32(e4m3(clamp(x/s,
# -448, 448))) * s)`` (``K:85``). ``"none"`` skips it (a bf16-mode caller that
# wants the un-quantised window rows).
KV_FAKE_QUANT_MODES = ("fp8_block32_ue8m0", "none")


@dataclass(frozen=True)
class YarnParams:
    """YaRN frequency correction, exactly the four ``ModelArgs`` fields
    ``precompute_freqs_cis`` consumes (``M:369``, ``M:379-386``).

    ``original_seq_len`` is the training context the ramp is anchored to; the
    model enables YaRN iff it is ``> 0`` (``M:378``). Passing ``YarnParams``
    with ``original_seq_len <= 0`` is rejected rather than silently building an
    un-corrected table -- pass ``yarn=None`` to mean "no YaRN".
    """

    factor: float
    beta_fast: float
    beta_slow: float
    original_seq_len: int

    def validate(self) -> None:
        if not self.factor > 0.0:
            raise ValueError(f"YarnParams.factor must be > 0, got {self.factor}")
        if not self.beta_fast > 0.0 or not self.beta_slow > 0.0:
            raise ValueError(f"YarnParams.beta_fast / beta_slow must be > 0, got {self.beta_fast} / {self.beta_slow}")
        if self.original_seq_len <= 0:
            raise ValueError(
                f"YarnParams.original_seq_len must be > 0 (the model applies YaRN iff original_seq_len > 0, M:378); "
                f"got {self.original_seq_len} -- pass yarn=None for a plain table"
            )


@dataclass(frozen=True)
class MqaSparseAttentionBlockGeometry:
    """Everything the stages specialize on. Plan-time only, never runtime data.

    Defaults are the released DeepSeek-V4.1-Flash shape at TP=1
    (``inference_config.json``: ``dim 5120, n_heads 64, head_dim 512,
    rope_head_dim 64, q_lora_rank 1280, o_lora_rank 1024, o_groups 8,
    window_size 128, index_topk 512, norm_eps 1e-20``). Field names are op
    geometry, not model names.

    ``has_compressed_kv`` is the caller's ``compress_ratio > 0``: layers 0-1 of
    the release attend the window only (``M:775`` skips ``_compress_kv``;
    ``MqaSparseAttentionBlockGeometry(has_compressed_kv=False)``); layers 2-39
    (ratio 2 on 2-19, ratio 1 on 20-39) add the Indexer's top-``index_topk``
    compressed rows -- the plain default. When ``has_compressed_kv=False`` the
    ``index_topk`` field is unused (``n_kv_slots_max`` ignores it) and may be 0. The
    ratio itself is NOT a field -- ``N_c = compress_kv.shape[1]`` and the index
    width ``K`` are runtime extents (``min(S, window) + min(index_topk,
    S // ratio)``, ``M:419``, ``M:578``).

    ``kv_fake_quant`` selects the bf16 fake-quantisation of the window KV rows
    (``M:707``); see :data:`KV_FAKE_QUANT_MODES`. The compressed rows are the
    caller's input and are never re-quantised here.

    ``attn_scale=None`` means ``head_dim ** -0.5`` (``self.softmax_scale =
    self.head_dim**-0.5``, ``M:651``); there is no mscale / temperature.
    """

    d_model: int = 5120
    n_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64  # the LAST rope_dim dims of every head rotate; [0, head_dim - rope_dim) pass through
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window: int = 128
    index_topk: int = 512
    has_compressed_kv: bool = True
    norm_eps: float = 1e-20
    attn_scale: Optional[float] = None
    kv_fake_quant: str = "fp8_block32_ue8m0"

    # -- derived scalars ----------------------------------------------------

    @property
    def scale(self) -> float:
        """Softmax scale actually used: ``attn_scale`` or ``head_dim ** -0.5`` (``M:651``)."""
        return self.attn_scale if self.attn_scale is not None else float(self.head_dim) ** -0.5

    @property
    def nope_dim(self) -> int:
        """Leading pass-through dims of every head (``self.nope_head_dim``, ``M:632``)."""
        return self.head_dim - self.rope_dim

    @property
    def heads_per_group(self) -> int:
        """Query heads one ``W_o_a`` group projects (``M:785``)."""
        return self.n_heads // self.o_groups

    @property
    def group_width(self) -> int:
        """K of each grouped ``W_o_a`` GEMM: ``n_heads * head_dim // o_groups``
        (``ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, ...)``, ``M:646``)."""
        return self.n_heads * self.head_dim // self.o_groups

    @property
    def n_o_lora(self) -> int:
        """Width of the concatenated grouped projection = K of ``W_o_b``:
        ``o_groups * o_lora_rank`` (``M:647``, ``M:650``)."""
        return self.o_groups * self.o_lora_rank

    @property
    def n_kv_slots_max(self) -> int:
        """Upper bound on the index-list width ``K`` per query, independent of S:
        ``window`` window slots plus ``index_topk`` compressed slots when the
        layer has compressed KV (``M:419``, ``M:578``)."""
        return self.window + (self.index_topk if self.has_compressed_kv else 0)

    # -- validation -----------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on any geometry the stages cannot express.

        Never a module-level ``assert`` (engine contract § 7): anything derived
        from user input raises, so it survives ``python -O`` and names itself.
        Each message says what the failure would have LOOKED like, because a
        config error that reaches a kernel does not announce itself.
        """
        for label, value in (
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("head_dim", self.head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("o_groups", self.o_groups),
            ("window", self.window),
        ) + ((("index_topk", self.index_topk),) if self.has_compressed_kv else ()):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive int, got {value!r}")
        if not self.has_compressed_kv and (not isinstance(self.index_topk, int) or isinstance(self.index_topk, bool) or self.index_topk < 0):
            # Window-only layer: the field is dead (n_kv_slots_max ignores it), so a ModelArgs-style 0 is legal.
            raise ValueError(f"index_topk must be a non-negative int (unused when has_compressed_kv=False), got {self.index_topk!r}")

        if self.kv_fake_quant not in KV_FAKE_QUANT_MODES:
            raise ValueError(f"kv_fake_quant must be one of {KV_FAKE_QUANT_MODES}, got {self.kv_fake_quant!r}")

        # The fake-quant rule is checked first so its more specific message wins
        # when both apply (they share the granule: KV_FAKE_QUANT_BLOCK == 32).
        if self.kv_fake_quant != "none" and self.head_dim % KV_FAKE_QUANT_BLOCK != 0:
            raise ValueError(
                f"kv_fake_quant={self.kv_fake_quant!r} needs head_dim % {KV_FAKE_QUANT_BLOCK} == 0 "
                f"(act_quant asserts N % block_size == 0, K:108; a partial trailing block has no scale), got head_dim={self.head_dim}"
            )
        if self.head_dim % 32 != 0:
            raise ValueError(
                f"head_dim must be a multiple of 32 (the block's head-tiling granule; the attention stage adds its own "
                f"head_dim constraint at plan time), got {self.head_dim}"
            )

        if not 0 < self.rope_dim <= self.head_dim:
            raise ValueError(
                f"rope_dim must be in (0, head_dim={self.head_dim}] (the block always rotates the LAST rope_dim dims, M:772/M:781), got {self.rope_dim}"
            )
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even (interleaved pairs (2k, 2k+1), M:397), got {self.rope_dim}")

        if (self.n_heads * self.head_dim) % self.o_groups != 0:
            raise ValueError(
                f"n_heads * head_dim ({self.n_heads * self.head_dim}) must be divisible by o_groups ({self.o_groups}) "
                f"for the grouped W_o_a GEMM width (M:646, M:785)"
            )
        if self.n_heads % self.o_groups != 0:
            # The model's `o.view(bsz, seqlen, n_groups, -1)` (M:785) would let a group straddle a head; this
            # package's documented per-head map (`heads_per_group`, `build_grouped_wo_a`) is only true for whole heads.
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by o_groups ({self.o_groups}): each W_o_a group projects whole "
                f"heads (heads_per_group = n_heads // o_groups), got a group of {self.group_width / self.head_dim:.2f} heads"
            )

        if not self.norm_eps > 0.0:
            raise ValueError(f"norm_eps must be > 0 (RMSNorm rsqrt(var + eps), M:292), got {self.norm_eps}")
        if self.attn_scale is not None and not self.attn_scale > 0.0:
            raise ValueError(f"attn_scale must be > 0 when given, got {self.attn_scale}")


# ---------------------------------------------------------------------------
# Load-time helpers (torch imported lazily -- never on the import path)
# ---------------------------------------------------------------------------


def build_grouped_wo_a(w_o_a: "torch.Tensor", geometry: MqaSparseAttentionBlockGeometry) -> "torch.Tensor":
    """View the checkpoint ``wo_a.weight [o_groups * o_lora_rank, group_width]`` as
    ``[o_groups, o_lora_rank, group_width]`` -- a VIEW, never a copy.

    **Load time only.** Mirrors ``self.wo_a.weight.view(self.n_local_groups,
    self.o_lora_rank, -1)`` (``M:786``) and the weight's declared shape
    ``ColumnParallelLinear(n_heads * head_dim // o_groups, o_groups *
    o_lora_rank)`` (``M:645-649``). Group ``g`` (``[g]`` of the result) projects
    heads ``g * heads_per_group .. (g+1) * heads_per_group - 1``, whose
    ``head_dim`` blocks are contiguous in ``o.view(B, S, o_groups, -1)``
    (``M:785``): element ``(h % heads_per_group) * head_dim + k`` of the group
    row. The weight is bf16 in BOTH dtype modes of the model (``M:648``,
    ``dtype=torch.bfloat16``; ``M:783-784`` "convert.py dequantizes it to
    bf16"), but this helper is dtype-agnostic.

    Only the leading dim is split, so the result shares storage with the input
    for ANY row stride (``data_ptr()`` is preserved). Shape mismatches raise
    ``ValueError`` naming the offending dim.
    """
    geometry.validate()
    if w_o_a.dim() != 2:
        raise ValueError(f"w_o_a must be 2-D [o_groups * o_lora_rank, group_width], got {tuple(w_o_a.shape)}")
    rows, cols = int(w_o_a.shape[0]), int(w_o_a.shape[1])
    if rows != geometry.n_o_lora:
        raise ValueError(f"w_o_a must have {geometry.n_o_lora} rows (o_groups * o_lora_rank = {geometry.o_groups} * {geometry.o_lora_rank}), got {rows}")
    if cols != geometry.group_width:
        raise ValueError(
            f"w_o_a must have {geometry.group_width} columns (group_width = n_heads * head_dim // o_groups = "
            f"{geometry.n_heads} * {geometry.head_dim} // {geometry.o_groups}), got {cols}"
        )
    return w_o_a.view(geometry.o_groups, geometry.o_lora_rank, geometry.group_width)


def build_rope_tables(
    seq_len: int,
    rope_dim: int,
    theta: float,
    *,
    yarn: Optional[Union[YarnParams, Tuple[float, float, float, int]]] = None,
    device=None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Interleaved-pair RoPE tables ``(cos, sin)``, each fp32 ``[seq_len, rope_dim // 2]``.

    A twin of ``precompute_freqs_cis`` (``M:368-389``): the same fp32 ops in
    the same order, so ``cos + 1j * sin`` equals the model's ``freqs_cis``
    buffer bit for bit WHEN BUILT ON THE SAME DEVICE (``polar()`` rounds
    differently on CPU vs CUDA by ~1 ulp; the release builds its buffer on CUDA
    under ``torch.set_default_device("cuda")``, ``generate.py:134`` /
    ``M:1297`` -- pass ``device=`` accordingly). The ops: ``freqs = 1 / theta **
    (arange(0, rope_dim, 2) / rope_dim)``; optional YaRN blend; ``polar(1,
    outer(arange(seq_len), freqs))``. Column ``k`` is the angle of pair ``(2k,
    2k+1)`` of the rotated
    slice -- the interleaved / GPT-J pairing ``apply_rotary_emb`` uses
    (``M:397``) -- NOT a duplicated-halves rotate_half table. Per pair, at
    position ``t``::

        x'[2k]   = x[2k] * cos[t, k] - x[2k+1] * sin[t, k]
        x'[2k+1] = x[2k] * sin[t, k] + x[2k+1] * cos[t, k]

    and the inverse rotation (``M:781``, ``freqs_cis.conj()``, ``M:399``) is
    the same table with ``sin`` negated.

    Which table a layer uses is the CALLER's decision (``M:680-698``): the
    release rotates window-only layers with ``theta = rope_theta`` (10000) and
    NO YaRN, and every compressed layer with ``theta = compress_rope_theta``
    (160000) under YaRN ``(factor 16, beta_fast 32, beta_slow 1,
    original_seq_len 65536)``. The block takes the TABLE, never positions, so
    it need not know. Row ``t`` is position ``t``; slice ``[start_pos :
    start_pos + S]`` for a chunk (``M:767``).

    ``yarn`` is a :class:`YarnParams` or the tuple ``(factor, beta_fast,
    beta_slow, original_seq_len)``; ``None`` disables the correction (the
    model's ``original_seq_len == 0`` branch, ``M:378``).
    """
    import math

    import torch

    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive int, got {seq_len!r}")
    if not isinstance(rope_dim, int) or isinstance(rope_dim, bool) or rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be a positive even int (interleaved pairs), got {rope_dim!r}")
    if not theta > 0.0:
        raise ValueError(f"theta must be > 0, got {theta}")
    if yarn is not None and not isinstance(yarn, YarnParams):
        if len(yarn) != 4:
            raise ValueError(f"yarn must be YarnParams or (factor, beta_fast, beta_slow, original_seq_len), got {yarn!r}")
        yarn = YarnParams(*yarn)
    if yarn is not None:
        yarn.validate()

    # M:376 -- fp32 frequencies, one per pair.
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    if yarn is not None:
        # M:378-386 -- YaRN: dims whose wavelength fits the training context keep
        # their frequency, dims far beyond it are divided by `factor`, and the
        # beta_fast..beta_slow band is faded across with a linear ramp.
        def corrected_dim(rotations: float) -> float:
            return rope_dim * math.log(yarn.original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(corrected_dim(yarn.beta_fast)), 0)
        high = min(math.ceil(corrected_dim(yarn.beta_slow)), rope_dim - 1)
        ramp = ((torch.arange(rope_dim // 2, dtype=torch.float32, device=device) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / yarn.factor * (1 - smooth) + freqs * smooth

    # M:388-389 -- angles fp32 [seq_len, rope_dim // 2]; polar() is the model's
    # exact cos/sin evaluation, so real/imag reproduce freqs_cis bit for bit on
    # the same device (CPU and CUDA polar() differ by ~1 ulp from each other).
    angles = torch.outer(torch.arange(seq_len, device=device), freqs)
    table = torch.polar(torch.ones_like(angles), angles)
    return table.real.contiguous(), table.imag.contiguous()
