# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MQA sparse-attention block, forward -- the UNFUSED path, v0.

One attention layer of the form (op geometry; the shipped instance is
DeepSeek-V4.1-Flash, which supplies :class:`MqaSparseAttentionBlockGeometry`'s
defaults) as ELEVEN stream-ordered stages over ONE caller workspace::

    1  q_a_proj    x [T, d_model]        @ W_q_a^T        -> qa [T, q_lora]            FROST GEMM
    2  q_norm      RMSNorm(qa, w_q_norm)  in place                                      pointwise (torch v0 / norm_rope kernel)
    3  q_b_proj    qa                     @ W_q_b^T        -> q  [T, H*D]               FROST GEMM
    4  q_rope      interleaved RoPE on the LAST rope_dim dims of every head, in place   pointwise
    5  kv_proj     x [B, S, d_model]      @ W_kv^T         -> kv_all[:, :S, :]  (D4)    FROST GEMM, batched, prefix-strided C
    6  kv_chain    RMSNorm -> RoPE -> block-32 ue8m0 fake-quant, in place on kv_all[:, :S]   pointwise
    6b (dsa) index staging when B > 1 or K % 64 != 0 (D5)                               torch out= ops into the workspace
    7  attention   q, kv_all, attn_sink, topk_idxs        -> o [T, H, D] (+ lse_th)     torch | dsa (in-tree H64) | d512 (fork, wave B)
    8  o_unrope    inverse RoPE in place on o; LSE fold (D1, dsa only)                  pointwise
    9  o_a_proj    8 x  o[:, g*4096:(g+1)*4096] @ W_o_a[g]^T -> o_lora[:, g*1024:(g+1)*1024]   FROST GEMM, declared row strides (D3)
    10 o_b_proj    o_lora [T, 8192]       @ W_o_b^T        -> out [T, d_model]          FROST GEMM

``T = B*S``.  Stages 1/3/5/9/10 drive the shipped FROST GEMM through
``gated_attention_block.kernels.proj_gemm`` (extended append-only with row /
batch strides so 5 writes the ``[:, :S]`` prefix of the caller's ``kv_all``
directly and 9 runs eight launches on column slices -- no repack anywhere).

**FLAGGED (temporary, v0 = W1): stages 2 / 4 / 6 / 8 run in torch and ALLOCATE
per execute**, as does the ``torch`` attention adapter.  ``pointwise_impl="frost"``
selects the single-tensor norm/RoPE/fake-quant kernel (``kernels/norm_rope.py``,
wave W2) by the agreed interface, imported lazily inside the stage; until it
lands that arm is a typed decline.  The ``dsa`` adapter is contract-clean (no
allocation, no sync on its execute path); ``d512`` declines until wave B.

Conventions (``geometry.py``): K == V, one ``[B, S + N_c, D]`` tensor; the sink
enters the denominator once; ``-1`` = no key, duplicates count twice; compressed
ids already carry the ``+S`` offset; RoPE is interleaved (GPT-J) on the LAST
``rope_dim`` dims; the returned ``lse [B, H, S]`` is the FROST convention (sink
INCLUDED) on every adapter (D7).  ``N_c == 0`` under ``has_compressed_kv=True`` is
the legal ``S < ratio`` degenerate (then ``K == min(S, window)``).

``execute`` VALIDATES every runtime tensor against the declaration and every
execute-time contract (alignment, workspace, ``lse``) before the first launch, so a
decline never leaves a half-run behind; ``stage_runners`` hands the same pipeline
out as per-stage launch units (``RUNNER_NAMES``) for the perf table's per-op rows.

The block sits OUTSIDE ``graph.plans`` for v0 (no engine row); it is a
frontend-only API in the ``gated_attention_block`` mould.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase
from cudnn.frost.workspace import WorkspaceLayout, align_up

from .geometry import KV_FAKE_QUANT_BLOCK, MqaSparseAttentionBlockGeometry

_SM107_CC = (10, 7)
_WS_ALIGN = 256

ATTENTION_IMPLS = ("torch", "dsa", "d512")
POINTWISE_IMPLS = ("torch", "frost")

# Fake-quant constants of the reference (K:44-46, K:76): an fp32 ``1/448`` MULTIPLY (not a divide --
# the two differ at power-of-two boundaries) and the amax floor.
_FP8_MAX = 448.0
# A Python float, NOT a CPU tensor: torch casts a wrapped scalar to the operand's fp32 before the multiply, so
# ``amax * _FP8_MAX_INV`` IS the fp32 ``amax * fp32(1/448)`` product (bitwise vs the tensor constant on CUDA and
# CPU: 20 x 2^20 draws over 12 decades + the 1e-4 / 448 / 896 / 3.4e38 corners, A100 2026-09-17) -- and unlike
# ``tensor.to(amax.device)`` it costs no pageable H2D copy, which is a SYNCHRONIZING op
# (``torch.cuda.set_sync_debug_mode('error')`` fired inside every kv_chain execute, Rubin 2026-09-17).
_FP8_MAX_INV = 1.0 / _FP8_MAX
_FP8_AMAX_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# Workspace views + stream scope
# ---------------------------------------------------------------------------


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _view(ws: torch.Tensor, offset: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """A typed, shaped VIEW of the caller's uint8 workspace at ``offset`` -- never a copy."""
    n = 1
    for x in shape:
        n *= int(x)
    nbytes = n * _itemsize(dtype)
    return ws[offset : offset + nbytes].view(dtype).view(*shape)


@contextmanager
def _torch_stream(stream: int, device: torch.device):
    """Order torch-issued work on the block's launch stream (a raw ``CUstream`` int)."""
    cur = torch.cuda.current_stream(device)
    if int(stream) == cur.cuda_stream:
        yield
        return
    ext = torch.cuda.get_stream_from_external(int(stream), device.index)
    with torch.cuda.stream(ext):
        yield


# ---------------------------------------------------------------------------
# Runtime-tensor validation + the per-stage runner (execute and stage_runners share both)
# ---------------------------------------------------------------------------

# execute() / stage_runners() operand order == the constructor's sample order.
TENSOR_NAMES = ("x", "w_q_a", "w_q_norm", "w_q_b", "w_kv", "w_kv_norm", "w_o_a", "w_o_b", "attn_sink", "cos", "sin", "kv_all", "topk_idxs", "out")


def _check_bound_tensors(descs: dict, tensors: dict, device: torch.device) -> None:
    """Every runtime tensor against its DECLARED descriptor: shape, dtype, contiguity and
    device, host-only, BEFORE anything is launched.  A typed ``ValueError`` naming the tensor.

    Nothing downstream re-checks these: the graph-route GEMM binds a pointer under the
    declared layout (``proj_gemm.run_proj_gemm`` validates shape/strides but not dtype or
    device), a torch stage would broadcast or die in ``.view()`` with a ``RuntimeError``,
    and a stage's ``torch.cuda.stream`` scope follows the DECLARED device -- so a tensor on
    another device would be ordered on the wrong stream.  The 14 descriptors already exist;
    this loop is the whole cost."""
    for name, t in tensors.items():
        d = descs[name]
        want = tuple(int(v) for v in d.shape)
        if tuple(int(v) for v in t.shape) != want:
            raise ValueError(f"{name} must have the declared shape {want}, got {tuple(t.shape)}")
        if t.dtype != d.dtype:
            raise ValueError(f"{name} must have the declared dtype {d.dtype}, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (declared contiguous; the block binds it as is, never repacks), got strides {tuple(t.stride())}")
        if t.device != device:
            raise ValueError(f"{name} must live on the declared device {device}, got {t.device}")


class _StageRunner:
    """One launch unit of the block, as :meth:`MqaSparseAttentionBlockFwd.stage_runners` hands it out.

    ``name`` is the pipeline name (``RUNNER_NAMES``), ``kind`` one of ``'mma'`` / ``'bw'`` /
    ``'prep'``, ``run()`` launches on the stream the runner was bound with (no allocation on
    the contract-clean stages), ``prepare`` is ``None`` or a callable that RESTORES the
    in-place input so a timing loop can call ``run()`` repeatedly on the same bytes.  The
    restore re-runs the PRODUCING stage (the GEMM / attention that wrote the buffer):
    bit-exact, no snapshot, no allocation -- the producer's inputs are caller tensors (or
    workspace state the pipeline order already fixed).
    """

    __slots__ = ("name", "kind", "run", "prepare")

    def __init__(self, name: str, kind: str, run, prepare=None) -> None:
        self.name, self.kind, self.run, self.prepare = name, kind, run, prepare

    def __repr__(self) -> str:
        return f"_StageRunner({self.name!r}, kind={self.kind!r}, prepare={'yes' if self.prepare else 'no'})"


# name -> kind of every runner the block can emit, in pipeline order (idx_staging only under dsa staging).
RUNNER_NAMES = (
    ("q_a_proj", "mma"),
    ("q_norm", "bw"),
    ("q_b_proj", "mma"),
    ("q_rope", "bw"),
    ("kv_proj", "mma"),
    ("kv_chain", "bw"),
    ("idx_staging", "prep"),
    ("sparse_attention", "mma"),
    ("o_unrope", "bw"),
    ("o_a_proj", "mma"),
    ("o_b_proj", "mma"),
)


# ---------------------------------------------------------------------------
# Torch pointwise math (the model's op order and bf16 rounding points; TEMPORARY)
# ---------------------------------------------------------------------------


def _rmsnorm_(x2: torch.Tensor, w: torch.Tensor, eps: float) -> None:
    """``RMSNorm.forward`` (M:289-293) in place on ``[R, d]``: fp32 variance, weight promoted, ONE rounding."""
    xf = x2.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    x2.copy_(w * xf)


def _rope_(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> None:
    """``apply_rotary_emb`` (M:392-406) in place on the rotated slice ``[T, heads, rope_dim]``
    (heads may be 1): adjacent pairs as complex, fp32 multiply, ONE rounding on write.
    ``cos`` / ``sin`` are ``[T, rope_dim // 2]`` fp32."""
    t, heads, rope_dim = (int(v) for v in x_rot.shape)
    fc = torch.complex(cos, sin)
    if inverse:
        fc = fc.conj()
    fc = fc.view(t, 1, rope_dim // 2)
    xc = torch.view_as_complex(x_rot.float().unflatten(-1, (-1, 2)).contiguous())
    x_rot.copy_(torch.view_as_real(xc * fc).flatten(-2))


def _fake_quant_(x2: torch.Tensor, block: int) -> None:
    """``act_quant(x, block, "ue8m0", e8m0, inplace=True)`` (K:41-124) in place on ``[R, d]``:
    per block ``amax = max(|x|, 1e-4)``; ``s = 2**ceil(log2(amax * fp32(1/448)))``;
    ``y = bf16(fp32(e4m3(clamp(x/s))) * s)``.  ``ceil(log2)`` via ``frexp`` (no ``exp2`` /
    ``log2``: those are jiterator ops that fail on cc 10.7)."""
    xb = x2.float().unflatten(-1, (-1, block))
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(_FP8_AMAX_FLOOR)
    m, e = torch.frexp(amax * _FP8_MAX_INV)  # fp32 multiply by fp32(1/448); see the constant
    k = e - (m == 0.5).to(e.dtype)  # x is a power of two iff its frexp mantissa is exactly 0.5
    s = torch.ldexp(torch.ones_like(amax), k)
    q = (xb / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float()
    x2.copy_((q * s).flatten(-2))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class _Stage(ABC):
    """One kernel of the block -- the same three-phase shape as ``APIBase``
    (support / compile / execute) so a stage can be promoted or absorbed without
    the block's API moving (mirror of ``gated_attention_block.api._Stage``)."""

    name: str

    @abstractmethod
    def check_support(self) -> None:
        """Raise ``NotImplementedError`` / ``ValueError`` if this stage cannot serve
        the declaration.  Never degrade silently, never adapt."""

    @abstractmethod
    def compile(self) -> None:
        """Build the artifact.  Plan-time keys only."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> None:
        """Launch on the block's stream.  Contract-clean stages allocate nothing."""


class _Projection(_Stage):
    """Stages 1 / 3 / 5 / 9 / 10 -- ONE dense-GEMM stage class at five shapes.

    Drives the shipped FROST GEMM through ``gated_attention_block.kernels.proj_gemm``
    (pinned ``frost_gemm`` plan; the forced 256-wide N tile where 256 divides N --
    every N of this block: 1280, 32768, 512, 1024, 5120).  Strided / batched forms
    (D3, D4, M6) are DECLARED at plan time and the runner refuses a bound view
    that disagrees.
    """

    def __init__(
        self,
        *,
        m: int,
        k: int,
        n: int,
        dtype: torch.dtype,
        name: str,
        a_row_stride: Optional[int] = None,
        c_row_stride: Optional[int] = None,
        batch: int = 1,
        a_batch_stride: Optional[int] = None,
        b_batch_stride: Optional[int] = None,
        c_batch_stride: Optional[int] = None,
    ) -> None:
        self.name = name
        self.m, self.k, self.n, self.batch = int(m), int(k), int(n), int(batch)
        self.dtype = dtype
        self.strides = dict(
            a_row_stride=a_row_stride, c_row_stride=c_row_stride, a_batch_stride=a_batch_stride, b_batch_stride=b_batch_stride, c_batch_stride=c_batch_stride
        )
        self._plan = None

    def check_support(self) -> None:
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"{self.name}: bf16 / f16 only, got {self.dtype}")

    def compile(self) -> None:
        from cudnn.gated_attention_block.kernels.proj_gemm import build_proj_gemm

        self._plan = build_proj_gemm(m=self.m, k=self.k, n=self.n, dtype=self.dtype, label=self.name, batch=self.batch, **self.strides)

    @property
    def plan(self):
        return self._plan

    def workspace_bytes(self) -> int:
        if self._plan is None:
            raise RuntimeError(f"{self.name}: call compile() before workspace_bytes()")
        return int(self._plan.workspace_bytes)

    def flops(self) -> int:
        return 2 * self.batch * self.m * self.n * self.k

    def execute(self, a: torch.Tensor, w: torch.Tensor, out: torch.Tensor, workspace: torch.Tensor, *, stream: int) -> None:
        from cudnn.gated_attention_block.kernels.proj_gemm import run_proj_gemm

        if self._plan is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        run_proj_gemm(self._plan, a, w, out, workspace, stream=stream)


class _NormRope(_Stage):
    """Stages 2 / 4 / 6 / 8 -- ONE row-local pointwise recipe at four settings.

    Row space: ``x`` is ``[T, heads, d]`` (``heads = 1`` for the ``[T, d]`` tensors);
    RoPE rotates the LAST ``rope_dim`` dims of every row as interleaved pairs with
    ``cos / sin [T, rope_dim // 2]`` (row ``(t, h)`` takes table row ``t``); the
    fake-quant is per contiguous 32-block over the whole row (the model quantizes
    the RoPE tail too, M:707); ``rope_inverse`` conjugates; ``lse_fold`` turns the
    H64 kernel's ``lse_in [T, heads]`` + ``sink [heads]`` into the FROST
    ``lse_out [B, heads, S]`` (D1 select).  Op order inside one row: norm -> RoPE ->
    fake-quant, with the model's THREE bf16 roundings.

    **``impl="torch"`` (v0 default) is FLAGGED temporary: it allocates per execute.**
    ``impl="frost"`` imports ``kernels/norm_rope.py`` lazily by the agreed interface
    (``NormRopeRecipe`` / ``build_norm_rope(recipe, n_rows_max=, device=)`` ->
    ``.run(x, w, cos, sin, *, lse_in=, sink=, lse_out=, stream=)`` in place on ``x``
    viewed ``[T*heads, d]``); a checkout without it is a typed decline.
    """

    def __init__(
        self,
        *,
        name: str,
        d: int,
        heads: int,
        apply_norm: bool,
        rope_dim: int,
        eps: float,
        impl: str,
        n_tokens: int,
        batch: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        rope_inverse: bool = False,
        fake_quant: bool = False,
        lse_fold: bool = False,
        three_roundings: bool = True,
    ) -> None:
        self.name = name
        self.d, self.heads, self.rope_dim = int(d), int(heads), int(rope_dim)
        self.apply_norm, self.rope_inverse, self.fake_quant, self.lse_fold = bool(apply_norm), bool(rope_inverse), bool(fake_quant), bool(lse_fold)
        self.eps, self.impl, self.dtype, self.device = float(eps), impl, dtype, torch.device(device)
        self.n_tokens, self.batch, self.seq_len = int(n_tokens), int(batch), int(seq_len)
        self.three_roundings = bool(three_roundings)
        self._kernel = None

    def check_support(self) -> None:
        if self.impl not in POINTWISE_IMPLS:
            raise ValueError(f"{self.name}: pointwise_impl must be one of {POINTWISE_IMPLS}, got {self.impl!r}")
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"{self.name}: bf16 only (v0), got {self.dtype}")
        if self.rope_dim < 0 or self.rope_dim % 2 or self.rope_dim > self.d:
            raise ValueError(f"{self.name}: rope_dim must be even and in [0, d={self.d}] (0 = no rotation), got {self.rope_dim}")
        if not (self.apply_norm or self.rope_dim or self.fake_quant or self.lse_fold):
            raise ValueError(f"{self.name}: a pointwise stage with nothing to do (no norm, no RoPE, no fake-quant, no LSE fold)")
        if self.fake_quant and self.d % KV_FAKE_QUANT_BLOCK:
            raise ValueError(f"{self.name}: fake_quant needs d % {KV_FAKE_QUANT_BLOCK} == 0, got d={self.d}")
        if self.impl == "torch":
            if not self.three_roundings:
                raise NotImplementedError(
                    f"{self.name}: three_roundings=False exists only on the norm_rope kernel (pointwise_impl='frost'); the torch arm is the model's arithmetic"
                )
            return
        try:
            from .kernels import norm_rope  # noqa: F401  -- wave W2; imported by the agreed interface
        except ImportError as exc:
            raise NotImplementedError(
                f"{self.name}: pointwise_impl='frost' needs kernels/norm_rope.py, which has not landed ({exc}); use pointwise_impl='torch'"
            ) from exc
        norm_rope.validate_shape(self._recipe())

    def _recipe(self):
        from .kernels.norm_rope import NormRopeRecipe

        return NormRopeRecipe(
            d=self.d,
            heads=self.heads,
            apply_norm=self.apply_norm,
            rope_dim=self.rope_dim,
            rope_inverse=self.rope_inverse,
            fake_quant=self.fake_quant,
            lse_fold=self.lse_fold,
            three_roundings=self.three_roundings,
            io_dtype=self.dtype,
        )

    def compile(self) -> None:
        if self.impl != "frost":
            return
        from .kernels.norm_rope import build_norm_rope

        # The kernel is shape-agnostic (symbolic extents); n_rows_max is the block's declared bound on B * S * heads.
        self._kernel = build_norm_rope(self._recipe(), n_rows_max=self.n_tokens * self.heads, device=self.device, eps=self.eps)

    def moved_bytes(self) -> int:
        return 2 * self.n_tokens * self.heads * self.d * _itemsize(self.dtype)

    def execute(
        self,
        x: torch.Tensor,  # [T, heads, d] (or [B, S, d] when heads == 1 -- a batch-strided prefix of kv_all is fine)
        w: Optional[torch.Tensor],  # [d] iff apply_norm
        cos: Optional[torch.Tensor],  # [T, rope_dim // 2] fp32 iff rope_dim > 0 (always here)
        sin: Optional[torch.Tensor],
        *,
        lse_in: Optional[torch.Tensor] = None,  # [T, heads] fp32 (the H64 LSE) iff lse_fold
        sink: Optional[torch.Tensor] = None,  # [heads] fp32 iff lse_fold
        lse_out: Optional[torch.Tensor] = None,  # [B, heads, S] fp32 iff lse_fold
        stream: int,
    ) -> None:
        if self.apply_norm and w is None:
            raise ValueError(f"{self.name}: apply_norm needs the norm weight")
        if self.lse_fold and (lse_in is None or sink is None or lse_out is None):
            raise ValueError(f"{self.name}: lse_fold needs lse_in, sink and lse_out")
        if self.impl == "frost":
            self._run_kernel(x, w, cos, sin, lse_in, sink, lse_out, stream)
            return
        with _torch_stream(stream, self.device):
            self._run_torch(x, w, cos, sin, lse_in, sink, lse_out)

    def _rows(self, x: torch.Tensor):
        """``x`` as ``[T, heads, d]`` row-major over (token, head); a ``[B, S, d]`` view keeps its batch stride."""
        if x.ndim == 3 and int(x.shape[1]) == self.heads and int(x.shape[2]) == self.d and int(x.shape[0]) == self.n_tokens:
            return x
        if x.ndim == 3 and self.heads == 1 and tuple(int(v) for v in x.shape) == (self.batch, self.seq_len, self.d):
            return x  # [B, S, d]: per-token rows, possibly batch-strided (the kv_all prefix)
        if x.ndim == 2 and tuple(int(v) for v in x.shape) == (self.n_tokens * self.heads, self.d):
            return x.view(self.n_tokens, self.heads, self.d)
        raise ValueError(f"{self.name}: x must be [T={self.n_tokens}, heads={self.heads}, d={self.d}] (or [B, S, d] at heads=1), got {tuple(x.shape)}")

    def _row_blocks(self, x: torch.Tensor):
        """``[(first_token, rows [n_tok * heads, d])]`` -- one block for a contiguous ``x``; the
        batch-strided ``[B, S, d]`` prefix of ``kv_all`` (``N_c > 0``, ``heads == 1``) is not viewable
        as ``[T, d]``, so it is walked per batch entry over its contiguous ``[S, d]`` rows."""
        x3 = self._rows(x)
        if x3.is_contiguous():
            return [(0, x3.view(-1, self.d))]
        return [(b_ * self.seq_len, x3[b_].view(-1, self.d)) for b_ in range(int(x3.shape[0]))]

    def _run_torch(self, x, w, cos, sin, lse_in, sink, lse_out) -> None:
        cos_t = cos.view(self.n_tokens, self.rope_dim // 2) if self.rope_dim else None
        sin_t = sin.view(self.n_tokens, self.rope_dim // 2) if self.rope_dim else None
        for row0, rows in self._row_blocks(x):
            n_tok = int(rows.shape[0]) // self.heads
            if self.apply_norm:
                _rmsnorm_(rows, w, self.eps)
            if self.rope_dim:
                _rope_(rows.view(n_tok, self.heads, self.d)[..., -self.rope_dim :], cos_t[row0 : row0 + n_tok], sin_t[row0 : row0 + n_tok], self.rope_inverse)
            if self.fake_quant:
                _fake_quant_(rows, KV_FAKE_QUANT_BLOCK)
        if self.lse_fold:
            from .kernels.sparse_attention import dsa_lse_fold

            folded = dsa_lse_fold(lse_in.view(self.n_tokens, self.heads), sink)  # [T, heads]
            lse_out.copy_(folded.view(self.batch, self.seq_len, self.heads).permute(0, 2, 1))

    def _run_kernel(self, x, w, cos, sin, lse_in, sink, lse_out, stream) -> None:
        """The ``kernels/norm_rope.py`` arm: ONE launch, in place.  ``NormRopeKernel.run`` takes ``x`` as
        ``[T, d]`` / ``[B, S, d]`` (heads == 1; the batch stride is free, so the ``kv_all[:, :S]`` prefix binds
        as is) or ``[B, S, heads, d]``, the tables as ``[B, S, rope_dim // 2]``, ``lse_in [B, S, heads]`` and
        ``lse_out [B, heads, S]`` -- and REFUSES operands the recipe does not use (``w`` without a norm, tables
        at ``rope_dim == 0``, LSE operands without the fold), so only what applies is passed."""
        if self._kernel is None:
            raise RuntimeError(f"{self.name}: call compile() before execute()")
        b, s, heads, d = self.batch, self.seq_len, self.heads, self.d
        x3 = self._rows(x)  # [T, heads, d] or [B, S, d]
        if heads == 1:
            x_k = x3 if tuple(int(v) for v in x3.shape) == (b, s, d) else x3.view(b * s, d)  # [B, S, d] (prefix) or [T, d]
        else:
            x_k = x3.view(b, s, heads, d)
        cos_k = cos.view(b, s, self.rope_dim // 2) if self.rope_dim else None
        sin_k = sin.view(b, s, self.rope_dim // 2) if self.rope_dim else None
        fold = self.lse_fold
        self._kernel.run(
            x_k,
            w if self.apply_norm else None,
            cos_k,
            sin_k,
            lse_in=lse_in.view(b, s, heads) if fold else None,
            sink=sink if fold else None,
            lse_out=lse_out if fold else None,
            stream=int(stream),
        )


class _SparseAttention(_Stage):
    """Stage 7 behind one of the three adapters of ``kernels/sparse_attention.py``."""

    def __init__(
        self, *, impl: str, geom: MqaSparseAttentionBlockGeometry, batch: int, seq_len: int, n_kv_rows: int, topk: int, dtype, device, want_lse: bool
    ) -> None:
        self.name = "sparse_attention"
        self.impl, self.geom = impl, geom
        self.batch, self.seq_len, self.n_kv_rows, self.topk = int(batch), int(seq_len), int(n_kv_rows), int(topk)
        self.dtype, self.device, self.want_lse = dtype, torch.device(device), bool(want_lse)
        self._adapter = None

    @property
    def needs_idx_staging(self) -> bool:
        from .kernels.sparse_attention import dsa_needs_index_staging

        return self.impl == "dsa" and dsa_needs_index_staging(self.batch, self.topk)

    @property
    def k_pad(self) -> int:
        from .kernels.sparse_attention import dsa_topk_padded

        return dsa_topk_padded(self.topk) if self.impl == "dsa" else self.topk

    def check_support(self) -> None:
        from .kernels import sparse_attention as sa

        if self.impl not in ATTENTION_IMPLS:
            raise ValueError(f"attention must be one of {ATTENTION_IMPLS}, got {self.impl!r}")
        g = self.geom
        common = dict(
            batch=self.batch,
            seq_len=self.seq_len,
            n_kv_rows=self.n_kv_rows,
            n_heads=g.n_heads,
            head_dim=g.head_dim,
            topk=self.topk,
            scale=g.scale,
            dtype=self.dtype,
            device=self.device,
        )
        if self.impl == "torch":
            if self.device.type != "cuda":
                raise ValueError(f"{self.name}: tensors must live on CUDA, got {self.device}")
            return
        if self.impl == "dsa":
            self._adapter = sa.DsaSparseAttention(**common)
        else:
            self._adapter = sa.D512SparseAttention(want_lse=self.want_lse, **common)
        self._adapter.check_support()

    def compile(self) -> None:
        if self._adapter is not None:
            self._adapter.compile()

    def stage_indices(self, topk_idxs: torch.Tensor, idx_tmp: Optional[torch.Tensor], idx_ws: torch.Tensor, *, stream: int) -> None:
        """6b (dsa, ``needs_idx_staging`` only): the D5 index pre-pass into the workspace --
        torch ``out=`` ops on the block's stream, no allocation."""
        if not self.needs_idx_staging:
            raise RuntimeError(f"{self.name}: stage_indices applies only to the dsa adapter's staging path (B > 1 or K % 64 != 0)")
        with _torch_stream(stream, self.device):
            self._adapter.stage_indices(topk_idxs, idx_tmp, idx_ws)

    def launch(
        self,
        q: torch.Tensor,  # [B, S, H, D] (workspace view)
        kv_all: torch.Tensor,  # [B, N, D] caller's, contiguous
        sink: torch.Tensor,  # [H] fp32
        topk_idxs: torch.Tensor,  # [B, S, K] int32 caller's (bound directly on the identity path)
        o: torch.Tensor,  # [B, S, H, D] (workspace view)
        *,
        lse: Optional[torch.Tensor],  # torch adapter: [B, H, S] written directly (None = not wanted)
        lse_th: Optional[torch.Tensor],  # dsa: [T, H] the kernel's LSE (sink excluded)
        max_logits: Optional[torch.Tensor],  # dsa: [T, H] dead output
        idx_ws: Optional[torch.Tensor],  # dsa staging target [B, S, K_pad], already staged when the path needs it
        stream: int,
    ) -> None:
        """7: the attention kernel ALONE (the ``8a kernel-only`` unit of the perf table)."""
        from .kernels import sparse_attention as sa

        b, s, h, d = (int(v) for v in q.shape)
        t = b * s
        if self.impl == "torch":
            with _torch_stream(stream, self.device):
                sa.torch_sparse_attention(q, kv_all, sink, topk_idxs, self.geom.scale, out=o, lse=lse)
            return
        if self.impl != "dsa":
            raise NotImplementedError("attention='d512' has not landed")
        ad = self._adapter
        if ad.needs_idx_staging:
            idx2d = idx_ws.view(t, ad.k_pad)
        else:
            sa.check_dsa_index_alignment(topk_idxs)  # the block checks this in its preamble too; the stage keeps its own contract
            idx2d = topk_idxs.view(t, self.topk)
        ad.execute(q.view(t, h, d), kv_all.view(b * self.n_kv_rows, d), idx2d, sink, o.view(t, h, d), lse_th, max_logits, stream=stream)

    def execute(
        self,
        q: torch.Tensor,
        kv_all: torch.Tensor,
        sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        o: torch.Tensor,
        *,
        lse: Optional[torch.Tensor],
        lse_th: Optional[torch.Tensor],
        max_logits: Optional[torch.Tensor],
        idx_tmp: Optional[torch.Tensor],  # dsa staging scratch [B, S, K] (B > 1)
        idx_ws: Optional[torch.Tensor],  # dsa staging target [B, S, K_pad]
        stream: int,
    ) -> None:
        """6b + 7: stage the indices when the path needs it, then launch."""
        if self.needs_idx_staging:
            self.stage_indices(topk_idxs, idx_tmp, idx_ws, stream=stream)
        self.launch(q, kv_all, sink, topk_idxs, o, lse=lse, lse_th=lse_th, max_logits=max_logits, idx_ws=idx_ws, stream=stream)


# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Slots:
    """Byte offsets of every intermediate in the caller's workspace (``WorkspaceLayout(align=256)``, stage order)."""

    qa: int
    q: int
    o: int
    o_lora: int
    lse_th: Optional[int]  # dsa only
    max_logits: Optional[int]  # dsa only (required dead output of the H64 kernel)
    idx_tmp: Optional[int]  # dsa, B > 1 (staging scratch)
    idx_ws: Optional[int]  # dsa, staging
    engine_scratch: int
    total_bytes: int


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


class MqaSparseAttentionBlockFwd(APIBase):
    """MQA sparse-attention block, forward: one call, one workspace, 11 stages.

    Sample tensors fix the DECLARATION (shapes, dtypes, device); the runtime
    tensors handed to :meth:`execute` must match it.  Weights are ``nn.Linear``
    ``[out, in]``; ``w_o_a`` is the grouped ``[o_groups, o_lora_rank, group_width]``
    VIEW (:func:`~cudnn.mqa_sparse_attention_block.build_grouped_wo_a`).

    Knobs: ``attention`` (``"dsa"`` default = the in-tree H64 kernel with the sink
    passed in and the LSE folded, ``"torch"`` = the oracle-shaped baseline,
    ``"d512"`` = the Rubin fork -- declines until it lands); ``wo_a_batched``
    (one batched ``o_a_proj`` launch instead of eight -- gated on M6, off by
    default); ``pointwise_impl`` (``"torch"`` default in v0 -- FLAGGED, allocates
    per execute; ``"frost"`` = ``kernels/norm_rope.py`` once it lands);
    ``three_roundings`` (the model's bf16 rounding points; ``False`` exists only
    on the kernel arm); ``return_lse`` (then ``execute(lse=)`` is a required
    ``[B, n_heads, S]`` fp32 tensor in the FROST convention).
    """

    def __init__(
        self,
        sample_x: torch.Tensor,  # [B, S, d_model] bf16
        sample_w_q_a: torch.Tensor,  # [q_lora_rank, d_model]
        sample_w_q_norm: torch.Tensor,  # [q_lora_rank]
        sample_w_q_b: torch.Tensor,  # [n_heads * head_dim, q_lora_rank]
        sample_w_kv: torch.Tensor,  # [head_dim, d_model]
        sample_w_kv_norm: torch.Tensor,  # [head_dim]
        sample_w_o_a: torch.Tensor,  # [o_groups, o_lora_rank, group_width]  (build_grouped_wo_a view)
        sample_w_o_b: torch.Tensor,  # [d_model, o_groups * o_lora_rank]
        sample_attn_sink: torch.Tensor,  # [n_heads] fp32
        sample_cos: torch.Tensor,  # [B, S, rope_dim // 2] fp32 interleaved-pair table
        sample_sin: torch.Tensor,  # [B, S, rope_dim // 2] fp32
        sample_kv_all: torch.Tensor,  # [B, S + N_c, head_dim]  -- rows [:, :S] are WRITTEN by the block, [:, S:] are the caller's compressed rows
        sample_topk_idxs: torch.Tensor,  # [B, S, K] int32, -1 = no key, compressed ids offset by S
        sample_out: torch.Tensor,  # [B, S, d_model]
        geometry: MqaSparseAttentionBlockGeometry,
        *,
        return_lse: bool = False,
        attention: str = "dsa",
        wo_a_batched: bool = False,
        pointwise_impl: str = "torch",
        three_roundings: bool = True,
    ):
        super().__init__()
        self._warn_experimental_api()
        if not isinstance(geometry, MqaSparseAttentionBlockGeometry):
            raise TypeError(f"geometry must be an MqaSparseAttentionBlockGeometry, got {type(geometry).__name__}")
        if attention not in ATTENTION_IMPLS:
            raise ValueError(f"attention must be one of {ATTENTION_IMPLS}, got {attention!r}")
        if pointwise_impl not in POINTWISE_IMPLS:
            raise ValueError(f"pointwise_impl must be one of {POINTWISE_IMPLS}, got {pointwise_impl!r}")
        self.geom = geometry
        self.attention = attention
        self.pointwise_impl = pointwise_impl
        self.wo_a_batched = bool(wo_a_batched)
        self.three_roundings = bool(three_roundings)
        self.return_lse = bool(return_lse)
        if sample_x.ndim != 3:
            raise ValueError(f"sample_x must be [B, S, d_model], got {tuple(sample_x.shape)}")
        self.dtype = sample_x.dtype
        self.device = sample_x.device
        self.batch, self.seq_len, self.d_model = (int(v) for v in sample_x.shape)
        self.tokens = self.batch * self.seq_len
        self._descs = {
            name: self._make_tensor_desc(t, name=name)
            for name, t in (
                ("x", sample_x),
                ("w_q_a", sample_w_q_a),
                ("w_q_norm", sample_w_q_norm),
                ("w_q_b", sample_w_q_b),
                ("w_kv", sample_w_kv),
                ("w_kv_norm", sample_w_kv_norm),
                ("w_o_a", sample_w_o_a),
                ("w_o_b", sample_w_o_b),
                ("attn_sink", sample_attn_sink),
                ("cos", sample_cos),
                ("sin", sample_sin),
                ("kv_all", sample_kv_all),
                ("topk_idxs", sample_topk_idxs),
                ("out", sample_out),
            )
        }
        for name, d in self._descs.items():
            if d is None:
                raise ValueError(f"sample_{name} is required (None given)")
        self.n_kv_rows = int(self._descs["kv_all"].shape[1]) if self._descs["kv_all"].ndim == 3 else -1
        self.topk = int(self._descs["topk_idxs"].shape[2]) if self._descs["topk_idxs"].ndim == 3 else -1
        self._stages: tuple = ()
        self._slots: Optional[_Slots] = None
        self._compiled = False

    # -- support ------------------------------------------------------------

    @staticmethod
    def _contiguous(desc) -> bool:
        expect, ok = 1, True
        for size, stride in zip(reversed(desc.shape), reversed(desc.stride)):
            if int(size) != 1 and int(stride) != expect:
                ok = False
            expect *= int(size)
        return ok

    def check_support(self) -> bool:
        """Validate the declaration (typed, in this order: geometry, dtypes, shapes,
        layouts, the KV / index contract, the arch, then every stage), then build
        the stages.  Everything accepted here is served natively by execute."""
        g = self.geom
        g.validate()
        d = self._descs
        b, s, t = self.batch, self.seq_len, self.tokens
        if self.d_model != g.d_model:
            raise ValueError(f"sample_x last dim {self.d_model} != geometry.d_model {g.d_model}")
        if self.dtype != torch.bfloat16:
            raise NotImplementedError(f"mqa_sparse_attention_block v0 is bf16 only, got x.dtype={self.dtype}")
        for nm in ("w_q_a", "w_q_norm", "w_q_b", "w_kv", "w_kv_norm", "w_o_a", "w_o_b", "kv_all", "out"):
            if d[nm].dtype != self.dtype:
                raise ValueError(f"{nm} must have x's dtype {self.dtype}, got {d[nm].dtype}")
        for nm in ("attn_sink", "cos", "sin"):
            if d[nm].dtype != torch.float32:
                raise ValueError(f"{nm} must be fp32, got {d[nm].dtype}")
        if d["topk_idxs"].dtype != torch.int32:
            raise ValueError(f"topk_idxs must be int32, got {d['topk_idxs'].dtype}")
        pairs = g.rope_dim // 2
        self._check_tensor_shape(d["w_q_a"], (g.q_lora_rank, g.d_model), "w_q_a")
        self._check_tensor_shape(d["w_q_norm"], (g.q_lora_rank,), "w_q_norm")
        self._check_tensor_shape(d["w_q_b"], (g.n_heads * g.head_dim, g.q_lora_rank), "w_q_b")
        self._check_tensor_shape(d["w_kv"], (g.head_dim, g.d_model), "w_kv")
        self._check_tensor_shape(d["w_kv_norm"], (g.head_dim,), "w_kv_norm")
        self._check_tensor_shape(d["w_o_a"], (g.o_groups, g.o_lora_rank, g.group_width), "w_o_a")
        self._check_tensor_shape(d["w_o_b"], (g.d_model, g.n_o_lora), "w_o_b")
        self._check_tensor_shape(d["attn_sink"], (g.n_heads,), "attn_sink")
        self._check_tensor_shape(d["cos"], (b, s, pairs), "cos")
        self._check_tensor_shape(d["sin"], (b, s, pairs), "sin")
        self._check_tensor_shape(d["out"], (b, s, g.d_model), "out")
        if d["kv_all"].ndim != 3 or int(d["kv_all"].shape[0]) != b or int(d["kv_all"].shape[2]) != g.head_dim or int(d["kv_all"].shape[1]) < s:
            raise ValueError(f"kv_all must be [B={b}, S + N_c >= {s}, head_dim={g.head_dim}], got {tuple(d['kv_all'].shape)}")
        if d["topk_idxs"].ndim != 3 or tuple(int(v) for v in d["topk_idxs"].shape[:2]) != (b, s):
            raise ValueError(f"topk_idxs must be [B={b}, S={s}, K], got {tuple(d['topk_idxs'].shape)}")
        k = self.topk
        if k < 1 or k > g.n_kv_slots_max:
            raise ValueError(f"topk_idxs lists K={k} slots per query; the geometry allows 1 <= K <= n_kv_slots_max={g.n_kv_slots_max} (window + index_topk)")
        for nm in ("w_q_a", "w_q_b", "w_kv", "w_o_a", "w_o_b", "w_q_norm", "w_kv_norm", "attn_sink", "cos", "sin", "kv_all", "topk_idxs", "x", "out"):
            if not self._contiguous(d[nm]):
                raise ValueError(
                    f"{nm} must be contiguous (shape {tuple(d[nm].shape)}, strides {tuple(d[nm].stride)}); the block binds it as is, never repacks"
                )
        n_c = self.n_kv_rows - s
        if n_c > 0 and not g.has_compressed_kv:
            raise ValueError(
                f"kv_all carries N_c={n_c} compressed rows but geometry.has_compressed_kv=False (a window-only layer takes kv_all [B, S, head_dim])"
            )
        if n_c == 0 and g.has_compressed_kv and k != min(s, g.window):
            # The legal S < ratio degenerate: no compressed rows -> the list is the window list alone.
            raise ValueError(
                f"kv_all has no compressed rows (N_c == 0) under has_compressed_kv=True, so topk_idxs must be the window list alone: "
                f"K == min(S, window) = {min(s, g.window)}, got K={k}"
            )
        dev = d["x"].device
        for nm, dd in d.items():
            if dd.device != dev:
                raise ValueError(f"{nm} must live on x's device {dev}, got {dd.device}")
        if dev.type != "cuda":
            raise ValueError(f"the block runs on CUDA tensors; x is on {dev}")
        self._check_knobs()
        # Arch AFTER the declaration and knob checks, so a bad declaration reads the same on every device.
        cc = tuple(torch.cuda.get_device_capability(dev))
        if cc != _SM107_CC:
            raise NotImplementedError(f"mqa_sparse_attention_block targets Rubin (SM{_SM107_CC[0]}{_SM107_CC[1]}) only for now; found SM{cc[0]}{cc[1]}")
        self._build_stages()
        for st in self._stages:
            st.check_support()
        self._is_supported = True
        return True

    def _check_knobs(self) -> None:
        """Device-independent knob declines (typed), ahead of the arch check."""
        if self.attention == "d512":
            from .kernels.sparse_attention import D512_FORK_MODULE, d512_fork_available

            if not d512_fork_available():
                raise NotImplementedError(
                    f"attention='d512': the gathered-list d512 sparse-attention fork ({D512_FORK_MODULE}) has not landed; use attention='dsa' or 'torch'"
                )
        if self.pointwise_impl == "frost":
            try:
                from .kernels import norm_rope  # noqa: F401
            except ImportError as exc:
                raise NotImplementedError(
                    f"pointwise_impl='frost' needs kernels/norm_rope.py, which has not landed ({exc}); use pointwise_impl='torch'"
                ) from exc
        elif not self.three_roundings:
            raise NotImplementedError(
                "three_roundings=False exists only on the norm_rope kernel (pointwise_impl='frost'); the torch arm is the model's arithmetic"
            )

    def _build_stages(self) -> None:
        g, b, s, t = self.geom, self.batch, self.seq_len, self.tokens
        hd = g.n_heads * g.head_dim
        dt, dev = self.dtype, self.device
        pw = dict(impl=self.pointwise_impl, eps=g.norm_eps, n_tokens=t, batch=b, seq_len=s, device=dev, dtype=dt, three_roundings=self.three_roundings)
        self._q_a_proj = _Projection(m=t, k=g.d_model, n=g.q_lora_rank, dtype=dt, name="q_a_proj")
        self._q_norm = _NormRope(name="q_norm", d=g.q_lora_rank, heads=1, apply_norm=True, rope_dim=0, **pw)  # RMSNorm only
        self._q_b_proj = _Projection(m=t, k=g.q_lora_rank, n=hd, dtype=dt, name="q_b_proj")
        self._q_rope = _NormRope(name="q_rope", d=g.head_dim, heads=g.n_heads, apply_norm=False, rope_dim=g.rope_dim, **pw)
        # D4: batched over B with M = S, C = the [:, :S] prefix of kv_all at batch stride (S + N_c) * head_dim.
        self._kv_proj = _Projection(
            m=s,
            k=g.d_model,
            n=g.head_dim,
            dtype=dt,
            name="kv_proj",
            batch=b,
            a_batch_stride=s * g.d_model,
            c_row_stride=g.head_dim,
            c_batch_stride=self.n_kv_rows * g.head_dim,
        )
        self._kv_chain = _NormRope(name="kv_chain", d=g.head_dim, heads=1, apply_norm=True, rope_dim=g.rope_dim, fake_quant=(g.kv_fake_quant != "none"), **pw)
        self._attention = _SparseAttention(
            impl=self.attention, geom=g, batch=b, seq_len=s, n_kv_rows=self.n_kv_rows, topk=self.topk, dtype=dt, device=dev, want_lse=self.return_lse
        )
        self._o_unrope = _NormRope(
            name="o_unrope",
            d=g.head_dim,
            heads=g.n_heads,
            apply_norm=False,
            rope_dim=g.rope_dim,
            rope_inverse=True,
            lse_fold=(self.attention == "dsa" and self.return_lse),
            **pw,
        )
        if self.wo_a_batched:
            # M6: ONE launch, batch = o_groups, interleaved batch stride (< the row stride) on A and C.
            self._o_a_proj = _Projection(
                m=t,
                k=g.group_width,
                n=g.o_lora_rank,
                dtype=dt,
                name="o_a_proj",
                batch=g.o_groups,
                a_row_stride=hd,
                a_batch_stride=g.group_width,
                b_batch_stride=g.o_lora_rank * g.group_width,
                c_row_stride=g.n_o_lora,
                c_batch_stride=g.o_lora_rank,
            )
        else:
            # D3: eight launches of ONE plan on column slices (declared row strides).
            self._o_a_proj = _Projection(m=t, k=g.group_width, n=g.o_lora_rank, dtype=dt, name="o_a_proj", a_row_stride=hd, c_row_stride=g.n_o_lora)
        self._o_b_proj = _Projection(m=t, k=g.n_o_lora, n=g.d_model, dtype=dt, name="o_b_proj")
        self._stages = (
            self._q_a_proj,
            self._q_norm,
            self._q_b_proj,
            self._q_rope,
            self._kv_proj,
            self._kv_chain,
            self._attention,
            self._o_unrope,
            self._o_a_proj,
            self._o_b_proj,
        )

    # -- workspace ----------------------------------------------------------

    def _layout(self) -> _Slots:
        g, b, s, t = self.geom, self.batch, self.seq_len, self.tokens
        bpe = _itemsize(self.dtype)
        lay = WorkspaceLayout(align=_WS_ALIGN)
        qa = lay.add(t * g.q_lora_rank * bpe)
        q = lay.add(t * g.n_heads * g.head_dim * bpe)
        o = lay.add(t * g.n_heads * g.head_dim * bpe)
        o_lora = lay.add(t * g.n_o_lora * bpe)
        lse_th = max_logits = idx_tmp = idx_ws = None
        if self.attention == "dsa":
            lse_th = lay.add(t * g.n_heads * 4)
            max_logits = lay.add(t * g.n_heads * 4)
            if self._attention.needs_idx_staging:
                if b > 1:
                    idx_tmp = lay.add(t * self.topk * 4)
                idx_ws = lay.add(t * self._attention.k_pad * 4)
        engine = max([st.workspace_bytes() for st in self._stages if isinstance(st, _Projection)] + [1])
        scratch = lay.add(align_up(engine, _WS_ALIGN))
        return _Slots(
            qa=qa, q=q, o=o, o_lora=o_lora, lse_th=lse_th, max_logits=max_logits, idx_tmp=idx_tmp, idx_ws=idx_ws, engine_scratch=scratch, total_bytes=lay.size
        )

    def get_workspace_size(self) -> int:
        """Bytes the caller must provide: every intermediate plus the GEMM engines'
        own scratch at a reserved offset.  Honest and never exceeded."""
        self._ensure_support_checked()
        if not self._compiled:
            raise RuntimeError("call compile() before get_workspace_size() (the GEMM plans report their scratch)")
        return self._layout().total_bytes

    # -- compile ------------------------------------------------------------

    def compile(self) -> None:
        self._ensure_support_checked()
        for st in self._stages:
            st.compile()
        self._compiled = True
        self._slots = self._layout()

    # -- execute ------------------------------------------------------------

    def _resolve_stream(self, current_stream) -> int:
        """The ONE launch stream, as a raw ``CUstream`` int, resolved on the DECLARED device."""
        if current_stream is None:
            return torch.cuda.current_stream(self.device).cuda_stream
        if isinstance(current_stream, (int, cuda.CUstream)):
            return int(current_stream)
        raise TypeError(f"current_stream must be a cuda.CUstream or a raw CUstream int (pass torch_stream.cuda_stream), got {type(current_stream).__name__}")

    def _validate_bound(self, tensors: dict, workspace: torch.Tensor, lse: Optional[torch.Tensor]) -> None:
        """Every execute-time check, host-only, BEFORE the first launch -- so a decline
        leaves the caller's ``kv_all`` / ``out`` / workspace untouched."""
        g, b, s = self.geom, self.batch, self.seq_len
        _check_bound_tensors(self._descs, tensors, self.device)
        kv_all, topk_idxs = tensors["kv_all"], tensors["topk_idxs"]
        if kv_all.data_ptr() % 16:
            raise ValueError(f"kv_all must be 16-byte aligned (TMA / gather operand), got 0x{kv_all.data_ptr():x}")
        if self.attention == "dsa" and not self._attention.needs_idx_staging:
            from .kernels.sparse_attention import check_dsa_index_alignment

            check_dsa_index_alignment(topk_idxs)  # identity index path (B == 1, K % 64 == 0): the list is bound as a [T, K] view
        if self.return_lse:
            if lse is None or tuple(lse.shape) != (b, g.n_heads, s) or lse.dtype != torch.float32 or not lse.is_contiguous() or lse.device != self.device:
                raise ValueError(
                    f"return_lse=True: lse must be a contiguous fp32 [B, n_heads, S] = {(b, g.n_heads, s)} tensor on {self.device}, "
                    f"got {None if lse is None else (tuple(lse.shape), lse.dtype, lse.device)}"
                )
        elif lse is not None:
            raise ValueError("lse was given but the block was declared with return_lse=False; declare return_lse=True to have it written")
        req = self._slots.total_bytes
        if workspace.dtype != torch.uint8 or workspace.ndim != 1:
            raise ValueError(f"workspace must be a 1-D uint8 tensor, got {tuple(workspace.shape)} {workspace.dtype}")
        if workspace.device != self.device:
            raise ValueError(f"workspace must live on the declared device {self.device}, got {workspace.device}")
        if workspace.numel() < req:
            raise ValueError(f"workspace is {workspace.numel()} bytes, need {req} (get_workspace_size())")
        if workspace.data_ptr() % _WS_ALIGN:
            raise ValueError(f"workspace must be {_WS_ALIGN}-byte aligned, got 0x{workspace.data_ptr():x}")

    def _bind(self, tensors: dict, workspace: torch.Tensor, lse: Optional[torch.Tensor], stream: int) -> list:
        """Validate, carve the workspace views, and return the pipeline as ``_StageRunner``s.

        ``execute`` runs them in order; ``stage_runners`` hands the list out.  Each in-place
        stage (2 / 4 / 6 / 8) carries ``prepare`` = the run of the stage that PRODUCES its
        input, so a timing loop restores the bytes bit-exactly without a snapshot."""
        if not self._compiled or self._slots is None:
            raise RuntimeError("call compile() before execute()")
        self._validate_bound(tensors, workspace, lse)
        g, b, s, t = self.geom, self.batch, self.seq_len, self.tokens
        x, w_q_a, w_q_norm, w_q_b, w_kv, w_kv_norm, w_o_a, w_o_b, attn_sink, cos, sin, kv_all, topk_idxs, out = (tensors[n] for n in TENSOR_NAMES)
        ws, sl = workspace, self._slots
        h, dh = g.n_heads, g.head_dim
        engine_ws = ws[sl.engine_scratch :]
        cos2, sin2 = cos.view(t, g.rope_dim // 2), sin.view(t, g.rope_dim // 2)

        qa = _view(ws, sl.qa, (t, g.q_lora_rank), self.dtype)
        q = _view(ws, sl.q, (t, h, dh), self.dtype)
        o = _view(ws, sl.o, (t, h, dh), self.dtype)
        o_lora = _view(ws, sl.o_lora, (t, g.n_o_lora), self.dtype)
        kv_prefix = kv_all[:, :s, :]
        lse_th = _view(ws, sl.lse_th, (t, h), torch.float32) if sl.lse_th is not None else None
        max_logits = _view(ws, sl.max_logits, (t, h), torch.float32) if sl.max_logits is not None else None
        idx_tmp = _view(ws, sl.idx_tmp, (b, s, self.topk), torch.int32) if sl.idx_tmp is not None else None
        idx_ws = _view(ws, sl.idx_ws, (b, s, self._attention.k_pad), torch.int32) if sl.idx_ws is not None else None
        fold = self.attention == "dsa" and self.return_lse

        # 1 + 2: q_a_proj -> RMSNorm in place
        def q_a_proj():
            self._q_a_proj.execute(x.view(t, g.d_model), w_q_a, qa, engine_ws, stream=stream)

        def q_norm():
            self._q_norm.execute(qa.view(t, 1, g.q_lora_rank), w_q_norm, None, None, stream=stream)

        # 3 + 4: q_b_proj -> RoPE on the last rope_dim dims of every head, in place
        def q_b_proj():
            self._q_b_proj.execute(qa, w_q_b, q.view(t, h * dh), engine_ws, stream=stream)

        def q_rope():
            self._q_rope.execute(q, None, cos2, sin2, stream=stream)

        # 5 + 6: kv_proj straight into kv_all[:, :S] (D4) -> RMSNorm, RoPE, fake-quant in place there
        def kv_proj():
            self._kv_proj.execute(x, w_kv, kv_prefix, engine_ws, stream=stream)

        def kv_chain():
            self._kv_chain.execute(kv_prefix, w_kv_norm, cos2, sin2, stream=stream)

        # 6b + 7: the attention adapter (dsa: index staging into the workspace, then the H64 kernel)
        def idx_staging():
            self._attention.stage_indices(topk_idxs, idx_tmp, idx_ws, stream=stream)

        def sparse_attention():
            self._attention.launch(
                q.view(b, s, h, dh),
                kv_all,
                attn_sink,
                topk_idxs,
                o.view(b, s, h, dh),
                lse=lse if (self.return_lse and self.attention == "torch") else None,
                lse_th=lse_th,
                max_logits=max_logits,
                idx_ws=idx_ws,
                stream=stream,
            )

        # 8: inverse RoPE in place on o (+ the D1 LSE fold for dsa)
        def o_unrope():
            self._o_unrope.execute(
                o, None, cos2, sin2, lse_in=lse_th if fold else None, sink=attn_sink if fold else None, lse_out=lse if fold else None, stream=stream
            )

        # 9: grouped o_a_proj (eight launches on column slices, or the one batched launch)
        o2 = o.view(t, h * dh)
        if self.wo_a_batched:
            a_b = o2.view(t, g.o_groups, g.group_width).permute(1, 0, 2)  # [G, T, group_width], batch stride < row stride
            c_b = o_lora.view(t, g.o_groups, g.o_lora_rank).permute(1, 0, 2)  # [G, T, o_lora_rank]

            def o_a_proj():
                self._o_a_proj.execute(a_b, w_o_a, c_b, engine_ws, stream=stream)

        else:
            gw, r = g.group_width, g.o_lora_rank
            slices = [(o2[:, grp * gw : (grp + 1) * gw], w_o_a[grp], o_lora[:, grp * r : (grp + 1) * r]) for grp in range(g.o_groups)]

            def o_a_proj():
                for a_g, w_g, c_g in slices:
                    self._o_a_proj.execute(a_g, w_g, c_g, engine_ws, stream=stream)

        # 10: o_b_proj -> out
        def o_b_proj():
            self._o_b_proj.execute(o_lora, w_o_b, out.view(t, g.d_model), engine_ws, stream=stream)

        kind = dict(RUNNER_NAMES)
        runners = [
            _StageRunner("q_a_proj", kind["q_a_proj"], q_a_proj),
            _StageRunner("q_norm", kind["q_norm"], q_norm, prepare=q_a_proj),
            _StageRunner("q_b_proj", kind["q_b_proj"], q_b_proj),
            _StageRunner("q_rope", kind["q_rope"], q_rope, prepare=q_b_proj),
            _StageRunner("kv_proj", kind["kv_proj"], kv_proj),
            _StageRunner("kv_chain", kind["kv_chain"], kv_chain, prepare=kv_proj),
        ]
        if self._attention.needs_idx_staging:
            runners.append(_StageRunner("idx_staging", kind["idx_staging"], idx_staging))
        runners += [
            _StageRunner("sparse_attention", kind["sparse_attention"], sparse_attention),
            _StageRunner("o_unrope", kind["o_unrope"], o_unrope, prepare=sparse_attention),
            _StageRunner("o_a_proj", kind["o_a_proj"], o_a_proj),
            _StageRunner("o_b_proj", kind["o_b_proj"], o_b_proj),
        ]
        return runners

    def execute(
        self,
        x: torch.Tensor,
        w_q_a: torch.Tensor,
        w_q_norm: torch.Tensor,
        w_q_b: torch.Tensor,
        w_kv: torch.Tensor,
        w_kv_norm: torch.Tensor,
        w_o_a: torch.Tensor,
        w_o_b: torch.Tensor,
        attn_sink: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_all: torch.Tensor,
        topk_idxs: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        lse: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch the stages in pipeline order on ONE stream.

        Validate-then-launch: every runtime tensor is checked against the declaration
        (shape, dtype, contiguity, device -- ``_check_bound_tensors``), plus the
        alignment / workspace / ``lse`` contracts, BEFORE the first launch; a decline
        leaves ``kv_all``, ``out`` and the workspace untouched.

        ``current_stream`` is a ``cuda.CUstream`` or a raw ``CUstream`` int (a
        ``torch.cuda.Stream`` is refused: pass its ``.cuda_stream``); ``None`` means
        torch's current stream on the DECLARED device.  The GEMMs take it through
        ``run_proj_gemm(stream=)`` (both routes), the DSA kernel as
        ``current_stream``, the torch stages under ``torch.cuda.stream``.  No
        stage derives its own stream.  The FLAGGED torch stages allocate; the
        GEMM and ``dsa`` stages do not, and nothing here syncs.
        """
        tensors = dict(zip(TENSOR_NAMES, (x, w_q_a, w_q_norm, w_q_b, w_kv, w_kv_norm, w_o_a, w_o_b, attn_sink, cos, sin, kv_all, topk_idxs, out)))
        for runner in self._bind(tensors, workspace, lse, self._resolve_stream(current_stream)):
            runner.run()

    def stage_runners(
        self,
        x: torch.Tensor,
        w_q_a: torch.Tensor,
        w_q_norm: torch.Tensor,
        w_q_b: torch.Tensor,
        w_kv: torch.Tensor,
        w_kv_norm: torch.Tensor,
        w_o_a: torch.Tensor,
        w_o_b: torch.Tensor,
        attn_sink: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_all: torch.Tensor,
        topk_idxs: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        lse: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> list:
        """The pipeline as per-stage launch units bound to these tensors -- the same
        validation, views and stream as :meth:`execute` (running them in order IS
        ``execute``).  For per-op attribution (the perf table's comparison mode):

        each ``_StageRunner`` has ``.name`` (``RUNNER_NAMES``: ``q_a_proj`` .. ``o_b_proj``;
        ``idx_staging`` only under ``dsa`` when ``B > 1`` or ``K % 64 != 0``), ``.kind``
        (``'mma'`` / ``'bw'`` / ``'prep'``), ``.run()`` and, on the in-place stages
        (``q_norm``, ``q_rope``, ``kv_chain``, ``o_unrope``), ``.prepare()`` -- the
        producing stage's run, restoring the input bit-exactly with no snapshot and no
        allocation.  ``sparse_attention.run()`` is the kernel alone (the staging is its
        own ``prep`` runner), so it is the ``kernel-only`` row.  The runners hold VIEWS of
        ``workspace`` / the tensors: keep them alive while the runners are in use, and run
        the whole list once before timing a stage in isolation (its inputs are workspace
        state).  ``lse`` is required exactly when ``return_lse=True`` (as for ``execute``).
        """
        tensors = dict(zip(TENSOR_NAMES, (x, w_q_a, w_q_norm, w_q_b, w_kv, w_kv_norm, w_o_a, w_o_b, attn_sink, cos, sin, kv_all, topk_idxs, out)))
        return self._bind(tensors, workspace, lse, self._resolve_stream(current_stream))

    # -- introspection ----------------------------------------------------------

    @property
    def stages(self) -> tuple:
        """The live stages in pipeline order (built by ``check_support``)."""
        return self._stages

    def workspace_slots(self) -> _Slots:
        """The workspace carve (offsets in bytes) -- for tests that read intermediates back."""
        if self._slots is None:
            raise RuntimeError("call compile() first")
        return self._slots
