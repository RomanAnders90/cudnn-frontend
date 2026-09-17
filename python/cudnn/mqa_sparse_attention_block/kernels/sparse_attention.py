# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 of the MQA sparse-attention block: sparse attention over an index list,
behind THREE swappable adapters with one contract.

The contract (the block's, not any kernel's -- ``geometry.py`` module docstring):
``q [B, S, H, D]`` bf16/f16; ``kv_all [B, N, D]`` (K == V, ONE tensor, ``N = S + N_c``);
``attn_sink [H]`` fp32 folded into the softmax DENOMINATOR once; ``topk_idxs
[B, S, K]`` int32 with ``-1`` = no key and duplicates counted twice; ``scale``
applied AFTER the QK product; fp32 online softmax, bf16 ``P``; ``o [B, S, H, D]``;
optional ``lse [B, H, S]`` fp32 natural-log in the FROST convention: sink
INCLUDED, ``== sink`` on a keyless row with a sink, ``-inf`` without one.
Ids outside ``[0, N)`` are "no key" on EVERY adapter: the ``torch`` adapter masks
them, the H64 kernel masks ``token < 0 or token >= kv.shape[0]`` itself (so the
``dsa`` identity path at ``B == 1`` is covered), and the ``dsa`` staging pass
(:func:`dsa_stage_indices`) folds the ``[0, N)`` test into its flat-id arithmetic
at ``B > 1`` -- without it an id ``>= N`` offset by ``b * N`` would alias another
batch entry's rows (the kernel's bound is the flat ``B * N``).  No D2H read
anywhere on the execute path.

Adapters:

``torch_sparse_attention``
    The oracle's op order in batch torch (``sparse_attention_reference`` of the
    test suite, kept independent there): fp32 scores from ``.float()`` operands,
    ``-inf`` on ``-1``, fp32 row-sum, bf16 ``P`` for the PV product, the empty
    denominator SELECTed to ``O = 0``.  **TEMPORARY / FLAGGED: allocates
    temporaries per call** (gathers, scores, P) -- it is the attribution baseline
    and the correctness fallback, not a contract-clean stage.

``DsaSparseAttention``
    The in-tree SM100 DSA H64 kernel (``cudnn.deepseek_sparse_attention``):
    ``attn_sink`` is passed INTO the kernel, which folds it into the O
    denominator once; its LSE EXCLUDES the sink and is ``+inf`` on a keyless row,
    so the block's stage 8 folds ``lse = select(lse_h64 == +inf, sink,
    select(sink == +inf, +inf, logaddexp(lse_h64, sink)))`` (a bare ``logaddexp``
    is NaN at ``sink = +inf`` on a live row).  Index staging (:meth:`stage_indices`)
    exists only when ``B > 1`` (flat global ids) or ``K % 64 != 0`` (the kernel's
    slot granule; the interface would otherwise ``torch.cat`` a pad per call);
    otherwise the caller's ``[B, S, K]`` list is bound as a ``[T, K]`` view.
    The JIT is primed at ``compile()`` with a one-token probe: the interface
    compiles lazily at first execute and keys its cache without shapes.

``D512SparseAttention``
    The gathered-list fork of the Rubin d512 SDPA.  A typed
    ``NotImplementedError("... has not landed")`` until the fork module
    (``kernels/sparse_attention_d512.py``) exists; feature-detected by import.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Optional

import torch

# The row-max floor of the reference kernel (K:355): finite, so an all -1 row gives O = 0, not NaN.
_MAX_FLOOR = -1e30

# The DSA H64 kernel's index granule: K is padded to a multiple of this many slots (docs/fe-oss-apis/dsa.md).
DSA_TOPK_GRANULE = 64
# The DSA interface re-aligns a ``topk_idxs`` whose base is not 32-byte aligned by CLONING it (a hidden
# per-execute allocation); the block refuses such a pointer instead.
DSA_IDX_ALIGN_BYTES = 32

D512_FORK_MODULE = "cudnn.mqa_sparse_attention_block.kernels.sparse_attention_d512"


# ---------------------------------------------------------------------------
# Shared operand checks (host arithmetic only)
# ---------------------------------------------------------------------------


def check_attention_operands(q, kv_all, sink, topk_idxs, out, lse=None) -> tuple:
    """``(B, S, H, D, N, K)`` of a consistent operand set, else ``ValueError`` naming the tensor."""
    if q.ndim != 4:
        raise ValueError(f"q must be [B, S, H, D], got {tuple(q.shape)}")
    b, s, h, d = (int(x) for x in q.shape)
    if kv_all.ndim != 3 or int(kv_all.shape[0]) != b or int(kv_all.shape[2]) != d:
        raise ValueError(f"kv_all must be [B={b}, N, D={d}], got {tuple(kv_all.shape)}")
    n = int(kv_all.shape[1])
    if n < s:
        raise ValueError(f"kv_all must hold at least the S={s} window rows, got N={n}")
    if kv_all.dtype != q.dtype:
        raise ValueError(f"kv_all must have q's dtype {q.dtype}, got {kv_all.dtype}")
    if topk_idxs.ndim != 3 or tuple(int(x) for x in topk_idxs.shape[:2]) != (b, s):
        raise ValueError(f"topk_idxs must be [B={b}, S={s}, K], got {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32, got {topk_idxs.dtype}")
    k = int(topk_idxs.shape[2])
    if k < 1:
        raise ValueError(f"topk_idxs must list at least one slot per query, got K={k}")
    if tuple(int(x) for x in sink.shape) != (h,) or sink.dtype != torch.float32:
        raise ValueError(f"attn_sink must be fp32 [{h}], got {tuple(sink.shape)} {sink.dtype}")
    if tuple(int(x) for x in out.shape) != (b, s, h, d) or out.dtype != q.dtype:
        raise ValueError(f"out must be [B, S, H, D] = {(b, s, h, d)} of dtype {q.dtype}, got {tuple(out.shape)} {out.dtype}")
    if lse is not None and (tuple(int(x) for x in lse.shape) != (b, h, s) or lse.dtype != torch.float32):
        raise ValueError(f"lse must be fp32 [B, H, S] = {(b, h, s)}, got {tuple(lse.shape)} {lse.dtype}")
    return b, s, h, d, n, k


# ---------------------------------------------------------------------------
# Adapter 1: torch (oracle op order; TEMPORARY -- allocates per call)
# ---------------------------------------------------------------------------


def torch_sparse_attention(
    q: torch.Tensor,
    kv_all: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    q_chunk: int = 128,
) -> None:
    """Gathered sparse attention in batch torch, written into ``out`` (and ``lse``).

    Op order per query chunk (the reference kernel's, ``K:350-387``): gather
    ``kv_all[b, idx]`` (a ZERO row for ``-1``); fp32 scores from the bf16 values;
    ``-inf`` where the slot is no key; ``scale`` AFTER the product; row max floored
    at ``-1e30``; ``p = exp(s - m)``; fp32 denominator ``sum p + exp(sink - m)`` (sink
    once, no value row); numerator with ``P`` rounded to ``q.dtype`` before the PV
    product; ``o = num / den`` with the empty denominator SELECTed to 0.  LSE (when
    asked) is ``logaddexp(m_raw + log(sum p), sink)``: sink included, ``sink`` on a
    keyless row, ``-inf`` with no key and no sink.

    **FLAGGED (temporary):** allocates the per-chunk gather / score / P tensors.
    Kept as the attribution baseline and the any-device correctness path; it is
    NOT a contract-clean stage (engine contract rule 10).
    """
    b, s, h, d, n, _ = check_attention_operands(q, kv_all, sink, topk_idxs, out, lse)
    dev = q.device
    sink_b = sink.view(1, 1, h)
    bidx = torch.arange(b, device=dev)[:, None, None]
    step = max(1, int(q_chunk))
    for lo in range(0, s, step):
        hi = min(s, lo + step)
        idx = topk_idxs[:, lo:hi].long()  # [B, C, K]
        valid = (idx >= 0) & (idx < n)  # -1 (and anything outside [0, N)) is "no key"
        kv_g = kv_all[bidx, idx.clamp(0, n - 1)].float()  # [B, C, K, D] -- gathered rows, exact in fp32
        kv_g = kv_g.masked_fill(~valid.unsqueeze(-1), 0.0)  # K:362: no key -> zero row
        scores = torch.einsum("bchd,bckd->bchk", q[:, lo:hi].float(), kv_g)
        scores = scores.masked_fill(~valid.unsqueeze(2), -math.inf)  # K:364: no key -> -inf
        scores = scores * scale  # K:366-367: AFTER the product
        m_raw = scores.amax(-1)  # -inf on a keyless row
        m = m_raw.clamp_min(_MAX_FLOOR)  # K:355
        p = torch.exp(scores - m.unsqueeze(-1))  # 0 on masked slots (and everywhere on a keyless row)
        sum_p = p.sum(-1)  # K:374-376: fp32 P
        num = torch.einsum("bchk,bckd->bchd", p.to(q.dtype).float(), kv_g)  # K:377-380: bf16 P
        sum_exp = sum_p + torch.exp(sink_b - m)  # K:382-383: the sink, once, in the denominator only
        o = torch.where((sum_exp > 0).unsqueeze(-1), num / sum_exp.unsqueeze(-1), torch.zeros_like(num))
        out[:, lo:hi].copy_(o)  # the ONE rounding to the output dtype (K:384-387)
        if lse is not None:
            lse_keys = m_raw + torch.log(sum_p)  # -inf on a keyless row (log 0)
            l = torch.logaddexp(lse_keys, sink_b)  # sink INCLUDED (FROST convention)
            l = torch.where(torch.isnan(l), torch.full_like(l, -math.inf), l)  # both -inf: no key, no sink
            lse[:, :, lo:hi].copy_(l.permute(0, 2, 1))


# ---------------------------------------------------------------------------
# Adapter 2: the in-tree SM100 DSA H64 kernel
# ---------------------------------------------------------------------------


def dsa_topk_padded(k: int) -> int:
    """``K`` rounded up to the H64 kernel's 64-slot granule."""
    return -(-int(k) // DSA_TOPK_GRANULE) * DSA_TOPK_GRANULE


def dsa_needs_index_staging(batch: int, k: int) -> bool:
    """D5: stage the list iff the kernel cannot take the caller's buffer as is --
    ``B > 1`` (it wants FLAT global row ids over ``kv_all.view(B*N, D)``) or
    ``K % 64`` (its slot granule; the interface would pad with a per-call ``torch.cat``)."""
    return int(batch) > 1 or int(k) % DSA_TOPK_GRANULE != 0


def check_dsa_index_alignment(topk_idxs: torch.Tensor) -> None:
    """The kernel reads its index rows with 32-byte vectors; the interface CLONES a
    misaligned buffer (a hidden per-execute allocation).  Refuse it, typed."""
    if not topk_idxs.is_contiguous():
        raise ValueError("topk_idxs must be contiguous for the dsa adapter (bound as a [T, K] view)")
    if topk_idxs.data_ptr() % DSA_IDX_ALIGN_BYTES:
        raise ValueError(
            f"topk_idxs must be {DSA_IDX_ALIGN_BYTES}-byte aligned for the dsa adapter (the interface would clone a misaligned buffer per "
            f"execute), got 0x{topk_idxs.data_ptr():x}"
        )


def dsa_lse_fold(lse_h64: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
    """D1: the H64 kernel's LSE (sink EXCLUDED, ``+inf`` on a keyless row) -> the FROST
    convention (sink INCLUDED, ``sink`` on a keyless row), as one select chain.

    ``lse_h64 [T, H]`` fp32, ``sink [H]`` fp32 -> ``[T, H]`` fp32 (a new tensor; the
    torch stage-8 arm copies it into the caller's ``[B, H, S]`` LSE).  The inner
    select keeps ``sink = +inf`` finite-safe: ``logaddexp(x, +inf)`` is ``+inf`` in
    torch but the branch is written out so the contract does not rest on it.
    """
    sink_b = sink.view(1, -1)
    keyless = lse_h64 == math.inf
    folded = torch.where(sink_b == math.inf, torch.full_like(lse_h64, math.inf), torch.logaddexp(lse_h64, sink_b))
    return torch.where(keyless, sink_b.expand_as(lse_h64), folded)


def dsa_stage_indices(
    topk_idxs: torch.Tensor, idx_tmp: Optional[torch.Tensor], idx_ws: torch.Tensor, *, base_col: Optional[torch.Tensor], n_kv_rows: int, topk: int, k_pad: int
) -> None:
    """D5 index staging with the ``[0, N)`` contract folded in: int32 ``out=`` ops only (no allocation, no D2H).

    ``idx_ws[b, s, :K] = (0 <= id < N) ? id + b * N : -1``, ``idx_ws[..., K:] = -1``.  ``topk_idxs
    [B, S, K]`` (caller), ``idx_tmp [B, S, K]`` int32 scratch (``None`` at ``B == 1``), ``idx_ws
    [B, S, K_pad]`` (the kernel's operand), ``base_col [B, 1, 1]`` int32 of ``b * N`` (``None`` at
    ``B == 1``, where flat ids == local ids and the kernel's own ``0 <= token < kv.shape[0]``
    mask is exactly the contract, so the list is copied as is).

    At ``B > 1`` the validity is sign arithmetic on the two buffers (``T := idx_ws[..., :K]``,
    ``X := idx_tmp``; the list is clamped to ``[-1, N]`` first so any out-of-range id lands on
    one of the two sentinels)::

        T = clamp(id, -1, N);  X = sign(T + 1)           # 1 iff T >= 0
        T = sign(N - T);       X = X * T                 # v = 1 iff 0 <= id < N, else 0
        T = clamp(id, -1, N) + b * N                     # the clamp re-read is cheaper than a third scratch
        T = T * v + (v - 1)                              # v ? id + b * N : -1
    """
    k = int(topk)
    tgt = idx_ws[:, :, :k]
    if base_col is None:
        tgt.copy_(topk_idxs)
    else:
        if idx_tmp is None:
            raise RuntimeError("dsa_stage_indices at B > 1 needs the idx_tmp scratch")
        n = int(n_kv_rows)
        torch.clamp(topk_idxs, -1, n, out=tgt)
        torch.add(tgt, 1, out=idx_tmp)
        torch.sign(idx_tmp, out=idx_tmp)  # 1 iff id >= 0
        torch.neg(tgt, out=tgt)
        torch.add(tgt, n, out=tgt)
        torch.sign(tgt, out=tgt)  # 1 iff id < N (0 at the clamp sentinel N)
        torch.mul(idx_tmp, tgt, out=idx_tmp)  # v
        torch.clamp(topk_idxs, -1, n, out=tgt)
        torch.add(tgt, base_col, out=tgt)  # id + b * N
        torch.mul(tgt, idx_tmp, out=tgt)  # v * (id + b * N)
        torch.sub(idx_tmp, 1, out=idx_tmp)  # v - 1
        torch.add(tgt, idx_tmp, out=tgt)  # v ? id + b * N : -1
    if int(k_pad) > k:
        idx_ws[:, :, k:].fill_(-1)


class DsaSparseAttention:
    """Stage-7 adapter over ``cudnn.deepseek_sparse_attention.SparseAttentionForward`` (H64 variant).

    Plan-time keys: ``(batch, seq_len, n_kv_rows, n_heads, head_dim, topk, dtype,
    device)``.  ``scale`` is a runtime float of the kernel.  Execute binds the
    block's views: ``q.view(T, H, D)``, ``kv_all.view(B*N, D)``, the ``[T, K_pad]``
    index view, ``attn_sink``, ``o.view(T, H, D)``, ``lse_th [T, H]`` and the
    kernel's required ``max_logits [T, H]`` dead output.  No allocation on that
    path: every buffer is the caller's or the block's workspace.
    """

    def __init__(self, *, batch: int, seq_len: int, n_kv_rows: int, n_heads: int, head_dim: int, topk: int, scale: float, dtype, device) -> None:
        self.batch, self.seq_len, self.n_kv_rows = int(batch), int(seq_len), int(n_kv_rows)
        self.n_heads, self.head_dim, self.topk = int(n_heads), int(head_dim), int(topk)
        self.scale, self.dtype, self.device = float(scale), dtype, torch.device(device)
        self.tokens = self.batch * self.seq_len
        self.k_pad = dsa_topk_padded(self.topk)
        self.needs_idx_staging = dsa_needs_index_staging(self.batch, self.topk)
        self._impl = None
        self._base_col = None  # [B, 1, 1] int32 of b * N -- the flat-row offset per batch entry (compile-time constant)

    # -- support / compile --------------------------------------------------

    def check_support(self) -> None:
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise NotImplementedError(f"dsa adapter: bf16 / f16 only, got {self.dtype}")
        if self.device.type != "cuda":
            raise ValueError(f"dsa adapter: q must live on CUDA, got {self.device}")
        if self.topk < 1 or self.seq_len < 1 or self.batch < 1 or self.n_kv_rows < self.seq_len:
            raise ValueError(f"dsa adapter: need B, S, K >= 1 and N >= S; got B={self.batch} S={self.seq_len} K={self.topk} N={self.n_kv_rows}")
        from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100 as iface
        from cudnn.deepseek_sparse_attention.sparse_attention_forward.api import _SUPPORTED_VARIANTS, SparseAttentionForward

        variant = (self.n_heads, self.head_dim)
        if variant not in _SUPPORTED_VARIANTS:
            raise NotImplementedError(f"dsa adapter: (n_heads, head_dim) must be one of {tuple(_SUPPORTED_VARIANTS)}, got {variant}")
        cc = tuple(torch.cuda.get_device_capability(self.device))
        if cc[0] != 10 or cc not in iface._ARCH_FLAGS:
            raise NotImplementedError(f"dsa adapter: the DSA kernels target the SM100 family {sorted(iface._ARCH_FLAGS)}; found SM{cc[0]}{cc[1]}")
        # Sample tensors at the smallest shapes the API validates against (plan time -- allowed).
        sample_q = torch.empty(1, self.n_heads, self.head_dim, dtype=self.dtype, device=self.device)
        sample_kv = torch.empty(1, self.head_dim, dtype=self.dtype, device=self.device)
        sample_idx = torch.empty(1, self.k_pad, dtype=torch.int32, device=self.device)
        sample_sink = torch.empty(self.n_heads, dtype=torch.float32, device=self.device)
        impl = SparseAttentionForward(sample_q, sample_kv, sample_idx, sample_attn_sink=sample_sink, softmax_scale=self.scale, indexer_topk=0)
        impl.check_support()  # the API's own typed declines (ValueError / RuntimeError on arch)
        self._impl = impl

    def compile(self) -> None:
        """D6: the interface compiles at first EXECUTE (its cache key carries no
        shapes), so prime it here with a one-token probe on the current stream."""
        if self._impl is None:
            raise RuntimeError("call check_support() before compile()")
        from cuda.bindings import driver as cuda

        self._impl.compile()
        dev = self.device
        q1 = torch.zeros(1, self.n_heads, self.head_dim, dtype=self.dtype, device=dev)
        kv1 = torch.zeros(1, self.head_dim, dtype=self.dtype, device=dev)
        idx1 = torch.zeros(1, DSA_TOPK_GRANULE, dtype=torch.int32, device=dev)  # slot 0 = row 0, the rest duplicates: a live row
        sink1 = torch.zeros(self.n_heads, dtype=torch.float32, device=dev)
        out1 = torch.empty_like(q1)
        lse1 = torch.empty(1, self.n_heads, dtype=torch.float32, device=dev)
        ml1 = torch.empty_like(lse1)
        stream = torch.cuda.current_stream(dev).cuda_stream
        self._impl.execute(q1, kv1, idx1, attn_sink=sink1, out=out1, lse=lse1, max_logits=ml1, current_stream=cuda.CUstream(stream))
        if self.batch > 1:
            # B > 1 ONLY: flat global row id of ``kv_all.view(B*N, D)`` = ``b * N + id`` for a valid id (dsa_stage_indices masks the
            # rest to -1).  At B == 1 the staging path (K % 64 != 0) is a plain copy + pad -- flat ids == local ids and the kernel's own
            # ``0 <= token < kv.shape[0]`` mask is the contract -- so ``base_col`` stays None and no ``idx_tmp`` scratch exists
            # (the block's ``_layout`` carves ``idx_tmp`` at B > 1 only; setting ``base_col`` here at B == 1 made
            # ``dsa_stage_indices`` demand it: RuntimeError on every B == 1, K % 64 != 0 shape, Rubin 2026-09-17).
            self._base_col = (torch.arange(self.batch, device=dev, dtype=torch.int32) * self.n_kv_rows).view(self.batch, 1, 1)

    # -- execute ------------------------------------------------------------

    def stage_indices(self, topk_idxs: torch.Tensor, idx_tmp: Optional[torch.Tensor], idx_ws: torch.Tensor) -> None:
        """D5: :func:`dsa_stage_indices` with this adapter's ``b * N`` column (set at ``compile``)."""
        if self.batch > 1 and (idx_tmp is None or self._base_col is None):
            raise RuntimeError("stage_indices at B > 1 needs the idx_tmp scratch and a compiled adapter")
        dsa_stage_indices(topk_idxs, idx_tmp, idx_ws, base_col=self._base_col, n_kv_rows=self.n_kv_rows, topk=self.topk, k_pad=self.k_pad)

    def execute(
        self,
        q_t: torch.Tensor,  # [T, H, D] contiguous (the block's q workspace view)
        kv2d: torch.Tensor,  # [B*N, D] contiguous (kv_all.view)
        idx2d: torch.Tensor,  # [T, K_pad] int32, 32-byte aligned
        sink: torch.Tensor,  # [H] fp32 contiguous
        out_t: torch.Tensor,  # [T, H, D] contiguous
        lse_th: torch.Tensor,  # [T, H] fp32 -- the kernel's LSE, sink EXCLUDED (fold it with dsa_lse_fold)
        max_logits: torch.Tensor,  # [T, H] fp32 -- required dead output
        *,
        stream: int,
    ) -> None:
        if self._impl is None:
            raise RuntimeError("call compile() before execute()")
        from cuda.bindings import driver as cuda

        self._impl.execute(q_t, kv2d, idx2d, attn_sink=sink, out=out_t, lse=lse_th, max_logits=max_logits, current_stream=cuda.CUstream(int(stream)))


# ---------------------------------------------------------------------------
# Adapter 3: the gathered-list fork of the Rubin d512 SDPA (wave B)
# ---------------------------------------------------------------------------


def d512_fork_available() -> bool:
    """True once ``kernels/sparse_attention_d512.py`` exists (feature-detected by import spec, never imported here)."""
    try:
        return importlib.util.find_spec(D512_FORK_MODULE) is not None
    except (ImportError, ValueError):
        return False


class D512SparseAttention:
    """Stage-7 adapter over the gathered-list d512 fork -- a typed decline until wave B lands it.

    Same plan-time keys as :class:`DsaSparseAttention` plus ``want_lse``; the
    execute ABI (``compiled(q, kv2d, o, lse, sinks, union_ids, union_bits,
    union_ntiles, scale_log2, stream)``) and the block-owned ``_UnionLists``
    pre-pass are wired when the fork module exists.
    """

    def __init__(
        self, *, batch: int, seq_len: int, n_kv_rows: int, n_heads: int, head_dim: int, topk: int, scale: float, dtype, device, want_lse: bool = False
    ) -> None:
        self.batch, self.seq_len, self.n_kv_rows = int(batch), int(seq_len), int(n_kv_rows)
        self.n_heads, self.head_dim, self.topk = int(n_heads), int(head_dim), int(topk)
        self.scale, self.dtype, self.device, self.want_lse = float(scale), dtype, torch.device(device), bool(want_lse)

    def check_support(self) -> None:
        if not d512_fork_available():
            raise NotImplementedError(
                f"attention='d512': the gathered-list d512 sparse-attention fork ({D512_FORK_MODULE}) has not landed; use attention='dsa' or 'torch'"
            )
        raise NotImplementedError(
            f"attention='d512': the fork module {D512_FORK_MODULE} is present but the block-side wiring (union-list pre-pass + compile ABI) has not landed"
        )

    def compile(self) -> None:
        raise NotImplementedError("attention='d512' has not landed (check_support declines first)")

    def execute(self, *args, **kwargs) -> None:
        raise NotImplementedError("attention='d512' has not landed (check_support declines first)")
