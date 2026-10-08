# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block selection of a block-sparse attention indexer: score every complete 4-token block, keep the top-k.

The sparse-attention indexer of the Qwen3.8 attention family scores a query token against the compressed
keys of the complete blocks it can see and keeps the ``top_k`` best blocks::

    score[t, b] = scale * sum_h relu(qi[t, h] . kbar[b])        for b < floor((pos_t + 1) / block_size)
    block_ids[t] = top_k(score[t, :])                            (local block ids of the token's sequence)

``qi`` holds a few indexer heads per token (4 for Qwen3.8-Flash-Next) and ``kbar`` one compressed key per
block (the block's mean-pooled, normalized, position-encoded raw keys -- produced upstream). The scale is the
constant ``1 / sqrt(head_dim)``: every head carries the same weight.

This is exactly the ratio-causal, ReLU-summed, head-reduced score the DSA indexer forward computes with its
compact-logits + fused top-k path (``cudnn.DSA.indexer_forward_top_k_wrapper``): the compression ratio is
the block size, the per-head weights are the constant scale, and the dense ``[T, n_blocks]`` score tensor is
never materialized -- the kernel writes only the ratio-causal candidates compactly and a radix pass selects
the top-k per token. :func:`qsa_select` is the thin host wrapper that spells the QSA geometry in that
vocabulary.

Contract (prefill, THD-packed queries):

* ``qi``: ``[T, n_heads, head_dim]`` bf16, the indexer queries of all sequences packed along ``T``; the
  token ``t`` of sequence ``s`` (``cu_seqlens_q[s] <= t < cu_seqlens_q[s + 1]``) sits at position
  ``pos_t = q_pos0[s] + (t - cu_seqlens_q[s])`` (``q_pos0 = None`` = every sequence starts at 0).
* ``kbar``: ``[B, n_blocks_max, head_dim]`` bf16, the compressed keys of every sequence, one row per
  complete block, padded to the longest sequence. A token reads rows ``[0, floor((pos_t + 1) / block_size))``
  of its sequence; the caller guarantees those rows are complete blocks (``n_blocks_s >=
  floor((q_pos0[s] + S_q[s]) / block_size)``) -- rows past them are never read.
* ``block_ids``: ``[T, top_k]`` int32, LOCAL block ids of the token's sequence, ``-1`` padded where a token
  sees fewer than ``top_k`` blocks (the first ``block_size * top_k + block_size - 1`` tokens of a sequence
  select every visible block -- the identity region), unsorted. ``scores`` are the selected logits
  (``-inf`` on the padding). With ``deterministic=True`` (the default) a tie at the k-th boundary keeps the
  smaller block id, so the selected SET is reproducible across runs; the slot order within a row is not.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional, Sequence, Union

import torch

from cudnn.api_base import TupleDict
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

__all__ = ["qsa_select", "QSA_BLOCK_SIZE", "QSA_TOP_K"]

QSA_BLOCK_SIZE = 4  # tokens per compressed block (the compression ratio of the indexer)
QSA_TOP_K = 512  # blocks kept per token (2048 selected tokens / 4)

_ones_cache: dict = {}
_ones_cache_lock = Lock()
_cu_k_cache: dict = {}
_cu_k_cache_lock = Lock()


def _constant_weights(numel: int, device: torch.device) -> torch.Tensor:
    """A cached bf16 ones buffer (power-of-two capacity buckets, never evicted) -- the constant head weight.

    Mirrors the DSA denominator placeholder: a CUDA graph may retain the address captured at warm-up, so a
    bucket is never replaced; one allocation per (device, capacity) for the life of the process.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    capacity = 1 << (max(1, numel) - 1).bit_length()
    key = (index, capacity)
    with _ones_cache_lock:
        buf = _ones_cache.get(key)
        if buf is None:
            buf = torch.ones((capacity,), dtype=torch.bfloat16, device=torch.device("cuda", index))
            _ones_cache[key] = buf
    return buf[:numel]


def _padded_cu_seqlens_k(batch: int, n_blocks_max: int, device: torch.device) -> torch.Tensor:
    """``[0, n, 2n, ..., B n]`` int32: the row offsets of a ``[B, n_blocks_max, D]`` key buffer seen as packed."""
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (index, batch, n_blocks_max)
    with _cu_k_cache_lock:
        t = _cu_k_cache.get(key)
        if t is None:
            t = torch.arange(batch + 1, dtype=torch.int64) * n_blocks_max
            if int(t[-1]) >= 2**31:
                raise ValueError(f"kbar holds {batch} x {n_blocks_max} rows; the packed row offsets must fit int32")
            t = t.to(dtype=torch.int32, device=torch.device("cuda", index))
            _cu_k_cache[key] = t
    return t


def _as_host_ints(values, name: str) -> list:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(values.shape)}")
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def qsa_select(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    top_k: int = QSA_TOP_K,
    block_size: int = QSA_BLOCK_SIZE,
    q_pos0: Optional[torch.Tensor] = None,
    n_blocks_per_seq: Optional[Union[Sequence[int], torch.Tensor]] = None,
    max_seqlen_q: Optional[int] = None,
    scale: Optional[float] = None,
    w: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    block_ids_out: Optional[torch.Tensor] = None,
    scores_out: Optional[torch.Tensor] = None,
    cand_buffer: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    stream=None,
) -> TupleDict:
    """Select the ``top_k`` compressed blocks per query token (see the module docstring for the contract).

    Args:
        qi: ``[T, n_heads, head_dim]`` bf16 indexer queries, THD-packed; ``n_heads`` in (4, 8, 16, 32, 64)
            (the groups the DSA unified scorer packs), ``head_dim == 128``. Any strides with a unit last
            stride are read in place (a column slice of a wider slab needs no copy).
        kbar: ``[B, n_blocks_max, head_dim]`` bf16 compressed keys, contiguous.
        cu_seqlens_q: ``[B + 1]`` int32 CUDA prefix of the query lengths (``T == cu_seqlens_q[-1]``).
        top_k: blocks kept per token (1..2048).
        block_size: tokens per compressed block (the compression ratio), 4 for QSA.
        q_pos0: ``[B]`` int32 CUDA, the position of each sequence's first query token (chunked prefill:
            the tokens already in the cache); ``None`` = 0 for every sequence.
        n_blocks_per_seq: optional HOST-visible block counts per sequence; when given, the geometry
            ``floor((q_pos0 + S_q) / block_size) <= n_blocks_per_seq <= n_blocks_max`` is checked on the
            host (a CUDA tensor here, or a CUDA ``q_pos0`` with it, is read back -- one sync; pass lists
            for a sync-free call). Device values are otherwise consumed as provided.
        max_seqlen_q: the longest query length; required under CUDA-graph capture (deriving it syncs).
        scale: the constant head weight; ``None`` = ``1 / sqrt(head_dim)``.
        w: optional ``[T, n_heads]`` bf16 or fp32 per-head weights replacing the constant (every head then
            weighs ``w[t, h] * scale``).
        deterministic: ties at the k-th boundary keep the smaller block id (reproducible sets).
        block_ids_out / scores_out: optional preallocated ``[T, top_k]`` int32 / fp32 outputs.
        cand_buffer / cand_batch_offsets: the compact-logits scratch from
            ``cudnn.DSA.compress_topk_cand_buffer_size_thd(cu_seqlens_q, cu_seqlens_k, block_size, q_pos0)``
            with ``cu_seqlens_k = arange(B + 1) * n_blocks_max``; both are required under graph capture.
        stream: optional ``cuda.CUstream``; ``None`` = torch's current stream.

    Returns:
        ``TupleDict(block_ids=[T, top_k] int32, scores=[T, top_k] fp32)``.

    Raises:
        NotImplementedError: a head count outside the packed groups, a device below the SM100 family, or a
            CuTe DSL below the floor / without the target (the message names the version or the set).
        ValueError: a malformed input (shape, dtype, device, prefix, geometry).
    """
    if not isinstance(qi, torch.Tensor) or qi.ndim != 3:
        raise ValueError(f"qi must be a [T, n_heads, head_dim] tensor, got {None if not isinstance(qi, torch.Tensor) else tuple(qi.shape)}")
    if not isinstance(kbar, torch.Tensor) or kbar.ndim != 3:
        raise ValueError(f"kbar must be a [B, n_blocks_max, head_dim] tensor, got {None if not isinstance(kbar, torch.Tensor) else tuple(kbar.shape)}")
    total_q, n_heads, head_dim = qi.shape
    batch, n_blocks_max, head_dim_k = kbar.shape
    if head_dim != 128 or head_dim_k != 128:
        raise ValueError(f"qsa_select scores head_dim 128 indexer heads, got qi head_dim {head_dim} and kbar head_dim {head_dim_k}")
    if qi.dtype != torch.bfloat16 or kbar.dtype != torch.bfloat16:
        raise ValueError(f"qi and kbar must be bfloat16, got {qi.dtype} and {kbar.dtype}")
    if not qi.is_cuda or kbar.device != qi.device:
        raise ValueError("qi and kbar must be CUDA tensors on one device")
    if qi.stride(-1) != 1:
        raise ValueError(f"qi needs a unit last stride, got strides {tuple(qi.stride())}")
    if not kbar.is_contiguous():
        raise ValueError(f"kbar must be contiguous, got strides {tuple(kbar.stride())}")
    if not isinstance(cu_seqlens_q, torch.Tensor) or cu_seqlens_q.ndim != 1 or cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_cuda:
        raise ValueError("cu_seqlens_q must be a 1-D int32 CUDA tensor of length B + 1")
    if cu_seqlens_q.numel() != batch + 1:
        raise ValueError(f"cu_seqlens_q has {cu_seqlens_q.numel()} entries; kbar has B = {batch} sequences, so B + 1 = {batch + 1} are needed")
    if cu_seqlens_q.device != qi.device:
        raise ValueError("cu_seqlens_q must be on qi's device")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not 1 <= top_k <= 2048:
        raise ValueError(f"top_k must be in [1, 2048], got {top_k}")
    if n_blocks_max < 1:
        raise ValueError("kbar must hold at least one block row")
    if q_pos0 is not None:
        if (
            not isinstance(q_pos0, torch.Tensor)
            or q_pos0.ndim != 1
            or q_pos0.numel() != batch
            or q_pos0.dtype != torch.int32
            or not q_pos0.is_cuda
            or q_pos0.device != qi.device
        ):
            raise ValueError(f"q_pos0 must be a [B={batch}] int32 CUDA tensor on qi's device")
    if w is not None:
        if not isinstance(w, torch.Tensor) or tuple(w.shape) != (total_q, n_heads) or w.dtype not in (torch.bfloat16, torch.float32) or w.device != qi.device:
            raise ValueError(f"w must be a [T={total_q}, n_heads={n_heads}] bf16 or fp32 tensor on qi's device")

    # Typed declines come before any device query or DSL import: the head count is a property of the request.
    from cudnn.deepseek_sparse_attention.indexer_forward._support import SUPPORTED_QHEAD_PER_KV_HEAD_BF16

    if n_heads not in SUPPORTED_QHEAD_PER_KV_HEAD_BF16:
        raise NotImplementedError(f"qsa_select packs n_heads in {SUPPORTED_QHEAD_PER_KV_HEAD_BF16} indexer heads per token, got {n_heads}")

    if n_blocks_per_seq is not None:
        counts = _as_host_ints(n_blocks_per_seq, "n_blocks_per_seq")
        if len(counts) != batch:
            raise ValueError(f"n_blocks_per_seq has {len(counts)} entries for B = {batch} sequences")
        cu_q = _as_host_ints(cu_seqlens_q, "cu_seqlens_q")
        pos0 = _as_host_ints(q_pos0, "q_pos0") if q_pos0 is not None else [0] * batch
        for s, n_blocks_s in enumerate(counts):
            seq_q = cu_q[s + 1] - cu_q[s]
            needed = (pos0[s] + seq_q) // block_size
            if not needed <= n_blocks_s <= n_blocks_max:
                raise ValueError(
                    f"sequence {s}: {seq_q} query tokens from position {pos0[s]} see floor(({pos0[s]} + {seq_q}) / {block_size}) = {needed} "
                    f"complete blocks, but n_blocks_per_seq = {n_blocks_s} and kbar holds {n_blocks_max} rows per sequence"
                )

    # The DSL floor and target, then the device family (never a DSL- or kernel-internal error).
    dsl_problem = cutedsl_requirement_error("qsa_select")
    if dsl_problem is not None:
        raise NotImplementedError(dsl_problem)
    device_cc = tuple(torch.cuda.get_device_capability(qi.device))
    arch_problem = cutedsl_arch_requirement_error(device_cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if device_cc[0] != 10:
        raise NotImplementedError(f"qsa_select runs the SM100-family indexer scorer (cc 10.x); found cc {device_cc[0]}.{device_cc[1]}")

    from cudnn.deepseek_sparse_attention.indexer_forward import indexer_forward_top_k_wrapper

    if scale is None:
        scale = head_dim**-0.5
    weights = w if w is not None else _constant_weights(total_q * n_heads, qi.device).view(total_q, n_heads)
    cu_seqlens_k = _padded_cu_seqlens_k(batch, n_blocks_max, qi.device)
    result = indexer_forward_top_k_wrapper(
        qi,
        kbar.view(batch * n_blocks_max, 1, head_dim),
        weights,
        top_k=top_k,
        ratio=block_size,
        qhead_per_kv_head=n_heads,
        sm_scale=float(scale),
        stream=stream,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=n_blocks_max,
        q_causal_offsets=q_pos0,
        topk_indices_global=False,
        cand_buffer=cand_buffer,
        out_indices=block_ids_out,
        out_logits=scores_out,
        cand_batch_offsets=cand_batch_offsets,
        return_softmax=False,
        deterministic=deterministic,
    )
    return TupleDict(block_ids=result["indices"], scores=result["logits"])
