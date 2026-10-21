# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_list_union``: the DSL-free host entry of the block-sparse decode mode's list-union kernel (``kernels/qsa_list_union.py``).

The union list form is the input contract of the block-sparse d256 forward's UNION rendering (``sdpa/fwd/sparse_gqa_sm107``,
``union_rows=S_q``): a sequence's ``S_q <= 4`` per-row block lists -- the decode top-k's canonical lists, or a caller's lists in any
order -- merged into ONE list of DISTINCT block ids in strictly descending order, each id carrying the nibble of rows that list it
(``word = (blk << 4) | mask``, ``mask`` bit ``j`` = row ``j`` lists ``blk``), ``-1`` past ``union_lens[b]``.  The union tile gathers a
sequence's listed KV ONCE for all its rows and masks each row's columns with its bit; the function it computes is the per-row one
(every row sees exactly its own list plus its own open block).  Duplicates INSIDE a row (a list contract violation) collapse to one
entry here -- the one place the union tile and the per-row tile differ (documented in both kernels).

Rule 7: this module imports no CuTe DSL -- torch and the version gate only -- so the documented import path resolves on every
installed DSL; :func:`qsa_block_list_union` checks every operand's FORM, gates the DSL floor (a typed ``NotImplementedError`` naming
the version) and only THEN imports the kernel module inside the function body.  Rule 5: the launch runs on the stream the caller
names (a ``torch.cuda.Stream``, a raw handle or a ``cuda.CUstream``; ``None`` = torch's current stream on the operands' device), no
allocation, no host read of a list value (Rule 3: the counts, the dummies and the ranks are device facts).
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch

from cudnn._torch_stream import launch_stream
from cudnn.frost.buffers import cutedsl_requirement_error

__all__ = [
    "QSA_LIST_UNION_KERNEL_MODULE",
    "UNION_LIST_WIDTH",
    "UNION_MAX_ROWS",
    "UNION_ID_BITS",
    "UNION_TOP_K_MAX",
    "qsa_block_list_union",
    "qsa_list_union_kernel_module",
    "union_list_words",
]

QSA_LIST_UNION_KERNEL_MODULE = "cudnn.gated_attention_block.kernels.qsa_list_union"
"""The kernel module's dotted path -- imported lazily by :func:`qsa_list_union_kernel_module`, never at import time here."""
UNION_LIST_WIDTH = 2048
"""The union list's width in int32 words (``4 rows x 512`` ids: the sort capacity of the kernel's one CTA per sequence)."""
UNION_MAX_ROWS = 4
"""The rows per sequence a union word's mask nibble carries."""
UNION_ID_BITS = 27
"""Block ids below ``2^27`` keep the word ``(blk << 4) | mask`` inside an int32 (the declaration's table bounds the ids)."""
UNION_TOP_K_MAX = 512


def qsa_list_union_kernel_module():
    """The kernel module, imported once the CuTe DSL floor is met (Rule 7: the gate runs HERE, before the import).  Raises
    ``NotImplementedError`` naming the installed version when it is below the floor."""
    too_old = cutedsl_requirement_error("qsa_list_union")
    if too_old is not None:
        raise NotImplementedError(too_old)
    return importlib.import_module(QSA_LIST_UNION_KERNEL_MODULE)


def union_list_words(block_ids: torch.Tensor, block_lens: Optional[torch.Tensor]) -> int:
    """The number of union words a ``[B, S_q, top_k]`` list block needs: the kernel's fixed width (every slot of a row is written)."""
    if block_ids.dim() != 3:
        raise ValueError(f"block_ids must be [B, S_q, top_k], got {tuple(block_ids.shape)}")
    return UNION_LIST_WIDTH


def _check_form(block_ids, block_lens, out_ids, out_lens, n_blocks_max) -> None:
    if not isinstance(block_ids, torch.Tensor) or block_ids.dim() != 3 or block_ids.dtype != torch.int32:
        raise ValueError(
            f"block_ids must be a [B, S_q, top_k] int32 tensor, got {getattr(block_ids, 'shape', type(block_ids).__name__)} {getattr(block_ids, 'dtype', '')}"
        )
    if not block_ids.is_cuda:
        raise ValueError(f"block_ids must be a CUDA tensor, got device {block_ids.device}")
    b, s_q, top_k = (int(x) for x in block_ids.shape)
    if not 1 <= s_q <= UNION_MAX_ROWS:
        raise NotImplementedError(f"qsa_list_union serves S_q in [1, {UNION_MAX_ROWS}] rows per sequence (the mask nibble); got S_q = {s_q}")
    if not 1 <= top_k <= UNION_TOP_K_MAX:
        raise NotImplementedError(f"qsa_list_union serves top_k in [1, {UNION_TOP_K_MAX}]; got top_k = {top_k}")
    if s_q * top_k > UNION_LIST_WIDTH:
        raise NotImplementedError(f"qsa_list_union: S_q x top_k = {s_q * top_k} exceeds the union capacity {UNION_LIST_WIDTH}")
    if block_ids.stride(2) != 1 or block_ids.stride(1) != top_k:
        raise ValueError(
            f"block_ids rows must be contiguous ([B, S_q, top_k] with the S_q x top_k entries of a sequence adjacent), got strides {tuple(block_ids.stride())}"
        )
    if block_ids.data_ptr() % 16 != 0 or (block_ids.stride(0) * 4) % 16 != 0:
        raise ValueError("block_ids must start 16-byte aligned with a 16-byte-aligned sequence stride")
    device = block_ids.device
    if block_lens is not None:
        if not isinstance(block_lens, torch.Tensor) or block_lens.dtype != torch.int32 or tuple(block_lens.shape) != (b, s_q) or not block_lens.is_contiguous():
            raise ValueError(
                f"block_lens must be a contiguous int32 [B, S_q] = {(b, s_q)} tensor or None, got {getattr(block_lens, 'shape', type(block_lens).__name__)} {getattr(block_lens, 'dtype', '')}"
            )
        if block_lens.device != device:
            raise ValueError(f"block_lens must live on {device}, got {block_lens.device}")
    for name, t, shape in (("out_ids", out_ids, (b, UNION_LIST_WIDTH)), ("out_lens", out_lens, (b,))):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous int32 {shape} tensor, got {getattr(t, 'shape', type(t).__name__)} {getattr(t, 'dtype', '')}")
        if t.device != device:
            raise ValueError(f"{name} must live on {device}, got {t.device}")
    if n_blocks_max is not None and int(n_blocks_max) > (1 << UNION_ID_BITS):
        raise NotImplementedError(
            f"qsa_list_union: block ids must be below 2^{UNION_ID_BITS} (the union word (blk << 4) | mask is an int32); n_blocks_max={n_blocks_max} reaches past it"
        )


def qsa_block_list_union(
    block_ids: torch.Tensor,
    block_lens: Optional[torch.Tensor] = None,
    *,
    out_ids: torch.Tensor,
    out_lens: torch.Tensor,
    n_blocks_max: Optional[int] = None,
    stream=None,
) -> torch.Tensor:
    """Merge the per-row block lists of every sequence into its canonical union list (one launch; the CALLER's buffers).

    ``block_ids [B, S_q, top_k]`` int32 (rows contiguous; any order inside a row; ``-1`` = no entry), ``block_lens [B, S_q]`` int32 or
    None (entries at index ``>= block_lens[b, j]`` ignored; None = every non-negative entry counts), ``out_ids [B, 2048]`` int32 -- the
    words ``(blk << 4) | mask`` in strictly descending block order over ``[0, out_lens[b])``, ``-1`` past it, every slot written --,
    ``out_lens [B]`` int32 = the distinct-block count; ``n_blocks_max`` (optional) is the caller's bound on the ids, declined by name
    past ``2^UNION_ID_BITS``; ``stream`` is the launch stream (``None`` = torch's current stream on the operands' device).  Returns
    ``out_ids``.  Two launches are bitwise.
    """
    _check_form(block_ids, block_lens, out_ids, out_lens, n_blocks_max)
    mod = qsa_list_union_kernel_module()
    if int(block_ids.shape[0]) == 0:
        return out_ids
    with torch.cuda.device(block_ids.device):
        launch = launch_stream(stream, block_ids.device)
        recipe = mod.compile_qsa_list_union(s_q=int(block_ids.shape[1]), top_k=int(block_ids.shape[2]), has_lens=block_lens is not None)
        mod.run_qsa_list_union(recipe, block_ids, block_lens, out_ids, out_lens, stream=launch.cuda_stream)
    return out_ids
