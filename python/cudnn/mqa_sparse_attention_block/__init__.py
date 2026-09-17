# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MQA sparse-attention block (one KV head shared by every query head, KV rows
selected per query by an index list, softmax sink, interleaved partial RoPE,
grouped LoRA output projection) -- a FROST frontend-only API.

Import-light on purpose: this module and ``geometry`` pull in dataclasses only;
``torch`` is imported lazily inside the load-time helpers, and the forward API
(``MqaSparseAttentionBlockFwd``, which imports ``torch`` and ``cuda.bindings``)
is resolved on first attribute access through ``__getattr__`` -- ``import
cudnn.mqa_sparse_attention_block`` stays torch-free.
"""

from .geometry import (
    KV_FAKE_QUANT_BLOCK,
    KV_FAKE_QUANT_MODES,
    MqaSparseAttentionBlockGeometry,
    YarnParams,
    build_grouped_wo_a,
    build_rope_tables,
)

# Torch-dependent public names, resolved lazily (name -> attribute of .api).
_LAZY_API = {
    "MqaSparseAttentionBlockFwd": "MqaSparseAttentionBlockFwd",
    "ATTENTION_IMPLS": "ATTENTION_IMPLS",
    "POINTWISE_IMPLS": "POINTWISE_IMPLS",
}


def __getattr__(name: str):
    if name in _LAZY_API:
        from . import api as _api

        value = getattr(_api, _LAZY_API[name])
        globals()[name] = value  # cache: the next access is a plain module attribute
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_API))


# Append-only.
__all__ = [
    "KV_FAKE_QUANT_BLOCK",
    "KV_FAKE_QUANT_MODES",
    "MqaSparseAttentionBlockGeometry",
    "YarnParams",
    "build_grouped_wo_a",
    "build_rope_tables",
    "MqaSparseAttentionBlockFwd",
    "ATTENTION_IMPLS",
    "POINTWISE_IMPLS",
]
