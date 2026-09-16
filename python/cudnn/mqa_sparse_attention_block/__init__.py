# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MQA sparse-attention block (one KV head shared by every query head, KV rows
selected per query by an index list, softmax sink, interleaved partial RoPE,
grouped LoRA output projection) -- a FROST frontend-only API.

Import-light on purpose: this module and ``geometry`` pull in dataclasses only;
``torch`` is imported lazily inside the load-time helpers.
"""

from .geometry import (
    KV_FAKE_QUANT_BLOCK,
    KV_FAKE_QUANT_MODES,
    MqaSparseAttentionBlockGeometry,
    YarnParams,
    build_grouped_wo_a,
    build_rope_tables,
)

# Append-only.
__all__ = [
    "KV_FAKE_QUANT_BLOCK",
    "KV_FAKE_QUANT_MODES",
    "MqaSparseAttentionBlockGeometry",
    "YarnParams",
    "build_grouped_wo_a",
    "build_rope_tables",
]
