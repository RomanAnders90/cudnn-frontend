# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernels and kernel adapters of the MQA sparse-attention block.

Import-light on purpose: nothing here is imported by the package's ``__init__``;
``api.py`` pulls each module in lazily inside the stage that needs it, so a
missing optional dependency (or a module that has not landed yet -- the d512
fork, the norm/RoPE kernel) surfaces as a typed decline at ``check_support``
and never as an import error at ``import cudnn``.

Modules (ownership per the W1 plan):

* ``sparse_attention.py`` -- stage 7, the three swappable attention adapters
  (``torch`` oracle-shaped, ``dsa`` = the in-tree SM100 H64 kernel, ``d512`` =
  the gathered-list fork of the Rubin d512 SDPA -- a typed decline until it lands).
* ``norm_rope.py`` -- stages 2/4/6/8 as ONE row-local kernel (wave W2; the
  block's ``pointwise_impl="frost"`` arm imports it by the agreed interface).
* ``union_lists.py`` -- the block-owned index pre-pass of the d512 adapter (W3).
"""
