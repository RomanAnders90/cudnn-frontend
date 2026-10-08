# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SM100-family contract helpers of the indexer forward (dense and compressed-logits paths).

One place for the rules every entry point of ``indexer_forward`` applies, so the APIBase class, the dense
interface and the compressed-logits orchestration cannot drift apart.
"""

from __future__ import annotations


def check_q_covered_by_k(seqlen_q: int, seqlen_k: int, ratio: int, *, what_q: str = "seqlen_q", what_k: str = "seqlen_k") -> None:
    """Geometry check: the longest query row must not see more compressed keys than ``seqlen_k`` holds.

    Under the top-left ratio-causal mask row ``r`` sees ``floor((r + 1) / ratio)`` compressed keys, so the
    last row of ``seqlen_q`` rows sees ``floor(seqlen_q / ratio)``; that count fits ``seqlen_k`` keys exactly
    when ``seqlen_q <= seqlen_k * ratio + (ratio - 1)``. The ``ratio - 1`` trailing tokens are the ones that
    have not completed a compressed block yet (a 4:1 compression of 2051 tokens holds 512 keys): they score
    against the complete blocks only and are a legitimate geometry, not a mismatch.
    """
    if seqlen_q > seqlen_k * ratio + (ratio - 1):
        raise ValueError(
            f"{what_q} ({seqlen_q}) must be <= {what_k} * ratio + (ratio - 1) ({seqlen_k * ratio + ratio - 1}): "
            f"its last row would see more compressed keys than {what_k} holds"
        )
