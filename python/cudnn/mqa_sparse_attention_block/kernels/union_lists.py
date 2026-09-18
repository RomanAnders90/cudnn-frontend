# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block-owned index pre-pass for the gathered-list d512 attention kernel: per 4-token
cluster, the MULTISET union of the four tokens' key lists (ascending flat KV rows), a
per-slot membership bit table, and the tile count -- three read-only int32 workspace
tensors the kernel's TMA-LDG warp gathers from and the softmax lanes mask with.

The kernel is a generic "gathered-list attention with per-row membership"; EVERY list rule
lives here: ``-1`` = no key, duplicates are distinct slots and count TWICE, compressed ids
arrive already offset by ``S`` (the block's ``kv_all`` row space), out-of-range ids are
dropped, tokens past ``seq_len`` (the tail cluster when ``seq_len % 4 != 0``) have no keys.

Layout contract (the kernel's ABI -- change both sides or neither):

``union_ids``    ``[B, NC, U_MAX]`` int32, ``U_MAX = 128 * u_max_tiles``: the cluster's union as FLAT
                 rows ``b * n_kv_rows + id`` of the ``[B * n_kv_rows, D]`` KV view, ascending by
                 ``(id, copy)``, ``-1``-padded to ``U_MAX``.  A ``-1`` is a TMA-OOB row: zero-filled,
                 bytes counted.
``union_bits``   ``[B, NC, u_max_tiles, 4, 4]`` int32 -- tile ``t``, token slot ``j`` (token ``4c + j``),
                 word ``w``: bit ``i`` set iff union column ``128 t + 32 w + i`` is one of token
                 ``4c + j``'s copies.  One tile's four slots are one 64-B line; a softmax lane loads
                 its slot's 16 B once per tile and chunk ``k`` (64 columns) uses words ``2k, 2k+1``.
``union_ntiles`` ``[B, NC]`` int32, ``max(1, ceil(|union| / 128))`` -- NEVER 0, so the kernel's
                 per-cluster KV loop always runs (an all-``-1`` cluster costs one zero-MMA tile
                 and the count-based empty-row select handles the rest).

Multiset rule: id ``x`` appears ``max_j mult_j(x)`` times; copy ``k`` (0-based, ascending) carries
bit ``j`` iff ``mult_j(x) > k``.  Sanitised ids (negative or ``>= n_kv_rows``) are dropped AND
their bits cleared -- a zero-filled row with a set bit would be a real logit of 0.

v1 = torch (this file): pad, sanitise, sort ``id * 4 + j`` keys, run-length + copy rank via
``diff`` / ``cummax``, ``scatter_add`` the bits.  FLAGGED: ~10 launches and several temporaries
per call, i.e. a per-execute allocation the stage that calls it cannot claim contract-clean
(frost-engine-contract.md section 10); the perf driver reports it as a ``kind='prep'`` row.
v2 (planned): one CuTe warp per cluster, bitonic sort in SMEM, coalesced 16-B stores into the
same three views -- same contract, no allocation.
"""

from __future__ import annotations

import math

TOKENS_PER_CLUSTER = 4
TILE_ROWS = 128
# The d512 fork indexes union columns with a 15-bit quantity (its `_validate_sparse_cfg` re-checks `u_max_tiles * TILE_ROWS <
# U_MAX_COLS_LIMIT`); the adapter's K bound and the pre-pass bound below import THIS constant rather than re-spelling 2**15.
U_MAX_COLS_LIMIT = 2**15
WORDS_PER_SLOT = TILE_ROWS // 32  # 4 x int32 = 128 membership bits per (tile, slot)
_INT32_MAX = 2**31 - 1


def u_max_tiles_for(topk: int) -> int:
    """Smallest tile count that holds a cluster's worst-case union (4 full lists, no overlap)."""
    if topk < 0:
        raise ValueError(f"topk must be >= 0, got {topk}")
    return max(1, math.ceil(TOKENS_PER_CLUSTER * topk / TILE_ROWS))


def n_clusters_for(seq_len: int) -> int:
    """Clusters of 4 adjacent tokens; the tail cluster is partial when ``seq_len % 4 != 0``."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    return math.ceil(seq_len / TOKENS_PER_CLUSTER)


def validate_union_lists_shapes(topk_idxs, *, seq_len: int, n_kv_rows: int, u_max_tiles: int, out_ids, out_bits, out_ntiles) -> None:
    """Raise ``ValueError`` unless the three workspace views match the contract for this list."""
    import torch

    if topk_idxs.dim() != 3:
        raise ValueError(f"topk_idxs must be [B, S, K], got shape {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32, got {topk_idxs.dtype}")
    b, s, k = topk_idxs.shape
    if s != seq_len:
        raise ValueError(f"topk_idxs.shape[1] ({s}) must equal seq_len ({seq_len})")
    if n_kv_rows < 1:
        raise ValueError(f"n_kv_rows must be >= 1, got {n_kv_rows}")
    if n_kv_rows * b > _INT32_MAX:
        raise ValueError(f"flat KV row space B * n_kv_rows = {b * n_kv_rows} does not fit int32")
    need = u_max_tiles_for(k)
    if u_max_tiles < need:
        raise ValueError(f"u_max_tiles ({u_max_tiles}) cannot hold 4 x K = {4 * k} rows; need >= {need}")
    if u_max_tiles * TILE_ROWS >= U_MAX_COLS_LIMIT:
        raise ValueError(f"u_max_tiles * 128 = {u_max_tiles * TILE_ROWS} must stay below 2**15 = {U_MAX_COLS_LIMIT} (the kernel's column index)")
    nc = n_clusters_for(seq_len)
    exp = {
        "union_ids": (out_ids, (b, nc, u_max_tiles * TILE_ROWS)),
        "union_bits": (out_bits, (b, nc, u_max_tiles, TOKENS_PER_CLUSTER, WORDS_PER_SLOT)),
        "union_ntiles": (out_ntiles, (b, nc)),
    }
    for name, (t, shape) in exp.items():
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.device != topk_idxs.device:
            raise ValueError(f"{name} must live on {topk_idxs.device}, got {t.device}")


def build_union_lists(topk_idxs, *, seq_len: int, n_kv_rows: int, u_max_tiles: int, out_ids, out_bits, out_ntiles) -> None:
    """Fill ``out_ids`` / ``out_bits`` / ``out_ntiles`` (see the module docstring) from ``topk_idxs`` [B, S, K] int32.

    ``n_kv_rows`` is the per-batch row count of the KV view (``S + N_c``); ids outside ``[0, n_kv_rows)``
    (including ``-1``) are dropped.  Stream-ordered torch ops on ``topk_idxs.device``; the outputs are
    fully overwritten.  FLAGGED v1: allocates temporaries (see the module docstring).
    """
    import torch

    validate_union_lists_shapes(
        topk_idxs, seq_len=seq_len, n_kv_rows=n_kv_rows, u_max_tiles=u_max_tiles, out_ids=out_ids, out_bits=out_bits, out_ntiles=out_ntiles
    )
    b, s, k = topk_idxs.shape
    nc = n_clusters_for(seq_len)
    u_max = u_max_tiles * TILE_ROWS
    dev = topk_idxs.device
    if nc == 0 or k == 0:
        out_ids.fill_(-1)
        out_bits.zero_()
        out_ntiles.fill_(1)
        return

    # 1. pad the token axis to NC * 4 (tokens >= S get no keys) -> [B, NC, 4*K] int64 with slot j = pos // K
    idx = topk_idxs.to(torch.int64)
    pad = nc * TOKENS_PER_CLUSTER - s
    if pad:
        idx = torch.cat([idx, torch.full((b, pad, k), -1, dtype=torch.int64, device=dev)], dim=1)
    idx = idx.reshape(b, nc, TOKENS_PER_CLUSTER * k)
    slot = torch.arange(TOKENS_PER_CLUSTER * k, device=dev) // k  # [4K]

    # 2. sanitise: everything outside [0, n_kv_rows) sorts to the tail as a sentinel
    valid = (idx >= 0) & (idx < n_kv_rows)
    ids = torch.where(valid, idx, torch.full_like(idx, _INT32_MAX))

    # 3. sort (id, slot) keys; the rank of an entry inside its (id, slot) run is its COPY index k
    key = ids * TOKENS_PER_CLUSTER + slot
    key_sorted, perm = key.sort(dim=-1)
    slot_sorted = slot.expand_as(key).gather(-1, perm)
    ar = torch.arange(TOKENS_PER_CLUSTER * k, device=dev).expand_as(key_sorted)
    run_start = torch.ones_like(key_sorted, dtype=torch.bool)
    run_start[..., 1:] = key_sorted[..., 1:] != key_sorted[..., :-1]
    first = torch.where(run_start, ar, torch.zeros_like(ar)).cummax(dim=-1).values
    copy_rank = ar - first  # < K
    id_sorted = key_sorted // TOKENS_PER_CLUSTER

    # 4. the union entry (id, copy) -- dedupe across slots, order ascending by (id, copy); its position
    #    is the union column.  Sentinel ids sort last, so the valid prefix is contiguous.
    key2 = id_sorted * k + copy_rank
    key2_sorted, perm2 = key2.sort(dim=-1)
    slot2 = slot_sorted.gather(-1, perm2)
    valid2 = key2_sorted < _INT32_MAX * k  # (any sentinel key2 >= _INT32_MAX * k)
    uniq = torch.ones_like(key2_sorted, dtype=torch.bool)
    uniq[..., 1:] = key2_sorted[..., 1:] != key2_sorted[..., :-1]
    col = uniq.to(torch.int64).cumsum(dim=-1) - 1  # union column of every entry
    count = (uniq & valid2).sum(dim=-1)  # |union| per cluster; <= 4K <= u_max by validate_union_lists_shapes, so no
    #    device->host check here (an `int(count.max())` would sync the stream and fail under CUDA-graph capture)

    # 5. ids: flat rows b * n_kv_rows + id at the union columns.  Every entry is scattered: duplicate
    #    (id, copy) entries write the SAME value to the same column, and the sentinel entries sort past
    #    the valid prefix (their columns are >= count, < 4K <= U_MAX) and write the -1 pad there.
    out_ids.fill_(-1)
    batch_off = (torch.arange(b, device=dev) * n_kv_rows).view(b, 1, 1)
    flat = (key2_sorted // k) + batch_off
    out_ids.view(b, nc, u_max).scatter_(-1, col, torch.where(valid2, flat, torch.full_like(flat, -1)).to(torch.int32))

    # 6. bits: one (id, copy, slot) entry per set bit -> tile = col // 128, word = (col % 128) // 32, bit = col % 32
    bits64 = torch.zeros(b, nc, u_max_tiles * TOKENS_PER_CLUSTER * WORDS_PER_SLOT, dtype=torch.int64, device=dev)
    tile = col // TILE_ROWS
    word = (col % TILE_ROWS) // 32
    bit = col % 32
    lin = (tile * TOKENS_PER_CLUSTER + slot2) * WORDS_PER_SLOT + word
    contrib = torch.where(valid2, torch.ones_like(col) << bit, torch.zeros_like(col))
    bits64.scatter_add_(-1, torch.where(valid2, lin, torch.zeros_like(lin)), contrib)
    bits32 = torch.where(bits64 >= 2**31, bits64 - 2**32, bits64)  # two's-complement int32 of the low word
    out_bits.view(b, nc, -1).copy_(bits32.to(torch.int32))

    # 7. tiles: never 0
    out_ntiles.copy_(torch.clamp((count + TILE_ROWS - 1) // TILE_ROWS, min=1).to(torch.int32))
