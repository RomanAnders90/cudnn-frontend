# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_list_union``: a sequence's per-row block lists merged into ONE canonical union list with a per-row membership nibble per
block (``kernels/qsa_list_union.py`` through its DSL-free host entry ``gated_attention_block/qsa_list_union.py``).

Every accept cell compares the kernel BITWISE against the plain-torch reference (``gated_block_qsa_reference.union_words_reference``:
the distinct ids over the rows' valid entries, descending, the mask OR, ``-1`` past the count) and launches twice (bitwise); the
inverse (``per_row_from_union``) recovers every canonical input, so the union LOSES nothing of a contract-clean list.  The kernel is
arch-agnostic (plain CuTe DSL, no tensor core), so the cells run on any CUDA device with the DSL target; a too-old DSL skips the
module with the version named (Rule 7), and the facade's decline below the floor is pinned without importing the kernel module.
"""

import importlib
import os
import sys

import pytest
import torch

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from gated_block_census import cupti_kernels  # noqa: E402
from gated_block_qsa_reference import UNION_LIST_WIDTH, full_block_ids, per_row_from_union, random_block_ids, union_words_reference  # noqa: E402

_CUTEDSL_REQUIREMENT = cutedsl_requirement_error("qsa_list_union")
pytestmark = [pytest.mark.L0, pytest.mark.skipif(_CUTEDSL_REQUIREMENT is not None, reason=str(_CUTEDSL_REQUIREMENT))]


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device is needed")
_TARGET = cutedsl_arch_requirement_error(_cc()) if _cc() is not None else None
needs_dsl_target = pytest.mark.skipif(_TARGET is not None, reason=str(_TARGET))

BS = 4
SENTINEL = -2  # never produced by the kernel (words are >= 0, the padding is -1): a slot still holding it after a launch was never written
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _entry():
    from cudnn.gated_attention_block.qsa_list_union import qsa_block_list_union

    return qsa_block_list_union


def _lists(B, s_q, top_k, kv_lens, *, kind, seed):
    """Per-row lists at the decode positions ``kv_len - s_q + j``: ``full`` (every complete block in order), ``shuffled`` (a random
    ``top_k``-subset of the complete blocks in random order), ``canonical`` (the shuffled set sorted descending = the top-k's output form),
    ``disjoint`` (row ``j`` lists the blocks congruent to ``j`` mod ``s_q``: no two rows share a block)."""
    g = torch.Generator(device=_DEV).manual_seed(seed)
    kvl = torch.tensor(kv_lens, device=_DEV, dtype=torch.long)
    pos = (kvl[:, None] - s_q) + torch.arange(s_q, device=_DEV)[None, :]
    if kind == "full":
        ids, lens = full_block_ids(pos, top_k, BS, kv_lens=kvl[:, None])
    elif kind in ("shuffled", "canonical"):
        ids, lens = random_block_ids(pos, top_k, BS, generator=g, kv_lens=kvl[:, None], shuffle=True)
        if kind == "canonical":
            srt = torch.sort(torch.where(ids >= 0, ids, torch.full_like(ids, -1)), dim=-1, descending=True).values
            ids = srt
    elif kind == "disjoint":
        ids = torch.full((B, s_q, top_k), -1, dtype=torch.int32, device=_DEV)
        lens = torch.zeros(B, s_q, dtype=torch.int32, device=_DEV)
        for b in range(B):
            for j in range(s_q):
                n_complete = max(int(pos[b, j]) + 1, 0) // BS
                mine = torch.arange(n_complete - 1, -1, -1, device=_DEV)
                mine = mine[(mine % s_q) == j][:top_k]
                ids[b, j, : mine.numel()] = mine.to(torch.int32)
                lens[b, j] = mine.numel()
    else:
        raise ValueError(kind)
    return ids.contiguous(), lens.contiguous()


def _run(ids, lens, *, n_blocks_max=None, launches=2):
    B = int(ids.shape[0])
    outs = []
    for _ in range(launches):
        out = torch.full((B, UNION_LIST_WIDTH), SENTINEL, dtype=torch.int32, device=ids.device)
        out_lens = torch.full((B,), SENTINEL, dtype=torch.int32, device=ids.device)
        _entry()(ids, lens, out_ids=out, out_lens=out_lens, n_blocks_max=n_blocks_max)
        torch.cuda.synchronize()
        outs.append((out, out_lens))
    return outs


def _check(ids, lens, *, s_q, top_k, n_blocks_max=None, inverse=True):
    (out, out_lens), *rest = _run(ids, lens, n_blocks_max=n_blocks_max)
    assert bool((out != SENTINEL).all()) and bool((out_lens != SENTINEL).all()), "a slot was never written (the sentinel survives)"
    ref, ref_lens = union_words_reference(ids, lens)
    assert torch.equal(out_lens, ref_lens), f"union_lens differ: {out_lens.tolist()[:8]} vs {ref_lens.tolist()[:8]}"
    if not torch.equal(out, ref):
        bad = (out != ref).nonzero()
        b, j = (int(x) for x in bad[0])
        raise AssertionError(
            f"{bad.shape[0]} words differ; first at sequence {b} slot {j}: got {int(out[b, j])}, want {int(ref[b, j])} (|U| {int(ref_lens[b])})"
        )
    for o2, l2 in rest:
        assert torch.equal(out, o2) and torch.equal(out_lens, l2), "two launches differ"
    # the words' structure: strictly descending ids, a non-zero nibble on every live word, -1 past the count
    for b in range(int(ids.shape[0])):
        n = int(out_lens[b])
        w = out[b, :n].to(torch.long)
        assert bool((w >= 0).all()) and bool(((w & 15) != 0).all())
        blk = w >> 4
        assert bool((blk[1:] < blk[:-1]).all()), "the union ids must be strictly descending"
        assert bool((out[b, n:] == -1).all())
    if inverse:
        back_ids, back_lens = per_row_from_union(out, out_lens, s_q, top_k)
        canon = torch.sort(torch.where(ids >= 0, ids, torch.full_like(ids, -1)), dim=-1, descending=True).values
        valid_lens = lens if lens is not None else (ids >= 0).sum(-1).to(torch.int32)
        if lens is not None:
            # entries at or past the count are ignored: canonicalise the reference the same way
            keep = torch.arange(top_k, device=ids.device)[None, None, :] < lens.to(torch.long).clamp(0, top_k)[:, :, None]
            canon = torch.sort(torch.where(keep & (ids >= 0), ids, torch.full_like(ids, -1)), dim=-1, descending=True).values
            valid_lens = (keep & (ids >= 0)).sum(-1).to(torch.int32)
        assert torch.equal(back_ids, canon) and torch.equal(back_lens, valid_lens), "the inverse must recover every row's canonical list"
    return out, out_lens


# --------------------------------------------------------------------------------------------- accept cells


@needs_cuda
@needs_dsl_target
@pytest.mark.parametrize("top_k", [4, 64, 512])
@pytest.mark.parametrize("s_q", [1, 2, 3, 4])
@pytest.mark.parametrize("B", [1, 2, 32])
def test_matches_the_reference_on_canonical_and_shuffled_lists(B, s_q, top_k):
    """The union of ``s_q`` decode-position lists per sequence: canonical input (the top-k's form) and shuffled input, per-row
    ``block_lens`` and None, kv lengths that saturate the list, sit inside the identity region and leave a row with no complete block."""
    kv_lens = (
        [8192 + s_q, 2048 + 3, 61, s_q + 2, 4 * top_k + 1][:B] if B <= 5 else [8192 + s_q, 2048 + 3, 61, s_q + 2] + [37 * (i + 1) + s_q for i in range(B - 4)]
    )
    for kind in ("canonical", "shuffled"):
        ids, lens = _lists(B, s_q, top_k, kv_lens, kind=kind, seed=7 * s_q + top_k + B)
        _check(ids, lens, s_q=s_q, top_k=top_k)
        _check(ids, None, s_q=s_q, top_k=top_k)


@needs_cuda
@needs_dsl_target
def test_full_lists_overlap_completely_and_disjoint_lists_fill_the_capacity():
    """Identical lists (the identity region: every row lists the same complete blocks -> |U| = top_k with every row's bit on every word
    the rows share); disjoint lists at S_q 4 x top_k 512 (|U| = 2048, no padding, each word ONE bit)."""
    ids, lens = _lists(2, 4, 512, [2051, 4096], kind="full", seed=1)
    out, out_lens = _check(ids, lens, s_q=4, top_k=512)
    assert int(out_lens[0]) == 512 and bool(((out[0, :512] & 15) == 15).all()), "the identity region: every row lists the same 512 blocks"
    ids, lens = _lists(3, 4, 512, [8192 + 4, 9000, 16384], kind="disjoint", seed=2)
    out, out_lens = _check(ids, lens, s_q=4, top_k=512)
    assert bool((out_lens == UNION_LIST_WIDTH).all()), out_lens.tolist()
    nib = out.to(torch.long) & 15
    assert bool(((nib & (nib - 1)) == 0).all()), "disjoint lists: exactly one bit per word"


@needs_cuda
@needs_dsl_target
def test_empty_rows_dead_sequences_and_counts_out_of_range():
    """A row of all -1 (no entry), a sequence shorter than the step (every row empty: |U| = 0, every slot -1), ``block_lens`` above the
    row's prefix (clamped: the -1 entries past the prefix never count), below it (the entries at or past the count ignored), negative
    (clamped to 0: the row contributes nothing)."""
    ids, lens = _lists(4, 4, 64, [300, 3, 0, 1000], kind="canonical", seed=3)
    ids[0, 2] = -1  # one empty row inside a live sequence
    _check(ids, lens, s_q=4, top_k=64)
    out, out_lens = _check(ids, None, s_q=4, top_k=64, inverse=False)
    assert int(out_lens[1]) == 0 and int(out_lens[2]) == 0 and bool((out[1] == -1).all()) and bool((out[2] == -1).all())
    big = lens.clone()
    big[:, 0] = 10_000  # above the prefix: clamped to top_k, the -1 entries never count
    big[:, 1] = torch.minimum(lens[:, 1], torch.tensor(5, dtype=torch.int32, device=_DEV))  # below the prefix: the tail is ignored
    big[:, 3] = -7  # negative: the row contributes nothing
    _check(ids, big, s_q=4, top_k=64)


@needs_cuda
@needs_dsl_target
def test_duplicates_inside_a_row_collapse_to_one_entry_the_documented_difference():
    """A list contract violation: block 5 listed twice by row 1 -- the union holds it ONCE with bit 1 set once (the per-row tile would
    attend it twice; the union tile once: the one case the two renderings differ on, documented in both kernels' headers)."""
    ids, lens = _lists(1, 4, 64, [4096], kind="canonical", seed=4)
    ids[0, 1, 0] = 5
    ids[0, 1, 1] = 5
    ids[0, 1, 2] = 5
    (out, out_lens), (out2, _) = _run(ids, lens)
    assert torch.equal(out, out2)
    ref, ref_lens = union_words_reference(ids, lens)
    assert torch.equal(out, ref) and torch.equal(out_lens, ref_lens)
    w = out[0, : int(out_lens[0])].to(torch.long)
    hit = w[(w >> 4) == 5]
    assert hit.numel() == 1, "block 5 appears once in the union"
    assert int(hit[0] & 15) & 2, "row 1's bit is set once"
    # a run longer than the row count (row 1 lists block 5 three times plus the other rows): every row's bit still lands
    ids2 = ids.clone()
    for j in range(4):
        ids2[0, j, 5] = 5
    o3, l3 = _run(ids2, lens, launches=1)[0]
    w3 = o3[0, : int(l3[0])].to(torch.long)
    assert int(w3[(w3 >> 4) == 5][0] & 15) == 15, "a run of six keys: the OR over the whole run"


@needs_cuda
@needs_dsl_target
def test_two_launches_capture_and_the_launch_count():
    """One CUDA kernel per call (the census: ``cudnn_qsa_list_union`` once, no memset / memcpy); a CUDA-graph capture of the first
    launch after the compile replays bitwise the eager launch over the same inputs AND over new inputs written into the same buffers."""
    ids, lens = _lists(32, 4, 512, [8192 + 4 + i for i in range(32)], kind="shuffled", seed=5)
    out = torch.full((32, UNION_LIST_WIDTH), SENTINEL, dtype=torch.int32, device=_DEV)
    out_lens = torch.full((32,), SENTINEL, dtype=torch.int32, device=_DEV)
    entry = _entry()

    def call():
        entry(ids, lens, out_ids=out, out_lens=out_lens)

    kernels, extras = cupti_kernels(call)
    assert kernels is not None and len(kernels) == 1 and "qsa_list_union" in kernels[0], kernels
    assert not extras, extras
    eager = out.clone()
    ref, ref_lens = union_words_reference(ids, lens)
    assert torch.equal(eager, ref) and torch.equal(out_lens, ref_lens)
    side = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    out.fill_(SENTINEL)
    with torch.cuda.stream(side):
        with torch.cuda.graph(graph, stream=side):
            call()
    torch.cuda.synchronize()
    out.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager) and torch.equal(out_lens, ref_lens), "the captured launch replays the eager one"
    new_ids, new_lens = _lists(32, 4, 512, [16384 + i for i in range(32)], kind="canonical", seed=6)
    ids.copy_(new_ids)
    lens.copy_(new_lens)
    graph.replay()
    torch.cuda.synchronize()
    ref2, ref_lens2 = union_words_reference(new_ids, new_lens)
    assert torch.equal(out, ref2) and torch.equal(out_lens, ref_lens2), "the replay reads the new lists"


# --------------------------------------------------------------------------------------------- reject cells


@needs_cuda
def test_form_refusals_are_typed_and_precede_the_kernel_import(monkeypatch):
    """Every malformed call is refused by name BEFORE the kernel module is imported: S_q 5, top_k 513 (the capacity 2048 = 4 x 512 is
    never exceeded inside those two bounds; the entry keeps its backstop), an id bound past 2^27, a non-contiguous row, a wrong dtype, a
    CPU tensor, block_lens of the wrong shape / dtype / device, outputs of the wrong shape."""
    import cudnn.gated_attention_block.qsa_list_union as facade

    def boom():
        raise AssertionError("the kernel module was imported before the form check refused the call")

    monkeypatch.setattr(facade, "qsa_list_union_kernel_module", boom)
    good = torch.zeros(2, 4, 64, dtype=torch.int32, device=_DEV)
    lens = torch.full((2, 4), 64, dtype=torch.int32, device=_DEV)
    out = torch.zeros(2, UNION_LIST_WIDTH, dtype=torch.int32, device=_DEV)
    out_lens = torch.zeros(2, dtype=torch.int32, device=_DEV)
    with pytest.raises(NotImplementedError, match="S_q in \\[1, 4\\]"):
        facade.qsa_block_list_union(torch.zeros(2, 5, 64, dtype=torch.int32, device=_DEV), None, out_ids=out, out_lens=out_lens)
    with pytest.raises(NotImplementedError, match="top_k in \\[1, 512\\]"):
        facade.qsa_block_list_union(torch.zeros(2, 1, 513, dtype=torch.int32, device=_DEV), None, out_ids=out, out_lens=out_lens)
    with pytest.raises(NotImplementedError, match="below 2\\^27"):
        facade.qsa_block_list_union(good, lens, out_ids=out, out_lens=out_lens, n_blocks_max=(1 << 27) + 1)
    wide = torch.zeros(2, 4, 128, dtype=torch.int32, device=_DEV)[:, :, :64]
    with pytest.raises(ValueError, match="contiguous"):
        facade.qsa_block_list_union(wide, None, out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="int32"):
        facade.qsa_block_list_union(good.to(torch.int64), None, out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="CUDA"):
        facade.qsa_block_list_union(good.cpu(), None, out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="block_lens"):
        facade.qsa_block_list_union(good, lens[:, :3].contiguous(), out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="block_lens"):
        facade.qsa_block_list_union(good, lens.to(torch.int64), out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="block_lens must live on"):
        facade.qsa_block_list_union(good, lens.cpu(), out_ids=out, out_lens=out_lens)
    with pytest.raises(ValueError, match="out_ids"):
        facade.qsa_block_list_union(good, lens, out_ids=out[:, :1024].contiguous(), out_lens=out_lens)
    with pytest.raises(ValueError, match="out_lens"):
        facade.qsa_block_list_union(good, lens, out_ids=out, out_lens=out_lens[:1])
    with pytest.raises(ValueError, match="rank|\\[B, S_q, top_k\\]"):
        facade.qsa_block_list_union(good.view(2, 256), None, out_ids=out, out_lens=out_lens)


def test_the_facade_declines_a_too_old_dsl_by_version_without_importing_the_kernel_module(monkeypatch):
    """Rule 7: with the floor modeled as unmet the facade raises ``NotImplementedError`` naming the version and the kernel module stays
    un-imported (the gate runs before the import)."""
    import cudnn.gated_attention_block.qsa_list_union as facade

    monkeypatch.setattr(facade, "cutedsl_requirement_error", lambda what: f"{what}: needs nvidia-cutlass-dsl >= 4.7.0 (installed 4.6.2 modeled)")
    sys.modules.pop(facade.QSA_LIST_UNION_KERNEL_MODULE, None)
    with pytest.raises(NotImplementedError, match="4.7.0"):
        facade.qsa_list_union_kernel_module()
    assert facade.QSA_LIST_UNION_KERNEL_MODULE not in sys.modules
    assert facade.UNION_LIST_WIDTH == UNION_LIST_WIDTH == 2048 and facade.UNION_MAX_ROWS == 4 and facade.UNION_ID_BITS == 27


def test_the_kernel_module_docstring_carries_the_tables_and_the_contract():
    """The pre-development tables (the barrier table, the SMEM table) and the contract sentences are in the kernel module's docstring, and
    the kernel's geometry validator refuses every out-of-range geometry by name."""
    if _CUTEDSL_REQUIREMENT is not None:
        pytest.skip(str(_CUTEDSL_REQUIREMENT))
    mod = importlib.import_module("cudnn.gated_attention_block.kernels.qsa_list_union")
    doc = " ".join(mod.__doc__.split()).lower()  # the docstring wraps at 160 columns: compare on normalised whitespace
    for frag in ("Barrier table", "SMEM table", "STRICTLY DESCENDING", "two launches are bitwise", "duplicates WITHIN a row collapse", "INT32_MIN"):
        assert frag.lower() in doc, frag
    assert mod.UNION_LIST_WIDTH == 2048 and mod.THREADS == 1024 and mod.SMEM_BYTES >= 2048 * 4 + 256
    for s_q, top_k in ((0, 64), (5, 64), (1, 0), (1, 513), (4, 513)):
        with pytest.raises(ValueError):
            mod.validate_union_geometry(s_q, top_k)
    mod.validate_union_geometry(4, 512)
