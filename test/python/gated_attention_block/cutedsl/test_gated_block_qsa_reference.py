# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins of the QSA oracle (``gated_block_qsa_reference.py``) -- the acceptance gate of every sparse kernel, checked
against itself and the dense oracle before any kernel exists.  CPU or any GPU; no FROST kernel, no ``transformers``.

What is pinned, and why each matters to a kernel test downstream:

* ``qsa_visible_mask``: the count rule (``n_sel_default`` from the position; ``block_lens`` clamped; a ``-1`` inside
  the count removes exactly that block; entries beyond the count are never read), the tail (always visible), the
  causal / length clip, idempotent duplicates, dead rows, and the ``pos0`` reading of the decode mode's shared list
  (the block completed after step 0 stays visible to the row that completes it).
* the identity bound: with the full list the QSA oracle is BITWISE the dense oracle at every ``S <= 2051`` (same fp32
  code path), and at ``S = 2052`` exactly row 2051 differs -- plus an adversarial plant that makes that row's
  difference large, so a kernel ignoring the list cannot pass the same cell.
* the indexer reference: the identity below the bound, one dropped block at 2052, the tie-break, the score formula
  against a hand computation of the HF arithmetic, and ``hf_rounding`` changing rounding only.
* ``make_qsa_inputs``: every list source satisfies the block-id contract; the dense draws are bitwise the dense block's.
"""

import dataclasses
import math
import os
import sys

import pytest
import torch

pytestmark = pytest.mark.L0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_qsa_reference import (  # noqa: E402
    GEOMETRY_FLASH_NEXT,
    GEOMETRY_QSA_SMALL,
    INDEX_SOURCES,
    RefQsaGeometry,
    RefQsaSpec,
    block_ids_contract_violations,
    dense_geometry,
    full_block_ids,
    gated_attention_block_qsa_reference,
    make_qsa_inputs,
    qsa_indexer_reference,
    qsa_visible_mask,
    random_block_ids,
)
from gated_block_reference import gated_attention_block_reference, make_inputs  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G = GEOMETRY_QSA_SMALL
BS = 4


def _vis(block_ids, p, *, L=None, S_kv=None, block_lens=None, top_k=None, pos0=None, bs=BS):
    """``qsa_visible_mask`` on a python list of rows -> list of visible-token lists."""
    ids = torch.as_tensor(block_ids, device=DEV, dtype=torch.int32)
    if ids.dim() == 1:
        ids = ids[None]
    pos = torch.as_tensor(p, device=DEV)
    rows = ids.shape[0]
    if pos.dim() == 0:
        pos = pos.expand(rows)
    if ids.shape[0] == 1 and rows == 1 and pos.numel() > 1:
        ids = ids.expand(pos.numel(), -1)
    s_kv = int(S_kv if S_kv is not None else (L if L is not None else int(pos.max()) + 1))
    kv = s_kv if L is None else L
    m = qsa_visible_mask(ids, block_lens, pos, kv, s_kv, bs, top_k=top_k, pos0=pos0)
    return [row.nonzero().flatten().tolist() for row in m]


def _run_qsa(inp, geom, **kw):
    return gated_attention_block_qsa_reference(
        inp["h"], inp["w_qkvg"], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], geom, block_ids=inp["block_ids"], **kw
    )


def _run_dense(inp, geom, **kw):
    g = dense_geometry(geom)
    n_dense = g.n_qkvg
    return gated_attention_block_reference(inp["h"], inp["w_qkvg"][:n_dense], inp["w_q_norm"], inp["w_k_norm"], inp["cos"], inp["sin"], inp["w_o"], g, **kw)


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """``max|a - b| / max|b|`` over the finite entries (``-inf`` LSEs must match exactly)."""
    a, b = a.float(), b.float()
    fin = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), fin), "finite / -inf pattern differs"
    assert torch.equal(a[~fin], b[~fin]), "the non-finite entries differ"
    scale = b[fin].abs().max().clamp_min(1e-30)
    return float((a[fin] - b[fin]).abs().max() / scale)


def _assert_oracles_equal(qsa, dense, *, rel=1e-5):
    """The QB00 bound: max relative diff <= 1e-5 on out / O / LSE (the two share the fp32 code path, so bitwise in practice)."""
    for name in ("out", "o", "lse"):
        a, b = getattr(qsa, name), getattr(dense, name)
        r = _max_rel(a, b)
        assert r <= rel, f"{name}: max rel diff {r:.3e} > {rel}"


def _differing_rows(qsa, dense) -> list:
    """Query rows where out / O / LSE differ at all (bitwise)."""
    d_out = (qsa.out.float() - dense.out.float()).abs().amax(dim=(0, 2))
    d_o = (qsa.o.float() - dense.o.float()).abs().amax(dim=(0, 2, 3))
    d_l = (qsa.lse - dense.lse).abs().amax(dim=(0, 1))
    d_l = torch.where(torch.isnan(d_l), torch.ones_like(d_l), d_l)  # -inf - -inf = nan: count a pattern change as a difference
    return (d_out + d_o + d_l > 0).nonzero().flatten().tolist()


# ---------------------------------------------------------------------------
# qsa_visible_mask -- the count rule, the tail, duplicates, pos0, dead rows
# ---------------------------------------------------------------------------


class TestVisibleMask:
    def test_listed_complete_blocks_plus_the_tail(self):
        # p = 13: n_vis 14, complete blocks 0..2, tail [12, 14); the list names blocks 0 and 2 -> 0..3, 8..11, 12, 13.
        assert _vis([0, 2, -1, -1], 13, L=16)[0] == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13]
        # p = 15: n_vis 16, (p + 1) % 4 == 0 -> no tail; block 3 (the query's own) is visible only if listed.
        assert _vis([0, 2, -1, -1], 15, L=16)[0] == [0, 1, 2, 3, 8, 9, 10, 11]
        assert _vis([0, 2, 3, -1], 15, L=16)[0] == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]

    def test_a_minus_one_inside_the_count_removes_exactly_that_block(self):
        """The count rule's first half: ``-1`` at an index below the count is 'no key', not a terminator."""
        with_hole = _vis([0, -1, 2, 3], 15, L=16)[0]
        without = _vis([0, 2, 3, -1], 15, L=16)[0]
        assert with_hole == without == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
        # ... and it is a VALUE: the block after the hole is still read (block 3 above), only block 1 is absent.
        assert 4 not in with_hole and 12 in with_hole

    def test_entries_at_or_beyond_the_count_are_never_read(self):
        """The count rule's second half, both forms of the count."""
        # (a) block_lens: list [1, 0] with block_lens 1 -> only block 1 (tokens 4..7) + the tail; block 0 is ignored.
        assert _vis([1, 0, -1], 9, L=12, block_lens=1)[0] == [4, 5, 6, 7, 8, 9]
        assert _vis([1, 0, -1], 9, L=12)[0] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # (b) n_sel_default = min(top_k, floor((p + 1) / 4)): at p = 9 only 2 complete blocks exist, so a third
        #     valid-looking entry is beyond the derived count and changes nothing.
        assert _vis([1, 0, 1], 9, L=12) == _vis([1, 0, -1], 9, L=12)
        # (c) the top_k cap: p = 100 has 25 complete blocks, top_k = 2 reads two entries.
        assert _vis([5, 7, 9, 11], 100, L=104, top_k=2)[0] == [20, 21, 22, 23, 28, 29, 30, 31, 100]

    def test_block_lens_is_clamped_to_zero_and_to_the_derived_count(self):
        full = _vis([0, 1, 2, 3], 15, L=16)[0]
        assert _vis([0, 1, 2, 3], 15, L=16, block_lens=99)[0] == full  # above the derived count: clamped, nothing extra
        assert _vis([0, 1, 2, 3], 15, L=16, block_lens=-3)[0] == []  # negative -> 0 blocks, and no tail at p = 15 -> dead
        assert _vis([0, 1, 2, 3], 14, L=16, block_lens=-3)[0] == [12, 13, 14]  # negative -> 0 blocks + the tail

    def test_duplicates_are_idempotent(self):
        a = _vis([3, 3, 5, 5, 5], 100, L=200, top_k=8)[0]
        b = _vis([3, 5, -1, -1, -1], 100, L=200, top_k=8)[0]
        assert a == b == [12, 13, 14, 15, 20, 21, 22, 23, 100]

    def test_a_listed_block_at_or_past_the_visible_range_contributes_nothing(self):
        # p = 9 (n_vis 10, blocks 0, 1 complete): block 2 holds tokens 8..11 of which 8, 9 are the tail anyway and
        # 10, 11 are future; block 7 is past the sequence; neither faults or leaks.
        assert _vis([2, 7, -1, -1], 9, L=12)[0] == [8, 9]
        # a listed block past the SEQUENCE LENGTH (kv_lens) is clipped too: L = 6, p = 9 (a padded row) sees 0..5 only.
        assert _vis([0, 1, 2, -1], 9, L=6, S_kv=12)[0] == [0, 1, 2, 3, 4, 5]

    def test_positions_0_to_2_attend_to_the_tail_only(self):
        for p in (0, 1, 2):
            assert _vis([0, 1, 2, 3], p, L=16)[0] == list(range(p + 1))  # no complete block yet: the list is never read
            assert _vis([-1, -1, -1, -1], p, L=16)[0] == list(range(p + 1))

    def test_dead_rows(self):
        assert _vis([-1, -1], 3, L=4)[0] == []  # an empty list and an empty tail: no visible key
        assert _vis([0, 1], 3, L=0, S_kv=4)[0] == []  # a zero-length sequence
        assert _vis([0, 1], -1, L=4)[0] == []  # a negative position (a padded decode row)

    def test_pos0_keeps_the_block_completed_after_step_0_visible(self):
        """The decode mode's shared list: pos0 = 12 (step 0; blocks 0..2 complete, tail {12}); rows at 12..15."""
        ids = [[0, 1, 2]] * 4
        pos = [12, 13, 14, 15]
        shared = _vis(ids, pos, L=16, pos0=12)
        per_row = _vis(ids, pos, L=16)
        # FROST's reading: every token from the step-0 tail start (12) to the row's own position stays visible.
        assert shared == [list(range(13)), list(range(14)), list(range(15)), list(range(16))]
        # The per-row-tail reading hides the just-completed block 3 (tokens 12..15) from the row at 4k + 3 = 15.
        assert per_row[3] == list(range(12))
        assert per_row[:3] == shared[:3]

    def test_pos0_takes_the_count_from_step_0(self):
        # At step 0 (pos0 = 6: n_vis 7) ONE complete block exists and the tail starts at 4; the shared list's second entry
        # is beyond the step-0 count and is never read by a later row even though that row (p = 11) has three.
        assert _vis([0, 1], 11, L=16, pos0=6)[0] == list(range(12))  # block 0 + the tail from 4 to 11 covers everything
        assert _vis([1, 0], 11, L=16, pos0=6)[0] == [4, 5, 6, 7, 8, 9, 10, 11]  # only entry 0 (block 1) is read; tail 4..11
        assert _vis([1, 0], 11, L=16, pos0=7)[0] == list(range(12))  # pos0 = 7: two complete blocks at step 0, both read

    def test_trailing_minus_ones_and_block_lens_form_are_inert(self):
        a = _vis([3, 1, 7] + [-1] * 5, 100, L=104)[0]
        b = _vis([3, 1, 7] + [-1] * 29, 100, L=104)[0]
        c = _vis([3, 1, 7] + [-1] * 5, 100, L=104, block_lens=3)[0]
        assert a == b == c

    def test_shapes_are_checked(self):
        with pytest.raises(ValueError, match="rows, top_k"):
            qsa_visible_mask(torch.zeros(4, dtype=torch.int32, device=DEV), None, 3, 4, 4, 4)
        with pytest.raises(ValueError, match="one entry per row"):
            qsa_visible_mask(torch.zeros(2, 4, dtype=torch.int32, device=DEV), None, torch.tensor([1, 2, 3], device=DEV), 4, 4, 4)


# ---------------------------------------------------------------------------
# The identity bound: the QSA oracle vs the dense oracle
# ---------------------------------------------------------------------------


class TestIdentityBound:
    @pytest.mark.parametrize("S", [4, 5, 7, 512, 2051])
    def test_full_list_equals_the_dense_oracle(self, S):
        inp = make_qsa_inputs(G, 2, S, device=DEV, seed=S, index_source="full")
        qsa = _run_qsa(inp, G, block_lens=inp["block_lens"])
        dense = _run_dense(inp, G)
        _assert_oracles_equal(qsa, dense)
        assert _differing_rows(qsa, dense) == [], "same fp32 code path: bitwise expected"
        assert int(qsa.n_visible[0, -1]) == S  # the last row sees everything
        # the tail of a non-multiple-of-4 length is visible: n_vis counts every token
        assert torch.equal(qsa.n_visible[0], torch.arange(1, S + 1, device=DEV))

    def test_without_block_lens_too(self):
        inp = make_qsa_inputs(G, 1, 300, device=DEV, seed=3, index_source="full")
        _assert_oracles_equal(_run_qsa(inp, G), _run_dense(inp, G))

    def test_padding_lengths_match_the_dense_oracle(self):
        S = 300
        seq_lens = torch.tensor([S, S - 37], dtype=torch.int32, device=DEV)
        inp = make_qsa_inputs(G, 2, S, device=DEV, seed=4, index_source="full", seq_lens=seq_lens)
        qsa = _run_qsa(inp, G, block_lens=inp["block_lens"], seq_lens=seq_lens)
        dense = _run_dense(inp, G, seq_lens=seq_lens)
        _assert_oracles_equal(qsa, dense)
        assert int(qsa.n_visible[1, -1]) == S - 37

    def test_at_s_2052_exactly_row_2051_differs(self):
        S = 2052
        inp = make_qsa_inputs(G, 1, S, device=DEV, seed=5, index_source="full")
        assert int(inp["block_lens"][0, 2051]) == 512 and int(inp["block_ids"][0, 2051, 511]) == 511
        qsa = _run_qsa(inp, G, block_lens=inp["block_lens"])
        dense = _run_dense(inp, G)
        assert _differing_rows(qsa, dense) == [2051]
        assert int(qsa.n_visible[0, 2051]) == 2048  # 512 blocks x 4, no tail: block 512 (its own) is the dropped one
        d_lse = float((qsa.lse[0, :, 2051] - dense.lse[0, :, 2051]).abs().max())
        print(f"S=2052 row 2051: max |LSE_list - LSE_dense| = {d_lse:.3e} (the dropped block holds ~4 of 2052 keys)")
        assert d_lse > 0

    def test_adversarial_plant_makes_the_dropped_block_matter(self):
        """A kernel that ignores the list must FAIL this cell: a dominant key is planted in the omitted block."""
        S = 2052
        g = dataclasses.replace(G, qk_norm=False)  # no RMSNorm: the key's magnitude is ours to set
        inp = make_qsa_inputs(g, 1, S, device=DEV, seed=6, index_source="full")
        w = inp["w_qkvg"].clone()
        o_q, o_g, o_k, o_v = g.offsets
        rep = g.h_q // g.h_kv
        for hq in range(g.h_q):  # every Q head projects like its KV head: q_t . k_t' is then |W h|^2-like when h_t' = c h_t
            w[o_q + hq * g.d_head : o_q + (hq + 1) * g.d_head] = w[o_k + (hq // rep) * g.d_head : o_k + (hq // rep + 1) * g.d_head]
        inp["w_qkvg"] = w
        h = inp["h"].clone()
        h[0, 2049] = (h[0, 2051].float() * 30).to(h.dtype)  # token 2049 is in block 512, the block row 2051 loses
        inp["h"] = h
        qsa = _run_qsa(inp, g, block_lens=inp["block_lens"])
        dense = _run_dense(inp, g)
        assert _differing_rows(qsa, dense) == [2051]
        margin = (dense.lse[0, :, 2051] - qsa.lse[0, :, 2051]).min()  # nats the planted key adds to the row's mass
        cos = torch.nn.functional.cosine_similarity(qsa.o[0, 2051].float().flatten(), dense.o[0, 2051].float().flatten(), dim=0)
        print(f"planted margin {float(margin):.2f} nats; cos(O_list, O_dense) at row 2051 = {float(cos):.4f}")
        assert float(margin) > 8.0, "the plant must dominate the row by >= 8 nats"
        assert float(cos) < 0.5, "with the dominant key dropped the row's O must change grossly"

    def test_dead_entry_is_exactly_zero(self):
        S = 64
        seq_lens = torch.tensor([S, 0], dtype=torch.int32, device=DEV)
        inp = make_qsa_inputs(G, 2, S, device=DEV, seed=7, index_source="full", seq_lens=seq_lens)
        qsa = _run_qsa(inp, G, block_lens=inp["block_lens"], seq_lens=seq_lens)
        assert torch.equal(qsa.out[1], torch.zeros_like(qsa.out[1]))
        assert torch.equal(qsa.o[1], torch.zeros_like(qsa.o[1]))
        assert torch.isneginf(qsa.lse[1]).all()
        assert int(qsa.n_visible[1].sum()) == 0
        assert torch.isfinite(qsa.out[0]).all() and torch.isfinite(qsa.lse[0]).all()

    def test_geometry_rejects_what_qsa_cannot_compose_with(self):
        inp = make_qsa_inputs(G, 1, 16, device=DEV, seed=0, index_source="full")
        with pytest.raises(ValueError, match="is_causal"):
            _run_qsa(inp, dataclasses.replace(G, is_causal=False))
        with pytest.raises(ValueError, match="window"):
            _run_qsa(inp, dataclasses.replace(G, window_left=8))
        with pytest.raises(ValueError, match="w_qkvg must be"):
            _run_qsa(inp, dataclasses.replace(G, qsa=RefQsaSpec(index_band=True)))

    def test_flash_next_geometry_numbers(self):
        g = GEOMETRY_FLASH_NEXT
        assert g.n_qkvg == 13312 and g.n_qkvg_dense == 13312 and g.index_offset is None
        assert g.qsa.identity_bound == 2051 and g.qsa.token_budget == 2048 and g.qsa.index_band_width == 640
        with_band = dataclasses.replace(g, qsa=RefQsaSpec(index_band=True))
        assert with_band.n_qkvg == 13952 and with_band.index_offset == 13312
        assert g.scale == pytest.approx(1.0 / 16.0) and g.rope_base == 1e7
        g.qsa.validate(g)
        with_band.qsa.validate(with_band)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(block_size=8), "block_size"),
            (dict(top_k=6), "top_k"),
            (dict(top_k=0), "top_k"),
            (dict(top_k=1024), "top_k"),
            (dict(index_source="tokens"), "index_source"),
            (dict(index_kv_heads=2), "index_kv_heads"),
            (dict(index_head_dim=100), "multiple of 64"),
            (dict(index_norm_eps=0.0), "index_norm_eps"),
        ],
    )
    def test_spec_validate_rejects(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RefQsaSpec(**kwargs).validate()

    def test_spec_validate_checks_the_rope_fit(self):
        with pytest.raises(ValueError, match="rope_dim"):
            RefQsaSpec(index_head_dim=64).validate(dataclasses.replace(GEOMETRY_FLASH_NEXT, rope_dim=128))


# ---------------------------------------------------------------------------
# The indexer reference
# ---------------------------------------------------------------------------


def _indexer(inp, geom, spec, **kw):
    return qsa_indexer_reference(inp["h"], inp["w_i"], inp["w_iq_norm"], inp["w_ik_norm"], None, None, inp["cos"], inp["sin"], spec, geom, **kw)


class TestIndexerReference:
    def test_selects_every_complete_block_below_the_identity_bound(self):
        S = 300
        inp = make_qsa_inputs(G, 2, S, device=DEV, seed=8, index_source="indexer")
        pos = torch.arange(S, device=DEV)
        expect_lens = torch.div(pos + 1, BS, rounding_mode="floor").to(torch.int32)
        assert torch.equal(inp["block_lens"][0], expect_lens) and torch.equal(inp["block_lens"][1], expect_lens)
        srt = torch.sort(inp["block_ids"].to(torch.long), dim=-1, descending=True).values  # -1s go last
        for t in (3, 4, 7, 100, S - 1):
            n = int(expect_lens[t])
            assert srt[0, t, :n].flip(0).tolist() == list(range(n)), f"row {t}: the set must be every complete block"
        assert block_ids_contract_violations(inp["block_ids"][0], pos, S, block_size=BS, block_lens=inp["block_lens"][0], top_k=512) == []

    def test_indexer_lists_reproduce_the_dense_oracle_below_the_bound(self):
        inp = make_qsa_inputs(G, 1, 2051, device=DEV, seed=9, index_source="indexer")
        _assert_oracles_equal(_run_qsa(inp, G, block_lens=inp["block_lens"]), _run_dense(inp, G))

    def test_drops_one_block_at_s_2052_row_2051(self):
        inp = make_qsa_inputs(G, 1, 2052, device=DEV, seed=10, index_source="indexer")
        lens = inp["block_lens"][0]
        assert int(lens[2051]) == 512 and int(lens[2050]) == 512 and int(lens[2047]) == 512 and int(lens[2046]) == 511
        ids = inp["block_ids"][0, 2051]
        assert int((ids >= 0).sum()) == 512 and len(set(ids.tolist())) == 512  # 512 distinct of the 513 complete blocks
        missing = set(range(513)) - set(ids.tolist())
        assert len(missing) == 1
        qsa = _run_qsa(inp, G, block_lens=inp["block_lens"])
        dense = _run_dense(inp, G)
        assert _differing_rows(qsa, dense) == [2051]

    def test_tie_break_prefers_the_smaller_block_id(self):
        S = 64
        inp = make_qsa_inputs(G, 1, S, device=DEV, seed=11, index_source="indexer")
        inp["w_i"] = torch.zeros_like(inp["w_i"])  # every score is relu(0) = 0: a 64-way tie per row
        spec = dataclasses.replace(G.qsa, top_k=8)
        scores, ids, lens, k_raw, k_c = _indexer(inp, G, spec)
        assert torch.equal(ids[0, S - 1], torch.arange(8, device=DEV, dtype=torch.int32))
        assert torch.equal(ids[0, 11], torch.tensor([0, 1, 2, -1, -1, -1, -1, -1], device=DEV, dtype=torch.int32))
        assert int(lens[0, 11]) == 3 and int(lens[0, 2]) == 0
        assert torch.isneginf(scores[0, 11, 3:]).all() and (scores[0, 11, :3] == 0).all()

    def test_score_formula_against_a_hand_computation(self):
        """``score_j = sum_h relu(q_h . kbar_j) / sqrt(d)`` with HF's norm, pooled fp32 mean and block-start RoPE, written
        out by hand in fp32 (no shared helper), at one query."""
        g = RefQsaGeometry(d_model=16, h_q=2, h_kv=1, d_head=8, rope_dim=4, qsa=RefQsaSpec(top_k=2, index_heads=2, index_head_dim=8))
        spec = g.qsa
        S, t = 12, 11
        torch.manual_seed(0)
        h = torch.randn(1, S, 16, device=DEV)  # fp32 io: no rounding anywhere in either computation
        w_i = torch.randn(spec.index_band_width, 16, device=DEV) * 0.3
        wq = torch.rand(8, device=DEV) + 0.5
        wk = torch.rand(8, device=DEV) + 0.5
        inv_freq = 1.0 / (g.rope_base ** (torch.arange(0, 4, 2, device=DEV).float() / 4))
        ang = torch.arange(S, device=DEV).float()[:, None] * inv_freq
        cos = torch.cat([ang.cos(), ang.cos()], -1)[None]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[None]
        scores, ids, lens, k_raw, k_c = qsa_indexer_reference(h, w_i, wq, wk, None, None, cos, sin, spec, g)

        def rms(x, w):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + spec.index_norm_eps) * w

        def rope(x, pos):  # NeoX rotate-half on the leading 4 dims, the rest untouched
            c, s_ = cos[0, pos], sin[0, pos]
            r = x[..., :4]
            rot = torch.cat([-r[..., 2:], r[..., :2]], -1)
            return torch.cat([r * c + rot * s_, x[..., 4:]], -1)

        qk = h[0] @ w_i.t()  # [S, 24]
        q = rope(rms(qk[t, :16].view(2, 8), wq), t)
        kr = qk[:, 16:]
        hand = []
        for j in range(3):  # n_vis = 12 -> 3 complete blocks
            pooled = kr[4 * j : 4 * j + 4].mean(0)
            kb = rope(rms(pooled, wk), 4 * j)
            hand.append(float(torch.relu(q @ kb).sum() / math.sqrt(8)))
        torch.testing.assert_close(scores[0, t, :3], torch.tensor(hand, device=DEV), rtol=1e-5, atol=1e-6)
        assert torch.isneginf(scores[0, t, 3:]).all() if scores.shape[-1] > 3 else True
        assert int(lens[0, t]) == 2 and sorted(ids[0, t].tolist()) == sorted(torch.tensor(hand).topk(2).indices.tolist())
        torch.testing.assert_close(k_raw[0], kr)

    def test_hf_rounding_changes_rounding_only(self):
        S = 512
        inp = make_qsa_inputs(G, 1, S, device=DEV, seed=12, index_source="indexer")
        s_ref, ids_ref, lens_ref, kr_ref, kc_ref = _indexer(inp, G, G.qsa)
        s_hf, ids_hf, lens_hf, kr_hf, kc_hf = _indexer(inp, G, G.qsa, hf_rounding=True)
        assert torch.equal(lens_ref, lens_hf) and torch.equal(kr_ref, kr_hf)
        assert kc_ref.dtype == torch.float32 and kc_hf.dtype == inp["h"].dtype
        fin = torch.isfinite(s_ref)
        assert torch.equal(fin, torch.isfinite(s_hf))
        rel = float((s_ref[fin] - s_hf[fin]).abs().max() / s_ref[fin].abs().max())
        print(f"hf_rounding: max rel score diff {rel:.3e}")
        assert 0 < rel < 2e-2  # a few bf16 roundings, not a different formula
        # Below the identity bound both select every complete block: the SETS agree whatever the rounding.
        assert torch.equal(torch.sort(ids_ref, dim=-1).values, torch.sort(ids_hf, dim=-1).values)

    def test_rejects_bad_declarations(self):
        inp = make_qsa_inputs(G, 1, 16, device=DEV, seed=0, index_source="indexer")
        with pytest.raises(ValueError, match="index_kv_heads"):
            _indexer(inp, G, dataclasses.replace(G.qsa, index_kv_heads=2))
        with pytest.raises(ValueError, match="w_i must be"):
            _indexer(inp, G, dataclasses.replace(G.qsa, index_heads=8))
        narrow = dataclasses.replace(G.qsa, index_head_dim=8)  # the attention tables rotate 16 dims: they do not fit an 8-wide index head
        with pytest.raises(ValueError, match="rope_dim"):
            qsa_indexer_reference(
                inp["h"],
                torch.zeros(narrow.index_band_width, G.d_model, device=DEV),
                torch.ones(8, device=DEV),
                torch.ones(8, device=DEV),
                None,
                None,
                inp["cos"],
                inp["sin"],
                narrow,
                G,
            )


# ---------------------------------------------------------------------------
# make_qsa_inputs and the list builders
# ---------------------------------------------------------------------------


class TestInputs:
    @pytest.mark.parametrize("index_source", INDEX_SOURCES)
    def test_every_list_source_satisfies_the_contract(self, index_source):
        S = 2100
        seq_lens = torch.tensor([S, 1337], dtype=torch.int32, device=DEV)
        inp = make_qsa_inputs(G, 2, S, device=DEV, seed=13, index_source=index_source, seq_lens=seq_lens)
        ids, lens = inp["block_ids"], inp["block_lens"]
        assert ids.shape == (2, S, 512) and ids.dtype == torch.int32 and lens.shape == (2, S) and lens.dtype == torch.int32
        pos = torch.arange(S, device=DEV)
        for b in range(2):
            assert block_ids_contract_violations(ids[b], pos, int(seq_lens[b]), block_size=BS, block_lens=lens[b], top_k=512) == [], index_source
        assert int(lens[0, 2051]) == 512 and int(lens[0, 2050]) == 512 and int(lens[1, -1]) == 1337 // 4

    def test_dense_draws_are_bitwise_the_dense_blocks(self):
        g = dataclasses.replace(G, qsa=RefQsaSpec(index_band=True))
        inp = make_qsa_inputs(g, 2, 64, device=DEV, seed=14, index_source="indexer")
        dense = make_inputs(dense_geometry(g), 2, 64, device=DEV, seed=14)
        for name in ("h", "cos", "sin", "w_o", "w_q_norm", "w_k_norm"):
            assert torch.equal(inp[name], dense[name]), name
        assert torch.equal(inp["w_qkvg"][: g.n_qkvg_dense], dense["w_qkvg"]) and inp["w_qkvg"].shape[0] == g.n_qkvg
        assert torch.equal(inp["w_i"], inp["w_qkvg"][g.n_qkvg_dense :])  # the indexer ran on the band rows

    def test_the_index_band_is_split_off_and_exposed(self):
        g = dataclasses.replace(G, qsa=RefQsaSpec(index_band=True))
        inp = make_qsa_inputs(g, 1, 64, device=DEV, seed=15, index_source="indexer")
        qsa = _run_qsa(inp, g, block_lens=inp["block_lens"])
        dense = _run_dense(inp, g)  # the dense oracle over the four dense bands
        _assert_oracles_equal(qsa, dense)  # S = 64 < 2051: the band changes nothing of out / O / LSE
        assert qsa.index_k_raw.shape == (1, 64, 128) and qsa.index_q_raw.shape == (1, 64, 4, 128)
        # the band is the slab's last 640 columns, raw: the indexer's own k_raw (pre-norm, un-rotated) within one bf16
        # rounding -- the same linear map, but a 1408-column GEMM and a 640-column one accumulate in a different order
        _s, _ids, _lens, k_raw, _kc = _indexer(inp, g, g.qsa)
        torch.testing.assert_close(qsa.index_k_raw.float(), k_raw.float(), rtol=2**-7, atol=1e-3)
        n_bitwise = int((qsa.index_k_raw == k_raw).sum())
        print(
            f"index band vs indexer k_raw: {n_bitwise}/{k_raw.numel()} bitwise, max |diff| {float((qsa.index_k_raw.float() - k_raw.float()).abs().max()):.3e}"
        )

    def test_full_and_random_builders(self):
        pos = torch.arange(2060, device=DEV)
        ids, lens = full_block_ids(pos, 512, BS)
        assert ids.shape == (2060, 512) and int(lens[2051]) == 512 and int(lens[2050]) == 512 and int(lens[6]) == 1 and int(lens[2]) == 0
        assert ids[2051].tolist() == list(range(512)) and ids[7, :3].tolist() == [0, 1, -1]
        gen = torch.Generator(device=DEV).manual_seed(0)
        r_ids, r_lens = random_block_ids(pos, 512, BS, generator=gen)
        assert torch.equal(r_lens, lens)
        assert block_ids_contract_violations(r_ids, pos, 2060, block_size=BS, block_lens=r_lens, top_k=512) == []
        assert sorted(r_ids[2051].tolist()) != list(range(512))  # a random 512-subset of 513 in random order ...
        assert sorted(r_ids[2047].tolist()) == list(range(512))  # ... and the full set below the bound
        assert r_ids[2047].tolist() != list(range(512))  # in random order
        gen2 = torch.Generator(device=DEV).manual_seed(0)
        a_ids, _ = random_block_ids(pos, 512, BS, generator=gen2, shuffle=False)
        assert a_ids[2047].tolist() == list(range(512))

    def test_contract_detector_names_each_violation(self):
        pos = torch.full((3,), 100, device=DEV)
        ids = torch.tensor([[0, -1, 2, -1], [3, 3, -1, -1], [30, 1, -1, -1]], device=DEV, dtype=torch.int32)
        out = block_ids_contract_violations(ids, pos, 101, block_size=BS, block_lens=torch.tensor([2, 2, 2], device=DEV))
        text = "\n".join(out)
        assert "not a valid prefix" in text and "duplicate" in text and "at or past floor" in text
        assert block_ids_contract_violations(ids[1:2, :1], pos[:1], 101, block_size=BS) == []

    def test_unknown_source_is_refused(self):
        with pytest.raises(ValueError, match="index_source"):
            make_qsa_inputs(G, 1, 8, device=DEV, index_source="real")
