# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 7 of the MQA block -- the ``torch``, ``dsa`` and ``d512`` attention adapters against the oracles.

Each adapter is driven through the block's ``_SparseAttention`` stage exactly as
``execute`` drives it (workspace-shaped buffers, the D5 index staging, the D1
LSE fold), and compared with ``mqa_block_reference.sparse_attention_reference``
under the bf16 budget that oracle documents (rtol ``2**-7``, atol ``2**-8 *
max|kv|``: a bitwise or 1e-4 bar on O is RED on 10-27 % of the elements of a
correct kernel), and with the exact fp64 arm for the LSE (``1e-4``).  The
degenerate rows the plan singles out -- keyless rows, ``sink = +-inf``,
duplicate slots, ``K % 64 != 0``, ``B = 2``, ``S = 1`` -- each have their own
test.  Accept tests need Rubin; the rejects and the pure-torch adapter's CPU
check run anywhere.  The ``d512`` rows (the gathered-list fork behind the
block-owned ``union_lists`` pre-pass) skip typed until the fork module exists;
its kernel-level sweep lives in ``test_mqa_block_sparse_attention_d512.py``.
"""

import math

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("MQA sparse-attention block stage tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

# Rootdir-qualified: a bare `import mqa_block_reference` would collide with other fe_api packages in one session.
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R  # noqa: E402
from fe_api.mqa_sparse_attention_block.test_mqa_block_reference_adversarial import _sparse_attn_kernel_emulation  # noqa: E402

from cudnn.mqa_sparse_attention_block import MqaSparseAttentionBlockGeometry  # noqa: E402
from cudnn.mqa_sparse_attention_block.api import _SparseAttention  # noqa: E402
from cudnn.mqa_sparse_attention_block.kernels import sparse_attention as SA  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
requires_d512_fork = pytest.mark.skipif(not SA.d512_fork_available(), reason=f"the d512 fork module {SA.D512_FORK_MODULE} has not landed (wave W3b)")

_H, _D, _WINDOW, _TOPK = 64, 512, 128, 512  # the DSA (64, 512) variant at the released list geometry
_SHAPES = [(1, 300, 0), (1, 300, 1), (2, 1024, 2), (1, 256, 1), (1, 1, 2)]
_ADAPTERS = ["torch", "dsa", pytest.param("d512", marks=requires_d512_fork)]
_D512_UNION = ("union_ids", "union_bits", "union_ntiles")  # the pre-pass views the d512 stage carves (``_SparseAttention.union_shapes``)
_EXACT = dict(p_dtype=torch.float32, out_dtype=torch.float32)


def _case(device, *, batch, seq_len, ratio, h=_H, d=_D, window=_WINDOW, topk=_TOPK, seed=0, sink=None):
    """Random q / kv_all / sink / topk_idxs in the block's layouts (window list ++ synthetic compressed list)."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, seq_len, h, d, generator=g, device=device).to(torch.bfloat16)
    kv = torch.randn(batch, seq_len, d, generator=g, device=device).to(torch.bfloat16)
    idxs = R.window_idxs(batch, seq_len, window, device=device)
    if ratio > 0 and seq_len // ratio > 0:
        n_c = seq_len // ratio
        kv = torch.cat([kv, torch.randn(batch, n_c, d, generator=g, device=device).to(torch.bfloat16)], dim=1)
        idxs = torch.cat([idxs, R.compressed_idxs_synthetic(batch, seq_len, ratio, topk, g, device=device)], dim=-1)
    if sink is None:
        sink = torch.randn(h, generator=g, device=device, dtype=torch.float32)
    else:
        sink = torch.full((h,), float(sink), device=device, dtype=torch.float32)
    return q, kv.contiguous(), sink, idxs.contiguous(), float(d) ** -0.5


def _assert_bf16_budget(o_a, o_b, kv, label=""):
    torch.testing.assert_close(o_a.float(), o_b.float(), rtol=2**-7, atol=2**-8 * float(kv.float().abs().max()), msg=lambda m: f"{label}: {m}")


def _stage(impl, q, kv, idxs, *, want_lse=True):
    b, s, h, d = (int(v) for v in q.shape)
    geom = MqaSparseAttentionBlockGeometry(n_heads=h, head_dim=d, window=_WINDOW, index_topk=_TOPK)
    st = _SparseAttention(
        impl=impl, geom=geom, batch=b, seq_len=s, n_kv_rows=int(kv.shape[1]), topk=int(idxs.shape[2]), dtype=q.dtype, device=q.device, want_lse=want_lse
    )
    st.check_support()
    st.compile()
    return st


def _buffers(st, q, idxs, *, want_lse=True):
    """Workspace-shaped buffers exactly as the block carves them."""
    b, s, h, d = (int(v) for v in q.shape)
    t = b * s
    bufs = dict(o=torch.empty_like(q), lse=torch.full((b, h, s), math.nan, device=q.device, dtype=torch.float32) if want_lse else None)
    if st.impl == "dsa":
        bufs["lse_th"] = torch.empty(t, h, device=q.device, dtype=torch.float32)
        bufs["max_logits"] = torch.empty(t, h, device=q.device, dtype=torch.float32)
        if st.needs_idx_staging:
            bufs["idx_ws"] = torch.empty(b, s, st.k_pad, device=q.device, dtype=torch.int32)
            if b > 1:
                bufs["idx_tmp"] = torch.empty(b, s, int(idxs.shape[2]), device=q.device, dtype=torch.int32)
    if st.impl == "d512":
        # The three int32 pre-pass views, shaped by the adapter (u_max_tiles = ceil(4K / 128) is the kernel's column pitch).
        assert set(st.union_shapes) == set(_D512_UNION), st.union_shapes
        for name in _D512_UNION:
            bufs[name] = torch.full(st.union_shapes[name], -7, device=q.device, dtype=torch.int32)  # poison: the pre-pass must overwrite all of it
    return bufs


def _run(st, q, kv, sink, idxs, bufs, *, stream=None):
    b, s, h, d = (int(v) for v in q.shape)
    want_lse = bufs["lse"] is not None
    stream = torch.cuda.current_stream(q.device).cuda_stream if stream is None else int(stream)
    st.execute(
        q,
        kv,
        sink,
        idxs,
        bufs["o"],
        lse=bufs["lse"] if (want_lse and st.impl in ("torch", "d512")) else None,  # these adapters write the FROST-convention LSE themselves
        lse_th=bufs.get("lse_th"),
        max_logits=bufs.get("max_logits"),
        idx_tmp=bufs.get("idx_tmp"),
        idx_ws=bufs.get("idx_ws"),
        union_ids=bufs.get("union_ids"),
        union_bits=bufs.get("union_bits"),
        union_ntiles=bufs.get("union_ntiles"),
        stream=stream,
    )
    if st.impl == "dsa" and want_lse:
        # The block's stage 8 does this fold (D1); here it is applied directly.
        bufs["lse"].copy_(SA.dsa_lse_fold(bufs["lse_th"], sink).view(b, s, h).permute(0, 2, 1))
    torch.cuda.synchronize()
    return bufs["o"], bufs["lse"]


def _run_adapter(impl, q, kv, sink, idxs, *, want_lse=True):
    st = _stage(impl, q, kv, idxs, want_lse=want_lse)
    return _run(st, q, kv, sink, idxs, _buffers(st, q, idxs, want_lse=want_lse))


# ---------------------------------------------------------------------------
# Accept -- Rubin
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("impl", _ADAPTERS)
@pytest.mark.parametrize("batch,seq_len,ratio", _SHAPES, ids=[f"B{b}_S{s}_r{r}" for b, s, r in _SHAPES])
def test_adapter_matches_the_gathered_oracle(impl, batch, seq_len, ratio):
    """O within the bf16 budget of the gathered oracle; LSE within 1e-4 of the exact
    fp64 arm, in the FROST convention (sink included).  The five shapes cover the
    identity index path (B=1, K % 64 == 0), the K % 64 staging path, the B=2 flat-id
    staging path and the one-token window-only degenerate (S=1 < ratio)."""
    q, kv, sink, idxs, scale = _case("cuda", batch=batch, seq_len=seq_len, ratio=ratio)
    o, lse = _run_adapter(impl, q, kv, sink, idxs)
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    _assert_bf16_budget(o, ref_o, kv, f"{impl} O")
    _, ref_lse = R.sparse_attention_reference(q, kv, sink, idxs, scale, **_EXACT)
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-4, msg=lambda m: f"{impl} LSE: {m}")


@requires_rubin
@pytest.mark.parametrize("impl", _ADAPTERS)
def test_adapter_o_matches_the_64_slot_kernel_emulation(impl):
    """O ONLY against the sequential K:350-387 emulation (its keyless-row LSE is +inf, not
    the FROST convention, so the LSE is checked against the oracle above instead)."""
    q, kv, sink, idxs, scale = _case("cuda", batch=1, seq_len=300, ratio=1)
    o, _ = _run_adapter(impl, q, kv, sink, idxs)
    o_k, _ = _sparse_attn_kernel_emulation(q, kv, sink, idxs, scale)
    _assert_bf16_budget(o, o_k, kv, f"{impl} O vs emulation")


@requires_rubin
@pytest.mark.parametrize("impl", _ADAPTERS)
def test_keyless_rows_give_zero_o_and_lse_equal_to_the_sink(impl):
    q, kv, sink, idxs, scale = _case("cuda", batch=1, seq_len=200, ratio=1)
    dead = [5, 100, 199]
    idxs[:, dead] = -1  # every slot of these rows is "no key"
    o, lse = _run_adapter(impl, q, kv, sink, idxs)
    for row in dead:
        assert torch.equal(o[:, row], torch.zeros_like(o[:, row])), f"{impl}: keyless row {row} has O != 0"
        assert torch.equal(lse[0, :, row], sink), f"{impl}: keyless row {row} LSE != sink"
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    _assert_bf16_budget(o, ref_o, kv, f"{impl} O with dead rows")


@requires_rubin
@pytest.mark.parametrize("impl", _ADAPTERS)
def test_sink_corners_on_live_rows(impl):
    """``sink = -inf`` == no sink (O and LSE against the oracle at -inf); ``sink = +inf``
    -> ``O == 0`` and ``LSE == +inf`` exactly, with no NaN anywhere (the D1 fold's
    inner select; the H64 kernel's ``exp2(+inf - m)`` denominator)."""
    q, kv, _, idxs, scale = _case("cuda", batch=1, seq_len=256, ratio=1)
    ninf = torch.full((_H,), -math.inf, device="cuda", dtype=torch.float32)
    o, lse = _run_adapter(impl, q, kv, ninf, idxs)
    ref_o, _ = R.sparse_attention_reference(q, kv, ninf, idxs, scale)
    _, ref_lse = R.sparse_attention_reference(q, kv, ninf, idxs, scale, **_EXACT)
    _assert_bf16_budget(o, ref_o, kv, f"{impl} O at sink=-inf")
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-4)
    pinf = torch.full((_H,), math.inf, device="cuda", dtype=torch.float32)
    o, lse = _run_adapter(impl, q, kv, pinf, idxs)
    assert torch.equal(o, torch.zeros_like(o)), f"{impl}: sink=+inf must give O == 0"
    assert torch.all(lse == math.inf), f"{impl}: sink=+inf must give LSE == +inf"


@requires_rubin
@pytest.mark.parametrize("impl", _ADAPTERS)
def test_duplicate_slots_count_twice(impl):
    q, kv, sink, idxs, scale = _case("cuda", batch=1, seq_len=130, ratio=0)  # K = 128
    dup = idxs.clone()
    dup[0, 129, 1] = dup[0, 129, 0]  # row 129: slot 1 repeats slot 0's id
    o_dup, _ = _run_adapter(impl, q, kv, sink, dup)
    o_one, _ = _run_adapter(impl, q, kv, sink, idxs)
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, dup, scale)
    _assert_bf16_budget(o_dup, ref_o, kv, f"{impl} O with a duplicate slot")
    assert not torch.equal(o_dup[0, 129], o_one[0, 129]), f"{impl}: a duplicated slot must change the row (duplicates are distinct slots)"
    # Rows without the duplicate are untouched.  d512 gathers per 4-token CLUSTER: row 128 shares cluster 32 with row 129,
    # whose extra union copy shifts the cluster's tile columns (a different MMA K-order for row 128 -> within budget, not
    # bitwise); every other cluster's gather is identical, so those rows ARE bitwise.  torch / dsa are per row: all bitwise.
    untouched = 128 if impl == "d512" else 129
    assert torch.equal(o_dup[0, :untouched], o_one[0, :untouched]), f"{impl}: rows without the duplicate must be untouched"
    _assert_bf16_budget(o_dup[0, 128], o_one[0, 128], kv, f"{impl} row 128 (cluster neighbour of the duplicate)")


@requires_rubin
@pytest.mark.parametrize("impl", [a for a in _ADAPTERS if a != "torch"])
def test_ids_outside_the_kv_range_are_no_key_at_batch_two(impl):
    """The block's contract says ids outside ``[0, N)`` are "no key" on every adapter.  At
    ``B = 2`` both kernel adapters work in the flat ``[B*N, D]`` row space (the dsa staging
    offsets ids by ``b * N``; the d512 pre-pass emits flat union rows), so an unmasked
    ``id >= N`` (or ``< -1``) of batch 0 would read batch 1's rows and the adapters would
    disagree with the torch one on identical inputs.  Planted ids: ``N``, ``N + 7``, ``-5``."""
    q, kv, sink, idxs, scale = _case("cuda", batch=2, seq_len=128, ratio=2)  # K = 128 + 64 = 192 -> dsa staging (B = 2)
    n = int(kv.shape[1])
    bad = idxs.clone()
    bad[0, 3, :4] = torch.tensor([n, n + 7, -5, n - 1], device="cuda", dtype=torch.int32)  # the last one is the largest VALID id
    bad[1, 100, 0] = n
    o_k, lse_k = _run_adapter(impl, q, kv, sink, bad)
    o_torch, lse_torch = _run_adapter("torch", q, kv, sink, bad)
    _assert_bf16_budget(o_k, o_torch, kv, f"{impl} vs torch adapter with out-of-range ids")
    torch.testing.assert_close(lse_k, lse_torch, rtol=0, atol=1e-4)
    # ... and both equal the oracle on the SANITISED list (the oracle knows only -1 as "no key").
    clean = torch.where((bad >= 0) & (bad < n), bad, torch.full_like(bad, -1))
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, clean, scale)
    _assert_bf16_budget(o_k, ref_o, kv, f"{impl} O vs oracle on the sanitised list")
    assert not torch.equal(o_k[0, 3], _run_adapter(impl, q, kv, sink, idxs)[0][0, 3]), "the planted slots must have changed row (0, 3) (they replaced live ids)"


@requires_rubin
@requires_d512_fork
def test_d512_launch_allocates_nothing_is_bitwise_repeatable_and_never_syncs():
    """The d512 stage is the FLAGGED torch ``union_lists`` pre-pass (allocates -- measured and reported, not asserted) followed
    by ONE kernel launch over zero-copy views: ``launch`` alone allocates nothing (no copy of q / kv_all / o, the ``[B*N, D]``
    KV view shares ``kv_all``'s pointer), writes into the caller's sentinel-filled ``o``, syncs nowhere
    (``set_sync_debug_mode("error")`` around the whole execute, pre-pass included), and a second execute is bitwise identical."""
    q, kv, sink, idxs, scale = _case("cuda", batch=2, seq_len=256, ratio=2)  # K = 128 + 128 = 256 -> u_max_tiles 8
    st = _stage("d512", q, kv, idxs)
    assert st.needs_union_lists and not st.needs_idx_staging
    bufs = _buffers(st, q, idxs)
    bufs["o"].fill_(1.5e30)
    torch.cuda.set_sync_debug_mode("error")
    try:
        o1, lse1 = _run(st, q, kv, sink, idxs, bufs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert not bool((bufs["o"] == torch.full((), 1.5e30, dtype=q.dtype, device="cuda")).any()), "sentinel cells survived: O was not written in place"
    assert int(bufs["union_ntiles"].min()) >= 1 and not bool((bufs["union_ntiles"] == -7).any()), "the pre-pass must overwrite every union view"
    o1, lse1 = o1.clone(), lse1.clone()
    ref_o, _ = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    _assert_bf16_budget(o1, ref_o, kv, "d512 O")
    # the kernel launch ALONE (the union views already built) allocates nothing
    stream = torch.cuda.current_stream(q.device).cuda_stream
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    st.launch(q, kv, sink, idxs, bufs["o"], lse=bufs["lse"], lse_th=None, max_logits=None, idx_ws=None, stream=stream, **{k: bufs[k] for k in _D512_UNION})
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, f"the d512 launch allocated {torch.cuda.memory_allocated() - before} bytes"
    assert torch.equal(bufs["o"], o1) and torch.equal(bufs["lse"], lse1), "a second launch must be bitwise identical"
    # the whole execute (pre-pass + launch) is bitwise repeatable too; the pre-pass's allocation is the FLAGGED v1 cost
    before = torch.cuda.memory_allocated()
    _run(st, q, kv, sink, idxs, bufs)
    print(f"\nd512 execute (pre-pass + launch) allocated {torch.cuda.memory_allocated() - before} bytes transiently (FLAGGED torch pre-pass)")
    assert torch.equal(bufs["o"], o1) and torch.equal(bufs["lse"], lse1), "a second execute must be bitwise identical"


@requires_rubin
@requires_d512_fork
def test_d512_union_views_agree_with_the_stage_geometry():
    """The stage's ``union_shapes`` are the pre-pass contract (``[B, NC, 128 * u_max_tiles]``, ``[B, NC, u_max_tiles, 4, 4]``,
    ``[B, NC]``) at ``NC = ceil(S / 4)`` and ``u_max_tiles = ceil(4K / 128)``; after an execute every ``n_tiles >= 1`` and the
    tail cluster of an ``S % 4 != 0`` sequence exists."""
    from cudnn.mqa_sparse_attention_block.kernels.union_lists import n_clusters_for, u_max_tiles_for

    q, kv, sink, idxs, scale = _case("cuda", batch=1, seq_len=301, ratio=1)  # S % 4 == 1 -> a partial tail cluster; K = 128 + 301 -> 14 tiles
    st = _stage("d512", q, kv, idxs)
    k, nc, ut = int(idxs.shape[2]), n_clusters_for(301), u_max_tiles_for(int(idxs.shape[2]))
    assert st.union_shapes == dict(union_ids=(1, nc, 128 * ut), union_bits=(1, nc, ut, 4, 4), union_ntiles=(1, nc)), st.union_shapes
    bufs = _buffers(st, q, idxs)
    o, lse = _run(st, q, kv, sink, idxs, bufs)
    assert int(bufs["union_ntiles"].min()) >= 1 and int(bufs["union_ntiles"].max()) <= ut
    _assert_bf16_budget(o, R.sparse_attention_reference(q, kv, sink, idxs, scale)[0], kv, "d512 O at S=301")


@requires_rubin
def test_dsa_execute_allocates_nothing_and_is_bitwise_repeatable():
    """The dsa adapter binds the caller's / the block's buffers by pointer (no copy,
    no per-execute allocation, incl. the B=2 index staging) and is deterministic."""
    q, kv, sink, idxs, scale = _case("cuda", batch=2, seq_len=1024, ratio=2)
    st = _stage("dsa", q, kv, idxs)
    assert st.needs_idx_staging
    bufs = _buffers(st, q, idxs)
    o1, lse1 = _run(st, q, kv, sink, idxs, bufs)
    o1, lse1 = o1.clone(), lse1.clone()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _run(st, q, kv, sink, idxs, bufs)
    after = torch.cuda.memory_allocated()
    assert after == before, f"the dsa execute path allocated {after - before} bytes"
    assert torch.equal(bufs["o"], o1) and torch.equal(bufs["lse"], lse1), "a second execute must be bitwise identical"


@requires_rubin
def test_dsa_identity_index_path_refuses_a_misaligned_or_strided_list():
    """B=1 and K % 64 == 0: the caller's list is bound as a ``[T, K]`` view, so it must
    be contiguous and 32-byte aligned (the interface would otherwise CLONE it per call)."""
    q, kv, sink, idxs, _ = _case("cuda", batch=1, seq_len=256, ratio=1)  # K = 384
    st = _stage("dsa", q, kv, idxs)
    assert not st.needs_idx_staging
    bufs = _buffers(st, q, idxs)
    wide = torch.empty(1, 256, 384 + 1, device="cuda", dtype=torch.int32)
    wide[..., 1:] = idxs
    with pytest.raises(ValueError, match="contiguous"):
        _run(st, q, kv, sink, wide[..., 1:], bufs)
    flat = torch.empty(1 * 256 * 384 + 8, device="cuda", dtype=torch.int32)
    misaligned = flat[1 : 1 + 256 * 384].view(1, 256, 384)  # contiguous, base 4 bytes past a 32-byte boundary
    misaligned.copy_(idxs)
    with pytest.raises(ValueError, match="32-byte"):
        _run(st, q, kv, sink, misaligned, bufs)


# ---------------------------------------------------------------------------
# Any device
# ---------------------------------------------------------------------------


def test_torch_adapter_matches_the_oracle_on_cpu_at_a_small_head_geometry():
    """The pure-torch adapter is the oracle's op order; it runs anywhere."""
    q, kv, sink, idxs, scale = _case("cpu", batch=2, seq_len=96, ratio=1, h=4, d=64)
    out = torch.empty_like(q)
    lse = torch.full((2, 4, 96), math.nan, dtype=torch.float32)
    SA.torch_sparse_attention(q, kv, sink, idxs, scale, out=out, lse=lse)
    ref_o, ref_lse = R.sparse_attention_reference(q, kv, sink, idxs, scale)
    _assert_bf16_budget(out, ref_o, kv, "torch adapter CPU")
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-5)


def test_torch_adapter_rejects_inconsistent_operands():
    q = torch.empty(1, 8, 4, 64, dtype=torch.bfloat16)
    kv = torch.empty(1, 8, 64, dtype=torch.bfloat16)
    sink = torch.empty(4, dtype=torch.float32)
    idxs = torch.zeros(1, 8, 3, dtype=torch.int32)
    out = torch.empty_like(q)
    with pytest.raises(ValueError, match="kv_all must be"):
        SA.torch_sparse_attention(q, torch.empty(1, 8, 32, dtype=torch.bfloat16), sink, idxs, 0.125, out=out)
    with pytest.raises(ValueError, match="int32"):
        SA.torch_sparse_attention(q, kv, sink, idxs.long(), 0.125, out=out)
    with pytest.raises(ValueError, match="attn_sink"):
        SA.torch_sparse_attention(q, kv, sink.to(torch.bfloat16), idxs, 0.125, out=out)
    with pytest.raises(ValueError, match="lse must be"):
        SA.torch_sparse_attention(q, kv, sink, idxs, 0.125, out=out, lse=torch.empty(1, 8, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="out must be"):
        SA.torch_sparse_attention(q, kv, sink, idxs, 0.125, out=torch.empty(1, 8, 4, 32, dtype=torch.bfloat16))


def test_dsa_lse_fold_is_the_d1_select_chain():
    lse_h64 = torch.tensor([[math.inf, 1.0, 2.0, math.inf, 3.0]], dtype=torch.float32)
    sink = torch.tensor([0.5, 0.5, math.inf, -math.inf, -math.inf], dtype=torch.float32)
    got = SA.dsa_lse_fold(lse_h64, sink)[0]
    assert got[0].item() == 0.5  # keyless row with a finite sink -> sink
    assert torch.isclose(got[1], torch.logaddexp(torch.tensor(1.0), torch.tensor(0.5)))  # live row -> sink folded in
    assert got[2].item() == math.inf  # sink = +inf on a live row -> +inf, never NaN
    assert got[3].item() == -math.inf  # keyless row, no sink -> -inf
    assert got[4].item() == 3.0  # sink = -inf on a live row -> the plain log-sum-exp
    assert not torch.isnan(got).any()


def test_dsa_staging_predicates_and_padding():
    assert not SA.dsa_needs_index_staging(1, 640) and SA.dsa_needs_index_staging(2, 640) and SA.dsa_needs_index_staging(1, 428)
    assert SA.dsa_topk_padded(428) == 448 and SA.dsa_topk_padded(640) == 640 and SA.dsa_topk_padded(1) == 64


def test_dsa_stage_indices_masks_ids_outside_the_kv_range_in_place():
    """The D5 arithmetic on CPU against the brute-force spelling: ``(0 <= id < N) ? id + b * N : -1``,
    the pad columns ``-1``; ids ``< -1`` and ``>= N`` (incl. exactly ``N`` and huge values) are no key.
    All ``out=`` ops into the two workspace-shaped buffers -- nothing else is allocated."""
    b, s, k, n = 3, 5, 7, 40
    k_pad = SA.dsa_topk_padded(k)
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(-4, n + 4, (b, s, k), generator=g, dtype=torch.int32)
    idx[0, 0, :5] = torch.tensor([-1, 0, n - 1, n, 2**30], dtype=torch.int32)
    idx[2, 4, :2] = torch.tensor([-(2**30), -2], dtype=torch.int32)
    base_col = (torch.arange(b, dtype=torch.int32) * n).view(b, 1, 1)
    idx_tmp = torch.full((b, s, k), 12345, dtype=torch.int32)
    idx_ws = torch.full((b, s, k_pad), 12345, dtype=torch.int32)
    SA.dsa_stage_indices(idx, idx_tmp, idx_ws, base_col=base_col, n_kv_rows=n, topk=k, k_pad=k_pad)
    valid = (idx >= 0) & (idx < n)
    want = torch.where(valid, idx + base_col, torch.full_like(idx, -1))
    assert torch.equal(idx_ws[:, :, :k], want)
    assert torch.all(idx_ws[:, :, k:] == -1)
    assert idx_ws[0, 0, :5].tolist() == [-1, 0, n - 1, -1, -1] and idx_ws[2, 4, :2].tolist() == [-1, -1]
    # B == 1 (base_col None): the list is copied as is -- the kernel's own 0 <= token < kv.shape[0] mask is the contract there.
    ws1 = torch.full((1, s, k_pad), 12345, dtype=torch.int32)
    SA.dsa_stage_indices(idx[:1], None, ws1, base_col=None, n_kv_rows=n, topk=k, k_pad=k_pad)
    assert torch.equal(ws1[:, :, :k], idx[:1]) and torch.all(ws1[:, :, k:] == -1)
    with pytest.raises(RuntimeError, match="idx_tmp"):
        SA.dsa_stage_indices(idx, None, idx_ws, base_col=base_col, n_kv_rows=n, topk=k, k_pad=k_pad)


def test_dsa_index_alignment_helper():
    idxs = torch.zeros(4, 64, dtype=torch.int32)
    SA.check_dsa_index_alignment(idxs)  # contiguous, allocator-aligned
    with pytest.raises(ValueError, match="32-byte"):
        SA.check_dsa_index_alignment(torch.zeros(4 * 64 + 8, dtype=torch.int32)[1 : 1 + 4 * 64].view(4, 64))
    with pytest.raises(ValueError, match="contiguous"):
        SA.check_dsa_index_alignment(torch.zeros(4, 65, dtype=torch.int32)[:, :64])


def test_d512_adapter_accepts_once_the_fork_exists_and_declines_typed_before():
    """INVERTED when the fork landed: with the module present a well-formed (64, 512) declaration passes ``check_support`` on
    Rubin (and off Rubin fails on the ARCH, typed); without it the decline is the typed "has not landed"."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ad = SA.D512SparseAttention(batch=1, seq_len=8, n_kv_rows=8, n_heads=64, head_dim=512, topk=8, scale=512**-0.5, dtype=torch.bfloat16, device=dev)
    if not SA.d512_fork_available():
        with pytest.raises(NotImplementedError, match="has not landed"):
            ad.check_support()
    elif _cc() == _SM107:
        ad.check_support()  # accept: the specialized module is loaded (its validator ran), nothing raised
        assert ad.params is not None and ad.u_max_tiles == 1 and ad.union_shapes["union_ids"] == (1, 2, 128)
    else:
        with pytest.raises(NotImplementedError, match="Rubin"):
            ad.check_support()


@requires_cuda
def test_dsa_adapter_declines_a_head_geometry_outside_its_variants():
    ad = SA.DsaSparseAttention(batch=1, seq_len=8, n_kv_rows=8, n_heads=4, head_dim=64, topk=8, scale=0.125, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="n_heads, head_dim"):
        ad.check_support()


def test_stage_rejects_an_unknown_adapter_name():
    geom = MqaSparseAttentionBlockGeometry(n_heads=64, head_dim=512)
    st = _SparseAttention(impl="bogus", geom=geom, batch=1, seq_len=8, n_kv_rows=8, topk=8, dtype=torch.bfloat16, device="cpu", want_lse=False)
    with pytest.raises(ValueError, match="attention must be one of"):
        st.check_support()
