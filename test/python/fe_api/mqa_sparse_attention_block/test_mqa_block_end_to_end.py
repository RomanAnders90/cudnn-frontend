# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole MQA sparse-attention block, one call, against the pure-torch oracle.

What is under test here is the ASSEMBLY: the workspace carve, the strided
column slices of the grouped ``o_a_proj``, the prefix write into the caller's
``kv_all``, the stage order, the index staging and LSE fold of the ``dsa``
adapter, the ``union_lists`` pre-pass slots of the ``d512`` adapter (rows skip
typed until the fork module exists), and that every stage runs on the caller's
stream.  Each stage is
checked on its own elsewhere; here every intermediate is read back through its
workspace view and the final output is held to ``cos > 0.999`` against
``mqa_block_reference.block_reference``.

Accept tests need Rubin (the block's target); every ``check_support`` decline of
plan section 2.4 is exercised on any device -- except the BLOCK-level
``attention="dsa"`` with a non-DSA head geometry, which the arch decline precedes
off Rubin; it is pinned at adapter level instead
(``test_mqa_block_sparse_attention_stage.py::test_dsa_adapter_declines_a_head_geometry_outside_its_variants``).
The execute-time preamble (``_check_bound_tensors``: every runtime tensor against
the declaration BEFORE the first launch) is unit-tested on CPU here and end to end
on Rubin.
"""

import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("MQA sparse-attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

# Rootdir-qualified: a bare `import mqa_block_reference` would collide with other fe_api packages in one session.
from fe_api.mqa_sparse_attention_block import mqa_block_reference as R  # noqa: E402

import cudnn.mqa_sparse_attention_block as M  # noqa: E402
from cudnn.mqa_sparse_attention_block import MqaSparseAttentionBlockFwd, MqaSparseAttentionBlockGeometry  # noqa: E402
from cudnn.mqa_sparse_attention_block.api import RUNNER_NAMES, TENSOR_NAMES, _check_bound_tensors, _view  # noqa: E402
from cudnn.mqa_sparse_attention_block.kernels import sparse_attention as SA  # noqa: E402
from cudnn.mqa_sparse_attention_block.kernels.union_lists import u_max_tiles_for  # noqa: E402

_SM107 = (10, 7)


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_rubin = pytest.mark.skipif(_cc() != _SM107, reason=f"the block targets SM107 only; found {_cc()}")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# 64 heads x 512 (the DSA variant and the released head geometry) at small d_model / LoRA ranks / lists.
_TINY = dict(d_model=512, n_heads=64, head_dim=512, rope_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=8, window=16, index_topk=8)
requires_d512_fork = pytest.mark.skipif(not SA.d512_fork_available(), reason=f"the d512 fork module {SA.D512_FORK_MODULE} has not landed (wave W3b)")
_D512 = pytest.param("d512", marks=requires_d512_fork)
_ADAPTERS = ["torch", "dsa", _D512]
_KERNEL_ADAPTERS = ["dsa", _D512]  # the contract-clean kernel adapters (no allocation, no sync on the launch)
_D512_UNION = ("union_ids", "union_bits", "union_ntiles")
_BF16 = torch.bfloat16


def _geoms(**kw):
    return MqaSparseAttentionBlockGeometry(**kw), R.RefGeometry(**kw)


def _build_inputs(ref_geom, batch, seq_len, ratio, device="cuda"):
    """The oracle's seeded inputs plus the block's ``kv_all`` (compressed rows filled, window rows zero)."""
    inp = R.make_inputs(ref_geom, batch, seq_len, ratio, device=device)
    n_c = 0 if inp["compress_kv"] is None else int(inp["compress_kv"].shape[1])
    kv_all = torch.zeros(batch, seq_len + n_c, ref_geom.head_dim, device=device, dtype=_BF16)
    if n_c:
        kv_all[:, seq_len:] = inp["compress_kv"]
    return inp, kv_all


def _args(inp, kv_all, out):
    """The 14 tensors in constructor / execute order."""
    return (
        inp["x"],
        inp["w_q_a"],
        inp["w_q_norm"],
        inp["w_q_b"],
        inp["w_kv"],
        inp["w_kv_norm"],
        inp["w_o_a"],
        inp["w_o_b"],
        inp["attn_sink"],
        inp["cos"],
        inp["sin"],
        kv_all,
        inp["topk_idxs"],
        out,
    )


def _run_block(geom_kw, *, batch, seq_len, ratio, return_lse=False, **blk_kw):
    bg, rg = _geoms(**geom_kw)
    inp, kv_all = _build_inputs(rg, batch, seq_len, ratio)
    ref = R.block_reference(rg, inp)
    out = torch.empty(batch, seq_len, bg.d_model, device="cuda", dtype=_BF16)
    args = _args(inp, kv_all, out)
    blk = MqaSparseAttentionBlockFwd(*args, bg, return_lse=return_lse, **blk_kw)
    blk.check_support()
    blk.compile()
    ws = torch.empty(blk.get_workspace_size(), dtype=torch.uint8, device="cuda")
    lse = torch.full((batch, bg.n_heads, seq_len), math.nan, device="cuda", dtype=torch.float32) if return_lse else None
    blk.execute(*args, ws, lse=lse)
    torch.cuda.synchronize()
    return SimpleNamespace(out=out, lse=lse, ref=ref, blk=blk, inp=inp, kv_all=kv_all, ws=ws, args=args, geom=bg)


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _assert_stage(got, want, label, *, rtol, atol_rel):
    """cos > 0.999 (the structural bar -- a wrong RoPE pairing, a swapped group or a
    skipped norm all drop it far below) plus a loose element bar sized to the bf16
    chain (each stage adds ~1 bf16 rounding on top of the GEMM's accumulation order)."""
    g, w = got.float(), want.float()
    assert torch.isfinite(g).all(), f"{label}: non-finite"
    c = _cos(g, w)
    assert c > 0.999, f"{label}: cos {c:.5f}"
    torch.testing.assert_close(g, w, rtol=rtol, atol=atol_rel * float(w.abs().max()), msg=lambda m: f"{label}: {m}")


def _check_intermediates(r):
    """Every workspace intermediate against the oracle's stage outputs."""
    bg, ref, ws, sl = r.geom, r.ref, r.ws, r.blk.workspace_slots()
    b, s = ref.out.shape[0], ref.out.shape[1]
    t = b * s
    qa = _view(ws, sl.qa, (t, bg.q_lora_rank), _BF16)
    _assert_stage(qa, ref.qr.reshape(t, bg.q_lora_rank), "qa (stages 1+2)", rtol=2**-5, atol_rel=2**-5)
    q = _view(ws, sl.q, (t, bg.n_heads, bg.head_dim), _BF16)
    _assert_stage(q, ref.q.reshape(t, bg.n_heads, bg.head_dim), "q (stages 3+4)", rtol=2**-5, atol_rel=2**-5)
    kv = r.kv_all[:, :s]
    # The fake-quant grid steps by an e4m3 ulp of the block's amax, so a 1-ulp GEMM difference can move one
    # element by ~12 %: a generous element bar, the cosine, and "most elements identical" carry this stage.
    _assert_stage(kv, ref.window_kv, "kv_all[:, :S] (stages 5+6)", rtol=2**-2, atol_rel=2**-4)
    assert (kv == ref.window_kv).float().mean().item() > 0.5, "kv_all[:, :S]: fewer than half the elements agree bit for bit"
    if ref.kv_all is not None and ref.kv_all.shape[1] > s:
        assert torch.equal(r.kv_all[:, s:], ref.kv_all[:, s:]), "the block touched the caller's compressed rows"
    o = _view(ws, sl.o, (t, bg.n_heads, bg.head_dim), _BF16)  # in place: post inverse-RoPE
    _assert_stage(o, ref.o_unrot.reshape(t, bg.n_heads, bg.head_dim), "o (stages 7+8)", rtol=2**-4, atol_rel=2**-5)
    o_lora = _view(ws, sl.o_lora, (t, bg.n_o_lora), _BF16)
    _assert_stage(o_lora, ref.o_lora.reshape(t, bg.n_o_lora), "o_lora (stage 9)", rtol=2**-4, atol_rel=2**-5)
    if r.blk.attention == "d512":
        # The pre-pass slots: every cluster has >= 1 union tile (never 0 -- the kernel's KV loop always runs) and the
        # flat union ids stay inside the [B * N, D] row space or are the -1 pad.
        nt = r.blk.workspace_view(ws, "union_ntiles")
        ids = r.blk.workspace_view(ws, "union_ids")
        assert nt.shape[0] == b and int(nt.min()) >= 1, nt
        assert int(ids.max()) < b * int(r.kv_all.shape[1]) and int(ids.min()) >= -1


# ---------------------------------------------------------------------------
# Accept -- Rubin
# ---------------------------------------------------------------------------


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_block_matches_the_reference_at_a_small_geometry(attention, ratio):
    r = _run_block(_TINY, batch=1, seq_len=256, ratio=ratio, attention=attention)
    _check_intermediates(r)
    c = _cos(r.out, r.ref.out)
    assert c > 0.999, f"{attention} ratio={ratio}: cos(out, ref) = {c:.5f}"


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_block_at_batch_two(attention):
    """B=2: the batched prefix ``kv_proj`` (batch stride ``(S + N_c) * 512``) and, under
    ``dsa``, the flat global-id staging."""
    r = _run_block(_TINY, batch=2, seq_len=128, ratio=2, attention=attention)
    if attention == "dsa":
        assert r.blk._attention.needs_idx_staging
    _check_intermediates(r)
    c = _cos(r.out, r.ref.out)
    assert c > 0.999, f"{attention} B=2: cos(out, ref) = {c:.5f}"


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_block_on_a_window_only_layer(attention):
    """``has_compressed_kv=False``: ``kv_all`` is ``[B, S, D]`` and the list is the window alone."""
    kw = dict(_TINY, has_compressed_kv=False, index_topk=0)
    r = _run_block(kw, batch=1, seq_len=200, ratio=0, attention=attention)
    assert r.kv_all.shape[1] == 200
    _check_intermediates(r)
    assert _cos(r.out, r.ref.out) > 0.999


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_block_accepts_s_smaller_than_the_compress_ratio(attention):
    """The legal degenerate: ``S=1 < ratio=2`` gives ``N_c == 0`` under ``has_compressed_kv=True``
    and a one-slot window list (``K == min(S, window)``)."""
    r = _run_block(_TINY, batch=1, seq_len=1, ratio=2, attention=attention)
    assert r.kv_all.shape[1] == 1 and r.inp["topk_idxs"].shape[2] == 1
    assert torch.isfinite(r.out.float()).all()
    assert _cos(r.out, r.ref.out) > 0.999


# The synthetic Indexer tail caps the compressed list at S // ratio ids (``compressed_idxs_synthetic``), so the released
# K = 640 (window 128 ++ top-k 512 at ratio 2) exists only from S = 1024 on; below it the d512 pitch is ceil(4K / 128) of the
# SHORTER list.  The d512 arm adds S = 1024 so the released union pitch (u_max_tiles = 20) is really exercised end to end.
_RELEASED_SHAPE_CASES = [
    pytest.param(256, "dsa", id="256-dsa"),
    pytest.param(512, "dsa", id="512-dsa"),
    pytest.param(256, "d512", marks=requires_d512_fork, id="256-d512"),
    pytest.param(512, "d512", marks=requires_d512_fork, id="512-d512"),
    pytest.param(1024, "d512", marks=requires_d512_fork, id="1024-d512"),
]


@requires_rubin
@pytest.mark.parametrize("seq_len, attention", _RELEASED_SHAPE_CASES)
def test_block_at_the_released_shape(seq_len, attention):
    """The released geometry (5120 x 64 heads x 512, LoRA 1280 / 1024 x 8, window 128, top-k 512)
    at a compressed layer (ratio 2): every GEMM at its real N; dsa on the identity index path,
    d512 at the union pitch of the list the generator can build at ``S`` -- K = 640 / u_max_tiles = 20
    (the released pitch) from S = 1024 on, K = 128 + S // 2 below it."""
    r = _run_block({}, batch=1, seq_len=seq_len, ratio=2, attention=attention)
    k = int(r.inp["topk_idxs"].shape[2])
    assert k == 128 + min(512, seq_len // 2), f"the generator's list width at S={seq_len}: {k}"
    if attention == "dsa":
        assert not r.blk._attention.needs_idx_staging
    else:
        info = r.blk.attention_info()
        assert info["u_max_tiles"] == u_max_tiles_for(k), f"u_max_tiles {info['u_max_tiles']} != ceil(4 x {k} / 128)"
        if seq_len >= 1024:
            assert k == 640 and info["u_max_tiles"] == 20, "the released union pitch"
    c = _cos(r.out, r.ref.out)
    assert c > 0.999, f"released shape S={seq_len} {attention}: cos(out, ref) = {c:.5f}"


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_lse_is_optional_and_returned_in_the_frost_convention(attention):
    r = _run_block(_TINY, batch=1, seq_len=256, ratio=1, attention=attention, return_lse=True)
    assert r.lse.shape == r.ref.lse.shape and torch.isfinite(r.lse).all()
    # Plumbing bar: the stage test pins the LSE at 1e-4; here q / kv carry the GEMMs' bf16 roundings into the logits.
    torch.testing.assert_close(r.lse, r.ref.lse, rtol=1e-2, atol=5e-2)
    # A block declared without return_lse refuses an lse buffer rather than ignoring it.
    r0 = _run_block(_TINY, batch=1, seq_len=64, ratio=0, attention=attention)
    with pytest.raises(ValueError, match="return_lse"):
        r0.blk.execute(*r0.args, r0.ws, lse=torch.empty(1, 64, 64, device="cuda", dtype=torch.float32))


@requires_rubin
@pytest.mark.parametrize("attention", _KERNEL_ADAPTERS)
def test_second_execute_is_bitwise_identical_and_never_syncs(attention):
    r = _run_block(_TINY, batch=2, seq_len=128, ratio=2, attention=attention, return_lse=True)
    out1, lse1 = r.out.clone(), r.lse.clone()
    torch.cuda.set_sync_debug_mode("error")
    try:
        r.blk.execute(*r.args, r.ws, lse=r.lse)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.equal(r.out, out1) and torch.equal(r.lse, lse1)


def _park_the_default_stream(seconds: float = 0.5) -> None:
    """Enqueue a long spin on the CURRENT (default) stream so anything a stage wrongly launches there runs LATE."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(seconds * 2.0e9))
        return
    x = torch.randn(8192, 8192, device="cuda", dtype=_BF16)
    for _ in range(16):
        x = x @ x


@requires_rubin
@pytest.mark.parametrize("attention", _KERNEL_ADAPTERS)
@pytest.mark.parametrize("how", ["ambient", "explicit"])
def test_a_caller_stream_orders_every_stage(how, attention):
    """Every stage -- five FROST GEMMs, the torch pointwise stages, the attention kernel
    (and the d512 union-list pre-pass) -- launches on ONE stream, the caller's (ambient
    ``torch.cuda.stream`` or explicit ``current_stream=``).  The workspace is zeroed and
    the default stream parked, so a stage enqueued there runs late and the output
    differs from the default-stream run; correct threading gives a BIT-IDENTICAL result."""
    import cuda.bindings.driver as cuda_drv

    r = _run_block(_TINY, batch=1, seq_len=256, ratio=1, attention=attention)
    ref_out = r.out.clone()
    ws = torch.zeros_like(r.ws)
    out = torch.zeros_like(r.out)
    args = r.args[:-1] + (out,)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    _park_the_default_stream()
    if how == "ambient":
        with torch.cuda.stream(side):
            r.blk.execute(*args, ws)
    else:
        r.blk.execute(*args, ws, current_stream=cuda_drv.CUstream(side.cuda_stream))
    with torch.cuda.stream(side):
        ws.zero_()  # ordered AFTER the block on the side stream; a consumer parked on the default stream would read this instead
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out), f"a stage ran off the caller's stream ({how}): max|diff| = {(out.float() - ref_out.float()).abs().max().item()}"


@requires_rubin
@pytest.mark.parametrize("attention", _KERNEL_ADAPTERS)
def test_workspace_is_sized_honestly(attention):
    r = _run_block(_TINY, batch=1, seq_len=128, ratio=1, attention=attention)
    req = r.blk.get_workspace_size()
    with pytest.raises(ValueError, match="workspace is"):
        r.blk.execute(*r.args, torch.empty(req - 1, dtype=torch.uint8, device="cuda"))
    padded = torch.full((req + 256,), 0xA5, dtype=torch.uint8, device="cuda")
    r.blk.execute(*r.args, padded)
    torch.cuda.synchronize()
    assert torch.all(padded[req:] == 0xA5), "the block wrote past get_workspace_size()"
    assert torch.equal(r.out, r.out), "sanity"


@requires_rubin
def test_execute_declines_a_bad_stream_type_and_a_misaligned_index_list():
    r = _run_block(_TINY, batch=1, seq_len=64, ratio=0, attention="dsa")  # K = 16 -> staging path
    with pytest.raises(TypeError, match="CUstream"):
        r.blk.execute(*r.args, r.ws, current_stream=torch.cuda.Stream())
    # The identity index path (B=1, K % 64 == 0) refuses a misaligned list rather than letting the interface clone it.
    kw = dict(_TINY, window=64, index_topk=0, has_compressed_kv=False)
    r1 = _run_block(kw, batch=1, seq_len=128, ratio=0, attention="dsa")  # K = 64
    assert not r1.blk._attention.needs_idx_staging
    idx = r1.inp["topk_idxs"]
    flat = torch.empty(idx.numel() + 8, dtype=torch.int32, device="cuda")
    misaligned = flat[1 : 1 + idx.numel()].view_as(idx)
    misaligned.copy_(idx)
    args = list(r1.args)
    args[12] = misaligned
    r1.kv_all[:, :128].fill_(7.0)  # the prefix the block would overwrite in stage 5
    with pytest.raises(ValueError, match="32-byte"):
        r1.blk.execute(*args, r1.ws)
    torch.cuda.synchronize()
    assert torch.all(r1.kv_all[:, :128] == 7.0), "the alignment decline fired AFTER stage 5 had launched (validate, then launch)"


@requires_rubin
def test_execute_declines_a_mismatched_tensor_before_any_launch():
    """Every runtime tensor is checked against the DECLARATION in the preamble -- shape,
    dtype, contiguity, device -- with a ValueError naming it, and nothing has launched:
    the caller's ``kv_all`` prefix / ``out`` (sentinel-filled) are untouched.  Without this a
    short ``w_kv`` reaches the graph-route GEMM as a bare pointer (read past its end), a
    non-contiguous ``x`` dies in ``.view()`` with a RuntimeError, a ``[B, S, rope_dim]`` cos
    broadcasts in the torch RoPE."""
    r = _run_block(_TINY, batch=1, seq_len=128, ratio=1, attention="dsa")
    g = r.geom

    def attempt(name, value, match):
        args = list(r.args)
        args[_IDX[name]] = value
        r.kv_all[:, :128].fill_(7.0)
        args[_IDX["out"]].fill_(7.0)
        with pytest.raises(ValueError, match=match):
            r.blk.execute(*args, r.ws)
        torch.cuda.synchronize()
        assert torch.all(r.kv_all[:, :128] == 7.0) and torch.all(args[_IDX["out"]] == 7.0), f"{name}: a stage launched before the decline"

    attempt("w_kv", torch.empty(g.head_dim, g.d_model // 2, device="cuda", dtype=_BF16), "w_kv must have the declared shape")
    attempt("x", r.inp["x"].transpose(1, 2).contiguous().transpose(1, 2), "x must be contiguous")
    attempt("cos", torch.empty(1, 128, g.rope_dim, device="cuda", dtype=torch.float32), "cos must have the declared shape")
    attempt("w_q_norm", r.inp["w_q_norm"].float(), "w_q_norm must have the declared dtype")
    attempt("w_o_b", r.inp["w_o_b"].cpu(), "w_o_b must live on the declared device")
    attempt("topk_idxs", r.inp["topk_idxs"].long(), "topk_idxs must have the declared dtype")


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_execute_declines_a_bad_lse_buffer_under_return_lse(attention):
    # S != n_heads on purpose: at S == H the "[B, S, H]" tensor below IS a [B, H, S] buffer and rightly passes
    # (the first cut used S = 64 = n_heads and this case could not fail).
    s, h = 48, _TINY["n_heads"]
    assert s != h
    r = _run_block(_TINY, batch=1, seq_len=s, ratio=0, attention=attention, return_lse=True)
    for bad in (
        torch.empty(1, s, h, device="cuda", dtype=torch.float32),  # [B, S, H]: the wrong convention
        torch.empty(1, h, s, device="cuda", dtype=torch.float16),  # fp16
        torch.empty(1, h, s + 1, device="cuda", dtype=torch.float32)[:, :, 1:],  # non-contiguous
        None,  # required when return_lse=True
    ):
        with pytest.raises(ValueError, match="lse must be"):
            r.blk.execute(*r.args, r.ws, lse=bad)


@requires_rubin
@pytest.mark.parametrize("attention", _ADAPTERS)
def test_stage_runners_reproduce_execute_and_restore_their_in_place_inputs(attention):
    """``stage_runners`` is ``execute`` cut at the stage seams: the names / kinds are the
    documented pipeline (``idx_staging`` present iff the dsa path stages), running the list
    in order is bit-identical to ``execute``, every in-place stage's ``prepare`` restores
    its input so ``prepare(); run()`` twice leaves the state bit-identical, and the
    contract-clean runners (GEMMs, dsa kernel, index staging) allocate nothing."""
    r = _run_block(_TINY, batch=2, seq_len=64, ratio=2, attention=attention, return_lse=True)
    bg, sl, t = r.geom, r.blk.workspace_slots(), 2 * 64
    slots = dict(qa=(sl.qa, (t, bg.q_lora_rank), _BF16), q=(sl.q, (t, bg.n_heads, bg.head_dim), _BF16), o=(sl.o, (t, bg.n_heads, bg.head_dim), _BF16))
    slots["o_lora"] = (sl.o_lora, (t, bg.n_o_lora), _BF16)
    if sl.lse_th is not None:
        slots["lse_th"] = (sl.lse_th, (t, bg.n_heads), torch.float32)
    if sl.idx_ws is not None:
        slots["idx_ws"] = (sl.idx_ws, (2, 64, r.blk._attention.k_pad), torch.int32)
    for name, shape in r.blk._attention.union_shapes.items():  # d512: the three pre-pass views
        slots[name] = (getattr(sl, name), tuple(shape), torch.int32)

    def intermediates():
        return {k: _view(r.ws, off, shape, dt).clone() for k, (off, shape, dt) in slots.items()}  # the SLOTS, not the padding / engine scratch

    ref_out, ref_lse, ref_ws = r.out.clone(), r.lse.clone(), intermediates()
    runners = r.blk.stage_runners(*r.args, r.ws, lse=r.lse)
    names = [x.name for x in runners]
    live = {"idx_staging": attention == "dsa" and r.blk._attention.needs_idx_staging, "union_lists": attention == "d512"}
    want = [n for n, _ in RUNNER_NAMES if live.get(n, True)]
    assert names == want, names
    assert all(x.kind == dict(RUNNER_NAMES)[x.name] for x in runners)
    assert [x.name for x in runners if x.prepare is not None] == ["q_norm", "q_rope", "kv_chain", "o_unrope"]
    r.ws.zero_()
    r.out.zero_()
    r.lse.zero_()
    for x in runners:
        x.run()
    torch.cuda.synchronize()
    assert torch.equal(r.out, ref_out) and torch.equal(r.lse, ref_lse), "the runner list is not execute()"
    assert all(torch.equal(v, ref_ws[k]) for k, v in intermediates().items()), "workspace state after the runner list differs from execute()'s"
    # prepare() restores the in-place input: a timing loop's prepare/run leaves everything where the chain left it.
    for x in runners:
        if x.prepare is None:
            continue
        for _ in range(2):
            x.prepare()
            x.run()
        torch.cuda.synchronize()
        got = intermediates()
        assert all(torch.equal(got[k], ref_ws[k]) for k in ref_ws) and torch.equal(r.lse, ref_lse), f"{x.name}: prepare()/run() drifted the workspace"
    # The contract-clean runners allocate nothing -- measured PER RUNNER while the WHOLE pipeline runs in order.
    # Re-running the clean subset alone is not a valid probe: q_a_proj / kv_proj overwrite qa / kv with un-normed
    # values that the dsa kernel (itself contract-clean) then consumes, so out != ref_out (the first cut did that
    # and only the torch arm, whose attention is not in the set, happened to pass).
    # (d512: the kernel launch is clean; its torch union_lists pre-pass is FLAGGED and allocates by design.)
    clean = {"q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"}
    clean |= {"idx_staging", "sparse_attention"} if attention == "dsa" else ({"sparse_attention"} if attention == "d512" else set())
    for x in runners:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        x.run()
        torch.cuda.synchronize()
        if x.name in clean:
            assert torch.cuda.memory_allocated() == before, f"contract-clean runner {x.name} allocated {torch.cuda.memory_allocated() - before} bytes"
    assert torch.equal(r.out, ref_out) and torch.equal(r.lse, ref_lse), "the full runner pass after the allocation probe is not execute()"
    # The hook validates like execute: return_lse=True needs the lse buffer.
    with pytest.raises(ValueError, match="lse must be"):
        r.blk.stage_runners(*r.args, r.ws)


# ---------------------------------------------------------------------------
# Reject -- any device (plan section 2.4)
# ---------------------------------------------------------------------------


def _samples(geom, *, batch=1, seq_len=32, k=None, n_c=None, device=None):
    """Empty tensors in the declared layouts (no values are read by check_support)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = geom
    n_c = (seq_len // 2 if g.has_compressed_kv else 0) if n_c is None else n_c
    k = (min(seq_len, g.window) + (min(g.index_topk, n_c) if g.has_compressed_kv else 0)) if k is None else k

    def t(*shape, dt=_BF16):
        return torch.empty(*shape, dtype=dt, device=device)

    return [
        t(batch, seq_len, g.d_model),
        t(g.q_lora_rank, g.d_model),
        t(g.q_lora_rank),
        t(g.n_heads * g.head_dim, g.q_lora_rank),
        t(g.head_dim, g.d_model),
        t(g.head_dim),
        t(g.o_groups, g.o_lora_rank, g.group_width),
        t(g.d_model, g.n_o_lora),
        t(g.n_heads, dt=torch.float32),
        t(batch, seq_len, g.rope_dim // 2, dt=torch.float32),
        t(batch, seq_len, g.rope_dim // 2, dt=torch.float32),
        t(batch, seq_len + n_c, g.head_dim),
        t(batch, seq_len, k, dt=torch.int32),
        t(batch, seq_len, g.d_model),
    ]


_TINY_GEOM = MqaSparseAttentionBlockGeometry(**_TINY)
_IDX = dict(x=0, w_q_a=1, w_q_norm=2, w_q_b=3, w_kv=4, w_kv_norm=5, w_o_a=6, w_o_b=7, attn_sink=8, cos=9, sin=10, kv_all=11, topk_idxs=12, out=13)


def _declines(exc, match, geom=_TINY_GEOM, replace=None, **blk_kw):
    args = _samples(geom)
    for name, value in (replace or {}).items():
        args[_IDX[name]] = value
    blk = MqaSparseAttentionBlockFwd(*args, geom, **blk_kw)
    with pytest.raises(exc, match=match):
        blk.check_support()


def test_declines_an_invalid_geometry():
    _declines(ValueError, "rope_dim", geom=MqaSparseAttentionBlockGeometry(**dict(_TINY, rope_dim=3)))


def test_declines_a_non_bf16_activation():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _declines(NotImplementedError, "bf16 only", replace=dict(x=torch.empty(1, 32, _TINY_GEOM.d_model, dtype=torch.float16, device=dev)))


def test_declines_a_weight_or_output_dtype_that_disagrees_with_x():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = _TINY_GEOM
    _declines(ValueError, "w_q_b must have x's dtype", replace=dict(w_q_b=torch.empty(g.n_heads * g.head_dim, g.q_lora_rank, dtype=torch.float16, device=dev)))
    _declines(ValueError, "kv_all must have x's dtype", replace=dict(kv_all=torch.empty(1, 48, g.head_dim, dtype=torch.float16, device=dev)))
    _declines(ValueError, "out must have x's dtype", replace=dict(out=torch.empty(1, 32, g.d_model, dtype=torch.float32, device=dev)))


def test_declines_a_bad_sink_or_rope_table():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = _TINY_GEOM
    _declines(ValueError, "attn_sink must be fp32", replace=dict(attn_sink=torch.empty(g.n_heads, dtype=_BF16, device=dev)))
    _declines(ValueError, "attn_sink tensor shape mismatch", replace=dict(attn_sink=torch.empty(g.n_heads + 1, dtype=torch.float32, device=dev)))
    _declines(ValueError, "cos tensor shape mismatch", replace=dict(cos=torch.empty(1, 32, g.rope_dim, dtype=torch.float32, device=dev)))
    _declines(ValueError, "sin must be fp32", replace=dict(sin=torch.empty(1, 32, g.rope_dim // 2, dtype=_BF16, device=dev)))


def test_declines_a_non_contiguous_kv_all_and_a_bad_index_list():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = _TINY_GEOM
    _declines(ValueError, "kv_all must be contiguous", replace=dict(kv_all=torch.empty(1, g.head_dim, 48, dtype=_BF16, device=dev).transpose(1, 2)))
    _declines(ValueError, "kv_all must be", replace=dict(kv_all=torch.empty(1, 16, g.head_dim, dtype=_BF16, device=dev)))  # fewer rows than S
    _declines(ValueError, "int32", replace=dict(topk_idxs=torch.empty(1, 32, 24, dtype=torch.int64, device=dev)))
    _declines(ValueError, "n_kv_slots_max", replace=dict(topk_idxs=torch.empty(1, 32, g.n_kv_slots_max + 1, dtype=torch.int32, device=dev)))
    _declines(ValueError, "topk_idxs must be", replace=dict(topk_idxs=torch.empty(1, 31, 24, dtype=torch.int32, device=dev)))


def test_declines_compressed_rows_on_a_window_only_layer():
    g = MqaSparseAttentionBlockGeometry(**dict(_TINY, has_compressed_kv=False, index_topk=0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = _samples(g, n_c=0)
    args[_IDX["kv_all"]] = torch.empty(1, 32 + 8, g.head_dim, dtype=_BF16, device=dev)  # N_c = 8 on a window-only layer
    with pytest.raises(ValueError, match="has_compressed_kv"):
        MqaSparseAttentionBlockFwd(*args, g).check_support()


def test_accepts_n_c_zero_under_has_compressed_kv_only_with_the_window_list():
    """``N_c == 0`` under ``has_compressed_kv=True`` is the legal ``S < ratio`` degenerate,
    accepted iff the list is the window list alone (``K == min(S, window)``)."""
    g = _TINY_GEOM
    args = _samples(g, n_c=0, k=min(32, g.window) + 1)
    with pytest.raises(ValueError, match="window list alone"):
        MqaSparseAttentionBlockFwd(*args, g).check_support()
    args = _samples(g, n_c=0, k=min(32, g.window))
    blk = MqaSparseAttentionBlockFwd(*args, g)
    if _cc() == _SM107:
        assert blk.check_support()
    else:
        with pytest.raises((NotImplementedError, ValueError)) as ei:  # the arch decline (CUDA) or the device decline (CPU)
            blk.check_support()
        assert "has_compressed_kv" not in str(ei.value) and "window list" not in str(ei.value)


def test_declines_unknown_knob_values_in_the_constructor():
    args = _samples(_TINY_GEOM)
    with pytest.raises(ValueError, match="attention must be one of"):
        MqaSparseAttentionBlockFwd(*args, _TINY_GEOM, attention="bogus")
    with pytest.raises(ValueError, match="pointwise_impl must be one of"):
        MqaSparseAttentionBlockFwd(*args, _TINY_GEOM, pointwise_impl="bogus")
    with pytest.raises(TypeError, match="geometry"):
        MqaSparseAttentionBlockFwd(*args, R.RefGeometry(**_TINY))


def test_d512_is_accepted_once_the_fork_exists_and_declined_typed_before():
    """INVERTED when the fork landed: the block-level ``attention="d512"`` declaration passes ``check_support`` on Rubin;
    off Rubin the ARCH decline follows (never "has not landed"); without the module the typed "has not landed"."""
    args = _samples(_TINY_GEOM)
    blk = MqaSparseAttentionBlockFwd(*args, _TINY_GEOM, attention="d512")
    if not SA.d512_fork_available():
        with pytest.raises(NotImplementedError, match="has not landed"):
            blk.check_support()
    elif _cc() == _SM107:
        assert blk.check_support()
        assert blk._attention.needs_union_lists and blk.attention_info()["adapter"] == "d512"
    else:
        with pytest.raises((NotImplementedError, ValueError)) as ei:  # the arch decline (CUDA) or the device decline (CPU)
            blk.check_support()
        assert "has not landed" not in str(ei.value)


def test_declines_k_above_n_kv_slots_max_under_d512_before_any_device_check():
    """``K > n_kv_slots_max`` is the geometry's ``ValueError`` on every adapter, ahead of the fork-present and arch checks."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = _TINY_GEOM
    _declines(ValueError, "n_kv_slots_max", replace=dict(topk_idxs=torch.empty(1, 32, g.n_kv_slots_max + 1, dtype=torch.int32, device=dev)), attention="d512")


@pytest.mark.skipif(not SA.d512_fork_available(), reason="with the fork absent the 'has not landed' decline precedes the head-geometry one")
def test_declines_d512_on_a_head_geometry_other_than_64x512():
    """The fork's work item is 4 tokens x 64 heads x 512 per cluster: 32 heads (TP=2) or another head_dim is a typed decline
    at knob-check time, device-independent (the block's ``_check_knobs`` runs ahead of the arch check)."""
    for kw in (dict(n_heads=32), dict(n_heads=128), dict(head_dim=256, q_lora_rank=256)):
        geom = MqaSparseAttentionBlockGeometry(**dict(_TINY, **kw))
        _declines(NotImplementedError, "64", geom=geom, attention="d512")


@pytest.mark.skipif(
    importlib.util.find_spec("cudnn.mqa_sparse_attention_block.kernels.norm_rope") is not None, reason="kernels/norm_rope.py exists in this checkout"
)
def test_declines_the_frost_pointwise_arm_until_the_kernel_lands():
    _declines(NotImplementedError, "has not landed", pointwise_impl="frost")


def test_declines_three_roundings_off_on_the_torch_pointwise_arm():
    _declines(NotImplementedError, "three_roundings", pointwise_impl="torch", three_roundings=False)


@requires_cuda
@pytest.mark.skipif(_cc() == _SM107, reason="this IS the target arch")
def test_declines_a_non_rubin_device_typed():
    _declines(NotImplementedError, "targets Rubin")


def test_bound_tensor_check_names_the_offender_on_any_device():
    """``_check_bound_tensors`` (the execute preamble) against the declaration: shape, dtype,
    contiguity, device -- one ValueError per defect, naming the tensor; a matching set passes."""
    dev = torch.device("cpu")
    args = _samples(_TINY_GEOM, device=dev)
    blk = MqaSparseAttentionBlockFwd(*args, _TINY_GEOM)
    good = dict(zip(TENSOR_NAMES, args))
    _check_bound_tensors(blk._descs, good, dev)
    g = _TINY_GEOM

    def bad(name, value, match):
        t = dict(good)
        t[name] = value
        with pytest.raises(ValueError, match=match):
            _check_bound_tensors(blk._descs, t, dev)

    bad("w_kv", torch.empty(g.head_dim, g.d_model // 2, dtype=_BF16), "w_kv must have the declared shape")
    bad("x", torch.empty(1, g.d_model, 32, dtype=_BF16).transpose(1, 2), "x must be contiguous")
    bad("w_q_norm", torch.empty(g.q_lora_rank, dtype=torch.float32), "w_q_norm must have the declared dtype")
    bad("cos", torch.empty(1, 32, g.rope_dim // 2, dtype=torch.float32, device="meta"), "cos must live on the declared device")
    bad("topk_idxs", good["topk_idxs"].long(), "topk_idxs must have the declared dtype")
    assert len(TENSOR_NAMES) == 14 and list(TENSOR_NAMES) == list(_IDX)


def test_runner_name_table_is_the_pipeline():
    """The names / kinds ``stage_runners`` emits are the perf driver's stage vocabulary: the ten fixed stages in pipeline
    order, plus the two adapter pre-passes (``union_lists`` for d512, ``idx_staging`` for dsa) as ``prep`` rows between
    ``kv_chain`` and ``sparse_attention``."""
    names = [n for n, _ in RUNNER_NAMES]
    fixed = ["q_a_proj", "q_norm", "q_b_proj", "q_rope", "kv_proj", "kv_chain", "sparse_attention", "o_unrope", "o_a_proj", "o_b_proj"]
    assert [n for n in names if n in fixed] == fixed, names
    preps = [n for n in names if n not in fixed]
    assert set(preps) == {"union_lists", "idx_staging"}, preps
    assert all(dict(RUNNER_NAMES)[n] == "prep" for n in preps)
    assert all(names.index("kv_chain") < names.index(n) < names.index("sparse_attention") for n in preps), names
    assert set(k for _, k in RUNNER_NAMES) == {"mma", "bw", "prep"}
    assert dict(RUNNER_NAMES)["sparse_attention"] == "mma"
    assert hasattr(MqaSparseAttentionBlockFwd, "stage_runners")


def test_package_exports_the_api_lazily():
    assert "MqaSparseAttentionBlockFwd" in M.__all__ and "MqaSparseAttentionBlockFwd" in dir(M)
    assert M.MqaSparseAttentionBlockFwd is MqaSparseAttentionBlockFwd
    assert M.ATTENTION_IMPLS == ("torch", "dsa", "d512") and M.POINTWISE_IMPLS == ("torch", "frost")
    with pytest.raises(AttributeError):
        M.does_not_exist
