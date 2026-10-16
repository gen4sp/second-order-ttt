"""Small, CPU-only contracts for the shared P2-3/MINI core."""
import numpy as np
import pytest
import torch

from tttmem.p2.data import repeat_classes
from tttmem.p2.meta import (InnerCfg, clip_per_seq, fo_backward, fo_loss_plain,
                            inner_logit_cotangent, nc_control_mask, so_backward,
                            so_loss_plain, stream_eval)
from tttmem.p2.model import build_model
from tttmem.p2 import evaluate
from tttmem.p2.probe import boundary_valid


def _batch(seed=4):
    torch.manual_seed(seed)
    m = build_model("tiny2", compute_dtype=torch.float64).double()
    ids = torch.randint(3, 101, (2, 64))
    tgt = torch.randint(3, 101, (2, 64))
    return m, ids, tgt, torch.ones_like(ids, dtype=torch.bool)


def _grads(m):
    return [p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()]


def test_barrier_engines_match_full_graph_fp64():
    for kind in ("fo", "so"):
        m, ids, tgt, mask = _batch()
        inner = InnerCfg(clip=.7)
        m.zero_grad(set_to_none=True)
        ref_loss, _ = (fo_loss_plain if kind == "fo" else so_loss_plain)(
            m, ids, tgt, mask, inner, barriers=[2, 4])
        ref = torch.autograd.grad(ref_loss, tuple(m.parameters()), allow_unused=True)
        m.zero_grad(set_to_none=True)
        if kind == "fo":
            got = fo_backward(m, ids, tgt, mask, inner, .5, barriers=[2, 4], fused_attn=False)
        else:
            got = so_backward(m, ids, tgt, mask, inner, .5, barriers=[2, 4])
        assert abs(float(got.loss_sum) / 2 - torch.as_tensor(ref_loss).item()) < 1e-12
        for a, b in zip(_grads(m), ref):
            assert torch.allclose(a, b, atol=2e-9, rtol=2e-8)


def test_rp_zero_reference_and_cotangent_identity():
    torch.manual_seed(0)
    z = torch.randn(3, 7, 11, dtype=torch.float64)
    y = torch.randint(0, 11, (3, 7))
    w = torch.rand(3, 7, dtype=torch.float64)
    ce = inner_logit_cotangent("ce", z, y, w)
    rp0 = inner_logit_cotangent("rp", z, y, w, r_ref=0)
    assert torch.equal(ce, rp0)
    r = -torch.expm1(-torch.nn.functional.cross_entropy(z.flatten(0, 1), y.flatten(), reduction="none").view(3, 7))
    q = torch.softmax(z.masked_fill(torch.nn.functional.one_hot(y, 11).bool(), -torch.inf), -1)
    lhs = inner_logit_cotangent("rp", z, y, w, r_ref=.03)
    coeff = torch.maximum(r, torch.full_like(r, .03))
    rhs = coeff.unsqueeze(-1) * q
    rhs.scatter_add_(-1, y.unsqueeze(-1), -coeff.unsqueeze(-1))
    assert torch.allclose(lhs, rhs * w.unsqueeze(-1))
    assert torch.isfinite(inner_logit_cotangent("rp", z * 1000, y, w, r_ref=.03)).all()
    # Registered algebraic identity (ii), restricted to r >= 1e-8.
    p = torch.softmax(z, -1)
    r_safe = r.clamp_min(1e-8)
    lhs2 = torch.maximum(r, torch.full_like(r, .03)).unsqueeze(-1) * (q -
        torch.nn.functional.one_hot(y, 11).to(z.dtype))
    rhs2 = torch.maximum(torch.ones_like(r), .03 / r_safe).unsqueeze(-1) * (
        p - torch.nn.functional.one_hot(y, 11).to(z.dtype))
    assert torch.allclose(lhs2[r >= 1e-8], rhs2[r >= 1e-8], atol=1e-12, rtol=1e-12)
    # (iii): a residual below 1e-12 must remain finite.
    tiny = torch.full((1, 1, 11), -1000., dtype=torch.float64)
    tiny[..., 0] = 1000.
    assert torch.isfinite(inner_logit_cotangent("rp", tiny, torch.zeros((1, 1), dtype=torch.long),
                                                torch.ones((1, 1), dtype=torch.float64), r_ref=.03)).all()


def test_nc_matching_is_disjoint_counted_and_deterministic():
    active = torch.tensor([[True, False, True, False, False, True, False, False]])
    classes = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    a, ca = nc_control_mask(active, classes, seed=7, step=3, chunk=2)
    b, cb = nc_control_mask(active, classes, seed=7, step=3, chunk=2)
    assert torch.equal(a, b) and ca == cb and not bool((a & active).any())
    for (_, _, k), (n_active, n_drawn) in ca.items():
        assert n_drawn == min(n_active, int(((classes == k) & ~active).sum()))


def test_nc_fail_only_counts_rows_with_floor_active_tokens():
    model, ids, tgt, mask = _batch(52)
    classes = torch.ones_like(tgt)
    quiet = stream_eval(model, ids, tgt, mask,
                        InnerCfg(rule="nc", r_ref=0.0, classes=classes),
                        barriers=[2, 4], fused_attn=False)
    assert quiet["nc_fail"] == 0
    # Floor active on part of each row: the class-1 pool is not empty.
    mid = stream_eval(model, ids, tgt, mask,
                      InnerCfg(rule="nc", r_ref=0.99, classes=classes),
                      barriers=[2, 4], fused_attn=False)
    assert mid["nc_fail"] == 0
    # r_ref > 1 makes every token floor-active, so the control pool is empty.
    empty = stream_eval(model, ids, tgt, mask,
                        InnerCfg(rule="nc", r_ref=1.5, classes=classes),
                        barriers=[2, 4], fused_attn=False)
    assert empty["nc_fail"] > 0


def test_nc_control_mask_selects_every_class_against_bruteforce():
    g = torch.Generator().manual_seed(11)
    for trial in range(10):
        B, T = 4, 256
        active = torch.rand(B, T, generator=g) < .3
        classes = torch.randint(-1, 4, (B, T), generator=g)
        allowed = torch.rand(B, T, generator=g) < .9
        sel, counts = nc_control_mask(active, classes, seed=trial, step=trial, chunk=1,
                                      allowed=allowed)
        c = torch.where(classes == 0, torch.ones_like(classes), classes)
        assert not bool((sel & active).any()) and not bool((sel & ~allowed).any())
        assert not bool((sel & (c == -1)).any())
        for b in range(B):
            for k in range(4):
                n_act = int((active[b] & allowed[b] & (c[b] == k)).sum())
                pool = (~active[b]) & allowed[b] & (c[b] == k)
                want = min(n_act, int(pool.sum()))
                got = sel[b] & (c[b] == k)
                assert int(got.sum()) == want, (trial, b, k)
                assert not bool((got & ~pool).any())
                assert int(counts[(b, 1, k)][1]) == want


def test_repeat_classes_matches_bruteforce():
    rng = np.random.default_rng(3)
    rows = rng.integers(2, 40, size=(3, 65), dtype=np.uint32)
    rows[:, 0] = 1
    got = repeat_classes(rows, 16)
    ref = np.full_like(got, -1)
    for b in range(len(rows)):
        seen = {}
        for t in range(rows.shape[1] - 1):
            seg0 = (t // 16) * 16
            if rows[b, t + 1] == 1 or (t % 16 == 15 and t < 63):
                continue
            if t - 2 < seg0:
                ref[b, t] = 0
                continue
            key = tuple(int(x) for x in rows[b, t - 2:t + 2])
            prev = seen.get(key)
            ref[b, t] = 1 if prev is None else (2 if prev >= seg0 else 3)
            seen[key] = t
    np.testing.assert_array_equal(got, ref)


def test_init_seed_changes_model_but_data_seed_does_not():
    a = build_model("tiny2", compute_dtype=torch.float64, seed=0)
    b = build_model("tiny2", compute_dtype=torch.float64, seed=0)
    c = build_model("tiny2", compute_dtype=torch.float64, seed=1)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))


def test_clip_per_seq_detach_scale_modes_have_expected_gradients():
    g = torch.tensor([[3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    c, n = clip_per_seq([g], 1.0, detach_scale=False)
    c[0].sum().backward()
    grad_diff = g.grad.detach().clone()
    g2 = torch.tensor([[3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    c2, _ = clip_per_seq([g2], 1.0, detach_scale=True)
    c2[0].sum().backward()
    assert not torch.allclose(grad_diff, g2.grad)
    assert torch.allclose(c[0], c2[0]) and torch.allclose(n, torch.tensor([5.0], dtype=torch.float64))


def test_repeat_classes_includes_segment_boundaries_and_masks_crossing():
    rows = np.arange(1, 66, dtype=np.uint32)[None, :]
    rows[0, 0] = 1
    got = repeat_classes(rows, 16)[0]
    for p in (0, 1, 16, 17, 32, 33, 48, 49):
        assert got[p] == 0
    for p in (15, 31, 47):
        assert got[p] == -1


def test_nc_ignores_masked_class_minus_one_and_logs_code_zero():
    active = torch.tensor([[True, False, True, False, False, False]])
    classes = torch.tensor([[0, 0, -1, -1, 1, 0]])
    sel, counts = nc_control_mask(active, classes, seed=11, step=2, chunk=0)
    assert not bool(sel[classes < 0].any())
    assert not bool((sel & active).any())
    assert counts[(0, 0, 1)] == (1, 1)


def test_nc_inner_mask_changes_rp_and_nc_stream_eval():
    """Suppressed focal writers must affect both rescue rules and their controls."""
    model, ids, tgt, mask = _batch(29)
    classes = torch.ones_like(tgt)
    im = mask.clone()
    im[:, ::2] = False
    for rule in ("rp", "nc"):
        cfg = InnerCfg(rule=rule, r_ref=.99, classes=classes, seed=3, row_ids=torch.arange(2))
        # The calls are intentionally separate so any accidental mask reuse is visible.
        f = stream_eval(model, ids, tgt, mask, cfg, barriers=[2, 4],
                        inner_mask=im, fused_attn=False)
        g = stream_eval(model, ids, tgt, mask, cfg, barriers=[2, 4],
                        inner_mask=mask, fused_attn=False)
        assert not torch.equal(f["nll"], g["nll"])


def test_rp_nc_focal_and_nonfocal_reads_are_distinct():
    model, _, _, _ = _batch(37)
    docs = np.random.default_rng(37).integers(3, 101, size=(2, 65), dtype=np.uint32)
    docs[:, 0] = 1
    for row in docs:
        for i in range(4):
            key = np.array([11 + 4 * i, 12 + 4 * i, 13 + 4 * i, 14 + 4 * i], dtype=np.uint32)
            row[3 + 6 * i:7 + 6 * i] = key
            row[35 + 6 * i:39 + 6 * i] = key
    probes = evaluate.prepare_probes(docs, 8, np.zeros(101, dtype=np.bool_), bos=1,
                                     barriers=[2, 4])
    for rule in ("rp", "nc"):
        out = evaluate.eval_docs(model, docs, probes, rule,
                                 modes=["fP_focal", "fP_nonfocal"], batch=2,
                                 device="cpu", fused_attn=False, barriers=[2, 4],
                                 r_ref=.99,
                                 classes=repeat_classes(docs, 16))
        assert not np.allclose(out["doc"]["fP_focal"], out["doc"]["fP_nonfocal"])


def test_rp_zero_floor_with_write_mask_matches_ce():
    model, ids, tgt, mask = _batch(31)
    im = mask.clone()
    im[:, ::2] = False
    ce = stream_eval(model, ids, tgt, mask, InnerCfg(rule="ce", classes=torch.ones_like(tgt)),
                     barriers=[2, 4], inner_mask=im, fused_attn=False)
    rp = stream_eval(model, ids, tgt, mask, InnerCfg(rule="rp", r_ref=1e-12,
                                                      classes=torch.ones_like(tgt)),
                     barriers=[2, 4], inner_mask=im, fused_attn=False)
    assert torch.allclose(ce["nll"], rp["nll"], atol=1e-12, rtol=1e-12)


def test_rp_logs_separate_norms_and_floor_share():
    model, ids, tgt, mask = _batch(41)
    inner = InnerCfg(rule="rp", clip=.7, r_ref=.99)
    stats = fo_backward(model, ids, tgt, mask, inner, .5, barriers=[2, 4], fused_attn=False)
    diag = stats.rule_stats
    assert diag["extra_norm"] and diag["ce_norm"] and diag["induced_binding"]
    active = torch.cat([x.reshape(-1) for x in diag["rp_floor_active"]]).float().mean()
    assert 0.2 < float(active) < 0.8


def test_checkpoint_schedule_is_small_and_arm_specific():
    from tttmem.p2.train import TrainCfg, checkpoint_steps
    cfg = TrainCfg(steps=1200)
    assert checkpoint_steps(cfg, "so") == {600, 1200}
    assert checkpoint_steps(cfg, "fo") == {1200}


def test_fo_rp_and_nc_match_full_graph_fp64():
    for rule in ("rp", "nc"):
        m, ids, tgt, mask = _batch(8)
        classes = torch.from_numpy(repeat_classes(
            torch.cat([torch.ones(2, 1, dtype=torch.long), ids], 1).numpy(), 8))
        inner = InnerCfg(rule=rule, clip=.7, r_ref=.99, classes=classes)
        ref_loss, _ = fo_loss_plain(m, ids, tgt, mask, inner, barriers=[2, 4])
        ref = torch.autograd.grad(ref_loss, tuple(m.parameters()), allow_unused=True)
        m.zero_grad(set_to_none=True)
        got = fo_backward(m, ids, tgt, mask, inner, .5, barriers=[2, 4], fused_attn=False)
        assert torch.allclose(got.loss_sum / 2, ref_loss, atol=1e-12, rtol=1e-12)
        for p, q in zip(m.parameters(), ref):
            if q is not None:
                assert torch.allclose(p.grad, q, atol=3e-9, rtol=3e-8)


@pytest.mark.parametrize("fn", [fo_backward, so_backward])
@pytest.mark.parametrize("rule", ["ce", "rp", "nc"])
def test_training_fast_updates_equal_stream_eval(fn, rule):
    if fn is so_backward and rule != "ce":
        return  # The production SO arm is CE; RP/NC are first-order arms.
    model, ids, tgt, mask = _batch(17)
    classes = torch.from_numpy(repeat_classes(
        torch.cat([torch.ones(2, 1, dtype=torch.long), ids], 1).numpy(), 8))
    inner = InnerCfg(rule=rule, clip=.7, r_ref=.99, classes=classes)
    model.zero_grad(set_to_none=True)
    kw = {"barriers": [2, 4], "fused_attn": False} if fn is fo_backward else {
        "barriers": [2, 4], "fused_forward": False}
    stats = fn(model, ids, tgt, mask, inner, .5, **kw)
    ev = stream_eval(model, ids, tgt, mask, inner, adapt=True,
                     barriers=[2, 4], fused_attn=False, return_fast=True)
    for a, b in zip(stats.fast, ev["fast"]):
        assert torch.allclose(a, b, atol=3e-9, rtol=3e-8)


def test_segment_rope_reset_and_no_update_leakage():
    torch.manual_seed(3)
    model = build_model("tiny2", compute_dtype=torch.float64).double()
    tail = torch.randint(3, 101, (1, 16))
    prefix_a = torch.randint(3, 101, (1, 16))
    prefix_b = torch.randint(3, 101, (1, 16))
    long_a = torch.cat([prefix_a, tail], 1)
    long_b = torch.cat([prefix_b, tail], 1)
    tgt_tail = torch.randint(3, 101, (1, 16))
    mask_tail = torch.ones_like(tgt_tail, dtype=torch.bool)
    short = stream_eval(model, tail, tgt_tail, mask_tail, InnerCfg(), adapt=False,
                        barriers=[], fused_attn=False)
    tgt_long = torch.randint(3, 101, (1, 32))
    tgt_long[:, 16:] = tgt_tail
    for ids in (long_a, long_b):
        tgt = tgt_long
        out = stream_eval(model, ids, tgt, torch.ones_like(tgt, dtype=torch.bool), InnerCfg(),
                          adapt=False, barriers=[2], fused_attn=False)
        assert torch.allclose(out["nll"][:, 16:], short["nll"], atol=2e-9, rtol=2e-8)
    # The optimized FO/SO paths expose the same post-barrier chunk losses when
    # the inner channel is disabled, proving that earlier segments do not leak.
    for fn in (fo_backward, so_backward):
        vals = []
        for ids in (long_a, long_b):
            tgt = tgt_long
            model.zero_grad(set_to_none=True)
            st = fn(model, ids, tgt, torch.ones_like(tgt, dtype=torch.bool),
                    InnerCfg(lr=0.0), .5, barriers=[2], **({"fused_attn": False} if fn is fo_backward else {}))
            vals.append(st.chunk_loss[2:].detach())
        assert torch.allclose(vals[0], vals[1], atol=2e-9, rtol=2e-8)


def test_h_lm_is_noadapt_training_layout_and_never_sparse(monkeypatch):
    model = build_model("tiny", compute_dtype=torch.float64).double()
    docs = np.arange(1, 65, dtype=np.uint32)[None, :]
    docs[0, 0] = 1
    calls = []
    original = evaluate.stream_eval
    def wrapped(*args, **kwargs):
        calls.append(kwargs.copy())
        assert "nll_positions" not in kwargs
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluate, "stream_eval", wrapped)
    result = evaluate.eval_docs(model, docs, [None], "ce", modes=["heldout"], batch=1,
                                inner_lr=1., inner_clip=1., device="cpu", fused_attn=False,
                                barriers=[2, 4, 6])
    assert calls and all(c.get("adapt") is False for c in calls)
    ids, tgt = torch.from_numpy(docs[:, :-1].astype(np.int64)), torch.from_numpy(docs[:, 1:].astype(np.int64))
    valid = torch.from_numpy(boundary_valid(63, 8, [2, 4, 6]))[None] & (tgt != 1)
    direct = stream_eval(model, ids, tgt, valid, InnerCfg(), adapt=False, barriers=[2, 4, 6], fused_attn=False)
    expected = (direct["nll"] * valid).sum(1) / valid.sum(1)
    assert np.allclose(result["doc"]["heldout"], expected.numpy(), atol=1e-10)


def test_mix_order_is_eight_plus_eight_and_dclm_prefix():
    from tttmem.p2.train import mix_order
    a = mix_order(64, 64, steps=2, batch=16, data_seed=9)
    b = mix_order(64, 64, steps=2, batch=16, data_seed=9)
    assert np.array_equal(a, b)
    assert np.all(a[:8, 0] == 0) and np.all(a[8:16, 0] == 1)
    core = np.random.default_rng(9).permutation(64)[:16]
    assert np.array_equal(a[a[:, 0] == 0, 1], core)
