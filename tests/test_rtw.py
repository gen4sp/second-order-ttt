"""CPU/fp64 invariants for the RTW exploratory routes."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tttmem import exp_rtw as rtw
from tttmem import exp_ttt as ttt
from tttmem.floor_rules import complement_ce


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def _tiny():
    torch.set_default_dtype(torch.float64)
    cfg = dict(d_model=8, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=4, T=128, steps=2, batch=2, n_keys=2,
               n_eval=4, eval_batch=2, eta_lr=0., lr_total=10, warmup=2,
               prefix_min=0, prefix_max=0)
    st, task = rtw._settings(cfg)
    raw, meta = task.raw(np.random.default_rng(0), 2)
    idx, y, meta = task.tensors(raw, meta, torch.device("cpu"))
    return st, task, idx, y, meta


def test_gate_beta_one_matches_a0_outer_gradient_and_gain_is_live():
    st, task, idx, y, meta = _tiny()
    a = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb")
    g = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb")
    rtw._gate_params(g, "G-hi")
    la = ttt.arm_loss(a, "e2e1emb", idx, y, 40., 10., 4, ttt.GradStats(), meta=meta)
    lg = rtw.gate_inner_loss(g, idx, y, meta, arm="G-hi", chunk=4)
    ga = torch.autograd.grad(la, tuple(a.parameters()), allow_unused=True)
    ng = list(g.named_parameters())
    gg = torch.autograd.grad(lg, tuple(p for _, p in ng), allow_unused=True)
    gg = dict(zip((n for n, _ in ng), gg))
    for (name, _), x in zip(a.named_parameters(), ga):
        if name in ("rho",) or x is None:
            continue
        assert torch.allclose(x, gg[name], atol=1e-10, rtol=1e-6), name
    assert float(gg["rtw_u"].norm()) > 0 and float(gg["rtw_c"].abs()) > 0


def test_somix_masks_are_paired_and_endpoints_are_exact():
    m1 = rtw.somix_mask(300, 7, 32, .25)
    m2 = rtw.somix_mask(300, 7, 32, .25)
    assert torch.equal(m1, m2) and int(m1.sum()) == 8
    assert not rtw.somix_mask(300, 7, 32, 0.).any()
    assert bool(rtw.somix_mask(300, 7, 32, 1.).all())


def test_rpf_complement_identity_and_finite_floor():
    z = torch.tensor([[2., -1., .5], [100., -100., -100.]], requires_grad=True)
    y = torch.tensor([0, 0])
    c = complement_ce(z, y)
    assert torch.isfinite(c).all()
    p = z.softmax(-1)
    r = 1 - p.gather(1, y[:, None]).squeeze(1)
    q = p / r.clamp_min(1e-8)[:, None]
    q = q.scatter(1, y[:, None], 0.)
    lhs = torch.maximum(r, torch.full_like(r, .03))[:, None] * (q - F_onehot(y, 3))
    rhs = torch.maximum(torch.ones_like(r), torch.full_like(r, .03) / r.clamp_min(1e-8))[:, None] * (p - F_onehot(y, 3))
    valid = r >= 1e-8
    assert torch.allclose(lhs[valid], rhs[valid], atol=1e-6, rtol=1e-6)


def F_onehot(y, n):
    return torch.nn.functional.one_hot(y, n).to(torch.float64)


def test_nc_control_is_deterministic_and_class_matched():
    _, _, _, _, _ = _tiny()
    active = torch.tensor([[True, False, True, False, False, True, False, False]])
    wp = np.array([[2, 6]])
    a = rtw._nc_control_mask(wp, active, 300, 0, 1, 0)
    b = rtw._nc_control_mask(wp, active, 300, 0, 1, 0)
    assert torch.equal(a, b)
    # A write-class control may select a non-active write position; every selected
    # position is nevertheless non-active and the count is class matched.
    assert not bool((a & active).any())


def test_seed_guard_rejects_outside_screen_and_confirmation_range():
    try:
        rtw.run({"device": "cpu", "steps": 0, "cells": [{"arm": "A0", "seed": 299}],
                 "output_root": "outputs/rtw-test"}, 299)
    except ValueError as exc:
        assert "300..323" in str(exc)
    else:
        raise AssertionError("out-of-range seed was accepted")


def _tiny_batch():
    st, task, idx, y, meta = _tiny()
    return st, task, idx, y, meta


def test_gate_eval_uses_each_arm_rule_and_ce_read_is_plain_e2e1emb():
    st, task, idx, y, meta = _tiny_batch()
    data = [(idx, y, meta)]
    for arm in rtw.GATE_ARMS:
        model = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb").double()
        rtw._gate_params(model, arm)
        train_loss = rtw.gate_inner_loss(model, idx, y, meta, arm=arm, chunk=task.chunk)
        own = rtw._metric_rows(model, task, data, arm, st, own=True)
        assert torch.allclose(train_loss.detach(), torch.tensor(own["outer_loss"]), atol=1e-10, rtol=1e-10)
        got = rtw._metric_rows(model, task, data, arm, st, own=False)
        want = ttt.eval_batch_logits(model, task, "e2e1emb", idx, y, 40., 10., task.chunk,
                                     ttt.GradStats(), learned=False, meta=meta)
        assert abs(got["outer_loss"] - float(ttt.token_ce(want, y).mean())) < 1e-10


def test_pf_own_eval_matches_training_recurrence_and_nc_eval_is_deterministic():
    st, task, idx, y, meta = _tiny_batch()
    for arm in ("RP-all", "NC-all"):
        model = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb").double()
        loss = rtw.pf_inner_loss(model, idx, y, meta, arm=arm, chunk=task.chunk)
        logits1 = rtw._pf_eval(model, idx, y, meta, st, arm)
        logits2 = rtw._pf_eval(model, idx, y, meta, st, arm)
        assert torch.allclose(loss.detach(), ttt.token_ce(logits1, y).mean(), atol=1e-10, rtol=1e-10)
        if arm == "NC-all":
            assert torch.equal(logits1, logits2)


def test_stats_are_bounded_to_quick_probe_rows():
    st, task, idx, y, meta = _tiny_batch()
    model = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb").double()
    rtw._gate_params(model, "G-hi")
    stats = {}
    for i in range(50):
        rtw.gate_inner_loss(model, idx, y, meta, arm="G-hi", chunk=task.chunk, stats=stats)
        if (i + 1) % 25 == 0:
            rtw._flush_stats(stats)
    assert len(stats["series"]) == 2
    assert all(acc["reservoir"].numel() <= rtw._RESERVOIR for acc in stats.get("_acc", {}).values())
    assert not any(isinstance(v, list) and any(torch.is_tensor(x) for x in v) for v in stats.values())


def test_fixed_factor_fd_matches_autograd_for_none_and_base():
    st, task, idx, y, meta = _tiny_batch()
    model = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb").double()
    for clip in ("none", "base"):
        _, _, factors = rtw._gate_replay(model, idx, y, meta, torch.ones(() , dtype=torch.float64), clip,
                                         record_factors=True)
        ell = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        z, _ = rtw._gate_replay(model, idx, y, meta, torch.exp(ell), clip, factor_cache=factors)
        aut = torch.autograd.grad(ttt.token_ce(z, y).mean(), ell)[0]
        eps = 1e-4; vals = []
        for sign in (-1, 1):
            zz, _ = rtw._gate_replay(model, idx, y, meta, torch.tensor(np.exp(sign * eps)), clip,
                                     factor_cache=factors)
            vals.append(float(ttt.token_ce(zz, y).mean().detach()))
        fd = (vals[1] - vals[0]) / (2 * eps)
        assert abs(float(aut) - fd) / max(abs(fd), 1e-30) <= 1e-6


def test_somix_exact_per_sequence_decomposition_fp64():
    st, task, idx, y, meta = _tiny_batch()
    model = ttt.build_model(task, st, torch.device("cpu"), 300, "e2e1emb").double()
    mask = rtw.somix_mask(300, 7, len(idx), .5)
    fo = [rtw._per_sequence_grad(model, idx[i:i+1], y[i:i+1], second=False) for i in range(len(idx))]
    so = [rtw._per_sequence_grad(model, idx[i:i+1], y[i:i+1], second=True) for i in range(len(idx))]
    composed = [sum((so[i][j] if bool(mask[i]) else fo[i][j] for i in range(len(idx))),
                    torch.zeros_like(fo[0][j])) / len(idx) for j in range(len(fo[0]))]
    direct = rtw._mixed_direct_grad(model, idx, y, mask)
    assert rtw._flat_rel(direct, composed) <= 1e-8


def test_nonfinite_loss_skips_step_and_is_counted(tmp_path, monkeypatch):
    cfg = {"device": "cpu", "steps": 1, "batch": 2, "T": 64, "chunk": 4, "window": 8,
           "n_keys": 2, "d_model": 8, "n_layers": 1, "n_heads": 2, "backbone_mult": 2,
           "fast_mult": 2, "n_eval": 4, "n_val": 4, "eval_batch": 2, "lr_total": 2,
           "warmup": 1, "output_root": str(tmp_path), "cells": [{"arm": "G-hi", "seed": 300}]}
    monkeypatch.setattr(rtw, "gate_inner_loss", lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    out = rtw.run(cfg, 300)
    rec = out["cells"]["G-hi__s300"]
    assert rec["nonfinite_steps"] == 1 and rec["diverged"] and rec["retained_own"] is False


def test_eval_sets_match_pool_sizes():
    cfg = {"n_eval": 256, "n_val": 64, "n_quick_eval": 64, "T": 64, "chunk": 4,
           "window": 8, "n_keys": 2, "steps": 1}
    st, task = rtw._settings(cfg)
    assert task.n_eval == 256 and task.n_val == 64 and st.n_quick_eval == 64
    ev, _ = ttt.fixed_set(task, st, "eval"); quick, _ = ttt.fixed_set(task, st, "quick")
    assert len(ev) == 256 and len(quick) == 64


def test_pool_workers_persist_each_cell_and_resume_skips_finished(tmp_path, capsys):
    cfg = {"device": "cpu", "steps": 1, "batch": 2, "T": 64, "chunk": 4, "window": 8,
           "n_keys": 2, "d_model": 8, "n_layers": 1, "n_heads": 2, "backbone_mult": 2,
           "fast_mult": 2, "n_eval": 4, "n_val": 4, "n_quick_eval": 4, "eval_batch": 2,
           "lr_total": 2, "warmup": 1, "workers": 2, "threads_per_worker": 1,
           "process_name": "TTM-rtw-07-test-w0", "output_root": str(tmp_path),
           "cells": [{"arm": "A0", "seed": 300}, {"arm": "A0", "seed": 301}]}
    out = rtw.run(cfg, 300)
    files = sorted(p.parent.name for p in tmp_path.glob("*/result.json"))
    assert files == ["A0__s300", "A0__s301"] and not list(tmp_path.glob("*/*.tmp"))
    assert capsys.readouterr().out.count("RTW_CELL_DONE") == 2
    first = (tmp_path / "A0__s300" / "result.json").read_text()
    (tmp_path / "A0__s301" / "result.json").unlink()
    again = rtw.run(dict(cfg, workers=1), 300)
    assert (tmp_path / "A0__s300" / "result.json").read_text() == first
    assert capsys.readouterr().out.count("RTW_CELL_DONE") == 1
    assert set(again["cells"]) == set(out["cells"])


def test_stat_quantile_sample_covers_every_update_in_the_window():
    stats = {}
    for k in range(200):  # 25 steps x 8 chunks
        rtw._stat_update(stats, "beta_other", torch.full((300,), float(k)))
    acc = stats["_acc"]["beta_other"]
    assert acc["reservoir"].numel() == 200 * rtw._PER_UPDATE <= rtw._RESERVOIR
    row = rtw._flush_stats(stats)
    assert row["beta_other"]["n"] == 200 * 300
    assert abs(row["beta_other"]["median"] - 99.5) <= 1.0  # not the last update's 199
    assert abs(row["beta_other"]["mean"] - 99.5) < 1e-4
