"""E1-VE task, metrics and cell loop on CPU (no CUDA/MPS, no spawned workers)."""
import os

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t, exp_ve as ve

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-ve-90-tests"))

LAY = ve.Layout()
TINY = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=8, chunk=32,
            T=128, n_loc=10, n_val=8, n_fill=4, per_type=2, n_scratch_loc=2, n_scratch=4,
            q_space_max=1, steps=3, batch=2, n_quick=2, n_eval=2, eval_batch=2, quick_every=1,
            warmup=1, lr_total=10, device="cpu", save_final=False)


def test_default_layout_fits_and_isolates():
    LAY.check()
    assert (LAY.V, LAY.c_start, LAY.s_start, LAY.r_start, LAY.receptive) == (59, 64, 128, 380, 253)
    s = ve.layout_sanity(LAY, 128)
    assert s["passed"], s


def test_tiny_layout_sanity():
    lay = ve.settings(TINY).layout()
    assert ve.layout_sanity(lay, 64)["passed"]


def test_layout_rejects_short_T_and_bad_type_count():
    with pytest.raises(ValueError):
        ve.Layout(T=448).check()
    with pytest.raises(ValueError):
        ve.Layout(n_loc=16).check()


def test_slip_changes_only_put_arguments_and_keeps_clean_stream():
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    clean, m0 = ve.ve_batch(LAY, 64, g1)
    noisy, m1 = ve.ve_batch(LAY, 64, g2, slip=0.999, slip_gen=torch.Generator().manual_seed(3))
    diff = clean != noisy
    args = torch.zeros_like(diff)
    args.scatter_(1, m0["c_write"] - 1, True)
    args.scatter_(1, m0["s_write"] - 1, True)
    assert not (diff & ~args).any()
    puts = (m0["c_type"] == ve.A) | (m0["c_type"] == ve.N)
    assert m1["c_slipped"][puts].float().mean() > .99
    assert not m1["c_slipped"][~puts].any()
    assert m1["s_slipped"].float().mean() > .99
    for k in ("query_pos", "query_tok", "query_type", "query_old", "c_type", "c_old"):
        assert torch.equal(m0[k], m1[k])
    for w, flag in (("c_write", "c_slipped"), ("s_write", "s_slipped")):
        arg = noisy.gather(1, m0[w] - 1)
        obs = noisy.gather(1, m0[w] + 1)
        assert not (arg[m1[flag]] == obs[m1[flag]]).any()


def test_types_targets_and_v1_rules():
    tok, m = ve.ve_batch(LAY, 256, torch.Generator().manual_seed(11))
    qt, old = m["query_type"], m["query_old"]
    assert ((qt == ve.A) | (qt == ve.X)).eq(old >= 0).all()
    assert (old[old >= 0] != m["query_tok"][old >= 0]).all()
    loc = tok.gather(1, m["query_pos"])
    assert ((loc >= LAY.loc0) & (loc < LAY.sloc0)).all()
    assert torch.equal(tok.gather(1, m["query_pos"] + 1), m["query_tok"])
    assert (loc.sort(1).values == torch.arange(LAY.loc0, LAY.sloc0)).all()
    ct = m["c_type"]
    put = (ct == ve.A) | (ct == ve.N)
    marker = tok.gather(1, m["c_write"] - 2)
    assert ((marker == ve.PUT) == put).all()
    arg, obs = tok.gather(1, m["c_write"] - 1), tok.gather(1, m["c_write"] + 1)
    assert (arg[put] == obs[put]).all()
    assert (arg[~put] >= LAY.fill0).all()
    # S re-observes the old value, A and X show a new one
    assert (obs[ct == ve.S] == m["c_old"][ct == ve.S]).all()
    assert (obs[(ct == ve.A) | (ct == ve.X)] != m["c_old"][(ct == ve.A) | (ct == ve.X)]).all()
    assert (m["c_old"][ct == ve.N] == -1).all()
    wb = tok.gather(1, m["s_write"])
    assert ((wb >= LAY.sloc0) & (wb < LAY.val0)).all()
    assert (tok.gather(1, m["s_write"] - 1) == tok.gather(1, m["s_write"] + 1)).all()


def test_selector_counts():
    _, m = ve.ve_batch(LAY, 8, torch.Generator().manual_seed(0))
    assert (ve.event_selector(LAY, m, 8).sum(1) == 32).all()


def test_finish_on_synthetic_row():
    s = dict.fromkeys(ve.METRIC_NAMES, 0.0)
    for k in ve.TYPES:
        s[f"n_{k}"] = 10.0
    for k in ve.C_TYPES:
        s[f"nc_{k}"] = 10.0
    s.update(hit_U=10, hit_A=4, hit_X=9, stale_A=5, stale_X=1, oldp_X=4.0, n_wb=4, resid0_wb=1)
    out = ve.finish(np.array([s[n] for n in ve.METRIC_NAMES]))
    assert out["acc_A"] == .4 and out["stale_A"] == .5 and out["acc_U"] == 1.0
    assert out["stale_err_A"] == pytest.approx(5 / 6)
    assert out["stale_err_X"] == pytest.approx(1.0)
    assert out["oldp_X"] == pytest.approx(.4) and out["resid0_wb"] == .25


def _cell(order="fo", rule="ce", slip=0.0, edge=False):
    return dict(order=order, rule=rule, eta=10.0, slip=slip, seed=0, edge=edge)


def test_lesion_blocks_exactly_the_copy_edge_at_c_writes():
    """With ``edge``, a C location token's features ignore its argument; elsewhere unchanged."""
    st = ve.settings(TINY)
    lay = st.layout()
    model = ve.build(st, lay, 0, torch.device("cpu")).double()
    tok, meta = ve.ve_batch(lay, 4, torch.Generator().manual_seed(3))
    mask = ve.c_edge_mask(lay, torch.device("cpu"))
    blocked = mask[0, 0].nonzero()
    cw = meta["c_write"][0]
    assert torch.equal(blocked[:, 0], cw) and torch.equal(blocked[:, 1], cw - 1)
    for j in range(lay.n_c):                          # one event at a time: a later C
        other = tok.clone()                           # location may see an earlier argument
        arg = meta["c_write"][:, j:j + 1] - 1
        other.scatter_(1, arg, lay.val0 + (tok.gather(1, arg) - lay.val0 + 1) % lay.n_val)
        with torch.no_grad():
            h, h2 = (model.backbone(x[:, :-1], edge_mask=mask) for x in (tok, other))
            g, g2 = (model.backbone(x[:, :-1]) for x in (tok, other))
        at = lambda z: ve._gather_last(z, meta["c_write"][:, j:j + 1])
        assert torch.equal(at(h), at(h2))             # no path from its argument to the write
        assert not torch.allclose(at(g), at(g2))      # the unlesioned model can copy
    before = slice(0, lay.c_start)
    assert torch.equal(h[:, before], g[:, before])    # E is untouched by the lesion


def test_lesion_cell_trains_and_is_named():
    st = ve.settings(TINY)
    cell = _cell(edge=True)
    rec = ve.train_cell(cell, st, torch.device("cpu"), ve.cell_id(cell))
    assert rec["edge"] and rec["cell_id"] == "fo-ce-edge/eta10/slip0/s0"
    assert rec["steps_done"] == 3 and not rec["diverged"]


@pytest.mark.parametrize("order", ve.ORDERS)
@pytest.mark.parametrize("rule", tuple(ve.RULES))
def test_train_cell_runs_every_arm(order, rule):
    st = ve.settings(TINY)
    rec = ve.train_cell(_cell(order, rule, slip=.1), st, torch.device("cpu"), "tiny")
    assert rec["steps_done"] == 3 and not rec["diverged"] and rec["nonfinite_steps"] == 0
    assert set(rec["final"]) >= {f"acc_{k}" for k in ve.TYPES}
    assert len(rec["quick"]) == 3 and len(rec["quick"][0]) == len(ve.QUICK_COLS)
    assert len(rec["per_seq_acc"]["A"]) == 2


def test_fo_and_so_share_init_and_forward():
    st = ve.settings(TINY)
    lay = st.layout()
    tok, meta = ve.ve_batch(lay, 2, torch.Generator().manual_seed(1))
    losses = []
    for order in ve.ORDERS:
        model = ve.build(st, lay, 0, torch.device("cpu"))
        stats = ve.DeviceStats(lay.T // lay.chunk, torch.device("cpu"))
        losses.append(ve.train_loss(model, lay, st, _cell(order), tok, meta, stats))
    assert torch.allclose(losses[0], losses[1])


def test_aice_equals_lin_in_chunk_zero():
    """AICE = lin while W = W0 (chunk 0); later chunks differ once E has been written."""
    st = ve.settings(TINY)
    lay = st.layout()
    tok, meta = ve.ve_batch(lay, 2, torch.Generator().manual_seed(2))
    model = ve.build(st, lay, 0, torch.device("cpu")).double()
    idx, y = tok[:, :-1], tok[:, 1:]
    h = model.backbone(idx)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(2)]
    sl = slice(0, lay.chunk)
    z = model.read(model.fast_block(h[:, sl], idx[:, sl], W))
    ref = t.reference_logits(model, h, idx, W)[:, sl]
    g_aice = torch.autograd.grad(t.inner_token_loss("aice", z, y[:, sl], reference=ref).sum(), W,
                                 retain_graph=True)
    g_lin = torch.autograd.grad(t.inner_token_loss("lin", z, y[:, sl]).sum(), W)
    for a, b in zip(g_aice, g_lin):
        assert torch.allclose(a, b, atol=1e-10)


def test_nonfinite_steps_are_skipped_on_cpu_path(monkeypatch):
    st = ve.settings({**TINY, "steps": 2})
    real = ve.train_loss
    monkeypatch.setattr(ve, "train_loss", lambda *a: real(*a) * float("nan"))
    lay = st.layout()
    before = {k: v.clone() for k, v in ve.build(st, lay, 0, torch.device("cpu")).state_dict().items()}
    captured = {}
    orig_build = ve.build

    def spy(*a):
        captured["model"] = orig_build(*a)
        return captured["model"]

    monkeypatch.setattr(ve, "build", spy)
    rec = ve.train_cell(_cell(), st, torch.device("cpu"), "nan")
    assert rec["nonfinite_steps"] == 2
    after = captured["model"].state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_run_single_worker_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {**TINY, "save_final": True, "cells": [_cell(), _cell("so", "aice")]}
    out = ve.run(cfg, 0)
    assert out["sanity"]["passed"]
    assert all("error" not in v for v in out["cells"].values())
    assert out["throughput"]["steps_done"] == 2 * TINY["steps"]
    assert out["throughput"]["agg_steps_per_s"] > 0
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2


def test_run_two_spawned_workers_return_every_cell(tmp_path, monkeypatch):
    """The multiprocess path: both cells come back, the pool is reaped, the summary is complete."""
    monkeypatch.chdir(tmp_path)
    cfg = {**TINY, "workers": 2, "cells": [_cell(), _cell("so", edge=True)]}
    out = ve.run(cfg, 0)
    assert all("error" not in v for v in out["cells"].values())
    assert out["throughput"]["complete"] and out["throughput"]["finished"] == 2


def test_settings_reject_unknown_keys_and_bad_cells():
    with pytest.raises(ValueError):
        ve.settings({"nope": 1})
    with pytest.raises(ValueError):
        ve.check_cell(dict(order="fo", rule="lin", eta=10, seed=0))
    with pytest.raises(ValueError):
        ve.check_cell(dict(order="fo", rule="ce", eta=10, seed=0, slip=1.0))
    with pytest.raises(ValueError):
        ve.check_cell(dict(order="fo", rule="ce", eta=10, seed=0, edge=1))
