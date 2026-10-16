"""E1-VE-D2 write budget and beta sweep on a tiny random model (CPU, no real checkpoints)."""
import math
import os

import pytest
import torch

from tttmem import exp_ttt as t, exp_ve as ve, exp_ve_budget as vb, exp_ve_diag as dg

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-vd2-90-tests"))

TINY = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=8, chunk=32,
            T=128, n_loc=10, n_val=8, n_fill=4, per_type=2, n_scratch_loc=2, n_scratch=4,
            q_space_max=1)


def _setup(rule="ce", edge=False, seed=3):
    vst = ve.VeSettings(**TINY, n_eval=8, eval_batch=4, device="cpu")
    lay = vst.layout()
    cell = dict(order="fo", rule=rule, eta=10.0, slip=0.0, seed=seed, edge=edge)
    model = ve.build(vst, lay, seed, torch.device("cpu"))
    with torch.no_grad():  # move the fast head off its tiny init so writes matter
        for p in model.head.parameters():
            p.add_(0.3 * torch.randn_like(p))
    model.eval()
    data = ve.fixed_set(lay, 8, ve.EVAL_SEED, torch.device("cpu"), 4)
    return vst, lay, cell, model, data


@pytest.mark.parametrize("rule,edge", [("ce", False), ("ce", True), ("linev", False),
                                       ("aice", False)])
def test_masked_updates_partition_the_stream_update(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    tok, meta = data[0]
    idx, y, sel, h, etas = vb._setup(model, lay, cell, tok, meta)
    rr = ve.RULES[rule]
    seen = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        seen[c] = ([w.detach() for w in W], [(-e * g).detach() for e, g in zip(etas, grads)],
                   norm.detach())

    t.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, t.GradStats(), sel, h=h,
                      rule=rr, observer=observer)
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    ref = t.reference_logits(model, h, idx, seen[0][0]) if rr == "aice" else None
    m_e, m_c = vb.own_masks(lay, meta, idx, ve.A)
    for c, m in ((0, m_e), (dg.c_chunk(lay), m_c)):
        W, dW, norm = seen[c]
        sc = vb.clip_scale(norm, vst.g_clip)
        upd = lambda mask: vb.masked_update(model, rr, h, x, y, sel, ref, W, etas, sc, c,
                                            lay.chunk, mask)
        full = upd(torch.ones_like(m))
        own, rest = upd(m), upd(~m)
        for f, d, a, b in zip(full, dW, own, rest):
            assert torch.allclose(f, d, atol=1e-6, rtol=1e-5)
            assert torch.allclose(a + b, d, atol=1e-6, rtol=1e-5)
        assert float(own[2].abs().sum()) > 0


def test_own_masks_mark_the_e_and_c_writes_of_the_same_locations():
    vst, lay, cell, model, data = _setup()
    tok, meta = data[0]
    idx = tok[:, :-1]
    for ti in (ve.A, ve.X):
        m_e, m_c = vb.own_masks(lay, meta, idx, ti)
        n = int((meta["c_type"] == ti).sum())
        assert int(m_e.sum()) == n and int(m_c.sum()) == n
        lo = dg.c_chunk(lay) * lay.chunk
        for b in range(idx.shape[0]):
            e_locs = sorted(idx[b, :lay.chunk][m_e[b]].tolist())
            c_locs = sorted(idx[b, lo:lo + lay.chunk][m_c[b]].tolist())
            assert e_locs == c_locs


@pytest.mark.parametrize("rule,edge", [("ce", True), ("linev", False)])
def test_budget_stream_probe_matches_the_stream(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    for tok, meta in data:
        out = vb.budget_stream(model, lay, vst, cell, tok, meta)
        row, _ = ve.batch_metrics(model, lay, vst, cell, tok, meta)
        s = dict(zip(ve.METRIC_NAMES, row.tolist()))
        for k in ve.TYPES:
            assert out["q"][f"hit_{k}"] == s[f"hit_{k}"]
        assert out["probe_maxdiff"] < 1e-5
        assert set(out["traj"]) >= {"share_E_A", "share_C_A", "share_Eown_A", "share_Cown_A",
                                    "share_Eown_X", "share_Cown_X", "margin_A", "acc_A"}


@pytest.mark.parametrize("cond", ["erase_v0", "add_v1", "erase_rand"])
def test_sweep_stream_reproduces_the_diag_stream(cond):
    vst, lay, cell, model, data = _setup("ce", True)
    tok, meta = data[0]
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    a = vb.sweep_stream(model, lay, vst, cell, tok, meta, cond, 0.5, g1)
    b = dg.stream(model, lay, vst, cell, tok, meta, cond, 0.5, g2)
    assert a["q"] == b["q"]
    assert a["margin"] == pytest.approx(b["margin"], abs=1e-6)
    assert all(0 < s <= 1 for s in a["scale_C"])


def test_beta90_and_spearman():
    rec = {"base.acc_A": 0.2, "add_v1@0.5.acc_A": 0.5, "add_v1@1.acc_A": 0.95}
    assert vb.beta90(rec, [0.5, 1.0]) == pytest.approx(0.5 + 0.5 * 0.4 / 0.45)
    rec2 = {"base.acc_A": 0.2, "add_v1@0.5.acc_A": 0.3, "add_v1@1.acc_A": 0.4}
    assert vb.beta90(rec2, [0.5, 1.0]) == math.inf
    assert vb.spearman([1, 2, 3, 4], [1, 2, 3, math.inf]) == pytest.approx(1.0)
    assert vb.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert vb.spearman([1, 2, 3, 4], [1, 1, math.inf, math.inf]) == pytest.approx(0.894, abs=1e-3)


def test_run_end_to_end_from_saved_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [dict(order="fo", rule="ce", eta=10.0, slip=0.0, seed=s, edge=True) for s in (1, 2)]
    vst = ve.VeSettings(**TINY, device="cpu")
    lay = vst.layout()
    for c in cells:
        m = ve.build(vst, lay, c["seed"], torch.device("cpu"))
        name = ve.cell_id(c).replace("/", "__")
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()},
                   tmp_path / f"ve__{name}.pt")
    cfg = dict(process_name="TTM-vd2-91-test", device="cpu", threads=2, ckpt_dir=str(tmp_path),
               n_eval=4, eval_batch=2, ve=TINY, cells=cells, betas=[0.5, 2.0])
    res = vb.run(cfg, 0)
    assert res["cells"] == 2
    assert res["scores"]["S2_eta10"]["verdict"] == "NE"   # k = 2 < 4
    with pytest.raises(ValueError):
        vb.settings(dict(cfg, bogus=1))
    with pytest.raises(ValueError):
        vb.run(dict(cfg, betas=[1.0]), 0)
