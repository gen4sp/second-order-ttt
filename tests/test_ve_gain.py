"""E1-VE-D4 read-out gain split on a tiny random model (CPU, no real checkpoints)."""
import math
import os

import pytest
import torch

from tttmem import exp_ttt as t, exp_ve as ve, exp_ve_gain as vg

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-vd4-90-tests"))

TINY = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=8, chunk=32,
            T=128, n_loc=10, n_val=8, n_fill=4, per_type=2, n_scratch_loc=2, n_scratch=4,
            q_space_max=1)


def _setup(rule="ce", edge=False, seed=3):
    vst = ve.VeSettings(**TINY, n_eval=8, eval_batch=4, device="cpu")
    lay = vst.layout()
    cell = dict(order="fo", rule=rule, eta=10.0, slip=0.0, seed=seed, edge=edge)
    model = ve.build(vst, lay, seed, torch.device("cpu"))
    with torch.no_grad():  # move the fast head and ln_f off their init so the factors vary
        for p in list(model.head.parameters()) + list(model.ln_f.parameters()):
            p.add_(0.3 * torch.randn_like(p))
    model.eval()
    data = ve.fixed_set(lay, 8, ve.EVAL_SEED, torch.device("cpu"), 4)
    return vst, lay, cell, model, data


def test_factors_multiply_to_the_autograd_gain():
    vst, lay, cell, model, data = _setup()
    o = torch.randn(5, 7, model.ln_f.normalized_shape[0], requires_grad=True)
    y = torch.randint(0, model.n_vocab, (5, 7))
    z = model.read(o)
    loss = torch.nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), y.reshape(-1),
                                             reduction="sum")
    g, delta = torch.autograd.grad(loss, [z, o])
    f = vg.gain_factors(model, o.detach(), g)
    assert torch.allclose(f["delta"], delta, atol=1e-6)
    prod = f["fU"] * f["fgamma"] * f["fP"] * f["fsigma"]
    assert torch.allclose(prod, f["gain"], rtol=1e-5)
    assert torch.allclose(f["gain"], delta.norm(dim=-1) / g.norm(dim=-1), rtol=1e-5)
    assert bool((f["fP"] <= 1 + 1e-5).all())


@pytest.mark.parametrize("rule,edge", [("ce", False), ("ce", True), ("linev", False),
                                       ("aice", False)])
def test_stream_reconstructs_delta_and_matches_metrics(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    for tok, meta in data:
        out = vg.gain_stream(model, lay, vst, cell, tok, meta)
        row, _ = ve.batch_metrics(model, lay, vst, cell, tok, meta)
        s = dict(zip(ve.METRIC_NAMES, row.tolist()))
        for k in ve.TYPES:
            assert out["q"][f"hit_{k}"] == s[f"hit_{k}"]
        assert out["delta_recon_maxdiff"] < 1e-5
        for k in ("C_A.gain", "C_X.fsigma", "E_U.fP", "C_N.sigma_h", "E_A.p_maxval"):
            assert k in out


def test_summary_geometric_means_multiply():
    vst, lay, cell, model, data = _setup("ce", True)
    rec = vg.summarize([vg.gain_stream(model, lay, vst, cell, tok, meta) for tok, meta in data])
    for t_ in ("C_A", "C_X", "E_U"):
        prod = math.prod(rec[f"{t_}.{k}"] for k in ("fU", "fgamma", "fP", "fsigma"))
        assert prod == pytest.approx(rec[f"{t_}.gain"], rel=1e-5)
    share = vg.ln_share(rec, "C_X", "C_A")
    total = math.log(rec["C_X.gain"] / rec["C_A.gain"])
    u = math.log(rec["C_X.fU"] / rec["C_A.fU"])
    assert share == pytest.approx((total - u) / total, rel=1e-5)


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
    cfg = dict(process_name="TTM-vd4-91-test", device="cpu", threads=2, ckpt_dir=str(tmp_path),
               n_eval=4, eval_batch=2, ve=TINY, cells=cells)
    res = vg.run(cfg, 0)
    assert res["cells"] == 2
    assert res["scores"]["L1_eta10"]["verdict"] == "NE"
    with pytest.raises(ValueError):
        vg.settings(dict(cfg, bogus=1))
