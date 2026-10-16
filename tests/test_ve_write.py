"""E1-VE-D3 write factors on a tiny random model (CPU, no real checkpoints)."""
import os

import pytest
import torch
import torch.nn.functional as F

from tttmem import exp_ttt as t, exp_ve as ve, exp_ve_budget as vb, exp_ve_diag as dg
from tttmem import exp_ve_write as vw

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-vd3-90-tests"))

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


def _arrive(model, lay, vst, cell, tok, meta):
    idx, y, sel, h, etas = vb._setup(model, lay, cell, tok, meta)
    rr = ve.RULES[cell["rule"]]
    seen = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        seen[c] = [w.detach() for w in W]

    logits = t.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, t.GradStats(), sel,
                               h=h, rule=rr, observer=observer)
    return idx, y, sel, h, rr, seen, logits


@pytest.mark.parametrize("rule,edge", [("ce", False), ("ce", True), ("linev", False),
                                       ("aice", False)])
def test_factors_reproduce_the_stream_and_the_per_position_w2_gradient(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    tok, meta = data[0]
    idx, y, sel, h, rr, seen, logits = _arrive(model, lay, vst, cell, tok, meta)
    xin = model.fast_input(h, idx).detach()
    ref = t.reference_logits(model, h, idx, seen[0]) if rr == "aice" else None
    cc = dg.c_chunk(lay)
    sl = slice(cc * lay.chunk, (cc + 1) * lay.chunk)
    f = vw.chunk_factors(model, rr, h, xin, y, sel, ref, seen[cc], seen[0], sl)
    assert torch.allclose(f["logits"], logits[:, sl], atol=1e-5)
    # one position: ||dL_t/dW2|| = ||delta_t|| * ||a_t|| (outer product)
    b, pos = 1, int(meta["c_write"][1, 0]) - cc * lay.chunk
    W = [w.detach().clone().requires_grad_(True) for w in seen[cc]]
    with torch.enable_grad():
        z = model.read(t.fast_block(h[:, sl], model.head.ln, W, xin[:, sl]))
        inner = t.inner_token_loss(rr, z, y[:, sl], sel=None if sel is None else sel[:, sl],
                                   reference=None if ref is None else ref[:, sl])
        gW2 = torch.autograd.grad(inner[b, pos], W[2])[0][b]
    assert float(gW2.norm()) == pytest.approx(float(f["delta"][b, pos] * f["a"][b, pos]),
                                              rel=1e-4)


def test_ce_logit_gradient_is_p_minus_onehot():
    vst, lay, cell, model, data = _setup("ce", True)
    tok, meta = data[0]
    idx, y, sel, h, rr, seen, logits = _arrive(model, lay, vst, cell, tok, meta)
    xin = model.fast_input(h, idx).detach()
    sl = slice(0, lay.chunk)
    f = vw.chunk_factors(model, rr, h, xin, y, sel, None, seen[0], seen[0], sl)
    ref = (f["p"] - F.one_hot(y[:, sl], f["p"].shape[-1])).norm(dim=-1)
    assert torch.allclose(f["g"], ref, atol=1e-6)
    assert torch.allclose(f["a"], f["a0"])          # chunk 0 arrives with W0


def test_e_types_and_kappa_is_the_rank_one_key_response():
    vst, lay, cell, model, data = _setup("ce", True)
    tok, meta = data[0]
    idx, y, sel, h, rr, seen, logits = _arrive(model, lay, vst, cell, tok, meta)
    et = vw.e_types(meta, idx)
    for ti in (ve.S, ve.A, ve.X):
        assert int((et == ti).sum()) == int((meta["c_type"] == ti).sum())
    assert int((et == ve.U).sum()) == et.numel() - int((meta["c_type"] != ve.N).sum())
    # a single E write's W1 update moves u at x by <x_E, x>/||x_E||^2 of its move at x_E
    xin = model.fast_input(h, idx).detach()
    W = [w.detach().clone().requires_grad_(True) for w in seen[0]]
    b, pe = 0, int(meta["e_write"][0, 3])
    with torch.enable_grad():
        z = model.read(t.fast_block(h[:, :lay.chunk], model.head.ln, W, xin[:, :lay.chunk]))
        inner = t.inner_token_loss(rr, z, y[:, :lay.chunk])
        gW1 = torch.autograd.grad(inner[b, pe], W[0])[0][b]
    xe, xc = xin[b, pe], xin[b, int(meta["c_write"][0, 2])]
    kappa = float(xe @ xc / (xe @ xe))
    assert float((gW1 @ xc).norm()) == pytest.approx(abs(kappa) * float((gW1 @ xe).norm()),
                                                     rel=1e-4)


@pytest.mark.parametrize("rule,edge", [("ce", True), ("linev", False), ("aice", False)])
def test_write_stream_runs_and_matches_metrics(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    for tok, meta in data:
        out = vw.write_stream(model, lay, vst, cell, tok, meta)
        row, _ = ve.batch_metrics(model, lay, vst, cell, tok, meta)
        s = dict(zip(ve.METRIC_NAMES, row.tolist()))
        for k in ve.TYPES:
            assert out["q"][f"hit_{k}"] == s[f"hit_{k}"]
        assert out["logit_maxdiff"] < 1e-5
        for k in ("C_A.size_over_Eown", "C_X.kappa_own", "C_N.G_over_Emean", "E_U.a"):
            assert k in out
        sz, g, gain, a = (out[f"C_A.{k}_over_Eown"] for k in ("size", "g", "gain", "a"))
        for v in zip(sz, g, gain, a):
            assert v[0] == pytest.approx(v[1] * v[2] * v[3], rel=1e-4)


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
    cfg = dict(process_name="TTM-vd3-91-test", device="cpu", threads=2, ckpt_dir=str(tmp_path),
               n_eval=4, eval_batch=2, ve=TINY, cells=cells)
    res = vw.run(cfg, 0)
    assert res["cells"] == 2
    assert res["scores"]["V1_eta10"]["verdict"] == "NE"
    with pytest.raises(ValueError):
        vw.settings(dict(cfg, bogus=1))
