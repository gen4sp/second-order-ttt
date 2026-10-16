"""E1-VE-D diagnostics on a tiny random model (CPU, no checkpoints from a real run)."""
import os

import pytest
import torch

from tttmem import exp_ttt as t, exp_ve as ve, exp_ve_diag as dg

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-ved-90-tests"))

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
def test_base_stream_matches_wave1_metrics_and_rebuild(rule, edge):
    vst, lay, cell, model, data = _setup(rule, edge)
    gen = torch.Generator().manual_seed(0)
    for tok, meta in data:
        out = dg.stream(model, lay, vst, cell, tok, meta, "base", 0.5, gen)
        row, _ = ve.batch_metrics(model, lay, vst, cell, tok, meta)
        s = dict(zip(ve.METRIC_NAMES, row.tolist()))
        for k in ve.TYPES:
            assert out["q"][f"hit_{k}"] == s[f"hit_{k}"]
            assert out["q"][f"n_{k}"] == s[f"n_{k}"]
        assert out["rebuild_maxdiff"] < 1e-5


def test_zero_beta_injection_is_the_base_stream():
    vst, lay, cell, model, data = _setup()
    tok, meta = data[0]
    gen = torch.Generator().manual_seed(0)
    base = dg.stream(model, lay, vst, cell, tok, meta, "base", 0.5, gen)
    for cond in ("erase_v0", "erase_rand", "add_v1"):
        out = dg.stream(model, lay, vst, cell, tok, meta, cond, 0.0, gen)
        assert out["q"] == base["q"]
        assert out["margin"] == pytest.approx(base["margin"], abs=1e-6)


def test_injection_touches_only_the_c_chunk_update():
    vst, lay, cell, model, data = _setup()
    tok, meta = data[0]
    idx, y = tok[:, :-1], tok[:, 1:]
    h = model.backbone(idx).detach()
    etas = [e.detach() for e in model.etas(10.0)]
    base = t.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, t.GradStats(), h=h)
    tf = dg.injection("erase_v0", lay, meta, y, 2.0, torch.Generator().manual_seed(0))
    inj = t.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, t.GradStats(), h=h,
                            inner_transform=tf)
    cut = (dg.c_chunk(lay) + 1) * lay.chunk
    assert torch.equal(base[:, :cut], inj[:, :cut])
    assert not torch.allclose(base[:, cut:], inj[:, cut:])


def test_injection_targets_only_a_writes_with_the_right_values():
    vst, lay, cell, model, data = _setup()
    tok, meta = data[0]
    y = tok[:, 1:]
    mask, local = dg._a_writes(lay, meta)
    assert int(mask.sum()) == int((meta["c_type"] == ve.A).sum())
    seen = {cond: dg.injection_targets(cond, lay, meta, y, torch.Generator().manual_seed(1))[1]
            for cond in ("erase_v0", "erase_rand", "add_v1")}
    is_a = meta["c_type"] == ve.A
    at = lambda tgt: tgt.gather(1, local)[is_a]
    v0 = meta["c_old"][is_a]
    v1 = y.gather(1, meta["c_write"])[is_a]
    assert torch.equal(at(seen["erase_v0"]), v0)
    assert torch.equal(at(seen["add_v1"]), v1)
    rand = at(seen["erase_rand"])
    assert bool(((rand != v0) & (rand != v1)).all())
    assert bool(((rand >= lay.val0) & (rand < lay.val0 + lay.n_val)).all())


def test_erase_v0_lowers_the_old_value_at_a_queries():
    vst, lay, cell, model, data = _setup()
    tok, meta = data[0]
    idx, y = tok[:, :-1], tok[:, 1:]
    h = model.backbone(idx).detach()
    etas = [e.detach() for e in model.etas(10.0)]
    zq = lambda tf: ve._gather_last(t.eval_sequential(
        model, idx, y, etas, vst.g_clip, lay.chunk, t.GradStats(), h=h, inner_transform=tf),
        meta["query_pos"])
    is_a = meta["query_type"] == ve.A
    old = meta["query_old"].clamp_min(0)
    p_old = lambda z: z.softmax(-1).gather(-1, old[..., None]).squeeze(-1)[is_a].mean()
    base = p_old(zq(None))
    erased = p_old(zq(dg.injection("erase_v0", lay, meta, y, 5.0, torch.Generator())))
    assert erased < base


def test_count_rule_matches_wave1():
    assert [dg.count_verdict(n, 6) for n in range(7)] == ["F"] * 4 + ["I", "S", "S"]
    assert [dg.count_verdict(n, 4) for n in range(5)] == ["F", "F", "F", "I", "S"]
    assert dg.count_verdict(3, 3) == "NE"


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
    cfg = dict(process_name="TTM-ved-91-test", device="cpu", threads=2, ckpt_dir=str(tmp_path),
               n_eval=4, eval_batch=2, ve=TINY, cells=cells)
    res = dg.run(cfg, 0)
    assert res["cells"] == 2
    assert res["scores"]["D1_eta10"]["verdict"] == "NE"   # k = 2 < 4
    with pytest.raises(ValueError):
        dg.settings(dict(cfg, bogus=1))
