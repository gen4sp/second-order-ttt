"""P3-a write-rule control: parity with the inherited paths, the own-residual gradient,
calibration, resume, the torch-free pool and the registered analysis."""
from __future__ import annotations

import json
import os
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_p3a as p3a
from tttmem import exp_ttt as ttt
from tttmem import p3a_analysis as pa
from tttmem import p3a_pool

DEV = torch.device("cpu")


@pytest.fixture
def f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.fixture
def f32():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


def tiny_cfg(**over) -> dict:
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=8,
               chunk=8, T=32, n_keys=2, steps=4, batch=2, n_eval=4, n_val=2, n_quick_eval=4,
               eval_batch=2, eta_lr=0.0, lr=1e-3, lr_total=10, warmup=2, lr_floor=0.1,
               grad_clip=1.0, g_clip=10.0, eta_base=40.0, tf32=False, cal_steps=2,
               quick_every=2, qual_every=2, log_every=1)
    cfg.update(over)
    return cfg


def tiny_batch(cfg: dict, task: str = "echo", n: int = 2, seed: int = 3):
    t = p3a.make_task(cfg, task)
    raw, meta = t.raw(np.random.default_rng(seed), n)
    return t, *t.tensors(raw, meta, DEV)


def slow_params(model):
    return [p for n, p in model.named_parameters() if n != "rho"]


@pytest.mark.parametrize("order,arm", [("FO", "e2e1emb"), ("SO", "e2e2emb")])
def test_ce_recurrence_is_bit_identical_to_inherited_arm(f64, order, arm):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, meta = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 0, arm)
    ref = ttt.arm_loss(model, arm, idx, y, 40., 10., task.chunk, ttt.GradStats(), meta=meta)
    g_ref = torch.autograd.grad(ref, slow_params(model))
    new = p3a.p3a_loss(model, idx, y, "ce", order == "SO", 1.0, chunk=task.chunk)
    g_new = torch.autograd.grad(new, slow_params(model))
    assert torch.equal(new, ref)
    assert all(torch.equal(a, b) for a, b in zip(g_new, g_ref))


@pytest.mark.parametrize("so", [False, True])
def test_lin_recurrence_is_bit_identical_to_inherited_lin(f64, so):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, meta = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 1, "e2e1emb")
    sel = torch.ones(idx.shape, dtype=torch.bool)
    ref = ttt.sequential_inner(model, model.backbone(idx), y, model.per_sequence(len(idx)),
                               ttt.regime_etas(model, 40.), 10., so, ttt.GradStats(), task.chunk,
                               idx, sel=sel, rule="lin")
    g_ref = torch.autograd.grad(ref, slow_params(model))
    new = p3a.p3a_loss(model, idx, y, "lin", so, 1.0, chunk=task.chunk)
    g_new = torch.autograd.grad(new, slow_params(model))
    assert torch.equal(new, ref)
    assert all(torch.equal(a, b) for a, b in zip(g_new, g_ref))


@pytest.mark.parametrize("mult", [1.0, 0.37, 4.0])
def test_own_gradient_is_the_closed_form_delta_rule(f64, mult):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, _ = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 2, "e2e1emb")
    with torch.no_grad():
        model.unembed.weight.normal_(0, 0.3)
        for p in model.head.parameters():
            p.add_(0.05 * torch.randn_like(p))
        h = model.backbone(idx)
        x = model.fast_input(h, idx)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    sl = slice(0, task.chunk)
    _, _, inner = p3a.chunk_terms(model, "own", h[:, sl], x[:, sl], W, y[:, sl])
    auto = torch.autograd.grad(mult * inner.mean(1).sum(), W)
    closed = p3a.own_closed_form(model, x[:, sl], W, y[:, sl], mult)
    for a, b in zip(auto, closed):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-13)
    # the per-token loss is exactly 1/2 ||n(r) - n(U[y])||^2 with r the fast MLP output and
    # n the parameter-free LayerNorm; it is bounded by 2d and invariant to the scale of r
    # and of the target row
    r = p3a.fast_residual(W, x[:, sl])

    def n(v):
        c = v - v.mean(-1, keepdim=True)
        return c / torch.sqrt(c.square().mean(-1, keepdim=True) + 1e-5)

    want = 0.5 * (n(r) - n(model.unembed.weight[y[:, sl]])).square().sum(-1)
    assert torch.allclose(inner, want)
    d = r.shape[-1]
    assert bool((inner >= 0).all()) and bool((inner <= 2 * d + 1e-9).all())
    # scale invariance once var(r) >> eps (the tiny model's r is near the eps scale)
    big = p3a.own_tokens(model, 1e3 * r.detach(), y[:, sl])
    assert torch.allclose(big, p3a.own_tokens(model, 1e4 * r.detach(), y[:, sl]), rtol=1e-6)
    with torch.no_grad():
        model.unembed.weight.mul_(1e3)
    # target-scale invariance up to eps / var(U[y]) (~1e-4 here)
    assert torch.allclose(p3a.own_tokens(model, r.detach(), y[:, sl]), inner.detach(),
                          rtol=1e-3)


def test_own_target_is_detached_and_independent_of_the_prediction(f64):
    """The own write ignores the readout and the attention path's prediction.

    With the fast input x held fixed, changing the residual stream h (which changes the
    whole-model prediction) or the readout LayerNorm leaves the own gradient unchanged,
    while the CE gradient moves.  The target U[y] receives no gradient.
    """
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, _ = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 4, "e2e1emb")
    with torch.no_grad():
        model.unembed.weight.normal_(0, 0.3)
        h = model.backbone(idx)
        x = model.fast_input(h, idx)
    sl = slice(0, task.chunk)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]

    def grads(rule, hh):
        _, _, inner = p3a.chunk_terms(model, rule, hh[:, sl], x[:, sl], W, y[:, sl])
        return torch.autograd.grad(inner.mean(1).sum(), W)

    own0, ce0 = grads("own", h), grads("ce", h)
    h2 = 3.0 * h + 0.7 * torch.randn_like(h)
    own1, ce1 = grads("own", h2), grads("ce", h2)
    assert all(torch.equal(a, b) for a, b in zip(own0, own1))
    assert max(float((a - b).abs().max()) for a, b in zip(ce0, ce1)) > 1e-6
    with torch.no_grad():
        model.ln_f.weight.mul_(1.7)
    assert all(torch.equal(a, b) for a, b in zip(own0, grads("own", h)))
    _, _, inner = p3a.chunk_terms(model, "own", h[:, sl], x[:, sl], W, y[:, sl])
    assert torch.autograd.grad(inner.sum(), model.unembed.weight, allow_unused=True)[0] is None


def test_eval_recurrence_matches_inherited_for_ce_and_lin(f64):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, meta = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 5, "e2e1emb")
    ref = ttt.eval_batch_logits(model, task, "e2e1emb", idx, y, 40., 10., task.chunk,
                                ttt.GradStats(), learned=False, meta=meta)
    assert torch.equal(p3a.eval_logits(model, idx, y, "ce", 1.0, chunk=task.chunk), ref)
    sel = torch.ones(idx.shape, dtype=torch.bool)
    ref = ttt.eval_sequential(model, idx, y, ttt._fixed_etas(model, 40.), 10., task.chunk,
                              ttt.GradStats(), sel, rule="lin")
    assert torch.equal(p3a.eval_logits(model, idx, y, "lin", 1.0, chunk=task.chunk), ref)
    assert not p3a.eval_logits(model, idx, y, "own", 2.0, chunk=task.chunk).requires_grad


def test_multiplier_scales_the_pre_clip_gradient_and_the_clip_binds_on_it(f64):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, _ = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 6, "e2e1emb")
    n_chunk = task.T // task.chunk
    norms = {}
    for mult in (1e-3, 1e3):
        tel = p3a.Telemetry(1, n_chunk, len(idx), 10.0, DEV)
        p3a.p3a_loss(model, idx, y, "own", False, mult, chunk=task.chunk, tel=tel)
        tel.end_step()
        norms[mult] = tel.buf[0].clone()
        summ, pre = tel.first_k()
        assert pre.shape == (1, n_chunk, len(idx))
    # chunk 0 is evaluated at W0 for both multipliers: exactly linear there (float32 buffer)
    assert torch.allclose(norms[1e3][0], 1e6 * norms[1e-3][0], rtol=1e-6)
    assert bool((norms[1e3] > 10.0).all()) and bool((norms[1e-3] < 10.0).all())
    assert summ["clip_frac"] == 1.0


@pytest.mark.parametrize("rule", p3a.RULES)
def test_second_order_path_exists_for_every_rule(f64, rule):
    cfg = tiny_cfg()
    st = ttt._settings(cfg)
    task, idx, y, meta = tiny_batch(cfg)
    model = ttt.build_model(task, st, DEV, 7, "e2e1emb")
    with torch.no_grad():
        model.unembed.weight.normal_(0, 0.3)
    mult = 1.0 if rule == "ce" else 1.3
    term = p3a.second_order_term(model, (idx, y, meta), rule, mult, eta=40., g_clip=10.,
                                 chunk=task.chunk)
    assert all(math.isfinite(v) for v in term.values())
    assert term["rel_all"] > 1e-8
    write = p3a.write_direction(model, (idx, y, meta), rule, mult, task.chunk)
    if rule == "ce":
        assert write["cos_all"] == pytest.approx(1.0, abs=1e-12)
        assert write["norm_ratio_all"] == pytest.approx(1.0, abs=1e-12)
    assert -1.0 - 1e-9 <= write["cos_all"] <= 1.0 + 1e-9


def test_next_multiplier_secant():
    # proportional response: one step lands exactly
    assert p3a.next_multiplier([(1.0, 4.0)], 1.0) == pytest.approx(0.25)
    # response med = m**0.5: the secant recovers the exponent from two points
    hist = [(1.0, 1.0), (0.25, 0.5)]
    assert p3a.next_multiplier(hist, 0.1) == pytest.approx(0.01)
    # slope clamp keeps a flat response from exploding
    assert p3a.next_multiplier([(1.0, 1.0), (2.0, 1.0)], 0.5) == pytest.approx(2.0 * 0.5 ** 4)
    with pytest.raises(ValueError):
        p3a.next_multiplier([(1.0, 0.0)], 1.0)


def test_cal_start_and_freeze_on_the_closest_tested_round():
    assert p3a.cal_start({}, "own") == 0.01 and p3a.cal_start({}, "lin") == 1.0
    assert p3a.cal_start(dict(cal_start=dict(own=0.5)), "own") == 0.5
    hist = [(0.01, 0.05), (0.008, 0.041), (0.0079, 0.0455)]
    # |log(0.0455/0.044)| = 0.034 < |log(0.041/0.044)| = 0.071 < |log(0.05/0.044)| = 0.128
    assert p3a.closest_round(hist, 0.044) == 0.0079
    assert p3a.closest_round(hist, 0.040) == 0.008
    assert p3a.closest_round([(1.0, float("nan")), (0.5, 0.2)], 0.1) == 0.5
    with pytest.raises(ValueError):
        p3a.closest_round([(1.0, 0.0)], 0.1)


def test_design_guard_enforces_registered_seeds():
    base = dict(seeds=[330, 331], cal_seeds=[340], eta_lr=0.0)
    p3a.check_design(base)
    for bad in (dict(seeds=[200]), dict(cal_seeds=[331], seeds=[331]), dict(cal_seeds=[343]),
                dict(eta_lr=0.01), dict(rules=["lin", "own"]), dict(seeds=[1])):
        with pytest.raises(ValueError):
            p3a.check_design({**base, **bad})
    p3a.check_design(dict(seeds=[0, 1], cal_seeds=[2], smoke=True))


def test_pool_runs_stdlib_jobs_in_spawned_children_and_reports_failures():
    jobs = [dict(key=f"k{i}", sleep=0.05) for i in range(4)] + [dict(key="bad", fail=True)]
    with p3a_pool.CellPool(2, "tttmem.p3a_pool:probe_job", "tttmem.p3a_pool:probe_init",
                           ("t",)) as pool:
        out = {k: (r, e) for k, r, e in pool.run(jobs, stall_s=120)}
    assert set(out) == {"k0", "k1", "k2", "k3", "bad"}
    assert out["bad"][0] is None and "probe failure bad" in out["bad"][1]
    assert {out[f"k{i}"][0]["index"] for i in range(4)} <= {1, 2}
    assert all(out[f"k{i}"][0]["tag"].startswith("t-w") for i in range(4))
    with p3a_pool.CellPool(1, "tttmem.p3a_pool:probe_job") as pool:
        (k, r, e), = list(pool.run([dict(key="x")]))
    assert k == "x" and e is None and r["index"] == 0


def test_pool_surfaces_an_abrupt_worker_death():
    from concurrent.futures.process import BrokenProcessPool
    with p3a_pool.CellPool(2, "tttmem.p3a_pool:probe_job") as pool:
        with pytest.raises(BrokenProcessPool):
            list(pool.run([dict(key="die", exit=True), dict(key="k", sleep=1.0)], stall_s=120))


def _fake_cell(root: Path, task, rule, order, seed, acc, *, retained=None, defined=True):
    rec = dict(key=f"{task}__{rule}__{order}__s{seed:03d}", task=task, rule=rule, order=order,
               seed=seed, mult=1.0, steps=1500, g_clip=10.0, defined=defined,
               retained=acc >= .9 if retained is None else retained, retained_ce=acc >= .9,
               ever_acquired=True, t_acq=300, t_loss=None, diverged=not defined, nonfinite_steps=0,
               ms_step=200.0, wall_s=330.0, first_k=dict(post_median=0.02, clip_frac=0.0),
               train_tel=dict(clip_frac=0.0), quick=[dict(step=1500, acc_own=acc)], qualifiers=[],
               final=dict(own=dict(acc=acc, query_ce=0.1, eval_clip_frac=0.0), ce=dict(acc=acc)))
    folder = root / "main" / rec["key"]
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(rec))


def _scenario(root: Path, accs: dict, seeds=range(330, 336), jitter=0.01):
    rng = np.random.default_rng(0)
    for s in seeds:
        for (task, rule, order), a in accs.items():
            _fake_cell(root, task, rule, order, s, float(np.clip(a + jitter * rng.standard_normal(), 0, 1)))


def _accs(echo: dict, ctrl: dict) -> dict:
    out = {}
    for rule in p3a.RULES:
        for order in p3a.ORDERS:
            out[("echo", rule, order)] = echo[rule][order]
            out[("echo_ctrl", rule, order)] = ctrl[rule][order]
    return out


def test_analysis_supported_not_supported_and_ne(tmp_path):
    good = {r: dict(FO=.995, SO=.995) for r in p3a.RULES}
    echo = dict(ce=dict(FO=.40, SO=.99), lin=dict(FO=.98, SO=.99), own=dict(FO=.97, SO=.98))
    _scenario(tmp_path / "a", _accs(echo, good))
    res = pa.analyze(tmp_path / "a")
    assert res["verdict"]["label"] == "D-SUPPORTED" and not res["verdict"]["interim"]
    assert {"lin-ce", "own-ce"} <= {q["pair"] for q in res["verdict"]["qualifying"] if q["ok"]}
    assert res["gaps"]["echo"]["ce"]["hi"] < -0.5
    md = pa.render_md(res)
    assert "D-SUPPORTED" in md and "| echo | ce | FO |" in md
    # same rule dependence on the copy-free control: not specific -> not supported
    ctrl = dict(ce=dict(FO=.40, SO=.99), lin=dict(FO=.98, SO=.99), own=dict(FO=.97, SO=.98))
    _scenario(tmp_path / "b", _accs(echo, ctrl))
    assert pa.analyze(tmp_path / "b")["verdict"]["label"] == "D-NOT-SUPPORTED"
    # anchor failure: FO-CE retains on echo -> NE
    flat = {r: dict(FO=.99, SO=.99) for r in p3a.RULES}
    _scenario(tmp_path / "c", _accs(flat, good))
    v = pa.analyze(tmp_path / "c")["verdict"]
    assert v["label"] == "NE" and "anchor" in v["reason"]
    # an undefined cell -> NE for the whole family
    _scenario(tmp_path / "d", _accs(echo, good))
    rec_file = tmp_path / "d" / "main" / "echo__own__SO__s330" / "result.json"
    rec = json.loads(rec_file.read_text())
    rec_file.write_text(json.dumps({**rec, "defined": False, "diverged": True}))
    assert pa.analyze(tmp_path / "d")["verdict"]["label"] == "NE"
    # three complete seeds -> interim
    _scenario(tmp_path / "e", _accs(echo, good), seeds=range(330, 333))
    v = pa.analyze(tmp_path / "e")["verdict"]
    assert v["label"] == "D-SUPPORTED" and v["interim"] and v["n_seeds"] == 3
    # a non-functional own rule (recall at floor everywhere): its pairs are NE, so a floor
    # DiD cannot carry the verdict; lin-ce (no DiD) decides
    floor = dict(FO=.02, SO=.02)
    echo_f = dict(ce=dict(FO=.40, SO=.99), lin=dict(FO=.40, SO=.99), own=floor)
    ctrl_f = dict(ce=dict(FO=.99, SO=.99), lin=dict(FO=.99, SO=.99), own=floor)
    _scenario(tmp_path / "f", _accs(echo_f, ctrl_f))
    res = pa.analyze(tmp_path / "f")
    v = res["verdict"]
    assert v["label"] == "D-NOT-SUPPORTED" and v["ne_pairs"] == ["own-ce", "own-lin"]
    assert not res["functional"]["own"]["ok"] and res["functional"]["lin"]["ok"]
    assert "NON-FUNCTIONAL" in pa.render_md(res)
    # both non-CE rules at floor -> NE overall
    _scenario(tmp_path / "g", _accs(dict(echo_f, lin=floor), dict(ctrl_f, lin=floor)))
    v = pa.analyze(tmp_path / "g")["verdict"]
    assert v["label"] == "NE" and "non-functional" in v["reason"]


def test_ci_uses_student_t():
    c = pa.ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    half = 2.0150 * np.std([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ddof=1) / math.sqrt(6)
    assert c["lo"] == pytest.approx(0.35 - half) and c["hi"] == pytest.approx(0.35 + half)
    z = pa.ci([0.0] * 6)
    assert pa.within(z) and not pa.excludes_zero(z)


def smoke_cfg(root: Path, **over) -> dict:
    cfg = tiny_cfg(process_name=os.environ.get("PROCESS_NAME", "TTM-s3a-03-test"), device="cpu", workers=1, threads_per_worker=1,
                   output_root=str(root), smoke=True, seeds=[0], cal_seeds=[1], cal_rounds=2,
                   steps=4, cal_steps=2, parity_batch=2, profile_steps=0, max_hours=1.0,
                   cutoff_margin_s=0)
    cfg.update(over)
    return cfg


def test_end_to_end_tiny_wave_freezes_before_main_and_resumes(tmp_path, f32, capsys):
    cfg = smoke_cfg(tmp_path)
    res = p3a.run(cfg)
    log = capsys.readouterr().out
    assert res["complete"] and res["counts"]["done"] == 12
    assert "P3A_PARITY ok" in log
    freeze_at, first_main = log.index("P3A_FREEZE "), log.index('"key": "echo__ce__FO__s000"')
    assert freeze_at < first_main
    mult = json.loads((tmp_path / "cal" / "multipliers.json").read_text())
    assert mult["ce"] == 1.0 and mult["lin"] > 0 and mult["own"] > 0 and len(mult["rounds"]) == 2
    cells = sorted((tmp_path / "main").glob("*/result.json"))
    assert len(cells) == 12
    rec = json.loads((tmp_path / "main" / "echo__own__SO__s000" / "result.json").read_text())
    assert rec["mult"] == mult["own"] and rec["steps_done"] == 4 and rec["quick"]
    assert rec["qualifiers"] and "cos_kv" in rec["qualifiers"][-1]["write"]
    assert (tmp_path / "analysis" / "analysis.md").exists()
    # resume: nothing retrains, the frozen multipliers are reused
    res2 = p3a.run(cfg)
    log2 = capsys.readouterr().out
    assert res2["counts"]["resumed"] == 12 and res2["counts"]["done"] == 0
    assert "P3A_FREEZE_RESUMED" in log2 and "P3A_CELL_DONE" not in log2
    # a changed science knob refuses to mix with the old outputs
    with pytest.raises(ValueError):
        p3a.run({**cfg, "lr": 2e-3})


def test_deadline_skips_unstarted_cells(tmp_path, f32):
    cfg = smoke_cfg(tmp_path, frozen_multipliers=dict(lin=0.5, own=2.0), cal_seeds=[], parity=False,
                    run_analysis=False)
    out = p3a.run_specs(p3a_pool.CellPool(1, "tttmem.exp_p3a:run_cell"), cfg,
                        p3a.main_specs(cfg, dict(ce=1.0, lin=0.5, own=2.0))[:2],
                        tmp_path / "main", cutoff=0.0, stall_s=None)
    assert set(out.values()) == {"deadline"}
    assert not list((tmp_path / "main").glob("*/result.json"))
