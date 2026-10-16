"""Tiny single-process CPU tests for R4-4a; never launch a run or a worker."""
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_p01 as p
from tttmem import exp_ttt as t

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-r44a-91-tests"))


def small(**extra):
    cfg = dict(d_model=16, n_layers=2, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=16, n_keys=4, T=96, n_eval=2, n_quick_eval=2,
               eval_batch=2, n_probe=2, batch=2, steps=3, quick_every=1, probe_every=1,
               lr_total=3000, kamin_switch=2, d1_n_eval=2, workers=1, device="cpu", tf32=False)
    cfg.update(extra)
    st = t._settings(cfg)
    return cfg, st, t._task(st, "echo")


def sample(task, seed=9, n=2):
    return task.tensors(*task.raw(np.random.default_rng(seed), n), torch.device("cpu"))


def analyzer():
    spec = importlib.util.spec_from_file_location("analyze_r44a", ROOT/"scripts/analyze_r44a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_edge_mask_removes_source_influence_in_all_layers():
    _, st, task = small()
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_edge")
    idx, y, meta = sample(task)
    assert p.copy_edge_mask is t.copy_edge_mask
    wp = torch.as_tensor(meta["write_pos"][:, :1])
    changed = idx.clone()
    changed[torch.arange(len(idx))[:, None], wp-1] += 1
    captured = []
    handle = model.blocks[-1].register_forward_hook(lambda m, a, out: captured.append(out.detach()))
    for inputs in (idx, changed):
        t.arm_loss(model, "e2e1emb_edge", inputs, y, 10., 10., task.chunk, t.GradStats(), meta=meta)
    handle.remove()
    assert torch.equal(p.gather(captured[0], wp), p.gather(captured[1], wp))
    assert not torch.equal(p.gather(model.backbone(idx), wp), p.gather(model.backbone(changed), wp))


@pytest.mark.parametrize("arm", t.COPY_Q)
def test_noise_properties_and_clean_stream_identity_every_step(arm):
    _, _, task = small()
    clean_rng, other_rng = [np.random.default_rng([3, t.TRAIN_SEED]) for _ in range(2)]
    for step in range(3):
        clean, meta = task.raw(clean_rng, 256)
        base, bm = task.raw(other_rng, 256)
        assert np.array_equal(clean, base)
        noisy, nm = t.corrupt_echo(base, bm, arm, 3, step)
        again, am = t.corrupt_echo(base, bm, arm, 3, step)
        assert np.array_equal(noisy, again) and np.array_equal(nm["corrupted"], am["corrupted"])
        rows = np.arange(len(base))[:, None]
        wp, flags = meta["write_pos"], nm["corrupted"]
        assert np.array_equal(noisy[rows, wp+1], clean[rows, wp+1])  # clean kv labels
        assert np.array_equal(noisy[rows, wp-1] != clean[rows, wp-1], flags)
        altered = noisy != clean
        allowed = np.zeros_like(altered)
        allowed[rows, wp-1] = True
        assert not (altered & ~allowed).any()
        assert abs(flags.mean()-(1-t.COPY_Q[arm])) < .04
        assert np.array_equal(base, clean)  # corruption never mutates input
        sliced = t._slice_meta(nm, 4, 7)
        assert sliced["corrupted"].shape == (3, task.n_keys)


def test_other_15_uniform_and_exact_rate_and_w_marker_label():
    st = t._settings(dict(T=512, n_keys=16, n_layers=4, window=64))
    task = t._task(st, "echo")
    raw, m = task.raw(np.random.default_rng(7), 4096)
    noisy, nm = t.corrupt_echo(raw, m, "e2e1emb_cn90", 4, 0)
    rows, keys = np.nonzero(nm["corrupted"])
    wp = m["write_pos"][rows, keys]
    offset = (noisy[rows, wp-1] - raw[rows, wp+1]) % 16
    assert np.all(offset != 0)
    assert abs(nm["corrupted"].mean()-.1) < .005
    hist = np.bincount(offset, minlength=16)[1:]
    assert (hist.max()-hist.min())/hist.mean() < .4
    # The stream label of the W marker is necessarily the changed copy input.
    idx, y, _ = task.tensors(noisy, nm, torch.device("cpu"))
    assert np.array_equal(y[rows, wp-2].numpy(), noisy[rows, wp-1])
    assert np.array_equal(y[rows, wp].numpy(), raw[rows, wp+1])


def test_rf_touches_only_filler_gap_and_same_event_rate():
    _, _, task = small()
    raw, meta = task.raw(np.random.default_rng(7), 2048)
    noisy, nm = t.corrupt_echo(raw, meta, "e2e1emb_rf95", 1, 0)
    rows, positions = np.nonzero(raw != noisy)
    assert np.all(positions > meta["write_pos"][rows].max(1)+1)
    assert np.all(positions < meta["query_pos"][rows].min(1)-1)
    assert np.all(raw[rows, positions] >= 2+2*task.n_keys)
    assert np.all((noisy[rows, positions] >= 2+task.n_keys) &
                  (noisy[rows, positions] < 2+2*task.n_keys))
    assert abs(nm["corrupted"].mean()-.05) < .01
    assert len(rows) == nm["corrupted"].sum()


@pytest.mark.parametrize("arm", t.KAMIN_ARMS)
def test_s5_fallback_restores_final_kamin_regime(arm, tmp_path, monkeypatch):
    cfg, st, task = small()
    cell = dict(task="echo", seed=0, arm=arm, eta_base=10, diag=False)
    cfg["cells"] = [cell]
    model = t.build_model(task, st, torch.device("cpu"), 0, arm)
    path = tmp_path/"final.pt"
    torch.save(model.state_dict(), path)
    ident = t._cell_id(cell, task.steps)
    spec = importlib.util.spec_from_file_location("r44a_s5", ROOT/"scripts/r42l_s5_fp32.py")
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    actual = t.r4_probe
    seen = []

    def check(model, *args, **kwargs):
        seen.append(model.inner_multiplier)
        return actual(model, *args, **kwargs)

    monkeypatch.setattr(t, "r4_probe", check)
    rows = fallback.recompute(cfg, tmp_path, {ident: dict(arm=arm, seed=0, task="echo",
                                                         steps=task.steps, checkpoint=str(path))})
    assert seen == [1.] and rows[0]["passed"]


@pytest.mark.parametrize("arm", t.KAMIN_ARMS)
def test_kamin_stage1_w_equals_w0_and_stage2_normal(arm, monkeypatch):
    _, st, task = small()
    model = t.build_model(task, st, torch.device("cpu"), 1, arm)
    idx, y, meta = sample(task)
    h = model.backbone(idx, edge_mask=t.arm_edge_mask(model, arm, idx, meta))
    original = model.read(model.fast_block(h, idx, model.slow()))
    visited = []
    real_block = t.fast_block

    def capture(h, ln, params, x=None):
        visited.append([w.detach().clone() for w in params])
        return real_block(h, ln, params, x)

    monkeypatch.setattr(t, "fast_block", capture)
    sel = t.lin_selector(arm, meta["write_pos"], len(idx), task.T, idx.device)
    t.arm_loss(model, arm, idx, y, 10., 10., task.chunk, t.GradStats(), sel=sel, meta=meta)
    got = t.eval_batch_logits(model, task, arm, idx, y, 10., 10., task.chunk, t.GradStats(), meta=meta)
    assert all(torch.equal(w, initial) for weights in visited
               for w, initial in zip(weights, model.per_sequence(len(idx))))
    # Full-sequence static matmul and chunked per-sequence matmul round differently.
    torch.testing.assert_close(original, got, atol=1e-7, rtol=0)
    t.set_kamin_regime(model, arm, st.kamin_switch, st)
    reference = "e2e1emb_linkv" if arm.endswith("kaminlin") else "e2e1emb"
    a = t.arm_loss(model, arm, idx, y, 10., 10., task.chunk, t.GradStats(), sel=sel, meta=meta)
    b = t.arm_loss(model, reference, idx, y, 10., 10., task.chunk, t.GradStats(), sel=sel)
    assert torch.equal(a, b)
    a = t.eval_batch_logits(model, task, arm, idx, y, 10., 10., task.chunk, t.GradStats(), meta=meta)
    b = t.eval_batch_logits(model, task, reference, idx, y, 10., 10., task.chunk, t.GradStats(), meta=meta)
    assert torch.equal(a, b)


@pytest.mark.parametrize("g_clip", [10., .001])
def test_kv_eta_equivalent_shared_and_nonzero_outer_gradient(g_clip):
    _, st, task = small()
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_kveta")
    assert torch.equal(model.rho_kvW2, model.rho[2])
    idx, y, meta = sample(task)
    losses = [t.arm_loss(model, arm, idx, y, 10., g_clip, task.chunk, t.GradStats(), meta=meta)
              for arm in ("e2e1emb", "e2e1emb_kveta")]
    assert torch.equal(*losses)
    logits = [t.eval_batch_logits(model, task, arm, idx, y, 10., g_clip, task.chunk,
                                  t.GradStats(), meta=meta) for arm in ("e2e1emb", "e2e1emb_kveta")]
    assert (logits[0]-logits[1]).abs().max() <= 1e-6
    losses[1].backward()
    assert model.rho_kvW2.grad.abs() > 0 and torch.isfinite(model.rho_kvW2.grad)
    groups = t.build_optim(model, st).param_groups
    own = next(g for g in groups if any(p is model.rho_kvW2 for p in g["params"]))
    assert own["base_lr"] == .01 and own["weight_decay"] == 0
    assert not any(p is model.rho_kvW2 for p in t.clipped_params(model))


def test_kv_eta_unequal_step_uses_full_gradient_clip():
    _, st, task = small()
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_kveta10")
    assert model.kv_eta_lr == .1
    idx, y, meta = sample(task)
    h = model.backbone(idx).detach()
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    ce = t.token_ce(model.read(model.fast_block(h[:, :16], idx[:, :16], W)), y[:, :16])
    kv = t.lin_selector("e2e1emb_linkv", meta["write_pos"], len(idx), 16, idx.device)
    full = torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=True)
    gkv = torch.autograd.grad((ce*kv).mean(1).sum(), W[2])[0]
    clipped, norm, _ = t._clip_fast(list(full), .001)
    scale = torch.where(norm > .001, .001/norm, torch.ones_like(norm))[:, None, None]
    expected = W[2]-scale*(3*gkv+2*(full[2]-gkv))
    got = t.kv_w2_step(W[2], clipped[2], gkv, 2., 3., norm, .001)
    assert (got-expected).abs().max() <= 1e-6


def test_all_end_cell_paths_tiny_cpu_and_kamin_freeze(tmp_path, monkeypatch):
    cfg, st, _ = small()
    cfg["cells"] = [dict(task="echo", seed=0, arm=a, eta_base=10, diag=False) for a in t.R44A_ARMS]
    monkeypatch.chdir(tmp_path)
    result = t.run_cells(cfg, st, torch.device("cpu"))
    cells = {c["arm"]: c for c in result["cells"].values()}
    assert len(cells) == 12
    for a, c in cells.items():
        assert "error" not in c, (a, c.get("error"))
        assert c["s5"]["passed"] and Path(c["checkpoint"]).exists()
        assert "bound_ever" in c and "bound_final" in c
    for a in t.KAMIN_ARMS:
        c = cells[a]
        assert c["switch_check"]["rho_unchanged"] and c["rho_stage1_unchanged"]
        assert c["switch_check"]["rho_init"] == c["switch_check"]["rho_at_switch"]
        assert c["stage2_first_bound"] is None
        assert c["probes"][0]["inner_multiplier"] == 0
        assert c["probes"][1]["inner_multiplier"] == 1
    assert "p_kv_masked_at_switch" in cells["e2e1emb_kaminctl"]["switch_check"]
    for a in t.NOISE_ARMS:
        c = cells[a]
        assert len(c["corrupted_quick"]) == 3
        assert c["corrupted_eval"]["n_clean_keys"]+c["corrupted_eval"]["n_corrupted_keys"] == 8
        assert "p_kv_corrupted" in c["probes"][0]
        assert c["first_clean_batch_sha1"] == cells["e2e1emb"]["first_batch_sha1"]
    assert cells["e2e1emb_edge"]["ablation_ref_match"]
    assert "mask_off" in cells["e2e1emb_edge"]["route_checks"]
    assert all(cells[a]["d1"]["n_seq"] == 2 for a in ("e2e1emb", "e2e1emb_edge"))
    assert len(cells["e2e1emb_kveta"]["eta_kvW2_trajectory"]) == 3


def test_config_100_seed_major_and_checkpoint_pairing():
    cfg = tomllib.loads((ROOT/"configs/toy/r4-4a.toml").read_text())
    assert cfg["experiment"] == "ttt"
    cfg = cfg["ttt"]
    want = [(s, a) for s in range(10) for a in t.R44A_ARMS if s < 6 or a not in t.NOISE_ARMS]
    assert [(c["seed"], c["arm"]) for c in cfg["cells"]] == want and len(want) == 100
    assert cfg["process_name"] == "TTM-r44a-02" and cfg["workers"] == 6
    assert cfg["kamin_switch"] == 500 and cfg["tf32"] is False
    assert cfg["copy_noise_q"] == list(t.COPY_Q.values())
    assert cfg["kv_eta_lr"] == .01 and cfg["kv_eta10_lr"] == .1
    for c in cfg["cells"]:
        assert c["task"] == "echo" and c["eta_base"] == 10 and c["diag"] is False
    st = t._settings(cfg)
    for arm in t.R44A_ARMS:
        expected = set(range(150, 401, 25)) if arm in cfg["checkpoint_dense_arms"] else (
            {475, 500, 525, 550, 600, 700, 1000} if arm in t.KAMIN_ARMS else set())
        assert t.checkpoint_plan(st, arm, 4) == expected
        assert t.checkpoint_plan(st, arm, 5) == set()
    smoke = tomllib.loads((ROOT/"configs/toy/smoke/r4-4a-gpu.toml").read_text())["ttt"]
    assert smoke["steps"] == 60 and smoke["kamin_switch"] == 30
    assert smoke["process_name"] == "TTM-r44a-01-podsmoke"
    assert [c["arm"] for c in smoke["cells"]] == list(t.R44A_ARMS)
    assert tomllib.loads((ROOT/"configs/toy/p0-1.toml").read_text())["experiment"] == "p01"


def synthetic_cells():
    cells = {}
    for s in range(10):
        for a in t.R44A_ARMS:
            if s >= 6 and a in t.NOISE_ARMS:
                continue
            acc = .5 if a == "e2e1emb" else .95
            cells[s, a] = dict(cell_id=f"{s}/{a}", seed=s, task="echo", arm=a, steps=1500,
                               first_batch_sha1=str(s), warmup=200,
                               lin_per_seq=16 if a.endswith("kaminlin") else 0, s5=dict(passed=True),
                               bound_ever=True, bound_final=acc >= .9, quick=[[200, 0, 0, .95], [250, 0, 0, acc]],
                               eval=dict(query_acc=acc, query_ce=.3, ce=.5),
                               stage2_ever_bound=a != "e2e1emb_kamin", stage2_first_bound=None,
                               switch_check=dict(step=500, passed=True, rho_unchanged=True, p_kv_at_switch=.995),
                               ablation=dict(all=acc, all_minus_kv=.1, kv_only=acc), ablation_ref_match=True,
                               route_checks=dict(p_v_kv_masked=.0625, mask_off=dict(query_acc=.7, p_kv=.99)),
                               corrupted_eval=dict(acc_clean_keys=.7, acc_corrupted_keys=.9,
                                                   n_clean_keys=4000, n_corrupted_keys=96))
    return cells


def test_analysis_directions_routes_holm_and_missing_s5(tmp_path):
    a = analyzer()
    cells = synthetic_cells()
    result = a.summarize(cells, {}, [], tmp_path/"missing.csv")
    assert result["GM2"]["passed"] and result["GM2"]["test"]["mcnemar_one_sided"] == 1/1024
    assert result["GM3"]["passed"] and result["GM3"]["test"]["positive"] == 10
    assert result["other_readings"]["R3"]["p"] == 2**-24
    assert all(result["other_readings"][r]["holm_p"] >= result["other_readings"][r]["p"]
               for r in ("R1", "R2", "R3"))
    cells[0, "e2e1emb_edge"]["ablation"]["kv_only"] = .1
    assert not a.summarize(cells, {}, [], tmp_path/"missing.csv")["GM2"]["passed"]
    cells[0, "e2e1emb_edge"].pop("s5")
    assert a.paired_test(cells, "e2e1emb_edge", "e2e1emb")["n"] == 9
    cells[0, "e2e1emb_kamin"]["switch_check"]["passed"] = False
    result = a.summarize(cells, {}, [], tmp_path/"missing.csv")
    assert result["GM3"]["statistical_pass"] and not result["GM3"]["passed"]


def test_analysis_cli_outputs_and_reproducibility(tmp_path, capsys):
    a = analyzer()
    cells = synthetic_cells()
    log = tmp_path/"log.txt"
    log.write_text("\n".join("CELL_RESULT "+json.dumps(c) for c in cells.values()))
    ref = tmp_path/"r42l.csv"
    ref.write_text("seed,e2e1emb_bound_ever,e2e1emb_bound_final,e2e1emb_t_lose\n"+
                   "\n".join(f"{s},True,False,250" for s in range(10))+"\n")
    out = tmp_path/"out"
    a.main([str(log), "--out", str(out), "--anchor-table", str(ref)])
    summary = json.loads((out/"summary.json").read_text())
    assert summary["reproducibility"]["differences"] == []
    assert summary["GM2"]["passed"]
    assert (out/"a6-trend.csv").exists() and (out/"gm2-routes.csv").exists()
    assert "R44A_SUMMARY " in capsys.readouterr().out
    cells[0, "e2e1emb_cn95"]["first_clean_batch_sha1"] = "bad"
    log.write_text("\n".join("CELL_RESULT "+json.dumps(c) for c in cells.values()))
    parsed, _, issues = a.records(log)
    assert issues and not a.paired_test(parsed, "e2e1emb_edge", "e2e1emb")["n"] == 10


def test_analysis_sign_ties_and_smoke_do_not_pass_gates(tmp_path):
    a = analyzer()
    cells = synthetic_cells()
    for c in cells.values():
        c["corrupted_eval"]["acc_corrupted_keys"] = .7
    out = a.summarize(cells, dict(process_name="TTM-r44a-01-podsmoke"), [], tmp_path/"none")
    assert out["status"] == "smoke_only" and not out["GM2"]["passed"] and not out["GM3"]["passed"]
    assert out["other_readings"]["R3"]["p"] == 1 and out["other_readings"]["R3"]["ties"] == 24


def test_holm_incomplete_reading_cannot_relax_the_other_thresholds():
    a = analyzer()
    out = a.holm(dict(R1=dict(p=.02, complete=True), R2=dict(p=.001, complete=False),
                      R3=dict(p=1., complete=True)))
    assert out["R1"]["holm_p"] == .06 and not out["R1"]["holm_pass"]
    assert out["R2"]["p"] == .001 and out["R2"]["holm_input_p"] == 1.


def test_r2_bonferroni_and_kveta_only_fallback():
    # Bonferroni over the two eta_lr arms; kveta alone
    # when kveta10 has fewer than 10 valid pairs
    a = analyzer()
    both = a.r2_reading({a.KV: dict(n=10, mcnemar_one_sided=.01), a.KV10: dict(n=10, mcnemar_one_sided=.03)})
    assert both["p"] == .02 and both["rule"] == "bonferroni_2" and both["complete"]
    capped = a.r2_reading({a.KV: dict(n=10, mcnemar_one_sided=.7), a.KV10: dict(n=10, mcnemar_one_sided=.6)})
    assert capped["p"] == 1.
    alone = a.r2_reading({a.KV: dict(n=10, mcnemar_one_sided=.01), a.KV10: dict(n=7, mcnemar_one_sided=.001)})
    assert alone["p"] == .01 and alone["rule"] == "kveta_only" and alone["complete"]
    assert not a.r2_reading({a.KV: dict(n=9, mcnemar_one_sided=.01),
                             a.KV10: dict(n=10, mcnemar_one_sided=.01)})["complete"]


def test_finite_maps_nonfinite_to_none_for_json():
    a = analyzer()
    out = a.finite(dict(x=float("nan"), y=[1., float("inf"), dict(z=-float("inf"))], s="a", n=3))
    assert out == dict(x=None, y=[1., None, dict(z=None)], s="a", n=3)
    json.dumps(out, allow_nan=False)
