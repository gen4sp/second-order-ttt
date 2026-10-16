"""R4-2L: the linear token loss, the one selector, the schedule, probes and analysis."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t

ROOT = Path(__file__).resolve().parents[1]
# the same expectation the analysis script applies to the six echo arms
LIN_PER_SEQ = {"e2e1emb_linkv": 16, "e2e1emb_linvW": 16, "e2e1emb_linxkv": 496,
               "e2e1emb_lin": 512}
R42L_ARMS = ("e2e1emb", "e2e1emb_linkv", "e2e1emb_lin", "e2e1emb_linvW", "e2e1emb_linxkv",
             "e2e1emb_w600", "e2e1emb_ctrl")


def script(name):
    """Load a repository script as a module (scripts live outside the package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def analysis():
    return script("analyze_r42l")


def small_echo(mode: str = "echo", **extra):
    base = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
                window=16, chunk=16, n_keys=4, T=128, n_eval=4, n_quick_eval=4,
                eval_batch=2, n_probe=4)
    base.update(extra)
    st = t._settings(base)
    return st, t._task(st, mode)


def batch(task, seed=7, n=2, dev=torch.device("cpu")):
    tokens, meta = task.raw(np.random.default_rng(seed), n)
    return task.tensors(tokens, meta, dev)


# --------------------------------------------------------------------- the sanity


def test_r42l_sanity_l1_to_l8(tmp_path):
    checks = t.r42l_sanity()
    path = Path(tmp_path) / "sanity-test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checks, indent=2) + "\n")
    assert all(v["passed"] for v in checks.values()), json.dumps(checks)
    assert checks["L1"]["existing_path_equal"] and checks["L1"]["all_ce_selector_equal"]
    assert checks["L2"]["max_abs"] <= 1e-7 and checks["L3"]["max_abs"] <= 1e-7
    assert checks["L4"]["e2e1emb_linkv"]["per_seq"] == [16] * 4
    assert checks["L4"]["e2e1emb_linvW"]["per_seq"] == [16] * 4
    assert checks["L4"]["e2e1emb_linxkv"]["per_seq"] == [240] * 4
    assert checks["L4"]["e2e1emb_lin"]["per_seq"] == [256] * 4
    assert checks["L4"]["kv_target_is_written_value"]["passed"]
    assert checks["L5"]["linkv_inner_gradient_split"]["rel"] <= 1e-6
    for arm in t.LIN_ARMS:
        assert checks["L5"][arm]["passed"], arm
    assert abs(checks["L6"]["w600_step199"] - 200 / 600) <= 1e-12
    assert checks["L6"]["w600_step599"] == 1.0 and checks["L6"]["other_step199"]["e2e1emb"] == 1.0
    assert .85 <= checks["L7"]["r_ne_f_frac"] <= .99
    for arm in t.LIN_ARMS:
        assert checks["L8"][arm]["rel"] <= 1e-6, (arm, checks["L8"][arm])


# ----------------------------------------------------------- token loss, selectors


def test_linear_token_loss_is_prediction_independent_and_matches_uniform_ce():
    gen = torch.Generator().manual_seed(1)
    y = torch.randint(0, 9, (3, 4), generator=gen)
    grads = []
    for seed in (2, 3):
        z = torch.randn(3, 4, 9, generator=torch.Generator().manual_seed(seed))
        z = z.detach().requires_grad_(True)
        grads.append(torch.autograd.grad(t.lin_token(z, y).sum(), z)[0])
    ref = 1.0 / 9 - torch.nn.functional.one_hot(y, 9).to(torch.float32)
    assert torch.equal(grads[0], grads[1])
    assert (grads[0] - ref).abs().max() <= 1e-7
    zeros = torch.zeros(3, 4, 9, requires_grad=True)
    assert (torch.autograd.grad(t.token_ce(zeros, y).sum(), zeros)[0] - grads[0]
            ).abs().max() <= 1e-7
    ce = torch.rand(3, 4, generator=gen)
    lin = torch.rand(3, 4, generator=gen)
    assert torch.equal(t.token_loss(None, lin, ce), ce)
    sel = torch.tensor([[True, False, True, False]] * 3)
    assert torch.equal(t.token_loss(sel, lin, ce), torch.where(sel, lin, ce))


def test_selectors_positions_and_counts():
    st, task = small_echo(batch=2)
    _, _, meta = batch(task, n=4)
    dev = torch.device("cpu")
    wp = np.asarray(meta["write_pos"])
    kv = t.lin_selector("e2e1emb_linkv", meta["write_pos"], 4, task.T, dev)
    vW = t.lin_selector("e2e1emb_linvW", meta["write_pos"], 4, task.T, dev)
    xkv = t.lin_selector("e2e1emb_linxkv", meta["write_pos"], 4, task.T, dev)
    every = t.lin_selector("e2e1emb_lin", meta["write_pos"], 4, task.T, dev)
    assert t.lin_selector("e2e1emb", meta["write_pos"], 4, task.T, dev) is None
    assert t.lin_selector("e2e1emb_w600", meta["write_pos"], 4, task.T, dev) is None
    for i in range(4):
        assert {int(v) for v in kv[i].nonzero().flatten()} == set(wp[i])
        assert {int(v) for v in vW[i].nonzero().flatten()} == set(wp[i] + 1)
        assert {int(v) for v in xkv[i].nonzero().flatten()} == set(range(task.T)) - set(wp[i])
        assert int(every[i].sum()) == task.T
    assert [t.lin_count(a, task.T, task.n_keys) for a in
            ("e2e1emb", "e2e1emb_linkv", "e2e1emb_linvW", "e2e1emb_linxkv", "e2e1emb_lin")] == \
        [0, 4, 4, task.T - 4, task.T]
    assert t.arm_inner_loss("e2e1emb_lin") == "lin"
    assert t.arm_inner_loss("e2e1emb_linvW") == "linvW"
    assert t.arm_inner_loss("e2e1emb_linxkv") == "linxkv"
    assert t.arm_inner_loss("e2e1emb_w600") == "ce" and t.arm_inner_loss("e2e1emb_ctrl") == "ce"


def test_lin_arm_needs_its_selector():
    st, task = small_echo()
    idx, y, _ = batch(task)
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_lin")
    with pytest.raises(ValueError):
        t.arm_loss(model, "e2e1emb_lin", idx, y, 10., 10., task.chunk, t.GradStats())


# --------------------------------------------------------- CE arms bit-identical


@pytest.mark.parametrize("arm", ["e2e1emb", "e2e1emb_kvm"])
def test_ce_arms_are_bit_identical_to_the_prechange_recurrence(arm):
    """An independent copy of the R4-2 recurrence: loss and every outer gradient."""
    dev = torch.device("cpu")
    st, task = small_echo(batch=2)
    _, _, meta = batch(task)
    keep = t.outer_keep(meta, (2, task.T), t.MASK_ARMS.get(arm, "none"), None, dev)
    old = t.build_model(task, st, dev, 0, arm)
    new = t.build_model(task, st, dev, 0, arm)
    opt_old, opt_new = t.build_optim(old, st), t.build_optim(new, st)
    rng = np.random.default_rng([0, t.TRAIN_SEED])
    losses = []
    for _ in range(3):
        idx, y, meta = task.tensors(*task.raw(rng, st.batch), dev)
        h = old.backbone(idx)
        x = old.fast_input(h, idx)
        weights = old.per_sequence(len(idx))
        etas = old.etas(10.)
        total = 0.0
        for c in range(task.T // task.chunk):
            sl = slice(c * task.chunk, (c + 1) * task.chunk)
            ce = t._ce_tokens(old, t.fast_block(h[:, sl], old.head.ln, weights, x[:, sl]),
                              y[:, sl])
            loss_c = ce.mean(1)
            g = torch.autograd.grad(loss_c.sum(), weights, retain_graph=True)
            g, _, _ = t._clip_fast(list(g), 10.)
            total = total + ((ce * keep[:, sl]).sum(1) / task.chunk if keep is not None
                             else loss_c).mean()
            weights = [w - eta * v for w, eta, v in zip(weights, etas, g)]
        original = total / (task.T // task.chunk)
        current = t.arm_loss(new, arm, idx, y, 10., 10., task.chunk, t.GradStats(), keep)
        assert torch.equal(original, current)
        opt_old.zero_grad(set_to_none=True)
        opt_new.zero_grad(set_to_none=True)
        original.backward()
        current.backward()
        for a, b in zip(old.parameters(), new.parameters()):
            assert torch.equal(a.grad, b.grad)
        opt_old.step()
        opt_new.step()
        losses.append(float(original.detach()))
    assert all(math.isfinite(v) for v in losses)


def test_lin_arms_change_only_the_inner_gradient():
    st, task = small_echo()
    idx, y, meta = batch(task)
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_lin")
    losses, grads = {}, {}
    for arm in ("e2e1emb", "e2e1emb_linkv", "e2e1emb_linvW", "e2e1emb_linxkv", "e2e1emb_lin"):
        sel = t.lin_selector(arm, meta["write_pos"], len(idx), y.shape[1], idx.device)
        losses[arm] = float(t.arm_loss(model, arm, idx, y, 10., 10., task.chunk,
                                       t.GradStats(), None, sel).detach())
        grads[arm] = t._capture_inner_grads(
            lambda: t.arm_loss(model, arm, idx, y, 10., 10., task.chunk, t.GradStats(),
                               None, sel))
    # the outer loss stays the per-chunk CE at every token, so only the weights it is
    # evaluated at differ; every lin arm's own outer value differs from the anchor's
    for arm in ("e2e1emb_linkv", "e2e1emb_linvW", "e2e1emb_linxkv", "e2e1emb_lin"):
        assert losses[arm] != losses["e2e1emb"]
        assert not torch.allclose(grads[arm][0], grads["e2e1emb"][0])
    # the four selectors are four different interventions at the first chunk
    firsts = [grads[arm][0] for arm in grads]
    assert all(not torch.allclose(a, b) for i, a in enumerate(firsts) for b in firsts[i + 1:])


# ------------------------------------------------------------- schedule and cells


def test_warmup_override_and_probe_lr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st, task = small_echo(probe_every=10, quick_every=10)
    out = {}
    for arm in ("e2e1emb", "e2e1emb_w600"):
        model = t.build_model(task, st, torch.device("cpu"), 0, arm)
        out[arm] = t.train_arm(model, task, arm, 10., st, torch.device("cpu"),
                               np.random.default_rng(3), None, f"echo/s0/{arm}",
                               steps=20, seed=0)
    assert out["e2e1emb"]["warmup"] == 200 and out["e2e1emb_w600"]["warmup"] == 600
    assert out["e2e1emb"]["outer_grad"]["from_step"] == 200
    assert out["e2e1emb_w600"]["outer_grad"]["from_step"] == 600
    for arm, res in out.items():
        assert res["probes"], arm
        for probe in res["probes"]:
            expect = st.lr * t.lr_scale(probe["step"] - 1, res["warmup"], 20, st.lr_floor)
            assert probe["lr"] == pytest.approx(expect, rel=1e-12), (arm, probe["step"])
    assert t.arm_warmup("e2e1emb_w600", st) == 600
    assert t.arm_warmup("e2e1emb", st) == 200


def test_lin_per_seq_and_checkpoint_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st, task = small_echo(checkpoint_steps=[10, 20], checkpoint_dense=[15],
                          checkpoint_dense_arms=["e2e1emb", "e2e1emb_linkv"],
                          checkpoint_dense_max_seed=4,
                          checkpoint_arm_steps=[{"arm": "e2e1emb_w600", "steps": [12]}])
    seen = {}
    for arm in R42L_ARMS[:-1]:
        model = t.build_model(task, st, torch.device("cpu"), 0, arm)
        res = t.train_arm(model, task, arm, 10., st, torch.device("cpu"),
                          np.random.default_rng(3), None, f"echo/s0/{arm}", steps=20, seed=0)
        seen[arm] = (res["lin_per_seq"], [Path(p).name for p in res["step_checkpoints"]],
                     res["inner_loss"])
    assert {arm: v[0] for arm, v in seen.items()} == {"e2e1emb": 0, "e2e1emb_linkv": 4,
                                                      "e2e1emb_linvW": 4,
                                                      "e2e1emb_linxkv": task.T - 4,
                                                      "e2e1emb_lin": task.T,
                                                      "e2e1emb_w600": 0}
    assert {arm: v[2] for arm, v in seen.items()} == {
        "e2e1emb": "ce", "e2e1emb_linkv": "linkv", "e2e1emb_lin": "lin",
        "e2e1emb_linvW": "linvW", "e2e1emb_linxkv": "linxkv", "e2e1emb_w600": "ce"}
    assert seen["e2e1emb"][1] == ["echo__s0__e2e1emb__step10.pt", "echo__s0__e2e1emb__step15.pt",
                                  "echo__s0__e2e1emb__step20.pt"]
    assert seen["e2e1emb_linkv"][1] == ["echo__s0__e2e1emb_linkv__step10.pt",
                                        "echo__s0__e2e1emb_linkv__step15.pt",
                                        "echo__s0__e2e1emb_linkv__step20.pt"]
    assert seen["e2e1emb_linxkv"][1] == ["echo__s0__e2e1emb_linxkv__step10.pt",
                                         "echo__s0__e2e1emb_linxkv__step20.pt"]
    assert seen["e2e1emb_w600"][1] == ["echo__s0__e2e1emb_w600__step10.pt",
                                       "echo__s0__e2e1emb_w600__step12.pt",
                                       "echo__s0__e2e1emb_w600__step20.pt"]
    # the seed limit keeps the dense window away from the later seeds
    assert t.checkpoint_plan(st, "e2e1emb", 4) == {10, 15, 20}
    assert t.checkpoint_plan(st, "e2e1emb", 5) == {10, 20}
    assert t.checkpoint_plan(st, "e2e1emb", None) == {10, 20}


def test_echo_ctrl_cell_gets_probes_flags_and_ablation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st, task = small_echo("echo_ctrl", probe_every=10, quick_every=10)
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_ctrl")
    res = t.train_arm(model, task, "e2e1emb_ctrl", 10., st, torch.device("cpu"),
                      np.random.default_rng(5), None, "echo_ctrl/s0/e2e1emb_ctrl",
                      steps=20, seed=0)
    assert res["probes"] and all("p_kv" in p for p in res["probes"])
    assert res["lin_per_seq"] == 0 and res["inner_loss"] == "ce"
    flags = t.r4_flags(res["quick"], res["probes"], 0.1, 0.30)
    assert {"bound_ever", "bound_final", "episodes", "t_addr_lost"} <= set(flags)
    tokens, meta = t.fixed_set(task, st, "eval")
    data = t.eval_batches(task, tokens, meta, torch.device("cpu"), st.eval_batch)
    ablation = t.r42_ablation(model, task, data, 10., st.g_clip, arm="e2e1emb_ctrl")
    reference = t.evaluate(model, task, "e2e1emb_ctrl", data, meta, 10., st)["eval"]["query_acc"]
    assert abs(ablation["all"] - reference) < .01


def test_ctrl_arm_is_refused_on_echo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = dict(process_name="TTM-r42L-test", device="cpu", workers=1, max_minutes=1,
               d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=16,
               chunk=16, T=128, n_keys=4, batch=2, n_eval=2, n_quick_eval=2, eval_batch=2,
               cells=[dict(task="echo", arm="e2e1emb_ctrl", seed=0, eta_base=1., steps=1,
                           diag=False)])
    with pytest.raises(ValueError, match="echo_ctrl"):
        t.run_cells(cfg, t._settings(cfg), torch.device("cpu"))


@pytest.mark.parametrize("arm", t.LIN_ARMS)
@pytest.mark.parametrize("workers", [1, 2])
def test_lin_cell_refuses_ce_diagnostics(arm, workers):
    cfg = dict(workers=workers, cells=[dict(task="echo", arm=arm, seed=0,
                                          eta_base=10., diag=True)])
    with pytest.raises(ValueError, match="diag paths use CE"):
        t.run_cells(cfg, t._settings(cfg), torch.device("cpu"))


@pytest.mark.parametrize("func", [t.r4_probe, t.r42_ablation])
def test_probe_and_ablation_require_keyword_arm(func):
    with pytest.raises(TypeError, match="arm"):
        func(None, None, [], 10., 10.)
    with pytest.raises(TypeError):
        func(None, None, [], 10., 10., "e2e1emb")


def test_tf32_key_sets_asserts_and_logs(capsys):
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        state = t.configure_tf32(False, "0")
        assert state == dict(tf32=False, tf32_matmul=False, tf32_cudnn=False,
                             float32_matmul_precision="highest")
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert "TF32 worker=0 matmul=False cudnn=False" in capsys.readouterr().out
        assert t.configure_tf32(True, "main")["tf32"] is True
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32) = saved
    assert t._settings(dict(tf32=False)).tf32 is False
    assert t._settings(dict()).tf32 is True


# ------------------------------------------------------------------ probe fields


def test_probe_fields_for_ce_and_lin_arms():
    st, task = small_echo()
    idx, y, meta = batch(task, n=4)
    data = [(idx, y, meta)]
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb")
    ce = t.r4_probe(model, task, data, 10., 10., arm="e2e1emb")
    new = ("e_kv_ce_abs", "e_vW_abs", "vW_write_eff", "upd_W1", "upd_b1", "upd_W2",
           "upd_b2", "clipped_frac_c0", "inner_norm_c0", "eta_W1", "eta_b1", "eta_W2",
           "eta_b2")
    for field in new:
        assert field in ce and math.isfinite(ce[field]), field
    # in a CE arm the arm's inner error is the CE error
    assert ce["e_kv_ce_abs"] == pytest.approx(ce["e_kv_abs"], rel=1e-9)
    assert 0 <= ce["clipped_frac_c0"] <= 1 and ce["inner_norm_c0"] > 0
    for arm in t.LIN_ARMS:
        probe = t.r4_probe(model, task, data, 10., 10., arm=arm)
        for field in new:
            assert math.isfinite(probe[field]), (arm, field)
        # e_kv_ce_abs depends only on the CE, so it is arm-independent
        assert probe["e_kv_ce_abs"] == pytest.approx(ce["e_kv_ce_abs"], rel=1e-9)
    assert t.r4_probe(model, task, data, 10., 10., arm="e2e1emb_lin")["lin_kv_max_diff"] < 1e-4


def test_ablation_all_reproduces_the_streaming_eval_of_a_lin_arm():
    st, task = small_echo()
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_lin")
    tokens, ev_meta = t.fixed_set(task, st, "eval")
    data = t.eval_batches(task, tokens, ev_meta, torch.device("cpu"), st.eval_batch)
    ablation = t.r42_ablation(model, task, data, 10., st.g_clip, arm="e2e1emb_lin")
    reference = t.evaluate(model, task, "e2e1emb_lin", data, ev_meta, 10., st)
    assert set(ablation) == {"all", "all_minus_kv", "kv_only"}
    assert abs(ablation["all"] - reference["eval"]["query_acc"]) < .01
    # the CE arm's ablation also matches its evaluation
    plain = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb")
    assert abs(t.r42_ablation(plain, task, data, 10., st.g_clip, arm="e2e1emb")["all"] -
               t.evaluate(plain, task, "e2e1emb", data, ev_meta, 10., st
                          )["eval"]["query_acc"]) < .01


# ----------------------------------------------------------------------- configs


def test_r42l_config_orders():
    with (ROOT / "configs/toy/r4-2l.toml").open("rb") as f:
        cfg = tomllib.load(f)["ttt"]
    with (ROOT / "configs/toy/smoke/r4-2l.toml").open("rb") as f:
        smoke = tomllib.load(f)["ttt"]
    arms = ("e2e1emb", "e2e1emb_linkv", "e2e1emb_lin", "e2e1emb_linvW", "e2e1emb_linxkv",
            "e2e1emb_w600", "e2e1emb_ctrl")
    want = [(seed, arm, "echo_ctrl" if arm == "e2e1emb_ctrl" else "echo")
            for seed in range(10) for arm in arms]
    assert [(c["seed"], c["arm"], c["task"]) for c in cfg["cells"]] == want
    assert len(cfg["cells"]) == 70 and cfg["workers"] == 6 and cfg["tf32"] is False
    assert cfg["process_name"] == "TTM-r42L-02" and cfg["max_minutes"] == 120
    assert cfg["mps_memory_fraction"] == .20 and cfg["lr_total"] == 3000
    assert cfg["warmup"] == 200 and cfg["lr_floor"] == .1 and cfg["device"] == "auto"
    assert cfg["checkpoint_steps"] == [100, 200, 250, 300, 500, 1000]
    assert cfg["checkpoint_dense"] == list(range(150, 401, 25))
    assert cfg["checkpoint_dense_arms"] == ["e2e1emb", "e2e1emb_linkv", "e2e1emb_w600"]
    assert cfg["checkpoint_dense_max_seed"] == 4
    assert cfg["checkpoint_arm_steps"] == [{"arm": "e2e1emb_w600",
                                            "steps": [600, 700, 800]}]
    assert all(c["eta_base"] == 10 and c["diag"] is False for c in cfg["cells"])
    assert [(c["seed"], c["arm"]) for c in smoke["cells"]] == [
        (seed, arm) for seed in range(2) for arm in arms]
    assert smoke["device"] == "cpu" and smoke["workers"] == 1 and smoke["steps"] == 30
    assert smoke["tf32"] is False and smoke["probe_every"] == 10


def test_podsmoke_config_and_checkpoint_keys():
    with (ROOT / "configs/toy/smoke/r4-2l-gpu.toml").open("rb") as stream:
        cfg = tomllib.load(stream)["ttt"]
    st = t._settings(cfg)
    assert len(cfg["cells"]) == 7
    assert [c["arm"] for c in cfg["cells"]] == list(R42L_ARMS)
    assert all(c["seed"] == 0 and c["diag"] is False for c in cfg["cells"])
    assert st.process_name == "TTM-r42L-00-podsmoke" and st.steps == 100 and st.workers == 6
    for key, value in cfg.items():
        if key.startswith("checkpoint_"):
            assert key in t.DEFAULTS and getattr(st, key) == value
    for c in cfg["cells"]:
        t.check_arm_task(c["arm"], c["task"])
        expected = {50, 100}
        if c["arm"] in st.checkpoint_dense_arms:
            expected.add(75)
        if c["arm"] == "e2e1emb_w600":
            expected.add(60)
        assert t.checkpoint_plan(st, c["arm"], c["seed"]) == expected
        assert all(0 < step <= st.steps for step in expected)
    for key in ("checkpoint_steps", "checkpoint_steps_so", "checkpoint_dense"):
        assert all(isinstance(step, int) and 0 < step <= st.steps for step in cfg[key])


@pytest.mark.parametrize("arm", R42L_ARMS)
def test_s5_fp32_uses_explicit_cell_arm(tmp_path, monkeypatch, arm):
    fallback = script("r42l_s5_fp32")
    task_name = "echo_ctrl" if arm == "e2e1emb_ctrl" else "echo"
    st, task = small_echo(task_name)
    cfg = {key: getattr(st, key) for key in ("d_model", "n_layers", "n_heads", "backbone_mult",
           "fast_mult", "window", "chunk", "n_keys", "T", "n_probe", "eval_batch")}
    cell_cfg = dict(task=task_name, arm=arm, seed=0, eta_base=10., steps=30, diag=False)
    cfg["cells"] = [cell_cfg]
    ident = t._cell_id(cell_cfg, 30)
    path = tmp_path / f"{ident.replace('/', '__')}.pt"
    model = t.build_model(task, st, torch.device("cpu"), 0, arm)
    state = model.state_dict()
    torch.save(state, path)
    log = tmp_path / "cells.log"
    record = dict(cell_id=ident, arm=arm, seed=0, task=task_name, checkpoint=str(path))
    log.write_text("CELL_RESULT " + json.dumps(record) + "\n")
    real_probe = t.r4_probe
    called = []

    def checked_probe(model, task, data, eta_base, g_clip, *, arm):
        assert next(model.parameters()).device.type == "cpu"
        assert next(model.parameters()).dtype == torch.float32
        called.append(arm)
        return real_probe(model, task, data, eta_base, g_clip, arm=arm)

    monkeypatch.setattr(t, "r4_probe", checked_probe)
    rows = fallback.recompute(cfg, tmp_path, fallback.cell_results(log))
    assert called == [arm]
    assert set(rows[0]) == {"cell_id", "arm", "seed", "s5_max_diff", "passed"}
    assert rows[0]["passed"] and rows[0]["s5_max_diff"] <= 1e-4
    assert rows[0]["arm"] == arm and rows[0]["cell_id"] == ident and rows[0]["seed"] == 0
    with pytest.raises(ValueError, match="arm required"):
        fallback.recompute(cfg, tmp_path, {})
    torch.save(dict(state_dict=state, metadata=dict(arm=arm)), path)
    assert fallback.recompute(cfg, tmp_path, {}) == rows
    wrong_arm = "e2e1emb_linkv" if arm == "e2e1emb" else "e2e1emb"
    with pytest.raises(ValueError, match="arms disagree"):
        fallback.recompute(cfg, tmp_path, {ident: {**record, "arm": wrong_arm}})


# ---------------------------------------------------------------------- analysis


def cell(seed, arm, task_name, *, ever, final, acc_value, quick, probes):
    """One synthetic CELL_RESULT with only the fields the analysis reads."""
    return dict(cell_id=f"{task_name}/s{seed}/{arm}/eta10/st1500", task=task_name, arm=arm,
                seed=seed, bound_ever=bool(ever), bound_final=bool(final), collapse=False,
                degraded=False, eval=dict(query_acc=acc_value, query_ce=1.0, ce=2.0),
                quick=quick, probes=probes, s5=dict(passed=True),
                inner_loss=t.arm_inner_loss(arm), lin_per_seq=LIN_PER_SEQ.get(arm, 0),
                warmup=600 if arm == "e2e1emb_w600" else 200,
                first_batch_sha1="hash-echo" if task_name == "echo" else "hash-ctrl",
                ablation=dict(all=acc_value, all_minus_kv=0.20, kv_only=0.95),
                ablation_ref_match=True, tf32=False, peak_cuda_mb=0.0, masked_per_seq=0)


def quick_rows(pairs):
    """Quick-eval rows ``[step, ce, query_ce, query_acc]`` from ``(step, acc)`` pairs."""
    return [[s, 2.0, 1.0, a] for s, a in pairs]


def probes_for(pairs, kv=1e-3, vW=1e-3):
    """Probe rows with the fields the rules read; ``kv``/``vW`` set the write efficiencies."""
    return [dict(step=s, p_kv=p, kv_write_eff=kv, vW_write_eff=vW, e_kv_ce_abs=1e-3,
                 e_kv_abs=1e-3, addr_a=0.9, ce_kv=0.1, ce_vk=0.5, ce_vW=0.5, lr=1e-3,
                 upd_W1=0.1, upd_b1=0.01, upd_W2=0.2, upd_b2=0.02, eta_W1=10.0, eta_b1=10.0,
                 eta_W2=10.0, eta_b2=10.0, clipped_frac_c0=0.5, inner_norm_c0=10.0)
            for s, p in pairs]


def synthetic_log(path: Path, seeds=range(10), over=None) -> Path:
    """A log that fires LK-rescue, LM-partial, LS kv-specific, NC-loses and SC-tracks-p.

    ``over`` replaces one arm's profile for every seed; a callable is evaluated at
    the seed, which is how the label matrix below moves one endpoint at a time.
    """
    over = over or {}
    refs = dict(chance=dict(acc=.0625), occ=dict(acc=.19), occ_minus=dict(acc=.30))
    lines = [f"REF echo {json.dumps(refs)}", f"REF echo_ctrl {json.dumps(refs)}"]
    for s in seeds:
        for arm, profile in BASE_PROFILE.items():
            spec_p = {**profile, **over.get(arm, {})}
            values = {k: (v(s) if callable(v) else v) for k, v in spec_p.items()}
            values["acc_value"] = values.pop("acc")
            task_name = "echo_ctrl" if arm == "e2e1emb_ctrl" else "echo"
            lines.append("CELL_RESULT " + json.dumps(cell(s, arm, task_name, **values)))
    path.write_text("\n".join(lines) + "\n")
    return path


# the anchor acquires at 200-300 and drops below 0.90 at 325; p_kv passes 0.90 at 250
LOSING = quick_rows([(200, .95), (225, .95), (250, .95), (275, .95), (300, .95), (325, .12)])
SHORT_LOSING = quick_rows([(200, .95), (250, .95), (300, .95), (325, .12)])
P90 = probes_for([(250, .98), (300, .99), (325, .99)])
BIG_WRITE = probes_for([(250, .98), (300, .99), (325, .99)], kv=.1)
BIG_VW_WRITE = probes_for([(250, .98), (300, .99), (325, .99)], vW=.1)
STARVED = probes_for([(step, .05) for step in range(200, 326, 25)])
NEVER_LOSES = quick_rows([(200, .95), (250, .95), (300, .95)])
LATE_LOSING = quick_rows([(200, .95), (250, .95), (300, .95), (625, .12)])
LATE_PROBES = probes_for([(250, .98), (300, .99), (600, .99), (625, .99)])

BASE_PROFILE = {
    "e2e1emb": dict(ever=True, final=False, acc=.10, quick=LOSING, probes=P90),
    "e2e1emb_linkv": dict(ever=True, final=lambda s: s < 8, acc=lambda s: .95 if s < 8 else .10,
                          quick=SHORT_LOSING, probes=BIG_WRITE),
    "e2e1emb_lin": dict(ever=True, final=lambda s: s < 4, acc=lambda s: .55 if s < 6 else .10,
                        quick=LOSING, probes=BIG_WRITE),
    "e2e1emb_linvW": dict(ever=True, final=False, acc=.10, quick=LOSING, probes=BIG_VW_WRITE),
    "e2e1emb_linxkv": dict(ever=True, final=False, acc=.10, quick=LOSING, probes=P90),
    "e2e1emb_w600": dict(ever=True, final=False, acc=.10, quick=LOSING, probes=P90),
    "e2e1emb_ctrl": dict(ever=lambda s: s < 8, final=False, acc=.10, quick=SHORT_LOSING,
                         probes=STARVED),
}


def test_analysis_labels_on_a_synthetic_log(tmp_path, monkeypatch, capsys):
    module = analysis()
    evidence = Path(tmp_path)
    evidence.mkdir(parents=True, exist_ok=True)
    path = synthetic_log(evidence / "synthetic.log")
    result = module.summarize(*module.records(str(path)))
    monkeypatch.setattr(sys, "argv", ["analyze_r42l.py", str(path), str(evidence / "synthetic")])
    module.main()
    (evidence / "synthetic_analysis_stdout.json").write_text(capsys.readouterr().out)
    assert json.loads((evidence / "synthetic/summary.json").read_text())["labels"] == result["labels"]
    labels = result["labels"]
    assert labels["V-L"] == "holds"
    assert labels["LK"] == "LK-rescue", labels["LK"]
    assert labels["LM"] == "LM-partial", labels["LM"]
    assert labels["LS"] == "kv-specific", labels["LS"]
    assert labels["NC"] == "NC-loses", labels["NC"]
    assert labels["SC"] == "SC-tracks-p", labels["SC"]
    assert labels["LX"] == "descriptive" and not labels["descriptive"]
    for key in ("D1", "D2", "D3"):
        assert labels["D"]["e2e1emb_linkv"][key] is True, key
    assert labels["ML3"]["e2e1emb_linkv"]["passed"] is True
    assert labels["ML3"]["e2e1emb"]["passed"] is None       # no retained seed
    assert result["LK"]["mcnemar"] <= .05 and result["LK"]["ci95"][1] > 0
    assert result["SC"]["delta_p"] == 0 and abs(result["SC"]["delta_lr"]) >= 200
    assert result["NC"]["window_ok"] and result["NC"]["n_losing"] == 8
    assert result["LS"]["vW_take_seeds"] == 10
    assert result["issues"] == [] and len(result["per_seed"]) == 10
    d4 = [r for rows in result["D4"].values() for r in rows]
    assert d4 and all("upd_W2_final_ratio" in r and "n_window_probes" in r for r in d4)


LABEL_MATRIX = [
    ({}, {"LK": "LK-rescue", "LM": "LM-partial", "LS": "kv-specific", "NC": "NC-loses",
          "SC": "SC-tracks-p", "V-L": "holds"}),
    ({"e2e1emb_linkv": dict(ever=False, final=False, acc=.10, quick=LOSING, probes=P90)},
     {"LK": "LK-blocked"}),
    ({"e2e1emb_linkv": dict(ever=True, final=lambda s: s < 1,
                            acc=lambda s: .95 if s < 1 else .10, quick=LOSING,
                            probes=BIG_WRITE)}, {"LK": "LK-no rescue shown"}),
    ({"e2e1emb_linkv": dict(ever=True, final=lambda s: s < 4,
                            acc=lambda s: .95 if s < 4 else .10, quick=LOSING,
                            probes=BIG_WRITE)}, {"LK": "LK-partial"}),
    ({"e2e1emb_linkv": dict(ever=True, final=lambda s: s < 3,
                            acc=lambda s: .40 if s < 3 else .10, quick=LOSING,
                            probes=BIG_WRITE)}, {"LK": "LK-inconclusive"}),
    ({"e2e1emb_ctrl": dict(ever=False, final=False, acc=.10, quick=SHORT_LOSING,
                           probes=STARVED)}, {"NC": "NC-blocked"}),
    ({"e2e1emb_ctrl": dict(ever=True, final=lambda s: s < 8, acc=lambda s: .95 if s < 8 else .10,
                           quick=SHORT_LOSING, probes=P90)}, {"NC": "NC-retains"}),
    ({"e2e1emb_ctrl": dict(ever=True, final=lambda s: s < 4, acc=lambda s: .95 if s < 4 else .10,
                           quick=SHORT_LOSING, probes=P90)}, {"NC": "NC-mixed"}),
    ({"e2e1emb_w600": dict(ever=True, final=lambda s: s < 8, acc=lambda s: .95 if s < 8 else .10,
                           quick=SHORT_LOSING, probes=P90)}, {"SC": "SC-retains"}),
    ({"e2e1emb_w600": dict(ever=False, final=False, acc=.10, quick=LOSING, probes=P90)},
     {"SC": "SC-uninformative"}),
    ({"e2e1emb_w600": dict(ever=True, final=False, acc=.10, quick=LATE_LOSING,
                           probes=LATE_PROBES)}, {"SC": "SC-tracks-LR"}),
    ({"e2e1emb_w600": dict(ever=True, final=False, acc=.10, quick=NEVER_LOSES, probes=P90)},
     {"SC": "SC-mixed"}),
    ({"e2e1emb_linvW": dict(ever=True, final=lambda s: s < 3, acc=lambda s: .95 if s < 3 else .10,
                            quick=SHORT_LOSING, probes=BIG_VW_WRITE)},
     {"LS": "specificity unresolved"}),
    ({"e2e1emb_linvW": dict(ever=True, final=lambda s: s < 8, acc=lambda s: .95 if s < 8 else .10,
                            quick=SHORT_LOSING, probes=BIG_VW_WRITE)},
     {"LS": "not kv-specific"}),
    ({"e2e1emb": dict(ever=True, final=lambda s: s < 6, acc=lambda s: .95 if s < 6 else .10,
                      quick=SHORT_LOSING, probes=P90)},
     {"LK": "LK-inconclusive (descriptive: V-L not reproduced)"}),
]


@pytest.mark.parametrize("over,expect", LABEL_MATRIX, ids=[str(i) for i in range(15)])
def test_analysis_label_matrix(tmp_path, over, expect):
    module = analysis()
    result = module.summarize(*module.records(str(synthetic_log(tmp_path / "m.log",
                                                                  over=over))))
    for key, value in expect.items():
        assert result["labels"][key] == value, (key, result["labels"][key])


def test_analysis_is_pending_below_ten_pairs(tmp_path):
    module = analysis()
    result = module.summarize(*module.records(str(synthetic_log(tmp_path / "s.log",
                                                                  seeds=range(2)))))
    assert result["labels"]["V-L"] == "pending"
    for key in ("LK", "LM", "LS", "NC", "SC"):
        assert result["labels"][key].startswith("pending"), (key, result["labels"][key])
    assert result["n_pairs"]["e2e1emb"] == 2
    assert result["labels"]["D"]["e2e1emb_linkv"]["D1"] is None
    assert result["labels"]["ML3"]["e2e1emb_linkv"]["retained"] == 2


def test_analysis_flags_broken_pairing(tmp_path):
    module = analysis()
    path = synthetic_log(tmp_path / "s.log")
    lines = []
    for line in path.read_text().split("\n"):
        if line.startswith("CELL_RESULT ") and '"arm": "e2e1emb_lin"' in line:
            record = json.loads(line[len("CELL_RESULT "):])
            record["first_batch_sha1"] = "other"
            line = "CELL_RESULT " + json.dumps(record)
        lines.append(line)
    path.write_text("\n".join(lines))
    cells, probes, refs, issues = module.records(str(path))
    assert any("first_batch_sha1" in i for i in issues)
    assert cells[(0, "e2e1emb_lin")]["unpaired"]
    assert not module.valid(cells[(0, "e2e1emb_lin")])


def test_analysis_keeps_diverged_lin_cell_in_pairs(tmp_path):
    module = analysis()
    path = synthetic_log(tmp_path / "diverged.log")
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("CELL_RESULT "):
            record = json.loads(line[len("CELL_RESULT "):])
            if record["arm"] == module.LIN and record["seed"] == 0:
                record.update(diverged=True, bound_ever=True, bound_final=False)
                line = "CELL_RESULT " + json.dumps(record)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    result = module.summarize(*module.records(path))
    assert result["LM"]["n_pairs"] == 10 and result["n_pairs"][module.LIN] == 10
    assert result["counts"][module.LIN]["ever"] == 10
    assert result["counts"][module.LIN]["final"] == 3
    assert result["issues"] == ["diverged cell: arm=e2e1emb_lin seed=0 task=echo"]


def test_analysis_flags_wrong_lin_counts_and_warmup(tmp_path):
    module = analysis()
    path = synthetic_log(tmp_path / "s.log")
    lines = []
    for line in path.read_text().split("\n"):
        if line.startswith("CELL_RESULT ") and '"arm": "e2e1emb_w600"' in line:
            record = json.loads(line[len("CELL_RESULT "):])
            record["warmup"] = 200
            record["lin_per_seq"] = 4
            line = "CELL_RESULT " + json.dumps(record)
        lines.append(line)
    path.write_text("\n".join(lines))
    _, _, _, issues = module.records(str(path))
    assert any("lin_per_seq" in i for i in issues) and any("warmup" in i for i in issues)


def test_median_ci_and_expected_counts():
    module = analysis()
    assert module.median_ci([]) is None
    assert module.median_ci([3.0]) == [None, None]
    assert module.median_ci([1, 2, 3, 4, 5]) == [None, None]
    assert module.median_ci(list(range(11))) == [1, 9]
    assert module.want_lin_per_seq("e2e1emb_linkv") == 16
    assert module.want_lin_per_seq("e2e1emb_linvW") == 16
    assert module.want_lin_per_seq("e2e1emb_linxkv") == 496
    assert module.want_lin_per_seq("e2e1emb_lin") == 512
    assert module.want_lin_per_seq("e2e1emb_ctrl") == 0
    assert module.T_ECHO == 512 and module.WINDOW == 150


def test_analysis_missing_evidence_is_not_a_pair(tmp_path):
    module = analysis()
    cells, _, _, _ = module.records(synthetic_log(tmp_path / "s.log"))
    for field in ("first_batch_sha1", "lin_per_seq", "warmup"):
        broken = {key: dict(value) for key, value in cells.items()}
        broken[0, module.LINKV].pop(field)
        assert module.pairing_issues(broken)
        assert not module.valid(broken[0, module.LINKV])


def test_analysis_incomplete_specificity_and_nc_window(tmp_path):
    module = analysis()
    cells, probes, refs, _ = module.records(synthetic_log(tmp_path / "s.log"))
    cells.pop((0, module.LINVW))
    assert module.specificity(cells, probes, "LK-rescue")["label"] == (
        "pending (e2e1emb_linvW not evaluable)")
    cells.pop((1, module.LINKV))
    assert module.specificity(cells, probes, "pending")["label"] == (
        "pending (e2e1emb_linkv, e2e1emb_linvW not evaluable)")
    for seed in range(8):
        probes[cells[seed, module.CTRL]["cell_id"]].pop(225)
    result = module.no_copy(cells, probes, .30)
    assert result["label"] == "NC-mixed"
    assert all(not row["complete_window"] for row in result["checks"])


def test_d4_large_change_thresholds_are_descriptive():
    module = analysis()
    for field in ("upd_W1", "upd_b1", "upd_W2", "upd_b2", "clipped_frac_c0"):
        assert module.alternative_route({f"{field}_final_ratio": 10.})
        assert module.alternative_route({f"{field}_final_ratio": .1})
        assert not module.alternative_route({f"{field}_final_ratio": 1.})
    assert module.alternative_route({"addr_a_final_ratio": .5})
    assert not module.alternative_route({"addr_a_final_ratio": .51})


def test_nonoracle_lin_eval_needs_no_position_metadata():
    st, task = small_echo()
    idx, y, _ = batch(task)
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb_lin")
    assert t.lin_selector("e2e1emb_lin", None, len(idx), task.T, idx.device).all()
    got = t.eval_batch_logits(model, task, "e2e1emb_lin", idx, y, 10., 10.,
                              task.chunk, t.GradStats())
    assert got.shape == (*y.shape, task.V)


def test_all_r42l_arms_share_initialization_and_stream():
    dev = torch.device("cpu")
    st, task = small_echo(batch=2)
    anchor = t.build_model(task, st, dev, 0, "e2e1emb").state_dict()
    hashes = []
    for arm in R42L_ARMS:
        arm_task = t._task(st, "echo_ctrl") if arm == "e2e1emb_ctrl" else task
        state = t.build_model(arm_task, st, dev, 0, arm).state_dict()
        assert all(torch.equal(v, state[k]) for k, v in anchor.items())
        assert arm not in t.SECOND_ORDER_ARMS and arm not in t.FACTOR_ARMS
        assert arm not in t.MASK_ARMS
        if arm != "e2e1emb_ctrl":
            rng = np.random.default_rng([0, t.TRAIN_SEED])
            hashes.append([task.raw(rng, 2)[0].tobytes() for _ in range(3)])
    assert all(stream == hashes[0] for stream in hashes)


def test_analysis_named_smoke_uses_its_logged_layout(tmp_path):
    module = analysis()
    path = synthetic_log(tmp_path / "smoke.log", seeds=range(2))
    config = dict(process_name="TTM-r42L-00-smoke", device="cpu", workers=1,
                  steps=30, T=128, n_keys=4, warmup=4, probe_every=10)
    lines = ["config: " + json.dumps({"ttt": config})]
    for line in path.read_text().splitlines():
        if line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            c["lin_per_seq"] = module.want_lin_per_seq(c["arm"], 128, 4)
            c["warmup"] = 600 if c["arm"] == module.W600 else 4
            line = "CELL_RESULT " + json.dumps(c)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    result = module.summarize(*module.records(path))
    assert result["validation_only"] and not result["issues"]
    assert all(v == 2 for v in result["n_pairs"].values())
    assert result["labels"]["LK"] == "pending"
