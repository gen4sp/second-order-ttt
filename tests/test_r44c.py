"""R4-4c checks: CPU only, no spawned torch process or experiment launch."""
import copy
import csv
import importlib.util
import json
import math
import queue
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tttmem import exp_c1 as c1
from tttmem import exp_ttt as t
from tttmem import run as entrypoint

ROOT = Path(__file__).resolve().parents[1]
FULL = tomllib.loads((ROOT / "configs/toy/r4-4c.toml").read_text())["c1"]  # full c1 config
CPU = torch.device("cpu")
torch.set_num_threads(2)
needs_checkpoints = pytest.mark.skipif(
    not (ROOT / FULL["checkpoint_dir"]).is_dir(),
    reason="needs the R4-3 final checkpoints in checkpoint_dir (outputs/r4-3/checkpoints)")


def local_config(**overrides):
    return {**FULL, "device": "cpu", "workers": 1,
            "checkpoint_dir": str(ROOT / FULL["checkpoint_dir"]), **overrides}


def one_cell(init="e2e2emb", rule="e2e2emb", mode="floor", seed=0):
    return next(copy.deepcopy(c) for c in FULL["cells"] if
                (c["init"], c["rule"], c["lr_mode"], c["seed"]) == (init, rule, mode, seed))


def analyzer():
    spec = importlib.util.spec_from_file_location("analyze_r44c", ROOT / "scripts/analyze_r44c.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_checkpoints
@pytest.mark.parametrize("init", ["e2e2emb", "e2e2"])
def test_real_strict_load_step0_and_rho(init, capsys):
    cfg = local_config(steps=3, n_quick_eval=4, eval_batch=4)
    cell = one_cell(init, init)
    model, task, st = c1.load_initial(cfg, cell, CPU)
    state = torch.load(c1.checkpoint_path(cfg, cell), weights_only=True, map_location="cpu")
    assert set(model.state_dict()) == set(state)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key]), key
    assert torch.equal(model.rho.detach(), state["rho"])
    assert torch.equal(torch.stack(model.etas(10.)), 10. * state["rho"].exp())
    quick = t.eval_batches(task, *t.fixed_set(task, st, "quick"), CPU, 4)
    record = c1.step0_check(model, task, st, cell, quick)
    assert record == dict(acc=1., ref=1., match=True, exact=True, source_rule=init)
    text = capsys.readouterr().out
    assert "C1_STEP0 " in text and "match=True" in text
    # Printed when the test runner uses -s, so validation logs retain the evidence.
    print(f"TEST_STEP0 init={init} seed=0 subset=4 acc=1 ref=1 match=True")
    assert t.build_optim(model, st).state == {}
    bad = dict(state, unknown_parameter=torch.zeros(1))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(bad, strict=True)


@needs_checkpoints
@pytest.mark.parametrize("cell", [c for c in FULL["cells"] if
                         c["rule"] == c["init"] and c["lr_mode"] == "floor"],
                         ids=lambda c: f"{c['init']}-s{c['seed']}")
def test_all_19_source_full_quick_matches(cell):
    cfg = local_config(steps=3)
    model, task, st = c1.load_initial(cfg, cell, CPU)
    quick = t.eval_batches(task, *t.fixed_set(task, st, "quick"), CPU, st.eval_batch)
    record = c1.step0_check(model, task, st, cell, quick)
    assert record["match"]


@needs_checkpoints
def test_gated_install_bitwise_and_effect():
    cfg = local_config(steps=3, n_quick_eval=2, eval_batch=2)
    cell = one_cell("e2e2", "e2e1_code")
    model, task, st = c1.load_initial(cfg, cell, CPU)
    model.eval()
    idx, y, meta = t.eval_batches(task, *t.fixed_set(task, st, "quick"), CPU, 2)[0]
    def logits():
        return t.eval_batch_logits(model, task, "e2e1", idx, y, 10., st.g_clip,
                                   task.chunk, t.GradStats(), meta=meta)
    original = logits()
    t.install_key_code(model, cell["seed"])
    assert torch.equal(original, logits())  # complete streaming logits, not just W0
    code, _ = t.frozen_code(task.V, st.d_model, cell["seed"])
    assert torch.equal(model.frozen_emb, code)
    assert not model.frozen_emb.requires_grad
    assert model.code_gate.ndim == 0 and model.code_gate.item() == 0.
    opt = t.build_optim(model, st)
    assert opt.state == {}
    group = next(g for g in opt.param_groups if any(p is model.code_gate for p in g["params"]))
    assert group["base_lr"] == st.gamma_lr and group["weight_decay"] == 0.
    assert len(group["params"]) == 1
    assert sum(p is model.code_gate for g in opt.param_groups for p in g["params"]) == 1
    assert any(p is model.code_gate for p in t.clipped_params(model))
    assert not any(p is model.rho for p in t.clipped_params(model))
    model.train()
    loss = t.arm_loss(model, "e2e1_code", idx, y, 10., st.g_clip, task.chunk, t.GradStats())
    loss.backward()
    assert model.code_gate.grad is not None and torch.isfinite(model.code_gate.grad)
    assert model.code_gate.grad.item() != 0.
    with torch.no_grad():
        model.code_gate.fill_(.2)
    assert not torch.equal(original, logits())
    restored = t.build_model(task, st, CPU, cell["seed"], "e2e1_code")
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.code_gate, model.code_gate)
    assert torch.equal(restored.frozen_emb, model.frozen_emb)
    with pytest.raises(ValueError, match="plain checkpoint"):
        t.install_key_code(model, cell["seed"])


@needs_checkpoints
def test_schedules_and_eta_schedule():
    assert all(1e-3 * c1.schedule_multiplier("floor", s) == 1e-4 for s in range(376))
    for step, expected in [(0, 1e-4), (25, 1e-3), (375, 1e-4)]:
        assert 1e-3 * c1.schedule_multiplier("rewarm", step) == pytest.approx(expected)
    cfg = local_config(steps=3)
    cell = one_cell(mode="rewarm")
    model, _, st = c1.load_initial(cfg, cell, CPU)
    opt = t.build_optim(model, st)
    for step in [0, 1, 25, 100, 200, 375]:
        c1.set_lrs(opt, cfg, cell, step)
        assert opt.param_groups[-1]["lr"] == pytest.approx(opt.param_groups[0]["lr"] * 10)
    assert not opt.state


@needs_checkpoints
@pytest.mark.parametrize("mode", ["floor", "rewarm"])
def test_gamma_own_group_and_shared_schedule(mode):
    cfg = local_config(steps=3, gamma_lr=.023)
    cell = one_cell("e2e2", "e2e1_code", mode)
    model, _, st = c1.load_initial(cfg, cell, CPU)
    t.install_key_code(model, 0)
    opt = t.build_optim(model, st)
    for step in (0, 20, 25, 40, 60, 375):
        c1.set_lrs(opt, cfg, cell, step)
        scale = c1.schedule_multiplier(mode, step)
        assert opt.param_groups[-2]["lr"] == pytest.approx(.023 * scale)
        assert opt.param_groups[-1]["lr"] == pytest.approx(.01 * scale)
        assert opt.param_groups[0]["lr"] == pytest.approx(.001 * scale)


@needs_checkpoints
@pytest.mark.parametrize("init,rule", [(init, rule) for init, rules in c1.RULES.items() for rule in rules])
def test_three_step_cell_each_rule(init, rule, tmp_path, monkeypatch, capsys):
    cfg = local_config(steps=3, batch=1, n_quick_eval=4, n_eval=4, eval_batch=4,
                       n_probe=2, quick_every=1, probe_every=1, checkpoint_steps=[1, 2])
    cell = one_cell(init, rule, "rewarm")
    cfg["cells"] = [cell]
    monkeypatch.chdir(tmp_path)
    c1.validate(cfg)
    record = c1.run_cell(cfg, cell, CPU)
    assert len(record["losses"]) == 3 and all(math.isfinite(x) for x in record["losses"])
    assert record["step0"]["match"] and record["optimizer_fresh"]
    assert [r[0] for r in record["quick"]] == [0, 1, 2, 3]
    assert len(record["step_checkpoints"]) == 2 and Path(record["checkpoint"]).is_file()
    assert record["ablation_ref_match"]
    assert record["final_eval_ok"]
    assert record["data_seed"] == 50000
    assert all("gamma" in p and p["lr"] > 0 and p["eta_lr"] > 0 for p in record["probes"])
    if rule == "e2e1_code":
        assert record["probes"][0]["gamma"] == cfg["gamma0"] == .5
        assert record["gamma_final"] != .5
        assert record["probes"][0]["gamma_lr"] == pytest.approx(.001)
    text = capsys.readouterr().out
    assert sum(line.startswith("CELL_RESULT ") for line in text.splitlines()) == 1
    assert "match=True" in text
    emitted = json.loads(next(line[12:] for line in text.splitlines() if line.startswith("CELL_RESULT ")))
    assert emitted["rule"] == rule


@needs_checkpoints
def test_step0_mismatch_fails_and_missing_checkpoint(tmp_path, capsys):
    cfg = local_config(steps=3, n_quick_eval=2, eval_batch=2)
    cell = one_cell()
    model, task, st = c1.load_initial(cfg, cell, CPU)
    cell["reference_quick_acc"] = 0.
    with pytest.raises(AssertionError, match="step-0 mismatch"):
        c1.step0_check(model, task, st, cell, t.eval_batches(
            task, *t.fixed_set(task, st, "quick"), CPU, 2))
    assert "match=False" in capsys.readouterr().out
    cfg["checkpoint_dir"] = str(tmp_path)
    with pytest.raises(FileNotFoundError):
        c1.load_initial(cfg, cell, CPU)


def test_configs_and_guards(monkeypatch):
    assert entrypoint.EXPERIMENTS["c1"] == "tttmem.exp_c1"
    assert len(FULL["cells"]) == 132 and len({c1.cell_id(c) for c in FULL["cells"]}) == 132
    assert len({c1.checkpoint_path(FULL, c) for c in FULL["cells"]}) == 19
    assert FULL["process_name"] == "TTM-r44c-02" and FULL["workers"] == 6
    smoke = tomllib.loads((ROOT / "configs/toy/smoke/r4-4c-gpu.toml").read_text())["c1"]
    assert len(smoke["cells"]) == 6 and smoke["steps"] == 20 and smoke["workers"] == 6
    assert smoke["checkpoint_steps"] == [10]
    assert FULL["max_minutes"] == 75 and smoke["max_minutes"] == 15
    assert FULL["gamma0"] == smoke["gamma0"] == .5
    assert FULL["gamma_lr"] == smoke["gamma_lr"] == .01
    assert {c["lr_mode"] for c in smoke["cells"]} == {"floor", "rewarm"}
    assert {c["rule"] for c in smoke["cells"]} == {"e2e2emb", "e2e1emb", "e2e1emb_lin", "e2e1_code", "e2e2"}
    for init, seed in [("e2e2emb", 8), ("e2e2", 15)]:
        assert dict(seed=seed, init=init, rule=init, lr_mode="floor",
                    reference_quick_acc=.999023) in smoke["cells"]
    cfg = local_config(steps=3)
    for changes in [dict(device="mps"), dict(workers=2), dict(steps=61), dict(tf32=True),
                    dict(eval_seed=123), dict(data_seed_offset=0), dict(max_minutes=0),
                    dict(max_minutes=float("inf")), dict(gamma0=0), dict(gamma_lr=0)]:
        with pytest.raises(ValueError):
            c1.validate(dict(cfg, **changes))
    # Validate CUDA pool configuration without creating workers or allocating tensors.
    monkeypatch.setattr(c1.sys, "platform", "linux")
    if not (ROOT / FULL["checkpoint_dir"]).is_dir():
        pytest.skip("needs the R4-3 final checkpoints in checkpoint_dir (outputs/r4-3/checkpoints)")
    assert c1.validate(FULL).type == "cuda"


def synthetic_log(path):
    lines = []
    for cell in FULL["cells"]:
        init, rule, mode, seed = (cell[k] for k in ("init", "rule", "lr_mode", "seed"))
        retained = (rule in ("e2e2emb", "e2e2", "e2e1emb_lin", "e2e1_lin") or
                    rule == "e2e1_code" and seed in (0,5,6,7,8,11) or
                    rule == "e2e1emb" and mode == "rewarm" and seed < 4)
        ident = c1.cell_id(cell)
        lines.append(f"C1_STEP0 cell={ident} acc=1 ref=1 match=True exact=True")
        acc = 1. if retained else .2
        row = dict(cell_id=ident, init=init, rule=rule, lr_mode=mode, seed=seed,
                   steps=375, bound_final=retained, bound_ever=True, min_acc=acc,
                   final_quick_acc=acc, final_acc=acc, t_lose=None if retained else 100,
                   final_eval_ok=True, gamma_final=.1 if rule == "e2e1_code" else None,
                   quick=[[0, .1, .1, 1., None, None, 1, 1, 1, 1],
                          [100, .1, .1, acc, .1, None, 1, 1, 1, 1],
                          [375, .1, .1, acc, .1, None, 1, 1, 1, 1]],
                   ablation=dict(all=acc, kv_only=acc, all_minus_kv=.2))
        lines.append("C1_QUICK " + json.dumps(dict(cell_id=ident, step=375, query_acc=acc,
                    lr=1e-4, gamma=.1 if rule == "e2e1_code" else None)))
        lines.append("CELL_RESULT " + json.dumps(row))
    path.write_text("\n".join(lines)+"\n")


def test_synthetic_analyzer_and_policy(tmp_path, capsys):
    a = analyzer()
    log = tmp_path / "synthetic.log"
    synthetic_log(log)
    summary = a.analyze(log, tmp_path / "out")
    assert summary["status"] == "complete" and summary["step0_matches"] == 132
    assert summary["chosen_lr_mode"] == "floor" and summary["p21_add_ce_code"] is True
    assert summary["p21_toy_equivalent_tokens"] == 100 * 32 * 512
    assert len(summary["headline_tests"]) == 15
    c1floor = next(r for r in summary["headline_tests"] if r["test"] == "C1_floor")
    assert c1floor["a_only"] == 10 and c1floor["p_exact"] == 1 / 1024
    assert c1floor["p_holm"] == pytest.approx(15 / 1024)
    assert c1floor["reject_holm_05"]
    assert a.exact_mcnemar(9,0,"greater") == 1 / 512
    assert a.exact_mcnemar(10,0,"two-sided") == 2 / 1024
    assert a.exact_mcnemar(0,0,"greater") == 1.
    for name in ["cells", "quick", "step0", "c1", "c2", "c3", "c4", "token-budget", "gamma"]:
        assert (tmp_path / "out/data" / f"r44c-{name}.csv").is_file()
    assert "R44C_SUMMARY " in capsys.readouterr().out
    # A prefix/smoke cannot silently set the design or make headline comparisons.
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:3])+"\n")
    pending = a.analyze(log, tmp_path / "partial")
    assert pending["status"] == "incomplete_or_smoke"
    assert pending["chosen_lr_mode"] is None and pending["p21_add_ce_code"] is None
    assert all(r["p_exact"] is None for r in pending["headline_tests"])


def test_analyzer_rejects_step0_failure(tmp_path):
    path = tmp_path / "bad.log"
    path.write_text("C1_STEP0 cell=bad acc=0 ref=1 match=False exact=False\n")
    with pytest.raises(ValueError, match="step-0 mismatch"):
        analyzer().analyze(path, tmp_path / "out")


@pytest.mark.parametrize("variant,expected_mode,expected_code", [
    ("tie", "floor", True), ("floor_control_loses", "rewarm", True),
    ("both_controls_lose", None, None), ("five_code_retainers", "floor", False)])
def test_analyzer_decision_boundaries(tmp_path, variant, expected_mode, expected_code):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)
    lines = []
    # Change a complete synthetic experiment; drop redundant quick stdout rows
    # so the analyzer recovers the modified trajectory from CELL_RESULT.
    for line in log.read_text().splitlines():
        if line.startswith("C1_QUICK "):
            continue
        if line.startswith("CELL_RESULT "):
            row = json.loads(line[12:])
            change = (variant == "tie" and row["rule"] == "e2e1emb" or
                      variant == "floor_control_loses" and row["rule"] == "e2e2emb" and
                      row["lr_mode"] == "floor" and row["seed"] == 0 or
                      variant == "both_controls_lose" and row["rule"] == "e2e2emb" and row["seed"] == 0 or
                      variant == "five_code_retainers" and row["rule"] == "e2e1_code" and row["seed"] == 11)
            if change:
                row.update(bound_final=False, min_acc=.2, final_quick_acc=.2, t_lose=100)
                row["quick"][1][3] = row["quick"][-1][3] = .2
            line = "CELL_RESULT " + json.dumps(row)
        lines.append(line)
    log.write_text("\n".join(lines)+"\n")
    result = analyzer().analyze(log, tmp_path / "out")
    assert result["chosen_lr_mode"] == expected_mode
    assert result["p21_add_ce_code"] is expected_code
    if expected_mode is None:
        assert result["lr_choice"] == "inconclusive"


def test_analyzer_detects_inconsistent_loss(tmp_path):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)
    log.write_text(log.read_text().replace('"t_lose": 100', '"t_lose": 25', 1))
    with pytest.raises(ValueError, match="inconsistent t_lose"):
        analyzer().analyze(log, tmp_path / "out")


@pytest.mark.parametrize("queries,exact,passes", [(0, True, True), (1, False, True), (2, False, False)])
def test_step0_one_query_tolerance_and_analyzer(tmp_path, monkeypatch, capsys, queries, exact, passes):
    cell = one_cell()
    acc = 1 - queries / 1024
    monkeypatch.setattr(t, "quick_summary", lambda *a: {"query_acc": acc})
    if passes:
        record = c1.step0_check(None, None, None, cell, None)
        assert record["match"] and record["exact"] is exact
    else:
        with pytest.raises(AssertionError, match="step-0 mismatch"):
            c1.step0_check(None, None, None, cell, None)
    line = capsys.readouterr().out
    assert f"exact={exact}" in line
    path = tmp_path / "step0.log"
    path.write_text(line)
    if passes:
        _, checks, _, _ = analyzer().records(path)
        assert checks[c1.cell_id(cell)]["exact"] is exact
    else:
        with pytest.raises(ValueError, match="step-0 mismatch"):
            analyzer().records(path)


@needs_checkpoints
@pytest.mark.parametrize("failure", ["evaluate", "ablation", "mismatch"])
def test_final_eval_failure_keeps_checkpoint_and_cell(tmp_path, monkeypatch, capsys, failure):
    cfg = local_config(steps=1, batch=1, n_quick_eval=2, n_eval=2, eval_batch=2,
                       n_probe=2, quick_every=1, probe_every=1, checkpoint_steps=[])
    cell = one_cell()
    monkeypatch.chdir(tmp_path)
    original_eval, original_ablation = t.evaluate, t.r42_ablation

    def evaluate(*args, **kwargs):
        assert list(tmp_path.rglob(f"{c1.cell_id(cell)}__step1.pt")), "checkpoint must precede final eval"
        if failure == "evaluate":
            raise RuntimeError("injected final eval error")
        return original_eval(*args, **kwargs)

    def ablation(*args, **kwargs):
        if failure == "ablation":
            raise RuntimeError("injected ablation error")
        row = original_ablation(*args, **kwargs)
        row["all"] = .25
        return row

    monkeypatch.setattr(t, "evaluate", evaluate)
    monkeypatch.setattr(t, "r42_ablation", ablation)
    result = c1.run_cell(cfg, cell, CPU)
    assert Path(result["checkpoint"]).is_file()
    assert len(result["losses"]) == 1 and result["quick"][-1][0] == 1
    assert result["bound_final"] == (result["final_quick_acc"] >= .9)
    assert result["t_lose"] == next((r[0] for r in result["quick"] if r[3] < .9), None)
    output = capsys.readouterr()
    assert output.out.count("CELL_RESULT ") == 1
    if failure == "mismatch":
        assert result["final_eval_ok"] and not result["ablation_ref_match"]
        assert "ABLATION_MISMATCH " in output.out
    else:
        assert result["final_eval_ok"] is False
        assert all(k not in result for k in analyzer().FINAL_EVAL_FIELDS)
        assert "FINAL_EVAL_ERROR " in output.out and "Traceback" in output.err


def rewrite_synthetic(log, transform):
    lines = []
    for line in log.read_text().splitlines():
        if line.startswith("C1_QUICK "):
            continue  # recover the edited training trajectory from CELL_RESULT
        if line.startswith("CELL_RESULT "):
            row = json.loads(line[12:])
            transform(row)
            line = "CELL_RESULT " + json.dumps(row)
        lines.append(line)
    log.write_text("\n".join(lines) + "\n")


def test_analyzer_keeps_final_eval_failed_cells(tmp_path):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)

    def fail_eval(row):
        row.update(final_eval_ok=False, final_acc=.123, eval={"query_acc": .123},
                   ablation_ref_match=False)  # stale final fields must be dropped

    rewrite_synthetic(log, fail_eval)
    summary = analyzer().analyze(log, tmp_path / "out")
    assert summary["status"] == "complete" and summary["cells"] == 132
    assert summary["chosen_lr_mode"] == "floor" and summary["p21_add_ce_code"]
    assert all(row["status"] == "complete" for row in summary["headline_tests"])
    rows = list(csv.DictReader((tmp_path / "out/data/r44c-cells.csv").open()))
    assert len(rows) == 132 and all(r["final_eval_ok"] == "False" for r in rows)
    assert all(not any(k.startswith("ablation") or k == "final_acc" for k in r) for r in rows)
    assert any(r["t_lose"] == "100" for r in rows)


@pytest.mark.parametrize("no_loss_rules", [("e2e1",), ("e2e1emb",), ("e2e1", "e2e1emb")])
def test_analyzer_censors_budget_if_either_fo_ce_group_never_loses(tmp_path, no_loss_rules):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)

    def change(row):
        if row["rule"] not in ("e2e1emb", "e2e1"):
            return
        retained = row["rule"] in no_loss_rules
        acc = 1. if retained else .2
        row.update(bound_final=retained, min_acc=acc, final_quick_acc=acc,
                   t_lose=None if retained else 50)
        row["quick"][1][0] = 50
        row["quick"][1][3] = row["quick"][-1][3] = acc

    rewrite_synthetic(log, change)
    summary = analyzer().analyze(log, tmp_path / "out")
    assert summary["chosen_lr_mode"] == "floor"
    assert summary["p21_toy_equivalent_tokens"] is None
    assert summary["token_budget_lower_bound"] == 6_144_000
    assert summary["token_budget_status"] == "censored"


@pytest.mark.parametrize("gamma,add_code", [(.099, False), (.1, True), (-.1, True), (None, False)])
def test_code_gate_requires_present_gamma(tmp_path, gamma, add_code):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)

    def change(row):
        if row["rule"] == "e2e1_code":
            row["gamma_final"] = gamma

    rewrite_synthetic(log, change)
    summary = analyzer().analyze(log, tmp_path / "out")
    assert summary["p21_add_ce_code"] is add_code
    assert summary["p21_code_median_abs_gamma375"] == (abs(gamma) if gamma is not None else None)


def test_analyzer_prints_inconclusive_lr_choice(tmp_path, capsys):
    log = tmp_path / "synthetic.log"
    synthetic_log(log)

    def change(row):
        if row["rule"] == "e2e2emb" and row["seed"] == 0:
            row.update(bound_final=False, min_acc=.2, final_quick_acc=.2, t_lose=100)
            row["quick"][1][3] = row["quick"][-1][3] = .2

    rewrite_synthetic(log, change)
    assert analyzer().analyze(log, tmp_path / "out")["lr_choice"] == "inconclusive"
    assert "lr_choice=inconclusive" in capsys.readouterr().out


@pytest.mark.parametrize("variant", ["late_result", "nonzero_exit", "missing", "cell_error",
                                     "deadline_wait", "deadline_join"])
def test_pool_shutdown_and_deadline_without_torch_workers(monkeypatch, capsys, variant):
    cfg = dict(FULL, workers=2, cells=FULL["cells"][:2], max_minutes=.01)
    ids = [c1.cell_id(cell) for cell in cfg["cells"]]
    rows = [(ident, {"cell_id": ident}) for ident in ids]
    if variant == "cell_error":
        rows[1][1]["error"] = "injected"
    calls, processes, timers = [], [], []

    class FakeResults:
        def __init__(self):
            self.rows = list(rows[:1] if variant in ("late_result", "missing") else rows)
            self.late = []

        def get(self, timeout):
            calls.append(("get", timeout))
            if variant == "deadline_wait":
                timers[0].callback()
                raise queue.Empty
            if self.rows:
                return self.rows.pop(0)
            raise queue.Empty

        def get_nowait(self):
            calls.append(("drain",))
            if self.late:
                return self.late.pop(0)
            raise queue.Empty

    results = FakeResults()

    class FakeProcess:
        def __init__(self, **kwargs):
            self.index, self.alive = len(processes), False
            self.exitcode = 1 if variant == "nonzero_exit" else 0
            processes.append(self)

        def start(self):
            self.alive = variant in ("deadline_wait", "deadline_join")

        def is_alive(self):
            return self.alive

        def terminate(self):
            calls.append(("terminate", self.index))
            self.alive, self.exitcode = False, -15

        def join(self, *args, **kwargs):
            calls.append(("join", args, kwargs))
            if variant == "deadline_join" and self.index == 0:
                timers[0].callback()
            if variant == "late_result" and self.index == 1:
                results.late.append(rows[1])
            self.alive = False

    class FakeTimer:
        ident = None

        def __init__(self, duration, callback):
            assert 0 < duration <= .6
            self.callback = callback
            timers.append(self)

        def start(self):
            self.ident = 1

        def cancel(self):
            calls.append(("cancel_timer",))

        def join(self):
            pass

    queues = iter([queue.Queue(), results])
    ctx = SimpleNamespace(Queue=lambda: next(queues), Lock=lambda: None, Process=FakeProcess)
    monkeypatch.setattr(c1.mp, "get_context", lambda method: ctx)
    monkeypatch.setattr(c1.threading, "Timer", FakeTimer)
    if variant.startswith("deadline"):
        with pytest.raises(RuntimeError, match="exceeded max_minutes"):
            c1._run_pool(cfg)
        assert "POOL_DEADLINE " in capsys.readouterr().out
        assert sum(c[0] == "terminate" for c in calls) == 2
    elif variant in ("missing", "cell_error"):
        with pytest.raises(RuntimeError, match="failed"):
            c1._run_pool(cfg)
    else:
        result = c1._run_pool(cfg)
        assert set(result["cells"]) == set(ids)
    joins = [c for c in calls if c[0] == "join"]
    assert len(joins) == 2 and all(c[1] == () and c[2] == {} for c in joins)
    assert any(c[0] == "drain" for c in calls)


def test_analyzer_rejects_pool_deadline(tmp_path):
    log = tmp_path / "deadline.log"
    synthetic_log(log)
    log.write_text(log.read_text() + "POOL_DEADLINE max_minutes=75\n")
    with pytest.raises(ValueError, match="pool deadline"):
        analyzer().analyze(log, tmp_path / "out")
