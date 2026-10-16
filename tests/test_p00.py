"""P0.0: the plain lin arm on icl_mc, its config and the non-inferiority gate."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("e2e1", "e2e1_lin", "e2e2")


def analysis():
    path = ROOT / "scripts" / "analyze_p00.py"
    spec = importlib.util.spec_from_file_location("analyze_p00", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_p00"] = module
    spec.loader.exec_module(module)
    return module


def tiny():
    st = t._settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
                          window=8, chunk=4))
    return st, t.Task("icl_mc", 8, 16, 8, 4, 5, 4, 2)


def test_arm_tables():
    assert "e2e1_lin" in t.SEQUENTIAL_ARMS and "e2e1_lin" in t.TTT_ARMS
    assert t.LIN_ARMS["e2e1_lin"] == "all" and t.arm_inner_loss("e2e1_lin") == "lin"
    others = t.EMB_ARMS | t.ABLATION_ARMS | set(t.R42L_ARMS) | t.SECOND_ORDER_ARMS
    assert "e2e1_lin" not in others and "e2e1_lin" not in t.MASK_ARMS


def test_plain_model_and_selector_need_no_metadata():
    st, task = tiny()
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1_lin")
    assert model.frozen_emb is None
    # same seed, same initial weights as the CE arm: the inner loss is the only difference
    ref = t.build_model(task, st, torch.device("cpu"), 0, "e2e1")
    for a, b in zip(model.state_dict().values(), ref.state_dict().values()):
        assert torch.equal(a, b)
    sel = t.lin_selector("e2e1_lin", None, 3, task.T, torch.device("cpu"))
    assert sel.shape == (3, task.T) and bool(sel.all())
    assert t.lin_count("e2e1_lin", task.T, 16) == task.T


def test_lin_arm_inner_gradient_is_the_linear_loss():
    """The chunk-0 fast gradient of e2e1_lin equals that of an explicit lin token loss."""
    st, task = tiny()
    torch.manual_seed(0)
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1_lin")
    rng = np.random.default_rng(3)
    idx, y, _ = task.tensors(*task.raw(rng, 4), torch.device("cpu"))
    h = model.backbone(idx).detach()
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    z = model.read(t.fast_block(h[:, :task.chunk], model.head.ln, W))
    lin = t.lin_token(z, y[:, :task.chunk]).mean(1).sum()
    g_lin = torch.autograd.grad(lin, W)
    zero = torch.zeros_like(z, requires_grad=True)
    g_logit = torch.autograd.grad(t.lin_token(zero, y[:, :task.chunk]).sum(), zero)[0]
    want = torch.full_like(g_logit, 1 / task.V) - torch.nn.functional.one_hot(
        y[:, :task.chunk], task.V).to(g_logit.dtype)
    assert torch.allclose(g_logit, want, atol=1e-7)
    # the training path with the all-ones selector gives the same fast gradient
    sel = t.lin_selector("e2e1_lin", None, len(idx), task.T, torch.device("cpu"))
    W2 = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    z2 = model.read(t.fast_block(h[:, :task.chunk], model.head.ln, W2))
    ce = t.token_ce(z2, y[:, :task.chunk])
    loss = t.token_loss(sel[:, :task.chunk], t.lin_token(z2, y[:, :task.chunk]), ce).mean(1).sum()
    for a, b in zip(g_lin, torch.autograd.grad(loss, W2)):
        assert torch.allclose(a, b, atol=1e-6)


def test_lin_arm_trains_and_evaluates_on_icl_mc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run_cells saves checkpoints under ./checkpoints
    st, task = tiny()
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2, window=8,
               chunk=4, workers=1, quick_every=0, tf32=False,
               cells=[dict(task="icl_mc", seed=0, arm=a, eta_base=0.3, diag=False, steps=3)
                      for a in ARMS],
               icl_mc=dict(T=16, n_vocab=8, steps=3, n_eval=4, n_val=4, n_quick_eval=4,
                           eval_batch=4))
    out = t.run_cells(cfg, t._settings(cfg), torch.device("cpu"))
    recs = {r["arm"]: r for r in out["cells"].values()}
    assert set(recs) == set(ARMS)
    for arm in ARMS:
        assert np.isfinite(recs[arm]["eval"]["ce"]) and np.isfinite(recs[arm]["eval"]["ce_late"])
    assert recs["e2e1_lin"]["inner_loss"] == "lin" and recs["e2e1"]["inner_loss"] == "ce"
    assert recs["e2e1_lin"]["lin_per_seq"] == 16


def test_config_pairs_every_seed():
    with (ROOT / "configs/toy/p0-0.toml").open("rb") as f:
        cfg = tomllib.load(f)["ttt"]
    cells = [(c["seed"], c["arm"]) for c in cfg["cells"]]
    assert cells == [(s, a) for s in range(6) for a in ARMS]
    assert all(c["task"] == "icl_mc" and c["eta_base"] == 0.3 and c["diag"] is False
               for c in cfg["cells"])
    st = t._settings(cfg, "icl_mc")
    task = t._task(st, "icl_mc")
    assert (task.T, task.V, task.steps, task.n_eval) == (1024, 32, 4000, 512)
    assert st.lr_total == 0 and st.warmup == 200 and st.tf32 is False
    assert st.process_name == "TTM-p00-02"
    for name in ("configs/toy/smoke/p0-0-gpu.toml", "configs/toy/smoke/p0-0-cpu.toml"):
        with (ROOT / name).open("rb") as f:
            smoke = tomllib.load(f)["ttt"]
        assert {c["arm"] for c in smoke["cells"]} == set(ARMS)


def record(arm, seed, ce, late, n=8, shift=0.0):
    rng = np.random.default_rng(seed)
    per = (late + shift + rng.normal(0, 1e-3, n)).tolist()
    return {"arm": arm, "seed": seed, "task": "icl_mc", "inner_loss": "lin" if "lin" in arm else "ce",
            "eval": {"ce": ce, "ce_late": late, "late_per_seq": per}, "ms_step": 1.0,
            "eta_final": [1, 0, 0, 0]}


@pytest.mark.parametrize("delta,expect", [(0.001, "PASS"), (0.03, "FAIL")])
def test_gate(tmp_path, delta, expect):
    a = analysis()
    lines = []
    for s in range(6):
        base = 2.02 + 0.002 * s
        for arm, d in (("e2e1", 0.0), ("e2e1_lin", delta + 0.0005 * (s % 2)), ("e2e2", -0.03)):
            lines.append("CELL_RESULT " + json.dumps(record(arm, s, base + 0.2 + d, base + d)))
    log = tmp_path / "run.log"
    log.write_text("\n".join(lines) + "\n")
    res = a.analyze(a.parse(log))
    assert res["gate"]["verdict"] == expect
    assert res["n_paired"] == 6
    assert res["sanity"]["e2e1_s0_late"] == pytest.approx(2.02 + 0.0, abs=1e-9)
