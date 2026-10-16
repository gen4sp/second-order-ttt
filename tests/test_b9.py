"""Seed-0 tensors and the real echo layout; no fresh-seed training."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tomllib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tttmem import exp_ttt as t, exp_frsb as f, exp_b9 as driver
from tttmem.aux_b9 import auxiliary, key_pairs, sham_pairs, dot_logits, loss_from_u, B9_BASE

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-b9-71-tests"))
ROOT = Path(__file__).resolve().parents[1]


def tiny():
    st = t._settings(dict(d_model=8, n_heads=2, n_layers=1, backbone_mult=1,
                         fast_mult=1, T=24, window=8, chunk=8, batch=1, n_eval=1,
                         n_quick_eval=1, eval_batch=1, steps=2, tf32=False))
    task = t._task(st, "icl_mc")
    return st, task, t.build_model(task, st, torch.device("cpu"), 0, "e2e1").double()


def test_finite_difference_both_endpoints_and_trained_affine_ln():
    torch.manual_seed(0)
    h = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([.4, 1.2, 2., .7], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([.7, -.8, .2, 1.], dtype=torch.float64, requires_grad=True)
    pairs = key_pairs(dict(write_pos=np.array([[0, 1]]), query_pos=np.array([[3, 4]])), torch.device("cpu"))
    def term(h, w, b):
        return loss_from_u(F.layer_norm(h, (4,), w, b), pairs)[0]
    assert torch.autograd.gradcheck(term, (h, weight, bias), eps=1e-6, atol=1e-5, rtol=1e-4)
    grad = torch.autograd.grad(term(h, weight, bias), h)[0]
    assert grad[:, :2].abs().sum() > 0 and grad[:, 3:].abs().sum() > 0
    assert grad[:, 2].abs().sum() == 0


def test_dot_argmax_identity_nonuniform_norms_and_cosine_disagreement():
    u = torch.tensor([[[.9, .1], [2., 2.], [1., 0.], [0., 1.]]], dtype=torch.float64)
    pairs = key_pairs(dict(write_pos=np.array([[0, 1]]), query_pos=np.array([[2, 3]])), u.device)
    logits = dot_logits(u, pairs[0], pairs[1])
    direct = torch.bmm(u[:, 2:], u[:, :2].transpose(1, 2))
    cosine = torch.bmm(F.normalize(u[:, 2:], dim=-1), F.normalize(u[:, :2], dim=-1).transpose(1, 2))
    assert torch.equal(logits.argmax(-1), direct.argmax(-1))
    assert logits[0, 0].argmax() == 1 and cosine[0, 0].argmax() == 0


def real_layout(n=256):
    raw, meta = t.echo_batch(np.random.default_rng([0, t.TRAIN_SEED]), n, 512, 64, "echo", 16, 4)
    return torch.as_tensor(raw[:, :512]), meta


def test_sham_v2_on_real_layout_is_nonempty_deterministic_and_lag_matched():
    idx, meta = real_layout()
    a, b = sham_pairs(idx, 16, 64), sham_pairs(idx, 16, 64)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    write, query, _, valid = a
    assert bool(valid.any(1).all())
    assert float(valid.sum(1).float().median()) >= 12
    assert bool((write[valid] // 64 == 1).all())
    assert bool(((query[valid] // 64 == 5) | (query[valid] // 64 == 6)).all())
    target_tokens = idx.gather(1, query)
    assert torch.equal(target_tokens[valid], idx.gather(1, write)[valid])
    key_lag = torch.as_tensor(meta["query_pos"]-meta["write_pos"]).float().median()
    assert float((query-write)[valid].float().median()/key_lag) >= .6


def test_aux_dot_metric_equals_actual_r4_addr_h_trained_affine():
    cfg = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    st, dev = t._settings(cfg), torch.device("cpu")
    task = t._task(st, "echo")
    model = t.build_model(task, st, dev, 0, "e2e1_b9")
    with torch.no_grad():
        model.head.ln.weight.copy_(torch.linspace(.2, 2., st.d_model))
        model.head.ln.bias.copy_(torch.linspace(-.8, .7, st.d_model))
    raw, meta = task.raw(np.random.default_rng([0, t.TRAIN_SEED]), 2)
    idx, y, meta = task.tensors(raw, meta, dev)
    h = model.backbone(idx)
    _, metric = loss_from_u(model.head.ln(h), key_pairs(meta, dev))
    probe = t.r4_probe(model, task, [(idx, y, meta)], 10., 10., arm="e2e1_b9")
    assert metric["dot_addr"] == pytest.approx(probe["addr_h"], abs=1e-7)


@pytest.mark.parametrize("arm", list(B9_BASE))
def test_outer_loss_includes_fixed_aux_and_no_frozen_code(arm):
    cfg = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    st, dev = t._settings(cfg), torch.device("cpu")
    task = t._task(st, "echo")
    model = t.build_model(task, st, dev, 0, arm)
    assert model.frozen_emb is None
    idx, y, meta = task.tensors(*task.raw(np.random.default_rng([0, t.TRAIN_SEED]), 1), dev)
    sel = t.lin_selector(arm, meta["write_pos"], 1, task.T, dev)
    total = t.arm_loss(model, arm, idx, y, 10., 10., task.chunk, t.GradStats(), sel=sel, meta=meta)
    base = t.arm_loss(model, B9_BASE[arm], idx, y, 10., 10., task.chunk, t.GradStats(), sel=sel, meta=meta)
    aux = auxiliary(model, model.backbone(idx), idx, meta, task.chunk, sham=arm.endswith("b9f"))
    torch.testing.assert_close(total, base+.1*aux, atol=0, rtol=0)
    assert all(k in model.b9_last_metrics for k in ("loss", "grad_norm", "anchor_count", "lag", "grad_ratio", "lag_ratio"))


def test_logging_every_training_step(tmp_path, monkeypatch, capsys):
    cfg = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    st, dev = t._settings(cfg), torch.device("cpu")
    st = t.dataclasses.replace(st, probe_every=0, checkpoint_steps=[], quick_every=10)
    task = t._task(st, "echo")
    model = t.build_model(task, st, dev, 0, "e2e1_lin_b9f")
    monkeypatch.chdir(tmp_path)
    out = t.train_arm(model, task, "e2e1_lin_b9f", 10., st, dev,
                      np.random.default_rng([0, t.TRAIN_SEED]), None, "fixture", 2, seed=0)
    assert [q["step"] for q in out["aux_log"]] == [1, 2]
    assert all(q["sham_anchor_count"] > 0 and q["grad_norm"] > 0 for q in out["aux_log"])
    assert capsys.readouterr().out.count("AUX_B9 ") == 1  # final step; files retain both


def test_auxiliary_ignores_value_labels_and_reference_grads_do_not_pollute_grad():
    cfg = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    st, dev = t._settings(cfg), torch.device("cpu")
    task = t._task(st, "echo")
    model = t.build_model(task, st, dev, 0, "e2e1_b9")
    idx, _, meta = task.tensors(*task.raw(np.random.default_rng([0, t.TRAIN_SEED]), 1), dev)
    h = model.backbone(idx)
    first = auxiliary(model, h, idx, meta, task.chunk, log_grad=True)
    assert all(p.grad is None for p in model.parameters())
    changed = dict(meta, query_tok=np.zeros_like(meta["query_tok"]))
    second = auxiliary(model, h, idx, changed, task.chunk)
    assert torch.equal(first, second)


def test_nontriggering_parent_never_calls_training(monkeypatch):
    cfg = tomllib.loads((ROOT/"configs/toy/b9.toml").read_text())["b9"]
    monkeypatch.setattr(f, "execution_guard", lambda *a, **kw: "synthetic seal")
    monkeypatch.setattr(driver, "parent_check", lambda cfg: ({}, dict(run_b9=False, reason="NOT EVALUABLE")))
    monkeypatch.setattr(f, "prepare", lambda *a, **kw: pytest.fail("must not prepare a B9 run"))
    assert driver.run(cfg)["skipped"]


def test_sealed_b9_config_validation_is_read_only():
    cfg = tomllib.loads((ROOT/"configs/toy/b9.toml").read_text())["b9"]
    f.validate(cfg, b9=True)
    assert len(cfg["cells"]) == 36
    smoke = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    assert all(c["seed"] == 0 for c in smoke["cells"])


def test_production_b9_fails_closed_without_seal(tmp_path, monkeypatch):
    cfg = tomllib.loads((ROOT/"configs/toy/b9.toml").read_text())["b9"]
    cfg["seal_file"] = str(tmp_path/"absent.json")
    cfg["require_seal"] = True
    monkeypatch.setattr(driver, "parent_check", lambda *a: pytest.fail("must guard first"))
    with pytest.raises(ValueError, match="combined seal"):
        driver.run(cfg)


def test_parent_result_absolute_path_is_independent_of_cwd(tmp_path, monkeypatch):
    from test_analyze_frsb import payload
    cfg = tomllib.loads((ROOT/"configs/toy/smoke/b9.toml").read_text())["b9"]
    p = payload()
    p["manifest"].update(smoke=True, seeds=cfg["seeds"], plan_sha256=None)
    path = tmp_path/"persistent-result.json"
    path.write_text(json.dumps(p))
    cfg["frsb_result_path"] = str(path)
    monkeypatch.chdir(tmp_path)
    assert driver.parent_check(cfg)[1]["reason"] == "smoke-only"
    production = tomllib.loads((ROOT/"configs/toy/b9.toml").read_text())["b9"]
    assert production["frsb_result_path"] == "outputs/frsb-output/result.json"


