"""Seed-0 CPU equivalence and sealed protocol fixtures."""
import hashlib
import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t, exp_r44b as wave
from tttmem import exp_frsb as f

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-frb-71-tests"))


def tiny_cfg():
    return tomllib.loads((ROOT/"configs/toy/smoke/frs-b.toml").read_text())["frsb"]


def test_ten_step_cells_driver_equals_wave_b_update_bitwise(tmp_path, monkeypatch):
    cfg = tiny_cfg()
    cfg.update(steps=10, probe_every=0, quick_every=20, checkpoint_steps=[])
    st = t._settings(cfg)
    task = t._task(st, "echo")
    dev = torch.device("cpu")
    a = t.build_model(task, st, dev, 0, "e2e1emb_lin")
    b = t.build_model(task, st, dev, 0, "e2e1emb_lin")
    rng_a, rng_b = [np.random.default_rng([0, t.TRAIN_SEED]) for _ in range(2)]
    monkeypatch.chdir(tmp_path)
    result = t.train_arm(a, task, "e2e1emb_lin", 10., st, dev, rng_a, None, "fixture", 10, seed=0)
    opt = t.build_optim(b, st)
    losses = []
    for step in range(10):
        _, ok, loss, _ = wave.update(b, opt, task, st, rng_b, "e2e1emb_lin", 10., step)
        assert ok
        losses.append(loss)
    assert np.array_equal(result["_trace"]["loss"], np.asarray(losses, dtype=np.float32))
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


def test_configs_are_only_fresh_seed_source_and_no_training():
    # Read-only validation: no RNG initialization or training from this config.
    cfg = tomllib.loads((ROOT/"configs/toy/frs-b.toml").read_text())["frsb"]
    f.validate(cfg)
    assert len(cfg["cells"]) == 84
    for cell in cfg["cells"]:
        assert f.effective(cfg, cell)["eta_lr"] == (.0 if cell["block"] == "B40" else .01)
    assert cfg["n_val"] == 256 and cfg["beta2"] == .95


def test_smoke_rejects_nonzero_seed_before_model_creation():
    cfg = tiny_cfg()
    cfg["cells"][0]["seed"] = 1
    with pytest.raises(ValueError, match="seed 0"):
        f.validate(cfg)


def test_production_rejects_cpu_before_training():
    cfg = tomllib.loads((ROOT/"configs/toy/frs-b.toml").read_text())["frsb"]
    cfg["device"] = "cpu"
    with pytest.raises(ValueError, match="no local production"):
        f.validate(cfg)


def test_cross_eval_retains_frozen_code_and_uses_same_full_set():
    cfg = tiny_cfg()
    st, dev = t._settings(cfg), torch.device("cpu")
    task = t._task(st, "echo")
    model = t.build_model(task, st, dev, 0, "e2e1emb_lin")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = f.frsb_cross_eval(model, task, st, 10.)
    assert out["ce"]["token_sha256"] == out["lin"]["token_sha256"]
    assert out["ce"]["sequences"] == st.n_eval
    assert not out["ce"]["authoritative"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_completed_receipt_reused_and_seal_mismatch_rejected(tmp_path):
    cfg = tiny_cfg()
    cell = cfg["cells"][0]
    path = tmp_path/"cells"/f.cell_key(cell)/"receipt.json"
    path.parent.mkdir(parents=True)
    files = {k: str((path.parent / k).relative_to(tmp_path)) for k in ("checkpoint", "trace")}
    for relative in files.values():
        (tmp_path / relative).write_bytes(b"fixture")
    record = dict(cell, trained=True, **files, file_sha256={p: hashlib.sha256(b"fixture").hexdigest() for p in files.values()})
    f.atomic_json(path, dict(identity="same", attempts=1, record=record))
    assert f.run_one(cfg, cell, tmp_path, dict(identity="same"))["trained"]
    with pytest.raises(ValueError, match="seal mismatch"):
        f.run_one(cfg, cell, tmp_path, dict(identity="different"))
    (tmp_path / files["checkpoint"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="sha256"):
        f.run_one(cfg, cell, tmp_path, dict(identity="same"))


def test_infrastructure_retry_once_and_never_retries_divergence(tmp_path, monkeypatch):
    cfg = tiny_cfg()
    cell = cfg["cells"][0]
    calls = []
    def broken(*args):
        calls.append(1)
        return dict(cells={"a": dict(cell, error="synthetic infrastructure error")})
    monkeypatch.setattr(t, "run_cells", broken)
    r = f.run_one(cfg, cell, tmp_path, dict(identity="same"))
    assert len(calls) == 2 and r["attempts"] == 2
    f.run_one(cfg, cell, tmp_path, dict(identity="same"))
    assert len(calls) == 2


def test_manifest_hashes_registration(tmp_path):
    cfg = tiny_cfg()
    cfg["output_dir"] = str(tmp_path)
    plan = tmp_path/"plan.md"
    plan.write_text("registered plan fixture")
    cfg["plan_path"] = str(plan)
    manifest, _ = f.prepare(cfg)
    assert manifest["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert len(manifest["source_sha256"]) == 64


def test_production_fails_closed_without_seal_before_model_creation(tmp_path, monkeypatch):
    cfg = tomllib.loads((ROOT/"configs/toy/frs-b.toml").read_text())["frsb"]
    cfg["seal_file"] = str(tmp_path/"absent.json")
    cfg["require_seal"] = True
    monkeypatch.setattr(t, "build_model", lambda *a, **kw: pytest.fail("model before seal"))
    with pytest.raises(ValueError, match="combined seal"):
        f.run(cfg)


def test_production_logs_health_only_and_keeps_series_on_disk(tmp_path, capsys):
    with f.training_log(dict(smoke=False), tmp_path):
        print("fixture step 25/1500 train=0.3 quick=0.2 acc=0.99")
        print('PROBE {"addr_h": 0.2}')
        print('CELL_RESULT {"query_acc": 0.99}')
        print('AUX_B9 {"dot_addr": 0.2}')
    public = capsys.readouterr().out
    assert "steps=25/1500 loss_finite=True" in public
    assert all(k not in public for k in ("acc", "addr", "0.99", "PROBE", "AUX_B9"))
    assert "query_acc" in (tmp_path/"train.log").read_text()


def test_receipts_relocate_and_ignore_working_directory(tmp_path, monkeypatch):
    import shutil
    original, moved = tmp_path/"original", tmp_path/"moved"
    cfg, cell = tiny_cfg(), tiny_cfg()["cells"][0]
    directory = original/"cells"/f.cell_key(cell)
    directory.mkdir(parents=True)
    files = {}
    for field in ("checkpoint", "trace"):
        file = directory/field
        file.write_bytes(field.encode())
        files[field] = str(file.relative_to(original))
    record = dict(cell, trained=True, **files, file_sha256={p: hashlib.sha256((original/p).read_bytes()).hexdigest() for p in files.values()})
    f.atomic_json(directory/"receipt.json", dict(identity="same", attempts=1, record=record))
    shutil.move(original, moved)
    monkeypatch.chdir(tmp_path)
    assert f.run_one(cfg, cell, moved, dict(identity="same")) == record


def test_schema_one_seal_checks_phase_coverage_hashes_and_runtime(tmp_path):
    import shutil
    from tttmem.frs_seal import execution_guard, source_paths
    cfg = tomllib.loads((ROOT/"configs/toy/frs-b.toml").read_text())["frsb"]
    cfg = {**cfg, "require_seal": True, "plan_path": "plan.md"}
    (tmp_path/"plan.md").write_text("registered plan fixture")
    required = source_paths(ROOT) + [ROOT/"configs/toy/frs-b.toml", ROOT/"configs/toy/smoke/frs-b.toml", ROOT/"configs/toy/smoke/frs-b-gpu.toml"]
    entries = []
    for file in required:
        relative = file.relative_to(ROOT)
        dest = tmp_path/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
        if relative.name == "frs-b.toml" and relative.parent.name == "toy":
            dest.write_text(dest.read_text().replace("[frsb]\n", '[frsb]\nplan_path = "plan.md"\n', 1))
        entries.append(dict(node="frsb", role="fixture", path=str(relative), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    entries.append(dict(node="frsb", role="frsb_plan", path="plan.md",
                        sha256=hashlib.sha256((tmp_path/"plan.md").read_bytes()).hexdigest()))
    # Synthetic test manifest only, never the production seal.
    path = tmp_path/"frs_seal.json"
    seal = dict(schema=1, phase="sealed", files=entries)
    path.write_text(json.dumps(seal))
    assert execution_guard(cfg, root=tmp_path)
    # Additive per-node entries preserve the top-level FRS-A files semantics.
    seal = dict(schema=1, phase="sealed", files=[], nodes=dict(frsb=dict(files=entries)))
    path.write_text(json.dumps(seal))
    assert execution_guard(cfg, root=tmp_path)
    bad = {**cfg, "batch": 1}
    with pytest.raises(ValueError, match="runtime protocol"):
        execution_guard(bad, root=tmp_path)
    seal["phase"] = "draft"
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="sealed manifest"):
        execution_guard(cfg, root=tmp_path)
    seal["phase"] = "sealed"
    path.write_text(json.dumps(seal))
    (tmp_path/"src/tttmem/exp_frsb.py").write_text("tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        execution_guard(cfg, root=tmp_path)
    seal["nodes"]["frsb"]["files"].pop(0)
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="lacks"):
        execution_guard(cfg, root=tmp_path)
