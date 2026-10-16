"""Regression fixtures for TRG stage (i); legacy/tiny inputs only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tttmem import dense_log as d, exp_trg as g, trg_triggers as tr
from test_trg import TINY


@pytest.fixture
def analyzer():
    spec = importlib.util.spec_from_file_location("trg_analyzer", g.ROOT/"scripts/analyze_trg.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    fork = tmp_path/"echo__s0__e2e1emb__eta40__st1500__fork_state_step360.pt"
    fork.write_bytes(b"metadata fixture; not loaded")
    series = tmp_path/"chr_series.csv"
    series.write_text("cell_id,branch,step,a\n")
    row = dict(cell_id="echo/s0/e2e1emb/eta40/st1500", file="/old/"+fork.name,
               step=360, next_batch_sha1="a"*40, nonfinite_at_fork=0)
    log = tmp_path/"chr.log"
    log.write_text("CHR_FORK "+json.dumps(row)+"\n")
    cfg = dict(fork_dir=str(tmp_path), chr_log_path=str(log), historical_series_path=str(series))
    return cfg, row, [dict(family="FO", arm="e2e1emb", seed=0)]


def test_historical_series_is_required_and_hashed(inputs):
    cfg, _, cells = inputs
    assert g.prepare_inputs(cfg, cells) == d.sha256_file(cfg["historical_series_path"])
    Path(cfg["historical_series_path"]).unlink()
    with pytest.raises(ValueError, match="CHR series"):
        g.prepare_inputs(cfg, cells)
    assert g.prepare_inputs(dict(cfg, smoke_only=True), cells) is None


@pytest.mark.parametrize("failure", ["missing", "duplicate", "basename", "fields", "cell-id"])
def test_chr_fork_matching_fails_closed(inputs, failure):
    cfg, row, cells = inputs
    if failure == "basename": row["file"] = "other.pt"
    if failure == "fields": row.pop("next_batch_sha1")
    if failure == "cell-id": row["cell_id"] = "renamed"
    rows = [] if failure == "missing" else [row, row] if failure == "duplicate" else [row]
    Path(cfg["chr_log_path"]).write_text("".join("CHR_FORK "+json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError, match="CHR_FORK"):
        g.prepare_inputs(cfg, cells)


def test_analyzer_series_override_and_integrity(inputs, analyzer, tmp_path):
    cfg, _, cells = inputs
    digest = g.prepare_inputs(cfg,cells)
    manifest = dict(effective_config=cfg, chr_series_sha256=digest)
    moved = tmp_path/"downloaded.csv"
    Path(cfg["historical_series_path"]).rename(moved)
    with pytest.raises(ValueError, match="CHR series missing"):
        analyzer.resolve_chr_series(manifest)
    assert analyzer.resolve_chr_series(manifest,moved) == str(moved)
    moved.write_text(moved.read_text()+"changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyzer.resolve_chr_series(manifest,moved)
    with pytest.raises(ValueError, match="hash missing"):
        analyzer.resolve_chr_series(dict(manifest,chr_series_sha256=None),moved)


def test_analyzer_override_never_relaxes_driver_or_schema(analyzer, capsys):
    current = g.source_hashes()
    manifest = dict(source_sha256=dict(current), schema_sha256=d.SCHEMA_SHA256)
    manifest["source_sha256"]["scripts/analyze_trg.py"] = "old-analyzer"
    with pytest.raises(ValueError, match="analyzer hash mismatch"):
        analyzer.verify_sources(manifest)
    evidence = analyzer.verify_sources(manifest, allow_analyzer_override=True)
    assert evidence["override_used"] and evidence["override_requested"]
    assert "TRG_ANALYZER_HASHES" in capsys.readouterr().out
    manifest["source_sha256"]["exp_trg.py"] = "old-driver"
    with pytest.raises(ValueError, match="driver/schema"):
        analyzer.verify_sources(manifest, allow_analyzer_override=True)
    manifest = dict(source_sha256=current, schema_sha256="old-schema")
    with pytest.raises(ValueError, match="driver/schema"):
        analyzer.verify_sources(manifest, allow_analyzer_override=True)


@pytest.fixture
def diverged_sham(tmp_path, monkeypatch):
    folder = tmp_path/"SO-s0"/"sham"
    folder.mkdir(parents=True)
    skip = folder.parent/"skip"
    skip.mkdir()
    (skip/"branch.json").write_text(json.dumps(dict(drops=[12,30])))
    steps = list(range(22))
    summaries = [dict(step=s, acc=1.) for s in steps]
    controller = tr.SpikeController(0)
    telemetry = [dict(step=s, grad_norm=1. if s < 21 else None,
                      **controller.observe(s, 1. if s < 21 else np.nan),
                      dropped=s == 15, lr=[.001], lr_before=[.001], masked_per_sequence=[])
                 for s in range(1,23)]
    (folder/"summaries.jsonl").write_text("".join(json.dumps(r)+"\n" for r in summaries))
    (folder/"telemetry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in telemetry))
    (folder/"acquisitions.jsonl").write_text("")
    (folder/"drops.json").write_bytes(tr.drop_list_bytes([15]))
    record = dict(family="SO", seed=0, branch="sham", output_dir="/old/run/sham",
                  disposition="DIVERGED", t_acq=0, start=0, end=50, t_div=22, certified=True,
                  error=None, nonfinite_local=2, steps=steps, accuracy=[1.]*22, kappa=5, t_l1=None,
                  drops=[15], drop_list_sha256=d.sha256_file(folder/"drops.json"),
                  source_drop_list_sha256=tr.realised_drop_hash([12,30]),
                  acquisition_sha256=d.sha256_file(folder/"acquisitions.jsonl"),
                  telemetry_sha256=d.sha256_file(folder/"telemetry.jsonl"))
    monkeypatch.setattr(d,"validate_shards",lambda _: ({"shards":[]},steps,[]))
    return tmp_path, record, folder


def test_diverged_sham_checks_only_schedule_until_stop(analyzer, diverged_sham):
    root, record, _ = diverged_sham
    summaries, telemetry = analyzer.contract_branch(record, root=root)
    assert len(summaries) == len(telemetry) == 22
    # COMPLETE would require all scheduled drops, including the future one.
    record = dict(record, disposition="COMPLETE", end=21, t_div=None, certified=False)
    with pytest.raises(ValueError, match="yoked realised schedule"):
        analyzer.contract_branch(record, root=root)


def test_online_dense_trigger_is_recomputed(analyzer, diverged_sham):
    root, record, _ = diverged_sham
    with pytest.raises(ValueError, match="t_l1 mismatch"):
        analyzer.contract_branch(dict(record,t_l1=1),root=root)


def test_online_acquisition_is_recomputed(analyzer, diverged_sham):
    root, record, folder = diverged_sham
    main = folder.parent/"main"
    folder.rename(main)
    (main/"acquisitions.jsonl").write_text(json.dumps(dict(step=5,acc=1.))+"\n")
    record = dict(record,branch="main",acquisition_sha256=d.sha256_file(main/"acquisitions.jsonl"))
    with pytest.raises(ValueError, match="t_acq mismatch"):
        analyzer.contract_branch(record,root=root)


def test_so_main_divergence_uses_training_segment_certificate(tmp_path, monkeypatch):
    original = g.train_step
    def nonfinite_tail(*args, **kwargs):
        row = original(*args, **kwargs)
        if row["step"] >= 20:
            row["finite"] = False
        return row
    monkeypatch.setattr(g,"train_step",nonfinite_tail)
    cfg = dict(TINY, smoke_only=True, smoke_fork=5, smoke_steps=30,
               output_dir=str(tmp_path), emit_dense=False)
    record = g.run_segment(cfg,dict(family="SO",seed=0,arm="e2e2emb"))
    assert record["t_acq"] == 5 and record["t_div"] == 21
    assert record["disposition"] == "DIVERGED"


def test_probes_resume_rejects_stale_kappa(tmp_path, monkeypatch):
    cfg = dict(TINY,process_name=os.environ["PROCESS_NAME"],output_dir=str(tmp_path),resume=True)
    cell = dict(family="SO",seed=0,arm="e2e2emb")
    marker = tmp_path/"SO-s0"/"cell-probes.json"
    g.write_json(marker,dict(cell=cell,phase="probes",branches=[],kappa=5,
                            config_sha256=g.config_digest(cfg),source_sha256=g.source_hashes()))
    with pytest.raises(FileNotFoundError,match="branch.json"):
        g._cell_worker((cfg,cell,"probes",3))
    assert g._cell_worker((cfg,cell,"probes",5))["kappa"] == 5


def test_soft_watchdog_stops_cleanly_without_work(tmp_path, monkeypatch, capsys):
    def elapsed(cfg, seed):
        g._check_deadline(dict(cfg,_deadline=0))
    monkeypatch.setattr(g,"_run_stage_i",elapsed)
    result = g.run(dict(output_dir=str(tmp_path),max_minutes=80,hard_grace_minutes=10),0)
    assert not result["complete"] and result["watchdog"]
    assert "TRG_STOP" in capsys.readouterr().out
    assert json.loads((tmp_path/"stop.json").read_text()) == result


def test_hard_watchdog_stops_children_and_emits_line(tmp_path, monkeypatch, capsys):
    calls = []
    class Child:
        def terminate(self): calls.append("terminate")
        def join(self, timeout): calls.append("join")
        def is_alive(self): return True
        def kill(self): calls.append("kill")
    monkeypatch.setattr(g.mp,"active_children",lambda:[Child()])
    monkeypatch.setattr(g.os,"_exit",lambda code:calls.append(code))
    g._hard_stop(tmp_path)
    assert calls == ["terminate","join","kill",124]
    assert "TRG_STOP" in capsys.readouterr().out
    assert json.loads((tmp_path/"stop.json").read_text())["exit_code"] == 124


def test_runtime_deadline_does_not_change_resume_hash():
    assert g.config_digest(dict(a=1,_deadline=1)) == g.config_digest(dict(a=1,_deadline=200))


def test_descriptive_coincidence_uses_active_domain(analyzer, monkeypatch):
    steps = np.arange(661)
    accuracy = np.ones(661)
    accuracy[220:223] = accuracy[300:303] = .5
    controller = tr.SpikeController(185)
    telemetry = [dict(step=int(s),**controller.observe(int(s),20. if s in (219,299,550,600) else 1.))
                 for s in steps]
    monkeypatch.setattr(d,"validate_shards",lambda _: ({"shards":[]},[],[]))
    record = dict(family="SO",seed=0,t_acq=185,t_l1=None,steps=steps,accuracy=accuracy,output_dir="unused")
    desc, _ = analyzer.descriptive(record, [], telemetry)
    wanted = tr.active_coincidence(telemetry,tr.dip_onsets(steps,accuracy,185),5)
    assert desc["coincidence"]["false_alarm"] == wanted["false_alarm"]
    unit = dict(steps=steps,accuracy=accuracy,t_acq=185,telemetry_steps=steps,norms=[20. if s in (219,299,550,600) else 1. for s in steps])
    assert tr.select_kappa([unit]*3)["median_j"]["5"] == wanted["hit"]-wanted["false_alarm"]
