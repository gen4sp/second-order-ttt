"""SOQ-N3B: split extension, FO-CM stream / schedule / orchestration, the from-scratch guard,
G-div, and the frontier reads (generic η, parity, κ, H_LM).  Tiny presets and synthetic data."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from tttmem.p2 import data as D
from tttmem.p2 import evaluate as E
from tttmem.p2 import matrix, soq_n3b
from tttmem.p2.guard import RunningMinGuard, probe_steps, running_min_rise, trip_action
from tttmem.p2.meta import InnerCfg, stream_eval
from tttmem.p2.mini import _g0_parity, _write_tiny_data, g0_read_spec, mini_g0
from tttmem.p2.model import build_model
from tttmem.p2.probe import KappaSet, find_probes, kappa_set, kappa_stats
from tttmem.p2.train import (TrainCfg, _resume_contract, checkpoint_steps, stream_order, train)
from test_p2 import _smoke_data, _write_split

REPO = Path(__file__).resolve().parents[1]


def _load(path):
    with open(REPO / path, "rb") as f:
        return tomllib.load(f)


# --------------------------------------------------------------------------- split extension
def _ws(texts):
    return [[int(w) for w in t.split()] for t in texts]


def _doc(tokens, row, shard="s0.parquet"):
    return {"text": " ".join(map(str, tokens)), "url": f"https://d{row % 5}.org/{row}",
            "shard": shard, "row": row}


def _store(tmp_path):
    """A prepared store (eval / ho / calib / A / B) written by ``prepare`` itself."""
    rng = np.random.default_rng(0)
    quotas = {"eval": 2, "ho": 1, "calib": 1, "A": 3, "B": 2}
    root = tmp_path / "p2data"
    docs = [_doc(rng.integers(3, 50000, size=D.SEQ + 5).tolist(), i) for i in range(80)]
    meta = D.prepare(iter(docs), _ws, root, quotas, min_chars=1, log_every=0,
                     tokenizer_revision="tok", dataset_revision="ds")
    assert meta["counts"] == quotas
    summary = {"sha256": {p.name: D.sha256_file(p) for p in sorted(root.iterdir())
                          if p.suffix in (".bin", ".jsonl")}}
    (root / "prepare_summary.json").write_text(json.dumps(summary))
    return root, docs


def _ext_cfg(root, tmp_path, quotas, **kw):
    return D.PrepareCfg(out_dir=str(root), extend=True, extend_quotas=quotas, min_chars=1,
                        seed=0, dataset_revision="ds", tokenizer_revision="tok", log_every=0,
                        extend_pack_path=str(tmp_path / f"pack-{'-'.join(sorted(quotas))}.tar"), **kw)


def test_extend_adds_c_then_d_and_keeps_every_store_byte(tmp_path, capsys):
    root, docs = _store(tmp_path)
    before = {p.name: D.sha256_file(p) for p in root.iterdir() if p.is_file()}
    a_sha = {m["sha1"] for m in D.open_split(root, "A")[1]}
    dup = next(d for d in docs if hashlib.sha1(d["text"].encode()).hexdigest() in a_sha)
    ev_row = np.asarray(D.open_split(root, "eval")[0][0])
    near = _doc(ev_row[1:].tolist() + [7, 7, 7], 900)        # same row as eval[0], new sha1
    rng = np.random.default_rng(5)
    fresh = [_doc(rng.integers(3, 50000, size=D.SEQ + 5).tolist(), 1000 + i) for i in range(4)]
    rec = D.extend_store(_ext_cfg(root, tmp_path, {"C": 2}), docs=iter([dup, near] + fresh),
                         tokenize=_ws, files=[])
    assert rec["counts"] == {"C": 2} and rec["existing_unchanged"]
    acc = rec["acceptance"]
    assert acc["duplicates"] == 1 and acc["rejected_row_filter"] == 1
    c_rows, c_man = D.open_split(root, "C")
    assert len(c_man) == 2 and {m["sha1"] for m in c_man} & a_sha == set()
    assert {p: D.sha256_file(root / p) for p in before} == before        # A/B/eval/ho/calib bytes
    log = json.loads((root / "extensions.json").read_text())
    assert [r["splits"] for r in log] == [["C"]]
    assert log[0]["sha256"]["C.bin"] == D.sha256_file(root / "C.bin")
    pack = tmp_path / "pack-C.tar"
    assert pack.exists() and (tmp_path / "pack-C.tar.sha256").read_text().split()[0] == D.sha256_file(pack)
    assert not list(root.glob(".extend-*"))
    # rerun = reuse; split D later with its own count; C stays byte-identical
    assert D.extend_store(_ext_cfg(root, tmp_path, {"C": 2}), docs=iter([]), tokenize=_ws,
                          files=[])["reused"]
    c_hash = D.sha256_file(root / "C.bin")
    more = [_doc(rng.integers(3, 50000, size=D.SEQ + 5).tolist(), 2000 + i) for i in range(3)]
    rec_d = D.extend_store(_ext_cfg(root, tmp_path, {"D": 3}), docs=iter([fresh[0]] + more),
                           tokenize=_ws, files=[])
    assert rec_d["counts"] == {"D": 3} and rec_d["acceptance"]["duplicates"] == 1  # C's doc too
    assert D.sha256_file(root / "C.bin") == c_hash
    assert [r["splits"] for r in json.loads((root / "extensions.json").read_text())] == [["C"], ["D"]]
    assert "P2_EXTEND" in capsys.readouterr().out


def test_extend_refuses_filter_mismatch_exhaustion_and_changed_store(tmp_path):
    root, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="filters differ"):
        D.extend_store(replace(_ext_cfg(root, tmp_path, {"C": 1}), min_chars=5),
                       docs=iter([]), tokenize=_ws, files=[])
    with pytest.raises(ValueError, match="source exhausted"):
        D.extend_store(_ext_cfg(root, tmp_path, {"C": 3}), docs=iter([]), tokenize=_ws, files=[])
    assert not (root / "C.bin").exists() and not list(root.glob(".extend-*"))
    with pytest.raises(ValueError, match="extend_quotas"):
        D.extend_store(_ext_cfg(root, tmp_path, {}), docs=iter([]), tokenize=_ws, files=[])
    with (root / "B.bin").open("r+b") as f:                               # corrupt a byte
        f.seek(100)
        f.write(b"\x01")
    with pytest.raises(ValueError, match="registered sha256"):
        D.extend_store(_ext_cfg(root, tmp_path, {"C": 1}), docs=iter([]), tokenize=_ws, files=[])


def test_prepare_config_dispatches_extend_and_root_config_is_split_c():
    cfg = D.PrepareCfg.from_dict(_load("configs/soq/n3b-prepare-c.toml")["p2"])
    assert cfg.extend and cfg.extend_quotas == {"C": 25600} and cfg.out_dir == "data/p2data"
    assert (cfg.min_chars, cfg.seed, cfg.near_dup_n, cfg.near_dup_stride) == (30000, 0, 64, 32)
    assert cfg.dataset_revision == "817d6752765f6a41261085171dd546b104f60626"
    assert cfg.tokenizer_revision == "315b20096dc791d381d514deb5f8bd9c8d6d3061"


def test_split_contamination_counts(tmp_path):
    data, T = _smoke_data(tmp_path)
    rng = np.random.default_rng(3)
    rows = rng.integers(3, 7, size=(4, T + 1))
    rows[:, 0] = 1
    _write_split(data, "C", rows)
    rec = D.split_contamination(data, "C")
    assert rec["hits"] == 0 and rec["rows"] == 4
    ev = np.asarray(D.open_split(data, "eval")[0][:1, :T + 1])
    _write_split(data, "C", np.concatenate([rows, ev]))
    rec = D.split_contamination(data, "C")
    assert rec["against"]["eval"]["prefix_hits"] == 1 and rec["hits"] == 1


def test_contamination_stops_only_on_extension_splits(tmp_path, capsys):
    """B (a registered prepare split, flagged near_dup in P2-1) is logged; C (extend) stops."""
    data, T = _smoke_data(tmp_path)
    _bc(data, T)
    ev = np.asarray(D.open_split(data, "eval")[0][:1, :T + 1])
    b = np.asarray(D.open_split(data, "B")[0][:, :T + 1])
    _write_split(data, "B", np.concatenate([b, ev]))
    (data / "meta.json").write_text(json.dumps({"quotas": {"A": 8, "B": 5, "eval": 6, "ho": 3,
                                                           "calib": 2}}))
    assert D.contamination_enforced(data, "C") and not D.contamination_enforced(data, "B")
    cfg = _tcfg(data, T, tmp_path / "x", steps=6, eval_steps=[6], full_eval_steps=[6],
                extra_train_splits=["B", "C"], contamination_check=True)
    assert not train(cfg)["aborted"]
    recs = [json.loads(x.split(" ", 1)[1]) for x in capsys.readouterr().out.splitlines()
            if x.startswith("P2_CONTAMINATION ")]
    assert [(r["split"], r["hits"], r["enforced"]) for r in recs] == [("B", 1, False), ("C", 0, True)]
    c = np.asarray(D.open_split(data, "C")[0][:, :T + 1])
    _write_split(data, "C", np.concatenate([c, ev]))
    with pytest.raises(ValueError, match="split 'C' overlaps"):
        train(replace(cfg, out_dir=str(tmp_path / "y"), resume_dir=str(tmp_path / "y-state")))
    base = {"data_dir": str(data), "global_batch": 2, "train_split": "A"}
    with pytest.raises(ValueError, match="split 'C' overlaps"):
        soq_n3b.preflight({"fo_cm": {"extra_train_splits": ["B", "C"]}}, base)
    _write_split(data, "C", c)
    assert soq_n3b.preflight({"fo_cm": {"extra_train_splits": ["B", "C"]}}, base)["boundary_step"] == 6


# --------------------------------------------------------------------------- FO-CM stream
def test_fo_cm_stream_is_a_then_b_then_c_and_pins_the_fo_ce_rows():
    A = B = C = 25600
    n_cm = 3552
    for seed, first, at_b, at_c in ((0, [16770, 11458, 17653, 13562], [41874, 36928], [73603, 65233]),
                                    (1, [4106, 2427, 3715, 17742], [47451, 31441], [65515, 52462]),
                                    (2, [20233, 14467, 14379, 6121], [28640, 38647], [58843, 65947])):
        o = stream_order([A, B, C], seed, n_cm * 16)
        assert o[:4].tolist() == first and o[A:A + 2].tolist() == at_b and o[A + B:A + B + 2].tolist() == at_c
        fo_ce = stream_order([A, B], seed, 1200 * 16)       # g43-g45: A (+B) at 1200 steps
        assert np.array_equal(o[:1200 * 16], fo_ce)          # steps 1-1200: identical rows
        assert o[:A].max() < A and A <= o[A:A + B].min() and o[A:A + B].max() < A + B
        assert o[A + B:].min() >= A + B                       # C only after the A+B rows
    assert (A + B) // 16 == 3200                              # only steps 3201.. touch C


def _tcfg(data, T, out, **kw):
    base = dict(arm="e2e1-ce", preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                global_batch=2, micro_batch=1, steps=4, warmup=1, lr_peak=1e-2,
                data_dir=str(data), out_dir=str(out), resume_dir=str(out) + "-state",
                n_eval=4, min_probes=1, assay_gate=False, ho_docs=3, ho_trajectory_docs=2,
                eval_batch=2, pilot_docs=2, pilot_steps=[], eval_steps=[4], full_eval_steps=[4],
                sizing_steps=[], trajectory_docs=2, crossover_rules=[], resume_every=2,
                log_every=1, save_checkpoints=False, eval_at_init=False, barriers=[2, 4, 6])
    base.update(kw)
    return TrainCfg(**base)


def _bc(data, T, nb=4, nc=4):
    rng = np.random.default_rng(1)
    for name, n in (("B", nb), ("C", nc)):
        rows = rng.integers(3, 7, size=(n, T + 1))
        rows[:, 0] = 1
        _write_split(data, name, rows)


def test_trainer_preconditions_name_the_split(tmp_path):
    data, T = _smoke_data(tmp_path)
    _bc(data, T)
    cfg = _tcfg(data, T, tmp_path / "x", extra_train_splits=["B", "C"])
    (data / "C.bin").unlink()
    with pytest.raises(ValueError, match="training split 'C'"):
        train(cfg)
    _write_split(data, "C", np.zeros((0, T + 1), dtype=np.int64))      # present but empty
    with pytest.raises(ValueError, match="training split 'C'"):
        train(cfg)
    _bc(data, T)
    with pytest.raises(ValueError, match=r"has 16 rows, needs 20.*split D"):
        train(replace(cfg, steps=10))
    assert not (tmp_path / "x" / "steps.jsonl").exists()


def test_contamination_check_stops_before_step_one(tmp_path, capsys):
    data, T = _smoke_data(tmp_path)
    _bc(data, T)
    ev = np.asarray(D.open_split(data, "eval")[0][:1, :T + 1])
    rows = np.asarray(D.open_split(data, "C")[0][:, :T + 1])
    _write_split(data, "C", np.concatenate([rows, ev]))
    with pytest.raises(ValueError, match="overlaps the eval/ho/calib"):
        train(_tcfg(data, T, tmp_path / "x", extra_train_splits=["B", "C"], contamination_check=True))
    assert "P2_CONTAMINATION" in capsys.readouterr().out
    assert not (tmp_path / "x" / "steps.jsonl").exists()


def test_fo_cm_schedule_and_n_cm():
    base = _load("configs/mini/e2e1-ce-m.toml")["p2"]
    over = soq_n3b.fo_cm_overrides(3552, base, fo_micro_batch=4, target_s=8800.0, boundary_step=3200)
    assert over["steps"] == 3552 and over["warmup"] == over["inner_warmup"] == 355
    assert over["eval_steps"] == [300, 600, 1200, 2400, 3552]
    assert over["full_eval_steps"] == [1200, 2400, 3552] and over["save_model_steps"] == [1200, 2400]
    assert over["keep_final_state"] and over["keep_resume_steps"] == [3200]
    assert over["micro_batch"] == 4 and over["train_seconds_target"] == 8800.0
    assert soq_n3b.n_cm_from_r(2.96) == 3552 and soq_n3b.n_cm_from_r(1.5) == 1800
    small = soq_n3b.fo_cm_overrides(1800, base, fo_micro_batch=2, target_s=1.0, boundary_step=3200)
    assert small["full_eval_steps"] == [1200, 1800] and small["keep_resume_steps"] == []
    cfg = TrainCfg.from_dict({**base, **over})
    assert checkpoint_steps(cfg, "fo") == {1200, 2400, 3552}          # mini-m: only these


def test_decide_n_cm_uses_the_fastest_fitting_settings_and_flags_r():
    pilot = {"settings": [
        {"arm": "e2e2", "micro_batch": 1, "status": "ok", "median_ms": 7300.0},
        {"arm": "e2e2", "micro_batch": 2, "status": "oom"},
        {"arm": "e2e1-ce", "micro_batch": 2, "status": "ok", "median_ms": 2500.0},
        {"arm": "e2e1-ce", "micro_batch": 4, "status": "ok", "median_ms": 2400.0}],
        "flops": {"e2e2": {"flops_per_seq": 3.0e12}, "e2e1-ce": {"flops_per_seq": 1.0e12}}}
    d = soq_n3b.decide_n_cm(pilot, r_flag_tol=.05)
    assert d["r"] == pytest.approx(7300 / 2400) and d["n_cm"] == round(1200 * 7300 / 2400)
    assert (d["so_micro_batch"], d["fo_micro_batch"]) == (1, 4)
    assert d["train_seconds_target"] == pytest.approx(7.3 * 1200)
    assert d["flop_ratio_counted"] == pytest.approx(3.0) and not d["r_flag"]
    pilot["settings"][3]["median_ms"] = 2000.0                         # r = 3.65: flagged, used
    d = soq_n3b.decide_n_cm(pilot, r_flag_tol=.05)
    assert d["r_flag"] and d["n_cm"] == round(1200 * 7300 / 2000)
    pilot["settings"] = [s for s in pilot["settings"] if s["arm"] == "e2e2"]
    with pytest.raises(RuntimeError, match="no usable"):
        soq_n3b.decide_n_cm(pilot)


def test_n3b_configs_parse_and_build_valid_cells():
    fo = _load("configs/soq/n3b-fo-cm.toml")["p2"]
    assert fo["mode"] == "soq_n3b_fo_cm" and fo["seeds"] == [0, 1, 2] and fo["process_name"].startswith("TTM-n3b-")
    assert fo["pilot"]["so_micro_batch"] == [1, 2] and fo["pilot"]["fo_micro_batch"] == [2, 4]
    assert (fo["pilot"]["warmup_steps"], fo["pilot"]["timed_steps"]) == (10, 50)
    assert fo["fo_cm"]["extra_train_splits"] == ["B", "C"] and fo["fo_cm"]["gdiv"]
    base = matrix._base_config(fo, {"preset": "mini-m", "arm": "e2e1-ce", "init_seed": 1,
                                    "data_seed": 1}, REPO)
    over = soq_n3b.fo_cm_overrides(3552, base, fo_micro_batch=2, target_s=1.0, boundary_step=3200,
                                   extra=fo["fo_cm"])
    cfg = TrainCfg.from_dict({**base, **over})
    assert (cfg.init_seed, cfg.data_seed, cfg.lr_peak, cfg.global_batch) == (1, 1, 3e-3, 16)
    assert cfg.guard_every == 50 and cfg.train_loss_guard == "running_min" and cfg.contamination_check
    reads = _load("configs/soq/n3b-reads.toml")["p2"]
    res = soq_n3b.resolve_reads(reads, 3552)
    assert len(res["items"]) == 12 and sum(len(i["reads"]) for i in res["items"]) == 105
    for it in res["items"]:
        assert all(g0_read_spec(r) is not None for r in it["reads"]) and it["reads"][0] == "ce"
        assert "{N_CM" not in json.dumps(it)
    so = [i for i in res["items"] if i["arm"] == "e2e2"]
    assert all(set(i["g0_ref"]) == {"ce", "eta150", "eta200"} and i["hlm"] for i in so)
    focm = [i for i in res["items"] if "/fo-cm/" in i["cell"]]
    assert {i["step"] for i in focm} == {3552, 2400}
    assert all(i["overrides"]["steps"] == 3552 for i in focm)
    assert [i["ckpt"].rsplit("/", 1)[1] for i in focm[:1]] == ["model_step3552.pt"]
    grid = sorted(g0_read_spec(r)[0] for r in focm[0]["reads"])
    assert grid == [.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12]
    only = soq_n3b.resolve_reads({**reads, "run_groups": ["existing"]}, None)
    assert len(only["items"]) == 6
    assert len(soq_n3b.resolve_reads(reads, 2400)["items"]) == 9       # 2400 items dropped
    with pytest.raises(ValueError, match="need N_CM"):
        soq_n3b.resolve_reads(reads, None)


def test_fo_cm_coordinator_pilot_n_cm_cells_attempt1_and_reuse(tmp_path, monkeypatch, capsys):
    """Coordinator logic with the workers stubbed: no torch process is started."""
    data, T = _smoke_data(tmp_path)
    _bc(data, T, nb=2, nc=8)
    ms = {("e2e2", 1): 30.0, ("e2e2", 2): None, ("e2e1-ce", 1): 12.0, ("e2e1-ce", 2): 10.0}
    names = []

    def worker(cell, path, log, name, *, flops=False):
        names.append(name)
        path.write_text(json.dumps(cell))
        if flops:
            log.write_text("N3B_FLOPS " + json.dumps({"flops_per_seq": 3e9 if cell["arm"] == "e2e2" else 1e9}))
            return 0
        t = ms[(cell["arm"], cell["micro_batch"])]
        if t is None:
            log.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
            return 1
        assert cell["stop_after"] == 4 and cell["eval_steps"] == [] and cell["guard_every"] == 0
        log.write_text("MINI_CELL_DONE " + json.dumps({"final": {"step_ms": [99.0, t, t, t + 1]}}) + "\n")
        return 0
    runs = []

    def run(mcfg, repo_root):
        runs.append(mcfg)
        for c in mcfg["cells"]:
            out = matrix._cell_out(Path(mcfg["out_dir"]), c)
            out.mkdir(parents=True, exist_ok=True)
            first = c["init_seed"] == 1 and "attempt" not in c["overrides"]
            fin = {"aborted": first, "label": "RESTART" if first else None, "steps": c["steps"],
                   "train_seconds": {"within_2pct": True}}
            (out / "MINI_CELL_DONE.json").write_text(json.dumps({"status": "done", "final": fin}))
    monkeypatch.setattr(soq_n3b, "_run_worker", worker)
    monkeypatch.setattr(matrix, "run", run)
    cfg = {"mode": "soq_n3b_fo_cm", "out_dir": str(tmp_path / "n3b"), "data_dir": str(data),
           "preset": "tiny", "steps": 4, "seeds": [0, 1], "base_steps": 2, "r_prior": 2.96,
           "r_flag_tol": .05, "process_nn": 2, "pilot": {"so_micro_batch": [1, 2],
           "fo_micro_batch": [1, 2], "warmup_steps": 1, "timed_steps": 3},
           "fo_cm": {"extra_train_splits": ["B", "C"], "guard_every": 1}}
    res = soq_n3b.soq_n3b_fo_cm(cfg)
    dec = json.loads((tmp_path / "n3b" / "n_cm.json").read_text())
    assert dec["r"] == pytest.approx(30 / 10) and dec["n_cm"] == 6 and not dec["r_flag"]
    assert (dec["so_micro_batch"], dec["fo_micro_batch"]) == (1, 2) and dec["flop_ratio_counted"] == 3
    assert [s["status"] for s in dec["pilot"]["settings"]] == ["ok", "oom", "ok", "ok"]
    assert names[:4] == [f"TTM-n3b-02-pilot{k}" for k in (1, 2, 3, 4)]
    cells = runs[0]["cells"]
    assert [(c["init_seed"], c["data_seed"], c["steps"]) for c in cells] == [(0, 0, 6), (1, 1, 6)]
    ov = cells[0]["overrides"]
    assert ov["steps"] == 6 and ov["micro_batch"] == 2 and ov["guard_every"] == 1
    assert ov["train_seconds_target"] == pytest.approx(.03 * 2) and ov["keep_resume_steps"] == [5]  # (8 + 2) // 2
    assert ov["extra_train_splits"] == ["B", "C"] and runs[0]["name_prefix"] == "TTM-n3b"
    assert runs[0]["workers"] == 1 and runs[0]["analyse"] is False
    assert [c["init_seed"] for c in runs[1]["cells"]] == [1]
    assert runs[1]["cells"][0]["overrides"]["attempt"] == 1 and runs[1]["cells"][0]["overrides"]["lr_peak"] == 1e-3
    assert res["so_attempt1_required"] and res["n_cm"] == 6 and res["within_2pct"]
    assert "N3B_CONTAMINATION" in capsys.readouterr().out
    names.clear()
    soq_n3b.soq_n3b_fo_cm(cfg)                                          # restart: n_cm.json reused
    assert names == [] and len(runs) == 4
    pod = dict(cfg, out_dir=str(tmp_path / "pod2"), seeds=[2], n_cm_file=str(tmp_path / "n3b" / "n_cm.json"))
    soq_n3b.soq_n3b_fo_cm(pod)
    assert names == [] and [c["init_seed"] for c in runs[-1]["cells"]] == [2]
    assert soq_n3b.soq_n3b_fo_cm(dict(cfg, pilot_only=True))["pilot_only"]
    (data / "C.jsonl").write_text("")
    with pytest.raises(ValueError, match="split 'C' is missing or empty"):
        soq_n3b.soq_n3b_fo_cm(cfg)


def test_matrix_name_prefix_defaults_unchanged(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(matrix, "_worker", lambda path, log, name: seen.append(name) or _Done())
    monkeypatch.setattr(matrix, "_analyse_root", lambda root, label: None)
    for prefix in (None, "TTM-n3b"):
        cfg = {"out_dir": str(tmp_path / str(prefix)), "preset": "tiny", "process_nn": 7,
               "cells": [{"arm": "e2e1-ce", "init_seed": 0}], "data_dir": str(tmp_path)}
        if prefix:
            cfg["name_prefix"] = prefix
        matrix.run(cfg, repo_root=REPO)
    assert seen == ["TTM-min-07-tiny-e2e1-ce-s0", "TTM-n3b-07-tiny-e2e1-ce-s0"]


class _Done:
    pid, returncode = 1, 1

    class _ttm_log_handle:
        @staticmethod
        def close():
            pass

    def poll(self):
        return 1


# --------------------------------------------------------------------------- guard rules
def test_running_min_guard_rules():
    g = RunningMinGuard(.10, 2)
    trace = [g.update(s, v) for s, v in ((50, 3.0), (100, 2.8), (150, 2.95), (200, 2.85),
                                         (250, 2.88), (300, 2.96))]
    assert [t["exceed"] for t in trace] == [False, False, True, False, False, True]
    assert [t["trip"] for t in trace] == [False] * 6                   # never 2 in a row
    assert trace[2]["running_min"] == 2.8 and trace[2]["rise"] == pytest.approx(.15)
    assert g.update(350, 2.95)["trip"]                                  # 2 consecutive
    assert not RunningMinGuard().update(50, 9.0)["exceed"]             # first probe: no baseline
    nan = RunningMinGuard()
    nan.update(50, 3.0)
    assert not nan.update(100, float("nan"))["finite"] and nan.update(150, 3.05)["running_min"] == 3.0
    assert probe_steps(0, 400, every=50) == [50, 100, 150, 200, 250, 300, 350, 400]
    assert probe_steps(0, 300, every=50, dense_every=10)[:3] == [50, 60, 70]
    assert probe_steps(0, 300, every=50, dense_every=10)[-3:] == [200, 250, 300]
    assert probe_steps(2400, 2600, every=50) == [2450, 2500, 2550, 2600] and probe_steps(0, 9, every=0) == []
    assert trip_action(150, "so", 0)["label"] == "RESTART" and trip_action(150, "fo", 0)["label"] == "RESTART"
    assert trip_action(200, "so", 1) == {"label": "UNSTABLE_AT_ATTEMPT_1", "stop": True,
                                         "next": "stop_replan", "final": True}
    assert trip_action(250, "so", 0)["label"] == "SO-UNSTABLE" and trip_action(250, "so", 0)["stop"]
    assert trip_action(250, "fo", 0) == {"label": "FO-UNSTABLE", "stop": False, "next": "continue_descriptive"}
    assert running_min_rise([], 3.0) == 0 and running_min_rise([3.0, 2.5], 2.7) == pytest.approx(.2)
    assert math.isnan(running_min_rise([3.0], float("inf")))


def _fake_probe(monkeypatch, values):
    """Replace the guard's adapted calib loss by a scripted series (one value per probe)."""
    it = iter(values)

    def fake(model, docs, rule, **kw):
        v = next(it)
        return np.array([v * 10.0]), np.array([10.0])
    monkeypatch.setattr(E, "adapted_loss", fake)


@pytest.mark.parametrize("arm,restart_until,label,aborted", [
    ("e2e1-ce", 200, "RESTART", True), ("e2e1-ce", 0, "FO-UNSTABLE", False),
    ("e2e2", 0, "SO-UNSTABLE", True)])
def test_guard_trip_actions_in_training(tmp_path, monkeypatch, capsys, arm, restart_until, label, aborted):
    data, T = _smoke_data(tmp_path)
    _fake_probe(monkeypatch, [3.0, 3.2, 3.3, 3.4])
    cfg = _tcfg(data, T, tmp_path / "g", arm=arm, guard_every=1, guard_first=1,
                guard_restart_until=restart_until)
    res = train(cfg)
    out = capsys.readouterr().out
    assert sum(ln.startswith("P2_GUARD ") for ln in out.splitlines()) == (3 if aborted else 4)
    if aborted:
        assert res["aborted"] and res["label"] == label and res["step"] == 3
        assert json.loads((tmp_path / "g" / "guard_abort.json").read_text())["label"] == label
        with pytest.raises(ValueError, match="guard-stopped attempt cannot resume"):
            train(replace(cfg, resume=True))
    else:
        assert not res["aborted"] and res["guard_label"] == label
        assert json.loads((tmp_path / "g" / "guard_label.json").read_text())["step"] == 3


def test_guard_fo_stop_turns_fo_unstable_into_a_stop(tmp_path, monkeypatch, capsys):
    data, T = _smoke_data(tmp_path)
    _fake_probe(monkeypatch, [3.0, 3.2, 3.3, 3.4])
    cfg = _tcfg(data, T, tmp_path / "g", arm="e2e1-ce", guard_every=1, guard_first=1,
                guard_restart_until=0, guard_fo_stop=True)
    res = train(cfg)
    assert res["aborted"] and res["label"] == "FO-UNSTABLE" and res["step"] == 3
    assert json.loads((tmp_path / "g" / "guard_abort.json").read_text())["next"] == "stop_user_decides"


def test_pilot_rerun_after_interruption_and_peak_cap(tmp_path, monkeypatch):
    """A half-done pilot setting (dir + log left behind) reruns fresh; a setting over the peak cap
    does not count as fitting."""
    calls = []

    def fake_worker(cell_cfg, path, log, name, *, flops=False):
        assert not Path(cell_cfg["out_dir"]).exists() and not log.exists()
        calls.append(cell_cfg["micro_batch"])
        peak = 21.0 if cell_cfg["micro_batch"] == 4 else 10.0
        log.write_text("MINI_CELL_DONE " + json.dumps(
            {"final": {"step_ms": [5.0] * 60}, "peak_gb": peak}) + "\n")
        return 0

    monkeypatch.setattr(soq_n3b, "_run_worker", fake_worker)
    monkeypatch.setattr(soq_n3b.matrix, "_base_config", lambda cfg, cell, repo: {"global_batch": 16})
    out = tmp_path / "n3b"
    stale = out / "pilot" / f"{soq_n3b.PILOT_FO}-mb4"
    stale.mkdir(parents=True)
    (out / "pilot" / f"{soq_n3b.PILOT_FO}-mb4.log").write_text("CUDA out of memory\n")
    cfg = {"pilot": {"so_micro_batch": [1], "fo_micro_batch": [2, 4], "count_flops": False,
                     "max_peak_gb": 20.0}}
    res = soq_n3b.timing_pilot(cfg, tmp_path, out, lambda s: f"TTM-n3b-test-{s}")
    st = {(r["arm"], r["micro_batch"]): r["status"] for r in res["settings"]}
    assert calls == [1, 2, 4]
    assert st[(soq_n3b.PILOT_FO, 2)] == "ok" and st[(soq_n3b.PILOT_FO, 4)] == "over_peak"


def test_gdiv_reloads_once_then_stops(tmp_path, monkeypatch, capsys):
    data, T = _smoke_data(tmp_path)
    _fake_probe(monkeypatch, [3.0, float("nan"), 3.0, 3.0, 3.0])           # one NaN, then finite
    res = train(_tcfg(data, T, tmp_path / "a", guard_every=1, guard_first=1, gdiv=True, resume_every=1))
    assert not res["aborted"] and res["steps"] == 4
    assert len(json.loads((tmp_path / "a" / "gdiv.json").read_text())) == 1
    assert "P2_GDIV" in capsys.readouterr().out
    _fake_probe(monkeypatch, [3.0, float("nan"), float("nan")])            # NaN again after reload
    res = train(_tcfg(data, T, tmp_path / "b", guard_every=1, guard_first=1, gdiv=True, resume_every=1))
    assert res["aborted"] and res["reason"] == "gdiv_diverged" and res["label"] == "DIVERGED"
    assert (tmp_path / "b" / "divergence_abort.json").exists()


def test_train_loss_guard_running_min_from_scratch_start_relative_for_continuations(tmp_path, capsys):
    data, T = _smoke_data(tmp_path)
    res = train(_tcfg(data, T, tmp_path / "s", train_loss_max_rise=-1.0))
    assert res["aborted"] and res["baseline_kind"] == "running_min" and res["step"] == 1
    parent = _tcfg(data, T, tmp_path / "p", steps=2, eval_steps=[2], full_eval_steps=[2],
                   keep_final_state=True)
    train(parent)
    cont = _tcfg(data, T, tmp_path / "c", continue_from=str(tmp_path / "p-state" / "resume_step0002.pt"),
                 steps=2, continue_steps=4, train_loss_max_rise=-1.0)
    res = train(cont)
    assert res["aborted"] and "baseline_kind" not in res and res["step"] == 3   # start-relative
    assert res["baseline"] == res["loss"]


def test_n3b_keys_stay_out_of_the_resume_contract_at_defaults():
    c = TrainCfg()
    contract = _resume_contract(c)
    for k in ("guard_every", "gdiv", "save_model_steps", "train_loss_guard", "log_train_seconds",
              "train_seconds_target", "keep_resume_steps", "contamination_check"):
        assert k not in contract
    assert _resume_contract(replace(c, guard_every=50))["guard_every"] == 50
    assert "keep_resume_steps" not in _resume_contract(replace(c, keep_resume_steps=[3200]))


def test_pilot_loss_uses_the_runs_barriers(tmp_path, monkeypatch):
    data, T = _smoke_data(tmp_path)
    seen = []
    real = E.eval_docs

    def spy(*a, **kw):
        if list(kw.get("modes", [])) == ["heldout"]:
            seen.append(kw.get("barriers"))
        return real(*a, **kw)
    monkeypatch.setattr(E, "eval_docs", spy)
    train(_tcfg(data, T, tmp_path / "p", arm="e2e2", steps=2, eval_steps=[], full_eval_steps=[],
                pilot_steps=[1, 2], pilot_max_rise=10.0))
    assert len(seen) == 3 and all(list(b) == [2, 4, 6] for b in seen)    # baseline + 2 probes


def test_fo_cm_training_saves_times_and_keeps_states(tmp_path, monkeypatch, capsys):
    data, T = _smoke_data(tmp_path)
    _bc(data, T, nb=4, nc=8)
    cfg = _tcfg(data, T, tmp_path / "f", steps=8, extra_train_splits=["B", "C"],
                eval_steps=[4, 8], full_eval_steps=[4, 8], save_model_steps=[4],
                keep_final_state=True, keep_resume_steps=[6], resume_every=3, guard_every=2,
                guard_first=2, gdiv=True, log_train_seconds=True, train_seconds_target=1e9,
                contamination_check=True, save_checkpoints=True)
    res = train(cfg)
    out = capsys.readouterr().out
    assert not res["aborted"] and res["steps"] == 8
    times = [json.loads(ln.split(" ", 1)[1]) for ln in out.splitlines() if ln.startswith("P2_TIME ")]
    assert [t["step"] for t in times] == list(range(1, 9))
    assert times[-1]["train_s"] == pytest.approx(res["train_seconds"]["train_s"])
    ts = res["train_seconds"]
    assert ts["steps"] == 8 and ts["within_2pct"] is False and ts["ratio"] < 1e-3
    state = Path(str(tmp_path / "f") + "-state")
    assert sorted(p.name for p in state.glob("resume_*.pt")) == ["resume_keep0006.pt", "resume_step0008.pt"]
    assert json.loads((tmp_path / "f" / "final_state.json").read_text())["step"] == 8
    assert (tmp_path / "f" / "model_step0004.pt").exists() and (tmp_path / "f" / "model_step0008.pt").exists()
    assert [json.loads(ln.split(" ", 1)[1])["step"] for ln in out.splitlines()
            if ln.startswith("P2_GUARD ")] == [2, 4, 6, 8]
    keep = torch.load(state / "resume_keep0006.pt", weights_only=False)
    assert keep["step"] == 6 and keep["cursor"] == 12 and "guard" in keep and keep["train_s"] > 0


# --------------------------------------------------------------------------- reads
def test_g0_read_spec_generic_eta():
    assert g0_read_spec("ce") == (1.0, "none", None) and g0_read_spec("sig4")[1] == "sigma"
    assert g0_read_spec("eta075") == (.75, "none", None) and g0_read_spec("eta160")[0] == 1.6
    assert g0_read_spec("eta450")[0] == 4.5 and g0_read_spec("eta1600")[0] == 16.0
    assert g0_read_spec("eta000") is None and g0_read_spec("eta16") is None and g0_read_spec("x") is None


def test_stream_eval_gather_leaves_the_nll_unchanged():
    torch.manual_seed(0)
    model = build_model("tiny", compute_dtype=torch.float32, seed=3)
    rng = np.random.default_rng(0)
    ids = torch.from_numpy(rng.integers(3, 30, size=(2, 65))).long()
    x, y = ids[:, :-1], ids[:, 1:]
    mask = torch.ones_like(x, dtype=torch.bool)
    inner = InnerCfg(rule="ce", lr=1.0, clip=1.0)
    im = mask.clone()
    im[:, 32:] = False
    nllp = torch.zeros_like(mask)
    nllp[:, 40:44] = True
    pos = torch.tensor([[40, 50, 63, -1], [33, 41, 41, 60]])
    cid = torch.tensor([[[3, 4, 5]] * 4, [[7, 8, 9]] * 4])
    kw = dict(barriers=[2, 4, 6], inner_mask=im, nll_positions=nllp, fused_attn=False)
    a = stream_eval(model, x, y, mask, inner, **kw)
    b = stream_eval(model, x, y, mask, inner, gather_positions=pos, gather_ids=cid, **kw)
    c = stream_eval(model, x, y, mask, inner, candidate_positions=pos.clamp_min(0),
                    candidate_ids=cid, **kw)
    assert torch.equal(a["nll"], b["nll"]) and torch.equal(a["clip_scale"], b["clip_scale"])
    lp, zc = b["gather_logp"], b["gather_centred"]
    assert torch.isnan(lp[0, 3]).all() and torch.isfinite(lp[:, :3]).all()
    torch.testing.assert_close(lp[1], c["candidate_log_probs"][1].to(lp.dtype), rtol=1e-6, atol=1e-6)
    gap = zc - lp                                     # logsumexp - mean logit: per position only
    torch.testing.assert_close(gap[:, :3] - gap[:, :3, :1], torch.zeros_like(gap[:, :3]), atol=1e-5, rtol=0)


def _kdoc(rng, T=64, C=8):
    base = rng.integers(3, 12, size=6)
    x = rng.integers(12, 40, size=T)
    x[4:10] = base
    x[36:42] = base
    x[50:56] = base
    y = np.concatenate([x[1:], [5]])
    return x, y


def test_kappa_set_foils_same_segment_context_and_tags():
    rng = np.random.default_rng(2)
    C, bars, seen = 8, (2, 4, 6), 0
    for d in range(10):
        x, y = _kdoc(rng)
        p = find_probes(x, y, C, barriers=bars)
        if not len(p.pos):
            continue
        s = kappa_set(x, y, p, C, barriers=bars, seed=d)
        assert np.array_equal(s.foil_q, kappa_set(x, y, p, C, barriers=bars, seed=d).foil_q)
        seen += int(s.foil_off[-1] > 0)
        ctx = lambda t: (x[t - 2], x[t - 1], x[t])              # noqa: E731
        for k, t in enumerate(p.pos):
            yk = y[t]
            f = s.pos[s.foil_q[s.foil_off[k]:s.foil_off[k + 1]]]
            assert len(f) <= 8 and np.all(f >= 4 * C) and np.all(y[f] != yk)
            assert np.all(np.searchsorted([0, 16, 32, 48], f, "right") ==
                          np.searchsorted([0, 16, 32, 48], t, "right"))     # the probe's segment
            src = {ctx(u) for u in range(2, 4 * C) if y[u] == yk}
            assert all(ctx(u) not in src for u in f)
            part = s.foil_partial[s.foil_off[k]:s.foil_off[k + 1]]
            assert np.array_equal(part, np.isin(x[f], [x[u] for u in range(4 * C) if y[u] == yk]))
            assert np.array_equal(s.foil_partial_own[s.foil_off[k]:s.foil_off[k + 1]], x[f] == x[t])
        assert np.array_equal(s.cand[s.probe_v], y[p.pos]) and np.array_equal(s.n_src, p.n_src)
    assert seen >= 3


def _synthetic_sets(K=4, V=3, F=2, docs=6):
    sets = []
    for i in range(docs):
        Q = K * (1 + F)
        probe_v = np.arange(K) % V
        sets.append(_kset(Q, V, K, F, probe_v))
    return sets


def _kset(Q, V, K, F, probe_v):
    foil_q = np.arange(K, K + K * F)
    return KappaSet(pos=np.arange(Q), cand=np.arange(V), probe_q=np.arange(K), probe_v=probe_v,
                    n_src=np.ones(K, np.int64), y_mult=np.ones(K, np.int64),
                    seg=np.full(K, 2), foil_off=np.arange(0, K * F + 1, F), foil_q=foil_q,
                    foil_partial=np.zeros(K * F, bool), foil_partial_own=np.zeros(K * F, bool))


def test_kappa_is_one_for_key_specific_and_zero_for_candidate_priming():
    rng = np.random.default_rng(0)
    sets = _synthetic_sets()
    base = [rng.normal(size=(len(s.pos), len(s.cand))) for s in sets]
    keyed = [b.copy() for b in base]
    for s, z in zip(sets, keyed):
        z[s.probe_q, s.probe_v] += 1.0                  # only y_k at t_k
    sm, arr = kappa_stats(sets, keyed, base, n_boot=500)
    assert sm["kappa"] == pytest.approx(1.0) and sm["kappa_x"] == pytest.approx(1.0)
    assert sm["kappa_lo90"] == pytest.approx(1.0) and not sm["kappa_x_na"]
    assert sm["rank_top1_nonfocal"] >= sm["rank_top1_focal"]
    primed = [b.copy() for b in base]
    for s, z in zip(sets, primed):
        z[:, np.unique(s.probe_v)] += 1.0               # every candidate y_k, at every position
    sm, arr = kappa_stats(sets, primed, base, n_boot=500)
    assert sm["kappa"] == pytest.approx(0.0, abs=1e-12) and sm["kappa_x"] == pytest.approx(0.0, abs=1e-12)
    assert sm["rank_gain_x"] == pytest.approx(0.0) and sm["cross_S_mean"] == pytest.approx(0.0, abs=1e-12)
    assert arr["x_nf"].size == sum(len(s.probe_q) * len(s.cand) for s in sets)
    assert len(arr["foil_nf"]) == sum(len(s.foil_q) for s in sets) and arr["n_foil"].tolist() == [2] * 24


def test_kappa_nan_not_clipped_and_rank_chance():
    rng = np.random.default_rng(1)
    sets = _synthetic_sets()
    base = [rng.normal(size=(len(s.pos), len(s.cand))) for s in sets]
    worse = [b.copy() for b in base]
    for s, z in zip(sets, worse):
        z[s.probe_q, s.probe_v] -= 1.0                  # ΣP < 0 in every document
    sm, arr = kappa_stats(sets, worse, base, n_boot=200)
    assert math.isnan(sm["kappa"]) and np.isnan(arr["doc_kappa"]).all() and sm["n_docs_sumP_pos"] == 0
    assert sm["rank_chance"] == pytest.approx(1 / 3) and sm["rank_n_probes"] == 24
    same = kappa_stats(sets, base, base, n_boot=200)[0]
    assert same["rank_top1_diff"] == 0 and math.isnan(same["kappa"]) and same["kappa_x_na"]


def test_parity_rule(tmp_path):
    rows = np.arange(4)
    ref = tmp_path / "ref.npz"
    M = np.array([.3, .4, .5, .6])
    Ls = np.array([[30.0, 10.0, 0], [40.0, 10.0, 0]])
    np.savez(ref, M_doc=M, L_adapt=np.asarray(3.5), n_ho=np.asarray(2), eval_rows=rows,
             L_adapt_sum=Ls[:, 0], L_adapt_count=Ls[:, 1])
    ok = _g0_parity("ce", M + 5e-4, Ls, 3.5 + 1e-4, ref, rows, 2, 1e-3, 2e-4)
    assert ok["pass"] and ok["dM"] == pytest.approx(5e-4)
    assert not _g0_parity("ce", M + 2e-3, Ls, 3.5, ref, rows, 2, 1e-3, 2e-4)["pass"]
    assert not _g0_parity("ce", M, Ls, 3.5 + 3e-4, ref, rows, 2, 1e-3, 2e-4)["pass"]
    assert not _g0_parity("ce", M, Ls, 3.5, ref, rows + 1, 2, 1e-3, 2e-4)["same_documents"]


def test_mini_g0_n3b_keys_end_to_end_tiny(tmp_path, capsys):
    from test_g0 import _tiny_cfg
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=8)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    base = {"process_nn": 91, "process_name": "TTM-n3b-91-test", "data_dir": str(data),
            "device": "cpu", "n_docs": 4, "n_ho_adapt": 2, "kappa_boot": 200, "setup_cache": True}
    item = {"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0, "data_seed": 0, "step": 2,
            "ckpt": str(cell / "model_step0002.pt")}
    first = tmp_path / "r1"
    mini_g0({**base, "out_dir": str(first), "items": [{**item, "out": str(first / "c"),
                                                      "reads": ["ce", "eta150"]}]})
    out = tmp_path / "r2"
    res = mini_g0({**base, "out_dir": str(out), "kappa_reads": "all", "hlm": True,
                   "items": [{**item, "out": str(out / "c"), "reads": ["ce", "eta150", "eta350"],
                              "overrides": {"steps": 2},
                              "read_modes": {"ce": ["frozenP", "fP_focal_refclip"]},
                              "g0_ref": {"ce": str(first / "c" / "g0_read_ce.npz"),
                                         "eta150": str(first / "c" / "g0_read_eta150.npz")}}]})
    assert res["done"] == 1
    lines = capsys.readouterr().out.splitlines()
    par = [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("G0_PARITY ")]
    assert [p["read"] for p in par] == ["ce", "eta150"] and all(p["pass"] and p["dM"] == 0 for p in par)
    kap = [json.loads(x.split(" ", 1)[1]) for x in lines if x.startswith("G0_KAPPA ")]
    assert [k["read"] for k in kap] == ["ce", "eta150", "eta350"] and "kappa_x" in kap[0]["kz"]
    with np.load(out / "c" / "g0_read_ce.npz") as z, np.load(first / "c" / "g0_read_ce.npz") as z1:
        np.testing.assert_array_equal(z["M_doc"], z1["M_doc"])          # κ gather changes nothing
        for k in ("kz_x_nf", "klp_x_f", "kz_foil_nf", "kz_n_foil", "kz_n_src", "kz_y_mult", "kz_seg",
                  "kz_foil_partial", "kz_doc_sum_P", "kz_doc_sum_F", "klp_p_f", "frozenP",
                  "fP_focal_refclip", "clip_bind_fP_focal", "L_clip_bind_doc", "L_adapt_sum"):
            assert k in z.files
        assert z["clip_bind_fP_nonfocal"].shape == (4,) and z["eta"] == 1.0
    with np.load(out / "c" / "g0_hlm.npz") as h, np.load(cell / "eval_step0002.npz") as logged:
        assert h["heldout_sum"].shape == (2,) and (h["heldout_count"] > 0).all()
        np.testing.assert_allclose(h["heldout_doc"], logged["doc_heldout_noadapt"], rtol=1e-6)
        np.testing.assert_allclose(h["heldout_sum"] / h["heldout_count"], h["heldout_doc"], rtol=1e-6)
    bad = tmp_path / "r3"
    with pytest.raises(RuntimeError, match="error"):
        mini_g0({**base, "out_dir": str(bad), "items": [{**item, "out": str(bad / "c"),
                 "reads": ["ce", "eta200"], "g0_ref": {"eta200": str(first / "c" / "g0_read_eta150.npz")}}]})
    assert any(x.startswith("G0_PARITY ") and '"pass": false' in x for x in capsys.readouterr().out.splitlines())


# --------------------------------------------------------------------------- CPU scripts
def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


