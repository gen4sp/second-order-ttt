"""SOQ-P1 preparation: continuation protocol, echo-injection plumbing, echo reads.

Tiny preset on CPU.  The real generator (``echo_text``) draws token ids from the Llama
vocabulary, so the tiny-model training tests use a deterministic stand-in with the same
interface; the real generator is exercised through the Prefetcher and the eval-set builder
at the production row shape (numpy only).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tomllib
import types

import numpy as np
import pytest
import torch

from tttmem.p2 import soq_p1, soq_p1_analysis
from tttmem.p2.echo_text import build_eval_documents, make_injector
from tttmem.p2.train import (CONTINUATION_IDENTITY, Prefetcher, StreamRows, TrainCfg, lr_at,
                             parity_check, schedule_lr, stream_order, train)
from test_p2 import _smoke_data, _write_split

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- helpers
class StubInjector:
    """Interface stand-in for ``echo_text.EchoInjector`` at the tiny shape (vocab 97).

    Selected iff a fixed hash of the global row index is in the lower half; a selected row
    gets six tokens 7..12 (never produced by the smoke data, ids 3..6) at positions 1..6."""

    def __init__(self, f=.5, p=.5, seed=0):
        self.f, self.p, self.seed = f, p, seed
        self.layout = types.SimpleNamespace(n_slots=8)
        self.calls = []

    def selected(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return (idx * 2654435761 + self.seed) % 4 < 2

    def manifest(self):
        params = {"stub": 1, "f": self.f, "p": self.p, "seed": self.seed}
        enc = json.dumps(params, sort_keys=True).encode()
        return {**params, "params_sha256": hashlib.sha256(enc).hexdigest()}

    def _write(self, rows, sel):
        out = np.asarray(rows).astype(np.uint32, copy=True)
        out[sel, 1:7] = np.arange(7, 13, dtype=np.uint32)
        return out

    def eval_documents(self, rows, row_ids=None, p=None):
        return self._write(rows, np.ones(len(rows), dtype=bool)), {"rows": []}

    def __call__(self, rows, idx):
        idx = np.asarray(idx, dtype=np.int64)
        self.calls.append((idx.copy(), np.asarray(rows).copy()))
        sel = self.selected(idx)
        return self._write(rows, sel), {"selected": sel.tolist()}


def _data(tmp_path):
    data, T = _smoke_data(tmp_path)
    rng = np.random.default_rng(1)
    rows = rng.integers(3, 7, size=(8, T + 1))
    rows[:, 0] = 1
    _write_split(data, "B", rows)
    return data, T


def _cfg(data, T, **kw):
    base = dict(arm="e2e2", preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                global_batch=2, micro_batch=1, steps=3, warmup=2, lr_peak=1e-2,
                data_dir=str(data), n_eval=4, min_probes=1, assay_gate=False,
                ho_docs=3, ho_trajectory_docs=2, eval_batch=2, pilot_docs=2, pilot_steps=[],
                eval_steps=[3], full_eval_steps=[], sizing_steps=[], trajectory_docs=2,
                crossover_rules=[], resume_every=1, log_every=1, save_checkpoints=False,
                eval_at_init=False, extra_train_splits=["B"], echo_f=.5, echo_p=.5)
    base.update(kw)
    return TrainCfg(**base)


def _stream(data, T, seed=0):
    from tttmem.p2.data import open_split
    a, _ = open_split(data, "A")
    b, _ = open_split(data, "B")
    return np.concatenate([np.asarray(a[:, :T + 1]), np.asarray(b[:, :T + 1])])


def _steps(root):
    return [json.loads(l.removeprefix("P2_STEP ")) for l in (root / "steps.jsonl").read_text().splitlines()]


def _phase1(tmp_path, data, T, stub, monkeypatch, **kw):
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: stub if cfg.echo_f > 0 else None)
    cfg = _cfg(data, T, out_dir=str(tmp_path / "p1"), resume_dir=str(tmp_path / "p1-state"),
               keep_final_state=True, **kw)
    res = train(cfg)
    return cfg, res


def _cont(tmp_path, data, T, name, **kw):
    return _cfg(data, T, continue_from=str(tmp_path / "p1-state" / "resume_step0003.pt"),
                continue_steps=6, out_dir=str(tmp_path / name),
                resume_dir=str(tmp_path / (name + "-state")), eval_steps=[5, 6],
                full_eval_steps=[6], keep_final_state=True, **kw)


# --------------------------------------------------------------------------- data order
def test_stream_order_is_the_single_run_permutation_and_prefix_stable():
    n = 25
    assert np.array_equal(stream_order([n], 7, 20), np.random.default_rng(7).permutation(n)[:20])
    long = stream_order([25, 25], 7, 38)
    assert np.array_equal(long[:25], np.random.default_rng(7).permutation(25))
    assert np.array_equal(np.sort(long[25:]) - 25, np.sort(long[25:] - 25)) and long[25:].min() >= 25
    assert np.array_equal(stream_order([25, 25], 7, 19), long[:19])        # horizon prefix
    assert len(np.unique(long)) == len(long)                               # no repeats
    with pytest.raises(ValueError, match="needs"):
        stream_order([25], 7, 26)


def test_stream_rows_matches_concatenation():
    a = np.arange(5 * 4, dtype=np.uint32).reshape(5, 4)
    b = 100 + np.arange(3 * 4, dtype=np.uint32).reshape(3, 4)
    s = StreamRows([a, b])
    cat = np.concatenate([a, b])
    rows = np.array([6, 0, 4, 5, 7])
    assert len(s) == 8 and np.array_equal(s[rows, :3], cat[rows, :3])


# --------------------------------------------------------------------------- LR
def test_lr_boundaries_single_phase_and_continuation():
    cfg = TrainCfg(steps=1200, warmup=120, lr_peak=3e-3, lr_floor=1e-5)

    def closed_form(step):  # single-phase linear warm-up, then cosine to the floor
        if step < cfg.warmup:
            return cfg.lr_floor + (cfg.lr_peak - cfg.lr_floor) * step / cfg.warmup
        t = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
        return cfg.lr_floor + 0.5 * (cfg.lr_peak - cfg.lr_floor) * (1 + math.cos(math.pi * t))
    assert all(lr_at(s, cfg) == closed_form(s) == schedule_lr(s, cfg) for s in range(0, 1202, 7))
    cont = replace(cfg, continue_from="x.pt", continue_steps=2400)
    floor, peak = cfg.lr_floor, cfg.lr_peak
    assert schedule_lr(1201, cont) == lr_at(1, cfg) == floor + (peak - floor) / 120
    assert schedule_lr(1320, cont) == peak                      # re-warm ends at the same peak
    assert schedule_lr(1200 + 120 + 1140, cont) == pytest.approx(floor)   # floor at 2400
    assert schedule_lr(2400, cont) == pytest.approx(floor)
    assert schedule_lr(1320 + 540, cont) == pytest.approx((peak + floor) / 2)
    fb = replace(cont, continue_lr_peak=1e-3)
    assert schedule_lr(1320, fb) == 1e-3 and schedule_lr(1201, fb) == floor + (1e-3 - floor) / 120


# --------------------------------------------------------------------------- end to end
def test_phase1_keep_final_state_then_continuation_reads_the_single_run_rows(tmp_path, monkeypatch):
    data, T = _data(tmp_path)
    stub = StubInjector()
    cfg1, res1 = _phase1(tmp_path, data, T, stub, monkeypatch)
    assert not res1["aborted"] and res1["final_state"]["step"] == 3
    states = sorted(p.name for p in (tmp_path / "p1-state").glob("resume_step*.pt"))
    assert states == ["resume_step0003.pt"]                    # rolling states dropped
    fs = json.loads((tmp_path / "p1" / "final_state.json").read_text())
    from tttmem.p2.data import sha256_file
    assert fs["sha256"] == sha256_file(tmp_path / "p1-state" / "resume_step0003.pt")
    parent = torch.load(tmp_path / "p1-state" / "resume_step0003.pt", weights_only=False)
    assert set(parent["optimizer"]["state"]) and parent["cursor"] == 6 and parent["step"] == 3
    phase1_calls = list(stub.calls)

    cfg2 = _cont(tmp_path, data, T, "c", parity_ref=str(tmp_path / "p1" / "eval_step0003.npz"))
    res2 = train(cfg2)
    assert not res2["aborted"] and (res2["start"], res2["horizon"]) == (3, 6)
    par = [json.loads(l[10:]) for l in (tmp_path / "c" / "events.jsonl").read_text().splitlines()
           if l.startswith("P2_PARITY ")]
    assert len(par) == 1 and par[0]["pass"] and par[0]["step"] == 3 and par[0]["n_docs"] == 4
    assert par[0]["mean_abs_delta"] == {"adapt": 0.0, "noadapt": 0.0}     # same weights, CPU
    cont_calls = stub.calls[len(phase1_calls):]
    # LR: re-warm from the floor over `warmup` steps from the start, floor at the new horizon.
    lr1 = [r["lr"] for r in _steps(tmp_path / "p1")]
    lr2 = [r["lr"] for r in _steps(tmp_path / "c")]
    assert lr1 == [lr_at(s, cfg1) for s in (1, 2, 3)]
    assert lr2 == lr1 == [cfg1.lr_floor + (cfg1.lr_peak - cfg1.lr_floor) / 2, cfg1.lr_peak,
                          cfg1.lr_floor]
    assert [r["step"] for r in _steps(tmp_path / "c")] == [4, 5, 6]
    # Rows: identical (position, content) to a single 6-step run's stream.
    straight_stub = StubInjector()
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: straight_stub)
    train(_cfg(data, T, steps=6, out_dir=str(tmp_path / "s"), resume_dir=str(tmp_path / "s-state")))
    got = phase1_calls + cont_calls
    assert [int(i[0]) for i, _ in got] == list(range(12))
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(got, straight_stub.calls))
    order = stream_order([8, 8], 0, 12)
    stream = _stream(data, T)
    assert all(np.array_equal(rows[0], stream[order[int(i[0])]]) for i, rows in got)
    # Order prefix, provenance and evals.
    final = torch.load(tmp_path / "c-state" / "resume_step0006.pt", weights_only=False)
    assert np.array_equal(final["order"], order) and np.array_equal(final["order"][:6], parent["order"])
    prov = json.loads((tmp_path / "c" / "continuation.json").read_text())
    assert prov["parent_state_sha256"] == fs["sha256"] and prov["rows"] == [6, 12]
    assert prov["parent_config"]["steps"] == 3
    assert {"eval_step0005.npz", "eval_step0006.npz"} <= {p.name for p in (tmp_path / "c").iterdir()}
    assert not (tmp_path / "c" / "eval_step0000.npz").exists()
    assert sum(r["echo_rows"] for r in _steps(tmp_path / "c")) == int(straight_stub.selected(np.arange(6, 12)).sum())


def test_interrupted_continuation_resumes_bitwise(tmp_path, monkeypatch):
    data, T = _data(tmp_path)
    _phase1(tmp_path, data, T, StubInjector(), monkeypatch)
    train(_cont(tmp_path, data, T, "a"))
    train(_cont(tmp_path, data, T, "b", stop_after=4))
    with (tmp_path / "b" / "steps.jsonl").open("a") as f:
        f.write("uncommitted partial record\n")
    train(_cont(tmp_path, data, T, "b", resume=True))
    a = torch.load(tmp_path / "a-state" / "resume_step0006.pt", weights_only=False)
    b = torch.load(tmp_path / "b-state" / "resume_step0006.pt", weights_only=False)
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    for k, st in a["optimizer"]["state"].items():
        for name, v in st.items():
            assert torch.equal(v, b["optimizer"]["state"][k][name]), (k, name)
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert np.array_equal(a["order"], b["order"]) and a["cursor"] == b["cursor"] == 12
    for name in ("steps.jsonl", "events.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    assert "continue_from" in a["config"] and a["config"]["continue_steps"] == 6
    assert sorted(p.name for p in (tmp_path / "b-state").glob("resume_step*.pt")) == ["resume_step0006.pt"]


def test_weights_only_continuation_fresh_adam_and_rows(tmp_path, monkeypatch):
    data, T = _data(tmp_path)
    stub = StubInjector()
    cfg1, res1 = _phase1(tmp_path, data, T, stub, monkeypatch)
    parent = torch.load(tmp_path / "p1-state" / "resume_step0003.pt", weights_only=False)
    weights = tmp_path / "weights.pt"
    torch.save({"model": parent["model"], "step": 3, "arm": "e2e2", "preset": "tiny",
                "init_seed": 0, "data_seed": 0}, weights)
    calls = list(stub.calls)
    cfg = _cfg(tmp_path / "data", T, continue_weights_from=str(weights), continue_steps=6,
               out_dir=str(tmp_path / "w"), resume_dir=str(tmp_path / "w-state"),
               eval_steps=[5, 6], full_eval_steps=[6],
               continue_parent_run_id="pilot", continue_lost_state_sha256="lost")
    res = train(cfg)
    assert not res["aborted"]
    assert [int(x[0][0]) for x in stub.calls[len(calls):]] == list(range(6, 12))
    final = torch.load(tmp_path / "w-state" / "resume_step0006.pt", weights_only=False)
    assert final["cursor"] == 12 and all(int(v["step"]) == 3 for v in final["optimizer"]["state"].values())
    assert "continue_weights_from" in final["config"]
    prov = json.loads((tmp_path / "w" / "continuation.json").read_text())
    assert prov["optimizer"] == "fresh (parent Adam state lost)" and prov["parent_run_id"] == "pilot"


def test_full_state_continuation_of_a_continued_run(tmp_path, monkeypatch):
    """P1 v3x: resume a weights-only continuation (3 -> 6) from its kept final state to 8."""
    data, T = _data(tmp_path)
    stub = StubInjector()
    _phase1(tmp_path, data, T, stub, monkeypatch)
    parent = torch.load(tmp_path / "p1-state" / "resume_step0003.pt", weights_only=False)
    weights = tmp_path / "weights.pt"
    torch.save({"model": parent["model"], "step": 3, "arm": "e2e2", "preset": "tiny",
                "init_seed": 0, "data_seed": 0}, weights)
    assert not train(_cfg(data, T, continue_weights_from=str(weights), continue_steps=6,
                          out_dir=str(tmp_path / "w"), resume_dir=str(tmp_path / "w-state"),
                          eval_steps=[6], keep_final_state=True))["aborted"]
    calls = len(stub.calls)
    x = _cfg(data, T, steps=6, continue_from=str(tmp_path / "w-state" / "resume_step0006.pt"),
             continue_steps=8, out_dir=str(tmp_path / "x"), resume_dir=str(tmp_path / "x-state"),
             eval_steps=[8], keep_final_state=True)
    assert not train(x)["aborted"]
    assert [int(c[0][0]) for c in stub.calls[calls:]] == list(range(12, 16))
    w6 = torch.load(tmp_path / "w-state" / "resume_step0006.pt", weights_only=False)
    x8 = torch.load(tmp_path / "x-state" / "resume_step0008.pt", weights_only=False)
    assert x8["cursor"] == 16
    assert all(int(v["step"]) == int(w6["optimizer"]["state"][k]["step"]) + 2
               for k, v in x8["optimizer"]["state"].items())
    prov = json.loads((tmp_path / "x" / "continuation.json").read_text())
    assert prov["parent_step"] == 6 and prov["rows"] == [12, 16]
    with pytest.raises(ValueError, match="not the parent endpoint"):
        train(replace(x, steps=5, out_dir=str(tmp_path / "x2"), resume_dir=str(tmp_path / "x2-s")))
    with pytest.raises(ValueError, match=r"changes parent training keys: \['echo_p'\]"):
        train(replace(x, echo_p=1.0, out_dir=str(tmp_path / "x3"), resume_dir=str(tmp_path / "x3-s")))


def test_weights_and_full_resume_are_mutually_exclusive(tmp_path):
    data, T = _data(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train(_cfg(data, T, continue_from="a", continue_weights_from="b", continue_steps=6,
                   out_dir=str(tmp_path / "x")))


def test_train_loss_guard_aborts(tmp_path, monkeypatch, capsys):
    data, T = _data(tmp_path)
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: StubInjector())
    res = train(_cfg(data, T, out_dir=str(tmp_path / "g"), resume_dir=str(tmp_path / "g-s"),
                     train_loss_max_rise=-1.0))
    assert res["aborted"] and res["reason"] == "train_loss_divergence"
    assert "P2_ABORT" in capsys.readouterr().out and (tmp_path / "g" / "divergence_abort.json").exists()


def test_max_hours_saves_state_and_resume_completes(tmp_path, monkeypatch, capsys):
    data, T = _data(tmp_path)
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: StubInjector())
    kw = dict(out_dir=str(tmp_path / "m"), resume_dir=str(tmp_path / "m-s"), resume_every=2)
    res = train(_cfg(data, T, max_hours=1e-9, **kw))
    assert res["aborted"] and res["reason"] == "max_hours" and res["step"] == 1
    assert "ABORT max_hours" in capsys.readouterr().out
    assert [p.name for p in (tmp_path / "m-s").glob("resume_step*.pt")] == ["resume_step0001.pt"]
    assert not train(_cfg(data, T, resume=True, **kw))["aborted"]
    train(_cfg(data, T, out_dir=str(tmp_path / "r"), resume_dir=str(tmp_path / "r-s"), resume_every=2))
    a = torch.load(tmp_path / "m-s" / "resume_step0003.pt", weights_only=False)
    b = torch.load(tmp_path / "r-s" / "resume_step0003.pt", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_continuation_guards(tmp_path, monkeypatch):
    data, T = _data(tmp_path)
    _phase1(tmp_path, data, T, StubInjector(), monkeypatch)
    with pytest.raises(ValueError, match=r"changes parent training keys: \['lr_peak'\]"):
        train(_cont(tmp_path, data, T, "g1", lr_peak=5e-3, continue_lr_peak=5e-3))
    with pytest.raises(ValueError, match="continue_lr_peak"):
        train(_cont(tmp_path, data, T, "g2", continue_lr_peak=2e-3))
    with pytest.raises(ValueError, match="changes parent training keys"):
        train(_cont(tmp_path, data, T, "g3", echo_p=1.0))
    state = torch.load(tmp_path / "p1-state" / "resume_step0003.pt", weights_only=False)
    state["step"] = 2
    torch.save(state, tmp_path / "early.pt")
    with pytest.raises(ValueError, match="not the parent endpoint"):
        train(replace(_cont(tmp_path, data, T, "g4"), continue_from=str(tmp_path / "early.pt")))
    with pytest.raises(ValueError, match="require continue_from or continue_weights_from"):
        train(_cfg(data, T, continue_steps=6, out_dir=str(tmp_path / "g5")))
    assert {"echo_f", "echo_p", "echo_seed", "data_seed", "extra_train_splits", "lr_peak",
            "micro_batch"} <= set(CONTINUATION_IDENTITY)


# --------------------------------------------------------------------------- injection
def test_real_injector_in_prefetcher_is_deterministic_and_keeps_order(tmp_path):
    T1 = 8193
    rng = np.random.default_rng(0)
    arr = rng.integers(1000, 30000, size=(12, T1)).astype(np.uint32)
    arr[:, 0] = 128000
    order = np.random.default_rng(3).permutation(12)
    inj = make_injector(split="train", f=.5, p=.5, seed=0, T=8192,
                        excluded_vocab=np.zeros(128256, dtype=bool))

    def drain(pf):
        out, counts = [], []
        for t in pf:
            out.append(t.numpy())
            counts.append(pf.last_injected)
        pf.close()
        return out, counts
    plain, _ = drain(Prefetcher(arr, order, 2, "cpu", width=T1))
    inj1, c1 = drain(Prefetcher(arr, order, 2, "cpu", width=T1, injector=inj))
    inj2, c2 = drain(Prefetcher(arr, order[4:], 2, "cpu", width=T1, injector=inj, base=4))
    sel = soq_p1.selected_mask(inj, 12)
    assert 0 < sel.sum() < 12
    for k, (p, q) in enumerate(zip(plain, inj1)):
        assert np.array_equal(p, arr[np.sort(order[2 * k:2 * k + 2])])          # order unchanged
        changed = (p != q).any(1)
        assert np.array_equal(changed, sel[2 * k:2 * k + 2])                    # injected iff selected
        assert c1[k] == int(changed.sum())
    assert all(np.array_equal(a, b) for a, b in zip(inj1[2:], inj2))           # resume-invariant
    assert c1[2:] == c2


def test_exposure_ledger_and_checks():
    inj = make_injector(split="train", f=.25, p=.5, seed=0, T=8192,
                        excluded_vocab=np.zeros(128256, dtype=bool))
    led = soq_p1.exposure(inj, 1200 * 16, 8193)
    assert led["injected_rows"] == int(soq_p1.selected_mask(inj, 19200).sum())
    assert abs(led["injected_rows"] / 19200 - .25) < .02
    assert led["source_writes"]["copy_available"] == led["source_writes"]["copy_blocked"] == 4 * led["injected_rows"]
    assert 0 < led["tokens_per_injected_row"] < 8192 and led["token_share"] < .05
    acc = inj.exposure_accounting(np.arange(1200 * 16, dtype=np.int64))      # exact per-row count
    assert led["injected_tokens"] == acc["injected_token_budget"]
    assert led["token_share"] == pytest.approx(acc["token_share"])
    soq_p1.check_exposure(led)
    with pytest.raises(ValueError, match="source writes"):
        soq_p1.check_exposure(soq_p1.exposure(inj, 100, 8193))
    man = soq_p1.compact_manifest(inj)
    assert "pools" not in man and len(man["pools_sha256"]) == 64 and len(man["params_sha256"]) == 64


def test_injection_does_not_change_stored_order_or_clean_evals(tmp_path, monkeypatch):
    data, T = _data(tmp_path)
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: StubInjector() if cfg.echo_f > 0 else None)
    train(_cfg(data, T, echo_f=.5, out_dir=str(tmp_path / "e"), resume_dir=str(tmp_path / "e-s"),
               eval_at_init=True, eval_steps=[]))
    train(_cfg(data, T, echo_f=0.0, out_dir=str(tmp_path / "n"), resume_dir=str(tmp_path / "n-s"),
               eval_at_init=True, eval_steps=[]))
    a = torch.load(tmp_path / "e-s" / "resume_step0003.pt", weights_only=False)
    b = torch.load(tmp_path / "n-s" / "resume_step0003.pt", weights_only=False)
    assert np.array_equal(a["order"], b["order"])
    za, zb = np.load(tmp_path / "e" / "eval_step0000.npz"), np.load(tmp_path / "n" / "eval_step0000.npz")
    assert np.array_equal(za["doc_adapt"], zb["doc_adapt"])                 # clean documents
    assert "echo_manifest" in a["data_sha256"] and "echo_manifest" not in b["data_sha256"]
    ev = (tmp_path / "e" / "events.jsonl").read_text()
    assert "P2_ECHO_EXPOSURE" in ev and "P2_ECHO_EXPOSURE" not in (tmp_path / "n" / "events.jsonl").read_text()


# --------------------------------------------------------------------------- echo reads
def test_return_greedy_sparse_matches_dense():
    from tttmem.p2.meta import InnerCfg, stream_eval
    from tttmem.p2.model import build_model
    torch.manual_seed(0)
    m = build_model("tiny", compute_dtype=torch.float32, seed=3)
    ids = torch.randint(3, 97, (2, 65))
    x, y = ids[:, :-1], ids[:, 1:]
    mask = torch.ones_like(x, dtype=torch.bool)
    sel = torch.zeros_like(mask)
    sel[:, [3, 10, 30, 50, 60, 63]] = True
    for adapt in (True, False):
        inner = InnerCfg(rule="ce", lr=1.0, clip=1.0)
        d = stream_eval(m, x, y, mask, inner, adapt=adapt, barriers=[2, 4], fused_attn=False,
                        return_greedy=True)
        s = stream_eval(m, x, y, mask, inner, adapt=adapt, barriers=[2, 4], fused_attn=False,
                        nll_positions=sel, return_greedy=True)
        assert torch.allclose(d["nll"][sel], s["nll"][sel], atol=1e-5)
        assert torch.equal(d["greedy"][sel], s["greedy"][sel])
    # Without the flag nothing changes in the result keys.
    assert "greedy" not in stream_eval(m, x, y, mask, InnerCfg(rule="ce"), barriers=[2, 4],
                                       fused_attn=False)


def _tiny_echo_set(rows):
    n = len(rows)
    slot = base_off = (np.arange(8)[:, None] % 4) * 3 + np.arange(3)[None, :]   # [8, 3], 0..11
    pos = lambda base: np.broadcast_to(base + base_off, (n, 8, 3)).copy()  # noqa: E731
    assert slot.max() == 11
    av = np.zeros((n, 8), dtype=bool)
    av[:, :4] = True
    positions = {"source": pos(2), "query": pos(40), "no_binding": pos(48), "no_barrier": pos(20),
                 "query_entity": pos(52)}
    vals = (np.arange(n * 8 * 3, dtype=np.int64).reshape(n, 8, 3) + 10)
    keys = {"query_key_positions": pos(12), "no_binding_key_positions": pos(16)}
    return soq_p1.EchoEvalSet(docs=np.asarray(rows, dtype=np.uint32), available=av,
                              positions=positions, row_ids=np.arange(n), docs_sha256="x" * 64,
                              manifest={"params_sha256": "stub"}, value_tokens=vals,
                              key_positions=keys)


def test_echo_eval_in_training_and_local_analysis(tmp_path, monkeypatch, capsys):
    data, T = _data(tmp_path)
    from tttmem.p2.data import open_split
    ho, _ = open_split(data, "ho")
    es = _tiny_echo_set(np.asarray(ho[:3, :T + 1]))
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: StubInjector())
    monkeypatch.setattr(soq_p1, "build_echo_eval_set", lambda cfg, ex: es)
    res = train(_cfg(data, T, echo_eval_docs=3, eval_at_init=True, eval_steps=[3],
                     out_dir=str(tmp_path / "o"), resume_dir=str(tmp_path / "o-s")))
    assert not res["aborted"]
    z = np.load(tmp_path / "o" / "eval_step0003.npz")
    for read in soq_p1.READS:
        assert z[f"echo_{read}_nll"].shape == (3, 8) and np.isfinite(z[f"echo_{read}_nll"]).all()
        assert z[f"echo_{read}_em"].dtype == bool
    assert np.isfinite(z["doc_adapt"]).all()                                  # clean evals intact
    ev = [json.loads(l[8:]) for l in (tmp_path / "o" / "events.jsonl").read_text().splitlines()
          if l.startswith("P2_ECHO ")]
    assert [e["step"] for e in ev] == [0, 3] and set(ev[0]["recall"]) == set(soq_p1.STRATA)
    # The echo read equals a direct stream_eval at the same positions.
    from tttmem.p2.model import build_model
    model = build_model("tiny", compute_dtype=torch.float32, seed=0)
    arrays, summ = soq_p1.echo_read(model, es, rule="ce", inner_lr=1.0, inner_clip=1.0,
                                    barriers=[2, 4], device="cpu", fused_attn=False, batch=2)
    diag_arrays, diag_summ = soq_p1.echo_read(model, es, rule="ce", inner_lr=1.0, inner_clip=1.0,
                                              barriers=[2, 4], device="cpu", fused_attn=False,
                                              batch=2, diagnostics=True)
    assert diag_arrays["echo_query_first_token_logprobs"].shape == (3, 8, 8)
    assert diag_arrays["echo_no_binding_first_token_rank"].shape == (3, 8)
    assert diag_arrays["echo_query_first_nll"].shape == (3, 8)
    assert "first_token_discrimination" in diag_summ
    from tttmem.p2.meta import InnerCfg, stream_eval
    from tttmem.p2.probe import boundary_valid
    t = torch.from_numpy(es.docs.astype(np.int64))
    mask = (t[:, 1:] != 1) & torch.from_numpy(boundary_valid(T, 8, [2, 4]))
    d = stream_eval(model, t[:, :-1], t[:, 1:], mask, InnerCfg(rule="ce", lr=1.0, clip=1.0),
                    barriers=[2, 4], fused_attn=False, return_greedy=True)
    q = es.positions["query"]
    rows = np.arange(3)[:, None, None]
    assert np.allclose(arrays["echo_query_nll"], d["nll"].numpy()[rows, q].mean(-1), atol=1e-5)
    assert np.array_equal(arrays["echo_query_em"], d["greedy"].numpy()[rows, q].all(-1))
    assert summ["recall"]["copy_available"] == pytest.approx(arrays["echo_query_em"][:, :4].mean())
    # With a write mask the read equals stream_eval with the same inner mask.
    wm = np.ones((3, T), dtype=bool)
    wm[:, 20:32] = False
    es_m = soq_p1.EchoEvalSet(**{**es.__dict__, "write_mask": wm})
    arr_m, _ = soq_p1.echo_read(model, es_m, rule="ce", inner_lr=1.0, inner_clip=1.0,
                                barriers=[2, 4], device="cpu", fused_attn=False, batch=2)
    dm = stream_eval(model, t[:, :-1], t[:, 1:], mask, InnerCfg(rule="ce", lr=1.0, clip=1.0),
                     barriers=[2, 4], inner_mask=torch.from_numpy(wm), fused_attn=False,
                     return_greedy=True)
    assert np.allclose(arr_m["echo_query_nll"], dm["nll"].numpy()[rows, q].mean(-1), atol=1e-5)
    assert not np.allclose(arr_m["echo_query_nll"], arrays["echo_query_nll"])
    # Local analysis entry point on the smoke outputs.
    capsys.readouterr()
    out = soq_p1_analysis.main([str(tmp_path / "o"), "--out", str(tmp_path / "a.json"),
                                "--md", str(tmp_path / "a.md")])
    assert out["endpoint"] == 3 and out["v2_endpoint"]["verdict"] in ("PASS", "FAIL")
    assert set(out["steps"]) == {"0", "3"} and out["exposure"]["injected_rows"] > 0
    assert "SOQ_P1_V2" in capsys.readouterr().out and (tmp_path / "a.md").exists()


def test_v2_rule_on_synthetic_reads():
    n = 200
    z = {"echo_available": np.tile(np.arange(8) < 4, (n, 1))}
    for r in soq_p1.READS:
        z[f"echo_{r}_nll"] = np.zeros((n, 8))
        z[f"echo_{r}_em"] = np.zeros((n, 8), dtype=bool)
    z["echo_query_em"][:, 4:] = np.random.default_rng(0).random((n, 4)) < .5   # blocked recall ~.5
    assert soq_p1_analysis.v2(soq_p1_analysis.doc_reads(z))["verdict"] == "PASS"
    z["echo_no_binding_em"][:] = np.random.default_rng(1).random((n, 8)) < .4   # margin ~.1
    assert soq_p1_analysis.v2(soq_p1_analysis.doc_reads(z))["verdict"] == "FAIL"


def test_first_token_discrimination_excludes_collisions_and_segments():
    n = 2
    av = np.tile(np.arange(8) < 4, (n, 1))
    a = {"echo_available": av}
    for r in soq_p1.READS:
        a[f"echo_{r}_nll"] = np.ones((n, 8), dtype=np.float32)
        a[f"echo_{r}_em"] = np.zeros((n, 8), dtype=bool)
        a[f"echo_{r}_first_nll"] = np.ones((n, 8), dtype=np.float32)
        a[f"echo_{r}_rest_nll"] = np.ones((n, 8), dtype=np.float32)
    for read in ("query", "no_binding"):
        a[f"echo_{read}_first_token_rank"] = np.ones((n, 8), dtype=np.int16)
        a[f"echo_{read}_first_token_collision"] = np.zeros((n, 8), dtype=bool)
    a["echo_query_first_token_collision"][0, 0] = True
    a["echo_query_segment"] = np.tile(np.arange(8) < 4, (n, 1))
    s = soq_p1.summarize(a)
    assert s["first_token_discrimination"]["query"]["accuracy"] == pytest.approx(1.0)
    assert s["first_token_discrimination"]["query"]["excluded_fraction"] == pytest.approx(1/16)
    assert set(s["first_token_discrimination"]) >= {"copy_available", "copy_blocked"}


def test_build_echo_eval_set_uses_generator_test_templates(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    rows = np.random.default_rng(0).integers(1000, 30000, size=(6, 8193))
    rows[:, 0] = 128000
    _write_split(root, "ho", rows)
    excl = np.zeros(128256, dtype=bool)
    cfg = TrainCfg(seq_len=8192, segment_len=2048, data_dir=str(root), echo_eval_docs=4,
                   echo_eval_offset=2, echo_seed=0)
    es = soq_p1.build_echo_eval_set(cfg, excl)
    ref, meta = build_eval_documents(rows[2:6].astype(np.uint32), split="test", p=.5, seed=0,
                                     excluded_vocab=excl, row_ids=np.arange(2, 6))
    assert np.array_equal(es.docs, ref) and list(es.row_ids) == [2, 3, 4, 5]
    assert es.available.shape == (4, 8) and (es.available.sum(1) == 4).all()
    assert np.array_equal(es.positions["query"][0], meta["rows"][0]["query_nll_positions"])
    assert es.manifest["split"] == "test"
    verb, vmeta = build_eval_documents(rows[2:6].astype(np.uint32), split="test", p=.5, seed=0,
                                       excluded_vocab=excl, row_ids=np.arange(2, 6),
                                       query_mode="verbatim")
    assert verb.shape == ref.shape and vmeta["rows"][0]["query_mode"] == "verbatim"
    # Control sentences do not write: the inner mask is False exactly on their targets.
    from tttmem.p2.echo_text import CONTROL_KINDS, write_mask
    assert np.array_equal(es.write_mask, write_mask(meta["rows"], 8192))
    for j, rec in enumerate(meta["rows"]):
        ctrl = [sp for sp in rec["spans"] if sp[2] in CONTROL_KINDS]
        assert len(ctrl) == 24 and (~es.write_mask[j]).sum() == sum(e - s for s, e, _, _ in ctrl)
        assert es.write_mask[j][es.positions["source"][j]].all()
        assert es.write_mask[j][es.positions["query"][j]].all()
        assert not es.write_mask[j][es.positions["no_binding"][j]].any()


def test_parity_check(tmp_path):
    sha = np.array([f"d{i}" for i in range(5)])
    np.savez(tmp_path / "ref.npz", doc_adapt=np.ones(5), doc_noadapt=np.full(5, 2.0), eval_sha1=sha)
    good = parity_check({"adapt": np.ones(4) + .005, "noadapt": np.full(4, 2.0)}, sha,
                        str(tmp_path / "ref.npz"), 64, .01)
    assert good["pass"] and good["n_docs"] == 4
    assert not parity_check({"adapt": np.ones(4) + .02, "noadapt": np.full(4, 2.0)}, sha,
                            str(tmp_path / "ref.npz"), 64, .01)["pass"]
    bad = parity_check({"adapt": np.ones(4)}, sha[::-1], str(tmp_path / "ref.npz"), 64, .01)
    assert not bad["pass"] and not bad["same_documents"]


def test_parity_abort_stops_before_training(tmp_path, monkeypatch, capsys):
    data, T = _data(tmp_path)
    monkeypatch.setattr(soq_p1, "make_train_injector", lambda cfg, ex: StubInjector())
    from tttmem.p2.data import open_split
    _, man = open_split(data, "eval")
    np.savez(tmp_path / "ref.npz", doc_adapt=np.zeros(6), doc_noadapt=np.zeros(6),
             eval_sha1=np.array([m["sha1"] for m in man]))
    res = train(_cfg(data, T, eval_at_init=True, parity_ref=str(tmp_path / "ref.npz"),
                     out_dir=str(tmp_path / "o"), resume_dir=str(tmp_path / "o-s")))
    assert res["aborted"] and res["reason"] == "parity"
    assert "ABORT parity" in capsys.readouterr().out
    assert not list((tmp_path / "o-s").glob("resume_step*.pt"))


# --------------------------------------------------------------------------- configs
def _load(path):
    with open(REPO / path, "rb") as f:
        return tomllib.load(f)


EXCLUDED = Path(__file__).resolve().parents[1] / "data/p2data" / "excluded_vocab.npy"


@pytest.mark.skipif(not EXCLUDED.exists(), reason="local copy of p2data/excluded_vocab.npy absent")
def test_registered_manifest_matches_current_generator():
    """Fails when echo_text changes without re-registering configs/soq/p1-pilot.toml."""
    p = TrainCfg.from_dict(_load("configs/soq/p1-pilot.toml")["p2"])
    inj = soq_p1.make_train_injector(p, np.load(EXCLUDED))
    assert soq_p1.manifest_id(soq_p1.compact_manifest(inj)) == p.echo_manifest_sha256


def test_pilot_and_continuation_configs():
    root, fo = _load("configs/soq/p1v3-so.toml"), _load("configs/soq/p1v3-fo.toml")
    a, b = TrainCfg.from_dict(root["p2"]), TrainCfg.from_dict(fo["p2"])
    assert a.echo_version == b.echo_version == 3 and a.echo_f == b.echo_f == .5
    assert a.echo_lookup_share == b.echo_lookup_share == .25 and a.echo_slots == b.echo_slots == 16
    assert a.arm == "e2e2" and b.arm == "e2e1-ce"
    assert a.process_name == "TTM-pv3-04" and b.process_name == "TTM-pv3-02"
    assert a.out_dir.endswith("so-s0-a1g") and b.out_dir.endswith("fo-s0-a1g")
    assert a.pilot_max_rise == b.pilot_max_rise == .5 and a.train_loss_max_rise == b.train_loss_max_rise == .5
    assert a.attempt == b.attempt == 1 and a.peak() == b.peak() == 1e-3
    assert a.max_hours == 3.6 and b.max_hours == 1.9
    keys = set(root["p2"]) | set(fo["p2"])
    common_diff = {k for k in keys if root["p2"].get(k) != fo["p2"].get(k)}
    assert common_diff <= {"arm", "process_name", "out_dir", "max_hours", "estimate_wall_s",
                           "continue_weights_from", "continue_parent_run_id", "parity_ref"}


def test_v3x_resume_config():
    """The p1v3p config is post-hoc-only while retaining the v3x model/eval contract."""
    so, x = _load("configs/soq/p1v3-so.toml")["p2"], _load("configs/soq/p1v3p.toml")["p2"]
    c = TrainCfg.from_dict(x)
    assert x["mode"] == "soq_p1v3_post" and c.process_name == "TTM-pv3-06" and c.out_dir.endswith("so-s0-a1g-x3200")
    assert x["post_out_dir"] == "outputs/soq/p1v3/post" and [p["step"] for p in x["post_checkpoints"]] == [2400, 3200]
    assert c.continue_from.endswith("so-s0-a1g/resume_step2400.pt") and not c.continue_weights_from
    assert (c.steps, c.horizon(), c.start(), c.peak(), c.attempt) == (2400, 3200, 2400, 1e-3, 1)
    assert c.horizon() * c.global_batch == 51200          # A + B rows: the data limit
    assert c.echo_eval_steps == [2400, 2800, 3200] and c.eval_steps == [2800, 3200]
    assert c.parity_ref.endswith("so-s0-a1g/eval_step2400.npz")
    assert {k for k in CONTINUATION_IDENTITY if k != "steps" and so.get(k) != x.get(k)} == set()
    assert {k for k in set(so) | set(x) if so.get(k) != x.get(k)} == {
        "mode", "process_name", "out_dir", "steps", "eval_steps", "full_eval_steps", "pilot_steps",
        "continue_weights_from", "continue_from", "continue_parent_run_id",
        "continue_lost_state_sha256", "continue_steps", "parity_ref", "echo_eval_steps",
        "max_hours", "estimate_wall_s", "post_checkpoints", "post_out_dir"}
