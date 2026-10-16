"""P2-1 launch gates and restart invariants; all data/checkpoints are synthetic."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem.p2 import data as D, decide
from tttmem.p2.train import TrainCfg, train
from test_p2 import _decision_dirs, _smoke_data


def _futility_dirs(tmp_path, delta):
    roots = []
    n = 256
    identities = dict(eval_sha1=np.array([f"doc{i}" for i in range(n)]),
                      eval_domain=np.array([f"d{i % 32}.org" for i in range(n)]),
                      eval_rows=np.arange(n))
    for name, offset in (("so", 0), ("fo", delta)):
        root = tmp_path / name
        root.mkdir()
        np.savez(root / "eval_meta.npz", **identities)
        np.savez(root / "eval_step0100.npz", doc_adapt=np.ones(n) + offset, **identities)
        roots.append(root)
    return roots


@pytest.mark.parametrize("delta,launch", [(0.01, False), (0.1, True), (0.05, True)])
def test_futility_upper_bound_and_write_once(tmp_path, delta, launch):
    so, fo = _futility_dirs(tmp_path, delta)
    result = decide.futility(so, fo, tmp_path / "decision")
    assert result["launch_rule_arms"] == launch
    assert result["U_100"] == pytest.approx(delta)
    assert result["n_boot"] == 10000 and result["seed"] == 0
    assert len(result["inputs"]) == 4
    with pytest.raises(FileExistsError):
        decide.futility(so, fo, tmp_path / "decision")


@pytest.mark.parametrize("bad", ["missing", "nan", "order", "domain", "short", "duplicate"])
def test_futility_invalid_inputs_launch_rule_arms(tmp_path, bad):
    so, fo = _futility_dirs(tmp_path, 0.01)
    path = fo / "eval_step0100.npz"
    with np.load(path) as z:
        arrays = dict(z)
    if bad == "missing":
        path.unlink()
    else:
        if bad == "nan":
            arrays["doc_adapt"][10] = np.nan
        if bad == "order":
            arrays["eval_sha1"] = arrays["eval_sha1"][::-1]
        if bad == "domain":
            arrays["eval_domain"][0] = "other.org"
        if bad == "short":
            arrays["doc_adapt"] = arrays["doc_adapt"][:-1]
        if bad == "duplicate":
            arrays["eval_sha1"][0] = arrays["eval_sha1"][1]
        np.savez(path, **arrays)
    r = decide.futility(so, fo, tmp_path / "decision")
    assert r["launch_rule_arms"] and r["U_100"] is None
    assert r["reason"].startswith("missing_or_invalid")


def test_futility_uses_cluster_bootstrap(tmp_path):
    so, fo = _futility_dirs(tmp_path, 0.03)
    path = fo / "eval_step0100.npz"
    with np.load(path) as z:
        arrays = dict(z)
    arrays["doc_adapt"] += (np.arange(256) % 32) * .01
    np.savez(path, **arrays)
    from tttmem.p2.probe import boot_mean, bootstrap_weights
    _, draws = boot_mean(arrays["doc_adapt"] - 1, bootstrap_weights(arrays["eval_domain"]))
    r = decide.futility(so, fo, tmp_path / "decision")
    assert r["U_100"] == np.quantile(draws, .95)


def test_sizing_and_endpoint_after_futility(tmp_path):
    arms = _decision_dirs(tmp_path)
    del arms["e2e1-lin"]
    out = tmp_path / "decision"
    with pytest.raises(ValueError, match="recorded futility"):
        decide.sizing(arms, decision_dir=out, ho_n=80, n_sim=30)
    out.mkdir()
    (out / "futility.json").write_text(json.dumps(dict(launch_rule_arms=False)))
    result = decide.sizing(arms, decision_dir=out, Ns=[40, 80], ho_n=80, n_sim=30)
    assert result["conjuncts"] == [1, 4]
    assert set(result["alternatives"]) == {"D", "dM"}
    final = decide.endpoint(arms, out, n_boot=100)
    assert final["G2"] is None and not final["evaluable"]
    assert final["D"]["est"] == pytest.approx(.1)
    assert final["paired_change_D"]["est"] == pytest.approx(.1)
    assert final["paired_change_delta_M"]["est"] == pytest.approx(.125)


@pytest.mark.parametrize("dtype", ["fp32", "fp64"])
@pytest.mark.parametrize("arm", ["e2e2", "e2e1-ce", "e2e1-lin", "e2e1-aice",
                                 "e2e1-rp", "e2e1-nc"])
def test_resume_matches_straight_parameters_optimizer_rng_and_logs(tmp_path, dtype, arm):
    data, T = _smoke_data(tmp_path)
    base = TrainCfg(arm=arm, preset="tiny", dtype=dtype, device="cpu", seq_len=T,
                    global_batch=2, micro_batch=1, steps=4, warmup=3,
                    data_dir=str(data), n_eval=4, min_probes=1, assay_gate=False,
                    ho_docs=3, ho_trajectory_docs=2, eval_batch=2, pilot_docs=2,
                    pilot_steps=[1, 3], pilot_max_rise=1e9,
                    eval_steps=[2, 4], full_eval_steps=[0, 4], sizing_steps=[2],
                    trajectory_docs=2, crossover_rules=[], resume_every=2,
                    log_every=3, save_checkpoints=False)
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    train(replace(base, out_dir=str(straight), resume_dir=str(tmp_path / "straight-state")))
    train(replace(base, out_dir=str(split), stop_after=2,
                 resume_dir=str(tmp_path / "split-state")))
    # Simulate an interrupted append ahead of the committed checkpoint.
    with (split / "steps.jsonl").open("a") as f:
        f.write("uncommitted partial record\n")
    train(replace(base, out_dir=str(split), resume=True,
                 resume_dir=str(tmp_path / "split-state")))
    a = torch.load(tmp_path / "straight-state" / "resume_step0004.pt", weights_only=False)
    b = torch.load(tmp_path / "split-state" / "resume_step0004.pt", weights_only=False)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    for key, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][key][name]), (key, name)
    assert a["cursor"] == b["cursor"] == 8
    assert np.array_equal(a["order"], b["order"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["pilot"] == b["pilot"]
    for name in ("steps.jsonl", "events.jsonl"):
        assert (straight / name).read_bytes() == (split / name).read_bytes()
    logged = [json.loads(line.removeprefix("P2_STEP "))
              for line in (straight / "steps.jsonl").read_text().splitlines()]
    assert logged[0]["step"] == 3 and logged[0]["lr"] == base.lr_peak
    assert logged[-1]["step"] == 4 and logged[-1]["lr"] == base.lr_floor
    assert len(list((tmp_path / "split-state").glob("resume_step*.pt"))) == 2
    assert not list(split.glob("*.tmp"))


def test_cgroup_worker_limit(tmp_path, monkeypatch):
    path = tmp_path / "cpu.max"
    monkeypatch.setattr(D.os, "sched_getaffinity", lambda _: set(range(112)), raising=False)
    path.write_text("2380000 100000\n")
    assert D.cpu_quota(path) == 23
    path.write_text("50000 100000\n")
    assert D.cpu_quota(path) == 1
    path.write_text("max 100000\n")
    assert D.cpu_quota(path) == 112


def test_packing_and_preencoded_layout(tmp_path):
    root = tmp_path / "store"
    doc = dict(text="z" * 100, _ids=np.arange(D.SEQ - 1, dtype=np.uint32),
               _n_tokens_full=D.SEQ + 50, url="https://a.org", row=0, shard="offline.parquet")

    def should_not_tokenize(_):
        raise AssertionError("already tokenized")

    result = D.prepare([doc], should_not_tokenize, root, {"A": 1}, min_chars=1)
    rows, metadata = D.open_split(root, "A")
    assert metadata[0]["n_tokens_full"] == D.SEQ + 51
    assert result["counts"] == {"A": 1}
    assert rows[0, 0] == D.BOS and rows[0, -1] == D.SEQ - 2
    archive = tmp_path / "p2data.tar"
    digest = D.pack_store(root, archive)
    assert digest == D.sha256_file(archive)
    import tarfile
    with tarfile.open(archive) as tar:
        assert "p2data/A.bin" in tar.getnames()
    assert archive.with_suffix(".tar.sha256").read_text() == f"{digest}  p2data.tar\n"


def test_sizing_waits_for_every_launched_arm_and_freezes_its_hash(tmp_path):
    import shutil
    arms = _decision_dirs(tmp_path)
    aice = tmp_path / "e2e1-aice"
    shutil.copytree(arms["e2e1-lin"], aice)
    arms["e2e1-aice"] = aice
    step = aice / "eval_step0060.npz"
    backup = step.read_bytes()
    step.unlink()
    with pytest.raises(FileNotFoundError):
        decide.sizing(arms, decision_dir=tmp_path / "decisions", Ns=[40], ho_n=80, n_sim=30)
    step.write_bytes(backup)
    size = decide.sizing(arms, decision_dir=tmp_path / "decisions", Ns=[40], ho_n=80, n_sim=30,
                         alpha=.025)
    assert "e2e1-aice" in size["meta_sha1"] and size["alpha"] == .025
    assert str(step.resolve()) in size["inputs"]
    with pytest.raises(ValueError, match="alpha must match"):
        decide.endpoint(arms, tmp_path / "decisions", n_boot=100)
    assert decide.endpoint(arms, tmp_path / "decisions", n_boot=100, alpha=.025)["G2"]


def test_resume_restores_rng_and_ignores_incomplete_temp_state(tmp_path):
    import random
    from tttmem.p2.train import _rng_state, _restore_rng, _latest_state
    state = _rng_state("cpu")
    expected = (random.random(), np.random.random(), torch.rand(4))
    _restore_rng(state, "cpu")
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(4), expected[2])
    payload = dict(step=2)
    torch.save(payload, tmp_path / "resume_step0002.pt")
    (tmp_path / "resume_step0004.pt.tmp").write_bytes(b"partial write")
    (tmp_path / "resume_step0003.pt").write_bytes(b"corrupt")
    recovered, path = _latest_state(tmp_path)
    assert recovered == payload and path.name == "resume_step0002.pt"


def test_p21_templates_match_fixed_contract_and_5090_batches():
    import tomllib
    root = Path(__file__).resolve().parents[1] / "configs" / "p21"
    for arm, mb in (("e2e2", 1), ("e2e1-ce", 2), ("e2e1-lin", 2), ("e2e1-aice", 2)):
        cfg = tomllib.loads((root / f"{arm}.toml").read_text())
        assert cfg["experiment"] == "p2"
        train_cfg = TrainCfg.from_dict(cfg["p2"])
        assert train_cfg.micro_batch == mb and train_cfg.global_batch == 64
        # e2e2.toml uses the attempt-0 settings; the FO arms use the attempt-1 peak 1e-3
        peak, attempt = (.003, 0) if arm == "e2e2" else (.001, 1)
        assert (train_cfg.steps, train_cfg.warmup, train_cfg.lr_peak, train_cfg.lr_floor,
                train_cfg.attempt) == (400, 30, peak, .00001, attempt)
        assert train_cfg.init == "data/ckpt/init.pt" and train_cfg.resume_every == 25
        assert train_cfg.eval_batch == 4 and not train_cfg.so_fused_forward
    prep = D.PrepareCfg.from_dict(tomllib.loads((root / "prepare.toml").read_text())["p2"])
    assert sum(prep.quotas.values()) == 55168
    assert len(prep.dataset_revision) == len(prep.tokenizer_revision) == 40


def test_smoke_coordinator_and_checkpoint_retention_without_gpu(tmp_path, monkeypatch):
    """Exercise the complete rehearsal orchestration with synthetic arm evidence.

    Training kernels are covered by the exact resume tests; no CUDA call is made here.
    """
    from tttmem.p2 import smoke as S
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text("{}")
    monkeypatch.setattr(S.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(S.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(S.torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(S.torch.cuda, "empty_cache", lambda: None)
    calls = []

    def fake_train(cfg):
        calls.append(cfg)
        root = Path(cfg.out_dir)
        root.mkdir(exist_ok=True)
        torch.save(dict(model={"weight": torch.ones(1)}), root / "resume_step0002.pt")
        torch.save(dict(model={"weight": torch.ones(1)}), root / "resume_step0003.pt")
        (root / "steps.jsonl").write_text("".join("P2_STEP " + json.dumps({"step": s, "lr": s * 1e-4}) + "\n"
                                                  for s in (1, 2, 3)))
        if cfg.pilot_max_rise < 0:
            return dict(aborted=True, final=bool(cfg.attempt))
        ids = dict(eval_rows=np.arange(8), eval_sha1=np.array([f"d{i}" for i in range(8)]),
                   eval_domain=np.array([f"d{i}.org" for i in range(8)]),
                   ho_sha1=np.array([f"h{i}" for i in range(8)]),
                   ho_domain=np.array([f"h{i}.org" for i in range(8)]))
        np.savez(root / "eval_meta.npz", **ids)
        loss, memory = {"e2e2": (1., .15), "e2e1-ce": (1.1, .025),
                        "e2e1-lin": (1.01, .125), "e2e1-aice": (1.01, .125)}[cfg.arm]
        for step in (0, 2, 3):
            l = np.full(8, 1. if step == 0 else loss)
            m = .10 if step == 0 else memory
            np.savez(root / f"eval_step{step:04d}.npz", **ids, doc_adapt=l,
                     doc_noadapt=l + .2, doc_fP_focal=l + m, doc_fP_nonfocal=l,
                     doc_heldout=np.ones(8))
        return dict(aborted=False, steps=cfg.stop_after or cfg.steps)

    monkeypatch.setattr(S, "train", fake_train)
    out = tmp_path / "smoke"
    result = S.smoke(dict(init="synthetic", data_dir=str(data), out_dir=str(out)))
    assert result["smoke"] and not result["scientific_verdict"]
    assert result["resume"]["identical"] and result["resume"]["max_abs"] == 0
    assert result["resume"]["same_lr"] and result["resume"]["noise_max_abs"] == 0
    assert result["sizing"]["smoke"] and result["futility"]["n_docs"] == 8
    assert len(list(out.rglob("resume_step*.pt"))) == 2
    assert len(calls) == 9 and calls[4].stop_after == 2 and calls[5].resume
    assert calls[6].out_dir.endswith("resume-twin") and not calls[6].resume
    assert [c.attempt for c in calls[7:]] == [0, 1]


def test_rule_lr_scale_validation_and_lookup():
    from tttmem.p2.train import check_rule_lr_scale
    cfg = TrainCfg(rule_lr_scale={"lin": .125})
    assert cfg.rule_lr("lin") == .125 and cfg.rule_lr("aice") == cfg.rule_lr("ce") == 1.0
    check_rule_lr_scale({"lin": .5, "aice": .25})
    for bad in ({"ce": .5}, {"lin": 0}, {"aice": -1.0}, {"lin": float("nan")}, {"lin": "x"}):
        with pytest.raises(ValueError):
            check_rule_lr_scale(bad)


def test_rule_lr_scale_reaches_training_and_every_evaluation(tmp_path, monkeypatch):
    """The per-rule inner-lr multiplier applies in training and in own/crossover evaluation;
    a multiplier of 1 is bitwise identical to no multiplier."""
    from tttmem.p2 import evaluate as E
    data, T = _smoke_data(tmp_path)
    base = TrainCfg(arm="e2e1-lin", preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                    global_batch=2, micro_batch=1, steps=2, warmup=1, data_dir=str(data),
                    n_eval=4, min_probes=1, assay_gate=False, ho_docs=3, ho_trajectory_docs=2,
                    eval_batch=2, pilot_docs=2, pilot_steps=[1], pilot_max_rise=1e9,
                    eval_steps=[2], full_eval_steps=[0, 2], sizing_steps=[2], trajectory_docs=2,
                    crossover_rules=["ce", "lin", "aice"], resume_every=2, log_every=1,
                    save_checkpoints=False)
    seen = []
    real = E.eval_docs

    def spy(model, docs, probes, rule, **kw):
        seen.append((rule, kw.get("inner_lr", 1.0)))
        return real(model, docs, probes, rule, **kw)

    monkeypatch.setattr(E, "eval_docs", spy)
    train(replace(base, out_dir=str(tmp_path / "plain")))
    train(replace(base, out_dir=str(tmp_path / "one"), rule_lr_scale={"lin": 1.0}))
    seen.clear()
    train(replace(base, out_dir=str(tmp_path / "dosed"), rule_lr_scale={"lin": .25, "aice": .5}))
    lrs = {rule: {lr for r, lr in seen if r == rule} for rule in ("ce", "lin", "aice")}
    assert lrs == {"ce": {1.0}, "lin": {.25}, "aice": {.5}}
    load = lambda d: torch.load(tmp_path / d / "resume_step0002.pt", weights_only=False)["model"]  # noqa: E731
    a, b, c = load("plain"), load("one"), load("dosed")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_calibrate_grid_selection_and_write_once(tmp_path):
    from tttmem.p2.calibrate import CalibCfg, calibrate, select
    assert select({1: {"mean": 2.0}, .5: {"mean": 2.0}, .25: {"mean": 2.1}}, [1, .5, .25]) == 1
    assert select({1: {"mean": 2.0}, .5: {"mean": 1.9}}, [1, .5]) == .5
    data, T = _smoke_data(tmp_path)
    cfg = CalibCfg(preset="tiny", dtype="fp32", device="cpu", seq_len=T, data_dir=str(data),
                   out_dir=str(tmp_path / "cal"), docs=2, batch=2, grid=[1, .5, .25],
                   ce_grid=[2, 1])
    res = calibrate(cfg)
    rec = json.loads((tmp_path / "cal" / "calibration.json").read_text())
    assert set(res["rule_lr_scale"]) == {"lin", "aice"}
    for rule in ("lin", "aice"):
        means = {g["s"]: g["mean"] for g in rec["grid"][rule]}
        assert res["rule_lr_scale"][rule] == min(means, key=lambda s: (means[s], -s))
        assert all(0 <= g["clip_frac"] <= 1 and len(g["update_norm"]) >= 1 for g in rec["grid"][rule])
    assert rec["doc_sha1"] == ["calib0000", "calib0001"] and len(rec["per_doc"]["noadapt"]) == 2
    with pytest.raises(FileExistsError):
        calibrate(cfg)


def _a7_fixture(tmp_path, n_docs=30, C=16):
    """Docs with planted E2c (both in S2), E2b and E2c (both in S1) probes; arms whose FO-CE
    deficit sits on E2c probes only and whose lin rescue sits on E2c only."""
    from tttmem.p2.evaluate import prepare_probes

    rng = np.random.default_rng(0)
    T = 8 * C
    docs = rng.integers(1, 5000, size=(n_docs, T + 1))
    for i in range(n_docs):
        for p_tok, places in (((6001, 6002, 6003, 6004), (2 * C + 2, 3 * C + 2, 5 * C + 2)),
                              ((6011, 6012, 6013, 6014), (4, 2 * C + 9, 6 * C + 4)),
                              ((6021, 6022, 6023, 6024), (10, C + 2, 7 * C + 2))):
            for p in places:
                docs[i, p:p + 4] = p_tok
    probes = prepare_probes(docs, C, None, bos=0)
    off = np.cumsum([0] + [len(p.pos) for p in probes])
    cat = lambda a: np.concatenate([getattr(p, a) for p in probes])  # noqa: E731
    meta = dict(eval_rows=np.arange(n_docs), eval_sha1=np.array([f"doc{i}" for i in range(n_docs)]),
                eval_domain=np.array([f"d{i}.org" for i in range(n_docs)]),
                ho_sha1=np.array(["h0"]), ho_domain=np.array(["h0.org"]),
                probe_offset=off, probe_pos=cat("pos"), probe_stratum=cat("stratum"),
                probe_n_src=cat("n_src"), probe_dist=cat("dist"), probe_match8=cat("match8"),
                focal_offset=np.cumsum([0] + [int(p.focal.sum()) for p in probes]),
                focal_pos=np.concatenate([np.flatnonzero(p.focal) for p in probes]))
    strat = meta["probe_stratum"]
    assert {2, 3} <= set(strat.tolist())
    base = rng.normal(2.0, 0.1, size=len(strat))
    shift = {"e2e2": 0.0, "e2e1-ce": np.where(strat == 2, 0.5, 0.0),
             "e2e1-lin": np.where(strat == 2, 0.2, 0.0)}
    nf = len(meta["focal_pos"])
    dirs = {}
    for arm, s in shift.items():
        root = tmp_path / arm
        root.mkdir()
        np.savez(root / "eval_meta.npz", **meta)
        for step, n in ((0, n_docs), (100, 20), (200, 20), (400, n_docs)):
            ids = {k: meta[k][:n] for k in ("eval_rows", "eval_sha1", "eval_domain")}
            np.savez(root / f"eval_step{step:04d}.npz", doc_adapt=np.ones(n),
                     tok_adapt=(base + s)[:off[n]].astype(np.float32),
                     src_tok_adapt=rng.normal(3.0, 1.0, nf).astype(np.float32),
                     src_cot_adapt=np.full(nf, 0.5 if arm == "e2e1-ce" else 1.0, np.float32), **ids)
        dirs[arm] = str(root)
    dec = tmp_path / "decisions"
    dec.mkdir()
    (dec / "sizing.json").write_text(json.dumps({"N_eval": 25, "smoke": False}))
    return dirs, dec, docs, C


def test_a7_contrasts_on_planted_strata_and_write_once(tmp_path):
    from tttmem.p2.a7 import a7

    dirs, dec, docs, C = _a7_fixture(tmp_path)
    r = a7(dirs, dec, docs, C=C, bos=0, n_boot=500)
    s400 = r["step0400"]
    assert r["rules"] == ["lin"] and s400["n_docs"] == 25 and r["step0100"]["n_docs"] == 20
    assert s400["P7.1"]["passed"] and abs(s400["P7.1"]["est"] - 0.5) < 1e-6
    assert s400["P7.1s"]["passed"] and abs(s400["P7.1s"]["est"] - 0.5) < 1e-6
    assert s400["P7.3_lin"]["passed"] and abs(s400["P7.3_lin"]["est"] - 0.3) < 1e-6
    assert s400["P7_4"]["E2c"]["cot_ratio_of_means"] == pytest.approx(0.5)
    assert s400["P7_4"]["E2c_S2"]["n_probes"] == 25
    with pytest.raises(FileExistsError):
        a7(dirs, dec, docs, C=C, bos=0, n_boot=500)


def test_a7_rejects_sources_that_disagree_with_saved_annotations(tmp_path):
    from tttmem.p2.a7 import a7

    dirs, dec, docs, C = _a7_fixture(tmp_path)
    tampered = docs.copy()
    tampered[0, 2 * C + 2:2 * C + 6] = (7001, 7002, 7003, 7004)  # first E2c source removed
    with pytest.raises(ValueError, match="disagree|no recomputed source"):
        a7(dirs, dec, tampered, C=C, bos=0, n_boot=200)
    with pytest.raises(ValueError, match="pool"):
        a7(dirs, dec, docs[:-1], C=C, bos=0, n_boot=200)
