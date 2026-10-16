"""Tests for scripts/analyze_chr.py on synthetic CHR logs and on a tiny real cell (CPU)."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_chr", ROOT / "scripts" / "analyze_chr.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _row(cid, step, acc, r, branch="main", bb5=1.0, bb6=1.0):
    geo = {c: dict(bb=b, slow=b, mem=.1, sigma=1.0, cos_bb_head0=-.5, n=10) for c, b in (("5", bb5), ("6", bb6))}
    return dict(cell_id=cid, branch=branch, rule="ce", step=step, loss=1.0,
                quick=dict(acc=acc, qce=.1, acc5=acc, acc6=acc, geo=geo),
                write=dict(r_kv_med=r, r_kv_mean=r, kv_write_med=1.0, g0_med=1.0, clip_c0=0.0))


def _trajectory(cid, acq, starve, loss, end=200, every=5):
    rows = []
    for t in range(every, end + every, every):
        acc = .95 if acq <= t < loss else .5
        r = .01 if starve is not None and t >= starve else .2
        rows.append(_row(cid, t, acc, r))
    return rows


def _cell(cid, acc=.5, steps=200):
    iv = {f"{p}{v}": {"5": .9, "6": .2, "all": .55} for v in ("BB", "SLOW", "HEAD")
          for p in ("", "RND_", "SHAM5_")}
    iv["NAT"] = {"5": .9, "6": .2, "all": .55}
    return dict(cell_id=cid, steps=steps, steps_done=steps, diverged=False, forks=[],
                final=dict(ce=dict(acc=acc, acc5=.9, acc6=.2, decomp_gate_ok=True,
                                   intervention=dict(iv, alpha_bb_median=.1, alpha_slow_median=.1,
                                                     alpha_head_median=.1))))


def _log(tmp_path, rows, cells, name="run.log", sanity=True, extra=()):
    lines = ["CHR_SETTINGS " + json.dumps(dict(chunk_early=5, chunk_late=6, log_every=5))]
    if sanity:
        lines.append("CHR_SANITY " + json.dumps({"x": {"passed": True}}))
    lines += ["CHR_LOG " + json.dumps(r) for r in rows]
    lines += ["CHR_CELL " + json.dumps(c) for c in cells]
    lines += list(extra)
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _score(*paths):
    return A.score(A.consolidate([A.parse(x) for x in paths], production=False))


def test_events_sustained_and_strict_order():
    rows = _trajectory("echo/s0/e2e1emb/eta40/st1500", acq=50, starve=80, loss=100)
    ev = A.events(A.series(rows))
    assert (ev["t_acq"], ev["t_starve"], ev["t_loss"]) == (50, 80, 100)
    # a one-log dip is not a sustained loss
    rows[11]["quick"]["acc"] = .5  # step 60
    ev = A.events(A.series(rows))
    assert ev["t_loss"] == 100
    assert A.events(A.series(rows), sustained=False)["t_loss"] == 60


def test_h1_counts_and_reserve_seeds(tmp_path):
    rows, cells = [], []
    # seeds 0-4 support (starve before loss); seed 5 never acquires (not evaluable);
    # seed 6 starves at the loss log (same log = not before); seeds 7-9 support
    plan = {0: (50, 80, 100), 1: (50, 60, 120), 2: (40, 45, 90), 3: (50, 80, 100), 4: (50, 95, 100),
            5: (10**6, None, 10**6), 6: (50, 100, 100), 7: (50, 80, 100), 8: (50, 80, 100),
            9: (50, 80, 100)}
    for seed, (acq, starve, loss) in plan.items():
        cid = f"echo/s{seed}/e2e1emb/eta40/st1500"
        rows += _trajectory(cid, acq, starve, loss)
        cells.append(_cell(cid))
    sc = _score(_log(tmp_path, rows, cells))
    h1 = sc["H1"]
    assert [u["seed"] for u in h1["units"]] == [0, 1, 2, 3, 4, 6]
    assert h1["k"] == 5 and h1["verdict"] == "SUPPORTED"
    assert h1["categories"] == {"before_persistent": 5, "simultaneous": 1}
    g25 = sc["H1_sensitivity"]["grid25"]
    assert g25["frozen"]["n"] == 6 and g25["reselected"]["n"] == 6
    disp = {d["seed"]: d for d in sc["disposition"]}
    assert disp[5]["eligible"] and not disp[5]["unit_H1"] and not disp[7]["unit_H1"] and disp[6]["unit_H1"]


def test_count_and_h5_verdicts():
    assert A.count_verdict(5, 6) == "SUPPORTED" and A.count_verdict(4, 6) == "INCONCLUSIVE"
    assert A.count_verdict(3, 6) == "FALSIFIED" and A.count_verdict(3, 3) == "NOT EVALUABLE"
    assert A.count_verdict(4, 5) == "SUPPORTED" and A.count_verdict(2, 4) == "FALSIFIED"
    assert A.count_verdict(3, 4) == "INCONCLUSIVE"
    assert A.h5_verdict(6, 0, 0, 6) == "SUPPORTED"
    assert A.h5_verdict(6, 0, 2, 6) == "INCONCLUSIVE"
    assert A.h5_verdict(5, 4, 0, 6) == "FALSIFIED"
    assert A.h5_verdict(2, 0, 0, 6) == "FALSIFIED"
    assert A.h5_verdict(4, 1, 0, 6) == "INCONCLUSIVE"
    assert A.exact_ci(5, 6) == [0.3588, 0.9958] and A.exact_ci(0, 6) == [0.0, 0.4593]
    assert A.exact_ci(3, 6) == [0.1181, 0.8819] and A.exact_ci(0, 0) is None


def test_h4_repaired_needs_rnd_margin(tmp_path):
    cid = "echo/s0/e2e1emb/eta40/st1500"
    cell = _cell(cid)
    iv = cell["final"]["ce"]["intervention"]
    iv["SLOW"]["6"] = .6       # Δ6 = .4 ≥ D/2 = .35
    iv["RND_SLOW"]["6"] = .58  # margin .02 < .05 → not repaired
    iv["BB"]["6"] = .6
    sc = _score(_log(tmp_path, _trajectory(cid, 50, 80, 100), [cell]))
    assert sc["H4a_SLOW"]["units"][0]["repaired"] is False
    assert sc["H4b_BB"]["units"][0]["repaired"] is True


def test_end_to_end_on_tiny_real_cell(tmp_path, monkeypatch, capsys):
    from tttmem import exp_chr as c
    from tttmem import exp_ttt as t
    torch.set_num_threads(2)
    t._mark_process(os.environ.get("PROCESS_NAME", "TTM-chr-91-tests"))
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=2, n_heads=2, backbone_mult=2, fast_mult=2, window=8, chunk=16,
               n_keys=4, T=96, n_eval=4, n_quick_eval=4, eval_batch=2, n_probe=2, batch=2, steps=12,
               lr_total=3000, eta_lr=0.0, workers=1, device="cpu", tf32=False,
               log_every=2, probe_every=4, cross_every=4, chunk_early=2, chunk_late=3,
               fork_arms=["e2e1emb", "e2e1emb_cn97"], fork_acc=0.0, rp_cap=1000.0,
               forks={"e2e1emb": [{"rule": "ce"}, {"rule": "rp", "r_min": 0.03}, {"rule": "nc", "r_min": 0.03},
                                  {"rule": "rp", "r_min": 0.10}, {"rule": "nc", "r_min": 0.10}],
                      "e2e1emb_cn97": [{"rule": "ce"}, {"rule": "rs", "r_cap": 0.01}, {"rule": "ns", "r_cap": 0.01}]},
               cross_rules={"e2e1emb": ["rp03"], "e2e1emb_cn97": ["rs01"]},
               final_rules={"e2e1emb": ["rp03", "rp10"], "e2e1emb_cn97": ["rs01", "rp03"]},
               ckpt_steps=[], branch_ckpt_every=0, branch_ckpt_tags=[], mediation_p=0.5)
    print("CHR_SETTINGS " + json.dumps(dataclasses.asdict(c.chr_settings(cfg)), default=str))
    print("CHR_SANITY " + json.dumps(c._finite(c.chr_sanity()), default=str))
    c.run_chr_cell(cfg, dict(task="echo", seed=0, arm="e2e1emb", eta_base=40.0), torch.device("cpu"))
    c.run_chr_cell(cfg, dict(task="echo", seed=0, arm="e2e2emb", eta_base=40.0), torch.device("cpu"))
    c.run_chr_cell(cfg, dict(task="echo", seed=0, arm="e2e1emb_cn97", eta_base=40.0), torch.device("cpu"))
    log = tmp_path / "run.log"
    log.write_text(capsys.readouterr().out)
    parsed = A.consolidate([A.parse(log)], production=False)
    assert A.chunks_of(parsed) == (2, 3) and A.grid_of(parsed) == 2
    sc = A.score(parsed)
    A.write_outputs(parsed, sc, tmp_path / "out")
    assert sc["ineligible"] == {} and sc["grid"] == {} and sc["chunks"] == [2, 3]
    # the tiny model never binds: every unit-based rule is NOT EVALUABLE, but nothing crashes
    assert sc["H1"]["verdict"] == "NOT EVALUABLE"
    assert {x["branch"] for x in sc["H4_no_harm"]} >= {"fork-rp03", "fork-nc10", "fork-rs01", "main"}
    assert sc["H6_r01"]["verdict"] == "NOT EVALUABLE" and sc["H8"]["verdict"] == "NOT EVALUABLE"
    assert {r["arm"] for r in sc["H8"]["descriptive"]} == {"e2e2emb", "e2e1emb_cn97"}
    series = (tmp_path / "out" / "chr_series.csv").read_text().splitlines()
    assert series[0].startswith("cell_id,branch,step,a,a5,a6,r,")
    assert any(",fork-rp03," in line for line in series)


def test_h6_verdict_table_and_selectivity():
    assert A.h6_verdict(0, 6, 6, 6) == "SUPPORTED" and A.h6_verdict(1, 5, 6, 6) == "SUPPORTED"
    assert A.h6_verdict(4, 6, 6, 6) == "FALSIFIED" and A.h6_verdict(2, 6, 6, 6) == "INCONCLUSIVE"
    assert A.h6_verdict(0, 3, 6, 6) == "INCONCLUSIVE"  # non-specific harm
    assert A.h6_verdict(0, 6, 4, 6) == "INCONCLUSIVE"  # the CE twin fails in 2 units: no stable reference
    assert A.h6_verdict(1, 4, 5, 5) == "SUPPORTED" and A.h6_verdict(4, 5, 5, 5) == "FALSIFIED"
    assert A.h6_verdict(0, 4, 4, 4) == "INCONCLUSIVE" and A.h6_verdict(3, 4, 4, 4) == "FALSIFIED"
    assert A.h6_verdict(0, 3, 3, 3) == "NOT EVALUABLE"
    assert A.selectivity(5, 0, 6, "SUPPORTED", True) == "eval-carried"
    assert A.selectivity(0, 5, 6, "SUPPORTED", True) == "training-selective"
    assert A.selectivity(3, 3, 6, "SUPPORTED", True) == "mixed"
    assert A.selectivity(0, 6, 6, "SUPPORTED", False) == "unknown"
    assert A.selectivity(0, 6, 6, "FALSIFIED", True) == "n/a"


def _branch(rule, own_acc, ce_acc, steps=200, fork=100, **evals):
    final = {rule: dict(acc=own_acc), "ce": dict(acc=ce_acc), **{k: dict(acc=v) for k, v in evals.items()}}
    return dict(branch=f"fork-{rule}" + ("-twin" if rule == "ce" else ""), rule=rule, fork_step=fork,
                state_identical=True, batch_identical=True, main_batch_identical=True, steps_done=steps,
                final=final)


def _btraj(cid, b, kv=1.0, kv_ce=1.0, end=200, **train):
    """A branch's complete exact grid (fork + 5 … end); non-CE rules carry the training counters."""
    rows = []
    for t in range(b["fork_step"] + 5, end + 5, 5):
        r = _row(cid, t, .95, .05, branch=b["branch"])
        r["rule"] = b["rule"]
        r["write"]["kv_write_med"] = kv
        if b["rule"] != "ce":
            r["write_ce"] = dict(kv_write_med=kv_ce)
            r.update(train_n=10, train_clip_count=0, train_match_fail_count=0, train_added_req_sum=1.0,
                     train_added_actual_sum=1.0, **train)
        rows.append(r)
    return rows


def _h6_world(tmp_path, n_seeds=6, mutate=None, name="run.log"):
    rows, cells, forks = [], [], []
    for seed in range(n_seeds):
        cid = f"echo/s{seed}/e2e1emb_cn97/eta40/st1500"
        rows += _trajectory(cid, 50, None, 10**6)
        cell = _cell(cid, acc=.97 if seed != 5 else .5)  # seed 5: main not retained -> not a unit
        cell["final"]["rs01"] = dict(acc=.2)
        cell["forks"] = [_branch("ce", .95, .95, rs01=.3), _branch("rs01", .2, .3), _branch("ns01", .95, .96, rs01=.5)]
        br = {b["rule"]: _btraj(cid, b) for b in cell["forks"][:1]}
        br["rs01"] = _btraj(cid, cell["forks"][1], kv=.33)
        br["ns01"] = _btraj(cid, cell["forks"][2], train_cap_bound_count=0)
        w = {"ce": dict(kv_write_med=1.0), "rs01": dict(kv_write_med=.33), "ns01": dict(kv_write_med=1.0, cap_bound=0.0)}
        fork = dict(cell_id=cid, step=100, write_at_fork=w, write_at_fork_corrupted=w)
        if mutate:
            mutate(seed, cell, br, fork)
        cells.append(cell)
        rows += [r for v in br.values() for r in v]
        forks.append(fork)
    return _log(tmp_path, rows, cells, name=name, extra=["CHR_FORK " + json.dumps(f) for f in forks])


def test_h6_units_and_selectivity(tmp_path):
    h6 = _score(_h6_world(tmp_path))["H6_r01"]
    assert (h6["n"], h6["rs"], h6["ns"], h6["twin"]) == (5, 0, 5, 5) and h6["verdict"] == "SUPPORTED"
    assert h6["selectivity"] == "training-selective" and h6["ce_main_retained_under_rs_eval"] == 0
    assert h6["manipulation"] == "OK" and all(not u["weak_manipulation"] for u in h6["units"])
    assert all(u["ns_attained_train"] == 1.0 and not u["ns_under_matched"] for u in h6["units"])


def test_h6_missing_twin_counters_and_under_matching(tmp_path):
    def mutate(seed, cell, br, fork):
        if seed == 0:
            cell["forks"][0]["final"]["ce"]["acc"] = None   # twin without a CE final -> unit invalid
        if seed == 1:
            del br["ns01"][3]["train_cap_bound_count"]      # one window lacks a counter -> invalid
        if seed in (2, 3):
            for r in br["ns01"]:
                r["train_added_actual_sum"] = .5            # NS removes half the requested norm
    h6 = _score(_h6_world(tmp_path, n_seeds=8, mutate=mutate))["H6_r01"]
    why = dict(h6["invalid"])
    assert set(why) == {"echo/s0/e2e1emb_cn97/eta40/st1500", "echo/s1/e2e1emb_cn97/eta40/st1500"}
    assert why["echo/s0/e2e1emb_cn97/eta40/st1500"]["ce"].startswith("no finite")
    assert why["echo/s1/e2e1emb_cn97/eta40/st1500"]["ns01"] == "training counters missing"
    assert [u["seed"] for u in h6["units"]] == [2, 3, 4, 6, 7]
    assert h6["manipulation"] == "LIMITED" and sum(u["ns_under_matched"] for u in h6["units"]) == 2


def _med(d, acc_clean=.5, inv=0, cap=0):
    return dict(d=d, acc_corr=None if d is None else acc_clean + d, acc_clean=acc_clean, clip_c0=.5, d5=d, d6=d,
                n_invalid_events=inv, n_capped_events=cap, n_nonfinite_norms=0, drive_corr_med=1.0,
                drive_clean_med=1.0)


def test_h8_clamp_sensitivity_validity_and_coupling(tmp_path):
    rows, cells = [], []
    for seed in range(8):
        cid = f"echo/s{seed}/e2e1emb/eta40/st1500"
        rows += _trajectory(cid, 50, 80, 100)
        cell = _cell(cid)
        nat, clamp, floor = _med(.2 if seed else .01), _med(.02, acc_clean=.45 if seed == 7 else .5), _med(.05)
        if seed == 1:
            clamp = _med(.02, inv=1)   # 1 of 10 corrupted events invalid > 5 % -> invalid unit
        if seed == 2:
            floor = _med(None)         # NaN written as null -> invalid unit
        cell["mediation"] = None if seed == 3 else dict(n_corrupted=10, n_clean=10, r_clean_med=1e-3,
                                                        r_corr_med=.9, NAT=nat, CLAMP=clamp, FLOOR=floor)
        cells.append(cell)
    h8 = _score(_log(tmp_path, rows, cells))["H8"]
    assert [u["seed"] for u in h8["units"]] == [4, 5, 6, 7]
    assert {c for c, _ in h8["invalid"]} == {f"echo/s{s}/e2e1emb/eta40/st1500" for s in (1, 2, 3)}
    assert h8["k_sensitive"] == 4 and h8["verdict"] == "CLAMP-SENSITIVE" and h8["n_coupled"] == 1
    u = h8["units"][0]
    assert abs(u["m_CLAMP"] - .9) < 1e-9 and abs(u["m_FLOOR"] - .75) < 1e-9 and abs(u["dacc_corr_CLAMP"] + .18) < 1e-9


def test_eligibility_and_sanity_abort(tmp_path):
    rows, cells = [], []
    for seed in range(8):
        cid = f"echo/s{seed}/e2e1emb/eta40/st200"
        rows += _trajectory(cid, 50, 80, 100)
        cells.append(_cell(cid))
    cells[0]["steps_done"] = 150          # incomplete -> ineligible, never scored as 0
    cells[1]["diverged"] = True
    cells[2]["final"]["ce"]["acc"] = None  # no finite final
    sc = _score(_log(tmp_path, rows, cells))
    assert set(sc["ineligible"]) == {c["cell_id"] for c in cells[:3]}
    assert [u["seed"] for u in sc["H1"]["units"]] == [3, 4, 5, 6, 7] and sc["H1"]["verdict"] == "SUPPORTED"
    with pytest.raises(SystemExit):
        _score(_log(tmp_path, rows, cells, name="nosanity.log", sanity=False))


def test_null_residual_is_not_evidence(tmp_path):
    rows = _trajectory("echo/s0/e2e1emb/eta40/st200", acq=50, starve=80, loss=100)
    rows[19]["write"]["r_kv_med"] = None   # step 100: the loss log has no finite residual
    ev = A.events(A.series(rows), terminal=200)
    assert ev["t_loss"] == 100 and ev["category"] == "unknown" and ev["r_at_loss"] is None
    sc = _score(_log(tmp_path, rows, [_cell("echo/s0/e2e1emb/eta40/st200")]))
    assert sc["grid"]["echo/s0/e2e1emb/eta40/st200"]["nonfinite"] == [100] and sc["H1"]["n"] == 0


def test_exact_grid_sustained_and_terminal():
    rows = _trajectory("echo/s0/e2e1emb/eta40/st200", acq=50, starve=None, loss=10**6)
    rows[-1]["quick"]["acc"] = .5           # a dip at the terminal step 200 alone counts
    s = A.series(rows)
    assert A.events(s, terminal=200)["t_loss"] == 200
    assert A.events(s, terminal=205)["t_loss"] is None  # not terminal -> needs t + 5
    rows = _trajectory("echo/s0/e2e1emb/eta40/st200", acq=50, starve=None, loss=10**6)
    for i in (11, 13):                       # steps 60 and 70 below; step 65 missing
        rows[i]["quick"]["acc"] = .5
    del rows[12]
    s = A.series(rows)
    assert A.events(s, terminal=200)["t_loss"] is None  # 60 -> 70 is not an exact t + 5 pair
    assert A.grid_check(rows, 5, 200)["missing"] == [65]


def test_h1_categories_episode_and_resolved():
    rows = _trajectory("echo/s0/e2e1emb/eta40/st200", acq=50, starve=None, loss=150)
    for r in rows:
        if 60 <= r["step"] <= 80:        # an early starved episode that resolves
            r["write"]["r_kv_med"] = .01
    ev = A.events(A.series(rows), terminal=200)
    assert ev["t_starve"] == 60 and ev["t_loss"] == 150 and ev["category"] == "resolved_earlier_episode"
    for r in rows:
        if r["step"] >= 140:             # a persistent episode from 140, before the loss
            r["write"]["r_kv_med"] = .01
    ev = A.events(A.series(rows), terminal=200)
    assert ev["t_onset_at_loss"] == 140 and ev["category"] == "before_persistent"


def test_retry_log_fills_failed_cell(tmp_path):
    cid0, cid1 = "echo/s0/e2e1emb/eta40/st200", "echo/s1/e2e1emb/eta40/st200"
    man = "CHR_MANIFEST " + json.dumps(dict(cells=[cid0, cid1], steps=200))
    err = "CHR_CELL_ERROR " + json.dumps(dict(cell_id=cid1, error="CUDA error"))
    first = _log(tmp_path, _trajectory(cid0, 50, 80, 100), [_cell(cid0)], extra=[man, err, "CHR_LOG {broken"])
    retry = _log(tmp_path, _trajectory(cid1, 50, 80, 100), [_cell(cid1)], name="retry.log")
    data = A.consolidate([A.parse(first), A.parse(retry)], production=False)
    assert data["attempt"] == {cid0: 0, cid1: 1} and data["status"][cid1].startswith("complete [retry")
    n_lines = len(first.read_text().splitlines())
    assert data["malformed"] == {str(first): [(n_lines, "CHR_LOG", "json")]}
    assert A.consolidate([A.parse(first)], production=False)["status"][cid1].startswith("error")


def test_retry_contract(tmp_path):
    cid = "echo/s0/e2e1emb/eta40/st200"
    man = "CHR_MANIFEST " + json.dumps(dict(cells=[cid], steps=200))
    good = _log(tmp_path, _trajectory(cid, 50, 80, 100), [_cell(cid)], name="good.log")
    diag = lambda *ps: A.consolidate([A.parse(x) for x in ps], production=False)  # noqa: E731
    with pytest.raises(SystemExit):           # the run log and at most one retry log
        diag(good, good, good)
    # a non-infrastructure error is a result of the code, not retried
    bad = _log(tmp_path, [], [], name="bad.log",
               extra=[man, "CHR_CELL_ERROR " + json.dumps(dict(cell_id=cid, error="ValueError('shape')"))])
    data = diag(bad, good)
    assert cid not in data["attempt"] and "non-infrastructure" in data["status"][cid]
    # a complete first attempt is never replaced
    first = _log(tmp_path, _trajectory(cid, 50, 80, 100), [_cell(cid)], name="first.log", extra=[man])
    data = diag(first, good)
    assert data["attempt"] == {cid: 0} and data["retry_ignored"] == [cid]
    # a branch OOM is infrastructure: the complete retry is used
    oom = _cell(cid)
    oom["forks"] = [dict(branch="fork-rp03", rule="rp03", fork_step=100, error="OutOfMemoryError('CUDA out of memory')")]
    data = diag(_log(tmp_path, _trajectory(cid, 50, 80, 100), [oom], name="oom.log", extra=[man]), good)
    assert data["attempt"] == {cid: 1} and "branch_error" in data["status"][cid]
    # a non-infrastructure branch error keeps the first attempt; the branch is ineligible
    oom["forks"][0]["error"] = "ValueError('nan')"
    data = diag(_log(tmp_path, _trajectory(cid, 50, 80, 100), [oom], name="berr.log", extra=[man]), good)
    assert data["attempt"] == {cid: 0} and data["status"][cid].startswith("branch_error")


def _production_log(tmp_path, name="prod.log", **over):
    cid = "echo/s0/e2e1emb/eta40/st1500"
    prot = dict(protocol="chr-full-v2", device="cuda:0", gpu="RTX 4090", analyzer_sha256=A.ANALYZER_SHA256,
                plan_sha256=A.PLAN_SHA256, science_sha256="abc", exp_chr_sha256="c", exp_ttt_sha256="t")
    prot.update({k: v for k, v in over.items() if k in prot})
    sanity = {k: {"passed": True} for k in A.EXPECTED_SANITY if k != over.get("drop_gate")}
    lines = ["CHR_PROTOCOL " + json.dumps(prot),
             "CHR_SETTINGS " + json.dumps(dict(chunk_early=5, chunk_late=6, log_every=over.get("log_every", 5))),
             "CHR_SANITY " + json.dumps(sanity),
             "CHR_MANIFEST " + json.dumps(dict(cells=[cid], steps=over.get("steps", 1500))),
             "CHR_CELL " + json.dumps(_cell(cid, steps=1500))]
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_production_contract(tmp_path):
    ok = _production_log(tmp_path)
    data = A.consolidate([A.parse(ok)])
    assert data["production"] and data["contract_violations"] == []
    for i, over in enumerate((dict(protocol="chr-smoke-v2"), dict(device="mps"), dict(analyzer_sha256="0" * 64),
                              dict(plan_sha256="0" * 64), dict(steps=200), dict(log_every=25),
                              dict(drop_gate="ns_norm_match"))):
        bad = _production_log(tmp_path, name=f"bad{i}.log", **over)
        with pytest.raises(SystemExit):
            A.consolidate([A.parse(bad)])
        if "drop_gate" not in over:   # diagnostic mode reports instead of aborting
            assert A.consolidate([A.parse(bad)], production=False)["contract_violations"]
    other = _production_log(tmp_path, name="other.log", science_sha256="def")
    with pytest.raises(SystemExit):   # a retry must share the scientific fingerprint
        A.consolidate([A.parse(ok), A.parse(other)])


def _h5_world(tmp_path, n_seeds=6, mutate=None):
    rows, cells, forks = [], [], []
    for seed in range(n_seeds):
        cid = f"echo/s{seed}/e2e1emb/eta40/st200"
        rows += _trajectory(cid, 50, 80, 100)
        cell = _cell(cid, acc=.5)
        cell["final"]["rp03"], cell["final"]["rp10"] = dict(acc=.5), dict(acc=.5)
        bs = [_branch("ce", .5, .5, rp03=.5, rp10=.5), _branch("rp03", .95, .6), _branch("nc03", .5, .5, rp03=.5),
              _branch("rp10", .95, .6), _branch("nc10", .5, .5, rp10=.5)]
        br = {b["rule"]: _btraj(cid, b, kv=3.0 if b["rule"].startswith("rp") else 1.0,
                                **({"train_cap_hit_mean": .1} if b["rule"].startswith("rp") else {}))
              for b in bs}
        cell["forks"] = bs
        fork = dict(cell_id=cid, step=100, write_at_fork={"ce": dict(kv_write_med=1.0, clip_c0=0.),
                                                          "rp03": dict(kv_write_med=3.0, clip_c0=0.),
                                                          "rp10": dict(kv_write_med=3.0, clip_c0=0.)},
                    quick_at_fork={"ce": dict(acc=.95), "rp03": dict(acc=.96)})
        if mutate:
            mutate(seed, cell, br, fork)
        cells.append(cell)
        rows += [r for v in br.values() for r in v]
        forks.append(fork)
    return _log(tmp_path, rows, cells, extra=["CHR_FORK " + json.dumps(f) for f in forks]), cells


def test_h5_pair_specific_validity_and_inherited_acquisition(tmp_path):
    def mutate(seed, cell, br, fork):
        if seed == 0:
            cell["forks"][4] = dict(branch="fork-nc10", rule="nc10", fork_step=100, error="boom")
            del br["nc10"]
    log, cells = _h5_world(tmp_path, mutate=mutate)
    sc = _score(log)
    h5, h5b = sc["H5_primary_r03"], sc["H5_secondary_r10"]
    assert h5["n"] == 6 and h5["verdict"] == "SUPPORTED" and h5["selectivity"] == "training-selective"
    assert h5["manipulation"] == "OK" and h5["units"][0]["M1_train"] == 3.0
    assert h5b["n"] == 5 and h5b["n_invalid"] == 1  # the nc10 error only invalidates the r10 pair
    data = A.consolidate([A.parse(log)], production=False)
    A.write_outputs(data, sc, tmp_path / "out")
    ev = (tmp_path / "out" / "chr_events.csv").read_text().splitlines()
    assert any(line.startswith(f"{cells[0]['cell_id']},fork-rp03,100,") for line in ev)  # acquisition inherited


def test_h5_missing_evidence(tmp_path):
    def mutate(seed, cell, br, fork):
        if seed == 0:
            cell["forks"][0]["final"]["ce"]["acc"] = None    # twin CE final missing -> both pairs invalid
        if seed == 1:
            del br["rp03"][5]["train_added_actual_sum"]      # rp03 counter missing -> r03 pair invalid
            for r in br["rp03"]:
                r["train_clip_count"] = 0
        if seed == 2:
            del cell["forks"][2]["final"]["rp03"]            # nc03 not scored under rp03 eval
        if seed == 3:
            del br["rp03"][-1]                               # rp03 branch grid incomplete
    log, _ = _h5_world(tmp_path, n_seeds=10, mutate=mutate)
    sc = _score(log)
    h5, h5b = sc["H5_primary_r03"], sc["H5_secondary_r10"]
    why = dict(h5["invalid"])
    assert set(why) == {f"echo/s{s}/e2e1emb/eta40/st200" for s in (0, 1, 3)}
    assert why["echo/s1/e2e1emb/eta40/st200"]["rp03"] == "training counters missing"
    assert why["echo/s3/e2e1emb/eta40/st200"]["rp03"].startswith("grid")
    assert [u["seed"] for u in h5["units"]] == [2, 4, 5, 6, 7, 8]
    assert h5["verdict"] == "SUPPORTED" and h5["selectivity"] == "unknown"   # missing ≠ "fails"
    assert [u["seed"] for u in h5b["units"]] == [1, 2, 3, 4, 5, 6] and h5b["selectivity"] == "training-selective"
