"""Registered R4-4a analysis: RUN.log [--out DIR]. No training or torch import.

Reuses R4-2L's complete-pair/S5 eligibility, exact McNemar, t_lose and CSV writer.
GM-2/GM-3 are separate conjunctive gates; Holm covers only R1/R2/R3.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_r42l import binom_cdf, mcnemar, t_lose, valid, write_csv

A1, EDGE = "e2e1emb", "e2e1emb_edge"
CN = {"e2e1emb_cn99": .99, "e2e1emb_cn97": .97,
      "e2e1emb_cn95": .95, "e2e1emb_cn90": .90}
RF = "e2e1emb_rf95"
KAMIN, KLIN, KCTL = "e2e1emb_kamin", "e2e1emb_kaminlin", "e2e1emb_kaminctl"
KV, KV10 = "e2e1emb_kveta", "e2e1emb_kveta10"
ARMS = (A1, EDGE, *CN, RF, KAMIN, KLIN, KCTL, KV, KV10)
A9_FIELDS = ("step", "eta_W2", "eta_kvW2", "kv_write_eff", "p_kv")
ANCHOR_TABLE = Path("outputs/r4-2l/r42l-paired-seeds.csv")


def records(path, s5_override=None):
    cells, settings, errors = {}, {}, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("config: "):
            settings = json.loads(line[8:]).get("ttt", {})
        elif line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            key = int(c["seed"]), c["arm"]
            if key in cells:
                errors.append(f"duplicate CELL_RESULT: {key}")
                c["unpaired"] = True
            cells[key] = c
        elif line.startswith("CELL_ERROR "):
            errors.append(line)
    if s5_override:
        for c in cells.values():
            if c["cell_id"] in s5_override:
                c["s5_in_run"] = c.get("s5", {})
                c["s5"] = dict(passed=bool(s5_override[c["cell_id"]]), source="fp32-cpu")
    for seed in range(10):
        group = [c for (s, a), c in cells.items() if s == seed and a in ARMS]
        hashes = {c.get("first_clean_batch_sha1", c.get("first_batch_sha1")) for c in group}
        if len(hashes) > 1 or None in hashes:
            errors.append(f"seed {seed}: clean training batches unpaired")
            for c in group:
                c["unpaired"] = True
        for c in group:
            if c.get("task") != "echo" or c.get("lin_per_seq") != (16 if c["arm"] == KLIN else 0):
                errors.append(f"{c['cell_id']}: wrong task or lin count")
                c["unpaired"] = True
            if c.get("warmup") != 200:
                errors.append(f"{c['cell_id']}: wrong warmup")
                c["unpaired"] = True
            if c.get("diverged"):
                errors.append(f"{c['cell_id']}: diverged (retained in pairing as in R4-2L)")
    return cells, settings, errors


def live(cells, arm):
    return [c for (s, a), c in sorted(cells.items()) if a == arm and valid(c)]


def paired_test(cells, a, b, field="bound_final"):
    eligible = {key: c for key, c in cells.items() if field in c}
    return mcnemar(eligible, a, b, field)


def reproducibility(cells, path):
    if not path.exists():
        return dict(status="missing_reference", path=str(path), rows=[])
    reference = {int(r["seed"]): r for r in csv.DictReader(path.open())}
    rows = []
    for seed in range(10):
        c, old = cells.get((seed, A1)), reference.get(seed)
        row = dict(seed=seed, available=bool(c and old))
        if c and old:
            for field in ("bound_ever", "bound_final", "t_lose"):
                raw = old[f"{A1}_{field}"]
                previous = (int(raw) if raw else None) if field == "t_lose" else raw == "True"
                current = t_lose(c) if field == "t_lose" else c.get(field)
                row.update({field: current, "r42l_"+field: previous, field+"_differs": current != previous})
            # exact check of the a1 training stream (registered pre-launch check)
            previous = old.get(f"{A1}_first_batch_sha1") or None
            row.update(first_batch_sha1=c.get("first_batch_sha1"), r42l_first_batch_sha1=previous,
                       first_batch_sha1_differs=previous is not None and c.get("first_batch_sha1") != previous)
        rows.append(row)
    return dict(status="descriptive", path=str(path), rows=rows,
                differences=[r for r in rows if any(r.get(f+"_differs") for f in
                                                  ("bound_ever", "bound_final", "t_lose", "first_batch_sha1"))])


def routes(cells):
    rows = []
    for seed in range(10):
        c = cells.get((seed, EDGE), {})
        route = c.get("route_checks") or {}
        pv = route.get("p_v_kv_masked")
        ab = c.get("ablation") or {}
        accuracy = c.get("eval", {}).get("query_acc")
        retained = c.get("bound_final", False)
        complete = valid(c)
        pv_ok = pv is not None and .04 <= pv <= .09
        removal_ok = bool(accuracy is not None and ab.get("all_minus_kv") is not None and
                          accuracy - ab["all_minus_kv"] >= .3)
        only_ok = bool(accuracy is not None and ab.get("kv_only") is not None and
                       ab["kv_only"] >= .9 * accuracy)
        rows.append(dict(seed=seed, retained=retained, complete=complete, p_v=pv,
                         p_v_pass=pv_ok, final_acc=accuracy,
                         all_minus_kv=ab.get("all_minus_kv"), kv_only=ab.get("kv_only"),
                         removal_pass=removal_ok, only_pass=only_ok,
                         mask_off_acc=route.get("mask_off", {}).get("query_acc"),
                         mask_off_p_kv=route.get("mask_off", {}).get("p_kv"),
                         passed=bool(complete and pv_ok and (not retained or
                                     (c.get("ablation_ref_match") is True and removal_ok and only_ok)))))
    return dict(passed=all(r["passed"] for r in rows), rows=rows)


def r2_reading(tests):
    """R2 (a9 re-binds): Bonferroni over the two eta_lr arms, p = min(1, 2 min p).

    Registered pre-launch amendment: if kveta10 has fewer than 10 valid pairs
    (e.g. non-finite steps from a runaway eta_kv), R2 uses kveta alone; kveta10 is then
    descriptive. R2 is complete only when kveta has 10 valid pairs.
    """
    kv, kv10 = tests[KV], tests[KV10]
    if kv10["n"] == 10:
        p, rule = min(1., 2*min(kv["mcnemar_one_sided"], kv10["mcnemar_one_sided"])), "bonferroni_2"
    else:
        p, rule = kv["mcnemar_one_sided"], "kveta_only"
    return dict(p=p, rule=rule, complete=kv["n"] == 10, tests=tests)


def finite(x):
    """Non-finite floats become None so one diverged cell cannot crash the JSON dump."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def holm(readings):
    """Three registered readings, including p=1 placeholders for incomplete ones."""
    running = 0.
    for r in readings.values():
        r["holm_input_p"] = r["p"] if r["complete"] else 1.
    ordered = sorted(readings, key=lambda k: readings[k]["holm_input_p"])
    for rank, key in enumerate(ordered):
        running = max(running, min(1., (3-rank)*readings[key]["holm_input_p"]))
        readings[key].update(holm_p=running,
                             holm_pass=readings[key]["complete"] and running < .05)
    return readings


def noise_rows(cells):
    rows = []
    for arm, q in CN.items():
        group = live(cells, arm)
        lam = (16*q-1)/15
        rows.append(dict(arm=arm, q=q, channel_lambda=lam,
                         bayes_residual=(1-lam*lam)*15/16, n=len(group),
                         bound_final=sum(c["bound_final"] for c in group),
                         final_query_acc=mean(c["eval"]["query_acc"] for c in group) if group else None,
                         clean_eval_ce=mean(c["eval"]["ce"] for c in group) if group else None,
                         clean_eval_query_ce=mean(c["eval"]["query_ce"] for c in group) if group else None))
    return rows


def summarize(cells, settings, issues, anchor_table=ANCHOR_TABLE):
    switches = [dict(arm=a, seed=s, **(cells.get((s, a), {}).get("switch_check") or {}))
                for s in range(10) for a in (KAMIN, KLIN, KCTL)]
    gm2_test = paired_test(cells, EDGE, A1)
    route = routes(cells)
    # GM-3 direction is control > Kamin on ever acquired after the switch.
    gm3_test = paired_test(cells, KCTL, KAMIN, "stage2_ever_bound")
    switch_ok = all(r.get("passed") is True and r.get("rho_unchanged") is True
                    for r in switches if r["arm"] == KAMIN)
    lost_cells = {key: dict(c, lost=bool(c["stage2_ever_bound"] and not c.get("bound_final")))
                  for key, c in cells.items() if "stage2_ever_bound" in c}
    r1 = paired_test(lost_cells, KAMIN, KCTL, "lost")
    r2_tests = {a: paired_test(cells, a, A1) for a in (KV, KV10)}
    contrasts = []
    for a in CN:
        for c in live(cells, a):
            ev = c.get("corrupted_eval", {})
            clean, corrupted = ev.get("acc_clean_keys"), ev.get("acc_corrupted_keys")
            if clean is not None and corrupted is not None and ev.get("n_clean_keys", 0) and ev.get("n_corrupted_keys", 0):
                contrasts.append(dict(arm=a, seed=c["seed"], d=corrupted-clean, **{
                    k: ev[k] for k in ("acc_clean_keys", "acc_corrupted_keys", "n_clean_keys", "n_corrupted_keys")}))
    positive = sum(r["d"] > 0 for r in contrasts)
    negative = sum(r["d"] < 0 for r in contrasts)
    r3_p = binom_cdf(negative, positive+negative, .5) if positive+negative else 1.
    readings = holm(dict(
        R1=dict(p=r1["mcnemar_one_sided"], complete=r1["n"] == 10, test=r1,
                learn_then_lose=sum(c["lost"] for c in live(lost_cells, KAMIN))),
        R2=r2_reading(r2_tests),
        R3=dict(p=r3_p, complete=len(contrasts) == 24, n=len(contrasts),
                positive=positive, negative=negative, ties=len(contrasts)-positive-negative,
                rows=contrasts)))
    counts = {a: dict(n=len(live(cells, a)), bound_ever=sum(c["bound_ever"] for c in live(cells, a)),
                      bound_final=sum(c["bound_final"] for c in live(cells, a))) for a in ARMS}
    for a in (KAMIN, KLIN, KCTL):
        counts[a]["stage2_ever_bound"] = sum(c.get("stage2_ever_bound", False) for c in live(cells, a))
    eta_rows = [dict(arm=a, seed=c["seed"], **{k: p.get(k) for k in A9_FIELDS})
                for a in (KV, KV10) for c in live(cells, a) for p in c.get("probes", [])]
    d1_rows = [dict(arm=a, seed=c["seed"], write=write, **c["d1"][write])
               for a in (A1, EDGE) for c in live(cells, a) if c.get("d1")
               for write in ("write1", "write2")]
    rf_comparison = [dict(seed=s, **{a+"_"+f: cells.get((s, a), {}).get("eval", {}).get(f)
                                    for a in (RF, "e2e1emb_cn95")
                                    for f in ("query_acc", "query_ce", "ce")}) for s in range(6)]
    smoke = "podsmoke" in settings.get("process_name", "")
    if smoke:
        for r in readings.values():
            r["holm_pass"] = False
    for r in readings.values():
        r["status"] = "smoke_only" if smoke else "complete" if r["complete"] else "pending"
    complete = all(counts[a]["n"] == (6 if a in (*CN, RF) else 10) for a in ARMS)
    return dict(status="smoke_only" if smoke else "complete" if complete and not issues else "pending_or_issues",
                issues=issues, counts=counts, reproducibility=reproducibility(cells, anchor_table),
                GM2=dict(test=gm2_test, routes=route,
                         status="smoke_only" if smoke else "evaluated" if gm2_test["n"] == 10 else "pending",
                         passed=not smoke and gm2_test["n"] == 10 and
                         gm2_test["mcnemar_one_sided"] < .05 and route["passed"]),
                GM3=dict(test=gm3_test, switch_check_passed=switch_ok,
                         status="smoke_only" if smoke else "evaluated" if gm3_test["n"] == 10 else "pending",
                         statistical_pass=not smoke and gm3_test["n"] == 10 and gm3_test["mcnemar_one_sided"] < .05,
                         passed=not smoke and gm3_test["n"] == 10 and
                         gm3_test["mcnemar_one_sided"] < .05 and switch_ok),
                other_readings=readings, a6_trend=noise_rows(cells), rf95_vs_cn95=rf_comparison,
                a9_trajectories=eta_rows, d1=d1_rows, switch_checks=switches,
                kaminlin_descriptive=[dict(seed=c["seed"], stage2_ever_bound=c.get("stage2_ever_bound"),
                                           stage2_first_bound=c.get("stage2_first_bound"),
                                           bound_final=c["bound_final"], query_acc=c["eval"]["query_acc"])
                                      for c in live(cells, KLIN)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--anchor-table", type=Path, default=ANCHOR_TABLE)
    parser.add_argument("--s5-fp32", type=Path)
    args = parser.parse_args(argv)
    # A no-torch analyzer still follows the project's process naming convention.
    import setproctitle
    setproctitle.setproctitle(os.environ.get("PROCESS_NAME", "TTM-r44a-90-analysis"))
    overrides = ({r["cell_id"]: r["passed"] for r in json.loads(args.s5_fp32.read_text())}
                 if args.s5_fp32 else None)
    cells, settings, issues = records(args.log, overrides)
    summary = finite(summarize(cells, settings, issues, args.anchor_table))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        tables = {"reproducibility": summary["reproducibility"]["rows"],
                  "gm2-routes": summary["GM2"]["routes"]["rows"], "switch-checks": summary["switch_checks"],
                  "a6-trend": summary["a6_trend"], "a6-within-model": summary["other_readings"]["R3"]["rows"],
                  "rf95-vs-cn95": summary["rf95_vs_cn95"], "a9-trajectories": summary["a9_trajectories"],
                  "d1": summary["d1"], "kaminlin": summary["kaminlin_descriptive"],
                  "cells": [dict(seed=c["seed"], arm=c["arm"], eligible=valid(c),
                                 bound_ever=c.get("bound_ever"), bound_final=c.get("bound_final"),
                                 t_lose=t_lose(c), query_acc=c.get("eval", {}).get("query_acc"))
                            for c in cells.values()]}
        for name, rows in tables.items():
            write_csv(args.out / (name+".csv"), rows)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print("R44A_SUMMARY " + json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
