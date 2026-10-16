"""Sealed FRS-B rules; stdlib only, independent of training and seed values."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

CORE_ARMS = ("e2e1emb", "e2e1emb_lin", "e2e1", "e2e1_lin", "e2e2")


def tail(k, n, p=.5):
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1))


def cp(k, n):
    if not n:
        return [0., 1.]
    def invert(k, target):
        lo, hi = 0., 1.
        for _ in range(70):
            mid = (lo+hi)/2
            if tail(k, n, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0. if not k else invert(k, .025), 1. if k == n else invert(k+1, .975)]


def count(rows, field):
    k, n = sum(bool(r[field]) for r in rows), len(rows)
    return dict(k=k, n=n, ci95=cp(k, n))


def mcnemar(a, b, field="bound_final"):
    shared = sorted(a.keys() & b.keys())
    pos = sum(bool(a[s][field]) and not b[s][field] for s in shared)
    neg = sum(bool(b[s][field]) and not a[s][field] for s in shared)
    return dict(n=len(shared), positive=pos, negative=neg,
                p=tail(pos, pos+neg) if len(shared) >= 10 else 1.,
                evaluable=len(shared) >= 10, seeds=shared)


def holm(values):
    out, previous = {}, 0.
    for i, name in enumerate(sorted(values, key=values.get)):
        previous = max(previous, min(1., (len(values)-i)*values[name]))
        out[name] = previous
    return out


def disposition(row, smoke=False):
    if not row or row.get("error") or row.get("skipped") or not row.get("trained"):
        return "MISSING"
    if row.get("diverged"):
        return "DIVERGED"
    horizon = int(row["steps"])
    if not smoke and horizon != 1500:
        return "TRUNCATED"
    quick, probes = row.get("quick", []), row.get("probes", [])
    expected = list(range(25, horizon+1, 25))
    if not smoke and ([q[0] for q in quick] != expected or
                      [p["step"] for p in probes] != expected):
        return "TRUNCATED"
    acc = row.get("eval", {}).get("query_acc")
    if not isinstance(acc, (int, float)) or not math.isfinite(acc):
        return "MISSING"
    if not quick or quick[-1][0] != horizon or not probes or probes[-1]["step"] != horizon:
        return "TRUNCATED"
    return "COMPLETE"


def index_units(payload, smoke=False):
    arms, units, seen = {}, [], set()
    for row in payload["cells"].values():
        key = row["block"], row["arm"], row["seed"]
        if key in seen:
            raise ValueError(f"duplicate unit {key}")
        seen.add(key)
        state = disposition(row, smoke)
        unit = dict(seed=row["seed"], arm=row["arm"], block=row["block"], disposition=state)
        if state in ("COMPLETE", "DIVERGED"):
            # Divergence is an adverse observation, never a missing/not-acquired substitution.
            ever = any(q[3] is not None and q[3] >= .9 for q in row.get("quick", []))
            accuracy = row.get("eval", {}).get("query_acc")
            final = isinstance(accuracy, (int, float)) and math.isfinite(accuracy) and accuracy >= .9
            quick_accuracy = row["quick"][-1][3] if row.get("quick") else None
            unit.update(bound_ever=ever if state == "COMPLETE" else False,
                        bound_final=final if state == "COMPLETE" else False,
                        t_addr_lost=row.get("t_addr_lost"),
                        quick_final_bound=isinstance(quick_accuracy, (int, float)) and math.isfinite(quick_accuracy) and quick_accuracy >= .9,
                        cross_eval=row.get("cross_eval", {}), probes=row.get("probes", []),
                        aux_log=row.get("aux_log", []))
            if state == "DIVERGED":
                observed_until = max([p["step"] for p in row.get("probes", [])] + [0])
                if unit["t_addr_lost"] is not None and unit["t_addr_lost"] > observed_until:
                    unit["t_addr_lost"] = None
            arms.setdefault((row["block"], row["arm"]), {})[row["seed"]] = unit
        units.append(unit)
    return arms, units


def selectivity(a, b, expected_n):
    def cross(rows, rule):
        out = {}
        for seed, r in rows.items():
            item = r.get("cross_eval", {}).get(rule, {})
            x = item.get("query_acc")
            if isinstance(x, (int, float)) and math.isfinite(x):
                out[seed] = dict(bound_final=x >= .9 if r["disposition"] == "COMPLETE" else False)
        return out
    al, bl = cross(a, "lin"), cross(b, "lin")
    pairs = mcnemar(al, bl)
    missing = 2*expected_n-len(al)-len(bl)
    k_l = sum(r["bound_final"] for s, r in al.items() if s in bl)
    k_c = sum(r["bound_final"] for s, r in bl.items() if s in al)
    n = pairs["n"]
    if missing > 2 or n < 10:
        label = "unknown"
    elif k_c <= math.floor(.25*n) and pairs["p"] <= .05:
        label = "training-selective"
    elif k_c >= math.ceil(.5*n):
        label = "eval-carried"
    else:
        label = "mixed"
    portable = list(cross(a, "ce").values())
    return dict(label=label, k_L=k_l, k_C=k_c, n=n, missing_cross_evals=missing,
                mcnemar=pairs, ce_portability=count(portable, "bound_final"))


def score_block(arms, expected_n):
    counts = {a: {f: count(list(rows.values()), f) for f in ("bound_ever", "bound_final")}
              for a, rows in arms.items()}
    ce, lin, emb, so = [arms.get(a, {}) for a in ("e2e1", "e2e1_lin", "e2e1emb_lin", "e2e2")]
    lin_ce, emb_lin = mcnemar(lin, ce), mcnemar(emb, lin)
    n = len(lin)
    e_ok = len(emb) >= 10 and sum(r["bound_final"] for r in emb.values()) >= math.ceil(.75*len(emb))
    pso = len(so) >= 10 and sum(r["bound_ever"] for r in so.values()) >= math.ceil(.5*len(so))
    unbound_ce = [r for r in ce.values() if not r["bound_ever"]]
    early_ce = sum(r["t_addr_lost"] is not None and r["t_addr_lost"] <= 100 for r in unbound_ce)
    pce = len(ce) >= 10 and sum(r["bound_ever"] for r in ce.values()) <= math.floor(len(ce)/12) and \
        bool(unbound_ce) and early_ce >= math.ceil(.75*len(unbound_ce))
    acq = [r for r in lin.values() if r["bound_ever"]]
    failed_retention = sum(not r["bound_final"] for r in acq)
    label, sublabel, reason = "INCONCLUSIVE", None, None
    if n < 10:
        label, reason = "NOT EVALUABLE", "fewer than 10 observations or complete comparison pairs"
    elif sum(r["bound_final"] for r in lin.values()) >= math.ceil(.75*n) and lin_ce["p"] <= .05:
        label = "LIN-ALONE"
    elif sum(r["bound_ever"] for r in lin.values()) <= math.floor(.25*n) and emb_lin["p"] <= .05 and e_ok:
        label = "CODE-NEEDED"
        unbound = [r for r in lin.values() if not r["bound_ever"]]
        early = sum(r["t_addr_lost"] is not None and r["t_addr_lost"] <= 200 for r in unbound)
        sublabel = "ADDRESS" if unbound and early >= math.ceil(.75*len(unbound)) else "OTHER"
    elif len(acq) >= 6 and failed_retention >= math.ceil(.75*len(acq)):
        label = "ACQUIRES-NOT-RETAINS"
    elif (sum(r["bound_final"] for r in lin.values()) >= math.ceil(.75*n) and not lin_ce["evaluable"]) or \
         (sum(r["bound_ever"] for r in lin.values()) <= math.floor(.25*n) and not emb_lin["evaluable"]):
        label, reason = "NOT EVALUABLE", "fewer than 10 observations or complete comparison pairs for a candidate row"
    elif not e_ok:
        reason = "no competent +emb-lin reference"
    return dict(verdict=label, sublabel=sublabel, reason=reason, counts=counts,
                P_CE=pce, P_SO=pso, E_LIN=e_ok,
                qualifiers=[] if pso else ["no working plain reference"],
                acquired=len(acq), acquired_retention_failures=failed_retention,
                lin_vs_ce=lin_ce, emb_lin_vs_lin=emb_lin,
                selectivity=selectivity(lin, ce, expected_n),
                rlin_selectivity=selectivity(emb, arms.get("e2e1emb", {}), expected_n),
                quick_full_agreement={a: sum(r["bound_final"] == r["quick_final_bound"] for r in rows.values())
                                      for a, rows in arms.items()})


def analyze(payload, smoke_only=False):
    smoke = bool(payload.get("manifest", {}).get("smoke"))
    if smoke and not smoke_only:
        raise ValueError("smoke output requires --smoke-only")
    if smoke_only and not smoke:
        raise ValueError("--smoke-only requires a smoke manifest")
    arms, units = index_units(payload, smoke)
    expected = len(payload["manifest"]["seeds"])
    scores = {}
    for block in ("core", "B40"):
        block_arms = {a: r for (b, a), r in arms.items() if b == block}
        if block == "core":
            scores[block] = score_block(block_arms, expected)
        elif block_arms:
            scores[block] = dict(counts={a: count(list(r.values()), "bound_final") for a, r in block_arms.items()},
                                 bound_ever={a: count(list(r.values()), "bound_ever") for a, r in block_arms.items()},
                                 cross_eval={a: {rule: count([dict(bound_final=r["disposition"] == "COMPLETE" and
                                                   r["cross_eval"][rule]["query_acc"] >= .9)
                                                   for r in rows.values() if isinstance(r.get("cross_eval", {}).get(rule, {}).get("query_acc"), (int, float))],
                                                   "bound_final") for rule in ("ce", "lin")}
                                             for a, rows in block_arms.items()},
                                 note="fixed-eta40 descriptive bridge; no CE40 reference")
    core = scores["core"]
    trigger = not smoke and core["verdict"] == "CODE-NEEDED" and core["sublabel"] == "ADDRESS"
    # Nominal R-LIN/R-ADDR inputs, for the pool analyzer's single Holm-5 family.
    replication = {"R-LIN": mcnemar(arms.get(("core", "e2e1emb_lin"), {}), arms.get(("core", "e2e1emb"), {})),
                   "R-ADDR": mcnemar(arms.get(("core", "e2e1emb"), {}), arms.get(("core", "e2e1"), {}), "bound_ever")}
    family = {}
    for name, test_arm, comparator_arm, field in (
            ("R-LIN", "e2e1emb_lin", "e2e1emb", "bound_final"),
            ("R-ADDR", "e2e1emb", "e2e1", "bound_ever")):
        a, b = arms.get(("core", test_arm), {}), arms.get(("core", comparator_arm), {})
        shared = sorted(a.keys() & b.keys())
        family[name] = dict(seeds=shared, test=[bool(a[s][field]) for s in shared],
                            comparator=[bool(b[s][field]) for s in shared],
                            disposition_test=[a[s]["disposition"] for s in shared],
                            disposition_comparator=[b[s]["disposition"] for s in shared],
                            diverged=[a[s]["disposition"] == "DIVERGED" or b[s]["disposition"] == "DIVERGED" for s in shared])
    gate_reason = "CODE-NEEDED/ADDRESS" if trigger else (
        "smoke-only" if smoke else core["reason"] if core["verdict"] == "NOT EVALUABLE"
        else "registered trigger not met")
    return dict(protocol="frsb-v2", smoke=smoke, scores=scores, units=units,
                replication_inputs=replication, family_inputs=family,
                b9_gate=dict(run_b9=trigger, reason=gate_reason))


def write_scores(out, scores, prefix="frsb"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/f"{prefix}_scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False)+"\n")
    brief = {k: v for k, v in scores.items() if k != "units"}
    (out/f"{prefix}_scores.txt").write_text(json.dumps(brief, indent=2, allow_nan=False)+"\n")
    with (out/f"{prefix}_units.csv").open("w") as f:
        fields = ["seed", "block", "arm", "disposition", "bound_ever", "bound_final", "t_addr_lost"]
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores["units"])
    if "b9_gate" in scores:
        (out/"b9_gate.json").write_text(json.dumps(scores["b9_gate"], indent=2)+"\n")
