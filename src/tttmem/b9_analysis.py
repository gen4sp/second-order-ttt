"""B9 scoring; manipulation and rescue are reported jointly."""
from __future__ import annotations

import math
import statistics

from .frsb_analysis import index_units, count, mcnemar, holm, selectivity


def manipulation(row):
    probes = row.get("probes", [])
    by_step = {p["step"]: p.get("addr_h") for p in probes}
    grid = list(range(25, 1501, 25))
    # Only an observed early address failure earns FAILED. A numerical
    # divergence still counts against binding; it cannot invent an address probe.
    if any(isinstance(a, (int, float)) and math.isfinite(a) and a < .8
           for s, a in by_step.items() if s <= 300):
        return "FAILED"
    if row.get("disposition") == "DIVERGED":
        return "LATE-COLLAPSE"
    if sorted(by_step) != grid or any(not isinstance(by_step[s], (int, float)) or
                                     not math.isfinite(by_step[s]) for s in grid):
        return "MISSING"
    late = [by_step[s] for s in grid if s > 300]
    if any(by_step[s] < .8 for s in grid[-5:]) or sum(a >= .8 for a in late) < .9*len(late):
        return "LATE-COLLAPSE"
    return "OK"


def matching(key_rows, sham_rows, *, steps=1500):
    """Compare the actual auxiliary gradients in paired key/sham-trained arms.

    Counterfactual same-model reference norms are diagnostic only. The gate
    compares each arm's realized dose, paired by seed and completed update.
    """
    keys_by_seed = {r["seed"]: r for r in key_rows}
    shams_by_seed = {r["seed"]: r for r in sham_rows}
    shared = sorted(keys_by_seed.keys() & shams_by_seed.keys())
    keys = ("sham_anchor_count", "lag_ratio", "grad_ratio")
    values = {k: [] for k in keys}
    included, excluded = [], []
    for seed in shared:
        key_logs, sham_logs = [r[seed].get("aux_log", []) for r in (keys_by_seed, shams_by_seed)]
        if any([q.get("step") for q in rows] != list(range(1, steps+1)) for rows in (key_logs, sham_logs)):
            excluded.append(seed)
            continue
        pair_values = {k: [] for k in keys}
        valid = True
        for key, sham in zip(key_logs, sham_logs):
            numbers = [key.get("grad_norm"), sham.get("grad_norm"), key.get("lag"),
                       sham.get("lag"), sham.get("anchor_count")]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in numbers) or numbers[0] <= 0 or numbers[2] <= 0:
                valid = False
                break
            pair_values["sham_anchor_count"].append(numbers[4])
            pair_values["lag_ratio"].append(numbers[3]/numbers[2])
            pair_values["grad_ratio"].append(numbers[1]/numbers[0])
        if not valid:
            excluded.append(seed)
            continue
        included.append(seed)
        for k in keys:
            values[k].extend(pair_values[k])
    complete = len(included) >= 10
    med = {}
    for key in keys:
        if not values[key]:
            complete = False
            med[key] = None
        else:
            med[key] = statistics.median(values[key])
    ok = complete and med["sham_anchor_count"] >= 12 and med["lag_ratio"] >= .6 and .5 <= med["grad_ratio"] <= 2
    return dict(passed=ok, complete=complete, medians=med, n_pairs=len(included),
                included_seeds=included, excluded_seeds=excluded,
                source="actual auxiliary gradients; paired key/sham-trained arms and steps")


def analyze(payload, frsb_payload, smoke_only=False):
    smoke = bool(payload.get("manifest", {}).get("smoke"))
    if smoke != smoke_only:
        raise ValueError("smoke output requires --smoke-only; production cannot use it")
    aa, units = index_units(payload, smoke)
    bb, _ = index_units(frsb_payload, smoke)
    treated, sham, ce = [aa.get(("core", a), {}) for a in ("e2e1_lin_b9", "e2e1_lin_b9f", "e2e1_b9")]
    baseline = bb.get(("core", "e2e1_lin"), {})
    labels = {s: manipulation(r) for s, r in treated.items()}
    missing_labels = [s for s, label in labels.items() if label == "MISSING"]
    treated = {s: r for s, r in treated.items() if s not in missing_labels}
    labels = {s: label for s, label in labels.items() if s not in missing_labels}
    n = len(treated)
    label_counts = {k: sum(v == k for v in labels.values()) for k in ("OK", "FAILED", "LATE-COLLAPSE", "MISSING")}
    if n < 10:
        gate = "NOT EVALUABLE"
    elif label_counts["OK"] >= math.ceil(.75*n):
        gate = "OK"
    elif label_counts["FAILED"] >= max(label_counts["OK"], label_counts["LATE-COLLAPSE"]):
        gate = "MANIPULATION FAILED"
    else:
        gate = "LATE-COLLAPSE"
    rescue, specificity = mcnemar(treated, baseline), mcnemar(treated, sham)
    adjusted = holm(dict(rescue=rescue["p"], specificity=specificity["p"]))
    match = matching(list(treated.values()), list(sham.values()))
    counts = {a: {f: count(list(r.values()), f) for f in ("bound_final", "bound_ever")}
              for a, r in (("lin_b9", treated), ("lin_b9f", sham), ("ce_b9", ce))}
    verdict, qualifier = "B9-INCONCLUSIVE", None
    if n < 10:
        verdict = "NOT EVALUABLE"
    elif gate == "MANIPULATION FAILED":
        verdict = "MANIPULATION FAILED"
    elif counts["lin_b9"]["bound_final"]["k"] >= math.ceil(.75*n) and rescue["evaluable"] and adjusted["rescue"] <= .05:
        verdict = "B9-RESCUE"
        if match["passed"] and specificity["evaluable"] and counts["lin_b9f"]["bound_final"]["k"] <= math.floor(.25*len(sham)) and adjusted["specificity"] <= .05:
            qualifier = "KEY-SPECIFIC"
        elif match["passed"] and counts["lin_b9f"]["bound_final"]["k"] >= math.ceil(.75*len(sham)):
            qualifier = "GENERIC-IDENTITY"
    elif counts["lin_b9"]["bound_ever"]["k"] <= math.floor(.25*n):
        verdict = "B9-NO-RESCUE" if gate == "OK" else "LATE-COLLAPSE"
    elif counts["lin_b9"]["bound_final"]["k"] >= math.ceil(.75*n) and not rescue["evaluable"]:
        verdict = "NOT EVALUABLE"
    joint = [dict(seed=s, address=labels[s], bound_ever=r["bound_ever"], bound_final=r["bound_final"])
             for s, r in treated.items()]
    # CE auxiliary's address status is secondary and does not change the lin gate.
    ce_joint = [dict(seed=s, address=manipulation(r), bound_ever=r["bound_ever"], bound_final=r["bound_final"])
                for s, r in ce.items()]
    expected_n = len(payload["manifest"]["seeds"])
    two_barrier = "NOT EVALUABLE" if len(ce) < 10 else (
        counts["ce_b9"]["bound_ever"]["k"] >= math.ceil(.75*len(ce)) and
        counts["ce_b9"]["bound_final"]["k"] <= math.floor(.25*len(ce)))
    return dict(protocol="b9-v2", smoke=smoke, verdict=verdict, qualifier=qualifier,
                manipulation_gate=gate, manipulation_counts=label_counts, joint=joint, ce_joint=ce_joint,
                missing_address_seeds=missing_labels,
                sham_matching=match, counts=counts, rescue=rescue, specificity=specificity, holm2=adjusted,
                two_barrier=two_barrier, units=units,
                selectivity=selectivity(treated, ce, expected_n),
                common_eval_descriptive={name: {k: v for k, v in selectivity(treated, reference, expected_n).items() if k != "label"}
                                         for name, reference in (("vs_plain", baseline), ("vs_sham", sham))},
                reading="manipulation not sustained" if verdict == "LATE-COLLAPSE" else
                "this auxiliary at lambda .1 preserved top-1 address without rescuing plain lin" if verdict == "B9-NO-RESCUE" else None)
