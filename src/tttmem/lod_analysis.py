"""Registered LOD inference, independent of training and optional plotting."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np

LOADS = (16, 32, 64, 128)
DELTA = .05
MIN_N = 6
NE = "NOT EVALUABLE"


def beta_cf(a, b, x):
    # Continued fraction for the regularized incomplete beta, in double precision.
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1., 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        aa = m * (b - m) * x / ((qam + 2 * m) * (a + 2 * m))
        d = 1 + aa * d
        c = 1 + aa / c
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + 2 * m) * (qap + 2 * m))
        d = 1 + aa * d
        c = 1 + aa / c
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        change = d * c
        h *= change
        if abs(change - 1) < 3e-14:
            break
    return h


def ibeta(a, b, x):
    if x <= 0:
        return 0.
    if x >= 1:
        return 1.
    scale = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return scale * beta_cf(a, b, x) / a
    return 1 - scale * beta_cf(b, a, 1 - x) / b


def beta_quantile(p, a, b):
    lo, hi = 0., 1.
    for _ in range(70):
        mid = (lo + hi) / 2
        if ibeta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t95(df):
    lo, hi = 0., 100.
    for _ in range(70):
        mid = (lo + hi) / 2
        cdf = 1 - .5 * ibeta(df / 2, .5, df / (df + mid * mid))
        if cdf < .95:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return dict(n=0, mean=None, LB=None, UB=None)
    mean = float(values.mean())
    constant = bool(np.all(values == values[0]))
    if constant:
        mean = float(values[0])
    width = t95(n - 1) * float(values.std(ddof=1)) / math.sqrt(n) if n > 1 and not constant else 0.
    return dict(n=n, mean=mean, LB=mean - width, UB=mean + width)


def ni(values, delta=DELTA, minimum=MIN_N):
    row = interval(values)
    row["label"] = NE if row["n"] < minimum else "FREE" if row["LB"] > -delta else "PRICED" if row["UB"] < -delta else "INCONCLUSIVE"
    row.update(margin=delta, scope="t-model; approximate normality of seed differences", multiplicity="unadjusted secondary")
    return row


def cp_bounds(k, n):
    if not n:
        return dict(k=k, n=n, upper95=None, CI95=None)
    return dict(k=k, n=n, upper95=1. if k == n else beta_quantile(.95, k + 1, n - k),
        CI95=[0. if k == 0 else beta_quantile(.025, k, n - k + 1),
              1. if k == n else beta_quantile(.975, k + 1, n - k)])


def failure_counts(cells):
    return cp_bounds(sum(c["composite"]["A"] < .5 for c in cells), len(cells))


def competent(cells):
    n = len(cells)
    return bool(n and np.mean([c["composite"]["A"] for c in cells]) >= .90 - 1e-12
                and sum(c["retained"] for c in cells) >= math.ceil(.75 * n))


def complete_families(index, arms, loads=LOADS):
    seeds = sorted({key[3] for key in index})
    return [s for s in seeds if all(("core" if a in ("lin", "so") else "aice", a, n, s) in index
                                   for a in arms for n in loads)]


def sign_p(slopes, direction):
    nonzero = np.asarray(slopes)[np.asarray(slopes) != 0]
    m = len(nonzero)
    k = int((direction * nonzero > 0).sum())
    return sum(math.comb(m, j) for j in range(k, m + 1)) / 2**m if m else 1.


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    previous, adjusted = 0., {}
    for i, name in enumerate(order):
        previous = max(previous, min(1., (len(order) - i) * pvalues[name]))
        adjusted[name] = previous
    return adjusted


def slope_rule(slopes, direction, adjusted_p=None):
    row = interval(slopes)
    p = sign_p(slopes, direction) if len(slopes) >= MIN_N else 1.
    adj = p if adjusted_p is None else adjusted_p
    row.update(raw_p=p, p_adj=adj, nonzero=int(np.count_nonzero(slopes)), slopes=list(map(float, slopes)))
    bound = DELTA / 3
    if row["n"] < MIN_N:
        row.update(label=NE, magnitude="UNRESOLVED")
        return row
    lb, ub = row["LB"], row["UB"]
    magnitude = "NEGLIGIBLE" if lb >= -bound and ub <= bound else "MATERIAL" if (
        ub < -bound if direction == -1 else lb > bound) else "UNRESOLVED"
    if adj <= .05:
        label = {"MATERIAL": "SUPPORTED-MATERIAL", "UNRESOLVED": "SUPPORTED-DIRECTION",
                 "NEGLIGIBLE": "NEGLIGIBLE-TREND"}[magnitude]
    else:
        label = "FALSIFIED" if magnitude == "NEGLIGIBLE" else "INCONCLUSIVE"
    row.update(label=label, magnitude=magnitude, estimand="directional consistency; t-model mean magnitude")
    return row


def interference_label(values):
    row = ni(values)
    row["label"] = {"FREE": "INTERFERENCE-FREE", "PRICED": "INTERFERENCE-PRICED"}.get(row["label"], row["label"])
    row["scope"] = "read-time older-write effect for the same trained models; coupled histories mildly off-distribution"
    return row


def bridge_label(values):
    values = np.asarray(values)
    label = "LEARNED-HIGHER" if len(values) == 4 and (values > DELTA).sum() >= 3 else (
        "NO-DIFFERENCE-SHOWN" if len(values) == 4 and (np.abs(values) <= DELTA).sum() >= 3 else "UNCERTAIN")
    return dict(label=label, n=len(values), differences=values.tolist(), scope="descriptive lin eta sensitivity; no transfer to historical recipe")


def indexed(cells):
    return {(c["group"], c["arm"], c["load"], c["seed"]): c for c in cells
            if c["status"] in ("COMPLETE", "DIVERGED") and c.get("composite", {}).get("A") is not None}


def pairs(index, group1, arm1, group2, arm2, load):
    seeds = sorted({k[3] for k in index})
    return [(index[(group1, arm1, load, s)], index[(group2, arm2, load, s)]) for s in seeds
            if (group1, arm1, load, s) in index and (group2, arm2, load, s) in index]


def plateau(cells):
    result = {}
    for group, arm, load in sorted({(c["group"], c["arm"], c["load"]) for c in cells}):
        cohort = [c for c in cells if (c["group"], c["arm"], c["load"]) == (group, arm, load)]
        changes = []
        for c in cohort:
            logs = c.get("logs", [])
            early = [r["quick"]["acc"] for r in logs if 1000 <= r["step"] <= 1200]
            late = [r["quick"]["acc"] for r in logs if 1300 <= r["step"] <= 1500]
            if early and late:
                changes.append(float(np.mean(late) - np.mean(early)))
        result[f"{group}/{arm}/{load}"] = dict(n=len(cohort), n_measured=len(changes), changes=changes,
            label="budget-limited" if sum(d > .02 for d in changes) > math.floor(.25 * len(cohort))
                  else "plateau" if len(changes) == len(cohort) else NE)
    return result


def score(cells, tier="T1", layout_valid=True):
    ix = indexed(cells)
    slope_loads = (16, 64, 128) if tier == "cheap" else LOADS
    out = {h: {} for h in ("H1a", "H1b", "H1c", "H3b", "H3_difference", "Q_SO", "Q_lin", "Bridge", "failures")}
    out["Q_plat"] = plateau(list(ix.values()))
    leaks = [c.get("observed", {}).get("W0", c["composite"].get("W0")) > c.get("B_nb", 1.) + .05
             for c in ix.values() if c.get("observed", {}).get("W0", c["composite"].get("W0")) is not None]
    valid = layout_valid and not any(leaks)
    out["valid"] = valid
    for n in LOADS:
        paired = pairs(ix, "core", "lin", "core", "so", n)
        qso, qlin = competent([b for a, b in paired]), competent([a for a, b in paired])
        out["Q_SO"][str(n)] = qso
        out["Q_lin"][str(n)] = qlin
        for arm in ("lin", "so", "aice"):
            cohort = [c for c in ix.values() if c["arm"] == arm and c["load"] == n and c["group"] in ("core", "aice")]
            out["failures"][f"{arm}/{n}"] = failure_counts(cohort)
        for hypothesis, endpoint in (("H1a", "A"), ("H1c", "A_last")):
            cohort = [(a, b) for a, b in paired if a["composite"].get(endpoint) is not None and b["composite"].get(endpoint) is not None]
            row = ni([a["composite"][endpoint] - b["composite"][endpoint] for a, b in cohort])
            row.update(seeds=[a["seed"] for a, b in cohort], Q_SO=competent([b for a, b in cohort]),
                       failures_lin=failure_counts([a for a, b in cohort]), failures_so=failure_counts([b for a, b in cohort]))
            if not valid or not row["Q_SO"]:
                row["label"] = NE
            if hypothesis == "H1a":
                row["delta02"] = ni([a["composite"][endpoint] - b["composite"][endpoint] for a, b in cohort], .02)
                row["delta02"].update(failures_lin=row["failures_lin"], failures_so=row["failures_so"])
                if not valid or not row["Q_SO"]:
                    row["delta02"]["label"] = NE
                la, so = [a["composite"]["A"] for a, b in cohort], [b["composite"]["A"] for a, b in cohort]
                denom = 1 - np.mean(so) if so else 0
                row["error_ratio"] = float((1 - np.mean(la)) / denom) if denom > 0 else None
                row["error_ratio_note"] = "undefined when SO has no errors" if denom == 0 else None
            out[hypothesis][str(n)] = row
        robust = pairs(ix, "core", "lin", "core", "lin", n)
        robust = [(a, ix[("core", "lin", 16, a["seed"])]) for a, _ in robust if ("core", "lin", 16, a["seed"]) in ix]
        out["H1b"][str(n)] = ni([a["composite"]["A"] - b["composite"]["A"] for a, b in robust])
        out["H1b"][str(n)]["failures_lin"] = failure_counts([a for a, b in robust])
        aice_pairs = pairs(ix, "aice", "aice", "core", "so", n)
        row = ni([a["composite"]["A"] - b["composite"]["A"] for a, b in aice_pairs])
        row.update(Q_SO=competent([b for a, b in aice_pairs]), seeds=[a["seed"] for a, b in aice_pairs],
                   failures_aice=failure_counts([a for a, b in aice_pairs]), failures_so=failure_counts([b for a, b in aice_pairs]))
        if not valid or not row["Q_SO"]:
            row["label"] = NE
        out["H3b"][str(n)] = row
        aice_lin = pairs(ix, "aice", "aice", "core", "lin", n)
        out["H3_difference"][str(n)] = dict(**interval([a["composite"]["A"] - b["composite"]["A"]
                                                       for a, b in aice_lin]), scope="descriptive e(N)")
    baseline = out["Q_lin"]["16"]
    joint_failure = not baseline and not out["Q_SO"]["16"]
    for n in LOADS:
        row = out["H1b"][str(n)]
        if not valid or not baseline:
            row["label"] = "VOID" if valid else NE
        else:
            row["label"] = {"FREE": "FREE-like", "PRICED": "degrades"}.get(row["label"], row["label"])
        if not baseline:
            out["H1a"][str(n)]["baseline_note"] = "lin baseline not competent; relative label only"
        if joint_failure and n == 16:
            for h in ("H1a", "H1c", "H3b"):
                out[h][str(n)]["label"] = NE
    # Decomposition cohort: both arms at all three schedules, then cohort-specific SO gates.
    seeds = sorted({k[3] for k in ix})
    schedules = (("core", 16), ("control", 16), ("core", 128))
    dseeds = [s for s in seeds if all((g, a, n, s) in ix for g, n in schedules for a in ("lin", "so"))]
    dvalid = valid and len(dseeds) >= MIN_N and all(competent([ix[(g, "so", n, s)] for s in dseeds]) for g, n in schedules)
    decomposition = dict(n=len(dseeds), seeds=dseeds, label="DESCRIPTIVE" if dvalid else NE,
                         scope="schedule decomposition; no capacity/interference reading",
                         Q_SO_by_schedule={f"{g}/{n}": competent([ix[(g, "so", n, s)] for s in dseeds])
                                           for g, n in schedules})
    for arm in ("lin", "so"):
        pos = [ix[("control", arm, 16, s)]["composite"]["A"] - ix[("core", arm, 16, s)]["composite"]["A"] for s in dseeds]
        old = [ix[("core", arm, 128, s)]["composite"]["A"] - ix[("control", arm, 16, s)]["composite"]["A"] for s in dseeds]
        decomposition[arm] = dict(c_pos=interval(pos), c_old=interval(old), pos=pos, old=old)
    decomposition["d_pos"] = interval(np.subtract(decomposition["lin"]["pos"], decomposition["so"]["pos"]))
    decomposition["d_old"] = interval(np.subtract(decomposition["lin"]["old"], decomposition["so"]["old"]))
    out["H1d"] = decomposition
    ipairs = [(a, b) for a, b in pairs(ix, "core", "lin", "core", "so", 128)
              if all(c["composite"].get("A_coupled") is not None for c in (a, b))]
    row = interference_label([(a["composite"]["A_last"] - a["composite"]["A_coupled"])
                            - (b["composite"]["A_last"] - b["composite"]["A_coupled"]) for a, b in ipairs])
    row.update(seeds=[a["seed"] for a, b in ipairs], Q_SO=competent([b for a, b in ipairs]),
               failures_lin=failure_counts([a for a, b in ipairs]), failures_so=failure_counts([b for a, b in ipairs]))
    if not valid or not row["Q_SO"]:
        row["label"] = NE
    row["L16_coupled_reference"] = {a: [c["composite"].get("A_coupled") for c in ix.values()
        if c["group"] == "control" and c["arm"] == a] for a in ("lin", "so")}
    out["H1i"] = row
    slopes = {}
    for h, arms, direction in (("H2", ("lin", "so"), -1), ("H3a", ("aice", "lin"), 1)):
        ss = complete_families(ix, arms, slope_loads)
        matrix = np.array([[ix[("aice" if a == "aice" else "core", a, n, s)]["composite"]["A"]
                            for n in slope_loads] for s in ss for a in arms]).reshape(len(ss), 2, len(slope_loads))
        diff = matrix[:, 0] - matrix[:, 1]
        x = np.log2(np.asarray(slope_loads) / 16)
        x -= x.mean()
        # Center the response too: an exactly constant difference has an exact
        # zero slope, so numerical cancellation cannot manufacture sign evidence.
        bs = (diff - diff.mean(1, keepdims=True)) @ x / (x @ x)
        slopes[h] = slope_rule(bs.tolist(), direction)
        slopes[h]["seeds"] = ss
        gate = valid and (h != "H2" or all(competent([ix[("core", "so", n, s)] for s in ss]) for n in slope_loads))
        if not gate:
            slopes[h].update(label=NE, raw_p=1.)
    family = ("H2",) if tier in ("T3", "cheap") else ("H2", "H3a")
    adjusted = holm({h: slopes[h]["raw_p"] for h in family})
    for h, direction in (("H2", -1), ("H3a", 1)):
        if slopes[h]["label"] != NE and h in family:
            slopes[h].update(slope_rule(slopes[h]["slopes"], direction, adjusted[h]))
        slopes[h]["p_adj"] = adjusted.get(h, 1.)
        slopes[h]["holm_family"] = list(family)
        out[h] = slopes[h]
    for n in (16, 128):
        bridge = pairs(ix, "bridge", "bridge", "core", "lin", n)
        brow = bridge_label([a["composite"]["A"] - b["composite"]["A"] for a, b in bridge])
        brow["learned_vs_fixed_SO_descriptive"] = [a["composite"]["A"] - b["composite"]["A"]
            for a, b in pairs(ix, "bridge", "bridge", "core", "so", n)]
        if not valid or (joint_failure and n == 16):
            brow["label"] = NE
        out["Bridge"][str(n)] = brow
    out["H4"] = {}
    for n in (16, 128):
        fo = [c for c in ix.values() if c["arm"] == "fo" and c["load"] == n]
        summaries = []
        for c in fo:
            label = "retained" if c["retained"] else "never-acquired" if c["acquired"] is None else "acquired-then-lost"
            after = [r for r in c.get("logs", []) if c["acquired"] is not None and r["step"] > c["acquired"] and r["quick"]["acc"] < .9]
            times = {"acquisition": c["acquired"], "loss": after[0]["step"] if after else None, "final": 1500}
            residuals = {name: next((r["write"] for r in c.get("logs", []) if r["step"] == step), None) for name, step in times.items()}
            summaries.append(dict(seed=c["seed"], label=label, residuals=residuals))
        out["H4"][str(n)] = dict(cells=summaries, changed_FO_regime=n == 16 and sum(c["retained"] for c in fo) >= 2,
                                     label="DESCRIPTIVE" if valid else NE)
    allfree = all(out["H1a"][str(n)]["label"] == "FREE" for n in LOADS)
    global_gate = valid and all(out["Q_SO"].values()) and all(out["H1a"][str(n)]["n"] >= MIN_N for n in LOADS)
    relative = "FREE" if global_gate and allfree else "NOT FREE" if global_gate else NE
    deployment = "STRENGTHENED" if relative == "FREE" and all(out["Q_lin"].values()) else "NO CLAIM" if global_gate else NE
    nstar = None
    if out["H1a"]["16"]["label"] == "FREE":
        for n in LOADS:
            if out["H1a"][str(n)]["label"] != "FREE":
                break
            nstar = n
    out["claims"] = dict(relative_global=relative, deployment=deployment, N_star=nstar,
        N_star_note="no FREE baseline" if nstar is None else None, lin_baseline_competent=baseline,
        precedence="INVALID" if not valid else "JOINT_BASELINE_FAILURE" if joint_failure else
                   "LIN_BASELINE_FAILURE" if not baseline else "SO_GATE_FAILURE" if not all(out["Q_SO"].values()) else "VALID",
        scope="fixed eta=10; streaming schedule; toy with frozen key code; t-model",
        trend_caveat="deficit grows" if out["H2"]["label"].startswith("SUPPORTED") else "unresolved" if out["H2"]["label"] in (NE, "INCONCLUSIVE") else "negligible trend",
        failures_lin={str(n): out["failures"][f"lin/{n}"] for n in LOADS})
    pooled = [c for c in ix.values() if c["group"] == "core" and c["arm"] == "lin"]
    out["pooled_lin_failures"] = dict(**failure_counts(pooled), scope="independence-dependent; shared seeds across loads; per-load bounds conservative")
    out["counts"] = {a: {str(n): {metric: cp_bounds(sum(bool(c[metric]) for c in ix.values() if c["arm"] == a and c["load"] == n and c["group"] == "core"),
        sum(c["arm"] == a and c["load"] == n and c["group"] == "core" for c in ix.values())) for metric in ("acquired", "retained")}
        for n in LOADS} for a in ("lin", "so")}
    out["cohort_counts"] = []
    for group, arm, load in sorted({(c["group"], c["arm"], c["load"]) for c in ix.values()}):
        cohort = [c for c in ix.values() if (c["group"], c["arm"], c["load"]) == (group, arm, load)]
        out["cohort_counts"].append(dict(group=group, arm=arm, load=load,
            **{metric: cp_bounds(sum(bool(c[metric]) for c in cohort), len(cohort))
               for metric in ("acquired", "retained")}))
    out["McNemar"] = {}
    for n in LOADS:
        cohort = pairs(ix, "core", "lin", "core", "so", n)
        discord = [(a["retained"], b["retained"]) for a, b in cohort if a["retained"] != b["retained"]]
        k = sum(a for a, b in discord)
        out["McNemar"][str(n)] = dict(discordant=len(discord), p=min(1., 2 * sum(math.comb(len(discord), j) for j in range(min(k, len(discord) - k) + 1)) / 2**len(discord)) if discord else 1., scope="descriptive")
    out["H5"] = exploratory(ix, valid)
    if tier == "cheap":
        for h in ("H1b", "H1c", "H3b", "Bridge"):
            for row in out[h].values():
                row["label"] = NE
        for h in ("H1d", "H1i", "H3a", "H5"):
            out[h]["label"] = NE
    out["interpretation"] = interpretation(out)
    if not valid:
        for h in ("H2", "H3a", "H1d", "H1i"):
            out[h]["label"] = NE
    return out


def interpretation(out):
    claims = out["claims"]
    if not out["valid"]:
        return "INVALID: layout or Q-mem failed; no scientific reading."
    if claims["precedence"] == "JOINT_BASELINE_FAILURE":
        return "Neither rule competent at 16; H1b void, global claims not evaluable."
    if claims["precedence"] == "LIN_BASELINE_FAILURE":
        return "Lin baseline not competent; relative labels only, H1b void; bridge descriptive."
    if claims["relative_global"] == "FREE":
        if claims["deployment"] != "STRENGTHENED":
            return "Relative global FREE (t-model); lin competence fails at a load, no deployment claim."
        return "Deployment strengthened up to 128 at fixed eta under this schedule and reported failure bounds; " + claims["trend_caveat"] + "."
    if claims["precedence"] == "SO_GATE_FAILURE":
        return "SO reference not competent at a load; no relative claim there, H1b carries the absolute reading."
    if claims["N_star"] is None:
        return "No FREE baseline; no load reading, per-load relative labels remain listed."
    if out["H1a"]["32"]["label"] == "PRICED":
        return "Fixed-eta lin priced already at 32; bridge is descriptive, with no price transfer to historical recipes."
    if any(out["H1a"][str(n)]["label"] == "PRICED" for n in (64, 128)):
        if out["H1i"]["label"] == "INTERFERENCE-PRICED":
            return "Lin qualified below N*: same-model read-time older-write interference shown; no training-time or capacity mechanism. AICE FREE may support a high-load alternative."
        return "Lin qualified below N*: a schedule price; read-time interference not shown. H1d descriptive only."
    return "Relative evidence inconclusive under the registered gates and complete-case minimums."


def exploratory(index, valid):
    cells = list(index.values())
    geometry = [{"cell_id": c.get("cell_id"), "arm": c["arm"], "load": c["load"], "seed": c["seed"],
                 "geometry": c.get("diagnostic", {}).get("geometry", {})} for c in cells]
    phase = [{"arm": c["arm"], "load": c["load"], "seed": c["seed"],
              "gap_effect": c["composite"].get("A_freeze", 0) - c["composite"].get("A_snap", 0)
                 if all(c["composite"].get(k) is not None for k in ("A_freeze", "A_snap")) else None,
              "probe_effect": c["composite"].get("A", 0) - c["composite"].get("A_freeze", 0)
                 if c["composite"].get("A_freeze") is not None else None,
              "profile": c.get("diagnostic", {}).get("profile", [])} for c in cells]
    associations = []
    # Seed-level within-load associations. Residualized load coefficient is
    # descriptive and unregularized; rank deficiency is reported explicitly.
    for n in (64, 128):
        ps = pairs(index, "core", "lin", "core", "so", n)
        rows = []
        for a, b in ps:
            g = a.get("diagnostic", {}).get("geometry", {}).get("x", {})
            keys = ("shared_mean_cos", "write_probe_cos", "participation_ratio", "write_norm")
            if all(k in g for k in keys):
                rows.append([a["composite"]["A"] - b["composite"]["A"], *[g[k] for k in keys]])
        data = np.array(rows)
        corrs = {}
        if len(rows) >= 3:
            for i, k in enumerate(keys, 1):
                corrs[k] = float(np.corrcoef(data[:, 0], data[:, i])[0, 1]) if data[:, 0].std() and data[:, i].std() else None
        associations.append(dict(load=n, n=len(rows), correlations=corrs))
    records = []
    for n in LOADS:
        for a, b in pairs(index, "core", "lin", "core", "so", n):
            g = a.get("diagnostic", {}).get("geometry", {}).get("x", {})
            keys = ("shared_mean_cos", "write_probe_cos", "participation_ratio", "write_norm")
            if all(k in g for k in keys):
                records.append([a["composite"]["A"] - b["composite"]["A"], math.log2(n / 16), *[g[k] for k in keys]])
    residualized = dict(n=len(records), load_coefficient=None)
    if records:
        data = np.asarray(records)
        design = np.column_stack((np.ones(len(data)), data[:, 1:]))
        coef, _, rank, _ = np.linalg.lstsq(design, data[:, 0], rcond=None)
        residualized.update(load_coefficient=float(coef[1]), rank=int(rank), columns=design.shape[1])
    return dict(label="DESCRIPTIVE" if valid else NE, geometry=geometry, phase=phase,
                within_load_associations=associations, residualized_load=residualized)


def contract_errors(manifest, cells, diagnostic=False):
    from . import exp_lod as lod
    errors = []
    if manifest.get("config_sha256") != lod.sha(manifest.get("config")):
        errors.append("config hash mismatch")
    if manifest.get("source_hashes") != lod.source_hashes():
        errors.append("source/plan/registry hash mismatch")
    if not manifest.get("created_before_production"):
        errors.append("tier was not fixed before production")
    admitted = manifest["cells"]
    expected = {lod.cell_id(c): c for c in admitted}
    if not diagnostic:
        lod.seal_guard(manifest["config"])
        tier = manifest["admission"]["tier"]
        if admitted != lod.manifest_cells(tier, manifest["admission"]["addon"]):
            errors.append("admitted manifest is not a sealed tier")
        if manifest["config"].get("device") != "cuda" or manifest["config"].get("tf32") is not False:
            errors.append("CUDA/TF32 contract")
    seen, pairing, inits, codes = set(), {}, {}, {}
    for cell in cells:
        cid = cell["cell_id"]
        if cid in seen or cid not in expected:
            errors.append("duplicate or unadmitted cell: " + cid)
        seen.add(cid)
        if cell.get("config_sha256") != manifest["config_sha256"] or cell.get("source_hashes") != manifest["source_hashes"]:
            errors.append("stale cell source/config: " + cid)
        if cid in expected and any(cell.get(k) != v for k, v in expected[cid].items()):
            errors.append("cell identity mismatch: " + cid)
        if cell["status"] == "LAYOUT_INVALID":
            errors.append("layout audit failed before training: " + cid)
        if cell["status"] not in ("COMPLETE", "DIVERGED"):
            continue
        if not diagnostic and (cell.get("device") != "cuda" or cell.get("tf32") is not False):
            errors.append("cell device contract: " + cid)
        if not all(cell.get("layout_audit", {}).values()) or not cell.get("layout_audit"):
            errors.append("layout audit: " + cid)
        if cell["status"] == "COMPLETE" and cell["completed_steps"] != manifest["config"]["steps"]:
            errors.append("incomplete final: " + cid)
        if cell["status"] == "DIVERGED" and (not cell.get("certified_divergence") or any(
                v != 1/16 for v in cell["composite"].values())):
            errors.append("uncertified numerical failure: " + cid)
        key = (cell["seed"], cell["load"], cell["layout"])
        for key_, field, table in ((key, "first_batch_sha256", pairing),
            (cell["seed"], "init_sha256", inits), (cell["seed"], "frozen_code_sha256", codes)):
            val = cell.get(field)
            if not val or key_ in table and table[key_] != val:
                errors.append("pairing " + field + ": " + cid)
            table[key_] = val
    return errors


def analyze_directory(path, diagnostic=False):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cells = [json.loads((path / (c["group"] + "_" + c["arm"] + f"_N{c['load']}_s{c['seed']}.json")).read_text())
             for c in manifest["cells"] if (path / (c["group"] + "_" + c["arm"] + f"_N{c['load']}_s{c['seed']}.json")).exists()]
    errors = contract_errors(manifest, cells, diagnostic)
    if errors:
        raise ValueError("production contract violations: " + "; ".join(errors))
    result = score(cells, manifest["admission"]["tier"], layout_valid=not manifest.get("infra_stop"))
    result["mode"] = "SMOKE ONLY; no scientific claims" if diagnostic else "production"
    if manifest.get("infra_stop"):
        result["mode"] = "INFRA_GPU_ERRORS; no scientific claims"
        result["infra_stop"] = manifest["infra_stop"]
    result["manifest_sha256"] = __import__("hashlib").sha256((path / "manifest.json").read_bytes()).hexdigest()
    result["missing"] = [c for c in manifest["cells"] if not any(r["cell_id"] ==
        f"{c['group']}_{c['arm']}_N{c['load']}_s{c['seed']}" and r["status"] in ("COMPLETE", "DIVERGED") for r in cells)]
    errors_by_value, error_rank_share = [], []
    for cell in indexed(cells).values():
        npz = path / (cell["cell_id"] + "_probes.npz")
        if not npz.exists():
            if cell["status"] == "COMPLETE":
                raise ValueError("missing probe records: " + cell["cell_id"])
            continue
        with np.load(npz) as data:
            from . import exp_lod as lod
            if lod.file_sha(npz) != cell.get("probes_sha256"):
                raise ValueError("probe file hash mismatch: " + cell["cell_id"])
            incorrect = data["prediction"] != data["query_tok"]
            hist = data["value_histogram"]
            val0 = manifest["config"].get("pool", 256) + 2
            probe_meta = dict(value_histogram=hist, query_tok=data["query_tok"], value0=val0)
            if not math.isclose(lod.binding_free_baseline(probe_meta), cell["B_nb"], abs_tol=1e-12):
                raise ValueError("B_nb mismatch: " + cell["cell_id"])
            pred_idx = data["prediction"] - val0
            valid_pred = (pred_idx >= 0) & (pred_idx < 16)
            pred_count = np.take_along_axis(hist, pred_idx.clip(0, 15), 1)
            above = valid_pred & (pred_count > np.median(hist, axis=1)[:, None])
            rank = ((hist[:, None, :] < pred_count[..., None]).sum(-1)
                    + .5 * (hist[:, None, :] == pred_count[..., None]).sum(-1)) / 16
            bins = np.where(valid_pred, np.minimum((rank * 4).astype(int), 3), 4)
            bin_counts = np.bincount(bins[incorrect], minlength=5)
            error_rank_share.append(dict(cell_id=cell["cell_id"], arm=cell["arm"], load=cell["load"],
                labels=["Q1", "Q2", "Q3", "Q4", "non-value"], counts=bin_counts.tolist(),
                shares=(bin_counts / bin_counts.sum()).tolist() if bin_counts.sum() else None))
            errors_by_value.append(dict(cell_id=cell["cell_id"], arm=cell["arm"], load=cell["load"],
                n_errors=int(incorrect.sum()), above_median_error_share=float(above[incorrect].mean()) if incorrect.any() else None))
    result["H5"]["error_share"] = errors_by_value
    result["H5"]["error_rank_share"] = error_rank_share
    if diagnostic:
        result["claims"] = dict(relative_global="SMOKE ONLY", deployment="SMOKE ONLY", N_star=None)
    return result


def write_outputs(path, result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lod_scores.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = []
    for h in ("H1a", "H1b", "H1c", "H3b", "Bridge"):
        for n, row in result[h].items():
            rows.append(dict(hypothesis=h, load=n, label=row["label"], n=row["n"],
                             mean=row.get("mean"), LB=row.get("LB"), UB=row.get("UB")))
    for h in ("H2", "H3a", "H1d", "H1i"):
        row = result[h]
        rows.append(dict(hypothesis=h, load="all", label=row["label"], n=row["n"],
                         mean=row.get("mean"), LB=row.get("LB"), UB=row.get("UB")))
    with (out / "lod_verdicts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [result["mode"], "", json.dumps(result["claims"], indent=2), "",
             "FREE is a t-model label. Per-load seed failures and CP upper bounds accompany each FREE.",
             "H1d is descriptive; H1i identifies a read-time history effect only.", ""]
    lines += [f"{r['hypothesis']} {r['load']}: {r['label']} (n={r['n']})" for r in rows]
    (out / "lod_scores.txt").write_text("\n".join(lines) + "\n")
    for name in ("geometry", "phase", "within_load_associations", "error_share", "error_rank_share"):
        (out / ("lod_" + name + ".json")).write_text(json.dumps(result["H5"].get(name, []), indent=2) + "\n")
