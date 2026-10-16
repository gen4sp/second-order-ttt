#!/usr/bin/env python
"""Per-checkpoint probe summary: apply the registered rules to the per-checkpoint probe outputs.

Pure numpy. For each probed toy checkpoint: point values with 95% percentile bootstrap intervals
(2000 draws, numpy seed 0) of every per-sequence quantity (accuracies, margins and CE of the
write-scaling variants a/b/c, exact and linearised kv/rest attributions Mkv/Mrest, matched-norm
CE/lin/alpha1 writes, write norms, residuals, the virtual-gain derivatives D_FO, D_SO, D_P and
R = -D_P/D_FO) and the identity checks. Registered counts Q1-Q11 per variant (i = later updates
frozen at recorded values, ii = recomputed), among them Q8: |D_SO - D_FO| / |D_FO| <= 1e-9 on every
checkpoint of the linear-kv set L; identity maxima; set-level medians with bootstrap intervals.
Sets (first letter of the label): E = FO early after loss, P = FO before loss, S = SO final,
F = FO final, L = FO linear-kv final.

Input (per-checkpoint probe outputs of the toy checkpoints, not included in the repository):
PROBE_DIR/*.json with ``label``, ``arm``, ``n_eval``, ``etas``, ``checks`` and ``per_seq``.
Outputs in RESULTS_DIR: probe_summary.json (verdicts, identities, set_medians, checkpoints; the input
of reduced_model_attribution.py and performative_replay.py) and probe_summary_checkpoints.csv;
verdicts, identities and medians on stdout.
Usage: python probe_summary.py PROBE_DIR [RESULTS_DIR]
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

B = 2000
RNG = np.random.default_rng(0)


def boot_mean(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_R(dfo, dso):
    dfo, dso = np.asarray(dfo, float), np.asarray(dso, float)
    dp = dso - dfo
    idx = RNG.integers(0, len(dfo), size=(B, len(dfo)))
    r = -dp[idx].mean(1) / dfo[idx].mean(1)
    return float(-dp.mean() / dfo.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def count_rule(flags):
    """flags: list of bool for eligible units."""
    n = len(flags)
    k = int(sum(flags))
    if n < 6:
        return dict(n=n, count=k, verdict="not evaluable")
    if k >= math.ceil(0.75 * n):
        v = "supported"
    elif k <= math.floor(0.5 * n):
        v = "falsified"
    else:
        v = "inconclusive"
    return dict(n=n, count=k, verdict=v)


def stats(d):
    ps = d["per_seq"]
    s = dict(label=d["label"], set=d["label"].split("-")[0], arm=d["arm"], n_eval=d["n_eval"],
             eta_W2=d["etas"][2], checks=d["checks"])
    has = lambda k: k in ps
    for v in ("i", "ii"):
        for name in ("a_0", "a_0.5", "a_1", "a_2", "a_4", "a_8", "a_16", "b_0", "b_0.5", "b_2", "b_4", "b_8",
                     "b_16", "c"):
            for met in ("acc", "margin", "ce"):
                k = f"{v}_{name}_{met}"
                if has(k):
                    s[k] = boot_mean(ps[k])
        for k in ("Mkv_exact", "Mrest_exact", "Minter_exact", "Mbase_nowrite", "Mkv_lin", "Mrest_lin"):
            if has(f"{v}_{k}"):
                s[f"{v}_{k}"] = boot_mean(ps[f"{v}_{k}"])
        for tgt in ("lin", "ce"):
            for rule in ("ce", "lin", "alpha1"):
                for met in ("acc", "margin", "proj"):
                    k = f"{v}_mn{tgt}_{rule}_{met}"
                    if has(k):
                        s[k] = boot_mean(ps[k])
    for k in ("kv_resid", "kv_resid_cv", "clip_c0", "clip_later", "dkv_norm", "drest_norm", "dkv_W2_norm",
              "drest_W2_norm", "cos_kv_rest", "comp_norm_kv", "comp_target_kv", "wnorm_ce", "wnorm_lin",
              "wnorm_alpha1", "cos_ce_lin", "cos_alpha1_lin", "clip_any_p3"):
        if has(k):
            s[k] = boot_mean(ps[k])
    if has("kv_resid_cv"):
        cvp = np.asarray(ps["kv_resid_cv"]) * math.sqrt(15 / 16)
        s["cos_alloc_pred"] = boot_mean(1 / np.sqrt(1 + cvp ** 2))
    if has("D_FO"):
        s["D_FO"] = boot_mean(ps["D_FO"])
        s["D_SO"] = boot_mean(ps["D_SO"])
        s["D_P"] = boot_mean(np.asarray(ps["D_SO"]) - np.asarray(ps["D_FO"]))
        s["R"] = boot_R(ps["D_FO"], ps["D_SO"])
        s["R_clipgrad"] = boot_R(ps["D_FO"], ps["D_SO_clipgrad"])
        dfo, dso = np.asarray(ps["D_FO"]), np.asarray(ps["D_SO"])
        s["DP_rel_max"] = float(np.max(np.abs(dso - dfo) / np.maximum(np.abs(dfo), 1e-300)))
        s["R_eligible"] = not (s["D_FO"][1] <= 0 <= s["D_FO"][2])
    return s


def g(s, k):
    return s[k][0]


def verdicts(S, v):
    out = {}
    E = [s for s in S if s["set"] == "E"]
    F = [s for s in S if s["set"] == "F"]
    SO = [s for s in S if s["set"] == "S"]
    P = [s for s in S if s["set"] == "P"]
    L = [s for s in S if s["set"] == "L"]
    E_lost = [s for s in E if g(s, f"{v}_a_1_acc") < .9]
    F_lost = [s for s in F if g(s, f"{v}_a_1_acc") < .9]
    S_bound = [s for s in SO if g(s, f"{v}_a_1_acc") >= .9]
    out["Q1"] = count_rule([g(s, f"{v}_a_16_acc") < .9 and g(s, f"{v}_a_16_acc") - g(s, f"{v}_a_1_acc") < .10
                            for s in E_lost])
    out["Q2"] = count_rule([g(s, f"{v}_b_16_acc") >= .9 for s in E_lost])
    out["Q3"] = count_rule([g(s, f"{v}_c_acc") - g(s, f"{v}_a_1_acc") >= .05 for s in E_lost])
    out["Q3m"] = count_rule([g(s, f"{v}_Mrest_exact") < 0 for s in E_lost])
    out["Q4a"] = count_rule([g(s, f"{v}_mnlin_ce_margin") < g(s, f"{v}_mnlin_lin_margin") for s in E])
    out["Q4b"] = count_rule([(g(s, f"{v}_mnlin_ce_acc") - g(s, f"{v}_a_1_acc")) >=
                             2 * (g(s, f"{v}_mnlin_lin_acc") - g(s, f"{v}_mnlin_ce_acc")) for s in E_lost])
    out["Q10"] = count_rule([g(s, f"{v}_b_16_acc") < .9 and g(s, f"{v}_mnlin_lin_acc") < .9 for s in F_lost])
    out["Q11"] = count_rule([g(s, f"{v}_c_acc") >= g(s, f"{v}_a_1_acc") - .02 for s in S_bound])
    if v == "i":
        out["Q4c"] = count_rule([abs(g(s, "cos_ce_lin") - g(s, "cos_alloc_pred")) <= .15 for s in E])
        out["Q5"] = count_rule([g(s, "R") > 1 for s in E if s.get("R_eligible")])
        out["Q6"] = count_rule([1 / 3 <= g(s, "R") <= 3 for s in SO if s.get("R_eligible")])
        out["Q7"] = count_rule([g(s, "R") < 1 for s in P if s.get("R_eligible")])
        RE = {s["label"].split("-")[2]: g(s, "R") for s in E if s["label"].startswith("E-anchor")}
        RP = {s["label"].split("-")[2]: g(s, "R") for s in P}
        RF = {s["label"].split("-")[2]: g(s, "R") for s in F}
        pairs = [(sd, RP.get(sd), RE.get(sd)) for sd in ("s0", "s1", "s3", "s4")]
        k = sum(1 for _, a, b in pairs if a is not None and b is not None and a < b)
        n = sum(1 for _, a, b in pairs if a is not None and b is not None)
        out["Q7b"] = dict(n=n, count=k, pairs=pairs, verdict=("not evaluable" if n < 4 else "supported" if k >= 3
                                                               else "falsified" if k <= 1 else "inconclusive"))
        pairs = [(sd, RF.get(sd), RE.get(sd)) for sd in ("s0", "s1", "s3", "s4")]
        k = sum(1 for _, a, b in pairs if a is not None and b is not None and a < b)
        n = sum(1 for _, a, b in pairs if a is not None and b is not None)
        out["Q10b"] = dict(n=n, count=k, pairs=pairs, verdict=("not evaluable" if n < 4 else "supported" if k >= 3
                                                                else "falsified" if k <= 1 else "inconclusive"))
        q8 = [s["DP_rel_max"] <= 1e-9 for s in L]
        out["Q8"] = dict(n=len(q8), count=int(sum(q8)), verdict="supported" if q8 and all(q8) else "falsified",
                         max_rel=max((s["DP_rel_max"] for s in L), default=None))
        q9 = [s["checks"].get("dfo_analytic_rel") for s in S if s["checks"].get("dfo_analytic_rel") is not None]
        out["Q9"] = dict(n=len(q9), max_rel=max(q9) if q9 else None,
                         verdict="supported" if q9 and max(q9) <= 1e-8 else "falsified")
    out["_eligible"] = dict(E_lost=[s["label"] for s in E_lost], F_lost=[s["label"] for s in F_lost],
                            S_bound=[s["label"] for s in S_bound],
                            E_not_lost=[s["label"] for s in E if s not in E_lost])
    return out


def unit_median_boot(vals):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    idx = RNG.integers(0, len(vals), size=(B, len(vals)))
    med = np.median(vals[idx], axis=1)
    return [float(np.median(vals)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="per-checkpoint probe outputs (*.json)")
    ap.add_argument("res_dir", nargs="?", default="outputs/analysis/theory-checks/probe-summary")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "probe-summary"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    out_dir, res_dir = args.out_dir, args.res_dir
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    S = [stats(json.load(open(f))) for f in files]
    order = {"E": 0, "P": 1, "S": 2, "F": 3, "L": 4}
    S.sort(key=lambda s: (order.get(s["set"], 9), s["label"]))
    V = {"i": verdicts(S, "i"), "ii": verdicts(S, "ii")}
    # identity checks across all checkpoints
    ident = {}
    for key in ("split_rel", "aice_lin_rel", "kvwrite_vs_split_rel", "gamma1_fo", "gamma1_so", "logits_vs_eval",
                "dfo_analytic_rel"):
        vals = [s["checks"].get(key) for s in S if s["checks"].get(key) is not None]
        ident[key] = dict(n=len(vals), max=max(vals) if vals else None)
    # set-level medians
    med = {}
    for st in ("E", "P", "S", "F", "L"):
        sub = [s for s in S if s["set"] == st]
        med[st] = {k: unit_median_boot([(s[k][0] if isinstance(s[k], (list, tuple)) else s[k]) for s in sub if k in s])
                   for k in ("i_a_1_acc", "i_a_16_acc", "i_b_16_acc", "i_c_acc", "ii_a_1_acc", "ii_a_16_acc",
                             "ii_b_16_acc", "ii_c_acc", "i_Mkv_exact", "i_Mrest_exact", "i_mnlin_ce_acc",
                             "i_mnlin_lin_acc", "i_mnlin_alpha1_acc", "R", "kv_resid", "dkv_norm", "drest_norm",
                             "cos_ce_lin", "cos_alloc_pred", "eta_W2")
                   if any(k in s for s in sub)}
    os.makedirs(res_dir, exist_ok=True)
    json.dump(dict(verdicts=V, identities=ident, set_medians=med, checkpoints=S),
              open(os.path.join(res_dir, "probe_summary.json"), "w"), indent=1)
    cols = ["label", "set", "arm", "n_eval", "eta_W2"]
    keys = ["i_a_0_acc", "i_a_1_acc", "i_a_2_acc", "i_a_4_acc", "i_a_8_acc", "i_a_16_acc", "i_b_0_acc", "i_b_2_acc",
            "i_b_4_acc", "i_b_8_acc", "i_b_16_acc", "i_c_acc", "ii_a_1_acc", "ii_a_16_acc", "ii_b_16_acc", "ii_c_acc",
            "i_a_1_margin", "i_Mkv_exact", "i_Mrest_exact", "i_Minter_exact", "i_Mbase_nowrite", "i_Mkv_lin",
            "i_Mrest_lin", "ii_Mkv_exact", "ii_Mrest_exact", "i_mnlin_ce_acc", "i_mnlin_lin_acc",
            "i_mnlin_alpha1_acc", "ii_mnlin_ce_acc", "ii_mnlin_lin_acc", "ii_mnlin_alpha1_acc", "i_mnlin_ce_margin",
            "i_mnlin_lin_margin", "i_mnce_lin_acc", "i_mnce_alpha1_acc", "kv_resid", "kv_resid_cv", "dkv_norm",
            "drest_norm", "cos_kv_rest", "wnorm_ce", "wnorm_lin", "cos_ce_lin", "cos_alloc_pred", "comp_target_kv",
            "clip_c0", "clip_later", "D_FO", "D_SO", "D_P", "R", "R_clipgrad", "clip_any_p3"]
    with open(os.path.join(res_dir, "probe_summary_checkpoints.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + [x for k in keys for x in (k, k + "_lo", k + "_hi")])
        for s in S:
            row = [s.get(c) for c in cols]
            for k in keys:
                row += list(s[k]) if k in s else [None, None, None]
            w.writerow(row)
    print(json.dumps(dict(verdicts=V, identities=ident), indent=1, default=str))
    print(json.dumps(med, indent=1))


if __name__ == "__main__":
    main()
