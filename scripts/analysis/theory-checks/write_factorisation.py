#!/usr/bin/env python
"""Write factorisation: apply the registered rules to the per-unit probe outputs.

Pure numpy. Point verdicts + cluster-by-seed-family / shared-sequence bootstraps
(write_factorisation_bootstrap). A unit that fails any identity check (ID_RULES, registered
thresholds) is invalid. --scope unit (default) drops an invalid unit from every test; --scope b
drops it from the (b) tests only (B_TESTS) and --scope none drops nothing; b and none are
descriptive sensitivity runs. The H_later / H_starve decision maps are applied to the point
verdicts. The released results/theory-checks/write_factorisation_{results.json,units.csv} are the
--scope b outputs.

Inputs (per-unit probe outputs of the toy checkpoints, not included in the repository): one JSON per
unit in PROBE_DIR, labelled <set>-eta<eta>-<seed> (set FO, SO or L; labels starting with "dev-" are
skipped), with ``per_seq`` (per-sequence arrays: accuracies ``i_*_acc`` / ``ii_*_acc``, chunk
derivatives ``dLq_e0..6``, ``dLo_e*``, ``dMq_e*``, write norms ``norm_*``, residuals, ...),
``n_eval``, ``checks`` (identity-check values), ``rho``, ``etas``, ``wall_s``, ``peak_rss_gb``,
``ckpt_sha256``.
Outputs in RESULTS_DIR: write_factorisation_results[-<scope>].json and
write_factorisation_units[-<scope>].csv (no suffix for --scope unit); a summary on stdout.
Usage: python write_factorisation.py PROBE_DIR [RESULTS_DIR] [--scope unit|b|none]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import write_factorisation_bootstrap as U  # noqa: E402

LATER = (1, 2, 3, 4, 5)        # chunks whose writes precede a query (chunk 6 is never read by a query)
WIN_F, WIN_QR = (1, 2, 3, 4), (5,)
# identity checks with the registered thresholds; a unit failing any of them is invalid
ID_RULES = {
    "logits_vs_eval<=1e-10": lambda c: c["logits_vs_eval"] <= 1e-10,
    "split_rel<=1e-12": lambda c: c["split_rel"] <= 1e-12,
    "base_ii_vs_i<=1e-10": lambda c: c["base_ii_vs_i"] <= 1e-10,
    "dLq_dead_chunks==0": lambda c: c["dLq_dead_chunks"] == 0.0,
    "fo_eps_zero==0": lambda c: c["fo_eps_zero"] == 0.0,
    "eps0_loss_vs_toy<=1e-12": lambda c: c["eps0_loss_vs_toy"] <= 1e-12,
    "fd_rest0(abs<=1e-6|rel<=1e-4)": lambda c: c["fd_rest0_abs"] <= 1e-6 or c["fd_rest0_rel"] <= 1e-4,
    "fd_later(abs<=1e-6|rel<=1e-4)": lambda c: c["fd_later_abs"] <= 1e-6 or c["fd_later_rel"] <= 1e-4,
}
B_TESTS = ("bFO_later", "bFO_rest0", "bL_later", "bSO_later", "bFO_later_neg")


def failed_identities(checks):
    return [k for k, f in ID_RULES.items() if not f(checks)]


def load(out_dir):
    units = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        d = json.load(open(f))
        if d["label"].startswith("dev-"):
            continue
        ps = {k: np.asarray(v, float) for k, v in d["per_seq"].items()}
        ps["Dq_later"] = sum(ps[f"dLq_e{c}"] for c in LATER)
        ps["Dq_rest0"] = ps["dLq_e0"]
        ps["Dq_F"] = sum(ps[f"dLq_e{c}"] for c in WIN_F)
        ps["Dq_Qr"] = sum(ps[f"dLq_e{c}"] for c in WIN_QR)
        ps["Do_later"] = sum(ps[f"dLo_e{c}"] for c in LATER)
        ps["Do_rest0"] = ps["dLo_e0"]
        ps["Dm_later"] = sum(ps[f"dMq_e{c}"] for c in LATER)
        ps["Dm_rest0"] = ps["dMq_e0"]
        st, eta, seed = d["label"].split("-")
        units[d["label"]] = dict(label=d["label"], set=st, eta=int(eta[3:]), seed=seed, ps=ps, n=d["n_eval"],
                                 checks=d["checks"], rho=d["rho"], etas=d["etas"], wall=d["wall_s"],
                                 rss=d["peak_rss_gb"], ckpt_sha=d["ckpt_sha256"])
    return units


def sign_class(x, idx, rng):
    """'pos' if the 95% percentile CI of the mean excludes 0 above, 'neg' below, else 'null'.
    Point: 2000 draws, numpy seed 0 (registered). Inside a bootstrap draw: nested 500 draws."""
    x = np.asarray(x, float)
    if idx is None:
        lo, hi = U.pct_ci(x)
        return np.array([lo > 0]), np.array([hi < 0])
    _, lo, hi = U.nested_pct_excl0(x[idx], rng)
    return lo > 0, hi < 0


def flags(units, q):
    def fn(l, idx, rng):
        ps = units[l]["ps"]
        g = lambda k: U.mean_of(ps, k, idx)
        base = g("i_base_acc")
        lost, bound = base < .9, base >= .9
        dL = g("i_noL_acc") - base
        if q == "A1":
            return lost, dL >= .03
        if q == "A2":
            return lost, dL - (g("i_noR_acc") - base) >= .02
        if q == "A2ii":
            return lost, dL - (g("ii_noR_acc") - g("ii_base_acc")) >= .02
        if q == "A3":
            return lost & (dL >= .03), g("i_rand_acc") - base >= .5 * dL
        if q == "A4":
            return lost & (dL >= .03), g("i_swap_acc") - base >= .5 * dL
        if q == "B1":
            return lost, g("i_kvonly_acc") < .9
        if q == "B2":
            return lost, g("i_k16_acc") - base >= .10
        if q == "B2ii":
            return lost, g("ii_k16_acc") - g("ii_base_acc") >= .10
        if q == "B3":
            return lost, np.abs(g("i_k16noL_acc") - g("i_k16_acc")) < .03
        if q == "B3ii":
            return lost, np.abs(g("i_k16noL_acc") - g("ii_k16_acc")) < .03
        if q == "S1":
            return bound, g("i_noL_acc") >= base - .02
        if q == "S3":
            return bound, g("norm_S6") / g("norm_kv") < 1
        if q == "L1":
            return bound, g("i_noL_acc") >= base - .02
        if q in ("bFO_later", "bFO_rest0", "bL_later", "bSO_later", "bFO_later_neg"):
            key = "Dq_rest0" if q.endswith("rest0") else "Dq_later"
            pos, neg = sign_class(ps[key], idx, rng)
            el = lost if q.startswith("bFO") else bound
            if q == "bSO_later":
                return el, ~pos
            if q == "bFO_later_neg":
                return el, neg
            return el, pos
        raise ValueError(q)
    return fn


TESTS = {  # id: (set, description)
    "A1": ("FO", "remove later writes helps: acc(noL) - acc(base) >= .03"),
    "A2": ("FO", "locus: [acc(noL)-acc(base)] - [acc(noR)-acc(base)] >= .02 (variant i)"),
    "A2ii": ("FO", "A2 with the chunk-0 rest removal recomputed (variant ii)"),
    "A3": ("FO", "among A1-positive lost units: random-direction later write (per-tensor norm-matched) removes >= half the harm"),
    "A4": ("FO", "among A1-positive lost units: other-sequence later write (per-tensor norm-matched) removes >= half the harm"),
    "B1": ("FO", "natural kv write alone not read: acc(kvonly) < .9"),
    "B2": ("FO", "read capacity: acc(k16) - acc(base) >= .10 (variant i)"),
    "B2ii": ("FO", "B2 with later writes recomputed (variant ii)"),
    "B3": ("FO", "later writes do not block an amplified kv write: |acc(k16noL) - acc(k16)| < .03 (variant i)"),
    "B3ii": ("FO", "B3 against the recomputed k16 (variant ii)"),
    "S1": ("SO", "removing later writes harmless: acc(noL) >= acc(base) - .02"),
    "S3": ("SO", "later write smaller than kv write: mean||S6|| / mean||d_kv|| < 1"),
    "L1": ("L", "removing later writes harmless: acc(noL) >= acc(base) - .02"),
    "bFO_later": ("FO", "(b) dLq/d eps_later > 0 (CI excludes 0 above)"),
    "bFO_rest0": ("FO", "(b) dLq/d eps_rest0 > 0 (CI excludes 0 above)"),
    "bL_later": ("L", "(b) dLq/d eps_later > 0 (CI excludes 0 above)"),
    "bSO_later": ("SO", "(b) dLq/d eps_later NOT > 0 (CI not entirely above 0)"),
    "bFO_later_neg": ("FO", "(b) [useful-writes account] dLq/d eps_later < 0 (CI excludes 0 below)"),
}


def families_of(units, st):
    fam = {}
    for l, u in units.items():
        if u["set"] == st:
            fam.setdefault(u["seed"], []).append(l)
    return [sorted(v) for k, v in sorted(fam.items())]


def point_stats(u):
    ps = u["ps"]
    s = {k: float(np.mean(v)) for k, v in ps.items()}
    for key in ("Dq_later", "Dq_rest0", "Dq_F", "Dq_Qr", "Do_later", "Do_rest0", "Dm_later", "Dm_rest0"):
        lo, hi = U.pct_ci(ps[key])
        s[key + "_ci"] = [lo, hi]
        s[key + "_class"] = "pos" if lo > 0 else "neg" if hi < 0 else "null"
    s["dL"] = s["i_noL_acc"] - s["i_base_acc"]
    s["dR"] = s["i_noR_acc"] - s["i_base_acc"]
    s["dR_ii"] = s["ii_noR_acc"] - s["ii_base_acc"]
    s["d_rand"] = s["i_rand_acc"] - s["i_base_acc"]
    s["d_swap"] = s["i_swap_acc"] - s["i_base_acc"]
    s["d_neg"] = s["i_neg_acc"] - s["i_base_acc"]
    s["dL_c6"] = s["i_noL_acc_c6"] - s["i_base_acc_c6"]
    s["d_noQr"] = s["i_noQr_acc"] - s["i_base_acc"]
    s["d_noF"] = s["i_noF_acc"] - s["i_base_acc"]
    s["d_k16"] = s["i_k16_acc"] - s["i_base_acc"]
    s["d_k16_noL"] = s["i_k16noL_acc"] - s["i_k16_acc"]
    s["ratio_S6_kv"] = s["norm_S6"] / s["norm_kv"]
    return s


def unit_median_ci(vals, rng, D=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    med = np.median(vals[rng.integers(0, len(vals), size=(D, len(vals)))], 1)
    return [float(np.median(vals)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def decision_maps(v):
    """Registered decision maps, applied to the point verdicts."""
    a1, a2 = v["A1"]["point"]["verdict"], v["A2"]["point"]["verdict"]
    if a1 == "supported" and a2 == "supported":
        h_later = "supported"
    elif a1 == "falsified":
        h_later = "falsified"
    elif a1 == "supported" and a2 == "falsified":
        h_later = "locus not later"
    else:
        h_later = "inconclusive"
    h_starve = (a1 == "falsified" and all(v[q]["point"]["verdict"] == "supported" for q in ("B1", "B2", "B3")))
    return dict(H_later=h_later, H_starve="supported" if h_starve else "not supported",
                H_starve_parts={q: v[q]["point"]["verdict"] for q in ("A1", "B1", "B2", "B3")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="per-unit probe outputs (*.json)")
    ap.add_argument("res_dir", nargs="?", default="outputs/analysis/theory-checks/write-factorisation")
    ap.add_argument("--scope", default="unit", choices=("unit", "b", "none"))
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "write-factorisation"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    out_dir, res_dir, scope = args.out_dir, args.res_dir, args.scope
    assert scope in ("unit", "b", "none"), scope
    tag = "" if scope == "unit" else f"-{scope}"
    all_units = load(out_dir)
    invalid = {l: failed_identities(u["checks"]) for l, u in all_units.items() if failed_identities(u["checks"])}
    units = all_units if scope in ("b", "none") else {l: u for l, u in all_units.items() if l not in invalid}
    units_b = {l: u for l, u in all_units.items() if l not in invalid} if scope == "b" else units
    rng = np.random.default_rng(U.SEED_OUT)
    P = {l: point_stats(u) for l, u in units.items()}
    res = dict(n_units={st: sum(u["set"] == st for u in units.values()) for st in ("FO", "SO", "L")},
               scope=scope, invalidated=invalid,
               verdicts={}, identities={}, eligibility={}, per_eta={}, set_medians={})
    for q, (st, desc) in TESTS.items():
        uq = units_b if q in B_TESTS else units
        fams = families_of(uq, st)
        if not fams:
            continue
        r = U.run_bootstraps(uq, fams, flags(uq, q), U.count_verdict, rng)
        r["set"], r["desc"] = st, desc
        res["verdicts"][q] = r
    res["decision_maps"] = decision_maps(res["verdicts"])
    # eligibility lists
    for st, rule in (("FO", "lost"), ("SO", "bound"), ("L", "bound")):
        ls = sorted(l for l in units if units[l]["set"] == st)
        res["eligibility"][st] = dict(
            eligible=[l for l in ls if (P[l]["i_base_acc"] < .9) == (rule == "lost")],
            other=[l for l in ls if (P[l]["i_base_acc"] < .9) != (rule == "lost")])
    # identity checks
    for key in ("logits_vs_eval", "split_rel", "base_ii_vs_i", "dLq_dead_chunks", "fd_rest0_rel", "fd_later_rel",
                "fd_rest0_abs", "fd_later_abs", "fo_eps_zero", "eps0_loss_vs_toy"):
        vals = [u["checks"][key] for u in units.values() if u["checks"].get(key) is not None]
        res["identities"][key] = dict(n=len(vals), max=max(vals) if vals else None)
    res["identities"]["rho_absmax"] = max(abs(r) for u in units.values() for r in u["rho"])
    # descriptive: per eta (FO, SO) and set medians
    keys = ("i_base_acc", "dL", "dR", "dR_ii", "d_rand", "d_swap", "d_neg", "dL_c6", "d_noQr", "d_noF",
            "i_kvonly_acc", "d_k16", "d_k16_noL", "i_linkvw_acc", "Dq_later", "Dq_rest0", "Dq_F", "Dq_Qr",
            "Do_later", "Do_rest0", "Dm_later", "Dm_rest0", "norm_kv", "norm_rest", "norm_S5", "norm_S6",
            "ratio_S6_kv", "kv_resid", "rest_resid", "clip_later", "b_clip", "b_Lq")
    for st in ("FO", "SO", "L"):
        for eta in (10, 40, 160):
            ls = [l for l in units if units[l]["set"] == st and units[l]["eta"] == eta]
            if not ls:
                continue
            res["per_eta"][f"{st}-eta{eta}"] = dict(
                units=ls, n_lost=sum(P[l]["i_base_acc"] < .9 for l in ls),
                medians={k: float(np.median([P[l][k] for l in ls])) for k in keys},
                Dq_later_class=[P[l]["Dq_later_class"] for l in ls],
                Dq_rest0_class=[P[l]["Dq_rest0_class"] for l in ls])
        el = res["eligibility"][st]["eligible"]
        res["set_medians"][st] = {k: unit_median_ci([P[l][k] for l in el], rng) for k in keys}
    res["units"] = {l: {k: v for k, v in P[l].items() if not k.startswith(("dLq_e", "dLo_e", "dMq_e"))}
                    for l in sorted(P)}
    res["chunk_derivs"] = {l: {f"dLq_e{c}": P[l][f"dLq_e{c}"] for c in range(7)} for l in sorted(P)}
    os.makedirs(res_dir, exist_ok=True)
    json.dump(res, open(os.path.join(res_dir, f"write_factorisation_results{tag}.json"), "w"), indent=1)
    cols = ["i_base_acc"] + list(keys[1:])
    with open(os.path.join(res_dir, f"write_factorisation_units{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "set", "eta", "seed"] + cols + ["Dq_later_class", "Dq_rest0_class"])
        for l in sorted(P):
            u = units[l]
            w.writerow([l, u["set"], u["eta"], u["seed"]] + [P[l][k] for k in cols]
                       + [P[l]["Dq_later_class"], P[l]["Dq_rest0_class"]])
    print("SCOPE", scope, "INVALIDATED (identity checks)", json.dumps(invalid))
    print("UNITS", res["n_units"])
    print("DECISION_MAPS", json.dumps(res["decision_maps"]))
    print("ELIGIBILITY", json.dumps(res["eligibility"]))
    print("IDENTITIES", json.dumps(res["identities"]))
    print(f"{'test':14s} {'set':3s} {'point':>18s} | P(same): {'seq':>5s} {'clust':>5s} {'hier':>5s} | hier share 95%")
    for q, r in res["verdicts"].items():
        p = r["point"]
        print(f"{q:14s} {r['set']:3s} {p['count']:>3d}/{p['n']:<3d} {p['verdict']:12s} | {r['seq']['p_same']:14.3f} "
              f"{r['cluster']['p_same']:5.3f} {r['hier']['p_same']:5.3f} | [{r['hier']['share_95'][0]:.2f}, "
              f"{r['hier']['share_95'][1]:.2f}]")
    for k, v in res["per_eta"].items():
        m = v["medians"]
        print(f"PER_ETA {k:11s} lost {v['n_lost']}/{len(v['units'])} base {m['i_base_acc']:.3f} dL {m['dL']:+.3f} "
              f"dR {m['dR']:+.3f} rand {m['d_rand']:+.3f} swap {m['d_swap']:+.3f} kvonly {m['i_kvonly_acc']:.3f} "
              f"k16 {m['d_k16']:+.3f} k16noL {m['d_k16_noL']:+.3f} Dq_later {m['Dq_later']:+.4g} "
              f"Dq_rest0 {m['Dq_rest0']:+.4g} |S6|/|kv| {m['ratio_S6_kv']:.3g} classes {v['Dq_later_class']}")


if __name__ == "__main__":
    main()
