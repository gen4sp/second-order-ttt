"""Pre-registered R4-2 seed-paired analysis.

Usage: python scripts/analyze_r42.py RUN.log OUTPUT_DIR [--s5-fp32 S5.json]

``--s5-fp32`` applies the pre-registered fallback (an R4-2 pre-launch
amendment): an +emb cell whose in-run S5 failed is judged by an fp32 CPU
recomputation on its final checkpoint (a list of {cell_id, passed} records).
Binary paired-difference intervals are conservative exact >=95% intervals:
Bonferroni projection of two exact 97.5% Clopper–Pearson intervals for the
positive and negative discordance probabilities (no asymptotic approximation).
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

ARMS = ("e2e1emb", "e2e1emb_kvm", "e2e1emb_rnd", "e2e1_kvm", "e2e1embP")
ANCHOR, KVM, RND, PLAIN, PERM = ARMS


def records(path, s5_override=None):
    cells, probes, refs = {}, {}, {}
    for line in Path(path).open():
        if line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            cells[(c["seed"], c["arm"])] = c
        elif line.startswith("PROBE "):
            p = json.loads(line[len("PROBE "):])
            probes.setdefault(p["cell_id"], {})[p["step"]] = p
        elif line.startswith("REF echo "):
            refs = json.loads(line[len("REF echo "):])
    for c in cells.values():
        probes.setdefault(c["cell_id"], {}).update({p["step"]: p for p in c.get("probes", [])})
        if s5_override and c["cell_id"] in s5_override and "s5" in c:
            c["s5_in_run"] = dict(c["s5"])
            c["s5"] = dict(c["s5"], passed=bool(s5_override[c["cell_id"]]), source="fp32-cpu")
    issues = []
    for s in range(10):
        group = [cells[s, a] for a in ARMS if (s, a) in cells and not cells[s, a].get("error")]
        hashes = {c.get("first_batch_sha1") for c in group}
        if len(hashes) > 1:
            issues.append(f"seed {s}: first_batch_sha1 differs across arms {sorted(map(str, hashes))}")
            for c in group:
                c["unpaired"] = True
        for c in group:
            want = 16 if c["arm"] in (KVM, RND, PLAIN) else 0
            if c.get("masked_per_seq") is not None and c["masked_per_seq"] != want:
                issues.append(f"{c['cell_id']}: masked_per_seq {c['masked_per_seq']} != {want}")
                c["unpaired"] = True
    return cells, probes, refs, issues


def valid(c):
    return bool(c and not c.get("error") and not c.get("skipped") and not c.get("unpaired") and
                "eval" in c and (c["arm"] not in (ANCHOR, KVM, RND, PERM) or
                                 c.get("s5", {}).get("passed") is True))


def paired(cells, a, b):
    return [(s, cells[s, a], cells[s, b]) for s in range(10)
            if valid(cells.get((s, a))) and valid(cells.get((s, b)))]


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))


def cp(k, n, alpha=.025):
    if not n:
        return (0., 1.)
    lo, hi = 0., 1.
    if k:
        a, b = 0., k/n
        for _ in range(60):
            mid = (a+b)/2
            if 1-binom_cdf(k-1, n, mid) > alpha/2:
                b = mid
            else:
                a = mid
        lo = (a+b)/2
    if k < n:
        a, b = k/n, 1.
        for _ in range(60):
            mid = (a+b)/2
            if binom_cdf(k, n, mid) > alpha/2:
                a = mid
            else:
                b = mid
        hi = (a+b)/2
    return lo, hi


def contrast(rows, field):
    positive = sum(bool(a[field]) and not bool(b[field]) for _, a, b in rows)
    negative = sum(bool(b[field]) and not bool(a[field]) for _, a, b in rows)
    n = len(rows)
    pl, ph = cp(positive, n)
    nl, nh = cp(negative, n)
    return dict(n=n, positive=positive, negative=negative,
                difference=(positive-negative)/n if n else None,
                ci95=[pl-nh, ph-nl] if n else None,
                mcnemar_one_sided=binom_cdf(negative, positive+negative, .5)
                if positive+negative else 1.)


def probe(probes, cell, step=None):
    series = probes.get(cell["cell_id"], {})
    return series.get(step) if step is not None else series[max(series)] if series else None


def ratio(a, b):
    if a is None or b is None:
        return None
    return a/b if b > 0 else (math.inf if a > 0 else None)


def summarize(cells, probes, refs, issues=()):
    contrasts = {name: contrast(paired(cells, *arms), field)
                 for name, arms, field in (("ret_kvm_anchor", (KVM, ANCHOR), "bound_final"),
                                           ("ret_kvm_rnd", (KVM, RND), "bound_final"),
                                           ("acq_anchor_perm", (ANCHOR, PERM), "bound_ever"),
                                           ("acq_kvm_anchor", (KVM, ANCHOR), "bound_ever"))}
    all_pairs = {a: paired(cells, a, ANCHOR) for a in ARMS[1:]}
    complete = [s for s in range(10) if all(valid(cells.get((s, a)))
                for a in (ANCHOR, KVM, RND))]
    anchor = [cells[s, ANCHOR] for s in range(10) if valid(cells.get((s, ANCHOR)))]
    v2 = "pending" if len(anchor) < 10 else (
        "anchor not reproduced" if sum(c["bound_ever"] for c in anchor) < 7 else
        "no retention deficit on this device" if sum(c["bound_final"] for c in anchor) >= 4 else "holds")
    windows = []
    acquired = [last_bound(c) for c in anchor if c["bound_ever"] and last_bound(c) is not None]
    # the fallback t_s is snapped to the 25-step probe grid, so every window has 7 probes
    fallback = 25 * round(median(acquired) / 25) if acquired else None
    per_seed = []
    ce_rows = []
    for s in range(10):
        row = {"seed": s}
        for arm in ARMS:
            c = cells.get((s, arm))
            row.update({f"{arm}_{key}": (c.get(key) if valid(c) else None)
                        for key in ("bound_ever", "bound_final", "collapse", "degraded",
                                    "mask", "mask_seed", "masked_per_seq", "first_batch_sha1",
                                    "peak_mps_mb")})
            for condition in ("all", "all_minus_kv", "kv_only"):
                row[f"{arm}_ablation_{condition}"] = (c.get("ablation", {}).get(condition)
                                                        if valid(c) else None)
            row[f"{arm}_last_bound_step"] = last_bound(c) if valid(c) else None
            if valid(c):
                p = probe(probes, c)
                ce_rows.append(dict(seed=s, arm=arm, **{k: p.get(k) if p else None
                              for k in ("ce_kv", "ce_vk", "ce_vW")}))
        if s in complete:
            a, k, r = (cells[s, arm] for arm in (ANCHOR, KVM, RND))
            ap, kp, rp = (probe(probes, c) for c in (a, k, r))
            if ap and kp and rp:
                row["p_kv_gap"] = ap["p_kv"]-kp["p_kv"]
                row["anchor_p_kv"] = ap["p_kv"]
                row["kvm_p_kv"] = kp["p_kv"]
                row["anchor_e_kv_abs"] = ap["e_kv_abs"]
                row["kvm_e_kv_abs"] = kp["e_kv_abs"]
                row["e_kv_ratio"] = ratio(kp["e_kv_abs"], ap["e_kv_abs"])
                row["kv_eff_ratio"] = ratio(kp["kv_write_eff"], ap["kv_write_eff"])
                row["rnd_eff_ratio"] = ratio(rp["kv_write_eff"], ap["kv_write_eff"])
                t = last_bound(a) if last_bound(a) is not None else fallback
                if t is not None:
                    series_a, series_k = (probes.get(c["cell_id"], {}) for c in (a, k))
                    steps = sorted(set(series_a) & set(series_k))
                    values = [ratio(series_k[step]["kv_write_eff"], series_a[step]["kv_write_eff"])
                              for step in steps if t <= step <= t+150]
                    values = [v for v in values if v is not None]
                    row["t_s"] = t
                    row["window_ratio"] = median(values) if values else None
                    row["first_ratio_gt2"] = next((step for step in steps
                        if ratio(series_k[step]["kv_write_eff"], series_a[step]["kv_write_eff"])
                        is not None and ratio(series_k[step]["kv_write_eff"],
                                              series_a[step]["kv_write_eff"]) > 2), None)
                    windows.append(row["window_ratio"])
        per_seed.append(row)
    full_rows = [r for r in per_seed if r["seed"] in complete]
    m1 = (sum(r.get("p_kv_gap", -math.inf) >= .3 for r in full_rows) >= 8 and
          sum(r.get("e_kv_ratio", -math.inf) >= 10 for r in full_rows) >= 8
          ) if len(full_rows) == 10 else None
    rnd_ratios = [r["rnd_eff_ratio"] for r in full_rows if r.get("rnd_eff_ratio") is not None]
    m2 = (sum(r.get("kv_eff_ratio", -math.inf) >= 10 for r in full_rows) >= 8 and
          bool(rnd_ratios) and median(rnd_ratios) < 3 and
          sum(v is not None and v >= 2 for v in windows) >= 7) if len(full_rows) == 10 else None
    retained = [cells[s, KVM] for s in complete if cells[s, KVM]["bound_final"]]
    floor = refs.get("occ_minus", {}).get("acc")
    # joint per-seed reading (pre-launch amendment): both ablation conditions in the same
    # retained seed; a missing or mismatched ablation fails that seed; no retained seed means
    # M3 is not evaluable and counts as not established
    def m3_seed(c):
        ab = c.get("ablation") or {}
        return bool(c.get("ablation_ref_match") is True and floor is not None and
                    ab.get("all_minus_kv", math.inf) <= floor + .10 and
                    ab.get("kv_only", -math.inf) >= .80)
    m3_pass = sum(m3_seed(c) for c in retained)
    m3 = (len(retained) > 0 and m3_pass * 3 >= 2 * len(retained)) if len(complete) == 10 else None
    kvm_acq = sum(cells[s, KVM]["bound_ever"] for s in complete)
    kvm_ret = sum(cells[s, KVM]["bound_final"] for s in complete)
    cka, ckr = contrasts["ret_kvm_anchor"], contrasts["ret_kvm_rnd"]
    common = [s for s in complete if cells[s, ANCHOR]["bound_ever"] and cells[s, KVM]["bound_ever"]]
    partial = sum(cells[s, KVM]["eval"]["query_acc"] >= .50 and
                  cells[s, ANCHOR]["eval"]["query_acc"] < .50 for s in common) - sum(
                  cells[s, ANCHOR]["eval"]["query_acc"] >= .50 and
                  cells[s, KVM]["eval"]["query_acc"] < .50 for s in common)
    if v2 != "holds" or len(complete) < 10:
        h = "pending" if len(complete) < 10 else "descriptive"
    elif kvm_acq <= 4:
        h = "H-blocked"
    elif cka["mcnemar_one_sided"] <= .05 and ckr["mcnemar_one_sided"] <= .05 and kvm_ret >= 5:
        h = "H-rescue"
    elif cka["mcnemar_one_sided"] <= .05 and ckr["mcnemar_one_sided"] > .05:
        h = "Specificity not established"
    elif partial >= 4:
        h = "H-partial"
    elif kvm_acq >= 7 and kvm_ret <= 2:
        h = "No rescue shown"
    else:
        h = "H-inconclusive"
    if h in ("No rescue shown", "H-inconclusive") and m1 is False:
        h = "untested (manipulation failed)"
    mediation = ("pending" if len(complete) < 10 else
                 "kv-write mechanism supported" if h in ("H-rescue", "H-partial") and
                 m1 and m2 and m3 else "kv-mask effect; mechanism not established" if
                 h in ("H-rescue", "H-partial") else "not applicable")
    plain = [cells[s, PLAIN] for s in range(10) if valid(cells.get((s, PLAIN)))]
    plain_acq = sum(c["bound_ever"] for c in plain)
    tb = "pending" if len(plain) < 10 else "TB-holds" if plain_acq <= 1 else (
        "TB-fails" if plain_acq >= 5 else "TB-mixed")
    p_rows = paired(cells, ANCHOR, PERM)
    geom = [c["probe_at_first_bound"] for _, c, _ in p_rows if c["bound_ever"]
            and c.get("probe_at_first_bound")]
    geometry = bool(geom) and sum(p["addr_a"] >= .80 and p["addr_margin"] > 0
                                      for p in geom) * 3 >= 2 * len(geom)
    validity = []
    for _, a, p in p_rows:
        a25, p25 = probe(probes, a, 25), probe(probes, p, 25)
        pf = probe(probes, p)
        validity.append(bool(a25 and p25 and pf and pf["emb_addr"] <= .10 and
            all(.8 <= ratio(p25[key], a25[key]) <= 1.2 for key in ("fast_input_norm", "code_norm"))))
    pvalid = len(p_rows) == 10 and all(validity) and geometry
    plabel = "pending" if len(p_rows) < 10 else "descriptive (anchor not reproduced)" if (
        v2 == "anchor not reproduced") else (
        "descriptive (validity failed)" if not pvalid else "P-supports" if
        contrasts["acq_anchor_perm"]["mcnemar_one_sided"] <= .05 else "P-fails")
    return dict(labels=dict(V2=v2, H=h, M1="pending" if m1 is None else m1,
                            M2="pending" if m2 is None else m2,
                            M3="pending" if m3 is None else m3, mediation=mediation,
                            TB=tb, P=plabel), n_complete=len(complete), contrasts=contrasts,
                counts={arm: dict(n=sum(valid(cells.get((s, arm))) for s in range(10)),
                                  ever=sum(bool(cells[s, arm]["bound_ever"]) for s in range(10)
                                           if valid(cells.get((s, arm)))),
                                  final=sum(bool(cells[s, arm]["bound_final"]) for s in range(10)
                                            if valid(cells.get((s, arm))))) for arm in ARMS},
                details=dict(m1_pass_count=sum(r.get("p_kv_gap", -math.inf) >= .3 and
                                                r.get("e_kv_ratio", -math.inf) >= 10 for r in full_rows),
                             m2_final_pass_count=sum(r.get("kv_eff_ratio", -math.inf) >= 10
                                                     for r in full_rows),
                             m2_window_pass_count=sum(v is not None and v >= 2 for v in windows),
                             m3_retained=len(retained), m3_pass=m3_pass, p_geometry=geometry,
                             p_validity_count=sum(validity), plain_t_addr_lost_median=median(
                                 [c["t_addr_lost"] for c in plain if c.get("t_addr_lost") is not None])
                                 if any(c.get("t_addr_lost") is not None for c in plain) else None,
                             plain_p_kv_median=median([probe(probes, c)["p_kv"] for c in plain
                                 if probe(probes, c)]) if plain else None),
                issues=list(issues), per_seed=per_seed, per_class_ce=ce_rows)


def last_bound(c):
    if not c:
        return None
    bound = [r[0] for r in c.get("quick", []) if r[3] is not None and r[3] >= .90]
    return max(bound) if bound else None


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--s5-fp32", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    override = ({r["cell_id"]: r["passed"] for r in json.loads(args.s5_fp32.read_text())}
                if args.s5_fp32 else None)
    result = summarize(*records(args.log, override))
    (args.out / "summary.json").write_text(json.dumps({k: v for k, v in result.items()
         if k not in ("per_seed", "per_class_ce")}, indent=2, sort_keys=True) + "\n")
    write_csv(args.out / "paired_seeds.csv", result["per_seed"])
    write_csv(args.out / "per_class_ce.csv", result["per_class_ce"])
    print(json.dumps({"labels": result["labels"], "n_complete": result["n_complete"],
                      "issues": result["issues"],
                      "counts": result["counts"], "contrasts": result["contrasts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
