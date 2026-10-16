"""MINI-D: where the FO-CE memory deficit sits at equal steps (mini-m and mini-s, seeds 0-2).

For each preset and seed, the per-probe memory m = fP_focal - fP_nonfocal of SO (e2e2) and
FO-CE (e2e1-ce) at step 1200 is split by probe level:
- B1 exposure stratum (E1, E2s, E2c, E2b, E3), B2 probe distance, B3 8-gram match,
  B4 chunk, B5 domain (domains with >= 500 probes);
- K1 copyable (E2s + E2c + E3) versus non-copyable (E1 + E2b).
Per level: M_SO, M_FO, dM = M_SO - M_FO, rho = dM / M_SO and the level's share of the total dM;
seed means with one-sided 90% t intervals (3 seeds). Contrasts:
- K1 = rho(copyable) - rho(non-copyable);
- K2 = rho(distance >= 4096) - rho(distance < 1024).
Label: STARVATION-LIKE if K1's lower bound > 0 and K1 >= .10; DISTANCE-LIKE if K2's lower bound
> 0 and K2 >= .05; BOTH, NEITHER; NE if the validity gate fails (same probes in both arms,
per-document M recomputed from the per-probe values within 1e-4 of the logged value).

Inputs (per-document reads of the MINI matrix, not included in the repository):
``<root>/<group>/runs/<preset>/<arm>/s<seed>/eval_step1200.npz`` with ``tok_fP_focal``,
``tok_fP_nonfocal``, ``doc_fP_focal``, ``doc_fP_nonfocal``, ``eval_sha1``, ``eval_domain``, and
``eval_meta.npz`` in the same folder with ``probe_offset``, ``probe_stratum``, ``probe_dist``,
``probe_pos``, ``probe_match8``. Groups: mini-s seeds 0-2 = g40-g42, mini-m seeds 0-2 = g43-g45.
Outputs: the JSON and a Markdown table; one summary line per preset on stdout.
Numpy only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

T_CRIT = 1.886
GROUPS = {"mini-s": {0: "g40", 1: "g41", 2: "g42"}, "mini-m": {0: "g43", 1: "g44", 2: "g45"}}
STRATA = ("E1", "E2s", "E2c", "E2b", "E3")
DIST_BINS = [(0, 1024, "<1024"), (1024, 2048, "1024-2047"), (2048, 4096, "2048-4095"),
             (4096, 10 ** 9, ">=4096")]
C = 1024


def interval(vals):
    v = np.asarray(vals, float)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    h = T_CRIT * sd / np.sqrt(len(v))
    return {"mean": m, "sd": sd, "lb": m - h, "ub": m + h, "values": v.tolist()}


def load_cell(root, preset, arm, k):
    d = Path(root) / GROUPS[preset][k] / "runs" / preset / arm / f"s{k}"
    z = np.load(d / "eval_step1200.npz", allow_pickle=False)
    meta = np.load(d / "eval_meta.npz", allow_pickle=False)
    m = z["tok_fP_focal"].astype(np.float64) - z["tok_fP_nonfocal"].astype(np.float64)
    off = meta["probe_offset"]
    doc_m = np.array([m[off[i]:off[i + 1]].mean() if off[i + 1] > off[i] else np.nan
                      for i in range(len(off) - 1)])
    doc_logged = z["doc_fP_focal"] - z["doc_fP_nonfocal"]
    v_doc = float(np.nanmax(np.abs(doc_m - doc_logged)))
    return {"m": m, "meta": meta, "sha1": z["eval_sha1"], "v_doc": v_doc,
            "M_doc_weighted": float(np.nanmean(doc_logged)), "M_recomputed": float(np.nanmean(doc_m)),
            "domain_doc": z["eval_domain"], "off": off}


def levels(meta, off, domains):
    st = meta["probe_stratum"]
    dist = meta["probe_dist"]
    chunk = meta["probe_pos"] // C
    m8 = meta["probe_match8"]
    doc_idx = np.repeat(np.arange(len(off) - 1), np.diff(off))
    dom = np.asarray(domains)[doc_idx]
    out = {}
    for i, s in enumerate(STRATA):
        out[f"B1:{s}"] = st == i
    for lo, hi, name in DIST_BINS:
        out[f"B2:{name}"] = (dist >= lo) & (dist < hi)
    out["B3:match8"] = m8.astype(bool)
    out["B3:no-match8"] = ~m8.astype(bool)
    for c in (4, 5, 6, 7):
        out[f"B4:chunk{c}"] = chunk == c
    u, cnt = np.unique(dom, return_counts=True)
    for d, n in zip(u, cnt):
        if n >= 500:
            out[f"B5:{d}"] = dom == d
    out["K1:copy(E2s+E2c+E3)"] = np.isin(st, [1, 2, 4])
    out["K1:nocopy(E1+E2b)"] = np.isin(st, [0, 3])
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default="outputs/mini/matrix",
                    help="MINI matrix root holding g40-g45")
    ap.add_argument("--out-json", default="outputs/analysis/text-copyability-distance/breakdown.json")
    ap.add_argument("--out-md", default="outputs/analysis/text-copyability-distance/breakdown.md")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "copyability-distance"))
    args = ap.parse_args(argv)
    root, out_json, out_md = args.root, args.out_json, args.out_md
    pname = args.process_name
    os.environ["PROCESS_NAME"] = pname
    try:
        import setproctitle
        setproctitle.setproctitle(pname)
    except ImportError:
        pass
    result = {"presets": {}}
    for preset in ("mini-m", "mini-s"):
        per_seed = []
        gate = {"v_doc_max": 0.0, "M_check": [], "same_probes": True}
        for k in (0, 1, 2):
            so = load_cell(root, preset, "e2e2", k)
            fo = load_cell(root, preset, "e2e1-ce", k)
            same = (np.array_equal(so["sha1"], fo["sha1"]) and
                    np.array_equal(so["meta"]["probe_pos"], fo["meta"]["probe_pos"]) and
                    np.array_equal(so["off"], fo["off"]))
            gate["same_probes"] &= bool(same)
            gate["v_doc_max"] = max(gate["v_doc_max"], so["v_doc"], fo["v_doc"])
            gate["M_check"].append({"seed": k, "SO": [so["M_doc_weighted"], so["M_recomputed"]],
                                    "FO": [fo["M_doc_weighted"], fo["M_recomputed"]]})
            lv = levels(so["meta"], so["off"], so["domain_doc"])
            n_all = len(so["m"])
            dm_all = float(so["m"].sum() - fo["m"].sum())
            row = {}
            for name, sel in lv.items():
                n = int(sel.sum())
                if n == 0:
                    continue
                Ms, Mf = float(so["m"][sel].mean()), float(fo["m"][sel].mean())
                row[name] = {"n": n, "probe_share": n / n_all, "M_SO": Ms, "M_FO": Mf,
                             "dM": Ms - Mf, "rho": (Ms - Mf) / Ms if Ms != 0 else np.nan,
                             "dM_share": float(so["m"][sel].sum() - fo["m"][sel].sum()) / dm_all}
            row["ALL(probe-weighted)"] = {"n": n_all, "M_SO": float(so["m"].mean()),
                                          "M_FO": float(fo["m"].mean()),
                                          "dM": float(so["m"].mean() - fo["m"].mean()),
                                          "rho": float(1 - fo["m"].mean() / so["m"].mean())}
            per_seed.append(row)
        gate["pass"] = bool(gate["same_probes"] and gate["v_doc_max"] <= 1e-4 and all(
            abs(c["SO"][0] - c["SO"][1]) <= 1e-4 and abs(c["FO"][0] - c["FO"][1]) <= 1e-4
            for c in gate["M_check"]))
        names = [n for n in per_seed[0] if all(n in r for r in per_seed)]
        pooled = {}
        for n in names:
            pooled[n] = {q: interval([r[n][q] for r in per_seed])
                         for q in ("M_SO", "M_FO", "dM", "rho", "dM_share") if q in per_seed[0][n]}
            pooled[n]["n_mean"] = float(np.mean([r[n]["n"] for r in per_seed]))
            if "probe_share" in per_seed[0][n]:
                pooled[n]["probe_share"] = float(np.mean([r[n]["probe_share"] for r in per_seed]))
        k1 = interval([r["K1:copy(E2s+E2c+E3)"]["rho"] - r["K1:nocopy(E1+E2b)"]["rho"] for r in per_seed])
        k2 = interval([r["B2:>=4096"]["rho"] - r["B2:<1024"]["rho"] for r in per_seed])
        starv = k1["lb"] > 0 and k1["mean"] >= .10
        dist = k2["lb"] > 0 and k2["mean"] >= .05
        label = ("NE" if not gate["pass"] else "BOTH" if starv and dist else
                 "STARVATION-LIKE" if starv else "DISTANCE-LIKE" if dist else "NEITHER")
        result["presets"][preset] = {"gate": gate, "K1": k1, "K2": k2, "label": label,
                                     "levels": pooled}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# MINI-D breakdown", ""]
    for preset, r in result["presets"].items():
        lines += [f"## {preset}: label **{r['label']}** (V-gate pass={r['gate']['pass']}, "
                  f"max doc diff {r['gate']['v_doc_max']:.2e})", "",
                  f"K1 = {r['K1']['mean']:+.3f} [{r['K1']['lb']:+.3f}, {r['K1']['ub']:+.3f}]; "
                  f"K2 = {r['K2']['mean']:+.3f} [{r['K2']['lb']:+.3f}, {r['K2']['ub']:+.3f}]", "",
                  "| level | probes/seed | probe share | M_SO | M_FO | ΔM | ρ [90% t] | ΔM share |",
                  "|---|---|---|---|---|---|---|---|"]
        for n, q in r["levels"].items():
            lines.append(f"| {n} | {q['n_mean']:.0f} | {q.get('probe_share', 1):.3f} | "
                         f"{q['M_SO']['mean']:.3f} | {q['M_FO']['mean']:.3f} | {q['dM']['mean']:.3f} | "
                         f"{q['rho']['mean']:.2f} [{q['rho']['lb']:.2f}, {q['rho']['ub']:.2f}] | "
                         + (f"{q['dM_share']['mean']:.3f}" if 'dM_share' in q else "—") + " |")
        lines.append("")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n")
    for preset, r in result["presets"].items():
        print("MINID " + json.dumps({"preset": preset, "label": r["label"], "gate": r["gate"]["pass"],
                                     "K1": r["K1"]["mean"], "K2": r["K2"]["mean"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
