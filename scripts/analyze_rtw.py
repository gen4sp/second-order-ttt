"""Mechanical RTW screening tables and pre-flight qualifiers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ARMS = ("A0","SO","G-hi","G-lo","G-scalar","G-sg","G-hi-c",
        "M-p125","M-p25","M-p50","M-late","M-early","RP-all","NC-all")
GATE = ("G-hi","G-lo","G-scalar","G-sg","G-hi-c")
SOMIX = ("M-p125","M-p25","M-p50","M-late","M-early")
PF = ("RP-all","NC-all")


def load(root):
    root = Path(root); summary = root / "summary.json"
    if summary.exists(): return json.loads(summary.read_text())
    cells = {}
    for f in root.glob("*/result.json"):
        x = json.loads(f.read_text()); cells[f.parent.name] = x
    return {"cells": cells}


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _stats(row, key=None):
    stats = row.get("stats", {})
    return stats if key is None else stats.get(key, {})


def _localisation(row):
    logs = [x for x in row.get("logs", []) if "quick" in x]
    acq = row.get("t_acq")
    post = [x for x in logs if acq is not None and x["step"] > acq]
    fallback = not post
    if fallback: post = logs[-20:]
    b = [_stats(x).get("beta_write", {}).get("median") for x in post]
    o = [_stats(x).get("beta_other", {}).get("median") for x in post]
    return (_median(b) / max(_median(o) or 1e-30, 1e-30) if _median(b) is not None and _median(o) is not None else None), fallback


def analyze(root):
    data = load(root); cells = list(data.get("cells", {}).values())
    out = {"retention": {}, "ever_acquired": {}, "gate": {}, "somix": {}, "pf": {}, "decisions": {}}
    for arm in ARMS:
        rs = [x for x in cells if x.get("arm") == arm]
        out["retention"][arm] = {
            "n": len(rs), "retained_own": sum(bool(x.get("retained_own", x.get("retained"))) for x in rs),
            "retained_ce": sum(bool(x.get("retained_ce")) for x in rs),
            "rate_own": (sum(bool(x.get("retained_own", x.get("retained"))) for x in rs) / len(rs) if rs else None),
            "rate_ce": (sum(bool(x.get("retained_ce")) for x in rs) / len(rs) if rs else None),
        }
        out["ever_acquired"][arm] = {"n": len(rs), "count": sum(bool(x.get("ever_acquired")) for x in rs)}
    for arm in GATE:
        rs = [x for x in cells if x.get("arm") == arm]
        loc = [_localisation(x)[0] for x in rs]
        out["gate"][arm] = {"beta_localisation_after_t_acq_median": _median(loc),
                             "fallback_count": sum(_localisation(x)[1] for x in rs),
                             "clip_binding_mean": _median([_stats(x, "clip").get("mean") for x in rs]),
                             "clip_without_gain_mean": _median([_stats(x, "clip_base").get("mean") for x in rs])}
    a0 = _median([x.get("ms_step") for x in cells if x.get("arm") == "A0"])
    so = _median([x.get("ms_step") for x in cells if x.get("arm") == "SO"])
    for arm in SOMIX:
        vals = [x.get("ms_step") for x in cells if x.get("arm") == arm]
        out["somix"][arm] = {"ms_step_median": _median(vals),
                              "ratio_to_A0": (_median(vals) / a0 if vals and a0 else None),
                              "ratio_to_SO": (_median(vals) / so if vals and so else None),
                              "so_norm_median": _median([_stats(x, "so_norm").get("median") for x in cells if x.get("arm") == arm]),
                              "fo_norm_median": _median([_stats(x, "fo_norm").get("median") for x in cells if x.get("arm") == arm])}
    for arm in PF:
        rs = [x for x in cells if x.get("arm") == arm]
        active_w = _median([_stats(x, "active_write").get("mean") for x in rs])
        active_o = _median([_stats(x, "active_other").get("mean") for x in rs])
        ratio = _median([_stats(x, "ratio").get("median") for x in rs])
        out["pf"][arm] = {"active_write_share": active_w, "active_other_share": active_o,
                          "active_share": _median([q for q in (active_w, active_o) if q is not None]),
                          "median_gex_over_gce": ratio,
                          "induced_clip_binding": _median([_stats(x, "induced_clip").get("mean") for x in rs]),
                          "matched_fraction": _median([_stats(x, "match").get("mean") for x in rs]) if arm == "NC-all" else None,
                          "nc_fail_count": sum(int(_stats(x, "nc_fail").get("count", 0)) for x in rs) if arm == "NC-all" else 0,
                          "manipulation_pass": bool(active_w is not None and active_o is not None and
                                                     max(active_w, active_o) >= .02 and ratio is not None and ratio >= .005)}
    # Mechanical GATE/SOMIX screen rows (registered); own-rule retention is primary.
    r = out["retention"]
    gate_hi, gate_scalar = r["G-hi"]["retained_own"], r["G-scalar"]["retained_own"]
    loc = out["gate"]["G-hi"]["beta_localisation_after_t_acq_median"]
    out["decisions"]["GATE"] = {"main": "localised" if gate_hi >= 10 and gate_scalar <= 3 and loc is not None and loc >= 3
                                  else "amplitude" if gate_hi >= 10 and gate_scalar >= 10
                                  else "cap_irrelevant" if r["G-lo"]["retained_own"] >= 10
                                  else "closed" if gate_hi <= 3 else "inconclusive"}
    small = [a for a in ("M-p125","M-p25") if r[a]["retained_own"] >= 10]
    out["decisions"]["SOMIX"] = {"main": "partial_so" if small else
                                   "only_half" if r["M-p50"]["retained_own"] >= 10 else
                                   "late" if r["M-late"]["retained_own"] >= 10 else
                                   "early" if r["M-early"]["retained_own"] >= 10 and r["M-late"]["retained_own"] <= 3 else "full_so_needed"}
    rp, nc = r["RP-all"]["retained_own"], r["NC-all"]["retained_own"]
    out["decisions"]["PF-T_R2"] = {"main": "keep_both" if rp >= 10 and nc <= 3 else "apply_R2_after_M0_and_mini",
                                     "rp_retained_own": rp, "nc_retained_own": nc,
                                     "rp_manipulation_pass": out["pf"]["RP-all"]["manipulation_pass"]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--json", dest="json_out"); ap.add_argument("--md", dest="md_out")
    a = ap.parse_args(); out = analyze(a.root)
    j = Path(a.json_out or Path(a.root) / "analysis.json"); m = Path(a.md_out or Path(a.root) / "analysis.md")
    j.write_text(json.dumps(out, indent=2, sort_keys=True))
    lines = ["# RTW analysis", "", "## Retention and ever-acquired", "", "|arm|n|own retained|CE retained|ever-acquired|", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        x, e = out["retention"][arm], out["ever_acquired"][arm]; lines.append(f"|{arm}|{x['n']}|{x['retained_own']}|{x['retained_ce']}|{e['count']}|")
    lines += ["", "## GATE", "", json.dumps(out["gate"], sort_keys=True), "", "## SOMIX", "", json.dumps(out["somix"], sort_keys=True), "", "## PF-T", "", json.dumps(out["pf"], sort_keys=True), "", "## Decisions", "", json.dumps(out["decisions"], sort_keys=True)]
    m.write_text("\n".join(lines) + "\n"); print("RTW_ANALYSIS " + json.dumps(out, sort_keys=True))

if __name__ == "__main__": main()
