"""Registered MINI pre-flight analysis (pre-flight rules R1-R3)."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

ARMS = ("e2e2", "e2e1-ce", "e2e12-p125", "e2e1-rp", "e2e1-nc", "e2e2-mix", "e2e1-ce-mix")
PRESET_DIM = {"mini-s": 256, "mini-m": 512}
T_CRIT = 1.886


def _mean(z, key, default=np.nan):
    if key not in z:
        return float(default)
    x = np.asarray(z[key], dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float(default)


def _mean_difference(z, a, b):
    if a not in z or b not in z:
        return np.nan
    d = np.asarray(z[a], dtype=float) - np.asarray(z[b], dtype=float)
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan


def _finite_mean(values):
    x = np.asarray(values, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _finite_std(values):
    x = np.asarray(values, dtype=float)
    return float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan


def _strata(z):
    akey = "tok_doc_adapt" if "tok_doc_adapt" in z else "tok_adapt"
    bkey = "tok_doc_noadapt" if "tok_doc_noadapt" in z else "tok_noadapt"
    if "probe_stratum" not in z or akey not in z or bkey not in z:
        return {}
    s, a, b = np.asarray(z["probe_stratum"]), np.asarray(z[akey]), np.asarray(z[bkey])
    out = {}
    for k, name in enumerate(("E1", "E2s", "E2c", "E2b", "E3")):
        sel = s == k
        out[name] = float(np.nanmean((b - a)[sel])) if sel.any() else np.nan
    return out


def _path_info(root: Path, p: Path):
    rel = p.relative_to(root).parts
    preset = next((x for x in rel if x in PRESET_DIM), "mini-s")
    arm = next((x for x in rel if x in ARMS), p.parent.name)
    seed = next((re.fullmatch(r"(?:seed|s)([0-9]+)", x).group(1) for x in rel
                 if re.fullmatch(r"(?:seed|s)([0-9]+)", x)), "0")
    return preset, arm, str(seed)


def _step_telemetry(path: Path):
    """Mean realised mix share and step time recorded by the trainer."""
    step_log = path.parent / "steps.jsonl"
    shares, times = [], []
    if step_log.exists():
        for line in step_log.read_text().splitlines():
            try:
                rec = json.loads(line.removeprefix("P2_STEP "))
            except (ValueError, json.JSONDecodeError):
                continue
            if "so_rows" in rec and rec.get("global_batch"):
                shares.append(float(rec["so_rows"]) / float(rec["global_batch"]))
            if "ms_per_step" in rec:
                times.append(float(rec["ms_per_step"]))
    marker = path.parent / "MINI_CELL_DONE.json"
    if not times and marker.exists():
        try:
            value = json.loads(marker.read_text()).get("ms_per_step")
            if value is not None:
                times.append(float(value))
        except (OSError, ValueError, TypeError):
            pass
    return (_finite_mean(shares), _finite_mean(times))

def _one(path: Path, arm: str, seed: str, step: int, corpus="dclm", preset="mini-s"):
    so_share, ms_step = _step_telemetry(path)
    with np.load(path, allow_pickle=False) as z:
        focal, nonfocal = _mean(z, "doc_fP_focal"), _mean(z, "doc_fP_nonfocal")
        row = {"arm": arm, "seed": str(seed), "step": int(step), "corpus": corpus,
               "preset": preset, "L": _mean(z, "doc_adapt"),
               "G": _mean(z, "doc_noadapt") - _mean(z, "doc_adapt"),
               "M": _mean_difference(z, "doc_fP_focal", "doc_fP_nonfocal"), "H_LM": _mean(z, "doc_heldout_noadapt"),
               "H_ad": _mean(z, "doc_heldout"), "strata": _strata(z),
               "rp_floor_active": _mean(z, "rp_floor_active"),
               "nc_f_match": _mean(z, "nc_f_match"), "path": str(path),
               "so_row_share": so_share, "ms_per_step": ms_step}
        if "x_ce_M" in z:
            row["M_ce"] = _mean(z, "x_ce_M")
        return row


def _abandoned(root: Path, p: Path) -> bool:
    """Stale-renamed or resume folders hold abandoned partial cells."""
    return any(".stale-" in x or x.endswith(".resume") or x == "resume"
               for x in p.relative_to(root).parts[:-1])


def collect(root: str | Path) -> list[dict]:
    root = Path(root)
    rows = []
    for p in sorted(root.glob("**/eval_step*.npz")):
        if _abandoned(root, p):
            continue
        preset, arm, seed = _path_info(root, p)
        if arm not in ARMS:
            continue
        try:
            step = int(p.stem.split("step", 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append(_one(p, arm, seed, step, "dclm", preset))
    for p in sorted(root.glob("**/code_eval_step*.npz")):
        if _abandoned(root, p):
            continue
        preset, arm, seed = _path_info(root, p)
        if arm not in ARMS:
            continue
        try:
            step = int(p.stem.split("step", 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append(_one(p, arm, seed, step, "code", preset))
    if not rows:
        raise FileNotFoundError(f"no MINI evaluation files below {root}")
    return rows


def _pairs(rows, preset, step, corpus, so_arm="e2e2", fo_arm="e2e1-ce"):
    so = {r["seed"]: r for r in rows if r["preset"] == preset and r["step"] == step
          and r["corpus"] == corpus and r["arm"] == so_arm}
    fo = {r["seed"]: r for r in rows if r["preset"] == preset and r["step"] == step
          and r["corpus"] == corpus and r["arm"] == fo_arm}
    keys = sorted(set(so) & set(fo), key=lambda x: int(x) if str(x).isdigit() else str(x))
    return [{"seed": k, "delta_M": so[k]["M"] - fo[k]["M"],
             "M_SO": so[k]["M"], "M_FO": fo[k]["M"]} for k in keys]


def _paired_stats(pairs, threshold=.145):
    d = np.asarray([p["delta_M"] for p in pairs], dtype=float)
    ms = np.asarray([p["M_SO"] for p in pairs], dtype=float)
    if len(d) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan, "rho": np.nan,
                "ub_rho": np.nan, "D": [], "pairs": []}
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    mean = float(np.mean(d))
    lb = mean - T_CRIT * sd / np.sqrt(len(d)) if np.isfinite(sd) else mean
    D = d - threshold * ms
    ds = float(np.std(D, ddof=1)) if len(D) > 1 else np.nan
    ub = float(np.mean(D) + T_CRIT * ds / np.sqrt(len(D))) if np.isfinite(ds) else float(np.mean(D))
    return {"n": len(d), "mean": mean, "sd": sd, "lb": lb,
            "rho": float(mean / np.mean(ms)) if np.mean(ms) else np.nan,
            "ub_rho": ub, "D": D.tolist(), "pairs": pairs}


def _partial_so_read(rows, preset, step=1200, corpus="dclm"):
    """Registered contrast read for the partial-SO arm."""
    so = {r["seed"]: r for r in rows if r.get("preset") == preset and r.get("step") == step
          and r.get("corpus") == corpus and r.get("arm") == "e2e2"}
    ce = {r["seed"]: r for r in rows if r.get("preset") == preset and r.get("step") == step
          and r.get("corpus") == corpus and r.get("arm") == "e2e1-ce"}
    part = {r["seed"]: r for r in rows if r.get("preset") == preset and r.get("step") == step
            and r.get("corpus") == corpus and r.get("arm") == "e2e12-p125"}
    keys = sorted(set(so) & set(ce) & set(part), key=lambda x: int(x) if str(x).isdigit() else str(x))
    pairs = []
    for k in keys:
        e = part[k]["M"] - ce[k]["M"]
        d = so[k]["M"] - ce[k]["M"]
        pairs.append({"seed": k, "E": e, "delta_M": d, "M_p": part[k]["M"],
                      "M_SO": so[k]["M"], "M_CE": ce[k]["M"]})
    e = np.asarray([x["E"] for x in pairs], dtype=float)
    d = np.asarray([x["delta_M"] for x in pairs], dtype=float)
    def contrast(a):
        if not len(a):
            return np.nan
        m = float(np.mean(a)); sd = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
        return m - T_CRIT * sd / np.sqrt(len(a)) if np.isfinite(sd) else m
    def contrast_ub(a):
        if not len(a):
            return np.nan
        m = float(np.mean(a)); sd = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
        return m + T_CRIT * sd / np.sqrt(len(a)) if np.isfinite(sd) else m
    # R1 row 2 is the SO-vs-CE deficit check, using the same paired contrast.
    r1 = _paired_stats([{"delta_M": x["delta_M"], "M_SO": x["M_SO"]} for x in pairs])
    lb_quarter = contrast(e - .25 * d)
    ub_half = contrast_ub(e - .5 * d)
    phi = float(np.mean(e) / np.mean(d)) if len(d) and np.mean(d) else np.nan
    precondition = bool(np.isfinite(r1["rho"]) and r1["rho"] >= .145 and
                        np.isfinite(r1["lb"]) and r1["lb"] > 0)
    if not precondition:
        row = "descriptive"
    elif np.isfinite(phi) and phi >= .5 and lb_quarter > 0:
        row = "transfers"
    elif np.isfinite(ub_half) and ub_half < 0:
        row = "does_not_transfer"
    else:
        row = "ambiguous"
    shares = [part[k].get("so_row_share", np.nan) for k in keys]
    ms = [part[k].get("ms_per_step", np.nan) for k in keys]
    return {"n": len(pairs), "pairs": pairs, "phi_hat": phi,
            "lb_E_minus_0.25_delta": lb_quarter,
            "ub_E_minus_0.5_delta": ub_half,
            "lb_E_minus_quarter_delta": lb_quarter,
            "ub_E_minus_half_delta": ub_half,
            "realised_so_row_share": _finite_mean(shares),
            "ms_per_step": _finite_mean(ms),
            "precondition": {"rho_hat": r1["rho"], "lb_delta_M": r1["lb"]},
            "R1_row2": precondition, "mechanical_row": row, "row": row}

def _m0(rows, preset, step, corpus, arm="e2e2"):
    q = [r for r in rows if r["preset"] == preset and r["step"] == step
         and r["corpus"] == corpus and r["arm"] == arm]
    ms, gs = np.asarray([r["M"] for r in q]), np.asarray([r["G"] for r in q])
    means = {"M_SO": float(np.mean(ms)) if len(ms) else np.nan,
             "G_SO": float(np.mean(gs)) if len(gs) else np.nan}
    count_need = 2 if len(ms) >= 3 else len(ms)
    parts = {"mean_M": bool(len(ms) and means["M_SO"] >= .02),
             "mean_G": bool(len(gs) and means["G_SO"] >= .02),
             "seed_M": bool(len(ms) and int(np.sum(ms >= .02)) >= count_need),
             "seed_need": count_need}
    return {"pass": bool(all(parts.values())), **means, "parts": parts,
            "seeds": [r["seed"] for r in q]}


def _rp_manipulation(root: Path, preset: str, seed: str):
    candidates = list(root.glob(f"**/{preset}/e2e1-rp/**/steps.jsonl"))
    target = next((p for p in candidates if f"seed{seed}" in str(p) or f"s{seed}" in str(p)), None)
    ratios, active = [], []
    if target and target.exists():
        for line in target.read_text().splitlines():
            if not line.startswith("P2_STEP "):
                continue
            try:
                rec = json.loads(line.split(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            if not 600 <= int(rec.get("step", 0)) <= 1200:
                continue
            if "rp_extra_norm" in rec and "rp_ce_norm" in rec:
                ex = np.asarray(rec["rp_extra_norm"], dtype=float)
                ce = np.asarray(rec["rp_ce_norm"], dtype=float)
                ratios.extend((ex / np.maximum(ce, 1e-12)).reshape(-1).tolist())
            if "rp_floor_active_share" in rec:
                active.append(float(rec["rp_floor_active_share"]))
    return {"pass": bool(active and ratios and np.mean(active) >= .02 and np.median(ratios) >= .005),
            "floor_active_share": float(np.mean(active)) if active else np.nan,
            "median_ratio": float(np.median(ratios)) if ratios else np.nan}


def analyse(rows: list[dict], root: str | Path | None = None) -> dict:
    root = Path(root) if root is not None else None
    steps = sorted({int(r["step"]) for r in rows})
    final = 1200
    presets = sorted({r.get("preset", "mini-s") for r in rows}, key=lambda x: PRESET_DIM.get(x, 0))
    stats = {p: _paired_stats(_pairs(rows, p, final, "dclm")) for p in presets}
    partial = {p: _partial_so_read(rows, p, final, "dclm") for p in presets
               if any(r.get("arm") == "e2e12-p125" and r.get("preset") == p for r in rows)}
    passing_pairs = [p for p in presets if stats[p]["n"] == 3]
    pooled_pairs = [pair for p in passing_pairs for pair in stats[p]["pairs"]]
    pooled = _paired_stats(pooled_pairs)
    m0 = {p: _m0(rows, p, final, "dclm") for p in presets}
    complete = bool(presets and all(stats[p]["n"] == 3 and
                    {x["seed"] for x in stats[p]["pairs"]} == {"0", "1", "2"} for p in presets))
    passing = [p for p in presets if stats[p]["n"] == 3 and m0[p]["pass"]]
    for p in presets:
        init = _m0(rows, p, 0, "dclm")
        m0[p].update(M_init=init["M_SO"], G_init=init["G_SO"],
                     M_gain=m0[p]["M_SO"]-init["M_SO"],
                     G_gain=m0[p]["G_SO"]-init["G_SO"])
        m0[p]["init_level_only"] = bool(m0[p]["pass"] and m0[p]["M_SO"] <= init["M_SO"])
    primary = max(passing, key=lambda x: PRESET_DIM.get(x, 0)) if passing else None
    null_all = bool(passing and all(stats[p]["rho"] < .05 and stats[p]["ub_rho"] < 0 for p in passing))
    r3 = {"row": "pending (no mix pairs)", "ordered_tree": ["switch-to-mix", "mix-only-exposure", "stay-dclm"]}
    if primary:
        core = _pairs(rows, primary, final, "dclm")
        mix = _pairs(rows, primary, final, "dclm", "e2e2-mix", "e2e1-ce-mix")
        code_mix = _pairs(rows, primary, final, "code", "e2e2-mix", "e2e1-ce-mix")
        core_d = {p["seed"]: p["delta_M"] for p in core if p["seed"] in {"0", "1"}}
        mix_d = {p["seed"]: p["delta_M"] for p in mix if p["seed"] in {"0", "1"}}
        if len(mix) == 2 and len(code_mix) == 2:
            r3["row"] = "stay-dclm"
        sd_core = stats[primary]["sd"]
        delta = [{"seed": k, "delta": mix_d[k] - core_d[k]}
                 for k in sorted(set(core_d) & set(mix_d), key=str)]
        both_positive = len(delta) >= 2 and all(x["delta"] > 0 for x in delta)
        strong = bool(np.isfinite(sd_core) and len(delta) >= 2 and all(x["delta"] > 2 * sd_core for x in delta))
        mix_m0 = _m0(rows, primary, final, "dclm", "e2e2-mix")
        mix_code_m0 = _m0(rows, primary, final, "code", "e2e2-mix")
        if (len(core) >= 2 and all(x > 0 for x in core_d.values()) and len(mix) >= 2 and
                all(mix_d[k] >= 2 * core_d[k] for k in set(core_d) & set(mix_d)) and
                both_positive and strong and mix_m0["pass"] and mix_code_m0["pass"]):
            r3["row"] = "switch-to-mix"
        elif (null_all and len(mix) == 2 and len(code_mix) == 2 and
              np.mean([x["delta_M"] for x in code_mix]) >= .02 and
              all(x["delta_M"] > 0 for x in code_mix)):
            r3["row"] = "mix-only-exposure"
        r3.update({"core_pairs": core, "mix_pairs": mix, "code_pairs": code_mix,
                   "paired_delta": delta, "mix_M0_dclm": mix_m0,
                   "mix_M0_code": mix_code_m0})
    null_all = bool(passing and all(stats[p]["rho"] < .05 and stats[p]["ub_rho"] < 0 for p in passing))
    r1 = "incomplete" if not complete else "no-passing-presets" if not passing else (
        "go" if stats[primary]["rho"] >= .145 and stats[primary]["lb"] > 0 else
        "reconsider" if null_all and r3["row"] == "stay-dclm" else
        "pending R3" if null_all and r3["row"] == "pending (no mix pairs)" else "ambiguous")
    manip = _rp_manipulation(root, primary, "0") if root and primary else {"pass": False}
    recovery = {}
    for p in passing:
        rp = [r for r in rows if r["preset"] == p and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-rp"]
        fo = [r for r in rows if r["preset"] == p and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-ce" and r["seed"] == "0"]
        own = float(np.mean([r["M"] for r in rp])) if rp else np.nan
        base = fo[0]["M"] if fo else np.nan
        recovery[p] = {"M_RP_rp": own, "M_FO_CE_s0": base,
                       "recovered": own - base if np.isfinite(base) else np.nan,
                       "M_RP_ce": _finite_mean([r.get("M_ce", np.nan) for r in rp]),
                       "M_NC_ce": _finite_mean([r.get("M_ce", np.nan) for r in rows if r["preset"] == p and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-nc"])}
    drop_a = bool(complete and null_all)
    drop_b = bool(complete and passing and manip.get("pass") and all(np.isfinite(recovery[p].get("recovered", np.nan)) and
                 recovery[p]["recovered"] < .25 * stats[p]["mean"] and stats[p]["mean"] >= 2 * stats[p]["sd"]
                 for p in passing))
    toy_rp = _finite_mean([r.get("toy_rp_retains", np.nan) for r in rows])
    toy_nc = _finite_mean([r.get("toy_nc_retains", np.nan) for r in rows])
    toy_drop = bool(np.isfinite(toy_rp) and toy_rp <= 3)
    toy_keep = bool(np.isfinite(toy_rp) and np.isfinite(toy_nc) and toy_rp >= 10 and toy_nc <= 3)
    keep_reason = "m0-fails" if not passing else ("rp-recovers" if any(
        recovery[p].get("recovered", -np.inf) >= .5 * stats[p]["mean"] for p in passing)
        else "toy-rp-recovers" if toy_keep else "otherwise")
    r2 = ("incomplete" if not complete else "drop-both" if drop_a or (drop_b and toy_drop)
          else "pending toy" if drop_b else "keep-both" if toy_keep or keep_reason == "rp-recovers" else "keep-both")
    code_strata = {}
    for name in ("E1", "E2s", "E2c", "E2b", "E3"):
        so_s = [r["strata"].get(name, np.nan) for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "code" and r["arm"] == "e2e2-mix"]
        fo_s = [r["strata"].get(name, np.nan) for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "code" and r["arm"] == "e2e1-ce-mix"]
        so_f, fo_f = np.asarray(so_s, dtype=float), np.asarray(fo_s, dtype=float)
        code_strata[name] = (float(np.nanmean(so_f) - np.nanmean(fo_f))
                             if np.isfinite(so_f).any() and np.isfinite(fo_f).any() else np.nan)
    h3 = {"C_own": np.nan, "C_T": np.nan, "C_E": np.nan, "C_NC": np.nan, "S": np.nan}
    if primary:
        ms = [r["M"] for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e2"]
        mf = [r["M"] for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-ce"]
        mr = [r["M"] for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-rp"]
        mn = [r["M"] for r in rows if r["preset"] == primary and r["step"] == final and r["corpus"] == "dclm" and r["arm"] == "e2e1-nc"]
        if ms and mf:
            dm = np.mean(ms) - np.mean(mf)
            h3.update(C_own=float(np.mean(mr) - np.mean(mf) - .5 * dm) if mr else np.nan,
                      C_NC=float(np.mean(mn) - np.mean(mf) - .5 * dm) if mn else np.nan,
                      S=float(np.mean(mr) - np.mean(mn)) if mr and mn else np.nan)
    return {"steps": steps, "final_step": final, "preset_dimension": PRESET_DIM,
            "complete": complete, "preset_stats": stats, "pooled_stats": pooled,
            "toy": {"rp_retains": toy_rp, "nc_retains": toy_nc},
            "M0_gate": {p: m0[p] for p in presets}, "passing_presets": passing,
            "primary_preset": primary, "preflight_rules": {"R1": r1,
             "R2": {"row": r2, "drop_a": drop_a, "drop_b_mini": drop_b, "toy": "observed" if np.isfinite(toy_rp) else "pending", "keep_reason": keep_reason,
                    "manipulation": manip, "recovery": recovery}, "R3": r3},
            "t_critical": T_CRIT, "contrasts": stats, "partial_so": partial,
            "H3_point_contrasts": h3, "code_deficit_strata": code_strata,
            "exploratory": False}


def _collect_minie(root: str | Path) -> list[dict]:
    """Read MINI-E's checkpoint-carried files without requiring checkpoints."""
    root = Path(root)
    rows = []
    for path in sorted(root.glob("**/eval_ckpt_*.npz")):
        if _abandoned(root, path):
            continue
        preset, arm, seed = _path_info(root, path)
        rule = path.stem.split("eval_ckpt_", 1)[1]
        try:
            with np.load(path, allow_pickle=False) as z:
                key = f"x_{rule}_M"
                if key in z:
                    m = float(np.nanmean(z[key]))
                else:
                    m = _mean_difference(z, "doc_fP_focal", "doc_fP_nonfocal")
                g = (_mean(z, "doc_noadapt") - _mean(z, "doc_adapt"))
                rows.append({"preset": preset, "arm": arm, "seed": str(seed),
                             "rule": rule, "M": m, "G": g,
                             "n_docs": int(np.asarray(z["n_docs"]).item()) if "n_docs" in z else len(z.get("doc_fP_focal", [])),
                             "v0_pass": (bool(np.asarray(z["v0_pass"]).item())
                                         if "v0_pass" in z else None),
                             "v0_abs_diff": (float(np.asarray(z["v0_abs_diff"]).item())
                                             if "v0_abs_diff" in z else np.nan),
                             "path": str(path)})
        except (OSError, ValueError, KeyError):
            continue
    return rows


def _interval(values, t=T_CRIT):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan, "ub": np.nan,
                "values": []}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    half = t * sd / np.sqrt(len(x)) if np.isfinite(sd) else np.nan
    return {"n": int(len(x)), "mean": mean, "sd": sd,
            "lb": mean - half if np.isfinite(half) else mean,
            "ub": mean + half if np.isfinite(half) else mean,
            "values": x.tolist()}


def _minil_rows(root: str | Path, *, return_issues: bool = False):
    """Read complete MINI-L cells and report every missing or unreadable input."""
    root = Path(root)
    rows = []
    issues = []
    labels = ("L-SO", "L-FO", "L-REF")

    def issue(label, seed, reason, path=None):
        rec = {"label": label, "seed": int(seed), "reason": reason}
        if path is not None:
            rec["path"] = str(path)
        issues.append(rec)

    for label in labels:
        for seed in range(3):
            dirs = [p for p in sorted(root.glob(f"**/{label}/s{seed}"))
                    if p.is_dir() and not _abandoned(root, p / "eval_step0300.npz")]
            if not dirs:
                issue(label, seed, "missing_cell")
                continue
            cell = dirs[0]
            marker = cell / "MINIL_CELL_DONE.json"
            marker_ok = True
            if not marker.exists():
                issue(label, seed, "missing_completion_marker", marker)
                marker_ok = False
            else:
                try:
                    done = json.loads(marker.read_text())
                    if done.get("status") != "done":
                        issue(label, seed, "completion_marker_not_done", marker)
                        marker_ok = False
                    if int(done.get("n_docs", -1)) != 1024:
                        issue(label, seed, "completion_marker_n_docs_not_1024", marker)
                        marker_ok = False
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    issue(label, seed, f"unreadable_completion_marker:{type(exc).__name__}", marker)
                    marker_ok = False
            for step in (150, 300):
                path = cell / f"eval_step{step:04d}.npz"
                valid = marker_ok
                if not path.exists():
                    issue(label, seed, f"missing_eval_step_{step}", path)
                    continue
                try:
                    with np.load(path, allow_pickle=False) as z:
                        if "doc_fP_focal" not in z or "doc_fP_nonfocal" not in z:
                            raise KeyError("doc_fP_focal/doc_fP_nonfocal")
                        n_docs = len(np.asarray(z["doc_fP_focal"]))
                        if len(np.asarray(z["doc_fP_nonfocal"])) != n_docs:
                            raise ValueError("focal/nonfocal document lengths differ")
                        if step == 300 and n_docs != 1024:
                            issue(label, seed, f"n_docs_at_300={n_docs}", path)
                            valid = False
                        m = _mean_difference(z, "doc_fP_focal", "doc_fP_nonfocal")
                        g = _mean(z, "doc_noadapt") - _mean(z, "doc_adapt")
                except Exception as exc:
                    issue(label, seed, f"unreadable_eval:{type(exc).__name__}", path)
                    continue
                if valid:
                    rows.append({"label": label, "seed": seed, "step": step,
                                 "M": m, "G": g, "n_docs": n_docs, "path": str(path)})
    if return_issues:
        return rows, issues
    return rows


def _minil_core(root: Path, seed: int,
                mini_root: str | Path = "outputs/mini/matrix") -> tuple[float, float]:
    """Return (M_SO, M_FO) at MINI step 1200 for one seed."""
    candidates = [
        root / "core" / "runs" / "mini-m",
        Path(mini_root) / f"g{43 + seed}" / "runs" / "mini-m",
    ]
    for base in candidates:
        vals = []
        for arm in ("e2e2", "e2e1-ce"):
            path = base / arm / f"s{seed}" / "eval_step1200.npz"
            if not path.exists():
                break
            try:
                with np.load(path, allow_pickle=False) as z:
                    vals.append(_mean_difference(z, "doc_fP_focal", "doc_fP_nonfocal"))
            except Exception:
                vals = []
                break
        if len(vals) == 2 and all(np.isfinite(v) for v in vals):
            return float(vals[0]), float(vals[1])
    return np.nan, np.nan


def minil_analysis(root: str | Path,
                   mini_root: str | Path = "outputs/mini/matrix") -> dict:
    """Apply the registered MINI-L rule to paired SO/FO continuations."""
    root = Path(root)
    rows, issues = _minil_rows(root, return_issues=True)
    by = {(r["label"], r["seed"], r["step"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    available = {(r["label"], r["seed"], r["step"]) for r in rows}
    complete_counts = {
        label: len({seed for seed in range(3)
                    if (label, seed, 150) in available and
                       (label, seed, 300) in available})
        for label in ("L-SO", "L-FO", "L-REF")
    }
    paired = []
    for seed in seeds:
        ls = by.get(("L-SO", seed, 300), {})
        lf = by.get(("L-FO", seed, 300), {})
        lr = by.get(("L-REF", seed, 300), {})
        ls150 = by.get(("L-SO", seed, 150), {})
        lf150 = by.get(("L-FO", seed, 150), {})
        lr150 = by.get(("L-REF", seed, 150), {})
        if all(np.isfinite(x.get("M", np.nan))
               for x in (ls, lf, lr, ls150, lf150, lr150)):
            so0, fo0 = _minil_core(root, seed, mini_root)
            d, r = ls["M"] - lf["M"], lr["M"] - lf["M"]
            paired.append({"seed": seed, "M_L_SO": ls["M"], "M_L_FO": lf["M"],
                           "M_L_REF": lr["M"], "D": d, "R": r,
                           "M_SO_1200": so0, "M_FO_1200": fo0,
                           "delta_M_1200": so0 - fo0})
    dvals = [x["D"] for x in paired]
    rvals = [x["R"] for x in paired]
    d_half_r = [d - .5 * r for d, r in zip(dvals, rvals)]
    d_quarter_r = [d - .25 * r for d, r in zip(dvals, rvals)]
    d_int, r_int = _interval(dvals), _interval(rvals)
    half_int, quarter_int = _interval(d_half_r), _interval(d_quarter_r)
    if any(complete_counts[x] < 3 for x in ("L-SO", "L-FO", "L-REF")) or len(paired) < 3:
        label = "NE"
    elif half_int["lb"] > 0:
        label = "LATE-REPAIRS"
    elif quarter_int["ub"] < 0:
        label = "LATE-FAILS"
    elif d_int["lb"] > 0:
        label = "LATE-PARTIAL"
    else:
        label = "AMBIGUOUS"
    psi = (float(np.mean(dvals)) / float(np.mean(rvals))
           if dvals and np.isfinite(np.mean(rvals)) and np.mean(rvals) != 0 else np.nan)
    baseline_missing = not paired or any(
        not (np.isfinite(x["M_SO_1200"]) and np.isfinite(x["M_FO_1200"]))
        for x in paired)
    psi_prime = ([] if baseline_missing else
                 [x["D"] / x["delta_M_1200"] for x in paired
                  if np.isfinite(x["delta_M_1200"]) and x["delta_M_1200"] != 0])
    rewarm_cells = ([] if baseline_missing else
                    [{"seed": x["seed"], "drop": x["M_L_FO"] - x["M_FO_1200"]}
                     for x in paired if x["M_L_FO"] < x["M_FO_1200"] - .03])
    m_at = {str(step): {label: [by[(label, s, step)]["M"] for s in seeds
                                  if (label, s, step) in by]
                        for label in ("L-SO", "L-FO", "L-REF")}
            for step in (150, 300)}
    result = {"label": label,
              "label_short": {"LATE-REPAIRS": "REPAIRS", "LATE-FAILS": "FAILS",
                              "LATE-PARTIAL": "PARTIAL"}.get(label, label),
              "mini_root": str(mini_root),
              "t_critical": T_CRIT, "n": len(paired),
              "complete_counts": complete_counts, "issues": issues,
              "paired": paired, "D": d_int, "R": r_int,
              "D_minus_half_R": half_int, "D_minus_quarter_R": quarter_int,
              "psi": psi, "psi_prime_values": psi_prime,
              "psi_prime": {**_interval(psi_prime),
                            **({"status": "NE", "reason": "missing MINI step-1200 baseline"}
                               if baseline_missing else {"status": "OK"})},
              "rewarm_check": ({"status": "NE", "triggered": None, "cells": [],
                                "reason": "missing MINI step-1200 baseline"}
                               if baseline_missing else
                               {"status": "OK", "triggered": bool(rewarm_cells),
                                "cells": rewarm_cells}),
              "M_at_150_300": m_at, "M_150": m_at["150"], "M_300": m_at["300"],
              "rows": rows}
    (root / "minil_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    md = ["# MINI-L analysis", "", f"Label: **{label}**", "",
          f"Paired seeds: {len(paired)}; D = M(L-SO) − M(L-FO); R = M(L-REF) − M(L-FO).",
          "", f"ψ = {psi:.6g}; ψ′ interval = {result['psi_prime']}",
          "", f"Re-warm check: {result['rewarm_check']}", "",
          "```json", json.dumps({"D": d_int, "R": r_int,
                                  "D_minus_half_R": half_int,
                                  "D_minus_quarter_R": quarter_int,
                                  "M_at_150_300": m_at}, indent=2, allow_nan=True),
          "```", ""]
    (root / "minil_analysis.md").write_text("\n".join(md))
    return result


def _point_tree(dm, c_own, c_t, c_e, c_nc, selectivity, recovery, *,
                h1_supported=True, required_status="OK"):
    """Registered decision tree applied to point estimates; MINI-E labels this rehearsal."""
    if not h1_supported:
        return "NOT-EVALUABLE (H1)"
    if required_status != "OK":
        return "NOT-EVALUABLE (NE)"
    vals = (dm, c_own, c_t, c_e, c_nc, selectivity, recovery)
    if not all(np.isfinite(x) for x in vals):
        return "NOT-EVALUABLE"
    if not (c_own > 0 or c_t > 0):
        if recovery > 0 and recovery >= .25 * dm:
            return "PARTIAL"
        if recovery < .25 * dm:
            return "NO-RESCUE"
        return "UNRESOLVED"
    if c_nc > 0 or selectivity < 0:
        return "NOT-SELECTIVE"
    if not selectivity > 0:
        return "UNRESOLVED"
    if c_t > 0 and c_e > 0:
        return "MIXED"
    if c_t > 0:
        return "TRAINING-SELECTIVE"
    if c_e > 0:
        return "EVAL-CARRIED"
    return "UNRESOLVED"


def _source_cot(root: Path, preset: str):
    """Return paired FO/SO E2c source cotangent sums by document.

    ``src_cot_adapt`` is stored in focal-source order, while probe positions are
    in probe order.  The last source is ``probe_pos - probe_dist``; map that
    position back into each document's focal slice before selecting E2c.
    """
    rows = []
    try:
        rows = collect(root)
    except FileNotFoundError:
        return None
    by = {}
    for row in rows:
        if row["preset"] != preset or row["step"] != 1200 or row["corpus"] != "dclm":
            continue
        if row["arm"] not in ("e2e2", "e2e1-ce"):
            continue
        by[(row["arm"], str(row["seed"]))] = Path(row["path"])
    out = {"e2e2": [], "e2e1-ce": []}
    seeds = sorted({seed for arm, seed in by if arm == "e2e2"} &
                   {seed for arm, seed in by if arm == "e2e1-ce"})
    required = {"probe_offset", "probe_pos", "probe_stratum", "probe_dist",
                "focal_offset", "focal_pos"}

    def sums(path):
        with np.load(path, allow_pickle=False) as z:
            if "src_cot_adapt" not in z:
                return None
            cot = np.asarray(z["src_cot_adapt"], dtype=float)
        meta_path = path.parent / "eval_meta.npz"
        if not meta_path.exists():
            return None
        with np.load(meta_path, allow_pickle=False) as m:
            if not required <= set(m.files):
                return None
            po, pp, ps, pd = (m["probe_offset"], m["probe_pos"],
                              m["probe_stratum"], m["probe_dist"])
            fo, fp = m["focal_offset"], m["focal_pos"]
            result = []
            for d in range(len(fo) - 1):
                psl = slice(int(po[d]), int(po[d + 1]))
                fsl = slice(int(fo[d]), int(fo[d + 1]))
                last = np.asarray(pp[psl], dtype=np.int64) - np.asarray(pd[psl], dtype=np.int64)
                focal = np.asarray(fp[fsl], dtype=np.int64)
                if not len(focal):
                    result.append(np.nan)
                    continue
                idx = np.searchsorted(focal, last)
                valid = ((np.asarray(ps[psl]) == 2) & (idx < len(focal)) &
                         (focal[np.minimum(idx, max(len(focal) - 1, 0))] == last))
                if not valid.any():
                    result.append(np.nan)
                    continue
                values = cot[int(fo[d]) + idx[valid]]
                result.append(float(np.nansum(values)) if np.isfinite(values).any() else np.nan)
            return np.asarray(result, dtype=float)

    for seed in seeds:
        try:
            so, fo = sums(by[("e2e2", seed)]), sums(by[("e2e1-ce", seed)])
            if so is None or fo is None or len(so) != len(fo):
                continue
            keep = np.isfinite(so) & np.isfinite(fo) & (so > 0)
            out["e2e2"].extend(so[keep].tolist())
            out["e2e1-ce"].extend(fo[keep].tolist())
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return out if out["e2e2"] and out["e2e1-ce"] else None


def _ratio_interval(so, fo, *, seed=0, n_boot=10000):
    so, fo = np.asarray(so, dtype=float), np.asarray(fo, dtype=float)
    keep = np.isfinite(so) & np.isfinite(fo) & (so > 0)
    so, fo = so[keep], fo[keep]
    if not len(so):
        return {"n": 0, "mean": np.nan, "lb": np.nan, "ub": np.nan}
    mean = float(fo.sum() / so.sum())
    if len(so) == 1:
        return {"n": 1, "mean": mean, "lb": mean, "ub": mean}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(so), len(so))
        draws[i] = fo[idx].sum() / so[idx].sum()
    return {"n": int(len(so)), "mean": mean,
            "lb": float(np.quantile(draws, .05)), "ub": float(np.quantile(draws, .95))}


def _starvation(root: Path, preset: str) -> dict:
    cot = _source_cot(root, preset)
    if cot is None:
        return {"status": "NE", "reason": "per-source src_cot_adapt or eval_meta strata unavailable"}
    ratios = []
    for log in root.glob(f"**/{preset}/e2e1-ce/**/steps.jsonl"):
        try:
            for line in log.read_text().splitlines():
                if not line.startswith("P2_STEP "):
                    continue
                rec = json.loads(line.split(" ", 1)[1])
                if not 1100 <= int(rec.get("step", 0)) <= 1200:
                    continue
                for key in ("ratio_r_within", "r_ratio_within", "mean_r_ratio_within"):
                    if key in rec:
                        ratios.append(float(rec[key]))
                        break
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not ratios:
        # The current MINI training log does not record r-bar by repeat class.
        # Keep the registered class NE, but report the available ratio_cot.
        rc = _ratio_interval(cot["e2e2"], cot["e2e1-ce"])
        return {"status": "NE", "reason": "ratio_r^within not logged in MINI",
                "ratio_cot": rc["mean"], "ratio_cot_interval": rc,
                "e2c_source_counts": {k: len(v) for k, v in cot.items()}}
    rr = float(np.mean(ratios))
    rc = _ratio_interval(cot["e2e2"], cot["e2e1-ce"])
    label = ("STARVATION-TRANSFERS" if rr < .8 and rc["ub"] < .9 else
             "NO-STARVATION" if rr >= .95 and rc["lb"] > .95 else "MIXED")
    return {"status": label, "ratio_r_within": rr, "ratio_cot": rc["mean"],
            "ratio_cot_interval": rc,
            "e2c_source_counts": {k: len(v) for k, v in cot.items()}, "rehearsal": True}


def minie_analysis(root: str | Path) -> dict:
    """Analyze MINI-E C_E reads and the registered rehearsal qualifiers."""
    root = Path(root)
    try:
        core = collect(root)
        base = analyse(core, root)
    except FileNotFoundError:
        core, base = [], {"passing_presets": [], "preset_stats": {}}
    carried = _collect_minie(root)
    presets = sorted(set([r["preset"] for r in core] + [r["preset"] for r in carried]),
                     key=lambda x: PRESET_DIM.get(x, 0))
    reports = {}
    for p in presets:
        final = [r for r in core if r["preset"] == p and r["step"] == 1200 and r["corpus"] == "dclm"]
        by = {(r["arm"], r["seed"]): r for r in final}
        fo0 = by.get(("e2e1-ce", "0"), {})
        base_m = fo0.get("M", np.nan)
        dm = base.get("preset_stats", {}).get(p, {}).get("mean", np.nan)
        rp = by.get(("e2e1-rp", "0"), {})
        nc = by.get(("e2e1-nc", "0"), {})
        m_rp_rp, m_rp_ce = rp.get("M", np.nan), rp.get("M_ce", np.nan)
        m_nc_nc, m_nc_ce = nc.get("M", np.nan), nc.get("M_ce", np.nan)
        e = {(r["seed"], r["rule"]): r for r in carried
             if r["preset"] == p and r["arm"] == "e2e1-ce"}
        v0_bad = [k for k, r in e.items() if k[1] == "ce" and r.get("v0_pass") is not True]
        v0_seed0 = e.get(("0", "ce"), {}).get("v0_pass") is True
        ce_vals = [e[(str(k), "rp")]["M"] - e[(str(k), "ce")]["M"]
                   for k in range(3) if (str(k), "rp") in e and (str(k), "ce") in e]
        ce_interval = _interval(ce_vals)
        if v0_bad:
            ce_interval = {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan,
                           "ub": np.nan, "values": [], "status": "NE",
                           "reason": "V0 failed for " + ",".join(str(x[0]) for x in v0_bad)}
        else:
            ce_interval["status"] = "OK" if ce_interval["n"] == 3 else "NE"
            if ce_interval["status"] == "NE":
                ce_interval["reason"] = "missing MINI-E seed read"
        if not v0_seed0:
            nc_interval = {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan,
                           "ub": np.nan, "values": [], "status": "NE",
                           "reason": "V0 failed for seed 0"}
        elif ("0", "nc") in e and ("0", "ce") in e:
            nc_interval = _interval([e[("0", "nc")]["M"] - e[("0", "ce")]["M"]])
            nc_interval["status"] = "POINT"
        else:
            nc_interval = {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan,
                           "ub": np.nan, "values": [], "status": "NE",
                           "reason": "missing MINI-E seed 0 NC read"}
        c_own = m_rp_rp - base_m if np.isfinite(m_rp_rp) and np.isfinite(base_m) else np.nan
        c_t = m_rp_ce - base_m if np.isfinite(m_rp_ce) and np.isfinite(base_m) else np.nan
        c_nc_own = m_nc_nc - base_m if np.isfinite(m_nc_nc) and np.isfinite(base_m) else np.nan
        c_nc_t = m_nc_ce - base_m if np.isfinite(m_nc_ce) and np.isfinite(base_m) else np.nan
        c_e = ce_interval["mean"]
        c_nc_e = nc_interval["mean"]
        selectivity = m_rp_rp - m_nc_nc if np.isfinite(m_rp_rp) and np.isfinite(m_nc_nc) else np.nan
        pstat = base.get("preset_stats", {}).get(p, {})
        h1_supported = bool(pstat.get("n") == 3 and
                            pstat.get("rho", np.nan) >= .145 and
                            pstat.get("lb", np.nan) > 0)
        required_status = ("OK" if ce_interval.get("status") == "OK" and
                           all(np.isfinite(x) for x in
                               (c_own, c_t, c_nc_own, selectivity, c_e)) else "NE")
        tree_c = {
            "C_own_minus_half_delta": c_own - .5 * dm if np.isfinite(c_own) and np.isfinite(dm) else np.nan,
            "C_T_minus_half_delta": c_t - .5 * dm if np.isfinite(c_t) and np.isfinite(dm) else np.nan,
            "C_E_minus_half_delta": c_e - .5 * dm if np.isfinite(c_e) and np.isfinite(dm) else np.nan,
            "C_NC_minus_half_delta": c_nc_own - .5 * dm if np.isfinite(c_nc_own) and np.isfinite(dm) else np.nan,
        }
        reports[p] = {
            "M_SO_minus_FO": dm, "M_CE_0_ce": base_m,
            "M_RP_rp": m_rp_rp, "M_RP_ce": m_rp_ce,
            "M_NC_nc": m_nc_nc, "M_NC_ce": m_nc_ce,
            "C_own": c_own, "C_T": c_t, "C_E": ce_interval,
            "C_NC_own": c_nc_own, "C_NC_T": c_nc_t, "C_NC_E": nc_interval,
            "S": selectivity,
            "C_own_raw": c_own, "C_T_raw": c_t, "C_E_raw": c_e,
            "C_NC_own_raw": c_nc_own, "C_NC_T_raw": c_nc_t, "C_NC_E_raw": c_nc_e,
            "tree_rehearsal": _point_tree(dm, tree_c["C_own_minus_half_delta"],
                                           tree_c["C_T_minus_half_delta"],
                                           tree_c["C_E_minus_half_delta"],
                                           tree_c["C_NC_minus_half_delta"],
                                           selectivity, c_own,
                                           h1_supported=h1_supported,
                                           required_status=required_status),
            "tree_inputs_minus_half_delta": tree_c,
            "tree_required_status": required_status,
            "h1_supported": h1_supported,
            "provenance": {
                "C_own": "core logged own-rule M values",
                "C_T": "core logged CE crossover M values",
                "C_E": "MINI-E recomputed CE checkpoint reads",
                "C_NC_E": "MINI-E CE checkpoint read under NC (seed 0 point)",
            },
            "v0_failed": v0_bad,
        }
    passing = [p for p in base.get("passing_presets", []) if p in reports]
    primary = max(passing, key=lambda x: PRESET_DIM.get(x, 0)) if passing else None
    r2 = {"row": False, "preset": primary, "inputs": {}}
    if primary:
        q = reports[primary]
        dm = q["M_SO_minus_FO"]
        fo0_rp = next((r["M"] for r in carried if r["preset"] == primary and r["arm"] == "e2e1-ce" and r["seed"] == "0" and r["rule"] == "rp"), np.nan)
        rec = q["M_RP_rp"] - fo0_rp if np.isfinite(q["M_RP_rp"]) and np.isfinite(fo0_rp) else np.nan
        r2["inputs"] = {"LB_C_E": q["C_E"].get("lb"), "C_E": q["C_E"].get("mean"),
                         "half_delta": .5 * dm if np.isfinite(dm) else np.nan,
                         "recovery_vs_fo0_rp": rec,
                         "quarter_delta": .25 * dm if np.isfinite(dm) else np.nan,
                         "delta_positive_required": True}
        r2["row"] = bool(np.isfinite(dm) and dm > 0 and q["C_E"].get("status") == "OK" and
                          q["C_E"].get("lb", np.nan) > 0 and q["C_E"].get("mean", np.nan) >= .5 * dm and
                          rec < .25 * dm)
        if not dm > 0:
            r2["reason"] = "nonpositive_delta_report_only"
    starvation = {p: _starvation(root, p) for p in presets}
    return {"available": bool(carried), "primary_preset": primary,
            "per_preset": reports, "r2_item4": r2,
            "starvation_rehearsal": starvation, "label": "rehearsal"}


def _minia_rank(x):
    x = np.asarray(x, dtype=float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[o[j]] == x[o[i]]:
            j += 1
        r[o[i:j]] = (i + j - 1) / 2
        i = j
    return r


def _minia_interval(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan, "ub": np.nan, "values": []}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    half = T_CRIT * sd / np.sqrt(len(x)) if np.isfinite(sd) else np.nan
    return {"n": int(len(x)), "mean": mean, "sd": sd,
            "lb": mean - half if np.isfinite(half) else mean,
            "ub": mean + half if np.isfinite(half) else mean, "values": x.tolist()}


def _minia_cell(path: Path):
    with np.load(path, allow_pickle=False) as z:
        def docs(values, n):
            x = np.asarray(values, dtype=float)
            if x.ndim == 0:
                return np.full(n, float(x))
            if len(x) == n:
                return x
            if len(x) == 1:
                return np.repeat(x, n)
            # Keep hand-built test fixtures readable while making malformed
            # production files visible through the paired metadata checks.
            return np.full(n, np.nanmean(x))

        v = np.asarray(z["normVF"], float)
        c = np.asarray(z["cosF"], float)
        m = np.asarray(z["M_doc"], float).reshape(-1)
        n = len(m)
        if "n_probes" in z:
            if v.ndim != 2 or c.ndim != 2 or v.shape != c.shape \
                    or v.shape[0] != n or np.asarray(z["normGr"]).reshape(-1).size != n \
                    or np.asarray(z["bF"]).reshape(-1).size != n \
                    or np.asarray(z["bN"]).reshape(-1).size != n \
                    or np.asarray(z["n_probes"]).reshape(-1).size != n:
                raise ValueError("align.npz document/chunk arrays are misaligned")
        vdoc = docs(v.sum(1) if v.ndim > 1 else v, n)
        if v.ndim > 1 and c.ndim > 1 and len(c) == 1 and len(v) > 1:
            c = np.broadcast_to(c, v.shape)
        cweighted = np.divide((v * c).sum(1), v.sum(1),
                              out=np.full(len(v), np.nan), where=v.sum(1) > 0) \
            if v.ndim > 1 and c.ndim > 1 else np.asarray(c, float)
        cdoc = docs(cweighted, n)
        b_f = docs(z["bF"], n)
        b_n = docs(z["bN"], n)
        grdoc = docs(z["normGr"], n) if "normGr" in z else np.full(n, np.nan)
        k = docs(z["n_probes"], n) if "n_probes" in z else np.ones(n)
        bdiff = b_f - b_n
        pred = np.divide(bdiff, k, out=np.full(n, np.nan), where=k > 0)

        def spearman(x, y):
            ok = np.isfinite(x) & np.isfinite(y)
            return (float(np.corrcoef(_minia_rank(x[ok]), _minia_rank(y[ok]))[0, 1])
                    if ok.sum() > 1 and np.std(x[ok]) and np.std(y[ok]) else np.nan)

        eval_sha1 = None
        if "eval_sha1" in z:
            eval_sha1 = tuple(str(x) for x in np.asarray(z["eval_sha1"]).reshape(-1))
        return {"preset": str(np.asarray(z["preset"]).item()) if "preset" in z else path.parts[-4],
                "arm": str(np.asarray(z["arm"]).item()) if "arm" in z else path.parts[-3],
                "seed": str(int(np.asarray(z["init_seed"]).item())) if "init_seed" in z else path.parts[-2].lstrip("s"),
                "cosF": float(np.nanmean(cdoc)),
                "normVF": float(np.nanmean(vdoc)),
                "normGr": float(np.nanmean(grdoc)),
                "bF": float(np.nanmean(b_f)), "bN": float(np.nanmean(b_n)),
                "n_bF_nonpositive": int(np.sum(np.isfinite(b_f) & (b_f <= 0))),
                "n_bN_nonpositive": int(np.sum(np.isfinite(b_n) & (b_n <= 0))),
                "spearman": spearman(pred, m), "spearman_raw": spearman(bdiff, m),
                "eval_sha1": eval_sha1, "path": str(path)}


def minia_analysis(root: str | Path, out_json: str | Path | None = None,
                   out_md: str | Path | None = None) -> dict:
    """Apply the registered MINI-A alignment rule to committed ``align.npz`` files."""
    root = Path(root)
    cells = []
    input_errors = []
    for p in sorted(root.glob("**/align.npz")):
        try:
            cells.append(_minia_cell(p))
        except (OSError, KeyError, ValueError) as exc:
            input_errors.append({"path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    by = {}
    for c in cells:
        by.setdefault(c["preset"], {}).setdefault(c["arm"], {})[c["seed"]] = c
    reports = {}
    for preset, arms in sorted(by.items(), key=lambda q: PRESET_DIM.get(q[0], 0)):
        so, fo = arms.get("e2e2", {}), arms.get("e2e1-ce", {})
        pair_errors = []
        seeds = []
        for seed in sorted(set(so) & set(fo), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            if so[seed].get("eval_sha1") is not None and fo[seed].get("eval_sha1") is not None \
                    and so[seed]["eval_sha1"] != fo[seed]["eval_sha1"]:
                pair_errors.append({"seed": seed, "error": "SO/FO eval_sha1 mismatch",
                                    "so": so[seed]["path"], "fo": fo[seed]["path"]})
                continue
            seeds.append(seed)
        spears = [so[s]["spearman"] for s in seeds if np.isfinite(so[s]["spearman"])]
        raw_spears = [so[s]["spearman_raw"] for s in seeds if np.isfinite(so[s]["spearman_raw"])]
        gate = {"n": len(spears), "mean": float(np.mean(spears)) if spears else np.nan,
                "pass": bool(len(spears) == 3 and np.mean(spears) >= .20), "values": spears,
                "raw_n": len(raw_spears), "raw_mean": float(np.mean(raw_spears)) if raw_spears else np.nan,
                "raw_values": raw_spears}
        dcos = [so[s]["cosF"] - fo[s]["cosF"] for s in seeds]
        delta = _minia_interval(dcos)
        cos_so = float(np.mean([so[s]["cosF"] for s in seeds])) if seeds else np.nan
        decomposition = {"write_size": np.nan, "read_sensitivity": np.nan, "direction": np.nan,
                         "residual": np.nan, "delta_log_b": np.nan, "status": "NE", "raw": {}}
        if seeds:
            pairs = [(so[s], fo[s]) for s in seeds]
            means = {"normVF": (np.mean([a["normVF"] for a, b in pairs]), np.mean([b["normVF"] for a, b in pairs])),
                     "normGr": (np.mean([a["normGr"] for a, b in pairs]), np.mean([b["normGr"] for a, b in pairs])),
                     "cosF": (np.mean([a["cosF"] for a, b in pairs]), np.mean([b["cosF"] for a, b in pairs]))}
            vals = {"write_size": np.log(means["normVF"][0] / means["normVF"][1])
                    if min(means["normVF"]) > 0 else np.nan,
                    "read_sensitivity": np.log(means["normGr"][0] / means["normGr"][1])
                    if min(means["normGr"]) > 0 else np.nan,
                    "direction": np.log(means["cosF"][0] / means["cosF"][1])
                    if min(means["cosF"]) > 0 else np.nan}
            mean_b = (np.mean([a["bF"] for a, b in pairs]), np.mean([b["bF"] for a, b in pairs]))
            delta_log_b = np.log(mean_b[0] / mean_b[1]) if min(mean_b) > 0 else np.nan
            finite_vals = [v for v in vals.values() if np.isfinite(v)]
            residual = delta_log_b - sum(finite_vals) if np.isfinite(delta_log_b) and len(finite_vals) == 3 else np.nan
            raw = {k: float(v) for k, v in vals.items()}
            raw.update({"delta_log_b": float(delta_log_b), "residual": float(residual)})
            decomposition["raw"] = raw
            decomposition.update({"delta_log_b": float(delta_log_b), "residual": float(residual)})
            if means["cosF"][0] > 0 and means["cosF"][1] > 0 and np.isfinite(delta_log_b) and delta_log_b != 0 and len(finite_vals) == 3:
                decomposition.update({k: float(v / delta_log_b) for k, v in vals.items()})
                decomposition["residual"] = float(residual / delta_log_b)
                decomposition["status"] = "OK"
        cell_errors = input_errors + pair_errors
        if not gate["pass"] or not seeds:
            label = "NE"
        elif not np.isfinite(delta["mean"]):
            label = "NE"
        elif delta["lb"] > 0 and delta["mean"] >= .10 * cos_so:
            label = "H-ALIGN SUPPORTED"
        elif delta["ub"] < .05 * cos_so:
            label = "H-ALIGN NOT SUPPORTED"
        else:
            label = "AMBIGUOUS"
        reports[preset] = {"seeds": seeds, "v_gate": gate, "cos_SO": cos_so,
                           "delta_cos": delta, "decomposition": decomposition,
                           "label": label, "cells": [c for c in cells if c["preset"] == preset],
                           "input_errors": cell_errors}
    result = {"root": str(root), "per_preset": reports,
              "labels": {p: r["label"] for p, r in reports.items()}, "n_cells": len(cells),
              "input_errors": input_errors}
    if out_json is None:
        out_json = root / "minia_analysis.json"
    if out_md is None:
        out_md = root / "minia_analysis.md"
    Path(out_json).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    lines = ["# MINI-A write–read gradient alignment", ""]
    for p, r in reports.items():
        lines.append(f"- `{p}`: **{r['label']}**; V-gate mean Spearman={r['v_gate']['mean']:.4g}; Δcos={r['delta_cos']['mean']:.4g}")
        lines.append("  decomposition shares: " + json.dumps(r["decomposition"], sort_keys=True, allow_nan=True))
        lines.append("  per-cell b≤0: " + json.dumps([
            {"arm": c["arm"], "seed": c["seed"], "bF": c["n_bF_nonpositive"],
             "bN": c["n_bN_nonpositive"]} for c in r["cells"]], sort_keys=True))
        if r["input_errors"]:
            lines.append("  input errors: " + json.dumps(r["input_errors"], sort_keys=True))
    Path(out_md).write_text("\n".join(lines) + "\n")
    return result


def write(root: str | Path, out_json: str | Path, out_md: str | Path) -> dict:
    result = analyse(collect(root), root)
    result["minie"] = minie_analysis(root)
    Path(out_json).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    lines = ["# MINI pre-flight analysis", "", f"primary preset: `{result['primary_preset']}`", "",
             "R1/R2/R3 (registered rules):",
             "M0 gate and descriptive init gains: " + json.dumps(result["M0_gate"], sort_keys=True), json.dumps(result["preflight_rules"], sort_keys=True)]
    lines += ["", "MINI-E C_E (rehearsal):",
              json.dumps(result["minie"], sort_keys=True, allow_nan=True)]
    Path(out_md).write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-min-05-analysis"))
    ap.add_argument("--minia", action="store_true")
    a = ap.parse_args(argv)
    os.environ["PROCESS_NAME"] = a.process_name
    try:
        import setproctitle
        setproctitle.setproctitle(a.process_name)
    except ImportError:
        pass
    root = Path(a.root)
    if a.minia:
        result = minia_analysis(root, a.out_json, a.out_md)
        print("MINIA_ANALYSIS " + json.dumps(result["labels"], sort_keys=True), flush=True)
        return
    result = write(root, a.out_json or root / "mini_analysis.json", a.out_md or root / "mini_analysis.md")
    for name in ("R1", "R2", "R3"):
        print(f"{name}_MATCHED " + json.dumps(result["preflight_rules"][name], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

# The registered TBB validation is kept beside the MINI-D helpers so both
# analyses share file discovery and interval conventions.
from .tbb_analysis import tbb_analysis
