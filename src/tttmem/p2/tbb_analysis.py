"""Registered V2 micro16k screen and F1--F6 analysis."""
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np

T_CRIT = 1.886
ARMS = ("e2e2", "e2e1-ce", "e2e12-p125", "e2e1-rp")
STRATA = ("E1", "E2s", "E2c", "E2b", "E3")

def _interval(x):
    a = np.asarray(x, dtype=float); a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": np.nan, "sd": np.nan, "lb": np.nan, "ub": np.nan, "values": []}
    m = float(a.mean()); sd = float(a.std(ddof=1)) if len(a) > 1 else np.nan
    h = T_CRIT * sd / math.sqrt(len(a)) if np.isfinite(sd) else 0.0
    return {"n": int(len(a)), "mean": m, "sd": sd, "lb": m-h, "ub": m+h, "values": a.tolist()}

def _path_info(root, path):
    parts = path.relative_to(root).parts
    arm = next((x for x in parts if x in ARMS), path.parent.name)
    seed = next((int(m.group(1)) for x in parts if (m := re.fullmatch(r"(?:s|seed)(\d+)", x))), 0)
    return arm, seed

def _m(z):
    if z is None:
        return np.nan
    if "M" in z: return float(np.nanmean(z["M"]))
    if "doc_M" in z: return float(np.nanmean(z["doc_M"]))
    return float(np.nanmean(np.asarray(z["doc_fP_focal"]) - np.asarray(z["doc_fP_nonfocal"])))

def _g(z):
    if "G" in z: return float(np.nanmean(z["G"]))
    return float(np.nanmean(z["doc_noadapt"]) - np.nanmean(z["doc_adapt"])) if "doc_noadapt" in z and "doc_adapt" in z else np.nan

def _load_cells(root):
    cells = {}
    for p in sorted(Path(root).glob("**/eval_step*.npz")):
        arm, seed = _path_info(Path(root), p)
        if arm not in ARMS: continue
        try: step = int(p.stem.split("step", 1)[1])
        except (ValueError, IndexError): continue
        with np.load(p, allow_pickle=False) as z:
            cells[(arm, seed, step)] = {k: np.array(z[k]) for k in z.files}
    return cells

def _probe_gain(z):
    if "probe_delta" in z:
        return np.asarray(z["probe_delta"], dtype=float)
    if "tok_fP_focal" in z and "tok_fP_nonfocal" in z:
        return (np.asarray(z["tok_fP_focal"], dtype=float) -
                np.asarray(z["tok_fP_nonfocal"], dtype=float))
    return None


def _probe_meta(z, meta, key):
    """Per-probe arrays live in eval_meta.npz in production; fixtures may inline them."""
    if z is not None and key in z:
        return np.asarray(z[key])
    if meta is not None and key in meta:
        return np.asarray(meta[key])
    return None


def _contrast_rho(so, fo, sel):
    """Registered rho for one probe subset, using paired SO/FO probe gains."""
    ds, df = _probe_gain(so), _probe_gain(fo)
    if ds is None or df is None or sel is None:
        return np.nan
    if ds.shape != df.shape or ds.shape != sel.shape or not np.any(sel):
        return np.nan
    keep = sel & np.isfinite(ds) & np.isfinite(df)
    if not np.any(keep):
        return np.nan
    denom = float(np.nanmean(ds[keep]))
    return float((denom - np.nanmean(df[keep])) / denom) if denom else np.nan

def _distance_rho(so, fo, lo, hi, chunk, meta=None):
    dist = _probe_meta(so, meta, "probe_dist")
    if dist is None:
        return np.nan
    dist = np.asarray(dist, float) / float(chunk)
    return _contrast_rho(so, fo, (dist >= lo) & (dist < hi))

def _ratio_cot(so, fo, meta):
    for key in ("ratio_cot", "ratio_cot_e2c"):
        if key in so and key in fo:
            return float(np.asarray(fo[key]).mean() / max(float(np.asarray(so[key]).mean()), 1e-12))
    if "src_cot_adapt" not in so or "src_cot_adapt" not in fo or meta is None:
        return np.nan
    cot_s, cot_f = np.asarray(so["src_cot_adapt"], float), np.asarray(fo["src_cot_adapt"], float)
    st = meta.get("probe_stratum")
    po, ppd = meta.get("probe_offset"), meta.get("probe_pos")
    pd, fo_off, fp = meta.get("probe_dist"), meta.get("focal_offset"), meta.get("focal_pos")
    if any(x is None for x in (st, po, ppd, pd, fo_off, fp)): return np.nan
    ss = []; ff = []
    for i in range(len(fo_off)-1):
        p0, p1 = int(po[i]), int(po[i+1]); f0, f1 = int(fo_off[i]), int(fo_off[i+1])
        focal = np.asarray(fp[int(fo_off[i]):int(fo_off[i+1])], dtype=int)
        if not len(focal): continue
        last = np.asarray(ppd[p0:p1], int) - np.asarray(pd[p0:p1], int)
        pick = np.asarray(st[p0:p1]) == 2
        idx = np.searchsorted(focal, last[pick])
        valid = (idx < len(focal)) & (focal[np.minimum(idx, len(focal)-1)] == last[pick])
        ss.extend(cot_s[f0 + idx[valid]].tolist()); ff.extend(cot_f[f0 + idx[valid]].tolist())
    return float(np.sum(ff) / max(np.sum(ss), 1e-12)) if ss else np.nan

def tbb_analysis(root, out_json=None, out_md=None, *, chunk=512):
    """Return and optionally write the exact V2 stop rule and F1--F6 label."""
    root = Path(root); cells = _load_cells(root); meta_by_seed = {}
    for p in root.glob("**/eval_meta.npz"):
        arm, seed = _path_info(root, p); meta_by_seed[seed] = dict(np.load(p, allow_pickle=False))
    seeds = sorted({s for a,s,t in cells if a == "e2e2" and t == 1200})
    def z(arm, seed, step=1200): return cells.get((arm, seed, step))
    so = [z("e2e2", s) for s in seeds]; fo = [z("e2e1-ce", s) for s in seeds]
    pairs = [{"seed": s, "M_SO": _m(a), "M_FO": _m(b), "delta_M": _m(a)-_m(b)}
             for s,a,b in zip(seeds, so, fo) if a is not None and b is not None]
    dm = [p["delta_M"] for p in pairs]; ms = [p["M_SO"] for p in pairs]
    rho = float(np.mean(dm)/np.mean(ms)) if ms and np.mean(ms) else np.nan
    delta_ci = _interval(dm)
    init = [(_m(z("e2e2", s, 0)) if z("e2e2", s, 0) is not None else np.nan) for s in seeds]
    m0 = {"mean_M_SO": float(np.nanmean(ms)) if ms else np.nan,
          "mean_G_SO": float(np.nanmean([_g(x) for x in so if x is not None])) if so else np.nan,
          "seed_M": int(np.sum(np.asarray(ms) >= .02)) if ms else 0,
          "n_seeds": len(pairs), "final_gt_init": bool(ms and np.nanmean(ms) > np.nanmean(init) if np.isfinite(init).any() else False)}
    m0["pass"] = bool(m0["n_seeds"] >= 3 and m0["mean_M_SO"] >= .02 and m0["mean_G_SO"] >= .02 and
                        m0["seed_M"] >= math.ceil(2*m0["n_seeds"]/3) and m0["final_gt_init"])
    f1 = {"rho": rho, "rho_range": bool(np.isfinite(rho) and .15 <= rho <= .50),
          "LB_delta_M": delta_ci["lb"], "pass": bool(.15 <= rho <= .50 and delta_ci["lb"] > 0)}
    f2_vals=[]; f2_pass=True
    for p in pairs:
        vals=[_m(z("e2e1-ce",p["seed"],step)) for step in (300,600,1200) if z("e2e1-ce",p["seed"],step) is not None]
        ok=bool(vals and vals[-1] >= .9*max(vals)); f2_vals.append({"seed":p["seed"],"values":vals,"pass":ok}); f2_pass &= ok
    f3_vals=[]
    for p in pairs:
        f3_vals.append(_ratio_cot(z("e2e2",p["seed"]), z("e2e1-ce",p["seed"]), meta_by_seed.get(p["seed"])))
    f3 = {"ratio_cot": _interval(f3_vals), "pass": bool(f3_vals and np.nanmean(f3_vals) >= .90)}
    part = [{"seed":s,"value":_m(z("e2e12-p125",s))-_m(z("e2e1-ce",s)),"delta":_m(z("e2e2",s))-_m(z("e2e1-ce",s))}
            for s in seeds if z("e2e12-p125",s) is not None and z("e2e1-ce",s) is not None]
    phi = float(sum(x["value"] for x in part)/max(sum(x["delta"] for x in part),1e-12)) if part else np.nan
    f4={"phi":phi,"pass":bool(np.isfinite(phi) and phi <= .5)}
    rp=z("e2e1-rp", seeds[0]) if seeds else None
    rec=( _m(rp)-_m(z("e2e1-ce",seeds[0])) ) if rp is not None and z("e2e1-ce",seeds[0]) is not None else np.nan
    # F5 compares the seed-0 RP recovery with a quarter of the preset mean ΔM.
    quarter = .25*float(np.mean(dm)) if dm else np.nan
    f5={"recovery":rec,"quarter_delta":quarter,"skipped":rp is None,
        "pass":True if rp is None else bool(np.isfinite(rec) and rec < quarter)}
    def k_for(s):
        zz=z("e2e2",s); ff=z("e2e1-ce",s); meta=meta_by_seed.get(s)
        st=_probe_meta(zz, meta, "probe_stratum")
        if zz is None or ff is None or st is None: return np.nan,np.nan
        # K1 = rho(E2s, E2c, E3) - rho(E1, E2b); K2 = rho(dist >= 4 chunks) - rho(dist < 1 chunk).
        k1=_contrast_rho(zz,ff,np.isin(st,[1,2,4]))-_contrast_rho(zz,ff,np.isin(st,[0,3]))
        k2=_distance_rho(zz,ff,4,np.inf,chunk,meta)-_distance_rho(zz,ff,0,1,chunk,meta)
        return k1,k2
    kk=[k_for(s) for s in seeds]; k1=_interval([x[0] for x in kk]); k2=_interval([x[1] for x in kk])
    # mini-m has K1 -.082 and K2 +.114 (both outside ±.05), so F6 requires the same signs.
    f6={"K1":k1,"K2":k2,"mini_m_signs":{"K1":-1,"K2":1},
        "pass":bool(np.isfinite(k1["mean"]) and np.isfinite(k2["mean"]) and
                    k1["mean"] < 0 and k2["mean"] > 0)}
    # V2 stop rule uses seed 0 alone: M_SO,0 < .02, ΔM_0 <= 0 or rho_0 = ΔM_0 / M_SO,0 < .10.
    screen0 = None
    if z("e2e2",0) is not None and z("e2e1-ce",0) is not None:
        m_so0 = _m(z("e2e2",0)); dm0 = m_so0 - _m(z("e2e1-ce",0))
        rho0 = dm0 / m_so0 if m_so0 else np.nan
        screen0 = bool(m_so0 < .02 or dm0 <= 0 or not np.isfinite(rho0) or rho0 < .10)
    failed=[]
    if not m0["pass"]: failed.append("M0")
    for name,val in (("F1",f1),("F2",{"pass":f2_pass}),("F3",f3),("F4",f4),("F5",f5),("F6",f6)):
        if not val["pass"]: failed.append(name)
    result={"label":"QUALIFIED" if not failed else "NOT QUALIFIED","failed":failed,
            "V2_stop":screen0,"M0":m0,"F1":f1,"F2":{"per_seed":f2_vals,"pass":f2_pass},
            "F3":f3,"F4":f4,"F5":f5,"F6":f6,"pairs":pairs,"chunk":chunk}
    if out_json is None: out_json=root/"tbb_analysis.json"
    if out_md is None: out_md=root/"tbb_analysis.md"
    Path(out_json).write_text(json.dumps(result,indent=2,allow_nan=True)+"\n")
    lines=["# TBB micro16k validation", "", f"**{result['label']}**", "", f"Failed: {', '.join(failed) if failed else 'none'}", "", f"V2 stop rule: {screen0}", ""]
    for n in ("M0","F1","F2","F3","F4","F5","F6"): lines.append(f"- {n}: {json.dumps(result[n], allow_nan=True, sort_keys=True)}")
    Path(out_md).write_text("\n".join(lines)+"\n")
    return result
