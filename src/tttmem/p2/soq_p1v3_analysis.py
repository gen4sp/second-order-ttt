"""Registered SOQ-P1 v3 analysis (document bootstrap, 10k draws)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MARGIN = .20
CPLUS = .50
CHANCE = 1 / 16
N_BOOT = 10_000
ALPHA = .10


def _doc_mean(x, mask=None):
    x = np.asarray(x, dtype=float)
    if mask is None:
        return x.mean(axis=1)
    mask = np.asarray(mask, dtype=bool)
    n = mask.sum(axis=1)
    return np.divide((x * mask).sum(axis=1), n, out=np.full(len(x), np.nan), where=n > 0)


def bootstrap_lb(x, *, draws=N_BOOT, alpha=ALPHA, seed=0):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return float("nan")
    idx = np.random.default_rng(seed).integers(0, len(x), size=(int(draws), len(x)))
    return float(np.quantile(x[idx].mean(axis=1), alpha))


def interval(x, *, seed=0):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return {"mean": None, "lb90": None, "n_docs": 0}
    return {"mean": float(x.mean()), "lb90": bootstrap_lb(x, seed=seed), "n_docs": int(len(x))}


def _key(fwd, read, measure="em"):
    return f"echo_{fwd}_{read}_{measure}"


def _read(z, fwd, read, measure="em"):
    key = _key(fwd, read, measure)
    if key not in z and measure == "em":
        key = _key(fwd, read, "R")
    if key not in z and measure == "top1_16":
        key = _key(fwd, read, "first_top1_16")
    if key not in z: return None
    return np.asarray(z[key])


def descriptive(z):
    """Descriptive tables split by forward/read/stratum/segment."""
    av = np.asarray(z["echo_available"], dtype=bool)
    out = {}
    for fwd, read in (("N", "cross_verbatim"), ("N", "lookup_verbatim"),
                      ("A", "cross_verbatim"), ("P", "cross_paraphrase")):
        em = _read(z, fwd, read, "em")
        if em is None: continue
        rank = _read(z, fwd, read, "first_token_rank")
        nll = _read(z, fwd, read, "nll")
        seg = np.asarray(z[f"echo_{fwd}_{read}_segment"]) if f"echo_{fwd}_{read}_segment" in z else None
        out.setdefault(fwd, {})[read] = {}
        masks = {"pooled": np.ones_like(av), "copy_available": av, "copy_blocked": ~av}
        if seg is not None:
            masks.update({f"segment{int(s)}": seg == s for s in np.unique(seg)})
        for name, mask in masks.items():
            row = {"R": interval(_doc_mean(em, mask)), "value_nll": interval(_doc_mean(nll, mask))}
            if rank is not None: row["first_rank_16"] = interval(_doc_mean(rank, mask)); row["top1_16"] = interval(_doc_mean(rank == 1, mask))
            out[fwd][read][name] = row
    return out


def registered_rules(z, *, step=2400, seed=0):
    """Apply C+, V2' and the top1 qualifier to one endpoint npz or mapping."""
    lookup = _read(z, "N", "lookup_verbatim", "em")
    cross_n = _read(z, "N", "cross_verbatim", "em")
    cross_a = _read(z, "A", "cross_verbatim", "em")
    top = _read(z, "N", "cross_verbatim", "top1_16")
    av = np.asarray(z["echo_available"], dtype=bool)
    if lookup is None or cross_n is None or cross_a is None:
        return {"verdict": "NE", "reason": "missing v3 N/A read arrays", "step": int(step)}
    lookup_docs = _doc_mean(lookup)
    cplus = interval(lookup_docs, seed=seed)
    cplus["threshold"] = CPLUS; cplus["pass"] = bool(cplus["mean"] is not None and cplus["mean"] >= CPLUS)
    blocked_n = _doc_mean(cross_n, ~av); blocked_a = _doc_mean(cross_a, ~av)
    delta = blocked_n - blocked_a
    v2 = interval(delta, seed=seed)
    v2.update({"threshold": MARGIN, "point": float(np.nanmean(delta)), "pass": bool(np.nanmean(delta) >= MARGIN and v2["lb90"] > 0)})
    qual = None
    if top is not None:
        qual = interval(_doc_mean(top, ~av), seed=seed)
        qual.update({"threshold": CHANCE, "pass": bool(qual["lb90"] > CHANCE)})
    if not cplus["pass"]:
        verdict = "LOOKUP-NOT-LEARNED"
    elif not v2["pass"]:
        verdict = "NOT-LEARNABLE"
    elif qual is not None and not qual["pass"]:
        verdict = "PASS-UNQUALIFIED"
    else:
        verdict = "PASS"
    return {"step": int(step), "C_plus": cplus, "V2_prime": v2, "top1_16_qualifier": qual,
            "verdict": verdict, "descriptive": descriptive(z)}


def d1(so, fo, *, seed=0):
    """D1 ADDRESS-BARRIER and seed-0 futility for an SO/FO endpoint pair."""
    def r(z, fwd, read, mask):
        av = np.asarray(z["echo_available"], dtype=bool)
        em = _read(z, fwd, read, "em")
        return _doc_mean(em, av if mask == "available" else ~av)
    so_av = r(so, "N", "cross_verbatim", "available"); so_bl = r(so, "N", "cross_verbatim", "blocked")
    fo_av = r(fo, "N", "cross_verbatim", "available"); fo_bl = r(fo, "N", "cross_verbatim", "blocked")
    address = float(np.nanmean(so_bl) - np.nanmean(fo_bl))
    delta_seed0 = float((np.nanmean(so_av) - np.nanmean(fo_av)) - (np.nanmean(so_bl) - np.nanmean(fo_bl)))
    return {"address_barrier_margin": address, "address_barrier": bool(np.nanmean(fo_bl) < np.nanmean(so_bl) - .10),
            "delta_seed0": delta_seed0, "futility": bool(delta_seed0 < .05),
            "node2": bool(delta_seed0 >= .05), "thresholds": {"address": .10, "futility": .05}}


def _load_steps(dirs):
    out = {}
    for root in map(Path, dirs):
        for p in root.glob("eval_step*.npz"):
            with np.load(p, allow_pickle=False) as z:
                if "echo_version" in z and int(np.asarray(z["echo_version"])) == 3:
                    out[int(p.stem.split("step")[-1])] = {k: z[k] for k in z.files}
    return out


def analyse(run_dirs):
    steps = _load_steps(run_dirs)
    if not steps: raise FileNotFoundError("no v3 eval npz files")
    end = max(steps); rules = registered_rules(steps[end], step=end)
    return {"version": 3, "endpoint": end, "steps": {str(k): {
        "rules": registered_rules(v, step=k), "descriptive": descriptive(v)} for k, v in sorted(steps.items())},
            "endpoint_rules": rules, "registered": {"C_plus": CPLUS, "V2_prime": MARGIN,
            "top1_chance": CHANCE, "n_boot": N_BOOT, "document_bootstrap": True}}


def markdown(res):
    r = res["endpoint_rules"]; c, v, q = r["C_plus"], r["V2_prime"], r["top1_16_qualifier"]
    lines = ["# SOQ P1 v3 analysis", "", f"Endpoint step {res['endpoint']}: **{r['verdict']}**.", "",
             f"C+ lookup R = {c.get('mean')} (threshold {CPLUS}); V2' margin = {v.get('point')} "
             f"(90% LB {v.get('lb90')}, threshold {MARGIN}); top1₁₆ LB = {None if q is None else q.get('lb90')} "
             f"(chance {CHANCE:.5f}).", "", "| forward | read | stratum | R | value NLL | top1₁₆ |", "|---|---|---|---:|---:|---:|"]
    for fwd, reads in r.get("descriptive", {}).items():
        for read, rows in reads.items():
            for stratum, x in rows.items():
                lines.append(f"| {fwd} | {read} | {stratum} | {x['R'].get('mean')} | {x['value_nll'].get('mean')} | {x.get('top1_16', {}).get('mean')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("run_dirs", nargs="+"); ap.add_argument("--out"); ap.add_argument("--md")
    a = ap.parse_args(argv); res = analyse(a.run_dirs); text = json.dumps(res, indent=1, sort_keys=True)
    if a.out: Path(a.out).write_text(text + "\n")
    if a.md: Path(a.md).write_text(markdown(res))
    print(markdown(res)); print("P2_ECHO3_ANALYSIS " + json.dumps(res["endpoint_rules"], sort_keys=True)); return res


if __name__ == "__main__": main()


# Short public aliases for notebook/CI callers.
v3 = registered_rules
analyze = registered_rules
d1_rule = d1
