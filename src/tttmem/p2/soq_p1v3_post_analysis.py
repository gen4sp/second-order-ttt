"""Analysis CLI for SOQ-P1 v3 post-hoc echo arrays.

All uncertainty intervals resample documents (not individual slots), with the
registered 10,000-draw one-sided 90% bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .soq_p1v3_analysis import N_BOOT, ALPHA, bootstrap_lb, registered_rules

CHANCE_8 = 1 / 8


def _doc_mean(x, mask=None):
    x = np.asarray(x, dtype=float)
    if mask is None:
        return np.nanmean(x, axis=1)
    mask = np.asarray(mask, dtype=bool)
    n = mask.sum(axis=1)
    return np.divide(np.nansum(np.where(mask, x, np.nan), axis=1), n,
                     out=np.full(len(x), np.nan), where=n > 0)


def _interval(x, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x): return {"mean": None, "lb90": None, "n_docs": 0}
    return {"mean": float(x.mean()), "lb90": float(bootstrap_lb(x, draws=N_BOOT,
                                                                   alpha=ALPHA, seed=seed)),
            "n_docs": int(len(x))}


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _load_inputs(items):
    paths = []
    for item in items:
        p = Path(item)
        if p.is_dir(): paths.extend(sorted(p.glob("post_step*.npz")))
        elif p.is_file(): paths.append(p)
        else: raise FileNotFoundError(p)
    out = {}
    for p in paths:
        z = _npz(p)
        step = int(np.asarray(z.get("checkpoint_step", int(p.stem.split("step")[-1]))))
        out[step] = z
    if not out: raise FileNotFoundError("no post_step*.npz files")
    return out


def _key(set_name, fwd, read, suffix):
    return f"echo_{set_name}_{fwd}_{read}_{suffix}"


def _get(z, set_name, fwd, read, suffix):
    names = [_key(set_name, fwd, read, suffix)]
    # Accept endpoint v3 arrays as a convenience for the v3x label checks.
    if set_name in (None, ""):
        names.insert(0, f"echo_{fwd}_{read}_{suffix}")
    aliases = {"R": "em", "top1_8": "within_stratum_top1_8",
               "first_rank_16": "first_token_rank", "value_nll": "nll"}
    if suffix in aliases:
        names += [_key(set_name, fwd, read, aliases[suffix])]
        if set_name in (None, ""): names += [f"echo_{fwd}_{read}_{aliases[suffix]}"]
    for name in names:
        if name in z: return np.asarray(z[name])
    if suffix == "top1_8":
        cand = _get(z, set_name, fwd, read, "cand_logp")
        if cand is not None:
            av = _available(z, set_name)
            own = np.take_along_axis(cand, np.arange(cand.shape[1])[None, :, None], axis=2)
            same = av[:, :, None] == av[:, None, :]
            return 1 + ((cand > own) & same).sum(2) == 1
    return None


def _available(z, set_name):
    k = f"echo_{set_name}_available"
    if k in z: return np.asarray(z[k], dtype=bool)
    if "echo_available" in z: return np.asarray(z["echo_available"], dtype=bool)
    raise KeyError(k)


def _mask(z, set_name, fwd, read, kind):
    av = _available(z, set_name)
    if kind == "pooled": return np.ones_like(av, dtype=bool)
    if kind == "copy_available": return av
    if kind == "copy_blocked": return ~av
    if kind.startswith("segment"):
        a = _get(z, set_name, fwd, read, "segment")
        return None if a is None else a == int(kind[7:])
    if kind == "order0":
        a = _get(z, set_name, fwd, read, "read_order")
        return None if a is None else a == 0
    return None


def _stat(z, set_name, fwd, read, suffix, mask_kind="pooled", seed=0):
    x = _get(z, set_name, fwd, read, suffix)
    if x is None: return None
    mask = _mask(z, set_name, fwd, read, mask_kind)
    if mask is None: return None
    return _interval(_doc_mean(x, mask), seed=seed)


def _contrast(z, set_name, fwd_a, fwd_b, read, suffix, mask_kind="pooled", seed=0):
    a = _get(z, set_name, fwd_a, read, suffix); b = _get(z, set_name, fwd_b, read, suffix)
    mask = _mask(z, set_name, fwd_a, read, mask_kind)
    if a is None or b is None or mask is None: return None
    return _interval(_doc_mean(a, mask) - _doc_mean(b, mask), seed=seed)


def _safe_cell(z, set_name, fwd, read, suffix, kind, seed):
    v = _stat(z, set_name, fwd, read, suffix, kind, seed)
    return v


def analyse_post(z, *, seed=0):
    """Return registered post-hoc contrasts and descriptive split tables."""
    out = {"version": 3, "checkpoint_step": int(np.asarray(z.get("checkpoint_step", -1))),
           "contrasts": {}, "within_stratum": {}, "order0": {}, "nll_split": {}, "P_vs_PA": {}}
    if "echo_test_available" in z:
        norm = {"echo_available": z["echo_test_available"]}
        norm.update({k.replace("echo_test_", "echo_", 1): v for k, v in z.items()
                     if k.startswith("echo_test_")})
        try: out["registered_rules"] = registered_rules(norm, step=out["checkpoint_step"], seed=seed)
        except (KeyError, TypeError, ValueError): pass
    def add(name, value):
        if value is not None: out["contrasts"][name] = value
    # (a) copy-blocked cross R contrasts.
    add("test_cross_R_N_minus_A", _contrast(z, "test", "N", "A", "cross_verbatim", "R", "copy_blocked", seed))
    add("test_cross_R_N_minus_Z", _contrast(z, "test", "N", "Z", "cross_verbatim", "R", "copy_blocked", seed + 1))
    add("train_cross_R_T_minus_TA", _contrast(z, "train", "T", "T-A", "cross_verbatim", "R", "copy_blocked", seed + 2))
    add("train_cross_R_N_minus_A", _contrast(z, "train", "N", "A", "cross_verbatim", "R", "copy_blocked", seed + 3))
    # (b) within-stratum top1_8, chance and matching-floor contrasts.
    for set_name, fwd, floor in (("test", "N", "A"), ("test", "N", "Z"),
                                 ("train", "T", "T-A"), ("train", "N", "A"),
                                 ("train", "N", "Z")):
        key = f"{set_name}_{fwd}_cross_top1_8_blocked"
        val = _stat(z, set_name, fwd, "cross_verbatim", "top1_8", "copy_blocked", seed + 10)
        if val is not None:
            out["within_stratum"][key] = {"estimate": val, "chance": CHANCE_8,
                "above_chance_lb": None if val["lb90"] is None else val["lb90"] - CHANCE_8,
                "contrast_floor": _contrast(z, set_name, fwd, floor, "cross_verbatim", "top1_8",
                                             "copy_blocked", seed + 11)}
    # (c) order-0 cross read top1 metrics.
    for set_name, fwds in (("test", ("N", "A", "Z")), ("train", ("N", "A", "Z", "T", "T-A"))):
        for fwd in fwds:
            for suffix in ("top1_16", "top1_8"):
                v = _stat(z, set_name, fwd, "cross_verbatim", suffix, "order0", seed + 20)
                if v is not None: out["order0"][f"{set_name}_{fwd}_{suffix}"] = v
    # (d) pooled lookup top1 contrasts.
    for b in ("A", "Z"):
        v = _contrast(z, "test", "N", b, "lookup_verbatim", "top1_16", "pooled", seed + 30)
        if v is not None: out["contrasts"][f"test_lookup_top1_16_N_minus_{b}"] = v
    # (e) first-vs-token2-3 NLL split, cross and lookup.
    for read in ("cross_verbatim", "lookup_verbatim"):
        for fwd in ("N", "A", "Z"):
            first = _stat(z, "test", fwd, read, "first_nll", "pooled", seed + 40)
            tail = _stat(z, "test", fwd, read, "tokens23_nll", "pooled", seed + 41)
            if first is not None or tail is not None:
                out["nll_split"][f"test_{fwd}_{read}"] = {"first_nll": first, "tokens23_nll": tail}
    # (f) P versus P-A on the paraphrase cross reads.
    for suffix in ("R", "top1_16", "top1_8", "first_nll", "tokens23_nll", "value_nll"):
        v = _contrast(z, "test", "P", "P-A", "cross_paraphrase", suffix, "pooled", seed + 50)
        if v is not None: out["P_vs_PA"][suffix] = v
    return out


def v3x_labels(end, start=None, parent=None, *, seed=0):
    """Apply the registered labels and, when available, the start-read check.

    LATE-SIGNAL (within-stratum arm): N's copy-blocked top1_8 LB > 1/8 and its point estimate
    exceeds A's (registered wording: "exceeds the same measure in A")."""
    end = _npz(end) if isinstance(end, (str, Path)) else end
    start = _npz(start) if isinstance(start, (str, Path)) else start
    parent = _npz(parent) if isinstance(parent, (str, Path)) else parent
    av = _available(end, "")
    n = _get(end, "", "N", "cross_verbatim", "R")
    a = _get(end, "", "A", "cross_verbatim", "R")
    top16 = _get(end, "", "N", "cross_verbatim", "top1_16")
    top8n = _get(end, "", "N", "cross_verbatim", "top1_8")
    top8a = _get(end, "", "A", "cross_verbatim", "top1_8")
    blocked = ~av
    v2 = None; qual = None; signal8 = None
    if n is not None and a is not None:
        v2 = _interval(_doc_mean(n, blocked) - _doc_mean(a, blocked), seed=seed)
        v2["point"] = float(np.nanmean(_doc_mean(n, blocked) - _doc_mean(a, blocked)))
    if top16 is not None: qual = _interval(_doc_mean(top16, blocked), seed=seed + 1)
    if top8n is not None and top8a is not None:
        nn = _doc_mean(top8n, blocked); aa = _doc_mean(top8a, blocked)
        signal8 = {"N": _interval(nn, seed=seed + 2), "N_minus_A": _interval(nn - aa, seed=seed + 3)}
    late_storage = bool(v2 and v2["point"] >= .20 and v2["lb90"] > 0 and qual and qual["lb90"] > 1/16)
    late_signal = bool(not late_storage and ((signal8 and signal8["N"]["lb90"] > CHANCE_8 and
                                               signal8["N_minus_A"]["mean"] > 0) or
                                              (v2 and v2["point"] >= .05 and v2["lb90"] > 0)))
    result = {"label": "LATE-STORAGE" if late_storage else ("LATE-SIGNAL" if late_signal else "NO-LATE-STORAGE"),
              "V2_prime": v2, "qualifier_top1_16": qual, "within_stratum": signal8}
    if start is not None and parent is not None:
        checks = {}
        for suffix, tol in (("R", .01), ("top1_16", .01), ("value_nll", .02)):
            xs = _get(start, "", "N", "cross_verbatim", suffix)
            xp = _get(parent, "", "N", "cross_verbatim", suffix)
            if xs is None or xp is None: continue
            ms = _doc_mean(xs, ~_available(start, "")); mp = _doc_mean(xp, ~_available(parent, ""))
            delta = float(abs(np.nanmean(ms) - np.nanmean(mp)))
            checks[suffix] = {"delta": delta, "tol": tol, "pass": delta <= tol}
        result["start_reproduction"] = {"checks": checks, "pass": bool(checks) and all(x["pass"] for x in checks.values())}
    else:
        result["start_reproduction"] = {"available": False}
    return result


def analyse(inputs, *, v3x_start=None, v3x_end=None, parent=None):
    steps = _load_inputs(inputs) if not isinstance(inputs, dict) else inputs
    out = {"version": 3, "document_bootstrap": {"draws": N_BOOT, "one_sided_lb": .90},
           "checkpoints": {str(step): analyse_post(z, seed=step) for step, z in sorted(steps.items())}}
    if v3x_end or v3x_start:
        end = v3x_end or (max(steps) if steps else None)
        endz = _npz(end) if end else None
        out["v3x"] = v3x_labels(endz, v3x_start, parent) if endz is not None else {"available": False}
    return out


def markdown(res):
    lines = ["# SOQ P1 v3 post-hoc analysis", "",
             f"Document bootstrap: {N_BOOT} draws, one-sided 90% lower bound.", ""]
    for step, chk in res.get("checkpoints", {}).items():
        lines += [f"## Step {step}", ""]
        for name, value in chk.get("contrasts", {}).items():
            lines.append(f"- `{name}`: mean={value.get('mean')} LB90={value.get('lb90')}")
        if chk.get("v3x"): lines.append(f"- v3x label: **{chk['v3x']['label']}**")
        lines.append("")
    if "v3x" in res: lines += [f"v3x label: **{res['v3x'].get('label')}**.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="post_out_dir or post_step*.npz")
    ap.add_argument("--v3x-start")
    ap.add_argument("--v3x-end")
    ap.add_argument("--parent", help="parent eval_step2400.npz for the start reproduction check")
    ap.add_argument("--out"); ap.add_argument("--md")
    a = ap.parse_args(argv)
    res = analyse(a.inputs, v3x_start=a.v3x_start, v3x_end=a.v3x_end, parent=a.parent)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out: Path(a.out).write_text(text + "\n")
    if a.md: Path(a.md).write_text(markdown(res) + "\n")
    print(markdown(res)); print("P2_ECHO3_POST_ANALYSIS " + json.dumps(res, sort_keys=True))
    return res


if __name__ == "__main__": main()


# Public spelling retained for callers that use American English.
analyze = analyse
