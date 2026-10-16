"""Local (CPU, numpy) analysis of SOQ P1 run directories: echo recall per stratum and V2.

Usage::

    python -m tttmem.p2.soq_p1_analysis RUN_DIR [RUN_DIR ...] [--out a.json] [--md a.md]

RUN_DIR is a trainer out_dir (``eval_step*.npz`` with ``echo_*`` keys, ``steps.jsonl``,
``events.jsonl``); a continuation directory may follow its parent, and the steps are merged.

Registered reads (fixed before any P1 outcome):

- recall = greedy exact match of all value tokens at the read, per stratum: copy-available /
  copy-blocked (barrier query), no-binding, no-barrier, query-entity counterfactual;
  source NLL per binding stratum (value tokens at the binding);
- **V2 (learnable)** at the endpoint: per document d = mean copy-blocked recall - mean
  no-binding recall; PASS iff mean(d) >= .20 and its one-sided 90 % lower bound > 0.  One seed
  (the pilot) -> the bound is a document-level bootstrap (10 000 draws, seed 0); with several
  seeds a seed-level t bound applies (not computed here);
- horizon slope: copy-blocked recall(2400) - recall(1800), reported when both exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

V2_MARGIN = 0.20
ALPHA = 0.10          # one-sided 90 % bounds
N_BOOT = 10_000
BOOT_SEED = 0


def _doc_mean(x: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if m is None:
        return x.mean(1)
    m = np.asarray(m, dtype=bool)
    cnt = m.sum(1)
    return np.divide((x * m).sum(1), cnt, out=np.full(len(x), np.nan), where=cnt > 0)


def doc_reads(z) -> dict:
    """Per-document means for every stratum (recall and NLL) from one eval npz."""
    av = np.asarray(z["echo_available"], dtype=bool)
    em = lambda r: np.asarray(z[f"echo_{r}_em"], dtype=np.float64)  # noqa: E731
    nl = lambda r: np.asarray(z[f"echo_{r}_nll"], dtype=np.float64)  # noqa: E731
    names = set(z.files) if hasattr(z, "files") else set(z)
    return {
        "recall": {"copy_available": _doc_mean(em("query"), av),
                   "copy_blocked": _doc_mean(em("query"), ~av),
                   "no_binding": _doc_mean(em("no_binding")),
                   "no_barrier": _doc_mean(em("no_barrier")),
                   "query_entity": _doc_mean(em("query_entity"))},
        "query_nll": {"copy_available": _doc_mean(nl("query"), av),
                      "copy_blocked": _doc_mean(nl("query"), ~av),
                      "no_binding": _doc_mean(nl("no_binding")),
                      "no_barrier": _doc_mean(nl("no_barrier")),
                      "query_entity": _doc_mean(nl("query_entity"))},
        "source_nll": {"copy_available": _doc_mean(nl("source"), av),
                       "copy_blocked": _doc_mean(nl("source"), ~av)},
        "first_nll": {r: _doc_mean(np.asarray(z[f"echo_{r}_first_nll"])) for r in READS
                      if f"echo_{r}_first_nll" in names},
        "rest_nll": {r: _doc_mean(np.asarray(z[f"echo_{r}_rest_nll"])) for r in READS
                     if f"echo_{r}_rest_nll" in names},
        "segment": _segment_reads(z),
    }


READS = ("source", "query", "no_binding", "no_barrier", "query_entity")


def _segment_reads(z) -> dict:
    names = set(z.files) if hasattr(z, "files") else set(z)
    if "echo_query_segment" not in names:
        return {}
    out = {}
    seg = np.asarray(z["echo_query_segment"], dtype=bool)
    em = np.asarray(z["echo_query_em"], dtype=bool)
    nl = np.asarray(z["echo_query_nll"], dtype=np.float64)
    for label, mask in (("segment3", seg), ("segment4", ~seg)):
        out[label] = {"recall": _doc_mean(em, mask), "query_nll": _doc_mean(nl, mask)}
    return out


def boot_means(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Bootstrap distribution of the mean over documents (vectorised, NaN documents dropped)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.full(n_boot, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(1)


def summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return {"mean": None, "lo90": None, "hi90": None, "n": 0}
    b = boot_means(x)
    return {"mean": float(x[ok].mean()), "lo90": float(np.quantile(b, ALPHA / 2)),
            "hi90": float(np.quantile(b, 1 - ALPHA / 2)), "n": int(ok.sum())}


def v2(reads: dict) -> dict:
    d = reads["recall"]["copy_blocked"] - reads["recall"]["no_binding"]
    d = d[np.isfinite(d)]
    if not len(d):
        return {"verdict": "NE", "reason": "no copy-blocked documents"}
    lb = float(np.quantile(boot_means(d), ALPHA))
    mean = float(d.mean())
    return {"mean": mean, "lb90_one_sided": lb, "n_docs": int(len(d)), "margin": V2_MARGIN,
            "verdict": "PASS" if mean >= V2_MARGIN and lb > 0 else "FAIL"}


def analyse_step(path: Path) -> dict | None:
    with np.load(path, allow_pickle=False) as z:
        if "echo_available" not in z.files:
            return None
        reads = doc_reads(z)
        diagnostics = {}
        for label in ("train", "verbatim"):
            key = f"echo_{label}_available"
            if key in z.files:
                # Store the same compact summaries for train-template and verbatim reads.
                av = np.asarray(z[key], dtype=bool)
                em = lambda r: np.asarray(z[f"echo_{label}_{r}_em"], dtype=np.float64)
                nl = lambda r: np.asarray(z[f"echo_{label}_{r}_nll"], dtype=np.float64)
                diagnostics[label] = {"recall": {"copy_available": _doc_mean(em("query"), av),
                    "copy_blocked": _doc_mean(em("query"), ~av),
                    "no_binding": _doc_mean(em("no_binding"))},
                    "query_nll": {"copy_available": _doc_mean(nl("query"), av),
                    "copy_blocked": _doc_mean(nl("query"), ~av),
                    "no_binding": _doc_mean(nl("no_binding"))},
                    "first_nll": {r: _doc_mean(np.asarray(z[f"echo_{label}_{r}_first_nll"]))
                                  for r in READS if f"echo_{label}_{r}_first_nll" in z.files},
                    "rest_nll": {r: _doc_mean(np.asarray(z[f"echo_{label}_{r}_rest_nll"]))
                                 for r in READS if f"echo_{label}_{r}_rest_nll" in z.files}}
        n = int(len(z["echo_available"]))
    def summarize_tree(x):
        return {k: summarize_tree(v) for k, v in x.items()} if isinstance(x, dict) else summary(x)
    return {"n_docs": n,
            **{kind: summarize_tree(by) for kind, by in reads.items()},
            "v2": v2(reads), "diagnostics": diagnostics}


def _jsonl(path: Path, prefix: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(prefix + " "):
            out.append(json.loads(line[len(prefix) + 1:]))
    return out


def analyse(run_dirs: list[Path]) -> dict:
    steps, train, exposure, provenance = {}, {}, None, []
    for root in map(Path, run_dirs):
        for path in sorted(root.glob("eval_step[0-9][0-9][0-9][0-9].npz")):
            res = analyse_step(path)
            if res is not None:
                steps[int(path.stem.split("step")[-1])] = {**res, "dir": str(root)}
        for rec in _jsonl(root / "steps.jsonl", "P2_STEP"):
            train[int(rec["step"])] = {k: rec.get(k) for k in ("loss", "lr", "grad_norm", "echo_rows")}
        exp = _jsonl(root / "events.jsonl", "P2_ECHO_EXPOSURE")
        exposure = exp[0] if exp and exposure is None else exposure
        if (root / "continuation.json").exists():
            cont = json.loads((root / "continuation.json").read_text())
            provenance.append({k: cont[k] for k in ("parent_state_sha256", "start", "horizon",
                                                    "lr_peak") if k in cont})
    if not steps:
        raise FileNotFoundError("no eval_step*.npz with echo_* keys in " + ", ".join(map(str, run_dirs)))
    end = max(steps)
    out = {"endpoint": end, "steps": {str(k): steps[k] for k in sorted(steps)},
           "v2_endpoint": steps[end]["v2"], "exposure": exposure, "continuation": provenance,
           "train_last": train[max(train)] if train else None,
           "registered": {"v2_margin": V2_MARGIN, "alpha_one_sided": ALPHA, "n_boot": N_BOOT,
                          "boot_seed": BOOT_SEED, "bound": "document-level bootstrap"}}
    if 1800 in steps and 2400 in steps:
        a = steps[1800]["recall"]["copy_blocked"]["mean"]
        b = steps[2400]["recall"]["copy_blocked"]["mean"]
        out["slope_1800_2400_copy_blocked"] = None if a is None or b is None else b - a
    return out


def _fmt(s: dict) -> str:
    if s.get("mean") is None:
        return "NE"
    return f"{s['mean']:.3f} [{s['lo90']:.3f}, {s['hi90']:.3f}]"


def markdown(res: dict) -> str:
    lines = ["# SOQ P1 echo reads", "",
             f"Endpoint step {res['endpoint']}; V2 = **{res['v2_endpoint']['verdict']}** "
             f"(mean {res['v2_endpoint'].get('mean', float('nan')):.3f}, one-sided 90 % LB "
             f"{res['v2_endpoint'].get('lb90_one_sided', float('nan')):.3f}, margin {V2_MARGIN}).", "",
             "| step | n | copy-avail | copy-blocked | no-binding | no-barrier | query-entity | "
             "src NLL avail | src NLL blocked |", "|" + "---|" * 9]
    for k, s in res["steps"].items():
        r, sn = s["recall"], s["source_nll"]
        lines.append(f"| {k} | {s['n_docs']} | " + " | ".join(_fmt(r[x]) for x in (
            "copy_available", "copy_blocked", "no_binding", "no_barrier", "query_entity"))
            + f" | {_fmt(sn['copy_available'])} | {_fmt(sn['copy_blocked'])} |")
    lines += ["", "Recall = greedy exact match of all value tokens; brackets = document-level "
              "bootstrap 90 % interval."]
    if any(s.get("diagnostics") for s in res["steps"].values()):
        lines += ["", "## Diagnostic reads", "", "| step | read | copy-blocked recall | no-binding recall | first-token NLL | rest NLL |", "|---|---|---:|---:|---:|---:|"]
        for k, s in res["steps"].items():
            for label, d in s.get("diagnostics", {}).items():
                rr, nn = d["recall"], d["query_nll"]
                f = d.get("first_nll", {}).get("query")
                q = d.get("rest_nll", {}).get("query")
                lines.append(f"| {k} | {label} | {_fmt(summary(rr['copy_blocked']))} | {_fmt(summary(rr['no_binding']))} | {_fmt(summary(f)) if f is not None else 'NE'} | {_fmt(summary(q)) if q is not None else 'NE'} |")
    if "slope_1800_2400_copy_blocked" in res:
        lines.append(f"Copy-blocked slope 1800 -> 2400: {res['slope_1800_2400_copy_blocked']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    res = analyse([Path(p) for p in args.run_dirs])
    text = json.dumps(res, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.md:
        Path(args.md).write_text(markdown(res))
    print(markdown(res))
    print("SOQ_P1_V2 " + json.dumps(res["v2_endpoint"], sort_keys=True))
    return res


if __name__ == "__main__":
    main()
