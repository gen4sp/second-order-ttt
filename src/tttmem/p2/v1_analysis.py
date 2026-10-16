"""CPU-only analysis and fixed V1 gate for ``soq_v1`` outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T90_DF2 = 1.886


def _bootstrap(x: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    means = np.mean(x[rng.integers(0, len(x), size=(n, len(x)))], axis=1)
    return float(np.quantile(means, .05)), float(np.quantile(means, .95))


def _item(path: Path, rng: np.random.Generator) -> dict:
    with np.load(path, allow_pickle=False) as z:
        src = np.asarray(z["source_nll"], float)
        masked = np.asarray(z["source_nll_masked"], float)
        avail = np.asarray(z["available"], bool)
        residual = np.asarray(z["source_residual"], float)
        exact = np.asarray(z["query_exact"], float)
        query_nll = np.asarray(z["query_nll"], float)
        no_barrier_nll = np.asarray(z["no_barrier_nll"], float)
        no_binding_nll = np.asarray(z["no_binding_nll"], float)
        entity_nll = np.asarray(z["query_entity_nll"], float)
        no_barrier = np.asarray(z["no_barrier_exact"], float)
        no_binding = np.asarray(z["no_binding_exact"], float)
        entity = np.asarray(z["query_entity_exact"], float)
        blocked = ~avail
        d = (np.nanmean(np.where(blocked, src, np.nan), axis=1)
             - np.nanmean(np.where(avail, src, np.nan), axis=1))
        md = (np.nanmean(np.where(avail, masked, np.nan), axis=1)
              - np.nanmean(np.where(blocked, src, np.nan), axis=1))
        seed = int(np.asarray(z.get("seed", -1))) if "seed" in z else -1
        arm = str(np.asarray(z.get("arm", path.parent.name)))
        return {
            "path": str(path), "arm": arm, "seed": seed, "n_docs": int(len(src)),
            "source_delta_blocked_minus_available": float(np.nanmean(d)),
            "source_delta_bootstrap90": _bootstrap(d, rng),
            "masked_available_minus_blocked": float(np.nanmean(md)),
            "masked_bootstrap90": _bootstrap(md, rng),
            "residual_available": float(np.nanmean(np.where(avail, residual, np.nan))),
            "residual_blocked": float(np.nanmean(np.where(blocked, residual, np.nan))),
            "query_exact_available": float(np.nanmean(np.where(avail, exact, np.nan))),
            "query_exact_blocked": float(np.nanmean(np.where(blocked, exact, np.nan))),
            "query_nll_available": float(np.nanmean(np.where(avail, query_nll, np.nan))),
            "query_nll_blocked": float(np.nanmean(np.where(blocked, query_nll, np.nan))),
            "no_barrier_nll": float(np.nanmean(no_barrier_nll)),
            "no_binding_nll": float(np.nanmean(no_binding_nll)),
            "query_entity_nll": float(np.nanmean(entity_nll)),
            "no_barrier_exact": float(np.nanmean(no_barrier)),
            "no_binding_exact": float(np.nanmean(no_binding)),
            "query_entity_exact": float(np.nanmean(entity)),
            "doc_delta": d.tolist(), "doc_masked_delta": md.tolist(),
        }


def analyze(root: str | Path, out: str | Path | None = None) -> dict:
    root = Path(root)
    paths = sorted(root.glob("**/v1_item.npz"))
    if not paths:
        raise FileNotFoundError(f"no v1_item.npz below {root}")
    rng = np.random.default_rng(20261010)
    items = [_item(p, rng) for p in paths]
    arms = {}
    for arm in sorted({x["arm"] for x in items}):
        rows = [x for x in items if x["arm"] == arm]
        d = np.asarray([x["source_delta_blocked_minus_available"] for x in rows], float)
        md = np.asarray([x["masked_available_minus_blocked"] for x in rows], float)
        lb = float(np.mean(d) - T90_DF2 * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        arms[arm] = {"seeds": len(rows), "source_seed_mean": float(np.mean(d)),
                     "source_seed_lb90": lb, "masked_seed_mean": float(np.mean(md)),
                     "source_gate": bool(np.isfinite(lb) and np.mean(d) >= .20 and lb > 0),
                     "mask_gate": bool(np.isfinite(md).all() and np.mean(np.abs(md)) <= .05),
                     "items": rows}
        arms[arm]["verdict"] = "PASS" if arms[arm]["source_gate"] and arms[arm]["mask_gate"] else "FAIL"
    overall = "PASS" if arms and all(x["verdict"] == "PASS" for x in arms.values()) else (
        "PARTIAL" if any(x["verdict"] == "PASS" for x in arms.values()) else "FAIL")
    generator = {}
    if (root / "v1_summary.json").exists():   # which echo generator produced the reads
        summ = json.loads((root / "v1_summary.json").read_text())
        generator = {k: summ.get(k) for k in ("manifest_sha256", "tables_sha256")}
    result = {"root": str(root), "n_items": len(items), "arms": arms, "verdict": overall,
              "generator": generator,
              "thresholds": {"source_mean_min": .20, "source_lb90_min": 0.0,
                             "masked_abs_mean_max": .05, "t90_df2": T90_DF2}}
    if out is None:
        out = root / "v1_analysis.json"
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    md = out.with_suffix(".md")
    lines = ["# V1 validity analysis", "", f"Verdict: **{overall}**", "",
             f"Generator: {generator or 'unknown (no v1_summary.json)'}", "",
             "Thresholds: source blocked−available ≥ 0.20 nats and one-sided 90% seed LB > 0; "
             "masked available−blocked absolute mean ≤ 0.05 nats.", ""]
    for arm, rec in arms.items():
        lines.append(f"- **{arm}**: {rec['verdict']}; source Δ={rec['source_seed_mean']:.4f} "
                     f"(LB90={rec['source_seed_lb90']:.4f}), masked Δ={rec['masked_seed_mean']:.4f}; "
                     f"residual available={np.mean([x['residual_available'] for x in rec['items']]):.4f}; "
                     f"recall controls (no-barrier/no-binding/entity)="
                     f"{np.mean([x['no_barrier_exact'] for x in rec['items']]):.4f}/"
                     f"{np.mean([x['no_binding_exact'] for x in rec['items']]):.4f}/"
                     f"{np.mean([x['query_entity_exact'] for x in rec['items']]):.4f}; "
                     f"control NLL (no-barrier/no-binding/entity)="
                     f"{np.mean([x['no_barrier_nll'] for x in rec['items']]):.4f}/"
                     f"{np.mean([x['no_binding_nll'] for x in rec['items']]):.4f}/"
                     f"{np.mean([x['query_entity_nll'] for x in rec['items']]):.4f}")
    md.write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    print(json.dumps(analyze(args.root, args.out), indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
