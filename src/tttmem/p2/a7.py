"""P2-1/A7: D split by copy availability at write. NumPy only.

Inputs are the saved evaluations of each arm (``eval_meta.npz``, ``eval_step{step}.npz``), the
frozen endpoint N (``decisions/sizing.json``) and the evaluation token rows. Per-probe source
positions are recomputed from the token ids with the ``find_probes`` n-gram rule and checked
against the saved probe annotations and focal sets before anything is computed.

- P7.1  K_i = D_E2c,i - D_E2b,i, D_s,i = mean over stratum-s probes of (NLL_FO-CE - NLL_SO);
        one-sided 95 % cluster-bootstrap lower bound > 0 (confirmatory, step 400, first N).
- P7.1s the same with E2c restricted to probes whose two sources both lie in S2; mean ``dist``
        per stratum.
- P7.2  K'_i = D_pred,i - D_unpred,i, "pred" = write-time NLL of the probe's last source in
        SO's step-0 adapt evaluation below the pool-wide step-0 median.
- P7.3  the P7.1 contrast for C = NLL_FO-CE - NLL_rule (lin; AICE reported alike).
- P7.4  write-time cotangent norm at the last source, FO-CE / SO, E2c vs E2b (descriptive);
        every contrast at the trajectory steps 100 and 200 (descriptive).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .decide import _destination, _ids, _load, _sha1, _write
from .probe import (BARRIERS, PROBE_START, STRATA, _gram_ids, boot_mean, boundary_valid,
                    bootstrap_weights, one_sided, segment_starts)

E2C, E2B = STRATA.index("E2c"), STRATA.index("E2b")
TRAJECTORY_STEPS = (100, 200)


def probe_sources(doc: np.ndarray, C: int, bos: int, pos: np.ndarray, n: int = 4) -> list[np.ndarray]:
    """Source positions (ascending) of each probe position in ``pos`` (``find_probes`` rule)."""
    x, y = np.asarray(doc[:-1], dtype=np.int64), np.asarray(doc[1:], dtype=np.int64)
    T = len(x)
    t = np.arange(T)
    starts = segment_starts(T // C, BARRIERS) * C
    seg_start = starts[np.searchsorted(starts, t, side="right") - 1]
    ok = (y != bos) & boundary_valid(T, C, BARRIERS)
    key = np.where(ok & (t - (n - 2) >= seg_start), _gram_ids(x, y, n), -1)
    src = (t < PROBE_START * C) & (key >= 0)
    return [t[src & (key == key[p])] for p in pos]


def _stratum(s: np.ndarray, C: int) -> int:
    if len(s) == 1:
        return 0
    if len(s) == 2:
        c0, c1 = s // C
        if c0 == c1:
            return 1
        seg = np.searchsorted(np.asarray(BARRIERS), s // C, side="right")
        return 2 if seg[0] == seg[1] else 3
    return 4


def annotate(meta: dict, tokens: np.ndarray, n_docs: int, C: int, bos: int) -> list[dict]:
    """Per-document probe table with last-source position and its index in the focal order.
    Raises if the recomputed sources disagree with the saved probe annotations."""
    off, foff = meta["probe_offset"], meta["focal_offset"]
    out = []
    for i in range(n_docs):
        sl = slice(off[i], off[i + 1])
        pos, strat = meta["probe_pos"][sl], meta["probe_stratum"][sl]
        focal = meta["focal_pos"][foff[i]:foff[i + 1]]
        srcs = probe_sources(tokens[i], C, bos, pos)
        if any(len(s) == 0 for s in srcs):
            raise ValueError(f"document {i}: a probe has no recomputed source")
        last = np.array([s[-1] for s in srcs], dtype=np.int64)
        checks = (
            np.array_equal([len(s) for s in srcs], meta["probe_n_src"][sl]),
            np.array_equal(pos - last, meta["probe_dist"][sl]),
            np.array_equal([_stratum(s, C) for s in srcs], strat),
            np.array_equal(np.unique(np.concatenate(srcs)) if srcs else focal[:0], focal),
        )
        if not all(checks):
            raise ValueError(f"document {i}: recomputed sources disagree with eval_meta {checks}")
        both_s2 = np.array([len(s) == 2 and (s >= 2 * C).all() and (s < PROBE_START * C).all()
                            for s in srcs], dtype=bool)
        out.append(dict(sl=sl, stratum=strat, dist=meta["probe_dist"][sl], both_s2=both_s2,
                        last_focal=foff[i] + np.searchsorted(focal, last)))
    return out


def _within(v: np.ndarray, table: list[dict], a: np.ndarray, b_sel) -> tuple[np.ndarray, np.ndarray]:
    """Per-document mean(v | a) - mean(v | b) over docs having both groups; returns values and
    the kept document indices. ``a`` and ``b_sel`` are per-probe boolean vectors."""
    vals, keep = [], []
    for i, d in enumerate(table):
        sa, sb = a[d["sl"]], b_sel[d["sl"]]
        if sa.any() and sb.any():
            x = v[d["sl"]]
            vals.append(float(x[sa].mean() - x[sb].mean()))
            keep.append(i)
    return np.asarray(vals), np.asarray(keep, dtype=np.int64)


def _test(vals, keep, clusters, *, n_boot, seed, alpha, min_clusters) -> dict:
    k = int(len(np.unique(clusters[keep]))) if len(keep) else 0
    r = dict(n_docs=int(len(keep)), clusters=k)
    if k < min_clusters:
        return dict(r, est=float(vals.mean()) if len(vals) else None, lo=None, hi=None,
                    unsupported=True, passed=False)
    b = one_sided(*boot_mean(vals, bootstrap_weights(clusters[keep], n_boot, seed)), alpha)
    nan_frac = b.pop("unsupported")  # share of bootstrap draws with no document
    return dict(r, **b, boot_nan_frac=nan_frac, unsupported=False, passed=bool(b["lo"] > 0))


def contrasts(tok: dict, table: list[dict], clusters, q_pred, *, rules=(), **kw) -> dict:
    """All A7 contrasts for one step. ``tok``: per-probe adapt NLL per arm key."""
    n = table[-1]["sl"].stop if table else 0
    strat = np.concatenate([d["stratum"] for d in table]) if table else np.zeros(0, int)
    both_s2 = np.concatenate([d["both_s2"] for d in table]) if table else np.zeros(0, bool)
    e2c, e2b = strat == E2C, strat == E2B
    D = tok["fo"][:n] - tok["so"][:n]
    out = {"P7.1": _test(*_within(D, table, e2c, e2b), clusters, **kw),
           "P7.1s": _test(*_within(D, table, e2c & both_s2, e2b), clusters, **kw),
           "P7.2": _test(*_within(D, table, q_pred[:n], ~q_pred[:n]), clusters, **kw)}
    for rule in rules:
        Cr = tok["fo"][:n] - tok[rule][:n]
        out[f"P7.3_{rule}"] = _test(*_within(Cr, table, e2c, e2b), clusters, **kw)
    return out


def a7(arm_dirs: dict, decision_dir, tokens: np.ndarray, *,
       step: int = 400, C: int = 1024, bos: int = 128000, n_boot: int = 10000, seed: int = 0,
       alpha: float = 0.05, min_clusters: int = 20) -> dict:
    """``tokens``: [pool, T + 1] evaluation rows in eligible order (``eval_rows`` of meta)."""
    path = _destination(decision_dir, "a7.json")
    size_path = Path(decision_dir) / "sizing.json"
    size = json.loads(size_path.read_text())
    N = size["N_eval"]
    names = {"so": "e2e2", "fo": "e2e1-ce", "lin": "e2e1-lin", "aice": "e2e1-aice"}
    present = {k: v for k, v in names.items() if v in arm_dirs}
    if not {"so", "fo"} <= set(present):
        raise ValueError("A7 needs the SO and FO-CE arms")
    rules = [r for r in ("lin", "aice") if r in present]
    meta, z0, inputs = _load(arm_dirs["e2e2"], 0)
    pool = len(meta["eval_sha1"])
    if len(tokens) != pool:
        raise ValueError(f"token rows ({len(tokens)}) differ from the eligible pool ({pool})")
    for k, name in present.items():
        m = _load(arm_dirs[name], 0)[0]
        if any(not np.array_equal(m[x], meta[x]) for x in
               ("eval_sha1", "eval_domain", "probe_offset", "probe_pos", "focal_pos")):
            raise ValueError(f"{name}: evaluation metadata differs from SO")
    table_pool = annotate(meta, tokens, pool, C, bos)
    _, clusters = _ids(meta, meta, "eval", pool)
    # P7.2 covariate: SO step-0 write-time NLL at the last source, pool-wide median split
    q = np.asarray(z0["src_tok_adapt"], dtype=np.float64)
    q_last = np.concatenate([q[d["last_focal"]] for d in table_pool])
    median = float(np.median(q_last))
    q_pred = q_last < median
    kw = dict(n_boot=n_boot, seed=seed, alpha=alpha, min_clusters=min_clusters)
    result = {"amendment": "P2-1/A7", "step": step, "N_eval": N, "rules": rules,
              "q_median": median, "alpha": alpha, "n_boot": n_boot, "seed": seed,
              "min_clusters": min_clusters, "smoke": size.get("smoke", False)}
    for s in [step, *TRAJECTORY_STEPS]:
        tok, cot, n_step = {}, {}, None
        for k, name in present.items():
            _, z, h = _load(arm_dirs[name], s)
            inputs.update(h)
            n_docs = len(z["doc_adapt"])
            if not np.array_equal(z["eval_sha1"], meta["eval_sha1"][:n_docs]):
                raise ValueError(f"{name} step {s}: document order differs")
            n_step = n_docs if n_step is None else min(n_step, n_docs)
            tok[k] = np.asarray(z["tok_adapt"], dtype=np.float64)
            cot[k] = np.asarray(z["src_cot_adapt"], dtype=np.float64)
        n_use = min(N, n_step) if s == step else n_step
        table = table_pool[:n_use]
        r = contrasts(tok, table, clusters[:n_use], q_pred, rules=rules, **kw)
        # P7.4: write-time cotangent norm at the last source, per stratum
        lf = np.concatenate([d["last_focal"] for d in table])
        strat = np.concatenate([d["stratum"] for d in table])
        both_s2 = np.concatenate([d["both_s2"] for d in table])
        dist = np.concatenate([d["dist"] for d in table])
        desc = {}
        for label, sel in (("E2c", strat == E2C), ("E2c_S2", (strat == E2C) & both_s2),
                           ("E2b", strat == E2B)):
            so, fo = cot["so"][lf[sel]], cot["fo"][lf[sel]]
            desc[label] = dict(n_probes=int(sel.sum()), dist_mean=float(dist[sel].mean()) if sel.any() else None,
                               cot_so_mean=float(so.mean()) if sel.any() else None,
                               cot_fo_mean=float(fo.mean()) if sel.any() else None,
                               cot_ratio_of_means=float(fo.mean() / so.mean()) if sel.any() and so.mean() > 0 else None)
        r.update(n_docs=int(n_use), P7_4=desc, confirmatory=s == step)
        result[f"step{s:04d}"] = r
    result["inputs"] = inputs | {str(size_path.resolve()): _sha1(size_path)}
    _write(path, result)
    return result


def main(argv=None):
    import argparse
    import os

    from .data import open_split

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME"))
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--step", type=int, default=400)
    args = parser.parse_args(argv)
    if args.process_name:
        try:
            import setproctitle
            setproctitle.setproctitle(args.process_name)
        except ImportError:
            pass
    arms = dict(item.partition("=")[::2] for item in args.arm)
    with np.load(Path(arms["e2e2"]) / "eval_meta.npz", allow_pickle=False) as m:
        rows = m["eval_rows"]
    arr, _ = open_split(args.data_dir, "eval")
    tokens = np.asarray(arr[rows])
    print(json.dumps(a7(arms, args.decision_dir, tokens, step=args.step), sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
