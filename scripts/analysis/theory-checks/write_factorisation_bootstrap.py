"""Bootstrap machinery for the write-factorisation verdicts (imported by write_factorisation.py).

- count_verdict: "supported" if k >= ceil(.75 n), "falsified" if k <= floor(.5 n), else
  "inconclusive"; "not evaluable" below n = 6.
- pct_ci: percentile bootstrap CI of a mean (2000 draws, numpy seed 0).
- nested_pct_excl0: the same percentile rule re-applied inside each outer bootstrap draw
  (500 inner draws) on the resampled sequences.
- run_bootstraps: the point verdict plus three bootstraps:
  seq     shared evaluation-sequence indices: one stratified index draw per outer draw (strata
          [0,32), [32,64), [64,128), [128,256)); a unit with n sequences uses the first n columns,
          so the same evaluation sequence gets the same weight in every checkpoint of that draw;
  cluster training-seed families: every unit produced by one training seed is drawn together;
  hier    families and shared sequences in the same draw.
"""
from __future__ import annotations

import math

import numpy as np

STRATA = (0, 32, 64, 128, 256)
D_OUT = 2000
D_IN = 500
SEED_OUT = 3


# ----------------------------------------------------------------------------- generic machinery

def count_verdict(k, n, n_min=6):
    if n < n_min:
        return "not evaluable"
    if k >= math.ceil(.75 * n):
        return "supported"
    if k <= math.floor(.5 * n):
        return "falsified"
    return "inconclusive"


def shared_seq_draws(rng, D, n_max):
    """Stratified bootstrap indices (D, n_max) shared by every checkpoint of a draw."""
    cols = []
    for lo, hi in zip(STRATA[:-1], STRATA[1:]):
        if lo >= n_max:
            break
        hi = min(hi, n_max)
        cols.append(rng.integers(lo, hi, size=(D, hi - lo)))
    return np.concatenate(cols, 1)


def pct_ci(x, D=2000, seed=0, level=95):
    """Registered percentile bootstrap CI of the mean of x (numpy seed 0, 2000 draws)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(D, len(x)))].mean(1)
    a = (100 - level) / 2
    return float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def nested_pct_excl0(xs, rng, D_in=D_IN):
    """xs: (D, n) resampled per-sequence values. Re-applies the percentile rule inside each draw:
    returns (lo>0 or hi<0, lo, hi) per draw from D_in inner resamples (vectorised)."""
    D, n = xs.shape
    inner = rng.integers(0, n, size=(D_in, n))
    cnt = np.zeros((n, D_in))
    for j in range(D_in):                    # resample-count matrix; means = xs @ cnt / n
        cnt[:, j] = np.bincount(inner[j], minlength=n)
    means = xs @ cnt / n                     # (D, D_in)
    lo = np.percentile(means, 2.5, axis=1)
    hi = np.percentile(means, 97.5, axis=1)
    return (lo > 0) | (hi < 0), lo, hi


def family_draws(rng, families, D):
    """families: list of lists of unit labels. Returns (D, F) family indices drawn with replacement."""
    return rng.integers(0, len(families), size=(D, len(families)))


def summarize(point_k, point_n, ks, ns, verdict_fn):
    v0 = verdict_fn(point_k, point_n)
    vs = np.array([verdict_fn(int(k), int(n)) for k, n in zip(ks, ns)])
    share = np.where(ns > 0, ks / np.maximum(ns, 1), np.nan)
    return dict(count=int(point_k), n=int(point_n), verdict=v0, p_same=float(np.mean(vs == v0)),
                p_supported=float(np.mean(vs == "supported")), p_falsified=float(np.mean(vs == "falsified")),
                share_95=[float(np.nanpercentile(share, 2.5)), float(np.nanpercentile(share, 97.5))],
                n_range=[int(ns.min()), int(ns.max())])


def run_bootstraps(units, families, flag_fn, verdict_fn, rng, D=D_OUT, label=""):
    """units: dict label -> unit (with 'ps' per-sequence arrays and 'n').
    families: list of lists of labels (one list per training-seed family, restricted to this test).
    flag_fn(unit_label, idx or None, rng) -> (eligible[D] bool, flag[D] bool); idx=None means point.
    Returns point + three bootstraps: seq (shared indices), cluster (families, point flags),
    hier (families + shared sequences)."""
    labels = [l for f in families for l in f]
    n_max = max(units[l]["n"] for l in labels)
    e0 = {l: flag_fn(l, None, rng) for l in labels}
    pk = sum(int(e0[l][0][0] and e0[l][1][0]) for l in labels)
    pn = sum(int(e0[l][0][0]) for l in labels)
    out = dict(point=dict(count=pk, n=pn, verdict=verdict_fn(pk, pn)))
    # sequence bootstrap (all units kept)
    idx = shared_seq_draws(rng, D, n_max)
    es = {l: flag_fn(l, idx[:, :units[l]["n"]], rng) for l in labels}
    ks = sum((es[l][0] & es[l][1]).astype(int) for l in labels)
    ns = sum(es[l][0].astype(int) for l in labels)
    out["seq"] = summarize(pk, pn, np.asarray(ks), np.asarray(ns), verdict_fn)
    # cluster bootstrap over families (point per-unit outcomes)
    fd = family_draws(rng, families, D)
    fk = np.array([sum(int(e0[l][0][0] and e0[l][1][0]) for l in f) for f in families])
    fn = np.array([sum(int(e0[l][0][0]) for l in f) for f in families])
    out["cluster"] = summarize(pk, pn, fk[fd].sum(1), fn[fd].sum(1), verdict_fn)
    # hierarchical: families + shared sequences in the same draw
    idx = shared_seq_draws(rng, D, n_max)
    eh = {l: flag_fn(l, idx[:, :units[l]["n"]], rng) for l in labels}
    fkh = np.stack([sum((eh[l][0] & eh[l][1]).astype(int) for l in f) for f in families], 1)  # (D, F)
    fnh = np.stack([sum(eh[l][0].astype(int) for l in f) for f in families], 1)
    rows = np.arange(D)[:, None]
    out["hier"] = summarize(pk, pn, fkh[rows, fd].sum(1), fnh[rows, fd].sum(1), verdict_fn)
    out["families"] = [list(f) for f in families]
    return out


def mean_of(ps, key, idx):
    x = np.asarray(ps[key], float)
    return np.array([x.mean()]) if idx is None else x[idx].mean(-1)
