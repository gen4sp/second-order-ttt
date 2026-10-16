"""Barrier metric: repeated-context targets, exposure strata, write masks, statistics.

Layout (per 8192-token document of 8 chunks): ``S1`` = chunks 0-1 | barrier |
``S2`` = chunks 2-3 | barrier | ``P`` = chunks 4-7. Inputs ``x``, targets ``y`` (``y_t =
x_{t+1}``). An n-gram *occurrence* at target position t is ``(x_{t-n+2}, ..., x_t -> y_t)``
inside one segment; it is written by the inner update after chunk ``t // C``.

Everything here is NumPy and arm-independent (token IDs only), so the target set, strata
and masks are computed once per document and reused for every arm, control and checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

# P2-1/P2-2 retain their registered default.  P2-3/MINI pass (2, 4, 6)
# explicitly through TrainCfg/eval configs.
BARRIERS = (2, 4)
PROBE_START = 4
STRATA = ("E1", "E2s", "E2c", "E2b", "E3")


def segment_starts(n_chunks: int, barriers=BARRIERS) -> np.ndarray:
    return np.array(sorted({0, *barriers}), dtype=np.int64)


def boundary_valid(T: int, C: int, barriers=BARRIERS) -> np.ndarray:
    """False at the last position before each barrier (its target crosses the barrier)."""
    ok = np.ones(T, dtype=bool)
    for b in barriers:
        if 0 < b * C < T:
            ok[b * C - 1] = False
    return ok


def _gram_ids(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Integer id per position for the n-gram (x[t-n+2..t], y[t]); -1 where t < n-2."""
    T = len(x)
    cols = [np.concatenate([np.full(k, -1), x[: T - k]]) for k in range(n - 2, -1, -1)]
    g = np.stack(cols + [y], 1).astype(np.int64)
    _, inv = np.unique(g, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    inv[: n - 2] = -1
    return inv


@dataclass
class DocProbes:
    pos: np.ndarray        # [K] probe target positions (in P)
    stratum: np.ndarray    # [K] index into STRATA
    n_src: np.ndarray      # [K] number of source occurrences
    dist: np.ndarray       # [K] probe position - last source position
    src_gap: np.ndarray    # [K] distance between the first two sources (0 if one)
    match8: np.ndarray     # [K] the 8-gram also occurs in S
    focal: np.ndarray      # [T] bool, every source occurrence of every probe n-gram
    nonfocal: np.ndarray   # [T] bool, count-matched random non-focal source positions
    nf_ok: bool           # exact count matching is possible in every source chunk


def find_probes(x: np.ndarray, y: np.ndarray, C: int, *, valid: np.ndarray | None = None,
                excluded: np.ndarray | None = None, barriers=BARRIERS,
                probe_start: int = PROBE_START, n: int = 4, seed: int = 0) -> DocProbes:
    """Probe targets of one document and their exposure strata."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    T = len(x)
    pos = np.arange(T)
    starts = segment_starts(T // C, barriers) * C
    seg = np.searchsorted(starts, pos, side="right") - 1
    seg_start = starts[seg]
    ok = boundary_valid(T, C, barriers) if valid is None else valid & boundary_valid(T, C, barriers)

    def occ(k):
        ids = _gram_ids(x, y, k)
        good = ok & (pos - (k - 2) >= seg_start)
        return np.where(good, ids, -1)

    key, key8 = occ(n), occ(8)
    p0 = probe_start * C
    in_s = (pos < p0) & (key >= 0)
    in_p = (pos >= p0) & (key >= 0)
    src_keys = set(key[in_s].tolist())
    src8 = set(key8[(pos < p0) & (key8 >= 0)].tolist())
    # first occurrence of each key inside P
    pk = key[in_p]
    first = np.zeros(T, dtype=bool)
    _, idx = np.unique(pk, return_index=True)
    first[pos[in_p][idx]] = True
    matched = in_p & np.isin(key, list(src_keys))
    cand = matched & first
    if excluded is not None:
        cand &= ~excluded[y]
    # one probe per maximal run of consecutive matched positions (overlapping repeats)
    prev = np.concatenate([[False], matched[:-1]])
    cand &= ~prev
    probes = pos[cand]
    chunk = pos // C
    src_pos = {}
    for t in pos[in_s]:
        src_pos.setdefault(int(key[t]), []).append(int(t))
    strat, nsrc, dist, gap, m8 = [], [], [], [], []
    focal = np.zeros(T, dtype=bool)
    for t in probes:
        s = src_pos[int(key[t])]
        focal[s] = True
        nsrc.append(len(s))
        dist.append(int(t - s[-1]))
        gap.append(int(s[1] - s[0]) if len(s) > 1 else 0)
        m8.append(int(key8[t]) in src8 if key8[t] >= 0 else False)
        if len(s) == 1:
            strat.append(0)
        elif len(s) == 2:
            if chunk[s[0]] == chunk[s[1]]:
                strat.append(1)
            elif seg[s[0]] == seg[s[1]]:
                strat.append(2)
            else:
                strat.append(3)
        else:
            strat.append(4)
    # count-matched non-focal control: same number per chunk, from non-focal written positions
    rng = np.random.default_rng(seed)
    nonfocal = np.zeros(T, dtype=bool)
    nf_ok = True
    writable = ok & (pos < p0)
    for c in range(probe_start):
        k = int(focal[c * C:(c + 1) * C].sum())
        if k == 0:
            continue
        pool = pos[c * C:(c + 1) * C][writable[c * C:(c + 1) * C] & ~focal[c * C:(c + 1) * C]]
        nf_ok &= len(pool) >= k
        nonfocal[rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
    as_arr = lambda v, dt=np.int64: np.asarray(v, dtype=dt)  # noqa: E731
    return DocProbes(pos=as_arr(probes), stratum=as_arr(strat), n_src=as_arr(nsrc),
                     dist=as_arr(dist), src_gap=as_arr(gap), match8=as_arr(m8, bool),
                     focal=focal, nonfocal=nonfocal, nf_ok=bool(nf_ok))


# ------------------------------------------------------------------------- statistics
def doc_mean(nll: np.ndarray, probes: list[DocProbes], stratum: int | None = None) -> np.ndarray:
    """Per-document mean NLL over its probe targets (optionally one stratum); NaN if none."""
    out = np.full(len(probes), np.nan)
    for i, p in enumerate(probes):
        sel = p.pos if stratum is None else p.pos[p.stratum == stratum]
        if len(sel):
            out[i] = float(nll[i, sel].mean())
    return out


def bootstrap_weights(clusters: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Cluster bootstrap: multinomial cluster counts [n_boot, n_clusters] and the
    document -> cluster index. The same draws are reused for every contrast."""
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(uniq)
    if k == 0:
        raise ValueError("no clusters")
    rng = np.random.default_rng(seed)
    return rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64), inv


def boot_mean(v: np.ndarray, weights) -> tuple[float, np.ndarray]:
    """Point estimate and bootstrap draws of a finite per-document contrast."""
    W, inv = weights
    v = _vector(v, len(inv))
    k = W.shape[1]
    s = np.bincount(inv, weights=v, minlength=k)
    c = np.bincount(inv, minlength=k).astype(np.float64)
    den = W @ c
    draws = np.full(len(W), np.nan)
    np.divide(W @ s, den, out=draws, where=den > 0)
    return float(v.mean()), draws


def one_sided(point: float, draws: np.ndarray, alpha: float = 0.05) -> dict:
    bounds = np.nanquantile(draws, [alpha, 1 - alpha]) if np.isfinite(draws).any() else [np.nan] * 2
    return {"est": point, "lo": float(bounds[0]), "hi": float(bounds[1]),
            "unsupported": float(np.isnan(draws).mean())}


def _vector(v, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (n,) or n == 0 or not np.isfinite(v).all():
        raise ValueError("expected aligned, finite document vectors")
    return v


def _clusters(clusters) -> np.ndarray:
    clusters = np.asarray(clusters)
    if clusters.ndim != 1 or not len(clusters):
        raise ValueError("expected a nonempty cluster vector")
    return clusters


def _arms(arrays: dict, n: int, keys) -> dict:
    if not set(keys).issubset(arrays):
        raise ValueError(f"missing arms: {sorted(set(keys) - arrays.keys())}")
    return {a: _vector(v, n) for a, v in arrays.items()}


def g2_decision(L: dict, G: dict, M: dict, clusters, L_ho: dict, clusters_ho, *,
                rule: str = "lin", n_boot: int = 10000, seed: int = 0,
                alpha: float = 0.05, min_clusters: int = 20) -> dict:
    """G2 = G2-loss and G2-memory. ``L``, ``G``, ``M``: arm -> per-document arrays
    over the eligible evaluation documents (arms ``so``, ``fo`` and ``rule``); ``L_ho``:
    arm -> per-document held-out losses on the separate ``ho`` split. One-sided bounds at
    ``alpha``; the same bootstrap draws for every contrast on a document set."""
    clusters, clusters_ho = _clusters(clusters), _clusters(clusters_ho)
    L, G, M = [_arms(a, len(clusters), ("so", "fo", rule)) for a in (L, G, M)]
    L_ho = _arms(L_ho, len(clusters_ho), ("so", rule))
    supported = len(np.unique(clusters)) >= min_clusters
    supported_ho = len(np.unique(clusters_ho)) >= min_clusters
    w = bootstrap_weights(clusters, n_boot, seed)
    wh = bootstrap_weights(clusters_ho, n_boot, seed + 1)
    D = boot_mean(L["fo"] - L["so"], w)
    CmD = boot_mean(L["fo"] - L[rule] - 0.5 * (L["fo"] - L["so"]), w)
    dho = boot_mean(L_ho[rule] - L_ho["so"], wh)
    dG = boot_mean(G["so"] - G["fo"], w)
    R = boot_mean(G[rule] - G["fo"] - 0.5 * (G["so"] - G["fo"]), w)
    dM = boot_mean(M["so"] - M["fo"], w)
    RM = boot_mean(M[rule] - M["fo"] - 0.5 * (M["so"] - M["fo"]), w)
    Mr = boot_mean(M[rule], w)
    r = {"D": one_sided(*D, alpha), "C_minus_half_D": one_sided(*CmD, alpha),
         "delta_ho": one_sided(*dho, alpha), "delta_G": one_sided(*dG, alpha),
         "rescue_G": one_sided(*R, alpha), "delta_M": one_sided(*dM, alpha),
         "rescue_M": one_sided(*RM, alpha), "M_rule": one_sided(*Mr, alpha)}
    r["pass"] = {"D": supported and r["D"]["est"] >= 0.05 and r["D"]["lo"] > 0,
                 "C": supported and r["C_minus_half_D"]["lo"] > 0,
                 "ho": supported_ho and r["delta_ho"]["hi"] <= 0.02,
                 "attribution": supported and r["delta_M"]["lo"] > 0,
                 "rescue": supported and r["rescue_M"]["lo"] > 0,
                 "source": supported and r["M_rule"]["lo"] > 0}
    r["unsupported_clusters"] = not (supported and supported_ho)
    p = r["pass"]
    r["G2_loss"] = p["D"] and p["C"] and p["ho"]
    r["G2_memory"] = p["attribution"] and p["rescue"] and p["source"]
    r["G2"] = r["G2_loss"] and r["G2_memory"]
    r["n_docs"], r["n_ho"] = int(len(clusters)), int(len(clusters_ho))
    return r


def assay(M0: np.ndarray, G0: np.ndarray, clusters, *, n_boot: int = 10000, seed: int = 0,
          alpha: float = 0.05, min_clusters: int = 20) -> dict:
    """Step-0 assay gate: source-specific memory and adaptation gain detectable."""
    clusters = _clusters(clusters)
    M0, G0 = _vector(M0, len(clusters)), _vector(G0, len(clusters))
    supported = len(np.unique(clusters)) >= min_clusters
    w = bootstrap_weights(clusters, n_boot, seed)
    m, g = one_sided(*boot_mean(M0, w), alpha), one_sided(*boot_mean(G0, w), alpha)
    return {"M0": m, "G0": g, "pass": supported and m["lo"] > 0 and g["lo"] > 0,
            "unsupported_clusters": not supported, "n_docs": len(clusters)}


def joint_power(contrasts: dict[str, np.ndarray], clusters, ho_contrast: np.ndarray,
                ho_clusters, alternatives: dict[str, float], Ns, *, pool_clusters,
                ho_pool_clusters, n_sim: int = 2000, alpha: float = 0.05, seed: int = 0,
                min_clusters: int = 20) -> dict[int, float]:
    """Joint clustered power of the endpoint at declared alternatives.

    The simulation follows the frozen endpoint design: for candidate N, the first N documents
    of the eligible pool (``pool_clusters``, pool order) and the fixed held-out split
    (``ho_pool_clusters``) define the number and sizes of the domain clusters. Each design
    cluster draws a pilot cluster at random and resamples that many documents from it with
    replacement (a two-stage bootstrap). Pilots with fewer than ``min_clusters`` domains are
    rejected; a design with fewer than ``min_clusters`` domains has power 0, because the
    endpoint (``g2_decision``) requires that support.
    """
    reduced = set(contrasts) == {"D", "dM"}
    keys = ("D", "dM") if reduced else ("D", "CmD", "dM", "RM", "M_rule")
    clusters, ho_clusters = _clusters(clusters), _clusters(ho_clusters)
    pool_clusters, ho_pool_clusters = _clusters(pool_clusters), _clusters(ho_pool_clusters)
    if set(contrasts) != set(keys) or not set(keys).issubset(alternatives):
        raise ValueError("declare all five contrasts and their alternatives")
    v = np.stack([_vector(contrasts[k], len(clusters)) for k in keys], 1)
    vh = _vector(ho_contrast, len(ho_clusters))[:, None]
    alt = np.array([alternatives[k] for k in keys], dtype=np.float64)
    ah = float(alternatives.get("ho", 0.0))
    if not np.isfinite(alt).all() or not np.isfinite(ah):
        raise ValueError("alternatives must be finite")
    Ns = list(Ns)
    if n_sim < 1 or any(N < 1 or int(N) != N for N in Ns) or not 0 < alpha < 0.5:
        raise ValueError("invalid simulation size or alpha")
    if any(N > len(pool_clusters) for N in Ns):
        raise ValueError("candidate N exceeds the eligible pool")
    if len(np.unique(clusters)) < min_clusters or (not reduced and len(np.unique(ho_clusters)) < min_clusters):
        raise ValueError(f"pilot has fewer than {min_clusters} domain clusters")
    v = v - v.mean(0) + alt
    vh = vh - vh.mean(0) + ah
    rng = np.random.default_rng(seed)
    z = NormalDist().inv_cdf(1 - alpha)

    def groups(cl):
        _, inv = np.unique(cl, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)

    def design(cl):  # cluster sizes of a frozen document set
        return np.unique(cl, return_counts=True)[1]

    gs, gh = groups(clusters), groups(ho_clusters)
    ho_sizes = design(ho_pool_clusters)

    def draw(a, g, sizes):
        picks = rng.integers(len(g), size=len(sizes))
        rows = [g[j][rng.integers(len(g[j]), size=n)] for j, n in zip(picks, sizes)]
        sample = a[np.concatenate(rows)]
        est = sample.mean(0)
        K, N = len(sizes), int(sizes.sum())
        sums = np.add.reduceat(sample - est, np.cumsum(np.r_[0, sizes[:-1]]), axis=0)
        se = np.sqrt(K / (K - 1) * (sums * sums).sum(0)) / N
        return est, se

    powers = {}
    for N in Ns:
        sizes = design(pool_clusters[:N])
        if len(sizes) < min_clusters or (not reduced and len(ho_sizes) < min_clusters):
            powers[int(N)] = 0.0
            continue
        passed = 0
        for _ in range(n_sim):
            est, se = draw(v, gs, sizes)
            eh, sh = (np.zeros(1), np.zeros(1)) if reduced else draw(vh, gh, ho_sizes)
            passed += bool(np.all(est - z * se > 0) and est[0] >= 0.05
                           and (reduced or eh[0] + z * sh[0] <= 0.02))
        powers[int(N)] = passed / n_sim
    return powers


# ------------------------------------------------------------------- key specificity (κ)
@dataclass
class KappaSet:
    """SOQ-N3B κ read of one document: the query positions whose candidate logits are gathered
    (every probe position t_k and its foils) and the candidate ids (Y_doc, the distinct probe
    targets).  ``probe_q``/``foil_q`` index ``pos``; ``probe_v`` indexes ``cand``."""

    pos: np.ndarray            # [Q] sorted unique query positions
    cand: np.ndarray           # [V] sorted distinct probe targets of the document (Y_doc)
    probe_q: np.ndarray        # [K] row of t_k in ``pos`` (DocProbes.pos order)
    probe_v: np.ndarray        # [K] column of y_k in ``cand``
    n_src: np.ndarray          # [K] source occurrences of the probe key (DocProbes.n_src)
    y_mult: np.ndarray         # [K] loss-valid positions of the document whose target is y_k
    seg: np.ndarray            # [K] barrier segment of t_k
    foil_off: np.ndarray       # [K + 1] offsets of each probe's foils in ``foil_q``
    foil_q: np.ndarray         # rows of the foil positions in ``pos``
    foil_partial: np.ndarray   # foil's x_f is the last token of a context preceding y_k in S
    foil_partial_own: np.ndarray  # foil's x_f is x_{t_k} (the last token of the probe's key)


def kappa_set(x: np.ndarray, y: np.ndarray, probes: DocProbes, C: int, *,
              valid: np.ndarray | None = None, barriers=BARRIERS,
              probe_start: int = PROBE_START, n_foils: int = 8, seed: int = 0) -> KappaSet:
    """Foils of probe k: up to ``n_foils`` loss-valid positions f in P (chunks >= probe_start),
    in the barrier segment of t_k, with target != y_k and 3-token context
    (x_{f-2}, x_{f-1}, x_f) different from every context that precedes y_k in the written
    chunks (< probe_start; any position, so a foil can never carry y_k's own key).  Drawn
    without replacement from ``seed`` (the document's eval row), so every arm, η and mode
    reads the same foils.  Foils whose x_f equals the last token of such a context are tagged
    (``foil_partial``; ``foil_partial_own``: x_f = x_{t_k}), the partial-key stratum."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    T = len(x)
    p0 = probe_start * C
    pos = np.arange(T)
    starts = segment_starts(T // C, barriers) * C
    seg = np.searchsorted(starts, pos, side="right") - 1
    ok = boundary_valid(T, C, barriers) if valid is None else \
        np.asarray(valid, dtype=bool) & boundary_valid(T, C, barriers)
    ctx = np.full(T, -1, dtype=np.int64)
    if T > 2:  # token ids < 2**17 (128256), so three ids pack into one int64
        ctx[2:] = (x[:-2] << 34) | (x[1:-1] << 17) | x[2:]
    pool = ok & (pos >= p0) & (pos >= 2)
    written = pos < p0
    rng = np.random.default_rng(seed)
    foils, part, own = [], [], []
    for t in probes.pos:
        yk = y[t]
        before = written & (y == yk)
        elig = pool & (seg == seg[t]) & (y != yk) & ~np.isin(ctx, ctx[before])
        f = pos[elig]
        if len(f) > n_foils:
            f = np.sort(rng.choice(f, size=n_foils, replace=False))
        foils.append(f)
        part.append(np.isin(x[f], x[before]))
        own.append(x[f] == x[t])
    empty = np.zeros(0, dtype=np.int64)
    q = np.unique(np.concatenate([probes.pos, *foils])).astype(np.int64) if len(probes.pos) else empty
    cand = np.unique(y[probes.pos]).astype(np.int64)
    yv = y[probes.pos]
    return KappaSet(pos=q, cand=cand,
                    probe_q=np.searchsorted(q, probes.pos).astype(np.int64),
                    probe_v=np.searchsorted(cand, yv).astype(np.int64),
                    n_src=np.asarray(probes.n_src, dtype=np.int64),
                    y_mult=np.array([int((ok & (y == v)).sum()) for v in yv], dtype=np.int64),
                    seg=seg[probes.pos].astype(np.int64),
                    foil_off=np.cumsum([0] + [len(f) for f in foils]).astype(np.int64),
                    foil_q=np.searchsorted(q, np.concatenate(foils)).astype(np.int64)
                    if foils else empty,
                    foil_partial=np.concatenate(part).astype(bool) if part else empty.astype(bool),
                    foil_partial_own=np.concatenate(own).astype(bool) if own else empty.astype(bool))


def _ratio_kappa(SP: np.ndarray, SF: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """κ = 1 - ΣF/ΣP (NaN, never clipped, if ΣP <= 0), per document and per bootstrap draw."""
    with np.errstate(invalid="ignore", divide="ignore"):
        kdoc = np.where(SP > 0, 1.0 - SF / np.where(SP > 0, SP, 1.0), np.nan)
        dsp, dsf = W @ SP, W @ SF
        kdraws = np.where(dsp > 0, 1.0 - dsf / np.where(dsp > 0, dsp, 1.0), np.nan)
    sp, sf = SP.sum(), SF.sum()
    return (float(1.0 - sf / sp) if sp > 0 else float("nan")), kdoc, kdraws


def kappa_stats(sets: list[KappaSet], z_nf: list[np.ndarray], z_f: list[np.ndarray], *,
                n_boot: int = 10000, seed: int = 0) -> tuple[dict, dict]:
    """κ reads from candidate values ``z[i]`` [Q_i, V_i] (centred logits, or log-probs: the
    same function reads both) of every document in the non-focal (``z_nf``) and focal
    (``z_f``) frozen-P modes.  Δ = nonfocal - focal.

    Cross-over (primary): the probe-row matrices Δ[k, v] = Δz_v(t_k) are
    returned whole (``x_nf``/``x_f`` flattened per document, offsets ``x_off``) and
    κ× = ΣS / Σ[Δ_{y_k}(t_k) + Δ_{y_j}(t_j)] over same-document pairs y_k != y_j, with
    S_kj = [Δ_{y_k}(t_k) + Δ_{y_j}(t_j)] - [Δ_{y_k}(t_j) + Δ_{y_j}(t_k)] (document bootstrap;
    ``kappa_x_na`` when the bootstrap LB90 of the diagonal sum is <= 0).  Rank: y_k's rank gain
    (focal - nonfocal rank within Y_doc) at t_k minus its mean gain at the other probes'
    positions t_j (y_j != y_k).

    Foil κ (secondary): P_k = Δz_{y_k}(t_k); F_k = mean over the foils of Δz_{y_k}(f); per
    document κ_doc = 1 - ΣF_k / ΣP_k over probes with foils (NaN if ΣP_k <= 0, never clipped);
    pooled κ = 1 - ΣF / ΣP with a ``n_boot`` document bootstrap (also on the partial-key
    stratum excluded).  Rank: 1 + the number of other candidates of Y_doc above y_k at t_k
    (documents with |Y_doc| >= 2; chance = mean 1/|Y_doc|).  Returns (summary, arrays)."""
    cols = {k: [] for k in ("p_nf", "p_f", "foilmean_nf", "foilmean_f", "n_foil",
                            "n_foil_nonpartial", "rank_nf", "rank_f", "ydoc", "foil_nf",
                            "foil_f", "foil_probe", "foil_partial", "foil_partial_own",
                            "x_nf", "x_f", "probe_v", "n_src", "y_mult", "seg", "rank_gain_x")}
    k0 = 0
    n = len(sets)
    SP, SF, SFn = np.zeros(n), np.zeros(n), np.zeros(n)
    top_d, cnt_d = np.zeros(n), np.zeros(n)
    diag_s, off_s, off_c, pair_s, pair_d, pair_c, rx_s, rx_c = (np.zeros(n) for _ in range(8))
    x_off, foil_doc_off = [0], [0]
    for i, (s, a, b) in enumerate(zip(sets, z_nf, z_f)):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        K, V = len(s.probe_q), len(s.cand)
        p_nf, p_f = a[s.probe_q, s.probe_v], b[s.probe_q, s.probe_v]
        n_foil = np.diff(s.foil_off)
        fk = np.repeat(np.arange(K), n_foil)
        foil_nf, foil_f = a[s.foil_q, s.probe_v[fk]], b[s.foil_q, s.probe_v[fk]]
        fm_nf, fm_f = np.full(K, np.nan), np.full(K, np.nan)
        fmn = np.full(K, np.nan)
        keep = ~s.foil_partial
        n_np = np.bincount(fk[keep], minlength=K) if K else np.zeros(0, np.int64)
        for k in np.flatnonzero(n_foil):
            sl = slice(s.foil_off[k], s.foil_off[k + 1])
            fm_nf[k], fm_f[k] = foil_nf[sl].mean(), foil_f[sl].mean()
            kk = keep[sl]
            if kk.any():
                fmn[k] = (foil_nf[sl][kk] - foil_f[sl][kk]).mean()
        has = n_foil > 0
        SP[i] = (p_nf - p_f)[has].sum()
        SF[i] = (fm_nf - fm_f)[has].sum()
        SFn[i] = fmn[n_np > 0].sum()
        rank_nf = 1 + (a[s.probe_q] > p_nf[:, None]).sum(1)
        rank_f = 1 + (b[s.probe_q] > p_f[:, None]).sum(1)
        if V >= 2:
            top_d[i] = float(((rank_nf == 1).astype(float) - (rank_f == 1)).sum())
            cnt_d[i] = K
        X_nf, X_f = a[s.probe_q], b[s.probe_q]      # [K, V] probe-row matrices
        D = X_nf - X_f
        dg = D[np.arange(K), s.probe_v]
        diag_s[i] = dg.sum()
        offm = np.ones((K, V), dtype=bool)
        offm[np.arange(K), s.probe_v] = False
        off_s[i], off_c[i] = D[offm].sum(), offm.sum()
        rx = np.full(K, np.nan)
        if K >= 2:
            # S_kj over pairs k < j with y_k != y_j; κ× = ΣS / Σ(Δ_{y_k}(t_k) + Δ_{y_j}(t_j))
            cross = D[:, s.probe_v]                  # cross[k, j] = Δ_{y_j}(t_k)
            S = dg[:, None] + dg[None, :] - cross.T - cross
            other = s.probe_v[:, None] != s.probe_v[None, :]
            pm = np.triu(other, 1)
            pair_s[i], pair_c[i] = S[pm].sum(), pm.sum()
            pair_d[i] = (dg[:, None] + dg[None, :])[pm].sum()
            # rank gain (focal rank - nonfocal rank) of y_k at t_k minus its mean at the other
            # probes' positions t_j (y_j != y_k)
            rk_nf = 1 + (X_nf[:, None, :] > X_nf[:, :, None]).sum(-1)   # [K, V]: rank of v at t_j
            rk_f = 1 + (X_f[:, None, :] > X_f[:, :, None]).sum(-1)
            gain = (rk_f - rk_nf)[:, s.probe_v]       # gain[j, k]: of y_k at t_j
            for k in range(K):
                js = np.flatnonzero(other[:, k])
                if len(js):
                    rx[k] = gain[k, k] - gain[js, k].mean()
            ok_rx = np.isfinite(rx)
            rx_s[i], rx_c[i] = rx[ok_rx].sum(), ok_rx.sum()
        for key, v in (("p_nf", p_nf), ("p_f", p_f), ("foilmean_nf", fm_nf),
                       ("foilmean_f", fm_f), ("n_foil", n_foil), ("n_foil_nonpartial", n_np),
                       ("rank_nf", rank_nf), ("rank_f", rank_f), ("ydoc", np.full(K, V)),
                       ("foil_nf", foil_nf), ("foil_f", foil_f), ("foil_probe", k0 + fk),
                       ("foil_partial", s.foil_partial), ("foil_partial_own", s.foil_partial_own),
                       ("x_nf", X_nf.ravel()), ("x_f", X_f.ravel()), ("probe_v", s.probe_v),
                       ("n_src", s.n_src), ("y_mult", s.y_mult), ("seg", s.seg),
                       ("rank_gain_x", rx)):
            cols[key].append(v)
        k0 += K
        x_off.append(x_off[-1] + K * V)
        foil_doc_off.append(foil_doc_off[-1] + len(fk))
    arr = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64) if n else np.zeros((0, 0))
    kappa, kdoc, kdraws = _ratio_kappa(SP, SF, W)
    SPn = np.zeros(n)  # ΣP over the probes that keep a non-partial foil
    p0 = 0
    for i, s in enumerate(sets):
        K = len(s.probe_q)
        sl = slice(p0, p0 + K)
        hn = arr["n_foil_nonpartial"][sl] > 0
        SPn[i] = (arr["p_nf"][sl] - arr["p_f"][sl])[hn].sum()
        p0 += K
    kappa_np, _, kdraws_np = _ratio_kappa(SPn, SFn, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        rdraws = (W @ top_d) / (W @ cnt_d)
        sdraws = (W @ pair_s) / (W @ pair_c)
        ddraws = W @ pair_d
        xdraws = np.where(ddraws != 0, (W @ pair_s) / np.where(ddraws != 0, ddraws, 1.0), np.nan)
        rxdraws = (W @ rx_s) / (W @ rx_c)
    multi = arr["ydoc"] >= 2
    q = lambda d, p: float(np.nanquantile(d, p)) if np.isfinite(d).any() else float("nan")  # noqa: E731
    mean = lambda x, m: float(x[m].mean()) if m.any() else float("nan")  # noqa: E731
    top_nf = mean((arr["rank_nf"] == 1).astype(float), multi)
    top_f = mean((arr["rank_f"] == 1).astype(float), multi)
    summary = {
        "kappa": kappa, "kappa_lo90": q(kdraws, .05), "kappa_hi90": q(kdraws, .95),
        "kappa_lo95": q(kdraws, .025), "kappa_hi95": q(kdraws, .975),
        "kappa_doc_median": float(np.nanmedian(kdoc)) if np.isfinite(kdoc).any() else float("nan"),
        "kappa_nonpartial": kappa_np, "kappa_nonpartial_lo90": q(kdraws_np, .05),
        "kappa_nonpartial_hi90": q(kdraws_np, .95),
        "sum_P": float(SP.sum()), "sum_F": float(SF.sum()), "n_docs": n,
        "n_docs_sumP_pos": int((SP > 0).sum()),
        "n_probes": int(len(arr["p_nf"])), "n_probes_with_foils": int((arr["n_foil"] > 0).sum()),
        "foils_mean": float(arr["n_foil"].mean()) if len(arr["n_foil"]) else float("nan"),
        "foil_partial_frac": float(arr["foil_partial"].mean()) if len(arr["foil_partial"]) else float("nan"),
        "cross_diag_mean": float(diag_s.sum() / max(1, len(arr["p_nf"]))),
        "cross_offdiag_mean": float(off_s.sum() / off_c.sum()) if off_c.sum() else float("nan"),
        "cross_S_mean": float(pair_s.sum() / pair_c.sum()) if pair_c.sum() else float("nan"),
        "cross_S_lo90": q(sdraws, .05), "cross_S_hi90": q(sdraws, .95),
        "cross_n_pairs": int(pair_c.sum()),
        "kappa_x": float(pair_s.sum() / pair_d.sum()) if pair_d.sum() != 0 else float("nan"),
        "kappa_x_lo90": q(xdraws, .05), "kappa_x_hi90": q(xdraws, .95),
        "kappa_x_sum_diag": float(pair_d.sum()), "kappa_x_sum_diag_lo90": q(ddraws, .05),
        "kappa_x_na": bool(not (q(ddraws, .05) > 0)),
        "rank_gain_x": float(rx_s.sum() / rx_c.sum()) if rx_c.sum() else float("nan"),
        "rank_gain_x_lo90": q(rxdraws, .05), "rank_gain_x_hi90": q(rxdraws, .95),
        "rank_top1_nonfocal": top_nf, "rank_top1_focal": top_f,
        "rank_top1_diff": (top_nf - top_f) if multi.any() else float("nan"),
        "rank_top1_diff_lo90": q(rdraws, .05), "rank_top1_diff_hi90": q(rdraws, .95),
        "rank_chance": mean(1.0 / np.maximum(arr["ydoc"], 1), multi),
        "rank_mrr_nonfocal": mean(1.0 / np.maximum(arr["rank_nf"], 1), multi),
        "rank_mrr_focal": mean(1.0 / np.maximum(arr["rank_f"], 1), multi),
        "rank_n_probes": int(multi.sum()), "n_boot": int(n_boot)}
    arr.update(doc_sum_P=SP, doc_sum_F=SF, doc_kappa=kdoc, doc_sum_F_nonpartial=SFn,
               doc_sum_P_nonpartial=SPn, doc_pair_S=pair_s, doc_pair_diag=pair_d, doc_pair_n=pair_c,
               doc_rank_gain_x_sum=rx_s, doc_rank_gain_x_n=rx_c,
               probe_offset=np.cumsum([0] + [len(s.probe_q) for s in sets]),
               x_off=np.asarray(x_off), foil_doc_offset=np.asarray(foil_doc_off),
               cand=np.concatenate([s.cand for s in sets]) if sets else np.zeros(0, np.int64),
               cand_offset=np.cumsum([0] + [len(s.cand) for s in sets]))
    return summary, arr
