"""TRG frozen scoring and descriptive estimators; usable without a pool."""
from __future__ import annotations

import math
from math import ceil, comb, floor

import numpy as np

from .episodes import COMPLETE, DIVERGED, TRUNCATED, coded_dense_series as coded_series, nl2_window


def exact_sign(wins, losses):
    n = wins + losses
    return sum(comb(n, k) for k in range(wins, n+1)) / 2**n if n else 1.0


def holm2(pvalues):
    keys = sorted(pvalues, key=lambda k: (pvalues[k], k))
    out, previous = {}, 0.
    for rank, key in enumerate(keys):
        previous = max(previous, min(1., (2-rank)*pvalues[key]))
        out[key] = previous
    return out


def _ht3_counts(pairs):
    k = np.array([p[0] for p in pairs], bool)
    r = np.array([p[1] for p in pairs], bool)
    b, c = int((k & ~r).sum()), int((~k & r).sum())
    n = len(k)
    return dict(n=n, kvm_nl2=int(k.sum()), rnd_nl2=int(r.sum()), discordant=b+c,
                favorable=b, adverse=c, p=exact_sign(b, c) if n >= 8 else 1.)


def score_ht3(units, *, attempted=None):
    """units: frozen enrolled rows with family, kvm/rnd={nl2: bool|None}.

    Missingness imputes ONLY the missing member of a pair. Holm-2 is recomputed
    jointly after adverse completion, including NOT EVALUABLE as p=1.
    """
    raw, worst = {}, {}
    for family in ("FO", "noise"):
        rows = [u for u in units if u["family"] == family]
        pairs = [(u["kvm"]["nl2"], u["rnd"]["nl2"]) for u in rows]
        raw[family] = _ht3_counts([p for p in pairs if None not in p])
        raw[family].update(enrolled=len(rows), attempted=(attempted or {}).get(family, len(rows)))
        worst[family] = _ht3_counts([(False if k is None else k, True if r is None else r)
                                    for k, r in pairs])
    hp = holm2({f: r["p"] for f, r in raw.items()})
    hw = holm2({f: r["p"] for f, r in worst.items()})
    for family, r in raw.items():
        w = worst[family]
        observed = hp[family] <= .05 and r["kvm_nl2"] >= ceil(.75*r["n"])
        robust = hw[family] <= .05 and w["kvm_nl2"] >= ceil(.75*w["n"])
        label = ("NOT EVALUABLE" if r["n"] < 8 else "SUPPORTED" if observed and robust else
                 "FALSIFIED" if r["kvm_nl2"] <= floor(.25*r["n"]) else "INCONCLUSIVE")
        r.update(label=label, holm_p=hp[family], worst=dict(w, holm_p=hw[family]),
                 worst_case_survives=robust, observed_support=observed,
                 missing_sensitive=observed and not robust,
                 non_specific=r["rnd_nl2"] >= ceil(.5*r["n"]) if r["n"] else None)
    return raw


def unbound_burden(steps, accuracy, t_acq, *, end=660, disposition=COMPLETE, t_div=None):
    s, a = coded_series(steps, accuracy, start=t_acq+10, end=end, grid=1,
                        disposition=disposition, t_div=t_div)
    if s is None:
        return dict(U=None, deficit=None, length=max(0, end-t_acq-10))
    return dict(U=int((a < .90).sum()), deficit=float(np.maximum(0., .90-a).sum()), length=len(s))


def burden_ratio(skip, sham):
    maximum = max(skip, sham)
    return (sham-skip)/maximum if maximum else None


def _ht4_counts(pairs):
    non_tied = [(s, h) for s, h in pairs if s != h]
    ratios = [burden_ratio(s, h) for s, h in non_tied]
    wins = sum(s < h for s, h in non_tied)
    n = len(non_tied)
    return dict(n=len(pairs), non_tied=n, ties=len(pairs)-n, wins=wins,
                p=exact_sign(wins, n-wins), median_R=float(np.median(ratios)) if n else None,
                nonpositive=sum(r <= 0 for r in ratios), ratios=ratios)


def score_ht4(units, *, attempted=None):
    pairs = [(u["skip"]["U"], u["sham"]["U"]) for u in units]
    r = _ht4_counts([p for p in pairs if None not in p])
    adverse = [(u["length"] if s is None else s, 0 if h is None else h)
               for u, (s, h) in zip(units, pairs)]
    w = _ht4_counts(adverse)
    def reduces(row):
        return row["non_tied"] >= 6 and row["p"] <= .05 and row["median_R"] >= .5
    robust = reduces(w)
    label = ("REDUCES" if reduces(r) and robust else "NO REDUCTION"
             if r["non_tied"] and r["nonpositive"] >= ceil(.75*r["non_tied"]) else "UNCLEAR")
    r.update(label=label, enrolled=len(units), descriptive=True, worst=w,
             worst_case_survives=robust, missing_sensitive=reduces(r) and not robust,
             engaged=sum(u.get("drops", 0) > 0 for u in units),
             attempted=len(units) if attempted is None else attempted)
    return r


def localisation(terms, margins, steps, chunks, t_acq, t_l1, *, late=6):
    """terms shape (steps, sequences, events, 6), means across steps x events."""
    terms, margins, steps = np.asarray(terms, float), np.asarray(margins, float), np.asarray(steps)
    if t_l1 is None or t_l1 < t_acq+21 or t_l1+5 > steps[-1]:
        return dict(eligible=False)
    base = (steps >= t_acq) & (steps <= t_acq+15)
    loss = (steps >= t_l1-5) & (steps <= t_l1+5)
    if base.sum() != 16 or loss.sum() != 11:
        return dict(eligible=False, reason="ladder coverage")
    event = np.asarray(chunks) == late
    bt, lt = terms[base][:, event], terms[loss][:, event]
    bm, lm = margins[base][:, event], margins[loss][:, event]
    if not bm.size:
        return dict(eligible=False)
    d = lt.mean((0, 1))-bt.mean((0, 1))
    total, sd = float(lm.mean()-bm.mean()), float(bm.std())
    cancelling = bool(np.abs(d).sum() > 2*abs(total))
    dominant = np.flatnonzero(d/total >= .5) if total else []
    label = ("NEGLIGIBLE" if abs(total) < .25*sd else "CANCELLING" if cancelling else
             f"DOMINANT {dominant[0]}" if len(dominant) == 1 else "DISTRIBUTED")
    return dict(eligible=True, D_j=d.tolist(), D_total=total, baseline_sd=sd,
                accounting_error=float(abs(d.sum()-total)), label=label, cancelling=cancelling)


def fractional_crossing(steps, values, t_acq, t_l1, *, fraction=.25, direction=1):
    s, x = np.asarray(steps, int), np.asarray(values, float)
    if t_l1 is None or t_l1 > t_acq+280:
        return dict(label="INELIGIBLE", onset=None, confirmed=None)
    req = np.arange(t_acq, t_l1+21)
    mask = (s >= req[0]) & (s <= req[-1])
    if not np.array_equal(s[mask], req) or not np.isfinite(x[mask]).all():
        return dict(label="INELIGIBLE", onset=None, confirmed=None)
    s, x = s[mask], x[mask]
    pre = float(np.median(x[:16]))
    post = float(np.median(x[(s >= t_l1+10) & (s <= t_l1+20)]))
    mad = float(np.median(np.abs(x[:16]-pre)))
    change = post-pre
    # A constant raw baseline can have MAD=0; equality is still FLAT.
    if change == 0 or abs(change) < 3*mad:
        return dict(label="FLAT", onset=None, confirmed=None)
    if change*direction < 0:
        return dict(label="REVERSED", onset=None, confirmed=None)
    trail = np.concatenate(([np.median(x[:i+1]) for i in range(min(4,len(x)))],
                            np.median(np.lib.stride_tricks.sliding_window_view(x,5), axis=1)))
    threshold = pre + fraction*change
    ok = direction*(trail-threshold) >= 0
    held = np.lib.stride_tricks.sliding_window_view(ok, 5).all(1)
    hit = np.flatnonzero(held & (s[:-4] >= t_acq+16))
    return dict(label="CROSSED" if len(hit) else "NO CROSSING",
                onset=int(s[hit[0]]) if len(hit) else None,
                confirmed=int(s[hit[0]]+4) if len(hit) else None,
                pre=pre, post=post, baseline_mad=mad)


def hinge_onset(steps, values, t_acq, t_l1):
    s, x = np.asarray(steps, int), np.asarray(values, float)
    mask = (s >= t_acq) & (s <= t_l1+20)
    s, x = s[mask], x[mask]
    if not np.array_equal(s, np.arange(t_acq, t_l1+21)) or not np.isfinite(x).all():
        return None
    grid = np.arange(t_acq+5, t_l1+16)
    h = np.maximum(s[None, :]-grid[:, None], 0).astype(float)
    hc, xc = h-h.mean(1, keepdims=True), x-x.mean()
    sx = (hc*hc).sum(1)
    loss = (xc*xc).sum() - (hc@xc)**2/sx
    # Numerical equality at a flat input must pick the earliest breakpoint.
    hits = np.flatnonzero(loss == loss.min())
    return int(grid[hits[0]])


def event_failures(steps, margins, t_acq):
    s, m = np.asarray(steps, int), np.asarray(margins, float)
    landmark = np.flatnonzero(s == t_acq+10)
    if not len(landmark):
        return None
    at_risk = m[landmark[0]] > 0
    wins = np.lib.stride_tricks.sliding_window_view(m < 0, 10, axis=0).sum(-1) >= 8
    grids = np.lib.stride_tricks.sliding_window_view(s, 10)
    wins &= (s[:-9] > t_acq+10).reshape((-1,)+ (1,)*(m.ndim-1))
    wins &= np.all(np.diff(grids, axis=1) == 1, axis=1).reshape((-1,)+(1,)*(m.ndim-1))
    observed = wins.any(0) & at_risk
    first = s[np.argmax(wins, axis=0)]
    return dict(at_risk=at_risk, observed=observed,
                times=np.where(observed, first, s[-1]-9))


def harrell_c(times, observed, covariate, strata=None):
    t, o, x = np.asarray(times), np.asarray(observed, bool), np.asarray(covariate, float)
    valid = np.isfinite(t) & np.isfinite(x)
    t, o, x = t[valid], o[valid], x[valid]
    comparable = (t[:, None] < t[None, :]) & o[:, None]
    if strata is not None:
        st = np.asarray(strata)[valid]
        comparable &= st[:, None] == st[None, :]
    n = int(comparable.sum())
    # Larger covariate = earlier failure; reversing x returns 1-C.
    credit = (x[:, None] > x[None, :]).astype(float) + .5*(x[:, None] == x[None, :])
    return dict(C=float(credit[comparable].mean()) if n >= 10 else None, comparable=n)


def clopper_pearson(k, n, alpha=.05):
    if not n:
        return [None, None]
    def solve(target, tail):
        lo, hi = 0., 1.
        for _ in range(60):
            p = (lo+hi)/2
            prob = sum(comb(n, j)*p**j*(1-p)**(n-j) for j in range(tail, n+1))
            if prob < target:
                lo = p
            else:
                hi = p
        return (lo+hi)/2
    return [0. if k == 0 else solve(alpha/2, k),
            1. if k == n else solve(1-alpha/2, k+1)]


def reference_category(onset, reference):
    if onset is None:
        return "NL2"
    if reference is None:
        return "REF-NL"
    return "EARLY" if onset < reference-25 else "DELAYED" if onset > reference+25 else "CONCURRENT"


def paired_onset_difference(kvm, rnd, t_acq, *, window=300):
    """Descriptive restricted onset difference; NL2 is censored at t_acq+300."""
    cap = t_acq+window
    if any(r.get("missing", False) or r.get("nl2") is None for r in (kvm, rnd)):
        return dict(difference=None, kvm=None, rnd=None, missing=True, censor_step=cap)
    def onset(row):
        return min(row["onset"], cap) if row["onset"] is not None else cap
    k, r = onset(kvm), onset(rnd)
    return dict(difference=k-r, kvm=k, rnd=r, kvm_censored=kvm["nl2"],
                rnd_censored=rnd["nl2"], missing=False, censor_step=cap)


def sequence_cluster_ci(terms, margins, steps, chunks, t_acq, t_l1, *, late=6, seed=0, draws=2000):
    """Bootstrap whole sequences; event/time correlation stays inside each cluster."""
    steps, terms, margins, chunks = map(np.asarray, (steps, terms, margins, chunks))
    base = (steps >= t_acq) & (steps <= t_acq+15)
    loss = (steps >= t_l1-5) & (steps <= t_l1+5)
    if base.sum() != 16 or loss.sum() != 11:
        return None
    mask = chunks == late
    n = mask.sum(1)
    if not n.sum():
        return None
    # Aggregate within each sequence first; never resample individual events.
    ts = ((terms[loss]*mask[None,:,:,None]).mean(0).sum(1)
          -(terms[base]*mask[None,:,:,None]).mean(0).sum(1)).astype(float)
    ms = ((margins[loss]*mask[None]).mean(0).sum(1)
          -(margins[base]*mask[None]).mean(0).sum(1)).astype(float)
    indices = np.random.default_rng([seed,4242,6]).integers(len(n),size=(draws,len(n)))
    denom = n[indices].sum(1)
    valid = denom > 0
    dt = ts[indices].sum(1)[valid]/denom[valid,None]
    dm = ms[indices].sum(1)[valid]/denom[valid]
    return dict(term_ci95=np.quantile(dt,[.025,.975],axis=0).T.tolist(),
                total_ci95=np.quantile(dm,[.025,.975]).tolist(), clusters=len(n), draws=draws,
                packed_input=True, note="fp16 shards; central D_j use fp64 summary sums")


def event_consistency(unit_descriptions):
    names = ["residual","write_norm","own_write","L1_margin","maximum_key_cosine",
             "address_margin","slow_margin","duplicate"]
    out = {}
    for name in names:
        fam = {}
        for family in ("FO","noise"):
            cs = [u["descriptive"]["harrell"][name]["C"] for u in unit_descriptions
                  if u["family"] == family and name in u.get("descriptive",{}).get("harrell",{})]
            cs = np.array([x for x in cs if x is not None])
            fam[family] = dict(n=len(cs), positive=int((cs >= .60).sum()),
                               negative=int((cs <= .40).sum()), evaluable=len(cs) >= 6)
        sign = [direction for direction in ("positive","negative")
                if all(r["evaluable"] and r[direction] >= ceil(.75*r["n"]) for r in fam.values())]
        out[name] = dict(families=fam, consistent=bool(sign), direction=sign[0] if sign else None,
                         multiplicity_corrected=False, descriptive=True)
    return out
