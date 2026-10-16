"""Named fresh-pool scoring adapters, independent of CHR's six-unit selection."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

from . import triggers
from .episodes import kaplan_meier

FO, SO, NOISE = "e2e1emb", "e2e2emb", "e2e1emb_cn97"
FAMILY = ("R-SO", "R-NOISE", "R-LIN", "R-RP", "R-ADDR")


def chr_analysis():
    name = "_frs_chr_analysis"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / "scripts/analyze_chr.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def exact_mcnemar(test, control) -> dict:
    test, control = np.asarray(test, dtype=bool), np.asarray(control, dtype=bool)
    if test.shape != control.shape or test.ndim != 1:
        raise ValueError("McNemar expects equally sized paired vectors")
    b, c = int((test & ~control).sum()), int((~test & control).sum())
    p = 1. if b + c == 0 else sum(math.comb(b+c, k) for k in range(b, b+c+1)) / 2 ** (b+c)
    return dict(n=len(test), test=int(test.sum()), comparator=int(control.sum()),
                favourable=b, adverse=c, both=int((test & control).sum()),
                neither=int((~test & ~control).sum()), p=p,
                ci95=dict(test=chr_analysis().exact_ci(int(test.sum()), len(test)),
                          comparator=chr_analysis().exact_ci(int(control.sum()), len(test))))


def holm5(pvalues: dict[str, float]) -> dict[str, float]:
    pairs = sorted(((name, float(pvalues.get(name, 1.))) for name in FAMILY), key=lambda x: (x[1], x[0]))
    out, maximum = {}, 0.
    for i, (name, p) in enumerate(pairs):
        maximum = max(maximum, min(1., (5-i)*p))
        out[name] = maximum
    return out


def paired_verdict(row: dict, adjusted: float) -> str:
    n, kt, kc = row["n"], row["test"], row["comparator"]
    if n < 10:
        return "NOT EVALUABLE"
    if adjusted <= .05:
        return "REPLICATED" if kt >= math.ceil(.75*n) and kc <= math.floor(.25*n) else "PARTIAL"
    if row["p"] > .05 and kt <= math.floor(.5*n):
        return "NOT REPLICATED"
    return "INCONCLUSIVE"


def count_rule(k: int, n: int, minimum: int = 6) -> dict:
    verdict = ("NOT EVALUABLE" if n < minimum else "REPLICATED" if k >= math.ceil(.75*n)
               else "NOT REPLICATED" if k <= math.floor(.5*n) else "INCONCLUSIVE")
    return dict(k=k, n=n, minimum=minimum, verdict=verdict, ci95=chr_analysis().exact_ci(k, n))


def outcome(record: dict | None, tag="ce") -> bool | None:
    if record is None or record.get("error") or record.get("status") == "MISSING":
        return None
    if record.get("diverged") or record.get("disposition") == "DIVERGED":
        return False
    acc = (record.get("final") or {}).get(tag, {}).get("acc")
    return None if acc is None or not np.isfinite(acc) else bool(acc >= .90)


def acquisition(record: dict | None) -> int | None:
    if record is None:
        return None
    event = triggers.detect(record.get("logs", []))
    return (event["t_acq"].event_step if "t_acq" in event else
            record.get('events', {}).get('t_acq', {}).get('event_step'))


def first_acquired(cells, limit=12, *, smoke=False):
    candidates = sorted([r for r in cells if r.get("arm") == FO], key=lambda r: int(r["seed"]))
    if smoke:
        return [r for r in candidates if "t_acq" in r.get("events", {}) or r.get('error')
                or r.get('status') == 'MISSING'][:limit]
    return [r for r in candidates if acquisition(r) is not None or r.get('error')
            or r.get('status') == 'MISSING'][:limit]


def paired_records(test, control, tag_test="ce", tag_control="ce") -> dict:
    units, missing = [], []
    for seed in sorted(set(test) | set(control)):
        a, b = outcome(test.get(seed), tag_test), outcome(control.get(seed), tag_control)
        if a is None or b is None:
            missing.append(seed)
        else:
            units.append(dict(seed=seed, test=a, comparator=b,
                              diverged=bool((test[seed].get("diverged") or control[seed].get("diverged")))))
    out = exact_mcnemar([u["test"] for u in units], [u["comparator"] for u in units])
    out.update(units=units, missing=missing)
    if out["n"] < 10:
        out["nominal_p"] = out["p"]
        out["p"] = 1.0  # NE enters the fixed family with p=1
    clean = [u for u in units if not u["diverged"]]
    out["without_divergence"] = exact_mcnemar([u["test"] for u in clean], [u["comparator"] for u in clean])
    return out


def h5_adapter(cells, jobs, *, limit=12, smoke=False, frozen_slots=None) -> dict:
    """Retained CE mains stay in; twins are not required; missing NC affects qualifiers only."""
    cohort = first_acquired(cells, limit, smoke=smoke)
    if frozen_slots is not None:
        byseed = {int(c['seed']): c for c in cells if c.get('arm') == FO}
        cohort = [dict(seed=int(slot['seed']), arm=FO, status='MISSING',
                       reason='frozen missing main slot') if slot.get('missing') else
                  byseed.get(int(slot['seed']), dict(seed=int(slot['seed']), arm=FO, status='MISSING'))
                  for slot in frozen_slots]
    by_job = {(int(j["seed"]), j["rule"]): j for j in jobs if "seed" in j and "rule" in j
              and (j.get('job_id', 'h5-').startswith('h5-'))}
    units = []
    test, comparator = {}, {}
    for main in cohort:
        seed = int(main["seed"])
        rp, nc = by_job.get((seed, "rp03")), by_job.get((seed, "nc03"))
        comparator[seed] = main
        if rp is not None:
            test[seed] = rp
        rp_rows = (rp or {}).get("logs", [])
        m1 = chr_analysis().train_write_ratio(rp_rows)
        clip = chr_analysis().train_fraction(rp_rows, "train_clip_count")
        cap = [r["train_cap_hit_mean"] for r in rp_rows if r.get("train_cap_hit_mean") is not None]
        cross_ce = outcome(main, "rp03")
        cross_nc = outcome(nc, "rp03")
        units.append(dict(seed=seed, acquired=acquisition(main), main_retained=outcome(main),
                          rp_retained=outcome(rp, "rp03"), nc_retained=outcome(nc, "nc03"),
                          ce_under_rp03=cross_ce, nc_under_rp03=cross_nc,
                          rp_under_ce=outcome(rp, "ce"), M1=m1, clip=clip,
                          nc_attained=chr_analysis().attained_ratio((nc or {}).get("logs", [])),
                          rp_cap_hit_median=float(np.median(cap)) if cap else None,
                          weak=m1 is None or m1 < 2, clip_confounded=clip is None or clip > .5))
    paired = paired_records(test, comparator, "rp03")
    paired.update(cohort=[int(m["seed"]) for m in cohort], h5_units=units)
    nc_seen = [u for u in units if u["nc_retained"] is not None]
    n_nc, k_nc = len(nc_seen), sum(u["nc_retained"] for u in nc_seen)
    paired["nc_qualifier"] = dict(n=n_nc, k=k_nc, ci95=chr_analysis().exact_ci(k_nc, n_nc),
                                   low=k_nc <= math.floor(.25*n_nc) if n_nc >= 10 else None,
                                   missing=[u["seed"] for u in units if u["nc_retained"] is None])
    paired["manipulation"] = chr_analysis().manipulation_qualifier([u["weak"] or u["clip_confounded"]
                                                                   for u in units])
    valid_ids = {u["seed"] for u in paired["units"]}
    used = [u for u in units if u["seed"] in valid_ids]
    if any(u["ce_under_rp03"] is None or u["nc_under_rp03"] is None for u in used):
        paired["selectivity"] = "unknown"
    elif not used:
        paired["selectivity"] = "unknown"
    else:
        k_eval = sum(u["ce_under_rp03"] for u in used)
        k_sel = sum(u["rp_retained"] and not u["ce_under_rp03"] and not u["nc_under_rp03"] for u in used)
        paired["selectivity"] = ("eval-carried" if k_eval >= len(used)-1 else
                                  "training-selective" if k_sel >= len(used)-1 else "mixed")
    nc_pairs = [u for u in units if u["rp_retained"] is not None and u["nc_retained"] is not None]
    paired["rp_vs_nc_secondary"] = exact_mcnemar([u["rp_retained"] for u in nc_pairs],
                                                 [u["nc_retained"] for u in nc_pairs])
    paired["rp_vs_nc_secondary"]["multiplicity"] = "unadjusted, outside Family A"
    paired['selectivity_descriptive'] = True
    return paired


def transient_adapter(cells, *, terminal=1500, grid=5) -> dict:
    ac = chr_analysis()
    groups = {}
    for arm in (FO, SO, NOISE):
        units = []
        for main in sorted((m for m in cells if m.get("arm") == arm), key=lambda m: int(m["seed"])):
            rows = main.get("logs", [])
            if main.get("error") or main.get("diverged") or not rows:
                continue
            events = triggers.detect(rows)
            if "t_acq" not in events:
                continue
            if not ac.grid_check(rows, grid, terminal, grid)["ok"]:
                continue
            loss = events.get("t_loss")
            t_re = None
            if loss:
                s = ac.series(rows)
                t_re = ac._first(s["a"] >= .90, s["step"], after=loss.event_step, gap=grid)
            units.append(dict(seed=int(main["seed"]), t_acq=events["t_acq"].event_step,
                              t_loss=None if loss is None else loss.event_step, t_re=t_re,
                              adequate=loss is not None and terminal-loss.event_step >= 300,
                              outage=None if loss is None else (t_re-loss.event_step if t_re is not None
                                                               else terminal-grid-loss.event_step),
                              recovered=t_re is not None, censored=loss is not None and t_re is None))
        groups[arm] = units
    fo = [u for u in groups[FO] if u["adequate"]]
    out = dict(fo_no_recovery=count_rule(sum(not u["recovered"] for u in fo), len(fo)),
               units=groups, incidence={}, recovery={}, km={})
    for arm in (SO, NOISE):
        all_units = groups[arm]
        lost = [u for u in all_units if u["t_loss"] is not None]
        n, m = len(all_units), len(lost)
        incidence_ok = n >= 6 and m >= math.ceil(.5*n)
        out["incidence"][arm] = dict(n_acq=n, losses=m, ci95=ac.exact_ci(m, n),
                                         verdict="REPLICATED" if incidence_ok else "NOT EVALUABLE",
                                         reason=None if incidence_ok else "loss too rare or n_acq < 6")
        row = count_rule(sum(u["recovered"] for u in lost), m)
        if not incidence_ok:
            row.update(verdict="NOT EVALUABLE", reason="loss too rare")
        out["recovery"][arm] = row
        out["km"][arm] = kaplan_meier([u["outage"] for u in lost], [u["recovered"] for u in lost])
    paired = []
    noise = {u["seed"]: u for u in groups[NOISE] if u["t_loss"] is not None}
    for so in groups[SO]:
        no = noise.get(so["seed"])
        if so["t_loss"] is None or no is None:
            continue
        if so["recovered"] and no["recovered"]:
            determined, success = True, so["outage"] < no["outage"]
        elif so["recovered"] and so["outage"] <= no["outage"]:
            determined, success = True, True
        elif no["recovered"] and no["outage"] <= so["outage"]:
            determined, success = True, False
        else:
            determined, success = False, False
        if determined:
            paired.append(dict(seed=so["seed"], success=success))
    out["ordinal"] = count_rule(sum(u["success"] for u in paired), len(paired))
    out["ordinal"]["units"] = paired
    parts = [out["fo_no_recovery"], *out["recovery"].values(), out["ordinal"]]
    evaluable = [r for r in parts if r['verdict'] != 'NOT EVALUABLE']
    out["verdict"] = ("NOT EVALUABLE" if not evaluable
                      else "REPLICATED" if all(r["verdict"] == "REPLICATED" for r in evaluable)
                      else "NOT REPLICATED" if any(r["verdict"] == "NOT REPLICATED" for r in parts)
                      else "INCONCLUSIVE")
    return out
