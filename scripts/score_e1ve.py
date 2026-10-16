"""Score E1-VE run logs against the registered predictions.

Usage: ``python scripts/score_e1ve.py wave1.log [wave2.log ...] [--json out.json]``.
Reads every ``CELL_RESULT`` line, applies the validity gates, prints a per-group
table and the verdicts S (supported), F (falsified), I (inconclusive) or NE (not
evaluable: fewer than 4 evaluable cells or pairs). ``--selftest`` checks the rules
on synthetic cells. Standard library only.

Registered count rule, for ``n`` of ``k`` evaluable cells (or seed pairs) showing the
predicted outcome: S if ``n >= ceil(5k/6)``, F if ``n <= floor(k/2)``, otherwise I.
The two never overlap, and k = 6 gives the registered 5/6 and 3/6.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict

TYPES = ("U", "S", "A", "X", "N")
RET = 0.90
MIN_EVAL = 4
ETAS = (10.0, 40.0)
DOSES = (0.0, 0.01, 0.03, 0.05, 0.10)   # the registered slip grid; 0 is wave 1's FO-CE


def load(paths: list[str]) -> list[dict]:
    cells = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if line.startswith("CELL_RESULT "):
                    rec = json.loads(line.split(" ", 1)[1])
                    cells[rec["cell_id"]] = rec          # a rerun of a cell replaces it
    return list(cells.values())


def gates(rec: dict) -> dict:
    f = rec.get("final") or {}
    v1 = ("final" in rec and not rec.get("diverged") and rec.get("stopped") is None
          and rec["steps_done"] == rec["steps"]
          and rec.get("nonfinite_steps", 0) <= 0.01 * rec["steps_done"])
    v2 = v1 and f.get("resid0_wb") is not None and f["resid0_wb"] <= 0.10
    v3 = v1 and f.get("acc_U", 0) >= RET and f.get("acc_S", 0) >= RET
    return dict(V1=v1, V2=v2, V3=v3)


def group(cells: list[dict]) -> dict:
    out = defaultdict(list)
    for rec in cells:
        if "final" in rec:
            key = (rec["order"], rec["rule"] + ("-edge" if rec.get("edge") else ""), rec["eta"],
                   rec["slip"])
            out[key].append(rec)
    return out


def need(k: int) -> int:
    """Support threshold: '>= 5/6' scaled to k."""
    return math.ceil(5 * k / 6)


def count_verdict(n: int, k: int) -> str:
    if k < MIN_EVAL:
        return "NE"
    return "S" if n >= need(k) else "F" if n <= k // 2 else "I"


def all_types_verdict(counts: dict, k: int) -> str:
    """S if every type meets the count rule, F if any type is at or below half."""
    if k < MIN_EVAL:
        return "NE"
    if all(n >= need(k) for n in counts.values()):
        return "S"
    return "F" if any(n <= k // 2 for n in counts.values()) else "I"


def retained(cells: list[dict], t: str) -> int:
    return sum(c["final"][f"acc_{t}"] >= RET for c in cells)


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def evaluable(cells: list[dict], need: tuple[str, ...]) -> list[dict]:
    return [c for c in cells if all(gates(c)[g] for g in need)]


def first_step(rows: list[list], cols: list[str], cond) -> int | None:
    for row in rows:
        r = dict(zip(cols, row))
        if cond(r):
            return r["step"]
    return None


def score(cells: list[dict]) -> dict:
    g = group(cells)
    out: dict = {}
    for eta in sorted({k[2] for k in g}):
        fo = g.get(("fo", "ce", eta, 0.0), [])
        so = g.get(("so", "ce", eta, 0.0), [])
        lin = g.get(("fo", "linev", eta, 0.0), [])
        aice = g.get(("fo", "aice", eta, 0.0), [])
        edge = g.get(("fo", "ce-edge", eta, 0.0), [])
        e = f"eta{eta:g}"
        fo3 = evaluable(fo, ("V1", "V2", "V3"))
        k = len(fo3)
        # the cheap path at the world writes themselves (W0 residual), for the reading of P1/P2
        cheap = dict(median_resid0_A=median(c["final"]["resid0_A"] for c in fo3),
                     median_resid0_N=median(c["final"]["resid0_N"] for c in fo3))
        cheap["cheap_path_at_world_writes"] = all(
            v is not None and v <= .10 for v in cheap.values())
        nN = retained(fo3, "N")
        out[f"P1_{e}"] = dict(verdict=count_verdict(k - nN, k), N_retained=nN, k=k, **cheap)
        med_stale = median(c["final"]["stale_err_A"] for c in fo3)
        nA = retained(fo3, "A")
        v = count_verdict(k - nA, k)
        if v == "S" and (med_stale is None or med_stale < .80):
            v = "F" if med_stale is not None and med_stale < .50 else "I"
        elif v == "I" and med_stale is not None and med_stale < .50:
            v = "F"
        out[f"P2_{e}"] = dict(verdict=v, A_retained=nA, median_stale_err_A=med_stale, k=k, **cheap)
        nX = retained(fo3, "X")
        out[f"P3_{e}"] = dict(verdict=count_verdict(nX, k), X_retained=nX, k=k)
        # remedies: the endpoint over every completed cell (V1), and the subset in which the
        # workbench copy formed (V1 and V2); a rescue is established only if both are S
        for name, arm, types in (("P4", so, ("A", "N")), ("P5", lin, TYPES), ("P6", aice, TYPES)):
            res = {}
            for label, need_gates in (("all", ("V1",)), ("copy", ("V1", "V2"))):
                ev = evaluable(arm, need_gates)
                counts = {t: retained(ev, t) for t in types}
                res[label] = dict(verdict=all_types_verdict(counts, len(ev)), retained=counts,
                                  k=len(ev))
            va, vc = res["all"]["verdict"], res["copy"]["verdict"]
            v = va if va in ("NE", "F") else "S" if va == vc == "S" else "I"
            out[f"{name}_{e}"] = dict(verdict=v, unconditional=res["all"], copy_formed=res["copy"])
        # P12, the lesion: same gates as FO-CE, plus a manipulation check that the lesion
        # removed the cheap path at the C writes
        ed3 = evaluable(edge, ("V1", "V2", "V3"))
        kk = len(ed3)
        r0 = [median(c["final"][f"resid0_{t}"] for c in ed3) for t in ("A", "N")]
        counts = {t: retained(ed3, t) for t in ("A", "N")}
        v = all_types_verdict(counts, kk)
        if v != "NE" and not all(r is not None and r >= .50 for r in r0):
            v = "NE"                                   # the lesion leaked: not a test
        out[f"P12_{e}"] = dict(verdict=v, retained=counts, k=kk, median_resid0_A=r0[0],
                               median_resid0_N=r0[1],
                               lesion_retained_USX={t: retained(ed3, t) for t in "USX"},
                               fo_ce_retained_USX={t: retained(fo3, t) for t in "USX"})
        so2 = {c["seed"]: c for c in evaluable(so, ("V1", "V2"))}
        fo_by = {c["seed"]: c for c in fo3}
        pairs = sorted(set(so2) & set(fo_by))
        # every P7 quantity comes from the seeds valid in both arms
        oldp_A, oldp_X = (median(fo_by[s]["final"]["oldp_A"] for s in pairs),
                          median(fo_by[s]["final"]["oldp_X"] for s in pairs))
        pa = sum(so2[s]["final"]["oldp_A"] > fo_by[s]["final"]["oldp_A"] for s in pairs)
        if k < MIN_EVAL or len(pairs) < MIN_EVAL:
            v = "NE"
        elif oldp_A >= .10:
            v = "F"
        elif oldp_A <= .02 and oldp_X >= .20 and pa >= need(len(pairs)):
            v = "S"
        else:
            v = "I"
        out[f"P7_{e}"] = dict(verdict=v, fo_oldp_A=oldp_A, fo_oldp_X=oldp_X, so_gt_fo_pairs=pa,
                              pairs=len(pairs), paired_seeds=pairs)
        pr = sum(so2[s]["final"]["resid0_A"] > fo_by[s]["final"]["resid0_A"] for s in pairs)
        out[f"P8_{e}"] = dict(verdict=count_verdict(pr, len(pairs)), so_gt_fo_pairs=pr,
                              pairs=len(pairs))
        # P9: acquisition, copy and loss are first crossings at the quick checks; a cell is
        # scorable once it acquired and then lost N. Ordered means t_acq < t_copy < t_loss;
        # a copy at the same check as either is a tie and is not scored
        tally = dict(ordered=0, not_ordered=0, tie=0, never_acquired=0, never_lost=0)
        crossings = {}
        for c in fo3:
            cols, rows = c["quick_cols"], c["quick"]
            t_acq = first_step(rows, cols, lambda r: r["acc_N"] >= RET)
            t_loss = None if t_acq is None else first_step(
                rows, cols, lambda r: r["step"] > t_acq and r["acc_N"] < .5)
            t_copy = first_step(rows, cols, lambda r: r["resid0_wb"] < .5)
            crossings[c["seed"]] = dict(acquired=t_acq, copy=t_copy, lost=t_loss)
            if t_acq is None:
                tally["never_acquired"] += 1
            elif t_loss is None:
                tally["never_lost"] += 1
            elif t_copy is not None and t_copy in (t_acq, t_loss):
                tally["tie"] += 1
            elif t_copy is not None and t_acq < t_copy < t_loss:
                tally["ordered"] += 1
            else:
                tally["not_ordered"] += 1
        m = tally["ordered"] + tally["not_ordered"]
        v = ("NE" if m < MIN_EVAL else "S" if tally["ordered"] >= math.ceil(2 * m / 3) else
             "F" if tally["not_ordered"] >= math.ceil(m / 2) else "I")
        out[f"P9_{e}"] = dict(verdict=v, scored=m, **tally, crossings=crossings)
    if any(k[:2] == ("fo", "ce") and k[3] > 0 for k in g):
        out.update(score_wave2(g))
    return out


def dose_status(cells: list[dict], t: str) -> dict:
    """'ok' (type retained in >= 4 evaluable cells), 'fail', or 'unknown' (< 4 evaluable)."""
    ev = evaluable(cells, ("V1", "V3"))
    n = retained(ev, t)
    status = "unknown" if len(ev) < MIN_EVAL else "ok" if n >= 4 else "fail"
    return dict(status=status, k=len(ev), retained=n)


def threshold(statuses: list[str]) -> tuple[int | None, str]:
    """Index of the smallest dose that is 'ok' with every larger dose.

    'measured' needs the next lower dose to be a measured failure (or the dose to be
    the lowest); 'censored' means the largest dose fails, so the threshold lies above
    the grid; 'NE' means missing evidence decides it.
    """
    if statuses[-1] == "unknown":
        return None, "NE"
    if statuses[-1] == "fail":
        return None, "censored"
    i = len(statuses) - 1
    while i > 0 and statuses[i - 1] == "ok":
        i -= 1
    if i > 0 and statuses[i - 1] == "unknown":
        return None, "NE"
    return i, "measured"


def combine(verdicts: list[str]) -> str:
    return "F" if "F" in verdicts else "S" if all(v == "S" for v in verdicts) else "NE"


def score_wave2(g: dict) -> dict:
    """P10 and P11 on the registered grid; score wave 1 and wave 2 logs together."""
    out, thr = {}, {}
    for eta in ETAS:
        for t in ("N", "A"):
            per = {f"{d:g}": dose_status(g.get(("fo", "ce", eta, d), []), t) for d in DOSES}
            thr[(eta, t)] = threshold([v["status"] for v in per.values()])
            i, kind = thr[(eta, t)]
            out[f"threshold_{t}_eta{eta:g}"] = dict(kind=kind, index=i,
                                                    dose=None if i is None else DOSES[i],
                                                    by_dose=per)
    p10a, p10b, p11 = [], None, []
    for eta in ETAS:
        i, kind = thr[(eta, "N")]
        p10a.append("NE" if kind == "NE" else "S" if kind == "measured" and DOSES[i] == .03
                    else "F")
        (ia, ka), (iN, kN) = thr[(eta, "A")], thr[(eta, "N")]
        if "NE" in (ka, kN) or ka == kN == "censored":
            p11.append("NE")
        elif kN == "censored":
            p11.append("F")                            # A measured, N above the grid
        elif ka == "censored":
            p11.append("S")
        else:
            p11.append("S" if ia >= iN else "F")
    kinds = [thr[(eta, "N")] for eta in ETAS]
    if any(k == "NE" for _, k in kinds) or all(k == "censored" for _, k in kinds):
        p10b = "NE"
    elif all(k == "measured" for _, k in kinds):
        p10b = "S" if kinds[0][0] == kinds[1][0] else "F"
    else:
        p10b = "F"                                     # one measured, one above the grid
    out["P10a"] = dict(verdict=combine(p10a), per_eta=p10a)
    out["P10b"] = dict(verdict=p10b, delta_index=None if p10b == "NE" or None in
                       (kinds[0][0], kinds[1][0]) else kinds[1][0] - kinds[0][0])
    out["P11"] = dict(verdict=combine(p11), per_eta=p11)
    return out


def exclusions(cells: list[dict]) -> list[str]:
    """Every cell that fails a gate or has no result, with the reason."""
    lines = []
    for rec in sorted(cells, key=lambda r: r["cell_id"]):
        if "final" not in rec:
            lines.append(f"{rec['cell_id']}: no result ({rec.get('error')})")
            continue
        failed = [k for k, ok in gates(rec).items() if not ok]
        if failed:
            lines.append(f"{rec['cell_id']}: fails {','.join(failed)} (stopped={rec.get('stopped')}, "
                         f"resid0_wb={rec['final'].get('resid0_wb')}, "
                         f"U={rec['final'].get('acc_U')}, S={rec['final'].get('acc_S')})")
    return lines


def table(cells: list[dict]) -> str:
    lines = ["order rule    eta slip  n  V1 V2 V3 | retained U S A X N | med stale_err_A oldp_A "
             "oldp_X resid0_A resid0_N resid0_wb act_A act_X ms/step"]
    for key, cs in sorted(group(cells).items()):
        gs = [gates(c) for c in cs]
        ev = [c for c, gg in zip(cs, gs) if gg["V1"]]
        med = lambda f: median(c["final"].get(f) for c in ev)
        fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
        lines.append(f"{key[0]:5} {key[1]:7} {key[2]:3g} {key[3]:4g} {len(cs):2d}  "
                     f"{sum(g['V1'] for g in gs)}  {sum(g['V2'] for g in gs)}  "
                     f"{sum(g['V3'] for g in gs)} | " +
                     " ".join(str(retained(ev, t)) for t in TYPES) + " | " +
                     " ".join(fmt(med(f)) for f in ("stale_err_A", "oldp_A", "oldp_X", "resid0_A",
                                                    "resid0_N", "resid0_wb", "act_A", "act_X")) +
                     f" {fmt(median(c.get('ms_step') for c in cs))}")
    return "\n".join(lines)


def _synthetic(order, rule, eta, slip, seed, acc, edge=False, **extra):
    final = {f"acc_{t}": a for t, a in zip(TYPES, acc)}
    final.update(stale_err_A=.95, oldp_A=.001, oldp_X=.5, resid0_A=.002, resid0_N=.002,
                 resid0_wb=.001, act_A=1.0, act_X=1.0)
    final.update(extra)
    rows = [[25 * i, 1.0, 1.0, .5, 1.0, 1.0 if i < 36 else .05, *[0] * 10,
             .9 if i < 33 else .01, 0, 0, 0, 0, 0] for i in range(1, 81)]
    cols = ["step", "acc_U", "acc_S", "acc_A", "acc_X", "acc_N", "stale_A", "stale_X", "oldp_X",
            "oldp_A", "resid0_A", "resid0_N", "x1", "x2", "x3", "x4", "resid0_wb", "act_A",
            "act_X", "act_N", "act_E", "train_loss"]
    rows = [r[:16] + [r[16]] + [0, 0, 0, 0, 0] for r in rows]
    tag = f"{order}-{rule}{'-edge' if edge else ''}/eta{eta:g}/slip{slip:g}/s{seed}"
    return dict(cell_id=tag, order=order, rule=rule, edge=edge,
                eta=eta, slip=slip, seed=seed, steps=2000, steps_done=2000, stopped=None,
                diverged=False, nonfinite_steps=0, final=final, quick_cols=cols, quick=rows,
                ms_step=100.0)


def selftest() -> None:
    for k in range(MIN_EVAL, 7):
        assert need(k) > k // 2                       # support and falsification are disjoint
    assert [count_verdict(n, 6) for n in (6, 5, 4, 3)] == ["S", "S", "I", "F"]
    assert [count_verdict(n, 4) for n in (4, 3, 2)] == ["S", "I", "F"]
    assert count_verdict(3, 3) == "NE"
    cells = []
    for eta in (10.0, 40.0):
        for s in range(10, 16):
            cells.append(_synthetic("fo", "ce", eta, 0.0, s, (1, 1, .01, 1, .08)))
            cells.append(_synthetic("so", "ce", eta, 0.0, s, (1, 1, 1, 1, 1), oldp_A=.1,
                                    resid0_A=.1))
            cells.append(_synthetic("fo", "linev", eta, 0.0, s, (1, 1, 1, 1, 1)))
            cells.append(_synthetic("fo", "aice", eta, 0.0, s, (1, 1, 1, 1, .5 if s < 13 else 1)))
            cells.append(_synthetic("fo", "ce", eta, 0.0, s, (1, 1, 1, 1, 1), edge=True,
                                    resid0_A=.9, resid0_N=.9))
            for slip, ok in ((.01, False), (.03, True), (.05, True), (.10, True)):
                cells.append(_synthetic("fo", "ce", eta, slip, s,
                                        (1, 1, 1 if ok else .2, 1, 1 if ok else .2)))
    res = score(cells)
    want = dict(P1_eta10="S", P2_eta10="S", P3_eta10="S", P4_eta10="S", P5_eta10="S",
                P6_eta10="F", P7_eta10="S", P8_eta10="S", P9_eta10="S", P12_eta10="S", P10a="S",
                P10b="S", P11="S")
    got = {k: res[k]["verdict"] for k in want}
    assert got == want, (got, want)
    # a lesion that leaves the copy in place is not a test of P12
    leaky = [dict(c, final=dict(c["final"], resid0_A=.01)) if c.get("edge") else c
             for c in cells]
    assert score(leaky)["P12_eta10"]["verdict"] == "NE"
    # two lost FO-CE cells: k = 4 still scores, with the scaled thresholds
    short = [dict(c, stopped="deadline") if (c["rule"], c["slip"], c["seed"]) == ("ce", 0.0, 10)
             or (c["rule"], c.get("edge"), c["slip"], c["seed"]) == ("ce", False, 0.0, 11)
             else c for c in cells]
    res4 = score(short)
    assert res4["P1_eta10"]["k"] == 4 and res4["P1_eta10"]["verdict"] == "S"
    # P7 medians come from the paired seeds only: S would be wrong
    fo_oldp = dict(zip(range(10, 16), (.001, .001, .2, .2, .001, .001)))
    p7 = [_synthetic("fo", "ce", 10.0, 0.0, s, (1, 1, .01, 1, .08), oldp_A=fo_oldp[s])
          for s in range(10, 16)]
    p7 += [_synthetic("so", "ce", 10.0, 0.0, s, (1, 1, 1, 1, 1), oldp_A=.3,
                      resid0_wb=.001 if s < 14 else .5) for s in range(10, 16)]
    assert score(p7)["P7_eta10"]["verdict"] == "F"
    # wave 2: a missing or unevaluable dose is not a threshold result
    w2 = lambda c: c["slip"] > 0
    drop = lambda d: [c for c in cells if not (w2(c) and c["slip"] == d)]
    for d in (.01, .10):
        assert score(drop(d))["P10a"]["verdict"] == "NE", d
    three = [c for c in cells if not (w2(c) and c["slip"] == .01 and c["seed"] > 12)]
    assert score(three)["P10a"]["verdict"] == "NE"
    assert threshold(["fail"] * 5) == (None, "censored")
    assert threshold(["fail", "fail", "ok", "ok", "ok"]) == (2, "measured")
    none = [dict(c, final=dict(c["final"], acc_A=.2, acc_N=.2)) if w2(c) else c for c in cells]
    r = score(none)
    assert r["P11"]["verdict"] == "NE" and r["P10a"]["verdict"] == "F"
    assert r["P10b"]["verdict"] == "NE"
    print("selftest ok:", got)


def main(argv: list[str]) -> None:
    if "--selftest" in argv:
        selftest()
        return
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        del argv[i:i + 2]
    cells = load(argv[1:])
    print(table(cells))
    print("exclusions:")
    for line in exclusions(cells) or ["none"]:
        print("  " + line)
    res = score(cells)
    for key, val in res.items():
        print(f"{key}: {json.dumps(val)}")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(dict(scores=res, n_cells=len(cells)), fh, indent=1)


if __name__ == "__main__":
    main(sys.argv)
