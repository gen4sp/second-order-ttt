"""P3-a analysis (numpy only): per-cell table, FO-SO gaps with seed-level 90 % CIs, verdict D.

Registered statistics:

- ``A`` = final binding accuracy at each key's first query on the 256 fixed eval
  sequences, read with the cell's own rule and multiplier (step 1500).
- ``gap_r = A_FO - A_SO`` per seed (paired: same init, frozen code and data stream).
- ``DiD_{r,r'} = gap_r - gap_r'`` per seed for (lin, ce), (own, ce), (own, lin).
- 90 % two-sided t-interval over seeds (df = n - 1).
- **D-SUPPORTED** iff some pair has an echo DiD interval excluding 0 with a point estimate
  outside the ROPE (|mean| > .10), *and* on echo_ctrl both rules of that pair have FO-SO
  gap intervals inside the ROPE [-.10, .10].
  **D-NOT-SUPPORTED** otherwise.  **NE** if any main cell is undefined (diverged or
  non-finite), if fewer than 3 seeds have all 12 configurations, or if the CE anchor
  fails (FO-CE echo retained > floor(n/6), or SO-CE echo retained < n - floor(n/6)).
  A pair is NE if one of its rules is non-functional (best (task, order) cell seed-mean
  recall < .50); NE overall if no pair is evaluable.
  With 3-5 complete seeds the verdict is marked interim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RULES = ("ce", "lin", "own")
ORDERS = ("FO", "SO")
TASKS = ("echo", "echo_ctrl")
PAIRS = (("lin", "ce"), ("own", "ce"), ("own", "lin"))
ROPE = 0.10
LEVEL = 0.90
# a rule is functional iff its best (task, order) cell has seed-mean final recall >= .50;
# pairs with a non-functional rule are NE (a rule that never writes usable memory cannot
# show FO/SO sensitivity, and its zero gap would make any DiD a floor artefact)
FUNCTIONAL = 0.50
FULL_SEEDS = 6
# one-sided 95 % Student-t quantiles (two-sided 90 % intervals)
T95 = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432, 7: 1.8946,
       8: 1.8595, 9: 1.8331, 10: 1.8125, 11: 1.7959, 12: 1.7823, 13: 1.7709, 14: 1.7613,
       15: 1.7531, 16: 1.7459, 17: 1.7396, 18: 1.7341, 19: 1.7291, 20: 1.7247, 25: 1.7081,
       30: 1.6973}


def t_quantile(df: int) -> float:
    if df < 1:
        raise ValueError("df must be positive")
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    for lo, hi in zip(keys, keys[1:]):
        if lo < df < hi:
            return T95[hi]  # conservative: the smaller df's quantile is larger
    return 1.6973


def ci(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return dict(n=0, mean=None, sd=None, lo=None, hi=None, values=[])
    mean = float(v.mean())
    if n < 2:
        return dict(n=n, mean=mean, sd=None, lo=None, hi=None, values=v.tolist())
    sd = float(v.std(ddof=1))
    half = t_quantile(n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, lo=mean - half, hi=mean + half, values=v.tolist())


def excludes_zero(c: dict) -> bool:
    return c.get("lo") is not None and (c["lo"] > 0 or c["hi"] < 0)


def within(c: dict, rope: float = ROPE) -> bool:
    return c.get("lo") is not None and c["lo"] >= -rope and c["hi"] <= rope


def load_cells(root: Path) -> dict[tuple, dict]:
    cells = {}
    for f in sorted((Path(root) / "main").glob("*/result.json")):
        rec = json.loads(f.read_text())
        cells[(rec["task"], rec["rule"], rec["order"], int(rec["seed"]))] = rec
    return cells


def _median(vals: list) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else None


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _qual(rec: dict, path: tuple[str, str], *, final: bool) -> float | None:
    rows = rec.get("qualifiers") or []
    vals = [r[path[0]].get(path[1]) for r in rows]
    if final:
        return vals[-1] if vals else None
    return _mean(vals)


def _pooled_first_k(root: Path, recs: list[dict], g_clip: float) -> float | None:
    vals = []
    for rec in recs:
        f = Path(root) / "main" / rec["key"] / "first_k.npz"
        if not f.exists():
            return None
        with np.load(f) as z:
            pre = z["first_k_preclip"]
        vals.append(np.minimum(pre[np.isfinite(pre)], g_clip).reshape(-1))
    return float(np.median(np.concatenate(vals))) if vals else None


def analyze(root: str | Path, rope: float = ROPE) -> dict:
    root = Path(root)
    cells = load_cells(root)
    mult_file = root / "cal" / "multipliers.json"
    multipliers = json.loads(mult_file.read_text()) if mult_file.exists() else None
    configs = [(t, r, o) for t in TASKS for r in RULES for o in ORDERS]
    seeds = sorted({k[3] for k in cells})
    complete = [s for s in seeds if all((t, r, o, s) in cells for t, r, o in configs)]
    undefined = sorted("/".join(map(str, k)) for k, rec in cells.items() if not rec.get("defined"))

    def acc(t, r, o, s):
        return cells[(t, r, o, s)]["final"]["own"]["acc"]

    table, trajectories = [], {}
    for t, r, o in configs:
        recs = [cells[(t, r, o, s)] for s in seeds if (t, r, o, s) in cells]
        if not recs:
            continue
        steps = recs[0].get("steps", 1500)
        accs = [x["final"]["own"]["acc"] for x in recs]
        late = [_mean([q["acc_own"] for q in x.get("quick", []) if q["step"] > 2 * steps / 3])
                for x in recs]
        table.append(dict(
            task=t, rule=r, order=o, n=len(recs), mult=recs[0]["mult"],
            acc_mean=_mean(accs), acc_sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
            acc_per_seed={str(x["seed"]): x["final"]["own"]["acc"] for x in recs},
            acc_ce_read_mean=_mean([x["final"]["ce"]["acc"] for x in recs]),
            query_ce_mean=_mean([x["final"]["own"]["query_ce"] for x in recs]),
            retained=sum(bool(x.get("retained")) for x in recs),
            retained_ce_read=sum(bool(x.get("retained_ce")) for x in recs),
            ever_acquired=sum(bool(x.get("ever_acquired")) for x in recs),
            t_acq_median=_median([x.get("t_acq") for x in recs]),
            t_loss_n=sum(x.get("t_loss") is not None for x in recs),
            late_quick_acc_mean=_mean(late),
            diverged=sum(bool(x.get("diverged")) for x in recs),
            nonfinite_steps=sum(int(x.get("nonfinite_steps", 0)) for x in recs),
            ms_step_median=_median([x.get("ms_step") for x in recs]),
            wall_s_median=_median([x.get("wall_s") for x in recs]),
            first_k_post_median=_median([x["first_k"]["post_median"] for x in recs]),
            first_k_clip_frac=_mean([x["first_k"]["clip_frac"] for x in recs]),
            train_clip_frac=_mean([x["train_tel"]["clip_frac"] for x in recs]),
            eval_clip_frac=_mean([x["final"]["own"]["eval_clip_frac"] for x in recs]),
            cos_all_final=_median([_qual(x, ("write", "cos_all"), final=True) for x in recs]),
            cos_kv_final=_median([_qual(x, ("write", "cos_kv"), final=True) for x in recs]),
            cos_all_traj=_median([_qual(x, ("write", "cos_all"), final=False) for x in recs]),
            cos_kv_traj=_median([_qual(x, ("write", "cos_kv"), final=False) for x in recs]),
            norm_ratio_all_final=_median([_qual(x, ("write", "norm_ratio_all"), final=True) for x in recs]),
            so_rel_all_final=_median([_qual(x, ("so_term", "rel_all"), final=True) for x in recs]),
            so_rel_head_final=_median([_qual(x, ("so_term", "rel_head"), final=True) for x in recs]),
            so_rel_all_traj=_median([_qual(x, ("so_term", "rel_all"), final=False) for x in recs]),
        ))
        cs = [cells[(t, r, o, s)] for s in complete]
        if cs and cs[0].get("quick"):
            grid = [q["step"] for q in cs[0]["quick"]]
            if all([q["step"] for q in c.get("quick", [])] == grid for c in cs):
                trajectories.setdefault(t, {}).setdefault(r, {})[o] = dict(
                    steps=grid, acc_own_mean=np.mean([[q["acc_own"] for q in c["quick"]] for c in cs],
                                                     axis=0).round(5).tolist())
    gaps = {t: {r: ci([acc(t, r, "FO", s) - acc(t, r, "SO", s) for s in complete]) for r in RULES}
            for t in TASKS}
    did = {t: {f"{a}-{b}": ci([(acc(t, a, "FO", s) - acc(t, a, "SO", s))
                               - (acc(t, b, "FO", s) - acc(t, b, "SO", s)) for s in complete])
               for a, b in PAIRS} for t in TASKS}
    n = len(complete)
    allowed = n // 6
    fo = sum(bool(cells[("echo", "ce", "FO", s)].get("retained")) for s in complete)
    so = sum(bool(cells[("echo", "ce", "SO", s)].get("retained")) for s in complete)
    anchor = dict(n=n, fo_ce_echo_retained=fo, so_ce_echo_retained=so, allowed_fo_max=allowed,
                  required_so_min=n - allowed, ok=bool(n and fo <= allowed and so >= n - allowed))
    g_clip = float(next(iter(cells.values()))["g_clip"]) if cells else 10.0
    realized = {}
    ce_recs = [cells[(t, "ce", o, s)] for t in TASKS for o in ORDERS for s in complete]
    ce_med = _pooled_first_k(root, ce_recs, g_clip) if ce_recs else None
    for r in RULES:
        med = _pooled_first_k(root, [cells[(t, r, o, s)] for t in TASKS for o in ORDERS for s in complete],
                              g_clip) if complete else None
        realized[r] = dict(first_k_post_median=med,
                           ratio_to_ce=(med / ce_med) if med is not None and ce_med else None)
    best = {r: (max(_mean([acc(t, r, o, s) for s in complete]) for t in TASKS for o in ORDERS)
                if complete else None) for r in RULES}
    functional = {r: dict(best_cell_acc_mean=best[r],
                          ok=bool(best[r] is not None and best[r] >= FUNCTIONAL)) for r in RULES}
    verdict = decide(n=n, undefined=undefined, anchor=anchor, gaps=gaps, did=did, rope=rope,
                     functional={r: v["ok"] for r, v in functional.items()})
    return dict(experiment="p3a", root=str(root), seeds=seeds, complete_seeds=complete,
                n_cells=len(cells), undefined=undefined, multipliers=multipliers, table=table,
                gaps=gaps, did=did, anchor=anchor, functional=functional,
                realized_norm=realized, verdict=verdict,
                trajectories=trajectories, rope=rope, level=LEVEL,
                wall_s_sum=float(sum(rec.get("wall_s", 0.0) for rec in cells.values())))


def decide(*, n: int, undefined: list, anchor: dict, gaps: dict, did: dict,
           rope: float = ROPE, functional: dict | None = None) -> dict:
    base = dict(n_seeds=n, interim=n < FULL_SEEDS, rope=rope, level=LEVEL, qualifying=[],
                ne_pairs=[])
    if undefined:
        return dict(base, label="NE", reason=f"undefined (diverged or non-finite) cells: {undefined}")
    if n < 3:
        return dict(base, label="NE", reason="fewer than 3 seeds with all 12 configurations")
    if not anchor["ok"]:
        return dict(base, label="NE", reason=f"CE anchor not reproduced: {anchor}")
    functional = functional or {r: True for r in RULES}
    ne_pairs = [f"{a}-{b}" for a, b in PAIRS if not (functional[a] and functional[b])]
    base["ne_pairs"] = ne_pairs
    if len(ne_pairs) == len(PAIRS):
        return dict(base, label="NE", reason="no evaluable pair: non-functional rule(s) "
                    f"{[r for r in RULES if not functional[r]]} (best cell mean recall < {FUNCTIONAL})")
    qualifying = []
    for a, b in PAIRS:
        name = f"{a}-{b}"
        if name in ne_pairs:
            continue
        e = did["echo"][name]
        if excludes_zero(e) and abs(e["mean"]) > rope:
            ctrl = {r: within(gaps["echo_ctrl"][r], rope) for r in (a, b)}
            qualifying.append(dict(pair=name, echo_did_mean=e["mean"], echo_did_lo=e["lo"],
                                   echo_did_hi=e["hi"], ctrl_gap_within_rope=ctrl,
                                   ctrl_did_within_rope=within(did["echo_ctrl"][name], rope),
                                   ok=all(ctrl.values())))
    supported = any(q["ok"] for q in qualifying)
    if supported:
        reason = "echo DiD excludes 0 (|mean| > ROPE) and both echo_ctrl gaps lie in the ROPE for " + \
                 ", ".join(q["pair"] for q in qualifying if q["ok"])
    elif qualifying:
        reason = "echo DiD excludes 0 only where an echo_ctrl gap is not inside the ROPE"
    else:
        reason = "no echo DiD interval excludes 0 with |mean| > ROPE"
    if ne_pairs:
        reason += f"; NE pairs (non-functional rule): {ne_pairs}"
    return dict(base, label="D-SUPPORTED" if supported else "D-NOT-SUPPORTED",
                reason=reason, qualifying=qualifying)


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _ci_str(c: dict) -> str:
    if c.get("lo") is None:
        return f"{_f(c.get('mean'))} (n {c.get('n')})"
    return f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def render_md(res: dict) -> str:
    v = res["verdict"]
    lines = [f"# P3-a analysis ({res['root']})", "",
             f"**Verdict: {v['label']}**{' (interim, n = ' + str(v['n_seeds']) + ')' if v.get('interim') else ''}"
             f" — {v.get('reason')}", "",
             f"Seeds with all 12 configurations: {res['complete_seeds']} (cells loaded: {res['n_cells']}).", ""]
    m = res.get("multipliers") or {}
    if m:
        lines += ["## Multipliers", "",
                  f"lin {_f(m.get('lin'), 5)}, own {_f(m.get('own'), 5)}, ce 1 (source {m.get('source')}; "
                  f"K = {m.get('K')}; CE target {_f(m.get('target'), 5)})", ""]
        for rd in m.get("rounds", []):
            lines.append(f"- CAL round {rd['round']}: m {json.dumps(rd['m'])}, median "
                         f"{json.dumps(rd.get('median'))}, ms/step {_f(rd.get('ms_step_median'), 1)}")
        lines.append("")
    lines += ["## Per-cell table (final eval binding accuracy, own-rule read; 256 sequences)", "",
              "| task | rule | order | m | n | acc mean ± sd | per seed | retained | CE-read acc | ever acq | t_acq med | late quick acc | diverged |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in res["table"]:
        per = " ".join(f"{float(a):.2f}" for a in row["acc_per_seed"].values())
        lines.append(f"| {row['task']} | {row['rule']} | {row['order']} | {_f(row['mult'], 4)} | {row['n']} | "
                     f"{_f(row['acc_mean'])} ± {_f(row['acc_sd'])} | {per} | {row['retained']}/{row['n']} | "
                     f"{_f(row['acc_ce_read_mean'])} | {row['ever_acquired']}/{row['n']} | "
                     f"{_f(row['t_acq_median'], 0)} | {_f(row['late_quick_acc_mean'])} | {row['diverged']} |")
    lines += ["", "## FO − SO gap per rule (A_FO − A_SO; mean and 90 % seed-level t-interval)", "",
              "| task | ce | lin | own |", "|---|---|---|---|"]
    for t in TASKS:
        lines.append(f"| {t} | " + " | ".join(_ci_str(res["gaps"][t][r]) for r in RULES) + " |")
    lines += ["", "## Difference of gaps between rules (DiD; 90 % interval)", "",
              "| task | lin−ce | own−ce | own−lin |", "|---|---|---|---|"]
    for t in TASKS:
        lines.append(f"| {t} | " + " | ".join(_ci_str(res["did"][t][f"{a}-{b}"]) for a, b in PAIRS) + " |")
    a = res["anchor"]
    lines += ["", f"CE anchor: FO-CE echo retained {a['fo_ce_echo_retained']}/{a['n']} (max {a['allowed_fo_max']}), "
              f"SO-CE echo retained {a['so_ce_echo_retained']}/{a['n']} (min {a['required_so_min']}) → "
              f"{'ok' if a['ok'] else 'FAILED'}", "",
              f"Functional rules (best cell seed-mean recall ≥ {FUNCTIONAL}): " +
              ", ".join(f"{r} {_f(x['best_cell_acc_mean'])} ({'ok' if x['ok'] else 'NON-FUNCTIONAL'})"
                        for r, x in res.get("functional", {}).items()), "",
              "## Qualifiers", "",
              "| task | rule | order | cos(write, CE) all / kv, final | traj | norm ratio | SO-term rel all / head, final | first-K post median | train clip | eval clip |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in res["table"]:
        lines.append(f"| {row['task']} | {row['rule']} | {row['order']} | {_f(row['cos_all_final'])} / "
                     f"{_f(row['cos_kv_final'])} | {_f(row['cos_all_traj'])} / {_f(row['cos_kv_traj'])} | "
                     f"{_f(row['norm_ratio_all_final'])} | {_f(row['so_rel_all_final'])} / "
                     f"{_f(row['so_rel_head_final'])} | {_f(row['first_k_post_median'], 5)} | "
                     f"{_f(row['train_clip_frac'], 4)} | {_f(row['eval_clip_frac'], 4)} |")
    lines += ["", "Realized first-K update-norm match on the main seeds (pooled median, ratio to CE): " +
              ", ".join(f"{r} {_f(x['first_k_post_median'], 5)} ({_f(x['ratio_to_ce'])})"
                        for r, x in res["realized_norm"].items()), "",
              f"Sum of cell wall times: {res['wall_s_sum'] / 3600:.2f} worker-hours.", ""]
    return "\n".join(lines)
