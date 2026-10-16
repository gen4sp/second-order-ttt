"""Pre-registered R4-2L seed-paired analysis.

Usage: python scripts/analyze_r42l.py RUN.log OUT_DIR [--s5-fp32 S5.json]

Every label is the pre-registered one: V-L; LK and LM with their five-way
precedence; LS with the vW manipulation check; LX descriptively; D1-D3 plus the D4
component table; ML3 read jointly per seed; NC with its window check; and SC with
evaluable losing seeds, Delta_p, Delta_LR and the always-reported per-seed times.
All tests are one-sided exact McNemar on paired seeds; binary intervals are the
conservative exact >=95% projection of two exact Clopper-Pearson intervals and the
median intervals are the exact order-statistic ones (no asymptotic approximation).
A label with fewer than 10 evaluable pairs is "pending".

``--s5-fp32`` applies the R4-2 fallback (an R4-2 pre-launch amendment): a cell
whose in-run S5 failed is judged by an fp32 CPU recomputation on its final checkpoint
(a list of {cell_id, passed} records).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_r42 import binom_cdf, contrast, last_bound, probe, ratio  # noqa: E402

ANCHOR = "e2e1emb"
LINKV = "e2e1emb_linkv"
LIN = "e2e1emb_lin"
LINVW = "e2e1emb_linvW"
LINXKV = "e2e1emb_linxkv"
W600 = "e2e1emb_w600"
CTRL = "e2e1emb_ctrl"
ARMS = (ANCHOR, LINKV, LIN, LINVW, LINXKV, W600, CTRL)
ECHO_ARMS = (ANCHOR, LINKV, LIN, LINVW, LINXKV, W600)
SEEDS = list(range(10))
N_KEYS = 16
T_ECHO = 512
WARMUP_ANCHOR = 200
WARMUP_W600 = 600
WINDOW = 150
D4_FIELDS = ("upd_W1", "upd_b1", "upd_W2", "upd_b2", "eta_W1", "eta_b1", "eta_W2",
             "eta_b2", "addr_a", "clipped_frac_c0", "inner_norm_c0")
PROBE_FIELDS = ("p_kv", "e_kv_abs", "e_kv_ce_abs", "kv_write_eff", "Wv_write_eff",
                "vW_write_eff", "e_vW_abs", "clipped_frac_c0", "addr_a", "ce_kv",
                "ce_vk", "ce_vW")


# --------------------------------------------------------------------------- log


def records(path, s5_override=None):
    """CELL_RESULT / PROBE / REF lines of one run log, plus the registered pairing checks."""
    cells, probes, refs, settings = {}, {}, {}, {}
    for line in Path(path).open():
        if line.startswith("config: "):
            settings = json.loads(line[len("config: "):]).get("ttt", {})
        elif line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            cells[(c["seed"], c["arm"])] = c
        elif line.startswith("PROBE "):
            p = json.loads(line[len("PROBE "):])
            probes.setdefault(p["cell_id"], {})[p["step"]] = p
        elif line.startswith("REF echo_ctrl "):
            refs["echo_ctrl"] = json.loads(line[len("REF echo_ctrl "):])
        elif line.startswith("REF echo "):
            refs["echo"] = json.loads(line[len("REF echo "):])
    for c in cells.values():
        # A named CPU smoke validates its own layout, never the registered endpoint.
        smoke = (settings.get("process_name") == "TTM-r42L-00-smoke" and
                 settings.get("device") == "cpu" and settings.get("workers") == 1 and
                 settings.get("steps", 1500) <= 30)
        c["_smoke"] = smoke
        c["_T"] = settings.get("T", T_ECHO) if smoke else T_ECHO
        c["_n_keys"] = settings.get("n_keys", N_KEYS) if smoke else N_KEYS
        c["_warmup"] = settings.get("warmup", WARMUP_ANCHOR) if smoke else WARMUP_ANCHOR
        c["_probe_every"] = settings.get("probe_every", 25)
        probes.setdefault(c["cell_id"], {}).update({p["step"]: p for p in c.get("probes", [])})
        if s5_override and c["cell_id"] in s5_override:
            c["s5_in_run"] = dict(c.get("s5", {}))
            c["s5"] = dict(c.get("s5", {}), passed=bool(s5_override[c["cell_id"]]),
                           source="fp32-cpu")
    issues = pairing_issues(cells)
    issues.extend(f"diverged cell: arm={c['arm']} seed={c['seed']} task={c['task']}"
                  for c in cells.values() if c.get("diverged"))
    return cells, probes, refs, issues


def want_lin_per_seq(arm: str, T=T_ECHO, n_keys=N_KEYS) -> int:
    """The lin positions per sequence the registered design expects of ``arm``."""
    if arm in (LINKV, LINVW):
        return n_keys
    return T - n_keys if arm == LINXKV else T if arm == LIN else 0


def pairing_issues(cells: dict) -> list[str]:
    """The three registered pairing checks; a violation un-pairs the offending cells."""
    issues = []
    for s in SEEDS:
        group = [cells[s, a] for a in ECHO_ARMS
                 if (s, a) in cells and not cells[s, a].get("error")]
        hashes = {c.get("first_batch_sha1") for c in group}
        if len(hashes) > 1 or (group and any(not c.get("first_batch_sha1") for c in group)):
            issues.append(f"seed {s}: first_batch_sha1 differs across the six echo arms "
                          f"{sorted(map(str, hashes))}")
            for c in group:
                c["unpaired"] = True
        for key, c in sorted(cells.items()):
            if key[0] != s or c.get("error"):
                continue
            if c["arm"] not in ARMS:
                continue
            want_task = "echo_ctrl" if c["arm"] == CTRL else "echo"
            if c.get("task") != want_task or not c.get("first_batch_sha1"):
                issues.append(f"{c['cell_id']}: missing hash or wrong task (want {want_task})")
                c["unpaired"] = True
            want = want_lin_per_seq(c["arm"], c.get("_T", T_ECHO), c.get("_n_keys", N_KEYS))
            if c.get("lin_per_seq") != want:
                issues.append(f"{c['cell_id']}: lin_per_seq {c.get('lin_per_seq')} != {want}")
                c["unpaired"] = True
            want_warmup = WARMUP_W600 if c["arm"] == W600 else c.get("_warmup", WARMUP_ANCHOR)
            if c.get("warmup") != want_warmup:
                issues.append(f"{c['cell_id']}: warmup {c.get('warmup')} != {want_warmup}")
                c["unpaired"] = True
    return issues


def valid(c) -> bool:
    """A cell a contrast may use: finished, correctly paired, scored and S5-passed."""
    return bool(c and not c.get("error") and not c.get("skipped") and not c.get("unpaired")
                and "eval" in c and "bound_ever" in c and
                "bound_final" in c and c.get("s5", {}).get("passed") is True)


def seeds_of(cells: dict, arm: str) -> list[int]:
    """Seeds with a usable cell for ``arm``."""
    return [s for s in SEEDS if valid(cells.get((s, arm)))]


def paired(cells: dict, a: str, b: str) -> list[tuple]:
    """Seeds where both arms are usable, with their two cells."""
    return [(s, cells[s, a], cells[s, b]) for s in SEEDS
            if valid(cells.get((s, a))) and valid(cells.get((s, b)))]


# ------------------------------------------------------------------ small helpers


def counts(cells: dict, arm: str) -> dict:
    """Acquisition, retention and the observed sample size of one arm."""
    live = seeds_of(cells, arm)
    return dict(n=len(live), seeds=live, ever=sum(cells[s, arm]["bound_ever"] for s in live),
                final=sum(cells[s, arm]["bound_final"] for s in live))


def acc(c) -> float | None:
    """A cell's final query accuracy."""
    return None if not c else (c.get("eval") or {}).get("query_acc")


def bound_steps(c) -> list[int]:
    """Quick-eval steps whose query accuracy is at least 0.90."""
    return [r[0] for r in (c or {}).get("quick", []) if r[3] is not None and r[3] >= .90]


def t_lose(c) -> int | None:
    """First quick eval below 0.90 after the cell's last bound step (registered SC/NC rule)."""
    live = bound_steps(c)
    if not live:
        return None
    later = [r[0] for r in c.get("quick", []) if r[0] > max(live) and r[3] is not None
             and r[3] < .90]
    return later[0] if later else None


def t_p90(probes: dict, c) -> int | None:
    """First probe whose median ``p_kv`` reaches 0.90 (registered SC rule)."""
    series = probes.get(c["cell_id"], {}) if c else {}
    return next((step for step in sorted(series) if (series[step].get("p_kv") or -1) >= .90),
                None)


def accuracy_at_p90(probes: dict, c) -> tuple[int | None, float | None]:
    """Quick accuracy at the first probe with ``p_kv`` >= 0.90 (both grids are 25 steps)."""
    step = t_p90(probes, c)
    if step is None:
        return None, None
    row = next((r for r in (c or {}).get("quick", []) if r[0] == step), None)
    return step, (row[3] if row else None)


def mcnemar(cells: dict, a: str, b: str, field: str = "bound_final") -> dict:
    """One-sided exact McNemar of ``a > b`` over the paired seeds, with its exact CI."""
    return contrast(paired(cells, a, b), field)


def median_ci(values, alpha: float = .05) -> list | None:
    """Exact distribution-free 95% interval for a median (order statistics)."""
    xs = sorted(v for v in values if v is not None)
    n = len(xs)
    if not n:
        return None
    ranks = [i for i in range(1, n + 1) if binom_cdf(i - 1, n, .5) <= alpha / 2]
    # With n < 6, even [min, max] has less than 95% coverage. None denotes
    # an unbounded endpoint rather than falsely claiming a finite exact interval.
    if not ranks:
        return [None, None]
    k = max(ranks)
    return [xs[k - 1], xs[n - k]]


def final_ratio(probes: dict, a, b, field: str):
    """``field`` of ``a`` over ``field`` of ``b`` at their final probes."""
    pa, pb = probe(probes, a), probe(probes, b)
    if not pa or not pb:
        return None
    return ratio(pa.get(field), pb.get(field))


# ----------------------------------------------------------------------- labels


def site_label(cells: dict, arm: str, prefix: str) -> dict:
    """LK or LM: the five registered rules, evaluated in order, first match is the label."""
    info = counts(cells, arm)
    ret = contrast(paired(cells, arm, ANCHOR), "bound_final")
    common = [s for s in SEEDS if valid(cells.get((s, arm))) and
              valid(cells.get((s, ANCHOR))) and cells[s, arm]["bound_ever"] and
              cells[s, ANCHOR]["bound_ever"]]
    partial = (sum((acc(cells[s, arm]) or 0.) >= .50 for s in common) -
               sum((acc(cells[s, ANCHOR]) or 0.) >= .50 for s in common))
    label = "pending"
    if ret["n"] == len(SEEDS) and not any(a.get("_smoke") for _, a, _ in
                                         paired(cells, arm, ANCHOR)):
        if info["ever"] <= 3:
            label = f"{prefix}-blocked"
        elif ret["mcnemar_one_sided"] <= .05 and info["final"] >= 5:
            label = f"{prefix}-rescue"
        elif partial >= 4:
            label = f"{prefix}-partial"
        elif info["final"] <= 2 and info["ever"] >= 7:
            label = f"{prefix}-no rescue shown"
        else:
            label = f"{prefix}-inconclusive"
    return dict(label=label, ever=info["ever"], final=info["final"], n=info["n"],
                n_pairs=ret["n"], mcnemar=ret["mcnemar_one_sided"], ci95=ret["ci95"],
                difference=ret["difference"], partial_diff=partial,
                n_both_acquired=len(common))


def specificity(cells: dict, probes: dict, lk: str) -> dict:
    """LS: site specificity at matched dose against linvW; needs LK-rescue."""
    info = counts(cells, LINVW)
    linkv_vs = mcnemar(cells, LINKV, LINVW)
    linvw_vs = mcnemar(cells, LINVW, ANCHOR)
    took = sorted(s for s, a, b in paired(cells, LINVW, ANCHOR)
                  if (final_ratio(probes, a, b, "vW_write_eff") or -1) >= 10)
    label = "specificity unresolved"
    if lk == "pending" or min(linkv_vs["n"], linvw_vs["n"]) < 10:
        missing = [arm for arm in (LINKV, LINVW) if len(paired(cells, arm, ANCHOR)) < 10]
        if lk == "pending" and LINKV not in missing:
            missing.insert(0, LINKV)
        label = f"pending ({', '.join(missing)} not evaluable)"
    elif lk != "LK-rescue":
        label = f"not applicable (needs LK-rescue, got {lk})"
    elif info["final"] <= 2 and linkv_vs["mcnemar_one_sided"] <= .05 and len(took) >= 8:
        label = "kv-specific"
    elif info["final"] >= 5 and linvw_vs["mcnemar_one_sided"] <= .05:
        label = "not kv-specific"
    return dict(label=label, linvW_ever=info["ever"], linvW_final=info["final"],
                linkv_vs_linvW=linkv_vs, linvW_vs_anchor=linvw_vs,
                vW_take_seeds=len(took), vW_take_need=8, seeds_vW_took=took)


def dissociation(cells: dict, probes: dict, arm: str) -> dict:
    """D1-D3, paired by seed at the final probe: value tests and paired ratios."""
    rows = paired(cells, arm, ANCHOR)
    checks = {
        "D1": ("p_kv", None, lambda v: v >= .95, "median p_kv(arm) >= 0.95 in >= 8/10 seeds"),
        "D2": ("e_kv_ce_abs", "e_kv_ce_abs", lambda v: v <= 3,
               "e_kv_ce_abs(arm)/e_kv_ce_abs(anchor) <= 3 in >= 8/10 seeds"),
        "D3": ("kv_write_eff", "kv_write_eff", lambda v: v >= 10,
               "kv_write_eff(arm)/kv_write_eff(anchor) >= 10 in >= 8/10 seeds"),
    }
    per_seed = {}
    for name, (field_a, field_b, test, _) in checks.items():
        flags = []
        for _, a, b in rows:
            pa, pb = probe(probes, a), probe(probes, b)
            if not pa or not pb or pa.get(field_a) is None or pb.get(field_b or field_a) is None:
                flags.append(False)
                continue
            value = ratio(pa[field_a], pb[field_b]) if field_b else pa[field_a]
            flags.append(bool(value is not None and test(value)))
        per_seed[name] = flags
    full = len(rows) == len(SEEDS)
    out = dict(n_pairs=len(rows))
    for name, (_, _, _, rule) in checks.items():
        out[name] = None if not full else sum(per_seed[name]) >= 8
        out[f"{name}_seeds"] = sum(per_seed[name])
        out[f"{name}_rule"] = rule
    out["per_seed"] = per_seed
    out["seeds"] = [s for s, _, _ in rows]
    out["label"] = "evaluable" if full else "pending"
    return out


def ml3(cells: dict, arm: str, floor: float | None) -> dict:
    """ML3 read jointly per seed: both ablation conditions inside the same seed."""
    retained = [s for s in seeds_of(cells, arm) if cells[s, arm]["bound_final"]]

    def passes(s) -> bool:
        c = cells[s, arm]
        ab = c.get("ablation") or {}
        return bool(c.get("ablation_ref_match") is True and floor is not None and
                    ab.get("all_minus_kv", math.inf) <= floor + .10 and
                    ab.get("kv_only", -math.inf) >= .80)

    ok = [s for s in retained if passes(s)]
    full = len(paired(cells, arm, ANCHOR)) == 10 and not any(
        a.get("_smoke") for _, a, _ in paired(cells, arm, ANCHOR))
    observed = None if not retained else len(ok) * 3 >= 2 * len(retained)
    return dict(passed=observed if full else None, observed_passed=observed,
                label="evaluable" if full else "pending",
                retained=len(retained), pass_count=len(ok), seeds_pass=ok, floor=floor,
                rule="all-kv <= occ_minus + 0.10 and kv-only >= 0.80 in >= 2/3 of retained "
                     "seeds; a missing, mismatched or failing ablation fails the seed")


def no_copy(cells: dict, probes: dict, floor_ctrl: float | None) -> dict:
    """NC: the no-copy task against the anchor, with the pre-loss starvation check."""
    info = counts(cells, CTRL)
    ret = mcnemar(cells, CTRL, ANCHOR)
    checks = []
    for s, c, _ in paired(cells, CTRL, ANCHOR):
        if not c["bound_ever"] or c["bound_final"]:
            continue
        series = probes.get(c["cell_id"], {})
        first, lose = min(bound_steps(c)), t_lose(c)
        window = ([series[step] for step in sorted(series) if first <= step <= lose]
                  if lose else [])
        interval = c.get("_probe_every", 25)
        expected = set(range(first, lose + 1, interval)) if lose and interval > 0 else set()
        complete = bool(expected) and expected <= set(series) and lose in series
        p_ok = complete and all((p.get("p_kv") if p.get("p_kv") is not None else 1.) <= .20
                                    for p in window)
        eff = ratio((series.get(lose) or {}).get("kv_write_eff"),
                    (series.get(first) or {}).get("kv_write_eff"))
        eff_ok = eff is not None and eff >= .5
        checks.append(dict(seed=s, t_first_bound=first, t_lose=lose, n_window_probes=len(window),
                           complete_window=complete,
                           p_kv_le_020=p_ok, eff_ratio_at_lose=eff, eff_ge_half=eff_ok,
                           passed=bool(p_ok and eff_ok)))
    kept = [s for s in seeds_of(cells, CTRL) if cells[s, CTRL]["bound_final"]]
    kept_ml3 = ml3(cells, CTRL, floor_ctrl)
    losing_ok = (sum(c["passed"] for c in checks) * 3 >= 2 * len(checks) if checks else None)
    window_ok = bool(losing_ok) and (not kept or kept_ml3["passed"] is True)
    label = "pending"
    if ret["n"] == len(SEEDS):
        if info["ever"] <= 3:
            label = "NC-blocked"
        elif info["ever"] >= 5 and info["final"] <= 2 and window_ok:
            label = "NC-loses"
        elif (info["ever"] >= 5 and info["final"] >= 5 and
              ret["mcnemar_one_sided"] <= .05):
            label = "NC-retains"
        else:
            label = "NC-mixed"
    return dict(label=label, ever=info["ever"], final=info["final"], n=info["n"],
                n_pairs=ret["n"], mcnemar=ret["mcnemar_one_sided"], ci95=ret["ci95"],
                n_losing=len(checks), losing_window_ok=losing_ok, window_ok=window_ok,
                checks=checks, retained_seeds=kept, retained_ml3=kept_ml3)


def schedule(cells: dict, probes: dict) -> dict:
    """SC: the schedule control with Delta_p, Delta_LR and the per-seed times."""
    info = counts(cells, W600)
    ret = mcnemar(cells, W600, ANCHOR)
    rows = []
    for arm in (W600, ANCHOR):
        for s in seeds_of(cells, arm):
            c = cells[s, arm]
            step90, at90 = accuracy_at_p90(probes, c)
            rows.append(dict(arm=arm, seed=s, t_lose=t_lose(c), t_p90=step90,
                             t_peak=c["warmup"],
                             bound_ever=c["bound_ever"], bound_final=c["bound_final"],
                             acc_at_p90=at90))
    evaluable = {arm: [r for r in rows if r["arm"] == arm and r["t_lose"] is not None and
                       r["t_p90"] is not None] for arm in (W600, ANCHOR)}
    d_p = d_lr = None
    if evaluable[W600] and evaluable[ANCHOR]:
        d_p = (median([r["t_lose"] - r["t_p90"] for r in evaluable[W600]]) -
               median([r["t_lose"] - r["t_p90"] for r in evaluable[ANCHOR]]))
        d_lr = (median([r["t_lose"] - r["t_peak"] for r in evaluable[W600]]) -
                median([r["t_lose"] - r["t_peak"] for r in evaluable[ANCHOR]]))
    label = "pending"
    if ret["n"] == len(SEEDS):
        if info["ever"] <= 3:
            label = "SC-uninformative"
        elif info["final"] >= 5 and ret["mcnemar_one_sided"] <= .05:
            label = "SC-retains"
        elif (len(evaluable[W600]) >= 5 and len(evaluable[ANCHOR]) >= 5 and
              d_p is not None and d_lr is not None):
            if abs(d_p) <= 100 and abs(d_lr) >= 200:
                label = "SC-tracks-p"
            elif abs(d_lr) <= 100 and abs(d_p) >= 200:
                label = "SC-tracks-LR"
            else:
                label = "SC-ambiguous"
        else:
            label = "SC-mixed"
    def median_record(values):
        values = [v for v in values if v is not None]
        return dict(n=len(values), median=median(values) if values else None,
                    ci95=median_ci(values))

    final_p = [probe(probes, cells[s, W600]).get("p_kv") for s in seeds_of(cells, W600)
               if probe(probes, cells[s, W600])]
    final_p = [p for p in final_p if p is not None]
    return dict(label=label, w600_ever=info["ever"], w600_final=info["final"],
                n_pairs=ret["n"], mcnemar=ret["mcnemar_one_sided"], ci95=ret["ci95"],
                delta_p=d_p, delta_lr=d_lr,
                n_evaluable={arm: len(evaluable[arm]) for arm in evaluable},
                n_losing={arm: sum(1 for r in rows if r["arm"] == arm and
                                   r["t_lose"] is not None) for arm in (W600, ANCHOR)},
                n_losing_without_p90={arm: sum(r["arm"] == arm and r["t_lose"] is not None
                                               and r["t_p90"] is None for r in rows)
                                       for arm in (W600, ANCHOR)},
                median_final_p_kv=median(final_p) if final_p else None,
                retains_high_p_kv=label == "SC-retains" and bool(final_p) and median(final_p) >= .95,
                times=rows,
                medians={f"{arm}_{key}": median_record([r[key] for r in rows if r["arm"] == arm])
                         for arm in (W600, ANCHOR)
                         for key in ("t_lose", "t_p90", "t_peak")},
                acc_at_p90={arm: median_record([r["acc_at_p90"] for r in rows if r["arm"] == arm])
                            for arm in (W600, ANCHOR)})


# ----------------------------------------------------------------- D4 and tables


def alternative_route(row: dict) -> bool:
    """Descriptive flag for large update, clipping or address changes (never gating)."""
    def moved(field):
        r = row.get(f"{field}_final_ratio")
        return r is not None and (r >= 10 or r <= .1)
    addr = row.get("addr_a_final_ratio")
    return bool(any(moved(field) for field in ("clipped_frac_c0", "upd_W1", "upd_b1",
                                              "upd_W2", "upd_b2")) or
                (addr is not None and addr <= .5))


def d4_table(cells: dict, probes: dict, arm: str, windows: dict) -> list[dict]:
    """D4: the arm against the anchor over the anchor's loss window and at the final probe."""
    rows = []
    for s, a, b in paired(cells, arm, ANCHOR):
        t_s = windows.get(s)
        sa, sb = probes.get(a["cell_id"], {}), probes.get(b["cell_id"], {})
        steps = sorted(set(sa) & set(sb))
        window = [step for step in steps if t_s is not None and t_s <= step <= t_s + WINDOW]
        final = max(steps) if steps else None
        row = dict(seed=s, arm=arm, t_s=t_s, n_window_probes=len(window), final_probe=final)
        for field in D4_FIELDS:
            fa = (sa.get(final) or {}).get(field)
            fb = (sb.get(final) or {}).get(field)
            wa = [sa[step][field] for step in window if sa[step].get(field) is not None]
            wb = [sb[step][field] for step in window if sb[step].get(field) is not None]
            row[f"{field}_final_arm"] = fa
            row[f"{field}_final_anchor"] = fb
            row[f"{field}_final_ratio"] = ratio(fa, fb)
            row[f"{field}_window_arm"] = float(median(wa)) if wa else None
            row[f"{field}_window_anchor"] = float(median(wb)) if wb else None
        row["alt_route_flag"] = alternative_route(row)
        rows.append(row)
    return rows


def per_seed_rows(cells: dict, probes: dict) -> list[dict]:
    """The per-seed paired table every R4-2L report shows."""
    rows = []
    for s in SEEDS:
        row = {"seed": s}
        for arm in ARMS:
            c = cells.get((s, arm))
            ok = valid(c)
            p = probe(probes, c) if ok else None
            row.update({f"{arm}_{key}": (c.get(key) if ok else None) for key in
                        ("bound_ever", "bound_final", "collapse", "degraded", "task",
                         "inner_loss", "lin_per_seq", "warmup", "first_batch_sha1",
                         "ablation_ref_match", "tf32", "peak_cuda_mb", "masked_per_seq")})
            row[f"{arm}_s5_passed"] = (c.get("s5") or {}).get("passed") if ok else None
            row[f"{arm}_query_acc"] = acc(c) if ok else None
            row[f"{arm}_last_bound_step"] = last_bound(c) if ok else None
            row[f"{arm}_t_lose"] = t_lose(c) if ok else None
            for condition in ("all", "all_minus_kv", "kv_only"):
                row[f"{arm}_ablation_{condition}"] = ((c.get("ablation") or {}).get(condition)
                                                      if ok else None)
            for field in PROBE_FIELDS + ("eta_W1", "eta_b1", "eta_W2", "eta_b2",
                                         "upd_W1", "upd_b1", "upd_W2", "upd_b2"):
                row[f"{arm}_{field}"] = (p.get(field) if p else None)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- main


def summarize(cells: dict, probes: dict, refs: dict, issues=()) -> dict:
    """Every pre-registered R4-2L label with the counts, p-values and intervals behind it."""
    anchor = counts(cells, ANCHOR)
    v = "pending" if anchor["n"] < len(SEEDS) else (
        "holds" if anchor["ever"] >= 7 and anchor["final"] <= 3 else "not reproduced")
    descriptive = v != "holds"
    floor = ((refs.get("echo") or {}).get("occ_minus") or {}).get("acc")
    floor_ctrl = ((refs.get("echo_ctrl") or {}).get("occ_minus") or {}).get("acc")

    lk = site_label(cells, LINKV, "LK")
    lm = site_label(cells, LIN, "LM")
    ls = specificity(cells, probes, lk["label"])
    lx = dict(counts(cells, LINXKV), vs_anchor=mcnemar(cells, LINXKV, ANCHOR),
              vs_lin=mcnemar(cells, LINXKV, LIN),
              note="descriptive: changes T - 16 positions, not a matched-dose control")
    d = {arm: dissociation(cells, probes, arm) for arm in (LINKV, LIN, LINVW, LINXKV)}
    ml = {arm: ml3(cells, arm, floor_ctrl if arm == CTRL else floor) for arm in ARMS}
    nc = no_copy(cells, probes, floor_ctrl)
    sc = schedule(cells, probes)

    anchor_windows = {s: last_bound(cells[s, ANCHOR]) for s in seeds_of(cells, ANCHOR)}
    acquired = [t for t in anchor_windows.values() if t is not None]
    fallback = 25 * round(median(acquired) / 25) if acquired else None
    windows = {s: (t if t is not None else fallback) for s, t in anchor_windows.items()}
    d4 = {arm: d4_table(cells, probes, arm, windows) for arm in (LINKV, LIN, LINVW, LINXKV)}

    def annotate(label: str) -> str:
        """Every contrast is descriptive unless V-L holds."""
        if descriptive and not label.startswith(("pending", "not applicable", "descriptive")):
            return f"{label} (descriptive: V-L {v})"
        return label

    raw = {"V-L": v, "LK": lk["label"], "LM": lm["label"], "LS": ls["label"],
           "LX": "descriptive", "NC": nc["label"], "SC": sc["label"]}
    labels = {k: annotate(val) for k, val in raw.items()}
    labels.update({
        "D": {arm: {key: q[key] for key in ("D1", "D2", "D3", "D1_seeds", "D2_seeds",
                                             "D3_seeds", "n_pairs")}
              for arm, q in d.items()},
        "ML3": {arm: {key: q[key] for key in ("passed", "retained", "pass_count")}
                for arm, q in ml.items()},
        "descriptive": descriptive,
        "joint_with_r42": ("R4-2L rescue = LK-rescue with D1-D3 and ML3; the joint reading "
                           "with R4-2's M1-M3 is pre-registered and applied once both "
                           "runs are analysed"),
    })
    contrasts = {name: mcnemar(cells, a, b) for name, a, b in (
        ("linkv_anchor", LINKV, ANCHOR), ("lin_anchor", LIN, ANCHOR),
        ("linkv_linvW", LINKV, LINVW), ("linvW_anchor", LINVW, ANCHOR),
        ("linxkv_anchor", LINXKV, ANCHOR), ("linxkv_lin", LINXKV, LIN),
        ("w600_anchor", W600, ANCHOR), ("ctrl_anchor", CTRL, ANCHOR))}
    contrasts.update({f"acq_{name}": mcnemar(cells, a, b, "bound_ever") for name, a, b in (
        ("linkv_anchor", LINKV, ANCHOR), ("lin_anchor", LIN, ANCHOR),
        ("w600_anchor", W600, ANCHOR), ("ctrl_anchor", CTRL, ANCHOR))})
    return dict(labels=labels, validation_only=any(c.get("_smoke") for c in cells.values()),
                median_ci_unbounded_endpoint=None,
                n_pairs={arm: len(paired(cells, arm, ANCHOR)) for arm in ARMS},
                counts={arm: counts(cells, arm) for arm in ARMS}, contrasts=contrasts,
                LK=lk, LM=lm, LS=ls, LX=lx, D=d, ML3=ml, NC=nc, SC=sc, D4=d4,
                anchor_windows=anchor_windows, anchor_window_fallback=fallback,
                occ_minus_echo=floor, occ_minus_echo_ctrl=floor_ctrl,
                issues=list(issues), per_seed=per_seed_rows(cells, probes),
                refs={k: v for k, v in refs.items() if k in ("echo", "echo_ctrl")})


def write_csv(path: Path, rows: list[dict]) -> None:
    """Flat CSV over the union of all row keys."""
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--s5-fp32", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    override = ({r["cell_id"]: r["passed"] for r in json.loads(args.s5_fp32.read_text())}
                if args.s5_fp32 else None)
    result = summarize(*records(args.log, override))
    flat_d4 = [row for rows in result["D4"].values() for row in rows]
    (args.out / "summary.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "per_seed"}, indent=2, sort_keys=True,
        default=str) + "\n")
    write_csv(args.out / "paired_seeds.csv", result["per_seed"])
    write_csv(args.out / "d4_components.csv", flat_d4)
    write_csv(args.out / "sc_times.csv", result["SC"]["times"])
    write_csv(args.out / "nc_window.csv", result["NC"]["checks"])
    print(json.dumps({"labels": result["labels"], "n_pairs": result["n_pairs"],
                      "issues": result["issues"]}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
