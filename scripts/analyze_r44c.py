"""Pre-registered R4-4c analysis. uv run --locked python scripts/analyze_r44c.py LOG OUT_DIR.

Pure standard library: importing/running this analyzer does not start torch.
OUT_DIR is a report folder; all tables go in its data/ subdirectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

RULES = {"e2e2emb": ("e2e2emb", "e2e1emb", "e2e1emb_lin"),
         "e2e2": ("e2e2", "e2e1", "e2e1_lin", "e2e1_code")}
SEEDS = {"e2e2emb": set(range(10)), "e2e2": {0, 5, 6, 7, 8, 11, 15, 17, 19}}
MODES = ("floor", "rewarm")
STEP0_TOLERANCE = 1 / 1024 + 1e-6
STEP0 = re.compile(r"^C1_STEP0 cell=(\S+) acc=(\S+) ref=(\S+) match=(True|False) exact=(True|False)$")
FINAL_EVAL_FIELDS = ("final_acc", "eval", "ablation", "ablation_ref_match")


def exact_mcnemar(a_only: int, b_only: int, alternative: str) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.
    if alternative == "greater":
        return sum(math.comb(n, k) for k in range(a_only, n + 1)) / 2**n
    if alternative == "two-sided":
        return min(1., 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n)
    raise ValueError(alternative)


def holm(rows: list[dict]) -> None:
    """All 15 planned tests form one family, even if some are unavailable."""
    ranked = sorted((r for r in rows if r["p_exact"] is not None), key=lambda r: r["p_exact"])
    previous = 0.
    for i, row in enumerate(ranked):
        previous = max(previous, min(1., (len(rows) - i) * row["p_exact"]))
        row["p_holm"] = previous
        row["reject_holm_05"] = previous < .05
    for row in rows:
        row.setdefault("p_holm", None)
        row.setdefault("reject_holm_05", None)


def records(path: Path):
    cells, checks, trajectories, probes = {}, {}, {}, {}
    for line in path.open():
        line = line.strip()
        if line.startswith("CELL_ERROR "):
            raise ValueError(f"failed cell in log: {line}")
        if line.startswith("POOL_DEADLINE"):
            raise ValueError(f"pool deadline in log: {line}")
        if line.startswith("CELL_RESULT "):
            row = json.loads(line[len("CELL_RESULT "):])
            ident = row["cell_id"]
            if ident in cells:
                raise ValueError(f"duplicate CELL_RESULT: {ident}")
            if row.get("final_eval_ok") is False:
                for key in FINAL_EVAL_FIELDS:
                    row.pop(key, None)
            cells[ident] = row
        elif line.startswith("C1_STEP0 "):
            match = STEP0.fullmatch(line)
            if match is None:
                raise ValueError(f"malformed step-0 check: {line}")
            ident, acc, ref, ok, exact = match.groups()
            row = dict(cell_id=ident, acc=float(acc), ref=float(ref), match=ok == "True",
                       exact=exact == "True")
            diff = abs(row["acc"] - row["ref"])
            if not row["match"] or not math.isfinite(diff) or diff > STEP0_TOLERANCE:
                raise ValueError(f"step-0 mismatch: {ident}")
            if row["exact"] != (diff <= 1e-6):
                raise ValueError(f"inconsistent step-0 exact flag: {ident}")
            if ident in checks:
                raise ValueError(f"duplicate C1_STEP0: {ident}")
            checks[ident] = row
        elif line.startswith("C1_QUICK "):
            row = json.loads(line[len("C1_QUICK "):])
            trajectories[(row["cell_id"], int(row["step"]))] = row
        elif line.startswith("PROBE "):
            row = json.loads(line[len("PROBE "):])
            probes[(row["cell_id"], int(row["step"]))] = row
    for ident, cell in cells.items():
        if ident not in checks:
            raise ValueError(f"missing C1_STEP0: {ident}")
        for row in cell.get("quick", []):
            key = (ident, int(row[0]))
            trajectories.setdefault(key, dict(cell_id=ident, step=int(row[0]), ce=row[1],
                query_ce=row[2], query_acc=row[3], train_loss=row[4],
                **dict(zip(("eta_W1", "eta_b1", "eta_W2", "eta_b2"), row[6:10]))))
        for row in cell.get("probes", []):
            probes.setdefault((ident, int(row["step"])), row)
    return cells, checks, trajectories, probes


def contrast(cells, label, init, a, b, mode_a, mode_b, alternative):
    left = {c["seed"]: c for c in cells.values() if c["init"] == init and
            c["rule"] == a and c["lr_mode"] == mode_a and c["steps"] == 375}
    right = {c["seed"]: c for c in cells.values() if c["init"] == init and
             c["rule"] == b and c["lr_mode"] == mode_b and c["steps"] == 375}
    paired = sorted(left.keys() & right.keys() & SEEDS[init])
    a_only = sum(left[s]["bound_final"] and not right[s]["bound_final"] for s in paired)
    b_only = sum(right[s]["bound_final"] and not left[s]["bound_final"] for s in paired)
    complete = set(paired) == SEEDS[init]
    return dict(test=label, init=init, a=a, b=b, mode_a=mode_a, mode_b=mode_b,
                alternative=alternative, n=len(paired), n_expected=len(SEEDS[init]),
                retained_a=sum(left[s]["bound_final"] for s in paired),
                retained_b=sum(right[s]["bound_final"] for s in paired),
                a_only=a_only, b_only=b_only, paired_seeds=json.dumps(paired),
                status="complete" if complete else "pending",
                p_exact=exact_mcnemar(a_only, b_only, alternative) if complete else None)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "NaN" if value is None else
                             json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def analyze(log_path: Path, out_dir: Path) -> dict:
    cells, checks, quick, probes = records(log_path)
    if not cells:
        raise ValueError("no CELL_RESULT records")
    keys = set()
    for ident, cell in cells.items():
        init, rule, mode, seed = (cell[k] for k in ("init", "rule", "lr_mode", "seed"))
        if init not in RULES or rule not in RULES[init] or mode not in MODES or seed not in SEEDS[init]:
            raise ValueError(f"unregistered cell: {ident}")
        expected_id = f"c1__s{seed}__{init}__{rule}__{mode}"
        if ident != expected_id or (init, rule, mode, seed) in keys:
            raise ValueError(f"incorrect/duplicate cell identity: {ident}")
        keys.add((init, rule, mode, seed))
        if "error" in cell or cell.get("diverged"):
            raise ValueError(f"failed cell: {ident}")
        if not math.isfinite(cell["final_quick_acc"]) or not 0 <= cell["final_quick_acc"] <= 1:
            raise ValueError(f"invalid final accuracy: {ident}")
        if cell["bound_final"] != (cell["final_quick_acc"] >= .90):
            raise ValueError(f"inconsistent bound_final: {ident}")
        trajectory = sorted((step, row["query_acc"]) for (name, step), row in quick.items()
                            if name == ident)
        if not trajectory or trajectory[0][0] != 0 or trajectory[-1][0] != cell["steps"]:
            raise ValueError(f"missing initial/final quick observation: {ident}")
        if any(not math.isfinite(acc) or not 0 <= acc <= 1 for _, acc in trajectory):
            raise ValueError(f"invalid quick trajectory: {ident}")
        observed_loss = next((step for step, acc in trajectory if acc < .90), None)
        if observed_loss != cell.get("t_lose"):
            raise ValueError(f"inconsistent t_lose: {ident}")
        if abs(trajectory[-1][1] - cell["final_quick_acc"]) > 1e-6:
            raise ValueError(f"inconsistent final quick: {ident}")
        if abs(min(acc for _, acc in trajectory) - cell["min_acc"]) > 1e-6:
            raise ValueError(f"inconsistent min_acc: {ident}")
    tests = []
    for mode in MODES:
        tests.append(contrast(cells, f"C1_{mode}", "e2e2emb", "e2e2emb", "e2e1emb",
                              mode, mode, "greater"))
    for init, rules in RULES.items():
        for rule in rules:
            tests.append(contrast(cells, f"C2_{init}_{rule}", init, rule, rule,
                                  "floor", "rewarm", "two-sided"))
    for mode in MODES:
        for baseline, alternative in (("e2e1emb", "greater"), ("e2e2emb", "two-sided")):
            tests.append(contrast(cells, f"C3_{baseline}_{mode}", "e2e2emb", "e2e1emb_lin",
                                  baseline, mode, mode, alternative))
        tests.append(contrast(cells, f"C4_{mode}", "e2e2", "e2e1_code", "e2e1",
                              mode, mode, "greater"))
    holm(tests)
    data = out_dir / "data"
    flat = []
    for cell in cells.values():
        row = {k: v for k, v in cell.items() if k not in ("quick", "probes", "losses", "eval")}
        row.update({f"ablation_{k}": v for k, v in cell.get("ablation", {}).items()})
        row.pop("ablation", None)
        flat.append(row)
    write_csv(data / "r44c-cells.csv", flat)
    write_csv(data / "r44c-step0.csv", list(checks.values()))
    trajectories = []
    for key in sorted(quick):
        ident, step = key
        if ident not in cells:
            continue
        trajectories.append(dict(quick[key], **{k: cells[ident][k] for k in
                    ("init", "rule", "lr_mode", "seed")}, **{
                    k: v for k, v in probes.get(key, {}).items() if k not in quick[key]}))
    write_csv(data / "r44c-quick.csv", trajectories)
    write_csv(data / "r44c-gamma.csv", [r for r in trajectories if r["rule"] == "e2e1_code"])
    write_csv(data / "r44c-tests.csv", tests)
    for family in ("C1", "C2", "C3", "C4"):
        write_csv(data / f"r44c-{family.lower()}.csv", [r for r in tests if r["test"].startswith(family)])
    groups = defaultdict(list)
    for cell in cells.values():
        if cell["steps"] == 375:
            groups[(cell["init"], cell["rule"], cell["lr_mode"])].append(cell)
    budgets, loss_rows = [], []
    for (init, rule, mode), group in sorted(groups.items()):
        losses = sorted(int(c["t_lose"]) for c in group if c.get("t_lose") is not None)
        t80 = losses[math.ceil(.8 * len(losses)) - 1] if losses else None
        budgets.append(dict(init=init, rule=rule, lr_mode=mode, n=len(group),
            n_losses=len(losses), n_never_lost=len(group)-len(losses), t80=t80,
            tokens_per_step=32*512, tokens80=t80*32*512 if t80 is not None else None,
            horizon_tokens=375*32*512, observed_loss_steps=losses))
        loss_rows.append(dict(init=init, rule=rule, lr_mode=mode, n=len(group),
            retained=sum(c["bound_final"] for c in group),
            loss_steps=losses, n_never_lost=len(group)-len(losses),
            min_acc=min(c["min_acc"] for c in group)))
    write_csv(data / "r44c-token-budget.csv", budgets)
    write_csv(data / "r44c-retention-loss.csv", loss_rows)
    complete = len(cells) == 132 and all(c["steps"] == 375 for c in cells.values())
    chosen, contrasts, eligible = None, {}, []
    if complete:
        for mode in MODES:
            so = groups[("e2e2emb", "e2e2emb", mode)]
            fo = groups[("e2e2emb", "e2e1emb", mode)]
            contrasts[mode] = sum(c["bound_final"] for c in so) - sum(c["bound_final"] for c in fo)
            if all(c["bound_final"] for c in so):
                eligible.append(mode)
        if eligible:
            chosen = max(eligible, key=lambda mode: (contrasts[mode], mode == "floor"))
    lr_choice = chosen or ("inconclusive" if complete else "pending")
    add_code, recommended, lower_bound, median_gamma = None, None, None, None
    budget_status = "inconclusive"
    if chosen:
        code_cells = groups[("e2e2", "e2e1_code", chosen)]
        gammas = [c.get("gamma_final", quick.get((c["cell_id"], 375), {}).get("gamma"))
                  for c in code_cells]
        if all(g is not None and math.isfinite(g) for g in gammas):
            median_gamma = statistics.median(abs(g) for g in gammas)
        add_code = (sum(c["bound_final"] for c in code_cells) >= 6 and
                    median_gamma is not None and median_gamma >= .1)
        candidates = [row["tokens80"] for row in budgets if row["lr_mode"] == chosen and
                      row["rule"] in ("e2e1emb", "e2e1")]
        if len(candidates) != 2 or any(value is None for value in candidates):
            lower_bound, budget_status = 375*32*512, "censored"
        else:
            recommended, budget_status = max(candidates), "available"
    summary = dict(status="complete" if complete else "incomplete_or_smoke",
        cells=len(cells), expected_cells=132, step0_matches=len(checks),
        headline_tests=tests, holm_family_size=15, eligible_lr_modes=eligible,
        so_minus_fo_contrast=contrasts, chosen_lr_mode=chosen,
        lr_choice=lr_choice, p21_code_median_abs_gamma375=median_gamma,
        p21_add_ce_code=add_code, p21_toy_equivalent_tokens=recommended,
        token_budget_lower_bound=lower_bound, token_budget_status=budget_status,
        horizon_tokens=375*32*512,
        interpretation="Small n=9–10; P2-1 design rehearsal, not research claims.",
        t_lose_resolution="first observed loss, interval-censored at probe cadence; rule-switch loss at 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    (data / "r44c-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"lr_choice={lr_choice}")
    print("R44C_SUMMARY " + json.dumps(summary, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--process-name", default="TTM-r44c-06-analysis")
    args = parser.parse_args()
    os.environ["PROCESS_NAME"] = args.process_name
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    analyze(args.log, args.out_dir)


if __name__ == "__main__":
    main()
