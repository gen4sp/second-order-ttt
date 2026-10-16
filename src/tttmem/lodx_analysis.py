"""LOD-X registered budget contrast; no adjustment or fresh-seed claim."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tomllib

import numpy as np
from . import lod_analysis as lod

LOADS = (16, 32, 64)
SEEDS = tuple(range(216, 228))
N = 12
NE = lod.NE
DELTA = .05


def interval(values, quantile=.95):
    """t interval with bounds at the given t quantile.

    The verdict bounds use LOD's convention, the .95 quantile: each bound is a
    one-sided 95 % bound (a 90 % two-sided interval), as in LOD's NI tests.
    quantile=.975 gives the descriptive 95 % two-sided interval.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    row = dict(n=n, mean=None, LB=None, UB=None, quantile=quantile,
               confidence=2 * quantile - 1,
               scope="t-model; approximate normality of seed differences")
    if not n:
        return row
    mean = float(values[0]) if np.all(values == values[0]) else float(values.mean())
    if n == 1:
        return dict(row, mean=mean)  # No variance estimate from a single seed.
    lo, hi = 0., 100.
    for _ in range(70):
        mid = (lo + hi) / 2
        cdf = 1 - .5 * lod.ibeta((n - 1) / 2, .5, (n - 1) / (n - 1 + mid * mid))
        if cdf < quantile:
            lo = mid
        else:
            hi = mid
    width = (lo + hi) / 2 * float(values.std(ddof=1)) / math.sqrt(n)
    if np.all(values == values[0]):
        width = 0.
    return dict(row, mean=mean, LB=mean - width, UB=mean + width)


def ni(values, evaluable=True):
    row = interval(values)
    two_sided = interval(values, .975)
    row.update(margin=DELTA, label=NE, LB95_two_sided=two_sided["LB"], UB95_two_sided=two_sided["UB"])
    if evaluable and row["n"] == N:
        row["label"] = ("FREE" if row["LB"] > -DELTA + 1e-12 else
                        "PRICED" if row["UB"] < -DELTA - 1e-12 else "INCONCLUSIVE")
    return row


def mcnemar(before, after):
    """Exact two-sided binomial McNemar on discordant retained labels."""
    lost = sum(bool(b) and not bool(a) for b, a in zip(before, after, strict=True))
    gained = sum(not bool(b) and bool(a) for b, a in zip(before, after, strict=True))
    discordant = lost + gained
    p = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(lost, gained) + 1))
            / 2**discordant) if discordant else 1.
    return dict(n=len(before), lost=lost, gained=gained, discordant=discordant, p_two_sided=p,
                method="exact binomial McNemar; descriptive; unadjusted")


def plateau(cells):
    result = {}
    early_grid, late_grid = set(range(2500, 2701, 25)), set(range(2800, 3001, 25))
    for arm in ("lin", "so"):
        for n in LOADS:
            cohort = [c for c in cells if c["group"] == "core" and c["arm"] == arm and c["load"] == n]
            changes = []
            for c in cohort:
                logs = {r["step"]: r["quick"]["acc"] for r in c.get("logs", [])}
                if early_grid | late_grid <= logs.keys():
                    change = float(np.mean([logs[s] for s in late_grid]) - np.mean([logs[s] for s in early_grid]))
                    if math.isfinite(change):
                        changes.append(dict(seed=c["seed"], delta=change))
            rising = sum(r["delta"] > .02 + 1e-12 for r in changes)
            label = "budget-limited" if rising > 3 else "plateau" if len(changes) == N else NE
            result[f"core/{arm}/{n}"] = dict(n=len(cohort), n_measured=len(changes),
                rising=rising, changes=changes, label=label, early=[2500, 2700], late=[2800, 3000])
    return result


def score(cells, layout_valid=True):
    ix = lod.indexed(cells)
    core = [c for c in ix.values() if c["group"] == "core" and c["seed"] in SEEDS]
    # The production contract rejects a missing audit; accepting absent audits
    # here keeps score() useful for small synthetic verdict fixtures.
    layout = layout_valid and all(c.get("layout_audit") is None or all(c["layout_audit"].values())
                                  for c in core)
    leaks = [c for c in core if c.get("observed", {}).get("W0", c["composite"].get("W0")) is not None
             and c.get("observed", {}).get("W0", c["composite"].get("W0")) > c.get("B_nb", 1.) + .05]
    valid = bool(layout and not leaks)
    out = dict(valid=valid, gates=dict(layout=bool(layout), Q_mem=not leaks),
               Q_SO={}, Q_plat=plateau(core), X2={}, X3={}, X4={})
    for n in LOADS:
        so = [c for c in core if c["arm"] == "so" and c["load"] == n]
        mean_so = float(np.mean([c["composite"]["A"] for c in so])) if so else None
        retained = sum(c["retained"] for c in so)
        qso = len(so) == N and mean_so >= .90 - 1e-12 and retained >= 9
        out["Q_SO"][str(n)] = dict(pass_gate=qso, n=len(so), mean_A=mean_so, retained=retained)
        paired = lod.pairs({(c["group"], c["arm"], c["load"], c["seed"]): c for c in core},
                           "core", "lin", "core", "so", n)
        differences = [a["composite"]["A"] - b["composite"]["A"] for a, b in paired]
        row = ni(differences, valid and qso)
        row.update(load=n, seeds=[a["seed"] for a, b in paired], differences=differences,
                   Q_SO=qso, multiplicity="primary" if n == 32 else "secondary; unadjusted",
                   failures_lin=lod.failure_counts([a for a, b in paired]),
                   failures_so=lod.failure_counts([b for a, b in paired]))
        if n == 32:
            out["X1"] = row
        else:
            out["X2"][str(n)] = row
        for arm, h in (("lin", "X3"), ("so", "X4")):
            cohort = [c for c in core if c["arm"] == arm and c["load"] == n
                      and c.get("composite_1500", {}).get("A") is not None]
            diff = [c["composite"]["A"] - c["composite_1500"]["A"] for c in cohort]
            two_sided = interval(diff, .975)
            out[h][str(n)] = dict(**interval(diff), LB95_two_sided=two_sided["LB"],
                UB95_two_sided=two_sided["UB"], label="DESCRIPTIVE" if valid and len(cohort) == N else NE,
                seeds=[c["seed"] for c in cohort], differences=diff,
                retained_1500=sum(c.get("retained_1500", False) for c in cohort),
                retained_3000=sum(c["retained"] for c in cohort),
                McNemar=mcnemar([c.get("retained_1500", False) for c in cohort],
                                [c.get("retained", False) for c in cohort]))
    label = out["X1"]["label"]
    lin_plat = out["Q_plat"]["core/lin/32"]["label"]
    readings = {"FREE": "LOD's N32 price was a budget artefact on this width, eta and schedule.",
        "INCONCLUSIVE": "Report the X1 CI and X3 budget effect.",
        NE: "No N32 contrast: invalid run, incomplete pairing or SO not competent at 3000."}
    if label == "PRICED":
        reading = ("The price is real at this width, eta and schedule." if lin_plat == "plateau" else
                   "Lin is slower; whether the price is real stays unresolved." if lin_plat == "budget-limited" else
                   "Lin is priced; the plateau reading is not evaluable.")
    else:
        reading = readings[label]
    out.update(interpretation=reading if valid else "INVALID: layout audit or Q-mem failed.",
               claims=dict(primary=label, relative_global=label,
                           seed_scope="same LOD seeds; no fresh-seed family", lin32_plateau=lin_plat))
    return out


def reproduction(cells, reference):
    """Read only JSON for matching LOD core cells; no checkpoint loads."""
    reference = Path(reference)
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest.get("config", {}).get("steps") != 1500:
        raise ValueError("LOD reference must have a 1500-step endpoint")
    expected = {f"core_{a}_N{n}_s{s}" for a in ("lin", "so") for n in LOADS for s in SEEDS}
    rows, missing = [], []
    for c in sorted(cells, key=lambda c: c["cell_id"]):
        cid = c["cell_id"]
        if cid not in expected or c.get("composite_1500", {}).get("A") is None:
            continue
        file = reference / (cid + ".json")
        if not file.exists():
            missing.append(cid)
            continue
        old = json.loads(file.read_text())
        if old.get("cell_id") != cid or old.get("status") not in ("COMPLETE", "DIVERGED"):
            missing.append(cid)
            continue
        value = old.get("composite", {}).get("A")
        if value is None or not math.isfinite(value):
            missing.append(cid)
            continue
        rows.append(dict(cell_id=cid, arm=c["arm"], load=c["load"], seed=c["seed"],
            A_1500=c["composite_1500"]["A"], A_LOD=value,
            absolute_difference=abs(c["composite_1500"]["A"] - value),
            retained_1500=c["retained_1500"], retained_LOD=old["retained"],
            retained_agreement=c["retained_1500"] == old["retained"]))
    return dict(label="DESCRIPTIVE", gate=False, reference=str(reference),
        reference_config_sha256=manifest.get("config_sha256"),
        n=len(rows), mean_absolute_difference=float(np.mean([r["absolute_difference"] for r in rows])) if rows else None,
        max_absolute_difference=max((r["absolute_difference"] for r in rows), default=None),
        retained_agreements=sum(r["retained_agreement"] for r in rows),
        retained_agreement_rate=float(np.mean([r["retained_agreement"] for r in rows])) if rows else None,
        cells=rows, missing=missing,
        note="Paired descriptive reproduction; CUDA nondeterminism allowed; not a gate.")


def contract_errors(manifest, cells, diagnostic=False):
    from . import exp_lod as driver
    errors = lod.contract_errors(manifest, cells, diagnostic)
    cfg = manifest["config"]
    points = driver.full_eval_steps(cfg)
    if cfg.get("protocol") != "lodx":
        errors.append("not a LOD-X manifest")
    plan = cfg.get("plan_file")
    if manifest.get("plan_sha256") != (driver.file_sha(driver.ROOT / plan) if plan else None):
        errors.append("LOD-X plan hash mismatch")
    if not diagnostic:
        registered = tomllib.loads((driver.ROOT / "configs/toy/lod-x.toml").read_text())["lod"]
        if cfg != registered or cfg["steps"] != 3000 or points != [1500, 3000]:
            errors.append("LOD-X production config/endpoints mismatch")
        if manifest["admission"].get("tier") != "X" or manifest["admission"].get("addon"):
            errors.append("LOD-X tier/add-on mismatch")
    for c in cells:
        if c["status"] not in ("COMPLETE", "DIVERGED"):
            continue
        if c["status"] == "COMPLETE" and not diagnostic:
            steps = [r["step"] for r in c.get("logs", [])]
            if steps != list(range(25, 3001, 25)):
                errors.append("quick step grid mismatch: " + c["cell_id"])
        for point in points[:-1]:
            comp = c.get(f"composite_{point}", {})
            if not comp or comp.get("A") is None:
                errors.append("missing intermediate composite: " + c["cell_id"])
                continue
            detail = c.get(f"details_{point}", {})
            failed_early = c["status"] == "DIVERGED" and c["completed_steps"] <= point
            if failed_early:
                if any(v != 1/16 for v in comp.values()) or c.get(f"retained_{point}") is not False:
                    errors.append("uncertified intermediate failure: " + c["cell_id"])
                continue
            if c.get(f"eval_step_{point}") != point:
                errors.append("intermediate step endpoint mismatch: " + c["cell_id"])
            if not detail.get("layout_audit") or not all(detail["layout_audit"].values()):
                errors.append("intermediate layout audit: " + c["cell_id"])
            if not detail.get("eval_sha256") or detail["eval_sha256"] != c.get("eval_sha256"):
                errors.append("intermediate/final evaluation set mismatch: " + c["cell_id"])
            acquired = c.get(f"acquired_{point}")
            if acquired is not None and (acquired > point or acquired % cfg["log_every"]):
                errors.append("intermediate acquisition endpoint: " + c["cell_id"])
            if c.get(f"retained_{point}") != (acquired is not None and comp["A"] >= .9):
                errors.append("intermediate retained label mismatch: " + c["cell_id"])
        if c["status"] == "COMPLETE":
            if not c.get("eval_sha256") or not math.isfinite(c["composite"].get("A", math.nan)):
                errors.append("missing final composite/set: " + c["cell_id"])
            acquired = c.get("acquired")
            if acquired is not None and (acquired > cfg["steps"] or acquired % cfg["log_every"]):
                errors.append("final acquisition endpoint: " + c["cell_id"])
            if c.get("retained") != (acquired is not None and c["composite"]["A"] >= .9):
                errors.append("final retained label mismatch: " + c["cell_id"])
    return errors


def validate_probes(path, cells, points, cfg):
    from . import exp_lod as driver
    for c in cells:
        if c["status"] not in ("COMPLETE", "DIVERGED"):
            continue
        for point in points:
            final = point == cfg["steps"]
            detail = c if final else c.get(f"details_{point}", {})
            suffix = "" if final else f"_{point}"
            probe = path / (c["cell_id"] + suffix + "_probes.npz")
            if not probe.exists():
                if c["status"] == "COMPLETE" or detail.get("probes_sha256"):
                    raise ValueError("missing probe records: " + probe.name)
                continue
            if driver.file_sha(probe) != detail.get("probes_sha256"):
                raise ValueError("probe file hash mismatch: " + probe.name)
            with np.load(probe) as data:
                meta = dict(value_histogram=data["value_histogram"], query_tok=data["query_tok"],
                            value0=cfg["pool"] + 2)
                if not math.isclose(driver.binding_free_baseline(meta), detail["B_nb"], abs_tol=1e-12):
                    raise ValueError("B_nb mismatch: " + probe.name)


def analyze_directory(path, diagnostic=False, lod_ref=None):
    from . import exp_lod as driver
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    cells = [json.loads(file.read_text()) for c in manifest["cells"]
             if (file := path / (driver.cell_id(c) + ".json")).exists()]
    errors = contract_errors(manifest, cells, diagnostic)
    if errors:
        raise ValueError("production contract violations: " + "; ".join(errors))
    validate_probes(path, cells, driver.full_eval_steps(manifest["config"]), manifest["config"])
    result = score(cells, layout_valid=not manifest.get("infra_stop"))
    result.update(mode="SMOKE ONLY; no scientific claims" if diagnostic else "production",
                  manifest_sha256=driver.file_sha(path / "manifest.json"),
                  counts={s: sum(c["status"] == s for c in cells)
                          for s in ("COMPLETE", "DIVERGED", "TRUNCATED", "MISSING")},
                  missing=[c for c in manifest["cells"] if driver.cell_id(c) not in lod.indexed_by_id(cells)]
                  if hasattr(lod, "indexed_by_id") else [c for c in manifest["cells"]
                      if not any(r["cell_id"] == driver.cell_id(c) and r["status"] in ("COMPLETE", "DIVERGED") for r in cells)])
    result["R"] = reproduction(cells, lod_ref) if lod_ref is not None else dict(label="NOT REQUESTED", gate=False)
    if diagnostic or manifest.get("infra_stop"):
        mode = "SMOKE ONLY" if diagnostic else "INFRA_GPU_ERRORS"
        if manifest.get("infra_stop"):
            result.update(valid=False, infra_stop=manifest["infra_stop"], mode="INFRA_GPU_ERRORS; no scientific claims")
        result["claims"] = dict(primary=mode, relative_global=mode)
        result["X1"]["label"] = NE
        for h in ("X2", "X3", "X4"):
            for row in result[h].values():
                row["label"] = NE
        result["interpretation"] = result["mode"]
    return result


def write_outputs(path, result, out):
    from . import exp_lod as driver
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    driver.atomic_json(out / "lodx_scores.json", result)
    rows = [dict(hypothesis="X1", load=32, **{k: result["X1"].get(k) for k in ("label", "n", "mean", "LB", "UB")})]
    for h in ("X2", "X3", "X4"):
        rows += [dict(hypothesis=h, load=n, **{k: row.get(k) for k in ("label", "n", "mean", "LB", "UB")})
                 for n, row in result[h].items()]
    with (out / "lodx_verdicts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [result["mode"], json.dumps(result["claims"], sort_keys=True), result["interpretation"],
             "95% two-sided t-model intervals; X2 unadjusted; X3/X4 and R descriptive."]
    lines += [f"{r['hypothesis']} N{r['load']}: {r['label']} n={r['n']} mean={r['mean']} CI=[{r['LB']}, {r['UB']}]" for r in rows]
    lines += ["Q-SO: " + json.dumps(result["Q_SO"], sort_keys=True),
              "Q-plat: " + json.dumps(result["Q_plat"], sort_keys=True)]
    for h in ("X3", "X4"):
        lines += [f"{h} N{n} McNemar: " + json.dumps(row["McNemar"], sort_keys=True) for n, row in result[h].items()]
    lines.append("R: " + json.dumps({k: v for k, v in result["R"].items() if k != "cells"}, sort_keys=True))
    (out / "lodx_scores.txt").write_text("\n".join(lines) + "\n")
