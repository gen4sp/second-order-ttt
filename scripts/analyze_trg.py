"""Validate persisted TRG evidence and score exploratory stage-(i) outputs.

Usage: uv run --locked python scripts/analyze_trg.py OUTPUT [--smoke-only]
       --process-name TTM-trg-74-analyze
Fresh-pool scoring imports tttmem.analyze_trg directly after its own seal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tttmem import analyze_trg as a
from tttmem import dense_log as d
from tttmem import exp_trg as driver
from tttmem import exp_ttt as t
from tttmem.episodes import COMPLETE, DIVERGED, TRUNCATED, classify_branch, nl2_window
from tttmem.trg_triggers import (SpikeController, active_coincidence, dense_onset,
                                 dip_onsets, realised_drop_hash, yoked_sham)

ANALYZER_FILES = {"analyze_trg.py", "scripts/analyze_trg.py"}


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def branch_folder(record, root=None):
    return (Path(root)/f"{record['family']}-s{record['seed']}"/record["branch"]
            if root is not None else Path(record["output_dir"]))


def contract_branch(record, smoke_only=False, *, root=None, cfg=None):
    folder = branch_folder(record, root)
    if record["disposition"] == TRUNCATED:
        # A branch with a certified gap / code fault is missing, after its retry.
        # Its incomplete reads cannot turn into a positive scientific endpoint.
        return (load_rows(folder/"summaries.jsonl") if (folder/"summaries.jsonl").exists() else [],
                load_rows(folder/"telemetry.jsonl") if (folder/"telemetry.jsonl").exists() else [])
    _, steps, ladder = d.validate_shards(folder)
    rows = load_rows(folder/"summaries.jsonl")
    if [r["step"] for r in rows] != steps or steps != record["steps"]:
        raise ValueError("summary vs shard vs branch grid mismatch")
    if [r["acc"] for r in rows] != record["accuracy"]:
        raise ValueError("branch accuracy differs from full-precision summary")
    if d.sha256_file(folder/"telemetry.jsonl") != record["telemetry_sha256"]:
        raise ValueError("telemetry hash mismatch")
    telemetry = load_rows(folder/"telemetry.jsonl")
    at = record["t_acq"] if record["t_acq"] is not None else record["end"]
    expected = classify_branch(steps, record["accuracy"], start=at,
                end=record["end"], grid=1, t_div=record["t_div"], certified=record["certified"],
                code_fault=record["error"] is not None, nonfinite=record["nonfinite_local"],
                continuation_steps=record["t_div"]-record["start"] if record["t_div"] else None,
                certificate_start=record["start"])
    if record["t_acq"] is not None and expected != record["disposition"]:
        raise ValueError("branch disposition mismatch")
    offline_onset = dense_onset(steps, [r["acc"] for r in rows], at)
    if offline_onset != record["t_l1"]:
        raise ValueError("online/offline t_l1 mismatch")
    if d.sha256_file(folder/"acquisitions.jsonl") != record["acquisition_sha256"]:
        raise ValueError("acquisition evidence hash mismatch")
    acquisitions = load_rows(folder/"acquisitions.jsonl")
    if record["branch"] == "main" and record["start"] == 0:
        forced = (cfg or {}).get("smoke_fork") if smoke_only else None
        offline_acq = next((r["step"] for r in acquisitions
                            if (r["step"] == forced if forced is not None else r["acc"] >= .90)), None)
        if offline_acq != record["t_acq"]:
            raise ValueError("online/offline t_acq mismatch")
    elif record["t_acq"] != record["start"]:
        raise ValueError("loaded fork acquisition/start mismatch")
    if d.sha256_file(folder/"drops.json") != record["drop_list_sha256"]:
        raise ValueError("drop-list file hash mismatch")
    if realised_drop_hash(record["drops"]) != record["drop_list_sha256"]:
        raise ValueError("drop-list payload hash mismatch")
    controller = SpikeController(at, record["kappa"])
    for row in telemetry:
        if row["step"] <= at:
            continue
        offline = controller.observe(row["step"], row["grad_norm"] if row["grad_norm"] is not None else np.nan)
        if offline["flags"] != row["flags"] or offline["active"] != row["active"]:
            raise ValueError("online/offline flag mismatch")
        if record["branch"] in ("kvm", "rnd") and any(c != (4 if smoke_only and len(row["masked_per_sequence"]) else 16)
                                                           for c in row["masked_per_sequence"]):
            # Tiny model counts come from the manifest and are checked separately below.
            if not smoke_only:
                raise ValueError("mask miscount")
        if row["dropped"] and row["lr"] != row["lr_before"]:
            raise ValueError("drop changed LR")
    if record["branch"] != "main" and ladder:
        raise ValueError("probe unexpectedly carries ladder")
    if record["branch"] == "sham":
        wanted = yoked_sham(json.loads((folder.parent/"skip"/"branch.json").read_text())["drops"],
                            record["source_drop_list_sha256"])
        if record["disposition"] == DIVERGED:
            wanted = [s for s in wanted if s <= record["t_div"]]
        if record["drops"] != wanted:
            raise ValueError("yoked realised schedule mismatch")
    return rows, telemetry


def localisation_summaries(rows, at, loss, late=6):
    """Exact means use fp64 stdout sufficient statistics, never packed fp16 terms."""
    if loss is None or loss < at+21:
        return dict(eligible=False)
    base = [r for r in rows if at <= r["step"] <= at+15 and r["ladder"]]
    fail = [r for r in rows if loss-5 <= r["step"] <= loss+5 and r["ladder"]]
    if len(base) != 16 or len(fail) != 11 or f"events{late}" not in base[0]:
        return dict(eligible=False, reason="coverage")
    def aggregate(block):
        n = sum(r[f"events{late}"] for r in block)
        terms = np.sum([r[f"term_sums{late}"] for r in block], axis=0)/n
        total = sum(r[f"margin_sum{late}"] for r in block)/n
        sq = sum(r[f"margin_sumsq{late}"] for r in block)/n
        return terms, total, max(0., sq-total**2)**.5
    bt, bm, sd = aggregate(base)
    lt, lm, _ = aggregate(fail)
    dj, dt = lt-bt, lm-bm
    if abs(dj.sum()-dt) > 1e-5:
        raise ValueError("aggregated telescoping failed")
    cancelling = bool(np.abs(dj).sum() > 2*abs(dt))
    dom = np.flatnonzero(dj/dt >= .5) if dt else []
    label = ("NEGLIGIBLE" if abs(dt) < .25*sd else "CANCELLING" if cancelling else
             f"DOMINANT {dom[0]}" if len(dom) == 1 else "DISTRIBUTED")
    return dict(eligible=True, D_j=dj.tolist(), D_total=float(dt), baseline_sd=sd,
                accounting_error=float(abs(dj.sum()-dt)), label=label)


def descriptive(record, rows, telemetry, kappa=5, *, root=None):
    at, loss = record["t_acq"], record["t_l1"]
    if at is None:
        return {}, []
    out = dict(localisation=localisation_summaries(rows, at, loss))
    series = {"C1": ("residual", -1), "C2": ("own_median6", -1),
              "C3": ("erosion_median6", 1), "C4": ("slow_median6", -1), "qce6": ("qce6", 1)}
    out["chronology"] = {}
    if loss is not None and loss <= at+280:
        for name, (field, direction) in series.items():
            subset = [r for r in rows if field in r]
            steps, values = [r["step"] for r in subset], [r[field] for r in subset]
            crosses = {str(f): a.fractional_crossing(steps, values, at, loss, fraction=f, direction=direction)
                       for f in (.10, .25, .50)}
            out["chronology"][name] = dict(crossings=crosses, hinge=a.hinge_onset(steps, values, at, loss))
        anchor = out["chronology"]["qce6"]["crossings"]
        for name, value in out["chronology"].items():
            for f, cross in value["crossings"].items():
                aa, bb = cross["onset"], anchor[f]["onset"]
                cross["lead_to_qce6"] = bb-aa if aa is not None and bb is not None else None
                cross["shape_bias_annotation"] = (abs(bb-aa) < 30) if aa is not None and bb is not None else None
    flags = [(r["step"], r["flags"].get(str(kappa), False)) for r in telemetry]
    if loss is not None:
        spike = any(f for s, f in flags if loss-3 <= s <= loss)
        noflag = not any(f for s, f in flags if loss-30 <= s <= loss)
        crossing = out["chronology"].get("qce6", {}).get("crossings", {}).get("0.25", {}).get("onset")
        out["signature"] = dict(spike_associated=spike,
                                drift_like=noflag and crossing is not None and crossing <= loss-10)
    if record["family"] == "SO":
        onsets = dip_onsets(record["steps"], record["accuracy"], at)
        coincidence_row = active_coincidence(telemetry, onsets, kappa)
        if coincidence_row["ratio"] == float("inf"):
            coincidence_row["ratio"] = "infinity"
        out["coincidence"] = dict(onsets=onsets.tolist(), **coincidence_row)
    event_rows = []
    folder = branch_folder(record, root)
    manifest, _, _ = d.validate_shards(folder)
    blocks = []
    for shard in manifest["shards"]:
        with np.load(folder/shard["file"], allow_pickle=False) as z:
            blocks.append({k: z[k].copy() for k in z.files})
    if not blocks:
        return out, event_rows
    if out["localisation"].get("eligible"):
        ladder_blocks = [b for b in blocks if len(b["ladder_steps"])]
        out["localisation"]["sequence_cluster_ci"] = a.sequence_cluster_ci(
            np.concatenate([b["terms"] for b in ladder_blocks]),
            np.concatenate([b["rungs"][...,5] for b in ladder_blocks]),
            np.concatenate([b["ladder_steps"] for b in ladder_blocks]), blocks[0]["chunks"][0],
            at, loss, seed=record["seed"])
    steps = np.concatenate([b["steps"] for b in blocks])
    margins = np.concatenate([b["margin"] for b in blocks])
    failure = a.event_failures(steps, margins, at) if len(steps) >= 10 else None
    if failure is None:
        return out, event_rows
    landmark = at+10
    block = next((b for b in blocks if landmark in b["ladder_steps"]), None)
    if block is None:
        out["event_landmark"] = "no ladder at landmark"
        return out, event_rows
    i, j = int(np.flatnonzero(block["ladder_steps"] == landmark)[0]), int(np.flatnonzero(block["steps"] == landmark)[0])
    covs = dict(residual=block["residual"][j], write_norm=block["write_norm"][i],
                own_write=block["terms"][i, ..., 1], L1_margin=block["rungs"][i, ..., 1],
                maximum_key_cosine=block["maximum_key_cosine"][i],
                address_margin=block["address_margin"][i], slow_margin=block["terms"][i, ..., 0],
                duplicate=block["duplicate"][j])
    chunks, order = block["chunks"][j], block["query_order"][j]
    risk, times, observed = failure["at_risk"], failure["times"], failure["observed"]
    out["harrell"] = {k: a.harrell_c(times[risk], observed[risk], v[risk], chunks[risk]) for k, v in covs.items()}
    for seq, ev in np.argwhere(risk):
        event_rows.append(dict(family=record["family"], seed=record["seed"], sequence=int(seq),
                               event=int(ev), chunk=int(chunks[seq, ev]), query_order=int(order[seq, ev]),
                               failure_step=int(times[seq, ev]), observed=bool(observed[seq, ev]),
                               **{k: float(v[seq, ev]) for k, v in covs.items()}))
    return out, event_rows


def replay_fidelity(record, rows, csv_path):
    if record["family"] == "SO":
        return dict(replay_defined=True, reason="SO rerun")
    if not csv_path or not Path(csv_path).exists():
        return dict(replay_defined=True, reason="historical grid unavailable")
    arm = "e2e1emb" if record["family"] == "FO" else "e2e1emb_cn97"
    with open(csv_path) as f:
        historical = [r for r in csv.DictReader(f) if r["branch"] == "main"
                      and f"/s{record['seed']}/{arm}/" in r["cell_id"]]
    hs = np.array([int(r["step"]) for r in historical])
    ha = np.array([float(r["a"]) for r in historical])
    at = record["t_acq"]
    h = nl2_window(hs, ha, at)
    rs, ra = [r["step"] for r in rows], [r["acc"] for r in rows]
    r = nl2_window(rs, ra, at)
    if h["onset"] is None or r["onset"] is None:
        return dict(replay_defined=True, reason="no matched onset", reference=h, replay=r)
    reference = {s: v for s, v in zip(hs, ha)}
    differences = [abs(v-reference[s]) for s, v in zip(rs, ra)
                   if s in reference and h["onset"]-25 <= s <= h["onset"]+25]
    maximum = max(differences, default=float("inf"))
    matches = abs(h["onset"]-r["onset"]) <= 10 and maximum <= .10
    return dict(replay_defined=not matches, reference_onset=h["onset"], replay_onset=r["onset"],
                max_accuracy_difference=maximum if np.isfinite(maximum) else None)


def verify_sources(manifest, *, allow_analyzer_override=False):
    recorded, current = manifest["source_sha256"], driver.source_hashes()
    if (set(recorded) != set(current) or manifest["schema_sha256"] != d.SCHEMA_SHA256
            or any(recorded[p] != current[p] for p in set(current)-ANALYZER_FILES)):
        raise ValueError("driver/schema hash mismatch")
    changed = sorted(p for p in ANALYZER_FILES if recorded[p] != current[p])
    if changed and not allow_analyzer_override:
        raise ValueError("analyzer hash mismatch; use --allow-analyzer-override to record the override")
    evidence = dict(recorded={p:recorded[p] for p in sorted(ANALYZER_FILES)},
                    current={p:current[p] for p in sorted(ANALYZER_FILES)},
                    changed=changed, override_requested=allow_analyzer_override, override_used=bool(changed))
    driver.emit("TRG_ANALYZER_HASHES", evidence)
    return evidence


def resolve_chr_series(manifest, override=None, *, smoke_only=False):
    path = override or manifest["effective_config"].get("historical_series_path")
    expected = manifest.get("chr_series_sha256")
    if not path or not Path(path).is_file():
        if not smoke_only:
            raise ValueError("CHR series missing; supply --chr-series PATH")
        return None
    if expected is None and not smoke_only:
        raise ValueError("CHR series hash missing in manifest")
    if expected is not None and d.sha256_file(path) != expected:
        raise ValueError("CHR series hash mismatch")
    return str(path)


def analyze_folder(folder, *, smoke_only=False, chr_series=None, allow_analyzer_override=False):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    if bool(manifest["smoke_only"]) != bool(smoke_only):
        raise ValueError("smoke-only mode must match manifest")
    plan = manifest.get("effective_config", {}).get("plan_file")
    if manifest["plan_sha256"] != (d.sha256_file(driver.ROOT/plan) if plan else None):
        raise ValueError("registration hash mismatch")
    analyzer_hashes = verify_sources(manifest, allow_analyzer_override=allow_analyzer_override)
    series_path = resolve_chr_series(manifest, chr_series, smoke_only=smoke_only)
    records = json.loads((folder/"branches.json").read_text())
    calibration = json.loads((folder/"calibration.json").read_text())
    scored, event_rows, units = {}, [], []
    for record in records:
        rows, telemetry = contract_branch(record, smoke_only, root=folder, cfg=manifest["effective_config"])
        if record["t_acq"] is None:
            continue
        at = record["t_acq"]
        endpoint = nl2_window(record["steps"], record["accuracy"], at,
                             disposition=record["disposition"], t_div=record["t_div"])
        burden = a.unbound_burden(record["steps"], record["accuracy"], at,
                                  disposition=record["disposition"], t_div=record["t_div"])
        desc, events = descriptive(record, rows, telemetry, calibration["kappa"], root=folder) if record["branch"] == "main" and record["disposition"] != TRUNCATED else ({}, [])
        scored[(record["family"], record["seed"], record["branch"])] = dict(record, endpoint=endpoint,
                                     burden=burden, descriptive=desc, rows=rows, telemetry=telemetry)
        event_rows.extend(events)
    pairs3, pairs4 = [], []
    for c in manifest["cells"]:
        key = c["family"], c["seed"]
        main = scored.get((*key, "main"))
        if main is None:
            units.append(dict(family=key[0], seed=key[1], disposition="no-acquisition"))
            continue
        at = main["t_acq"]
        unit = dict(family=key[0], seed=key[1], t_acq=at, t_l1=main["t_l1"],
                    descriptive=main["descriptive"], disposition="enrolled")
        if key[0] in ("FO", "noise") and at <= 1100:
            pair = dict(unit)
            ref = main["endpoint"]["onset"]
            for tag in ("kvm", "rnd"):
                branch = scored.get((*key, tag))
                pair[tag] = branch["endpoint"] if branch else dict(nl2=None, onset=None)
                unit[tag] = dict(pair[tag], disposition=branch["disposition"] if branch else TRUNCATED,
                                 category="REF-MISSING" if main["endpoint"]["missing"] else
                                          a.reference_category(pair[tag]["onset"], ref))
            unit["paired_onset_difference"] = a.paired_onset_difference(pair["kvm"], pair["rnd"], at)
            losses = scored.get((*key, "kvm"))
            onset = pair["kvm"]["onset"]
            residuals = [r["residual"] for r in losses["rows"] if onset-10 <= r["step"] <= onset] if onset is not None and losses else []
            pair["own_onset_residual"] = float(np.median(residuals)) if residuals else None
            unit["replay_fidelity"] = replay_fidelity(main, main["rows"], series_path)
            pairs3.append(pair)
        elif key[0] == "SO" and at <= 450:
            pair = dict(unit, length=660-at-10)
            for tag in ("skip", "sham"):
                branch = scored.get((*key, tag))
                pair[tag] = branch["burden"] if branch else dict(U=None)
                unit[tag] = dict(pair[tag], disposition=branch["disposition"] if branch else TRUNCATED)
                if branch:
                    active = sum(r["active"] for r in branch["telemetry"])
                    n_drop = len(branch["drops"])
                    dose = dict(active_steps=active, drops=n_drop,
                                drop_fraction=n_drop/active if active else None,
                                own_flags=sum(r["flags"].get(str(calibration["kappa"]),False) for r in branch["telemetry"]),
                                summed_update_norm=sum(sum(v["update_norm"] for v in r["update"].values()) for r in branch["telemetry"]))
                    unit[tag]["dose"] = dose
            skip = scored.get((*key, "skip"))
            pair["drops"] = len(skip["drops"]) if skip else 0
            pair["R"] = a.burden_ratio(pair["skip"]["U"], pair["sham"]["U"]) if pair["skip"]["U"] is not None and pair["sham"]["U"] is not None else None
            unit["R"] = pair["R"]
            pairs4.append(pair)
        else:
            unit["disposition"] = "t_acq over limit"
        units.append(unit)
    attempted = {f: sum(c["family"] == f for c in manifest["cells"]) for f in ("FO", "noise")}
    ht3 = a.score_ht3(pairs3, attempted=attempted)
    for family, row in ht3.items():
        losses = [u for u in pairs3 if u["family"] == family and u["kvm"]["nl2"] is False]
        held = sum(u["own_onset_residual"] is not None and u["own_onset_residual"] >= .5 for u in losses)
        row["residual_held"] = dict(n=len(losses), held=held,
                   flag=held >= int(np.ceil(.75*len(losses))) if len(losses) >= 4 else None)
    scores = dict(stage="i", exploratory=True, smoke_only=smoke_only, H_T3=ht3,
                  H_T4=a.score_ht4(pairs4,attempted=sum(c["family"] == "SO" for c in manifest["cells"])),
                  units=units, calibration=calibration,
                  confirmatory_family="H-T3 FO/noise Holm-2 only (stage ii)",
                  plan_sha256=manifest["plan_sha256"], schema_sha256=manifest["schema_sha256"],
                  analyzer_hashes=analyzer_hashes, chr_series_sha256=manifest.get("chr_series_sha256"),
                  contract="PASS", event_export="CSV (no parquet dependency)")
    scores["H_T5"] = a.event_consistency(units)
    engaged = [u["skip"]["dose"] for u in units if u["family"] == "SO" and "dose" in u.get("skip",{})
               and u["skip"]["dose"]["drops"] > 0]
    scores["H_T4"]["manipulation"] = ("LIMITED" if any(v["drop_fraction"] > .05 for v in engaged) else
              "OK" if engaged and all(.002 <= v["drop_fraction"] <= .05 for v in engaged) else "NOT EVALUABLE")
    scores["descriptive_counts"] = {f: {name: dict(k=sum(bool(u.get("descriptive",{}).get("signature",{}).get(name)) for u in units if u["family"] == f),
            n=sum(u["family"] == f and "signature" in u.get("descriptive",{}) for u in units))
            for name in ("spike_associated","drift_like")} for f in ("FO","noise","SO")}
    for rows in scores["descriptive_counts"].values():
        for r in rows.values():
            r["ci95"] = a.clopper_pearson(r["k"],r["n"])
    driver.write_json(folder/"trg_scores.json", scores)
    (folder/"trg_scores.txt").write_text(json.dumps(scores, indent=2)+"\n")
    with (folder/"trg_units.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "seed", "t_acq", "t_l1", "disposition", "details"])
        writer.writeheader()
        for u in units:
            writer.writerow({**{k: u.get(k) for k in writer.fieldnames[:-1]}, "details": json.dumps(u)})
    fields = list(event_rows[0]) if event_rows else ["family", "seed", "sequence", "event", "observed"]
    with (folder/"trg_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(event_rows)
    print("TRG_ANALYSIS "+json.dumps(dict(contract="PASS", smoke_only=smoke_only, units=len(units),
                                        event_rows=len(event_rows), H_T3=ht3, H_T4=scores["H_T4"])))
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--chr-series", type=Path)
    p.add_argument("--allow-analyzer-override", action="store_true")
    p.add_argument("--process-name", required=True)
    args = p.parse_args()
    t._mark_process(args.process_name)
    t.torch.set_num_threads(2)
    analyze_folder(args.output, smoke_only=args.smoke_only, chr_series=args.chr_series,
                   allow_analyzer_override=args.allow_analyzer_override)
