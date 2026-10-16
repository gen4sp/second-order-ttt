"""FRS-A analyzer; smoke evidence never receives a confirmatory verdict."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tttmem import frs_cohorts as cohorts
from tttmem import triggers


def contract(pool, *, smoke=False) -> list[str]:
    issues = []
    cfg = pool.get("provenance", {}).get("effective_config", {})
    if pool.get("schema") != 1:
        issues.append("unsupported pool schema")
    if bool(pool.get("smoke_only")) != smoke:
        issues.append("explicit analyzer mode must match the driver mode")
    grid, terminal = int(cfg.get("log_every", 5)), int(cfg.get("steps", 1500))
    ac = cohorts.chr_analysis()
    for main in pool.get("cells", {}).values():
        if main.get("error"):
            continue
        rows = main.get("logs", [])
        if not main.get("diverged") and not ac.grid_check(rows, grid, terminal, grid,
                            int(cfg.get("chunk_early", 5)), int(cfg.get("chunk_late", 6)))["ok"]:
            issues.append(f"{main['cell_id']}: main grid mismatch")
        if not smoke:
            found = triggers.detect(rows)
            for name in ("t_acq", "t_loss", "S1"):
                online = main.get("events", {}).get(name)
                offline = found.get(name)
                if (online is None) != (offline is None) or (online is not None and
                    (online["event_step"], online["detected_step"]) != (offline.event_step, offline.detected_step)):
                    issues.append(f"{main['cell_id']}: online/offline {name} mismatch")
    if not smoke:
        if cfg.get("protocol") != "frs-a-full-v2" or terminal != 1500 or grid != 5:
            issues.append("production protocol mismatch")
        if not pool.get("provenance", {}).get("seal_sha256"):
            issues.append("production evidence lacks seal hash")
        want = {cohorts.FO: set(range(200,216)), cohorts.SO: set(range(200,212)),
                cohorts.NOISE: set(range(200,212))}
        for arm, seeds in want.items():
            if {int(r["seed"]) for r in pool["cells"].values() if r.get("arm") == arm} != seeds:
                issues.append(f"{arm}: seed manifest mismatch")
    return issues


def score(pool, *, smoke=False, frsb=None) -> dict:
    cells = list(pool.get("cells", {}).values())
    jobs = list(pool.get("jobs", {}).values())
    errors = contract(pool, smoke=smoke)
    arms = {arm: {int(r["seed"]): r for r in cells if r.get("arm") == arm}
            for arm in (cohorts.FO, cohorts.SO, cohorts.NOISE)}
    # R-SO and R-NOISE remain on 200–211 even if SO 212–215 is later added for TRG.
    if not smoke:
        arms = {arm: {s:r for s,r in values.items() if s <= 211} for arm,values in arms.items()}
    family = {"R-SO": cohorts.paired_records(arms[cohorts.SO], arms[cohorts.FO]),
              "R-NOISE": cohorts.paired_records(arms[cohorts.NOISE], arms[cohorts.FO]),
              "R-RP": cohorts.h5_adapter(cells, jobs, smoke=smoke,
                                          frozen_slots=pool.get('frozen_h5_slots'))}
    from tttmem.frs_family import validate_family_inputs, family_pair
    if not smoke or frsb is not None:
        validate_family_inputs(frsb, smoke=smoke)
    for name in ("R-LIN", "R-ADDR"):
        source = (frsb or {}).get("family_inputs", {}).get(name)
        family[name] = family_pair(source,smoke=smoke) if source else cohorts.exact_mcnemar([], [])
        if family[name]["n"] < 10:
            family[name]["nominal_p"], family[name]["p"] = family[name]["p"], 1.
        family[name]["reason"] = None if source else "FRS-B evidence not supplied"
    adjusted = cohorts.holm5({name:r["p"] for name,r in family.items()})
    for name,row in family.items():
        row["holm_p"] = adjusted[name]
        row["verdict"] = cohorts.paired_verdict(row, adjusted[name])
        clean = row.get("without_divergence")
        if clean:
            alternative = {n:r["p"] for n,r in family.items()}
            alternative[name] = clean["p"] if clean["n"] >= 10 else 1.
            row["divergence_sensitive"] = cohorts.paired_verdict(clean, cohorts.holm5(alternative)[name]) != row["verdict"]
    ac = cohorts.chr_analysis()
    acquired = cohorts.first_acquired(cells, limit=16, smoke=smoke)
    observed = [r for r in acquired if cohorts.outcome(r) is not None]
    retained = sum(cohorts.outcome(r) for r in observed)
    r_fo = dict(n=len(observed), retained=retained, ci95=ac.exact_ci(retained,len(observed)),
                verdict="NOT EVALUABLE" if len(observed)<8 else "REPLICATED" if retained<=1
                else "NOT REPLICATED" if retained>=3 else "INCONCLUSIVE")
    h1_units, h3_units, h4 = [], [], {"SLOW":[], "BB":[]}
    h4_excluded = []
    early = int(pool.get("provenance",{}).get("effective_config",{}).get("chunk_early",5))
    late = int(pool.get("provenance",{}).get("effective_config",{}).get("chunk_late",6))
    for main in acquired:
        if main.get("diverged") or main.get("error") or not main.get("logs"):
            continue
        event = ac.events(ac.series(main["logs"], early, late))
        if event["t_loss"] is not None and event["category"] != "unknown":
            h1_units.append(dict(seed=main["seed"], **event))
            h3_units.append(dict(seed=main["seed"], t_bb=event["t_bb"], t_loss=event["t_loss"],
                                 after_loss=event["t_bb"] is not None and event["t_bb"]>event["t_loss"]))
        if cohorts.outcome(main) is False:
            fin = main.get("final",{}).get("ce",{})
            iv = fin.get("intervention",{})
            a5, a6 = fin.get(f"acc{early}"), fin.get(f"acc{late}")
            if a5 is None or a6 is None or (a5 - a6 < .20 and
                    not math.isclose(a5-a6,.20,rel_tol=0.,abs_tol=1e-12)):
                h4_excluded.append(dict(seed=main['seed'], reason='D < .20 or missing',
                                        D=None if a5 is None or a6 is None else a5-a6))
                continue
            for variant in h4:
                treatment = iv.get(variant,{}).get(str(late))
                random = iv.get("RND_"+variant,{}).get(str(late))
                sham = iv.get("SHAM5_"+variant,{}).get(str(early))
                if fin.get("decomp_gate_ok") and all(v is not None for v in (a5,a6,treatment,random,sham)):
                    h4[variant].append(dict(seed=main["seed"], repaired=treatment-a6 >= (a5-a6)/2 and
                                           treatment-random>=.05))
    n, after, before = len(h1_units), sum(u["category"]=="after_loss" for u in h1_units), sum(u["category"]=="before_persistent" for u in h1_units)
    r_h1 = dict(n=n, after_loss=after, before_persistent=before, ci95=ac.exact_ci(after,n), units=h1_units,
                 verdict="NOT EVALUABLE" if n<6 else "REPLICATED" if after>=math.ceil(.75*n)
                 else "NOT REPLICATED" if before>=math.ceil(.5*n) else "INCONCLUSIVE")
    r_h4 = {v:dict(n=len(us),repaired=sum(u["repaired"] for u in us),units=us,
                        ci95=ac.exact_ci(sum(u["repaired"] for u in us),len(us)),
                        verdict="NOT EVALUABLE" if len(us)<6 else
                        "REPLICATED" if sum(u["repaired"] for u in us)<=math.floor(.5*len(us)) else
                        "NOT REPLICATED" if sum(u["repaired"] for u in us)>=math.ceil(.75*len(us)) else "INCONCLUSIVE")
              for v,us in h4.items()}
    out = dict(smoke_only=smoke, contract_issues=errors, family=family, R_FO=r_fo, R_H1=r_h1,
               R_H4_excluded=h4_excluded,
               R_H3=dict(n=len(h3_units),units=h3_units,descriptive=True), R_H4=r_h4,
               R_TR=cohorts.transient_adapter([r for r in cells if smoke or r.get("arm") == cohorts.FO
                                               or int(r.get("seed", 0)) <= 211]))
    fo_deltas, so_deltas = [], []
    for main in cells:
        final = (main.get("final") or {}).get("ce", {})
        a5, a6 = final.get(f"acc{early}"), final.get(f"acc{late}")
        if main.get("diverged") or a5 is None or a6 is None:
            continue
        if main.get("arm") == cohorts.FO and cohorts.acquisition(main) is not None and cohorts.outcome(main) is False:
            fo_deltas.append(a5-a6)
        elif main.get("arm") == cohorts.SO and cohorts.outcome(main) is True:
            so_deltas.append(a5-a6)
    out["R_CH6"] = dict(FO=cohorts.count_rule(sum(d>=.30 for d in fo_deltas),len(fo_deltas)),
                         SO=cohorts.count_rule(sum(d<=.05 for d in so_deltas),len(so_deltas)),
                         descriptive=True)
    out["R_H3"]["after_loss"] = cohorts.count_rule(sum(u["after_loss"] for u in h3_units),len(h3_units))
    if smoke:
        for row in family.values():
            row["verdict"] = "SMOKE ONLY — NOT SCORED"
        out["status"] = "SMOKE ONLY — NOT SCORED"
    else:
        out["status"] = "CONTRACT FAILED" if any("mismatch" in e or "lacks" in e for e in errors) else "SCORED"
        if out["status"] == "CONTRACT FAILED":
            for row in family.values():
                row["verdict"] = "CONTRACT FAILED — NOT SCORED"
    return out


def write_outputs(pool, scores, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"frs_scores.json").write_text(json.dumps(scores,sort_keys=True,indent=2))
    text = [scores["status"]] + [f"{name}: {r['verdict']} n={r['n']} p={r['p']:.6g} Holm={r['holm_p']:.6g}"
                                 for name,r in scores["family"].items()]
    text += [f"contract: {issue}" for issue in scores["contract_issues"]]
    (out/"frs_scores.txt").write_text("\n".join(text)+"\n")
    with open(out/"frs_units.csv","w",newline="") as f:
        fields = ["claim","seed","included","test","comparator","reason"]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name,row in scores["family"].items():
            for unit in row.get("units",[]):
                writer.writerow(dict(claim=name,seed=unit["seed"],included=True,test=unit["test"],comparator=unit["comparator"],reason=""))
            for seed in row.get("missing",[]):
                writer.writerow(dict(claim=name,seed=seed,included=False,reason="missing pair"))
    print("\n".join(text))


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("input",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--frsb",type=Path)
    parser.add_argument("--process-name",default=os.environ.get("PROCESS_NAME","TTM-frs-78-analyze"))
    args=parser.parse_args(argv)
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    pool=json.loads(args.input.read_text())
    frsb=json.loads(args.frsb.read_text()) if args.frsb else None
    scores=score(pool,smoke=args.smoke_only,frsb=frsb)
    write_outputs(pool,scores,args.out)
    if scores["contract_issues"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
