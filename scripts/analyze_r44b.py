"""Wave-B confirmatory analysis. Pure stdlib; training seeds are the units.

Usage: analyze_r44b.py [--process-name NAME] LOG OUT_DIR
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics


def finite(x):
    if isinstance(x,dict):
        return {k:finite(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [finite(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x


def tail(k,n,p=.5):
    return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1)) if n else 1.


def mcnemar(pairs):
    b=sum(a and not c for a,c in pairs); c=sum(c and not a for a,c in pairs)
    return dict(n=len(pairs),positive=b,negative=c,discordances=b+c,p=tail(b,b+c))


def binomial_interval(k,n,alpha=.05):
    if not n:
        return [0.,1.]
    def invert(fn,target):
        lo,hi=0.,1.
        for _ in range(80):
            mid=(lo+hi)/2
            if fn(mid)<target:
                lo=mid
            else:
                hi=mid
        return (lo+hi)/2
    lower=0. if k==0 else invert(lambda p:tail(k,n,p),alpha/2)
    upper=1. if k==n else invert(lambda p:tail(k+1,n,p),1-alpha/2)
    return [lower,upper]


def sign_test(count_pairs,events=4096):
    deltas=[(a-b)/events for a,b in count_pairs]
    pos=sum(a>b for a,b in count_pairs); neg=sum(a<b for a,b in count_pairs)
    return dict(n=len(deltas),m=pos+neg,positive=pos,negative=neg,ties=len(deltas)-pos-neg,
                differences=deltas,mean=statistics.mean(deltas) if deltas else None,
                median=statistics.median(deltas) if deltas else None,p=tail(pos,pos+neg),
                positive_probability_ci95=binomial_interval(pos,pos+neg))


def _beta_fraction(a,b,x):
    # Lentz continued fraction for regularized incomplete beta.
    qab,qap,qam=a+b,a+1,a-1
    c=1.; d=1-qab*x/qap
    tiny=1e-300
    if abs(d)<tiny: d=tiny
    d=1/d; h=d
    for m in range(1,301):
        aa=m*(b-m)*x/((qam+2*m)*(a+2*m))
        d=1+aa*d; d=tiny if abs(d)<tiny else d
        c=1+aa/c; c=tiny if abs(c)<tiny else c
        d=1/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m))
        d=1+aa*d; d=tiny if abs(d)<tiny else d
        c=1+aa/c; c=tiny if abs(c)<tiny else c
        d=1/d; factor=d*c; h*=factor
        if abs(factor-1)<1e-14: break
    return h


def regularized_beta(x,a,b):
    if x<=0: return 0.
    if x>=1: return 1.
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x<(a+1)/(a+b+2): return bt*_beta_fraction(a,b,x)/a
    return 1-bt*_beta_fraction(b,a,1-x)/b


def t5_cdf(value):
    if value==0: return .5
    prob=.5*regularized_beta(5/(5+value*value),2.5,.5)
    return prob if value<0 else 1-prob


def ni(diffs):
    valid=len(diffs)==6 and all(isinstance(d,(float,int)) and math.isfinite(d) for d in diffs)
    if not valid: return dict(n=len(diffs),differences=diffs,p=1.,ub95=None,valid=False)
    mean=statistics.mean(diffs); sd=statistics.stdev(diffs)
    p=(0. if mean<.01 else 1.) if sd==0 else t5_cdf((mean-.01)/(sd/math.sqrt(6)))
    bound=mean+2.015048*sd/math.sqrt(6)
    return dict(n=6,differences=diffs,mean=mean,sd=sd,ub95=bound,p=p,valid=True,
                passed=bound<.01,assumption="approximately normal seed-level paired differences")


def holm(values):
    order=sorted(values,key=values.get)
    out={}; previous=0.; proceed=True
    for i,name in enumerate(order):
        p=values[name]; adjusted=max(previous,min(1.,(2-i)*p)); previous=adjusted
        passed=proceed and p<.05/(2-i)
        proceed=passed
        out[name]=dict(p=p,adjusted_p=adjusted,threshold=.05/(2-i),passed=passed)
    return out


def _num(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def _good(row):
    return bool(row and row.get("complete") and not row.get("error") and not row.get("failure") and
                not row.get("forced_for_smoke") and row.get("eval") and
                all(isinstance(row["eval"].get(k),(int,float)) and math.isfinite(row["eval"][k]) for k in ("ce","ce_late")))


def _paired_hash(a,b):
    if not a or not b: return False
    return all(a.get(k) is not None and a.get(k)==b.get(k) for k in
               ("initial_sha256","code_sha256","rolling_data","first_batch_sha1")) and \
           a.get("eval",{}).get("token_sha256") is not None and \
           a["eval"]["token_sha256"]==b.get("eval",{}).get("token_sha256") and \
           [q.get("token_sha256") for q in a.get("quick",[])]==[q.get("token_sha256") for q in b.get("quick",[])]


def analyze(payload):
    rows=payload.get("cells",[]); config=payload.get("config",{})
    indexed={}; duplicates=[]
    for row in rows:
        key=(row.get("task"),row.get("stage"),row.get("seed"),row.get("arm"))
        if key in indexed: duplicates.append(key)
        indexed[key]=row
    def cell(task,stage,s,arm): return indexed.get((task,stage,s,arm))
    smoke=config.get("smoke",True) or any(r.get("forced_for_smoke") for r in rows)
    retention=[]; echo_valid=[]; missing=[]
    for s in range(10):
        a,c,l=[cell("echo","fresh",s,arm) for arm in ("e2e1emb_aice","e2e1emb","e2e1emb_lin")]
        valid=all(_good(r) and r.get("successful_steps")==1500 and r.get("s5",{}).get("passed") for r in (a,c,l)) and _paired_hash(a,c) and _paired_hash(a,l)
        if valid:
            valid=all(r.get("quick") and r["quick"][-1].get("local_step")==1500 and
                      r["quick"][-1].get("query_events")==1024 and
                      type(r["quick"][-1].get("query_hits")) is int and
                      0<=r["quick"][-1]["query_hits"]<=1024 for r in (a,c,l))
        echo_valid.append(valid)
        if valid:
            def retained(r):
                q=r.get("quick",[])[-1] if r.get("quick") else {}
                return q.get("local_step")==1500 and q.get("query_events")==1024 and q.get("query_hits",0)*10>=9*1024
            retention.append((retained(a),retained(c)))
        else: missing.append(f"echo/s{s}")
    R=mcnemar(retention)
    diffs={k:[] for k in ("ce","ce_late")}; icl_valid=[]; ni_seeds=[]
    for s in range(6):
        a,c=[cell("icl_mc","fresh",s,arm) for arm in ("e2e1_aice","e2e1")]
        valid=all(_good(r) and r.get("successful_steps")==4000 and r.get("plain") for r in (a,c)) and _paired_hash(a,c)
        icl_valid.append(valid)
        if valid:
            for k in diffs: diffs[k].append(a["eval"][k]-c["eval"][k])
            ni_seeds.append(dict(seed=s,overall=a["eval"]["ce"]-c["eval"]["ce"],late=a["eval"]["ce_late"]-c["eval"]["ce_late"]))
        else: missing.append(f"icl_mc/s{s}")
    O,L=ni(diffs["ce"]),ni(diffs["ce_late"])
    reassigned={}; reassignment_valid=[]; count_pairs=[]; reassign_seeds=[]
    for r in payload.get("reassign",[]):
        key=(r.get("historical",False),r.get("seed"),r.get("arm"))
        if key in reassigned: duplicates.append(key)
        reassigned[key]=r
    expected={(c,m) for c in ("reassign","removed","unchanged") for m in ("streaming","query_freeze","write_snapshot")}
    def conditions_of(r):
        if not (r and not r.get("error") and len(r.get("conditions",[]))==9 and r.get("mechanism")):
            return None
        conditions={(c["condition"],c["mode"]):c for c in r["conditions"]}
        ok=set(conditions)==expected and all(c.get("events")==4096 and type(c.get("new_hits")) is int and
                                              0<=c["new_hits"]<=4096 and type(c.get("old_hits")) is int and
                                              0<=c["old_hits"]<=4096 and c["new_hits"]+c["old_hits"]<=4096 and
                                              c.get("tokens_sha256") is not None for c in conditions.values())
        return conditions if ok else None
    def hashes(conditions):
        return {key:row.get("tokens_sha256") for key,row in conditions.items()}
    historical_lin_valid=[]
    for s in range(10):
        # Primary: fresh AICE vs fresh lin (C08/C09). The historical R4-2L lin
        # re-evaluation is an audit: reported, never a gate on the primary.
        a,l=[conditions_of(reassigned.get((False,s,arm))) for arm in ("e2e1emb_aice","e2e1emb_lin")]
        h=conditions_of(reassigned.get((True,s,"e2e1emb_lin")))
        valid=bool(a and l and hashes(a)==hashes(l))
        audit=bool(valid and h and hashes(h)==hashes(a))
        historical_lin_valid.append(audit)
        if valid:
            pa,pl=a[("reassign","streaming")],l[("reassign","streaming")]
            ph=h[("reassign","streaming")] if audit else {}
            count_pairs.append((pa["new_hits"],pl["new_hits"]))
            reassign_seeds.append(dict(seed=s,aice_hits=pa["new_hits"],lin_hits=pl["new_hits"],
                                        historical_lin_hits=ph.get("new_hits"),difference=(pa["new_hits"]-pl["new_hits"])/4096,
                                        fresh_minus_historical_lin=(pl["new_hits"]-ph["new_hits"])/4096 if audit else None))
        reassignment_valid.append(valid)
        if not valid: missing.append(f"reassign/s{s}")
        elif not audit: missing.append(f"historical_lin_audit/s{s}")
    B=sign_test(count_pairs)
    kpairs=[]; conditional=[]; kvalid=[]
    for s in range(10):
        one=cell("echo","stage1",s,"e2e1emb"); ctl=cell("echo","masked",s,"e2e1emb")
        ce,lin,control=[cell("echo","stage2",s,a) for a in ("kamin2","kaminlin2","kaminctl2")]
        sw=one.get("switch",{}) if one else {}; m=ctl.get("switch",{}) if ctl else {}
        twins=[ce,lin,control]
        # Stage-2 algorithmic failure counts as not acquired; infrastructure missing stays incomplete.
        finished=lambda r: bool(r and not r.get("error") and (
            (r.get("complete") and r.get("successful_steps")==1000) or r.get("failure")=="algorithmic_nonfinite"))
        valid=bool(one and one.get("complete") and sw.get("passed") and sw.get("rho_unchanged") and
                   not sw.get("forced_for_smoke") and sw.get("audit",{}).get("events")==4096 and
                   _num(sw.get("audit",{}).get("mean")) and sw["audit"]["mean"]>=.99 and
                   len(sw.get("trigger",[]))>=2 and all(_num(x.get("mean")) and x["mean"]>=.99 for x in sw["trigger"][-2:]) and
                   ctl and ctl.get("complete") and m.get("chance_pass") and m.get("rho_unchanged") and
                   _num(m.get("masked",{}).get("mean")) and .04<=m["masked"]["mean"]<=.09 and m.get("masked",{}).get("events")==512 and
                   one.get("rolling_data")==ctl.get("rolling_data") and all(finished(r) for r in twins))
        if valid:
            same=all(ce.get("clone",{}).get(k)==lin.get("clone",{}).get(k) and ce.get("clone",{}).get(k) is not None for k in
                     ("model_sha256","optimizer_sha256","rng_sha256","snapshot_sha256","audit"))
            valid=valid and same and all(r.get("quick",[{}])[0].get("local_step")==0 for r in twins)
        acquired=lambda r: bool(r and not r.get("failure") and r.get("ever_bound"))
        if all(finished(r) for r in (ce,control)):
            conditional.append((acquired(control),acquired(ce)))
        if valid: kpairs.append((acquired(control),acquired(ce)))
        else: missing.append(f"kamin/s{s}")
        kvalid.append(valid)
    K=mcnemar(kpairs)
    history=payload.get("historical_icl",[])
    historical_valid=len(history)==6 and {r.get("seed") for r in history}==set(range(6)) and all(r.get("eval") and not r.get("error") for r in history)
    if not historical_valid: missing.append("historical_icl_audit")
    validity=dict(echo=all(echo_valid),icl_mc=all(icl_valid),reassign=all(reassignment_valid),
                  historical_icl=historical_valid,historical_lin_audit=all(historical_lin_valid),
                  kamin=all(kvalid),no_duplicates=not duplicates,scientific=not smoke)
    # Historical icl_mc (C12) and lin reassign (C08) re-evaluations are audits only:
    # reported in validity/missing, not conjuncts of the AICE headline.
    a_valid=all(validity[k] for k in ("echo","icl_mc","reassign","no_duplicates","scientific"))
    k_valid=all(validity[k] for k in ("kamin","no_duplicates","scientific"))
    favorable_A=R["positive"]>R["negative"] and B["median"] is not None and B["median"]>0 and O.get("ub95") is not None and O["ub95"]<.01 and L.get("ub95") is not None and L["ub95"]<.01
    p_A=max(R["p"],O["p"],L["p"],B["p"]) if a_valid and favorable_A else 1.
    p_K=K["p"] if k_valid and K["positive"]>K["negative"] else 1.
    family=holm(dict(AICE=p_A,GM3r=p_K))
    return finite(dict(status="smoke_only" if smoke else "complete" if a_valid and k_valid else "incomplete_or_invalid",
                       p_R=R["p"] if validity["echo"] and not smoke else 1.,
                       p_O=O["p"] if validity["icl_mc"] and not smoke else 1.,
                       p_L=L["p"] if validity["icl_mc"] and not smoke else 1.,
                       p_B=B["p"] if validity["reassign"] and not smoke else 1.,p_A=p_A,p_K=p_K,
                       retention=R,NI_overall=O,NI_late=L,reassign_sign=B,Kamin=K,
                       NI_seed_differences=ni_seeds,reassign_seed_differences=reassign_seeds,
                       conditional_Kamin=mcnemar(conditional),validity=validity,missing=missing,duplicates=duplicates,
                       holm=family,primary_rule="aice" if family["AICE"]["passed"] else "lin",
                       GM3r="PASS" if family["GM3r"]["passed"] else "INCONCLUSIVE" if not k_valid else "NONPASS",
                       nominal_AICE_eligibility=a_valid and max(R["p"],O["p"],L["p"])<.05,
                       nominal_RB=a_valid and B["p"]<.05 and B["median"]>0,
                       infrastructure=dict(phases=payload.get("phases"),driver_error=payload.get("driver_error"))))


def parse(path):
    data=dict(config={},cells=[],reassign=[],historical_icl=[])
    for line in Path(path).read_text().splitlines():
        if line.startswith("R44B_WAVE_A_AUDIT "):
            data["historical_wave_a"]=json.loads(line.removeprefix("R44B_WAVE_A_AUDIT "))
        for kind,key in (("R44B_CONFIG ","config"),("R44B_CELL ","cells"),("R44B_REASSIGN ","reassign"),("R44B_HISTORICAL_ICL ","historical_icl")):
            if line.startswith(kind):
                row=json.loads(line[len(kind):])
                if key=="config": data[key]=row.get("wave_b",{})
                else: data[key].append(row)
    return data


def write_csv(path,rows):
    rows=finite(rows)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys or ["status"]); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def write_tables(payload,summary,out):
    out=Path(out); data=out/"data"; data.mkdir(parents=True,exist_ok=True)
    cells=payload.get("cells",[])
    write_csv(data/"cells.csv",cells)
    for key,name in (("quick","quick-trajectories"),("probes","probes")):
        write_csv(data/(name+".csv"),[dict(cell_id=c.get("cell_id"),seed=c.get("seed"),arm=c.get("arm"),**r) for c in cells for r in c.get(key,[])])
    write_csv(data/"eta.csv",[dict(cell_id=c.get("cell_id"),eta=c.get("eta")) for c in cells])
    write_csv(data/"kamin-switch-audit.csv",[dict(seed=c.get("seed"),stage=c.get("stage"),**c.get("switch",{})) for c in cells if c.get("stage") in ("stage1","masked")])
    write_csv(data/"kamin-acquisition.csv",[c for c in cells if c.get("stage")=="stage2"])
    write_csv(data/"kaminlin2.csv",[c for c in cells if c.get("arm")=="kaminlin2"])
    for key,name in (("conditions","reassign-per-seed"),("mechanism","write2-mechanism")):
        write_csv(data/(name+".csv"),[dict(seed=c.get("seed"),arm=c.get("arm"),historical=c.get("historical"),**r) for c in payload.get("reassign",[]) for r in c.get(key,[])])
    write_csv(data/"historical-lin-audit.csv",[c for c in payload.get("reassign",[]) if c.get("historical")])
    write_csv(data/"historical-icl-audit.csv",payload.get("historical_icl",[]))
    historical=payload.get("historical_wave_a",{})
    anchors=[]
    for old in historical.get("cells",[]):
        fresh=next((c for c in cells if c.get("task")=="echo" and c.get("stage")=="fresh" and
                   c.get("arm")=="e2e1emb" and c.get("seed")==old["seed"]),{})
        anchors.append(dict(seed=old["seed"],source_run=historical.get("source_run"),source_commit=historical.get("source_commit"),
                            historical_bound_final=old.get("bound_final"),fresh_bound_final=fresh.get("bound_final"),
                            historical_bound_ever=old.get("bound_ever"),fresh_bound_ever=fresh.get("ever_bound"),
                            first_batch_matches=old.get("first_batch_sha1")==fresh.get("first_batch_sha1")))
    write_csv(data/"historical-wave-a-audit.csv",anchors)
    write_csv(data/"NI-seed-differences.csv",summary["NI_seed_differences"])
    write_csv(data/"reassign-seed-differences.csv",summary["reassign_seed_differences"])
    transitions=[]
    for c in cells:
        previous=False; losses=[]; recoveries=[]; first=None
        for q in c.get("quick",[]):
            if q.get("local_step")==0 or not q.get("query_events"):
                continue
            bound=q["query_hits"]*10>=9*q["query_events"]
            if bound and first is None: first=q["local_step"]
            elif bound and not previous: recoveries.append(q["local_step"])
            if previous and not bound: losses.append(q["local_step"])
            previous=bound
        transitions.append(dict(cell_id=c.get("cell_id"),seed=c.get("seed"),arm=c.get("arm"),
                                first_bound=first,losses=losses,recoveries=recoveries,bound_final=previous))
    write_csv(data/"acquisition-losses-recoveries.csv",transitions)
    (out/"summary.json").write_text(json.dumps(finite(summary),allow_nan=False,indent=2)+"\n")


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--process-name",default="TTM-r44b-92-eval")
    ap.add_argument("log"); ap.add_argument("out_dir"); args=ap.parse_args(argv)
    os.environ["PROCESS_NAME"]=args.process_name
    # Keep the analyzer stdlib-only: process-name argument/environment are searchable metadata.
    payload=parse(args.log); summary=analyze(payload); write_tables(payload,summary,args.out_dir)
    print("R44B_SUMMARY "+json.dumps(summary,allow_nan=False))


if __name__=="__main__": main()
