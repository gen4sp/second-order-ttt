"""Persisted-job adapter. Every disposition and endpoint uses tttmem.episodes."""
from __future__ import annotations
import csv,json,math
from pathlib import Path

import numpy as np
from . import episodes as ep,rdo_analysis,rpe_analysis


def indexed_reads(record):
    out={}
    def put(step,rule,kind,q):
        key=(int(step),rule,kind)
        if key in out and out[key].get('acc')!=q.get('acc'):
            raise ValueError(f'conflicting read {key}')
        out[key]=q
    own=record.get('rule')
    for r in record.get('logs',[]):
        put(r['step'],own,'quick',r['quick'])
        for tag,q in r.get('cross',{}).items():put(r['step'],tag,'quick',q)
    for r in record.get('reads',[]):put(r['step'],r['rule'],r['set'],r['quick'])
    return out


def certificate(record):
    if not record.get('divergence_certified'):return None
    return ep.Divergence(int(record['t_div']),int(record['attempted']),int(record['nonfinite_local']))


def rows_for(index,rule,start,stop,every=5):
    return [dict(step=t,quick=index[(t,rule,'quick')]) for t in range(start+every,stop+1,every)
            if (t,rule,'quick') in index]


def branch_events(record,spec,*,smoke=False):
    start=int(record.get('fork_step',spec['metadata']['start']));H=int(spec['horizon']);tag=spec['rule']
    W=min(300,H-5) if smoke else 300
    if record.get('status')!='DONE' or not record.get('next_batch_verified',False):
        return {tag:dict(disposition='TRUNCATED')}
    try:index=indexed_reads(record)
    except (ValueError,KeyError):return {tag:dict(disposition='TRUNCATED')}
    own=rows_for(index,tag,start,start+H)
    cert=certificate(record)
    needed=[]
    # Dense gaps affect only the fine endpoint; all other required reads gate disposition.
    for schedule in spec['reads']:
        if schedule['every']==1 and not schedule['full']:continue
        first=schedule['start'];last=schedule['stop'];every=schedule['every']
        grid=list(range(first,last+1,every))
        kind='full' if schedule['full'] else 'quick'
        sr=[dict(step=t,quick=index[(t,schedule['rule'],kind)]) for t in grid
            if (t,schedule['rule'],kind) in index]
        needed.append((sr,grid))
    finals=record.get('final',{})
    if set(spec['final_rules'])-finals.keys():return {tag:dict(disposition='TRUNCATED')}
    disposition=ep.disposition(own,start,H,certificate=cert,needed=tuple(needed),finals=finals)
    if disposition==ep.Disposition.TRUNCATED:return {tag:dict(disposition='TRUNCATED')}
    dense=rows_for(index,tag,start,start+min(W+9,H),every=1)
    rules={tag}
    for schedule in spec['reads']:
        if schedule['every']==5 and not schedule['full']:rules.add(schedule['rule'])
    events={}
    for rule in rules:
        last=start+H if spec['metadata']['pack']!='rpe' or rule==tag else start+W+5
        horizon=last-start
        data=rows_for(index,rule,start,last)
        full=finals.get(rule,{}).get('acc')
        endpoint=ep.endpoints(data,start,horizon,window=min(W,horizon-5),certificate=cert,
                              full_acc=full,dense=dense if rule==tag else None,
                              require_final=rule==tag or spec['metadata']['pack']!='rpe')
        # The full branch contract was checked first, including finals and common reads.
        endpoint['disposition']=disposition.value
        endpoint.update(reacquired=endpoint.get('t_re') is not None,retention=endpoint.get('retained'),
                        NL2=endpoint.get('nl2'),NL_fine=endpoint.get('nl_fine'),L=endpoint.get('lost_time'),
                        first_outage=endpoint['episodes'][0]['outage'] if endpoint.get('episodes') else None,
                        eval_immediate=record.get('pre',{}).get(rule,{}).get('quick',{}).get('acc',0)>=.90,
                        start=start,rule=rule,branch=tag,seed=spec['metadata']['seed'],horizon=H)
        last_two=[index.get((t,rule,'quick'),{}).get('acc',float('nan')) for t in (start+H-5,start+H)]
        endpoint['bound_at_H']=bool(disposition!=ep.Disposition.DIVERGED and all(v>=.90 for v in last_two))
        if spec['metadata']['pack']=='rpe' and rule!=tag:
            endpoint['L']=None;endpoint['retention']=None
        if spec['metadata']['pack']!='rpe' and rule==tag:
            fine=ep.endpoints(rows_for(index,tag,start,start+min(25,H),1),start,min(25,H),
                              every=1,window=min(25,H)-1,certificate=cert,require_final=False)
            endpoint['tau_fine']=fine.get('tau')
        events[rule]=endpoint
    return events


def manipulation(rp,nc,start,H,*,prevention=False):
    rlogs=[r for r in rp.get('logs',[]) if start<int(r['step'])<=start+H]
    nlogs=[r for r in nc.get('logs',[]) if start<int(r['step'])<=start+H]
    if not rlogs or not nlogs:return True
    writes=[r.get('write',{}) for r in rlogs]
    nw=[r.get('write',{}) for r in nlogs]
    floor=all(w.get('floor_rel_error_max',math.inf)<=1e-4 for w in writes)
    def training_fraction(rows,count):
        den=sum(r.get('train_n',0) for r in rows)
        return sum(r.get(count,0) for r in rows)/den if den else math.inf
    clip=all(training_fraction(logs,'train_clip_count')<= (.05 if prevention else .01)
             for logs in (rlogs,nlogs))
    req=sum(r.get('train_added_req_sum',0) for r in nlogs)
    act=sum(r.get('train_added_actual_sum',0) for r in nlogs)
    match=(req==act==0 or req>0 and abs(act/req-1)<=1e-4)
    match &= training_fraction(nlogs,'train_match_fail_count')<=.001
    match &= all(w.get('match_fail',1)<=.001 and abs(w.get('attained_ratio',0)-1)<=1e-4 for w in nw)
    active=True if prevention else rp.get('pre_write',{}).get('floor_active',0)>=.5
    return not(floor and clip and match and active)


def write_gate(twin,rp,start,lead):
    own=branch_events(twin,dict(rule='ce',horizon=twin.get('horizon',400),metadata=dict(start=start,seed=0,pack='rpe'),
                             reads=[],final_rules=['ce']),smoke=False).get('ce',{})
    t_ref=own.get('t_loss')
    fallback=t_ref is None
    if fallback:t_ref=start+lead
    peaklogs=[r['write']['kv_write_med'] for r in twin.get('logs',[]) if start<r['step']<=t_ref and r.get('write',{}).get('kv_write_med') is not None]
    window=[r for r in rp.get('logs',[]) if t_ref-10<=r['step']<=t_ref+25]
    if not peaklogs or not window:return dict(passed=False,fallback=fallback,t_ref=t_ref)
    peak=max(peaklogs)
    if peak<=0:return dict(passed=False,fallback=fallback,t_ref=t_ref)
    ratios=np.array([r.get('write',{}).get('kv_write_med',np.nan)/peak for r in window])
    clip=max(r.get('train_clip_c0',r.get('write',{}).get('clip_c0',1)) for r in window)
    wm=float(np.min(ratios));over=float(np.median(ratios))>3 or clip>.05
    return dict(passed=bool(np.isfinite(ratios).all() and wm>=.8 and not over),WM=wm,
                over_write=bool(over),peak=peak,t_ref=t_ref,fallback=fallback,short_lead=t_ref-start<=25)


def analyze(evidence,output,*,smoke=False):
    manifest=evidence['manifest'];registered=manifest['provenance'].get('smoke_only',False)
    if registered and not smoke:raise ValueError('smoke output cannot receive production analysis')
    H=int(manifest['provenance']['effective_config']['horizon'])
    results=evidence['jobs'];specs=manifest['jobs'];cohorts=manifest['cohorts']
    units={block:{int(r['seed']):dict(seed=int(r['seed']),events={},manipulation={},write_gate={})
                  for r in cohorts[trigger]} for block,trigger in (('rdo','S1'),('rpe','t_acq'),('late','S1'))}
    series=[];events=[]
    for spec in specs:
        pack=spec['metadata']['pack'];seed=int(spec['metadata']['seed']);tag=spec['rule']
        rec=results.get(spec['job_id'],{})
        scored=branch_events(rec,spec,smoke=smoke)
        units[pack][seed]['events'][tag]=scored
        for rule,e in scored.items():events.append({**e,'block':pack,'seed':seed,'branch':tag,'eval_rule':rule})
        for r in rec.get('logs',[]):
            series.append(dict(block=pack,seed=seed,branch=tag,step=r['step'],start=spec['metadata']['start'],
                               acc=r['quick'].get('acc'),**r.get('write',{})))
        index=indexed_reads(rec)
        for (step,rule,kind),q in index.items():
            if rule!=tag or kind=='full':
                series.append(dict(block=pack,seed=seed,branch=tag,eval_rule=rule,set=kind,
                                   step=step,start=spec['metadata']['start'],acc=q.get('acc')))
    for block in ('rdo','rpe'):
        doses=('01','03','10','30') if block=='rdo' else ('30','100')
        leads=[e['t_loss']-e['start'] for u in units['rpe'].values() for e in [u['events'].get('ce',{}).get('ce',{})] if e.get('t_loss') is not None]
        lead=float(np.median(leads)) if leads else 300
        for seed,u in units[block].items():
            for d in doses:
                rp=results.get(f'{block}-s{seed}-rpf{d}',{});nc=results.get(f'{block}-s{seed}-ncf{d}',{})
                start=rp.get('fork_step',0)
                u['manipulation'][d]=manipulation(rp,nc,start,min(H,300) if block=='rpe' else H,prevention=block=='rpe')
                if block=='rpe':
                    gate=write_gate(results.get(f'rpe-s{seed}-ce',{}),rp,start,lead)
                    u['write_gate'][d]=gate['passed'];u.setdefault('write_diagnostics',{})[d]=gate
    rdo=rdo_analysis.score(list(units['rdo'].values()),H)
    rpe=rpe_analysis.score(list(units['rpe'].values()),H)
    for rec in (rdo,rpe):rec['smoke_only']=smoke;rec['provenance']=manifest['provenance']
    # Registered start-factor and late comparisons remain descriptive.
    paired_starts=[]
    for seed in sorted(set(units['rdo'])&set(units['rpe'])):
        for tag in ('rpf30','ncf30'):
            pair=[units[b][seed]['events'].get(tag,{}).get(tag,{}) for b in ('rdo','rpe')]
            paired_starts.append(dict(seed=seed,branch=tag,S1=pair[0],t_acq=pair[1]))
    late_comparisons=[]
    for spec in specs:
        if spec['metadata']['pack']!='late' or spec['rule']=='ce':continue
        seed=spec['metadata']['seed'];tag=spec['rule'];late=units['late'][seed]['events'].get(tag,{}).get(tag,{})
        early=results.get(f'rdo-s{seed}-{tag}',{});at=int(early.get('fork_step',0))
        sr=rows_for(indexed_reads(early),tag,at,at+300)
        cert=certificate(early)
        if cert and cert.t_div>at+300:cert=None
        first=ep.endpoints(sr,at,300,certificate=cert,require_final=False)
        late_comparisons.append(dict(seed=seed,branch=tag,S1_tau_300=first.get('tau'),S2_tau_300=late.get('tau'),
                                     delta=None if first.get('tau') is None or late.get('tau') is None else late['tau']-first['tau'],
                                     S1_disposition=first['disposition'],S2_disposition=late.get('disposition')))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,score in (('rdo',rdo),('rpe',rpe)):
        (output/f'{name}_scores.json').write_text(json.dumps(score,sort_keys=True,indent=2))
        (output/f'{name}_scores.txt').write_text(('SMOKE ONLY — no scientific verdict\n' if smoke else '')+json.dumps(score,indent=2)+'\n')
        _csv(output/f'{name}_events.csv',[e for e in events if e['block']==name])
        _csv(output/f'{name}_series.csv',[r for r in series if r['block']==name])
    _csv(output/'rdo_disposition.csv',cohorts['dispositions'])
    (output/'late_and_start_factor.json').write_text(json.dumps(dict(late=list(units['late'].values()),
                                                       late_comparisons=late_comparisons,start_factor=paired_starts),indent=2))
    (output/'units.json').write_text(json.dumps(units,indent=2))
    print('RDO_ANALYSIS '+json.dumps(dict(smoke_only=smoke,headline=rdo['headline'])),flush=True)
    print('RPE_ANALYSIS '+json.dumps(dict(smoke_only=smoke,verdicts={d:r['verdict'] for d,r in rpe['doses'].items()})),flush=True)
    return rdo,rpe


def _csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
