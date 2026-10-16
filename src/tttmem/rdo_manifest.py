"""Merged registered manifest; the sole schedule source."""
from __future__ import annotations
from pathlib import Path

from .floor_rules import parse_rule, CE

S1_TAGS=('ce','rpf01','rpf03','rpf10','rpf30','ncf01','ncf03','ncf10','ncf30')
RPE_TAGS=('ce','rp10','rpf30','ncf30','rpf100','ncf100')
FLOORS=('rpf01','rpf03','rpf10','rpf30')
MATRIX=('ce','rp10','rpf30','rpf100')


def cohorts(states,cells,config):
    """Read pre-fork contract fields only. No continuation outcomes are consulted."""
    pool=config.get('candidate_seeds',list(range(200,216)))
    limit=int(config.get('limit',12))
    rows={}
    for r in states:
        if r.get('arm')=='e2e1emb': rows[(int(r['seed']),r['trigger'])]=r
    byseed={int(c['seed']):c for c in cells if c.get('arm')=='e2e1emb'}
    out={'S1':[],'t_acq':[],'dispositions':[]}
    occupied = {'S1': 0, 't_acq': 0}
    for seed in sorted(pool):
        cell=byseed.get(seed,{})
        for trigger in ('S1','t_acq'):
            r=rows.get((seed,trigger)); reason='eligible' if trigger=='S1' else 'enrolled'
            if not r and (cell.get('error') or cell.get('status') == 'MISSING'):
                reason = 'missing main; frozen seed slot'
                if occupied[trigger] < limit:
                    occupied[trigger] += 1
                    out[trigger].append(dict(seed=seed,cell_id=cell.get('cell_id'),trigger=trigger,
                                             state_step=None,missing=True))
            elif not r:
                reason=('no-acquisition' if (seed,'t_acq') not in rows else
                        'no-loss' if not cell.get('t_loss',cell.get('events',{}).get('t_loss')) else 'no-S1')
            elif r.get('verified') is False or not r.get('sha256'):
                reason='state-validation failed'
                if int(r['state_step']) <= (1000 if trigger=='S1' else 1100) and occupied[trigger]<limit:
                    occupied[trigger]+=1
                    out[trigger].append(dict(r,missing=True))
            elif int(r['state_step'])>(1000 if trigger=='S1' else 1100):
                reason='S1 > 1000' if trigger=='S1' else 't_acq > 1100'
            elif trigger=='S1' and r.get('ce_bound_at_S1',False):
                reason='CE-bound-at-S1'
            elif trigger=='S1' and cell.get('logs') and not config.get('smoke_only') and any(
                x.get('quick',{}).get('acc',1)>=.90 for x in cell['logs']
                if x['step'] in (int(r['state_step'])-5,int(r['state_step']))):
                reason='CE-bound-at-S1'
            elif occupied[trigger]>=limit: reason='not needed'
            else:
                out[trigger].append(r)
                occupied[trigger] += 1
            out['dispositions'].append(dict(seed=seed,block=trigger,disposition=reason))
    return out


def build_jobs(states,cells,config):
    from .frs_jobs import JobSpec,ReadSchedule
    config={**config,**config.get('rdo_rpe',{})}
    selected=cohorts(states,cells,config)
    H=int(config.get('horizon',400)); late=int(config.get('late_horizon',300))
    smoke=bool(config.get('smoke_only',False))
    W=min(300,H-5) if smoke else 300
    dense=min(309,H) if smoke else 309
    root=Path(config.get('pool_output_dir','outputs/frs-a'))
    jobs=[]
    def schedules(tags,every,start,stop,full=False):
        return [ReadSchedule(parse_rule(t),every=every,start=start,stop=stop,full=full) for t in tags]
    def spec(block,row,tag,horizon,reads,pre,final,after=(),state=None,save=()):
        seed=int(row['seed']); start=int(row['state_step'])+(H if block=='late' else 0)
        file=Path(row['file'])
        if not file.is_absolute() and not file.is_relative_to(root): file=root/file
        return JobSpec(f'{block}-s{seed}-{tag}',str(file) if state is None else state,
                       row['sha256'] if state is None else '',parse_rule(tag),horizon,
                       after=tuple(after),reads=tuple(reads),final_rules=tuple(parse_rule(t) for t in final),
                       save_at=tuple(save),pre_rules=tuple(parse_rule(t) for t in pre),
                       expected_hashes=row.get('hashes',{k:row[k] for k in ('config','source','rules','quick','eval','probe') if k in row}),
                       cell_id=row['cell_id'],metadata=dict(pack=block,seed=seed,start=start,arm=row['arm'],
                       backend_factory='tttmem.exp_rdo:create_backend'))
    for row in selected['S1']:
        if row.get('missing'):continue
        start=int(row['state_step']); s2=start+H
        dolate=bool(config.get('late',True)) and start<=800
        twin_id=f'rdo-s{row["seed"]}-ce'
        for tag in S1_TAGS:
            common=FLOORS if tag=='ce' else ('rpf'+tag[3:],) if tag.startswith('ncf') else ()
            reads=schedules(common,5,start+5,start+H)
            if tag!='ce': reads+=schedules(('ce',),25,start+25,start+H)
            reads+=schedules((tag,),1,start+1,start+min(25,H))
            finals=list(('ce',)+FLOORS)+( [tag] if tag.startswith('ncf') else [])
            jobs.append(spec('rdo',row,tag,H,reads,S1_TAGS if tag=='ce' else (),finals,
                             save=(s2,) if tag=='ce' and dolate else ()))
        if dolate:
            for tag in ('ce','rpf03','rpf30'):
                reads=(schedules(('rpf03','rpf30'),5,s2+5,s2+late) if tag=='ce' else
                       schedules(('ce',),25,s2+25,s2+late))
                reads+=schedules((tag,),1,s2+1,s2+min(25,late))
                jobs.append(spec('late',row,tag,late,reads,('ce',)+FLOORS if tag=='ce' else (),('ce',)+FLOORS,after=(twin_id,),
                                 state=f'job:{twin_id}:{s2}'))
    for row in selected['t_acq']:
        if row.get('missing'):continue
        start=int(row['state_step'])
        for tag in RPE_TAGS:
            common=('rp10','rpf30','rpf100') if tag=='ce' else ('rpf'+tag[3:],) if tag.startswith('ncf') else ()
            reads=schedules(common,5,start+5,start+W+5)
            reads+=schedules(tuple(t for t in MATRIX if t!=tag),25,start+25,start+H)
            reads+=schedules((tag,),1,start+1,start+dense)
            finals=tuple(dict.fromkeys(MATRIX+(tag,)))
            jobs.append(spec('rpe',row,tag,H,reads,RPE_TAGS if tag=='ce' else (),finals))
    return jobs


def distinct_reads(jobs,starts):
    """Schedule accounting independent of tensors; includes only additional quick reads."""
    out={}
    for job in jobs:
        start=starts[job.job_id]
        own={(t,job.rule.tag,False) for t in range(start+5,start+job.horizon+1,5)}
        scheduled=set()
        for s in job.reads:
            scheduled.update((t,s.rule.tag,s.full) for t in range(s.start,s.stop+1) if s.active(t,start))
        out[job.job_id]=scheduled-own
    return out
