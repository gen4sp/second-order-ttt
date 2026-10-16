"""Merged recovery/prevention driver over the verified pool state and job APIs."""
from __future__ import annotations
import csv,dataclasses,json,tomllib
from pathlib import Path

import numpy as np
import torch
from . import exp_chr as chr,exp_ttt as ttt,exp_frs as frs,floor_rules as floors,frs_state as state
from .frs_registry import verify_seal
from .rdo_manifest import build_jobs as manifest_jobs,cohorts

ROOT=Path(__file__).resolve().parents[2]


class FloorBackend:
    drop_r4_probe=True

    def __init__(self):
        self.pre_cache={}
        self.references={}

    def begin_job(self,job,payload,ctx,model):
        ctx._route_pre=True;ctx._route_step=payload['state_step']
        ctx._route_source=job.state_sha256
        ctx._route_end=payload['state_step']+job.horizon
        ctx.cs=dataclasses.replace(ctx.cs,cross_rules={},branch_ckpt_tags=['rpf03','rpf30','rpf100'],
                                  branch_ckpt_every=100,ckpt_steps=[])
        if ctx._route_source not in self.references:
            vectors=[]
            floors.write_log(model,ctx.probe,ctx.eta,ctx.st.g_clip,ctx.task.chunk,chr.CE,vector_out=vectors)
            self.references[ctx._route_source]=torch.cat(vectors)

    def loss(self,model,idx,y,meta,ctx,rule,stats,diag):
        ctx._route_pre=False;ctx._route_step+=1
        return floors.chr_arm_loss(model,ctx.arm,idx,y,ctx.eta,ctx.st.g_clip,ctx.task.chunk,stats,meta,rule,diag)

    def quick(self,model,ctx,rule,*,geometry_on=False):
        geometry_on=geometry_on and (ctx._route_pre or ctx._route_step%25==0 or
                                     ctx._route_step==getattr(ctx,'_route_end',-1))
        key=(ctx._route_source,rule.tag,'quick')
        if ctx._route_pre and key in self.pre_cache:return self.pre_cache[key]
        out=floors.quick_log(model,ctx.quick,ctx.eta,ctx.st.g_clip,ctx.task.chunk,rule,
                            (ctx.cs.chunk_early,ctx.cs.chunk_late),geometry_on)
        if ctx._route_pre:self.pre_cache[key]=out
        return out

    def write(self,model,ctx,rule):
        reference=self.references.get(ctx._route_source) if isinstance(rule,floors.FloorRule) else None
        out=floors.write_log(model,ctx.probe,ctx.eta,ctx.st.g_clip,ctx.task.chunk,rule,reference=reference)
        if isinstance(rule,floors.FloorRule):
            ce=floors.write_log(model,ctx.probe,ctx.eta,ctx.st.g_clip,ctx.task.chunk,chr.CE)
            out['M1']=out['kv_write_med']/max(ce['kv_write_med'],1e-30)
        out['lr_scale']=ttt.lr_scale(ctx._route_step,ttt.arm_warmup(ctx.arm,ctx.st),ctx.st.lr_total,ctx.st.lr_floor)
        return out

    def pre_write(self,model,ctx,rule):
        key=(ctx._route_source,rule.tag,'write')
        if key not in self.pre_cache:self.pre_cache[key]=self.write(model,ctx,rule)
        return self.pre_cache[key]

    def final(self,model,ctx,rule,rules):
        out={}
        # Exactly the requested rules; callers explicitly provide own and CE in final_rules.
        for r in {r.tag:r for r in rules}.values():
            key=(ctx._route_source,r.tag,'full')
            if ctx._route_pre and key in self.pre_cache: out[r.tag]=self.pre_cache[key];continue
            out[r.tag]=floors.quick_log(model,ctx.ev,ctx.eta,ctx.st.g_clip,ctx.task.chunk,r,
                                       (ctx.cs.chunk_early,ctx.cs.chunk_late),False)
            if ctx._route_pre:self.pre_cache[key]=out[r.tag]
        return out


def create_backend(config=None):return FloorBackend()


def build_jobs(states,cells,config):
    """Provider imported by the pool at seal; validates sources before freezing cohorts."""
    if config.get('cells') and any(r.get('verified') is None for r in states):
        route={**config,**config.get('rdo_rpe',{}),'pool_output_dir':config['output_dir']}
        states,cells=verified_inputs(dict(cells={c['cell_id']:c for c in cells if c.get('cell_id')},states=states),
                                     route,config,ttt._device(config['device']))
    config['_rdo_cohorts']=cohorts(states,cells,{**config,**config.get('rdo_rpe',{})})
    return manifest_jobs(states,cells,config)


def guard(cfg,pool_cfg):
    seeds=cfg.get('candidate_seeds',list(range(200,216)))
    if cfg.get('smoke_only'):
        if any(not 0<=int(s)<=9 for s in seeds) or int(cfg.get('horizon',400))>40:
            raise ValueError('pre-seal route smoke requires legacy seeds and horizon <=40')
        if not pool_cfg.get('smoke_only') or int(pool_cfg.get('steps',1500))>80:
            raise ValueError('route smoke requires a tiny legacy pool config')
    elif cfg.get('require_seal',False):
        seal=verify_seal(cfg.get('seal_file','frs_seal.json'))
        modules=seal['job_modules']
        if 'tttmem.exp_rdo' not in modules:
            raise ValueError('route manifest provider is not sealed')
        if list(seeds)!=list(range(200,216)) or cfg.get('horizon')!=400 or cfg.get('late_horizon')!=300:
            raise ValueError('production route protocol differs from registrations')


def verified_inputs(pool,cfg,pool_cfg,dev):
    cells=list(pool['cells'].values());byid={c['cell_id']:c for c in cells}
    root=Path(cfg['pool_output_dir'])
    manifest=root/'states_manifest.csv'
    relocated={}
    if manifest.exists():
        with manifest.open(newline='') as stream:
            relocated={(r['cell_id'],r['trigger'],int(r['state_step'])):r
                       for r in state.resolve_manifest(list(csv.DictReader(stream)),root)}
    rows=[]
    for original in pool['states']:
        if original['arm']!='e2e1emb' or original['trigger'] not in ('S1','t_acq'):continue
        r=dict(original);file=Path(r['file'])
        if not file.exists():
            if not file.is_absolute() and not file.is_relative_to(root):file=root/file
            if not file.exists():
                saved=relocated.get((r['cell_id'],r['trigger'],int(r['state_step'])))
                if saved is not None and saved['sha256']==r['sha256']:file=Path(saved['file'])
        r['file']=str(file)
        try:
            c=byid[r['cell_id']]
            cell={k:c[k] for k in ('task','seed','arm','eta_base')}
            ctx=frs.make_context(pool_cfg,cell,dev)
            payload=state.verify_state(file,ctx,sha256=r['sha256'],
                                      expected_hashes={k:r[k] for k in state.HASH_KEYS})
            # Require the sealed pool source/config/rule hashes, not just a self-consistent file.
            if c.get('hashes') and payload['hashes']!=c['hashes']:raise ValueError('source state differs from pool provenance')
            current=state.context_hashes(ctx,pool_cfg)
            if any(payload['hashes'][k]!=current[k] for k in ('config','rules')):
                raise ValueError('source pool config/rule identity differs from continuation context')
            if r['trigger']=='S1':
                tail=payload['detector_tail'];at=int(r['state_step'])
                previous={int(x['step']):x for x in tail}
                checks=[previous.get(t,{}).get('quick',{}).get('acc') for t in (at-5,at)]
                r['ce_bound_at_S1']=any(v is None or v>=.90 for v in checks) and not cfg.get('smoke_only')
            r['verified']=True
        except (ValueError,OSError,KeyError) as exc:
            r['verified']=False;r['validation_error']=repr(exc)
        rows.append(r)
    return rows,cells


def reanalyze_pool(pool_root, output):
    """Read-only reuse of frozen jobs/cohorts; never construct or execute jobs."""
    pool_root=Path(pool_root)
    pool=json.loads((pool_root/'pool.json').read_text())
    combined=json.loads((pool_root/'job_manifest.json').read_text())
    specs=[s for s in combined['jobs'] if s['metadata'].get('pack') in ('rdo','rpe','late')]
    if not combined.get('route_cohorts'):
        raise ValueError('pool manifest lacks frozen RDO/RPE cohorts')
    cfg=pool['provenance']['effective_config']
    route={**cfg,**cfg.get('rdo_rpe',{})}
    provenance=dict(smoke_only=pool['smoke_only'],effective_config=route,pool_provenance=pool['provenance'])
    manifest=dict(schema=1,provenance=provenance,cohorts=combined['route_cohorts'],jobs=specs,
                  contract_patch=dict(rpe_enrolment='outcome-free',ledger_worker_seconds=31592.6))
    results={s['job_id']:pool['jobs'].get(s['job_id'],dict(status='MISSING',disposition='TRUNCATED')) for s in specs}
    if {k for k in pool['jobs'] if k.startswith(('rdo-','rpe-','late-'))}-{s['job_id'] for s in specs}:
        raise ValueError('pool route result set differs from frozen manifest')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        base=output
        n=1
        while output.exists():
            output=base.with_name(base.name+f'-attempt-{n:02d}');n+=1
    output.mkdir(parents=True,exist_ok=True)
    evidence=dict(schema=1,manifest=manifest,jobs=results)
    state.atomic_json(output/'results.json',chr._finite(evidence))
    from .rdo_evidence import analyze
    rdo,rpe=analyze(evidence,output/'analysis',smoke=pool['smoke_only'])
    return dict(complete=all(r.get('status')=='DONE' for r in results.values()),jobs=len(results),
                reanalysis_only=True,smoke_only=pool['smoke_only'],output_dir=str(output),
                rdo_headline=rdo['headline'],rpe_verdicts={k:v['verdict'] for k,v in rpe['doses'].items()})


def run(cfg,seed):
    ttt._mark_process(cfg['process_name']);torch.set_num_threads(2)
    return reanalyze_pool(cfg['pool_output_dir'],cfg['output_dir'])
