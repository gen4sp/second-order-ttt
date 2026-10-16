"""Sealed TRG adapter for the single pool continuation queue."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from . import dense_log as dense, exp_chr as chr, exp_ttt as ttt, frs_state as state
from .frs_backend import ChrBackend
from .frs_jobs import JobSpec
from .trg_jobs import make_job_specs, ARM_FAMILIES
from .trg_triggers import SpikeController
from .pool_dense import PoolDenseLogger, nonfinite_fields


class PoolLogger:
    def __init__(self, ctx, output_dir, config):
        self.ctx,self.config=ctx,config
        self.folder=Path(output_dir)/'trg'
        self.writer=PoolDenseLogger(self.folder,identity=dict(cell_id=ctx.ident,arm=ctx.arm,seed=ctx.seed),
                                      emit=bool(config.get('emit_dense',False)))
        self.stream=(self.folder/'telemetry.jsonl').open('w')
        self.tel=None;self.pre_step=None;self.controller=None
        self.costs={'light':[],'ladder':[]}
        self.cached=None
        self.diagnostics=[]

    def loss_observer(self, step, model, ctx):
        self.current_step=step
        if self.tel is None:self.tel=dense.Telemetry(model)
        self.tel.chunks={}
        return self.tel.observer

    def acquisition(self, step):
        self.controller=SpikeController(step,int(self.config['trg_kappa']))

    def pre_clip(self, step, norms, model, opt, ctx):
        self.gradient_groups=self.tel.before()
        self.pre_step=step

    def telemetry(self, step, data):
        model,opt,ctx=data['model'],data['opt'],data['ctx']
        if self.tel is None:self.tel=dense.Telemetry(model)
        if self.pre_step!=step:self.gradient_groups=self.tel.before()
        chunks=self.tel.chunks
        row={k:v for k,v in data.items() if k not in ('model','opt','ctx','idx','y','meta')}
        if 'flags' not in row:
            row.update(self.controller.observe(step,row['grad_norm'] if row['grad_norm'] is not None else float('nan'))
                       if self.controller else dict(flags={},flag=False,active=False,baseline=None))
        keep=data.get('meta',{}).get('_trg_keep')
        masked=(1-keep).sum(1) if keep is not None else None
        removed=sum(v['ce']-v['outer_ce'] for v in chunks.values())/(ctx.task.T/ctx.task.chunk) if keep is not None else 0.
        row.update(step=step,finite=not row['nonfinite'],dropped=row['skipped'],chunks=chunks,
                   gradient_groups=self.gradient_groups,update=self.tel.after(opt),
                   removed_ce_mass=removed,removed_ce_sum=removed*ctx.st.batch*ctx.task.T,
                   removed_count=int(masked.sum()) if masked is not None else 0,
                   masked_per_sequence=masked.tolist() if masked is not None else [],
                   outer_denominator=ctx.st.batch*ctx.task.T,lr=[g['lr'] for g in opt.param_groups],
                   batch_sha1=getattr(ctx,'_frs_batch_sha',None))
        fields = nonfinite_fields(row)
        if fields:
            row.update(telemetry_nonfinite=True, nonfinite_fields=fields)
        self.stream.write(json.dumps(chr._finite(row),allow_nan=False)+'\n');self.stream.flush()

    def _read(self, model, ctx, step, ladder):
        if self.cached is not None and self.cached[0]==step:
            return self.cached[1]
        ttt._sync(ctx.dev);begin=time.perf_counter()
        read=(dense.ladder_log if ladder else dense.light_read)(model,ctx.quick,ctx.eta,ctx.st.g_clip,ctx.task.chunk)
        ttt._sync(ctx.dev)
        self.costs['ladder' if ladder else 'light'].append(time.perf_counter()-begin)
        summary = self.writer.append(step,read)
        # Scalars for the pool window; event payloads are owned by the writer once.
        result=dict(summary,shard_manifest=str(self.folder/'shards.json'))
        self.cached=(step,result)
        return result

    def light(self, model, ctx, step):return self._read(model,ctx,step,False)
    def ladder(self, model, ctx, step):return self._read(model,ctx,step,True)

    def finalize(self):
        self.stream.close()
        manifest=self.writer.close()
        return dict(shard_manifest=str(self.folder/'shards.json'),shard_manifest_sha256=state.file_sha256(self.folder/'shards.json'),
                    telemetry_file=str(self.folder/'telemetry.jsonl'),telemetry_sha256=state.file_sha256(self.folder/'telemetry.jsonl'),
                    gradient_diagnostics=self.diagnostics,
                    read_costs={k:dict(calls=len(v),wall_s=sum(v),median_s=float(np.median(v)) if v else None,samples_s=v)
                                for k,v in self.costs.items()},schema_sha256=manifest['schema_sha256'])


def create_logger(ctx, output_dir, config):return PoolLogger(ctx,output_dir,config)


@dataclass
class OuterMask:
    seed:int
    fork_step:int
    outer:str
    def __post_init__(self):self.rng=np.random.default_rng([self.seed,4242,self.fork_step])
    def __call__(self,idx,y,meta,*,step,ctx):
        keep=ttt.outer_keep(meta,tuple(y.shape),self.outer,self.rng,idx.device)
        if keep is not None and bool(((1-keep).sum(1)!=ctx.task.n_keys).any()):
            raise ValueError('TRG mask miscount')
        return idx,y,dict(meta,_trg_keep=keep)
    def state_dict(self):return self.rng.bit_generator.state
    def load_state_dict(self,value):self.rng.bit_generator.state=value


class SpikeSkip:
    def __init__(self,start,kappa,policy):
        self.controller=SpikeController(start,kappa);self.policy=policy
    def __call__(self,step,telemetry):
        norm=telemetry['grad_norm']
        flags=self.controller.observe(step,float('nan') if norm is None else norm)
        telemetry.update(flags)
        return self.policy=='skip' and flags['flag']
    def state_dict(self):
        return dict(history=list(self.controller.history),finite_count=self.controller.finite_count)
    def load_state_dict(self,value):
        self.controller.history.clear();self.controller.history.extend(value['history'])
        self.controller.finite_count=value['finite_count']


class TrgBackend(ChrBackend):
    def begin_job(self,job,payload,ctx,model):
        # H5/mains retain CHR cross reads; intervention jobs have only CE quick reads.
        if job.metadata.get('pack')=='TRG':
            from dataclasses import replace
            ctx.cs=replace(ctx.cs,cross_rules={},branch_ckpt_tags=[],ckpt_steps=[])
            self.drop_r4_probe=True
            ctx._trg_job=True
            ctx._trg_start=job.metadata['start']
    def quick(self,model,ctx,rule,*,geometry_on=True):
        logger=getattr(ctx,'_trg_logger',None)
        if getattr(ctx,'_trg_job',False) and logger is not None:
            return logger.light(model,ctx,getattr(logger,'current_step',ctx._trg_start))
        return super().quick(model,ctx,rule,geometry_on=geometry_on)
    def write(self,model,ctx,rule):
        if getattr(ctx,'_trg_job',False):return {}
        return super().write(model,ctx,rule)
    def final(self,model,ctx,rule,rules):
        if getattr(ctx,'_trg_job',False):
            # TRG stage (ii) has dense quick outcomes, no registered full-set matrix.
            return {rule.tag:dict(self.quick(model,ctx,rule),set='quick')}
        return super().final(model,ctx,rule,rules)
    def loss(self,model,idx,y,meta,ctx,rule,stats,diag,*,observer=None):
        if meta.get('_trg_keep') is not None:
            return ttt.arm_loss(model,ctx.arm,idx,y,ctx.eta,ctx.st.g_clip,ctx.task.chunk,stats,
                                keep=meta['_trg_keep'],meta=meta,observer=observer)
        return super().loss(model,idx,y,meta,ctx,rule,stats,diag,observer=observer)


def create_backend(config=None):return TrgBackend()


class JobHooks:
    def __init__(self,ctx,folder,job):
        self.logger=PoolLogger(ctx,folder,ctx._frs_cfg)
        ctx._trg_logger=self.logger
        self.job=job
        self.failed={}
    def loss_observer(self,step,model,ctx):return self.logger.loss_observer(step,model,ctx)
    def pre_clip(self,step,norms,model,opt,ctx):self.logger.pre_clip(step,norms,model,opt,ctx)
    def after_update(self,step,data,model,opt,rng,ctx):
        self.logger.telemetry(step,dict(data,model=model,opt=opt,ctx=ctx,**ctx._frs_update_inputs))
    def at_start(self,step,model,opt,rng,ctx):self.logger.light(model,ctx,step)
    def after_step(self,step,model,opt,rng,ctx,cache):
        self.logger.light(model,ctx,step)
        if step%25==0 and self.job.metadata['branch'] in ('kvm','rnd'):
            from .exp_trg import gradient_diagnostic
            self.logger.diagnostics.append(dict(step=step,records=gradient_diagnostic(model,
                ctx.probe,ctx.arm,ctx.eta,ctx.st,self.job.outer_mask.outer,ctx.seed,
                self.job.metadata['start'],step)))
    def finalize(self):return self.logger.finalize()


def create_job_hooks(ctx,folder,job):return JobHooks(ctx,folder,job)


def build_jobs(states,cells,config):
    smoke=bool(config.get('smoke_only'))
    kappa=config.get('trg_kappa')
    digest=config.get('trg_addendum_sha256')
    if smoke:digest='0'*64
    if not smoke and config.get('require_seal',False):
        from .trg_registration import read_addendum
        read_addendum(config)
    jobs=[];dispositions=[]
    by_cell={c['cell_id']:c for c in cells}
    acquired={r['cell_id']:r for r in states if r['trigger']=='t_acq'}
    for cid,cell in sorted(by_cell.items()):
        row=acquired.get(cid)
        if row is None:
            dispositions.append(dict(cell_id=cid,seed=cell['seed'],arm=cell['arm'],
                family=ARM_FAMILIES[cell['arm']],
                disposition='missing-main' if cell.get('status') == 'MISSING' or cell.get('error') else 'no-acquisition'))
            continue
        native=make_job_specs(row,dict(kappa=kappa,addendum_sha256=digest))
        dispositions.append({k:v for k,v in native.items() if k!='jobs'})
        for spec in native['jobs']:
            start=int(row['state_step']);end=spec['end']
            if smoke:end=min(config['steps'],config.get('trg_smoke_so_stop',40) if spec['family']=='SO'
                             else start+config.get('trg_smoke_horizon',20))
            if end-start<5:continue
            jobs.append(JobSpec(spec['job_id'],row['file'],row['sha256'],chr.CE,end-start,
                log_every=1,
                after=tuple(spec['after']),sham_from=spec['after'][0] if spec['branch']=='sham' else None,
                cell_id=cid,final_rules=(chr.CE,),expected_hashes={k:row[k] for k in state.HASH_KEYS},
                outer_mask=OuterMask(cell['seed'],start,spec.get('outer','none')),
                skip_controller=SpikeSkip(start,kappa,spec['branch']),hooks_factory=create_job_hooks,
                metadata=dict(pack='TRG',seed=cell['seed'],arm=cell['arm'],family=spec['family'],
                              branch=spec['branch'],start=start,end=end,backend_factory='tttmem.trg_pool:create_backend')))
    config['_trg_dispositions']=dispositions
    return jobs


def analyze_pool(pool,manifest,output,*,smoke=False,pool_root=None):
    from .analyze_trg import score_ht3, score_ht4, unbound_burden
    from .episodes import nl2_window,classify_branch
    specs=[s for s in manifest['jobs'] if s['metadata'].get('pack')=='TRG']
    grouped={}
    for spec in specs:
        meta=spec['metadata'];record=pool['jobs'].get(spec['job_id'],{})
        start=meta['start'];end=start+spec['horizon']
        root=Path(pool_root or pool['output_dir'])/'jobs'
        evidence=record.get('logger_final') or {}
        folders=[root/spec['job_id']/'trg',root/'retry'/spec['job_id']/'trg']
        folder=next((d for d in folders if (d/'shards.json').is_file() and
                     state.file_sha256(d/'shards.json')==evidence.get('shard_manifest_sha256')),None)
        dense_rows=[];evidence_error=None
        disposition='TRUNCATED'
        if record.get('status')=='DONE':
            try:
                if folder is None:raise ValueError('missing logger manifest')
                dense.validate_shards(folder)
                if state.file_sha256(folder/'telemetry.jsonl')!=evidence['telemetry_sha256']:
                    raise ValueError('telemetry hash mismatch')
                dense_rows=[json.loads(line) for line in (folder/'summaries.jsonl').read_text().splitlines()]
                disposition=classify_branch([r['step'] for r in dense_rows],
                    [r['acc'] for r in dense_rows],start=start,end=end,grid=1,
                    t_div=record.get('t_div'),certified=record.get('divergence_certified',False),
                    continuation_steps=record.get('attempted'),nonfinite=record.get('nonfinite_local'))
            except (OSError,ValueError,KeyError) as exc:
                evidence_error=str(exc)
        rows=[r for r in record.get('logs',[]) if r['step']>start and (r['step']-start)%5==0]
        steps=[r['step'] for r in rows];acc=[r['quick']['acc'] for r in rows]
        if smoke:
            # Tiny smoke checks the contract without giving the 300-step endpoint a verdict.
            endpoint=dict(nl2=None,disposition=disposition)
        else:
            endpoint=nl2_window(steps,acc,start,disposition=disposition,t_div=record.get('t_div'))
        endpoint['evidence_error']=evidence_error
        unit=grouped.setdefault((meta['family'],meta['seed']),dict(family=meta['family'],seed=meta['seed']))
        unit[meta['branch']]=endpoint
        if meta['family']=='SO':
            unit[meta['branch']]=unbound_burden([r['step'] for r in dense_rows],[r['acc'] for r in dense_rows],start,
                end=end,disposition=disposition,t_div=record.get('t_div'))
            unit[meta['branch']]['evidence_error']=evidence_error
            unit.update(length=max(0,end-start-10))
            if meta['branch']=='skip':unit['drops']=len(record.get('drops',[]))
    ht3=score_ht3([u for (family,_),u in grouped.items() if family!='SO'])
    ht4=score_ht4([u for (family,_),u in grouped.items() if family=='SO'])
    result=dict(smoke_only=smoke,H_T3=ht3,H_T4=ht4,units=list(grouped.values()),
                dispositions=manifest.get('trg_dispositions',[]),status='SMOKE ONLY — NOT SCORED' if smoke else 'SCORED')
    state.atomic_json(Path(output)/'trg_scores.json',result)
    print('TRG_POOL_ANALYSIS '+json.dumps(dict(smoke_only=smoke,H_T3={k:v['label'] for k,v in ht3.items()},H_T4=ht4['label'])),flush=True)
    return result
