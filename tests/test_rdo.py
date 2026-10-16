import copy,csv,json,os,tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from tttmem import exp_ttt as t,exp_frs as frs,exp_chr as chr,frs_state as state
from tttmem import floor_rules as f,exp_rdo as driver,rdo_manifest as manifest
from tttmem.frs_jobs import JobSpec,ReadSchedule,run_job,job_record
from tttmem.rdo_evidence import branch_events

torch.set_num_threads(2)
t._mark_process(os.environ.get('PROCESS_NAME','TTM-rdo-72'))
ROOT=Path(__file__).resolve().parents[1]


def config():
    cfg=tomllib.loads((ROOT/'configs/toy/smoke/rdo-pool.toml').read_text())['frs']
    cfg.update(n_eval=2,n_quick_eval=2,n_probe=2,mediation_p=0.,ckpt_steps=[],ckpt_steps_sparse=[],
               branch_ckpt_tags=[],cells=[dict(task='echo',arm='e2e1emb',seed=0,eta_base=40)])
    return cfg


def setup_state(folder,start=5,nonfinite=0):
    cfg=config();ctx=frs.make_context(cfg,cfg['cells'][0],torch.device('cpu'))
    model=t.build_model(ctx.task,ctx.st,ctx.dev,0,ctx.arm);opt=t.build_optim(model,ctx.st)
    rng=np.random.default_rng([0,t.TRAIN_SEED])
    frs.frs_segment(model,opt,rng,ctx,chr.CE,'main',0,stop=start,output_dir=folder/'prep')
    row=state.save_state(folder/'start.pt',model,opt,rng,ctx,trigger='S1',event_step=start,
                         detected_step=start,state_step=start,nonfinite=nonfinite,
                         hashes=state.context_hashes(ctx,cfg),detector_tail=[])
    return cfg,ctx,model,opt,rng,row


def test_pre_reads_and_file_roundtrip_equal_in_memory_bitwise(tmp_path):
    cfg,ctx,model,opt,rng,row=setup_state(tmp_path)
    own=f.parse_rule('rpf30');backend=driver.FloorBackend()
    spec=JobSpec('rdo-s0-rpf30',row['file'],row['sha256'],own,20,
                 reads=(ReadSchedule(own,1,6,25),),final_rules=(chr.CE,own),save_at=(25,),
                 metadata=dict(pack='rdo',seed=0,start=5))
    pre=f.quick_log(model,ctx.quick,40.,10.,16,own,(2,3),False)
    result=run_job(spec,ctx,tmp_path/'jobs',backend=backend)
    assert result['pre']['rpf30']['quick']==pre
    assert result['next_batch_verified'] and result['first_batch_sha1']==row['next_batch_sha1']
    payload=state.verify_state(row['file'],ctx,sha256=row['sha256'])
    state.restore_rngs(payload['rngs'])
    baseline=driver.FloorBackend();baseline.begin_job(spec,payload,ctx,model)
    frs.frs_segment(model,opt,rng,ctx,own,'ref',5,stop=25,backend=baseline,output_dir=tmp_path/'ref')
    loaded=torch.load(result['final_checkpoint'],weights_only=False)
    assert chr._state_equal(loaded,state._cpu(model.state_dict()))
    actual=state.verify_state(result['states'][0]['file'],ctx,sha256=result['states'][0]['sha256'])
    assert chr._deep_equal(actual['opt'],state._cpu(opt.state_dict()))
    assert actual['rng']==rng.bit_generator.state
    assert branch_events(result,job_record(spec),smoke=True)['rpf30']['disposition']=='COMPLETE'


def test_moved_pool_paths_and_config_identity(tmp_path):
    cfg,ctx,model,opt,rng,row=setup_state(tmp_path)
    moved=tmp_path/'download';moved.mkdir()
    Path(row['file']).rename(moved/'start.pt')
    with (moved/'states_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=state.MANIFEST_FIELDS);writer.writeheader()
        writer.writerow({k:('start.pt' if k=='file' else row[k]) for k in state.MANIFEST_FIELDS})
    cell={**cfg['cells'][0],'cell_id':ctx.ident,'hashes':state.context_hashes(ctx,cfg)}
    pool=dict(cells={ctx.ident:cell},states=[row])
    route=dict(pool_output_dir=str(moved),smoke_only=True)
    verified,_=driver.verified_inputs(pool,route,cfg,ctx.dev)
    assert verified[0]['verified'] and Path(verified[0]['file'])==moved/'start.pt'
    changed={**cfg,'lr':cfg['lr']*2}
    verified,_=driver.verified_inputs(pool,route,changed,ctx.dev)
    assert not verified[0]['verified'] and 'config/rule identity' in verified[0]['validation_error']


def test_continuation_local_divergence_and_cumulative_counter(tmp_path):
    cfg,ctx,model,opt,rng,row=setup_state(tmp_path,start=20,nonfinite=4)
    class Backend:
        drop_r4_probe=True
        def loss(self,model,idx,y,meta,ctx,rule,stats,diag):return next(model.parameters()).sum()*float('nan')
        def quick(self,*args,**kw):return dict(acc=.95)
        def write(self,*args):return dict(r_kv_med=.1)
        def final(self,model,ctx,rule,rules):return {r.tag:dict(acc=.95) for r in (rule,*rules)}
    spec=JobSpec('rdo-s0-ce',row['file'],row['sha256'],chr.CE,30,final_rules=(chr.CE,))
    r=run_job(spec,ctx,tmp_path/'jobs',backend=Backend())
    assert r['t_div']==40 and r['nonfinite_local']==20 and r['nonfinite_cumulative']==24
    assert r['disposition']=='DIVERGED' and r['divergence_certified']
    class Finite(Backend):
        def loss(self,model,*args):return next(model.parameters()).sum()*0
    r=run_job(JobSpec('finite',row['file'],row['sha256'],chr.CE,20,final_rules=(chr.CE,)),
              ctx,tmp_path/'jobs',backend=Finite())
    assert r['nonfinite_local']==0 and r['nonfinite_cumulative']==4 and not r['diverged']


def rows():
    return [dict(seed=0,arm='e2e1emb',cell_id='unit',trigger=trigger,state_step=at,file='state.pt',sha256='abc')
            for trigger,at in (('t_acq',350),('S1',600))]


def test_single_source_manifest_and_distinct_ledger():
    jobs=manifest.build_jobs(rows(),[],dict(pool_output_dir='pool',candidate_seeds=[0]))
    assert len(jobs)==18
    blocks={b:[j for j in jobs if j.metadata['pack']==b] for b in ('rdo','late','rpe')}
    assert {b:len(js) for b,js in blocks.items()}==dict(rdo=9,late=3,rpe=6)
    extra=manifest.distinct_reads(jobs,{j.job_id:j.metadata['start'] for j in jobs})
    assert sum(len(extra[j.job_id]) for j in blocks['rdo'])==640+128+180
    assert sum(len(extra[j.job_id]) for j in blocks['late'])==120+24+60
    assert sum(len(extra[j.job_id]) for j in blocks['rpe'])==305+260+1488
    assert sum(len(j.final_rules) for j in blocks['rdo'])==49
    assert sum(len(j.final_rules) for j in blocks['late'])==15
    assert sum(len(j.final_rules) for j in blocks['rpe'])==26 # four own-rule duplicates in the ledger's 30
    assert set(j.rule.tag for j in blocks['rpe'])==set(manifest.RPE_TAGS)
    assert all(len(j.after)==1 for j in blocks['late'])
    assert all(j.metadata['start']==1000 for j in blocks['late'])


def test_outcome_free_enrolment_and_no_refill():
    rs=[dict(seed=s,arm='e2e1emb',cell_id=f'u{s}',trigger='t_acq',state_step=350,file='x',sha256='x') for s in range(10)]
    cells=[dict(seed=s,arm='e2e1emb',t_loss=None,branch_success=s%2) for s in range(10)]
    cfg=dict(candidate_seeds=list(range(10)),limit=6)
    c=manifest.cohorts(rs,cells,cfg)
    assert [r['seed'] for r in c['t_acq']]==list(range(6))
    cells.reverse()
    assert manifest.cohorts(rs,cells,cfg)['t_acq']==c['t_acq']
    assert sum(d['disposition']=='not needed' for d in c['dispositions'])==4


def test_phase_boundary_and_smoke_production_analysis_block(tmp_path):
    pool=config()
    with pytest.raises(ValueError):driver.guard(dict(smoke_only=True,candidate_seeds=[200],horizon=20),pool)
    with pytest.raises(ValueError):driver.guard(dict(smoke_only=True,candidate_seeds=[0],horizon=400),pool)
    with pytest.raises((ValueError,FileNotFoundError)):driver.guard(dict(candidate_seeds=[200],horizon=400,require_seal=True),pool)


def test_backend_precache_deduplicates_same_start(tmp_path):
    cfg,ctx,model,opt,rng,row=setup_state(tmp_path)
    spec=JobSpec('precache',row['file'],row['sha256'],f.parse_rule('rpf03'),20)
    backend=driver.FloorBackend();payload=state.verify_state(row['file'],ctx,sha256=row['sha256'])
    backend.begin_job(spec,payload,ctx,model)
    first=backend.quick(model,ctx,spec.rule)
    assert backend.quick(model,ctx,spec.rule) is first
    full=backend.final(model,ctx,spec.rule,[spec.rule])
    assert backend.final(model,ctx,spec.rule,[spec.rule])['rpf03'] is full['rpf03']
