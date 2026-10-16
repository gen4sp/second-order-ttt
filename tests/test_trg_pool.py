"""Legacy seed-zero parity of the route-native and global queue paths."""
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_chr as c, exp_frs as f, exp_ttt as t, exp_trg as g
from tttmem import frs_state as s, trg_pool as p, trg_triggers as tr, triggers
from tttmem.frs_jobs import run_queue


@pytest.fixture(autouse=True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # job continuations default to ./checkpoints


def fixture(tmp_path, arm, horizon):
    cfg=dict(smoke_only=True,protocol='frs-a-smoke-v2',d_model=16,n_layers=2,
        n_heads=2,backbone_mult=2,fast_mult=2,window=8,chunk=16,T=96,n_keys=4,
        n_eval=2,n_quick_eval=2,eval_batch=2,n_probe=2,batch=2,steps=65,
        lr_total=3000,eta_lr=0.,workers=1,device='cpu',tf32=False,log_every=5,
        probe_every=10000,cross_every=10000,chunk_early=2,chunk_late=3,
        fork_arms=[],cross_rules=[],final_rules=[],ckpt_steps=[],branch_ckpt_tags=[],
        mediation_p=0.,trg_kappa=3,trg_smoke_horizon=horizon,trg_smoke_so_stop=5+horizon,
        output_dir=str(tmp_path/'out'))
    cell=dict(task='echo',seed=0,arm=arm,eta_base=40.)
    cfg['cells']=[cell.copy()]
    ctx=f.make_context(cfg,cell,torch.device('cpu'))
    model=t.build_model(ctx.task,ctx.st,ctx.dev,0,arm);opt=t.build_optim(model,ctx.st)
    rng=np.random.default_rng([0,t.TRAIN_SEED])
    f.frs_segment(model,opt,rng,ctx,c.CE,'prepare',0,stop=5)
    row=s.save_state(tmp_path/'fork.pt',model,opt,rng,ctx,trigger='t_acq',event_step=5,
        detected_step=5,state_step=5,nonfinite=0,hashes=s.context_hashes(ctx,cfg),detector_tail=[])
    cell.update(cell_id=ctx.ident)
    jobs=[replace(j,save_at=(5+horizon,)) for j in p.build_jobs([row],[cell],cfg)]
    return cfg,ctx,row,jobs


def native(ctx,row,branch,horizon,schedule=()):
    model=t.build_model(ctx.task,ctx.st,ctx.dev,0,ctx.arm);opt=t.build_optim(model,ctx.st)
    rng=np.random.default_rng();s.load_state(row['file'],model,opt,rng,ctx,sha256=row['sha256'])
    controller=tr.SpikeController(5,3);mask_rng=np.random.default_rng([0,4242,5])
    rows=[]
    for step in range(6,6+horizon):
        rows.append(g.train_step(model,opt,ctx.task,ctx.st,ctx.arm,0,step,rng,eta=40.,
            outer={'kvm':'kv','rnd':'rnd'}.get(branch,'none'),mask_rng=mask_rng,
            controller=controller,policy=branch if branch in ('skip','sham') else 'none',
            schedule=schedule,telemetry=True))
    return model.state_dict(),opt.state_dict(),rng.bit_generator.state,rows


@pytest.mark.parametrize('arm,branch',[(a,b) for a in ('e2e1emb','e2e1emb_cn97') for b in ('kvm','rnd')]
                         + [('e2e2emb','skip'),('e2e2emb','sham')])
def test_queue_native_bitwise_20(tmp_path,arm,branch):
    cfg,ctx,row,jobs=fixture(tmp_path,arm,20)
    selected=[j for j in jobs if j.metadata['branch']==branch or (branch=='sham' and j.metadata['branch']=='skip')]
    results=run_queue(selected,lambda j:ctx,tmp_path/'jobs',backend=p.TrgBackend())
    record=next(results[j.job_id] for j in selected if j.metadata['branch']==branch)
    assert record['status']=='DONE',record
    schedule=()
    if branch=='sham':
        skip=next(results[j.job_id] for j in selected if j.metadata['branch']=='skip')
        schedule=tr.yoked_sham(skip['drops'],skip['drop_sha256'])
    model,opt,rng,rows=native(ctx,row,branch,20,schedule)
    payload=torch.load(record['states'][0]['file'],weights_only=False)
    assert c._deep_equal(model,payload['model'])
    assert c._deep_equal(opt,payload['opt'])
    assert c._deep_equal(rng,payload['rng'])
    telemetry=Path(record['logger_final']['telemetry_file'])
    queue_rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
    assert [r['flags'] for r in queue_rows]==[r['flags'] for r in rows]
    assert record['drops']==[r['step'] for r in rows if r['dropped']]


def test_online_flags_and_drops_native_40(tmp_path):
    cfg,ctx,row,jobs=fixture(tmp_path,'e2e2emb',40)
    job=next(j for j in jobs if j.metadata['branch']=='skip')
    result=run_queue([job],lambda j:ctx,tmp_path/'jobs',backend=p.TrgBackend())[job.job_id]
    assert result['status']=='DONE',result
    _,_,_,rows=native(ctx,row,'skip',40)
    saved=[json.loads(line) for line in Path(result['logger_final']['telemetry_file']).read_text().splitlines()]
    assert [(r['flags'],r['active'],r['baseline'],r['dropped']) for r in saved]==[
           (r['flags'],r['active'],r['baseline'],r['dropped']) for r in rows]
    assert result['drops']==[r['step'] for r in rows if r['dropped']]


def test_dropped_norms_stay_in_history_and_shared_onset():
    skip=p.SpikeSkip(5,3,'skip');native=tr.SpikeController(5,3)
    for step,norm in enumerate([1.]*10+[20.,30.,float('nan')]+[1.]*27,6):
        expected=native.observe(step,norm);row=dict(grad_norm=norm)
        assert skip(step,row)==expected['flag']
        assert row['flags']==expected['flags']
    assert list(skip.controller.history)==list(native.history)
    steps=list(range(5,46));acc=[.95]*5+[.89]*36
    hit=triggers.detect_l1([dict(step=s,acc=a) for s,a in zip(steps,acc)],5)
    assert tr.dense_onset(steps,acc,5)==hit.event_step
    window=tr.WindowController(5,'FO')
    for step,a in zip(steps,acc):window.observe(step,a)
    assert window.onset==hit.event_step


@pytest.mark.parametrize('horizon', [20, 40])
@pytest.mark.parametrize('branch', ['skip', 'sham'])
def test_forced_flags_skip_sham_native_bitwise(tmp_path, monkeypatch, horizon, branch):
    original = tr.SpikeController.observe
    forced = {16, 18, 19, 30, 31}
    def observe(self, step, norm):
        result = original(self, step, norm)
        result.update(flag=step in forced)
        return result
    monkeypatch.setattr(tr.SpikeController, 'observe', observe)
    cfg, ctx, row, jobs = fixture(tmp_path, 'e2e2emb', horizon)
    selected = [j for j in jobs if j.metadata['branch'] == branch or branch == 'sham']
    results = run_queue(selected, lambda j: ctx, tmp_path/'jobs', backend=p.TrgBackend())
    record = next(results[j.job_id] for j in selected if j.metadata['branch'] == branch)
    assert record['status'] == 'DONE', record
    schedule = ()
    if branch == 'sham':
        skip = next(results[j.job_id] for j in selected if j.metadata['branch'] == 'skip')
        schedule = tr.yoked_sham(skip['drops'], skip['drop_sha256'])
        assert record['equal_realised_count']
    model, opt, rng, rows = native(ctx, row, branch, horizon, schedule)
    payload = torch.load(record['states'][0]['file'], weights_only=False)
    assert record['drops'] and record['drops'] == [r['step'] for r in rows if r['dropped']]
    assert c._deep_equal(model, payload['model'])
    assert c._deep_equal(opt, payload['opt'])  # includes AdamW steps and held LR
    assert c._deep_equal(rng, payload['rng'])


@pytest.mark.parametrize('steps,acc', [
    (list(range(5,46)), [.95]*5+[.89]*36),
    (list(range(5,46)), [.95]*41),
    (list(range(5,46)), [.95]*5+[.89]*9+[.95]*27),
])
def test_native_dense_onset_equals_pool_detector(steps, acc):
    hit = triggers.detect_l1([dict(step=s, acc=a) for s,a in zip(steps, acc)], 5)
    assert tr.dense_onset(steps, acc, 5) == (None if hit is None else hit.event_step)
