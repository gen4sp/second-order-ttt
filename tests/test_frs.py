"""FRS CPU fixtures. Every training cell uses legacy seed 0."""
import copy
from dataclasses import replace
import gzip
import importlib.util
import json
from pathlib import Path
import random
import os
import tomllib

import numpy as np
import pytest
import torch

from tttmem import exp_chr as c
from tttmem import exp_ttt as t
from tttmem import exp_frs as f
from tttmem import frs_state as s
from tttmem import frs_registry as reg
from tttmem import frs_cohorts as co
from tttmem import triggers as tr
from tttmem import episodes as ep
from tttmem.frs_jobs import JobSpec, ReadSchedule, run_job, run_queue, write_drop_list, yoked_schedule, build_manifest
from tttmem.frs_backend import ChrBackend

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    previous_dtype=torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    monkeypatch.chdir(tmp_path)
    cfg = dict(smoke_only=True, protocol="frs-a-smoke-v2", process_name=os.environ.get('PROCESS_NAME','TTM-pool-74'),
               d_model=16, n_layers=2, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=16, n_keys=4, T=96, n_eval=2, n_quick_eval=2,
               eval_batch=2, n_probe=2, batch=2, steps=30, lr_total=3000, eta_lr=0.,
               workers=1, device="cpu", tf32=False, log_every=5, probe_every=25,
               cross_every=25, chunk_early=2, chunk_late=3, fork_arms=[],
               rp_cap=1000., cross_rules=["rp03"], final_rules=["rp03"],
               ckpt_steps=[], branch_ckpt_every=100, branch_ckpt_tags=[], mediation_p=0.,
               output_dir=str(tmp_path / "out"), cells=[dict(task="echo",seed=0,arm="e2e1emb",eta_base=40.)])
    try:
        yield cfg
    finally:
        torch.set_default_dtype(previous_dtype)


def setup(cfg, arm="e2e1emb"):
    ctx=f.make_context(cfg,dict(task="echo",seed=0,arm=arm,eta_base=40.),torch.device("cpu"))
    model=t.build_model(ctx.task,ctx.st,ctx.dev,0,arm)
    return ctx,model,t.build_optim(model,ctx.st),np.random.default_rng([0,t.TRAIN_SEED])


@pytest.mark.parametrize("arm", ["e2e1emb", "e2e2emb", "e2e1emb_cn97"])
def test_hooks_off_equals_chr_bitwise_30(tiny, arm):
    ctx,model,opt,rng=setup(tiny,arm)
    c.train_segment(model,opt,rng,ctx,c.CE,"ref",0,allow_fork=False)
    ref,refopt,refrng=copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(rng.bit_generator.state)
    ctx,mine,o2,r2=setup(tiny,arm)
    f.frs_segment(mine,o2,r2,ctx,c.CE,"new",0)
    assert equal(ref,mine.state_dict()) and equal(refopt,o2.state_dict()) and equal(refrng,r2.bit_generator.state)


@pytest.mark.parametrize("arm", ["e2e1emb", "e2e2emb", "e2e1emb_cn97"])
def test_hooks_on_do_not_change_parameters(tiny, arm, tmp_path):
    ctx,ref,opt,rng=setup(tiny,arm)
    f.frs_segment(ref,opt,rng,ctx,c.CE,"ref",0,output_dir=tmp_path/"ref")
    reference=copy.deepcopy(ref.state_dict())
    cfg={**tiny,"trigger_bind":0.,"trigger_residual":2.,"smoke_force_loss":True,
         "coverage_span":20,"so_coverage_stop":30}
    ctx,model,opt,rng=setup(cfg,arm)
    hooks=f.PoolHooks(ctx,cfg,tmp_path/"hooks",s.context_hashes(ctx,cfg))
    f.frs_segment(model,opt,rng,ctx,c.CE,"main",0,hooks=hooks,output_dir=tmp_path/"hooks")
    assert equal(reference,model.state_dict())
    assert hooks.states and hooks.coverage[0]["step"] == hooks.events["t_acq"]["event_step"]
    if arm == "e2e1emb":
        row=next(r for r in hooks.states if r["trigger"]=="S1")
        assert row["event_step"] == row["detected_step"] == row["state_step"] == 25
        p=s.verify_state(row["file"],ctx,sha256=row["sha256"])
        assert len(p["detector_tail"]) == 5 and p["completed_updates"] == 25


def test_complete_pool_cell_persists_hook_states(tiny,tmp_path):
    cfg={**tiny,"trigger_bind":0.,"trigger_residual":2.,"smoke_force_loss":True,
         "coverage_span":20}
    record=f.run_frs_cell(cfg,cfg["cells"][0],torch.device("cpu"))
    assert record["steps_done"]==30 and len(record["states"])==2
    assert {r["trigger"] for r in record["states"]}=={"t_acq","S1"}
    record_path=Path(cfg["output_dir"])/"mains"/record["cell_id"].replace("/","__")/"cell.json"
    assert json.loads(record_path.read_text())["states"]==record["states"]


def test_s2_saved_state_dependency_and_resume(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    c.train_segment(m,opt,rng,ctx,c.CE,"memory",5,allow_fork=False)
    first=JobSpec("first",row["file"],row["sha256"],c.CE,10,save_at=(15,),cell_id=ctx.ident)
    second=JobSpec("second","job:first:15","",c.CE,15,after=("first",),cell_id=ctx.ident)
    result=run_queue([second,first],lambda j:ctx,tmp_path/"jobs")
    assert all(r["status"]=="DONE" for r in result.values())
    assert equal(m.state_dict(),torch.load(result["second"]["final_checkpoint"],weights_only=True))
    replay=run_queue([second,first],lambda j:ctx,tmp_path/"jobs",resume=True)
    assert replay["first"]["final_sha256"]==result["first"]["final_sha256"]
    assert replay["second"]["final_sha256"]==result["second"]["final_sha256"]
    import shutil
    shutil.move(tmp_path/'jobs',tmp_path/'moved-jobs')
    moved=run_queue([second,first],lambda j:ctx,tmp_path/'moved-jobs',resume=True)
    assert moved['first']['status']==moved['second']['status']=='DONE'
    assert moved['second']['final_sha256']==result['second']['final_sha256']
    assert str(tmp_path/'moved-jobs') in moved['first']['states'][0]['file']


def test_combined_rdo_pool_freezes_cohorts_and_reanalyzes_after_move(tiny,tmp_path,monkeypatch):
    import shutil
    from tttmem import exp_rdo as rdo
    cfg={**tiny,'steps':65,'max_minutes':15,'trigger_bind':0.,'trigger_residual':2.,
         'smoke_force_loss':True,'coverage_span':10,'h5_limit':1,'job_modules':['tttmem.exp_rdo'],
         'rdo_rpe':dict(candidate_seeds=[0],limit=1,horizon=20,late_horizon=20,late=True)}
    summary=f.run(cfg,0)
    assert summary['complete'] and summary['jobs']==20
    root=Path(cfg['output_dir'])
    pool=json.loads((root/'pool.json').read_text())
    routes={k:v for k,v in pool['jobs'].items() if k.startswith(('rdo-','rpe-','late-'))}
    assert len(routes)==18 and all(r['status']=='DONE' for r in routes.values())
    moved=tmp_path/'downloaded';shutil.move(root,moved)
    assert not hasattr(rdo, 'execute')
    reused=rdo.reanalyze_pool(moved,tmp_path/'reanalysis')
    assert reused['complete'] and reused['jobs']==18
    repeated=rdo.reanalyze_pool(moved,tmp_path/'reanalysis')
    assert repeated['output_dir']!=reused['output_dir']


def test_rdo_failed_eligible_seed_is_not_refilled():
    from tttmem.rdo_manifest import cohorts
    states=[dict(seed=1,arm=co.FO,trigger=tag,state_step=25,sha256='a'*64,cell_id='later')
            for tag in ('S1','t_acq')]
    cells=[dict(seed=0,arm=co.FO,cell_id='failed',error='lost'),dict(seed=1,arm=co.FO,cell_id='later')]
    result=cohorts(states,cells,dict(candidate_seeds=[0,1],limit=1,smoke_only=True))
    assert result['S1'][0]['seed']==result['t_acq'][0]['seed']==0
    assert result['S1'][0]['missing'] and all(r['seed']==0 for r in result['t_acq'])


def test_global_dispatch_spans_cells_and_waits_for_dependencies(tiny,monkeypatch,tmp_path):
    import queue,time
    cells=[dict(task='echo',seed=0,arm=arm,eta_base=40.) for arm in (co.FO,co.SO,co.NOISE)]
    ids=[t._cell_id(cell,65) for cell in cells]
    jobs=[JobSpec(f'job{i}','unused','x',c.CE,h,cell_id=ids[i%3])
          for i,h in enumerate((100,400,200,300,150,350))]
    jobs.append(JobSpec('dependent','unused','x',c.CE,50,cell_id=ids[1],after=('job1',)))
    events=[]
    class Context:
        def Queue(self):
            q=queue.Queue();q.close=lambda:None;q.cancel_join_thread=lambda:None
            return q
        def Pipe(self, **kw):
            q=queue.Queue()
            class Receiver:
                def poll(self):return not q.empty()
                def recv(self):return q.get_nowait()
                def close(self):pass
            class Sender:
                def send(self,item):q.put(item)
                def close(self):pass
            return Receiver(),Sender()
        class Process:
            def __init__(self,**kw):
                self.alive=False
                index,cfg,mode,inbox,results,lock=kw['args']
                def put(item):
                    if item is None:return
                    key,payload,attempt=item
                    events.append(('dispatch',key,sorted(payload['prior'])))
                    results.put((index,key,dict(status='DONE',job_id=key)))
                inbox.put=put
            def start(self):self.alive=True
            def is_alive(self):return self.alive
            def join(self,**kw):self.alive=False
            def terminate(self):self.alive=False
    monkeypatch.setattr(f.mp,'get_context',lambda _:Context())
    from tttmem import pool_dispatch
    monkeypatch.setattr(pool_dispatch,'wait',lambda *a,**k:[])
    cfg={**tiny,'cells':cells,'steps':65,'workers':5,'_deadline':time.time()+60}
    def progress(results):events.append(('complete',tuple(results)))
    result=f._global_jobs(cfg,jobs,progress)
    assert len(result)==7 and all(r['status']=='DONE' for r in result.values())
    dispatch=[e for e in events if e[0]=='dispatch']
    assert [e[1] for e in dispatch[:5]]==['job1','job5','job3','job2','job4']
    dependent=next(i for i,e in enumerate(events) if e[:2]==('dispatch','dependent'))
    assert any(e[0]=='complete' and 'job1' in e[1] for e in events[:dependent])
    assert dispatch[-1][2]==['job1']


def test_infrastructure_error_one_retry_same_state(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    class Once(ChrBackend):
        def __init__(self): self.calls=0
        def loss(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:raise OSError("simulated worker I/O error")
            return super().loss(*args,**kwargs)
    job=JobSpec("retry",row["file"],row["sha256"],c.CE,25)
    result=run_queue([job],lambda j:ctx,tmp_path/"jobs",backend=Once())["retry"]
    assert result["status"]=="DONE" and result["attempts"]==2
    assert result["first_batch_sha1"]==row["next_batch_sha1"]
    assert s.file_sha256(row["file"])==row["sha256"]
    resumed=run_queue([job],lambda j:ctx,tmp_path/"jobs",backend=Once(),resume=True)['retry']
    assert resumed['status']=='DONE' and resumed['attempts']==2
    assert resumed['final_sha256']==result['final_sha256']


def test_preclip_skip_history_and_adam_state(tiny, tmp_path):
    ctx, model, opt, rng = setup(tiny)
    seen, snapshots = [], {}
    class Hooks:
        def after_update(self, step, tel, *args):
            if step in snapshots:
                assert equal(snapshots[step], opt.state_dict())
    def controller(step, tel):
        assert all(p.grad is not None for p in t.clipped_params(model))
        norm=float(f.preclip_norm(t.clipped_params(model)))
        assert tel['grad_norm']==norm
        seen.append(norm)
        drop=step in (2, 3, 8)
        if drop:snapshots[step]=copy.deepcopy(opt.state_dict())
        return drop
    result=f.frs_segment(model,opt,rng,ctx,c.CE,'skip',0,skip_controller=controller,
                         hooks=Hooks(),output_dir=tmp_path/'skip')
    assert result['drops']==[2,3,8] and len(seen)==30
    assert all(float(v['step'])==27 for v in opt.state_dict()['state'].values())


def test_failed_main_preserves_state_slot_and_resume(tiny,tmp_path,monkeypatch):
    cfg={**tiny,'trigger_bind':0.,'resume':True}
    original=f.frs_segment
    def interrupted(*args,**kwargs):
        original(*args,**{**kwargs,'stop':10})
        raise RuntimeError('interrupted after acquisition')
    monkeypatch.setattr(f,'frs_segment',interrupted)
    rec=f.run_frs_cell(cfg,cfg['cells'][0],torch.device('cpu'))
    assert rec['seed']==0 and rec['status']=='MISSING' and rec['states']
    assert co.first_acquired([rec,dict(seed=1,arm=co.FO,events={'t_acq':{'event_step':5}})],1,smoke=True)==[rec]
    monkeypatch.setattr(f,'frs_segment',original)
    resumed=f.run_frs_cell(cfg,cfg['cells'][0],torch.device('cpu'))
    assert resumed['status']=='DONE' and resumed['steps_done']==30
    assert list((Path(cfg['output_dir'])/'mains').glob('*.partial-1'))


def test_failed_observer_keeps_main_bitwise(tiny,tmp_path,monkeypatch):
    cfg={**tiny,'trigger_bind':0.,'coverage_span':20}
    ctx,m,opt,rng=setup(cfg)
    original=copy.deepcopy(m.state_dict())
    f.frs_segment(m,opt,rng,ctx,c.CE,'baseline',0,output_dir=tmp_path/'ref')
    reference=copy.deepcopy(m.state_dict())
    class Broken(f.SmokeLogger):
        def telemetry(self,*args):raise RuntimeError('telemetry fault')
        def ladder(self,*args):raise RuntimeError('ladder fault')
    monkeypatch.setattr(f,'SmokeLogger',Broken)
    ctx,m,opt,rng=setup(cfg)
    hooks=f.PoolHooks(ctx,cfg,tmp_path/'broken',s.context_hashes(ctx,cfg))
    f.frs_segment(m,opt,rng,ctx,c.CE,'main',0,hooks=hooks,output_dir=tmp_path/'broken')
    assert equal(reference,m.state_dict()) and hooks.window.fail_open
    assert set(hooks.logger_errors)>={'telemetry','ladder'}


def test_non_infrastructure_main_failure_is_terminal_on_resume(tiny,monkeypatch):
    def bad(*args,**kwargs):raise ValueError('malformed route adapter')
    monkeypatch.setattr(f,'frs_segment',bad)
    cfg={**tiny,'resume':True}
    first=f.run_frs_cell(cfg,cfg['cells'][0],torch.device('cpu'))
    assert first['status']=='MISSING' and not first['retryable']
    monkeypatch.setattr(f,'frs_segment',lambda *a,**kw:pytest.fail('terminal seed must not rerun'))
    resumed=f.run_frs_cell(cfg,cfg['cells'][0],torch.device('cpu'))
    assert resumed['error']==first['error'] and not resumed['retryable']


def test_h5_terminal_acquisition_has_disposition():
    dispositions=[]
    cells=[dict(seed=0,arm=co.FO,events={'t_acq':{'event_step':30}})]
    rows=[dict(seed=0,arm=co.FO,trigger='t_acq',state_step=30)]
    assert build_manifest(rows,cells,horizon=30,config={'smoke_only':True},dispositions=dispositions)==[]
    assert dispositions[0]['disposition']=='no continuation'


def test_job_route_observer_hook(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    seen=[]
    class Observer:
        def after_update(self,step,*args):seen.append(step)
    job=JobSpec("observed",row["file"],row["sha256"],c.CE,25,
                hooks_factory=lambda ctx,folder,job:Observer())
    result=run_job(job,ctx,tmp_path/"jobs")
    assert result["status"]=="DONE" and seen==list(range(6,31))


@pytest.mark.parametrize("confirmed",[True,False])
def test_insurance_onset_global_50_preserves_weights_if_rejected(tiny,tmp_path,confirmed):
    cfg={**tiny,"steps":80,"ckpt_steps":[50],"coverage_span":0}
    ctx,m,opt,rng=setup(cfg)
    hooks=f.PoolHooks(ctx,cfg,tmp_path/"insurance",s.context_hashes(ctx,cfg))
    hooks.nonfinite=0
    for step in range(5,51,5):
        row=dict(step=step,quick=dict(acc=.8 if step==50 else .95),write=dict(r_kv_med=.5))
        hooks.after_log(row,m,opt,rng,ctx)
    hooks.after_step(50,m,opt,rng,ctx,{})
    assert hooks.candidate and hooks.has_state(50)
    hooks.after_log(dict(step=55,quick=dict(acc=.8 if confirmed else .95),write=dict(r_kv_med=.5)),
                    m,opt,rng,ctx)
    assert hooks.candidate is None
    if confirmed:
        insurance=next(r for r in hooks.states if r["trigger"]=="insurance")
        assert insurance["state_step"]==50 and hooks.events["t_loss"]["detected_step"]==55
    else:
        assert not any(r["trigger"]=="insurance" for r in hooks.states)
        assert equal(m.state_dict(),torch.load(hooks.pending_weights[0],weights_only=True))


def test_runner_divergence_counter_is_continuation_local(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    class Nonfinite(ChrBackend):
        def __init__(self):self.calls=0
        def loss(self,model,*args,**kwargs):
            self.calls+=1
            if self.calls<=2:return model.head.W2.sum()*float("nan")
            return super().loss(model,*args,**kwargs)
    result=f.frs_segment(m,opt,rng,ctx,c.CE,"divergence",5,nonfinite0=1,
                         backend=Nonfinite(),output_dir=tmp_path/"div")
    assert result["t_div"]==25 and result["attempted"]==20 and result["nonfinite_local"]==2
    assert result["nonfinite_cumulative"]==3 and result["divergence_certified"]


def start_file(cfg, tmp_path, *, at=5, rule=c.CE, outer_mask=None, skip_controller=None):
    ctx,m,opt,rng=setup(cfg)
    f.frs_segment(m,opt,rng,ctx,rule,"start",0,stop=at,outer_mask=outer_mask,
                  skip_controller=skip_controller,output_dir=tmp_path/"prefix")
    hashes=s.context_hashes(ctx,cfg)
    extra={}
    for name,value in (("outer_mask",outer_mask),("skip_controller",skip_controller)):
        if hasattr(value,"state_dict"):
            extra[name]=value.state_dict()
    row=s.save_state(tmp_path/"start.pt",m,opt,rng,ctx,trigger="t_acq",event_step=at,
                     detected_step=at,state_step=at,nonfinite=0,hashes=hashes,detector_tail=[],extra=extra)
    return ctx,m,opt,rng,row


@pytest.mark.parametrize("tag", ["ce", "rp03", "nc03"])
def test_job_file_equals_chr_in_memory_20(tiny,tmp_path,tag):
    cfg={**tiny,"steps":25}
    ctx,m,opt,rng,row=start_file(cfg,tmp_path)
    rule=c.parse_rule(tag,1000.)
    c.train_segment(m,opt,rng,ctx,rule,"memory",5,allow_fork=False)
    expected=copy.deepcopy(m.state_dict())
    job=JobSpec("file-fork",row["file"],row["sha256"],rule,20,expected_hashes=s.context_hashes(ctx,cfg))
    result=run_job(job,ctx,tmp_path/"jobs")
    got=torch.load(result["final_checkpoint"],weights_only=True)
    assert equal(expected,got) and result["next_batch_verified"]
    assert result["steps_done"]==25 and result["status"]=="DONE"


def test_all_22_chr_triggers_and_verified_s1():
    with gzip.open(ROOT/"tests/fixtures/frs_chr_mains.json.gz","rt") as stream:
        fixture=json.load(stream)
    assert len(fixture["rows"])==22
    ac=co.chr_analysis()
    for cid,rows in fixture["rows"].items():
        found=tr.detect(rows)
        old=ac.events(ac.series(rows))
        assert (found["t_acq"].event_step if "t_acq" in found else None)==old["t_acq"]
        assert (found["t_loss"].event_step if "t_loss" in found else None)==old["t_loss"]
        if "e2e1emb/" in cid or "|e2e1emb|" in cid or ac.arm_of(cid)=="e2e1emb":
            seed=ac.seed_of(cid)
            expect=fixture["s1_expected"].get(str(seed))
            assert (found["S1"].event_step if "S1" in found else None)==expect
        for event in found.values():
            prefix=[r for r in rows if r["step"]<=event.detected_step]
            assert tr.detect(prefix)[event.name]==event
            if event.name != "t_acq":
                assert event.name not in tr.detect(prefix[:-1])


def test_all_rng_adamw_and_next_batch_roundtrip(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    saved=s.capture_rngs()
    m2=t.build_model(ctx.task,ctx.st,ctx.dev,0,ctx.arm)
    opt2=t.build_optim(m2,ctx.st);rng2=np.random.default_rng(7)
    p=s.load_state(row["file"],m2,opt2,rng2,ctx,sha256=row["sha256"],expected_hashes=s.context_hashes(ctx,tiny))
    assert equal(saved,s.capture_rngs())
    assert equal(m.state_dict(),m2.state_dict()) and equal(opt.state_dict(),opt2.state_dict())
    assert equal(rng.bit_generator.state,rng2.bit_generator.state)
    assert p["copy_noise"]["identity"]==[0,4406,5]
    assert c.rng_peek(rng2.bit_generator.state,ctx.task,ctx.st.batch)==p["next_batch_sha1"]
    assert all(float(v["step"])==5 for v in p["opt"]["state"].values())
    f.frs_segment(m,opt,rng,ctx,c.CE,"ref",5,output_dir=tmp_path/"r1")
    f.frs_segment(m2,opt2,rng2,ctx,c.CE,"loaded",5,output_dir=tmp_path/"r2")
    assert equal(m.state_dict(),m2.state_dict()) and equal(opt.state_dict(),opt2.state_dict())


def test_state_hash_schema_fixed_sets_and_immutability(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    with pytest.raises(ValueError,match="sha256"):
        s.load_state(row["file"],m,opt,rng,ctx,sha256="0"*64)
    with pytest.raises(ValueError,match="source"):
        s.verify_state(row["file"],ctx,sha256=row["sha256"],expected_hashes={"source":"wrong"})
    with pytest.raises(FileExistsError):
        s.save_state(row["file"],m,opt,rng,ctx,trigger="S1",event_step=5,detected_step=5,state_step=5,
                     nonfinite=0,hashes=s.context_hashes(ctx,tiny),detector_tail=[])
    payload=torch.load(row["file"],weights_only=False)
    for field, value, pattern in [("schema",2,"schema"),("next_batch_sha1","bad","next-batch"),
                                  ("completed_updates",4,"count")]:
        bad=copy.deepcopy(payload);bad[field]=value;path=tmp_path/f"bad-{field}.pt";torch.save(bad,path)
        with pytest.raises(ValueError,match=pattern):
            s.verify_state(path,ctx,sha256=s.file_sha256(path))


class Mask:
    def __init__(self, at=0):
        self.rng=np.random.default_rng([0,4242,at])
    def __call__(self,idx,y,meta,*,step,ctx):
        mask=torch.as_tensor(self.rng.random(y.shape)<.2,device=y.device)
        y=torch.where(mask,(y+1)%ctx.task.V,y)
        return idx,y,meta
    def state_dict(self):
        return copy.deepcopy(self.rng.bit_generator.state)
    def load_state_dict(self,state):
        self.rng.bit_generator.state=copy.deepcopy(state)


def test_masked_continuation_roundtrip(tiny,tmp_path):
    mask=Mask()
    ctx,m,opt,rng,row=start_file(tiny,tmp_path,outer_mask=mask)
    saved_mask=Mask();saved_mask.load_state_dict(mask.state_dict())
    f.frs_segment(m,opt,rng,ctx,c.CE,"memory-mask",5,outer_mask=mask,output_dir=tmp_path/"mem")
    job=JobSpec("masked",row["file"],row["sha256"],c.CE,25,outer_mask=saved_mask)
    result=run_job(job,ctx,tmp_path/"jobs")
    assert equal(m.state_dict(),torch.load(result["final_checkpoint"],weights_only=True))


def test_skip_and_yoked_sham_dependencies(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    skip=JobSpec("skip",row["file"],row["sha256"],c.CE,25,
                 skip_controller=lambda step,tel: step in {8,9,15},cell_id=ctx.ident)
    sham=JobSpec("sham",row["file"],row["sha256"],c.CE,25,after=("skip",),sham_from="skip",cell_id=ctx.ident)
    result=run_queue([sham,skip],lambda job:ctx,tmp_path/"jobs")
    assert result["skip"]["drops"]==[8,9,15] and result["sham"]["drops"]==[11,12,18]
    assert result["skip"]["status"]==result["sham"]["status"]=="DONE"
    assert len(result["skip"]["drops"])==len(result["sham"]["drops"])
    assert s.file_sha256(row["file"])==row["sha256"]
    with pytest.raises(ValueError,match="sha256"):
        yoked_schedule(result["skip"]["drop_file"],"0"*64,start=5,stop=30)
    direct=replace(skip,job_id="direct-skip")
    again=run_job(direct,ctx,tmp_path/"again")
    assert equal(torch.load(again["final_checkpoint"],weights_only=True),
                 torch.load(result["skip"]["final_checkpoint"],weights_only=True))


def test_failed_skip_means_sham_missing_not_run(tiny,tmp_path):
    ctx,m,opt,rng,row=start_file(tiny,tmp_path)
    skip=JobSpec("skip",row["file"],"0"*64,c.CE,25)
    sham=JobSpec("sham",row["file"],row["sha256"],c.CE,25,after=("skip",),sham_from="skip")
    result=run_queue([sham,skip],lambda job:ctx,tmp_path/"jobs")
    assert result["skip"]["status"]==result["sham"]["status"]=="MISSING"
    assert result["sham"]["attempts"]==0 and not (tmp_path/"jobs/sham").exists()


def test_dependency_cycle_rejected(tiny,tmp_path):
    ctx,_,_,_,row=start_file(tiny,tmp_path)
    a=JobSpec("a",row["file"],row["sha256"],c.CE,25,after=("b",))
    b=replace(a,job_id="b",after=("a",))
    with pytest.raises(ValueError,match="cycle"):
        run_queue([a,b],lambda job:ctx,tmp_path/"jobs")


def test_yoked_overflow_and_first_free_step(tmp_path):
    path,sha=write_drop_list(tmp_path/"drops.json",[5,6,7])
    assert yoked_schedule(path,sha,start=0,stop=20)==[8,9,10]
    with pytest.raises(ValueError,match="overflow"):
        yoked_schedule(path,sha,start=0,stop=9)


def test_registry_stream_typing_all_fresh_without_training():
    for seed in range(200,240):
        reg.guard_seed(seed)
    for stream in ("init","frozen_code","data","copy_noise","chr_directions"):
        for identity in reg.streams(0)[stream]:
            with pytest.raises(ValueError,match="historical"):
                reg.guard(stream,identity)
    with pytest.raises(ValueError,match="historical"):
        reg.guard("rnd_mask",(0,4242))
    reg.guard("rnd_mask",(0,4242,5))  # fork-local arity is a different stream identity
    reg.guard("init",(7919,))  # never compare unrelated purposes
    with pytest.raises(ValueError,match="no stream"):
        reg.guard("b9_sham",(0,))
    assert reg.registry()["reserved_r5"]==[216,227] and reg.registry()["free"]==[228,239]


def test_preseal_fresh_rejected_before_model_build(tiny,tmp_path):
    cfg=copy.deepcopy(tiny);cfg["cells"][0]["seed"]=200;cfg["require_seal"]=True
    with pytest.raises(ValueError,match="legacy"):
        f.run(cfg,0)
    with pytest.raises(ValueError,match="legacy"):
        f.make_context(tiny,dict(task="echo",seed=200,arm="e2e1emb",eta_base=40.),torch.device("cpu"))
    cfg["smoke_only"]=False;cfg["seal_file"]=str(tmp_path/"absent.json");cfg["require_seal"]=True
    with pytest.raises(FileNotFoundError):
        reg.execution_guard(cfg)
    with pytest.raises(ValueError,match="roles"):
        file=tmp_path/"seal.json";file.write_text(json.dumps(dict(schema=1,phase="sealed",files=[])))
        reg.verify_seal(file)


def test_coverage_inclusive_tail_so_and_fail_open():
    window=tr.CoverageWindow("e2e1emb",5)
    for step in range(5,16):
        window.observe(dict(step=step,acc=1. if step==5 else .8))
    assert window.t_l1.event_step==6 and window.stop==66
    so=tr.CoverageWindow("e2e2emb",5)
    for step in range(5,16):
        so.observe(dict(step=step,acc=.8))
    assert so.stop==660 and so.active(660) and not so.active(661)
    broken=tr.CoverageWindow("e2e1emb",5)
    broken.observe(dict(step=5,acc=1.));broken.observe(dict(step=7,acc=1.))
    assert broken.fail_open and broken.stop==305
    assert tr.insurance_steps(400)==list(range(400,1001,50))
    assert tr.insurance_steps(405)[0]==450


def synth_cell(seed, retained=False, acquired=True):
    return dict(seed=seed,arm="e2e1emb",logs=[dict(step=5,quick=dict(acc=.95 if acquired else .1))],
                final={"ce":{"acc":.95 if retained else .1},"rp03":{"acc":.1}})


@pytest.mark.parametrize("case", ["retained-main","no-twin","n>6","divergence","missing-nc","missing-cross","nine-pairs"])
def test_h5_scoring_adapter_registered_fixtures(case):
    n=9 if case=="nine-pairs" else 10
    cells=[synth_cell(seed,retained=case=="retained-main" and seed==0) for seed in range(n)]
    jobs=[dict(seed=seed,rule=tag,final={tag:{"acc":.95 if tag=="rp03" else .1},"rp03":{"acc":.95 if tag=="rp03" else .1}})
          for seed in range(n) for tag in ("rp03","nc03")]
    if case=="divergence":
        jobs[0]["diverged"]=True
    if case=="missing-nc":
        jobs=[j for j in jobs if not (j["seed"]==0 and j["rule"]=="nc03")]
    if case=="missing-cross":
        cells[0]["final"].pop("rp03")
    row=co.h5_adapter(cells,jobs)
    assert row["n"]==n and len(row["cohort"])==n
    if case=="retained-main":
        assert row["comparator"]==1 and row["units"][0]["seed"]==0
    if case=="divergence":
        assert row["test"]==9 and row["n"]==10
    if case=="missing-nc":
        assert row["nc_qualifier"]["missing"]==[0] and row["selectivity"]=="unknown"
    if case=="missing-cross":
        assert row["selectivity"]=="unknown"
    if case=="nine-pairs":
        assert row["p"]==1 and co.paired_verdict(row,1)=="NOT EVALUABLE"


def test_holm_fixed_five_and_exact_pairs():
    row=co.exact_mcnemar([True]*12,[False]*12)
    assert row["p"]==2**-12
    adj=co.holm5({"R-SO":row["p"],"R-NOISE":.04})
    assert adj["R-SO"]==5*2**-12 and adj["R-NOISE"]==.16
    assert adj["R-LIN"]==adj["R-RP"]==adj["R-ADDR"]==1
    assert co.paired_verdict(row,adj["R-SO"])=="REPLICATED"


def series(start=0,horizon=400,acc=1.,every=5):
    return [dict(step=i,acc=acc) for i in range(start+every,start+horizon+1,every)]


@pytest.mark.parametrize("common",[False,True])
def test_nl2_boundary_300_confirmed_305(common):
    rows=series()
    for r in rows:
        if r["step"] in (300,305):r["acc"]=.8
    extra=((rows[:61],list(range(5,306,5))),) if common else ()
    out=ep.endpoints(rows,0,full_acc=1.,needed=extra)
    assert out["nl2"] is False and out["t_loss"]==300


def test_dense_boundary_300_through_309_and_gap():
    dense=series(horizon=309,every=1)
    for r in dense:
        if r["step"]>=300:r["acc"]=.8
    assert ep.nl_fine(dense,0) is False
    assert ep.nl_fine(dense[:302]+dense[303:],0) is None


def test_recovery_boundary_395_confirmed_400_and_truncation():
    rows=series(acc=.8)
    rows[-2]["acc"]=rows[-1]["acc"]=.95
    out=ep.endpoints(rows,0,full_acc=.95)
    assert out["t_re"]==395 and out["tau"]==395 and out["retained"]
    assert ep.endpoints(rows[:-1],0,full_acc=.95)["disposition"]=="TRUNCATED"


@pytest.mark.parametrize("div_at",[20,100,310])
def test_divergence_endpoints_and_adverse_control(div_at):
    rows=series(horizon=div_at)
    cert=ep.Divergence(div_at,div_at,int(div_at*.1))
    out=ep.endpoints(rows,0,certificate=cert,full_acc=1.)
    assert out["disposition"]=="DIVERGED" and not out["retained"] and not out["censored"]
    assert out["nl2"] == (div_at>300)
    assert out["lost_time"]==400-div_at
    adverse=ep.adverse_completion(out,treatment=False)
    assert adverse["nl2"] and adverse["lost_time"]==0


def test_divergence_keeps_previously_confirmed_recovery():
    rows=series(horizon=100,acc=.8)
    for r in rows:
        if r["step"]>=30:r["acc"]=.95
    out=ep.endpoints(rows,0,certificate=ep.Divergence(100,100,10),full_acc=1.)
    assert out["t_re"]==30 and out["tau"]==30 and not out["retained"]


def test_uncertified_nan_is_missing_and_common_label_unknown():
    rows=series(horizon=20,acc=.8);rows[-2]["acc"]=rows[-1]["acc"]=np.nan
    out=ep.endpoints(rows,0,horizon=20,window=10,full_acc=.1)
    assert out["disposition"]=="TRUNCATED" and out["nl2"] is None
    treated=ep.adverse_completion(out,treatment=True)
    control=ep.adverse_completion(out,treatment=False)
    assert not treated["nl2"] and control["nl2"]
    assert ep.common_label(("a","b"),lambda x:x)=="unknown"


def test_episode_missing_common_read_truncates():
    rows=series()
    assert ep.endpoints(rows,0,full_acc=1.,needed=((rows[:-1],list(range(5,401,5))),))["disposition"]=="TRUNCATED"


def test_empty_transient_denominators_not_evaluable():
    result=co.transient_adapter([])
    assert result["verdict"]=="NOT EVALUABLE"
    assert result["fo_no_recovery"]["n"]==result["ordinal"]["n"]==0
    assert all(r["verdict"]=="NOT EVALUABLE" for r in result["recovery"].values())


def test_prod_config_and_registration_pins():
    cfg=tomllib.loads((ROOT/"configs/toy/pool.toml").read_text())
    assert cfg["experiment"]=="frs"
    config=cfg["frs"]
    count={arm:[c["seed"] for c in config["cells"] if c["arm"]==arm] for arm in c.CHR_ARMS}
    assert count=={"e2e1emb":list(range(200,216)),"e2e2emb":list(range(200,212)),"e2e1emb_cn97":list(range(200,212))}
    assert config["steps"]==1500 and config["eta_lr"]==0 and config["lr_total"]==3000
    assert not config["tf32"] and config["log_every"]==5 and config["cross_every"]==25
    assert config["fork_arms"]==[] and config["workers"]==5 and config["process_name"]=="TTM-pool-70"
    assert config["ckpt_steps"]==tomllib.loads((ROOT/"configs/toy/chr.toml").read_text())["chr"]["ckpt_steps"]
