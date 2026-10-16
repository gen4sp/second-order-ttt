"""R4-4b float64 fixtures, no CUDA/MPS and no spawned torch processes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import types

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t, exp_r44b as b, exp_p01 as p
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
t._mark_process(os.environ.get('PROCESS_NAME','TTM-r44b-90-tests'))


def tiny(emb=True):
    st=t._settings(dict(d_model=8,n_heads=2,n_layers=1,backbone_mult=2,fast_mult=2,window=8,
                        chunk=8,T=24,batch=2,n_keys=2,n_eval=2,n_quick_eval=2,eval_batch=2,
                        n_probe=2,probe_every=0,quick_every=1,lr_total=3000,steps=2,device='cpu',tf32=False))
    task=t._task(st,'icl_mc')
    model=t.build_model(task,st,torch.device('cpu'),0,'e2e1').double()
    if emb:
        model.frozen_emb=torch.randn(task.V,8,dtype=torch.float64)
    idx,y,m=task.tensors(*task.raw(np.random.default_rng(9),2),torch.device('cpu'))
    return st,task,model,idx,y


@pytest.mark.parametrize('emb',[False,True])
@pytest.mark.parametrize('clip',[10.,.001])
def test_reference_detachment_chunk0_lin_update_and_meta_paths(emb,clip):
    st,task,model,idx,y=tiny(emb)
    h=model.backbone(idx)
    W=model.per_sequence(2)
    ref=t.reference_logits(model,h,idx,W)
    assert not ref.requires_grad and ref.grad_fn is None
    z=model.read(model.fast_block(h[:,:8],idx[:,:8],W))
    losses=[t.inner_token_loss(rule,z,y[:,:8],reference=ref[:,:8]).mean(1).sum() for rule in ('lin','aice')]
    grads=[torch.autograd.grad(loss,W,retain_graph=True) for loss in losses]
    for a,c in zip(*grads): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-8)
    clipped=[t._clip_fast(list(g),clip) for g in grads]
    assert all(not g.requires_grad for g in clipped[0][0])
    for a,c in zip(clipped[0][0],clipped[1][0]): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-8)
    eta=model.etas(.3)
    updated=[[w-e*g for w,e,g in zip(W,eta,c[0])] for c in clipped]
    for a,c in zip(*updated): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-8)
    outer=sum(v.square().sum() for v in updated[1])
    identity=torch.autograd.grad(outer,model.slow(),retain_graph=True)
    rho=torch.autograd.grad(outer,model.rho)[0]
    assert rho.abs().sum()>0 and all(g.abs().sum()>0 for g in identity)
    z1=model.read(model.fast_block(h[:,8:16],idx[:,8:16],updated[1]))
    a=t.inner_token_loss('aice',z1,y[:,8:16],reference=ref[:,8:16]).mean(1).sum()
    l=t.lin_token(z1,y[:,8:16]).mean(1).sum()
    ga=torch.autograd.grad(a,updated[1],retain_graph=True)
    gl=torch.autograd.grad(l,updated[1])
    assert sum((x-y).square().sum() for x,y in zip(ga,gl))>1e-12


def test_so_hessian_and_fixed_reference_finite_difference():
    _,_,model,idx,y=tiny(False)
    h=model.backbone(idx).detach()[:,:8]; idx=idx[:,:8]; y=y[:,:8]
    W=[w.detach().clone().requires_grad_() for w in model.per_sequence(2)]
    ref=t.reference_logits(model,h,idx,W)
    direction=[torch.randn_like(w) for w in W]
    def gradient(weights,rule,graph=False):
        z=model.read(model.fast_block(h,idx,weights))
        return torch.autograd.grad(t.inner_token_loss(rule,z,y,reference=ref).mean(1).sum(),weights,create_graph=graph)
    g=gradient(W,'aice',True)
    hv=torch.autograd.grad(sum((x*d).sum() for x,d in zip(g,direction)),W)
    gl=gradient(W,'lin',True)
    hl=torch.autograd.grad(sum((x*d).sum() for x,d in zip(gl,direction)),W)
    assert sum((a-l).square().sum() for a,l in zip(hv,hl))>1e-8
    eps=1e-5
    shifted=[gradient([(w+scale*eps*d).detach().requires_grad_() for w,d in zip(W,direction)],'aice')
             for scale in (-2,-1,1,2)]
    # Fourth-order centered derivative controls truncation near LayerNorm.
    for exact,a,c,d,e in zip(hv,*shifted):
        torch.testing.assert_close(exact,(a-8*c+8*d-e)/(12*eps),atol=1e-10,rtol=1e-8)


@pytest.mark.parametrize('emb',[False,True])
def test_training_streaming_preupdate_logits_and_ordinary_outer_ce(emb,monkeypatch):
    _,task,model,idx,y=tiny(emb)
    h=model.backbone(idx); observed=[]
    real=t.token_ce
    def capture(z,target):
        if z.requires_grad: observed.append(z.detach().clone())
        return real(z,target)
    monkeypatch.setattr(t,'token_ce',capture)
    loss=t.sequential_inner(model,h,y,model.per_sequence(2),model.etas(.3),10.,False,t.GradStats(),8,idx,rule='aice')
    ordinary=torch.cat(observed[::2],dim=1)  # ordinary CE then reference-subtracted CE per chunk
    monkeypatch.setattr(t,'token_ce',real)
    stream=t.eval_sequential(model,idx,y,[e.detach() for e in model.etas(.3)],10.,8,t.GradStats(),rule='aice')
    torch.testing.assert_close(ordinary,stream,atol=1e-10,rtol=1e-8)
    torch.testing.assert_close(loss,real(stream,y).mean(),atol=1e-10,rtol=1e-8)


def test_arm_routing_unknown_rules_config_and_manifest():
    cfg=tomllib.loads((ROOT/'configs/toy/r4-4b.toml').read_text())['ttt']; w=b.validate_config(cfg)
    assert w.echo_seeds==list(range(10)) and w.icl_seeds==list(range(6))
    b.validate_config(tomllib.loads((ROOT/'configs/toy/smoke/r4-4b-gpu.toml').read_text())['ttt'])
    for key,val in [('enabled',False),('smoke',True),('oops',1),('echo_steps',60)]:
        bad=copy.deepcopy(cfg); bad['wave_b'][key]=val
        with pytest.raises(ValueError): b.validate_config(bad)
    with pytest.raises(ValueError): b.validate_config({'process_name':'TTM-r44b-02'})
    for arm,code,taskname in [('e2e1_aice',False,'icl_mc'),('e2e1emb_aice',True,'echo')]:
        st=t._settings(cfg,taskname); task=t._task(st,taskname)
        model=t.build_model(task,st,torch.device('cpu'),0,arm)
        assert (model.frozen_emb is not None)==code and t.arm_inner_loss(arm)=='aice'
        with pytest.raises(ValueError): t.check_arm_task(arm,'echo' if taskname=='icl_mc' else 'icl_mc')
    with pytest.raises(ValueError): t.arm_inner_loss('bogus')
    with pytest.raises(ValueError): t.inner_token_loss('bogus',None,None)
    with pytest.raises(ValueError): t.inner_token_loss('aice',None,None)
    manifest=json.loads((ROOT/'configs/toy/r4-4b/historical-manifest.json').read_text())
    assert len(manifest['checkpoints'])==16
    # The historical checkpoints are not distributed; hash whichever are present locally.
    for e in manifest['checkpoints']:
        if (ROOT/e['path']).exists(): assert b.sha_file(ROOT/e['path'])==e['sha256']


def test_rule_aware_probe_ablation_and_dual_write_diagnostics(tmp_path):
    st=t._settings(dict(d_model=8,n_heads=2,n_layers=1,backbone_mult=2,fast_mult=2,window=8,
                        chunk=8,T=64,n_keys=2,batch=2,n_eval=2,eval_batch=2,steps=2))
    task=t._task(st,'echo'); model=t.build_model(task,st,torch.device('cpu'),0,'e2e1emb_aice').double()
    raw,meta=task.raw(np.random.default_rng(7),2); data=t.eval_batches(task,raw,meta,torch.device('cpu'),2)
    captures=[]
    for arm in ('e2e1emb_lin','e2e1emb_aice'):
        capture={}; t.r4_probe(model,task,data,10.,10.,arm=arm,capture=capture); captures.append(capture)
        ab=t.r42_ablation(model,task,data,10.,10.,arm=arm)
        z=t.eval_batch_logits(model,task,arm,*data[0][:2],10.,10.,8,t.GradStats(),meta=meta)
        hit=(p.gather(z,meta['query_pos']).argmax(-1)==torch.as_tensor(meta['query_tok'])).double().mean()
        assert abs(ab['all']-float(hit))<1e-10
    for a,c in zip(captures[0]['inner_grads'],captures[1]['inner_grads']): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-8)
    conditions,_=p.reassign_set(2,512,64,4,16,64,20250308)
    assert conditions['reassign'][1]['write_pos'].shape==(2,32)
    assert conditions['reassign'][1]['query_pos'].shape==(2,16)
    assert np.all(conditions['reassign'][1]['old_tok']!=conditions['reassign'][1]['new_tok'])


def test_trigger_audit_cap_and_stage_clock_rho_freeze_snapshot_resume(tmp_path):
    trigger=b.Trigger()
    assert not trigger.observe(25,.99)
    assert not trigger.observe(50,.98)
    assert not trigger.observe(75,.999)
    assert trigger.observe(100,.99)
    st,task,model,idx,y=tiny(False)
    opt=t.build_optim(model,st); rng=np.random.default_rng([1,t.TRAIN_SEED]); rho=model.rho.detach().clone()
    raw,ok,_,_=b.update(model,opt,task,st,rng,'e2e1',.3,0,stage1=True)
    assert ok and torch.equal(rho,model.rho) and model.rho.grad is None and model.rho not in opt.state
    entry=b.save_snapshot(tmp_path/'continuation.pt',model,opt,rng,dict(completed_steps=1))
    before=b.state_hash(model.state_dict()); optim=b.state_hash(opt.state_dict())
    # Uninterrupted next update versus restored model and next data.
    raw1,ok,_,_=b.update(model,opt,task,st,rng,'e2e1',.3,0,stage2=True)
    resumed=t.build_model(task,st,torch.device('cpu'),0,'e2e1').double(); ropt=t.build_optim(resumed,st); rrng=np.random.default_rng()
    assert b.load_snapshot(entry,resumed,ropt,rrng)==dict(completed_steps=1)
    assert b.state_hash(resumed.state_dict())==before and b.state_hash(ropt.state_dict())==optim
    raw2,ok2,_,lrs=b.update(resumed,ropt,task,st,rrng,'e2e1',.3,0,stage2=True)
    assert ok and ok2 and np.array_equal(raw1,raw2)
    for a,c in zip(model.parameters(),resumed.parameters()): torch.testing.assert_close(a,c,atol=1e-10,rtol=1e-8)
    assert lrs==[st.lr/25,st.lr/25,st.eta_lr/25]
    b.set_clock(ropt,25,st,True); assert [g['lr'] for g in ropt.param_groups]==[st.lr,st.lr,st.eta_lr]
    with open(entry['path'],'ab') as f: f.write(b'bad')
    with pytest.raises(ValueError): b.load_snapshot(entry,resumed,ropt,rrng)


def analyzer():
    spec=importlib.util.spec_from_file_location('analyze_r44b',ROOT/'scripts/analyze_r44b.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def fixture():
    cells=[]; reassignment=[]; history=[]
    for s in range(10):
        shared=dict(seed=s,initial_sha256=str(s),code_sha256=str(s),first_batch_sha1=str(s),rolling_data=[dict(step=1500,sha256=str(s))],
                    complete=True,failure=None,s5=dict(passed=True))
        for a in ('e2e1emb','e2e1emb_lin','e2e1emb_aice'):
            hits=0 if a=='e2e1emb' else 1024
            cells.append(dict(**shared,task='echo',stage='fresh',arm=a,successful_steps=1500,eval=dict(ce=1.,ce_late=1.,token_sha256='eval'),
                              quick=[dict(local_step=1500,query_hits=hits,query_events=1024,token_sha256='quick')]))
        sw=dict(passed=True,rho_unchanged=True,audit=dict(events=4096,mean=.999),trigger=[dict(step=100,mean=.99),dict(step=125,mean=.99)])
        cells.append(dict(**shared,task='echo',stage='stage1',arm='e2e1emb',switch=sw))
        cells.append(dict(**shared,task='echo',stage='masked',arm='e2e1emb',switch=dict(chance_pass=True,rho_unchanged=True,masked=dict(mean=.0625,events=512))))
        for a in ('kamin2','kaminlin2','kaminctl2'):
            clone={k:'same' for k in ('model_sha256','optimizer_sha256','rng_sha256','snapshot_sha256','audit')}
            cells.append(dict(**shared,task='echo',stage='stage2',arm=a,clone=clone,successful_steps=1000,ever_bound=a!='kamin2',quick=[dict(local_step=0)]))
        for historical,a in ((False,'e2e1emb_aice'),(False,'e2e1emb_lin'),(True,'e2e1emb_lin')):
            rows=[dict(condition=c,mode=m,new_hits=3000 if a=='e2e1emb_aice' else 2000,old_hits=500,events=4096,tokens_sha256='tokens')
                  for c in ('reassign','removed','unchanged') for m in ('streaming','query_freeze','write_snapshot')]
            reassignment.append(dict(seed=s,arm=a,historical=historical,conditions=rows,mechanism=[dict(ok=True)]))
    for s in range(6):
        shared=dict(seed=s,initial_sha256=str(s),code_sha256='none',first_batch_sha1=str(s),rolling_data=[1],complete=True,
                    task='icl_mc',stage='fresh',plain=True,successful_steps=4000,quick=[dict(token_sha256='quick')])
        for a in ('e2e1','e2e1_aice'):
            cells.append(dict(**shared,arm=a,eval=dict(ce=1.,ce_late=1.,token_sha256='eval')))
        history.append(dict(seed=s,eval=dict(ce=1.)))
    return dict(config=dict(smoke=False),cells=cells,reassign=reassignment,historical_icl=history)


def test_analyzer_exact_ties_no_discordances_ni_holm_missing_and_smoke(tmp_path,capsys):
    a=analyzer(); data=fixture(); result=a.analyze(data)
    assert result['primary_rule']=='aice' and result['GM3r']=='PASS'
    assert result['p_R']==result['p_B']==result['p_K']==2**-10
    assert a.mcnemar([(True,True)]*10)['p']==1
    tied=a.sign_test([(1,1)]*10); assert tied['p']==1 and tied['ties']==10 and tied['m']==0 and tied['positive_probability_ci95']==[0,1]
    assert abs(a.binomial_interval(10,10)[0]-(.025**.1))<1e-12
    assert a.ni([.005]*6)['p']==0 and a.ni([.01]*6)['p']==1
    assert a.ni([0]*5)['p']==1 and a.t5_cdf(0)==.5 and abs(a.t5_cdf(-2.015048)-.05)<1e-7
    assert not a.holm(dict(A=.025,K=.04))['A']['passed']
    assert a.holm(dict(A=.024,K=.049))['K']['passed']
    a.write_tables(data,result,tmp_path/'out'); assert (tmp_path/'out/data/cells.csv').exists()
    assert a.finite(dict(n=float('nan'),i=float('inf')))==dict(n=None,i=None)
    data['cells'].pop(0); assert a.analyze(data)['p_A']==1
    data=fixture(); data['cells'][0]['forced_for_smoke']=True
    assert a.analyze(data)['status']=='smoke_only' and a.analyze(data)['p_A']==1
    data=fixture(); data['reassign'][0]['conditions'][0]['new_hits']=2000
    assert a.analyze(data)['reassign_sign']['ties']==1
    data['cells'][0]['rolling_data']=['bad']; assert a.analyze(data)['p_A']==1


def test_stage_driver_cap_failure_forced_forks_yoking_and_reassign_cpu(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg=dict(d_model=8,n_layers=1,n_heads=2,backbone_mult=2,fast_mult=2,window=8,chunk=16,T=96,
             n_keys=4,n_eval=2,n_quick_eval=2,eval_batch=2,n_probe=2,batch=2,steps=2,quick_every=1,
             probe_every=1,lr_total=3000,tf32=False)
    w=b.WaveB(smoke=True,echo_steps=2,kamin_smoke_fixed_switch=2,kamin_stage1_cap=2,
              kamin_probe_every=1,kamin_trigger_sequences=2,kamin_audit_sequences=2,kamin_stage2_steps=2,
              output_dir=str(tmp_path/'snapshots'))
    stage1=b.train_cell(cfg,w,dict(task='echo',seed=0,arm='e2e1emb',stage='stage1'),torch.device('cpu'))
    assert stage1['complete'] and stage1['forced_for_smoke'] and stage1['switch']['rho_unchanged']
    replay=stage1['continuation_check']
    assert replay['passed'] and replay['cloned_state_exact'] and replay['next_batch_exact'], replay
    cap=b.train_cell(cfg,w,dict(task='echo',seed=1,arm='e2e1emb',stage='stage1',cap_failure=True,cap=2),torch.device('cpu'))
    assert cap['failure']=='cap_miss' and not cap['complete']
    ctl=b.train_cell(cfg,w,dict(task='echo',seed=0,arm='e2e1emb',stage='masked',duration=2,
                    rolling_data=stage1['rolling_data'],first_batch_sha1=stage1['first_batch_sha1']),torch.device('cpu'))
    assert ctl['complete'] and ctl['rolling_data']==stage1['rolling_data'] and ctl['initial_sha256']==stage1['initial_sha256']
    twins=[b.train_cell(cfg,w,dict(task='echo',seed=0,arm=a,stage='stage2',snapshot=stage1['snapshot']),torch.device('cpu'))
           for a in ('kamin2','kaminlin2')]
    assert twins[0]['clone']==twins[1]['clone']
    assert all(r['quick'][0]['local_step']==0 and r['quick'][-1]['global_step']==4 for r in twins)
    assert all(r['quick'][1]['lrs']==[.001/25,.001/25,.01/25] for r in twins)
    # Reassign uses the original generator and full sequence shape, with tiny model width.
    cfg.update(T=512,chunk=64,window=64,n_keys=16,n_layers=4)
    st=t._settings(cfg); task=t._task(st,'echo'); model=t.build_model(task,st,torch.device('cpu'),0,'e2e1emb_aice')
    opt=t.build_optim(model,st)
    snapshot=b.save_snapshot(tmp_path/'reassign.pt',model,opt,np.random.default_rng(),{})
    w.reassign_sequences=2
    evaluated=b.reassign_eval(cfg,w,dict(seed=0,arm='e2e1emb_aice',snapshot=snapshot),torch.device('cpu'))
    assert len(evaluated['conditions'])==9 and len(evaluated['mechanism'])==2
    for r in evaluated['conditions']:
        assert r['events']==32 and type(r['new_hits']) is int
        if r['condition']=='reassign': assert r['other_hits']+r['new_hits']+r['old_hits']==32
    mechanism=evaluated['mechanism'][0]
    assert len(mechanism['raw_gradient_norms'])==len(mechanism['delta_norms'])==4
    assert len(mechanism['increment_p_g'])==2


def test_failure_audit_is_terminal_no_search_for_another_crossing(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg=dict(d_model=8,n_layers=1,n_heads=2,backbone_mult=2,fast_mult=2,window=8,chunk=16,T=96,
             n_keys=4,n_eval=2,n_quick_eval=2,eval_batch=2,batch=2,steps=2,lr_total=3000)
    w=b.WaveB(echo_steps=2,kamin_stage1_cap=5,kamin_probe_every=1,kamin_trigger_sequences=2,kamin_audit_sequences=2,
              output_dir=str(tmp_path/'snapshots'))
    calls=[]
    def audit(model,task,st,n,split,masked=False):
        calls.append(split)
        return dict(mean=.999 if split==0 else .98,events=n*4,quantiles=[],per_sequence=[.99]*n,token_sha256=str(split))
    monkeypatch.setattr(b,'copy_audit',audit)
    result=b.train_cell(cfg,w,dict(task='echo',seed=0,arm='e2e1emb',stage='stage1'),torch.device('cpu'))
    assert result['failure']=='audit_failure' and result['successful_steps']==2 and calls==[0,0,1]
    assert Path(result['snapshot']['path']).is_file()


def test_phase_order_seed_groups_and_dependency_manifest_without_launches(tmp_path,monkeypatch):
    cfg=tomllib.loads((ROOT/'configs/toy/r4-4b.toml').read_text())['ttt']
    cfg['wave_b']['output_dir']=str(tmp_path/'out')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(t,'_mark_process',lambda name:None)
    cfg['wave_b']['lock_path']=str(tmp_path/'workspace/TTM-r44b.lock')
    monkeypatch.setattr(b,'run_state_guard',lambda:None)
    monkeypatch.setattr(b,'device_guard',lambda:None)
    monkeypatch.setattr(b,'verify_smoke_receipt',lambda w:None)
    historical=[dict(task=task,seed=s,arm='e2e1emb_lin' if task=='echo' else 'e2e1')
                for task,n in [('echo',10),('icl_mc',6)] for s in range(n)]
    monkeypatch.setattr(b,'verify_historical',lambda *a:historical)
    called=[]
    def phase(cfg,w,number,groups,deadline,workers=None):
        called.append((number,copy.deepcopy(groups)))
        rows=[]
        for group in groups:
            assert len({j['seed'] for j in group})==1
            for job in group:
                row=dict(job)
                row.update(complete=True,snapshot=dict(path=f"{number}/{job['seed']}",sha256='hash'),
                         rolling_data=[dict(step=50,sha256=str(job['seed']))],first_batch_sha1='batch',switch=dict(n_s=50))
                if number==4 and job['seed']==1:
                    row['complete']=False; row['failure']='cap_miss'
                rows.append(row)
        return rows
    monkeypatch.setattr(b,'phase',phase)
    monkeypatch.setattr(b,'load_analyzer',lambda:types.SimpleNamespace(analyze=lambda p:dict(status='fixture'),write_tables=lambda *a:None))
    assert b.run(cfg,0)['status']=='fixture'
    assert [n for n,_ in called]==[1,2,3,4,5,6,7]
    assert [g[0]['seed'] for g in called[0][1]]==list(range(10))
    assert all([j['arm'] for j in group]==list(b.ECHO) for group in called[0][1])
    masked=called[4][1]; stage2=called[5][1]
    assert 1 not in [g[0]['seed'] for g in masked] and len(masked)==9
    assert all(g[0]['duration']==50 for g in masked)
    for group in stage2:
        assert [j['arm'] for j in group]==list(b.TWINS)
        assert group[0]['snapshot']==group[1]['snapshot']
        assert group[2]['snapshot']!=group[0]['snapshot']


def test_analyzer_cli_tables_and_invalid_manipulations(tmp_path,capsys):
    a=analyzer(); data=fixture(); log=tmp_path/'log'
    lines=['R44B_CONFIG '+json.dumps(dict(wave_b=data['config']))]
    for prefix,key in [('R44B_CELL','cells'),('R44B_REASSIGN','reassign'),('R44B_HISTORICAL_ICL','historical_icl')]:
        lines.extend(prefix+' '+json.dumps(row) for row in data[key])
    log.write_text('\n'.join(lines)+'\n')
    a.main(['--process-name','TTM-r44b-97-analyzer',str(log),str(tmp_path/'tables')])
    out=capsys.readouterr().out
    assert out.count('R44B_SUMMARY ')==1
    summary=json.loads((tmp_path/'tables/summary.json').read_text())
    assert summary['primary_rule']=='aice'
    for filename in ['cells','quick-trajectories','kamin-switch-audit','kaminlin2','write2-mechanism','eta','historical-lin-audit','historical-icl-audit']:
        assert (tmp_path/f'tables/data/{filename}.csv').is_file()
    assert a.analyze(dict(config=dict(smoke=False)))['p_A']==1
    invalid=copy.deepcopy(data)
    next(c for c in invalid['cells'] if c['stage']=='masked')['switch']['masked']['mean']=.2
    assert a.analyze(invalid)['p_K']==1
    invalid=copy.deepcopy(data)
    next(c for c in invalid['cells'] if c['arm']=='kaminlin2')['clone']['optimizer_sha256']='wrong'
    assert a.analyze(invalid)['p_K']==1
    invalid=copy.deepcopy(data)
    bad=next(c for c in invalid['cells'] if c['arm']=='kaminctl2')
    bad.update(failure='algorithmic_nonfinite',complete=False,successful_steps=12,ever_bound=True)
    result=a.analyze(invalid)
    assert result['Kamin']['n']==10 and result['Kamin']['positive']==9


def test_run_state_and_smoke_receipt_guards(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    import time
    assert b.run_state_guard() is None
    # The receipt lives at the configured path, not under the run's output directory.
    w=b.WaveB(smoke_receipt_path=str(tmp_path/'workspace/r44b-podsmoke/smoke-receipt.json'))
    with pytest.raises(ValueError,match='missing'): b.verify_smoke_receipt(w)
    files=b.source_files()
    receipt=dict(project_id=b.PROJECT,passed=True,memory_workers=4,six_worker_tranche_attempted=True,
                 aggregate_peak_gib=12.,workers_exited=True,continuation_passed=True,
                 source_sha256=b.state_hash(files),source_files=files)
    b.atomic_json(w.smoke_receipt_path,receipt)
    assert b.verify_smoke_receipt(w)['memory_workers']==4
    changed=dict(files); changed['uv.lock']='0'*64
    receipt.update(source_sha256=b.state_hash(changed),source_files=changed)
    b.atomic_json(w.smoke_receipt_path,receipt)
    with pytest.raises(ValueError,match='uv.lock'): b.verify_smoke_receipt(w)
    receipt.update(source_sha256=b.state_hash(files),source_files=files,passed=False)
    b.atomic_json(w.smoke_receipt_path,receipt)
    with pytest.raises(ValueError,match='failed'): b.verify_smoke_receipt(w)
    assert not b.smoke_complete([],[],[],dict(workers=6,records=[]))
    cfg=tomllib.loads((ROOT/'configs/toy/r4-4b.toml').read_text())['ttt']
    smoke=tomllib.loads((ROOT/'configs/toy/smoke/r4-4b-gpu.toml').read_text())['ttt']
    assert cfg['wave_b']['smoke_receipt_path']==smoke['wave_b']['smoke_receipt_path']=='outputs/r44b-podsmoke/smoke-receipt.json'
    assert cfg['wave_b']['lock_path']==smoke['wave_b']['lock_path']=='outputs/TTM-r44b.lock'


def test_source_hash_ignores_untracked_state(tmp_path):
    root=tmp_path/'repo'
    for name in ('src/tttmem/a.py','scripts/analyze_r44b.py','configs/toy/r4-4b.toml','pyproject.toml','uv.lock',
                 '.python-version','configs/toy/r4-4b/historical-manifest.json','configs/toy/r4-4b/wave-a-anchor-audit.json'):
        (root/name).parent.mkdir(parents=True,exist_ok=True); (root/name).write_text(name)
    before=b.source_files(root)
    assert set(before)=={'src/tttmem/a.py','scripts/analyze_r44b.py','configs/toy/r4-4b.toml','pyproject.toml','uv.lock',
                         '.python-version','configs/toy/r4-4b/historical-manifest.json','configs/toy/r4-4b/wave-a-anchor-audit.json'}
    for name in ('.git/HEAD','.venv/lib/x.py','src/tttmem/__pycache__/a.cpython-312.pyc','checkpoints/r44b/x.pt',
                 'inputs/run-state.json'):
        (root/name).parent.mkdir(parents=True,exist_ok=True); (root/name).write_text('noise')
    os.utime(root/'uv.lock',(1,1))
    assert b.source_files(root)==before
    (root/'src/tttmem/a.py').write_text('changed')
    assert b.source_files(root)!=before


def test_workspace_lock_and_device_guard(tmp_path,monkeypatch):
    path=tmp_path/'workspace/TTM-r44b.lock'
    mine=b.acquire_lock(path,'TTM-r44b-02')
    assert json.loads(path.read_text())['pid']==os.getpid()
    with pytest.raises(RuntimeError,match='holds'): b.acquire_lock(path,'TTM-r44b-02')
    b.release_lock(path,mine); assert not path.exists()
    dead=subprocess.Popen([sys.executable,'-c','pass']); dead.wait()
    path.write_text(json.dumps(dict(pid=dead.pid,host=b.socket.gethostname())))
    mine=b.acquire_lock(path,'TTM-r44b-02')  # stale local holder is replaced
    assert json.loads(path.read_text())['pid']==os.getpid()
    b.release_lock(path,dict(pid=-1)); assert path.exists()  # never removes another holder's lock
    b.release_lock(path,mine)
    path.write_text(json.dumps(dict(pid=dead.pid,host='another-host')))
    with pytest.raises(RuntimeError): b.acquire_lock(path,'TTM-r44b-02')
    path.write_text('')  # creation interrupted before the record was written
    b.release_lock(path,b.acquire_lock(path,'TTM-r44b-02'))
    outputs={}
    def smi(args,**kwargs):
        assert args[0]=='nvidia-smi'
        key='apps' if any('compute-apps' in a for a in args) else 'used'
        return types.SimpleNamespace(stdout=outputs[key])
    monkeypatch.setattr(b.subprocess,'run',smi)
    outputs.update(apps='',used='120\n')
    b.device_guard()
    outputs.update(apps='4242, 900 MiB')
    with pytest.raises(RuntimeError,match='compute process'): b.device_guard()
    outputs.update(apps='',used='2048\n')
    with pytest.raises(RuntimeError,match='already holds'): b.device_guard()
    outputs.update(used='')
    with pytest.raises(RuntimeError,match='unavailable'): b.device_guard()


class _FakeProcess:
    """Thread-backed stand-in for a spawned worker: no extra torch process."""
    def __init__(self,target,args,stop):
        import threading
        self._target,self._args,self._stop,self.exitcode=target,args,stop,None
        self._thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        try:
            self._target(*self._args); self.exitcode=0
        except BaseException:
            self.exitcode=1
    def start(self): self._thread.start()
    def is_alive(self): return self._thread.is_alive()
    def join(self,timeout=None): self._thread.join(5 if timeout is None else timeout)
    def terminate(self): self._stop.set()


class _FakeCtx:
    def __init__(self):
        import threading; self.stop=threading.Event()
    def Queue(self):
        import queue as q
        class Q(q.Queue):
            def close(self): pass
        out=Q(); out.stop=self.stop; return out
    def Lock(self):
        import threading; return threading.Lock()
    def Barrier(self,n):
        import threading; return threading.Barrier(n)
    def Process(self,target,args,name=None): return _FakeProcess(target,args,self.stop)


@pytest.mark.parametrize('scenario',['complete','crash','deadline'])
def test_phase_records_terminal_status_without_raising(tmp_path,monkeypatch,scenario):
    import queue as q, time as clock
    monkeypatch.setattr(b.mp,'get_context',lambda kind:_FakeCtx())
    monkeypatch.setattr(b,'occupancy',lambda:None)
    def worker(index,cfg,w,jobs,results,lock,barrier=None):
        while not jobs.stop.is_set():
            try:
                group=jobs.get(timeout=.02)
            except q.Empty:
                continue
            if group is None: return
            if scenario=='crash' and group[0]['seed']==1: raise RuntimeError('simulated hard crash')
            if scenario=='deadline' and group[0]['seed']==1:
                jobs.stop.wait(5); return
            results.put([dict(seed=group[0]['seed'],complete=True)])
    monkeypatch.setattr(b,'_worker',worker)
    w=b.WaveB(output_dir=str(tmp_path/'out'))
    deadline=clock.time()+60
    if scenario=='deadline':
        real=clock.time; start=real()
        # The hard grace expires after the first groups are admitted.
        monkeypatch.setattr(b.time,'time',lambda:start if real()-start<.3 else start+10**4)
    rows=b.phase(dict(workers=2,process_name='TTM-r44b-02'),w,1,[[dict(seed=s)] for s in range(2)],deadline)
    want=dict(complete='complete',crash='failed',deadline='deadline_terminated')[scenario]
    assert rows.status==want
    assert sorted(r['seed'] for r in rows)==([0,1] if scenario=='complete' else [0])
    assert (rows.error is None)==(scenario!='crash')
    manifest=json.loads((tmp_path/'out/phase-1-manifest.json').read_text())
    assert manifest['status']==want and manifest['workers_exited']==(scenario=='complete')
    if scenario=='crash':
        with pytest.raises(RuntimeError,match='memory tranche'):
            b.phase(dict(workers=2,process_name='TTM-r44b-01-podsmoke'),w,'preflight',[[dict(seed=s)] for s in range(2)],deadline)


@pytest.mark.parametrize('status',['failed','deadline_terminated'])
def test_driver_writes_evidence_and_analysis_after_phase_trouble(tmp_path,monkeypatch,status):
    cfg=tomllib.loads((ROOT/'configs/toy/r4-4b.toml').read_text())['ttt']
    cfg['wave_b'].update(output_dir=str(tmp_path/'out'),lock_path=str(tmp_path/'workspace/lock'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(t,'_mark_process',lambda name:None)
    for name in ('run_state_guard','device_guard'):
        monkeypatch.setattr(b,name,lambda:None)
    monkeypatch.setattr(b,'verify_smoke_receipt',lambda w:dict(memory_workers=6))
    monkeypatch.setattr(b,'verify_historical',lambda *a:[])
    def phase(cfg,w,number,groups,deadline,workers=None):
        rows=b.PhaseRows()
        if number==1: rows.status,rows.error=status,'simulated'
        return rows
    monkeypatch.setattr(b,'phase',phase)
    if status=='failed':
        with pytest.raises(RuntimeError,match='evidence and analysis written'): b.run(cfg,0)
    else:
        assert b.run(cfg,0)['status']=='incomplete_or_invalid'
    evidence=json.loads((tmp_path/'out/evidence.json').read_text())
    assert evidence['phases'][0]==dict(phase=1,status=status,error='simulated',groups=10,rows=0)
    summary=json.loads((tmp_path/'out/analysis/summary.json').read_text())
    assert summary['p_A']==summary['p_K']==1 and summary['infrastructure']['phases'][0]['status']==status
    assert not (tmp_path/'workspace/lock').exists()


def test_historical_audits_reported_but_not_gating_the_aice_headline():
    a=analyzer(); data=fixture()
    data['historical_icl']=[]
    data['reassign']=[r for r in data['reassign'] if not (r['historical'] and r['seed']==3)]
    result=a.analyze(data)
    assert result['p_A']==2**-10 and result['primary_rule']=='aice'
    assert not result['validity']['historical_icl'] and not result['validity']['historical_lin_audit']
    assert 'historical_icl_audit' in result['missing'] and 'historical_lin_audit/s3' in result['missing']
    assert result['reassign_seed_differences'][3]['historical_lin_hits'] is None
    data=fixture(); data['reassign']=[r for r in data['reassign'] if not (r['arm']=='e2e1emb_lin' and not r['historical'] and r['seed']==3)]
    assert a.analyze(data)['p_A']==1
    data=fixture(); next(c for c in data['cells'] if c['stage']=='stage1')['switch']['trigger'][-1]['mean']=None
    assert a.analyze(data)['p_K']==1


def test_smoke_driver_writes_receipt_to_absolute_path_and_four_worker_fallback(tmp_path,monkeypatch):
    cfg=tomllib.loads((ROOT/'configs/toy/smoke/r4-4b-gpu.toml').read_text())['ttt']
    receipt_path=tmp_path/'workspace/r44b-podsmoke/smoke-receipt.json'
    cfg['wave_b'].update(output_dir=str(tmp_path/'out'),lock_path=str(tmp_path/'workspace/lock'),
                         smoke_receipt_path=str(receipt_path))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(t,'_mark_process',lambda name:None)
    for name in ('run_state_guard','device_guard'):
        monkeypatch.setattr(b,name,lambda:None)
    monkeypatch.setattr(b,'verify_historical',lambda *a:[])
    seen=[]
    def phase(cfg,w,number,groups,deadline,workers=None):
        seen.append((number,workers or cfg['workers']))
        if number=='memory':
            raise b.MemorySizingFailure('simulated')
        rows=b.PhaseRows()
        if number=='memory4':
            b.atomic_json(Path(w.output_dir)/'phase-memory4-manifest.json',
                          dict(workers=4,records=[],aggregate_peak_gib=13.,workers_exited=True))
        return rows
    monkeypatch.setattr(b,'phase',phase)
    assert b.run(cfg,0)['status']=='smoke_only'
    assert seen[:3]==[('preflight',2),('memory',6),('memory4',4)]
    assert all(n==(1 if num=='cap_failure' else 4) for num,n in seen[3:])  # four-worker certificate caps later phases
    receipt=json.loads(receipt_path.read_text())
    assert receipt==json.loads((tmp_path/'out/smoke-receipt.json').read_text())
    assert receipt['memory_workers']==4 and not receipt['passed']  # incomplete smoke never certifies
    assert receipt['source_sha256']==b.source_hash() and receipt['source_files']==b.source_files()
