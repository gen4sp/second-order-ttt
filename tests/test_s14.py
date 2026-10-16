"""S14 CPU contracts. Workers are stubbed here; the real subprocess smoke is separate."""
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib
import types

import numpy as np
import pytest
import torch

from tttmem.p2 import s14, s14_data, matrix
from tttmem.p2.train import (TrainCfg, ARMS, checkpoint_steps, inner_lr_multiplier,
                             schedule_lr, stream_order, build, partial_so_step)
from tttmem.p2.meta import InnerCfg
from test_p2 import _smoke_data

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n", [1200, 3469])
def test_wsd_schedule_key_coordinates(n):
    cfg = TrainCfg(arm="wsd", lr_schedule="wsd", steps=n, warmup=120, inner_warmup=120, wsd_decay=240)
    cfg.validate_layout()
    for step, fraction in ((0, 0), (60, .5), (120, 1), (n-240, 1), (n-120, .5), (n, 0), (n+1, 0)):
        assert schedule_lr(step, cfg) == pytest.approx(cfg.lr_floor + fraction*(cfg.lr_peak-cfg.lr_floor))
    for step, mult in ((0, .1), (60, .55), (120, 1), (n-240, 1),
                       (n-120, 1), (n, 1), (n+1, 1)):
        assert inner_lr_multiplier(step, cfg) == pytest.approx(mult)
    assert ARMS["wsd"] == ARMS["e2e1-ce"]


@pytest.mark.parametrize("updates", [dict(lr_schedule="bad"), dict(wsd_decay=0), dict(steps=100),
                                      dict(lr_schedule="cosine")])
def test_invalid_wsd_fails(updates):
    cfg = TrainCfg(arm="wsd", lr_schedule="wsd", steps=1200, warmup=120)
    with pytest.raises(ValueError):
        replace(cfg, **updates).validate_layout()


@pytest.mark.parametrize("seed", range(6))
def test_seed_values_match_mini_g43_g45_convention(seed):
    cell = dict(preset="mini-m", arm="e2e2", init_seed=seed, data_seed=seed)
    cfg = matrix._base_config({}, cell, REPO)
    TrainCfg.from_dict(cfg)  # no seed registry refusal for 3–5
    assert cfg['init_seed'] == cfg['data_seed'] == seed
    prefix = stream_order([25600], seed, 1200*16)
    longer = stream_order([25600,25600,25600], seed,3469*16)
    assert np.array_equal(prefix, longer[:len(prefix)])
    assert (longer[3200*16:] >= 51200).all()


def _pilot():
    settings = [dict(arm=a, micro_batch=m, so_micro_batch=sm, status="ok", n_timed=50,
                     median_ms=t, peak_gb=peak) for a,m,sm,t,peak in (
                     ('e2e2',1,1,7200,9),('e2e2',2,1,6600,15),
                     ('e2e12-p125',2,1,2900,12),('e2e12-p125',4,2,2800,19),
                     ('e2e12-p125',4,1,2000,21))]
    return dict(settings=settings,timed_steps=50,warmup_steps=10)


def test_pilot_rounding_fastest_with_headroom():
    d = s14.decide_n_p(_pilot())
    assert d['n_p'] == round(1200*6600/2800)
    assert (d['p_micro_batch'],d['p_so_micro_batch']) == (4,2)
    assert d['train_seconds_target'] == 7920


@pytest.mark.parametrize("field,value", [('peak_gb',None),('peak_gb',float('nan')),
                                        ('n_timed',49),('median_ms',0),('median_ms',float('nan'))])
def test_pilot_no_valid_setting_fails(field,value):
    p = _pilot()
    for r in p['settings']:
        if r['arm']=='e2e12-p125':
            r[field]=value
    with pytest.raises(RuntimeError):
        s14.decide_n_p(p)


def _cfg(tmp_path):
    data,_ = _smoke_data(tmp_path)
    return dict(mode='s14_pod_job',preset='tiny',device='cpu',max_hours=1,
                process_name='TTM-s14-02-test',process_nn=51,out_dir=str(tmp_path/'s14'),
                data_dir=str(data),n_p=2,p_micro_batch=1,p_so_micro_batch=1,
                cells=[dict(armdir='mini-p',seed=3),dict(armdir='wsd-1200',seed=3)])


def _fake_worker(calls, fail=None):
    def run(phase,payload,root,name,deadline):
        calls.append(phase)
        if phase == fail:
            raise RuntimeError(f'injected {phase} failure')
        if phase == 'train':
            out=Path(payload['out_dir']);out.mkdir(parents=True,exist_ok=True)
            for f in (f"model_step{payload['steps']:04d}.pt",f"eval_step{payload['steps']:04d}.npz",
                      'eval_meta.npz','steps.jsonl','events.jsonl','compute.json',f"resume_step{payload['steps']:04d}.pt"):
                (out/f).write_bytes(b'fixture')
            return dict(final=dict(steps=payload['steps'],aborted=False),peak_gb=0.)
        if phase == 'reads':
            for item in payload['items']:
                out=Path(item['out']);out.mkdir(parents=True,exist_ok=True)
                for n in [f'g0_read_{r}.npz' for r in item['reads']]+['g0_hlm.npz']:
                    (out/n).write_bytes(b'fixture')
        if phase == 'passkey' and not payload.get('preflight'):
            out=Path(payload['out_dir'])
            names=[s14.read_name(e) for e in payload['eta_mults']]
            if payload.get('noadapt',True):names+=['noadapt']
            for n in names:(out/f'passkey_{n}.npz').write_bytes(b'fixture')
        return dict(done=1,skipped=0,errors=0)
    return run


def test_pod_job_order_and_manifest(tmp_path,monkeypatch):
    cfg=_cfg(tmp_path);cfg['passkey']=True
    calls=[];monkeypatch.setattr(s14,'execute_worker',_fake_worker(calls))
    result=s14.pod_job(cfg)
    assert calls==['data','passkey','train','train','reads','passkey','reads','passkey']
    assert result['status']=='done'
    root=Path(cfg['out_dir']);manifest=json.loads((root/'manifest.json').read_text())
    assert (root/'DONE').is_file()
    paths=[r['path'] for r in manifest['files']]
    assert 'runs/tiny/mini-p/s3/model_step0002.pt' in paths
    assert '.cell-configs/tiny-wsd-1200-s3.json' in paths
    assert not any('resume_' in p for p in paths)
    assert all(s14.sha256_file(root/r['path'])==r['sha256'] for r in manifest['files'])


@pytest.mark.parametrize('fail',['data','train','reads','passkey'])
def test_pod_job_fails_closed(tmp_path,monkeypatch,fail):
    cfg=_cfg(tmp_path);cfg['passkey']=True
    calls=[];monkeypatch.setattr(s14,'execute_worker',_fake_worker(calls,fail))
    with pytest.raises(RuntimeError):s14.pod_job(cfg)
    root=Path(cfg['out_dir'])
    assert not (root/'DONE').exists() and not (root/'manifest.json').exists()
    assert (root/'FAILED.json').is_file()
    assert calls[-1]==fail
    if fail=='passkey': assert calls==['data','passkey']  # after verified data, before training


def test_missing_n_p_stops_before_data(tmp_path,monkeypatch):
    cfg=_cfg(tmp_path);del cfg['n_p'];calls=[]
    monkeypatch.setattr(s14,'execute_worker',_fake_worker(calls))
    with pytest.raises(ValueError,match='before this pod starts'):s14.pod_job(cfg)
    assert not calls


@pytest.mark.parametrize('armdir',s14.ARMDIRS)
def test_output_layout_and_read_grids(tmp_path,armdir):
    cfg=_cfg(tmp_path);root=Path(cfg['out_dir'])
    base=s14.cell_config(cfg,dict(armdir=armdir,seed=4),root,dict(n_p=2,p_micro_batch=1,p_so_micro_batch=1))
    rc=s14.read_config(cfg,base,armdir,root)
    assert base['out_dir']==str(root/f'runs/tiny/{armdir}/s4')
    assert rc['items'][0]['out']==str(root/f'reads/tiny/{armdir}/s4')
    assert rc['process_nn']==1
    assert rc['items'][0]['reads'][0]=='ce'
    assert set(rc['kappa_reads'])=={'ce','eta200','eta400'}
    wanted=set(s14.SO_GRID if armdir=='so' else s14.FO_GRID)|{1.,2.,4.}
    assert set(rc['items'][0]['reads'])=={s14.read_name(e) for e in wanted}
    assert checkpoint_steps(TrainCfg.from_dict(base),'so')=={2}


def test_passkey_hook_stub_and_missing_outputs(tmp_path,monkeypatch):
    calls=[];mod=types.ModuleType('tttmem.p2.passkey')
    def stub(model_path,cell_config_path,eta_mults,out_dir,*,n_docs=256,seed=1014,device='cuda',
             data_dir='data/p2data',noadapt=True,batch=1):
        calls.append((model_path,cell_config_path,eta_mults,n_docs,seed,device,data_dir,noadapt))
        for n in [s14.read_name(e) for e in eta_mults]+['noadapt']:
            np.savez(Path(out_dir)/f'passkey_{n}.npz',fixture=True)
        return {'stub':True}
    mod.passkey_read=stub;monkeypatch.setitem(sys.modules,'tttmem.p2.passkey',mod)
    mod._tokenizer=lambda *a:object()
    mod._host_rows=lambda *a:(np.zeros((256,1)),None,None)
    mod.candidates=lambda *a:None
    payload=dict(model_path='m.pt',cell_config_path='c.json',eta_mults=[1.,2.75],out_dir=str(tmp_path))
    assert s14.passkey_hook({**payload,'preflight':True})['passkey_available'] and not calls
    assert s14.passkey_hook(payload)=={'stub':True}
    assert calls[0][3:]==(256,1014,'cuda','data/p2data',True)
    (tmp_path/'passkey_eta275.npz').unlink()
    mod.passkey_read=lambda *a,**k: {}
    with pytest.raises(RuntimeError,match='eta275'):s14.passkey_hook(payload)


def test_missing_passkey_module_raises(monkeypatch):
    real=importlib.import_module
    def missing(name):
        if name=='tttmem.p2.passkey':raise ModuleNotFoundError(name)
        return real(name)
    monkeypatch.setattr(s14.importlib,'import_module',missing)
    with pytest.raises(ModuleNotFoundError):
        s14.passkey_hook(dict(model_path='',cell_config_path='',eta_mults=[1.],out_dir='',preflight=True))


def test_resolve_n_p_refuses_conflict(tmp_path):
    p=tmp_path/'pilot/n_p.json';p.parent.mkdir();p.write_text(json.dumps(dict(n_p=2829,base_steps=1200)))
    with pytest.raises(ValueError,match='disagrees'):s14.resolve_n_p(dict(n_p=2800),tmp_path)
    assert s14.resolve_n_p({},tmp_path)['n_p']==2829


def test_config_n_p_keeps_pilot_microbatch_selection(tmp_path):
    with pytest.raises(ValueError,match='pilot-selected'):
        s14.resolve_n_p(dict(n_p=2829),tmp_path)
    with pytest.raises(ValueError,match='actual pilot times'):
        s14.resolve_n_p(dict(n_p=2829,p_micro_batch=4,p_so_micro_batch=2),tmp_path)
    rec=s14.resolve_n_p(dict(n_p=2829,p_micro_batch=4,p_so_micro_batch=2,
                            t_so_pilot_s=6.6,t_arm_pilot_s=2.8),tmp_path)
    assert rec['p_micro_batch']==4 and rec['p_so_micro_batch']==2


def test_reused_training_outputs_fail_on_mutation(tmp_path):
    base=dict(steps=2,out_dir=str(tmp_path))
    files=('model_step0002.pt','eval_step0002.npz','eval_meta.npz','steps.jsonl','events.jsonl',
           'resume_step0002.pt','compute.json')
    for n in files:(tmp_path/n).write_bytes(b'fixture')
    result=dict(final=dict(steps=2,aborted=False),output_sha256={n:s14.sha256_file(tmp_path/n) for n in files})
    s14._require_training(base,result)
    (tmp_path/'model_step0002.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='output changed'):s14._require_training(base,result)


def test_all_production_configs_layouts():
    paths=sorted((REPO/'configs/s14').glob('r-pod*.toml'))
    assert len(paths)==12
    found=set()
    for p,path in enumerate(paths,1):
        with path.open('rb') as f:cfg=tomllib.load(f)['p2']
        assert cfg['process_name']==f'TTM-s14-{50+p:02d}'
        assert cfg['pilot']['enabled']==(p==1)
        assert cfg['data']['mode']=='uploaded' and cfg['max_hours']>0
        assert not cfg['data']['rebuild_c'] and cfg['data']['verify_c']
        assert cfg['read_batch']==2 and cfg['passkey_batch']==1
        assert cfg['data']['expected_c_hashes']=='configs/s14/data/expected-c-sha256.json'
        found|={c['seed'] for c in cfg['cells']}
    assert found=={3,4,5}


def test_metadata_provenance_check_only_allows_runtime_diagnostics():
    s14_data._contains({'a':1,'wall_time':9,'extra':3},{'a':1,'wall_time':2})
    with pytest.raises(ValueError,match='provenance'):s14_data._contains({'a':2},{'a':1})


def test_uploaded_data_verification_and_mismatch_stops(tmp_path):
    sys.path.insert(0,str(REPO/'scripts'))
    from s14_test_support import tiny_store
    data=tmp_path/'p2data';expected=tiny_store(data);root=tmp_path/'job';root.mkdir()
    cfg=dict(data_dir=str(data),data=dict(mode='uploaded',expected_hashes=str(expected)))
    good=s14_data.prepare_data(cfg,root,need_c=False)
    assert good['verification']['verified'] and len(good['verification']['sha256'])==13
    with (data/'A.bin').open('r+b') as f:f.write(b'bad!')
    with pytest.raises(ValueError,match='sha256 mismatch'):
        s14_data.prepare_data(cfg,root,need_c=False)


def test_uploaded_long_cell_requires_c(tmp_path):
    sys.path.insert(0,str(REPO/'scripts'))
    from s14_test_support import tiny_store
    data=tmp_path/'p2data';expected=tiny_store(data);root=tmp_path/'job';root.mkdir()
    cfg=dict(data_dir=str(data),data=dict(mode='uploaded',expected_hashes=str(expected)))
    with pytest.raises(ValueError,match='split C'):
        s14_data.prepare_data(cfg,root,need_c=True)


@pytest.mark.parametrize('armdir',s14.ARMDIRS)
def test_production_recipes_and_c_boundary(tmp_path,monkeypatch,armdir):
    # No large model/tokens are allocated for production config validation.
    class Rows:
        def __len__(self):return 25600
    monkeypatch.setattr(s14,'open_split',lambda *a:(Rows(),[]))
    cfg=dict(preset='mini-m',device='cpu',process_name='TTM-s14-03-test',data_dir=str(tmp_path))
    p=dict(n_p=2829,p_micro_batch=4,p_so_micro_batch=1)
    base=s14.cell_config(cfg,dict(armdir=armdir,seed=5),tmp_path,p)
    tc=TrainCfg.from_dict(base)
    assert tc.global_batch==16 and tc.lr_peak==.003 and tc.lr_floor==1e-5
    assert tc.barriers==[2,4,6] and not tc.fused_attn
    assert tc.init_seed==tc.data_seed==5
    assert tc.guard_fo_stop and tc.guard_consecutive==2 and tc.guard_max_rise==.1
    if armdir in ('fo-cm','mini-p'):
        assert tc.warmup==tc.inner_warmup==round(.1*tc.steps)
        assert tc.lr_schedule=='cosine'
    if armdir=='mini-p':assert tc.so_fraction==.125 and tc.steps==2829
    if armdir.startswith('wsd-'):
        assert tc.warmup==tc.inner_warmup==120 and tc.wsd_decay==240 and tc.micro_batch==4
    if armdir in ('fo-cm','wsd-3469'):assert tc.train_seconds_target==1200*s14.N3B_T_SO
    if armdir in ('fo-cm','wsd-3469'):assert tc.extra_train_splits==['B','C'] and tc.steps==3469
    else:assert 'C' not in tc.extra_train_splits


def test_pilot_workers_use_fair_microbatches_and_exact_window(tmp_path,monkeypatch):
    cfg=dict(preset='mini-m',data_dir=str(tmp_path),process_name='TTM-s14-04-test',device='cpu')
    calls=[]
    monkeypatch.setattr(s14,'warm_page_cache',lambda d,s:{'splits':s,'bytes':123})
    def worker(phase,base,root,name,deadline):
        assert phase=='train' and base['global_batch']==16 and base['stop_after']==60
        assert base['eval_steps']==[] and not base['save_checkpoints'] and base['guard_every']==0
        calls.append((base['arm'],base['micro_batch'],base['so_micro_batch']))
        time=6600 if base['arm']=='e2e2' else 2300 if base['arm']=='e2e1-ce' else 2800
        return dict(final=dict(step_ms=[100000]*10+[time]*50,steps=60),peak_gb=19.)
    monkeypatch.setattr(s14,'execute_worker',worker)
    result=s14.timing_pilot(cfg,tmp_path,100000000.)
    assert calls==[('e2e2',1,1),('e2e2',2,1),('e2e1-ce',2,1),('e2e1-ce',4,1),
                   ('e2e12-p125',2,1),('e2e12-p125',2,2),
                   ('e2e12-p125',4,1),('e2e12-p125',4,2)]
    assert result['n_p']==round(1200*6600/2800)
    assert result['pilot']['page_cache']['splits']==['A','B']
    assert (tmp_path/'pilot/n_p.json').is_file()


def test_job_prepares_c_after_pilot_if_p_horizon_crosses_boundary(tmp_path,monkeypatch):
    class Rows:
        def __len__(self):return 25600
    cfg=dict(preset='mini-m',data_dir=str(tmp_path),out_dir=str(tmp_path/'s14'),max_hours=1,
             process_name='TTM-s14-05-test',cells=[dict(armdir='mini-p',seed=3)],pilot=dict(enabled=True))
    calls=[];data_flags=[]
    fake=_fake_worker(calls)
    def execute(phase,payload,*args):
        if phase=='data':data_flags.append(payload['need_c'])
        return fake(phase,payload,*args)
    monkeypatch.setattr(s14,'execute_worker',execute)
    monkeypatch.setattr(s14,'timing_pilot',lambda *a:dict(n_p=3500,p_micro_batch=2,p_so_micro_batch=1,
                                                        t_so_ms=7000,t_p_ms=2400))
    monkeypatch.setattr(s14,'open_split',lambda *a:(Rows(),[]))
    assert s14.pod_job(cfg)['status']=='done'
    assert data_flags==[False,True]
