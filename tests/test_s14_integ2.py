"""Pre-launch fail-closed integrity and automatic batch recovery regressions."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tttmem.p2 import s14, s14_batch, s14_analysis, passkey, mini
from tttmem.p2.s14_evidence import PullIntegrity


@pytest.mark.parametrize('failure',[False,True])
def test_frontier_selfcheck_recovers_all_reads(tmp_path,monkeypatch,failure):
    calls=[]
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'weights')
    cell=tmp_path/'cell.json';cell.write_text('{}')
    out=tmp_path/'reads'
    item=dict(out=str(out),ckpt=str(checkpoint),cell_config=str(cell),reads=['ce','eta200','eta400'],hlm=True)
    payload=dict(items=[item],read_batch=8,kappa_reads=['ce','eta200','eta400'],endpoint_parity=True)
    def reader(cfg):
        calls.append(deepcopy(cfg))
        folder=Path(cfg['items'][0]['out']);folder.mkdir(parents=True,exist_ok=True)
        value=.001 if failure and cfg['read_batch']==8 else 0.
        for name in cfg['items'][0]['reads']:
            np.savez(folder/f'g0_read_{name}.npz',eval_rows=np.arange(4),M_doc=np.full(4,value),
                L_adapt_count=np.ones(2),L_adapt_doc=np.ones(2),kz_probe_offset=np.arange(5),
                kz_rank_f=np.ones(4),kz_rank_nf=np.ones(4))
        np.savez(folder/'g0_hlm.npz',heldout_count=np.ones(2),heldout_sum=np.ones(2))
        return dict(done=1,skipped=0,errors=0)
    monkeypatch.setattr(mini,'mini_g0',reader)
    s14.frontier_reads(payload)
    assert calls[0]['read_batch']==2 and calls[0]['endpoint_parity']
    assert calls[0]['items'][0]['reads']==['ce']
    assert calls[1]['read_batch']==8 and calls[1]['endpoint_parity_read'] is False
    assert calls[-1]['items'][0]['reads']==item['reads']
    assert calls[-1]['read_batch']==(2 if failure else 8)
    check=json.loads((out/'batch_check.json').read_text())['frontier']
    assert check['status']==('FALLBACK' if failure else 'PASS')
    for name in item['reads']:
        with np.load(out/f'g0_read_{name}.npz') as z:assert np.all(z['M_doc']==0)


def test_batch_two_is_identical_execution(monkeypatch):
    payload=dict(read_batch=2,items=[dict(reads=['ce','eta200'])])
    calls=[]
    result=dict(done=1,skipped=0,errors=0)
    monkeypatch.setattr(mini,'mini_g0',lambda cfg:(calls.append(cfg),result)[1])
    assert s14.frontier_reads(payload) is result
    assert calls==[payload] and calls[0] is payload


@pytest.mark.parametrize('failing_stage',['_tokenizer','_host_rows','candidates'])
def test_deep_passkey_preflight_failure_stops_before_training(tmp_path,monkeypatch,failing_stage):
    from test_s14 import _cfg, _fake_worker
    cfg=_cfg(tmp_path);cfg['passkey']=True
    def failure(*a,**k):raise ValueError('mocked tokenizer/host failure')
    monkeypatch.setattr(passkey,'_tokenizer',lambda *a:object())
    monkeypatch.setattr(passkey,'_host_rows',lambda *a:(np.zeros((256,1)),None,None))
    monkeypatch.setattr(passkey,'candidates',lambda *a:None)
    monkeypatch.setattr(passkey,failing_stage,failure)
    calls=[]
    stub=_fake_worker(calls)
    def worker(phase,payload,*args):
        if phase=='passkey':
            calls.append(phase)
            return s14.passkey_hook(payload)
        return stub(phase,payload,*args)
    monkeypatch.setattr(s14,'execute_worker',worker)
    with pytest.raises(ValueError,match='mocked'):s14.pod_job(cfg)
    assert calls==['data','passkey']
    assert json.loads((Path(cfg['out_dir'])/'FAILED.json').read_text())['phase']=='passkey_preflight'


def test_manifest_checks_only_read_files_and_corruption_is_local(tmp_path):
    a=tmp_path/'a.json';a.write_text('{}')
    b=tmp_path/'b.json';b.write_text('{}')
    from tttmem.p2.s14_evidence import sha256
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=[dict(path=p.name,sha256=sha256(p)) for p in (a,b)])))
    integrity=PullIntegrity(tmp_path)
    b.write_text('changed')
    assert integrity.verify(a)==a
    with pytest.raises(ValueError,match='hash mismatch'):integrity.verify(b)


def test_production_manifest_corruption_makes_only_that_cell_unreadable(tmp_path):
    from test_s14_analysis import synthetic_store
    from tttmem.p2.s14_evidence import sha256
    synthetic_store(tmp_path,[3],arms=['so','fo-ce'],passkeys=False)
    files=[dict(path=str(p.relative_to(tmp_path)),sha256=sha256(p)) for p in tmp_path.rglob('*') if p.is_file()]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=files)))
    store=s14_analysis.ReadStore(tmp_path)
    # The fixture has no marker, which a production manifest requires.
    with pytest.raises((FileNotFoundError,ValueError)):store.curve('so',3)
    target=tmp_path/'runs/mini-m/so/s3/compute.json'
    target.write_text('{}')
    from tttmem.p2.s14_checks import training_check
    assert training_check(tmp_path,'so',3,verify=store.verify)['status']=='NOT-EVALUABLE'
    assert training_check(tmp_path,'fo-ce',3,verify=store.verify)['status']=='OK'


def production_evidence(root,batch=8):
    from test_s14_analysis import synthetic_store
    from tttmem.p2.s14_evidence import sha256, fingerprint
    synthetic_store(root,[3],arms=['so'],passkeys=False)
    directory=root/'reads/mini-m/so/s3'
    cell=root/'.cell-configs/mini-m-so-s3.json';cell.parent.mkdir()
    base=dict(steps=1200)
    cell.write_text(json.dumps(base))
    ckpt=root/'runs/mini-m/so/s3/model_step1200.pt';ckpt.write_bytes(b'checkpoint')
    for path in directory.glob('*.npz'):
        with np.load(path) as z:values={k:z[k] for k in z.files}
        values['ckpt_sha256']=np.asarray(sha256(ckpt))
        np.savez(path,**values)
    item=dict(ckpt=str(ckpt),cell_config=str(cell),overrides=base,step=1200,init_seed=3,
              cell='mini-m/so/s3',out=str(directory),reads=s14.read_plan('so'))
    names=[f'g0_read_{n}.npz' for n in item['reads']]+['g0_hlm.npz']
    config=dict(item=item,read_batch=batch,passkey_batch=1,reads=names,passkey_docs=256,passkey_seed=1014)
    marker=dict(config=config,config_sha256=fingerprint(config),cell_config_sha256=sha256(cell),
                checkpoint_sha256=sha256(ckpt),complete=True,read_list=names,
                npz_sha256={n:sha256(directory/n) for n in names})
    if batch>2:
        check=dict(status='PASS',requested_batch=batch,effective_batch=batch,reference_batch=2,
                   checkpoint_sha256=sha256(ckpt),cell_config_sha256=sha256(cell),
                   n_docs=1024,n_ho=128,n_hlm=256,limits=s14_batch.FRONTIER_LIMITS,
                   values=dict(mean_M=0.,mean_L=0.,H_LM=0.,kappa_top1_agreement=1.,max_abs_M_doc=0.),
                   max_abs_M_doc_flag=False)
        s14_batch.save_check(directory,'frontier',check)
        marker.update(batch_check=dict(frontier=check),batch_check_sha256=sha256(directory/'batch_check.json'))
    s14._json(directory/'READS_DONE.json',marker)
    s14.publish_manifest(root,dict(cells=[dict(armdir='so',seed=3)]))
    return directory,cell,ckpt


@pytest.mark.parametrize('corruption',['npz','checkpoint','cell_config','config_hash','npz_hash','batch_missing','batch_mismatch'])
def test_production_checkpoint_config_and_batch_bindings(tmp_path,corruption):
    directory,cell,ckpt=production_evidence(tmp_path)
    store=s14_analysis.ReadStore(tmp_path)
    assert store.evidence('so',3)['batch_check']['status']=='PASS'
    if corruption=='npz':(directory/'g0_hlm.npz').write_bytes(b'changed')
    elif corruption=='checkpoint':ckpt.write_bytes(b'changed checkpoint')
    elif corruption=='cell_config':cell.write_text(json.dumps(dict(steps=1201)))
    elif corruption=='batch_missing':(directory/'batch_check.json').unlink()
    elif corruption=='batch_mismatch':
        check=json.loads((directory/'batch_check.json').read_text());check['frontier']['effective_batch']=2
        s14._json(directory/'batch_check.json',check)
    else:
        marker=json.loads((directory/'READS_DONE.json').read_text())
        if corruption=='config_hash':marker['config_sha256']='0'*64
        else:marker['npz_sha256']['g0_hlm.npz']='0'*64
        s14._json(directory/'READS_DONE.json',marker)
    # Refresh the external envelope: internal checkpoint/config/check bindings
    # must still catch corruption, independently of the pull transport hashes.
    s14.publish_manifest(tmp_path,dict(cells=[dict(armdir='so',seed=3)]))
    store=s14_analysis.ReadStore(tmp_path)
    with pytest.raises((ValueError,OSError)):store.evidence('so',3)
    assert s14_analysis._hlm(store,'so',3)['status']=='NOT-EVALUABLE'


def test_batch_check_max_per_doc_is_flag_only(tmp_path):
    directory,_,_=production_evidence(tmp_path)
    marker=json.loads((directory/'READS_DONE.json').read_text())
    check=marker['batch_check']['frontier']
    check['values']['max_abs_M_doc']=.1;check['max_abs_M_doc_flag']=True
    s14_batch.save_check(directory,'frontier',check)
    marker['batch_check_sha256']=s14.sha256_file(directory/'batch_check.json')
    s14._json(directory/'READS_DONE.json',marker)
    s14.publish_manifest(tmp_path,dict(cells=[dict(armdir='so',seed=3)]))
    assert s14_analysis.ReadStore(tmp_path).evidence('so',3)['batch_check']['status']=='PASS'


@pytest.mark.parametrize('failure',[None,'rank','lift'])
def test_passkey_bounds_apply_at_each_position(failure):
    ref={p:[dict(rank=1,lift=0.) for _ in range(32)] for p in ('in_window','beyond')}
    cand=deepcopy(ref)
    if failure=='rank':cand['in_window'][0]['rank']=2  # 31/32 < .97
    if failure=='lift':
        for doc in cand['beyond']:doc['lift']=.0011
    check=s14_batch.passkey_comparison(ref,cand)
    assert check['status']==('PASS' if failure is None else 'FALLBACK')


def test_pilot_rows_carry_only_measured_timing(tmp_path,monkeypatch):
    from test_s14 import _cfg
    cfg=_cfg(tmp_path);cfg['tiny_overrides']=dict(global_batch=16)
    root=Path(cfg['out_dir']);root.mkdir()
    monkeypatch.setattr(s14,'warm_page_cache',lambda *a:dict(test_only=True))
    def worker(phase,payload,*args):
        return dict(final=dict(step_ms=[10.]*60),peak_gb=1.,compute=dict(
            t_so_pilot_s=s14.N3B_T_SO,t_arm_pilot_s=s14.N3B_T_FO,pilot_source='n3b-50',
            device_name='cpu'))
    monkeypatch.setattr(s14,'execute_worker',worker)
    selected=s14.timing_pilot(cfg,root,0.)
    assert selected['common_timing']['fo_micro_batch']==4
    assert selected['common_timing']['rho_s14']==pytest.approx(3469/1200)
    for row in selected['pilot']['settings']:
        assert not {'t_so_pilot_s','t_arm_pilot_s','pilot_source'} & row.keys()
        assert row['median_ms']==10.
