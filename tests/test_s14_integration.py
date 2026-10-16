"""S14 writer-to-reader contracts, real tiny training, and batch invariance."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from tttmem.p2 import s14, s14_analysis, s14_checks, passkey
from tttmem.p2.s14_protocol import COMPUTE_FIELDS, N3B_FO_PILOT_S
from tttmem.p2.s14_test_contract import TestContract

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import s14_test_support as support
from s14_test_support import tiny_store


def test_real_pod_writers_all_six_arms_contract_and_batches(tmp_path,monkeypatch):
    torch.set_num_threads(4)
    data=tmp_path/'p2data'
    expected=tiny_store(data, quotas=dict(A=96,B=32,eval=12,ho=8,calib=2))
    hosts=tmp_path/'hosts.json'
    # Save and restore every test-only substitution, including PRESETS.
    from tttmem.p2.model import PRESETS
    original=PRESETS['tiny']
    monkeypatch.setitem(PRESETS,'tiny',original)
    monkeypatch.setattr(passkey,'HOSTS_FILE',hosts)
    monkeypatch.setattr(passkey,'_tokenizer',lambda *a:support.TinyDigitTokenizer())
    support.install(data,hosts)
    root=tmp_path/'s14'
    cfg=dict(preset='tiny',device='cpu',max_hours=1,process_name='TTM-s14-75-contract',
        process_nn=51,out_dir=str(root),data_dir=str(data),steps=4,n_p=4,
        p_micro_batch=4,p_so_micro_batch=1,t_so_pilot_s=.02,t_arm_pilot_s=.01,
        passkey=True,passkey_docs=4,tiny_overrides=support.overrides(),read_batch=2,
        data=dict(mode='uploaded',expected_hashes=str(expected)),
        cells=[dict(armdir=a,seed=3) for a in s14.ARMDIRS])
    # Real production worker/writers, run inline so no extra torch process is started.
    monkeypatch.setattr(s14,'execute_worker',lambda phase,payload,root,name,deadline:
                        s14.worker(phase,payload,name))
    assert s14.pod_job(cfg)['status']=='done'
    tc=support.contract(hosts)
    result=s14_analysis.analyze_s14(root,tmp_path/'analysis',test_contract=tc)
    assert result['status']=='OK',result
    for cell in result['cells']:
        assert cell['checks']['compute_v2']['status']=='PASS'
        assert cell['checks']['timing']['covered_steps']==4
        assert cell['checks']['guard']['probes']==2
        compute=json.loads((root/f"runs/tiny/{cell['arm']}/s3/compute.json").read_text())
        assert set(COMPUTE_FIELDS)<=compute.keys()
        assert 'RUNPOD_POD_ID' not in compute
        assert compute['so_micro_batch'] is None if cell['arm']!='mini-p' else compute['so_micro_batch']==1
    # Re-read the same checkpoint at 8 rows, with its real in-training parity gate.
    base=json.loads((root/'.cell-configs/tiny-so-s3.json').read_text())
    rc=s14.read_config({**cfg,'read_batch':8},base,'so',root)
    rc['items'][0]['out']=str(tmp_path/'batch8')
    s14.worker('reads',rc,'TTM-s14-75-batch8')
    for eta in s14.SO_GRID:
        filename=f'g0_read_{s14.read_name(eta)}.npz'
        with np.load(root/'reads/tiny/so/s3'/filename) as two, np.load(tmp_path/'batch8'/filename) as eight:
            for key in ('M_doc','L_adapt_sum','L_adapt_doc','kz_rank_f','kz_rank_nf'):
                if key in two:
                    np.testing.assert_allclose(two[key],eight[key],atol=1e-5,rtol=0)
    with np.load(root/'reads/tiny/so/s3/g0_hlm.npz') as two, np.load(tmp_path/'batch8/g0_hlm.npz') as eight:
        np.testing.assert_allclose(two['heldout_doc'],eight['heldout_doc'],atol=1e-5,rtol=0)
        np.testing.assert_array_equal(two['heldout_count'],eight['heldout_count'])
    for batch in (2,8):
        s14.passkey_hook(dict(model_path=str(Path(base['out_dir'])/'model_step0004.pt'),
            cell_config_path=str(root/'.cell-configs/tiny-so-s3.json'),eta_mults=[1,2],
            out_dir=str(tmp_path/f'passkey-b{batch}'),n_docs=8,device='cpu',data_dir=str(data),batch=batch))
    for condition in ('ce','eta200','noadapt'):
        with np.load(tmp_path/'passkey-b2'/f'passkey_{condition}.npz') as two, np.load(tmp_path/'passkey-b8'/f'passkey_{condition}.npz') as eight:
            for pos in ('in_window','beyond'):
                np.testing.assert_array_equal(two[pos+'_rank'],eight[pos+'_rank'])
                np.testing.assert_allclose(two[pos+'_lift'],eight[pos+'_lift'],atol=1e-5,rtol=0)
    # Real trainer resume: persisted non-log tail plus new timing must cover all steps.
    payload={**base,'out_dir':str(tmp_path/'resumed'),'resume_dir':str(tmp_path/'resumed'),
             'stop_after':2,'save_checkpoints':False,'keep_final_state':False,
             '_compute_context':s14.compute_context(cfg,dict(armdir='so'))}
    s14.worker('train',dict(payload),'TTM-s14-75-tail')
    payload.update(stop_after=None,resume=True)
    final=s14.worker('train',dict(payload),'TTM-s14-75-resume')
    records=s14_checks._records(tmp_path/'resumed/events.jsonl')
    assert s14_checks.timing_check(records,4,test_only=True)['status']=='OK'
    assert final['final']['steps']==4
    # Real recovered G-div writer, replayed timing, and C2 FLAG adjudication.
    from test_soq_n3b import _fake_probe
    _fake_probe(monkeypatch,[4.,float('nan'),4.])
    gcfg={**cfg,'out_dir':str(tmp_path/'gdiv'), 'cells':[dict(armdir='so',seed=3)]}
    assert s14.pod_job(gcfg)['status']=='done'
    evidence=json.loads((tmp_path/'gdiv/runs/tiny/so/s3/gdiv.json').read_text())
    assert len(evidence)==1 and set(evidence[0])=={'event','step','loss','grad_norm'}
    checked=s14_checks.training_check(tmp_path/'gdiv','so',3,test_contract=tc)
    assert checked['status']=='OK' and checked['gdiv']['status']=='FLAG',checked
    # Execute the canary benchmark code itself on the same tiny real checkpoint.
    from tttmem.p2.s14_canary import benchmark
    bcfg={**cfg,'canary':dict(benchmark_docs=8,benchmark_ho_docs=8,
        read_batches=[2,4,8,16],passkey_batches=[1,8,16],passkey_etas=[1,2])}
    assert benchmark(bcfg,root)['status']=='OK'
    bench=json.loads((root/'canary_bench.json').read_text())
    assert [v['batch'] for v in bench['read']]==[2,4,8,16]
    assert all(v['max_abs_delta_M_doc']<=1e-5 and v['max_abs_delta_L']<=1e-5 for v in bench['read'])
    assert all(v['rank_agreement']==1 for v in bench['passkey'])
    # Corruption must fail the exact same contract analysis.
    marker=root/'reads/tiny/so/s3/READS_DONE.json'
    record=json.loads(marker.read_text());record['npz_sha256']['g0_hlm.npz']='0'*64
    marker.write_text(json.dumps(record))
    failed=s14_analysis.analyze_s14(root,tmp_path/'bad',test_contract=tc)
    assert failed['status']=='FAIL'
    assert failed['cells'][0]['checks']['READS_DONE']['status']=='NOT-EVALUABLE'


def test_override_requires_explicit_test_only():
    with pytest.raises(ValueError):TestContract(test_only=False,expected_steps={'so':4})
    with pytest.raises(TypeError):s14_analysis.analyze_s14('.', '.',test_contract={})


def test_compute_context_uses_actual_mini_p_pilot():
    selected=dict(n_p=2800,t_so_ms=6607.372670434415,t_p_ms=2831.731144471892,
                  p_micro_batch=4,p_so_micro_batch=2)
    context=s14.compute_context(dict(preset='mini-m'),dict(armdir='mini-p'),selected)
    assert context['pilot_source']=='s14-pilot'
    assert context['t_arm_pilot_s']==selected['t_p_ms']/1000
    assert context['selected_pilot_micro_batch']==4 and context['selected_pilot_so_micro_batch']==2
    assert s14.compute_context({},dict(armdir='fo-cm'))['t_arm_pilot_s']==N3B_FO_PILOT_S[4]


def test_canary_resolves_six_real_mini_m_recipes_without_pilot(tmp_path,monkeypatch):
    import tomllib
    repo=Path(__file__).resolve().parents[1]
    cfg=tomllib.loads((repo/'configs/s14/canary.toml').read_text())['p2']
    class Rows:
        def __len__(self):return 25600
    monkeypatch.setattr(s14,'open_split',lambda *a:(Rows(),[]))
    selected=dict(n_p=30,p_micro_batch=4,p_so_micro_batch=1,
                  t_so_ms=s14.N3B_T_SO*1000,t_p_ms=s14.N3B_T_FO*1000)
    assert cfg['process_name']=='TTM-s14-63' and round(cfg['max_hours']*3600)<=2400
    assert cfg['data']['mode']=='uploaded' and not cfg['data']['rebuild_c']
    for cell in cfg['cells']:
        base=s14.cell_config(cfg,cell,tmp_path,selected)
        assert base['steps']==30 and base['preset']=='mini-m' and base['guard_every']==10
        read=s14.read_config(cfg,base,cell['armdir'],tmp_path)
        assert read['items'][0]['reads']==['ce','eta200','eta400']
        if cell['armdir'].startswith('wsd'):
            from tttmem.p2 import matrix
            assert matrix._base_config(read,read['items'][0],repo)==base
        assert s14.passkey_etas_for(cfg,cell['armdir'])==[1,2]
