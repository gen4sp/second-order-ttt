"""Canary benchmarks and contract analysis; called only by explicit canary jobs."""
from dataclasses import replace, asdict
import json
from pathlib import Path
import time

import numpy as np

from .s14 import _json, read_name, passkey_hook


def analyze(cfg, root):
    from .s14_analysis import analyze_s14
    from .s14_test_contract import TestContract
    c=cfg['canary']
    contract=TestContract(test_only=True,preset=cfg['preset'],
        seeds=tuple(sorted(set(cell['seed'] for cell in cfg['cells']))),
        expected_steps={cell['armdir']:c.get('steps',30) for cell in cfg['cells']},
        guard_every=c.get('guard_every',10),n_docs=c.get('n_docs',8),
        n_ho_adapt=c.get('ho_docs',8),ho_docs=c.get('ho_docs',8),passkey_docs=cfg['passkey_docs'],
        grids={arm:c.get('read_etas',[1,2,4]) for arm in ('so','fo')},
        passkey_etas={arm:c.get('passkey_etas',[1,2]) for arm in ('so','fo')},
        passkey_hosts_file=c.get('test_hosts_file','') if cfg['preset']=='tiny' else '')
    _json(root/'test-contract.json',asdict(contract))
    result=analyze_s14(root,root/'analysis',test_contract=contract)
    if result['status']!='OK':raise RuntimeError('canary writer/reader contract failed')
    return dict(status='OK',contract_only=True,cells=len(result['cells']))


def benchmark(cfg, root):
    import torch
    from .data import repeat_classes
    from .evaluate import eval_docs
    from .meta import InnerCfg
    from .mini import _load_checkpoint, _g0_ho_adapt
    from .train import TrainCfg, prepare_eval_setup
    torch.set_num_threads(4)
    c=cfg['canary'];seed=int(cfg['cells'][0]['seed']);preset=cfg['preset']
    config=root/'.cell-configs'/f'{preset}-so-s{seed}.json'
    base=json.loads(config.read_text())
    checkpoint=Path(base['out_dir'])/f'model_step{base["steps"]:04d}.pt'
    n=int(c.get('benchmark_docs',256));hn=int(c.get('benchmark_ho_docs',128))
    hln=int(c.get('benchmark_hlm_docs',hn))
    tc=replace(TrainCfg.from_dict(base),n_eval=n,ho_docs=max(hn,hln),init='',resume=False)
    model=_load_checkpoint(checkpoint,tc)
    setup=prepare_eval_setup(tc,model,np.load(Path(tc.data_dir)/'excluded_vocab.npy'),write_meta=False)
    if len(setup.eval_docs_arr)!=n or len(setup.ho_docs)!=max(hn,hln):
        raise RuntimeError('canary benchmark cannot cover requested rows')
    docs=np.asarray(setup.eval_docs_arr);ho=np.asarray(setup.ho_docs[:hn])
    hlm_docs=np.asarray(setup.ho_docs[:hln])
    classes=repeat_classes(docs,tc.segment_len)
    cuda=torch.device(tc.device).type=='cuda'

    def sync():
        if cuda:torch.cuda.synchronize()

    def peak():
        if cuda:return torch.cuda.max_memory_allocated()/2**30
        from .passkey import _peak_gb
        return _peak_gb()

    result=dict(status='OK',measured=True,device=tc.device,torch_version=torch.__version__,
                checkpoint=str(checkpoint),n_docs=n,read=[],passkey=[],
                tolerance=1e-5,production_batches='set after canary',n_ho_adapt=hn,n_hlm=hln,
                warmup='one full row batch per read setting; eta1 passkey warmup before each setting')
    baseline=None
    started=time.monotonic()
    try:
        for batch in c.get('read_batches',[2,4,8,16]):
            warm_n=min(batch,n)
            began=time.perf_counter()
            eval_docs(model,docs[:warm_n],setup.probes[:warm_n],'ce',modes=('fP_focal','fP_nonfocal'),
                classes=classes[:warm_n],batch=batch,inner_lr=tc.rule_lr('ce'),inner_clip=tc.inner_clip,
                device=tc.device,fused_attn=tc.fused_attn,barriers=tc.barriers,r_ref=tc.r_ref,
                data_seed=tc.data_seed,inner_extra=dict(tok_mult='none',sigma_cap=4.))
            sync();warmup_s=time.perf_counter()-began
            sync()
            if cuda:torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter()
            r=eval_docs(model,docs,setup.probes,'ce',modes=('fP_focal','fP_nonfocal'),
                classes=classes,batch=batch,inner_lr=tc.rule_lr('ce'),inner_clip=tc.inner_clip,
                device=tc.device,fused_attn=tc.fused_attn,barriers=tc.barriers,r_ref=tc.r_ref,
                data_seed=tc.data_seed,inner_extra=dict(tok_mult='none',sigma_cap=4.))
            m=np.asarray(r['doc']['fP_focal']-r['doc']['fP_nonfocal'],float)
            l=_g0_ho_adapt(model,ho,InnerCfg(rule='ce',lr=tc.rule_lr('ce'),clip=tc.inner_clip,
                             seed=tc.data_seed),tc,batch)
            ld=l[:,0]/l[:,1]
            rh=eval_docs(model,hlm_docs,[None]*hln,'ce',modes=('heldout',),
                classes=repeat_classes(hlm_docs,tc.segment_len),batch=batch,
                inner_lr=tc.rule_lr('ce'),inner_clip=tc.inner_clip,device=tc.device,
                fused_attn=tc.fused_attn,barriers=tc.barriers,r_ref=tc.r_ref,data_seed=tc.data_seed)
            hd=np.asarray(rh['doc']['heldout_noadapt'],float)
            sync();seconds=time.perf_counter()-began
            if baseline is None:baseline=(m.copy(),ld.copy(),hd.copy())
            dm=float(np.max(np.abs(m-baseline[0])));dl=float(np.max(np.abs(ld-baseline[1])))
            dh=float(np.max(np.abs(hd-baseline[2])))
            if not np.isfinite([seconds,dm,dl,dh]).all():raise RuntimeError('nonfinite canary read benchmark')
            result['read'].append(dict(batch=batch,seconds=seconds,warmup_seconds=warmup_s,
                peak_gb=peak(),max_abs_delta_M_doc=dm,
                max_abs_delta_L=dl,max_abs_delta_H_LM=dh,within_tolerance=max(dm,dl,dh)<=1e-5))
        # Release the first model before passkey creates its checkpoint model.
        del model,r
        if cuda:torch.cuda.empty_cache()
        pbase=None
        for batch in c.get('passkey_batches',[1,8,16]):
            began=time.perf_counter()
            passkey_hook(dict(model_path=str(checkpoint),cell_config_path=str(config),eta_mults=[1],
                out_dir=str(root/'bench'/f'passkey-warmup-b{batch}'),n_docs=min(batch,cfg['passkey_docs']),
                seed=cfg.get('passkey_seed',1014),device=tc.device,data_dir=tc.data_dir,noadapt=False,batch=batch))
            sync();warmup_s=time.perf_counter()-began
            sync()
            if cuda:torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter();out=root/'bench'/f'passkey-b{batch}'
            pk=passkey_hook(dict(model_path=str(checkpoint),cell_config_path=str(config),
                eta_mults=c.get('passkey_etas',[1,2]),out_dir=str(out),n_docs=cfg['passkey_docs'],
                seed=cfg.get('passkey_seed',1014),device=tc.device,data_dir=tc.data_dir,noadapt=True,batch=batch))
            sync();seconds=time.perf_counter()-began
            lifts=[];ranks=[]
            for condition in pk['conditions']:
                with np.load(out/f'passkey_{condition}.npz') as z:
                    for pos in ('in_window','beyond'):
                        lifts.extend(z[pos+'_lift']);ranks.extend(z[pos+'_rank'])
            lifts=np.asarray(lifts);ranks=np.asarray(ranks)
            if pbase is None:pbase=(lifts.copy(),ranks.copy())
            delta=float(np.max(np.abs(lifts-pbase[0])))
            agreement=float(np.mean(ranks==pbase[1]))
            result['passkey'].append(dict(batch=batch,seconds=seconds,warmup_seconds=warmup_s,peak_gb=peak(),
                max_abs_delta_lift=delta,rank_agreement=agreement,leak_check=pk['leak_check'],
                within_tolerance=delta<=1e-5 and agreement==1.))
            if pk['leak_check']!='PASS':raise RuntimeError('canary passkey benchmark leak audit failed')
        result['wall_s']=time.monotonic()-started
        _json(root/'canary_bench.json',result)
        return dict(status='OK',wall_s=result['wall_s'],file=str(root/'canary_bench.json'))
    except BaseException as exc:
        result.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',wall_s=time.monotonic()-started)
        _json(root/'canary_bench.json',result)
        raise
