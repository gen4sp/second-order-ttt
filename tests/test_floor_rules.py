import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tttmem import exp_chr as c, exp_ttt as t, floor_rules as f

torch.set_num_threads(2)
t._mark_process(os.environ.get('PROCESS_NAME','TTM-rdo-72'))


@pytest.mark.parametrize('tag',['ce','rp10','nc03','rs01','ns01','rpf00','rpf100','ncf30'])
def test_parser(tag):
    assert f.parse_rule(tag).tag==tag


@pytest.mark.parametrize('tag',['rpf101','ncf-1','rpfoo','rpf','rpf30extra'])
def test_parser_reject(tag):
    with pytest.raises(ValueError): f.parse_rule(tag)


@pytest.mark.parametrize('target_logit',[1.,20.,100.])
@pytest.mark.parametrize('dose',[0.,.03,1.])
def test_floor_logit_gradient_including_zero_ce(target_logit,dose):
    z=torch.tensor([[[target_logit,0.,-1.]]],requires_grad=True)
    y=torch.zeros((1,1),dtype=torch.long)
    ce=t.token_ce(z,y)
    extra,_=f.rule_extra(f.FloorRule('rpf',dose),ce,[z],torch.ones_like(ce),
                          torch.zeros_like(ce),1,True,logits=z,y=y)
    base=torch.autograd.grad(ce.sum(),z)[0]
    r=-torch.expm1(-ce.detach())
    q=z.detach().scatter(-1,y[...,None],float('-inf')).softmax(-1)
    q.scatter_add_(-1,y[...,None],-torch.ones_like(y[...,None],dtype=z.dtype))
    expected=torch.maximum(r,r.new_tensor(dose))[...,None]*q
    assert torch.allclose(base+extra[0],expected,atol=1e-7,rtol=1e-5)
    if target_logit==100 and dose>0:
        assert ce.item()==0 and extra[0].abs().max()>0


def test_agrees_with_uncapped_rp_float64():
    z=torch.tensor([[[4.,0.,1.],[0.,2.,1.]]],dtype=torch.float64,requires_grad=True)
    y=torch.zeros((1,2),dtype=torch.long)
    ce=F.cross_entropy(z.flatten(0,1),y.flatten(),reduction='none').reshape(1,2)
    kv=torch.ones_like(ce)
    r=-torch.expm1(-ce.detach()); w=(.3/r).clamp_min(1)-1
    reference=torch.autograd.grad((w*ce).sum()/2,z,retain_graph=True)[0]
    new,_=f.rule_extra(f.FloorRule('rpf',.3),ce,[z],kv,kv*0,2,True,logits=z,y=y)
    assert torch.allclose(reference,new[0],atol=1e-14,rtol=1e-12)


def test_norm_match_zero_cases():
    extra=[torch.ones((3,2,2)),torch.ones((3,2)),torch.ones((3,2,2)),torch.ones((3,2))]
    control=[torch.ones_like(g)*2 for g in extra]
    control[0][1]=0
    for g in control: g[1]=0
    for g in extra: g[2]=0
    _,info=f.match_control(extra,control)
    assert torch.allclose(info['actual'][0],info['added'][0])
    assert info['match_fail'].tolist()==[False,True,False]
    assert info['actual'][2]==0


def tiny():
    st=t._settings(dict(d_model=16,n_layers=2,n_heads=2,backbone_mult=2,fast_mult=2,
                       window=8,chunk=16,n_keys=4,T=96,n_eval=2,eval_batch=2))
    task=t._task(st,'echo'); dev=torch.device('cpu')
    model=t.build_model(task,st,dev,0,'e2e1emb')
    data=t.eval_batches(task,*task.raw(np.random.default_rng(3),2),dev,2)
    return model,data,st


@pytest.mark.parametrize('tag',['ce','rp10','nc10','rs01','ns01'])
def test_chr_and_floor_rules_dispatch_bit_identical(tag):
    model,data,st=tiny(); idx,y,meta=data[0]; rule=f.parse_rule(tag)
    a=c.chr_stream(model,idx,y,meta['write_pos'],40.,10.,16,rule)[0]
    b=f.chr_stream(model,idx,y,meta['write_pos'],40.,10.,16,rule)[0]
    assert torch.equal(a,b)
    assert c.write_log(model,data,40.,10.,16,rule)==f.write_log(model,data,40.,10.,16,rule)


def test_floor_zero_and_anchor_before_cap():
    model,data,st=tiny(); idx,y,meta=data[0]
    def read(rule):return f.chr_stream(model,idx,y,meta['write_pos'],40.,10.,16,f.parse_rule(rule))[0]
    assert torch.equal(read('ce'),read('rpf00'))
    assert torch.allclose(read('rp10'),read('rpf10'),atol=2e-5,rtol=2e-5)


def test_active_floor_same_state_write_vector_matches_resolved_rp():
    model,data,st=tiny();vectors=[]
    rows=[]
    for tag in ('rp99','rpf99'):
        part=[]
        rows.append(f.write_log(model,data,40.,10.,16,f.parse_rule(tag),vector_out=part))
        vectors.append(torch.cat(part))
    assert rows[1]['floor_active']>.5 and rows[1]['cap_hit']==0
    relative=(vectors[0]-vectors[1]).norm(dim=1)/vectors[0].norm(dim=1).clamp_min(1e-12)
    assert float(relative.max())<1e-4


def test_write_log_kv_term_and_ncf_per_sequence_match():
    model,data,st=tiny()
    a=f.write_log(model,data,40.,10.,16,f.parse_rule('ce'))
    b=f.write_log(model,data,40.,10.,16,f.parse_rule('rpf100'))
    n=f.write_log(model,data,40.,10.,16,f.parse_rule('ncf100'))
    assert b['kv_write_med']>a['kv_write_med']
    assert b['floor_rel_error_max']<=1e-4 and b['floor_active']==1
    assert abs(n['attained_ratio']-1)<1e-4 and n['match_fail']==0
    assert n['r_eff_hypothetical']==1


def test_realized_floor_check_detects_construction_error(monkeypatch):
    model,data,st=tiny()
    original=f.complement_ce
    monkeypatch.setattr(f,'complement_ce',lambda z,y:.9*original(z,y))
    row=f.write_log(model,data,40.,10.,16,f.parse_rule('rpf100'))
    assert row['floor_active']==1 and row['floor_rel_error_max']>1e-4
