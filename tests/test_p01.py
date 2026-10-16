"""Tiny P0.1 CPU checks; no checkpoint sweep or child torch processes."""
import os

import numpy as np
import pytest
import torch

from tttmem import exp_p01 as p
from tttmem import exp_ttt as t

# Also governs test_r42l when collected in this one process.
torch.set_num_threads(4)
t._mark_process(os.environ.get("PROCESS_NAME","TTM-p01-60-tests"))


def fixture():
    st = t._settings(dict(d_model=16,n_layers=3,n_heads=2,backbone_mult=2,fast_mult=2,
                         window=8,chunk=16,n_keys=4,T=96,n_eval=2))
    task = t._task(st,"echo")
    idx,y,meta = task.tensors(*task.raw(np.random.default_rng(11),2),torch.device("cpu"))
    model = t.build_model(task,st,torch.device("cpu"),3,"e2e1emb")
    return model,task,idx,y,meta


def test_backbone_default_path_bit_identical():
    model,_,idx,_,_ = fixture()
    # Independent reference: the blocks applied directly, without the edge_mask argument.
    with torch.no_grad():
        original = model.tok(idx)
        rope,mask = model._rope(idx.shape[1],idx.device),model._mask(idx.shape[1],idx.device)
        for block in model.blocks:
            original = block(original,rope,mask)
        assert torch.equal(original,model.backbone(idx))
        assert torch.equal(original,model.backbone(idx,edge_mask=None))
        empty = torch.zeros(idx.shape[1],idx.shape[1],dtype=torch.bool)
        assert torch.equal(original,model.backbone(idx,edge_mask=empty))


@pytest.mark.parametrize("full_attention",[False,True])
def test_edge_mask_removes_source_influence_in_every_layer(full_attention):
    model,_,idx,_,meta = fixture()
    if full_attention:
        model.window = None
        model._mask_cache.clear()
    # Perturb just one source per sequence. Adjacent repeated tokens are untouched.
    wp = torch.as_tensor(meta["write_pos"][:, :1])
    altered = idx.clone()
    rows = torch.arange(len(idx))[:,None]
    altered[rows,wp-1] = (altered[rows,wp-1]+1)%model.n_vocab
    mask = p.copy_edge_mask(wp,idx.shape[1])
    with torch.no_grad():
        plain = p.gather(model.backbone(idx),wp)
        changed = p.gather(model.backbone(altered),wp)
        assert float((plain-changed).abs().max())>1e-7
        masked = p.gather(model.backbone(idx,edge_mask=mask),wp)
        masked_changed = p.gather(model.backbone(altered,edge_mask=mask),wp)
        assert torch.equal(masked,masked_changed)
        # Cache was not mutated by a counterfactual call.
        assert torch.equal(plain,p.gather(model.backbone(idx),wp))
        for shape in (mask[:,0],mask):
            assert torch.equal(model.backbone(idx,edge_mask=shape),model.backbone(idx,edge_mask=mask))


def test_alpha_zero_reproduces_chunk0_step_and_full_stream_exactly():
    model,task,idx,y,meta = fixture()
    with torch.no_grad():
        h = model.backbone(idx)
        competitor,_,_ = p.copy_competitor(model,idx[:,:task.chunk],meta["write_pos"],h[:,:task.chunk])
        etas = [e.detach() for e in model.etas(10.)]
    kv = t.lin_selector("e2e1emb_linkv",meta["write_pos"],len(idx),task.chunk,idx.device)
    writes = []
    for probe in (False,True):
        weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
        z = model.read(model.fast_block(h[:,:task.chunk],idx[:,:task.chunk],weights))
        loss = (p.titrated_loss(z,y[:,:task.chunk],kv,competitor,0.) if probe else
                t.token_ce(z,y[:,:task.chunk]))
        grads = torch.autograd.grad(loss.mean(1).sum(),weights)
        clipped,_,_ = t._clip_fast(list(grads),10.)
        writes.append([w-e*g for w,e,g in zip(weights,etas,clipped)])
    assert all(torch.equal(a,b) for a,b in zip(*writes))
    normal = t.eval_sequential(model,idx,y,etas,10.,task.chunk,t.GradStats(),h=h)
    alpha0 = t.eval_sequential(model,idx,y,etas,10.,task.chunk,t.GradStats(),h=h,
                             inner_transform=p.chunk0_transform(kv,competitor,0.))
    assert torch.equal(normal,alpha0)
    # A nonzero titration changes the write but never changes chunk-0 predictions.
    alpha1 = t.eval_sequential(model,idx,y,etas,10.,task.chunk,t.GradStats(),h=h,
                             inner_transform=p.chunk0_transform(kv,competitor,1.))
    assert torch.equal(normal[:,:task.chunk],alpha1[:,:task.chunk])
    assert not torch.equal(normal[:,task.chunk:],alpha1[:,task.chunk:])


@pytest.mark.parametrize("g_clip",[10.,.001])
@pytest.mark.parametrize("arm",["e2e1emb","e2e1emb_linkv","e2e1emb_lin","e2e1emb_kvm"])
def test_eta_split_sums_to_independent_normal_training_gradient(g_clip,arm):
    model,task,idx,y,meta = fixture()
    h = model.backbone(idx).detach()
    sel = t.lin_selector(arm,meta["write_pos"],len(idx),idx.shape[1],idx.device)
    keep = t.outer_keep(meta,tuple(y.shape),t.MASK_ARMS.get(arm,"none"),None,idx.device)
    ordinary = t.sequential_inner(model,h,y,model.per_sequence(len(idx)),model.etas(10.),
                                 g_clip,False,t.GradStats(),task.chunk,idx=idx,sel=sel,keep=keep)
    expected = float(torch.autograd.grad(ordinary,model.rho)[0][2])
    result = p.eta_gradient_split(model,idx,y,meta,10.,g_clip,task.chunk,h,arm)
    assert result["relative_error"]<=1e-5
    assert result["total"]==expected
    assert result["outer_ce"]==float(ordinary.detach())
    assert abs(sum(result[k] for k in p.CLASSES)-expected)<=1e-5*max(abs(expected),1e-12)


@pytest.mark.parametrize("emb",[True,False])
def test_unknown_arm_architecture_is_read_from_checkpoint_content(tmp_path,emb):
    model,task,idx,_,_ = fixture()
    if not emb:
        model.frozen_emb = None
    path = tmp_path/"unknown.pt"
    torch.save(model.state_dict(),path)
    cfg = dict(n_heads=2,n_layers=3,window=8,chunk=16,T=96,n_keys=4)
    restored,_ = p.load_checkpoint(dict(path=str(path),arm="unknown_arch"),cfg)
    assert (restored.frozen_emb is not None)==emb
    with torch.no_grad():
        assert torch.equal(model.backbone(idx),restored.backbone(idx))
        assert torch.equal(model.read(model.fast_block(model.backbone(idx),idx,model.slow())),
                           restored.read(restored.fast_block(restored.backbone(idx),idx,restored.slow())))


def test_reassign_layout_and_controls_are_paired_and_fit_512():
    conditions,layout = p.reassign_set(2,512,64,4,16,64,42)
    assert layout["T"]==512 and layout["filler_tokens"]>layout["receptive_field"]
    a,ma = conditions["reassign"]
    removed,mr = conditions["removed"]
    unchanged,mu = conditions["unchanged"]
    assert np.array_equal(a[:,:64],removed[:,:64])
    assert np.array_equal(a[:,:64],unchanged[:,:64])
    assert np.all(ma["new_tok"]!=ma["old_tok"])
    assert np.array_equal(ma["query_pos"],mr["query_pos"])
    assert np.array_equal(ma["query_pos"],mu["query_pos"])
    assert np.all(removed[:,64:128]>=34)
    for raw,meta in conditions.values():
        rows = np.arange(2)[:,None]
        assert np.array_equal(raw[rows,meta["query_pos"]+1],meta["query_tok"])
        wp = meta["write_pos"]
        assert np.array_equal(raw[rows,wp-1],raw[rows,wp+1])
        sel = t.lin_selector("e2e1emb_linkv",wp,2,512,torch.device("cpu"))
        assert np.all(sel.sum(1).numpy()==wp.shape[1])


def test_titration_detaches_competitor():
    z = torch.randn(2,4,7,requires_grad=True)
    c = torch.randn(2,4,7,requires_grad=True)
    y = torch.zeros(2,4,dtype=torch.long)
    kv = torch.tensor([[True,False,False,True]]*2)
    p.titrated_loss(z,y,kv,c,1.).sum().backward()
    assert c.grad is None and z.grad is not None
    expected = t.token_ce(z-c.detach(),y)
    assert torch.equal(p.titrated_loss(z,y,kv,c,1.)[kv],expected[kv])


def test_gate_unit_average_and_inconclusive_exclusion():
    rows = []
    for arm in ("e2e1emb","e2e1emb_w600"):
        for seed in range(5):
            for step in (300,350):
                for a in (0.,1.):
                    rows.append(dict(source="r42l",phase="post",arm=arm,seed=seed,step=step,
                                     condition="alpha",alpha=a,query_acc=.5 if a==0 else .95))
                rows.append(dict(source="r42l",phase="post",arm=arm,seed=seed,step=step,
                                 condition="lin_kv",query_acc=1.))
    result = p.gate(rows)
    assert result["status"]=="pass" and result["one_sided_binomial_p"]==1/1024
    assert len(result["units"])==10 and result["n_conclusive"]==10
    assert p.gate(rows,smoke=True)["status"]=="smoke_only"
    result = p.gate([r for r in rows if r["seed"]<2])
    assert result["status"]=="inconclusive" and result["inconclusive"]==6


def test_manifest_does_not_extend_anchor_seed2_past_registered_cutoff(tmp_path):
    times = tmp_path / "times.csv"
    times.write_text("arm,seed,t_lose\ne2e1emb,2,425\ne2e1emb,0,250\ne2e1emb_w600,0,375\n")
    for seed,arm,steps in ((2,"e2e1emb",(400,500)),(0,"e2e1emb",(200,300)),
                           (0,"e2e1emb_w600",(350,400,600,700))):
        for step in steps:
            (tmp_path/f"echo__s{seed}__{arm}__eta10__st1500__step{step}.pt").touch()
    cfg = dict(loss_times=str(times),sources=[dict(name="r42l",root=str(tmp_path),
                                                arms=["e2e1emb","e2e1emb_w600"])])
    entries = p.checkpoint_manifest(cfg)
    assert not any(e["arm"]=="e2e1emb" and e["seed"]==2 and e["phase"]=="post" for e in entries)
    for arm in ("e2e1emb","e2e1emb_w600"):
        post = [e for e in entries if e["phase"]=="post" and e["arm"]==arm]
        assert post and max(e["step"] for e in post)<= (400 if arm=="e2e1emb" else 600)
