"""P0.1: CPU/fp32 evaluation of immutable echo checkpoints (no training).

The shared backbone edge mask and streaming hooks live in exp_ttt. This module
provides the copy competitor and titrated token loss for subsequent experiments.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import resource
import sys
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as t
from .exp_ttt import copy_edge_mask

ALPHAS = (0., .5, .9, 1.)
CLASSES = ("kv", "W_marker", "other")
CKPT = re.compile(r"echo__s(\d+)__(.+)__eta([\d.]+)__st(\d+)(?:__step(\d+))?\.pt$")


def gather(x, positions):
    pos = torch.as_tensor(positions, device=x.device, dtype=torch.long)
    index = pos.reshape(*pos.shape, *([1] * (x.ndim - 2))).expand(*pos.shape, *x.shape[2:])
    return x.gather(1, index)


@torch.no_grad()
def copy_competitor(model, idx, write_pos, h=None):
    """Exact vocabulary-vector counterfactual at W0, detached for titration."""
    h = model.backbone(idx) if h is None else h
    off = model.backbone(idx, edge_mask=copy_edge_mask(write_pos, idx.shape[1], idx.device))
    on_z = model.read(model.fast_block(h, idx, model.slow()))
    off_z = model.read(model.fast_block(off, idx, model.slow()))
    return on_z - off_z, on_z, off_z


def titrated_loss(logits, targets, kv, competitor, alpha):
    """CE everywhere, CE(z-alpha*sg(c)) at kv. alpha=0 uses the exact CE path."""
    ce = t.token_ce(logits, targets)
    if alpha == 0:
        return ce
    return torch.where(kv, t.token_ce(logits - alpha * competitor.detach(), targets), ce)


def chunk0_transform(kv, competitor, alpha=0., positive=False):
    def transform(c, logits, targets, original):
        if c != 0:
            return original
        if positive:
            return t.token_loss(kv, t.lin_token(logits, targets), original)
        if alpha == 0:
            return original
        return titrated_loss(logits, targets, kv, competitor, alpha)
    return transform


def masks_for(idx, meta):
    kv = t.lin_selector("e2e1emb_linkv", meta["write_pos"], len(idx), idx.shape[1], idx.device)
    marker = idx == 0
    assert not bool((kv & marker).any())
    return dict(kv=kv, W_marker=marker, other=~(kv | marker))


def float64_copy(model):
    """Double-precision copy whose slow fast-weights and rho alone require grad."""
    m = copy.deepcopy(model).double()
    for prm in m.parameters():
        prm.requires_grad_(False)
    for prm in (*m.slow(), m.rho):
        prm.requires_grad_(True)
    return m


def eta_gradient_split(model, idx, y, meta, eta_base, g_clip, chunk, h=None, arm="e2e1emb"):
    """Exact first-order outer CE gradient, credited to ALL chunks' write sites.

    Zero-valued gates on detached class gradients preserve the forward recurrence
    bit-for-bit. Their derivatives are the corresponding log-eta_W2 contributions.
    The original learned rho remains attached and supplies the independent total.
    No derivative of the scalar clip or of any inner gradient is retained.
    """
    h = model.backbone(idx).detach() if h is None else h.detach()
    classes = masks_for(idx, meta)
    sel = t.lin_selector(arm,meta["write_pos"],len(idx),idx.shape[1],idx.device)
    keep = t.outer_keep(meta,tuple(y.shape),t.MASK_ARMS.get(arm,"none"),None,idx.device)
    gates = torch.zeros(3, requires_grad=True, device=idx.device, dtype=h.dtype)

    def tag(c, ce, weights, raw, clipped):
        sl = slice(c * chunk, (c + 1) * chunk)
        _, norm, _ = t._clip_fast(raw, g_clip)
        scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
        tagged = list(clipped)
        for j, name in enumerate(CLASSES):
            class_loss = (ce * classes[name][:, sl]).mean(1).sum()
            part = torch.autograd.grad(class_loss, weights[2], retain_graph=True)[0].detach()
            tagged[2] = tagged[2] + torch.expm1(gates[j]) * part * scale[:, None, None]
        return tagged

    loss = t.sequential_inner(model, h, y, model.per_sequence(len(idx)), model.etas(eta_base),
                              g_clip, False, t.GradStats(), chunk, idx=idx,
                              sel=sel,keep=keep,gradient_transform=tag)
    rho_grad, parts = torch.autograd.grad(loss, (model.rho, gates))
    total = float(rho_grad[2])
    parts = parts.detach().double().cpu().numpy()
    rel = abs(float(parts.sum()) - total) / max(abs(total), 1e-12)
    assert rel <= 1e-5, f"eta split relative error {rel}: {parts} vs {total}"
    return dict(outer_ce=float(loss.detach()), total=total,
                **{name: float(v) for name, v in zip(CLASSES, parts)}, relative_error=rel)


def write_sizes(model, h, idx, y, kv, competitor, etas, g_clip, alpha):
    """Token W2 Frobenius write sizes, with the actual full-chunk scalar clip."""
    with torch.enable_grad():
        weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
        out = model.fast_block(h, idx, weights)
        z = model.read(out)
        losses = titrated_loss(z, y, kv, competitor, alpha)
        grads = torch.autograd.grad(losses.mean(1).sum(), weights, retain_graph=True)
        _, norm, _ = t._clip_fast(list(grads), g_clip)
        scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
        delta = torch.autograd.grad(losses.sum(), out, retain_graph=True)[0].detach()
        kv_grad = torch.autograd.grad((losses * kv).mean(1).sum(), weights[2])[0].detach()
    with torch.no_grad():
        code = F.gelu(F.linear(model.fast_input(h, idx), model.slow()[0], model.slow()[1]))
        size = (etas[2].abs() * scale[:, None] * delta.norm(dim=-1) * code.norm(dim=-1) / y.shape[1])
        p = (-(t.token_ce(z.detach() - alpha * competitor.detach(), y))).exp()
        return dict(size=size[kv].cpu().numpy(), error=(1-p)[kv].cpu().numpy(),
                    aggregate=(etas[2] * scale[:, None, None] * kv_grad).flatten(1).norm(dim=1).cpu().numpy())


def jacobian_probe(model, out, targets):
    with torch.enable_grad():
        leaf = out.detach().requires_grad_(True)
        e = torch.full((*targets.shape, model.n_vocab), 1/model.n_vocab, device=out.device)
        e.scatter_(-1, targets[..., None], 1/model.n_vocab - 1)
        u = F.linear(e, model.unembed.weight.detach().T)
        jt = torch.autograd.grad(model.ln_f(leaf), leaf, grad_outputs=u)[0]
    return dict(cosine=F.cosine_similarity(jt, u, dim=-1).detach().cpu().numpy(),
                norm_ratio=(jt.norm(dim=-1)/u.norm(dim=-1).clamp_min(1e-30)).detach().cpu().numpy())


def reassign_set(n, T, window, n_layers, n_keys, chunk, seed):
    """Paired conditions. Insert a full second chunk into exp_ttt.echo_batch.

    T=512 fits: 128 write tokens + 268 filler + at most 108 query tokens.
    The single-write control retains the same filler gap and query positions.
    """
    receptive = n_layers * (window - 1) + 1
    if chunk != 4*n_keys:
        raise ValueError("reassign requires one full write phase per chunk")
    minimum = 2*chunk + n_layers*(window-1)+n_keys + 3*n_keys+4*(n_keys-1)+1
    valid_T = max(T, math.ceil(minimum/chunk)*chunk)
    rng = np.random.default_rng([seed, 7])
    raw, meta = t.echo_batch(rng, n, valid_T-chunk, window, "echo", n_keys, n_layers)
    vocab = t.EchoVocab(n_keys)
    old = meta["query_tok"].copy()
    new = vocab.value[(old-vocab.value[0]+rng.integers(1,n_keys,size=old.shape)) % n_keys]
    middle = rng.choice(vocab.filler, size=(n, chunk))
    single = np.concatenate((raw[:, :chunk], middle, raw[:, chunk:]), axis=1)
    qp = meta["query_pos"] + chunk
    wp1 = meta["write_pos"]
    wp2 = np.empty_like(wp1)
    reassign, unchanged = single.copy(), single.copy()
    for i in range(n):
        for j,k in enumerate(rng.permutation(n_keys)):
            pos = chunk+4*j
            wp2[i,k] = pos+2
            reassign[i,pos:pos+4] = [vocab.write,new[i,k],vocab.key[k],new[i,k]]
            unchanged[i,pos:pos+4] = [vocab.write,old[i,k],vocab.key[k],old[i,k]]
        reassign[i,qp[i]+1] = new[i]
    assert np.all(new != old)
    assert np.all(qp+1 <= valid_T)
    assert np.all(wp1 < chunk) and np.all((wp2 >= chunk) & (wp2 < 2*chunk))
    assert int(qp.min())-int(wp2.max()+1) > receptive
    base = dict(query_pos=qp, old_tok=old, new_tok=new, write_start=np.zeros(n,dtype=np.int64))
    conditions = {
        "reassign": (reassign, {**base,"write_pos":np.concatenate((wp1,wp2),axis=1),"query_tok":new}),
        "removed": (single, {**base,"write_pos":wp1,"query_tok":old}),
        "unchanged": (unchanged, {**base,"write_pos":np.concatenate((wp1,wp2),axis=1),"query_tok":old})}
    return conditions, dict(T_requested=T,T=valid_T,receptive_field=receptive,
                            filler_tokens=n_layers*(window-1)+n_keys,
                            first_query=int(qp.min()),last_query=int(qp.max()))


def checkpoint_manifest(cfg):
    times = {}
    with open(cfg["loss_times"], newline="") as handle:
        for row in csv.DictReader(handle):
            times[row["arm"],int(row["seed"])] = int(row["t_lose"]) if row["t_lose"] else None
    entries = []
    for source in cfg["sources"]:
        root = Path(source["root"])
        if not root.is_dir():
            raise ValueError(f"checkpoint root must be an existing directory: {root}")
        for path in sorted(root.glob("echo__*.pt")):
            match = CKPT.fullmatch(path.name)
            if match is None:
                continue
            seed,arm,eta,final,step = match.groups()
            seed,final = int(seed),int(final)
            is_final = step is None
            step = int(step) if step is not None else final
            if arm not in source["arms"]:
                continue
            phase = None
            if source["name"] == "r42l" and arm in ("e2e1emb","e2e1emb_w600") and seed < 5:
                cutoff = 400 if arm == "e2e1emb" else 600
                # Dense 150..400 plus both saved w600 points through step 600.
                if step in range(150,401,25) or (arm == "e2e1emb_w600" and step in (500,600)):
                    lose = times.get((arm,seed))
                    if lose is not None and step <= cutoff:
                        phase = "post" if step >= lose else "pre"
            # R42L: dense titration, same-step anchor comparisons, finals for reassign;
            # linkv/lin: finals suffice. Other sources: all saved specified arms.
            if source["name"] == "r42l" and not (phase or is_final or
                                                  (arm == "e2e1emb" and step in (100,500,1000))):
                continue
            if arm == "e2e1emb_kvm" and not is_final:
                continue
            entries.append(dict(path=str(path),source=source["name"],arm=arm,seed=seed,
                                step=step,final=is_final,eta_base=float(eta),phase=phase,
                                residual=arm in ("e2e2","e2e2emb","e2e1emb"),
                                eta_share=arm in ("e2e1","e2e1emb","e2e1emb_w600",
                                                 "e2e1emb_linkv","e2e1emb_lin","e2e1emb_kvm"),
                                reassign=is_final and arm in ("e2e2emb","e2e1emb_linkv",
                                            "e2e1emb_lin","e2e1emb","e2e1emb_kvm")))
    if not entries:
        raise ValueError("no checkpoints selected")
    return entries


def load_checkpoint(entry, cfg):
    state = torch.load(entry["path"], map_location="cpu", weights_only=True)
    # State tensors determine architecture, including unknown R4-3 arm names.
    d = state["tok.weight"].shape[1]
    layers = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
    st = t._settings(dict(d_model=d,n_layers=layers,n_heads=cfg["n_heads"],
                         backbone_mult=state["blocks.0.fc1.weight"].shape[0]//d,
                         fast_mult=state["head.W1"].shape[0]//d,window=cfg["window"],
                         chunk=cfg["chunk"],T=cfg["T"],n_keys=cfg["n_keys"]))
    task = t._task(st,"echo")
    arch = "e2e1emb" if "frozen_emb" in state else "e2e1"
    model = t.build_model(task,st,torch.device("cpu"),0,arch)
    if "key_permutation" in state:
        model.key_permutation = state["key_permutation"].clone()
    model.load_state_dict(state,strict=True)
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Required for the common training recurrence's inner derivatives and rho total.
    for parameter in (*model.slow(),model.rho):
        parameter.requires_grad_(True)
    assert next(model.parameters()).device.type == "cpu"
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert task.n_layers == cfg["n_layers"] and task.V == state["tok.weight"].shape[0]
    return model,task


def slice_meta(meta,lo,hi):
    return {key:value[lo:hi] if isinstance(value,np.ndarray) and value.ndim else value
            for key,value in meta.items()}


def batches(raw,meta,batch):
    for lo in range(0,len(raw),batch):
        hi = min(len(raw),lo+batch)
        yield (torch.from_numpy(raw[lo:hi,:-1]),torch.from_numpy(raw[lo:hi,1:]),slice_meta(meta,lo,hi))


def mean(parts):
    return float(np.concatenate([np.asarray(v).reshape(-1) for v in parts]).mean())


def correlation(x,y):
    x,y = np.concatenate(x).reshape(-1),np.concatenate(y).reshape(-1)
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def peak_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss/(1024**3 if sys.platform == "darwin" else 1024**2)


class Output:
    def __init__(self,path):
        self.path = path
        self.rows = {i:[] for i in range(1,8)}
        self.path.mkdir(parents=True,exist_ok=False)
        self.log = (path/"p01.jsonl").open("w")

    def emit(self,step,record):
        line = f"P01_{step} " + json.dumps(record,sort_keys=True,allow_nan=False)
        print(line,flush=True)
        self.log.write(line+"\n")
        self.log.flush()

    def save(self,step,identity,rows):
        self.rows[step].extend({**identity,**row} for row in rows)
        self.csv(f"p01-step{step}.csv",self.rows[step])

    def csv(self,name,rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (self.path/name).open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


class Recorder:
    """Collects Output calls in a pool worker; the parent replays them in manifest order."""
    def __init__(self):
        self.calls = []

    def emit(self,step,record):
        self.calls.append(("emit",(step,record)))

    def save(self,step,identity,rows):
        self.calls.append(("save",(step,identity,rows)))


_POOL = {}


def _pool_init(cfg,raw,meta,reassign):
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    t._mark_process(f"{os.environ.get('PROCESS_NAME',cfg['process_name'])}-w{os.getpid()}")
    _POOL.update(cfg=cfg,raw=raw,meta=meta,reassign=reassign)


def _pool_evaluate(entry):
    recorder = Recorder()
    timing = evaluate_checkpoint(entry,_POOL["cfg"],_POOL["raw"],_POOL["meta"],_POOL["reassign"],recorder)
    return timing,recorder.calls


def evaluate_all(entries,cfg,raw,meta,reassign,output):
    """Serial, or a spawn pool over checkpoints (Linux CPU host). Each checkpoint is independent and
    uses 4 threads either way; imap keeps manifest order, so outputs match the serial run."""
    workers = int(cfg.get("workers",1))
    if workers <= 1:
        return [evaluate_checkpoint(e,cfg,raw,meta,reassign,output) for e in entries]
    if sys.platform == "darwin":
        raise ValueError("P0.1 workers > 1 requires a Linux CPU host, not macOS")
    timings = []
    with mp.get_context("spawn").Pool(workers,initializer=_pool_init,
                                      initargs=(cfg,raw,meta,reassign)) as pool:
        for timing,calls in pool.imap(_pool_evaluate,entries,chunksize=1):
            for kind,args in calls:
                getattr(output,kind)(*args)
            timings.append(timing)
    return timings


def evaluate_checkpoint(entry,cfg,raw,meta,reassign,output):
    start = time.perf_counter()
    model,task = load_checkpoint(entry,cfg)
    profile = dict(load=time.perf_counter()-start,descriptive=0.,write_checks=0.,
                   titration=0.,eta_split=0.,reassign=0.)
    chunk,g_clip = cfg["chunk"],cfg["g_clip"]
    with torch.no_grad():
        etas = [v.detach() for v in model.etas(entry["eta_base"])]
    identity = {key:entry[key] for key in ("source","arm","seed","step","path")}
    collected = {i:[] for i in range(1,8)}
    alpha_parts = {str(a):dict(hit=[],ce=[],size=[],error=[],aggregate=[]) for a in ALPHAS}
    positive = dict(hit=[],ce=[])
    split_parts = []
    model64 = None
    for idx,y,bmeta in batches(raw,meta,cfg["eval_batch"]):
        tick = time.perf_counter()
        with torch.no_grad():
            h = model.backbone(idx).detach()
            competitor,on,off = copy_competitor(model,idx,bmeta["write_pos"],h)
            competitor,on,off = (z[:,:chunk] for z in (competitor,on,off))
            kv = t.lin_selector("e2e1emb_linkv",bmeta["write_pos"],len(idx),chunk,idx.device)
            for condition,z in (("full",on),("edge_off",off),("head_removed",model.read(h[:,:chunk]))):
                zv = z.gather(-1,y[:,:chunk,None]).squeeze(-1)
                wrong = z.scatter(-1,y[:,:chunk,None],-torch.inf).max(-1).values
                pv = (-t.token_ce(z,y[:,:chunk])).exp()
                collected[1].append(dict(condition=condition,margin=(zv-wrong)[kv].numpy(),p_v=pv[kv].numpy()))
            ct = competitor[kv]
            collected[2].append(dict(norm=ct.norm(dim=-1).numpy(),
                                    target=competitor.gather(-1,y[:,:chunk,None]).squeeze(-1)[kv].numpy()))
            if entry["residual"]:
                z = model.read(model.fast_block(h,idx,model.slow()))
                probs = z.softmax(-1)
                error = probs-F.one_hot(y,model.n_vocab)
                cm = masks_for(idx,bmeta)
                # Actual filler inputs, not the heterogeneous 'other' write class.
                cm["filler"] = idx >= 2+2*task.n_keys
                for name in ("kv","W_marker","filler"):
                    select = cm[name]
                    collected[4].append(dict(position_class=name,
                         p_v=probs.gather(-1,y[...,None]).squeeze(-1)[select].numpy(),
                         residual_norm=error.norm(dim=-1)[select].numpy()))
            out_kv = gather(model.fast_block(h[:,:chunk],idx[:,:chunk],model.slow()),bmeta["write_pos"])
            ykv = gather(y,bmeta["write_pos"])
        collected[6].append(jacobian_probe(model,out_kv,ykv))
        profile["descriptive"] += time.perf_counter()-tick
        if entry["phase"]:
            for a in ALPHAS:
                tick = time.perf_counter()
                size = write_sizes(model,h[:,:chunk],idx[:,:chunk],y[:,:chunk],kv,competitor,etas,g_clip,a)
                for key,value in size.items():
                    alpha_parts[str(a)][key].append(value)
                profile["write_checks"] += time.perf_counter()-tick
                if entry["phase"] == "post":
                    tick = time.perf_counter()
                    logits = t.eval_sequential(model,idx,y,etas,g_clip,chunk,t.GradStats(),h=h,
                                 inner_transform=chunk0_transform(kv,competitor,a))
                    q = gather(logits,bmeta["query_pos"])
                    target = torch.as_tensor(bmeta["query_tok"])
                    alpha_parts[str(a)]["hit"].append((q.argmax(-1)==target).numpy())
                    alpha_parts[str(a)]["ce"].append(t.token_ce(q,target).numpy())
                    profile["titration"] += time.perf_counter()-tick
            if entry["phase"] == "post":
                tick = time.perf_counter()
                logits = t.eval_sequential(model,idx,y,etas,g_clip,chunk,t.GradStats(),h=h,
                              inner_transform=chunk0_transform(kv,competitor,positive=True))
                q,target = gather(logits,bmeta["query_pos"]),torch.as_tensor(bmeta["query_tok"])
                positive["hit"].append((q.argmax(-1)==target).numpy())
                positive["ce"].append(t.token_ce(q,target).numpy())
                profile["titration"] += time.perf_counter()-tick
        if entry["eta_share"]:
            tick = time.perf_counter()
            # float64 copy: near-cancelling totals exceed the 1e-5 split check in
            # float32 (1.1e-5 on s0 e2e1emb step 1000); fp64 gives ~1e-13.
            if model64 is None:
                model64 = float64_copy(model)
            split_parts.append(eta_gradient_split(model64,idx,y,bmeta,entry["eta_base"],g_clip,chunk,
                                                  h.double(),entry["arm"]))
            profile["eta_split"] += time.perf_counter()-tick
        assert peak_rss_gb()<3, "P0.1 exceeded 3 GiB RSS"
    logit_rows = []
    for condition in ("full","edge_off","head_removed"):
        rows = [row for row in collected[1] if row["condition"] == condition]
        logit_rows.append(dict(condition=condition,margin=mean([r["margin"] for r in rows]),
                              p_v=mean([r["p_v"] for r in rows])))
    output.save(1,identity,logit_rows)
    output.emit(1,{**identity,"conditions":logit_rows})
    row2 = dict(competitor_norm=mean([r["norm"] for r in collected[2]]),
                target_competitor=mean([r["target"] for r in collected[2]]))
    output.save(2,identity,[row2]); output.emit(2,{**identity,**row2})
    titration = []
    if entry["phase"]:
        pooled_correlation = correlation(
            [v for a in ALPHAS for v in alpha_parts[str(a)]["size"]],
            [v for a in ALPHAS for v in alpha_parts[str(a)]["error"]])
        for a in ALPHAS:
            p = alpha_parts[str(a)]
            row = dict(phase=entry["phase"],condition="alpha",alpha=a,
                       token_W2_write=mean(p["size"]),one_minus_p=mean(p["error"]),
                       correlation=correlation(p["size"],p["error"]),
                       pooled_alpha_correlation=pooled_correlation,
                       aggregate_kv_W2_write=mean(p["aggregate"]))
            if p["hit"]:
                row.update(query_acc=mean(p["hit"]),query_ce=mean(p["ce"]))
            titration.append(row)
        if positive["hit"]:
            titration.append(dict(phase="post",condition="lin_kv",alpha=None,
                                  query_acc=mean(positive["hit"]),query_ce=mean(positive["ce"])))
    output.save(3,identity,titration or [dict(skipped="outside titration set")])
    output.emit(3,{**identity,"phase":entry["phase"],"conditions":titration})
    residual = []
    for name in ("kv","W_marker","filler"):
        rows = [r for r in collected[4] if r["position_class"]==name]
        if rows:
            residual.append(dict(position_class=name,p_v=mean([r["p_v"] for r in rows]),
                                 residual_norm=mean([r["residual_norm"] for r in rows])))
    output.save(4,identity,residual or [dict(skipped="outside residual set")])
    output.emit(4,{**identity,"positions":residual})
    split = {}
    if split_parts:
        # Weight by batch sequence count (the last batch may be shorter).
        weights = [min(cfg["eval_batch"],len(raw)-i*cfg["eval_batch"]) for i in range(len(split_parts))]
        split = {key:float(np.average([row[key] for row in split_parts],weights=weights))
                 for key in ("outer_ce","total",*CLASSES)}
        rel = abs(sum(split[k] for k in CLASSES)-split["total"])/max(abs(split["total"]),1e-12)
        assert rel <= 1e-5, f"aggregate eta split relative error {rel}"
        split.update(relative_error=rel,max_batch_relative_error=max(r["relative_error"] for r in split_parts),
                     W_marker_sign=int(np.sign(split["W_marker"])))
        abs_denom = sum(abs(split[k]) for k in CLASSES)
        for name in CLASSES:
            # part/total is unstable when the classes nearly cancel; the raw signed parts
            # and the absolute shares are primary.
            stable = abs_denom and abs(split["total"]) >= 0.05*abs_denom
            split[name+"_signed_share"] = split[name]/split["total"] if stable else None
            split[name+"_absolute_share"] = abs(split[name])/abs_denom if abs_denom else None
    output.save(5,identity,[split or dict(skipped="not a first-order arm")])
    output.emit(5,{**identity,**(split or dict(skipped="not a first-order arm"))})
    jac = {key:mean([r[key] for r in collected[6]]) for key in ("cosine","norm_ratio")}
    output.save(6,identity,[jac]); output.emit(6,{**identity,**jac})
    reassignment = []
    if entry["reassign"]:
        tick = time.perf_counter()
        for condition,(tokens,rmeta) in reassign.items():
            scores = {key:[] for key in ("new","old","ce")}
            for idx,y,bmeta in batches(tokens,rmeta,cfg["eval_batch"]):
                sel = t.lin_selector(entry["arm"],bmeta["write_pos"],len(idx),idx.shape[1],idx.device)
                with torch.no_grad():
                    logits = t.eval_sequential(model,idx,y,etas,g_clip,chunk,t.GradStats(),sel)
                    q = gather(logits,bmeta["query_pos"])
                    scores["new"].append((q.argmax(-1)==torch.as_tensor(bmeta["new_tok"])).numpy())
                    scores["old"].append((q.argmax(-1)==torch.as_tensor(bmeta["old_tok"])).numpy())
                    scores["ce"].append(t.token_ce(q,torch.as_tensor(bmeta["query_tok"])).numpy())
                assert peak_rss_gb()<3
            reassignment.append(dict(condition=condition,new_map_recall=mean(scores["new"]),
                        old_map_recall=mean(scores["old"]),
                        intrusion_f=mean(scores["old"]) if condition=="reassign" else None,
                        query_ce=mean(scores["ce"]),smoke_intermediate=cfg.get("smoke",False) and not entry["final"]))
        baseline = next(r["old_map_recall"] for r in reassignment if r["condition"]=="removed")
        for row in reassignment:
            row["baseline_recall"] = baseline
        profile["reassign"] += time.perf_counter()-tick
    output.save(7,identity,reassignment or [dict(skipped="not a selected final")])
    output.emit(7,{**identity,"conditions":reassignment})
    timing = dict(**identity,wall_s=time.perf_counter()-start,peak_rss_gb=peak_rss_gb(),
                  phase=entry["phase"],reassign=entry["reassign"],eta_share=entry["eta_share"],
                  **{name+"_s":value for name,value in profile.items()})
    output.emit("TIME",timing)
    del model
    gc.collect()
    return timing


def gate(rows,smoke=False):
    units = []
    for arm in ("e2e1emb","e2e1emb_w600"):
        for seed in range(5):
            selected = [r for r in rows if r.get("source")=="r42l" and r.get("phase")=="post"
                        and r["arm"]==arm and r["seed"]==seed]
            groups = {a:[r["query_acc"] for r in selected if r.get("condition")=="alpha" and r.get("alpha")==a]
                      for a in (0.,1.)}
            controls = [r["query_acc"] for r in selected if r.get("condition")=="lin_kv"]
            unit = dict(arm=arm,seed=seed,n_checkpoints=len(controls))
            if not controls:
                unit.update(status="inconclusive",reason="no saved post-loss checkpoints in specified range")
            else:
                control = float(np.mean(controls))
                a0,a1 = (float(np.mean(groups[a])) for a in (0.,1.))
                unit.update(acc0=a0,acc1=a1,positive_control_acc=control,
                            status="conclusive" if control>=.9 else "inconclusive",success=a1>a0,
                            prediction_reaches_90=a1>=.9)
            units.append(unit)
    valid = [u for u in units if u["status"]=="conclusive"]
    n,k = len(valid),sum(u["success"] for u in valid)
    p = sum(math.comb(n,j) for j in range(k,n+1))/2**n if n else None
    status = ("smoke_only" if smoke else "inconclusive" if 10-n>5 else
              "pass" if p is not None and p<.05 else "fail")
    return dict(status=status,n_units=10,n_conclusive=n,inconclusive=10-n,successes=k,
                one_sided_binomial_p=p,prediction_reaches_90=sum(u["prediction_reaches_90"] for u in valid),units=units)


def full_sizing(entries,timings,n_smoke):
    """Linear component extrapolation only; never evaluates another checkpoint."""
    counts = dict(descriptive=len(entries),write_checks=sum(bool(e["phase"]) for e in entries),
                  titration=sum(e["phase"]=="post" for e in entries),
                  eta_split=sum(e["eta_share"] for e in entries),reassign=sum(e["reassign"] for e in entries))
    rates = {}
    for key in ("load",*counts):
        values = [row[key+"_s"] for row in timings if row[key+"_s"]>0]
        rates[key] = float(np.mean(values)) if values else 0.
    predicted = rates["load"]*len(entries) + 256/n_smoke*sum(rates[k]*n for k,n in counts.items())
    return dict(n_checkpoints=len(entries),component_checkpoint_counts=counts,
                estimated_wall_s=predicted,planning_range_s=[predicted*.75,predicted*1.5],
                measured_peak_rss_gb=peak_rss_gb(),planning_peak_rss_gb=1.,
                method="component timings scaled 32->256 at unchanged batch 4; no full evaluation")


def run(cfg,seed):
    if cfg.get("device")!="cpu":
        raise ValueError("P0.1 is CPU only")
    if cfg["eval_batch"]>4:
        raise ValueError("P0.1 eval_batch is capped at 4")
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    t._mark_process(os.environ.get("PROCESS_NAME",cfg["process_name"]))
    output = Output(Path(cfg["out_dir"]))
    entries = checkpoint_manifest(cfg)
    full_entries = [dict(e) for e in entries]
    full_count = len(entries)
    if cfg.get("smoke"):
        requested = cfg["smoke_checkpoints"]
        entries = [next(e for e in entries if e["path"]==path) for path in requested]
        assert 2<=len(entries)<=3 and cfg["n_eval"]==32
        for e in entries:
            # Exercise reassign on the intermediate smoke anchors as well.
            e["reassign"] = True
    elif cfg["n_eval"]!=256:
        raise ValueError("full P0.1 evaluation is fixed at 256 sequences")
    # Generate the same full set first; smoke is its exact 32-sequence prefix.
    raw,meta = t.echo_batch(np.random.default_rng([cfg["eval_seed"],0]),256,cfg["T"],
                            cfg["window"],"echo",cfg["n_keys"],cfg["n_layers"])
    raw,meta = raw[:cfg["n_eval"]],slice_meta(meta,0,cfg["n_eval"])
    reassignment,layout = reassign_set(256,cfg["T"],cfg["window"],cfg["n_layers"],
                                     cfg["n_keys"],cfg["chunk"],cfg["eval_seed"])
    reassignment = {name:(r[:cfg["n_eval"]],slice_meta(m,0,cfg["n_eval"]))
                    for name,(r,m) in reassignment.items()}
    output.emit("CONFIG",dict(**cfg,eval_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                              n_checkpoints=len(entries),full_manifest_count=full_count,reassign_layout=layout,
                              torch_threads=torch.get_num_threads(),dtype="float32"))
    output.csv("p01-manifest.csv",entries)
    output.csv("p01-full-manifest.csv",full_entries)
    (output.path/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    timings = evaluate_all(entries,cfg,raw,meta,reassignment,output)
    # Match SO/anchor residuals by source, seed, step and position class.
    comparisons = []
    for row in output.rows[4]:
        if row["arm"] not in ("e2e2","e2e2emb") or "p_v" not in row:
            continue
        anchor = next((a for a in output.rows[4] if a["arm"]=="e2e1emb" and
                      all(a.get(k)==row.get(k) for k in ("source","seed","step","position_class"))),None)
        comparisons.append({**row,"anchor_available":anchor is not None,
                            "delta_p_v":row["p_v"]-anchor["p_v"] if anchor else None,
                            "delta_residual_norm":row["residual_norm"]-anchor["residual_norm"] if anchor else None})
    output.csv("p01-step4-comparisons.csv",comparisons)
    result = dict(smoke=cfg.get("smoke",False),n_checkpoints=len(entries),
                  gate=gate(output.rows[3],cfg.get("smoke",False)),
                  checkpoint_wall_s=[dict(source=x["source"],arm=x["arm"],seed=x["seed"],step=x["step"],wall_s=x["wall_s"]) for x in timings],
                  peak_rss_gb=peak_rss_gb(),out_dir=str(output.path),reassign_layout=layout)
    if cfg.get("smoke"):
        result["full_sizing"] = full_sizing(full_entries,timings,len(raw))
    output.csv("p01-timings.csv",timings)
    output.csv("p01-gate-units.csv",result["gate"]["units"])
    (output.path/"p01-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    compact = {**result,"gate":{k:v for k,v in result["gate"].items() if k!="units"}}
    if not cfg.get("smoke"):
        compact.pop("checkpoint_wall_s")
    output.emit("SUMMARY",compact)
    output.log.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/toy/p0-1.toml")
    parser.add_argument("--smoke",action="store_true",required=True,help="only lightweight direct checks are permitted")
    parser.add_argument("--process-name",required=True)
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    with open(args.config,"rb") as handle:
        config = tomllib.load(handle)
    cfg = {**config["p01"],"smoke":True,"n_eval":32,"process_name":args.process_name,"out_dir":args.out_dir,
           "workers":args.workers}
    os.environ["PROCESS_NAME"] = args.process_name
    run(cfg,int(config.get("seed",0)))


if __name__=="__main__":
    main()
