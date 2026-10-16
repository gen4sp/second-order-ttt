"""Importable, non-mutating TRG readers, telemetry and hashed fifty-step shards."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_chr as chr
from . import exp_ttt as ttt

SCHEMA = dict(version="trg-dense-v2", shard_steps=50, event_float="float16", flags="int8",
              margin="z_v-logsumexp(z_wrong)",
              rungs=["L0", "L1", "L1_without_e", "L2", "L3", "L4"],
              terms=["slow", "own", "other_kv", "rest_chunk0", "chunks1_4", "chunk5"],
              summary_float="float64", write_norm="joint |eta*c0*g_event|", descriptive=True)
SCHEMA_SHA256 = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def margin(logits, targets):
    correct = logits.gather(-1, targets[..., None]).squeeze(-1)
    wrong = logits.scatter(-1, targets[..., None], -float("inf"))
    return correct-torch.logsumexp(wrong, dim=-1)


def gather(values, positions):
    return values.gather(1, positions[..., None].expand(*positions.shape, values.shape[-1]))


@torch.no_grad()
def event_write_factors(model, h, x, targets, W, chunk):
    """Closed CE gradients factorised as s⊗x, s, e⊗a, e (chunk mean)."""
    u = torch.einsum("bkd,bmd->bkm", x, W[0])+W[1][:, None]
    a = F.gelu(u)
    v = h+torch.einsum("bkm,bdm->bkd", a, W[2])+W[3][:, None]
    logits = model.read(v)
    dz = logits.softmax(-1).scatter_add(-1, targets[..., None],
                       -torch.ones_like(targets[..., None], dtype=logits.dtype))
    centred = v-v.mean(-1, keepdim=True)
    sigma = (centred.square().mean(-1, keepdim=True)+model.ln_f.eps).sqrt()
    normalized = centred/sigma
    upstream = (dz@model.unembed.weight)*model.ln_f.weight
    e = (upstream-upstream.mean(-1, keepdim=True)
         -normalized*(upstream*normalized).mean(-1, keepdim=True))/sigma
    prime = .5*(1+torch.erf(u/math.sqrt(2)))+u*torch.exp(-u.square()/2)/math.sqrt(2*math.pi)
    s = torch.einsum("bkd,bdm->bkm", e, W[2])*prime
    norm = (s.square().sum(-1)*(x.square().sum(-1)+1)
            +e.square().sum(-1)*(a.square().sum(-1)+1)).sqrt()/chunk
    joint = [torch.einsum("bkm,bkd->bmd", s, x)/chunk, s.sum(1)/chunk,
             torch.einsum("bkd,bkm->bdm", e, a)/chunk, e.sum(1)/chunk]
    return dict(x=x, a=a, s=s, e=e, norm=norm, joint=joint)


@torch.no_grad()
def own_removed_read(model, hq, xq, L1, factors, etas, clip_scale, chunk):
    """Read L1∖e without materialising one weight matrix per event."""
    f, c = factors, clip_scale[:, None, None]/chunk
    u = torch.einsum("bkd,bmd->bkm", xq, L1[0])+L1[1][:, None]
    u = u+c*f["s"]*(etas[0]*(f["x"]*xq).sum(-1, keepdim=True)+etas[1])
    a = F.gelu(u)
    v = hq+torch.einsum("bkm,bdm->bkd", a, L1[2])+L1[3][:, None]
    return v+c*f["e"]*(etas[2]*(f["a"]*a).sum(-1, keepdim=True)+etas[3])


@torch.no_grad()
def _read_batch(model, data, eta, g_clip, chunk, ladder=False):
    idx, y, meta = data
    qp = torch.as_tensor(meta["query_pos"], device=idx.device, dtype=torch.long)
    wp = torch.as_tensor(meta["write_pos"], device=idx.device, dtype=torch.long)
    qc = qp//chunk
    query_chunks = tuple(sorted(set(qc.flatten().tolist())))
    capture = tuple(sorted(set((0, 1, 5, 6, *query_chunks)))) if ladder else (0, *query_chunks)
    logits, states, tensors = chr.chr_stream(model, idx, y, meta["write_pos"], eta, g_clip,
                                            chunk, chr.CE, capture=capture)
    hq = gather(tensors["h"], qp)
    xq = gather(tensors["x"], qp) if tensors["x"] is not None else model.head.ln(hq)
    hw = gather(tensors["h"], wp)
    xw = gather(tensors["x"], wp) if tensors["x"] is not None else model.head.ln(hw)
    targets = y.gather(1, qp)
    zq = gather(logits, qp)
    zw = model.read(ttt.fast_block(hw, model.head.ln, states[0], xw))
    residual = 1-zw.softmax(-1).gather(-1, y.gather(1, wp)[..., None]).squeeze(-1)
    out = dict(margin=margin(zq, targets), correct=(zq.argmax(-1) == targets).to(torch.int8),
               probability=zq.softmax(-1).gather(-1, targets[..., None]).squeeze(-1),
               query_ce=ttt.token_ce(zq, targets), residual=residual, chunks=qc,
               query_order=qp.argsort(1).argsort(1),
               duplicate=(targets[:, :, None] == targets[:, None, :]).sum(-1).gt(1).to(torch.int8))
    if not ladder:
        return out
    if bool((wp//chunk != 0).any()) or 5 not in states or 6 not in states:
        raise ValueError("TRG ladder needs writes in chunk0 and query chunks5/6")
    W0, etas = states[0], ttt.regime_etas(model, eta)
    with torch.enable_grad():
        leaves = [w.detach().requires_grad_(True) for w in W0]
        h0 = tensors["h"][:, :chunk]
        x0 = None if tensors["x"] is None else tensors["x"][:, :chunk]
        ce = ttt._ce_tokens(model, ttt.fast_block(h0, model.head.ln, leaves, x0), y[:, :chunk])
        g_all = torch.autograd.grad(ce.mean(1).sum(), leaves, retain_graph=True)
        mask = torch.zeros_like(ce).scatter_(1, wp, 1.)
        g_kv = torch.autograd.grad((ce*mask).mean(1).sum(), leaves)
    _, norm, _ = ttt._clip_fast(list(g_all), g_clip)
    scale = torch.where(norm > g_clip, g_clip/norm.clamp_min(1e-24), torch.ones_like(norm))
    L1 = [w-et*scale.reshape(-1, *([1]*(g.ndim-1)))*g for w, et, g in zip(W0, etas, g_kv)]
    factors = event_write_factors(model, hw, xw, y.gather(1, wp), W0, chunk)
    m0 = margin(model.read(ttt.fast_block(hq, model.head.ln, W0, xq)), targets)
    m1 = margin(model.read(ttt.fast_block(hq, model.head.ln, L1, xq)), targets)
    m1e = margin(model.read(own_removed_read(model, hq, xq, L1, factors, etas, scale, chunk)), targets)
    m2 = margin(model.read(ttt.fast_block(hq, model.head.ln, states[1], xq)), targets)
    m3 = margin(model.read(ttt.fast_block(hq, model.head.ln, states[5], xq)), targets)
    m4 = out["margin"]
    terms = torch.stack((m0, m1-m1e, m1e-m0, m2-m1, m3-m2, m4-m3), -1)
    actual = torch.zeros_like(hq)
    for c in query_chunks:
        v = ttt.fast_block(hq, model.head.ln, states[c], xq)
        actual = torch.where((qc == c)[..., None], v, actual)
    slow = ttt.fast_block(hq, model.head.ln, W0, xq)
    keys = F.normalize(xw, dim=-1)
    diagonal = torch.eye(wp.shape[1], device=idx.device, dtype=torch.bool)[None]
    maximum = (keys@keys.transpose(1, 2)).masked_fill(diagonal, -float("inf")).amax(-1)
    address = F.normalize(xq, dim=-1)@keys.transpose(1, 2)
    addr_margin = address.diagonal(dim1=-2, dim2=-1)-address.masked_fill(diagonal, -float("inf")).amax(-1)
    s2, e2 = factors["s"].square().sum(-1), factors["e"].square().sum(-1)
    write_norm = (s2*(etas[0]**2*factors["x"].square().sum(-1)+etas[1]**2)
                  +e2*(etas[2]**2*factors["a"].square().sum(-1)+etas[3]**2)).sqrt()/chunk
    out.update(terms=terms, rungs=torch.stack((m0, m1, m1e, m2, m3, m4), -1),
               sigma=chr._sigma(model, actual), memory_norm=chr._centred(actual-slow).norm(dim=-1),
               write_norm=write_norm*scale[:, None],
               maximum_key_cosine=maximum, address_margin=addr_margin,
               inner_clip_scale=scale[:, None].expand_as(m0))
    return out


def _reader(model, data, eta, g_clip, chunk, ladder):
    training = model.training
    model.eval()
    try:
        batches = [_read_batch(model, b, eta, g_clip, chunk, ladder) for b in data]
        events = {k: torch.cat([b[k] for b in batches]).detach().cpu().numpy() for k in batches[0]}
    finally:
        model.train(training)
    summary = dict(acc=float(events["correct"].mean()), ce=float(events["query_ce"].mean()),
                   residual=float(np.median(events["residual"])),
                   residual_q10=float(np.quantile(events["residual"], .1)))
    for c in np.unique(events["chunks"]):
        sel = events["chunks"] == c
        summary[f"acc{c}"] = float(events["correct"][sel].mean())
        summary[f"qce{c}"] = float(events["query_ce"][sel].mean())
        if ladder:
            term, mar = events["terms"][sel].astype(np.float64), events["margin"][sel].astype(np.float64)
            summary[f"term_sums{c}"] = term.sum(0).tolist()
            summary[f"margin_sum{c}"], summary[f"margin_sumsq{c}"] = float(mar.sum()), float(mar@mar)
            summary[f"events{c}"] = int(sel.sum())
            summary[f"own_median{c}"] = float(np.median(term[:, 1]))
            summary[f"erosion_median{c}"] = float(np.median(events["rungs"][sel, 3]-mar))
            summary[f"slow_median{c}"] = float(np.median(term[:, 0]))
    return dict(events=events, summary=summary, ladder=ladder)


def light_read(model, data, eta, g_clip, chunk):
    return _reader(model, data, eta, g_clip, chunk, False)


def ladder_log(model, data, eta, g_clip, chunk):
    return _reader(model, data, eta, g_clip, chunk, True)


class DenseLogger:
    """One branch: append(step, read); close() -> manifest. DenseWriter is an alias."""
    def __init__(self, output_dir, *, identity=None, shard_steps=50, emit=True):
        self.folder = Path(output_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.identity, self.shard_steps, self.emit = identity or {}, shard_steps, emit
        self.rows, self.shards, self.last_step = [], [], None
        self.stream = (self.folder/"summaries.jsonl").open("w")

    def append(self, step, read):
        if self.last_step is not None and step != self.last_step+1:
            raise ValueError("dense grid gap")
        self.last_step = step
        self.rows.append((step, read))
        row = dict(self.identity, step=step, ladder=read["ladder"], **read["summary"])
        line = json.dumps(row, allow_nan=False)
        self.stream.write(line+"\n")
        self.stream.flush()
        if self.emit:
            print("TRG_DENSE "+line, flush=True)
        if len(self.rows) >= self.shard_steps:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        out = {"steps": np.array([s for s, _ in self.rows], np.int32),
               "ladder_steps": np.array([s for s, r in self.rows if r["ladder"]], np.int32)}
        keys = set().union(*(r["events"] for _, r in self.rows))
        for key in sorted(keys):
            data = np.stack([r["events"][key] for _, r in self.rows if key in r["events"]])
            if key in ("correct", "duplicate", "chunks", "query_order"):
                dtype = np.int8
            else:
                if not np.isfinite(data).all() or (np.abs(data) > np.finfo(np.float16).max).any():
                    raise ValueError(f"nonfinite / float16 overflow in {key}")
                dtype = np.float16
            out[key] = data.astype(dtype)
        name = f"dense-{self.rows[0][0]:06d}-{self.rows[-1][0]:06d}.npz"
        path = self.folder/name
        np.savez_compressed(path, **out)
        self.shards.append(dict(file=name, sha256=sha256_file(path), steps=out["steps"].tolist(),
                                ladder_steps=out["ladder_steps"].tolist(), fields=sorted(out)))
        self.rows.clear()

    def close(self):
        self.flush()
        self.stream.close()
        manifest = dict(schema=SCHEMA, schema_sha256=SCHEMA_SHA256, identity=self.identity,
                        shards=self.shards, summaries_sha256=sha256_file(self.folder/"summaries.jsonl"))
        (self.folder/"shards.json").write_text(json.dumps(manifest, indent=2))
        return manifest


DenseWriter = DenseLogger


def validate_shards(folder):
    folder = Path(folder)
    manifest = json.loads((folder/"shards.json").read_text())
    if manifest["schema_sha256"] != SCHEMA_SHA256:
        raise ValueError("schema hash mismatch")
    if sha256_file(folder/"summaries.jsonl") != manifest["summaries_sha256"]:
        raise ValueError("summary hash mismatch")
    steps, ladder = [], []
    for shard in manifest["shards"]:
        path = folder/shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("shard hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            if data["steps"].tolist() != shard["steps"] or data["ladder_steps"].tolist() != shard["ladder_steps"]:
                raise ValueError("shard coverage mismatch")
        steps.extend(shard["steps"])
        ladder.extend(shard["ladder_steps"])
    if steps and not np.array_equal(steps, np.arange(steps[0], steps[-1]+1)):
        raise ValueError("shard grid gap")
    return manifest, steps, ladder


def parameter_group(name):
    if name.startswith("head."):
        return "slow_head"
    if name.startswith(("ln_f.", "unembed.")):
        return "readout"
    return "eta" if name.startswith("rho") else "backbone"


class Telemetry:
    def __init__(self, model):
        self.groups, self.previous, self.chunks = {}, {}, {}
        for name, p in model.named_parameters():
            self.groups.setdefault(parameter_group(name), []).append(p)

    def observer(self, c, ce, h, W, model, x, y, keep):
        with torch.no_grad():
            z = model.read(ttt.fast_block(h.detach(), model.head.ln, [w.detach() for w in W],
                                         None if x is None else x.detach()))
            self.chunks[str(c)] = dict(ce=float(ce.detach().mean()),
                                      acc=float((z.argmax(-1) == y).float().mean()),
                                      outer_ce=float((ce.detach()*keep).mean()) if keep is not None
                                      else float(ce.detach().mean()))

    def before(self):
        self.snapshot = {key: torch.cat([p.detach().reshape(-1) for p in ps]).clone()
                         for key, ps in self.groups.items()}
        return {key: float(torch.cat([(p.grad.detach() if p.grad is not None else
                 torch.zeros_like(p)).reshape(-1) for p in ps]).norm()) for key, ps in self.groups.items()}

    def after(self, opt):
        out = {}
        for key, ps in self.groups.items():
            theta = torch.cat([p.detach().reshape(-1) for p in ps])
            update = theta-self.snapshot[key]
            prev = self.previous.get(key)
            sqrtv, ratios = [], []
            for p in ps:
                state = opt.state.get(p, {})
                if "exp_avg" not in state:
                    continue
                group = next(g for g in opt.param_groups if any(q is p for q in g["params"]))
                b1, b2 = group["betas"]
                step = float(state["step"])
                v, m = (state["exp_avg_sq"]/(1-b2**step)).sqrt(), state["exp_avg"]/(1-b1**step)
                sqrtv.append(v.reshape(-1))
                ratios.append((m.abs()/(v+group["eps"])).reshape(-1))
            out[key] = dict(update_norm=float(update.norm()),
                            relative_update=float(update.norm()/self.snapshot[key].norm().clamp_min(1e-30)),
                            cosine=float(F.cosine_similarity(update, prev, dim=0)) if prev is not None else None,
                            sqrt_vhat_median=float(torch.cat(sqrtv).median()) if sqrtv else None,
                            adam_ratio_median=float(torch.cat(ratios).median()) if ratios else None)
            self.previous[key] = update.clone()
        return out
