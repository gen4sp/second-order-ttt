"""Cap-free residual floors (rpf, ncf). The CHR rules dispatch directly to the frozen CHR kernels."""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_chr as chr
from . import exp_ttt as ttt
from .exp_chr import (position_masks, write_chunks, _joint_norm, _bcast,
                      read_parts, geometry, FORK_ARMS_ALLOWED)
from . import exp_kag as kag


@dataclasses.dataclass(frozen=True)
class FloorRule:
    name: str
    r_ref: float
    cap: float = 1000.0

    def __post_init__(self):
        if self.name not in ('rpf', 'ncf') or not 0 <= self.r_ref <= 1:
            raise ValueError(f'invalid cap-free rule: {self}')

    @property
    def tag(self):
        return f'{self.name}{round(100 * self.r_ref):02d}'


Rule = chr.Rule | FloorRule
CE = chr.CE


def parse_rule(spec: str | dict, cap: float = 1000.0) -> Rule:
    if isinstance(spec, str):
        if spec == 'ce':
            return CE
        m = re.fullmatch(r'(rp|nc|rs|ns|rpf|ncf)(\d+)', spec)
        if m is None:
            raise ValueError(f'invalid rule tag {spec!r}')
        name, pct = m.groups()
        return (FloorRule if name in ('rpf', 'ncf') else chr.Rule)(name, int(pct) / 100, cap)
    name = spec['rule']
    if name not in ('rpf', 'ncf'):
        return chr.parse_rule(spec, cap)
    if 'r_cap' in spec:
        raise ValueError('cap-free floors accept r_min only')
    return FloorRule(name, float(spec['r_min']))


def complement_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Log-sum-exp of non-target logits minus target; gradient q - e_target.

    Preserve float64 for reference checks; promote half precision to float32.
    Unlike weighted CE, this remains active when CE rounds to zero.
    """
    z = logits if logits.dtype == torch.float64 else logits.float()
    target = z.gather(-1, y[..., None]).squeeze(-1)
    return z.scatter(-1, y[..., None], float('-inf')).logsumexp(-1) - target


def match_control(extra, control):
    requested = _joint_norm(list(extra)).detach()
    norm = _joint_norm(list(control)).detach()
    ok = torch.isfinite(norm) & (norm > 1e-12)
    coef = torch.where(ok, requested / norm.clamp_min(1e-12), torch.zeros_like(norm))
    out = [_bcast(coef, g) * g for g in control]
    actual = _joint_norm(out).detach()
    fail = ((requested > 0) & ~ok) | ~torch.isfinite(actual)
    fail |= (actual - requested).abs() > 1e-4 * requested + 1e-12
    return out, dict(added=requested, actual=actual, coef=coef, match_fail=fail)


def rule_extra(rule, ce, W, kv, vw, chunk, retain, *, logits=None, y=None, measure_logits=False):
    if not isinstance(rule, FloorRule):
        return chr.rule_extra(rule, ce, W, kv, vw, chunk, retain)
    if logits is None or y is None:
        raise ValueError('cap-free extra requires logits and targets')
    r = -torch.expm1(-ce.detach())
    coeff = (rule.r_ref - r).clamp_min(0).detach() * kv
    objective = ((coeff * complement_ce(logits, y)).sum(1) / chunk).sum()
    attained = None
    if measure_logits:
        gz = torch.autograd.grad(ce.mean(1).sum() + objective, logits, retain_graph=True)[0]
        attained = -chunk * gz.gather(-1, y[..., None]).squeeze(-1).detach()
    extra = torch.autograd.grad(objective, W, retain_graph=True)
    requested = _joint_norm(list(extra)).detach()
    info = dict(added=requested, actual=requested,
                match_fail=torch.zeros_like(requested, dtype=torch.bool),
                r_kv=r, kv=kv, r_eff=torch.where(kv > 0, torch.maximum(r, r.new_tensor(rule.r_ref)),
                                               torch.full_like(r, float('nan'))),
                cap_hit=((r < rule.r_ref / 1000) & (kv > 0)).sum(1) / kv.sum(1).clamp_min(1))
    if attained is not None:
        info['attained_preclip'] = attained
    if rule.name == 'ncf':
        ctl = torch.autograd.grad(((vw * ce).sum(1) / chunk).sum(), W, retain_graph=retain)
        extra, matched = match_control(extra, ctl)
        info.update(matched)
    return list(extra), info


def chr_sequential_inner(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                         write_pos: Any, etas: list[torch.Tensor], g_clip: float, chunk: int,
                         stats: ttt.GradStats, rule: Rule, diag: dict | None = None) -> torch.Tensor:
    """First-order chunk recurrence of ``exp_ttt.sequential_inner`` plus ``rule``'s extra.

    The CE path (fast block, token CE, chunk mean, retained first-order gradient,
    joint clip, ``W - eta g``, per-chunk CE outer loss) is the inherited one; ``rp``
    and ``nc`` add their extra gradient before the clip in the chunks that hold a
    kv write. ``diag`` (optional) receives the detached added norms and clip flags
    of those chunks.
    """
    if not isinstance(rule, FloorRule):
        return chr.chr_sequential_inner(model, h, y, idx, write_pos, etas, g_clip, chunk, stats, rule, diag)
    T, b = h.shape[1], chunk
    W, total, n_chunk = model.per_sequence(idx.shape[0]), 0.0, T // b
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    active = write_chunks(write_pos, b) if rule.name != "ce" else set()
    masks = position_masks(write_pos, y, h.device) if active else None
    for c in range(n_chunk):
        sl = slice(c * b, (c + 1) * b)
        logits = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                                   None if x is None else x[:, sl]))
        ce = ttt.token_ce(logits, y[:, sl])
        loss_c = ce.mean(1)
        grads = torch.autograd.grad(loss_c.sum(), W, create_graph=False, retain_graph=True)
        raw = list(grads)
        if c in active:
            extra, info = rule_extra(rule, ce, W, masks["kv"][:, sl], masks["vw"][:, sl], b, True, logits=logits, y=y[:, sl])
            raw = [g + e for g, e in zip(raw, extra)]
        grads, norm, clipped = ttt._clip_fast(raw, g_clip)
        if c in active and diag is not None:
            for key, val in (("added", info["added"]), ("actual", info["actual"]), ("clipped", clipped.detach()),
                             ("match_fail", info["match_fail"]), ("cap_bound", info.get("cap_bound")),
                             ("cap_hit", info.get("cap_hit"))):
                if val is not None:
                    diag.setdefault(key, []).append(val.detach())
        stats.add(norm, clipped, c)
        total = total + loss_c.mean()
        W = [w - eta * g for w, eta, g in zip(W, etas, grads)]
    return total / n_chunk


def chr_arm_loss(model: ttt.TTTModel, arm: str, idx: torch.Tensor, y: torch.Tensor, eta_base: float,
                 g_clip: float, chunk: int, stats: ttt.GradStats, meta: dict, rule: Rule,
                 diag: dict | None = None) -> torch.Tensor:
    """The inherited ``exp_ttt.arm_loss`` for ``ce``; the FO rule loop otherwise."""
    if not isinstance(rule, FloorRule):
        return chr.chr_arm_loss(model, arm, idx, y, eta_base, g_clip, chunk, stats, meta, rule, diag)
    if rule.name == "ce":
        return ttt.arm_loss(model, arm, idx, y, eta_base, g_clip, chunk, stats, meta=meta)
    if arm not in FORK_ARMS_ALLOWED:
        raise ValueError(f"rule {rule.tag} is defined for {FORK_ARMS_ALLOWED} only, not {arm}")
    h = model.backbone(idx, edge_mask=ttt.arm_edge_mask(model, arm, idx, meta))
    etas = ttt.regime_etas(model, eta_base)
    return chr_sequential_inner(model, h, y, idx, meta["write_pos"], etas, g_clip, chunk, stats,
                                rule, diag)


@torch.no_grad()
def chr_stream(model: ttt.TTTModel, idx: torch.Tensor, y: torch.Tensor, write_pos: Any,
               eta_base: float, g_clip: float, chunk: int, rule: Rule,
               capture: tuple[int, ...] = (), kv_weights: torch.Tensor | None = None,
               clip_out: list | None = None, norm_out: list | None = None
               ) -> tuple[torch.Tensor, dict[int, list[torch.Tensor]], dict]:
    """``exp_ttt.eval_sequential``'s CE streaming loop with ``rule``'s write extra.

    ``kv_weights`` (B, K), only with ``ce``, weights each write event's kv CE term
    by a given detached factor (adds ``sum_kv (w_i - 1) ce_i / chunk``'s gradient).
    ``clip_out`` / ``norm_out`` (optional) receive the per-sequence clip flags / pre-clip
    joint norms of the write chunks.
    Returns logits ``(B, T, V)``, the fast weights each captured chunk was read with,
    and the backbone/fast-input tensors used (for the read decomposition).
    """
    if not isinstance(rule, FloorRule):
        return chr.chr_stream(model, idx, y, write_pos, eta_base, g_clip, chunk, rule, capture, kv_weights, clip_out, norm_out)
    if kv_weights is not None and rule.name != "ce":
        raise ValueError("kv_weights replace the rule; use them with ce only")
    h = model.backbone(idx).detach()
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    T, b = h.shape[1], chunk
    etas = ttt.regime_etas(model, eta_base)
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(idx.shape[0])]
    active = (write_chunks(write_pos, b) if rule.name != "ce" or kv_weights is not None
              or clip_out is not None or norm_out is not None else set())
    masks = position_masks(write_pos, y, h.device) if active else None
    tokw = None
    if kv_weights is not None:
        wp = torch.as_tensor(np.asarray(write_pos), device=h.device, dtype=torch.long)
        tokw = torch.zeros(idx.shape[0], T, device=h.device).scatter_(1, wp, kv_weights.float() - 1.0)
    out, captured = [], {}
    with torch.enable_grad():
        for c in range(T // b):
            sl = slice(c * b, (c + 1) * b)
            if c in capture:
                captured[c] = [w.detach() for w in W]
            logits = model.read(ttt.fast_block(h[:, sl], model.head.ln, W,
                                               None if x is None else x[:, sl]))
            out.append(logits.detach())
            ce = ttt.token_ce(logits, y[:, sl])
            extra_on = c in active and (rule.name != "ce" or tokw is not None)
            grads = torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=extra_on)
            raw = list(grads)
            if extra_on and tokw is not None:
                extra = torch.autograd.grad(((tokw[:, sl] * ce).sum(1) / b).sum(), W)
                raw = [g + e for g, e in zip(raw, extra)]
            elif extra_on:
                extra, _ = rule_extra(rule, ce, W, masks["kv"][:, sl], masks["vw"][:, sl], b, False, logits=logits, y=y[:, sl])
                raw = [g + e for g, e in zip(raw, extra)]
            grads, norm, clipped = ttt._clip_fast(raw, g_clip)
            if clip_out is not None and c in active:
                clip_out.append(clipped.detach().float())
            if norm_out is not None and c in active:
                norm_out.append(norm.detach().float())
            W = [(w - eta * g).detach().requires_grad_(True) for w, eta, g in zip(W, etas, grads)]
    return torch.cat(out, dim=1), captured, dict(h=h, x=x)


@torch.no_grad()
def quick_log(model: ttt.TTTModel, data: list, eta_base: float, g_clip: float, chunk: int,
              rule: Rule, chunks: tuple[int, ...], geometry_on: bool = True) -> dict:
    """Quick-set accuracy per query chunk under ``rule`` plus the read geometry."""
    if not isinstance(rule, FloorRule):
        return chr.quick_log(model, data, eta_base, g_clip, chunk, rule, chunks, geometry_on)
    hits, ces, cids, geo_parts = [], [], [], []
    for idx, y, meta in data:
        logits, captured, tens = chr_stream(model, idx, y, meta["write_pos"], eta_base, g_clip,
                                            chunk, rule, capture=chunks if geometry_on else ())
        qp = torch.as_tensor(np.asarray(meta["query_pos"]), device=idx.device)
        tgt = torch.as_tensor(np.asarray(meta["query_tok"]), device=idx.device)
        lq = logits.gather(1, qp[..., None].expand(-1, -1, logits.shape[-1]))
        hits.append((lq.argmax(-1) == tgt).float())
        ces.append(F.cross_entropy(lq.reshape(-1, lq.shape[-1]), tgt.reshape(-1),
                                   reduction="none").view(tgt.shape))
        cids.append(qp // chunk)
        if geometry_on:
            geo_parts.append(read_parts(model, tens, captured, meta["query_pos"], chunk))
    hit, ce, cid = torch.cat(hits), torch.cat(ces), torch.cat(cids)
    out = dict(acc=round(float(hit.mean()), 6), qce=round(float(ce.mean()), 6))
    for c in torch.unique(cid).tolist():
        sel = cid == c
        out[f"acc{c}"] = round(float(hit[sel].mean()), 6)
        out[f"qce{c}"] = round(float(ce[sel].mean()), 6)
    if geometry_on:
        merged = {k: torch.cat([p[k] for p in geo_parts]) for k in geo_parts[0]}
        out["geo"] = geometry(model, merged, chunks)
    return out


@torch.no_grad()
def write_log(model: ttt.TTTModel, data: list, eta_base: float, g_clip: float, chunk: int,
              rule: Rule, *, vector_out=None, reference=None) -> dict:
    """Write-side state at the slow weights, chunk 0 of the probe set, under ``rule``.

    Residuals ``r = 1 - p(target)`` at the kv positions (target: the value), the
    value->W control positions and the W-marker positions; the chunk-0 inner
    gradient's joint norm and clip flag under the rule; the kv tokens' own write
    norm ``|eta * s * g_kv|`` (``s`` the rule's clip scale; for ``rp``/``rs`` the kv
    term includes the extra), the norm the rule adds (rp/nc) or removes (rs/ns), and
    for ``ns`` the fraction of sequences whose removal scale hit its cap.
    """
    if not isinstance(rule, FloorRule) and vector_out is None:
        return chr.write_log(model, data, eta_base, g_clip, chunk, rule)
    r_kv, r_vw, r_wv, g0, clip, kvw, added, actual, capb = [], [], [], [], [], [], [], [], []
    scale_ratio, eff, fails, reff, caph = [], [], [], [], []
    active_frac, err, shares, cosines = [], [], [], []
    eta = float(ttt.regime_etas(model, eta_base)[2].detach())
    for idx, y, meta in data:
        h = model.backbone(idx).detach()
        wp = meta["write_pos"]
        if write_chunks(wp, chunk) != {0}:
            raise ValueError("write_log expects every write in chunk 0")
        masks = position_masks(wp, y, idx.device)
        kv, vw = masks["kv"][:, :chunk], masks["vw"][:, :chunk]
        with torch.enable_grad():
            W = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
            x = model.fast_input(h, idx) if model.frozen_emb is not None else None
            logits = model.read(ttt.fast_block(h[:, :chunk], model.head.ln, W,
                                               None if x is None else x[:, :chunk]))
            ce = ttt.token_ce(logits, y[:, :chunk])
            base = torch.autograd.grad(ce.mean(1).sum(), W, retain_graph=True)
            g_kv = torch.autograd.grad(((kv * ce).sum(1) / chunk).sum(), W, retain_graph=True)
            extra, info = rule_extra(rule, ce, W, kv, vw, chunk, False, logits=logits, y=y[:, :chunk],
                                     measure_logits=isinstance(rule, FloorRule))
        total = list(base) if extra is None else [g + e for g, e in zip(base, extra)]
        _, norm, clipped = ttt._clip_fast(total, g_clip)
        scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
        base_norm = _joint_norm(list(base))
        s_ce = torch.where(base_norm > g_clip, g_clip / base_norm.clamp_min(1e-24), torch.ones_like(norm))
        scale_ratio.append(scale / s_ce)
        # post-clip update change relative to CE: |s (g + e) - s_ce g|
        eff.append(eta * _joint_norm([_bcast(scale, a) * a - _bcast(s_ce, g) * g
                                      for a, g in zip(total, base)]))
        if info and isinstance(rule,FloorRule):
            active_frac.append((info["r_kv"] < rule.r_ref)[kv > 0].float())
            attained = info["attained_preclip"][kv > 0]
            target = torch.maximum(info["r_kv"][kv > 0], ce.new_tensor(rule.r_ref))
            err.append(((attained-target).abs() / target.clamp_min(1e-30)))
            denom = info["actual"].square().clamp_min(1e-30)
            shares.append(torch.stack([e.flatten(1).square().sum(1) / denom for e in extra], 1))
            fails.append(info["match_fail"].float())
            reff.append(info["r_eff"][kv > 0])
            if "cap_hit" in info:
                caph.append(info["cap_hit"])
        kv_term = list(g_kv) if rule.name not in ("rp", "rs", "rpf") else [a + e for a, e in zip(g_kv, extra)]
        if vector_out is not None or reference is not None:
            vec=torch.cat([(_bcast(eta*scale,g)*g).flatten(1) for g in kv_term],1).detach().cpu()
            if vector_out is not None: vector_out.append(vec)
            if reference is not None:
                offset=sum(len(v) for v in kvw)
                cosines.append(F.cosine_similarity(vec,reference[offset:offset+len(vec)],dim=1,eps=1e-12))
        kvw.append(eta * scale * _joint_norm(kv_term))
        g0.append(norm)
        clip.append(clipped.float())
        added.append(info["added"] if info else torch.zeros_like(norm))
        actual.append(info["actual"] if info else torch.zeros_like(norm))
        if "cap_bound" in info:
            capb.append(info["cap_bound"].float())
        r = -torch.expm1(-ce)
        r_kv.append(kag._gather(r, wp).flatten())
        r_vw.append(r[vw > 0])  # true value->W transitions only
        r_wv.append(kag._gather(r, np.asarray(wp) - 2).flatten())
    rk = torch.cat(r_kv).float().cpu().double()  # MPS has no float64
    out = dict(r_kv_med=float(rk.median()), r_kv_mean=float(rk.mean()),
               r_kv_q01=float(torch.quantile(rk, .01)), r_kv_lt1e6=float((rk < 1e-6).double().mean()),
               r_kv_q10=float(torch.quantile(rk, .10)), r_kv_q90=float(torch.quantile(rk, .90)),
               r_kv_lt01=float((rk < .01).double().mean()), r_kv_lt02=float((rk < .02).double().mean()),
               r_kv_lt03=float((rk < .03).double().mean()),
               r_vw_med=float(torch.cat(r_vw).median()), r_wv_med=float(torch.cat(r_wv).median()),
               g0_med=float(torch.cat(g0).median()), clip_c0=float(torch.cat(clip).mean()),
               kv_write_med=float(torch.cat(kvw).median()),
               added_req_med=float(torch.cat(added).median()),
               added_actual_med=float(torch.cat(actual).median()))
    out["clip_scale_ratio_med"] = float(torch.cat(scale_ratio).median())
    out["eff_change_med"] = float(torch.cat(eff).median())
    if cosines:
        out["kv_cos_twin_start_med"]=float(torch.cat(cosines).median())
    if active_frac:
        out["floor_active"] = float(torch.cat(active_frac).mean())
        out["floor_rel_error_max"] = float(torch.cat(err).max())
        out["attained_ratio"] = float(torch.cat(actual).sum() / torch.cat(added).sum().clamp_min(1e-30)) if float(torch.cat(added).sum()) > 0 else 1.0
        out["r_eff_hypothetical"] = float(rule.name == "ncf")
        for i, v in enumerate(torch.cat(shares).mean(0)):
            out[f"extra_share_{ttt.FAST_KEYS[i]}"] = float(v)
    if fails:
        out["match_fail"] = float(torch.cat(fails).mean())
        out["r_eff_med"] = float(torch.cat(reff).median())
    if caph:
        out["cap_hit"] = float(torch.cat(caph).mean())
    if capb:
        out["cap_bound"] = float(torch.cat(capb).mean())
    return {k: (round(v, 8) if math.isfinite(v) else None) for k, v in out.items()}
