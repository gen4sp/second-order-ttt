"""Inner rules and outer-gradient engines for TTT-E2E at the real shape.

Per sequence, chunk ``c`` is predicted with ``W_c`` (pre-update), its mean token CE
``l_c`` is the outer loss term, and ``W_{c+1} = W_c - lr * clip(g_c)`` with ``g_c`` the
partial gradient of the inner loss w.r.t. ``W_c`` (prefix output and past K/V fixed).
The outer objective of one sequence is ``(1/n) sum_c l_c``.

Inner rules (``rule``):
- ``ce``:   inner loss = l_c (the release).
- ``lin``:  mean_t [ mean_j z_tj - z_t,y ]  (logit gradient 1/V - onehot(y)).
- ``aice``: mean_t CE(z_t - sg z0_t, y_t), z0 = logits of the never-adapted stream
            (W_0 in every chunk, its own K/V, same barriers).

Engines (all accumulate into ``.grad`` and return detached statistics):
- ``so_backward``: exact second order (gradients through the inner gradient and the
  clip), rematerialized per chunk: O(1 chunk) of double-backward graph.
- ``fo_backward``: exact first order (``W_{c+1} = W_c - lr * sg(clip(g_c))``, all other
  paths kept, including K/V of earlier chunks). One backward of ``l_c`` per chunk gives
  the chunk's outer contributions and, for ``ce``, the inner gradient itself; the
  cotangents on earlier chunks' K/V are pushed back afterwards by recomputing only the
  K/V path (no LM head). ``trunc_kv=True`` drops that sweep (ouroboros "shared").
- ``so_loss_plain`` / ``fo_loss_plain``: un-optimized full-graph references for tests.

Token weights: ``mask`` marks loss-valid targets (target != BOS). ``inner_mask`` (eval
only, focal-write removal) zeroes some tokens in the inner loss *without* changing the
denominator, so a removed token simply stops contributing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import TTTE2E, _acc

RULES = ("ce", "lin", "aice", "rp", "nc")
TOK_MULTS = ("none", "const", "sigma", "sigma_shuf", "sigma_inv", "sigma_inv_shuf")
SIGMA_MULTS = ("sigma", "sigma_shuf", "sigma_inv", "sigma_inv_shuf")


@dataclass
class InnerCfg:
    rule: str = "ce"
    lr: float = 1.0
    clip: float = 1.0
    block_chunks: int = 1
    r_ref: float = 0.03
    # NC's selection is supplied by the caller so training and evaluation can
    # use the same counter-based draw after a resume.
    classes: torch.Tensor | None = None
    seed: int = 0
    step: int = 0
    row_ids: torch.Tensor | None = None
    # G0 eval-time token multipliers m_t on the CE write (detached; the inner
    # denominator is unchanged): "none"; "const" (m = const_mult); "sigma"
    # (m = min(sigma_cap, sigma_t / sigma_ref), sigma_t = RMS of the residual
    # before ln_f); "sigma_inv" (G0b: m = min(sigma_cap,
    # sigma_ref / sigma_t)); "sigma_shuf" / "sigma_inv_shuf" (the same values permuted
    # among each row's valid tokens of the chunk, shared by suppression modes).
    tok_mult: str = "none"
    sigma_ref: float = 1.0
    sigma_cap: float = 4.0
    const_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.tok_mult not in TOK_MULTS:
            raise ValueError(f"tok_mult must be one of {TOK_MULTS}, got {self.tok_mult!r}")
        if self.tok_mult != "none" and self.rule != "ce":
            raise ValueError("token multipliers are defined for rule='ce' only")
        if self.tok_mult in SIGMA_MULTS and not (self.sigma_ref > 0 and self.sigma_cap > 0):
            raise ValueError("sigma multipliers need sigma_ref > 0 and sigma_cap > 0")
        if self.block_chunks != 1:
            # K > 1 (ouroboros' block schedule: one update per K chunks, the inner gradient
            # of the block objective includes the in-block K/V paths) is not a P2-1 arm;
            # use ttt_pt.block_inner for it.
            raise NotImplementedError("block_chunks > 1 is not implemented in tttmem.p2")


# --------------------------------------------------------------------------- helpers
def clip_per_seq(grads: list[torch.Tensor], clip: float, detach_scale: bool = False):
    """optax ``clip_by_global_norm`` per sequence over all fast tensors (differentiable).
    Returns (clipped grads, pre-clip norms [B])."""
    B = grads[0].shape[0]
    acc = _acc(grads[0].dtype)
    sq = sum(g.to(acc).pow(2).reshape(B, -1).sum(-1) for g in grads)
    norm = sq.detach().sqrt()
    if clip <= 0:
        return list(grads), norm
    # = clip / max(norm, clip), but never differentiates sqrt at 0 (all-masked chunks)
    scale = clip / sq.clamp_min(clip * clip).sqrt()
    if detach_scale:
        scale = scale.detach()
    return [g * scale.view(B, *([1] * (g.dim() - 1))).to(g.dtype) for g in grads], norm


def token_nll(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape)


def denom(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(1).to(torch.float32).clamp_min(1e-10)


def seq_mean(tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence masked mean over the chunk ([B])."""
    return (tok * mask.to(tok.dtype)).sum(1) / denom(mask).to(tok.dtype)


def residual_sigma(h_pre: torch.Tensor, eps: float) -> torch.Tensor:
    """sigma_t of the final RMSNorm: sqrt(mean(h_t^2) + eps) in the accumulation dtype [B, C]."""
    hf = h_pre.to(_acc(h_pre.dtype))
    return (hf.pow(2).mean(-1) + eps).sqrt()


def token_multiplier(inner: InnerCfg, sigma: torch.Tensor | None, write: torch.Tensor, *,
                     chunk: int = 0, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Detached G0 multipliers m_t [B, C] for the CE write (see ``InnerCfg.tok_mult``).

    ``sigma_shuf`` permutes the sigma multipliers among each row's valid tokens of
    the chunk with a counter-seeded generator (row id, chunk, ``inner.seed``), so the
    multiplier distribution is kept and its token alignment is broken.  ``sigma_inv``
    (G0b) uses ``sigma_ref / sigma_t`` instead; ``sigma_inv_shuf`` permutes it the same way.
    """
    B, C = write.shape
    acc = dtype or (sigma.dtype if sigma is not None and sigma.is_floating_point()
                    else torch.float32)
    if inner.tok_mult == "none":
        return torch.ones(B, C, dtype=acc, device=write.device)
    if inner.tok_mult == "const":
        return torch.full((B, C), float(inner.const_mult), dtype=acc, device=write.device)
    if sigma is None:
        raise ValueError("sigma multipliers need the residual sigma")
    ratio = sigma.detach().to(acc) / float(inner.sigma_ref)
    if inner.tok_mult in ("sigma_inv", "sigma_inv_shuf"):
        ratio = 1.0 / ratio
    m = ratio.clamp_max(float(inner.sigma_cap))
    if inner.tok_mult in ("sigma_shuf", "sigma_inv_shuf"):
        rows = (inner.row_ids.tolist() if inner.row_ids is not None else list(range(B)))
        m = m.clone()
        for b in range(B):
            idx = torch.nonzero(write[b], as_tuple=False).flatten()
            if idx.numel() < 2:
                continue
            g = torch.Generator(device="cpu")
            g.manual_seed(_counter_u32(inner.seed, 0, int(rows[b]), int(chunk), 0x5EED))
            perm = torch.randperm(idx.numel(), generator=g).to(idx.device)
            m[b, idx] = m[b, idx[perm]]
    return m


def inner_logit_cotangent(rule: str, logits: torch.Tensor, y: torch.Tensor,
                          w: torch.Tensor, z0: torch.Tensor | None = None,
                          r_ref: float = 0.03) -> torch.Tensor:
    """d(sum_b inner_b)/d logits for weights ``w`` [B, C] (= inner_mask / denom)."""
    V = logits.shape[-1]
    if rule == "lin":
        cot = torch.full_like(logits, 1.0 / V)
    elif rule in ("ce", "nc"):
        cot = torch.softmax(logits, -1)
    elif rule == "rp":
        # Complement-CE form.  Keeping the target at -inf avoids a division by
        # a rounded-away residual and makes the cap-free rule finite.
        p = torch.softmax(logits, -1)
        pt = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        # r = 1 - p_y; deriving it from the already materialised softmax avoids
        # a second log-sum-exp over the full vocabulary.
        r = (1.0 - pt).detach()
        coeff = torch.maximum(r, torch.as_tensor(r_ref, dtype=r.dtype, device=r.device)) \
            if r_ref > 0 else r
        # RP and CE are algebraically identical above the floor.  Materialise
        # the complement only for rows where the floor is active; this is the
        # dominant memory saving at the usual r_ref=.03.
        cot = p.clone()
        cot.scatter_add_(-1, y.unsqueeze(-1), torch.full_like(y, -1, dtype=cot.dtype).unsqueeze(-1))
        if r_ref > 0 and bool((r < r_ref).any()):
            active = (r < r_ref)
            q = p[active] / (1.0 - pt[active]).clamp_min(torch.finfo(p.dtype).tiny).unsqueeze(-1)
            q.scatter_(-1, y[active].unsqueeze(-1), 0.0)
            q = r_ref * q
            q.scatter_add_(-1, y[active].unsqueeze(-1),
                           torch.full_like(y[active], -r_ref, dtype=q.dtype).unsqueeze(-1))
            cot[active] = q
        if r_ref == 0:
            # The explicit CE path preserves exact equality for the registered
            # r_ref=0 identity test (including tiny residuals).
            cot = p
            # already has the target coefficient from the CE expression
    elif rule == "aice":
        cot = torch.softmax(logits - z0, -1)
    else:
        raise ValueError(rule)
    if rule == "rp" and r_ref == 0:
        cot.scatter_add_(-1, y.unsqueeze(-1), torch.full_like(y, -1,
                         dtype=cot.dtype).unsqueeze(-1))
    elif rule != "rp":
        cot.scatter_add_(-1, y.unsqueeze(-1), torch.full_like(y, -1,
                         dtype=cot.dtype).unsqueeze(-1))
    return cot * w.unsqueeze(-1).to(cot.dtype)


def rp_logit_cotangent(logits: torch.Tensor, y: torch.Tensor, *, r_ref: float = 0.03,
                       w: torch.Tensor | None = None) -> torch.Tensor:
    """Unweighted/weighted cap-free residual-floor cotangent."""
    if w is None:
        w = torch.ones(logits.shape[:-1], dtype=logits.dtype, device=logits.device)
    return inner_logit_cotangent("rp", logits, y, w, r_ref=r_ref)


def _counter_u32(seed: int, step: int, row: int, chunk: int, value: int) -> int:
    # Integer-only counter hash.  It is stable across processes and resumes and
    # avoids touching the global torch/NumPy RNG streams.
    z = (int(seed) * 0x9E3779B1 + int(step) * 0x85EBCA77 + int(row) *
         0xC2B2AE3D + int(chunk) * 0x27D4EB2F + int(value) * 0x165667B1)
    z ^= z >> 16
    z = (z * 0x7FEB352D) & 0xFFFFFFFF
    z ^= z >> 15
    return z & 0xFFFFFFFF


def nc_control_mask(active: torch.Tensor, classes: torch.Tensor, *, seed: int = 0,
                    step: int = 0, row_ids: torch.Tensor | None = None,
                    chunk: int = 0, num_classes: int = 4,
                    allowed: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Select class-matched non-active tokens without Python per-token sampling.

    A deterministic hash score is sorted inside each (row, class) cell; taking
    the first ``min(active, nonactive)`` gives the exact registered counts.
    """
    if active.shape != classes.shape:
        raise ValueError("active and classes must have the same shape")
    B, T = active.shape
    device = active.device
    classes = torch.where(classes == 0, torch.ones_like(classes), classes)
    rows = torch.arange(B, device=device)[:, None] if row_ids is None else row_ids.to(device).view(B, 1)
    pos = torch.arange(T, device=device).view(1, T)
    # vectorised deterministic score; ties are broken by position in stable sort
    score = ((rows * 0x9E3779B1 + (step + 1) * 0x85EBCA77 +
              (chunk + 1) * 0xC2B2AE3D + (pos + 1) * 0x27D4EB2F + seed) & 0xFFFFFFFF)
    score = score ^ (score >> 16)
    score = score.to(torch.int64)
    allowed = torch.ones_like(active) if allowed is None else allowed.to(torch.bool)
    selected = torch.zeros_like(active)
    # Sorting the whole row once per class keeps all selection on-device.  In
    # particular, there is no int(tensor.sum()) inside the B x class loop.
    # Counts are materialised once after the chunk, for diagnostics only.
    act_counts = torch.stack([(active & allowed & (classes == cls)).sum(1)
                              for cls in range(num_classes)], dim=1)
    pool_masks = [(~active) & allowed & (classes == cls) for cls in range(num_classes)]
    for cls, pool in enumerate(pool_masks):
        # 1 << 32 is above every 32-bit score, so pool positions always sort first.
        order = torch.argsort(score.masked_fill(~pool, 1 << 32), dim=1, stable=True)
        rank = torch.arange(T, device=device).view(1, T)
        n = torch.minimum(act_counts[:, cls], pool.sum(1))
        take = pool.gather(1, order) & (rank < n.view(-1, 1))
        # Each class pass permutes all T positions, so a plain scatter would
        # overwrite earlier classes; accumulate this class's picks instead.
        selected |= torch.zeros_like(selected).scatter_(1, order, take)
    pool_counts = torch.stack([x.sum(1) for x in pool_masks], dim=1)
    matched = torch.minimum(act_counts, pool_counts)
    # Leave diagnostics on the device until the caller logs the completed
    # step/read. No host synchronisation occurs inside the chunk loop.
    counts = {(b, int(chunk), cls): (act_counts[b, cls].detach(), matched[b, cls].detach())
              for b in range(B) for cls in range(num_classes)}
    return selected, counts


def _inner_cotangent(model: TTTE2E, inner: InnerCfg, logits: torch.Tensor,
                     y: torch.Tensor, mask: torch.Tensor, z0=None,
                     classes: torch.Tensor | None = None, *, chunk: int = 0,
                     inner_mask: torch.Tensor | None = None):
    """Return the logit cotangent for CE/RP/NC and diagnostic counts."""
    acc = _acc(logits.dtype)
    write_mask = mask if inner_mask is None else inner_mask & mask
    w = write_mask.to(acc) / denom(mask).to(acc).unsqueeze(1)
    if inner.rule == "nc":
        active = (-torch.expm1(-F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                                 reduction="none").view_as(mask))).detach() < inner.r_ref
        cls = classes if classes is not None else torch.zeros_like(mask, dtype=torch.long)
        # Code 0 is the undefined-key cell and is deliberately matched as
        # non-repeat (cell 1). Keep the original count separately below.
        cls = torch.where(cls == 0, torch.ones_like(cls), cls)
        allowed = write_mask
        ctl, counts = nc_control_mask(active, cls, seed=inner.seed, step=inner.step,
                                       row_ids=inner.row_ids, chunk=chunk, allowed=allowed)
        # Reuse the softmax: p becomes the CE cotangent, while the RP
        # complement is overwritten in-place before becoming the extra term.
        p = torch.softmax(logits, -1)
        pt = p.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        coeff = torch.maximum((1.0 - pt).detach(),
                              torch.as_tensor(inner.r_ref, dtype=p.dtype, device=p.device))
        rp = p / (1.0 - pt).clamp_min(torch.finfo(p.dtype).tiny).unsqueeze(-1)
        rp.scatter_(-1, y.unsqueeze(-1), 0.0)
        rp.mul_(coeff.unsqueeze(-1)).scatter_add_(-1, y.unsqueeze(-1), -coeff.unsqueeze(-1))
        rp.mul_(w.unsqueeze(-1).to(rp.dtype))
        p.scatter_add_(-1, y.unsqueeze(-1), torch.full_like(y, -1, dtype=p.dtype).unsqueeze(-1))
        p.mul_(w.unsqueeze(-1).to(p.dtype))
        extra = rp.sub_(p)
        # The selected control is CE-shaped and is scaled later by the joint
        # fast-tensor norm in ``inner_gradients``.
        ctl_cot = p * ctl.unsqueeze(-1).to(p.dtype)
        return p, extra, ctl_cot, (counts, (active & write_mask).any(1))
    cot = inner_logit_cotangent(inner.rule, logits, y, w, z0, r_ref=inner.r_ref)
    return cot, None, None, {}


def _ce_and_extra_x(model, logits, y, w, r_ref, nll=None):
    """Project CE and only the floor-active residual into hidden space.

    The complement softmax is evaluated on active token rows only. Its target
    is masked before softmax, so even p_y rounded to one remains finite.
    """
    dt = model.compute_dtype
    E = model.wte.weight.detach().to(dt)
    with torch.no_grad():
        p = torch.softmax(logits.detach(), -1)
        r = (1.0 - p.gather(-1, y.unsqueeze(-1)).squeeze(-1) if nll is None
             else -torch.expm1(-nll.detach()))
        active = (r < r_ref) & (w != 0)
        p.scatter_add_(-1, y.unsqueeze(-1), -torch.ones_like(y, dtype=p.dtype).unsqueeze(-1))
        p.mul_(w.unsqueeze(-1).to(p.dtype))
        ce_x = p.to(dt) @ E
        za = logits.detach()[active].clone()
        ya = y[active].unsqueeze(-1)
        za.scatter_(-1, ya, -torch.inf)
        q = torch.softmax(za, -1)
        del za
        q.mul_(r_ref).scatter_add_(-1, ya, -torch.full_like(ya, r_ref, dtype=q.dtype))
        q.mul_(w[active].unsqueeze(-1).to(q.dtype)).sub_(p[active])
        del p
        extra_x = torch.zeros_like(ce_x)
        extra_x[active] = q.to(dt) @ E
        # This is the same final-hidden-state cotangent used by stream_eval's
        # source diagnostic, but without rebuilding a [B,C,V] tensor.
        cot_norm = (ce_x + extra_x).norm(dim=-1) / w.clamp_min(torch.finfo(w.dtype).tiny)
    return ce_x, extra_x, active, cot_norm


def inner_gradients(model: TTTE2E, x: torch.Tensor, Wc: list[torch.Tensor], logits,
                    y: torch.Tensor, mask: torch.Tensor, inner: InnerCfg, *,
                    z0=None, classes=None, chunk: int = 0,
                    inner_mask: torch.Tensor | None = None, nll=None):
    """VJP for an inner rule; returns gradients and rule diagnostics."""
    write_mask = mask if inner_mask is None else inner_mask & mask
    if inner.rule not in ("rp", "nc"):
        cot = inner_x_cotangent(model, inner.rule, logits, y,
            write_mask.to(_acc(logits.dtype)) / denom(mask).to(_acc(logits.dtype)).unsqueeze(1),
            z0, r_ref=inner.r_ref)
        return list(torch.autograd.grad(x, Wc, grad_outputs=cot, retain_graph=True)), {"counts": {}}
    w = write_mask.to(_acc(logits.dtype)) / denom(mask).to(_acc(logits.dtype)).unsqueeze(1)
    ce_x, ex_x, active, cot_norm = _ce_and_extra_x(model, logits, y, w, inner.r_ref, nll)
    if inner.rule == "rp":
        # Keep the CE and floor residual VJPs separate so the registered RP
        # manipulation check can be reconstructed from steps.jsonl.
        gce = list(torch.autograd.grad(x, Wc, grad_outputs=ce_x, retain_graph=True))
        gex = list(torch.autograd.grad(x, Wc, grad_outputs=ex_x, retain_graph=True))
        ne = torch.sqrt(sum(t.to(_acc(t.dtype)).pow(2).reshape(t.shape[0], -1).sum(-1) for t in gex))
        nce = torch.sqrt(sum(t.to(_acc(t.dtype)).pow(2).reshape(t.shape[0], -1).sum(-1) for t in gce))
        nt = torch.sqrt(sum((a + b).to(_acc(a.dtype)).pow(2).reshape(a.shape[0], -1).sum(-1)
                            for a, b in zip(gce, gex)))
        induced = (nt > inner.clip) & (nce <= inner.clip) if inner.clip > 0 else torch.zeros_like(nt, dtype=torch.bool)
        return [a + b for a, b in zip(gce, gex)], {
            "counts": {}, "extra_norm": ne.detach(), "ce_norm": nce.detach(),
            "induced_binding": induced.detach(), "floor_active": active,
            "cot_norm": cot_norm.detach(),
        }
    cls = classes if classes is not None else torch.zeros_like(mask, dtype=torch.long)
    ctl, counts = nc_control_mask(active, cls, seed=inner.seed, step=inner.step,
                                  row_ids=inner.row_ids, chunk=chunk, allowed=write_mask)
    ctl_x = ce_x * ctl.unsqueeze(-1).to(ce_x.dtype)
    g = list(torch.autograd.grad(x, Wc, grad_outputs=ce_x, retain_graph=True))
    ge = list(torch.autograd.grad(x, Wc, grad_outputs=ex_x, retain_graph=True))
    gc = list(torch.autograd.grad(x, Wc, grad_outputs=ctl_x, retain_graph=True))
    ne = torch.sqrt(sum(t.to(_acc(t.dtype)).pow(2).reshape(t.shape[0], -1).sum(-1) for t in ge))
    raw_nc = torch.sqrt(sum(t.to(_acc(t.dtype)).pow(2).reshape(t.shape[0], -1).sum(-1) for t in gc))
    kappa = (ne / raw_nc.clamp_min(1e-12)).detach()
    scaled_gc = [kappa.view(-1, *([1] * (b.dim() - 1))).to(b.dtype) * b for b in gc]
    actual_nc = torch.sqrt(sum(t.to(_acc(t.dtype)).pow(2).reshape(t.shape[0], -1).sum(-1)
                               for t in scaled_gc))
    norm_miss = (actual_nc - ne).abs() / ne.clamp_min(1e-12)
    active_row = active.any(1)
    nc_fail = active_row & (((raw_nc < 1e-12) & (ne > 0)) | (norm_miss > 1e-4))
    g = [a + b for a, b in zip(g, scaled_gc)]
    code0 = ((classes == 0) if classes is not None else
             torch.zeros_like(mask, dtype=torch.bool)).sum().detach()
    return g, {"counts": counts, "kappa": kappa.detach(), "extra_norm": ne.detach(),
               "control_norm": raw_nc.detach(), "code0_count": code0,
               "nc_fail": nc_fail.detach(), "floor_active": active,
               "cot_norm": cot_norm.detach()}


def inner_x_cotangent(model: TTTE2E, rule: str, logits: torch.Tensor | None,
                      y: torch.Tensor, w: torch.Tensor, z0: torch.Tensor | None = None,
                      emb_mean: torch.Tensor | None = None,
                      r_ref: float = 0.03) -> torch.Tensor:
    """Cotangent of the inner loss at the final hidden state (after ``ln_f``).

    For ``ce``/``aice``: ``cot_x = cot_logits.to(dt) @ E.to(dt)`` (= autograd through the
    tied head). For ``lin`` the [B, C, V] tensor is never built: ``cot_x = w * (mean(E) -
    E_y)`` with bf16-rounded rows accumulated in fp32 — lin's canonical write. It equals
    autograd exactly in fp32/fp64; in bf16 autograd would also round 1/V and accumulate
    the head matmul in bf16, so AICE's first write equals lin's only up to that rounding.
    """
    dt = model.compute_dtype
    E = model.wte.weight.detach()
    if rule == "lin":
        if emb_mean is None:
            emb_mean = E.to(dt).to(_acc(dt)).mean(0)
        ey = E[y].to(dt).to(_acc(dt))
        return ((emb_mean - ey) * w.unsqueeze(-1).to(ey.dtype)).to(dt)
    cot = inner_logit_cotangent(rule, logits.detach(), y, w, z0, r_ref=r_ref)
    return cot.to(dt) @ E.to(dt)


def _fast_values(model: TTTE2E, batch: int) -> list[torch.Tensor]:
    return [p.detach().unsqueeze(0).repeat(batch, 1, 1) for p in model.prime_params()]


def _past_leaves(past_chunks: list[list[tuple[torch.Tensor, torch.Tensor]]], c0: int,
                 c1: int, n_layers: int, requires_grad: bool):
    """Concatenate stored per-chunk K/V of chunks [c0, c1) per suffix layer."""
    out = []
    for j in range(n_layers):
        if c1 <= c0:
            out.append(None)
            continue
        k = torch.cat([past_chunks[c][j][0] for c in range(c0, c1)], 1)
        v = torch.cat([past_chunks[c][j][1] for c in range(c0, c1)], 1)
        if requires_grad:
            k, v = k.detach().requires_grad_(True), v.detach().requires_grad_(True)
        out.append((k, v))
    return out


def _as_past(leaves, ref: torch.Tensor, model: TTTE2E):
    """None entries -> empty [B, 0, H, d] tensors."""
    B, H, d = ref.shape[0], model.cfg.num_heads, model.cfg.head_dim
    out = []
    for e in leaves:
        if e is None:
            z = ref.new_zeros(B, 0, H, d, dtype=model.compute_dtype)
            out.append((z, z))
        else:
            out.append(e)
    return out


def _scatter_past_grads(leaves, c0: int, chunk: int, dpast: list, n_layers: int,
                        acc=torch.float32) -> None:
    """Split the grads of concatenated past leaves back into per-chunk accumulators."""
    for j in range(n_layers):
        if leaves[j] is None:
            continue
        k, v = leaves[j]
        for t, acc_idx in ((k, 0), (v, 1)):
            if t.grad is None:
                continue
            g = t.grad
            for i in range(g.shape[1] // chunk):
                c = c0 + i
                part = g[:, i * chunk:(i + 1) * chunk].to(acc)
                slot = dpast[c][j]
                if slot[acc_idx] is None:
                    slot[acc_idx] = part.clone()
                else:
                    slot[acc_idx] += part


def barrier_mask(mask: torch.Tensor, starts: list[int], chunk: int) -> torch.Tensor:
    """Boundary policy: the target that crosses a barrier (last position before
    it) is masked in the inner and the outer loss."""
    inner_starts = [s for s in starts if s > 0]
    if not inner_starts:
        return mask
    mask = mask.clone()
    for s in inner_starts:
        mask[:, s * chunk - 1] = False
    return mask


def _segment_starts(n_chunks: int, barriers: list[int] | None) -> list[int]:
    raw = list(barriers or [])
    if any(s < 0 for s in raw):
        raise ValueError("barrier chunk indices must be nonnegative")
    # MINI/P2-3 has six barriers for an 8-chunk sequence; tiny unit presets
    # have fewer chunks and simply omit barriers beyond their horizon.
    starts = sorted(set([0] + [s for s in raw if s < n_chunks]))
    return starts


# ------------------------------------------------------------------ plain references
def _plain_forward(model: TTTE2E, ids, tgt, mask, inner: InnerCfg, second_order: bool,
                   barriers=None, trunc_kv: bool = False):
    """Full-graph reference. Returns (mean loss over sequences and chunks, per-chunk [n, B]).
    ``trunc_kv`` detaches earlier chunks' K/V where later chunks read them."""
    B, T = ids.shape
    C = model.cfg.chunk
    n = T // C
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    fast = model.init_fast(B)
    fast0 = [f.detach() for f in fast]
    S = model.cfg.suffix_len
    losses = []
    past, past0 = [None] * S, [None] * S
    h = h0 = None
    for c in range(n):
        sl = slice(c * C, (c + 1) * C)
        if c in starts:
            seg_end = min([s for s in starts if s > c] + [n])
            h = model.prefix_forward(ids[:, c * C: seg_end * C])
            seg0 = c
            past, past0 = [None] * S, [None] * S
        hc = h[:, (c - seg0) * C: (c - seg0 + 1) * C]
        read = [None if p is None else (p[0].detach(), p[1].detach()) for p in past] \
            if trunc_kv else past
        x, new = model.suffix_chunk(hc, fast, _as_past(read, hc, model), math_backend=True,
                                    segment_pos=(c - seg0) * C)
        logits = model.head(x)
        nll = token_nll(logits, tgt[:, sl])
        l_b = seq_mean(nll, mask[:, sl])
        losses.append(l_b)
        past = [(torch.cat((p[0], k), 1), torch.cat((p[1], v), 1)) if p is not None else (k, v)
                for p, (k, v) in zip(past, new)]
        z0 = None
        if inner.rule == "aice":
            with torch.no_grad():
                x0, new0 = model.suffix_chunk(hc.detach(), fast0, _as_past(past0, hc, model),
                                              math_backend=False, segment_pos=(c - seg0) * C)
                z0 = model.head(x0)
            past0 = [(torch.cat((p[0], k), 1), torch.cat((p[1], v), 1)) if p is not None
                     else (k, v) for p, (k, v) in zip(past0, new0)]
        if c == n - 1:
            break
        m = mask[:, sl]
        if inner.rule in ("ce", "rp", "nc"):
            inner_b = l_b
        elif inner.rule == "lin":
            lin = logits.mean(-1) - logits.gather(-1, tgt[:, sl].unsqueeze(-1)).squeeze(-1)
            inner_b = seq_mean(lin, m)
        elif inner.rule == "aice":
            inner_b = seq_mean(token_nll(logits - z0, tgt[:, sl]), m)
        if inner.rule == "rp":
            w = m.to(_acc(logits.dtype)) / denom(m).to(_acc(logits.dtype)).unsqueeze(1)
            cot = inner_logit_cotangent("rp", logits, tgt[:, sl], w, r_ref=inner.r_ref)
            g = torch.autograd.grad(logits, fast, grad_outputs=cot,
                                    create_graph=second_order, retain_graph=True)
        elif inner.rule == "nc":
            classes = inner.classes[:, sl] if inner.classes is not None else None
            g, _ = inner_gradients(model, x, list(fast), logits, tgt[:, sl], m, inner,
                                   classes=classes, chunk=c)
        else:
            g = torch.autograd.grad(inner_b.sum(), fast, create_graph=second_order,
                                    retain_graph=True)
        g, _ = clip_per_seq(list(g), inner.clip)
        if not second_order:
            g = [x.detach() for x in g]
        fast = [f - inner.lr * x for f, x in zip(fast, g)]
    per_chunk = torch.stack(losses)
    return per_chunk.mean(), per_chunk


def so_loss_plain(model, ids, tgt, mask, inner: InnerCfg, barriers=None):
    if inner.rule not in ("ce", "rp", "nc"):
        raise ValueError("second order is defined for CE-like rules only")
    return _plain_forward(model, ids, tgt, mask, inner, True, barriers)


def fo_loss_plain(model, ids, tgt, mask, inner: InnerCfg, barriers=None, trunc_kv=False):
    return _plain_forward(model, ids, tgt, mask, inner, False, barriers, trunc_kv)


# ------------------------------------------------------------------- optimized engines
@dataclass
class StepStats:
    loss_sum: torch.Tensor          # sum over sequences of the per-sequence outer loss
    chunk_loss: torch.Tensor        # [n] mean over sequences
    inner_norm: torch.Tensor        # [n_updates, B] pre-clip inner-gradient norms
    n_seq: int
    rule_stats: dict | None = None
    fast: list[torch.Tensor] | None = None


def _segment_prefix(model: TTTE2E, ids: torch.Tensor, starts: list[int], n: int, C: int):
    parts = []
    for i, seg0 in enumerate(starts):
        seg1 = starts[i + 1] if i + 1 < len(starts) else n
        parts.append(model.prefix_forward(ids[:, seg0 * C:seg1 * C]))
    return torch.cat(parts, dim=1)


def fo_backward(model: TTTE2E, ids, tgt, mask, inner: InnerCfg, outer_scale: float, *,
                barriers: list[int] | None = None, trunc_kv: bool = False,
                fused_attn: bool = True) -> StepStats:
    """Exact first-order outer gradient (see module doc). Accumulates into ``.grad`` the
    gradient of ``outer_scale * sum_b (1/n) sum_c l_{c,b}``."""
    B, T = ids.shape
    cfg = model.cfg
    C, S = cfg.chunk, cfg.suffix_len
    n = T // C
    if n * C != T:
        raise ValueError("sequence length must be a multiple of the chunk")
    w_chunk = outer_scale / n
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    math_backend = not fused_attn
    acc = _acc(model.compute_dtype)
    h = _segment_prefix(model, ids, starts, n, C)
    h_val = h.detach()
    dh = torch.zeros(h_val.shape, dtype=acc, device=h_val.device)
    W = _fast_values(model, B)
    W0 = [w.clone() for w in W] if inner.rule == "aice" else None
    dW0 = [torch.zeros_like(p, dtype=acc) for p in model.prime_params()]
    stored_kv: list = []      # per chunk: per layer (k, v) detached
    stored_W: list = []       # per chunk: fast values used by that chunk
    stored_kv0: list = []
    seg0 = 0
    dpast = [[[None, None] for _ in range(S)] for _ in range(n)]
    chunk_losses, norms = [], []
    rule_stats = {"rp_floor_active": [], "nc_counts": {}, "nc_code0_count": 0,
                  "nc_fail": [],
                  "extra_norm": [],
                  "ce_norm": [], "induced_binding": []}
    emb_mean = None
    if inner.rule == "lin":
        dt = model.compute_dtype
        emb_mean = model.wte.weight.detach().to(dt).to(_acc(dt)).mean(0)
    for c in range(n):
        sl = slice(c * C, (c + 1) * C)
        hc = h_val[:, sl].detach().requires_grad_(True)
        Wc = [w.detach().requires_grad_(True) for w in W]
        if c in starts:
            seg0 = c
        leaves = _past_leaves(stored_kv, seg0, c, S, True)
        x, new = model.suffix_chunk(hc, Wc, _as_past(leaves, hc, model), math_backend=math_backend,
                                    segment_pos=(c - seg0) * C)
        logits = model.head(x)
        m = mask[:, sl]
        nll = token_nll(logits, tgt[:, sl])
        l_b = seq_mean(nll, m)
        chunk_losses.append(l_b.detach().mean())
        need_inner = c < n - 1
        g = None
        if need_inner and inner.rule != "ce":
            z0 = None
            if inner.rule == "aice":
                with torch.no_grad():
                    l0 = _past_leaves(stored_kv0, seg0, c, S, False)
                    x0, new0 = model.suffix_chunk(hc.detach(), W0, _as_past(l0, hc, model),
                                                  math_backend=False, segment_pos=(c - seg0) * C)
                    z0 = model.head(x0)
                    stored_kv0.append([(k.detach(), v.detach()) for k, v in new0])
            if inner.rule in ("rp", "nc"):
                classes = inner.classes[:, sl] if inner.classes is not None else None
                g, diag = inner_gradients(model, x, Wc, logits, tgt[:, sl], m, inner,
                                       z0=z0, classes=classes, chunk=c, nll=nll)
                if inner.rule == "nc":
                    rule_stats["nc_counts"].update(diag.get("counts", {}))
                    rule_stats["extra_norm"].append(diag.get("extra_norm"))
                    rule_stats["nc_code0_count"] += diag.get("code0_count", 0)
                    rule_stats["nc_fail"].append(diag.get("nc_fail"))
                if inner.rule == "rp":
                    rule_stats["extra_norm"].append(diag.get("extra_norm"))
                    rule_stats["ce_norm"].append(diag.get("ce_norm"))
                    rule_stats["induced_binding"].append(diag.get("induced_binding"))
                if inner.rule in ("rp", "nc"):
                    rule_stats["rp_floor_active"].append(diag["floor_active"])
            else:
                w = m.to(acc) / denom(m).to(acc).unsqueeze(1)
                cot = inner_x_cotangent(model, inner.rule, logits, tgt[:, sl], w, z0, emb_mean)
                g = list(torch.autograd.grad(x, Wc, grad_outputs=cot, retain_graph=True))
                if inner.rule == "rp":
                    rr = (-torch.expm1(-F.cross_entropy(logits.flatten(0, 1),
                         tgt[:, sl].flatten(), reduction="none").view_as(m))).detach()
                    rule_stats["rp_floor_active"].append((rr < inner.r_ref) & m)
            del z0
        (l_b.sum() * w_chunk).backward()
        if need_inner and inner.rule == "ce":
            g = [wc.grad / w_chunk for wc in Wc]
        del logits, nll, x
        dh[:, sl] += hc.grad.to(acc)
        for acc_, wc in zip(dW0, Wc):
            acc_ += wc.grad.to(acc).sum(0)
        _scatter_past_grads(leaves, seg0, C, dpast, S, acc)
        stored_kv.append([(k.detach(), v.detach()) for k, v in new])
        stored_W.append([w.detach() for w in W])
        if need_inner:
            gc, nrm = clip_per_seq([x.detach() for x in g], inner.clip)
            norms.append(nrm.detach())
            W = [w - inner.lr * x for w, x in zip(W, gc)]
        del Wc, leaves, new
    if not trunc_kv:
        for c in reversed(range(n - 1)):
            slots = dpast[c]
            if all(s[0] is None and s[1] is None for s in slots):
                continue
            hc = h_val[:, c * C:(c + 1) * C].detach().requires_grad_(True)
            Wc = [w.detach().requires_grad_(True) for w in stored_W[c]]
            seg0 = max(s for s in starts if s <= c)
            leaves = _past_leaves(stored_kv, seg0, c, S, True)
            _, new = model.suffix_chunk(hc, Wc, _as_past(leaves, hc, model),
                                        math_backend=math_backend, kv_only=True,
                                        segment_pos=(c - seg0) * C)
            outs, cots = [], []
            for j, (k, v) in enumerate(new):
                for t, gt in ((k, slots[j][0]), (v, slots[j][1])):
                    if gt is not None:
                        outs.append(t)
                        cots.append(gt.to(t.dtype))
            torch.autograd.backward(outs, cots)
            if hc.grad is not None:
                dh[:, c * C:(c + 1) * C] += hc.grad.to(acc)
            for acc_, wc in zip(dW0, Wc):
                if wc.grad is not None:
                    acc_ += wc.grad.to(acc).sum(0)
            _scatter_past_grads(leaves, seg0, C, dpast, S, acc)
            dpast[c] = None
    h.backward(dh.to(h.dtype))
    for p, d in zip(model.prime_params(), dW0):
        p.grad = d.to(p.dtype) if p.grad is None else p.grad + d.to(p.dtype)
    cl = torch.stack(chunk_losses)
    return StepStats(loss_sum=cl.mean() * B, chunk_loss=cl,
                     inner_norm=torch.stack(norms) if norms else torch.zeros(0, B),
                     n_seq=B, rule_stats=rule_stats, fast=[w.detach() for w in W])


def so_backward(model: TTTE2E, ids, tgt, mask, inner: InnerCfg, outer_scale: float, *,
                barriers: list[int] | None = None, fused_forward: bool = False) -> StepStats:
    """Exact second-order gradient with per-chunk rematerialization.

    Forward sweep: no outer graph; each chunk builds a local first-order graph for its
    inner gradient and stores W_c and K/V. It uses the math attention backend like the
    reverse sweep, so the stored states and their adjoint come from one numerical map
    (``fused_forward=True`` is a profiled speed option, not the default).
    Reverse sweep: rebuild chunk c with the math attention backend, take the inner
    gradient with ``create_graph=True``, and backpropagate the cotangents of
    (l_c, W_{c+1}, this chunk's K/V) into (h_c, W_c, earlier K/V, slow parameters).
    """
    if inner.rule != "ce":
        raise ValueError("second order is defined for the release rule (ce) only")
    B, T = ids.shape
    cfg = model.cfg
    C, S = cfg.chunk, cfg.suffix_len
    n = T // C
    w_chunk = outer_scale / n
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    acc = _acc(model.compute_dtype)
    h = _segment_prefix(model, ids, starts, n, C)
    h_val = h.detach()
    dh = torch.zeros(h_val.shape, dtype=acc, device=h_val.device)
    W = _fast_values(model, B)
    stored_kv, stored_W, chunk_losses, norms = [], [], [], []
    seg0 = 0
    for c in range(n):
        sl = slice(c * C, (c + 1) * C)
        stored_W.append([w.detach() for w in W])
        Wc = [w.detach().requires_grad_(True) for w in W]
        if c in starts:
            seg0 = c
        past = _as_past(_past_leaves(stored_kv, seg0, c, S, False), h_val, model)
        with torch.enable_grad():
            x, new = model.suffix_chunk(h_val[:, sl], Wc, past, math_backend=not fused_forward,
                                        segment_pos=(c - seg0) * C)
            l_b = seq_mean(token_nll(model.head(x), tgt[:, sl]), mask[:, sl])
            if c < n - 1:
                g = torch.autograd.grad(l_b.sum(), Wc)
        chunk_losses.append(l_b.detach().mean())
        stored_kv.append([(k.detach(), v.detach()) for k, v in new])
        if c < n - 1:
            gc, nrm = clip_per_seq(list(g), inner.clip)
            norms.append(nrm.detach())
            W = [w.detach() - inner.lr * x.detach() for w, x in zip(W, gc)]
        del x, new, l_b
    dpast = [[[None, None] for _ in range(S)] for _ in range(n)]
    d_fast = None
    for c in reversed(range(n)):
        sl = slice(c * C, (c + 1) * C)
        hc = h_val[:, sl].detach().requires_grad_(True)
        Wc = [w.detach().requires_grad_(True) for w in stored_W[c]]
        seg0 = max(s for s in starts if s <= c)
        leaves = _past_leaves(stored_kv, seg0, c, S, True)
        with torch.enable_grad():
            x, new = model.suffix_chunk(hc, Wc, _as_past(leaves, hc, model), math_backend=True,
                                        segment_pos=(c - seg0) * C)
            l_b = seq_mean(token_nll(model.head(x), tgt[:, sl]), mask[:, sl])
            outs, cots = [l_b.sum() * w_chunk], [None]
            if c < n - 1:
                g = torch.autograd.grad(l_b.sum(), Wc, create_graph=True)
                gc, _ = clip_per_seq(list(g), inner.clip)
                Wn = [w - inner.lr * x_ for w, x_ in zip(Wc, gc)]
                outs += Wn
                cots += d_fast
                for j, (k, v) in enumerate(new):
                    for t, gt in ((k, dpast[c][j][0]), (v, dpast[c][j][1])):
                        if gt is not None:
                            outs.append(t)
                            cots.append(gt.to(t.dtype))
            cots[0] = torch.ones_like(outs[0])
            torch.autograd.backward(outs, cots)
        dh[:, sl] += hc.grad.to(acc)
        d_fast = [wc.grad if wc.grad is not None else torch.zeros_like(wc) for wc in Wc]
        _scatter_past_grads(leaves, seg0, C, dpast, S, acc)
        dpast[c] = None
        del x, new, l_b, outs, cots
    h.backward(dh.to(h.dtype))
    for p, d in zip(model.prime_params(), d_fast):
        s = d.to(acc).sum(0).to(p.dtype)
        p.grad = s if p.grad is None else p.grad + s
    cl = torch.stack(chunk_losses)
    return StepStats(loss_sum=cl.mean() * B, chunk_loss=cl,
                     inner_norm=torch.stack(norms) if norms else torch.zeros(0, B), n_seq=B,
                     fast=[w.detach() for w in W])


# ---------------------------------------------------------------------------- eval
@torch.no_grad()
def stream_eval(model: TTTE2E, ids, tgt, mask, inner: InnerCfg, *, adapt: bool = True,
                barriers: list[int] | None = None, inner_mask: torch.Tensor | None = None,
                fused_attn: bool = True, return_fast: bool = False,
                prefix_cache: dict | None = None,
                suffix_key_masks: dict[int, torch.Tensor] | None = None,
                nll_positions: torch.Tensor | None = None,
                clip_scales: torch.Tensor | None = None,
                return_cot_norm: bool = False,
                return_ce_residual: bool = False,
                return_greedy: bool = False,
                candidate_positions: torch.Tensor | None = None,
                candidate_ids: torch.Tensor | None = None,
                fast_init: list[torch.Tensor] | None = None,
                return_sigma: bool = False,
                gather_positions: torch.Tensor | None = None,
                gather_ids: torch.Tensor | None = None) -> dict:
    """Streaming TTT evaluation. Returns per-token NLL [B, T] at pre-update weights.

    ``barriers``: chunk indices where the attention context is reset in every layer
    (prefix recomputed on the new segment, suffix K/V cleared); fast weights carry over.
    ``adapt=False``: no inner updates (W stays W_0). ``inner_mask``: tokens allowed to
    write (default ``mask``); the inner denominator stays ``mask.sum`` per chunk.
    ``return_fast`` also returns the fast weights after the last executed update.
    ``prefix_cache`` (a dict, reused across calls on the same ids and barriers) shares the
    prefix pass between evaluation modes; the prefix does not depend on W.
    ``suffix_key_masks`` optionally supplies a per-chunk [B,C,P+C] or [C,P+C]
    boolean math-attention mask, used by the echo value-mask ablation.
    ``nll_positions`` ([B, T] bool): in non-writing chunks and for ``lin``, evaluate
    the head only at these positions (NLL is 0 elsewhere).
    ``candidate_positions`` [B,Q] and ``candidate_ids`` [B,Q,K] request only gathered
    pre-update log probabilities [B,Q,K]. No logits are retained across chunks.
    ``clip_scales`` ([n-1, B]): replay reference scales instead of recomputing them.
    ``return_cot_norm`` also returns ``cot_norm`` [B, T]: the per-token write strength, the
    norm of the unweighted inner-loss cotangent at the final hidden state (0 where no write).
    ``fast_init`` (one [B, ...] tensor per prime parameter) starts the stream from these fast
    weights instead of the model's own W_0 (P2-2 swap reads); AICE's reference stays W_0.
    ``inner.tok_mult`` (G0) multiplies each CE write token's weight by a detached m_t;
    ``return_sigma`` also returns ``sigma`` [B, T], the final-norm sigma_t of every token, and
    any multiplier run returns ``mult_mean``/``mult_count`` over the write tokens
    (``sigma_shuf`` permutes over the valid tokens, so suppression modes share it).
    ``gather_positions`` [B,Q] (-1 = padding) and ``gather_ids`` [B,Q,K] (SOQ-N3B κ) also
    return ``gather_centred`` (the logit minus its vocabulary mean) and ``gather_logp``
    [B,Q,K] at pre-update weights, from a separate no-grad head call on the detached hidden
    states, so every other output is unchanged.
    """
    B, T = ids.shape
    if (gather_positions is None) != (gather_ids is None):
        raise ValueError("gather_positions and gather_ids must be supplied together")
    if gather_positions is not None and (gather_positions.shape[0] != B or gather_ids.ndim != 3
                                         or gather_ids.shape[:2] != gather_positions.shape):
        raise ValueError("gather shapes must be [B,Q] and [B,Q,K]")
    if (candidate_positions is None) != (candidate_ids is None):
        raise ValueError("candidate_positions and candidate_ids must be supplied together")
    if candidate_positions is not None:
        if candidate_positions.ndim != 2 or candidate_positions.shape[0] != B or \
                candidate_ids.ndim != 3 or candidate_ids.shape[:2] != candidate_positions.shape:
            raise ValueError("candidate shapes must be [B,Q] and [B,Q,K]")
        if candidate_positions.dtype != torch.long or candidate_ids.dtype != torch.long or \
                bool(((candidate_positions < 0) | (candidate_positions >= T)).any()) or \
                bool(((candidate_ids < 0) | (candidate_ids >= model.cfg.vocab_size)).any()):
            raise ValueError("candidate positions/ids must be int64 and in range")
        candidate_positions = candidate_positions.to(ids.device)
        candidate_ids = candidate_ids.to(ids.device)
    cfg = model.cfg
    C, S = cfg.chunk, cfg.suffix_len
    n = T // C
    acc = _acc(model.compute_dtype)
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    imask = mask if inner_mask is None else inner_mask & mask
    writes = [adapt and c < n - 1 and bool(imask[:, c * C:(c + 1) * C].any())
              for c in range(n)]
    if clip_scales is not None:
        if clip_scales.shape != (n - 1, B) or not bool(torch.isfinite(clip_scales).all()) \
                or bool((clip_scales < 0).any()):
            raise ValueError("clip_scales must be finite, nonnegative [n-1, B]")
        clip_scales = clip_scales.to(device=ids.device, dtype=acc)
    req = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        W0 = _fast_values(model, B)
        if fast_init is None:
            W = [w.clone() for w in W0]
        else:
            if [tuple(t.shape) for t in fast_init] != [tuple(w.shape) for w in W0]:
                raise ValueError("fast_init must match the model's batched prime parameters")
            W = [t.detach().to(device=w.device, dtype=w.dtype).clone() for t, w in zip(fast_init, W0)]
        out = torch.zeros(B, T, dtype=acc, device=ids.device)
        norms, upd, scales = [], [], []
        rule_diags = {"rp_floor_active": [], "nc_counts": {}, "nc_code0_count": 0,
                      "nc_fail": [],
                      "induced_binding": []}
        cot_norm = torch.zeros(B, T, dtype=acc, device=ids.device) if return_cot_norm else None
        ce_residual = torch.zeros(B, T, dtype=acc, device=ids.device) if return_ce_residual else None
        greedy = torch.zeros(B, T, dtype=torch.bool, device=ids.device) if return_greedy else None
        candidate_log_probs = (torch.empty(candidate_ids.shape, dtype=torch.float32, device=ids.device)
                               if candidate_ids is not None else None)
        if gather_positions is not None:
            gather_positions = gather_positions.to(ids.device)
            gather_ids = gather_ids.to(ids.device)
            gather_centred = torch.full(gather_ids.shape, float("nan"), dtype=acc, device=ids.device)
            gather_logp = torch.full_like(gather_centred, float("nan"))
        need_sigma = return_sigma or inner.tok_mult in SIGMA_MULTS
        sigma_all = torch.zeros(B, T, dtype=acc, device=ids.device) if return_sigma else None
        if inner.tok_mult != "none":
            mult_sum = torch.zeros((), dtype=torch.float64, device=ids.device)
            mult_count = torch.zeros((), dtype=torch.float64, device=ids.device)
        emb_mean = None
        if inner.rule == "lin":
            dt = model.compute_dtype
            emb_mean = model.wte.weight.to(dt).to(acc).mean(0)
        for c in range(n):
            sl = slice(c * C, (c + 1) * C)
            if c in starts:
                seg_end = min([s for s in starts if s > c] + [n])
                key = (c, seg_end)
                if prefix_cache is not None and key in prefix_cache:
                    h = prefix_cache[key]
                else:
                    h = model.prefix_forward(ids[:, c * C: seg_end * C])
                    if prefix_cache is not None:
                        prefix_cache[key] = h
                seg0, past, past0 = c, [], []
            hc = h[:, (c - seg0) * C: (c - seg0 + 1) * C]
            need_inner = writes[c]
            sparse_head = nll_positions is not None and (not need_inner or inner.rule == "lin")
            Wc = [w.detach().requires_grad_(need_inner) for w in W]
            with torch.enable_grad():
                cap = {} if need_sigma else None
                x, new = model.suffix_chunk(hc, Wc, _as_past(_cat(past, S), hc, model),
                                            math_backend=not fused_attn,
                                            key_mask=(suffix_key_masks.get(c)
                                                      if suffix_key_masks is not None else None),
                                            segment_pos=(c - seg0) * C, capture=cap)
                sig = residual_sigma(cap["h_pre"].detach(), cfg.rms_eps) if need_sigma else None
                if cap is not None:
                    del cap
                if sigma_all is not None:
                    sigma_all[:, sl] = sig.to(acc)
                past.append([(k.detach(), v.detach()) for k, v in new])
                logits = None
                pairs = (torch.nonzero((candidate_positions >= c * C) &
                                       (candidate_positions < (c + 1) * C), as_tuple=True)
                         if candidate_positions is not None else None)
                if sparse_head:
                    sel = nll_positions[:, sl]
                    if pairs is not None and pairs[0].numel():
                        sel = sel.clone()
                        sel[pairs[0], candidate_positions[pairs] - c * C] = True
                    if bool(sel.any()):
                        zs = model.head(x.detach()[sel])
                        tok = F.cross_entropy(zs, tgt[:, sl][sel], reduction="none")
                        out[:, sl][sel] = tok.to(acc)
                        if greedy is not None:
                            greedy[:, sl][sel] = zs.argmax(-1).eq(tgt[:, sl][sel])
                        if ce_residual is not None:
                            p = torch.softmax(zs.to(acc), dim=-1)
                            yy = tgt[:, sl][sel]
                            py = p.gather(1, yy[:, None]).squeeze(1)
                            ce_residual[:, sl][sel] = (p.square().sum(1) - 2 * py + 1).clamp_min(0).sqrt()
                        if pairs is not None and pairs[0].numel():
                            lookup = sel.flatten().long().cumsum(0).view(B, C) - 1
                            cz = zs[lookup[pairs[0], candidate_positions[pairs] - c * C]]
                else:  # no graph through the head: the inner VJP starts at x
                    logits = model.head(x.detach())
                    out[:, sl] = token_nll(logits, tgt[:, sl]).to(acc)
                    if greedy is not None:
                        greedy[:, sl] = logits.argmax(-1).eq(tgt[:, sl])
                    if ce_residual is not None:
                        p = torch.softmax(logits.to(acc), dim=-1)
                        py = p.gather(-1, tgt[:, sl].unsqueeze(-1)).squeeze(-1)
                        ce_residual[:, sl] = (p.square().sum(-1) - 2 * py + 1).clamp_min(0).sqrt()
                    if pairs is not None and pairs[0].numel():
                        cz = logits[pairs[0], candidate_positions[pairs] - c * C]
                if pairs is not None and pairs[0].numel():
                    candidate_log_probs[pairs] = F.log_softmax(cz.float(), dim=-1).gather(
                        1, candidate_ids[pairs]).detach()
                    del cz
                if gather_positions is not None:
                    gp = torch.nonzero((gather_positions >= c * C) &
                                       (gather_positions < (c + 1) * C), as_tuple=True)
                    with torch.no_grad():
                        for j0 in range(0, gp[0].numel(), 256):
                            gb, gq = gp[0][j0:j0 + 256], gp[1][j0:j0 + 256]
                            gz = model.head(x.detach()[gb, gather_positions[gb, gq] - c * C]).to(acc)
                            gid = gather_ids[gb, gq]
                            gather_centred[gb, gq] = (gz - gz.mean(-1, keepdim=True)).gather(1, gid)
                            gather_logp[gb, gq] = F.log_softmax(gz, dim=-1).gather(1, gid)
                            del gz
                z0 = None
                if inner.rule == "aice" and any(writes[c:seg_end]):
                    # Earlier non-writing chunks still supply K/V to a later AICE write.
                    with torch.no_grad():
                        x0, new0 = model.suffix_chunk(hc, W0, _as_past(_cat(past0, S), hc, model),
                                                      math_backend=not fused_attn,
                                                      segment_pos=(c - seg0) * C)
                        past0.append([(k.detach(), v.detach()) for k, v in new0])
                        if need_inner:
                            z0 = model.head(x0)
                if need_inner:
                    m = mask[:, sl]
                    im = imask[:, sl]
                    w = im.to(acc) / denom(m).to(acc).unsqueeze(1)
                    if inner.tok_mult != "none":
                        # Permute over the chunk's valid tokens (not the mode's write
                        # set) so suppression modes share one assignment.
                        mt = token_multiplier(inner, sig, m, chunk=c, dtype=acc)
                        w = w * mt
                        mult_sum = mult_sum + (mt * im.to(acc)).sum().double()
                        mult_count = mult_count + im.sum().double()
                    if inner.rule in ("rp", "nc"):
                        classes = inner.classes[:, sl] if inner.classes is not None else None
                        g, diag = inner_gradients(model, x, Wc, logits, tgt[:, sl], m, inner,
                                                   z0=z0, classes=classes, chunk=c,
                                                   inner_mask=im, nll=F.cross_entropy(
                                                       logits.flatten(0, 1), tgt[:, sl].flatten(),
                                                       reduction="none").view_as(m))
                        if inner.rule == "nc":
                            rule_diags["nc_counts"].update(diag.get("counts", {}))
                            rule_diags["nc_code0_count"] += diag.get("code0_count", 0)
                            rule_diags["nc_fail"].append(diag.get("nc_fail"))
                        cot = None
                    else:
                        cot = inner_x_cotangent(model, inner.rule, logits, tgt[:, sl], w, z0, emb_mean)
                    if cot_norm is not None:  # the cotangent is linear in the token weight
                        if cot is not None:
                            ctn = cot.to(acc).norm(dim=-1) / w.clamp_min(torch.finfo(acc).tiny)
                        else:
                            ctn = diag["cot_norm"].to(acc)
                        cot_norm[:, sl] = torch.where(w > 0, ctn, torch.zeros_like(w))
                    if cot is not None:
                        g = list(torch.autograd.grad(x, Wc, grad_outputs=cot))
                    if logits is not None:
                        r = (-torch.expm1(-F.cross_entropy(logits.flatten(0, 1),
                             tgt[:, sl].flatten(), reduction="none").view_as(m))).detach()
                        if inner.rule in ("rp", "nc"):
                            rule_diags["rp_floor_active"].append((r < inner.r_ref) & m)
                    gc, nrm = clip_per_seq(g, inner.clip)
                    scale = inner.clip / nrm.clamp_min(inner.clip) if inner.clip > 0 \
                        else torch.ones_like(nrm)
                    if clip_scales is not None:
                        scale = clip_scales[c]
                        gc = [t * scale.view(B, *([1] * (t.dim() - 1))).to(t.dtype) for t in g]
                    norms.append(nrm)
                    scales.append(scale)
                    upd.append(torch.stack([
                        sum(t.to(acc).pow(2).flatten(1).sum(1) for t in gc[3 * j: 3 * j + 3]).sqrt()
                        for j in range(S)], 1) * inner.lr)
                    W = [w_ - inner.lr * t for w_, t in zip(W, gc)]
                elif adapt and c < n - 1:
                    norms.append(torch.zeros(B, dtype=acc, device=ids.device))
                    scales.append(torch.ones(B, dtype=acc, device=ids.device))
                    upd.append(torch.zeros(B, S, dtype=acc, device=ids.device))
            del x, logits
        res = {"nll": out, "inner_norm": torch.stack(norms) if norms else
               (out.new_zeros(0, B) if adapt else None),
               "update_norm": torch.stack(upd) if upd else
               (out.new_zeros(0, B, S) if adapt else None),
               "clip_scale": torch.stack(scales) if scales else out.new_ones(n - 1, B)}
        if return_fast:
            res["fast"] = [w.detach() for w in W]
        if candidate_log_probs is not None:
            res["candidate_log_probs"] = candidate_log_probs
        if gather_positions is not None:
            res["gather_centred"], res["gather_logp"] = gather_centred, gather_logp
        if sigma_all is not None:
            res["sigma"] = sigma_all
        if inner.tok_mult != "none":
            res["mult_mean"] = float(mult_sum / mult_count.clamp_min(1.0))
            res["mult_count"] = int(mult_count.item())
        if cot_norm is not None:
            res["cot_norm"] = cot_norm
        if ce_residual is not None:
            res["ce_residual"] = ce_residual
        if greedy is not None:
            res["greedy"] = greedy
        if rule_diags["rp_floor_active"]:
            res["rp_floor_active"] = torch.stack(rule_diags["rp_floor_active"]).cpu()
        if rule_diags["nc_counts"]:
            res["nc_counts"] = rule_diags["nc_counts"]
        res["nc_code0_count"] = int(rule_diags["nc_code0_count"])
        res["nc_fail"] = (int(torch.cat(rule_diags["nc_fail"]).sum().item())
                           if rule_diags["nc_fail"] else 0)
        return res
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)


def stream_align(model: TTTE2E, ids: torch.Tensor, tgt: torch.Tensor,
                 mask: torch.Tensor, focal: torch.Tensor,
                 nonfocal: torch.Tensor, inner: InnerCfg, *,
                 barriers: list[int] | None = None, fused_attn: bool = False,
                 probe_targets: torch.Tensor | None = None,
                 return_vectors: bool = False) -> dict:
    """Run one frozen-P trajectory and return write/read gradient components.

    The three source masks are evaluated at the same pre-update fast weights.  The
    full update alone advances the trajectory; focal and non-focal gradients are
    diagnostic VJPs and never affect it.  Slow model parameters are frozen and all
    returned tensors are detached.
    """
    B, T = ids.shape
    cfg = model.cfg
    C, S = cfg.chunk, cfg.suffix_len
    n = T // C
    p0 = min(4, n)
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    focal = focal & mask
    nonfocal = nonfocal & mask
    if probe_targets is None:
        probe_targets = torch.zeros_like(mask)
    probe_targets = probe_targets & mask
    req = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    try:
        W = [w.clone() for w in _fast_values(model, B)]
        W_initial = [w.clone() for w in W]
        h = None
        past = []
        write_scales, vf, vn, vr = [], [], [], []
        for c in range(n):
            sl = slice(c * C, (c + 1) * C)
            if c in starts:
                seg_end = min([s for s in starts if s > c] + [n])
                h = model.prefix_forward(ids[:, c * C:seg_end * C])
                seg0, past = c, []
            hc = h[:, (c - seg0) * C:(c - seg0 + 1) * C]
            write = c < p0 and c < n - 1
            Wc = [w.detach().requires_grad_(write) for w in W]
            with torch.enable_grad():
                x, new = model.suffix_chunk(hc, Wc, _as_past(_cat(past, S), hc, model),
                                            math_backend=not fused_attn,
                                            segment_pos=(c - seg0) * C)
                m = mask[:, sl]
                if write:
                    logits = model.head(x.detach())
                    w = m.to(x.dtype) / denom(m).to(x.dtype).unsqueeze(1)
                    cot = inner_x_cotangent(model, inner.rule, logits, tgt[:, sl], w,
                                            None, None)
                    gfull = list(torch.autograd.grad(x, Wc, grad_outputs=cot,
                                                     retain_graph=True))
                    def masked_grad(source):
                        ww = source.to(x.dtype) / denom(m).to(x.dtype).unsqueeze(1)
                        cc = inner_x_cotangent(model, inner.rule, logits, tgt[:, sl], ww,
                                               None, None)
                        return list(torch.autograd.grad(x, Wc, grad_outputs=cc,
                                                        retain_graph=True, allow_unused=False))
                    gf = masked_grad(focal[:, sl])
                    gn = masked_grad(nonfocal[:, sl])
                    full_clipped, norm = clip_per_seq(gfull, inner.clip)
                    scale = inner.clip / norm.clamp_min(inner.clip) if inner.clip > 0 else torch.ones_like(norm)
                    write_scales.append(scale.detach())
                    vf.append(-inner.lr * scale[:, None] *
                              torch.cat([q.reshape(B, -1) for q in gf], 1))
                    vn.append(-inner.lr * scale[:, None] *
                              torch.cat([q.reshape(B, -1) for q in gn], 1))
                    if return_vectors:
                        vr.append(-inner.lr * scale[:, None] *
                                  (torch.cat([q.reshape(B, -1) for q in gfull], 1) -
                                   torch.cat([q.reshape(B, -1) for q in gf], 1) -
                                   torch.cat([q.reshape(B, -1) for q in gn], 1)))
                    W = [w_ - inner.lr * q for w_, q in zip(W, full_clipped)]
            past.append([(k.detach(), v.detach()) for k, v in new])
            del x, new
        # Re-read P with the same barrier segmentation as stream_eval.  In MINI,
        # P is chunks 4--7, split into 4--5 and 6--7 by the chunk-6 barrier.
        # Keeping one prefix/past stream here would give g_r the wrong function.
        if p0 not in starts:
            raise ValueError("stream_align requires a barrier at the P start")
        read_W = [w.detach().requires_grad_(True) for w in W]
        past = []
        h = None
        seg0 = None
        total = None
        for c in range(p0, n):
            if c in starts:
                seg_end = min([s for s in starts if s > c] + [n])
                h = model.prefix_forward(ids[:, c * C:seg_end * C])
                seg0, past = c, []
            sl = slice(c * C, (c + 1) * C)
            hc = h[:, (c - seg0) * C:(c - seg0 + 1) * C]
            with torch.enable_grad():
                x, new = model.suffix_chunk(hc, read_W, _as_past(_cat(past, S), hc, model),
                                            math_backend=not fused_attn,
                                            segment_pos=(c - seg0) * C)
                q = probe_targets[:, c * C:(c + 1) * C]
                if bool(q.any()):
                    # Probe targets are sparse; avoid materialising a full-vocabulary
                    # head for the other P positions.
                    val = F.cross_entropy(model.head(x[q]), tgt[:, sl][q], reduction="sum")
                    total = val if total is None else total + val
            past.append([(k, v) for k, v in new])
        if total is None:
            gr = [torch.zeros_like(w) for w in read_W]
        else:
            gr = list(torch.autograd.grad(total, read_W, allow_unused=True))
            gr = [g if g is not None else torch.zeros_like(w) for g, w in zip(gr, read_W)]
        gr_flat = torch.cat([g.reshape(B, -1) for g in gr], 1)
        vf_t, vn_t = torch.stack(vf, 1), torch.stack(vn, 1)
        grn = gr_flat.norm(dim=1).clamp_min(torch.finfo(gr_flat.dtype).tiny)
        vfn, vnn = vf_t.norm(dim=2), vn_t.norm(dim=2)
        cosf = (vf_t * (-gr_flat[:, None, :])).sum(2) / (vfn * grn[:, None]).clamp_min(1e-30)
        cosn = (vn_t * (-gr_flat[:, None, :])).sum(2) / (vnn * grn[:, None]).clamp_min(1e-30)
        result = {"vf": vfn.detach(), "vn": vnn.detach(), "cosf": cosf.detach(),
                  "cosn": cosn.detach(), "bf": (vf_t * (-gr_flat[:, None, :])).sum(2).sum(1).detach(),
                  "bn": (vn_t * (-gr_flat[:, None, :])).sum(2).sum(1).detach(),
                  "gr": grn.detach(), "scales": torch.stack(write_scales, 0).detach()}
        if return_vectors:
            vv = torch.stack(vr, 1)
            result.update({"vf_vec": vf_t.detach(), "vn_vec": vn_t.detach(),
                           "vr_vec": vv.detach(), "full_vec": (vf_t + vn_t + vv).detach(),
                           "full_applied": (torch.cat([w.reshape(B, -1) for w in W], 1) -
                                             torch.cat([w.reshape(B, -1) for w in W_initial], 1)).detach()})
        return result
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)


def stream_gain_split(model: TTTE2E, ids: torch.Tensor, tgt: torch.Tensor,
                      mask: torch.Tensor, sel_focal: torch.Tensor, sel_nonfocal: torch.Tensor,
                      inner: InnerCfg, *, barriers: list[int] | None = None,
                      fused_attn: bool = False) -> dict:
    """G0: per-token log-chain of the CE write at selected source tokens.

    Runs the frozen-P trajectory of ``stream_align`` (CE writes in chunks < min(4, n),
    each advancing W by the full clipped chunk update) and, at every selected token t of a
    write chunk, factorises the norm of its unweighted per-token write gradient
    g_t = d CE_t / d W (all fast tensors, pre-clip) exactly:

        ||g_t|| = e_t * u_t * gam_t * proj_t * s_t * R_t

    e = ||p - e_y|| (output error), u = ||c0 E|| / e (tied-unembedding pull-back),
    gam = ||gamma_f * c1|| / ||c1||, proj = ||RMSNorm projection|| / ||gamma_f * c1||,
    s = 1 / sigma_t (final-norm sigma), R = ||g_t|| / ||c_h|| (the body: suffix blocks and
    prime activations).  c_h, the cotangent at the residual before ``ln_f``, is computed in
    closed form and checked against autograd (``ch_err``).  Descriptive body terms per
    suffix block j: ||g^(j)||, the cotangent norms at the prime post-norm output (q_j) and
    input (p_j), and sigma_p of the last block's prime output at t (position-local there).
    Slow parameters are frozen; nothing here changes the trajectory.
    """
    if inner.rule != "ce" or inner.tok_mult != "none":
        raise ValueError("stream_gain_split measures the plain CE write")
    B, T = ids.shape
    cfg = model.cfg
    C, S = cfg.chunk, cfg.suffix_len
    n = T // C
    p0 = min(4, n)
    starts = _segment_starts(n, barriers)
    mask = barrier_mask(mask, starts, C)
    sel = (sel_focal | sel_nonfocal) & mask
    acc = _acc(model.compute_dtype)
    E = model.wte.weight.detach().to(acc)
    gam_f = model.ln_f.weight.detach().to(acc)
    eps = cfg.rms_eps
    req = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    rec = {k: [] for k in ("row", "pos", "chunk", "focal", "e", "u", "gam", "proj", "s",
                           "gnorm", "gblk", "cq", "cp", "sigp_last", "H", "nll", "ch_err",
                           "clip_scale", "pre_clip_norm")}
    try:
        W = [w.clone() for w in _fast_values(model, B)]
        h = None
        past = []
        for c in range(p0):
            if c >= n - 1:
                break
            sl = slice(c * C, (c + 1) * C)
            if c in starts:
                seg_end = min([s_ for s_ in starts if s_ > c] + [n])
                h = model.prefix_forward(ids[:, c * C:seg_end * C])
                seg0, past = c, []
            hc = h[:, (c - seg0) * C:(c - seg0 + 1) * C]
            Wc = [w.detach().requires_grad_(True) for w in W]
            cap: dict = {}
            with torch.enable_grad():
                x, new = model.suffix_chunk(hc, Wc, _as_past(_cat(past, S), hc, model),
                                            math_backend=not fused_attn,
                                            segment_pos=(c - seg0) * C, capture=cap)
                logits = model.head(x.detach())
                m = mask[:, sl]
                w = m.to(acc) / denom(m).to(acc).unsqueeze(1)
                cot = inner_x_cotangent(model, "ce", logits, tgt[:, sl], w)
                gfull = list(torch.autograd.grad(x, Wc, grad_outputs=cot, retain_graph=True))
                full_clipped, norm = clip_per_seq(gfull, inner.clip)
                scale = (inner.clip / norm.clamp_min(inner.clip) if inner.clip > 0
                         else torch.ones_like(norm))
                inputs = (list(Wc) + [cap["h_pre"]] + [cap[("q", j)] for j in range(S)] +
                          [cap[("p", j)] for j in range(S)])
                local = [torch.nonzero(sel[b, sl], as_tuple=False).flatten() for b in range(B)]
                slots = max((int(t.numel()) for t in local), default=0)
                for k in range(slots):
                    rb = [b for b in range(B) if local[b].numel() > k]
                    tb = [int(local[b][k]) for b in rb]
                    rb_t = torch.as_tensor(rb, device=ids.device)
                    tb_t = torch.as_tensor(tb, device=ids.device)
                    z = logits[rb_t, tb_t].to(acc)
                    y = tgt[:, sl][rb_t, tb_t]
                    pr = torch.softmax(z, -1)
                    c0 = pr.clone()
                    c0[torch.arange(len(rb), device=ids.device), y] -= 1.0
                    e = c0.norm(dim=-1)
                    c1 = c0 @ E
                    c2 = gam_f * c1
                    hp = cap["h_pre"][rb_t, tb_t].detach().to(acc)
                    sig = (hp.pow(2).mean(-1, keepdim=True) + eps).sqrt()
                    xh = hp / sig
                    c3 = c2 - xh * (c2 * xh).mean(-1, keepdim=True)
                    ch = c3 / sig
                    go = torch.zeros_like(x)
                    go[rb_t, tb_t] = c1.to(x.dtype)
                    grads = torch.autograd.grad(x, inputs, grad_outputs=go, retain_graph=True,
                                                allow_unused=True)
                    gW = grads[:3 * S]
                    gh = grads[3 * S]
                    gq = grads[3 * S + 1:3 * S + 1 + S]
                    gp = grads[3 * S + 1 + S:]
                    blk = torch.stack([
                        torch.sqrt(sum(t[rb_t].to(acc).pow(2).flatten(1).sum(1)
                                       for t in gW[3 * j:3 * j + 3])) for j in range(S)], 1)
                    gnorm = blk.pow(2).sum(1).sqrt()
                    ch_auto = gh[rb_t, tb_t].to(acc)
                    cq = torch.stack([gq[j][rb_t].to(acc).flatten(1).norm(dim=1)
                                      for j in range(S)], 1)
                    cp = torch.stack([gp[j][rb_t].to(acc).flatten(1).norm(dim=1)
                                      for j in range(S)], 1)
                    plast = cap[("p", S - 1)][rb_t, tb_t].detach().to(acc)
                    rec["row"].append(rb_t); rec["pos"].append(tb_t + c * C)
                    rec["chunk"].append(torch.full_like(rb_t, c))
                    rec["focal"].append(sel_focal[rb_t, tb_t + c * C])
                    rec["e"].append(e); rec["u"].append(c1.norm(dim=-1) / e)
                    rec["gam"].append(c2.norm(dim=-1) / c1.norm(dim=-1))
                    rec["proj"].append(c3.norm(dim=-1) / c2.norm(dim=-1))
                    rec["s"].append(1.0 / sig.squeeze(-1)); rec["gnorm"].append(gnorm)
                    rec["gblk"].append(blk); rec["cq"].append(cq); rec["cp"].append(cp)
                    rec["sigp_last"].append((plast.pow(2).mean(-1) + eps).sqrt())
                    rec["H"].append(-(pr * torch.log(pr.clamp_min(torch.finfo(acc).tiny))).sum(-1))
                    rec["nll"].append(-torch.log(pr[torch.arange(len(rb), device=ids.device), y]
                                                 .clamp_min(torch.finfo(acc).tiny)))
                    rec["ch_err"].append((ch_auto - ch).norm(dim=-1) /
                                         ch.norm(dim=-1).clamp_min(torch.finfo(acc).tiny))
                    rec["clip_scale"].append(scale[rb_t].to(acc))
                    rec["pre_clip_norm"].append(norm[rb_t].to(acc))
                    del grads, gW, gh, gq, gp, go
                W = [w_ - inner.lr * q for w_, q in zip(W, full_clipped)]
            past.append([(k_.detach(), v_.detach()) for k_, v_ in new])
            del x, new, logits, cap, inputs, gfull, full_clipped
        out = {}
        for k, v in rec.items():
            out[k] = (torch.cat(v).detach().cpu() if v else
                      torch.zeros((0, S) if k in ("gblk", "cq", "cp") else (0,)))
        return out
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)


def _cat(past_chunks: list, S: int):
    if not past_chunks:
        return [None] * S
    return [(torch.cat([pc[j][0] for pc in past_chunks], 1),
             torch.cat([pc[j][1] for pc in past_chunks], 1)) for j in range(S)]
