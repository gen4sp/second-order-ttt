"""Value-label-free B9 address auxiliary and deterministic filler sham."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

B9_BASE = {"e2e1_b9": "e2e1", "e2e1_lin_b9": "e2e1_lin", "e2e1_lin_b9f": "e2e1_lin"}
LAMBDA = .1


def gather(u, pos):
    return u.gather(1, pos[..., None].expand(-1, -1, u.shape[-1]))


def sham_pairs(idx, n_keys, chunk):
    """First-occurrence filler candidates; chunk-1 anchors recur first in chunks 5–6.

    Return padded pairs and masks, vectorised over batch, ids and tokens. Missing
    candidate ids are excluded from the softmax; missing anchors are excluded
    from the same global anchor-mean reduction used by the key auxiliary.
    """
    batch, horizon = idx.shape
    fillers = torch.arange(2+2*n_keys, 2+3*n_keys, device=idx.device)
    positions = torch.arange(horizon, device=idx.device)
    match = idx[:, None, :] == fillers[None, :, None]
    first = torch.where(match, positions, horizon).amin(-1)
    candidate_valid = first < horizon
    recurrent = match & (positions[None, None, :] > first[..., None]) & \
        ((positions // chunk == 5) | (positions // chunk == 6))[None, None, :]
    anchor = torch.where(recurrent, positions, horizon).amin(-1)
    valid = candidate_valid & (first // chunk == 1) & (anchor < horizon)
    return first.clamp_max(horizon-1), anchor.clamp_max(horizon-1), candidate_valid, valid


def key_pairs(meta, device):
    write = torch.as_tensor(meta["write_pos"], device=device, dtype=torch.long)
    query = torch.as_tensor(meta["query_pos"], device=device, dtype=torch.long)
    mask = torch.ones_like(write, dtype=torch.bool)
    return write, query, mask, mask


def dot_logits(u, write, query):
    return torch.bmm(gather(u, query), gather(u, write).transpose(1, 2))/math.sqrt(u.shape[-1])


def loss_from_u(u, pairs):
    write, query, candidate_valid, anchor_valid = pairs
    if not bool(anchor_valid.any()) or not bool(candidate_valid.any(1).all()):
        raise ValueError("B9 auxiliary has an empty anchor/candidate set")
    logits = dot_logits(u, write, query).masked_fill(~candidate_valid[:, None, :], -torch.inf)
    target = torch.arange(write.shape[1], device=u.device).expand(len(u), -1)
    # Selecting anchors before CE avoids 0*inf for missing filler candidates.
    loss = F.cross_entropy(logits[anchor_valid], target[anchor_valid])
    w, q = gather(u, write), gather(u, query)
    cosine = torch.bmm(F.normalize(q, dim=-1), F.normalize(w, dim=-1).transpose(1, 2))
    cosine = cosine.masked_fill(~candidate_valid[:, None, :], -torch.inf)
    with torch.no_grad():
        lags = (query-write)[anchor_valid]
        metrics = dict(loss=float(loss.detach()),
                       dot_addr=float((logits.argmax(-1)[anchor_valid] == target[anchor_valid]).float().mean()),
                       cosine_addr=float((cosine.argmax(-1)[anchor_valid] == target[anchor_valid]).float().mean()),
                       anchor_count=float(anchor_valid.sum(1).float().median()),
                       lag=float(lags.float().median()), anchors_total=int(anchor_valid.sum()))
    return loss, metrics


def auxiliary(model, h, idx, meta, chunk, sham=False, log_grad=False):
    """Uses trained affine LN(h), with gradients through both endpoints."""
    u = model.head.ln(h)
    pairs = sham_pairs(idx, meta["write_pos"].shape[1], chunk) if sham else key_pairs(meta, idx.device)
    loss, metrics = loss_from_u(u, pairs)
    if log_grad:
        # Compute both terms on the very same model/batch, before its update.
        # Reference gradients never reach .grad; only the chosen term joins CE.
        other, other_metrics = loss_from_u(u, key_pairs(meta, idx.device) if sham else
                                          sham_pairs(idx, meta["write_pos"].shape[1], chunk))
        params = tuple(model.parameters())
        def grad_norm(term):
            grads = torch.autograd.grad(LAMBDA*term, params, retain_graph=True, allow_unused=True)
            values = [g.detach().square().sum() for g in grads if g is not None]
            return float(torch.stack(values).sum().sqrt())
        norm, reference_norm = grad_norm(loss), grad_norm(other)
        key_metrics, filler_metrics = (other_metrics, metrics) if sham else (metrics, other_metrics)
        key_norm, sham_norm = (reference_norm, norm) if sham else (norm, reference_norm)
        metrics.update(grad_norm=norm, key_grad_norm=key_norm, sham_grad_norm=sham_norm,
                       grad_ratio=sham_norm/key_norm if key_norm > 0 else None,
                       key_loss=key_metrics["loss"], sham_loss=filler_metrics["loss"],
                       key_anchor_count=key_metrics["anchor_count"],
                       sham_anchor_count=filler_metrics["anchor_count"],
                       key_lag=key_metrics["lag"], sham_lag=filler_metrics["lag"],
                       lag_ratio=filler_metrics["lag"]/key_metrics["lag"] if key_metrics["lag"] > 0 else None,
                       lambda_aux=LAMBDA)
        model.b9_last_metrics = metrics
    return loss
