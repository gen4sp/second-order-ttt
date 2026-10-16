"""Rematerialized (checkpointed) inner-loop scan for TTT-E2E meta-training.

Equivalent to JAX `scan_remat_chunk` with inner_remat_freq=1: the forward pass
runs the chunk scan without building a graph, storing only the carries (fast
weights + KV caches) at each chunk boundary. The backward pass replays chunks
in reverse, rebuilding each chunk's second-order graph one at a time, and
propagates cotangents through the scan carry. Exact gradients, O(1-chunk)
graph memory instead of O(n_chunks).
"""

from __future__ import annotations

import torch

from ttt_pt.meta import clip_per_sample, masked_ce


def _chunk_step(
    model,
    h_c,
    fast,
    kv_flat,
    chunk_id,
    targets_c,
    mask_c,
    inner_lr,
    inner_clip,
    do_update,
    create_graph=True,
):
    """One inner-loop step. kv is passed flattened [k0,v0,k1,v1,...]."""
    kv = [(kv_flat[2 * i], kv_flat[2 * i + 1]) for i in range(len(kv_flat) // 2)]
    logits, new_kv = model.suffix_chunk_forward(h_c, fast, kv, chunk_id)
    loss_b, _ = masked_ce(logits, targets_c, mask_c)
    loss_c = loss_b.mean()
    if do_update:
        grads = torch.autograd.grad(loss_b.sum(), fast, create_graph=create_graph)
        if inner_clip > 0:
            grads = clip_per_sample(grads, inner_clip)
        new_fast = [f - inner_lr * g for f, g in zip(fast, grads)]
    else:
        new_fast = fast
    new_kv_flat = [t for pair in new_kv for t in pair]
    return loss_c, new_fast, new_kv_flat


class ChunkScan(torch.autograd.Function):
    """Checkpointed scan over chunks. Non-tensor context is passed via `meta` dict."""

    @staticmethod
    def forward(ctx, meta, h, *flat):
        model = meta["model"]
        n_chunks = meta["n_chunks"]
        CS = model.cfg.mini_batch_size
        n_fast = meta["n_fast"]
        n_kv = meta["n_kv"]

        fast = list(flat[:n_fast])
        kv_flat = list(flat[n_fast : n_fast + n_kv])
        params = list(flat[n_fast + n_kv :])

        carries_fast, carries_kv = [], []
        chunk_losses = []
        fast = [f.detach() for f in fast]
        kv_flat = [t.detach() for t in kv_flat]
        for c in range(n_chunks):
            carries_fast.append(fast)
            carries_kv.append(kv_flat)
            # The inner update needs first-order grads w.r.t. fast weights, so
            # each chunk builds a local graph (create_graph=False) and we drop
            # it immediately by detaching the carries.
            fast_leaves = [f.requires_grad_(True) for f in [f.clone() for f in fast]]
            with torch.enable_grad():
                loss_c, new_fast, new_kv_flat = _chunk_step(
                    model,
                    h[:, c * CS : (c + 1) * CS],
                    fast_leaves,
                    kv_flat,
                    c,
                    meta["targets"][:, c * CS : (c + 1) * CS],
                    meta["mask"][:, c * CS : (c + 1) * CS],
                    meta["inner_lr"],
                    meta["inner_clip"],
                    do_update=c < n_chunks - 1,
                    create_graph=False,
                )
            fast = [f.detach() for f in new_fast]
            kv_flat = [t.detach() for t in new_kv_flat]
            chunk_losses.append(loss_c.detach())

        ctx.meta = meta
        ctx.carries_fast = carries_fast
        ctx.carries_kv = carries_kv
        ctx.save_for_backward(h, *params)
        return torch.stack(chunk_losses)

    @staticmethod
    def backward(ctx, d_losses):
        meta = ctx.meta
        model = meta["model"]
        n_chunks = meta["n_chunks"]
        CS = model.cfg.mini_batch_size
        h = ctx.saved_tensors[0]
        params = list(ctx.saved_tensors[1:])

        d_fast = None  # cotangent w.r.t. fast entering chunk c+1 (None => zeros)
        d_kv = None
        dh = torch.zeros_like(h)
        d_params = [torch.zeros_like(p) for p in params]

        for c in reversed(range(n_chunks)):
            fast_c = [f.detach().requires_grad_(True) for f in ctx.carries_fast[c]]
            kv_c = [t.detach().requires_grad_(True) for t in ctx.carries_kv[c]]
            h_c = h[:, c * CS : (c + 1) * CS].detach().requires_grad_(True)

            with torch.enable_grad():
                loss_c, new_fast, new_kv = _chunk_step(
                    model,
                    h_c,
                    fast_c,
                    kv_c,
                    c,
                    meta["targets"][:, c * CS : (c + 1) * CS],
                    meta["mask"][:, c * CS : (c + 1) * CS],
                    meta["inner_lr"],
                    meta["inner_clip"],
                    do_update=c < n_chunks - 1,
                )

            outputs = [loss_c]
            cotangents = [d_losses[c]]
            if c < n_chunks - 1:
                outputs += new_fast
                cotangents += d_fast
                # KV caches: the first chunk(s) of cache are zeros rolled out; grads
                # flow through the concatenated cache tensors.
                outputs += new_kv
                cotangents += d_kv

            inputs = fast_c + kv_c + [h_c] + params
            grads = torch.autograd.grad(outputs, inputs, grad_outputs=cotangents, allow_unused=True)
            n_f, n_k = len(fast_c), len(kv_c)
            d_fast = [
                g if g is not None else torch.zeros_like(t) for g, t in zip(grads[:n_f], fast_c)
            ]
            d_kv = [
                g if g is not None else torch.zeros_like(t)
                for g, t in zip(grads[n_f : n_f + n_k], kv_c)
            ]
            g_h = grads[n_f + n_k]
            if g_h is not None:
                dh[:, c * CS : (c + 1) * CS] = g_h
            for i, g in enumerate(grads[n_f + n_k + 1 :]):
                if g is not None:
                    d_params[i] += g

        d_kv0 = [None] * len(d_kv)  # initial caches are zeros buffers, no grad needed
        return (None, dh, *d_fast, *d_kv0, *d_params)


def suffix_scan_params(model) -> list[torch.Tensor]:
    """All params used inside the suffix scan EXCEPT feed_forward_prime
    (prime enters via the fast-weight init path) : suffix block attn/ffn/norms,
    ln_f, and wte (logits head)."""
    out = []
    for blk in model.suffix_blocks():
        for name, p in blk.named_parameters():
            if "feed_forward_prime" not in name:
                out.append(p)
    out.extend(model.ln_f.parameters())
    out.append(model.wte.weight)
    return out


def loss_for_sequence_meta_remat(model, input_ids, targets, loss_mask, ilr_mult, cfg):
    """Memory-efficient TTT-E2E loss (training path). Returns (loss, aux)."""
    mcfg = model.cfg
    B, T = input_ids.shape
    CS = mcfg.mini_batch_size
    assert T % CS == 0
    n_chunks = T // CS

    h = model.prefix_forward(input_ids)

    fast = model.init_fast_weights(B)
    kv = model.init_kv_caches(B, input_ids.device)
    kv_flat = [t for pair in kv for t in pair]
    params = suffix_scan_params(model)

    meta = {
        "model": model,
        "n_chunks": n_chunks,
        "n_fast": len(fast),
        "n_kv": len(kv_flat),
        "targets": targets,
        "mask": loss_mask,
        "inner_lr": cfg.training.optimizer_inner.lr * ilr_mult,
        "inner_clip": cfg.training.optimizer_inner.clip_gradient,
    }
    chunk_losses = ChunkScan.apply(meta, h, *fast, *kv_flat, *params)
    return chunk_losses.mean(), {"chunk_losses": chunk_losses.detach()}
