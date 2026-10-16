"""Experimental block-Jacobi inner update shared by training and streaming.

Within a block, every chunk reads the same fast weights while causal KV still
advances. The block's *mean* weighted loss produces one clipped gradient and
one write. Thus the update operator is identical in outer training and at
test time. Block boundaries remain sequential; this is not an exact parallel
implementation of the original per-chunk SGD recurrence.
"""

from __future__ import annotations

import torch

from ttt_pt.meta import clip_per_sample, masked_ce
from ttt_pt.model import COMPUTE_DTYPE, apply_rope, sdpa
from ttt_pt.stream import StreamState


def block_step(fast, weighted_loss, lr, clip, create_graph, retain_graph=None):
    grads = torch.autograd.grad(
        weighted_loss,
        fast,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    if clip > 0:
        grads = clip_per_sample(grads, clip)
    return [f - lr * g for f, g in zip(fast, grads)]


def suffix_block_forward(model, h_block, fast, kv, start_chunk_id, *, double_backward):
    """Evaluate fixed-weight causal suffix attention for a whole chunk block.

    This is the actual chunk-parallel training read path. It uses one masked
    attention call per suffix layer, followed by one block backward. The cache
    stays pre-RoPE, as in the original per-chunk implementation.
    """
    B, T, _ = h_block.shape
    CS = model.cfg.mini_batch_size
    W = model.cfg.sliding_window_size
    if T % CS:
        raise ValueError("block token length must be a multiple of chunk size")
    if W + T > model.rope_cos.shape[0]:
        raise ValueError("model RoPE table is too short for this block")
    start = start_chunk_id * CS
    qi = torch.arange(start, start + T, device=h_block.device)[:, None]
    ki = torch.arange(start - W, start + T, device=h_block.device)[None, :]
    mask = (qi >= ki) & (qi < ki + W) & (ki >= 0)
    x = h_block
    new_kv = []
    for j, blk in enumerate(model.suffix_blocks()):
        h = blk.seq_norm(x) if model.cfg.pre_norm else x
        q, k_new, v_new = blk.attn.project_qkv(h)
        k_prev, v_prev = kv[j]
        k = torch.cat((k_prev, k_new), dim=1)
        v = torch.cat((v_prev, v_new), dim=1)
        new_kv.append((k[:, -W:], v[:, -W:]))
        qr = apply_rope(q, model.rope_cos[W : W + T], model.rope_sin[W : W + T])
        kr = apply_rope(k, model.rope_cos[: W + T], model.rope_sin[: W + T])
        a = blk.attn.wo(
            sdpa(qr, kr, v, attn_mask=mask[None, None], double_backward=double_backward)
        )
        if model.cfg.post_norm:
            a = blk.seq_post_norm(a)
        x = x + a
        pw = fast[3 * j : 3 * j + 3] if model.cfg.prime else None
        x = blk.forward_ffn_part(x, prime_weights=pw)
    x = model.ln_f(x)
    logits = x.to(COMPUTE_DTYPE) @ model.wte.weight.to(COMPUTE_DTYPE).T
    return logits.float(), new_kv


def loss_for_sequence_block(
    model,
    input_ids,
    targets,
    loss_mask,
    ilr_mult,
    cfg,
    block_chunks: int,
    create_graph: bool = True,
    return_fast: bool = False,
    parallel_read: bool = False,
    first_order: bool = False,
):
    """Outer-training objective with the same block update as BlockStreamState.

    ``first_order`` stops the outer gradient through the inner gradient while
    preserving the identical forward update at training and test time. This
    may permit fused attention in outer training, but changes the meta-gradient
    and requires a separate quality study. A production long-context path also
    needs rematerialization.
    """
    if block_chunks < 1:
        raise ValueError("block_chunks must be positive")
    B, T = input_ids.shape
    CS = cfg.model.mini_batch_size
    if T % CS:
        raise ValueError("sequence length must be divisible by chunk size")
    n_chunks = T // CS
    with torch.enable_grad() if create_graph else torch.no_grad():
        h = model.prefix_forward(input_ids)
    fast = model.init_fast_weights(B)
    if not create_graph:
        fast = [f.detach().requires_grad_(True) for f in fast]
    kv = model.init_kv_caches(B, input_ids.device)
    losses = []
    for start in range(0, n_chunks, block_chunks):
        stop = min(start + block_chunks, n_chunks)
        block_losses = []
        with torch.enable_grad():
            if parallel_read:
                sl_block = slice(start * CS, stop * CS)
                block_logits, kv = suffix_block_forward(
                    model,
                    h[:, sl_block],
                    fast,
                    kv,
                    start,
                    double_backward=create_graph and not first_order,
                )
            for c in range(start, stop):
                sl = slice(c * CS, (c + 1) * CS)
                if parallel_read:
                    logits = block_logits[:, (c - start) * CS : (c - start + 1) * CS]
                else:
                    logits, kv = model.suffix_chunk_forward(h[:, sl], fast, kv, c)
                loss_b, _ = masked_ce(logits, targets[:, sl], loss_mask[:, sl])
                block_losses.append(loss_b)
                losses.append(loss_b.mean())
            if stop < n_chunks or return_fast:
                objective = torch.stack(block_losses).sum(0).sum() / len(block_losses)
                fast = block_step(
                    fast,
                    objective,
                    cfg.training.optimizer_inner.lr * ilr_mult,
                    cfg.training.optimizer_inner.clip_gradient,
                    create_graph and not first_order,
                    retain_graph=create_graph,
                )
        if not create_graph:
            kv = [(k.detach(), v.detach()) for k, v in kv]
            fast = [f.detach().requires_grad_(True) for f in fast]
    loss = torch.stack(losses).mean()
    if not create_graph:
        loss = loss.detach()
    return loss, {
        "chunk_losses": torch.stack([x.detach() for x in losses]),
        "fast": [f.detach() for f in fast] if return_fast else None,
    }


class BlockStreamState(StreamState):
    """Streaming read with one delayed update per fixed-size chunk block."""

    def __init__(
        self,
        model,
        B,
        device,
        block_chunks: int,
        parallel_update: bool = False,
        defer_updates: bool = False,
    ):
        if block_chunks < 1:
            raise ValueError("block_chunks must be positive")
        super().__init__(model, B, device)
        self.block_chunks = block_chunks
        self.parallel_update = parallel_update
        self.defer_updates = defer_updates
        self.pending_blocks = []
        self.fast_version = 0
        self._block = []
        self._block_start_kv = None
        self._block_lr_mult = None

    def snapshot(self):
        snap = super().snapshot()
        snap["block"] = [
            (h.clone(), targets.clone(), weight, chunk_id)
            for h, targets, weight, chunk_id in self._block
        ]
        snap["block_start_kv"] = (
            [(k.clone(), v.clone()) for k, v in self._block_start_kv]
            if self._block_start_kv is not None
            else None
        )
        snap["block_lr_mult"] = self._block_lr_mult
        snap["pending_blocks"] = [[d.clone() for d in delta] for delta in self.pending_blocks]
        snap["fast_version"] = self.fast_version
        return snap

    def restore(self, snap):
        super().restore(snap)
        self._block = [
            (h.clone(), targets.clone(), weight, chunk_id)
            for h, targets, weight, chunk_id in snap["block"]
        ]
        self._block_start_kv = (
            [(k.clone(), v.clone()) for k, v in snap["block_start_kv"]]
            if snap["block_start_kv"] is not None
            else None
        )
        self._block_lr_mult = snap["block_lr_mult"]
        self.pending_blocks = [[d.clone() for d in delta] for delta in snap["pending_blocks"]]
        self.fast_version = snap["fast_version"]

    def process_real_chunk(self, tokens_in, tokens_tgt, provenance_w, ilr_mult, cfg, **kwargs):
        # Reject before prefix_chunk advances KV or global_pos. The matching
        # guard in commit_chunk also covers token-wise generation callers.
        if self.defer_updates and self.pending_blocks and provenance_w:
            raise RuntimeError(
                "settle the pending block before proposing another write; "
                "read-only chunks may continue with provenance_w=0"
            )
        return super().process_real_chunk(
            tokens_in,
            tokens_tgt,
            provenance_w,
            ilr_mult,
            cfg,
            **kwargs,
        )

    def commit_chunk(
        self,
        targets,
        provenance_w,
        ilr_mult,
        cfg,
        token_w=None,
        lam=1.0,
        grad_at_init=False,
        drift_cap=0.0,
        logit_hook=None,
    ):
        if token_w is not None or lam != 1 or grad_at_init or drift_cap:
            raise ValueError(
                "experimental block mode does not support token weighting or other update rules"
            )
        if self.defer_updates and self.pending_blocks and provenance_w:
            raise RuntimeError(
                "settle the pending block before proposing another write; "
                "read-only chunks may continue with provenance_w=0"
            )
        if not self._block:
            self._block_start_kv = [(k.detach(), v.detach()) for k, v in self.suf_kv]
            self._block_lr_mult = ilr_mult
        elif ilr_mult != self._block_lr_mult:
            raise ValueError("inner learning-rate multiplier must be fixed within a block")
        h = torch.cat(self.h_buffer, dim=1).detach()
        self._block.append((h, targets.detach(), float(provenance_w), self.chunk_id))
        # Read and advance KV at the unchanged weights; do not write yet.
        nll = super().commit_chunk(
            targets,
            0.0,
            ilr_mult,
            cfg,
            logit_hook=logit_hook,
        )
        if len(self._block) == self.block_chunks:
            self.flush_block(cfg)
        return nll

    def flush_block(self, cfg):
        """Apply a partial final block explicitly at end of stream, if needed."""
        if not self._block:
            return
        block = self._block
        if any(weight for _, _, weight, _ in block):
            fast = [f.detach().requires_grad_(True) for f in self.fast]
            kv = self._block_start_kv
            with torch.enable_grad():
                terms = []
                if self.parallel_update:
                    block_logits, _ = suffix_block_forward(
                        self.model,
                        torch.cat([x[0] for x in block], dim=1),
                        fast,
                        kv,
                        block[0][3],
                        double_backward=False,
                    )
                for h, targets, weight, chunk_id in block:
                    if self.parallel_update:
                        offset = (chunk_id - block[0][3]) * self.CS
                        logits = block_logits[:, offset : offset + self.CS]
                    else:
                        logits, kv = self.model.suffix_chunk_forward(h, fast, kv, chunk_id)
                    mask = targets != cfg.model.bos_token_id
                    loss_b, _ = masked_ce(logits, targets, mask)
                    terms.append(loss_b * weight)
                objective = torch.stack(terms).sum(0).sum() / len(block)
                updated = block_step(
                    fast,
                    objective,
                    cfg.training.optimizer_inner.lr * self._block_lr_mult,
                    cfg.training.optimizer_inner.clip_gradient,
                    False,
                )
            if self.defer_updates:
                self.pending_blocks.append(
                    [(u.detach() - f.detach()) for u, f in zip(updated, fast)]
                )
            else:
                self.fast = [f.detach() for f in updated]
                self.fast_version += 1
        self._block = []
        self._block_start_kv = None
        self._block_lr_mult = None

    @torch.no_grad()
    def settle_on_external(
        self, tokens_in, tokens_tgt, cfg, doses=(0.5, 1.0), margin=0.0, max_branches=4
    ):
        """Score one held block before consuming independent evidence text.

        Returns the selected dose per row and all validation losses. This
        method deliberately does not process the evidence chunk or report its
        NLL as an independent outcome. The caller must do those separately.
        """
        if not self.defer_updates:
            raise ValueError("settlement requires defer_updates=True")
        if self._block:
            raise ValueError("settle only at a block boundary; flush the partial block first")
        if not self.pending_blocks:
            return None
        if len(self.pending_blocks) != 1:
            raise RuntimeError("atomic settlement requires exactly one pending block")
        from ttt_pt.parallel_probe import score_block_settlement

        version = self.fast_version
        dose, scores, candidate_delta = score_block_settlement(
            self,
            tokens_in,
            tokens_tgt,
            self.pending_blocks,
            cfg,
            doses=doses,
            margin=margin,
            max_branches=max_branches,
        )
        if self.fast_version != version:
            raise RuntimeError("fast-weight base changed during candidate validation")
        if torch.any(dose > 0):
            self.fast = [
                (f + dose.reshape(self.B, *([1] * (f.ndim - 1))) * d).detach()
                for f, d in zip(self.fast, candidate_delta)
            ]
            self.fast_version += 1
        count = len(self.pending_blocks)
        self.pending_blocks = []
        return {
            "dose": dose,
            "scores": scores,
            "candidates": count,
            "base_version": version,
            "committed_version": self.fast_version,
        }
