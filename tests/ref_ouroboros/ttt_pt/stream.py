"""Streaming TTT processor: chunk-wise stream consumption with fast-weight
state, plus incremental single-token decoding for generation.

Prefix layers: sliding-window attention processed chunk-by-chunk with a rolling
post-RoPE KV cache (global positions) — mathematically identical to the
full-sequence SWA pass. Suffix layers: the standard pre-RoPE window cache and
chunk machinery from model.py, extended with a token-by-token partial-chunk
path for generation. Inner updates run at chunk boundaries exactly as in
training (pre-update weights process the chunk; update after), with a
per-chunk provenance weight deciding whether the chunk may write.
"""

from __future__ import annotations

import torch

from ttt_pt.meta import clip_per_sample, masked_ce
from ttt_pt.model import COMPUTE_DTYPE, TTTModel, apply_rope, sdpa


def sample_nucleus(probs, top_p, generator=None, row_seeds=None, token_index=0):
    """Sample with the decoder's exact nucleus rule.

    `row_seeds` makes each batch row a function of its stable book/replicate/
    chunk identity rather than its current batch position. The legacy shared
    generator remains the default so existing protocols are unchanged.
    """
    sp, si = torch.sort(probs, descending=True, dim=-1)
    keep = sp.cumsum(-1) - sp < top_p
    sp = sp * keep
    sp = sp / sp.sum(-1, keepdim=True)
    if row_seeds is None:
        # A CUDA generator keeps the full [B, vocab] distribution on device.
        # The legacy CPU generator remains supported for bitwise-compatible
        # reproduction of existing runs.
        sample_probs = (
            sp if generator is not None and str(generator.device).startswith("cuda") else sp.cpu()
        )
        idx = torch.multinomial(sample_probs, 1, generator=generator).to(probs.device)
    else:
        assert len(row_seeds) == probs.shape[0]
        ranks = []
        generator_device = (
            probs.device
            if generator is not None and str(generator.device).startswith("cuda")
            else "cpu"
        )
        for row, base in enumerate(row_seeds):
            g = torch.Generator(device=generator_device)
            # All factors are odd and the result is bounded to torch's seed range.
            g.manual_seed((int(base) * 1000003 + int(token_index) * 9176 + 104729) % (2**63 - 1))
            row_probs = sp[row] if str(generator_device).startswith("cuda") else sp[row].cpu()
            ranks.append(torch.multinomial(row_probs, 1, generator=g))
        idx = torch.stack(ranks).to(probs.device)
    return si.gather(-1, idx), keep, sp, si


class StreamState:
    """Per-stream state for a batch of B parallel streams."""

    def __init__(self, model: TTTModel, B: int, device):
        cfg = model.cfg
        H = cfg.num_attention_heads
        d = cfg.hidden_size // H
        self.model = model
        self.B = B
        self.device = device
        self.W = cfg.sliding_window_size
        self.CS = cfg.mini_batch_size

        # prefix: rolling post-RoPE caches, one (k, v) per prefix layer
        def z():
            return torch.zeros(B, 0, H, d, device=device, dtype=COMPUTE_DTYPE)

        self.pre_kv = [(z(), z()) for _ in range(model.n_prefix)]
        # suffix: pre-RoPE window caches + partial-chunk buffers
        self.suf_kv = model.init_kv_caches(B, device)
        self.suf_partial = [(z(), z()) for _ in range(cfg.suffix_len)]
        self.fast = [f.detach() for f in model.init_fast_weights(B)]
        self.fast_init = [f.clone() for f in self.fast]  # pristine W0 snapshot
        self.chunk_id = 0
        self.global_pos = 0
        self.pending = []  # deltas held for deferred verification
        self.h_buffer = []  # prefix outputs of current partial chunk (for update)
        self.tok_buffer = []  # token ids of current partial chunk

    # ------------------------------------------------------- branch control

    def snapshot(self):
        """Full state snapshot so a probe can be evaluated on a branch and
        discarded (probe text must not enter the stream's KV caches)."""
        return {
            "pre_kv": [(k.clone(), v.clone()) for k, v in self.pre_kv],
            "suf_kv": [(k.clone(), v.clone()) for k, v in self.suf_kv],
            "suf_partial": [(k.clone(), v.clone()) for k, v in self.suf_partial],
            "fast": [f.clone() for f in self.fast],
            # pending must branch with everything else: probe_score() snapshots,
            # runs a w=0 chunk and restores, and a w=0 chunk appends nothing --
            # but a probe taken while updates are held would otherwise leave the
            # buffer aliased to the live list.
            "pending": [[d.clone() for d in delta] for delta in self.pending],
            "ref_kv": (
                [(k.clone(), v.clone()) for k, v in self.ref_kv]
                if getattr(self, "ref_kv", None) is not None
                else None
            ),
            "chunk_id": self.chunk_id,
            "global_pos": self.global_pos,
            "h_buffer": list(self.h_buffer),
            "tok_buffer": list(self.tok_buffer),
        }

    def restore(self, snap):
        self.pre_kv = [(k.clone(), v.clone()) for k, v in snap["pre_kv"]]
        self.suf_kv = [(k.clone(), v.clone()) for k, v in snap["suf_kv"]]
        self.suf_partial = [(k.clone(), v.clone()) for k, v in snap["suf_partial"]]
        self.fast = [f.clone() for f in snap["fast"]]
        self.pending = [[d.clone() for d in delta] for delta in snap.get("pending", [])]
        self.ref_kv = (
            [(k.clone(), v.clone()) for k, v in snap["ref_kv"]]
            if snap.get("ref_kv") is not None
            else None
        )
        self.chunk_id = snap["chunk_id"]
        self.global_pos = snap["global_pos"]
        self.h_buffer = list(snap["h_buffer"])
        self.tok_buffer = list(snap["tok_buffer"])

    # ------------------------------------------------------------ prefix

    def _prefix_attn_bulk(self, blk, h, pos0):
        """h [B, T, D] chunk through one prefix layer's SWA with rolling cache."""
        i = self._layer_i
        cs = h.shape[1]
        q, k, v = blk.attn.project_qkv(h)
        cos, sin = self.model.rope_cos, self.model.rope_sin
        pos = torch.arange(pos0, pos0 + cs, device=h.device)
        q = apply_rope(q, cos[pos], sin[pos])
        k = apply_rope(k, cos[pos], sin[pos])
        ck, cv = self.pre_kv[i]
        nc = ck.shape[1]
        k_all = torch.cat([ck, k], 1)
        v_all = torch.cat([cv, v], 1)
        qpos = pos[:, None]
        kpos = torch.arange(pos0 - nc, pos0 + cs, device=h.device)[None, :]
        mask = (qpos >= kpos) & (qpos - kpos < self.W)
        o = sdpa(q, k_all, v_all, attn_mask=mask[None, None])
        self.pre_kv[i] = (k_all[:, -self.W :], v_all[:, -self.W :])
        return blk.attn.wo(o)

    def prefix_chunk(self, input_ids):
        """input_ids [B, T] -> prefix hidden states [B, T, D], caches advanced."""
        x = self.model.embed(input_ids)
        for i in range(self.model.n_prefix):
            blk = self.model.layers[i]
            self._layer_i = i
            h = blk.seq_norm(x)
            a = self._prefix_attn_bulk(blk, h, self.global_pos)
            a = blk.seq_post_norm(a)
            x = x + a
            x = blk.forward_ffn_part(x)
        return x

    # ------------------------------------------------------------ suffix

    def _suffix_attn_step(self, j, blk, h):
        """One-token suffix attention. h [B, 1, D]. Window semantics identical
        to sw_causal_mask: query at partial position p attends cache entries
        j > p (and j >= W - chunk_id*CS) plus all partial entries."""
        q, k_new, v_new = blk.attn.project_qkv(h)
        pk, pv = self.suf_partial[j]
        pk = torch.cat([pk, k_new], 1)
        pv = torch.cat([pv, v_new], 1)
        self.suf_partial[j] = (pk, pv)
        ck, cv = self.suf_kv[j]
        p = pk.shape[1] - 1  # current partial index
        W, CS = self.W, self.CS
        cos, sin = self.model.rope_cos, self.model.rope_sin

        k_all = torch.cat([ck, pk], 1)  # [B, W+p+1, H, d] pre-RoPE
        pos_k = torch.arange(W + p + 1, device=h.device)
        k_all = apply_rope(k_all, cos[pos_k], sin[pos_k])
        qr = apply_rope(
            q,
            cos[torch.tensor([W + p], device=h.device)],
            sin[torch.tensor([W + p], device=h.device)],
        )
        jj = torch.arange(W + p + 1, device=h.device)
        cache_part = (jj < W) & ((jj > p) & (jj >= W - self.chunk_id * CS))
        partial_part = jj >= W
        mask = (cache_part | partial_part)[None, None, None, :]
        o = sdpa(qr, k_all, torch.cat([cv, pv], 1), attn_mask=mask)
        return blk.attn.wo(o)

    def suffix_token(self, h, fast=None):
        """h [B, 1, D] prefix output -> logits [B, V] (single decode step).
        fast: optional fast-weight override (e.g. pristine W0 for the
        open-loop frozen-generator control)."""
        m = self.model
        fw = fast if fast is not None else self.fast
        x = h
        for j, blk in enumerate(m.suffix_blocks()):
            hn = blk.seq_norm(x)
            a = self._suffix_attn_step(j, blk, hn)
            a = blk.seq_post_norm(a)
            x = x + a
            pw = fw[3 * j : 3 * j + 3] if m.cfg.prime else None
            x = blk.forward_ffn_part(x, prime_weights=pw)
        x = m.ln_f(x)
        logits = x.to(COMPUTE_DTYPE) @ m.wte.weight.to(COMPUTE_DTYPE).T
        return logits[:, 0].float()

    # ------------------------------------------------------- chunk update

    def commit_chunk(
        self,
        targets,
        provenance_w: float,
        ilr_mult: float,
        cfg,
        token_w: torch.Tensor | None = None,
        lam: float = 1.0,
        grad_at_init: bool = False,
        drift_cap: float = 0.0,
        logit_hook=None,
    ):
        """Finish the current chunk: inner update (if allowed) using the
        standard training path, advance suffix caches, reset partial buffers.

        targets [B, CS]: next-token targets for the chunk's positions.
        token_w [B, CS] (optional): per-token write weights multiplying the
        inner loss (e.g. repetition-aware reweighting); the reported NLL is
        always unweighted. Returns per-sample chunk NLL [B].

        grad_at_init evaluates the update direction at the meta-learned init
        rather than at the running state: Delta_t = -eta * grad_W l(x_t; W_0)
        instead of grad at W_t. The chunk still *reads* through W_t -- only the
        gradient's reference point moves -- so this costs a second suffix
        forward. What it buys is that Delta_t no longer depends on M_t, which is
        the precondition for computing every chunk's update in parallel.
        """
        m = self.model
        h_chunk = torch.cat(self.h_buffer, 1)  # [B, CS, D]
        assert h_chunk.shape[1] == self.CS
        fast = [f.requires_grad_(True) for f in [f.clone() for f in self.fast]]
        with torch.enable_grad():
            logits, new_kv = m.suffix_chunk_forward(h_chunk, fast, self.suf_kv, self.chunk_id)
            if logit_hook is not None:
                # Pure observer: reads the pre-softmax logits (e.g. to score an
                # external memory's logit bias against the same forward), never
                # alters them, so every existing path is bit-identical with or
                # without a hook installed.
                with torch.no_grad():
                    logit_hook(logits.detach(), targets)
            mask = targets != cfg.model.bos_token_id
            loss_b, token_nll = masked_ce(logits, targets, mask)
            # Pure observation: the chunk's own loss under the state that is
            # about to write it. This is the headroom the read path has not
            # already taken, and unlike a prospective-advantage gate it is free
            # -- the forward pass has happened either way. Recorded, never read
            # back by any update path, so every existing result is unchanged.
            self.last_nll = float(loss_b.detach().mean())
            self.last_token_nll = token_nll.detach()
            if provenance_w > 0.0:
                inner_lr = cfg.training.optimizer_inner.lr * ilr_mult * provenance_w
                if grad_at_init:
                    # Same chunk, same targets, but the loss is re-evaluated at
                    # the reference point. That is W_0 by default; setting
                    # self.grad_ref every k chunks instead gives a block-serial
                    # rule, parallel within a block and sequential across blocks.
                    #
                    # The reference forward runs on its OWN kv cache, carried
                    # forward by reference-weight forwards only. Reusing
                    # self.suf_kv here would leave the update depending on M_t
                    # through the cache -- only the first suffix layer's cache is
                    # fast-weight-free, since every later layer reads the FFN
                    # output of the one before -- so the update would not be
                    # chunk-independent and could not be computed in parallel,
                    # which is the entire point of evaluating at W_0. It would
                    # also feed the reference weights a context built by
                    # different weights than their own.
                    ref = getattr(self, "grad_ref", None) or self.fast_init
                    if getattr(self, "ref_kv", None) is None:
                        self.ref_kv = m.init_kv_caches(self.B, self.device)
                    wrt = [f.requires_grad_(True) for f in [f.clone() for f in ref]]
                    ref_logits, ref_new_kv = m.suffix_chunk_forward(
                        h_chunk, wrt, self.ref_kv, self.chunk_id
                    )
                    self._pending_ref_kv = [(k.detach(), v.detach()) for k, v in ref_new_kv]
                    upd_b, upd_nll = masked_ce(ref_logits, targets, mask)
                else:
                    wrt, upd_b, upd_nll = fast, loss_b, token_nll
                if token_w is not None:
                    mf = mask.float() * token_w.to(upd_nll.dtype)
                    upd_loss = (upd_nll * mf).sum(-1) / mask.float().sum(-1).clamp_min(1e-10)
                else:
                    upd_loss = upd_b
                grads = torch.autograd.grad(upd_loss.sum(), wrt)
                raw_sq = sum(g.float().flatten(1).pow(2).sum(-1) for g in grads)
                raw_norm = raw_sq.sqrt()
                clip = cfg.training.optimizer_inner.clip_gradient
                if clip > 0:
                    grads = clip_per_sample(grads, clip)
                clipped_sq = sum(g.float().flatten(1).pow(2).sum(-1) for g in grads)
                clipped_norm = clipped_sq.sqrt()
                delta = [(-inner_lr * g).detach() for g in grads]
                self.last_update_stats = {
                    "raw_grad_norm": raw_norm.detach().cpu().tolist(),
                    "clip_factor": (clipped_norm / raw_norm.clamp_min(1e-30))
                    .detach()
                    .cpu()
                    .tolist(),
                    "update_norm": (clipped_norm * inner_lr).detach().cpu().tolist(),
                }
                if getattr(self, "defer", False):
                    # Deferred-verification writing: hold the update instead of
                    # applying it, so a later probe on real text can decide
                    # whether it earned its place. Because the held updates are
                    # never applied, the chunks after this one are generated from
                    # an unchanged state -- which is what stops the loop from
                    # manufacturing the next, worse batch while the verdict is
                    # pending. commit_pending() applies the ones that survive.
                    self.pending.append(delta)
                else:
                    self.fast = [(f + d).detach() for f, d in zip(fast, delta)]
            else:
                self.last_update_stats = {
                    "raw_grad_norm": [0.0] * self.B,
                    "clip_factor": [0.0] * self.B,
                    "update_norm": [0.0] * self.B,
                }
        if lam < 1.0:  # forgetting factor: decay toward the meta-learned init
            self.fast = [(i0 + lam * (f - i0)).detach() for f, i0 in zip(self.fast, self.fast_init)]
        if drift_cap > 0:
            # Project the accumulated memory back inside a ball. The frozen
            # reference rule fails by overshoot rather than by pointing the wrong
            # way -- its drift ends 2.7x above the sequential rule's -- and a cap
            # on ||M|| is one global operation on the accumulated sum, so it can
            # be applied after a parallel scan rather than at every step.
            num = sum(float(((f - i0) ** 2).sum()) for f, i0 in zip(self.fast, self.fast_init))
            den = sum(float((i0**2).sum()) for i0 in self.fast_init)
            d = (num / den) ** 0.5
            if d > drift_cap:
                sc = drift_cap / d
                self.fast = [
                    (i0 + sc * (f - i0)).detach() for f, i0 in zip(self.fast, self.fast_init)
                ]
        self.suf_kv = [(k.detach(), v.detach()) for k, v in new_kv]
        if getattr(self, "_pending_ref_kv", None) is not None:
            self.ref_kv = self._pending_ref_kv
            self._pending_ref_kv = None
        z = self.suf_partial[0][0][:, :0]
        self.suf_partial = [(z, z) for _ in range(m.cfg.suffix_len)]
        self.h_buffer, self.tok_buffer = [], []
        self.chunk_id += 1
        return loss_b.detach()

    def probe_score(self, tokens_in, tokens_tgt, cfg):
        """Clean-text NLL of the current state, without disturbing the stream."""
        snap = self.snapshot()
        nll = self.process_real_chunk(tokens_in, tokens_tgt, 0.0, 1.0, cfg)
        self.restore(snap)
        return float(nll.mean())

    def commit_pending(self, keep):
        """Apply held updates whose index is in `keep`; discard the rest.

        The held deltas were all computed against the same state, so dropping
        one does not invalidate the others -- the property that makes selective
        commit possible here and impossible for the sequential rule, where every
        update was taken on weights the previous update had already moved.
        """
        for i, delta in enumerate(self.pending):
            if i in keep:
                self.fast = [(f + d).detach() for f, d in zip(self.fast, delta)]
        n = len(self.pending)
        self.pending = []
        return n

    # ------------------------------------------------------------ drivers

    @torch.no_grad()
    def process_real_chunk(
        self,
        tokens_in,
        tokens_tgt,
        provenance_w,
        ilr_mult,
        cfg,
        lam=1.0,
        grad_at_init=False,
        drift_cap=0.0,
        logit_hook=None,
        token_w=None,
    ):
        """Teacher-forced chunk. tokens_in/tokens_tgt [B, CS].

        token_w [B, CS] weights the inner loss per position, exactly as it
        already does for generated chunks. Passing None is the previous
        behaviour bit-for-bit; it exists so that part of a chunk can be held
        out of its own update and then scored, which is a within-chunk
        substitute for a probe the deployment may not have.
        """
        h = self.prefix_chunk(tokens_in)
        self.global_pos += self.CS
        self.h_buffer = [h]
        nll = self.commit_chunk(
            tokens_tgt,
            provenance_w,
            ilr_mult,
            cfg,
            lam=lam,
            grad_at_init=grad_at_init,
            drift_cap=drift_cap,
            logit_hook=logit_hook,
            token_w=token_w,
        )
        return nll

    @torch.no_grad()
    def generate_chunk(
        self,
        first_input,
        provenance_w,
        ilr_mult,
        cfg,
        temperature=1.0,
        top_p=0.95,
        seed=None,
        weight_fn=None,
        gen_fast=None,
        lam=1.0,
        row_seeds=None,
        observer=None,
        sampling_device="cpu",
    ):
        """Generate CS tokens starting from `first_input` [B, 1] (the token whose
        prediction begins the chunk). Returns generated ids [B, CS]."""
        generator_device = first_input.device if sampling_device == "cuda" else "cpu"
        g = torch.Generator(device=generator_device)
        if seed is not None:
            g.manual_seed(seed)
        cur = first_input
        out = []
        for token_index in range(self.CS):
            h = self.prefix_chunk(cur)
            self.global_pos += 1
            self.h_buffer.append(h)
            logits = self.suffix_token(h, fast=gen_fast) / max(temperature, 1e-6)
            probs = torch.softmax(logits, -1)
            nxt, keep, sp, si = sample_nucleus(
                probs, top_p, generator=g, row_seeds=row_seeds, token_index=token_index
            )
            if observer is not None:
                observer(
                    token_index,
                    logits.detach(),
                    probs.detach(),
                    keep.detach(),
                    sp.detach(),
                    si.detach(),
                    nxt.detach(),
                )
            out.append(nxt)
            cur = nxt
        gen = torch.cat(out, 1)  # [B, CS]
        # chunk inputs were [first_input, gen[:, :-1]]; targets are gen
        token_w = weight_fn(gen) if weight_fn is not None else None
        self.commit_chunk(gen, provenance_w, ilr_mult, cfg, token_w=token_w, lam=lam)
        return gen
