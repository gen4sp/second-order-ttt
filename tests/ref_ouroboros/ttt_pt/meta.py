"""Inner-loop (test-time training) and sequence-level loss for TTT-E2E.

Meta mode per sequence (mirrors MetaModel.loss_for_sequence, train_mode="meta"):
  1. Run prefix layers over the full sequence once.
  2. Initialize per-sequence fast weights from the meta-learned prime init.
  3. For each chunk of `mini_batch_size` tokens:
       a. forward suffix layers -> chunk CE loss (this pre-update loss is the
          outer objective contribution for the chunk),
       b. grad of chunk loss w.r.t. fast weights (create_graph=True during
          meta-training for second-order E2E gradients),
       c. per-sequence global-norm clip, then SGD step on the fast weights.
  4. Outer loss = mean over chunks of the pre-update chunk losses.

Fast weights carry a leading batch dim: each sequence in the batch evolves its
own weights (the JAX code achieves this via vmap over sequences).
"""

from __future__ import annotations

import torch

from ttt_pt.config import Config
from ttt_pt.model import TTTModel


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    """Per-sample CE normalized by valid token count.

    logits [B, T, V] fp32, targets [B, T], mask [B, T] -> (loss [B], token_nll [B, T])

    Uses F.cross_entropy (fused log_softmax, supports double backward) instead of
    materializing a second [B, T, V] fp32 log-prob tensor.
    """
    B, T, V = logits.shape
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="none"
    ).view(B, T)
    m = mask.float()
    valid = m.sum(-1).clamp_min(1e-10)
    loss = (token_nll * m).sum(-1) / valid
    return loss, token_nll


def clip_per_sample(grads: list[torch.Tensor], clip: float) -> list[torch.Tensor]:
    """optax clip_by_global_norm applied per sequence (vmap semantics):
    g * clip / max(norm, clip), differentiable for second-order grads."""
    B = grads[0].shape[0]
    sq = sum(g.float().pow(2).reshape(B, -1).sum(-1) for g in grads)
    norm = sq.sqrt()
    scale = clip / norm.clamp_min(clip)  # [B]
    return [g * scale.view(B, *([1] * (g.dim() - 1))).to(g.dtype) for g in grads]


def ilr_multiplier(step: int, cfg: Config) -> float:
    """Inner-LR warmup multiplier (MetaModel.get_ilr_multiplier)."""
    t = cfg.training
    if t.ilr_warmup_steps == 0 or t.optimizer_inner.lr == 0.0:
        return 1.0
    progress = min(1.0, (step + 1) / t.ilr_warmup_steps)
    ilr = t.ilr_init + (t.optimizer_inner.lr - t.ilr_init) * progress
    return ilr / t.optimizer_inner.lr


def loss_for_sequence_meta(
    model: TTTModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    ilr_mult: float,
    cfg: Config,
    create_graph: bool = True,
    return_token_nll: bool = False,
    fast_init: list | None = None,
    return_fast: bool = False,
    outer_chunk_mask: list | None = None,
):
    """TTT-E2E loss for a batch of sequences. Returns (loss, aux dict).

    fast_init: carry fast weights in from a previous sequence (eval only).
    return_fast: run the last chunk's update too and return final fast weights
    in aux["fast"] (eval only), for cross-sequence carry experiments.
    outer_chunk_mask: per-chunk flag deciding whether a chunk contributes to
    the OUTER loss. The inner update always runs. This is what closed-loop
    meta-training needs: a self-generated chunk must be written into the fast
    weights, because that is the deployment behaviour being trained for, but it
    must not be an outer target -- optimising the model to predict its own
    samples is the degeneracy, not the fix. None keeps every chunk, which is
    the exogenous objective and must stay bit-identical.
    """
    mcfg = model.cfg
    B, T = input_ids.shape
    CS = mcfg.mini_batch_size
    assert T % CS == 0, f"seq len {T} must be divisible by chunk {CS}"
    n_chunks = T // CS
    inner_lr = cfg.training.optimizer_inner.lr * ilr_mult
    inner_clip = cfg.training.optimizer_inner.clip_gradient

    # During eval (create_graph=False) only the fast weights need gradients:
    # run the prefix in no_grad and keep each chunk's graph local, detaching
    # the carries after every inner update. Same math, O(1-chunk) memory.
    with torch.enable_grad() if create_graph else torch.no_grad():
        h = model.prefix_forward(input_ids)  # [B, T, D]

    fast = model.init_fast_weights(B) if fast_init is None else list(fast_init)
    if not create_graph:
        fast = [f.detach().requires_grad_(True) for f in fast]
    kv = model.init_kv_caches(B, input_ids.device)

    chunk_losses = []
    chunk_losses_b = []
    token_nlls = [] if return_token_nll else None
    for c in range(n_chunks):
        sl = slice(c * CS, (c + 1) * CS)
        with torch.enable_grad():
            logits, new_kv = model.suffix_chunk_forward(h[:, sl], fast, kv, chunk_id=c)
            loss_b, token_nll = masked_ce(logits, targets[:, sl], loss_mask[:, sl])

            # JAX updates on every chunk; the last update never affects the
            # outer loss (it depends only on pre-update losses), so skip it
            # unless the caller wants the final fast weights.
            if c < n_chunks - 1 or return_fast:
                grads = torch.autograd.grad(loss_b.sum(), fast, create_graph=create_graph)
                if inner_clip > 0:
                    grads = clip_per_sample(grads, inner_clip)
                fast = [f - inner_lr * g for f, g in zip(fast, grads)]

        keep = outer_chunk_mask is None or outer_chunk_mask[c]
        if create_graph:
            if keep:
                chunk_losses.append(loss_b.mean())
            kv = new_kv
        else:
            if keep:
                chunk_losses.append(loss_b.mean().detach())
            if keep:
                chunk_losses_b.append(loss_b.detach())
            kv = [(k.detach(), v.detach()) for k, v in new_kv]
            fast = [f.detach().requires_grad_(True) for f in fast]
        if return_token_nll:
            token_nlls.append(token_nll.detach() if not create_graph else token_nll)

    loss = torch.stack(chunk_losses).mean()
    aux = {"chunk_losses": torch.stack(chunk_losses).detach()}
    if chunk_losses_b:
        aux["chunk_losses_b"] = torch.stack(chunk_losses_b)  # [n_chunks, B]
    if return_fast:
        aux["fast"] = [f.detach() for f in fast]
    if return_token_nll:
        aux["token_nll"] = torch.cat(token_nlls, dim=1).detach()
    return loss, aux


def loss_for_sequence_pretrain(
    model: TTTModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    return_token_nll: bool = False,
):
    """Plain LM loss (FA baseline)."""
    logits = model.forward_plain(input_ids)
    loss_b, token_nll = masked_ce(logits, targets, loss_mask)
    aux = {}
    if return_token_nll:
        aux["token_nll"] = token_nll.detach()
    return loss_b.mean(), aux


def compute_loss(
    model,
    batch_tokens,
    cfg: Config,
    step: int,
    create_graph=True,
    return_token_nll=False,
    use_remat=True,
    outer_chunk_mask=None,
    inner_block_chunks=1,
    inner_block_parallel=False,
    inner_first_order=False,
):
    """batch_tokens: [B, T+1] int64. Splits into inputs/targets and dispatches."""
    input_ids = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    loss_mask = targets != cfg.model.bos_token_id
    if cfg.training.train_mode == "meta":
        if inner_block_chunks != 1 or inner_block_parallel or inner_first_order:
            if outer_chunk_mask is not None or return_token_nll:
                raise ValueError(
                    "experimental block inner loop does not support "
                    "masked outer chunks or token NLL"
                )
            from ttt_pt.block_inner import loss_for_sequence_block

            return loss_for_sequence_block(
                model,
                input_ids,
                targets,
                loss_mask,
                ilr_multiplier(step, cfg),
                cfg,
                block_chunks=inner_block_chunks,
                create_graph=create_graph,
                parallel_read=inner_block_parallel,
                first_order=inner_first_order,
            )
        if use_remat and create_graph and not return_token_nll:
            assert outer_chunk_mask is None, (
                "the remat scan accumulates every chunk loss inside a custom "
                "autograd Function; masking the outer loss there means editing "
                "a second-order checkpointed backward, which is where a silent "
                "error would be hardest to see. Closed-loop training runs the "
                "plain path at a shorter sequence length instead."
            )
            from ttt_pt.remat import loss_for_sequence_meta_remat

            return loss_for_sequence_meta_remat(
                model, input_ids, targets, loss_mask, ilr_multiplier(step, cfg), cfg
            )
        return loss_for_sequence_meta(
            model,
            input_ids,
            targets,
            loss_mask,
            ilr_multiplier(step, cfg),
            cfg,
            outer_chunk_mask=outer_chunk_mask,
            create_graph=create_graph,
            return_token_nll=return_token_nll,
        )
    return loss_for_sequence_pretrain(
        model, input_ids, targets, loss_mask, return_token_nll=return_token_nll
    )
