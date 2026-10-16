"""PyTorch port of TTT-E2E (End-to-End Test-Time Training for Long Context).

Faithful to the official JAX implementation (test-time-training/e2e):
- Params kept in fp32, matmuls computed in bf16 (promote-at-use).
- Interleaved-pair RoPE (complex formulation), QK-RMSNorm before RoPE.
- Blocks: pre-norm and post-norm around attention / prime-FFN / FFN sublayers.
- Prefix layers: full sliding-window attention over the whole sequence.
- Suffix layers: chunk-wise attention with a KV cache of `sliding_window_size`
  tokens stored PRE-RoPE; RoPE re-applied each chunk with window-local positions.
- Suffix layers carry a `feed_forward_prime` SwiGLU whose weights are the
  TTT fast weights, updated by inner-loop SGD (see meta.py). The prime weights
  are passed functionally with a leading batch dim so each sequence in the
  batch evolves its own fast weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttt_pt.config import ModelConfig

COMPUTE_DTYPE = torch.bfloat16


# ---------------------------------------------------------------- primitives


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm computed in fp32, output cast back to input dtype."""
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dt)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class NormalLinear(nn.Module):
    """Linear with weight stored [in, out] (JAX layout), computed x @ W in bf16."""

    def __init__(self, in_features: int, out_features: int, std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * std)

    def forward(self, x):
        return x.to(COMPUTE_DTYPE) @ self.weight.to(COMPUTE_DTYPE)


def swiglu(x, w1, w2, w3):
    """Functional SwiGLU: w2(silu(w1 x) * (w3 x)). Weights [.., D, I] / [.., I, D]."""
    x = x.to(COMPUTE_DTYPE)
    w1, w2, w3 = (w.to(COMPUTE_DTYPE) for w in (w1, w2, w3))
    if w1.dim() == 3:  # batched fast weights: [B, D, I]
        z1 = torch.einsum("btd,bdi->bti", x, w1)
        z3 = torch.einsum("btd,bdi->bti", x, w3)
        return torch.einsum("bti,bid->btd", F.silu(z1) * z3, w2)
    return (F.silu(x @ w1) * (x @ w3)) @ w2


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        std = cfg.initializer_range
        self.w1 = NormalLinear(cfg.hidden_size, cfg.intermediate_size, std)
        self.w2 = NormalLinear(cfg.intermediate_size, cfg.hidden_size, std)
        self.w3 = NormalLinear(cfg.hidden_size, cfg.intermediate_size, std)

    def forward(self, x):
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


# ---------------------------------------------------------------------- RoPE


def precompute_freqs(head_dim: int, end: int, theta: float, device=None):
    """Returns cos, sin of shape [end, head_dim // 2] in fp32."""
    inv = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotary embedding, matching the JAX complex formulation.

    x: [..., T, H, d]; cos/sin: [T, d/2] (already gathered at position ids).
    """
    dt = x.dtype
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    # broadcast cos/sin over the head axis: [T, 1, d/2]
    c = cos.unsqueeze(-2)
    s = sin.unsqueeze(-2)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(dt)


# ----------------------------------------------------------------- attention


class Attention(nn.Module):
    """QKV projections + QK-norm + RoPE. Core op dispatched by the caller."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = d // self.num_heads
        std = cfg.initializer_range
        self.wq = NormalLinear(d, d, std)
        self.wk = NormalLinear(d, d, std)
        self.wv = NormalLinear(d, d, std)
        self.wo = NormalLinear(d, d, std)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def _split(self, x):  # [B, T, D] -> [B, T, H, d]
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim)

    def project_qkv(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if self.cfg.qk_norm:
            q = rms_norm(q, self.q_norm.weight, self.cfg.rms_norm_eps)
            k = rms_norm(k, self.k_norm.weight, self.cfg.rms_norm_eps)
        return q, k, v


def sdpa(q, k, v, attn_mask=None, is_causal=False, double_backward=False):
    """q,k,v: [B, T, H, d] -> [B, T, H*d].

    double_backward=True forces the math backend: flash/efficient SDPA kernels
    do not support second-order gradients, which the TTT inner loop requires.
    """
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, d]
    if double_backward:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    B, H, T, d = o.shape
    return o.transpose(1, 2).reshape(B, T, H * d)


_FLEX = {"fn": None, "masks": {}}


def _flex_swa(q, k, v, window: int):
    """Sliding-window attention via flex_attention (long sequences, window < T)."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    if _FLEX["fn"] is None:
        _FLEX["fn"] = torch.compile(flex_attention, dynamic=False)
    T = q.shape[-2]
    key = (T, window, str(q.device))
    if key not in _FLEX["masks"]:

        def mask_mod(b, h, qi, ki):
            return (qi >= ki) & (qi - ki < window)

        _FLEX["masks"][key] = create_block_mask(mask_mod, None, None, T, T, device=q.device)
    return _FLEX["fn"](q, k, v, block_mask=_FLEX["masks"][key])


def full_swa_attention(attn: Attention, x, cos, sin, window: int):
    """Full-sequence sliding-window causal attention (prefix path)."""
    B, T, _ = x.shape
    q, k, v = attn.project_qkv(x)
    q = apply_rope(q, cos[:T], sin[:T])
    k = apply_rope(k, cos[:T], sin[:T])
    if window >= T:
        o = sdpa(q, k, v, is_causal=True)
    elif T <= 8192:
        idx = torch.arange(T, device=x.device)
        mask = (idx[:, None] >= idx[None, :]) & (idx[:, None] < idx[None, :] + window)
        o = sdpa(q, k, v, attn_mask=mask[None, None])
    else:
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, d]
        ot = _flex_swa(qt, kt, vt, window)
        o = ot.transpose(1, 2).reshape(B, T, -1)
    return attn.wo(o)


def sw_causal_mask(chunk_id: int, cs: int, window: int, device) -> torch.Tensor:
    """[CS, WS+CS] mask for chunk `chunk_id` (matches SWA.sw_causal_mask)."""
    nk = window + cs
    start_q = chunk_id * cs
    end_k = start_q + cs
    qi = (torch.arange(cs, device=device) + start_q)[:, None]
    ki = (torch.arange(-nk, 0, device=device) + end_k)[None, :]
    return (qi >= ki) & (qi < ki + window) & (ki >= 0)


def chunked_swa_attention(attn: Attention, x, kv_cache, chunk_id: int, cos, sin, window: int):
    """Suffix path: one chunk with a pre-RoPE KV cache of `window` tokens.

    x: [B, CS, D]; kv_cache: (k, v) each [B, WS, H, d] (pre-RoPE).
    Returns output [B, CS, D] and the new cache.
    """
    B, CS, _ = x.shape
    q, k_new, v_new = attn.project_qkv(x)
    k_prev, v_prev = kv_cache
    k = torch.cat([k_prev, k_new], dim=1)  # [B, WS+CS, H, d]
    v = torch.cat([v_prev, v_new], dim=1)
    new_cache = (k[:, -window:], v[:, -window:])

    pos_all = torch.arange(window + CS, device=x.device)
    q = apply_rope(q, cos[pos_all[-CS:]], sin[pos_all[-CS:]])
    k = apply_rope(k, cos[pos_all], sin[pos_all])

    mask = sw_causal_mask(chunk_id, CS, window, x.device)
    import os

    row_microbatch = int(os.environ.get("TTT_ATTN_ROW_MICROBATCH", "0"))
    if row_microbatch > 0 and B > row_microbatch:
        pieces = []
        for start in range(0, B, row_microbatch):
            stop = min(B, start + row_microbatch)
            pieces.append(
                sdpa(
                    q[start:stop],
                    k[start:stop],
                    v[start:stop],
                    attn_mask=mask[None, None],
                    double_backward=True,
                )
            )
        o = torch.cat(pieces, dim=0)
    else:
        o = sdpa(q, k, v, attn_mask=mask[None, None], double_backward=True)
    return attn.wo(o), new_cache


# -------------------------------------------------------------------- blocks


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, is_suffix: bool):
        super().__init__()
        self.cfg = cfg
        self.is_suffix = is_suffix
        eps = cfg.rms_norm_eps
        d = cfg.hidden_size
        self.attn = Attention(cfg)
        self.feed_forward = SwiGLUMLP(cfg)
        self.seq_norm = RMSNorm(d, eps)
        self.ffn_norm = RMSNorm(d, eps)
        self.seq_post_norm = RMSNorm(d, eps)
        self.ffn_post_norm = RMSNorm(d, eps)
        self.has_prime = is_suffix and cfg.prime
        if self.has_prime:
            self.feed_forward_prime = SwiGLUMLP(cfg)
            self.ffn_prime_norm = RMSNorm(d, eps)
            self.ffn_prime_post_norm = RMSNorm(d, eps)

    def _sublayer_ffn(self, x, norm, ffn_fn, post_norm):
        h = norm(x) if self.cfg.pre_norm else x
        h = ffn_fn(h)
        if self.cfg.post_norm:
            h = post_norm(h)
        return h

    def forward_ffn_part(self, x, prime_weights=None):
        """Residual prime-FFN (optional) + residual FFN, after attention."""
        if self.has_prime:
            if prime_weights is None:
                fn = self.feed_forward.forward  # unreachable; prime always passed
                raise ValueError("suffix block with prime requires prime_weights")
            w1, w2, w3 = prime_weights

            def fn(h):
                return swiglu(h, w1, w2, w3)

            x = x + self._sublayer_ffn(x, self.ffn_prime_norm, fn, self.ffn_prime_post_norm)
        x = x + self._sublayer_ffn(x, self.ffn_norm, self.feed_forward.forward, self.ffn_post_norm)
        return x

    def forward_prefix(self, x, cos, sin, window: int, full_causal: bool):
        h = self.seq_norm(x) if self.cfg.pre_norm else x
        if full_causal:
            q, k, v = self.attn.project_qkv(h)
            T = x.shape[1]
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])
            a = self.attn.wo(sdpa(q, k, v, is_causal=True))
        else:
            a = full_swa_attention(self.attn, h, cos, sin, window)
        if self.cfg.post_norm:
            a = self.seq_post_norm(a)
        x = x + a
        return self.forward_ffn_part(x)

    def forward_suffix_chunk(self, x, kv_cache, chunk_id, cos, sin, window, prime_weights):
        h = self.seq_norm(x) if self.cfg.pre_norm else x
        a, new_cache = chunked_swa_attention(self.attn, h, kv_cache, chunk_id, cos, sin, window)
        if self.cfg.post_norm:
            a = self.seq_post_norm(a)
        x = x + a
        x = self.forward_ffn_part(x, prime_weights=prime_weights)
        return x, new_cache


# --------------------------------------------------------------------- model


class TTTModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.n_prefix = cfg.num_hidden_layers - cfg.suffix_len
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.wte.weight, std=cfg.initializer_range)
        self.layers = nn.ModuleList(
            Block(cfg, is_suffix=(i >= self.n_prefix)) for i in range(cfg.num_hidden_layers)
        )
        self.ln_f = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        cos, sin = precompute_freqs(head_dim, 2 * max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        assert cfg.tie_word_embeddings, "only tied embeddings supported (125m/1b configs)"

    # -- param plumbing ------------------------------------------------------
    def suffix_blocks(self):
        return [self.layers[i] for i in range(self.n_prefix, self.cfg.num_hidden_layers)]

    def prime_params(self) -> list[torch.Tensor]:
        """Fast-weight init (meta-learned): [w1, w2, w3] per suffix layer, flattened."""
        out = []
        for blk in self.suffix_blocks():
            p = blk.feed_forward_prime
            out.extend([p.w1.weight, p.w2.weight, p.w3.weight])
        return out

    def init_fast_weights(self, batch_size: int) -> list[torch.Tensor]:
        """Per-sequence fast weights [B, ...] initialized from the meta init.

        Uses repeat (not expand) so autograd flows meta-gradients back to the init.
        """
        return [p.unsqueeze(0).repeat(batch_size, 1, 1) for p in self.prime_params()]

    def init_kv_caches(self, batch_size: int, device, dtype=COMPUTE_DTYPE):
        cfg = self.cfg
        H = cfg.num_attention_heads
        d = cfg.hidden_size // H
        W = cfg.sliding_window_size
        return [
            (
                torch.zeros(batch_size, W, H, d, device=device, dtype=dtype),
                torch.zeros(batch_size, W, H, d, device=device, dtype=dtype),
            )
            for _ in range(cfg.suffix_len)
        ]

    # -- forward paths -------------------------------------------------------
    def embed(self, input_ids):
        return self.wte(input_ids).to(COMPUTE_DTYPE)

    def prefix_forward(self, input_ids):
        """Embed + run all prefix layers over the full sequence. [B, T, D]

        Set TTT_CKPT_PREFIX=1 to activation-checkpoint each prefix layer
        (prefix needs only first-order grads, so this is safe) — required for
        760M-scale 32K meta-training on 141GB GPUs.
        """
        import os

        cfg = self.cfg
        full_causal = cfg.seq_modeling_block == "self_attention"
        use_ckpt = os.environ.get("TTT_CKPT_PREFIX") == "1" and torch.is_grad_enabled()
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
        x = self.embed(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for i in range(self.n_prefix):
            if use_ckpt:
                x = checkpoint(
                    lambda x_, b=self.layers[i]: b.forward_prefix(
                        x_, cos, sin, cfg.sliding_window_size, full_causal=full_causal
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.layers[i].forward_prefix(
                    x, cos, sin, cfg.sliding_window_size, full_causal=full_causal
                )
        return x

    def suffix_chunk_forward(self, h_chunk, fast_weights, kv_caches, chunk_id):
        """Run suffix layers on one chunk. Returns (logits_fp32, new_kv_caches)."""
        cfg = self.cfg
        cos, sin = self.rope_cos, self.rope_sin
        x = h_chunk
        new_caches = []
        for j, blk in enumerate(self.suffix_blocks()):
            pw = fast_weights[3 * j : 3 * j + 3] if cfg.prime else None
            x, cache = blk.forward_suffix_chunk(
                x, kv_caches[j], chunk_id, cos, sin, cfg.sliding_window_size, pw
            )
            new_caches.append(cache)
        x = self.ln_f(x)
        logits = x.to(COMPUTE_DTYPE) @ self.wte.weight.to(COMPUTE_DTYPE).T
        return logits.float(), new_caches

    def forward_plain(self, input_ids):
        """Plain LM forward (pretrain mode / FA baseline): full sequence, no TTT."""
        cfg = self.cfg
        assert cfg.suffix_len == 0, "forward_plain expects no suffix layers"
        full_causal = cfg.seq_modeling_block == "self_attention"
        x = self.embed(input_ids)
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.layers:
            x = blk.forward_prefix(x, cos, sin, cfg.sliding_window_size, full_causal=full_causal)
        x = self.ln_f(x)
        logits = x.to(COMPUTE_DTYPE) @ self.wte.weight.to(COMPUTE_DTYPE).T
        return logits.float()
