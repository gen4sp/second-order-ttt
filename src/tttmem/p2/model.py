"""TTT-E2E transformer (prefix + chunked suffix with fast-weight prime MLPs).

Numerics mirror the official JAX code: fp32 parameters, matmuls in ``compute_dtype``
(bf16 by default, promote at use), RMSNorm / RoPE / CE in fp32 (or fp64 in tests). The
residual stream is carried in ``compute_dtype``. Parameter names and ``[in, out]``
layouts match ttt-ouroboros ``ttt_pt.model.TTTModel``, so its state dicts load with
``strict=True``.

Suffix attention differs from the reference only in bookkeeping: chunk ``c`` attends to
its *valid* past keys instead of a zero-padded ``window``-slot cache. RoPE angles are the
official ones (keys at cache slots ``window - P .. window + chunk - 1``, queries at
``window .. window + chunk - 1``), so the scores are identical and the dropped keys were
masked anyway.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class P2Config:
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 1664
    num_layers: int = 12
    num_heads: int = 12
    chunk: int = 1024
    window: int = 8192
    rms_eps: float = 1e-6
    rope_theta: float = 500000.0
    suffix_len: int = 3
    bos_id: int = 128000
    eos_id: int = 128001
    init_std: float = 0.02
    max_len: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def as_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml (official)
    "125m": P2Config(),
    # unit-test shape
    "tiny": P2Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_layers=3,
                     num_heads=4, chunk=8, window=64, suffix_len=1, bos_id=1, eos_id=2,
                     max_len=64),
    "tiny2": P2Config(vocab_size=101, hidden_size=32, intermediate_size=40, num_layers=4,
                      num_heads=4, chunk=8, window=64, suffix_len=2, bos_id=1, eos_id=2,
                      max_len=64),
    # MINI keeps the P2-3 8K/1K geometry and tokenizer vocabulary while making the
    # from-scratch rehearsal small enough for a local CPU smoke.
    "mini-s": P2Config(hidden_size=256, intermediate_size=576, num_layers=6,
                       num_heads=4, chunk=1024, window=8192, suffix_len=2,
                       max_len=8192),
    "mini-m": P2Config(hidden_size=512, intermediate_size=1120, num_layers=8,
                       num_heads=8, chunk=1024, window=8192, suffix_len=2,
                       max_len=8192),
    "micro16k": P2Config(vocab_size=16384, hidden_size=128, intermediate_size=288,
                          num_layers=4, num_heads=2, chunk=512, window=4096,
                          suffix_len=2, bos_id=0, eos_id=1, max_len=4096),
}


def _acc(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype: fp32, or fp64 when the model runs in fp64 (tests)."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(_acc(x.dtype))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.to(xf.dtype)).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair RoPE (JAX complex form). x [B, T, H, d]; cos/sin [T, d/2]."""
    dt = x.dtype
    xf = x.to(cos.dtype)
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2).to(dt)


class Lin(nn.Module):
    """Weight stored ``[in, out]`` (JAX layout); ``x @ W`` in the compute dtype."""

    def __init__(self, n_in: int, n_out: int, std: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out) * std)


class Norm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class MLP(nn.Module):
    def __init__(self, d: int, i: int, std: float) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = Lin(d, i, std), Lin(i, d, std), Lin(d, i, std)


class Attn(nn.Module):
    def __init__(self, cfg: P2Config) -> None:
        super().__init__()
        d, s = cfg.hidden_size, cfg.init_std
        self.wq, self.wk, self.wv, self.wo = Lin(d, d, s), Lin(d, d, s), Lin(d, d, s), Lin(d, d, s)
        self.q_norm, self.k_norm = Norm(cfg.head_dim), Norm(cfg.head_dim)


class Block(nn.Module):
    def __init__(self, cfg: P2Config, suffix: bool) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.attn = Attn(cfg)
        self.feed_forward = MLP(d, cfg.intermediate_size, cfg.init_std)
        self.seq_norm, self.ffn_norm = Norm(d), Norm(d)
        self.seq_post_norm, self.ffn_post_norm = Norm(d), Norm(d)
        self.suffix = suffix
        if suffix:
            self.feed_forward_prime = MLP(d, cfg.intermediate_size, cfg.init_std)
            self.ffn_prime_norm, self.ffn_prime_post_norm = Norm(d), Norm(d)


def _swiglu(x, w1, w2, w3, dt):
    x = x.to(dt)
    if w1.dim() == 3:  # per-sequence fast weights [B, D, I]
        return torch.bmm(F.silu(torch.bmm(x, w1.to(dt))) * torch.bmm(x, w3.to(dt)), w2.to(dt))
    return (F.silu(x @ w1.to(dt)) * (x @ w3.to(dt))) @ w2.to(dt)


def _sdpa(q, k, v, *, causal: str, math_backend: bool, window_mask=None):
    """q [B, Tq, H, d], k/v [B, Tk, H, d] -> [B, Tq, H*d].

    ``causal``: ``"square"`` (Tq == Tk, prefix) or ``"lower_right"`` (queries are the
    last Tq of Tk positions). ``math_backend`` is required for double backward (SO).
    ``window_mask`` ([Tq, Tk] bool) overrides when a sliding window actually binds.
    """
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    Tq, Tk = q.shape[-2], k.shape[-2]
    if window_mask is not None:
        mask, is_causal = window_mask, False
    elif causal == "square" or Tq == Tk:
        mask, is_causal = None, True
    elif math_backend or not q.is_cuda:
        i = torch.arange(Tq, device=q.device)[:, None]
        j = torch.arange(Tk, device=q.device)[None, :]
        mask, is_causal = j <= i + (Tk - Tq), False
    else:
        from torch.nn.attention.bias import causal_lower_right
        mask, is_causal = causal_lower_right(Tq, Tk), False
    if math_backend:
        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
    B, H, T, d = o.shape
    return o.transpose(1, 2).reshape(B, T, H * d)


class TTTE2E(nn.Module):
    """The model. ``prefix_forward`` runs blocks ``[0, L - S)`` over a whole segment;
    ``suffix_chunk`` runs the ``S`` suffix blocks on one chunk with given fast weights and
    past keys/values and returns the final hidden state (after ``ln_f``)."""

    def __init__(self, cfg: P2Config, compute_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.cfg = cfg
        self.compute_dtype = compute_dtype
        self.n_prefix = cfg.num_layers - cfg.suffix_len
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.wte.weight, std=cfg.init_std)
        self.layers = nn.ModuleList(Block(cfg, i >= self.n_prefix) for i in range(cfg.num_layers))
        self.ln_f = Norm(cfg.hidden_size)
        self.prefix_checkpoint = False
        self._rope_cache: dict = {}

    # ------------------------------------------------------------------ plumbing
    def suffix_blocks(self) -> list[Block]:
        return list(self.layers[self.n_prefix:])

    def prime_params(self) -> list[nn.Parameter]:
        out = []
        for blk in self.suffix_blocks():
            p = blk.feed_forward_prime
            out += [p.w1.weight, p.w2.weight, p.w3.weight]
        return out

    def slow_suffix_params(self) -> list[nn.Parameter]:
        """Every parameter the suffix chunk uses except the prime (fast) weights."""
        out = []
        for blk in self.suffix_blocks():
            out += [p for n, p in blk.named_parameters() if "feed_forward_prime" not in n]
        return out + [self.ln_f.weight, self.wte.weight]

    def init_fast(self, batch: int) -> list[torch.Tensor]:
        """Per-sequence copies of W_0 (``repeat`` keeps the meta-gradient path)."""
        return [p.unsqueeze(0).repeat(batch, 1, 1) for p in self.prime_params()]

    def rope(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        acc = _acc(self.compute_dtype)
        key = (n, str(device), acc)
        if key not in self._rope_cache:
            d = self.cfg.head_dim
            inv = 1.0 / (self.cfg.rope_theta ** (
                torch.arange(0, d, 2, device=device, dtype=acc)[: d // 2] / d))
            t = torch.arange(n, device=device, dtype=acc)
            f = torch.outer(t, inv)  # fp32 as in the reference; fp64 in fp64 tests
            self._rope_cache[key] = (torch.cos(f), torch.sin(f))
        return self._rope_cache[key]

    def _qkv(self, blk: Block, x: torch.Tensor):
        dt, eps = self.compute_dtype, self.cfg.rms_eps
        B, T, _ = x.shape
        H, d = self.cfg.num_heads, self.cfg.head_dim
        a = blk.attn
        xd = x.to(dt)
        q = (xd @ a.wq.weight.to(dt)).view(B, T, H, d)
        k = (xd @ a.wk.weight.to(dt)).view(B, T, H, d)
        v = (xd @ a.wv.weight.to(dt)).view(B, T, H, d)
        return rms_norm(q, a.q_norm.weight, eps), rms_norm(k, a.k_norm.weight, eps), v

    def _ffn_tail(self, blk: Block, x: torch.Tensor, fast=None, capture=None,
                  j: int = 0) -> torch.Tensor:
        dt, eps = self.compute_dtype, self.cfg.rms_eps
        if blk.suffix:
            w1, w2, w3 = fast
            p = _swiglu(rms_norm(x, blk.ffn_prime_norm.weight, eps), w1, w2, w3, dt)
            q = rms_norm(p, blk.ffn_prime_post_norm.weight, eps)
            if capture is not None:  # G0 gain split: prime output before/after its post-norm
                capture[("p", j)], capture[("q", j)] = p, q
            x = x + q
        f = blk.feed_forward
        y = _swiglu(rms_norm(x, blk.ffn_norm.weight, eps), f.w1.weight, f.w2.weight, f.w3.weight, dt)
        return x + rms_norm(y, blk.ffn_post_norm.weight, eps)

    # ------------------------------------------------------------------- forward
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.wte(ids).to(self.compute_dtype)

    def _prefix_block(self, blk: Block, x: torch.Tensor, extra_mask=None) -> torch.Tensor:
        eps, W = self.cfg.rms_eps, self.cfg.window
        T = x.shape[1]
        q, k, v = self._qkv(blk, rms_norm(x, blk.seq_norm.weight, eps))
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        wm = None
        if T > W or extra_mask is not None:
            i = torch.arange(T, device=x.device)
            wm = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < W)
            if extra_mask is not None:  # test-only explicit barrier mask
                wm = wm & extra_mask
        o = _sdpa(q, k, v, causal="square", math_backend=False, window_mask=wm)
        o = o.to(self.compute_dtype) @ blk.attn.wo.weight.to(self.compute_dtype)
        x = x + rms_norm(o, blk.seq_post_norm.weight, eps)
        return self._ffn_tail(blk, x)

    def prefix_forward(self, ids: torch.Tensor, extra_mask=None) -> torch.Tensor:
        """Embed and run the prefix blocks over one attention segment. [B, T, D].
        ``extra_mask`` ([T, T] bool, ANDed with causal) is for the barrier test only."""
        x = self.embed(ids)
        for blk in self.layers[: self.n_prefix]:
            if self.prefix_checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self._prefix_block, blk, x, extra_mask, use_reentrant=False)
            else:
                x = self._prefix_block(blk, x, extra_mask)
        return x

    def suffix_chunk(self, h: torch.Tensor, fast: list[torch.Tensor],
                     past: list[tuple[torch.Tensor, torch.Tensor]] | None,
                     *, math_backend: bool, kv_only: bool = False, key_mask=None,
                     segment_pos: int | None = None, seg0: int | None = None,
                     capture: dict | None = None):
        """One chunk through the suffix blocks.

        h: [B, C, D] prefix output for the chunk. fast: 3 tensors per suffix block.
        past: per suffix block ``(k, v)`` of earlier chunks in this segment, each
        [B, P, H, d] pre-RoPE (or None / P = 0). Returns ``(x_final, new_kv)`` where
        ``new_kv`` lists this chunk's pre-RoPE ``(k, v)`` per block. With ``kv_only``
        the computation stops after the last block's K/V projection and ``x_final`` is
        None (the reverse KV sweep needs only that path). ``key_mask`` may be [C, P + C]
        or [B, C, P + C] bool and is ANDed with causal attention. The batched form is
        used by the echo value-mask ablation. ``capture`` (a dict) receives the
        residual before ``ln_f`` under ``"h_pre"`` and each suffix block's prime output before
        and after its post-norm under ``("p", j)`` / ``("q", j)``; the forward is unchanged.
        """
        cfg, eps, dt = self.cfg, self.cfg.rms_eps, self.compute_dtype
        # segment_pos is relative to the explicit segment origin. Keeping it
        # in the suffix API makes the RoPE reset part of every path.
        legacy_positions = segment_pos is None and seg0 is None
        if segment_pos is None:
            segment_pos = 0
        if seg0 is not None:
            segment_pos = int(seg0)
        if segment_pos < 0:
            raise ValueError("segment_pos must be nonnegative")
        W, C = cfg.window, h.shape[1]
        x, new = h, []
        blocks = self.suffix_blocks()
        for j, blk in enumerate(blocks):
            q, k, v = self._qkv(blk, rms_norm(x, blk.seq_norm.weight, eps))
            new.append((k, v))
            if kv_only and j == len(blocks) - 1:
                return None, new
            if past is not None and past[j][0].shape[1] > 0:
                pk, pv = past[j]
                if pk.shape[1] > W:
                    pk, pv = pk[:, -W:], pv[:, -W:]
                kk, vv = torch.cat((pk, k), 1), torch.cat((pv, v), 1)
            else:
                kk, vv = k, v
            P = kk.shape[1] - C
            rope_len = W + C if legacy_positions else max(W + C, segment_pos + C)
            cos, sin = self.rope(rope_len, x.device)
            q0 = W if legacy_positions else segment_pos
            qr = apply_rope(q, cos[q0: q0 + C], sin[q0: q0 + C])
            k0 = W - P if legacy_positions else max(0, segment_pos - P)
            k1 = W + C if legacy_positions else segment_pos + C
            kr = apply_rope(kk, cos[k0: k1], sin[k0: k1])
            wm = None
            if P + C > W or key_mask is not None:  # window binds, or explicit test mask
                i = torch.arange(C, device=x.device)[:, None] + P
                jj = torch.arange(P + C, device=x.device)[None, :]
                wm = (jj <= i) & (i - jj < W)
                if key_mask is not None:
                    if key_mask.ndim == 2:
                        wm = wm & key_mask
                    elif key_mask.ndim == 3:
                        wm = wm.unsqueeze(0).unsqueeze(1) & key_mask.unsqueeze(1)
                    else:
                        raise ValueError("key_mask must have shape [C,K] or [B,C,K]")
            o = _sdpa(qr, kr, vv, causal="lower_right", math_backend=math_backend, window_mask=wm)
            o = o.to(dt) @ blk.attn.wo.weight.to(dt)
            x = x + rms_norm(o, blk.seq_post_norm.weight, eps)
            x = self._ffn_tail(blk, x, fast[3 * j: 3 * j + 3], capture, j)
        if capture is not None:
            capture["h_pre"] = x
        return rms_norm(x, self.ln_f.weight, eps), new

    def head(self, x_final: torch.Tensor) -> torch.Tensor:
        """Tied-embedding logits, cast to fp32 (fp64 in fp64 mode) for the loss."""
        dt = self.compute_dtype
        return (x_final.to(dt) @ self.wte.weight.to(dt).T).to(_acc(dt))


def build_model(preset: str | P2Config, compute_dtype=torch.bfloat16, seed: int = 0,
                device="cpu") -> TTTE2E:
    cfg = PRESETS[preset] if isinstance(preset, str) else preset
    g = torch.random.fork_rng(devices=[])
    with g:
        torch.manual_seed(seed)
        m = TTTE2E(cfg, compute_dtype)
    return m.to(device)


def load_state(model: TTTE2E, path: str) -> dict:
    """Load an ouroboros-format checkpoint (``{'model': state_dict, ...}`` or a bare
    state dict) with ``strict=True``. Returns metadata (step, cfg keys)."""
    # ouroboros checkpoints pickle their config class: unpickling needs the vendored package
    ref = Path(__file__).resolve().parents[3] / "tests" / "ref_ouroboros"
    if ref.is_dir() and str(ref) not in sys.path:
        sys.path.append(str(ref))
    # mmap: optimizer state in the file is never read into memory
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    meta = {"keys": len(state)}
    if isinstance(payload, dict):
        meta["step"] = payload.get("step")
        meta["has_opt"] = "opt" in payload
        for key in ("arm", "preset", "init_seed", "data_seed", "init_label"):
            if key in payload:
                meta[key] = payload[key]
    return meta


def param_count(model: TTTE2E) -> dict:
    emb = model.wte.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "non_embedding": total - emb,
            "fast": sum(p.numel() for p in model.prime_params())}


__all__ = ["P2Config", "PRESETS", "TTTE2E", "build_model", "load_state", "rms_norm",
           "apply_rope", "param_count", "math"]
