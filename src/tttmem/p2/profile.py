"""P2-0 GPU profile: throughput and peak memory per arm and eval mode, validation of each
optimization against an unoptimized reference, and the checkpoint loader check.

Every measurement prints one ``P2_PROFILE {json}`` line; the run's result summarizes them.
Blocks run in a fixed order under a wall-clock cap (``max_minutes``); a skipped block is
reported as skipped, never silently dropped.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field, fields

import torch

from .meta import (InnerCfg, fo_backward, fo_loss_plain, inner_x_cotangent, so_backward,
                   so_loss_plain, stream_eval, token_nll, seq_mean, denom, _fast_values)
from .model import _acc, build_model, load_state, param_count
from .train import ARMS

CKPT_URL = ("https://www.dropbox.com/scl/fi/b3hdau2dn5s95kdydycp6/ext-125m-e2e-32k-pt"
            "?rlkey=rdyy0wbn4c5zwxj69p1ywp0xi&dl=1")
TEXT_URLS = ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
             "https://www.gutenberg.org/cache/epub/2701/pg2701.txt")


@dataclass
class ProfileCfg:
    preset: str = "125m"
    seq_len: int = 8192
    device: str = "cuda"
    arms: list = field(default_factory=lambda: list(ARMS))
    micro_batches: list = field(default_factory=lambda: [1, 2, 4, 8])
    timed_steps: int = 3
    eval_batches: list = field(default_factory=lambda: [4, 8, 16])
    eval_modes: list = field(default_factory=lambda: ["ce", "lin", "aice", "noadapt", "heldout"])
    validate: bool = True
    val_seq: int = 4096
    exact_so_seq: int = 3072
    compile_check: bool = True
    loader_check: bool = True
    loader_precisions: list = field(default_factory=lambda: ["bf16", "fp32"])
    ckpt_url: str = CKPT_URL
    cache_dir: str = "data/p2cache"
    max_minutes: float = 40.0
    trunc_check: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ProfileCfg":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"mode", "process_name"}
        if unknown:
            raise ValueError(f"unknown p2 profile keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})


def emit(kind: str, **rec) -> dict:
    rec = {"kind": kind, **rec}
    print("P2_PROFILE " + json.dumps(rec), flush=True)
    return rec


def _cuda() -> bool:
    return torch.cuda.is_available()


def _free():
    gc.collect()
    if _cuda():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _reset_peak():
    if _cuda():
        torch.cuda.reset_peak_memory_stats()


def _peak() -> tuple[float, float]:
    """(allocated, reserved) peak in GiB; on CPU the process max RSS for both."""
    if _cuda():
        return (torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    return rss, rss


def _rand_batch(V, bos, B, T, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq = torch.randint(0, V, (B, T + 1), generator=g)
    seq[:, 0] = bos
    seq = seq.to(device)
    ids, tgt = seq[:, :-1], seq[:, 1:]
    return ids, tgt, tgt != bos


def _engine(model, kind, rule, ids, tgt, mask, *, fused=True, trunc=False, scale=1 / 64,
            so_fused=False):
    """``fused`` selects the FO attention backend; SO uses math in both sweeps unless
    ``so_fused`` (the profiled speed option)."""
    inner = InnerCfg(rule=rule)
    if kind == "so":
        return so_backward(model, ids, tgt, mask, inner, scale, fused_forward=so_fused)
    return fo_backward(model, ids, tgt, mask, inner, scale, trunc_kv=trunc, fused_attn=fused)


def _timed(fn, n):
    if not _cuda():
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000 / n


def profile_train(model, opt, cfg: ProfileCfg, arm: str, mb: int, *, trunc=False,
                  so_fused=False) -> dict:
    kind, rule = ARMS[arm]
    T = cfg.seq_len
    ids, tgt, mask = _rand_batch(model.cfg.vocab_size, model.cfg.bos_id, mb, T, cfg.device)
    params = list(model.parameters())

    def micro():
        _engine(model, kind, rule, ids, tgt, mask, trunc=trunc, so_fused=so_fused)

    def opt_step():
        torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        opt.step()
        opt.zero_grad(set_to_none=True)

    _free()
    micro()
    opt_step()                      # allocates the AdamW state before the peak is measured
    _reset_peak()
    # one complete step: timed_steps accumulated microbatches, then the optimizer step on
    # the accumulated gradients; the peaks cover both
    t_micro = _timed(micro, cfg.timed_steps)
    peak_micro = _peak()[0]
    t_opt = _timed(opt_step, 1)
    peak, reserved = _peak()
    return dict(arm=arm, micro_batch=mb, trunc_kv=trunc, so_fused_forward=so_fused,
                s_per_micro=t_micro, s_opt=t_opt, peak_alloc_micro_gb=peak_micro,
                tok_per_s=mb * T / t_micro,
                s_per_step_gb64=(64 / mb) * t_micro + t_opt,
                peak_alloc_gb=peak, peak_reserved_gb=reserved)


def profile_eval(model, cfg: ProfileCfg, mode: str, B: int) -> dict:
    T, C = cfg.seq_len, model.cfg.chunk
    ids, tgt, mask = _rand_batch(model.cfg.vocab_size, model.cfg.bos_id, B, T, cfg.device, 1)
    probe = torch.zeros_like(mask)
    probe[:, 4 * C::50] = True                                     # ~2 % of P, like real probes
    barriers = [2, 4]

    def run():
        cache: dict = {}
        if mode == "heldout":
            stream_eval(model, ids, tgt, mask, InnerCfg(rule="ce"))
        elif mode == "noadapt":
            stream_eval(model, ids, tgt, mask, InnerCfg(), adapt=False, barriers=barriers,
                        prefix_cache=cache, nll_positions=probe)
        else:
            stream_eval(model, ids, tgt, mask, InnerCfg(rule=mode), barriers=barriers,
                        prefix_cache=cache, nll_positions=probe)

    _free()
    run()
    _reset_peak()
    t = _timed(run, 2)
    return dict(mode=mode, batch=B, s_per_batch=t, tok_per_s=B * T / t, peak_alloc_gb=_peak()[0])


def _grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def _zero(model):
    for p in model.parameters():
        p.grad = None


def _rel(a: dict, b: dict) -> float:
    num = sum(float((a[k] - b[k]).pow(2).sum()) for k in b) ** 0.5
    den = sum(float(b[k].pow(2).sum()) for k in b) ** 0.5
    return num / den


def _cos(a: dict, b: dict) -> float:
    dot = sum(float((a[k] * b[k]).sum()) for k in b)
    na = sum(float(a[k].pow(2).sum()) for k in b) ** 0.5
    nb = sum(float(b[k].pow(2).sum()) for k in b) ** 0.5
    return dot / (na * nb)


def validate(model, cfg: ProfileCfg) -> list[dict]:
    """Each optimized path vs its unoptimized reference at the real width (B = 1)."""
    out = []
    T = cfg.val_seq
    ids, tgt, mask = _rand_batch(model.cfg.vocab_size, model.cfg.bos_id, 1, T, cfg.device, 2)
    ref = copy.deepcopy(model)
    ref.compute_dtype = torch.float32

    def grads_of(m, fn):
        _zero(m)
        fn(m)
        g = _grads(m)
        _zero(m)
        return g

    # V1: FO engine, fused vs math attention, bf16 vs fp32
    for rule in ("ce", "lin", "aice"):
        g_f = grads_of(model, lambda m: _engine(m, "fo", rule, ids, tgt, mask, fused=True))
        g_m = grads_of(model, lambda m: _engine(m, "fo", rule, ids, tgt, mask, fused=False))
        g_32 = grads_of(ref, lambda m: _engine(m, "fo", rule, ids, tgt, mask, fused=False))
        out.append(emit("validate", what=f"fo_{rule}_fused_vs_math", T=T,
                        rel_fused_vs_fp32=_rel(g_f, g_32), rel_math_vs_fp32=_rel(g_m, g_32),
                        rel_fused_vs_math=_rel(g_f, g_m), cos_fused_fp32=_cos(g_f, g_32)))
        del g_f, g_m, g_32
        _free()
    # V2: SO engine, fused forward sweep (speed option) vs math in both sweeps (default)
    g_f = grads_of(model, lambda m: _engine(m, "so", "ce", ids, tgt, mask, so_fused=True))
    g_m = grads_of(model, lambda m: _engine(m, "so", "ce", ids, tgt, mask))
    g_32 = grads_of(ref, lambda m: _engine(m, "so", "ce", ids, tgt, mask))
    out.append(emit("validate", what="so_fusedfwd_vs_math", T=T,
                    rel_fused_vs_fp32=_rel(g_f, g_32), rel_math_vs_fp32=_rel(g_m, g_32),
                    rel_fused_vs_math=_rel(g_f, g_m), cos_fused_fp32=_cos(g_f, g_32)))
    del g_f, g_m
    _free()
    # V3/V4: exact engines vs full-graph autograd (fp32, math attention)
    inner = InnerCfg()

    def plain(fn, T_, **kw):
        def f(m):
            i, t, k = ids[:, :T_], tgt[:, :T_], mask[:, :T_]
            loss, per_chunk = fn(m, i, t, k, inner, **kw)
            (per_chunk.mean(0).sum() / 64).backward()
        return f

    try:
        g_p = grads_of(ref, plain(fo_loss_plain, T))
        g_e = grads_of(ref, lambda m: _engine(m, "fo", "ce", ids, tgt, mask, fused=False))
        out.append(emit("validate", what="fo_exact_vs_plain_fp32", T=T, rel=_rel(g_e, g_p)))
        del g_p, g_e
    except torch.OutOfMemoryError:
        out.append(emit("validate", what="fo_exact_vs_plain_fp32", T=T, oom=True))
    _free()
    Ts = cfg.exact_so_seq
    try:
        g_p = grads_of(ref, plain(so_loss_plain, Ts))
        g_e = grads_of(ref, lambda m: _engine(m, "so", "ce", ids[:, :Ts], tgt[:, :Ts],
                                              mask[:, :Ts]))
        out.append(emit("validate", what="so_remat_vs_plain_fp32", T=Ts, rel=_rel(g_e, g_p)))
        del g_p, g_e
    except torch.OutOfMemoryError:
        out.append(emit("validate", what="so_remat_vs_plain_fp32", T=Ts, oom=True))
    _free()
    # V5: lin cotangent injection vs autograd through the bf16 head, one chunk
    C = model.cfg.chunk
    with torch.no_grad():
        h = model.prefix_forward(ids[:, :C])
    W = [w.requires_grad_(True) for w in _fast_values(model, 1)]
    x, _ = model.suffix_chunk(h, W, None, math_backend=False)
    m_ = mask[:, :C]
    w = m_.float() / denom(m_).unsqueeze(1)
    cot = inner_x_cotangent(model, "lin", None, tgt[:, :C], w)
    g_inj = torch.autograd.grad(x, W, grad_outputs=cot, retain_graph=True)
    z = model.head(x)
    lin = z.mean(-1) - z.gather(-1, tgt[:, :C].unsqueeze(-1)).squeeze(-1)
    g_ad = torch.autograd.grad(seq_mean(lin, m_).sum(), W)
    num = sum(float((a - b).float().pow(2).sum()) for a, b in zip(g_inj, g_ad)) ** 0.5
    den = sum(float(b.float().pow(2).sum()) for b in g_ad) ** 0.5
    out.append(emit("validate", what="lin_injection_vs_bf16_autograd", rel=num / den))
    del x, z, W, g_inj, g_ad, h
    _free()
    # attention kernels actually dispatched by the fused FO path
    if _cuda():
        from torch.profiler import ProfilerActivity, profile as tprof
        with tprof(activities=[ProfilerActivity.CUDA]) as prof:
            _engine(model, "fo", "ce", ids[:, :2048], tgt[:, :2048], mask[:, :2048])
            torch.cuda.synchronize()
        _zero(model)
        names = sorted({e.key for e in prof.key_averages() if any(
            s_ in e.key.lower() for s_ in ("flash", "fmha", "efficient", "cudnn"))})
        out.append(emit("validate", what="fused_attention_kernels", kernels=names[:12]))
        _free()
    # V6: eval NLL bf16 vs fp32 (CE rule, no barrier)
    with torch.no_grad():
        a = stream_eval(model, ids, tgt, mask, InnerCfg())["nll"]
        b = stream_eval(ref, ids, tgt, mask, InnerCfg(), fused_attn=False)["nll"]
    d = (a - b)[mask]
    out.append(emit("validate", what="eval_bf16_vs_fp32", mean_abs=float(d.abs().mean()),
                    mean=float(d.mean()), max_abs=float(d.abs().max())))
    del ref
    _free()
    return out


def compile_check(model, opt, cfg: ProfileCfg) -> dict:
    """Regional torch.compile of the prefix (FO-CE): throughput and gradient agreement."""
    T = cfg.val_seq
    ids, tgt, mask = _rand_batch(model.cfg.vocab_size, model.cfg.bos_id, 1, T, cfg.device, 3)
    _zero(model)
    _engine(model, "fo", "ce", ids, tgt, mask)
    g_e = _grads(model)
    _zero(model)
    eager = profile_train(model, opt, cfg, "e2e1-ce", 1)
    t0 = time.time()
    model.prefix_forward = torch.compile(model.prefix_forward, dynamic=False)
    try:
        _engine(model, "fo", "ce", ids, tgt, mask)
        compile_s = time.time() - t0
        g_c = _grads(model)
        _zero(model)
        comp = profile_train(model, opt, cfg, "e2e1-ce", 1)
        rec = emit("compile", eager_tok_s=eager["tok_per_s"], compiled_tok_s=comp["tok_per_s"],
                   compile_s=compile_s, rel_grad_vs_eager=_rel(g_c, g_e),
                   peak_eager=eager["peak_alloc_gb"], peak_compiled=comp["peak_alloc_gb"])
    finally:
        del model.prefix_forward            # back to the class method
        _zero(model)
        _free()
    return rec


# ------------------------------------------------------------------------ loader check
def _download(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 24):
            f.write(chunk)
    os.replace(tmp, path)


def _text_tokens(cfg: ProfileCfg, n_seq: int, T: int) -> torch.Tensor:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("NousResearch/Meta-Llama-3-8B", "tokenizer.json"))
    rows = []
    for i, url in enumerate(TEXT_URLS[:n_seq]):
        p = os.path.join(cfg.cache_dir, f"text{i}.txt")
        _download(url, p)
        txt = open(p, encoding="utf-8", errors="ignore").read()
        a = txt.find("*** START OF")
        b = txt.find("*** END OF")
        txt = txt[txt.find("\n", a) + 1 if a >= 0 else 0: b if b > 0 else None]
        ids = tok.encode(txt, add_special_tokens=False).ids
        rows.append([128000] + ids[: T])
    return torch.tensor(rows)


def loader_check(cfg: ProfileCfg, ckpt_path: str, precision: str = "bf16",
                 n_seq: int = 2) -> dict:
    """Released checkpoint: our TTT eval vs the vendored ouroboros port on held-out text.

    Acceptance: in fp32 the per-token mean |dNLL| within every chunk is <= 1e-4
    (weight mapping and forward semantics); in bf16 the chunk-mean |dNLL| is <= 2e-3 (the two
    ports round differently per token, so per-token bf16 differences are reported only).
    The two models are evaluated one after the other to bound host memory.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.insert(0, os.path.join(root, "tests", "ref_ouroboros"))
    import ttt_pt.block_inner as obi
    import ttt_pt.meta as omm
    import ttt_pt.model as om
    import ttt_pt.stream as ost
    from ttt_pt.config import preset_125m_e2e_ext32k

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    T = cfg.seq_len
    seq = _text_tokens(cfg, n_seq, T).to(cfg.device)
    ids, tgt = seq[:, :-1], seq[:, 1:]
    mask = tgt != 128000
    mine = build_model(cfg.preset, compute_dtype=dt)
    meta = load_state(mine, ckpt_path)
    mine = mine.to(cfg.device)
    params = param_count(mine)
    C = mine.cfg.chunk
    with torch.no_grad():
        a = stream_eval(mine, ids, tgt, mask, InnerCfg())
        noad = stream_eval(mine, ids, tgt, mask, InnerCfg(), adapt=False)["nll"]
    a_nll, inner_med = a["nll"].float(), float(a["inner_norm"].median())
    del mine, a
    _free()
    saved = [(m, m.COMPUTE_DTYPE) for m in (om, obi, ost)]
    try:
        for m, _ in saved:
            m.COMPUTE_DTYPE = dt
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        ref_cfg = preset_125m_e2e_ext32k()
        # its RoPE table spans the sliding-window cache plus a chunk, whatever T is
        ref = om.TTTModel(ref_cfg.model, max_seq_len=max(T, ref_cfg.model.sliding_window_size))
        ref.load_state_dict(payload["model"], strict=True)
        del payload
        ref = ref.to(cfg.device)
        _, aux = omm.loss_for_sequence_meta(ref, ids, tgt, mask, 1.0, ref_cfg, create_graph=False,
                                            return_token_nll=True)
        b = aux["token_nll"].detach().float()
        del ref, aux
    finally:
        for m, d in saved:
            m.COMPUTE_DTYPE = d
    _free()
    mk = mask.float().view(n_seq, -1, C)
    ca = (a_nll.view(n_seq, -1, C) * mk).sum(-1) / mk.sum(-1)
    cb = (b.view(n_seq, -1, C) * mk).sum(-1) / mk.sum(-1)
    tok = ((a_nll - b).abs().view(n_seq, -1, C) * mk).sum(-1) / mk.sum(-1)
    chunk_diff = float((ca - cb).abs().max())
    tok_max = float(tok.max())
    accept = tok_max <= 1e-4 if precision == "fp32" else chunk_diff <= 2e-3
    rec = emit("loader", ckpt=os.path.basename(ckpt_path), precision=precision, T=T,
               n_seq=n_seq, keys=meta.get("keys"), step=meta.get("step"), params=params,
               ours_nll=float(a_nll.mean()), ouroboros_nll=float(b.mean()),
               noadapt_nll=float(noad.float().mean()),
               max_abs_chunk_diff=chunk_diff,
               chunk_token_mean_abs_diff_max=tok_max,
               chunk_token_mean_abs_diff=tok.mean(0).tolist(),
               mean_abs_token_diff=float((a_nll - b).abs().mean()),
               chunk_nll_ours=ca.mean(0).tolist(), chunk_nll_ouroboros=cb.mean(0).tolist(),
               inner_norm_med=inner_med, accept=bool(accept))
    return rec


# ---------------------------------------------------------------------------- driver
def profile(cfg: ProfileCfg) -> dict:
    t_start = time.time()
    # fp32 references must be true fp32 (the bf16 paths do not use fp32 matmuls)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if _cuda():
        dev = torch.cuda.get_device_properties(0)
        gpu, mem = dev.name, dev.total_memory / 2**30
    else:
        import platform
        gpu, mem = f"cpu:{platform.machine()}", 0.0
    env = emit("env", gpu=gpu, mem_gb=mem, torch=torch.__version__, cuda=torch.version.cuda,
               threads=torch.get_num_threads(), cfg=asdict(cfg))
    ckpt_path = os.path.join(cfg.cache_dir, "125m-ext32k.pt")
    dl = None
    if cfg.loader_check:
        dl = threading.Thread(target=_download, args=(cfg.ckpt_url, ckpt_path), daemon=True)
        dl.start()
    left = lambda: cfg.max_minutes * 60 - (time.time() - t_start)  # noqa: E731
    res: dict = {"env": env, "train": [], "eval": [], "validate": [], "skipped": []}
    model = build_model(cfg.preset, compute_dtype=torch.bfloat16, device=cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=_cuda())
    emit("model", params=param_count(model))
    for arm in cfg.arms:
        last = None
        for mb in cfg.micro_batches:
            # skip a larger microbatch when it would take > 3 min to measure, or time is short
            est = None if last is None else last["s_per_micro"] * mb / last["micro_batch"] * (
                cfg.timed_steps + 1)
            if left() < 120 or (est is not None and est > 180):
                res["skipped"].append(f"train {arm} mb={mb}")
                continue
            try:
                last = profile_train(model, opt, cfg, arm, mb)
                res["train"].append(emit("train", **last))
            except torch.OutOfMemoryError:
                _zero(model)
                _free()
                res["train"].append(emit("train", arm=arm, micro_batch=mb, oom=True))
                break
    for kw in ([dict(arm="e2e1-ce", trunc=True)] if cfg.trunc_check else []) + \
            [dict(arm="e2e2", so_fused=True)]:
        if left() < 120:
            res["skipped"].append(f"train {kw}")
            continue
        arm = kw.pop("arm")
        try:
            res["train"].append(emit("train", **profile_train(model, opt, cfg, arm, 1, **kw)))
        except torch.OutOfMemoryError:
            _zero(model)
            _free()
            res["train"].append(emit("train", arm=arm, micro_batch=1, oom=True, **kw))
    for mode in cfg.eval_modes:
        for B in cfg.eval_batches:
            if left() < 90:
                res["skipped"].append(f"eval {mode} B={B}")
                continue
            try:
                res["eval"].append(emit("eval", **profile_eval(model, cfg, mode, B)))
            except torch.OutOfMemoryError:
                _free()
                res["eval"].append(emit("eval", mode=mode, batch=B, oom=True))
                break
    if cfg.validate and left() > 240:
        res["validate"] = validate(model, cfg)
    elif cfg.validate:
        res["skipped"].append("validate")
    if cfg.compile_check and left() > 300:
        try:
            res["compile"] = compile_check(model, opt, cfg)
        except Exception as e:  # report, do not lose the rest of the profile
            res["compile"] = emit("compile", error=repr(e)[:400])
    elif cfg.compile_check:
        res["skipped"].append("compile")
    del model, opt
    _free()
    if cfg.loader_check:
        dl.join(timeout=max(0.0, left() - 120))
        if os.path.exists(ckpt_path) and left() > 90:
            res["loader"] = []
            for precision in cfg.loader_precisions:
                try:
                    res["loader"].append(loader_check(cfg, ckpt_path, precision))
                except Exception as e:
                    res["loader"].append(emit("loader", precision=precision,
                                              error=repr(e)[:400]))
        else:
            res["skipped"].append("loader")
    res["wall_s"] = time.time() - t_start
    emit("done", wall_s=res["wall_s"], skipped=res["skipped"])
    return {"profile": res}
