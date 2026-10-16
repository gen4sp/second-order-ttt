"""Phase 2 port: tiny-shape correctness tests (CPU).

References: full-graph autograd (fp64), the vendored ttt-ouroboros port (fp32), and the
official JAX reference fixture ``tests/jax_parity/official_tiny.npz``.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from tttmem.p2 import meta as M
from tttmem.p2.model import PRESETS, build_model, load_state, _acc

torch.set_num_threads(4)
try:  # optional process title, set via PROCESS_NAME
    import os
    import setproctitle
    if os.environ.get("PROCESS_NAME"):
        setproctitle.setproctitle(os.environ["PROCESS_NAME"])
except ImportError:
    pass

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "ref_ouroboros"))
import ttt_pt.block_inner as obi  # noqa: E402
import ttt_pt.meta as omm  # noqa: E402
import ttt_pt.model as om  # noqa: E402
from ttt_pt.config import Config, ModelConfig  # noqa: E402


# ---------------------------------------------------------------------------- helpers
def make_batch(cfg, B, T, seed=0, n_bos=1):
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(3, cfg.vocab_size, (B, T + 1), generator=g)
    seq[:, 0] = cfg.bos_id
    for b in range(B):  # a few BOS inside -> masked targets (one inside chunk 0)
        for p in torch.randint(1, T, (n_bos,), generator=g).tolist() + [cfg.chunk // 2]:
            seq[b, p] = cfg.bos_id
    ids, tgt = seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
    return ids, tgt, tgt != cfg.bos_id


def model64(preset="tiny2", seed=0, cfg=None, fast_scale=1.0):
    m = build_model(cfg or preset, compute_dtype=torch.float64, seed=seed).double()
    with torch.no_grad():
        for p in m.prime_params():
            p.mul_(fast_scale)
    return m


def grads(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in model.named_parameters()}


def zero(model):
    for p in model.parameters():
        p.grad = None


def max_rel(a: dict, b: dict):
    num = max(float((a[k] - b[k]).abs().max()) for k in a)
    den = max(float(b[k].abs().max()) for k in b)
    return num / den


def binding_clip(model, ids, tgt, mask, rule="ce"):
    """A clip threshold that binds on some inner updates and not on others."""
    n = M.stream_eval(model, ids, tgt, mask, M.InnerCfg(rule=rule, clip=0.0))["inner_norm"]
    return float(n.flatten().median())


def assert_mixed(norms, clip):
    """The clip binds on at least one inner update."""
    assert float(norms.max()) > clip, (norms, clip)


def plain_grads(model, fn, *args, **kw):
    zero(model)
    loss, per_chunk = fn(model, *args, **kw)
    B = args[0].shape[0]
    (per_chunk.mean(0).sum() / B).backward()  # outer_scale = 1/B, mean over chunks
    return grads(model), float(loss)


# ------------------------------------------------------------- ouroboros (fp32) setup
def ouro_pair(window=64, seed=1, fast_scale=1.0, clip=1.0):
    cfg = replace(PRESETS["tiny2"], window=window)
    mine = build_model(cfg, compute_dtype=torch.float32, seed=seed)
    with torch.no_grad():
        for p in mine.prime_params():
            p.mul_(fast_scale)
    mc = ModelConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                     intermediate_size=cfg.intermediate_size, num_hidden_layers=cfg.num_layers,
                     num_attention_heads=cfg.num_heads, mini_batch_size=cfg.chunk,
                     sliding_window_size=window, rms_norm_eps=cfg.rms_eps,
                     bos_token_id=cfg.bos_id, eos_token_id=cfg.eos_id,
                     rope_theta=cfg.rope_theta, suffix_len=cfg.suffix_len, prime=True,
                     seq_modeling_block="SWA")
    ref = om.TTTModel(mc, max_seq_len=cfg.max_len)
    ref.load_state_dict(mine.state_dict(), strict=True)
    full = Config()
    full.model = mc
    full.training.optimizer_inner.lr = 1.0
    full.training.optimizer_inner.clip_gradient = clip
    return mine, ref, full


@pytest.fixture
def fp32_reference(monkeypatch):
    for mod in (om, obi):
        monkeypatch.setattr(mod, "COMPUTE_DTYPE", torch.float32)
    import ttt_pt.stream as ost
    monkeypatch.setattr(ost, "COMPUTE_DTYPE", torch.float32)


# ------------------------------------------------------------------------- forward/eval
@pytest.mark.parametrize("window", [64, 16])
def test_stream_eval_matches_ouroboros(fp32_reference, window):
    mine, ref, full = ouro_pair(window=window, clip=0.45)
    ids, tgt, mask = make_batch(mine.cfg, 2, 32, seed=3)
    _, aux = omm.loss_for_sequence_meta(ref, ids, tgt, mask, 1.0, full, create_graph=False,
                                        return_token_nll=True)
    out = M.stream_eval(mine, ids, tgt, mask, M.InnerCfg(clip=0.45))
    err = float((out["nll"] - aux["token_nll"].detach()).abs().max())
    assert err < 2e-5, err
    assert_mixed(out["inner_norm"], 0.45)


def test_so_backward_matches_ouroboros(fp32_reference):
    mine, ref, full = ouro_pair(clip=0.45)
    ids, tgt, mask = make_batch(mine.cfg, 2, 32, seed=4)
    loss, _ = omm.loss_for_sequence_meta(ref, ids, tgt, mask, 1.0, full, create_graph=True)
    loss.backward()
    g_ref = {n: p.grad for n, p in ref.named_parameters()}
    st = M.so_backward(mine, ids, tgt, mask, M.InnerCfg(clip=0.45), outer_scale=1 / 2)
    assert abs(float(st.loss_sum) / 2 - float(loss)) < 1e-5
    assert_mixed(st.inner_norm, 0.45)
    assert max_rel(grads(mine), g_ref) < 5e-5


def test_block_schedule_not_implemented():
    with pytest.raises(NotImplementedError):
        M.InnerCfg(block_chunks=2)


# --------------------------------------------------------------- engines vs plain (fp64)
def test_so_backward_matches_plain_fp64():
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 3, 32, seed=6)
    inner = M.InnerCfg(clip=binding_clip(m, ids, tgt, mask))
    g_ref, _ = plain_grads(m, M.so_loss_plain, ids, tgt, mask, inner)
    for fused in (True, False):
        zero(m)
        st = M.so_backward(m, ids, tgt, mask, inner, outer_scale=1 / 3, fused_forward=fused)
        assert max_rel(grads(m), g_ref) < 1e-9
    assert_mixed(st.inner_norm, inner.clip)


@pytest.mark.parametrize("rule", ["ce", "lin", "aice"])
@pytest.mark.parametrize("trunc", [False, True])
def test_fo_backward_matches_plain_fp64(rule, trunc):
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 3, 32, seed=7)
    inner = M.InnerCfg(rule=rule, clip=binding_clip(m, ids, tgt, mask, rule))
    g_ref, loss_ref = plain_grads(m, M.fo_loss_plain, ids, tgt, mask, inner, trunc_kv=trunc)
    zero(m)
    st = M.fo_backward(m, ids, tgt, mask, inner, outer_scale=1 / 3, trunc_kv=trunc)
    assert abs(float(st.loss_sum) / 3 - loss_ref) < 1e-12
    assert_mixed(st.inner_norm, inner.clip)
    assert max_rel(grads(m), g_ref) < 1e-9
    # every parameter group receives gradient (prefix, suffix, W_0, embedding)
    g = grads(m)
    assert all(float(v.abs().max()) > 0 for v in g.values())


def test_fo_differs_from_so_and_trunc_matters():
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 2, 32, seed=8)
    inner = M.InnerCfg(clip=binding_clip(m, ids, tgt, mask))
    so, _ = plain_grads(m, M.so_loss_plain, ids, tgt, mask, inner)
    fo, _ = plain_grads(m, M.fo_loss_plain, ids, tgt, mask, inner)
    tr, _ = plain_grads(m, M.fo_loss_plain, ids, tgt, mask, inner, trunc_kv=True)
    assert max_rel(fo, so) > 1e-4 and max_rel(tr, fo) > 1e-4


def test_lin_cotangent_matches_autograd_fp64():
    m = model64()
    B, C, D = 2, 8, m.cfg.hidden_size
    x = torch.randn(B, C, D, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, m.cfg.vocab_size, (B, C))
    msk = torch.rand(B, C) > 0.2
    w = msk.double() / M.denom(msk).double().unsqueeze(1)
    z = m.head(x)
    lin = (z.mean(-1) - z.gather(-1, y.unsqueeze(-1)).squeeze(-1))
    (gx,) = torch.autograd.grad((lin * w).sum(), x)
    cot = M.inner_x_cotangent(m, "lin", None, y, w)
    assert float((cot - gx).abs().max()) < 1e-12


def test_aice_equals_lin_on_first_chunk_only():
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 24, seed=9)
    a = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule="aice"))["nll"]
    lin = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule="lin"))["nll"]
    C = m.cfg.chunk
    assert float((a[:, :2 * C] - lin[:, :2 * C]).abs().max()) < 1e-12   # W_1 identical
    assert float((a[:, 2 * C:] - lin[:, 2 * C:]).abs().max()) > 1e-8    # W_2 differs


def test_stream_eval_matches_plain_forward_losses():
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 32, seed=10)
    for rule in ("ce", "lin", "aice"):
        _, per_chunk = M.fo_loss_plain(m, ids, tgt, mask, M.InnerCfg(rule=rule))
        nll = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule=rule))["nll"]
        mine = (nll.double() * mask).view(2, 4, 8).sum(-1) / mask.view(2, 4, 8).sum(-1)
        assert float((mine.T - per_chunk.detach()).abs().max()) < 1e-6  # nll stored fp32


def test_no_adapt_and_masks():
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 32, seed=11)
    base = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(), adapt=False)["nll"]
    lr0 = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(lr=0.0))["nll"]
    none = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(),
                         inner_mask=torch.zeros_like(mask))["nll"]
    assert float((base - lr0).abs().max()) < 1e-12
    assert float((base - none).abs().max()) < 1e-12
    full = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(), inner_mask=torch.ones_like(mask))
    adapt = M.stream_eval(m, ids, tgt, mask, M.InnerCfg())
    assert float((full["nll"] - adapt["nll"]).abs().max()) < 1e-12


# ---------------------------------------------------------------------------- barrier
def masked_scan(model, ids, tgt, mask, inner, starts):
    """Stateful full-sequence scan with explicit block masks (reference for barriers):
    one prefix pass with a block-diagonal mask, absolute positions, all past keys kept,
    the segment restriction applied as a key mask; CE inner rule."""
    B, T = ids.shape
    C, S = model.cfg.chunk, model.cfg.suffix_len
    n = T // C
    seg = torch.zeros(T, dtype=torch.long)
    for s in starts:
        seg[s * C:] += 1
    h = model.prefix_forward(ids, extra_mask=seg[:, None] == seg[None, :])
    W = [p.detach().unsqueeze(0).repeat(B, 1, 1) for p in model.prime_params()]
    past, out, fasts = [], torch.zeros(B, T, dtype=torch.float64), []
    for c in range(n):
        sl = slice(c * C, (c + 1) * C)
        Wc = [w.detach().requires_grad_(True) for w in W]
        km = seg[sl][:, None] == seg[: (c + 1) * C][None, :]
        x, new = model.suffix_chunk(h[:, sl], Wc, M._as_past(M._cat(past, S), h, model),
                                    math_backend=True, key_mask=km)
        nll = M.token_nll(model.head(x), tgt[:, sl])
        out[:, sl] = nll.detach()
        past.append([(k.detach(), v.detach()) for k, v in new])
        if c < n - 1:
            g = torch.autograd.grad(M.seq_mean(nll, mask[:, sl]).sum(), Wc)
            gc, _ = M.clip_per_seq(list(g), inner.clip)
            W = [w - inner.lr * t for w, t in zip(W, gc)]
            fasts.append([w.detach() for w in W])
    return out, fasts


def test_barrier_equals_masked_scan_fp64():
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 64, seed=12)
    C, starts = m.cfg.chunk, [2, 4]
    for s in starts:  # boundary policy: the target crossing a barrier is masked
        mask[:, s * C - 1] = False
    inner = M.InnerCfg(clip=0.4)
    ref_nll, ref_fast = masked_scan(m, ids, tgt, mask, inner, starts)
    out = M.stream_eval(m, ids, tgt, mask, inner, barriers=starts, return_fast=True)
    assert_mixed(out["inner_norm"], inner.clip)
    assert float((out["nll"].double() - ref_nll).abs().max()) < 1e-6
    for a, b in zip(out["fast"], ref_fast[-1]):
        assert float((a - b).abs().max()) < 1e-10
    # and the barrier matters
    nob = M.stream_eval(m, ids, tgt, mask, inner)["nll"]
    assert float((nob[:, 2 * C:] - out["nll"][:, 2 * C:]).abs().max()) > 1e-4


# ----------------------------------------------------------------------------- loader
def test_load_state_roundtrip(tmp_path):
    a = build_model("tiny", compute_dtype=torch.float32, seed=1)
    b = build_model("tiny", compute_dtype=torch.float32, seed=2)
    torch.save({"model": a.state_dict(), "step": 7, "opt": {}}, tmp_path / "c.pt")
    meta = load_state(b, str(tmp_path / "c.pt"))
    assert meta["step"] == 7 and meta["has_opt"]
    for (n, p), q in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(p, q), n


# ------------------------------------------------------------------- official JAX parity
NPZ = HERE / "jax_parity" / "official_tiny.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="tests/jax_parity/official_tiny.npz missing")
def test_matches_official_jax():
    z = np.load(NPZ, allow_pickle=False)
    jc = json.loads(str(z["config_json"]))
    cfg = replace(PRESETS["tiny2"], **{k: jc[k] for k in jc if k in PRESETS["tiny2"].as_dict()})
    m = build_model(cfg, compute_dtype=torch.float32)
    sd = {k[len("param/"):]: torch.from_numpy(z[k]) for k in z.files if k.startswith("param/")}
    m.load_state_dict(sd, strict=True)
    ids = torch.from_numpy(z["input_ids"]).long()
    tgt = torch.from_numpy(z["target_ids"]).long()
    mask = torch.from_numpy(z["loss_mask"]) > 0
    inner = M.InnerCfg(clip=float(jc.get("inner_clip", 1.0)), lr=float(jc.get("inner_lr", 1.0)))
    B = ids.shape[0]
    out = M.stream_eval(m, ids, tgt, mask, inner)
    assert float((out["nll"] - torch.from_numpy(z["token_nll"])).abs().max()) < 1e-4
    st = M.so_backward(m, ids, tgt, mask, inner, outer_scale=1 / B)
    assert abs(float(st.loss_sum) / B - float(z["loss_so"])) < 1e-5
    g_ref = {k[len("grad_so/"):]: torch.from_numpy(z[k]) for k in z.files if k.startswith("grad_so/")}
    assert max_rel(grads(m), g_ref) < 1e-4
    if any(k.startswith("grad_fo/") for k in z.files):
        zero(m)
        M.fo_backward(m, ids, tgt, mask, inner, outer_scale=1 / B, fused_attn=False)
        g_fo = {k[len("grad_fo/"):]: torch.from_numpy(z[k]) for k in z.files
                if k.startswith("grad_fo/")}
        assert max_rel(grads(m), g_fo) < 1e-4


# ------------------------------------------------------- accumulation, masks, numerics
@pytest.mark.parametrize("kind", ["fo", "so"])
def test_microbatch_accumulation_parity_fp64(kind):
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 4, 32, seed=13)
    inner = M.InnerCfg(clip=binding_clip(m, ids, tgt, mask))
    eng = M.fo_backward if kind == "fo" else M.so_backward
    zero(m)
    eng(m, ids, tgt, mask, inner, outer_scale=1 / 4)
    whole = grads(m)
    zero(m)
    for sl in (slice(0, 2), slice(2, 4)):
        eng(m, ids[sl], tgt[sl], mask[sl], inner, outer_scale=1 / 4)
    assert max_rel(grads(m), whole) < 1e-12


def test_all_masked_chunk_is_finite_and_exact():
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 2, 32, seed=14)
    mask[0, 8:16] = False                       # sequence 0, chunk 1: no valid target
    inner = M.InnerCfg(clip=binding_clip(m, ids, tgt, mask))
    g_ref, _ = plain_grads(m, M.so_loss_plain, ids, tgt, mask, inner)
    zero(m)
    M.so_backward(m, ids, tgt, mask, inner, outer_scale=1 / 2)
    g = grads(m)
    assert all(bool(torch.isfinite(v).all()) for v in g.values())
    assert max_rel(g, g_ref) < 1e-9


def test_stream_eval_masks_barrier_crossing_targets():
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 2, 64, seed=15)
    C = m.cfg.chunk
    a = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(), barriers=[2, 4], return_fast=True)
    mk = mask.clone()
    mk[:, 2 * C - 1] = False
    mk[:, 4 * C - 1] = False
    b = M.stream_eval(m, ids, tgt, mk, M.InnerCfg(), barriers=[2, 4], return_fast=True)
    for x, y in zip(a["fast"], b["fast"]):
        assert float((x - y).abs().max()) == 0.0


def test_sparse_head_matches_full_head():
    m = model64()
    ids, tgt, mask = make_batch(m.cfg, 2, 64, seed=16)
    pos = torch.zeros_like(mask)
    pos[:, 33::5] = True
    for rule, adapt in (("lin", True), ("ce", False)):
        full = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule=rule), adapt=adapt,
                             barriers=[2, 4])["nll"]
        cache: dict = {}
        sp = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule=rule), adapt=adapt,
                           barriers=[2, 4], nll_positions=pos, prefix_cache=cache)["nll"]
        assert float((full[pos] - sp[pos]).abs().max()) < 1e-6
        assert float(sp[~pos].abs().max()) == 0.0
        again = M.stream_eval(m, ids, tgt, mask, M.InnerCfg(rule=rule), adapt=adapt,
                              barriers=[2, 4], nll_positions=pos, prefix_cache=cache)["nll"]
        assert torch.equal(again, sp)            # shared prefix gives identical results


# ------------------------------------------------------------------------------ probes
def _doc(C=8, n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(10, 10_000, size=C * n + 1)


def _plant(d, pos, gram):
    d[pos - 2: pos + 2] = gram       # x[t-2], x[t-1], x[t], y[t] = d[t+1]


def test_find_probes_strata_and_exclusions():
    from tttmem.p2.probe import STRATA, find_probes
    C = 16                                    # S1 = [0, 32), S2 = [32, 64), P = [64, 128)
    d = _doc(C, seed=1)
    g = {k: 20000 + 4 * k + np.arange(4) for k in range(6)}
    plants = {0: [3, 70], 1: [18, 23, 76], 2: [8, 28, 82], 3: [13, 40, 88],
              4: [46, 94, 100], 5: [33, 106]}
    for k, ts in plants.items():
        for t in ts:
            _plant(d, t, g[k])
    x, y = d[:-1], d[1:]
    p = find_probes(x, y, C)
    assert p.nf_ok
    got = {int(t): STRATA[s] for t, s in zip(p.pos, p.stratum)}
    assert got == {70: "E1", 76: "E2s", 82: "E2c", 88: "E2b", 94: "E1"}, got
    assert p.focal[[3, 18, 23, 8, 28, 13, 40, 46]].all() and p.focal.sum() == 8
    for c in range(4):                        # count-matched, disjoint non-focal control
        sl = slice(c * C, (c + 1) * C)
        assert p.nonfocal[sl].sum() == p.focal[sl].sum()
    assert not (p.nonfocal & p.focal).any() and not p.nonfocal[4 * C:].any()
    excl = np.zeros(30_000, dtype=bool)
    excl[g[0][3]] = True                      # target token of g0 excluded
    assert 70 not in set(find_probes(x, y, C, excluded=excl).pos.tolist())


def test_bootstrap_and_decision_shapes():
    from tttmem.p2.probe import g2_decision
    rng = np.random.default_rng(0)
    N = 200
    cl = rng.integers(0, 50, N)
    base = rng.normal(3, 0.5, N)
    L = {"so": base, "fo": base + 0.2 + rng.normal(0, 0.1, N), "lin": base + 0.02}
    Lho = {"so": base[:100], "lin": base[:100] + 0.001}
    G = {"so": np.full(N, 0.3), "fo": np.full(N, 0.1), "lin": np.full(N, 0.25)}
    noise = rng.normal(0, 0.05, N)
    Mm = {"so": noise + 0.10, "fo": noise + 0.02, "lin": noise + 0.09}
    r = g2_decision(L, G, Mm, cl, Lho, cl[:100], n_boot=2000)
    assert r["G2_loss"] and r["G2_memory"] and r["G2"]
    G["lin"] = np.full(N, 0.1)                     # total gain is reported, not gating
    assert g2_decision(L, G, Mm, cl, Lho, cl[:100], n_boot=2000)["G2_memory"]
    Mm["lin"] = Mm["fo"]                           # no source-memory rescue
    assert not g2_decision(L, G, Mm, cl, Lho, cl[:100], n_boot=2000)["G2_memory"]
    Mm = {a: np.full(N, 0.05) for a in ("so", "fo", "lin")}
    r = g2_decision(L, G, Mm, cl, Lho, cl[:100], n_boot=2000)
    assert r["G2_loss"] and not r["G2_memory"]     # equal memory in every arm


# ---------------------------------------------------------------------- P2-1 loop smoke
def _write_split(root, name, rows):
    from tttmem.p2 import data as D
    arr = np.zeros((len(rows), D.SEQ), dtype=np.uint32)
    arr[:, : rows.shape[1]] = rows
    arr.tofile(root / f"{name}.bin")
    with open(root / f"{name}.jsonl", "w") as f:
        for i in range(len(rows)):
            f.write(json.dumps({"row": i, "sha1": f"{name}{i:04d}",
                                "domain": f"d{i % 3}.org"}) + "\n")


def _smoke_data(tmp_path):
    cfg0 = PRESETS["tiny"]
    T = 8 * cfg0.chunk
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    excluded = np.zeros(cfg0.vocab_size, dtype=bool)
    excluded[[cfg0.bos_id, cfg0.eos_id]] = True
    np.save(data / "excluded_vocab.npy", excluded)
    for name, n in (("A", 8), ("eval", 6), ("ho", 3), ("calib", 2)):
        rows = rng.integers(3, 7, size=(n, T + 1))  # 4 symbols: many repeated 4-grams
        rows[:, 0] = cfg0.bos_id
        _write_split(data, name, rows)
    return data, T


@pytest.mark.parametrize("arm", ["e2e2", "e2e1-ce", "e2e1-lin", "e2e1-aice"])
def test_train_loop_smoke(tmp_path, arm, capsys):
    """End-to-end P2-1 loop at the tiny shape: data, pilot, eval artifacts, stdout evidence."""
    from tttmem.p2.train import TrainCfg, train
    data, T = _smoke_data(tmp_path)
    cfg = TrainCfg(arm=arm, preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                   global_batch=4, micro_batch=2, steps=2, warmup=1, lr_peak=1e-3,
                   data_dir=str(data), out_dir=str(tmp_path / "out"), n_eval=4, min_probes=1,
                   eval_steps=[0, 1, 2], full_eval_steps=[0, 2], trajectory_docs=2,
                   sizing_steps=[1], assay_gate=False,
                   ho_docs=3, ho_trajectory_docs=2, eval_batch=2, pilot_steps=[1],
                   pilot_docs=2, pilot_max_rise=1e9, log_every=1, save_checkpoints=False)
    res = train(cfg)
    out = capsys.readouterr().out
    assert not res["aborted"] and res["tokens"] == 2 * 4 * T
    steps = [json.loads(l[8:]) for l in out.splitlines() if l.startswith("P2_STEP ")]
    assert [s["step"] for s in steps] == [1, 2]
    assert all(np.isfinite(s["loss"]) and np.isfinite(s["grad_norm"]) for s in steps)
    assert ("P2_PILOT step=1" in out) == (arm == "e2e2")
    a = next(json.loads(l[9:]) for l in out.splitlines() if l.startswith("P2_ASSAY "))
    assert a["descriptive"] == (arm not in ("e2e2", "e2e1-ce"))
    meta = np.load(tmp_path / "out" / "eval_meta.npz")
    n_docs = len(meta["eval_rows"])
    assert n_docs >= 2 and meta["probe_offset"][-1] == len(meta["probe_pos"])
    assert list(meta["ho_domain"]) == ["d0.org", "d1.org", "d2.org"]
    full = np.load(tmp_path / "out" / "eval_step0002.npz")
    for m in cfg.eval_modes:
        assert np.isfinite(full[f"doc_{m}"]).all()
        assert len(full[f"tok_{m}"]) == meta["probe_offset"][-1]
    assert np.isfinite(full["doc_fP_focal_refclip"]).all()
    for mode in ("adapt", "frozenP"):
        assert len(full[f"src_nll_{mode}"]) == n_docs
        assert np.isfinite(full[f"src_nll_{mode}"]).all()
        # one write-time NLL and cotangent norm per focal source position
        assert len(full[f"src_tok_{mode}"]) == len(full[f"src_cot_{mode}"]) \
            == meta["focal_offset"][-1] == len(meta["focal_pos"]) > 0
        assert (full[f"src_cot_{mode}"] > 0).all()
    assert full["doc_clip_frac"].shape == (n_docs,)
    assert full["clip_scale"].shape == (7,) and full["clip_frac"].shape == (7,)
    assert full["update_norm"].shape == (1,)
    for rule in cfg.crossover_rules:
        if rule != {"e2e2": "ce", "e2e1-ce": "ce", "e2e1-lin": "lin", "e2e1-aice": "aice"}[arm]:
            assert full[f"x_{rule}_adapt"].shape == (n_docs,)
            assert full[f"x_{rule}_heldout"].shape == (3,)
    assert len(full["doc_heldout"]) == 3 and np.isfinite(full["doc_heldout"]).all()
    traj = np.load(tmp_path / "out" / "eval_step0001.npz")
    assert len(traj["doc_adapt"]) == 2 and len(traj["doc_fP_focal"]) == 2
    assert len(traj["doc_fP_nonfocal"]) == 2 and "doc_fP_focal_refclip" not in traj
    # adaptation is on: the no-adapt control differs from the adapted stream
    assert not np.allclose(full["doc_adapt"], full["doc_noadapt"])


@pytest.mark.parametrize("attempt", [0, 1])
def test_train_pilot_abort(tmp_path, capsys, attempt):
    from tttmem.p2.train import TrainCfg, train
    data, T = _smoke_data(tmp_path)
    cfg = TrainCfg(arm="e2e2", preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                   global_batch=4, micro_batch=2, steps=2, warmup=1, data_dir=str(data),
                   out_dir=str(tmp_path / "out"), n_eval=4, min_probes=1, eval_steps=[],
                   assay_gate=False, attempt=attempt, lr_peak=1e-3 if attempt else 3e-3,
                   ho_docs=3, pilot_steps=[1], pilot_docs=2, pilot_max_rise=-1e9,
                   save_checkpoints=False)
    res = train(cfg)
    assert res["aborted"] and res["step"] == 1
    out = capsys.readouterr().out
    abort = next(json.loads(l[9:]) for l in out.splitlines() if l.startswith("P2_ABORT "))
    assert abort["next"] == ("stop_replan" if attempt else "restart_so_lr_1e-3_attempt_1_then_launch_fo")
    assert abort.get("final", False) == bool(attempt)


def test_find_probes_rejects_unmatched_nonfocal_pool():
    from tttmem.p2.probe import find_probes
    C = 16
    d = np.full(8 * C + 1, 10)
    p = find_probes(d[:-1], d[1:], C)
    assert len(p.pos) and not p.nf_ok
    assert p.focal[:C].sum() > p.nonfocal[:C].sum()
    assert not (p.focal & p.nonfocal).any()       # diagnostic best-effort mask retained


def test_prepare_probes_uses_effective_write_validity():
    from tttmem.p2.evaluate import prepare_probes
    from tttmem.p2.probe import find_probes
    C, bos = 16, 99999
    d = _doc(C, seed=21)
    for t in (3, 70):
        _plant(d, t, np.arange(20000, 20004))
    d[[12, 21, 48]] = bos
    p = prepare_probes(d[None], C, None, 7, bos)[0]
    ref = find_probes(d[:-1], d[1:], C, valid=d[1:] != bos, seed=7)
    assert p.nf_ok and np.array_equal(p.pos, ref.pos)
    assert np.array_equal(p.nonfocal, ref.nonfocal)
    assert not p.nonfocal[d[1:] == bos].any()
    for c in range(4):
        sl = slice(c * C, (c + 1) * C)
        assert p.focal[sl].sum() == p.nonfocal[sl].sum()


def test_boot_mean_unsupported_draws_and_nonfinite():
    from tttmem.p2.probe import boot_mean, one_sided
    weights = (np.array([[1, 1], [0, 0], [0, 1]]), np.array([0, 1]))
    point, draws = boot_mean(np.array([2.0, 4.0]), weights)
    assert point == 3.0 and np.isnan(draws[1])
    r = one_sided(point, draws)
    assert r["unsupported"] == 1 / 3
    assert r["lo"] == np.nanquantile(draws, 0.05)
    assert np.isnan(one_sided(point, np.full(3, np.nan))["lo"])
    for v in ([np.nan, 1.0], [np.inf, 1.0], [1.0]):
        with pytest.raises(ValueError):
            boot_mean(np.array(v), weights)


def _g2_arrays(N=40):
    L = {"so": np.ones(N), "fo": np.full(N, 1.1), "lin": np.full(N, 1.01)}
    G = {"so": np.full(N, 0.3), "fo": np.full(N, 0.1), "lin": np.full(N, 0.25)}
    Mm = {"so": np.full(N, 0.10), "fo": np.full(N, 0.02), "lin": np.full(N, 0.09)}
    ho = {"so": np.ones(N), "lin": np.ones(N)}
    return L, G, Mm, ho


@pytest.mark.parametrize("group", [0, 1, 2, 3])
@pytest.mark.parametrize("bad", ["nan", "length"])
def test_g2_rejects_unaligned_or_nonfinite_arms(group, bad):
    from tttmem.p2.probe import g2_decision
    arrays = _g2_arrays()
    arrays[group]["lin"] = np.full(39 if bad == "length" else 40, np.nan if bad == "nan" else 1)
    L, G, Mm, ho = arrays
    with pytest.raises(ValueError):
        g2_decision(L, G, Mm, np.arange(40), ho, np.arange(40), n_boot=100)


@pytest.mark.parametrize("main_ok,ho_ok", [(False, True), (True, False), (False, False)])
def test_g2_minimum_cluster_support(main_ok, ho_ok):
    from tttmem.p2.probe import g2_decision
    L, G, Mm, ho = _g2_arrays()
    r = g2_decision(L, G, Mm, np.arange(40) if main_ok else np.zeros(40),
                    ho, np.arange(40) if ho_ok else np.zeros(40), n_boot=100)
    assert r["unsupported_clusters"] and not r["G2"]
    assert r["pass"]["D"] == main_ok and r["pass"]["ho"] == ho_ok
    assert np.isfinite(r["delta_M"]["lo"])       # bounds remain descriptive


def test_assay_support_and_finiteness():
    from tttmem.p2.probe import assay
    v, clusters = np.ones(40), np.arange(40)
    assert assay(v, v, clusters, n_boot=100)["pass"]
    r = assay(v, v, clusters % 19, n_boot=100)
    assert not r["pass"] and r["unsupported_clusters"] and r["M0"]["lo"] == 1
    for a, g in ((v[:-1], v), (v, np.full(40, np.inf))):
        with pytest.raises(ValueError):
            assay(a, g, clusters, n_boot=100)


def test_joint_power_increases_with_N_and_null_is_alpha_level():
    from tttmem.p2.probe import joint_power
    rng = np.random.default_rng(22)
    clusters = np.repeat(np.arange(100), 2)
    pool = np.repeat(np.arange(512), 2)          # frozen endpoint pool, 2 docs per domain
    ho_pool = np.arange(200)
    noise = rng.normal(0, 0.4, 100)[clusters] + rng.normal(0, 0.1, len(clusters))
    contrasts = {k: noise.copy() for k in ("D", "CmD", "dM", "RM", "M_rule")}
    alt = {k: 0.15 for k in contrasts}
    kw = dict(pool_clusters=pool, ho_pool_clusters=ho_pool)
    p = joint_power(contrasts, clusters, np.zeros(200), clusters, alt,
                    [64, 256, 1024], n_sim=500, seed=4, **kw)
    assert p[64] < p[256] < p[1024] and p[1024] > 0.8
    # Perfectly correlated null conjuncts give a joint rejection near alpha.
    p0 = joint_power(contrasts, clusters, np.zeros(200), clusters,
                     {k: 0.0 for k in contrasts}, [128], n_sim=1000, seed=5, **kw)
    assert 0.015 <= p0[128] <= 0.085
    # the held-out conjunct is sized at the endpoint's fixed held-out split
    ho = rng.normal(0, 0.15, 200)
    small = joint_power(contrasts, clusters, ho, clusters, alt, [1024], n_sim=300, seed=6, **kw)
    big = joint_power(contrasts, clusters, ho, clusters, alt, [1024], n_sim=300, seed=6,
                      pool_clusters=pool, ho_pool_clusters=np.arange(800))
    assert big[1024] > small[1024] + 0.1
    # the endpoint design sets the cluster count: a pool whose prefix has 10 domains
    # cannot pass the 20-cluster gate, so its power is 0 even though the pilot is rich
    few = joint_power(contrasts, clusters, np.zeros(200), clusters, alt, [1024], n_sim=50,
                      pool_clusters=np.repeat(np.arange(10), 103), ho_pool_clusters=ho_pool)
    assert few[1024] == 0.0
    # a one-domain pilot is unsupported (it would give zero SE and power 1)
    with pytest.raises(ValueError, match="pilot"):
        joint_power(contrasts, np.zeros(200, int), np.zeros(200), clusters, alt, [64],
                    n_sim=10, **kw)
    with pytest.raises(ValueError, match="exceeds"):
        joint_power(contrasts, clusters, np.zeros(200), clusters, alt, [2048], n_sim=10, **kw)
    # larger design clusters (fewer, bigger domains) lower power at the same N
    coarse = joint_power(contrasts, clusters, np.zeros(200), clusters, alt, [256], n_sim=500,
                         seed=4, pool_clusters=np.repeat(np.arange(256), 4)[:1024],
                         ho_pool_clusters=ho_pool)
    assert coarse[256] < p[256]
    contrasts["D"][0] = np.nan
    with pytest.raises(ValueError):
        joint_power(contrasts, clusters, np.zeros(200), clusters, alt, [64], n_sim=10, **kw)


@torch.no_grad()
def _dense_eval_reference(model, ids, tgt, mask, inner, im, barriers):
    """Dense head and zero-gradient VJPs, including the never-adapted AICE stream."""
    B, T = ids.shape
    C, S = model.cfg.chunk, model.cfg.suffix_len
    n = T // C
    starts = M._segment_starts(n, barriers)
    mask = M.barrier_mask(mask, starts, C)
    W = M._fast_values(model, B)
    W0 = [w.clone() for w in W]
    out = torch.zeros(B, T, dtype=torch.float64)
    cot_norm = torch.zeros(B, T, dtype=torch.float64)
    for c in range(n):
        sl = slice(c * C, (c + 1) * C)
        if c in starts:
            end = min([s for s in starts if s > c] + [n])
            h = model.prefix_forward(ids[:, c * C:end * C])
            seg0, past, past0 = c, [], []
        hc = h[:, (c - seg0) * C:(c - seg0 + 1) * C]
        Wc = [w.detach().requires_grad_(c < n - 1) for w in W]
        with torch.enable_grad():
            x, new = model.suffix_chunk(hc, Wc, M._as_past(M._cat(past, S), hc, model),
                                        math_backend=True)
            past.append([(k.detach(), v.detach()) for k, v in new])
            z = model.head(x.detach())
            out[:, sl] = M.token_nll(z, tgt[:, sl])
            if c < n - 1:
                z0 = None
                if inner.rule == "aice":
                    with torch.no_grad():
                        x0, new0 = model.suffix_chunk(hc, W0, M._as_past(M._cat(past0, S), hc, model),
                                                      math_backend=True)
                        past0.append([(k.detach(), v.detach()) for k, v in new0])
                        z0 = model.head(x0)
                m = mask[:, sl]
                w = (im[:, sl] & m).double() / M.denom(m).double().unsqueeze(1)
                cot = M.inner_x_cotangent(model, inner.rule, z, tgt[:, sl], w, z0)
                # per-token write strength from the unweighted single-token cotangent
                for t in range(C):
                    for b in range(B):
                        if w[b, t] > 0:
                            one = torch.zeros_like(w)
                            one[b, t] = 1.0
                            cot_norm[b, c * C + t] = M.inner_x_cotangent(
                                model, inner.rule, z, tgt[:, sl], one, z0)[b, t].norm()
                g = torch.autograd.grad(x, Wc, grad_outputs=cot)
                gc, _ = M.clip_per_seq(list(g), inner.clip)
                W = [w_ - inner.lr * t for w_, t in zip(W, gc)]
    return out, W, cot_norm


@pytest.mark.parametrize("rule", ["ce", "aice"])
def test_frozenP_sparse_matches_dense_and_skips_zero_VJP(rule, monkeypatch):
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 64, seed=23)
    im = torch.ones_like(mask)
    im[:, 32:] = False
    im[:, :8] = False                           # AICE must advance past0 before chunk 1
    pos = torch.zeros_like(mask)
    pos[:, 3::5] = True
    inner = M.InnerCfg(rule=rule, clip=0.2)
    ref, fast, ref_cot = _dense_eval_reference(m, ids, tgt, mask, inner, im, [2, 4])
    dense = M.stream_eval(m, ids, tgt, mask, inner, inner_mask=im, barriers=[2, 4], fused_attn=False)
    calls = []
    grad = torch.autograd.grad

    def count_grad(*args, **kw):
        calls.append(1)
        return grad(*args, **kw)

    monkeypatch.setattr(torch.autograd, "grad", count_grad)
    sp = M.stream_eval(m, ids, tgt, mask, inner, inner_mask=im, barriers=[2, 4],
                       nll_positions=pos, fused_attn=False, return_fast=True,
                       return_cot_norm=True)
    assert len(calls) == 3 and sp["inner_norm"].shape == (7, 2)
    assert torch.allclose(sp["cot_norm"], ref_cot, atol=1e-12, rtol=1e-10)
    bm = M.barrier_mask(mask, M._segment_starts(8, [2, 4]), 8)
    assert (ref_cot > 0).sum() == (im & bm)[:, :56].sum() > 0
    assert torch.allclose(sp["nll"][pos], ref[pos], atol=1e-12, rtol=0)
    assert torch.allclose(dense["nll"], ref, atol=1e-12, rtol=0)
    assert torch.allclose(sp["nll"][pos], dense["nll"][pos], atol=1e-12, rtol=0)
    skip = torch.tensor([0, 4, 5, 6])
    assert sp["inner_norm"][skip].eq(0).all() and sp["clip_scale"][skip].eq(1).all()
    assert sp["update_norm"][skip].eq(0).all()
    for a, b in zip(sp["fast"], fast):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


@pytest.mark.parametrize("rule", ["ce", "lin", "aice"])
def test_clip_scale_replay_is_exact(rule):
    m = model64(fast_scale=4.0)
    ids, tgt, mask = make_batch(m.cfg, 2, 64, seed=24)
    inner = M.InnerCfg(rule=rule, clip=0.15)
    r = M.stream_eval(m, ids, tgt, mask, inner, barriers=[2, 4], return_fast=True)
    assert (r["clip_scale"] < 1).any() and r["clip_scale"].shape == (7, 2)
    replay = M.stream_eval(m, ids, tgt, mask, inner, barriers=[2, 4], return_fast=True,
                           clip_scales=r["clip_scale"])
    for key in ("nll", "clip_scale", "inner_norm", "update_norm"):
        assert torch.equal(r[key], replay[key]), key
    assert all(torch.equal(a, b) for a, b in zip(r["fast"], replay["fast"]))
    with pytest.raises(ValueError):
        M.stream_eval(m, ids, tgt, mask, inner, clip_scales=torch.ones(6, 2))


def test_eval_source_nll_clip_statistics_and_reference_mode():
    from tttmem.p2.evaluate import eval_docs, prepare_probes
    m = model64(fast_scale=4.0)
    rng = np.random.default_rng(25)
    docs = rng.integers(3, m.cfg.vocab_size, size=(3, 65))
    for d in docs:
        for t in (3, 35):
            _plant(d, t, np.arange(10, 14))
    probes = prepare_probes(docs, 8, bos=m.cfg.bos_id)
    assert all(len(p.pos) and p.nf_ok for p in probes)
    kw = dict(device="cpu", batch=2, inner_clip=0.15, fused_attn=False)
    r = eval_docs(m, docs, probes, "ce", modes=["fP_focal_refclip", "adapt", "frozenP"], **kw)
    implicit = eval_docs(m, docs, probes, "ce", modes=["fP_focal_refclip"], **kw)
    np.testing.assert_array_equal(r["doc"]["fP_focal_refclip"], implicit["doc"]["fP_focal_refclip"])
    norms, scales = [], []
    for b0 in range(0, 3, 2):
        ids = torch.from_numpy(docs[b0:b0 + 2, :-1])
        tgt = torch.from_numpy(docs[b0:b0 + 2, 1:])
        im = torch.ones_like(tgt, dtype=torch.bool)
        im[:, 32:] = False
        for mode in ("adapt", "frozenP"):
            full = M.stream_eval(m, ids, tgt, tgt != m.cfg.bos_id, M.InnerCfg(clip=0.15),
                                 barriers=[2, 4], inner_mask=im if mode == "frozenP" else None,
                                 fused_attn=False)
            for i, p in enumerate(probes[b0:b0 + 2]):
                assert abs(r["src_nll"][mode][b0 + i] - float(full["nll"][i, p.focal].mean())) < 1e-12
            if mode == "adapt":
                norms.append(full["inner_norm"])
                scales.append(full["clip_scale"])
    clipped = (torch.cat(norms, 1) > 0.15).double()
    np.testing.assert_allclose(r["doc_clip_frac"], clipped.mean(0).numpy(), atol=1e-12)
    np.testing.assert_allclose(r["clip_frac"], clipped.mean(1).numpy(), atol=1e-12)
    np.testing.assert_allclose(r["clip_scale"], torch.cat(scales, 1).mean(1).numpy(), atol=1e-12)


def test_train_assay_gate_aborts_before_training(tmp_path, monkeypatch, capsys):
    from tttmem.p2 import probe
    from tttmem.p2 import train as T
    data, seq_len = _smoke_data(tmp_path)
    monkeypatch.setattr(probe, "assay", lambda *a, **kw: {"pass": False, "M0": {"lo": -1}})

    def no_training(*args, **kw):
        raise AssertionError("assay failure must prevent training")

    monkeypatch.setattr(T, "outer_microstep", no_training)
    cfg = T.TrainCfg(preset="tiny", dtype="fp32", device="cpu", seq_len=seq_len,
                     steps=1, global_batch=4, micro_batch=2, data_dir=str(data),
                     out_dir=str(tmp_path / "out"), min_probes=1, n_eval=4, ho_docs=3,
                     pilot_docs=2, eval_steps=[], save_checkpoints=False)
    r = T.train(cfg)
    assert r["aborted"] and not r["assay"]["pass"]
    out = capsys.readouterr().out
    assert "P2_ASSAY " in out and "P2_STEP " not in out


def test_train_rejects_wrong_fallback_lr(tmp_path):
    from tttmem.p2.train import TrainCfg, train
    with pytest.raises(ValueError, match="attempt 1 requires"):
        train(TrainCfg(attempt=1, lr_peak=3e-3))


def test_train_requires_exclusions(tmp_path):
    from tttmem.p2.train import TrainCfg, train
    with pytest.raises(ValueError, match="required exclusions missing"):
        train(TrainCfg(data_dir=str(tmp_path)))


def _decision_dirs(tmp_path):
    rng = np.random.default_rng(26)
    N = 80
    sha = np.array([f"eval{i}" for i in range(N)])
    domain = np.array([f"d{i % 40}.org" for i in range(N)])
    ho_sha = np.array([f"ho{i}" for i in range(80)])
    ho_domain = domain.copy()
    noise = rng.normal(0, 0.002, N)
    arms = {}
    for arm, loss, memory, gain in (("e2e2", 1.0, 0.15, 0.3),
                                   ("e2e1-ce", 1.1, 0.025, 0.1),
                                   ("e2e1-lin", 1.01, 0.125, 0.25)):
        root = tmp_path / arm
        root.mkdir()
        arms[arm] = root
        np.savez(root / "eval_meta.npz", eval_rows=np.arange(N), eval_sha1=sha,
                 eval_domain=domain, ho_sha1=ho_sha, ho_domain=ho_domain)
        for step, n, nh in ((0, N, 80), (60, 40, 40), (400, N, 80)):
            l = np.full(n, 1.0 if step == 0 else loss) + noise[:n]
            m = np.full(n, 0.10 if step == 0 else memory) + noise[:n]
            np.savez(root / f"eval_step{step:04d}.npz", doc_adapt=l,
                     doc_noadapt=l + (0.2 if step == 0 else gain),
                     doc_fP_focal=l + m, doc_fP_nonfocal=l, doc_heldout=np.ones(nh),
                     eval_sha1=sha[:n], eval_domain=domain[:n],
                     ho_sha1=ho_sha[:nh], ho_domain=ho_domain[:nh])
    return arms


def test_decision_coordinator_sizing_and_endpoint(tmp_path):
    from tttmem.p2.decide import assay_decision, sizing, endpoint
    arms = _decision_dirs(tmp_path)
    decision_dir = tmp_path / "decision"
    assert assay_decision(arms["e2e1-ce"], n_boot=200)["pass"]
    with pytest.raises(ValueError, match="requires sizing"):
        endpoint(arms, decision_dir)
    with pytest.raises(ValueError, match="held-out split has 80"):
        sizing(arms, decision_dir=decision_dir, Ns=(24,), n_sim=10)   # default ho_n 512
    r = sizing(arms, decision_dir=decision_dir, Ns=(24, 40, 128), n_sim=100, ho_n=80)
    assert r["N_eval"] <= 80 and set(r["powers"]) == {24, 40, 80}
    assert r["domains"] == {24: 24, 40: 40, 80: 40} and r["ho_domains"] == 40
    assert all(lo == max(0.0, 2 * r["powers"][n] - 1) and hi == r["powers"][n]
               for n, (lo, hi) in r["two_stage_bounds"].items())
    assert r["alternatives"]["dM"] == 0.5 * r["assay"]["M0"]["est"]
    assert all(len(v) == 40 for v in r["inputs"].values())
    assert json.loads((decision_dir / "sizing.json").read_text())["N_eval"] == r["N_eval"]
    with pytest.raises(FileExistsError):
        sizing(arms, decision_dir=decision_dir, n_sim=10, ho_n=80)
    g = endpoint(arms, decision_dir, n_boot=200)
    assert g["n_docs"] == r["N_eval"] and g["n_ho"] == 80 and g["G2"]
    assert json.loads((decision_dir / "g2.json").read_text())["G2"]
    with pytest.raises(FileExistsError):
        endpoint(arms, decision_dir, n_boot=200)


def test_decision_rejects_misaligned_documents_and_changed_pool(tmp_path):
    from tttmem.p2.decide import sizing, endpoint
    arms = _decision_dirs(tmp_path)
    size_dir = tmp_path / "decision"
    sizing(arms, decision_dir=size_dir, Ns=(24,), n_sim=50, ho_n=80)
    # the endpoint must use exactly the frozen held-out count
    path = arms["e2e2"] / "eval_step0400.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    short = {**data, "doc_heldout": data["doc_heldout"][:79], "ho_sha1": data["ho_sha1"][:79],
             "ho_domain": data["ho_domain"][:79]}
    for root in arms.values():
        np.savez(root / "eval_step0400.npz", **{**dict(np.load(root / "eval_step0400.npz")),
                                                **{k: short[k] for k in
                                                   ("doc_heldout", "ho_sha1", "ho_domain")}})
    with pytest.raises(ValueError, match="sizing froze 80"):
        endpoint(arms, size_dir, n_boot=100)
    for root in arms.values():
        np.savez(root / "eval_step0400.npz", **{**dict(np.load(root / "eval_step0400.npz")),
                                                **{k: data[k] for k in
                                                   ("doc_heldout", "ho_sha1", "ho_domain")}})
    path = arms["e2e1-lin"] / "eval_step0400.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["eval_sha1"] = data["eval_sha1"][::-1]
    np.savez(path, **data)
    with pytest.raises(ValueError, match="document order"):
        endpoint(arms, size_dir, n_boot=100)
    data["eval_sha1"] = data["eval_sha1"][::-1]
    np.savez(path, **data)
    for root in arms.values():
        path = root / "eval_meta.npz"
        with np.load(path) as z:
            meta = {k: z[k] for k in z.files}
        meta["annotation"] = np.array([1])
        np.savez(path, **meta)
    with pytest.raises(ValueError, match="pool changed"):
        endpoint(arms, size_dir, n_boot=100)
    assert not (size_dir / "g2.json").exists()


def test_decision_CLI(tmp_path, monkeypatch, capsys):
    from tttmem.p2 import decide
    arms = _decision_dirs(tmp_path)
    monkeypatch.setattr(sys, "argv", ["decide", "assay", str(arms["e2e1-ce"])])
    decide.main()
    assert json.loads(capsys.readouterr().out)["pass"]
    common = [arg for name, path in arms.items() for arg in ("--arm", f"{name}={path}")]
    common += ["--decision-dir", str(tmp_path / "decision")]
    monkeypatch.setattr(sys, "argv", ["decide", "size", *common, "--Ns", "24", "40",
                                      "--ho-n", "80"])
    decide.main()
    assert json.loads(capsys.readouterr().out)["N_eval"] == 24
    monkeypatch.setattr(sys, "argv", ["decide", "endpoint", *common])
    decide.main()
    assert json.loads(capsys.readouterr().out)["G2"]
