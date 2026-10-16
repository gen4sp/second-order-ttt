import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tttmem.p2.evaluate import eval_docs
from tttmem.p2.meta import (InnerCfg, _as_past, _fast_values, residual_sigma, stream_eval,
                            stream_gain_split, token_multiplier)
from tttmem.p2.mini import _write_tiny_data, mini_g0
from tttmem.p2.model import build_model
from tttmem.p2.probe import DocProbes, boundary_valid
from tttmem.p2.train import TrainCfg, train


BARRIERS = [2, 4, 6]


def _case(seed=3, B=2):
    model = build_model("tiny2", compute_dtype=torch.float64, seed=seed).double()
    rng = np.random.default_rng(seed)
    docs = rng.integers(3, 101, size=(B, 65)).astype(np.int64)
    docs[:, 0] = 1
    ids = torch.from_numpy(docs[:, :-1])
    tgt = torch.from_numpy(docs[:, 1:])
    return model, docs, ids, tgt, tgt != 1


def _probe(T=64):
    focal = np.zeros(T, dtype=bool)
    focal[[3, 9, 17, 26]] = True
    nonfocal = np.zeros(T, dtype=bool)
    nonfocal[[4, 12, 19, 27]] = True
    pos = np.asarray([40, 52, 60], dtype=np.int64)
    return DocProbes(pos=pos, stratum=np.zeros(3, dtype=np.int64),
                     n_src=np.ones(3, dtype=np.int64), dist=np.ones(3, dtype=np.int64),
                     src_gap=np.zeros(3, dtype=np.int64), match8=np.zeros(3, dtype=bool),
                     focal=focal, nonfocal=nonfocal, nf_ok=True)


def _frozen_mask(valid, C):
    im = torch.ones_like(valid)
    im[:, 4 * C:] = False
    return im


def test_none_multiplier_is_bitwise_baseline():
    model, docs, ids, tgt, valid = _case()
    C = model.cfg.chunk
    a = stream_eval(model, ids, tgt, valid, InnerCfg(lr=0.7, clip=1.0), barriers=BARRIERS,
                    fused_attn=False, inner_mask=_frozen_mask(valid, C))
    b = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.7, clip=1.0, tok_mult="none", sigma_ref=3.0, const_mult=2.0),
                    barriers=BARRIERS, fused_attn=False, inner_mask=_frozen_mask(valid, C),
                    return_sigma=True)
    assert torch.equal(a["nll"], b["nll"])
    assert torch.equal(a["clip_scale"], b["clip_scale"])
    assert "mult_mean" not in b and b["sigma"].shape == valid.shape
    probes = [_probe(), _probe()]
    kw = dict(modes=("fP_focal", "fP_nonfocal"), batch=2, inner_lr=0.7, inner_clip=1.0,
              device="cpu", fused_attn=False, barriers=BARRIERS)
    ra = eval_docs(model, docs, probes, "ce", **kw)
    rb = eval_docs(model, docs, probes, "ce", inner_extra={"tok_mult": "none"}, **kw)
    for m in ("fP_focal", "fP_nonfocal"):
        np.testing.assert_array_equal(ra["doc"][m], rb["doc"][m])
    assert set(rb["rule_stats"]) == {"fP_focal", "fP_nonfocal"}


@pytest.mark.parametrize("clip", [0.0, 1e6])
def test_const_multiplier_equals_scaled_eta_without_binding_clip(clip):
    model, docs, ids, tgt, valid = _case(seed=5)
    C = model.cfg.chunk
    im = _frozen_mask(valid, C)
    a = stream_eval(model, ids, tgt, valid, InnerCfg(lr=0.3 * 1.7, clip=clip),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    b = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.3, clip=clip, tok_mult="const", const_mult=1.7),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    torch.testing.assert_close(a["nll"], b["nll"], rtol=1e-10, atol=1e-12)
    assert b["mult_mean"] == pytest.approx(1.7)
    bvalid = torch.from_numpy(boundary_valid(64, C, BARRIERS))
    assert b["mult_count"] == int((im & valid & bvalid).sum())  # barrier_mask drops pre-barrier targets


def test_sigma_cap_saturates_to_const_and_sigma_reports_final_norm():
    model, docs, ids, tgt, valid = _case(seed=6)
    C = model.cfg.chunk
    im = _frozen_mask(valid, C)
    # sigma / 1e-9 >> cap everywhere: the sigma rule degenerates to the constant cap.
    a = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.5, clip=1.0, tok_mult="sigma", sigma_ref=1e-9, sigma_cap=0.6),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    b = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.5, clip=1.0, tok_mult="const", const_mult=0.6),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    torch.testing.assert_close(a["nll"], b["nll"], rtol=1e-12, atol=1e-12)
    # adapt=False: sigma at W_0 equals the residual sigma of a direct suffix pass.
    r = stream_eval(model, ids, tgt, valid, InnerCfg(), adapt=False, barriers=BARRIERS,
                    fused_attn=False, return_sigma=True)
    h = model.prefix_forward(ids[:, :2 * C])
    cap = {}
    model.suffix_chunk(h[:, :C], _fast_values(model, ids.shape[0]),
                       _as_past([None] * model.cfg.suffix_len, h[:, :C], model),
                       math_backend=True, segment_pos=0, capture=cap)
    torch.testing.assert_close(r["sigma"][:, :C], residual_sigma(cap["h_pre"], model.cfg.rms_eps))


def test_sigma_shuffle_keeps_row_multiset_and_is_deterministic():
    B, C = 3, 16
    sigma = torch.rand(B, C, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) + .2
    write = torch.rand(B, C, generator=torch.Generator().manual_seed(1)) > .3
    inner = InnerCfg(tok_mult="sigma_shuf", sigma_ref=.5, sigma_cap=1.8, seed=11,
                     row_ids=torch.tensor([4, 5, 6]))
    base = token_multiplier(InnerCfg(tok_mult="sigma", sigma_ref=.5, sigma_cap=1.8), sigma, write)
    torch.testing.assert_close(base, (sigma / .5).clamp_max(1.8))
    s1 = token_multiplier(inner, sigma, write, chunk=2)
    s2 = token_multiplier(inner, sigma, write, chunk=2)
    s3 = token_multiplier(inner, sigma, write, chunk=3)
    assert torch.equal(s1, s2) and not torch.equal(s1, s3)
    for b in range(B):
        w = write[b]
        assert torch.equal(torch.sort(s1[b, w]).values, torch.sort(base[b, w]).values)
        assert torch.equal(s1[b, ~w], base[b, ~w])
    assert not torch.equal(s1, base)
    with pytest.raises(ValueError):
        InnerCfg(rule="lin", tok_mult="sigma")
    with pytest.raises(ValueError):
        InnerCfg(tok_mult="sigma", sigma_ref=0.0)


def test_gain_split_matches_autograd_and_direct_per_token_gradient():
    model, docs, ids, tgt, valid = _case(seed=9)
    C, S = model.cfg.chunk, model.cfg.suffix_len
    B = ids.shape[0]
    sf = torch.zeros_like(valid)
    sn = torch.zeros_like(valid)
    sf[0, [2, 5, 11]] = True
    sn[0, [3, 14, 30]] = True
    sf[1, [1, 9, 25]] = True
    inner = InnerCfg(lr=0.8, clip=1.0)
    r = stream_gain_split(model, ids, tgt, valid, sf, sn, inner, barriers=BARRIERS)
    n = int((sf | sn).sum())
    assert len(r["gnorm"]) == n and r["gblk"].shape == (n, S)
    assert float(r["ch_err"].max()) < 1e-10
    chain = r["e"] * r["u"] * r["gam"] * r["proj"] * r["s"]
    assert torch.all(chain > 0) and torch.all(torch.isfinite(r["gnorm"] / chain))
    torch.testing.assert_close(r["gblk"].pow(2).sum(1).sqrt(), r["gnorm"])
    assert torch.equal(r["focal"], sf[r["row"], r["pos"]])
    assert set(r["chunk"].tolist()) <= {0, 1, 2, 3}

    # Direct per-token gradients in segment 0 (chunks 0-1): chunk 0 at W_0; chunk 1 at
    # W_1 = W_0 - lr * clip(g_chunk0), with chunk 0's K/V computed at W_0.
    W0 = [w.detach().clone() for w in _fast_values(model, B)]
    h = model.prefix_forward(ids[:, :2 * C])
    m0 = valid[:, :C]
    Wg = [w.clone().requires_grad_(True) for w in W0]
    x0, new0 = model.suffix_chunk(h[:, :C], Wg, _as_past([None] * S, h[:, :C], model),
                                  math_backend=True, segment_pos=0)
    nll0 = F.cross_entropy(model.head(x0).flatten(0, 1), tgt[:, :C].flatten(),
                           reduction="none").view(B, C)
    loss0 = ((nll0 * m0) / m0.sum(1, keepdim=True).clamp_min(1)).sum()
    g0 = torch.autograd.grad(loss0, Wg)
    sq = sum(g.pow(2).reshape(B, -1).sum(1) for g in g0)
    scale = 1.0 / sq.sqrt().clamp_min(1.0)
    W1 = [w - 0.8 * g * scale.view(B, *([1] * (g.dim() - 1))) for w, g in zip(W0, g0)]
    past = [[(k.detach(), v.detach()) for k, v in new0]]
    for i in range(n):
        b, t = int(r["row"][i]), int(r["pos"][i])
        if t >= 2 * C:
            continue
        c = t // C
        Wc = [(W0 if c == 0 else W1)[j].detach().clone().requires_grad_(True) for j in range(3 * S)]
        hc = h[:, c * C:(c + 1) * C]
        pst = _as_past([None] * S, hc, model) if c == 0 else _as_past(
            [(torch.cat([pc[j][0] for pc in past], 1), torch.cat([pc[j][1] for pc in past], 1))
             for j in range(S)], hc, model)
        x, _ = model.suffix_chunk(hc, Wc, pst, math_backend=True, segment_pos=c * C)
        ce = F.cross_entropy(model.head(x[b, t - c * C][None]), tgt[b, t][None])
        g = torch.autograd.grad(ce, Wc)
        direct = float(torch.sqrt(sum(q[b].pow(2).sum() for q in g)))
        assert direct == pytest.approx(float(r["gnorm"][i]), rel=1e-9)
        assert float(r["nll"][i]) == pytest.approx(float(ce), rel=1e-9)


def _tiny_cfg(data, out, *, arm="e2e1-ce"):
    return TrainCfg(
        arm=arm, preset="tiny", dtype="fp32", device="cpu", init_seed=0,
        seq_len=64, global_batch=2, micro_batch=1, steps=2,
        warmup=1, lr_peak=1e-2, inner_lr=1.0, data_dir=str(data), data_seed=0,
        out_dir=str(out), n_eval=4, min_probes=1,
        eval_steps=[2], full_eval_steps=[2], sizing_steps=[], trajectory_docs=4,
        ho_docs=2, ho_trajectory_docs=2, pilot_docs=2, eval_batch=2,
        pilot_steps=[], assay_gate=False, crossover_rules=[],
        log_every=1, resume_every=2, save_checkpoints=True,
        barriers=[2], segment_len=16, fused_attn=False, eval_at_init=False,
        eval_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
        trajectory_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
        full_extra_modes=[], process_name="TTM-g0-91-test",
    )


def test_mini_g0_end_to_end_tiny(tmp_path, capsys):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=8)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    out = tmp_path / "g0"
    reads = ["ce", "eta150", "sig4", "sig4shuf", "const4"]
    result = mini_g0({
        "process_nn": 91, "process_name": "TTM-g0-91-test", "data_dir": str(data),
        "device": "cpu", "out_dir": str(out), "reads": reads, "n_docs": 4,
        "n_split_docs": 4, "split_k": 4, "split_batch": 2, "n_ho_adapt": 2,
        "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0, "data_seed": 0,
                   "step": 2, "split": True, "ckpt": str(cell / "model_step0002.pt"),
                   "out": str(out / "tiny" / "e2e1-ce" / "s0")}],
    })
    assert result["done"] == 1
    lines = capsys.readouterr().out.splitlines()
    assert any(x.startswith("G0_V0 ") and '"pass": true' in x for x in lines)
    got = {json.loads(x[len("G0_READ "):])["read"]: json.loads(x[len("G0_READ "):])
           for x in lines if x.startswith("G0_READ ")}
    assert list(got) == reads
    calib = [json.loads(x[len("G0_CALIB "):]) for x in lines if x.startswith("G0_CALIB ")][0]
    assert calib["sigma_ref"] > 0 and calib["const_mult"] > 0
    assert got["const4"]["stats"]["fP_focal"]["mult_mean"] == pytest.approx(calib["const_mult"])
    assert np.isnan(got["ce"]["stats"]["fP_focal"]["mult_mean"])  # no multiplier, no mean
    cell_out = out / "tiny" / "e2e1-ce" / "s0"
    with np.load(cell / "eval_step0002.npz") as logged, \
            np.load(cell_out / "g0_read_ce.npz") as z:
        np.testing.assert_allclose(z["M_doc"], logged["doc_fP_focal"] - logged["doc_fP_nonfocal"],
                                   rtol=1e-6, atol=1e-7)
        assert z["L_adapt_doc"].shape == (2,)
    with np.load(cell_out / "gain_split.npz") as z:
        assert z["gnorm"].size > 0 and float(np.max(z["ch_err"])) < 1e-3
    assert (out / "g0_summary.json").exists()


def test_mini_g0_missing_checkpoint_fails_closed(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="skipped"):
        mini_g0({"process_nn": 91, "out_dir": str(tmp_path), "reads": ["ce"],
                 "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0,
                            "ckpt": str(tmp_path / "nope.pt")}]})
    assert "G0_SKIP" in capsys.readouterr().out


def _fake_cell(root, arm, seed, M, L, split_logs=None):
    from pathlib import Path
    d = Path(root) / "mini-m" / arm / f"s{seed}"
    d.mkdir(parents=True)
    for name, m in M.items():
        np.savez(d / f"g0_read_{name}.npz", M_doc=np.full(4, m), L_adapt_doc=np.full(2, L[name]),
                 stats=np.asarray(json.dumps({"fP_focal": {"mult_mean": 1.0, "clip_bind_frac": .5}})),
                 ckpt_sha256=np.asarray(f"{arm}{seed}"), eval_rows=np.arange(4))
    if split_logs is not None:
        n = 6
        rec = {f: np.full(n, np.exp(v)) for f, v in split_logs.items()}
        rec["gnorm"] = np.full(n, np.exp(sum(split_logs.values())))
        rec["focal"] = np.array([1, 1, 1, 0, 0, 0], bool)
        rec["gblk"] = np.ones((n, 2))
        rec["H"] = np.linspace(1, 3, n)
        np.savez(d / "gain_split.npz", **rec)


def test_g0_analysis_labels(tmp_path):
    from tttmem.p2.g0_analysis import analyse
    fo_logs = {"e": 0., "u": 0., "gam": 0., "proj": 0., "s": 0.}
    for s, jit in enumerate((0., .01, -.01)):
        base = .30 + jit
        _fake_cell(tmp_path, "e2e2", s, {"ce": base + .17, "eta200": base + .20},
                   {"ce": 3.0, "eta200": 3.0},
                   {**fo_logs, "s": .30 + jit, "e": .02})
        _fake_cell(tmp_path, "e2e1-ce", s,
                   {"ce": base, "eta125": base + .05, "eta150": base + .10, "eta200": base + .12,
                    "sig4": base + .02, "sig4shuf": base + .02 + jit, "const4": base + .02},
                   {"ce": 3.0, "eta125": 3.0, "eta150": 3.005, "eta200": 3.05,
                    "sig4": 3.0, "sig4shuf": 3.0, "const4": 3.0}, fo_logs)
    res = analyse(tmp_path)
    t = res["titration"]
    assert t["B1"]["dose"] == "eta150"            # eta200 fails the LM guard
    assert t["B1"]["label"] == "GAIN-LIMITED"     # kappa(eta150) = .10 / .17
    assert t["B2"]["label"] == "SIGMA-NULL"
    d = res["decomposition"]
    assert d["label"] == "SIGMA-MAJOR"
    assert d["shares"]["SIGMA"] == pytest.approx(.30 / .32, rel=1e-6)


def test_mini_g0_split_gate_and_read_order_fail_closed(tmp_path, capsys):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=8)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    out = tmp_path / "g0"
    item = {"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0, "data_seed": 0, "step": 2,
            "split": True, "ckpt": str(cell / "model_step0002.pt"),
            "out": str(out / "tiny" / "e2e1-ce" / "s0")}
    base = {"process_nn": 91, "data_dir": str(data), "device": "cpu", "out_dir": str(out),
            "n_docs": 4, "n_split_docs": 2, "split_k": 2, "split_batch": 2, "n_ho_adapt": 2}
    with pytest.raises(RuntimeError, match="error"):
        mini_g0({**base, "reads": ["ce"], "ch_err_median_max": -1.0, "items": [item]})
    cell_out = out / "tiny" / "e2e1-ce" / "s0"
    assert (cell_out / "g0_read_ce.npz").exists()            # reads come first
    assert (cell_out / "gain_split_FAILED.npz").exists()
    assert not (cell_out / "gain_split.npz").exists()
    assert "G0_ERROR" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="error"):
        mini_g0({**base, "reads": ["eta150", "ce"], "items": [{**item, "split": False}]})
    assert "must start with the V0" in capsys.readouterr().out


def test_inverse_sigma_multipliers_and_saturation():
    B, C = 2, 12
    sigma = torch.rand(B, C, dtype=torch.float64, generator=torch.Generator().manual_seed(3)) + .3
    write = torch.ones(B, C, dtype=torch.bool)
    inv = token_multiplier(InnerCfg(tok_mult="sigma_inv", sigma_ref=.8, sigma_cap=2.5), sigma, write)
    torch.testing.assert_close(inv, (.8 / sigma).clamp_max(2.5))
    shuf = token_multiplier(InnerCfg(tok_mult="sigma_inv_shuf", sigma_ref=.8, sigma_cap=2.5, seed=4),
                            sigma, write, chunk=1)
    for b in range(B):
        assert torch.equal(torch.sort(shuf[b]).values, torch.sort(inv[b]).values)
    assert not torch.equal(shuf, inv)
    # sigma_ref / sigma >> cap everywhere: the inverse rule degenerates to the constant cap.
    model, docs, ids, tgt, valid = _case(seed=12)
    im = _frozen_mask(valid, model.cfg.chunk)
    a = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.5, clip=1.0, tok_mult="sigma_inv", sigma_ref=1e9, sigma_cap=0.7),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    b = stream_eval(model, ids, tgt, valid,
                    InnerCfg(lr=0.5, clip=1.0, tok_mult="const", const_mult=0.7),
                    barriers=BARRIERS, fused_attn=False, inner_mask=im)
    torch.testing.assert_close(a["nll"], b["nll"], rtol=1e-12, atol=1e-12)


def test_mini_g0_inverse_and_frontier_reads_tiny(tmp_path, capsys):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=8)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    out = tmp_path / "g0b"
    reads = ["ce", "eta050", "eta250", "isig4", "isig4shuf", "iconst4"]
    mini_g0({"process_nn": 91, "data_dir": str(data), "device": "cpu", "out_dir": str(out),
             "reads": reads, "n_docs": 4, "n_ho_adapt": 2,
             "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0, "data_seed": 0,
                        "step": 2, "ckpt": str(cell / "model_step0002.pt"),
                        "out": str(out / "tiny" / "e2e1-ce" / "s0")}]})
    lines = capsys.readouterr().out.splitlines()
    got = {json.loads(x[8:])["read"]: json.loads(x[8:]) for x in lines if x.startswith("G0_READ ")}
    calib = [json.loads(x[len("G0_CALIB "):]) for x in lines if x.startswith("G0_CALIB ")][0]
    assert list(got) == reads and calib["const_mult_inv"] > 0
    assert got["iconst4"]["stats"]["fP_focal"]["mult_mean"] == pytest.approx(calib["const_mult_inv"])
    assert got["eta050"]["eta"] == .5 and got["eta250"]["eta"] == 2.5


def _fake_read(root, arm, seed, name, eta, M, L):
    from pathlib import Path
    d = Path(root) / "mini-m" / arm / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"g0_read_{name}.npz", M_doc=np.full(4, M), L_adapt=np.asarray(L), eta=np.asarray(eta))


def test_g0b_frontier_and_inverse_labels(tmp_path):
    from tttmem.p2.g0b_analysis import analyse, interp_at
    assert interp_at([{"eta": 1, "L": 1., "M": 0.}, {"eta": 2, "L": 2., "M": 1.}], 1.5) == pytest.approx(.5)
    assert np.isnan(interp_at([{"eta": 1, "L": 1., "M": 0.}, {"eta": 2, "L": 2., "M": 1.}], 2.5))
    g0, g0b = tmp_path / "g0", tmp_path / "g0b"
    for s, j in enumerate((0., .004, -.004)):
        # FO line M = .30 + 4 (L - 4.20); SO sits on the same line at L = 4.25 -> FRONTIER-MATCH
        for name, eta, L in (("ce", 1, 4.20), ("eta125", 1.25, 4.205), ("eta150", 1.5, 4.21), ("eta200", 2, 4.22)):
            _fake_read(g0, "e2e1-ce", s, name, eta, .30 + 4 * (L - 4.20) + j, L)
        for name, eta, L in (("ce", 1, 4.20), ("eta250", 2.5, 4.23), ("eta300", 3, 4.24), ("eta400", 4, 4.26)):
            _fake_read(g0b, "e2e1-ce", s, name, eta, .30 + 4 * (L - 4.20) + j, L)
        for name, eta, L, M in (("ce", 1, 4.25, .50 + j), ("eta200", 2, 4.27, .60)):
            _fake_read(g0, "e2e2", s, name, eta, M, L)
        for name, eta, L, M in (("ce", 1, 4.25, .50 + j), ("eta050", .5, 4.27, .40), ("eta075", .75, 4.26, .45)):
            _fake_read(g0b, "e2e2", s, name, eta, M, L)
        for name, M in (("isig4", .33), ("isig4shuf", .30), ("iconst4", .302)):
            _fake_read(g0b, "e2e1-ce", s, name, 1, M + j * 0, 4.2005)
    res = analyse(g0, g0b)
    fr, inv = res["frontier"], res["inverse_sigma"]
    assert fr["ce_identical"] and fr["label"] == "FRONTIER-MATCH"
    assert fr["rho_match"]["mean"] == pytest.approx(0, abs=1e-9)
    assert np.isnan(fr["rho_match_rev"]["mean"])          # SO cannot reach FO's L by lowering eta
    assert fr["per_seed"][0]["so_L_min_eta"] == 1
    assert inv["label"] == "SIGMA-INV-SELECTIVE"


def test_mini_g0_inverse_sigma_at_larger_eta_tiny(tmp_path, capsys):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=9)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    out = tmp_path / "g0c"
    reads = ["ce", "isig4", "isig4_eta150", "isig4_eta400"]
    mini_g0({"process_nn": 92, "data_dir": str(data), "device": "cpu", "out_dir": str(out),
             "reads": reads, "n_docs": 4, "n_ho_adapt": 2,
             "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0, "data_seed": 0,
                        "step": 2, "ckpt": str(cell / "model_step0002.pt"),
                        "out": str(out / "tiny" / "e2e1-ce" / "s0")}]})
    lines = capsys.readouterr().out.splitlines()
    got = {json.loads(x[8:])["read"]: json.loads(x[8:]) for x in lines if x.startswith("G0_READ ")}
    assert list(got) == reads
    for n, eta in (("isig4_eta150", 1.5), ("isig4_eta400", 4.0)):
        assert got[n]["eta"] == eta and got[n]["tok_mult"] == "sigma_inv"
        assert np.isfinite(got[n]["stats"]["fP_focal"]["mult_mean"])
    assert got["isig4_eta150"]["M"] != got["isig4"]["M"]


def test_g0c_lift_labels(tmp_path):
    from tttmem.p2.g0c_analysis import analyse, quad_at
    # L(eta) = eta^2, M(eta) = 2 eta - .1 eta^2: exact at L = 2.25 (eta = 1.5); no bracket beyond the curve
    q = [{"eta": e, "L": e * e, "M": 2 * e - .1 * e * e} for e in (1., 2., 3., 4.)]
    assert quad_at(q, 2.25) == pytest.approx(3 - .225, abs=1e-6)
    assert np.isnan(quad_at(q, 17.))

    def build(root, lift):
        g0, g0b, g0c = root / "g0", root / "g0b", root / "g0c"
        for s, j in enumerate((0., .004, -.004)):
            line = lambda L: .30 + 4 * (L - 4.20) + j  # noqa: E731
            for name, eta, L in (("ce", 1, 4.20), ("eta125", 1.25, 4.205), ("eta150", 1.5, 4.21), ("eta200", 2, 4.22)):
                _fake_read(g0, "e2e1-ce", s, name, eta, line(L), L)
            for name, eta, L in (("eta250", 2.5, 4.23), ("eta300", 3, 4.24), ("eta400", 4, 4.26)):
                _fake_read(g0b, "e2e1-ce", s, name, eta, line(L), L)
            _fake_read(g0b, "e2e1-ce", s, "isig4", 1, line(4.2005) + lift, 4.2005)
            for name, eta, L in (("isig4_eta150", 1.5, 4.212), ("isig4_eta250", 2.5, 4.232),
                                 ("isig4_eta300", 3, 4.243), ("isig4_eta400", 4, 4.265)):
                _fake_read(g0c, "e2e1-ce", s, name, eta, line(L) + lift, L)
            for root_, name, eta, L, M in ((g0, "ce", 1, 4.25, .50 + j), (g0, "eta200", 2, 4.27, .60),
                                           (g0b, "isig4", 1, 4.251, .53), (g0c, "eta150", 1.5, 4.26, .56),
                                           (g0c, "isig4_eta150", 1.5, 4.262, .59), (g0c, "isig4_eta200", 2, 4.268, .62)):
                _fake_read(root_, "e2e2", s, name, eta, M, L)
            for arm, M, L in (("e2e1-ce", line(4.20), 4.20), ("e2e2", .50 + j, 4.25)):
                _fake_read(g0c, arm, s, "ce", 1, M, L)
        return analyse(g0, g0b, g0c)

    up = build(tmp_path / "up", .03)                      # Delta M = .20 -> lambda = .15
    assert up["ce_identical"] and up["label"] == "ISIG-LIFT"
    assert up["lambda_lin"]["mean"] == pytest.approx(.15) and up["lambda_quad"]["mean"] == pytest.approx(.15)
    flat = build(tmp_path / "flat", 0.)
    assert flat["label"] == "ISIG-NO-LIFT" and flat["lambda_lin"]["mean"] == pytest.approx(0, abs=1e-9)
    assert build(tmp_path / "down", -.03)["label"] == "ISIG-BELOW"
    assert np.isfinite(up["rho_match_isig"]["mean"]) and np.isfinite(up["rho_match_both_isig"]["mean"])
