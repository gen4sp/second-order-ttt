import json

import numpy as np
import pytest
import torch

from tttmem.p2.evaluate import eval_docs
from tttmem.p2.meta import InnerCfg, stream_align, stream_eval
from tttmem.p2.mini import mini_align
from tttmem.p2.mini_analysis import minia_analysis
from tttmem.p2.model import build_model
from tttmem.p2.probe import DocProbes


BARRIERS = [2, 4, 6]


def _alignment_case():
    """Tiny fp64 case with probe targets in P chunks 4, 6 and 7."""
    rng = np.random.default_rng(7)
    docs = rng.integers(40, 97, size=(1, 65), dtype=np.uint32)
    docs[:, 0] = 1
    docs[0, 3:7] = [11, 12, 13, 14]
    docs[0, 36:40] = [11, 12, 13, 14]
    docs[0, 10:14] = [21, 22, 23, 24]
    docs[0, 50:54] = [21, 22, 23, 24]
    docs[0, 20:24] = [31, 32, 33, 34]
    docs[0, 58:62] = [31, 32, 33, 34]
    T = 64
    pos = np.asarray([34, 50, 58], dtype=np.int64)
    focal = np.zeros(T, dtype=bool)
    focal[[3, 10, 20]] = True
    nonfocal = np.zeros(T, dtype=bool)
    nonfocal[[4, 11, 21]] = True
    probe = DocProbes(pos=pos, stratum=np.zeros(3, dtype=np.int64),
                      n_src=np.ones(3, dtype=np.int64), dist=np.ones(3, dtype=np.int64),
                      src_gap=np.zeros(3, dtype=np.int64), match8=np.zeros(3, dtype=bool),
                      focal=focal, nonfocal=nonfocal, nf_ok=True)
    model = build_model("tiny", compute_dtype=torch.float64, seed=4).double()
    ids = torch.from_numpy(docs[:, :-1].astype(np.int64))
    tgt = torch.from_numpy(docs[:, 1:].astype(np.int64))
    valid = tgt != 1
    f = torch.from_numpy(focal[None])
    nf = torch.from_numpy(nonfocal[None])
    targets = torch.zeros((1, T), dtype=torch.bool)
    targets[0, torch.from_numpy(pos)] = True
    return model, docs, ids, tgt, valid, f, nf, targets, probe


def _unflat(flat, model):
    out, i = [], 0
    for p in model.prime_params():
        k = p.numel()
        out.append(flat[:, i:i + k].reshape(flat.shape[0], *p.shape))
        i += k
    return out


def _probe_nll(model, ids, tgt, valid, fast, targets, eta):
    r = stream_eval(model, ids, tgt, valid, InnerCfg(lr=eta, clip=1.0),
                    adapt=False, barriers=BARRIERS, fused_attn=False,
                    fast_init=fast, nll_positions=targets)
    return (r["nll"] * targets).sum(1)


def test_minia_finite_difference_and_barriers():
    model, docs, ids, tgt, valid, f, nf, targets, probe = _alignment_case()
    K = len(probe.pos)
    rels = []
    ratio_errors = []
    for eta in (1e-2, 1e-3):
        got = stream_align(model, ids, tgt, valid, f, nf,
                           InnerCfg(lr=eta, clip=1e6), barriers=BARRIERS,
                           probe_targets=targets, return_vectors=True)
        Wfin = torch.cat([p.detach().reshape(1, -1) for p in model.prime_params()], 1)
        Wfin = Wfin + got["full_applied"]
        dF, dN = got["vf_vec"].sum(1), got["vn_vec"].sum(1)
        step = 1e-5 / float(torch.cat((dF, dN), 1).norm(dim=1).max())
        fdF = (_probe_nll(model, ids, tgt, valid, _unflat(Wfin - step * dF, model), targets, eta)
               - _probe_nll(model, ids, tgt, valid, _unflat(Wfin + step * dF, model), targets, eta)) / (2 * step)
        fdN = (_probe_nll(model, ids, tgt, valid, _unflat(Wfin - step * dN, model), targets, eta)
               - _probe_nll(model, ids, tgt, valid, _unflat(Wfin + step * dN, model), targets, eta)) / (2 * step)
        relF = float(((fdF - got["bf"]).abs() / got["bf"].abs().clamp_min(1e-30)).max())
        relN = float(((fdN - got["bn"]).abs() / got["bn"].abs().clamp_min(1e-30)).max())
        rels.append(max(relF, relN))
        measured = eval_docs(model, docs, [probe], "ce",
                             modes=("fP_focal", "fP_nonfocal"), batch=1,
                             inner_lr=eta, inner_clip=1e6, device="cpu",
                             fused_attn=False, barriers=BARRIERS)
        M = float(measured["doc"]["fP_focal"][0] - measured["doc"]["fP_nonfocal"][0])
        pred = float((got["bf"][0] - got["bn"][0]) / K)
        ratio_errors.append(abs(pred / M - 1.0))
    # An unsplit reread fails this finite-difference assertion because chunk 6
    # would retain chunk 4--5's K/V and RoPE positions.
    assert max(rels) < 1e-6
    assert ratio_errors[1] < ratio_errors[0]
    assert ratio_errors[1] < .05


def test_minia_missing_checkpoint_is_skip(monkeypatch, tmp_path, capsys):
    from tttmem.p2 import mini as mini_mod

    def fake_item(item, cfg, repo, name):
        if item.get("good"):
            return {"cell": item["cell"], "status": "done"}
        rec = {"cell": item["cell"], "reason": "missing_checkpoint"}
        print("MINIA_SKIP " + json.dumps(rec), flush=True)
        return {**rec, "status": "skip"}

    monkeypatch.setattr(mini_mod, "_minia_item", fake_item)
    with pytest.raises(RuntimeError, match="skipped"):
        mini_align({"process_nn": 80, "out_dir": str(tmp_path / "out"),
                    "items": [{"cell": "good", "good": True},
                              {"cell": "missing", "good": False}]})
    output = capsys.readouterr().out
    assert "MINIA_SKIP" in output
    assert 'MINIA_DONE {"done": 1, "errors": 0, "skipped": 1}' in output


def test_minia_analysis_labels_and_ne(tmp_path):
    root = tmp_path / "a"
    for preset, so_cos, fo_cos in (("mini-s", .8, .4), ("mini-m", .5, .49)):
        for seed in range(3):
            for arm, cos in (("e2e2", so_cos), ("e2e1-ce", fo_cos)):
                p = root / preset / arm / f"s{seed}"; p.mkdir(parents=True)
                np.savez(p / "align.npz", preset=np.asarray(preset), arm=np.asarray(arm),
                         init_seed=np.asarray(seed), normVF=np.ones((2, 2)),
                         normGr=np.ones(2), cosF=np.full((2, 2), cos),
                         bF=np.array([.5, .3]), bN=np.array([.1, .1]),
                         M_doc=np.array([.1, .2]), n_probes=np.array([4, 1]),
                         eval_sha1=np.array(["same-0", "same-1"]))
    out = minia_analysis(root)
    assert out["labels"]["mini-s"] == "H-ALIGN SUPPORTED"
    assert out["labels"]["mini-m"] == "H-ALIGN NOT SUPPORTED"
    assert "raw_mean" in out["per_preset"]["mini-s"]["v_gate"]
    assert "residual" in out["per_preset"]["mini-s"]["decomposition"]
    (root / "mini-m" / "e2e2" / "s0" / "align.npz").unlink()
    assert minia_analysis(root)["labels"]["mini-m"] == "NE"
