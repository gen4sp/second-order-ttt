from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import pytest
import tomllib

from tttmem.p2.echo_text import build_eval_documents, make_injector
from tttmem.p2.model import build_model
from tttmem.p2.train import TrainCfg
from tttmem.p2.v1 import _mask_for_source, _read_batch
from tttmem.p2.v1_analysis import analyze

REPO = Path(__file__).resolve().parents[1]
EXCLUDED = Path(__file__).resolve().parents[1] / "data/p2data" / "excluded_vocab.npy"


def _tiny_cfg(tmp_path):
    return TrainCfg(preset="tiny", arm="e2e2", dtype="fp32", device="cpu", seq_len=64,
                    data_dir=str(tmp_path), out_dir=str(tmp_path / "out"), barriers=[2, 4, 6],
                    segment_len=8, eval_batch=2, save_checkpoints=False)


def _records(n):
    out = []
    for i in range(n):
        out.append({
            "mention_positions": [[1, 2], [2, 3]],
            "source_nll_positions": [[5, 6, 7], [5, 6, 7]],
            "source_value_positions": [[6, 7, 8], [6, 7, 8]],
            "query_value_positions": [[24, 25, 26], [32, 33, 34]],
            "no_barrier_value_positions": [[12, 13, 14], [16, 17, 18]],
            "no_binding_value_positions": [[40, 41, 42], [48, 49, 50]],
            "query_entity_value_positions": [[56, 57, 58], [60, 61, 62]],
            "available": [True, False],
            "spans": [],
        })
    return out


def test_v1_math_attention_value_mask_smoke(tmp_path):
    torch.set_num_threads(2)
    cfg = _tiny_cfg(tmp_path)
    model = build_model("tiny", compute_dtype=torch.float32, device="cpu", seed=0)
    rng = np.random.default_rng(2)
    rows = rng.integers(3, 97, size=(2, 65), dtype=np.uint32)
    rows[:, 0] = 1
    rec = _records(2)
    result = _read_batch(model, rows, rec, cfg, masked=True)
    assert result["nll"].shape == (2, 64)
    assert result["greedy"].shape == (2, 64)
    assert np.isfinite(result["nll"].numpy()).all()


def test_v1_analysis_gate_and_ci(tmp_path):
    root = tmp_path / "v1"
    for seed, delta in enumerate((.24, .25, .26)):
        d = root / f"e2e2/s{seed}"
        d.mkdir(parents=True)
        src = np.full((64, 8), 1.0, np.float32)
        src[:, 4:] += delta
        masked = src.copy(); masked[:, :4] = src[:, 4:]
        np.savez_compressed(d / "v1_item.npz", source_nll=src,
                            source_nll_masked=masked, source_residual=np.ones_like(src),
                            query_nll=np.ones_like(src), no_barrier_nll=np.ones_like(src),
                            no_binding_nll=np.ones_like(src), query_entity_nll=np.ones_like(src),
                            query_exact=np.ones_like(src, bool),
                            no_barrier_exact=np.ones_like(src, bool),
                            no_binding_exact=np.zeros_like(src, bool),
                            query_entity_exact=np.ones_like(src, bool),
                            available=np.array([[True] * 4 + [False] * 4] * 64),
                            arm=np.asarray("e2e2"), seed=np.asarray(seed))
    result = analyze(root)
    assert result["verdict"] == "PASS"
    assert result["arms"]["e2e2"]["source_gate"]
    assert (root / "v1_analysis.json").exists()


def test_v1_control_spans_do_not_write(tmp_path):
    """A control span in chunk 0 leaves chunk-0 NLL unchanged and changes later chunks
    (its tokens are removed from the inner loss)."""
    torch.set_num_threads(2)
    cfg = _tiny_cfg(tmp_path)
    model = build_model("tiny", compute_dtype=torch.float32, device="cpu", seed=0)
    rows = np.random.default_rng(3).integers(3, 97, size=(2, 65), dtype=np.uint32)
    rows[:, 0] = 1
    plain = _read_batch(model, rows, _records(2), cfg, masked=False)["nll"].numpy()
    rec = _records(2)
    for r in rec:
        r["spans"] = [[2, 6, "no_binding", 0]]
    ctrl = _read_batch(model, rows, rec, cfg, masked=False)["nll"].numpy()
    assert np.array_equal(plain[:, :8], ctrl[:, :8])
    assert not np.allclose(plain[:, 8:16], ctrl[:, 8:16])
    rec[0]["spans"] = rec[1]["spans"] = [[2, 6, "binding", 0]]
    assert np.array_equal(plain, _read_batch(model, rows, rec, cfg, masked=False)["nll"].numpy())


def test_v1_value_mask_on_generator_records():
    rows = np.random.default_rng(0).integers(1000, 30000, size=(2, 8193)).astype(np.uint32)
    rows[:, 0] = 128000
    _, meta = build_eval_documents(rows, split="cal", p=.5, seed=1,
                                   excluded_vocab=np.zeros(128256, dtype=bool))
    km = _mask_for_source(meta["rows"], 8192, 1024, "cpu")
    assert km.shape == (2, 1024, 1024)
    for i, rec in enumerate(meta["rows"]):
        hidden = (~km[i]).nonzero().numpy()
        src = set(np.asarray(rec["source_nll_positions"]).ravel().tolist())
        men = set(np.asarray(rec["mention_positions"]).ravel().tolist())
        assert set(hidden[:, 0].tolist()) == src and set(hidden[:, 1].tolist()) == men
        assert len(hidden) == len(src) * len(men)
    bad = [dict(meta["rows"][0], source_nll_positions=[[1500, 1501, 1502]] * 8)]
    with pytest.raises(ValueError, match="chunk 0"):
        _mask_for_source(bad, 8192, 1024, "cpu")


def test_v1_config_registration():
    reg = (REPO / "configs/soq/v1.toml").read_bytes()
    root = tomllib.loads(reg.decode())
    if tomllib.loads((REPO / "configs/soq/v1.toml").read_text())["p2"].get("mode") == "soq_v1":
        assert (REPO / "configs/soq/v1.toml").read_bytes() == reg      # the V1 root config
    p = root["p2"]
    assert p["mode"] == "soq_v1" and p["n_docs"] == 256 and p["out_dir"] == "outputs/soq/v1"
    assert p["max_hours"] == 1.5 and p["device"] == "cuda" and len(p["items"]) == 6
    if EXCLUDED.exists():
        inj = make_injector(split="cal", f=1.0, p=p["p"], seed=p["echo_seed"], T=p["T"], S=p["S"],
                            excluded_vocab=np.load(EXCLUDED))
        assert inj.manifest()["manifest_sha256"] == p["echo_manifest_sha256"]
