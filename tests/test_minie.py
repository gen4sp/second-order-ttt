import json
from pathlib import Path

import numpy as np
import pytest

from tttmem.p2.mini import _write_tiny_data, mini_ckpt_eval
from tttmem.p2.mini_analysis import minie_analysis
from tttmem.p2.train import TrainCfg, train


def _tiny_cfg(data, out, *, arm="e2e1-ce", crossover=(), init_seed=0, data_seed=0):
    return TrainCfg(
        arm=arm, preset="tiny", dtype="fp32", device="cpu", init_seed=init_seed,
        seq_len=64, global_batch=2, micro_batch=1, steps=2,
        warmup=1, lr_peak=1e-2, inner_lr=1.0, data_dir=str(data), data_seed=data_seed,
        out_dir=str(out), n_eval=4, min_probes=1,
        eval_steps=[2], full_eval_steps=[2], sizing_steps=[], trajectory_docs=4,
        ho_docs=2, ho_trajectory_docs=2, pilot_docs=2, eval_batch=2,
        pilot_steps=[], assay_gate=False, crossover_rules=list(crossover),
        log_every=1, resume_every=2, save_checkpoints=True,
        barriers=[2], segment_len=16, fused_attn=False, eval_at_init=False,
        eval_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
        trajectory_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
        full_extra_modes=[], process_name="TTM-min-62-test",
    )


def test_eval_setup_refactor_is_deterministic_bit_identical(tmp_path):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=7)
    a, b = tmp_path / "a", tmp_path / "b"
    train(_tiny_cfg(data, a))
    train(_tiny_cfg(data, b))
    with np.load(a / "eval_step0002.npz") as x, np.load(b / "eval_step0002.npz") as y:
        assert x.files == y.files
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])


def test_minie_ce_reproduces_logged_m_and_crossovers(tmp_path, capsys):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=8)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    out = tmp_path / "minie"
    result = mini_ckpt_eval({
        "process_nn": 62, "process_name": "TTM-min-62-minie", "data_dir": str(data),
        "device": "cpu", "out_dir": str(out),
        "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0,
                    "data_seed": 0, "step": 2,
                    "ckpt": str(cell / "model_step0002.pt"),
                    "rules": ["ce"], "out": str(out / "tiny" / "e2e1-ce" / "s0")}],
    })
    assert result["items"][0]["status"] == "done"
    lines = capsys.readouterr().out.splitlines()
    assert any(x.startswith("MINIE_V0 ") and '"pass": true' in x for x in lines)
    with np.load(cell / "eval_step0002.npz") as logged:
        with np.load(out / "tiny" / "e2e1-ce" / "s0" / "eval_ckpt_ce.npz") as got:
            np.testing.assert_array_equal(got["x_ce_M"], logged["doc_fP_focal"] - logged["doc_fP_nonfocal"])
    # RP/NC crossover parity is defined by the in-training read of an arm under
    # the common CE rule (run_eval stores x_ce_M for these arms).
    for arm in ("e2e1-rp", "e2e1-nc"):
        arm_cell = tmp_path / arm
        train(_tiny_cfg(data, arm_cell, arm=arm, crossover=("ce",)))
        arm_out = out / "tiny" / arm / "s0"
        mini_ckpt_eval({
            "process_nn": 62, "process_name": "TTM-min-62-minie", "data_dir": str(data),
            "device": "cpu", "items": [{"preset": "tiny", "arm": arm, "init_seed": 0,
                        "data_seed": 0, "step": 2,
                        "ckpt": str(arm_cell / "model_step0002.pt"),
                        "rules": ["ce"], "out": str(arm_out)}],
        })
        with np.load(arm_cell / "eval_step0002.npz") as logged:
            with np.load(arm_out / "eval_ckpt_ce.npz") as got:
                np.testing.assert_array_equal(got["x_ce_M"], logged["x_ce_M"])


def test_minie_missing_checkpoint_is_skip(tmp_path, capsys):
    with pytest.raises(RuntimeError, match="nothing evaluated"):
        mini_ckpt_eval({
            "process_nn": 63, "process_name": "TTM-min-63-minie", "data_dir": str(tmp_path),
            "device": "cpu", "out_dir": str(tmp_path / "out"),
            "items": [{"preset": "tiny", "arm": "e2e1-ce",
                       "ckpt": str(tmp_path / "missing.pt"), "rules": ["ce"]}],
        })
    assert "MINIE_SKIP" in capsys.readouterr().out
    summary = json.loads((tmp_path / "out" / "minie_summary.json").read_text())
    assert summary["items"][0]["status"] == "skip"


def test_minie_error_is_fail_closed_but_other_items_run(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=12)
    cell = tmp_path / "cell"
    train(_tiny_cfg(data, cell))
    from tttmem.p2 import evaluate
    original = evaluate.eval_docs
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA out of memory (injected)")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate, "eval_docs", flaky)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="item error"):
        mini_ckpt_eval({
            "process_nn": 64, "process_name": "TTM-min-64-minie", "data_dir": str(data),
            "device": "cpu", "out_dir": str(out),
            "items": [
                {"preset": "tiny", "arm": "e2e1-ce", "step": 2,
                 "ckpt": str(cell / "model_step0002.pt"), "rules": ["ce"]},
                {"preset": "tiny", "arm": "e2e1-ce", "step": 2,
                 "ckpt": str(cell / "model_step0002.pt"), "rules": ["ce"],
                 "out": str(out / "second")},
            ],
        })
    summary = json.loads((out / "minie_summary.json").read_text())
    assert summary["errors"] == 1 and summary["done"] == 1
    assert summary["items"][0]["status"] == "error"
    assert summary["items"][1]["status"] == "done"


def test_minie_identity_rejects_seed_mixup_and_eval_sha1(tmp_path):
    data = tmp_path / "data"
    _write_tiny_data(data, rows=16, seed=13)
    source = tmp_path / "source"
    train(_tiny_cfg(data, source, init_seed=1, data_seed=1))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="nothing evaluated"):
        mini_ckpt_eval({
            "process_nn": 65, "process_name": "TTM-min-65-minie", "data_dir": str(data),
            "device": "cpu", "out_dir": str(out),
            "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 0,
                        "data_seed": 0, "step": 2,
                        "ckpt": str(source / "model_step0002.pt"), "rules": ["ce"]}],
        })
    summary = json.loads((out / "minie_summary.json").read_text())
    assert "init_seed" in summary["items"][0]["reason"]

    meta = source / "eval_meta.npz"
    with np.load(meta) as z:
        values = {k: z[k] for k in z.files}
    values["eval_sha1"] = values["eval_sha1"].copy()
    values["eval_sha1"][0] = "wrong"
    np.savez_compressed(meta, **values)
    with pytest.raises(RuntimeError, match="nothing evaluated"):
        mini_ckpt_eval({
            "process_nn": 66, "process_name": "TTM-min-66-minie", "data_dir": str(data),
            "device": "cpu", "out_dir": str(tmp_path / "out-sha"),
            "items": [{"preset": "tiny", "arm": "e2e1-ce", "init_seed": 1,
                        "data_seed": 1, "step": 2,
                        "ckpt": str(source / "model_step0002.pt"), "rules": ["ce"]}],
        })
    summary = json.loads((tmp_path / "out-sha" / "minie_summary.json").read_text())
    assert "eval_meta_mismatch:eval_sha1" in summary["items"][0]["reason"]


def test_minie_starvation_ratio_cot_maps_e2c_sources(tmp_path):
    root = tmp_path / "matrix"
    meta = dict(probe_offset=np.array([0, 1, 2]), probe_pos=np.array([20, 20]),
                probe_stratum=np.array([2, 2]), probe_dist=np.array([4, 4]),
                focal_offset=np.array([0, 1, 2]), focal_pos=np.array([16, 16]))
    for arm, cot in (("e2e2", [2., 3.]), ("e2e1-ce", [1., 1.])):
        path = root / "core" / "runs" / "mini-s" / arm / "s0"
        path.mkdir(parents=True)
        np.savez(path / "eval_meta.npz", **meta,
                 eval_rows=np.array([0, 1]), eval_sha1=np.array(["a", "b"]),
                 eval_domain=np.array(["a", "b"]), ho_sha1=np.array(["h"]))
        np.savez(path / "eval_step1200.npz",
                 doc_fP_focal=np.array([.2, .2]), doc_fP_nonfocal=np.array([.1, .1]),
                 doc_adapt=np.array([.2, .2]), doc_noadapt=np.array([.3, .3]),
                 doc_heldout=np.array([.3, .3]), doc_heldout_noadapt=np.array([.3, .3]),
                 src_cot_adapt=np.array(cot))
    result = minie_analysis(root)["starvation_rehearsal"]["mini-s"]
    assert result["status"] == "NE"
    assert np.isclose(result["ratio_cot"], .4)


def test_minie_analysis_contrasts_r2_and_v0_ne(tmp_path):
    root = tmp_path / "matrix"

    def write_core(arm, seed, m, xce=None):
        p = root / "core" / "runs" / "mini-s" / arm / f"s{seed}"
        p.mkdir(parents=True, exist_ok=True)
        d = dict(doc_fP_focal=np.array([m + .1]), doc_fP_nonfocal=np.array([.1]),
                 doc_adapt=np.array([.2]), doc_noadapt=np.array([.25]),
                 doc_heldout=np.array([.3]), doc_heldout_noadapt=np.array([.3]),
                 probe_stratum=np.array([0]))
        if xce is not None:
            d["x_ce_M"] = np.array([xce])
        np.savez(p / "eval_step1200.npz", **d)

    for seed in range(3):
        write_core("e2e2", seed, .05)
        write_core("e2e1-ce", seed, .02)
    write_core("e2e1-rp", 0, .04, .035)
    write_core("e2e1-nc", 0, .025, .022)
    for seed in range(3):
        p = root / "minie" / "mini-s" / "e2e1-ce" / f"s{seed}"
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "eval_ckpt_ce.npz", x_ce_M=np.array([.02]), v0_pass=np.array(True), n_docs=np.array(1))
        np.savez(p / "eval_ckpt_rp.npz", x_rp_M=np.array([.03]), n_docs=np.array(1))
    np.savez(root / "minie" / "mini-s" / "e2e1-ce" / "s0" / "eval_ckpt_nc.npz",
             x_nc_M=np.array([.025]), n_docs=np.array(1))
    out = minie_analysis(root)
    q = out["per_preset"]["mini-s"]
    assert np.isclose(q["C_own"], .02)
    assert np.isclose(q["C_T"], .015)
    assert q["C_E"]["status"] == "OK"
    assert q["C_NC_E"]["status"] == "POINT"
    assert np.isclose(q["tree_inputs_minus_half_delta"]["C_NC_minus_half_delta"],
                      q["C_NC_own"] - .5 * q["M_SO_minus_FO"])
    assert "row" in out["r2_item4"]

    with np.load(root / "minie" / "mini-s" / "e2e1-ce" / "s1" / "eval_ckpt_ce.npz") as z:
        np.savez(root / "minie" / "mini-s" / "e2e1-ce" / "s1" / "eval_ckpt_ce.npz",
                 x_ce_M=z["x_ce_M"], v0_pass=np.array(False), n_docs=np.array(1))
    ne = minie_analysis(root)["per_preset"]["mini-s"]["C_E"]
    assert ne["status"] == "NE" and "V0" in ne["reason"]
