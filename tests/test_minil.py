import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem.p2.mini import _IDENTITY_FIELDS, _checkpoint_skip, _write_tiny_data, mini_late
from tttmem.p2.mini_analysis import minil_analysis
from tttmem.p2.train import TrainCfg, training_order, train


def _tiny(data, out, *, steps=2, eval_steps=None, full_eval_steps=None,
          eval_at_init=False, save_checkpoints=True, micro_batch=1,
          process_name="TTM-mnl-1N-test", **kw):
    eval_steps = [steps] if eval_steps is None else eval_steps
    full_eval_steps = [steps] if full_eval_steps is None else full_eval_steps
    return TrainCfg(arm="e2e1-ce", preset="tiny", dtype="fp32", device="cpu",
                    seq_len=64, global_batch=2, micro_batch=micro_batch, steps=steps,
                    warmup=1, lr_peak=1e-2, lr_floor=1e-5, inner_warmup=1,
                    data_dir=str(data), out_dir=str(out), n_eval=4, min_probes=1,
                    eval_steps=eval_steps, full_eval_steps=full_eval_steps, sizing_steps=[],
                    trajectory_docs=4, ho_docs=2, ho_trajectory_docs=2,
                    eval_batch=2, pilot_docs=2, pilot_steps=[], assay_gate=False,
                    crossover_rules=[], log_every=1, resume_every=2,
                    save_checkpoints=save_checkpoints, barriers=[2], segment_len=16,
                    fused_attn=False, eval_at_init=eval_at_init,
                    eval_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
                    trajectory_modes=["adapt", "noadapt", "fP_focal", "fP_nonfocal"],
                    full_extra_modes=[], process_name=process_name, **kw)


def test_default_stream_is_bit_identical_and_continuation_is_disjoint():
    original = training_order(32, steps=4, batch=2, data_seed=9)
    continuation = training_order(32, steps=2, batch=2, data_seed=9, data_start_step=4)
    expected = np.random.default_rng(9).permutation(32)
    np.testing.assert_array_equal(original, expected[:8])
    np.testing.assert_array_equal(continuation, expected[8:12])
    assert not set(original.tolist()) & set(continuation.tolist())


@pytest.mark.parametrize("micro_batch", [1, 2])
def test_train_continuation_consumes_original_stream_once(tmp_path, monkeypatch, micro_batch):
    import tttmem.p2.train as train_mod

    data = tmp_path / "data"; _write_tiny_data(data, rows=16, seed=11)
    seen = []

    class SpyPrefetcher:
        def __init__(self, arr, order, micro, device, *, width=None, **kwargs):
            self.arr, self.order, self.micro = arr, np.asarray(order), micro
            seen.extend(self.order.tolist())
            self.width = width or arr.shape[1]
            self.last_injected = 0  # the trainer's prefetcher interface (echo injection off)

        def __iter__(self):
            for i in range(0, len(self.order), self.micro):
                rows = np.sort(self.order[i:i + self.micro])
                yield torch.from_numpy(self.arr[rows, :self.width].astype(np.int64))

        def close(self):
            return None

    monkeypatch.setattr(train_mod, "Prefetcher", SpyPrefetcher)
    train(_tiny(data, tmp_path / f"run-{micro_batch}", steps=2,
                data_start_step=2, micro_batch=micro_batch))
    expected = training_order(16, steps=2, batch=2, data_seed=0, data_start_step=2)
    earlier = training_order(16, steps=2, batch=2, data_seed=0)
    np.testing.assert_array_equal(np.asarray(seen), expected)
    assert not set(seen) & set(earlier)


def test_default_run_is_bit_identical_with_explicit_zero(tmp_path):
    data = tmp_path / "data"; _write_tiny_data(data, rows=16, seed=10)
    a, b = tmp_path / "a", tmp_path / "b"
    train(_tiny(data, a))
    train(_tiny(data, b, data_start_step=0))
    assert (a / "steps.jsonl").read_bytes() == (b / "steps.jsonl").read_bytes()
    with np.load(a / "eval_step0002.npz") as x, np.load(b / "eval_step0002.npz") as y:
        assert x.files == y.files
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key])


def test_inner_warmup_one_starts_at_one(tmp_path):
    data = tmp_path / "data"; _write_tiny_data(data, rows=16, seed=4)
    out = tmp_path / "run"
    train(_tiny(data, out))
    rec = json.loads(next(x.split(" ", 1)[1] for x in (out / "steps.jsonl").read_text().splitlines()
                          if x.startswith("P2_STEP ")))
    assert rec["inner_lr_mult"] == 1.0


def test_init_eval_reproduces_checkpoint_eval(tmp_path):
    data = tmp_path / "data"; _write_tiny_data(data, rows=16, seed=5)
    source = tmp_path / "source"; train(_tiny(data, source))
    with np.load(source / "eval_step0002.npz") as z:
        expected = float(np.nanmean(z["doc_fP_focal"] - z["doc_fP_nonfocal"]))
    cont = tmp_path / "cont"
    train(_tiny(data, cont, init=str(source / "model_step0002.pt"), eval_at_init=True,
                steps=1, eval_steps=[1], full_eval_steps=[1], save_checkpoints=False))
    with np.load(cont / "eval_step0000.npz") as z:
        got = float(np.nanmean(z["doc_fP_focal"] - z["doc_fP_nonfocal"]))
    assert got == expected


def test_minil_identity_mismatch_prints_skip(tmp_path, capsys):
    data = tmp_path / "data"; _write_tiny_data(data, rows=16, seed=6)
    source = tmp_path / "source"; train(_tiny(data, source, init_seed=0, data_seed=0))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        mini_late({"process_nn": 81, "process_name": "TTM-mnl-1N-test",
                   "data_dir": str(data), "device": "cpu", "out_dir": str(out),
                   "items": [{"preset": "tiny", "init_arm": "e2e2", "arm": "e2e1-ce",
                              "init_seed": 0, "data_seed": 0,
                              "ckpt": str(source / "model_step0002.pt"),
                              "out": str(out / "L-FO" / "s0")} ]})
    assert "MINIL_SKIP" in capsys.readouterr().out


def test_minil_checkpoint_without_config_uses_cell_config_and_payload_identity(tmp_path):
    from dataclasses import asdict

    cfg = TrainCfg(preset="mini-m", arm="e2e2", steps=1200, barriers=[2, 4, 6],
                   segment_len=2048, init_seed=2, data_seed=2)
    cell_config = tmp_path / "cell.json"
    cell_config.write_text(json.dumps({key: asdict(cfg)[key] for key in _IDENTITY_FIELDS}))
    path = tmp_path / "model_step1200.pt"
    payload = {"model": {"weight": torch.zeros(1)}, "step": 1200,
               "arm": "e2e2", "init_label": ""}
    torch.save(payload, path)
    item = {"step": 1200, "cell_config": str(cell_config)}
    assert _checkpoint_skip(path, item, cfg, payload=payload, mini_e_only=False) is None
    wrong_arm = {**payload, "arm": "e2e1-ce"}
    assert _checkpoint_skip(path, item, cfg, payload=wrong_arm, mini_e_only=False) == \
        "checkpoint_config_mismatch:arm=e2e1-ce"


def _write_minil_fixture(root, values, *, complete_seeds=(0, 1, 2),
                         docs=1024, baseline=True):
    for label, vals in values.items():
        for seed, m in enumerate(vals):
            d = root / "mini-m" / label / f"s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            if seed in complete_seeds:
                (d / "MINIL_CELL_DONE.json").write_text(
                    json.dumps({"status": "done", "n_docs": docs}) + "\n")
            for step in (150, 300):
                np.savez(d / f"eval_step{step:04d}.npz",
                         doc_fP_focal=np.full(docs, m + .1),
                         doc_fP_nonfocal=np.full(docs, .1),
                         doc_adapt=np.full(docs, .2),
                         doc_noadapt=np.full(docs, .25))
    if baseline:
        core = root / "core" / "runs" / "mini-m"
        for seed in range(3):
            for arm, m in (("e2e2", .30), ("e2e1-ce", .20)):
                d = core / arm / f"s{seed}"; d.mkdir(parents=True, exist_ok=True)
                np.savez(d / "eval_step1200.npz",
                         doc_fP_focal=np.full(1024, m + .1),
                         doc_fP_nonfocal=np.full(1024, .1))


@pytest.mark.parametrize(
    "expected,so,fo,ref",
    [("LATE-REPAIRS", .40, .20, .30),
     ("LATE-PARTIAL", .40, .20, .70),
     ("LATE-FAILS", .10, .20, .20),
     ("AMBIGUOUS", .20, .20, .20)],
)
def test_minil_analysis_registered_labels(tmp_path, expected, so, fo, ref):
    root = tmp_path / "minil"
    _write_minil_fixture(root, {"L-SO": [so] * 3, "L-FO": [fo] * 3,
                                "L-REF": [ref] * 3})
    result = minil_analysis(root)
    assert result["label"] == expected
    assert result["D"]["n"] == 3 and result["R"]["n"] == 3
    assert result["M_at_150_300"]["150"]["L-SO"]


def test_minil_analysis_incomplete_is_ne_and_reports_unreadable(tmp_path):
    root = tmp_path / "minil"
    _write_minil_fixture(root, {"L-SO": [.4] * 3, "L-FO": [.2] * 3,
                                "L-REF": [.3] * 3}, complete_seeds=(0, 1),
                       baseline=False)
    bad = root / "mini-m" / "L-SO" / "s0" / "eval_step0300.npz"
    bad.write_bytes(b"not an npz")
    result = minil_analysis(root, mini_root=tmp_path / "missing-baseline")
    assert result["label"] == "NE"
    assert any("unreadable_eval" in x["reason"] for x in result["issues"])
    assert result["complete_counts"]["L-SO"] < 3
    assert result["psi_prime"]["status"] == "NE"
    assert result["rewarm_check"]["status"] == "NE"
