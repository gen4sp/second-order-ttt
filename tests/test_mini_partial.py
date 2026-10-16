"""CPU contracts for the MINI partial-SO arm."""
from dataclasses import replace
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch

from tttmem.p2.meta import InnerCfg
from tttmem.p2.model import build_model
from tttmem.p2.train import (TrainCfg, outer_microstep, partial_so_step,
                             somix_mask)


def _cfg(p, batch=16):
    return TrainCfg(arm="e2e12-p125", preset="tiny2", dtype="fp64", device="cpu",
                    global_batch=batch, micro_batch=2, so_micro_batch=1, so_fraction=p,
                    barriers=[2, 4], segment_len=16)


def _grads(model):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()]


def test_partial_mask_is_counter_deterministic_and_rtw_compatible():
    for seed, step, p in ((0, 0, .125), (3, 17, .125), (9, 4, .5)):
        got = somix_mask(seed, step, 16, p)
        rng = np.random.default_rng([seed, step, 0x534F4D49])
        want = np.zeros(16, dtype=bool)
        want[rng.permutation(16)[:round(p * 16)]] = True
        assert np.array_equal(got, want)
        assert int(got.sum()) == round(p * 16)
        assert np.array_equal(got, somix_mask(seed, step, 16, p))


def test_partial_p0_and_p1_match_fo_and_so_gradients():
    batch = torch.randint(3, 101, (4, 65))
    for p, kind in ((0., "fo"), (1., "so")):
        torch.manual_seed(20)
        mixed = build_model("tiny2", compute_dtype=torch.float64).double()
        mixed.zero_grad(set_to_none=True)
        partial_so_step(mixed, batch, _cfg(p, 4), InnerCfg(clip=1., step=0))
        got = _grads(mixed)
        torch.manual_seed(20)
        ref = build_model("tiny2", compute_dtype=torch.float64).double()
        ref.zero_grad(set_to_none=True)
        for i in range(0, 4, 2):
            outer_microstep(ref, batch[i:i + 2], _cfg(p, 4), kind, InnerCfg(clip=1., step=0))
        assert all(torch.allclose(a, b, atol=3e-11, rtol=3e-10)
                   for a, b in zip(got, _grads(ref)))


def test_partial_p125_is_per_row_so_fo_sum():
    batch = torch.randint(3, 101, (16, 65))
    cfg = _cfg(.125)
    torch.manual_seed(21)
    mixed = build_model("tiny2", compute_dtype=torch.float64).double()
    mixed.zero_grad(set_to_none=True)
    partial_so_step(mixed, batch, cfg, InnerCfg(clip=1., step=0))
    got = _grads(mixed)
    mask = somix_mask(0, 0, 16, .125)
    torch.manual_seed(21)
    ref = build_model("tiny2", compute_dtype=torch.float64).double()
    ref.zero_grad(set_to_none=True)
    for i in range(16):
        outer_microstep(ref, batch[i:i + 1], cfg, "so" if mask[i] else "fo", InnerCfg(clip=1., step=0))
    assert all(torch.allclose(a, b, atol=2e-11, rtol=2e-10) for a, b in zip(got, _grads(ref)))


def test_matrix_resolves_partial_configs_and_estimates():
    from tttmem.p2.matrix import _base_config, _cells, _estimate_hours, _load, _lookup
    root = Path(__file__).parents[1]
    cfg = {"preset": "mini-m", "cells": [{"arm": "e2e12-p125", "init_seed": s} for s in range(3)]}
    cells = _cells(cfg)
    assert len(cells) == 3 and {c["init_seed"] for c in cells} == {0, 1, 2}
    base = _base_config(cfg, cells[0], root)
    assert base["arm"] == "e2e12-p125" and base["so_fraction"] == .125
    assert _estimate_hours(cells[0], cfg, []) == 1.10
    assert _lookup(cfg, "memory_peaks_gb", cells[0]) == 11.5
    for name in ("e2e12-p125.toml", "e2e12-p125-m.toml"):
        assert (root / "configs" / "mini" / name).exists()


def test_partial_analysis_computes_addendum_v14_read():
    from tttmem.p2.mini_analysis import analyse
    rows = []
    for seed in range(3):
        for arm, value in (("e2e1-ce", .10), ("e2e2", .30), ("e2e12-p125", .20)):
            rows.append({"preset": "mini-m", "arm": arm, "seed": str(seed), "step": 1200,
                         "corpus": "dclm", "M": value, "G": .02,
                         "so_row_share": .125, "ms_per_step": 12.})
    read = analyse(rows)["partial_so"]["mini-m"]
    assert read["phi_hat"] == pytest.approx(.5)
    assert read["lb_E_minus_quarter_delta"] > 0
    assert read["realised_so_row_share"] == .125 and read["ms_per_step"] == 12.
    assert read["row"] == "transfers"


def test_partial_resume_preserves_masks_and_loss_trajectory(tmp_path):
    from test_p2 import _smoke_data, _write_split
    data, T = _smoke_data(tmp_path)
    rows = np.random.default_rng(44).integers(3, 7, size=(32, T + 1), dtype=np.uint32)
    rows[:, 0] = 1
    _write_split(data, "A", rows)
    base = TrainCfg(arm="e2e12-p125", preset="tiny", dtype="fp32", device="cpu",
                    seq_len=T, global_batch=16, micro_batch=2, so_micro_batch=1,
                    so_fraction=.125, steps=2, warmup=2, lr_peak=1e-3,
                    data_dir=str(data), out_dir=str(tmp_path / "straight"),
                    resume_dir=str(tmp_path / "straight-state"), n_eval=4,
                    min_probes=1, assay_gate=False, ho_docs=3, ho_trajectory_docs=2,
                    eval_batch=2, pilot_steps=[], pilot_docs=2, eval_steps=[2],
                    full_eval_steps=[2], trajectory_docs=2, sizing_steps=[],
                    crossover_rules=[], log_every=1, resume_every=1,
                    save_checkpoints=False, eval_at_init=False)
    from tttmem.p2.train import train
    train(base)
    split = replace(base, out_dir=str(tmp_path / "split"),
                    resume_dir=str(tmp_path / "split-state"), stop_after=1)
    train(split)
    train(replace(split, stop_after=None, resume=True))
    assert (tmp_path / "straight" / "steps.jsonl").read_bytes() == \
           (tmp_path / "split" / "steps.jsonl").read_bytes()
    rows = [json.loads(x.removeprefix("P2_STEP ")) for x in
            (tmp_path / "split" / "steps.jsonl").read_text().splitlines()]
    assert [r["so_rows"] for r in rows] == [2, 2]
