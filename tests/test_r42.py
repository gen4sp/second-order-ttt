"""R4-2 mask identities, pairing, probe and fixed-ablation behavior on CPU."""

import math
import tomllib
from pathlib import Path

import numpy as np
import torch

from tttmem import exp_ttt as t


def test_n1_n2_n3_n4():
    checks = t.r42_sanity()
    assert all(v["passed"] for v in checks.values()), checks
    assert checks["N1"]["gradient_rel"] <= 1e-6
    assert checks["N2"]["kv_outer_zero"]
    assert len(set(checks["N3"]["hashes"])) == 5
    assert checks["N4"]["rnd_counts"] == [4, 4]


def test_first_three_unmasked_losses_match_prechange_recurrence():
    """Independent copy of the original e2e1emb recurrence, including optimizer updates."""
    dev = torch.device("cpu")
    st = t._settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                          fast_mult=2, window=16, chunk=16, n_keys=4, T=128))
    task = t._task(st, "echo")
    old = t.build_model(task, st, dev, 0, "e2e1emb")
    new = t.build_model(task, st, dev, 0, "e2e1emb")
    opt_old, opt_new = t.build_optim(old, st), t.build_optim(new, st)
    rng = np.random.default_rng([0, t.TRAIN_SEED])
    losses = []
    for _ in range(3):
        idx, y, _ = task.tensors(*task.raw(rng, 2), dev)
        h = old.backbone(idx)
        x = old.fast_input(h, idx)
        weights = old.per_sequence(len(idx))
        etas = old.etas(10.)
        total = 0.
        for c in range(task.T // task.chunk):
            sl = slice(c * task.chunk, (c+1) * task.chunk)
            loss_c = t._ce_tokens(old, t.fast_block(h[:, sl], old.head.ln, weights, x[:, sl]),
                                  y[:, sl]).mean(1)
            g = torch.autograd.grad(loss_c.sum(), weights, retain_graph=True)
            g, _, _ = t._clip_fast(list(g), 10.)
            total = total + loss_c.mean()
            weights = [w - eta * v for w, eta, v in zip(weights, etas, g)]
        original = total / (task.T // task.chunk)
        current = t.arm_loss(new, "e2e1emb", idx, y, 10., 10., task.chunk, t.GradStats())
        losses.append((float(original.detach()), float(current.detach())))
        assert torch.equal(original, current)
        opt_old.zero_grad(set_to_none=True)
        opt_new.zero_grad(set_to_none=True)
        original.backward()
        current.backward()
        for a, b in zip(old.parameters(), new.parameters()):
            assert torch.equal(a.grad, b.grad)
        opt_old.step()
        opt_new.step()
    assert all(math.isfinite(a) and a == b for a, b in losses)


def test_probe_ablation_and_config():
    with (Path(__file__).resolve().parents[1] / "configs/toy/r4-2.toml").open("rb") as f:
        cfg = tomllib.load(f)["ttt"]
    assert len(cfg["cells"]) == 50 and cfg["workers"] == 3
    st = t._settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                          fast_mult=2, window=16, chunk=16, n_keys=4, T=128,
                          n_eval=4, eval_batch=2))
    task = t._task(st, "echo")
    model = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb")
    tok, meta = t.fixed_set(task, st, "eval")
    data = t.eval_batches(task, tok, meta, torch.device("cpu"), 2)
    ablation = t.r42_ablation(model, task, data, 10., 10., arm="e2e1emb")
    reference = t.evaluate(model, task, "e2e1emb", data, meta, 10., st)["eval"]["query_acc"]
    assert abs(ablation["all"] - reference) < .01
    assert set(ablation) == {"all", "all_minus_kv", "kv_only"}
    p = t.r4_probe(model, task, data, 10., 10., arm="e2e1emb")
    assert all(math.isfinite(p[k]) for k in ("ce_kv", "ce_vk", "ce_vW", "p_kv",
               "e_kv_abs", "p_wv", "e_wv_abs", "kv_write_eff", "Wv_write_eff"))
