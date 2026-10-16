"""Fast CPU checks of the write-class gradient and streaming interventions."""

import numpy as np
import torch

from tttmem import exp_ttt as ttt
from tttmem import exp_wab as wab


def test_write_classes_and_streaming_equivalences():
    dev = torch.device("cpu")
    st = ttt._settings({"d_model": 16, "n_layers": 1, "n_heads": 2,
                        "backbone_mult": 2, "fast_mult": 2, "window": 4,
                        "chunk": 8, "echo": {"T": 32, "n_keys": 2, "n_eval": 2}}, "echo")
    task = ttt._task(st, "echo")
    tokens, meta = task.raw(np.random.default_rng(7), 2)
    idx, y, _ = task.tensors(tokens, meta, dev)
    model = ttt.build_model(task, st, dev, 3, "e2e2").eval()
    with torch.no_grad():
        h = model.backbone(idx).detach()
        etas = [e.detach() for e in model.etas(3.)]
    masks = wab.class_masks(meta, task.chunk, dev)
    assert torch.all(masks.sum(0) == 1)
    weights = [p.detach().requires_grad_(True) for p in model.per_sequence(len(idx))]
    z, _, grads = wab._class_grads(model, h[:, :8], y[:, :8], weights, masks)
    full = wab._sum_grads(grads, wab.ALL)
    with torch.enable_grad():
        direct = model.read(ttt.fast_block(h[:, :8], model.head.ln, weights))
        direct_grad = torch.autograd.grad(ttt.token_ce(direct, y[:, :8]).mean(1).sum(), weights)
    for actual, expected in zip(full, direct_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def evaluate(subset, family="fixed", alpha=1.):
        return wab.eval_condition(model, h, y, z, grads, etas, st.g_clip, task.chunk,
                                  family, subset, alpha)

    all_logits = evaluate(wab.ALL)
    torch.testing.assert_close(all_logits, ttt.eval_sequential(
        model, idx, y, etas, st.g_clip, task.chunk, ttt.GradStats()), rtol=0, atol=1e-5)
    none = evaluate(frozenset())
    zeros = [[torch.zeros_like(g) for g in row] for row in grads]
    no_first_update = wab.eval_condition(model, h, y, z, zeros, etas, st.g_clip,
                                          task.chunk, "fixed", wab.ALL)
    torch.testing.assert_close(none, no_first_update, rtol=0, atol=1e-5)

    frozen = evaluate(wab.ALL, "frozen")
    clipped, _, _ = ttt._clip_fast(full, st.g_clip)
    w1 = [w.detach() - eta * g for w, eta, g in zip(weights, etas, clipped)]
    expected = torch.cat([z] + [model.read(ttt.fast_block(
        h[:, c:c + 8], model.head.ln, w1)).detach() for c in range(8, task.T, 8)], 1)
    torch.testing.assert_close(frozen, expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(evaluate(wab.ALL, "scale", 1.), all_logits, rtol=0, atol=1e-5)
    torch.testing.assert_close(evaluate(wab.ALL, "scale", 0.),
                               evaluate(wab.ALL - {"kv"}), rtol=0, atol=1e-5)
    pi = wab._permutations(len(idx), task.n_keys, task.V,
                           torch.Generator().manual_seed(0), dev)
    swapped = wab._swapped_kv(model, h[:, :8], y[:, :8], weights, masks[2], pi)
    for family, tensors, replacement in (("reclip", ttt.FAST_KEYS, None),
                                          ("tensor", ("W2",), None),
                                          ("transplant", ttt.FAST_KEYS, None),
                                          ("dswap", ttt.FAST_KEYS, swapped)):
        logits = wab.eval_condition(model, h, y, z, grads, etas, st.g_clip, task.chunk,
                                    family, wab.ALL, tensors=tensors, swapped=replacement)
        assert logits.shape == (len(idx), task.T, task.V)
        assert torch.isfinite(logits).all()


def test_condition_matrix_and_verdict_rules():
    conditions = wab._conditions(False, False)
    assert {family: sum(row[0] == family for row in conditions) for family in
            ("fixed", "frozen", "tensor", "transplant", "dswap", "scale")} == {
                "fixed": 16, "frozen": 4, "tensor": 6, "transplant": 1,
                "dswap": 1, "scale": 9}
    assert len(wab._conditions(False, True)) == len(conditions) + 16
    fixed = {label: (.96 if "kv" in subset else .1)
             for family, label, subset, _, _ in conditions if family == "fixed"}
    models = [{"label": f"m{i}", "bound": True, "accuracy": {"fixed": fixed},
               "occ_minus_threshold": .3668} for i in range(3)]
    pre = wab._verdict(models, {"occ_minus": {"acc": .3168}}, "fixed")
    assert pre["verdict"] == "F"
    assert pre["necessary_counts"]["kv"] == 3
    assert pre["sufficient_counts"]["kv"] == 3
    assert wab._post_hoc_verdict(models, pre)["verdict"] == "F-strong"
