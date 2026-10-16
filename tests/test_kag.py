"""Small-model identities and key-indexing checks for KAG."""

import numpy as np
import torch

from tttmem import exp_kag as kag
from tttmem import exp_ttt as ttt
from tttmem import exp_wab as wab


def test_linear_and_kv_gradient_are_exact():
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
        o = ttt.fast_block(h[:, :task.chunk], model.head.ln, model.slow())
    delta = kag._delta(model, o, y[:, :task.chunk])
    masks = wab.class_masks(meta, task.chunk, dev)
    weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
    z, _, grads = wab._class_grads(model, h[:, :task.chunk], y[:, :task.chunk], weights, masks)
    a = kag._gather(kag._code(model, h), meta["write_pos"])
    d = kag._gather(delta, meta["write_pos"])
    outer = torch.einsum("bkd,bkm->bdm", d, a) / task.chunk
    torch.testing.assert_close(outer, grads[2][2], rtol=1e-5, atol=1e-7)
    full = wab._sum_grads(grads, wab.ALL)
    _, norm, clipped = ttt._clip_fast(full, st.g_clip)
    scale = torch.where(clipped, st.g_clip / norm, torch.ones_like(norm))
    x = kag._linear_terms(model, h, meta, delta, etas[2], scale, task.chunk)
    linear = model.read(x["o0"] + x["factor"] * x["memory"])
    direct = wab.eval_condition(model, h, y, z, grads, etas, st.g_clip,
                                task.chunk, "frozen", kag.KV, tensors=("W2",))
    torch.testing.assert_close(linear, kag._gather(direct, meta["query_pos"]), rtol=0,
                               atol=1e-5)


def test_address_tracks_non_involutive_permutation():
    write = torch.eye(4).unsqueeze(0)
    sigma = torch.tensor([1, 2, 0, 3])  # sigma(sigma(0)) != 0
    query = write[:, sigma]
    dot, cos, effective, _, _ = kag._address(write, query, torch.ones(1, 4, 4))
    for scores in (dot, cos, effective):
        torch.testing.assert_close(scores.argmax(-1)[0], sigma)


def test_content_inf_uses_only_matched_write():
    delta = torch.eye(3).unsqueeze(0)
    o0 = torch.zeros_like(delta)
    scale = torch.ones(1, 1, 1)
    similarity = torch.ones(1, 3)
    oracle = kag._oracle_content(o0, scale, similarity, delta)
    changed = delta.clone()
    changed[:, 1] *= 100
    modified = kag._oracle_content(o0, scale, similarity, changed)
    torch.testing.assert_close(oracle[:, 0], modified[:, 0])
    torch.testing.assert_close(oracle[:, 2], modified[:, 2])
    assert not torch.equal(oracle[:, 1], modified[:, 1])
    torch.testing.assert_close(oracle, -delta)


def test_echo_off_gathers_without_class_masks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("class_masks must not be called on echo_off")
    monkeypatch.setattr(wab, "class_masks", forbidden)
    activations = torch.arange(30.).reshape(2, 5, 3)
    pos = np.array([[4, 2], [3, 2]])
    gate = kag._gate_positions(activations, pos, 2)
    np.testing.assert_allclose(gate["write_norm"],
                               np.linalg.norm(activations.numpy()[np.arange(2)[:, None], pos], axis=-1))


def test_norm_match_preserves_original_clip_before_kv_scaling():
    st = ttt._settings({"d_model": 16, "n_layers": 1, "n_heads": 2,
                        "backbone_mult": 2, "fast_mult": 2, "window": 4,
                        "chunk": 8, "echo": {"T": 32, "n_keys": 2}}, "echo")
    task = ttt._task(st, "echo")
    tokens, meta = task.raw(np.random.default_rng(8), 1)
    idx, y, _ = task.tensors(tokens, meta, torch.device("cpu"))
    model = ttt.build_model(task, st, torch.device("cpu"), 5, "e2e2").eval()
    with torch.no_grad():
        h = model.backbone(idx).detach()
        etas = [e.detach() for e in model.etas(3.)]
    masks = wab.class_masks(meta, task.chunk, torch.device("cpu"))
    weights = [w.detach().requires_grad_(True) for w in model.per_sequence(1)]
    z, _, grads = wab._class_grads(model, h[:, :8], y[:, :8], weights, masks)
    full = wab._sum_grads(grads, wab.ALL)
    g_clip = .01
    _, norm, clipped = ttt._clip_fast(full, g_clip)
    assert clipped.item()
    scale = torch.where(clipped, g_clip / norm, torch.ones_like(norm))
    own = wab._joint_norm(wab._weighted(grads[2], etas, scale)).item()
    matched = kag._norm_match(model, h, y, z, grads, etas, scale, own, task.chunk, g_clip)
    direct = wab.eval_condition(model, h, y, z, grads, etas, g_clip,
                                task.chunk, "fixed", wab.ALL)
    torch.testing.assert_close(matched, direct, rtol=0, atol=1e-5)
