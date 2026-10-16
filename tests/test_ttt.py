"""Unit checks for the TTT toy: ceilings, inner-loop algebra, causality, task layout."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as ttt
from tttmem.exp_ttt import EchoVocab, Settings, Task, bayes_ce, build_model, echo_batch

DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def double_precision_identities():
    """Keep algebra checks in float64 without changing other modules' tests."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def tiny_settings(**over) -> Settings:
    """Small settings: one layer, narrow model, default knobs otherwise."""
    st = ttt._settings({"d_model": 16, "n_layers": 1, "n_heads": 2, "backbone_mult": 2,
                        "fast_mult": 2, "window": 8, "chunk": 4, "n_vocab": 8, "T": 16,
                        "steps": 2, "batch": 2, "n_eval": 4, "n_val": 2, "n_quick_eval": 2,
                        "eval_batch": 2})
    base = {f.name: getattr(st, f.name) for f in dataclasses.fields(Settings)}
    return Settings(**{**base, **over})


def tiny_task(**over) -> Task:
    """A tiny Markov-chain task for forward-pass tests."""
    fields = dict(name="icl_mc", V=8, T=16, window=8, chunk=4, steps=2, n_eval=4, n_val=2)
    return Task(**{**fields, **over})


def batch_of(task: Task, seed: int, n: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic evaluation batch for a task."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, task.V, size=(n, task.T + 1))
    return task.tensors(tokens, {}, DEV)[:2]


# --------------------------------------------------------------------------- ceilings


def test_mc_chains_follow_their_transition_matrix():
    """Chains are Markov: pooled transitions and tokens are uniform, alpha changes stickiness.

    ``E[P[i]] = 1 / V`` for every Dirichlet row, so the pooled next-token
    distribution over many chains must be uniform whatever ``alpha`` is, and a
    smaller ``alpha`` must trap chains in fewer states.
    """
    V, T, n_seq = 4, 64, 4000
    uniform_tol = 0.02
    seen = {}
    for alpha in (0.5, 0.1):
        x = ttt.mc_batch(np.random.default_rng(0), n_seq, T, V, alpha)
        assert x.shape == (n_seq, T + 1)
        assert x.min() >= 0 and x.max() < V
        assert np.array_equal(np.bincount(x[:, 0], minlength=V).sum(), n_seq)
        emp = np.zeros(V)
        for i in range(V):
            sel = x[:, :-1] == i
            emp += np.bincount(x[:, 1:][sel], minlength=V)
        assert np.abs(emp / emp.sum() - 1 / V).max() < uniform_tol, alpha
        seen[alpha] = np.mean([len(np.unique(r[:-1])) for r in x])
    assert seen[0.1] < seen[0.5], "a sparser prior must make chains stickier"


def test_mc_bayes_is_attained_only_by_a_causal_predictor():
    """A trained model stays above the exact Bayes curve on fresh batches.

    This is the leak check: if the forward pass ever saw a target, the model
    would sit *below* the Bayes ceiling.
    """
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=0, arm="fa")
    rng = np.random.default_rng(12)
    opt = ttt.build_optim(model, st)
    gaps = []
    for _ in range(25):
        tokens, _ = task.raw(rng, 4)
        bce, _ = bayes_ce(tokens, task.V, st.alpha, task.window)
        idx, y = task.tensors(tokens, {}, DEV)[:2]
        loss = ttt.arm_loss(model, "fa", idx, y, 1.0, st.g_clip, task.chunk, ttt.GradStats())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), st.grad_clip)
        opt.step()
        logits = ttt.eval_batch_logits(model, task, "fa", idx, y, 1.0, st.g_clip, task.chunk,
                                       ttt.GradStats())
        gaps.append(float(ttt.token_ce(logits, y).mean().detach()) - float(bce.mean()))
    assert np.mean(gaps) > 0.0, "model below the Bayes ceiling on fresh data means a leak"


def test_bayes_matches_brute_force():
    """Full-context posterior predictive equals an explicit Python count."""
    rng = np.random.default_rng(0)
    V, alpha, T = 5, 0.3, 24
    tokens = ttt.mc_batch(rng, 3, T, V, alpha)
    ce, _ = bayes_ce(tokens, V, alpha)
    assert ce.shape == (3, T)
    for b in range(tokens.shape[0]):
        counts = np.zeros((V, V))
        for t in range(T):
            row = counts[tokens[b, t]].copy()
            p = (row + alpha) / (row.sum() + V * alpha)
            expect = -np.log(p[tokens[b, t + 1]])
            assert ce[b, t] == pytest.approx(expect, abs=1e-10)
            counts[tokens[b, t], tokens[b, t + 1]] += 1


def test_bayes_window_matches_brute_force():
    """Windowed curve counts only transitions fully inside the last ``w`` tokens."""
    rng = np.random.default_rng(1)
    V, alpha, T, w = 4, 0.2, 20, 5
    tokens = ttt.mc_batch(rng, 3, T, V, alpha)
    _, ce_w = bayes_ce(tokens, V, alpha, window=w)
    for b in range(tokens.shape[0]):
        for t in range(T):
            counts = np.zeros(V)
            for s in range(max(0, t - w + 1), t):
                if tokens[b, s] == tokens[b, t]:
                    counts[tokens[b, s + 1]] += 1
            p = (counts + alpha) / (counts.sum() + V * alpha)
            expect = -np.log(p[tokens[b, t + 1]])
            assert ce_w[b, t] == pytest.approx(expect, abs=1e-10)


def test_bayes_window_is_worse_than_full_context():
    """A shorter window can only lose information."""
    rng = np.random.default_rng(2)
    V, alpha, T = 6, 0.2, 30
    tokens = ttt.mc_batch(rng, 4, T, V, alpha)
    full, win = bayes_ce(tokens, V, alpha, window=6)
    assert win[:, 10:].mean() > full[:, 10:].mean()


# --------------------------------------------------------------------------- inner loops


def test_jacobi_full_sweeps_match_sequential():
    """Jacobi with ``C`` (or ``C - 1``) sweeps reproduces the sequential forward."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=0, arm="e2e1")
    idx, y = batch_of(task, seed=3)
    h = model.backbone(idx)
    etas = model.etas(1.0)
    seq = ttt.sequential_inner(model, h, y, model.per_sequence(2), etas, st.g_clip, False,
                              ttt.GradStats(), task.chunk).detach()
    C = task.T // task.chunk
    for sweeps in (C, C - 1):
        par = ttt.jacobi_inner(model, h, y, model.slow(), etas, st.g_clip, sweeps,
                               ttt.GradStats(), task.chunk).detach()
        assert float(seq) == pytest.approx(float(par), abs=1e-5)
    one = float(ttt.jacobi_inner(model, h, y, model.slow(), etas, st.g_clip, 1, ttt.GradStats(),
                                 task.chunk).detach())
    assert abs(float(seq) - one) > 1e-5, "one sweep must not already be the sequential loop"


def test_zero_eta_ttt_equals_static_head():
    """With ``eta = 0`` the fast weights never move, so TTT is the static head."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=1, arm="e2e1")
    idx, y = batch_of(task, seed=4)
    h = model.backbone(idx)
    static = ttt._ce_tokens(model, ttt.fast_block(h, model.head.ln, model.slow()), y).mean().detach()
    loss = ttt.sequential_inner(model, h, y, model.per_sequence(2), model.etas(0.0), st.g_clip,
                                False, ttt.GradStats(), task.chunk).detach()
    assert float(loss.detach()) == pytest.approx(float(static), abs=1e-10)


def test_second_order_grads_differ_from_first_order():
    """``e2e2`` gradients are finite and differ from the detached ``e2e1`` ones."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=2, arm="e2e2")
    idx, y = batch_of(task, seed=5)
    grads = {}
    for arm in ("e2e1", "e2e2"):
        model.zero_grad(set_to_none=True)
        loss = ttt.arm_loss(model, arm, idx, y, 1.0, st.g_clip, task.chunk, ttt.GradStats())
        loss.backward()
        grads[arm] = model.head.W1.grad.detach().clone()
        assert torch.isfinite(grads[arm]).all()
        assert float(grads[arm].abs().sum()) > 0
    delta = float((grads["e2e2"] - grads["e2e1"]).abs().max())
    assert delta > 1e-8, "second-order and first-order W0 gradients must differ"


def test_first_order_still_updates_slow_weights_and_eta():
    """``e2e1`` is first order in the inner step only: ``W0`` and ``rho`` still move."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=3, arm="e2e1")
    idx, y = batch_of(task, seed=6)
    loss = ttt.arm_loss(model, "e2e1", idx, y, 1.0, st.g_clip, task.chunk, ttt.GradStats())
    loss.backward()
    for name in ("W1", "W2", "b1", "b2"):
        grad = getattr(model.head, name).grad
        assert grad is not None and float(grad.abs().sum()) > 0, name
    assert model.rho.grad is not None and float(model.rho.grad.abs().sum()) > 0


def test_eta_logits_train_on_their_own_group():
    """``rho`` sits in its own group: lr 1e-2, no weight decay, no gradient clip."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=3, arm="e2e1")
    opt = ttt.build_optim(model, st)
    group = [g for g in opt.param_groups if any(p is model.rho for p in g["params"])][0]
    assert group["base_lr"] == st.eta_lr
    assert group["weight_decay"] == 0.0
    assert [g for g in opt.param_groups if g["weight_decay"] == st.weight_decay][0][
        "params"] == [p for p in model.parameters() if p.ndim >= 2 and p is not model.rho]


@pytest.mark.parametrize("arm", ["fa", "swa", "naive", "e2e1", "e2e2", "jac1", "jac2", "jac2so"])
def test_evaluate_keeps_no_autograd_graph(arm, monkeypatch):
    """Evaluation runs under ``no_grad``: no eval logits may require grad, any arm.

    Holding sixteen batches of ``(B, T, V)`` logits with their graph alive
    exhausts GPU memory at the full-scale config, so the static arms matter as much
    as the inner-loop ones here. The Jacobi arms also get their parallel eval
    pass, since ``evaluate`` runs both loops for them.
    """
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=9, arm="swa" if arm == "naive" else arm)
    tokens, _ = task.raw(np.random.default_rng(13), 2)
    data = ttt.eval_batches(task, tokens, {}, DEV, 2)
    seen: list[torch.Tensor] = []
    real = ttt.eval_batch_logits

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(ttt, "eval_batch_logits", spy)
    ttt.evaluate(model, task, arm, data, {}, 1.0, st)
    assert seen, "the spy never saw a forward pass"
    assert not any(bool(out.requires_grad) for out in seen), arm


def test_eval_inner_loop_still_computes_gradients():
    """``no_grad`` at eval time must not disable the inner loops' own gradients."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=10, arm="e2e1")
    tokens, _ = task.raw(np.random.default_rng(14), 2)
    data = ttt.eval_batches(task, tokens, {}, DEV, 2)
    out = ttt.evaluate(model, task, "e2e1", data, {}, 1.0, st)
    assert out["eval_grad"]["n_pairs"] > 0, "the eval-time inner loop recorded no gradients"
    assert np.isfinite(out["eval"]["ce"])


def test_jacobi_and_sequential_agree_at_eval():
    """A fully converged Jacobi arm matches the streaming eval inner loop."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=4, arm="jac2")
    idx, y = batch_of(task, seed=7)
    etas = model.etas(1.0)
    seq = ttt.eval_sequential(model, idx, y, etas, st.g_clip, task.chunk, ttt.GradStats())
    par = ttt.eval_jacobi(model, idx, y, etas, st.g_clip, task.chunk, task.T // task.chunk,
                          ttt.GradStats())
    assert torch.allclose(seq, par, atol=1e-5)


def test_gradient_clip_is_per_sequence():
    """Clipping is per leading index: one huge gradient does not shrink the other."""
    grads = [0.5 * torch.ones(2, 3, 4), 0.5 * torch.ones(2, 3), 0.5 * torch.ones(2, 4, 3),
             torch.zeros(2, 4)]
    grads[1][1] = 1000.0
    clipped, norm, hit = ttt._clip_fast(grads, g_clip=5.0)
    assert hit.tolist() == [False, True]
    assert float(norm[0]) == pytest.approx(6.75**0.5, rel=1e-6)
    joint = sum(g.reshape(2, -1).pow(2).sum(1) for g in clipped).sqrt()
    assert float(joint[1]) == pytest.approx(5.0, rel=1e-6)
    assert float(joint[0]) == pytest.approx(6.75**0.5, rel=1e-6)


# --------------------------------------------------------------------------- causality


@pytest.mark.parametrize("arm", ["fa", "swa", "naive", "e2e1", "e2e2", "jac1", "jac2", "jac2so"])
@pytest.mark.parametrize("jacobi", [False, True])
def test_causality_every_arm(arm, jacobi):
    """Perturbing token ``t + 1`` leaves the logits at positions ``<= t`` unchanged."""
    if jacobi and arm not in ttt.JACOBI_ARMS:
        pytest.skip("only the Jacobi arms have a parallel eval mode")
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=5, arm="swa" if arm == "naive" else arm)
    rng = np.random.default_rng(8)
    tokens = rng.integers(0, task.V, size=(2, task.T + 1))
    for t in (0, 3, task.T - 2):
        idx, y = task.tensors(tokens, {}, DEV)[:2]
        base = ttt.eval_batch_logits(model, task, arm, idx, y, 1.0, st.g_clip, task.chunk,
                                     ttt.GradStats(), jacobi=jacobi)
        other = tokens.copy()
        other[:, t + 1] = (other[:, t + 1] + 1) % task.V
        idx2, y2 = task.tensors(other, {}, DEV)[:2]
        pert = ttt.eval_batch_logits(model, task, arm, idx2, y2, 1.0, st.g_clip, task.chunk,
                                     ttt.GradStats(), jacobi=jacobi)
        assert torch.allclose(base[:, : t + 1], pert[:, : t + 1], atol=1e-5), (arm, t, jacobi)
        assert float((base[:, t + 1:] - pert[:, t + 1:]).abs().max().detach()) > 0.0


def test_causality_echo_task():
    """The same check on the echo layout, where the copyable pair sits inside a chunk."""
    st = tiny_settings(window=16, chunk=8)
    task = Task(name="echo", V=EchoVocab(4).size, T=96, window=16, chunk=8, steps=1, n_eval=2,
                n_val=2, mode="echo", n_keys=4)
    model = build_model(task, st, DEV, seed=6, arm="e2e2")
    tokens, meta = echo_batch(np.random.default_rng(9), 2, task.T, task.window, "echo", 4)
    for t in (int(meta["query_pos"][0, 0]), int(meta["write_pos"][0, 0]) + 1):
        idx, y = task.tensors(tokens, meta, DEV)[:2]
        base = ttt.eval_batch_logits(model, task, "e2e2", idx, y, 1.0, st.g_clip, task.chunk,
                                     ttt.GradStats())
        other = tokens.copy()
        other[:, t + 1] = (other[:, t + 1] + 1) % task.V
        idx2, y2 = task.tensors(other, meta, DEV)[:2]
        pert = ttt.eval_batch_logits(model, task, "e2e2", idx2, y2, 1.0, st.g_clip, task.chunk,
                                     ttt.GradStats())
        assert torch.allclose(base[:, : t + 1], pert[:, : t + 1], atol=1e-5), t


# --------------------------------------------------------------------------- tasks


def test_echo_generator_layout():
    """Queries sit outside the stacked receptive field and the value lands where it should."""
    window, T, n_keys, n_seq, n_layers = 16, 512, 16, 8, 4
    vocab = EchoVocab(n_keys)
    tokens, meta = echo_batch(np.random.default_rng(10), n_seq, T, window, "echo", n_keys, n_layers)
    assert tokens.shape == (n_seq, T + 1)
    assert set(np.unique(tokens)).issubset(set(range(vocab.size)))
    gaps, same = [], []
    for b in range(n_seq):
        gaps.append((meta["query_pos"][b] - meta["write_pos"][b]).min())
        for k in range(n_keys):
            pos = int(meta["write_pos"][b, k])
            assert tokens[b, pos] == vocab.key[k]
            assert tokens[b, pos + 1] == meta["query_tok"][b, k]
            assert tokens[b, int(meta["query_pos"][b, k]) + 1] == meta["query_tok"][b, k]
            same.append(tokens[b, pos + 1] == tokens[b, pos - 1])
    reach = n_layers * (window - 1)
    assert min(gaps) > reach, "every first query must be more than n_layers*(w-1) after its write"
    assert all(same), "in echo the value after k must be the value two tokens back"
    ctrl, _ = echo_batch(np.random.default_rng(11), n_seq, T, window, "echo_ctrl", n_keys, n_layers)
    copyable = [ctrl[b, int(meta["write_pos"][b, k]) + 1] == ctrl[b, int(meta["write_pos"][b, k]) - 1]
                for b in range(n_seq) for k in range(n_keys)]
    assert np.mean(copyable) < 0.2, "in echo_ctrl the value after k must not be copyable"


def test_echo_rejects_too_short_sequences():
    """The layout check fires instead of silently truncating the query phase."""
    with pytest.raises(ValueError):
        echo_batch(np.random.default_rng(12), 1, 128, 16, "echo", 16)
    with pytest.raises(ValueError):
        echo_batch(np.random.default_rng(12), 1, 200, 16, "echo", 16, n_layers=4)


def test_query_metric_uses_value_token():
    """``query_ce`` is the CE of the position whose target is the query's value."""
    logits = torch.full((2, 32, 4), -8.0)
    logits[:, 7, 2] = 8.0
    logits[1, 7, 2] = -8.0
    logits[1, 7, 1] = 8.0
    y = torch.zeros(2, 32, dtype=torch.long)
    meta = dict(query_pos=np.array([[7, 9], [7, 9]]), query_tok=np.zeros((2, 2), dtype=np.int64))
    ce = ttt.token_ce(logits, y)
    summ = ttt.summarize_ce(ce, n_bins=4, late_frac=0.25, meta=meta, preds=logits.argmax(-1))
    assert summ["query_ce"] == pytest.approx(float((ce[0, 7] + ce[1, 7] + 2 * ce[0, 9]) / 4))
    assert summ["query_ce"] > 4 * summ["ce"], "only queried positions count, not filler"
    assert summ["query_acc"] == pytest.approx(0.5)


def test_fixed_sets_do_not_depend_on_the_training_seed():
    """Eval, val and quick sets are seeded by ``EVAL_SEED`` alone.

    The control rule is a paired test on per-sequence late CEs, and the ``fa`` and
    Bayes references are single models, so every seed has to be scored on the same
    sequences.
    """
    st, task = tiny_settings(), tiny_task()
    for which in ("eval", "val", "quick"):
        a, ma = ttt.fixed_set(task, st, which)
        b, mb = ttt.fixed_set(task, st, which)
        assert np.array_equal(a, b), which
        c, _ = ttt.fixed_set(task, st, which)
        assert np.array_equal(a, c), which
    ev, _ = ttt.fixed_set(task, st, "eval")
    val, _ = ttt.fixed_set(task, st, "val")
    assert not np.array_equal(ev[: val.shape[0]], val), "val must stay disjoint from eval"


def test_effective_receptive_field_is_stacked():
    """``n_layers`` stacked windows of width ``w`` see ``n_layers * (w - 1) + 1``."""
    task = tiny_task(n_layers=4)
    assert task.receptive_field == 4 * (task.window - 1) + 1


def test_paired_test_needs_two_standard_errors():
    """A small mean difference with a large spread does not count as beating ``swa``."""
    tres = {"runs": {"seed0": {
        "swa": {"eval": {"late_per_seq": [1.0] * 20}},
        "e2e2": {"eval": {"late_per_seq": [0.98] * 20}},
        "e2e1": {"eval": {"late_per_seq": [0.0, 2.0] * 10}},
    }}}
    res = ttt.paired_vs_swa(tres, "e2e2")
    assert res["mean_diff"] == pytest.approx(0.02)
    assert res["passed"] is True, "a constant positive difference has zero spread"
    noisy = ttt.paired_vs_swa(tres, "e2e1")
    assert noisy["mean_diff"] == pytest.approx(0.0)
    assert noisy["passed"] is False, "mean difference inside its own noise"
    assert ttt.paired_vs_swa({"runs": {}}, "e2e2")["passed"] is None


def test_lr_schedule_warms_up_then_decays():
    """Warmup rises linearly, then cosine decay ends at the floor."""
    assert ttt.lr_scale(0, 200, 1000, 0.1) == pytest.approx(1 / 200)
    assert ttt.lr_scale(199, 200, 1000, 0.1) == pytest.approx(1.0)
    assert ttt.lr_scale(1000, 200, 1000, 0.1) == pytest.approx(0.1)


def test_gap_needs_a_resolvable_denominator():
    """A gap fraction is ``None`` unless the reference gap clears two standard errors."""
    assert ttt._gap(1.5, 2.0, 0.01, 1.0, 0.01) == pytest.approx(0.5)
    assert ttt._gap(1.5, 2.0, 0.4, 1.9, 0.4) is None, "reference gap inside the noise"
    assert ttt._gap(None, 2.0, 0.01, 1.0, 0.01) is None
    assert ttt._gap(1.5, 1.0, 0.01, 1.0, 0.01) is None


def test_settings_merge_task_table():
    """A ``[ttt.<task>]`` table overrides the shared table for that task only."""
    cfg = {"T": 100, "window": 64, "icl_mc": {"T": 512, "arms": ["fa"]}}
    assert ttt._settings(cfg, "icl_mc").T == 512
    assert ttt._settings(cfg, "icl_mc").task_table("icl_mc")["arms"] == ["fa"]
    assert ttt._settings(cfg, "echo").T == 100
    assert ttt._settings(cfg, "icl_mc").task_table("echo") == {}


def test_round1_data_hashes():
    """Train streams and fixed eval splits match their recorded hashes byte for byte."""
    def digest(array):
        """Hash contiguous array bytes, independent of metadata additions."""
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]
    expected = {"echo": ("4367e83f0c8a7d70", "c7dbc0d2ea48ae1c", "659ccbe05e5bbb6f"),
                "echo_ctrl": ("3de384f1d410fbd4", "7d2be0c9d53dff16", "3f4b306e47b8120c")}
    for mode, hashes in expected.items():
        tokens, meta = echo_batch(np.random.default_rng([0, ttt.TRAIN_SEED]),
                                  32, 512, 64, mode, 16, 4)
        assert tuple(digest(a) for a in (tokens, meta["query_pos"], meta["write_pos"])) == hashes
    assert digest(ttt.mc_batch(np.random.default_rng([0, ttt.TRAIN_SEED]),
                               32, 1024, 32, 0.1)) == "8240cd4231082d33"
    cfg = {"n_layers": 4, "window": 64, "n_keys": 16, "n_vocab": 32, "alpha": 0.1,
           "icl_mc": {"T": 1024, "n_eval": 512, "n_vocab": 32, "alpha": 0.1},
           "echo": {"T": 512, "n_eval": 256, "n_keys": 16},
           "echo_ctrl": {"T": 512, "n_eval": 256, "n_keys": 16}}
    st = ttt._settings(cfg)
    for mode, expected_hash in (("echo", "57edf31400c721fe"),
                                ("echo_ctrl", "3872d3c18a3c8536"),
                                ("icl_mc", "a04c4436a76a70b7")):
        assert digest(ttt.fixed_set(ttt._task(st, mode), st, "eval")[0]) == expected_hash


@pytest.mark.parametrize("clip", [1e6, 0.01])
def test_jac2so_exact_outer_gradients(clip):
    """At C-1 sweeps all slow, backbone and rho gradients match sequential SO."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=2, arm="jac2so")
    idx, y = batch_of(task, seed=5)
    ref = ttt.outer_grads(model, "sequential", 0, True, idx, y, 1.0, clip, task.chunk)
    par = ttt.outer_grads(model, "jacobi", task.T // task.chunk - 1, True,
                          idx, y, 1.0, clip, task.chunk)
    assert torch.isfinite(par).all()
    assert float((par - ref).norm() / ref.norm()) < 1e-6
    assert float(torch.dot(par, ref) / (par.norm() * ref.norm())) > 0.999999
    first = ttt.outer_grads(model, "jacobi", 2, False, idx, y, 1.0, clip, task.chunk)
    second = ttt.outer_grads(model, "jacobi", 2, True, idx, y, 1.0, clip, task.chunk)
    assert not torch.allclose(first, second, atol=1e-9)


def test_picard_diagnostics_converge():
    """Iterate k has exact prefix chunks 0..k and all chunks by C-1."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, 0, "e2e2")
    idx, y = batch_of(task, seed=1)
    diag = ttt.picard_diagnostics(model, idx, y, 1.0, st.g_clip, task.chunk)
    for k in (1, 2, 3):
        assert max(diag[f"rel_err_{k}"][:k + 1]) < 1e-6


def test_new_echo_layouts():
    """Offset writes stay outside the receptive field; no-r writes a filler."""
    vocab = EchoVocab(16)
    tokens, meta = echo_batch(np.random.default_rng(7), 512, 640, 64, "echo_off", 16, 4)
    assert (meta["write_start"] % 64 != 0).mean() > 0.9
    assert {1, 2}.issubset(set(meta["write_start"] // 64))
    assert (meta["query_pos"] - meta["write_pos"]).min() > 4 * 63
    for b in range(512):
        for k in range(16):
            pos = meta["write_pos"][b, k]
            assert tokens[b, pos + 1] == tokens[b, pos - 1] == meta["query_tok"][b, k]
    control, cm = echo_batch(np.random.default_rng(7), 16, 512, 64, "echo_ctrl_nor", 16, 4)
    _, old_meta = echo_batch(np.random.default_rng(7), 16, 512, 64, "echo_ctrl", 16, 4)
    assert np.array_equal(cm["write_pos"], old_meta["write_pos"])
    assert np.array_equal(cm["query_pos"], old_meta["query_pos"])
    for b in range(16):
        assert cm["write_start"][b] == 0
        for k in range(16):
            pos = cm["write_pos"][b, k]
            assert control[b, pos - 1] in vocab.filler
            assert control[b, pos] == vocab.key[k]
            assert control[b, pos + 1] == cm["query_tok"][b, k]


def test_round2b_config_order_and_settings():
    """The archived R2b smoke config retains its original cell and task settings."""
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/toy/smoke/r2b.toml").open("rb") as handle:
        smoke = tomllib.load(handle)["ttt"]
    assert len(smoke["cells"]) == 19
    assert ttt._settings(smoke, "icl_mc").lr_total == 0
    assert not any("gate_min_acc" in cell for cell in smoke["cells"])
    assert smoke["echo"]["lr_total"] == 8
    assert smoke["echo"]["n_keys"] == 4
    assert all(cell["arm"] in ttt.STATIC_ARMS + ttt.TTT_ARMS for cell in smoke["cells"])


def test_echo_references_exact_and_monte_carlo():
    """Exact unbound distributions agree with hand counts and independent draws."""
    meta = dict(query_tok=np.array([[2, 2, 3]]), query_pos=np.array([[0, 1, 2]]))
    refs = ttt.echo_references(meta, 3)
    assert refs["chance"]["ce"] == pytest.approx(math.log(3))
    assert refs["occ"]["ce"] == pytest.approx(-(2 * math.log(2 / 3) + math.log(1 / 3)) / 3)
    assert refs["occ"]["acc"] == pytest.approx(2 / 3)
    assert refs["occ_minus"]["ce"] == pytest.approx(-(math.log(2 / 3) + math.log(1 / 2)) / 3)
    assert refs["occ_minus"]["acc"] == pytest.approx(5 / 6)
    rng = np.random.default_rng(13)
    values = rng.integers(0, 16, size=(4000, 16))
    positions = np.argsort(rng.random((4000, 16)), axis=1)
    refs = ttt.echo_references(dict(query_tok=values, query_pos=positions), 16)
    for key in ("occ", "occ_minus"):
        assert abs(refs[key]["ce"] - (2.226 if key == "occ" else 1.628)) < 0.02
        assert abs(refs[key]["acc"] - (0.192 if key == "occ" else 0.319)) < 0.01
    simulated = {key: [[], []] for key in ("occ", "occ_minus")}
    for row, pos in zip(values, positions):
        counts = np.bincount(row, minlength=16)
        remaining = counts.copy()
        for j, k in enumerate(np.argsort(pos)):
            for key, dist in (("occ", counts / 16), ("occ_minus", remaining / (16 - j))):
                simulated[key][0].append(-math.log(dist[row[k]]))
                winners = np.flatnonzero(dist == dist.max())
                simulated[key][1].append(int(rng.choice(winners) == row[k]))
            remaining[row[k]] -= 1
    for key, (ce, hits) in simulated.items():
        assert abs(refs[key]["ce"] - np.mean(ce)) < 0.02
        assert abs(refs[key]["acc"] - np.mean(hits)) < 0.01


def test_cells_gate_and_isolation(monkeypatch, tmp_path):
    """A failed anchor gates ordinary cells; marked replications survive an error."""
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=8, n_keys=4, batch=2, n_eval=2, n_quick_eval=2,
               eval_batch=2, device="cpu", quick_every=2, max_minutes=10,
               echo=dict(T=64), cells=[
                   dict(task="echo", seed=0, arm="swa", steps=1, gate_min_acc=1.1),
                   dict(task="echo", seed=1, arm="swa", steps=1),
                   dict(task="echo", seed=2, arm="swa", steps=1, run_if_gate_fails=True),
                   dict(task="echo", seed=3, arm="swa", steps=1, run_if_gate_fails=True)])
    real_train = ttt.train_arm
    def broken(model, task, arm, eta, st, dev, rng, quick, tag, steps=None, seed=None):
        """Fail one replication after the gate without hiding the next cell."""
        if "s2/" in tag:
            raise RuntimeError("isolated")
        return real_train(model, task, arm, eta, st, dev, rng, quick, tag, steps, seed=seed)
    monkeypatch.setattr(ttt, "train_arm", broken)
    result = ttt.run_cells(cfg, ttt._settings(cfg), DEV)
    records = list(result["cells"].values())
    assert records[0]["eval"]["query_acc"] < 1.1
    assert records[1]["reason"] == "gate"
    assert "isolated" in records[2]["error"]
    assert records[3]["skipped"] is False
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 2


def test_cells_eta_ids_and_budget(monkeypatch, tmp_path):
    """Static and two pinned-eta cells train independently; expired budget skips all."""
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=8, n_vocab=8, batch=2, n_eval=2, n_quick_eval=2,
               eval_batch=2, device="cpu", quick_every=2, max_minutes=10,
               icl_mc=dict(T=16), cells=[
                   dict(task="icl_mc", seed=0, arm="swa", steps=1),
                   dict(task="icl_mc", seed=0, arm="e2e1", eta_base=0.3, steps=1),
                   dict(task="icl_mc", seed=0, arm="e2e1", eta_base=1.0, steps=1)])
    out = ttt.run_cells(cfg, ttt._settings(cfg), DEV)
    assert len(out["cells"]) == 3 and all(not x.get("error") for x in out["cells"].values())
    assert all("checkpoint" in x for x in out["cells"].values())
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 3
    cfg["max_minutes"] = -1
    skipped = ttt.run_cells(cfg, ttt._settings(cfg), DEV)["cells"]
    assert all(x.get("reason") == "budget" for x in skipped.values())


def test_nan_guard_skips_update(monkeypatch):
    """A non-finite loss does not step the optimizer or poison parameters."""
    st, task = tiny_settings(), tiny_task(steps=3)
    model = build_model(task, st, DEV, 0, "e2e1")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    real_loss = ttt.arm_loss
    calls = [0]
    def nan_once(*args, **kwargs):
        """Return a NaN on the first call only."""
        calls[0] += 1
        return real_loss(*args, **kwargs) * float("nan") if calls[0] == 1 else real_loss(*args, **kwargs)
    monkeypatch.setattr(ttt, "arm_loss", nan_once)
    out = ttt.train_arm(model, task, "e2e1", 1.0, st, DEV,
                        np.random.default_rng(4), None, "nan-test")
    assert out["nonfinite_steps"] == 1 and out["first_nonfinite"] == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_nan_guard_catches_a_nonfinite_rho_gradient(monkeypatch):
    """``rho`` sits outside the clip, so its gradient needs its own finiteness check."""
    st, task = tiny_settings(), tiny_task(steps=2)
    model = build_model(task, st, DEV, 0, "e2e1")
    real_loss = ttt.arm_loss
    calls = [0]

    def poison_rho(m, *args, **kwargs):
        """Add a term whose gradient reaches only ``rho`` and is infinite once."""
        calls[0] += 1
        loss = real_loss(m, *args, **kwargs)
        if calls[0] == 1:
            # sqrt(0) keeps the loss finite but has an infinite gradient
            loss = loss + torch.sqrt(m.rho[0] - m.rho[0].detach())
            assert torch.isfinite(loss)
        return loss
    monkeypatch.setattr(ttt, "arm_loss", poison_rho)
    out = ttt.train_arm(model, task, "e2e1", 1.0, st, DEV, np.random.default_rng(5), None, "rho")
    assert out["nonfinite_steps"] == 1
    assert torch.isfinite(model.rho).all()


def test_contraction_is_masked_on_exact_chunks():
    """Chunks ``c <= k`` are exact after ``k`` sweeps, so their ratio is reported as None."""
    st, task = tiny_settings(), tiny_task()
    model = build_model(task, st, DEV, seed=4, arm="e2e1")
    idx, y = batch_of(task, seed=7)
    diag = ttt.picard_diagnostics(model, idx, y, 1.0, st.g_clip, task.chunk)
    for k in (1, 2):
        ratio = diag[f"contraction_{k}"]
        assert all(r is None for r in ratio[: k + 1])
        assert all(r is not None for r in ratio[k + 1:])
        assert diag[f"contraction_{k}_median"] is not None


def test_a_failing_diagnostic_keeps_the_cell_and_a_failing_gate_cell_gates(monkeypatch, tmp_path):
    """Diagnostics run after the record is secured; a gate cell that errors trips the gate."""
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=8, n_keys=4, batch=2, n_eval=2, n_quick_eval=2,
               eval_batch=2, device="cpu", quick_every=2, max_minutes=10,
               echo=dict(T=64), cells=[
                   dict(task="echo", seed=0, arm="e2e1", eta_base=1.0, steps=1)])

    def broken(*args, **kwargs):
        """Stand in for a diagnostic that fails on the real hardware."""
        raise RuntimeError("diag")
    monkeypatch.setattr(ttt, "metagrad_fidelity", broken)
    rec = list(ttt.run_cells(cfg, ttt._settings(cfg), DEV)["cells"].values())[0]
    assert "eval" in rec and "checkpoint" in rec and "diag" in rec["diag_error"]
    assert not rec.get("error")

    cfg["cells"] = [dict(task="echo", seed=0, arm="swa", steps=1, gate_min_acc=0.0),
                    dict(task="echo", seed=1, arm="swa", steps=1)]
    real_train = ttt.train_arm

    def fail_anchor(model, task, arm, eta, st, dev, rng, quick, tag, steps=None):
        """Fail the gate cell only."""
        if "s0/" in tag:
            raise RuntimeError("anchor")
        return real_train(model, task, arm, eta, st, dev, rng, quick, tag, steps)
    monkeypatch.setattr(ttt, "train_arm", fail_anchor)
    recs = list(ttt.run_cells(cfg, ttt._settings(cfg), DEV)["cells"].values())
    assert "anchor" in recs[0]["error"] and recs[1]["reason"] == "gate"


def test_cell_grad_clip_override_diag_flag_and_spike_log(monkeypatch, tmp_path):
    """A per-cell outer clip reaches training only for that cell; diag=False skips diagnostics."""
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=8, n_keys=4, batch=2, n_eval=2, n_quick_eval=2,
               eval_batch=2, device="cpu", quick_every=2, max_minutes=10,
               echo=dict(T=64), cells=[
                   dict(task="echo", seed=0, arm="e2e2", eta_base=1.0, steps=3),
                   dict(task="echo", seed=0, arm="e2e2", eta_base=1.0, steps=3,
                        grad_clip=0.05, diag=False)])
    seen = []
    real_train = ttt.train_arm

    def spy(model, task, arm, eta, st, dev, rng, quick, tag, steps=None, seed=None):
        """Record the outer clip each cell trains with."""
        seen.append(st.grad_clip)
        return real_train(model, task, arm, eta, st, dev, rng, quick, tag, steps, seed=seed)
    monkeypatch.setattr(ttt, "train_arm", spy)
    cells = ttt.run_cells(cfg, ttt._settings(cfg), DEV)["cells"]
    plain, clipped = cells["echo/s0/e2e2/eta1/st3"], cells["echo/s0/e2e2/eta1/st3/clip0.05"]
    assert seen == [1.0, 0.05]
    assert plain["grad_clip"] == 1.0 and clipped["grad_clip"] == 0.05
    assert "metagrad_fidelity" in plain and "picard_diag" in plain
    assert "metagrad_fidelity" not in clipped and "picard_diag" not in clipped
    top = clipped["outer_grad"]["top"]
    assert 0 < len(top) <= 10 and all(a[1] >= b[1] for a, b in zip(top, top[1:]))
    assert ttt._settings(cfg).grad_clip == 1.0
    for rec in (plain, clipped):
        trace = np.load(rec["trace"])
        assert len(trace["loss"]) == 3 and len(trace["outer_norm"]) == len(trace["step"])
    assert "_trace" not in plain and "_trace" not in clipped


def test_spike_skip_drops_only_the_outlier_step_and_lr_total_extends_the_schedule(monkeypatch):
    """A 1000x loss on one post-warmup step is skipped; lr follows the longer schedule."""
    st, task = tiny_settings(), tiny_task(steps=16)
    st = dataclasses.replace(st, warmup=0, spike_skip=10.0, lr_total=40)
    real_loss = ttt.arm_loss
    calls = [0]

    def spike_once(*args, **kwargs):
        """Scale the loss (hence the outer gradient) by 1000 on the 14th step only."""
        calls[0] += 1
        return real_loss(*args, **kwargs) * (1000.0 if calls[0] == 14 else 1.0)
    monkeypatch.setattr(ttt, "arm_loss", spike_once)
    totals = []
    real_scale = ttt.lr_scale

    def spy(step, warmup, total, floor):
        """Record the schedule length each step uses."""
        totals.append(total)
        return real_scale(step, warmup, total, floor)
    monkeypatch.setattr(ttt, "lr_scale", spy)
    model = build_model(task, st, DEV, 0, "e2e1")
    out = ttt.train_arm(model, task, "e2e1", 1.0, st, DEV, np.random.default_rng(6), None, "spike")
    assert out["spike_skips"] == 1 and out["spike_skip_steps"] == [13]
    assert set(totals) == {40} and out["lr_total"] == 40
    trace = out["_trace"]
    assert len(trace["loss"]) == 16 and list(trace["spike_step"]) == [13]
    assert trace["outer_norm"][13] > 10 * np.median(trace["outer_norm"][3:13])
    with pytest.raises(ValueError):
        ttt.train_arm(model, task, "e2e1", 1.0, dataclasses.replace(st, lr_total=8), DEV,
                      np.random.default_rng(6), None, "short")


def test_cell_lr_override_and_jacobi_parallel_quick_eval(tmp_path, monkeypatch):
    """A cell's lr/lr_total reach train_arm and the record; Jacobi arms log quick_par."""
    monkeypatch.chdir(tmp_path)
    cfg = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=8, chunk=8, n_keys=4, batch=2, n_eval=2, n_quick_eval=2,
               eval_batch=2, device="cpu", quick_every=2, max_minutes=10, quick_par=True,
               echo=dict(T=64, lr_total=8), cells=[
                   dict(task="echo", seed=0, arm="jac2so", eta_base=1.0, steps=4,
                        lr=5e-4, lr_total=6, diag=False),
                   dict(task="echo", seed=0, arm="e2e2", eta_base=1.0, steps=4, diag=False)])
    seen = []
    real_train = ttt.train_arm

    def spy(model, task, arm, eta, st, dev, rng, quick, tag, steps=None, seed=None):
        """Record the lr settings each cell trains with."""
        seen.append((arm, st.lr, st.lr_total, type(st.lr_total)))
        return real_train(model, task, arm, eta, st, dev, rng, quick, tag, steps, seed=seed)
    monkeypatch.setattr(ttt, "train_arm", spy)
    cells = ttt.run_cells(cfg, ttt._settings(cfg), DEV)["cells"]
    assert seen == [("jac2so", 5e-4, 6, int), ("e2e2", 1e-3, 8, int)]
    jac, seq = cells["echo/s0/jac2so/eta1/st4/lr0.0005"], cells["echo/s0/e2e2/eta1/st4"]
    assert (jac["lr"], jac["lr_total"], seq["lr"], seq["lr_total"]) == (5e-4, 6, 1e-3, 8)
    assert jac["eta_lr"] == seq["eta_lr"] == 0.01
    assert [row[0] for row in jac["quick_par"]] == [2, 4]
    assert all(0.0 <= row[2] <= 1.0 for row in jac["quick_par"])
    assert [row[0] for row in jac["quick"]] == [2, 4] and len(jac["quick"][0]) == len(seq["quick"][0])
    assert seq["quick_par"] == []


def test_quick_par_is_off_by_default():
    """Without quick_par a Jacobi cell keeps the plain record (no parallel quick eval)."""
    st, task = tiny_settings(), tiny_task(steps=2)
    assert st.quick_par is False
    model = build_model(task, st, DEV, 0, "jac2so")
    quick = ttt.eval_batches(task, *task.raw(np.random.default_rng(5), 2), DEV, 2)
    out = ttt.train_arm(model, task, "jac2so", 1.0, st, DEV, np.random.default_rng(4), quick, "jac")
    assert "quick_par" not in out


def test_spike_skip_off_is_the_default_path():
    """With spike_skip = 0 and lr_total = 0 training matches the plain settings exactly."""
    st, task = tiny_settings(), tiny_task(steps=6)
    hashes = []
    for variant in (st, dataclasses.replace(st, spike_skip=0.0, lr_total=0),
                    dataclasses.replace(st, lr_total=6)):
        model = build_model(task, variant, DEV, 0, "e2e2")
        ttt.train_arm(model, task, "e2e2", 1.0, variant, DEV, np.random.default_rng(8), None, "d")
        hashes.append(torch.cat([p.detach().flatten() for p in model.parameters()]).sum().item())
    assert hashes[0] == hashes[1] == hashes[2]
