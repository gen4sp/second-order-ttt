"""R4 factor algebra, embedding paths, episode bookkeeping and worker isolation."""

import math
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_kag as kag
from tttmem import exp_ttt as ttt


def small_echo():
    st = ttt._settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                            fast_mult=2, window=16, chunk=16, n_keys=4, T=128,
                            n_eval=4, n_quick_eval=4, eval_batch=2, n_probe=4))
    return st, ttt._task(st, "echo")


def test_s1_s2_s3_at_init_and_five_steps():
    checks = ttt.r4_sanity()
    assert all(v["passed"] for v in checks.values()), checks
    assert checks["S2_trained_clip0.001"]["both_on_rel"] < 1e-4


def test_plain_arm_reference_loss_and_initialisation():
    st, task = small_echo()
    models = [ttt.build_model(task, st, torch.device("cpu"), 3, arm)
              for arm in ("e2e1", "e2e2", "e2e1k", "e2e1e", "e2e1emb")]
    assert all(torch.equal(models[0].tok.weight, m.tok.weight) for m in models[1:])
    idx, y, _ = task.tensors(*task.raw(np.random.default_rng(7), 2), torch.device("cpu"))
    h = models[0].backbone(idx)
    reference = ttt._ce_tokens(models[0], ttt.fast_block(h, models[0].head.ln,
                                 models[0].slow()), y).mean()
    actual = ttt.arm_loss(models[0], "e2e1", idx, y, 0., 10., task.chunk,
                          ttt.GradStats())
    assert torch.equal(reference, actual)
    losses = [ttt.arm_loss(m, arm, idx, y, 10., 10., task.chunk, ttt.GradStats())
              for m, arm in zip(models[:4], ("e2e1", "e2e2", "e2e1k", "e2e1e"))]
    assert torch.allclose(torch.stack(losses), losses[0].expand(4), rtol=1e-5)


def test_embedding_in_all_forward_paths_and_checkpoint(tmp_path, monkeypatch):
    st, task = small_echo()
    dev = torch.device("cpu")
    model = ttt.build_model(task, st, dev, 2, "e2e1emb")
    tokens, meta = task.raw(np.random.default_rng(9), 4)
    data = ttt.eval_batches(task, tokens, meta, dev, 2)
    idx, y, _ = data[0]
    h = model.backbone(idx)
    x = model.fast_input(h, idx)
    manual = h + torch.einsum("btm,dm->btd", torch.nn.functional.gelu(
        torch.nn.functional.linear(x, *model.slow()[:2])), model.slow()[2]) + model.slow()[3]
    assert torch.allclose(model.fast_block(h, idx, model.slow()), manual)
    assert torch.allclose(kag._code(model, h, idx=idx), torch.nn.functional.gelu(
        torch.nn.functional.linear(x, *model.slow()[:2])))
    logits = ttt.eval_batch_logits(model, task, "e2e1emb", idx, y, 0., 10.,
                                    task.chunk, ttt.GradStats())
    assert torch.allclose(logits, model.read(manual), atol=1e-6)
    train_loss = ttt.arm_loss(model, "e2e1emb", idx, y, 0., 10., task.chunk,
                              ttt.GradStats())
    assert torch.allclose(train_loss, ttt.token_ce(model.read(manual), y).mean(), atol=1e-6)
    original_eval = ttt.eval_batch_logits
    seen = []

    def checked_eval(m, t, arm, ix, yy, *args, **kwargs):
        result = original_eval(m, t, arm, ix, yy, *args, **kwargs)
        reference = m.read(m.fast_block(m.backbone(ix), ix, m.slow()))
        assert torch.allclose(result, reference, atol=1e-6)
        seen.append(result)
        return result

    monkeypatch.setattr(ttt, "eval_batch_logits", checked_eval)
    assert math.isfinite(ttt.quick_summary(model, task, "e2e1emb", 0., st, data)["ce"])
    assert math.isfinite(ttt.evaluate(model, task, "e2e1emb", data, meta, 0., st)["eval"]["ce"])
    assert len(seen) == 4
    monkeypatch.setattr(ttt, "eval_batch_logits", original_eval)
    probe = ttt.r4_probe(model, task, data, 10., 10., arm="e2e1emb")
    assert probe["lin_kv_max_diff"] < 1e-4 and probe["lin_kv_acc_diff"] <= .002
    with torch.no_grad():
        zero_delta = torch.zeros(idx.shape[0], task.chunk, h.shape[-1])
        terms = kag._linear_terms(model, h, data[0][2], zero_delta, model.rho.new_tensor(0.),
                                  torch.ones(len(idx)), task.chunk, idx)
        positions = data[0][2]["query_pos"]
        assert torch.allclose(model.read(terms["o0"]),
                              kag._gather(model.read(manual), positions), atol=1e-6)
    checkpoint = tmp_path / "embed.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = ttt.build_model(task, st, dev, 2, "e2e1emb")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(restored.frozen_emb, model.frozen_emb)
    assert torch.allclose(ttt.eval_batch_logits(restored, task, "e2e1emb", idx, y, 0., 10.,
                                                 task.chunk, ttt.GradStats()), logits)
    assert not any(n == "frozen_emb" for n, _ in model.named_parameters())


def test_permuted_query_embedding_only_after_q():
    st, task = small_echo()
    dev = torch.device("cpu")
    model = ttt.build_model(task, st, dev, 7, "e2e1embP")
    pi = model.key_permutation
    assert sorted(pi.tolist()) == list(range(2, 2 + task.n_keys))
    assert all(pi[k] != k + 2 for k in range(task.n_keys))
    idx = torch.tensor([[0, 2, 1, 2, 2, 1, 3]])
    h = model.backbone(idx)
    x = model.fast_input(h, idx)
    emb_ids = idx.clone()
    emb_ids[0, 3] = pi[0]
    emb_ids[0, 6] = pi[1]
    expected = (model.head.ln(h) + model.frozen_emb[emb_ids]) / math.sqrt(2)
    assert torch.allclose(x, expected)


def test_flags_episodes_and_ties():
    def rows(acc):
        return [[s, 0., 0., a] for s, a in acc]
    probes = [dict(step=s, addr_a=a) for s, a in
              [(25, .95), (50, .75), (75, .49), (100, .29), (125, .8), (150, .9)]]
    q = rows([(25, .90), (50, .70), (75, .30), (100, .95), (125, .40), (150, .90)])
    result = ttt.r4_flags(q, probes, .49, .30)
    assert result["bound_ever"] and result["collapse"] and not result["degraded"]
    assert result["first_bound_step"] == 25 and result["probe_at_first_bound"] == probes[0]
    assert result["episodes"] == [
        dict(t_bound_last=25, t_acc_drop=75, t_addr_drop=75, min_acc=.30, recovered=True),
        dict(t_bound_last=100, t_acc_drop=125, t_addr_drop=100, min_acc=.40, recovered=True)]
    never = ttt.r4_flags(rows([(25, .40), (50, .40)]), probes[:2], .50, .30)
    assert never["partial"] and not never["bound_ever"] and not never["collapse"]
    assert never["t_addr_lost"] is None
    assert ttt.r4_flags(rows([(25, .30), (50, .20)]),
                        [dict(step=25, addr_a=.30)], .20, .30)["t_addr_lost"] == 25
    assert ttt.r4_flags(rows([(25, .90), (50, .50)]), probes[:2], .50, .30)["degraded"]


def test_r42_config_order():
    root = Path(__file__).resolve().parents[1]
    with (root / "configs/toy/r4-2.toml").open("rb") as f:
        r42 = tomllib.load(f)["ttt"]
    with (root / "configs/toy/smoke/r4-2.toml").open("rb") as f:
        smoke = tomllib.load(f)["ttt"]
    with (root / "configs/toy/smoke/r4-3.toml").open("rb") as f:
        r43_smoke = tomllib.load(f)["ttt"]
    arms = ("e2e1emb", "e2e1emb_kvm", "e2e1emb_rnd", "e2e1_kvm", "e2e1embP")
    assert [(c["seed"], c["arm"]) for c in r42["cells"]] == [
        (seed, arm) for seed in range(10) for arm in arms]
    assert len(r42["cells"]) == 50 and r42["workers"] == 3
    assert [(c["seed"], c["arm"]) for c in smoke["cells"]] == [
        (seed, arm) for seed in range(2) for arm in arms]
    assert all(not c["diag"] for c in r42["cells"] + smoke["cells"])
    r43 = ("e2e2", "e2e2_kvm", "e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb", "e2e2emb")
    assert [(c["seed"], c["arm"]) for c in r43_smoke["cells"]] == [
        (seed, arm) for seed in range(2) for arm in r43]
    assert all(not c["diag"] for c in r43_smoke["cells"])


def test_two_spawned_cpu_workers_emit_two_cell_results(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    cfg = dict(process_name="TTM-r41-20-test", device="cpu", workers=2,
               max_minutes=5, d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
               fast_mult=2, window=8, chunk=8, T=16, batch=2, n_eval=2,
               n_quick_eval=2, eval_batch=2, cells=[
                   dict(task="icl_mc", arm="e2e1", seed=i, eta_base=1., steps=1,
                        diag=False) for i in range(2)])
    out = ttt.run_cells(cfg, ttt._settings(cfg), torch.device("cpu"))
    assert len(out["cells"]) == 2
    assert all(c["device"] == "cpu" and "eval" in c for c in out["cells"].values())
    # Spawned stdout bypasses pytest's in-process capture on some platforms;
    # records returned through the queue are the authoritative duplicate count.
    captured = capfd.readouterr().out
    if "CELL_RESULT" in captured:
        assert captured.count("CELL_RESULT ") == 2


def test_step_checkpoints_and_final_probe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st, task = small_echo()
    st.checkpoint_steps = [1, 3]
    st.checkpoint_steps_so = [2]
    st.probe_every = 2
    st.quick_every = 2
    model = ttt.build_model(task, st, torch.device("cpu"), 0, "e2e2emb")
    result = ttt.train_arm(model, task, "e2e2emb", 1., st, torch.device("cpu"),
                           np.random.default_rng(8), None, "echo/s0/e2e2emb", steps=3)
    assert [Path(p).name for p in result["step_checkpoints"]] == [
        f"echo__s0__e2e2emb__step{i}.pt" for i in (1, 2, 3)]
    assert [p["step"] for p in result["probes"]] == [2, 3]
    for path in result["step_checkpoints"]:
        state = torch.load(path, weights_only=True)
        assert torch.equal(state["frozen_emb"], model.frozen_emb)
    assert result["probes"][-1]["lin_kv_max_diff"] < 1e-4


def test_worker_failure_does_not_stop_other_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = dict(process_name="TTM-r41-21-test", device="cpu", workers=2,
               max_minutes=5, d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
               fast_mult=2, window=8, chunk=8, T=16, n_keys=4, batch=2,
               n_eval=2, n_quick_eval=2, eval_batch=2, cells=[
                   dict(task="echo", arm="e2e1", seed=0, eta_base=1., steps=1,
                        diag=False),
                   *[dict(task="icl_mc", arm="e2e1", seed=i, eta_base=1., steps=1,
                          diag=False) for i in range(2)]])
    out = ttt.run_cells(cfg, ttt._settings(cfg), torch.device("cpu"))["cells"]
    assert len(out) == 3
    assert "too short" in out["echo/s0/e2e1/eta1/st1"]["error"]
    assert all("eval" in r for k, r in out.items() if k.startswith("icl_mc"))
