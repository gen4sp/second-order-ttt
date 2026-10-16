"""CPU legacy-seed tests of every TRG implementation fixture; no fresh training."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import analyze_trg as a
from tttmem import dense_log as d
from tttmem import episodes as e
from tttmem import exp_chr as c
from tttmem import exp_trg as g
from tttmem import exp_ttt as t
from tttmem import trg_jobs as jobs
from tttmem import trg_triggers as tr

torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-trg-77-tests"))

TINY = dict(d_model=16, n_layers=4, n_heads=2, backbone_mult=2, fast_mult=2, window=16,
            chunk=16, T=128, n_keys=4, n_quick_eval=2, eval_batch=2, n_probe=2, batch=1,
            steps=100, lr_total=3000, eta_lr=0., device="cpu", tf32=False, workers=1)


def setup(**extra):
    st = t._settings(dict(TINY, **extra))
    task = t._task(st, "echo")
    m = t.build_model(task, st, torch.device("cpu"), 0, "e2e1emb")
    return st, task, m, t.build_optim(m, st)


def quick(task, st):
    return t.eval_batches(task, *t.fixed_set(task, st, "quick"), torch.device("cpu"), st.eval_batch)


def test_nonmutation_20_updates(tmp_path):
    final = []
    for observers in (False, True):
        st, task, model, opt = setup()
        rng = np.random.default_rng([0, t.TRAIN_SEED])
        reader = quick(task, st)
        logger = d.DenseLogger(tmp_path/str(observers), emit=False) if observers else None
        telemetry = d.Telemetry(model)
        for step in range(1, 21):
            g.train_step(model, opt, task, st, "e2e1emb", 0, step, rng,
                         telemetry=observers, telemetry_instance=telemetry)
            if observers:
                logger.append(step, d.ladder_log(model, reader, 40, st.g_clip, st.chunk))
        if logger:
            logger.close()
        final.append((copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), rng.bit_generator.state))
    assert c._deep_equal(final[0], final[1])


def test_saved_state_replay_matches_continuous(tmp_path):
    st, task, m, opt = setup()
    rng = np.random.default_rng([0, t.TRAIN_SEED])
    for s in range(1, 11):
        g.train_step(m, opt, task, st, "e2e1emb", 0, s, rng, telemetry=False)
    path = tmp_path/"fork.pt"
    g.save_state(path, m, opt, rng, 10, meta=dict(task=task, batch=st.batch))
    for s in range(11, 21):
        g.train_step(m, opt, task, st, "e2e1emb", 0, s, rng, telemetry=False)
    _, _, replica, ro = setup()
    rr, start, info = g.load_fork(path, replica, ro, task, st.batch)
    assert start == 10 and not info["legacy_state"]
    for s in range(11, 21):
        g.train_step(replica, ro, task, st, "e2e1emb", 0, s, rr, telemetry=False)
    assert c._deep_equal(m.state_dict(), replica.state_dict())
    assert c._deep_equal(opt.state_dict(), ro.state_dict())
    payload = torch.load(path,weights_only=False)
    payload["next_batch_sha1"] = "wrong"
    torch.save(payload,path)
    with pytest.raises(ValueError,match="state next-batch SHA1"):
        g.load_fork(path,replica,ro,task,st.batch)
    torch.save(m.state_dict(), tmp_path/"weights.pt")
    with pytest.raises(ValueError, match="full model"):
        g.load_fork(tmp_path/"weights.pt", replica, ro, task, st.batch)


def test_ladder_per_event_and_L0_L4_identities():
    st, task, m, _ = setup()
    data = quick(task, st)
    read = d.ladder_log(m, data, 40, st.g_clip, st.chunk)
    ev = read["events"]
    assert np.max(np.abs(ev["terms"].sum(-1)-ev["margin"])) < 1e-5
    actual = d.light_read(m, data, 40, st.g_clip, st.chunk)
    np.testing.assert_array_equal(ev["margin"], actual["events"]["margin"])
    idx, y, meta = data[0]
    with torch.no_grad():
        h = m.backbone(idx)
        qp = torch.as_tensor(meta["query_pos"])
        hq = d.gather(h, qp)
        xq = d.gather(m.fast_input(h, idx), qp)
        slow = d.margin(m.read(t.fast_block(hq, m.head.ln, m.per_sequence(len(idx)), xq)), y.gather(1, qp))
    np.testing.assert_allclose(ev["rungs"][..., 0], slow.numpy(), atol=1e-5)


@pytest.mark.parametrize("clip", [10., .001])
def test_descent_sign_closed_norms_own_removal_against_autograd(clip):
    st, task, model, _ = setup(g_clip=clip)
    idx, y, meta = quick(task, st)[0]
    wp, qp = torch.as_tensor(meta["write_pos"]), torch.as_tensor(meta["query_pos"])
    logits, states, tensors = c.chr_stream(model, idx, y, meta["write_pos"], 40, clip, st.chunk,
                                          c.CE, capture=(0, 1))
    h, x, W0 = tensors["h"], tensors["x"], states[0]
    leaves = [w.detach().requires_grad_(True) for w in W0]
    ce = t._ce_tokens(model, t.fast_block(h[:, :st.chunk], model.head.ln, leaves, x[:, :st.chunk]), y[:, :st.chunk])
    ga = list(torch.autograd.grad(ce.mean(1).sum(), leaves, retain_graph=True))
    mk = torch.zeros_like(ce).scatter_(1, wp, 1.)
    gkv = list(torch.autograd.grad((ce*mk).mean(1).sum(), leaves, retain_graph=True))
    clipped, norm, _ = t._clip_fast(ga, clip)
    scale = (clip/(norm+1e-12)).clamp(max=1.)
    etas = t.regime_etas(model, 40)
    L1 = [w-et*c._bcast(scale, grad)*grad for w, et, grad in zip(W0, etas, gkv)]
    reconstructed = [l-et*c._bcast(scale, grad)*(grad-kv)
                     for l, et, grad, kv in zip(L1, etas, ga, gkv)]
    assert max(float((aa-bb).detach().abs().max()) for aa, bb in zip(reconstructed, states[1])) < 1e-5
    f = d.event_write_factors(model, d.gather(h, wp), d.gather(x, wp), y.gather(1, wp), W0, st.chunk)
    hq, xq = d.gather(h, qp), d.gather(x, qp)
    actual = d.ladder_log(model, [(idx,y,meta)], 40, clip, st.chunk)["events"]["rungs"]
    targets = y.gather(1, qp)
    expected_l1 = d.margin(model.read(t.fast_block(hq, model.head.ln, L1, xq)), targets)
    expected_l2 = d.margin(model.read(t.fast_block(hq, model.head.ln, states[1], xq)), targets)
    np.testing.assert_allclose(actual[...,1], expected_l1.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(actual[...,3], expected_l2.detach().numpy(), atol=1e-5)
    removed = d.own_removed_read(model, hq, xq, L1, f, etas, scale, st.chunk)
    for ev in range(task.n_keys):
        selection = torch.zeros_like(ce).scatter_(1, wp[:, ev:ev+1], 1.)
        ge = list(torch.autograd.grad((ce*selection).mean(1).sum(), leaves, retain_graph=True))
        np.testing.assert_allclose(f["norm"][:, ev].numpy(), c._joint_norm(ge).detach().numpy(), atol=1e-5)
        Wo = [w-et*c._bcast(scale, grad)*(grad-own)
              for w, et, grad, own in zip(W0, etas, gkv, ge)]
        expected = t.fast_block(hq[:, ev:ev+1], model.head.ln, Wo, xq[:, ev:ev+1])
        assert float((expected[:, 0]-removed[:, ev]).detach().abs().max()) < 1e-5




@pytest.mark.parametrize("mask", ["kv", "rnd"])
def test_mask_bitwise_count_and_denominator(mask):
    assert c.parse_rule({"rule":"ce", "outer":mask},1000.).outer == mask
    st, task, model, _ = setup(T=512, chunk=64, window=64, n_keys=16)
    idx, y, meta = task.tensors(*task.raw(np.random.default_rng(0), 2), torch.device("cpu"))
    keep = t.outer_keep(meta, tuple(y.shape), mask, np.random.default_rng([0, 4242, 10]), idx.device)
    assert (1-keep).sum(1).tolist() == [16, 16]
    reference = t.arm_loss(model, "e2e1emb", idx, y, 40, st.g_clip, st.chunk, t.GradStats(), keep=keep, meta=meta)
    mine = c.chr_arm_loss(model, "e2e1emb", idx, y, 40, st.g_clip, st.chunk, t.GradStats(), meta,
                          c.Rule(outer=mask), mask_rng=np.random.default_rng([0, 4242, 10]))
    assert torch.equal(reference, mine)
    losses = []
    def obs(chunk, ce, h, W, model, x, y, keep):
        losses.append((ce*keep).sum(1).mean()/st.chunk)
    got = t.arm_loss(model, "e2e1emb", idx, y, 40, st.g_clip, st.chunk, t.GradStats(), keep=keep, meta=meta, observer=obs)
    assert torch.equal(got, sum(losses)/(task.T/st.chunk))


def test_mask_miscount_is_code_fault():
    st, task, model, opt = setup()
    original = task.raw
    def bad(rng, n):
        raw, meta = original(rng, n)
        meta["write_pos"][:, 1] = meta["write_pos"][:, 0]
        return raw, meta
    task.raw = bad
    with pytest.raises(ValueError, match="miscount"):
        g.train_step(model, opt, task, st, "e2e1emb", 0, 1, np.random.default_rng(0), outer="kv")
    assert e.classify_branch([5], [1.], start=0, end=5, code_fault=True) == e.TRUNCATED


def test_skip_pre_step_moments_counter_and_lr_unchanged():
    st, task, model, opt = setup()
    rng = np.random.default_rng([0, t.TRAIN_SEED])
    for step in range(1, 11):
        g.train_step(model, opt, task, st, "e2e1emb", 0, step, rng, telemetry=False)
    original_m, original_o = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    controller = tr.SpikeController(10, 3)
    for step in range(11, 21):
        controller.observe(step, 1e-20)
    row = g.train_step(model, opt, task, st, "e2e1emb", 0, 21, rng, controller=controller, policy="skip")
    assert row["flag"] and row["dropped"]
    assert c._deep_equal(original_m, model.state_dict())
    assert c._deep_equal(original_o, opt.state_dict())
    assert row["lr_before"] == row["lr"]


























def test_dense_shard_hash_and_exact_grid(tmp_path):
    st, task, m, _ = setup()
    read = d.ladder_log(m, quick(task,st),40,st.g_clip,st.chunk)
    log = d.DenseLogger(tmp_path,emit=False,shard_steps=2)
    log.append(10,read)
    with pytest.raises(ValueError, match="gap"):
        log.append(12,read)
    log.append(11,read)
    log.close()
    _, s, ladder = d.validate_shards(tmp_path)
    assert s == ladder == [10,11]
    with (tmp_path/"dense-000010-000011.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        d.validate_shards(tmp_path)




def test_fresh_seed_training_guard(tmp_path):
    with pytest.raises(ValueError, match="legacy seeds"):
        g.run(dict(TINY,output_dir=str(tmp_path),cells=[dict(seed=200,family="FO",arm="e2e1emb")]),0)


def test_resume_keeps_retry_exhausted_missing(tmp_path,monkeypatch):
    cfg = dict(TINY,process_name=os.environ.get("PROCESS_NAME", "TTM-trg-77-tests"),output_dir=str(tmp_path),resume=True)
    cell = dict(family="FO",seed=0,arm="e2e1emb")
    marker = tmp_path/"FO-s0"/"cell-main.json"
    saved = dict(cell=cell,phase="main",branches=[dict(disposition="TRUNCATED",attempts=2,t_acq=10)],
                 config_sha256=g.digest_json(cfg),source_sha256=g.source_hashes())
    g.write_json(marker,saved)
    def forbidden(*args,**kwargs):
        raise AssertionError("must not give an exhausted branch another retry")
    monkeypatch.setattr(g,"run_segment",forbidden)
    assert g._cell_worker((cfg,cell,"main",5)) == saved


def test_full_driver_and_analyzer_smoke(tmp_path):
    cfg = dict(TINY,process_name=os.environ.get("PROCESS_NAME", "TTM-trg-77-tests"),threads_per_worker=2,smoke_only=True,
               smoke_steps=15,smoke_fork=5,so_end=15,ladder_cap=10,output_dir=str(tmp_path),
               emit_dense=False,probes=True,resume=False,
               cells=[dict(family="FO",seed=0,arm="e2e1emb",rerun=True),
                      dict(family="SO",seed=0,arm="e2e2emb",rerun=True)])
    result = g.run(cfg,0)
    assert result["dispositions"]["TRUNCATED"] == 0, json.loads((tmp_path/"branches.json").read_text())
    spec = importlib.util.spec_from_file_location("analyze_trg_script",g.ROOT/"scripts/analyze_trg.py")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    scores = analyzer.analyze_folder(tmp_path,smoke_only=True)
    assert scores["contract"] == "PASS" and result["branches"] == 6
    # Download portability: stale absolute branch paths are never opened.
    moved = tmp_path.parent/(tmp_path.name+"-downloaded")
    tmp_path.rename(moved)
    scores = analyzer.analyze_folder(moved,smoke_only=True)
    assert scores["contract"] == "PASS"
