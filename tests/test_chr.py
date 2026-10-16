"""Tiny single-process CPU tests for CHR; never launch a run or a worker."""
import copy
import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_chr as c
from tttmem import exp_ttt as t

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
t._mark_process(os.environ.get("PROCESS_NAME", "TTM-chr-91-tests"))

TINY = dict(d_model=16, n_layers=2, n_heads=2, backbone_mult=2, fast_mult=2, window=8, chunk=16,
            n_keys=4, T=96, n_eval=4, n_quick_eval=4, eval_batch=2, n_probe=2, batch=2, steps=12,
            lr_total=3000, eta_lr=0.0, workers=1, device="cpu", tf32=False,
            log_every=2, probe_every=4, cross_every=4, chunk_early=2, chunk_late=3,
            fork_arms=["e2e1emb"], fork_acc=0.0, rp_cap=1000.0,
            forks=[{"rule": "ce"}, {"rule": "rp", "r_min": 0.99}, {"rule": "nc", "r_min": 0.99}],
            cross_rules=["rp50"], final_rules=["rp50"], ckpt_steps=[4], branch_ckpt_every=4,
            branch_ckpt_tags=["rp99", "nc99"], mediation_p=0.5)


def tiny(**extra):
    cfg = dict(TINY)
    cfg.update(extra)
    return cfg, t._settings(cfg), t._task(t._settings(cfg), "echo")


def batch(task, seed=3, n=2):
    return task.tensors(*task.raw(np.random.default_rng(seed), n), torch.device("cpu"))


def test_sanity_all_pass():
    out = c.chr_sanity()
    assert all(v["passed"] for v in out.values()), out


def test_train_segment_ce_bit_identical_to_train_arm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg, st, task = tiny(fork_arms=[], ckpt_steps=[])
    dev = torch.device("cpu")
    ref = t.build_model(task, st, dev, 1, "e2e1emb")
    st_ref = t._settings({**cfg, "quick_every": 10**6, "probe_every": 0})
    t.train_arm(ref, task, "e2e1emb", 40.0, st_ref, dev, np.random.default_rng([1, t.TRAIN_SEED]),
                None, "ref", task.steps, seed=1)
    cs = c.chr_settings(cfg)
    quick = t.eval_batches(task, *t.fixed_set(task, st, "quick"), dev, st.eval_batch)
    probe = t.eval_batches(task, *task.raw(np.random.default_rng([t.PROBE_SEED, 0]), st.n_probe),
                           dev, st.eval_batch)
    ev_tokens, ev_meta = t.fixed_set(task, st, "eval")
    ev = t.eval_batches(task, ev_tokens, ev_meta, dev, st.eval_batch)
    ctx = c.Ctx(task, "e2e1emb", 40.0, st, cs, dev, 1, quick, probe, None, ev, ev_meta, "x")
    mine = t.build_model(task, st, dev, 1, "e2e1emb")
    res = c.train_segment(mine, t.build_optim(mine, st), np.random.default_rng([1, t.TRAIN_SEED]),
                          ctx, c.CE, "main", 0, allow_fork=False)
    assert res["logs"] and res["logs"][-1]["step"] == task.steps
    a, b = ref.state_dict(), mine.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("arm", ["e2e1emb", "e2e1emb_cn97"])
def test_logging_does_not_change_trajectory(tmp_path, monkeypatch, arm):
    monkeypatch.chdir(tmp_path)
    states = []
    for every in (2, 1000):
        cfg, st, task = tiny(fork_arms=[], ckpt_steps=[], log_every=every, probe_every=every * 2,
                             cross_every=every * 2)
        cs = c.chr_settings(cfg)
        dev = torch.device("cpu")
        quick = t.eval_batches(task, *t.fixed_set(task, st, "quick"), dev, st.eval_batch)
        probe = t.eval_batches(task, *task.raw(np.random.default_rng([t.PROBE_SEED, 0]), st.n_probe),
                               dev, st.eval_batch)
        ctx = c.Ctx(task, arm, 40.0, st, cs, dev, 2, quick, probe, None, [], {}, "x")
        m = t.build_model(task, st, dev, 2, arm)
        c.train_segment(m, t.build_optim(m, st), np.random.default_rng([2, t.TRAIN_SEED]), ctx,
                        c.CE, "main", 0, allow_fork=False)
        states.append(m.state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_cell_with_forks_twin_reproduces_main_and_identities(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg, st, task = tiny()
    rec = c.run_chr_cell(cfg, dict(task="echo", seed=0, arm="e2e1emb", eta_base=40.0),
                         torch.device("cpu"))
    assert rec["forked"] and len(rec["forks"]) == 3
    for b in rec["forks"]:
        assert "error" not in b, b
        assert b["state_identical"] and b["batch_identical"] and b["main_batch_identical"]
        assert b["fork_step"] == 2
    main = torch.load(next(Path("checkpoints").glob("*__main__step12.pt")))
    twin = torch.load(next(Path("checkpoints").glob("*__fork-ce-twin__step12.pt")))
    rp = torch.load(next(Path("checkpoints").glob("*__fork-rp99__step12.pt")))
    assert all(torch.equal(main[k], twin[k]) for k in main)
    assert not all(torch.equal(main[k], rp[k]) for k in main)
    out = capsys.readouterr().out
    lines = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("CHR_LOG ")]
    branches = {r["branch"] for r in lines}
    assert branches == {"main", "fork-ce-twin", "fork-rp99", "fork-nc99"}
    rp_rows = [r for r in lines if r["branch"] == "fork-rp99"]
    assert all(r["write"]["added_req_med"] > 0 for r in rp_rows)
    assert all(r["write"]["added_actual_med"] == r["write"]["added_req_med"] for r in rp_rows)
    assert all("train_added_req_med" in r and r["train_added_actual_sum"] > 0 for r in rp_rows)
    # forked at step 2: branch logs start after the fork step and reach the last step
    assert min(r["step"] for r in rp_rows) == 4 and max(r["step"] for r in rp_rows) == 12
    fork = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("CHR_FORK ")]
    assert len(fork) == 1 and set(fork[0]["write_at_fork"]) == {"ce", "rp99", "nc99"}
    w = fork[0]["write_at_fork"]
    assert w["rp99"]["kv_write_med"] > w["ce"]["kv_write_med"]
    assert w["nc99"]["match_fail"] == 0.0 and "clip_scale_ratio_med" in w["rp99"] and "cap_hit" in w["rp99"]
    assert set(fork[0]["quick_at_fork"]) == {"ce", "rp99", "nc99"}
    assert all("write_ce" in r for r in rp_rows if "cross" in r)
    assert all(r["train_n"] > 0 and "train_clip_count" in r for r in rp_rows)
    # the emitted *applied* NC norm matches what RP adds (not merely the requested copy)
    assert abs(w["nc99"]["added_actual_med"] - w["rp99"]["added_actual_med"]) <= 1e-5 * w["rp99"]["added_actual_med"]
    assert sorted(p.name.split("__")[-2] for p in Path("checkpoints").glob("*__fork-*__step6.pt")) == \
        ["fork-nc99", "fork-rp99"]  # periodic branch checkpoints only for branch_ckpt_tags
    med = rec["mediation"]
    assert set(med) >= {"NAT", "CLAMP", "FLOOR"} and med["n_corrupted"] > 0 and med["n_clean"] > 0
    assert med["NAT"]["w_corr_med"] == 1.0 and med["FLOOR"]["w_corr_med"] == 1.0
    assert med["CLAMP"]["w_clean_med"] == 1.0
    fin = rec["final"]
    assert set(fin) == {"ce", "rp50"} and fin["ce"]["decomp_gate_ok"]
    iv = fin["ce"]["intervention"]
    assert set(iv) >= {"NAT", "BB", "SLOW", "HEAD", "RND_BB", "RND_SLOW", "RND_HEAD", "SHAM5_BB",
                       "SHAM5_SLOW", "SHAM5_HEAD", "alpha_head_median"}
    assert fin["ce"]["decomp_pred_agree"] == 1.0
    assert abs(iv["NAT"]["all"] - fin["ce"]["acc"]) < 1e-9


def test_rule_stream_identities():
    cfg, st, task = tiny()
    model = t.build_model(task, st, torch.device("cpu"), 4, "e2e1emb")
    idx, y, meta = batch(task)
    ref, _, _ = c.chr_stream(model, idx, y, meta["write_pos"], 40.0, st.g_clip, task.chunk, c.CE)
    for rule in (c.Rule("rp", 0.0), c.Rule("nc", 0.0)):
        got, _, _ = c.chr_stream(model, idx, y, meta["write_pos"], 40.0, st.g_clip, task.chunk, rule)
        assert torch.equal(ref, got)
    got, _, _ = c.chr_stream(model, idx, y, meta["write_pos"], 40.0, st.g_clip, task.chunk,
                             c.Rule("rp", 0.999))
    assert not torch.equal(ref, got)
    # chunk 0 is read before any write, so only later chunks can change
    assert torch.equal(ref[:, :task.chunk], got[:, :task.chunk])


def test_rule_weights_floor_cap_and_residual_cap():
    ce = torch.tensor([[-np.log(1 - 1e-4), -np.log(1 - .2), -np.log(1 - 1e-9)]], dtype=torch.float32)
    r, w = c.rule_weights(ce, c.Rule("rp", 0.03, 1000.0))
    assert torch.allclose(r, torch.tensor([[1e-4, .2, 1e-9]]), rtol=1e-3)
    assert torch.allclose(w, torch.tensor([[300.0, 1.0, 1000.0]]), rtol=1e-3)
    _, w = c.rule_weights(ce, c.Rule("rs", 0.01))
    assert torch.allclose(w, torch.tensor([[1.0, .05, 1.0]]), rtol=1e-3)
    _, w = c.rule_weights(ce, c.Rule("ns", 0.01))
    assert torch.allclose(w, torch.tensor([[1.0, .05, 1.0]]), rtol=1e-3)


def test_cn97_cell_forks_rs_ns(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg, st, task = tiny(fork_arms=["e2e1emb_cn97"], branch_ckpt_tags=[], mediation_p=0.0,
                         forks={"e2e1emb_cn97": [{"rule": "rs", "r_cap": 0.5}, {"rule": "ns", "r_cap": 0.5}]},
                         final_rules={"e2e1emb_cn97": ["rs50"]}, cross_rules={"e2e1emb_cn97": ["rs50"]})
    rec = c.run_chr_cell(cfg, dict(task="echo", seed=0, arm="e2e1emb_cn97", eta_base=40.0),
                         torch.device("cpu"))
    assert rec["forked"] and [b["branch"] for b in rec["forks"]] == ["fork-rs50", "fork-ns50"]
    for b in rec["forks"]:
        assert "error" not in b, b
        assert b["state_identical"] and b["batch_identical"] and b["main_batch_identical"]
    assert rec["mediation"] is None and set(rec["final"]) == {"ce", "rs50"}
    out = capsys.readouterr().out
    fork = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("CHR_FORK ")][0]
    assert set(fork["write_at_fork"]) == set(fork["write_at_fork_corrupted"]) == {"ce", "rs50", "ns50"}
    w = fork["write_at_fork"]
    assert w["rs50"]["kv_write_med"] < w["ce"]["kv_write_med"]
    assert "cap_bound" in w["ns50"]
    if w["ns50"]["cap_bound"] == 0.0:  # unbound: the applied removal matches RS's
        assert abs(w["ns50"]["added_actual_med"] - w["rs50"]["added_actual_med"]) <= 1e-5 * w["rs50"]["added_actual_med"]
    else:  # bound: never more than requested
        assert w["ns50"]["added_actual_med"] <= w["rs50"]["added_actual_med"] * (1 + 1e-6)
    rows = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("CHR_LOG ")]
    ns_rows = [r for r in rows if r["branch"] == "fork-ns50"]
    assert ns_rows and all("train_cap_bound" in r for r in ns_rows)
    main_cross = [r for r in rows if r["branch"] == "main" and "cross" in r]
    assert main_cross and all(set(r["cross"]) == {"rs50"} for r in main_cross)


def test_corrupt_events_matches_corrupt_echo_channel():
    _, st, task = tiny()
    raw, meta = task.raw(np.random.default_rng(9), 6)
    a, ma = t.corrupt_echo(raw, meta, "e2e1emb_cn97", 11, 7)
    b, mb = c.corrupt_events(raw, meta, 1 - t.COPY_Q["e2e1emb_cn97"], 11, 7)
    assert np.array_equal(a, b) and np.array_equal(ma["corrupted"], mb["corrupted"])
    hi, mh = c.corrupt_events(raw, meta, 0.5, 11, 7)
    f = mh["corrupted"]
    wp = meta["write_pos"]
    rows = np.arange(len(wp))[:, None]
    assert f.any() and (~f).any()
    assert np.array_equal(hi[rows, wp + 1], raw[rows, wp + 1])  # kv labels unchanged
    assert (hi[rows, wp - 1] != raw[rows, wp - 1])[f].all() and (hi[rows, wp - 1] == raw[rows, wp - 1])[~f].all()


def test_mediation_weights():
    r0 = torch.tensor([[.01, .02, .04, .01]])
    # the clean events' residuals in the mixed stream differ from their twin's (context spillover)
    rc = torch.tensor([[.5, .03, .8, .02]])
    f = torch.tensor([[True, False, True, False]])
    w, valid = c.mediation_weights(r0, rc, f, 1000.0)
    assert torch.equal(w["NAT"], torch.ones(1, 4))
    assert torch.allclose(w["CLAMP"], torch.tensor([[.02, 1.0, .05, 1.0]]))
    # torch takes the lower middle value as the median of an even count
    rt = float(torch.tensor([.5, .8]).median())
    # FLOOR divides by the residual in the stream it acts on (achieved drive = rt), not the twin's
    assert torch.allclose(w["FLOOR"], torch.tensor([[1.0, rt / .03, 1.0, rt / .02]]))
    assert torch.allclose((w["FLOOR"] * rc)[~f], torch.full((2,), rt))
    assert not any(bool(v[k].any()) for v in valid.values() for k in v)
    none, _ = c.mediation_weights(r0, rc, torch.zeros(1, 4, dtype=torch.bool), 1000.0)
    assert torch.equal(none["FLOOR"], torch.ones(1, 4))
    # saturated corrupted residual: no attainable ratio -> weight 1, flagged; huge ratio -> capped
    w, valid = c.mediation_weights(torch.tensor([[.1, .5]]), torch.tensor([[1e-9, 1e-5]]),
                                   torch.tensor([[True, True]]), 1000.0)
    assert torch.allclose(w["CLAMP"], torch.tensor([[1.0, 1000.0]]))
    assert valid["CLAMP"]["invalid"].tolist() == [[True, False]]
    assert valid["CLAMP"]["capped"].tolist() == [[False, True]]


def test_position_masks_and_layout_all_writes_in_chunk0():
    _, st, task = tiny()
    _, y, meta = batch(task, n=3)
    m = c.position_masks(meta["write_pos"], y, torch.device("cpu"))
    # 16 kv writes; 15 value->W controls (the last value predicts the first filler)
    assert m["kv"].sum(1).tolist() == [task.n_keys] * 3 and m["vw"].sum(1).tolist() == [task.n_keys - 1] * 3
    assert bool((y[m["vw"] > 0] == c.W_TOKEN).all())
    assert c.write_chunks(meta["write_pos"], task.chunk) == {0}
    full = t._task(t._settings(dict(T=512, chunk=64, window=64, n_layers=4, n_keys=16)), "echo")
    _, fm = full.raw(np.random.default_rng(1), 64)
    assert c.write_chunks(fm["write_pos"], 64) == {0}
    assert set(np.unique(fm["query_pos"] // 64).tolist()) == {5, 6}


def test_intervention_alpha_label_free_and_identity_when_norms_equal():
    torch.manual_seed(0)
    N, K, d = 3, 6, 8
    h = torch.randn(N, K, d)
    chunk = torch.tensor([[5, 5, 5, 6, 6, 6]] * N)
    h[:, 3:] *= 10.0
    ratio = c._per_seq_ratio(c._centred(h).norm(dim=-1), chunk, 5, 6)
    assert torch.all((ratio > 0.05) & (ratio < 0.2))
    same = c._per_seq_ratio(torch.ones(N, K), chunk, 5, 6)
    assert torch.equal(same, torch.ones(N))


def test_config_full_run_matches_plan():
    cfg = tomllib.load(open(ROOT / "configs/toy/chr.toml", "rb"))
    assert cfg["experiment"] == "chr"
    chr_cfg = cfg["chr"]
    cs = c.chr_settings(chr_cfg)
    st = t._settings(chr_cfg)
    assert st.steps == 1500 and st.eta_lr == 0.0 and not st.tf32 and st.T == 512 and st.chunk == 64
    assert [r.tag for r in cs.rules("forks", "e2e1emb")] == ["ce", "rp03", "nc03", "rp10", "nc10"]
    assert [r.tag for r in cs.rules("forks", "e2e1emb_cn97")] == ["ce", "rs01", "ns01"]
    assert cs.rules("forks", "e2e2emb") == [] and cs.fork_arms == ["e2e1emb", "e2e1emb_cn97"]
    assert [r.tag for r in cs.rules("cross_rules", "e2e1emb")] == ["rp03"]
    assert [r.tag for r in cs.rules("cross_rules", "e2e1emb_cn97")] == ["rs01"]
    assert [r.tag for r in cs.rules("final_rules", "e2e1emb")] == ["rp03", "rp10"]
    assert [r.tag for r in cs.rules("final_rules", "e2e1emb_cn97")] == ["rs01", "rp03"]
    assert cs.branch_ckpt_tags == ["rp03", "nc03"] and cs.mediation_p == 0.5
    assert cs.ckpt_dense_arms == ["e2e1emb"] and cs.ckpt_steps_sparse == list(range(100, 1501, 100))
    assert cs.log_every == 5 and cs.fork_acc == 0.90 and (cs.chunk_early, cs.chunk_late) == (5, 6)
    cells = chr_cfg["cells"]
    count = {}
    for cell in cells:
        assert cell["eta_base"] == 40
        count.setdefault(cell["arm"], []).append(cell["seed"])
    assert count == {"e2e1emb": list(range(10)), "e2e2emb": list(range(6)),
                     "e2e1emb_cn97": list(range(6))}
    assert chr_cfg["process_name"] == "TTM-chr-01"


def test_parse_rule_tags():
    assert c.parse_rule("rp03", 1000.0) == c.Rule("rp", 0.03, 1000.0)
    assert c.parse_rule({"rule": "ce"}, 1000.0) == c.CE
    assert c.parse_rule({"rule": "rs", "r_cap": 0.01}, 1000.0).tag == "rs01"
    assert c.parse_rule("ns01", 1000.0) == c.Rule("ns", 0.01, 1000.0)
    with pytest.raises(ValueError):
        c.Rule("xx")
    with pytest.raises(ValueError):
        c.Rule("rs", 0.0)
    with pytest.raises(ValueError):
        c.parse_rule({"rule": "rs", "r_cap": 0.01, "r_min": 0.01}, 1000.0)


def test_nc_without_control_direction_is_flagged():
    torch.manual_seed(1)
    W = [torch.randn(2, 3, requires_grad=True)]
    logits = W[0] @ torch.randn(3, 4)  # (2, 4) as (B, chunk) token "CE" stand-in
    ce = torch.nn.functional.softplus(logits)
    kv = torch.zeros(2, 4)
    kv[:, 0] = 1.0
    zero_ctl = torch.zeros(2, 4)
    orig = c._joint_norm
    c._joint_norm = lambda gs: torch.stack([g.reshape(g.shape[0], -1).norm(dim=1) for g in gs]).pow(2).sum(0).sqrt()
    try:
        extra, info = c.rule_extra(c.Rule("nc", 0.999), ce, W, kv, zero_ctl, 4, True)
    finally:
        c._joint_norm = orig
    assert bool(info["match_fail"].all()) and float(info["actual"].abs().max()) == 0.0
    assert bool((info["added"] > 0).all())
