"""Legacy-seed CPU fixtures for the sealed load protocol; no fresh-seed training."""
import copy
import json
import math
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest
import torch
from tttmem import exp_lod as l, exp_ttt as t, lod_analysis as a


@pytest.fixture(scope="session", autouse=True)
def threads():
    import setproctitle
    setproctitle.setproctitle(os.environ.get("PROCESS_NAME", "TTM-lod-78-tests"))
    torch.set_num_threads(2)


def cfg(**updates):
    c = dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
             window=16, chunk=64, pool=256, gap=268, n_probe=16, events_per_chunk=16,
             control_load=128, steps=2, batch=2, eval_batch=2, n_eval=4,
             n_quick_eval=4, mode="diagnostic", device="cpu", eta_lr=0., lr_total=30)
    c.update(updates)
    return c


def fixture_cell(arm, load, i, acc=.98, group=None, retained=True, **endpoints):
    # Synthetic unit ids are labels, not RNG seeds. Actual torch tests use seed 0.
    group = group or ("aice" if arm == "aice" else "core")
    c = dict(group=group, arm=arm, load=load, seed=f"fixture-{i:02d}", status="COMPLETE",
             composite=dict(A=acc, A_last=acc, A_coupled=acc, A_snap=acc, A_freeze=acc,
                            W0=.06, **endpoints), acquired=25, retained=retained,
             B_nb=.2, logs=[])
    c["cell_id"] = l.cell_id(c)
    return c


def grid(count=8, lin=.98, so=.99, aice=.99):
    return [fixture_cell(arm, n, i, acc) for i in range(count) for n in l.MAIN
            for arm, acc in (("lin", lin), ("so", so), ("aice", aice))]


@pytest.mark.parametrize("n,T,snapshot", [(16,448,1), (32,512,2), (64,640,4), (128,896,8), (256,1408,16)])
def test_layout(n, T, snapshot):
    task = l.make_task(cfg(), dict(load=n, layout="main"))
    raw, m = task.raw(np.random.default_rng(0), 8)
    assert task.T == T and m["snapshot_chunk"] == snapshot
    assert all(l.layout_audit(raw, m, task).values())
    assert m["query_pos"].shape == (8, 16) and m["write_pos"].shape == (8, n)
    assert raw.shape == (8, T + 1)
    assert (T - (m["query_pos"][:, -1] + 1) < 64).all()


@pytest.mark.parametrize("shift, snapshot", [(0, 8), (2, 10)])
def test_L16_shifted(shift, snapshot):
    raw, m = l.echo_load_batch(np.random.default_rng(0), 4, n_pairs=16,
                               layout="L16", shift_chunks=shift)
    task = l.make_task(cfg(), dict(load=16, layout="L16"))
    task.T += 64 * shift
    assert m["snapshot_chunk"] == snapshot
    assert all(l.layout_audit(raw, m, task).values())
    assert not (raw[:, :64 * (7 + shift)] == 0).any()


def test_coupled_history():
    task = l.make_task(cfg(), dict(load=128, layout="last"))
    raw, m = task.raw(np.random.default_rng(0), 4)
    other, cm = l.echo_load_batch(np.random.default_rng(1), 4, n_pairs=128,
                                 layout="coupled", given=(raw, m))
    assert np.array_equal(raw[:, 448:], other[:, 448:])
    assert not (other[:, :448] == 0).any()
    assert cm["snapshot_chunk"] == 8 and cm["write_pos"].shape == (4,16)
    assert all(l.layout_audit(other, cm, task).values())
    assert np.array_equal(m["query_write_pos"], cm["query_write_pos"])
    assert np.array_equal(m["query_tok"], cm["query_tok"])
    with pytest.raises(ValueError, match="last-chunk"):
        l.echo_load_batch(np.random.default_rng(0), 4, n_pairs=128, layout="coupled",
                          given=l.make_task(cfg(), dict(load=128)).raw(np.random.default_rng(0), 4))


def test_binding_free_ties():
    m = dict(value_histogram=np.array([[2,2,0], [1,2,1]]), query_tok=np.array([[10,11], [11,12]]), value0=10)
    assert l.binding_free_baseline(m) == .5


@pytest.mark.parametrize("n", [32, 128])
def test_batched_quick_full_metadata(n):
    c = cfg()
    task = l.make_task(c, dict(load=n))
    raw, m = task.raw(np.random.default_rng(0), 4)
    sliced = [l.slice_meta_load(m, 0, 2), l.slice_meta_load(m, 2, 4)]
    joined = l.concat_meta_load(sliced)
    assert all(np.array_equal(joined[k], m[k]) for k in l.ARRAY_FIELDS)
    model = t.build_model(task, t._settings(c), torch.device("cpu"), 0, l.ARMS["lin"])
    quick = l.evaluate_load(model, task, t._settings(c), l.ARMS["lin"], raw[:2], sliced[0])[0]
    full = l.evaluate_load(model, task, t._settings(c), l.ARMS["lin"], raw, m)[0]
    assert quick["n_probes"] == 32 and full["n_probes"] == 64
    assert set(full["by_write_chunk"]) <= {str(i) for i in range(n//16)}
    assert math.isfinite(full["ce"])


def test_identical_init_and_code_across_loads_arms():
    states = []
    for n, arm in ((16,"lin"), (128,"so"), (256,"aice")):
        task = l.make_task(cfg(), dict(load=n))
        states.append(t.build_model(task, t._settings(cfg()), torch.device("cpu"), 0, l.ARMS[arm]).state_dict())
    assert all(torch.equal(states[0][k], s[k]) for s in states[1:] for k in states[0])


def test_aice_lin_first_gradient_equal_then_filler_differs():
    c, dev = cfg(), torch.device("cpu")
    task = l.make_task(c, dict(load=16))
    raw, m = task.raw(np.random.default_rng(0), 2)
    idx, y, _ = task.tensors(raw, m, dev)
    model = t.build_model(task, t._settings(c), dev, 0, l.ARMS["lin"])
    grads = {}
    for arm in ("lin", "aice"):
        calls = []
        def observe(c, logits, reference, W, raw_g, g, norm, clip):
            calls.append(torch.cat([v.detach().flatten() for v in raw_g]))
        t.eval_sequential(model, idx[:, :128], y[:, :128], model.etas(10), 10, 64, t.GradStats(),
                          rule=t.arm_inner_loss(l.ARMS[arm]), observer=observe)
        grads[arm] = calls
    assert torch.allclose(grads["lin"][0], grads["aice"][0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(grads["lin"][1], grads["aice"][1], atol=1e-6, rtol=1e-5)


def test_freeze_snapshot_no_following_updates():
    c, dev = cfg(), torch.device("cpu")
    task = l.make_task(c, dict(load=16))
    idx, y, _ = task.tensors(*task.raw(np.random.default_rng(0), 2), dev)
    model = t.build_model(task, t._settings(c), dev, 0, l.ARMS["lin"])
    def call(**kwargs):
        return t.eval_sequential(model, idx[:, :128], y[:, :128], model.etas(10), 10, 64,
                                 t.GradStats(), rule="lin", **kwargs)
    assert torch.equal(call(), call(freeze_from=1))
    assert torch.equal(call(), call(snapshot_chunk=1))


def test_write_log_multichunk_geometry():
    c, dev = cfg(), torch.device("cpu")
    task = l.make_task(c, dict(load=32))
    raw, m = task.raw(np.random.default_rng(0), 4)
    model = t.build_model(task, t._settings(c), dev, 0, l.ARMS["lin"])
    log = l.write_log_load(model, task, t._settings(c), l.ARMS["lin"], raw, m, geometry=True)
    assert len([p for p in log["profile"] if p["type"] == "write"]) == 2
    assert {"x", "hidden_w0", "hidden_stream"} == set(log["geometry"])
    assert all(math.isfinite(v) for g in log["geometry"].values() for v in g.values())


@pytest.mark.parametrize("values,label", [([0]*8,"FREE"), ([-.1]*8,"PRICED"), ([-.04,-.1,0,.05,-.2,.01,.02,-.06],"INCONCLUSIVE"), ([0]*5,a.NE)])
def test_ni(values,label):
    assert a.ni(values)["label"] == label


def test_t_quantile_and_CP():
    assert a.t95(5) == pytest.approx(2.01504837, abs=1e-7)
    assert a.cp_bounds(0,12)["upper95"] == pytest.approx(.220922, abs=1e-6)
    assert a.cp_bounds(1,12)["upper95"] == pytest.approx(.338681, abs=1e-6)
    assert a.cp_bounds(0,48)["upper95"] == pytest.approx(.060503, abs=1e-6)
    assert a.cp_bounds(0,0)["upper95"] is None


def test_holm_7_of_8_neither_supported_and_allzero():
    bs = [-.03]*7 + [.03]
    p = a.sign_p(bs, -1)
    assert p == 9/256
    adj = a.holm(dict(H2=p, H3a=p))
    assert adj == dict(H2=18/256, H3a=18/256)
    assert not a.slope_rule(bs, -1, adj["H2"])["label"].startswith("SUPPORTED")
    assert a.slope_rule([0]*8, -1)["raw_p"] == 1
    assert a.slope_rule([0]*8, -1)["label"] == "FALSIFIED"


@pytest.mark.parametrize("bs,p,label", [([-.03]*8,.01,"SUPPORTED-MATERIAL"),
    ([-.001]*8,.01,"NEGLIGIBLE-TREND"), ([-.001]*8,.2,"FALSIFIED"),
    ([-.03]*8,.2,"INCONCLUSIVE"), ([-.03]*5,.01,a.NE),
    ([-.03]*6+[-.002]*2,.01,"SUPPORTED-DIRECTION")])
def test_slope_precedence(bs,p,label):
    assert a.slope_rule(bs,-1,p)["label"] == label


def test_complete_families_and_truncated_AICE():
    cells = grid(8)
    for k in (0, 5, 6, 8):
        sub = [c for c in cells if c["arm"] != "aice" or int(c["seed"].split("-")[-1]) < k]
        result = a.score(sub)
        assert result["H3a"]["n"] == k
        assert (result["H3a"]["label"] == a.NE) == (k < 6)
    incomplete = [c for c in cells if not (c["arm"] == "lin" and c["load"] == 32)]
    assert a.complete_families(a.indexed(incomplete), ("lin","so")) == []
    assert a.score(incomplete)["H2"]["label"] == a.NE


def test_T3_holm_family_singleton():
    result = a.score(grid(8), "T3")
    assert result["H2"]["holm_family"] == ["H2"]


def test_QSO_cohort_specific():
    cells = grid(8)
    # Main lin/SO cohort includes low SO cells; AICE has only the six competent SOs.
    for c in cells:
        if c["arm"] == "so" and int(c["seed"].split("-")[-1]) >= 6:
            c["composite"]["A"] = .1
            c["retained"] = False
    cells = [c for c in cells if c["arm"] != "aice" or int(c["seed"].split("-")[-1]) < 6]
    result = a.score(cells)
    assert result["H1a"]["128"]["label"] == a.NE
    assert result["H3b"]["128"]["Q_SO"] and result["H3b"]["128"]["label"] == "FREE"


def test_Qlin16_mean_gate_9_good_3_chance():
    cells = grid(12,lin=.9)
    for c in cells:
        if c["arm"] == "lin" and int(c["seed"].split("-")[-1]) >= 9:
            c["composite"]["A"] = 1/16
            c["retained"] = False
    result = a.score(cells)
    assert not result["Q_lin"]["16"]
    assert result["H1b"]["128"]["label"] == "VOID"
    assert result["claims"]["deployment"] != "STRENGTHENED"


@pytest.mark.parametrize("kind", ["valid", "layout", "memory", "lin_baseline", "joint", "so_high", "lin_high"])
def test_every_gate_precedence(kind):
    cells = grid(8)
    for c in cells:
        if kind == "memory":
            c["composite"]["W0"] = .8
        if kind in ("lin_baseline", "joint") and c["load"] == 16 and c["arm"] == "lin":
            c["composite"]["A"] = .89
        if kind == "joint" and c["load"] == 16 and c["arm"] == "so":
            c["composite"]["A"] = .89
        if kind == "so_high" and c["load"] == 128 and c["arm"] == "so":
            c["composite"]["A"] = .89
        if kind == "lin_high" and c["load"] == 128 and c["arm"] == "lin":
            c["composite"]["A"] = .89
            # SO is competent and lin remains within the NI margin.
        if kind == "lin_high" and c["load"] == 128 and c["arm"] == "so":
            c["composite"]["A"] = .91
    result = a.score(cells, layout_valid=kind != "layout")
    expected = {"valid":"VALID", "layout":"INVALID", "memory":"INVALID",
        "lin_baseline":"LIN_BASELINE_FAILURE", "joint":"JOINT_BASELINE_FAILURE",
        "so_high":"SO_GATE_FAILURE", "lin_high":"VALID"}[kind]
    assert result["claims"]["precedence"] == expected
    if kind == "valid":
        assert result["claims"]["relative_global"] == "FREE"
        assert result["claims"]["deployment"] == "STRENGTHENED"
        assert result["claims"]["N_star"] == 128
    elif kind in ("layout", "memory", "joint", "so_high"):
        assert result["claims"]["relative_global"] == a.NE
    if kind in ("lin_baseline", "joint"):
        assert result["H1b"]["128"]["label"] == "VOID"
    if kind == "so_high":
        assert result["H1b"]["128"]["label"] == "FREE-like"
    if kind == "lin_high":
        assert result["claims"]["relative_global"] == "FREE"
        assert result["claims"]["deployment"] == "NO CLAIM"


def test_Nstar_undefined_and_counts_attached_FREE():
    result = a.score(grid(8,lin=.8))
    assert result["claims"]["N_star"] is None
    assert result["claims"]["N_star_note"] == "no FREE baseline"
    result = a.score(grid(8))
    assert result["H1a"]["16"]["failures_lin"]["n"] == 8
    assert result["H1a"]["16"]["failures_lin"]["upper95"] > 0


@pytest.mark.parametrize("d,label", [(-.1,"INTERFERENCE-PRICED"), (0,"INTERFERENCE-FREE"), (-.05,"INCONCLUSIVE")])
def test_H1i_labels(d,label):
    assert a.interference_label([d]*8)["label"] == label


@pytest.mark.parametrize("d,label", [([.06,.07,.08,0],"LEARNED-HIGHER"),
    ([.02,-.02,0,.07],"NO-DIFFERENCE-SHOWN"), ([-.1]*4,"UNCERTAIN"), ([0]*3,"UNCERTAIN")])
def test_bridge_labels(d,label):
    assert a.bridge_label(d)["label"] == label


def test_nonfinite_gradient_with_finite_loss_skipped():
    model = torch.nn.Linear(2,2)
    opt = torch.optim.AdamW(model.parameters())
    before = copy.deepcopy(model.state_dict())
    model.weight.register_hook(lambda g: g * float("nan"))
    loss = model(torch.ones(1,2)).sum()
    updated, norm, bad = l.finite_update(model,opt,loss,1.)
    assert not updated and not bad and norm is None
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert not l.certified_divergence(19,2)
    assert l.certified_divergence(20,2)
    assert not l.certified_divergence(20,1)
    assert l.composite_endpoints({"A":.95},True) == {"A":1/16}


def test_eta_gradient_nonfinite_and_clip_norm(monkeypatch):
    task = l.make_task(cfg(),dict(load=16))
    model = t.build_model(task,t._settings(cfg()),torch.device("cpu"),0,l.ARMS["lin"])
    opt = t.build_optim(model,t._settings(cfg(eta_lr=.01)))
    hook = model.rho.register_hook(lambda g: g * float("nan"))
    assert not l.finite_update(model,opt,model.rho.sum(),1.)[0]
    hook.remove()
    monkeypatch.setattr(torch.nn.utils,"clip_grad_norm_",lambda *args: torch.tensor(float("inf")))
    assert not l.finite_update(model,opt,model.rho.sum(),1.)[0]


def test_OOM_once_and_divergence_never_retried(tmp_path):
    c = cfg(output_dir=str(tmp_path))
    cell = dict(group="smoke",arm="lin",load=16,seed=0)
    calls=[]
    def runner(*args):
        calls.append(1)
        if len(calls)==1:
            raise torch.OutOfMemoryError("fixture")
        return dict(status="COMPLETE")
    assert l.execute_with_retry(c,cell,math.inf,runner)["status"] == "RETRY"
    assert len(calls)==1
    assert l.execute_with_retry(c,cell,math.inf,runner)["status"] == "COMPLETE"
    assert len(calls)==2
    c["output_dir"] = str(tmp_path / "diverged")
    calls.clear()
    def diverged(*args):
        calls.append(1)
        return dict(status="DIVERGED",composite={"A":1/16})
    assert l.execute_with_retry(c,cell,math.inf,diverged)["status"] == "DIVERGED"
    assert len(calls)==1
    c["output_dir"] = str(tmp_path / "failed")
    def failed(*args):
        raise torch.OutOfMemoryError("fixture")
    assert l.execute_with_retry(c,cell,math.inf,failed)["status"] == "RETRY"
    assert l.execute_with_retry(c,cell,math.inf,failed)["status"] == "MISSING"


@pytest.mark.parametrize("tier,count", [("T1",160), ("T2",152), ("T3",128), ("T4",128), ("T5",120), ("cheap",40)])
def test_sealed_manifests_order(tier,count):
    cells=l.manifest_cells(tier)
    assert len(cells)==count
    assert len(set(l.cell_id(c) for c in cells))==count
    if tier!="cheap":
        bridge=max(i for i,c in enumerate(cells) if c["group"]=="bridge")
        assert all(i>bridge for i,c in enumerate(cells) if c["arm"]=="aice")
    assert set(c["seed"] for c in cells)<=set(range(216,228))


def smoke_receipt(scale=1):
    cells=[dict(arm=a,load=n,layout=layout,seed=0,ms_step=ms*scale,peak_gb=2.) for a,n,layout,ms in
           (("so",256,"main",160),("lin",256,"main",140),("so",128,"main",80),
            ("aice",128,"main",75),("so",16,"L16",78),("lin",16,"main",32))]
    return dict(cells=cells,workers=5,total_gb=24.)


def test_timing_ladder_pure_no_outcomes_and_none_fits():
    production=tomllib.loads(Path("configs/toy/lod-x.toml").read_text())["lod"]
    receipt=smoke_receipt()
    first=l.timing_admission(receipt,production)
    assert first["tier"]=="T1"
    assert first==l.timing_admission(copy.deepcopy(receipt),production)
    assert l.timing_admission(smoke_receipt(100),production)["tier"] is None
    # Select by projected cost alone, in the exact sealed preference order.
    for budget in (.7,.6,.5,.4):
        c=dict(production,budget_usd=budget)
        result=l.timing_admission(receipt,c)
        assert result["tier"]==next((k for k,v in result["costs_usd"].items() if v<=budget),None)
    receipt["total_gb"]=1.
    assert l.timing_admission(receipt,production)["tier"] is None


@pytest.mark.parametrize("long_peak,cap", [(6.,3), (30.,0)])
def test_main_vram_cap_independent_of_stretch(long_peak,cap):
    production=tomllib.loads(Path("configs/toy/lod-x.toml").read_text())["lod"]
    baseline=l.timing_admission(smoke_receipt(),production)
    receipt=smoke_receipt()
    for row in receipt["cells"]:
        if row["load"]==256:
            row["peak_gb"]=long_peak
    result=l.timing_admission(receipt,production)
    assert result["workers"]==5 and result["long_workers"]==cap
    assert result["tier"]==baseline["tier"]=="T1"
    assert result["costs_usd"]==baseline["costs_usd"]
    if cap==0:
        assert not result["addon"] and result["addon_cost_usd"] is None


def test_retry_allowance_is_aggregate_worker_share():
    production=tomllib.loads(Path("configs/toy/lod-x.toml").read_text())["lod"]
    receipt=smoke_receipt()
    result=l.timing_admission(receipt,production)
    overhead=production["eval_factor"]*production["vocab_factor"]*production["diagnostic_factor"]
    expected=(80*1500*overhead/1000+l.PROCESS_START_SECONDS)/(5*3600)
    assert result["retry_hours"]==pytest.approx(expected)
    assert result["allowance_hours"]==pytest.approx(expected+5/60)
    # LOD-X keeps the five-minute shutdown allowance at its own hourly rate, so
    # only an upper bound is checked; LOD checks a narrow range.
    if production.get("protocol") == "lodx":
        assert result["allowance_hours"] * production["hourly_usd"] <= .06
    else:
        assert .02<=result["allowance_hours"]*production["hourly_usd"]<=.03


def test_queue_projection_includes_tail_without_group_barriers():
    cells=[dict(group="core",load=16,duration=d) for d in (10,1,1)]
    cells += [dict(group="aice",load=16,duration=d) for d in (1,1)]
    assert l.ordered_queue_hours(cells,lambda c:c["duration"],2,1)*3600==10
    resources=dict(main_peak_gb=2.,long_peak_gb=6.,memory_budget_gb=8.)
    active=[dict(load=256),dict(load=16)]
    assert not l.queue_can_submit(dict(load=256),active,5,3,resources)


def test_seed_guard_phase_boundary():
    with pytest.raises(ValueError,match="phase boundary"):
        l.seed_guard(cfg(),[dict(seed=216)])
    l.seed_guard(cfg(),[dict(seed=0)])
    l.seed_guard(cfg(mode="production"),[dict(seed=216),dict(seed=227)])
    with pytest.raises(ValueError,match="phase boundary"):
        l.seed_guard(cfg(mode="production"),[dict(seed=200)])


def test_smoke_explicit_density_probes_all_cell_types():
    c=tomllib.loads(Path("configs/toy/smoke/lod.toml").read_text())["lod"]
    assert c["events_per_chunk"]==4 and c["n_probe"]==4
    assert {x["arm"] for x in l.smoke_jobs(c)}==set(l.ARMS)
    for cell in l.smoke_jobs(c):
        task=l.make_task(c,cell)
        raw,m=task.raw(np.random.default_rng(0),2)
        assert all(l.layout_audit(raw,m,task).values())
        assert m["query_pos"].shape==(2,4)


def test_cheap_three_load_slope_no_interference_reading():
    cells=[c for c in grid(6) if c["arm"]!="aice" and c["load"]!=32]
    result=a.score(cells,"cheap")
    assert result["H2"]["n"]==6
    assert result["H2"]["label"]!="NOT EVALUABLE"
    assert result["H2"]["holm_family"]==["H2"]
    assert result["H1i"]["label"]==a.NE


def test_divergence_is_failed_cell_not_missing():
    cells=grid(8)
    for c in cells:
        if c["arm"]=="lin" and c["seed"]=="fixture-00":
            c["status"]="DIVERGED"
            c["retained"]=False
            c["composite"]={k:1/16 for k in c["composite"]}
    result=a.score(cells)
    assert result["H1a"]["16"]["n"]==8
    assert result["failures"]["lin/16"]["k"]==1


def test_nonfinite_weights_certify():
    assert l.certified_divergence(1,0,bad_weights=True)


def test_Qplat_fixed_windows_and_budget_label():
    cells=grid(8)
    for c in cells:
        c["logs"]=[dict(step=step,quick=dict(acc=.8 if step<=1200 else .9)) for step in (1000,1100,1200,1300,1400,1500)]
    assert all(r["label"]=="budget-limited" for r in a.plateau(cells).values())


def test_layout_invalidates_all_readings():
    result=a.score(grid(),layout_valid=False)
    assert all(result[h]["label"]==a.NE for h in ("H2","H3a","H1d","H1i","H5"))
    assert result["interpretation"].startswith("INVALID")


@pytest.mark.parametrize("eval_error,checkpoint_error", [(False,False),(True,False),(True,True)])
def test_full_driver_certifies_nonfinite_gradient_never_retries(tmp_path,monkeypatch,eval_error,checkpoint_error):
    c=cfg(output_dir=str(tmp_path),steps=25,n_probe=4,events_per_chunk=4,chunk=16,
          gap=68,control_load=32,n_quick_eval=2,n_eval=2,process_name=os.environ.get("PROCESS_NAME","TTM-lod-78")+"-policy")
    original=t.build_model
    def build(*args,**kwargs):
        model=original(*args,**kwargs)
        model.rho.register_hook(lambda g:g*float("nan"))
        return model
    monkeypatch.setattr(t,"build_model",build)
    monkeypatch.setattr(t,"arm_loss",lambda model,*args,**kwargs:model.rho.sum()+1.)
    monkeypatch.setattr(l,"evaluate_load",lambda *args,**kwargs:(dict(acc=.4,ce=1.),np.zeros((2,4),dtype=int)))
    monkeypatch.setattr(l,"write_log_load",lambda *args,**kwargs:dict(r=.5,profile=[]))
    def final(*args,**kwargs):
        if eval_error:
            raise RuntimeError("fixture diagnostic failure after certification")
        return {"A":.4,"W0":.06},{"B_nb":.2,"layout_audit":{"valid":True}}
    monkeypatch.setattr(l,"final_eval",final)
    if checkpoint_error:
        def failed_save(*args,**kwargs):
            raise OSError("fixture checkpoint write failure")
        monkeypatch.setattr(torch,"save",failed_save)
    cell=dict(group="smoke",arm="bridge",load=8,seed=0,layout="main")
    result=l.execute_with_retry(c,cell,math.inf)
    assert result["status"]=="DIVERGED" and result["completed_steps"]==20
    assert result["skipped_steps"]==20
    if eval_error:
        assert result["observed"]=={} and "fixture diagnostic failure" in result["final_eval_error"]
    else:
        assert result["observed"]["A"]==.4
    assert all(v==1/16 for v in result["composite"].values())
    if checkpoint_error:
        assert "fixture checkpoint" in result["post_certification_error"]
    attempts=json.loads((tmp_path/(l.cell_id(cell)+"_attempts.json")).read_text())
    assert attempts["attempts"]==1
    assert l.execute_with_retry(c,cell,math.inf)["status"]=="DIVERGED"
    assert json.loads((tmp_path/(l.cell_id(cell)+"_attempts.json")).read_text())["attempts"]==1


def test_production_contract_pairing_sources_cuda_and_step_checks(monkeypatch):
    cells=grid(6)
    config=cfg()
    source=l.source_hashes()
    admitted=[]
    for c in cells:
        c.update(layout="main",config_sha256=l.sha(config),source_hashes=source,layout_audit={"valid":True},
            completed_steps=2,first_batch_sha256=f"data/{c['seed']}/{c['load']}",
            init_sha256=f"init/{c['seed']}",frozen_code_sha256=f"code/{c['seed']}",device="cpu",tf32=False)
        admitted.append({k:c[k] for k in ("group","arm","load","seed","layout")})
    manifest=dict(config=config,config_sha256=l.sha(config),source_hashes=source,cells=admitted,
                  created_before_production=True,admission=dict(tier="T1",addon=False))
    assert not a.contract_errors(manifest,cells,diagnostic=True)
    broken=copy.deepcopy(cells)
    broken[0]["first_batch_sha256"]="different"
    assert any("pairing" in s for s in a.contract_errors(manifest,broken,diagnostic=True))
    broken=copy.deepcopy(cells)
    broken[0]["completed_steps"]=1
    assert any("incomplete final" in s for s in a.contract_errors(manifest,broken,diagnostic=True))
    broken=copy.deepcopy(cells)
    broken[0]["source_hashes"]={}
    assert any("stale" in s for s in a.contract_errors(manifest,broken,diagnostic=True))
    monkeypatch.setattr(l,"seal_guard",lambda *a:None)
    assert any("CUDA" in s for s in a.contract_errors(manifest,cells,diagnostic=False))


def test_manifest_output_precedes_first_cell_and_absolute_deadline_resume(tmp_path,monkeypatch):
    c=tomllib.loads(Path("configs/toy/smoke/lod.toml").read_text())["lod"]
    c["output_dir"]=str(tmp_path)
    c["process_name"]=os.environ.get("PROCESS_NAME","TTM-lod-78")+"-manifest"
    def dispatch(config,cells,workers,deadline,**kwargs):
        manifest=json.loads((tmp_path/"manifest.json").read_text())
        assert manifest["created_before_production"] and manifest["cells"]==l.smoke_jobs(config)
        return []
    monkeypatch.setattr(l,"dispatch",dispatch)
    result=l.run(c,0)
    assert result["claims"]["relative_global"]=="SMOKE ONLY"


def test_constant_paired_differences_are_exact_zero_slopes():
    result=a.score(grid(8,lin=.98,so=.99,aice=.97))
    for h in ("H2","H3a"):
        assert result[h]["nonzero"]==0 and result[h]["raw_p"]==1
        assert result[h]["slopes"]==[0.]*8


def test_acquisition_retention_CP_for_every_arm_cohort():
    result=a.score(grid(8))
    assert {c["arm"] for c in result["cohort_counts"]}=={"lin","so","aice"}
    for c in result["cohort_counts"]:
        assert c["retained"]["k"]==8 and c["retained"]["n"]==8
        assert c["retained"]["CI95"][0]<1.


def test_finite_update_nonfinite_weight_certifies(monkeypatch):
    model=torch.nn.Linear(2,2)
    opt=torch.optim.AdamW(model.parameters())
    def step():
        with torch.no_grad():
            model.weight[0,0]=float("nan")
    monkeypatch.setattr(opt,"step",step)
    updated,norm,bad=l.finite_update(model,opt,model(torch.ones(1,2)).sum(),1.)
    assert updated and bad and l.certified_divergence(1,0,bad)


def install_fake_pool(monkeypatch, handler=None):
    """Exercise supervisor recovery without spawning torch processes or a GPU."""
    import concurrent.futures as cf
    pools=[]
    class Pool:
        def __init__(self,**kwargs):
            self.options=kwargs
            self.jobs=[]
            self.closed=False
            pools.append(self)
        def submit(self,fn,*args,**kwargs):
            future=cf.Future()
            self.jobs.append((future,args,kwargs))
            try:
                result=handler(self,fn,args,kwargs) if handler else fn(*args,**kwargs)
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            return future
        def shutdown(self,**kwargs):
            self.closed=True
    monkeypatch.setattr(cf,"ProcessPoolExecutor",Pool)
    return pools


def save_fake_complete(config,cell):
    result=dict(**cell,cell_id=l.cell_id(cell),status="COMPLETE",config_sha256=l.sha(config),source_hashes=l.source_hashes())
    l.atomic_json(Path(config["output_dir"])/(l.cell_id(cell)+".json"),result)
    return result


def test_retry_is_resubmitted_to_fresh_worker(tmp_path,monkeypatch):
    c=cfg(device="cuda",output_dir=str(tmp_path))
    cell=dict(group="smoke",arm="lin",load=16,seed=0,layout="main")
    calls=[]
    def runner(config,identity,deadline):
        calls.append((copy.deepcopy(identity),l.read_attempt(config,identity)["attempts"]))
        if len(calls)==1:
            raise torch.OutOfMemoryError("CUDA fixture OOM")
        return save_fake_complete(config,identity)
    monkeypatch.setattr(l,"train_cell",runner)
    pools=install_fake_pool(monkeypatch)
    assert l.dispatch(c,[cell],1,math.inf)[0]["status"]=="COMPLETE"
    assert calls==[(cell,1),(cell,2)]
    assert len(pools)==1 and len(pools[0].jobs)==2
    assert pools[0].options["max_tasks_per_child"]==1
    assert pools[0].options["mp_context"].get_start_method()=="spawn"
    assert pools[0].options["initializer"] is l.init_worker
    assert all(job[2]["attempt_reserved"] for job in pools[0].jobs)
    assert l.read_attempt(c,cell)["attempts"]==2 and len(l.read_attempt(c,cell)["errors"])==1
    l.dispatch(c,[cell],1,math.inf)
    assert len(calls)==2  # Resume reuses COMPLETE without another attempt.


@pytest.mark.parametrize("death_on_submit", [False,True])
def test_worker_crash_rebuilds_pool_once_only(tmp_path,monkeypatch,death_on_submit):
    from concurrent.futures.process import BrokenProcessPool
    import concurrent.futures as cf
    c=cfg(device="cuda",output_dir=str(tmp_path))
    cell=dict(group="smoke",arm="lin",load=16,seed=0,layout="main")
    def handler(pool,fn,args,kwargs):
        if len(pools)==1:
            raise BrokenProcessPool("fixture worker death")
        return save_fake_complete(args[0],args[1])
    pools=install_fake_pool(monkeypatch,handler)
    if death_on_submit:
        original=cf.ProcessPoolExecutor.submit
        def submit(pool,*args,**kwargs):
            if len(pools)==1:
                raise BrokenProcessPool("fixture worker death during submit")
            return original(pool,*args,**kwargs)
        monkeypatch.setattr(cf.ProcessPoolExecutor,"submit",submit)
    result=l.dispatch(c,[cell],1,math.inf)
    assert result[0]["status"]=="COMPLETE" and len(pools)==2
    assert all(p.closed for p in pools)
    assert pools[0].options["initargs"][0] is not pools[1].options["initargs"][0]
    assert l.read_attempt(c,cell)["attempts"]==2
    assert len(l.read_attempt(c,cell)["errors"])==1


def test_cuda_circuit_breaker_stops_and_persists(tmp_path,monkeypatch,capsys):
    c=cfg(device="cuda",output_dir=str(tmp_path))
    cells=[dict(group="smoke",arm=arm,load=16,seed=0,layout="main") for arm in ("lin","so","aice","fo")]
    def poisoned(*args,**kwargs):
        raise RuntimeError("CUDA illegal memory access fixture")
    monkeypatch.setattr(l,"train_cell",poisoned)
    pools=install_fake_pool(monkeypatch)
    results=l.dispatch(c,cells,1,math.inf)
    assert len(pools[0].jobs)==3
    assert all(r["status"]=="MISSING" for r in results)
    assert not l.attempt_path(c,cells[-1]).exists()
    stop=json.loads((tmp_path/"session_infra.json").read_text())
    assert stop["status"]=="INFRA_GPU_ERRORS"
    assert '"status":"INFRA_GPU_ERRORS"' in capsys.readouterr().out
    l.dispatch(c,cells,1,math.inf)
    assert len(pools)==1  # Resume cannot silently clear the circuit breaker.


def test_ordered_queue_submits_aice_before_mandatory_completion(tmp_path,monkeypatch):
    import concurrent.futures as cf
    c=cfg(device="cuda",output_dir=str(tmp_path))
    cells=[dict(group=g,arm="lin",load=16,seed=0,layout="main") for g in ("core","control","anchor","bridge","aice")]
    submitted=[]
    futures=[]
    pools=install_fake_pool(monkeypatch)
    def submit(pool,fn,config,cell,deadline,**kwargs):
        future=cf.Future()
        if cell["group"]=="aice":
            assert submitted==["core","control","anchor","bridge"]
            assert not futures[0][0].done()  # A long mandatory cell still runs.
        submitted.append(cell["group"])
        futures.append((future,cell))
        return future
    def wait(pending,**kwargs):
        future=list(pending)[-1]  # Keep the first mandatory cell running.
        cell=next(cell for f,cell in futures if f is future)
        future.set_result(save_fake_complete(c,cell))
        return {future},set(pending)-{future}
    monkeypatch.setattr(cf.ProcessPoolExecutor,"submit",submit)
    monkeypatch.setattr(cf,"wait",wait)
    result=l.dispatch(c,cells,2,math.inf,long_workers=1)
    assert submitted==[cell["group"] for cell in cells]
    assert all(r["status"]=="COMPLETE" for r in result) and len(pools)==1


def test_pretraining_layout_failure_invalidates_without_retry(tmp_path,monkeypatch):
    c=cfg(output_dir=str(tmp_path),process_name=os.environ.get("PROCESS_NAME","TTM-lod-78")+"-layout")
    cell=dict(group="smoke",arm="lin",load=16,seed=0,layout="main")
    monkeypatch.setattr(l,"layout_audit",lambda *args:dict(valid=False))
    result=l.execute_with_retry(c,cell,math.inf)
    assert result["status"]=="LAYOUT_INVALID" and l.read_attempt(c,cell)["attempts"]==1
    assert l.execute_with_retry(c,cell,math.inf)["status"]=="LAYOUT_INVALID"
    manifest=dict(config=c,config_sha256=l.sha(c),source_hashes=l.source_hashes(),cells=[cell],created_before_production=True)
    assert any("layout audit failed before training" in error for error in a.contract_errors(manifest,[result],diagnostic=True))


def test_joint_baseline_failure_suppresses_h3b_and_bridge16():
    cells=grid(8)
    for cell in cells:
        if cell["load"]==16 and cell["arm"] in ("lin","so") and int(cell["seed"].split("-")[-1])>=6:
            cell["composite"]["A"]=.1
            cell["retained"]=False
    cells=[c for c in cells if c["arm"]!="aice" or int(c["seed"].split("-")[-1])<6]
    cells += [fixture_cell("bridge",16,i,.98,group="bridge") for i in range(4)]
    result=a.score(cells)
    assert result["claims"]["precedence"]=="JOINT_BASELINE_FAILURE"
    assert result["H3b"]["16"]["Q_SO"]  # Its smaller cohort would pass alone.
    assert result["H3b"]["16"]["label"]==result["Bridge"]["16"]["label"]==a.NE
    assert result["H3b"]["128"]["label"]=="FREE"


def test_h2_competence_gate_keeps_holm_two():
    cells=grid(8,lin=.94)
    for c in cells:
        if c["arm"]=="so" and c["load"]==128:
            c["composite"]["A"]=.89
        if c["arm"]=="aice":
            positive=c["seed"]!="fixture-07"
            c["composite"]["A"]=(.90 if positive else .99)+(1 if positive else -1)*.03*math.log2(c["load"]/16)
    result=a.score(cells)
    assert result["H2"]["label"]==a.NE and result["H2"]["raw_p"]==1
    assert result["H3a"]["holm_family"]==["H2","H3a"]
    assert result["H3a"]["raw_p"]==9/256 and result["H3a"]["p_adj"]==18/256


def test_fresh_run_directory_regenerates_smoke_then_admits(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c=tomllib.loads((l.ROOT/"configs/toy/lod-x.toml").read_text())["lod"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(l,"seal_guard",lambda *a:None)
    monkeypatch.setattr(t,"_mark_process",lambda name:None)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda *args:SimpleNamespace(total_memory=24e9))
    calls=[]
    def dispatch(config,cells,workers,deadline,**kwargs):
        calls.append(config["mode"])
        if config["mode"]=="pod_smoke":
            assert all(cell["seed"]==0 for cell in cells) and workers==5
            rows={(r["arm"],r["load"],r["layout"]):r for r in
                  (smoke_receipt()["cells"] if config.get("protocol") != "lodx" else
                   [dict(arm=a,load=n,layout="main",ms_step=30.,peak_gb=2.)
                    for n in (16,32,64) for a in ("lin","so")])}
            return [dict(**cell,status="COMPLETE",ms_step=rows[(cell["arm"],cell["load"],cell["layout"])]["ms_step"],peak_gb=2.) for cell in cells]
        assert Path(config["smoke_receipt"]).exists()
        manifest=json.loads(Path(config["output_dir"],"manifest.json").read_text())
        assert manifest["admission"]["tier"] == ("X" if config.get("protocol") == "lodx" else "T1") and manifest["created_before_production"]
        return []  # No production cell or model is executed by this fixture.
    monkeypatch.setattr(l,"dispatch",dispatch)
    assert not Path(c["smoke_receipt"]).exists()
    result=l.run(c,0)
    assert calls==["pod_smoke","production"]
    receipt=json.loads(Path(c["smoke_receipt"]).read_text())
    assert "setup_elapsed_hours" not in receipt
    assert result["admission"]["setup_hours"]==.45
    receipt["setup_elapsed_hours"]=.6
    assert l.timing_admission(receipt,c)["setup_hours"]==.6


def test_infrastructure_stop_suppresses_scientific_claims(tmp_path):
    c=cfg(output_dir=str(tmp_path))
    manifest=dict(config=c,config_sha256=l.sha(c),source_hashes=l.source_hashes(),cells=[],
        created_before_production=True,admission=dict(tier="T1",addon=False),infra_stop=dict(status="INFRA_GPU_ERRORS"))
    l.atomic_json(tmp_path/"manifest.json",manifest)
    result=a.analyze_directory(tmp_path,diagnostic=True)
    assert result["mode"]=="INFRA_GPU_ERRORS; no scientific claims" and not result["valid"]


def test_stdout_lines_hold_lock_and_keep_full_json(monkeypatch,capsys):
    entries=[]
    class Lock:
        def __enter__(self):
            entries.append("enter")
        def __exit__(self,*args):
            entries.append("exit")
    monkeypatch.setattr(l,"_EMIT_LOCK",Lock())
    row=dict(cell_id="fixture",values=list(range(4000)))
    l.emit("LOG",row)
    line=capsys.readouterr().out
    assert entries==["enter","exit"]
    assert line.count("\n")==1 and json.loads(line.removeprefix("LOD_LOG "))==row
