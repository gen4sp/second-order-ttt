"""R4-3 arms, registration traps, second-order mask identities and the analysis labels."""

import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from tttmem import exp_ttt as t

ROOT = Path(__file__).resolve().parents[1]
R43_ARMS = ("e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb", "e2e2emb", "e2e2", "e2e2_kvm")


def tiny():
    st = t._settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
                          window=16, chunk=16, n_keys=4, T=128, n_eval=4, n_quick_eval=4,
                          eval_batch=2, n_probe=4))
    return st, t._task(st, "echo")


# ------------------------------------------------------------------- registrations
def test_arm_registration():
    for arm in R43_ARMS:
        assert arm in t.SEQUENTIAL_ARMS, arm
    assert t.MASK_ARMS["e2e2_kvm"] == "kv"
    assert "e2e2_kvm" in t.SECOND_ORDER_ARMS and "e2e2_kvm" in t.R43_ARMS
    assert t.FACTOR_ARMS["e2e1e_emb"] == t.FACTOR_ARMS["e2e1e"] == (False, True)
    assert t.FACTOR_ARMS["e2e1k_emb"] == t.FACTOR_ARMS["e2e1k"] == (True, False)
    for arm in ("e2e1e_emb", "e2e1k_emb", "e2e2emb"):
        assert arm in t.EMB_ARMS
    # the +emb arms take the +emb input path, e2e2_kvm the plain one
    st, task = tiny()
    assert t.build_model(task, st, torch.device("cpu"), 0, "e2e1e_emb").frozen_emb is not None
    assert t.build_model(task, st, torch.device("cpu"), 0, "e2e1k_emb").frozen_emb is not None
    assert t.build_model(task, st, torch.device("cpu"), 0, "e2e2_kvm").frozen_emb is None
    assert {"e2e1emb", "e2e1emb_kvm", "e2e1emb_rnd", "e2e2emb", "e2e2"} <= (t.ABLATION_ARMS |
                                                                       t.R43_ABLATION_EXTRA)
    # the same init and frozen code for a seed across all the +emb arms
    models = [t.build_model(task, st, torch.device("cpu"), 3, arm) for arm in R43_ARMS
              if arm in t.EMB_ARMS]
    assert all(torch.equal(models[0].frozen_emb, m.frozen_emb) for m in models[1:])
    assert all(torch.equal(models[0].tok.weight, m.tok.weight) for m in models[1:])


def test_factor_arm_with_a_mask_raises():
    st, task = tiny()
    dev = torch.device("cpu")
    model = t.build_model(task, st, dev, 0, "e2e1e_emb")
    idx, y, meta = task.tensors(*task.raw(np.random.default_rng(3), 2), dev)
    keep = t.outer_keep(meta, y.shape, "kv", None, dev)
    with pytest.raises(ValueError, match="outer mask"):
        t.arm_loss(model, "e2e1e_emb", idx, y, 10., 10., task.chunk, t.GradStats(), keep)
    with pytest.raises(ValueError, match="outer mask"):
        t.sequential_inner(model, model.backbone(idx), y, model.per_sequence(2),
                           model.etas(10.), 10., False, t.GradStats(), task.chunk, idx,
                           (False, True), keep)
    # the same arm without a mask trains
    assert math.isfinite(float(t.arm_loss(model, "e2e1e_emb", idx, y, 10., 10., task.chunk,
                                         t.GradStats()).detach()))


def test_second_order_checkpoint_list_covers_every_second_order_arm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st, task = tiny()
    st.checkpoint_steps = [1, 3]
    st.checkpoint_steps_so = [2]
    dev = torch.device("cpu")
    for arm in ("e2e2", "e2e2emb", "e2e2_kvm"):
        model = t.build_model(task, st, dev, 0, arm)
        result = t.train_arm(model, task, arm, 1., st, dev, np.random.default_rng(8), None,
                             f"echo/s0/{arm}", steps=3)
        assert [Path(p).name for p in result["step_checkpoints"]] == [
            f"echo__s0__{arm}__step{i}.pt" for i in (1, 2, 3)], arm
    # a first-order arm takes only the first-order list
    model = t.build_model(task, st, dev, 0, "e2e1emb")
    result = t.train_arm(model, task, "e2e1emb", 1., st, dev, np.random.default_rng(8), None,
                         "echo/s0/e2e1emb", steps=3)
    assert [Path(p).name for p in result["step_checkpoints"]] == [
        "echo__s0__e2e1emb__step1.pt", "echo__s0__e2e1emb__step3.pt"]


def test_probe_logs_all_four_etas():
    st, task = tiny()
    dev = torch.device("cpu")
    model = t.build_model(task, st, dev, 0, "e2e1e_emb")
    with torch.no_grad():
        model.rho.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    data = t.eval_batches(task, *t.fixed_set(task, st, "eval"), dev, 2)
    p = t.r4_probe(model, task, data, 10., 10., arm="e2e1e_emb")
    expected = [float(e.detach()) for e in model.etas(10.)]
    assert [p[f"eta_{key}"] for key in t.FAST_KEYS] == pytest.approx(expected, abs=1e-6)
    assert all(math.isfinite(p[k]) for k in ("p_kv", "e_kv_abs", "kv_write_eff", "addr_a",
                                             "pr_kv", "ce_kv"))


def test_tf32_key_pins_fp32_matmuls():
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    try:
        off = t._log_tf32(False, "w0")
        assert off["matmul"] is False and off["cudnn"] is False
        assert off["precision"] == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        on = t._log_tf32(True, "w0")
        assert on["matmul"] is True and on["cudnn"] is True
        assert t._settings(dict(tf32=False)).tf32 is False
        assert t._settings(dict()).tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def test_r43_sanity_passes_and_reports_the_tiny_share():
    checks = t.r43_sanity()
    assert all(v["passed"] for v in checks.values()), checks
    assert checks["registration"]["factor_mask_raises"]
    assert checks["registration"]["second_order_arms"] == sorted(t.SECOND_ORDER_ARMS)
    assert checks["N2_so"]["create_graph"]
    assert checks["N2_so"]["inner_equal"] and checks["N2_so"]["kv_logits_grad_ok"]
    assert checks["N2_so"]["linearity_loss_rel"] <= 1e-6
    assert checks["N2_so"]["linearity_gradient_rel"] <= 1e-6
    # the real path: the mask kills the direct kv-logit gradient in first order and cannot
    # in second order, which is what proves create_graph runs through e on the masked arm
    kv = checks["N2_so"]["kv_logits_grad"]
    assert checks["N2_so"]["kv_targets_per_seq"] == 4
    assert kv["e2e1_kvm"] == 0. and kv["e2e1emb_kvm"] == 0.
    assert kv["e2e2_kvm"] > 0. and kv["e2e2_unmasked"] > kv["e2e2_kvm"]
    assert checks["N3"]["groups_equal"] == {"second_order": True, "emb": True}
    assert set(checks["N3"]["groups_equal"]) == {"second_order", "emb"}
    assert checks["N4"]["kv_counts"] == [4, 4]
    # S1/S2 must have run on an +emb model in fp64, with R4-1's tolerances
    assert checks["S1_init"]["passed"] and checks["S1_trained"]["passed"]
    assert checks["S2_init_clip10"]["additivity_rel"] < 1e-4
    assert checks["S2_trained_clip0.001"]["passed"]
    # the tiny-model numbers are descriptive only; the gate is the real-size share record
    assert set(checks["EK_tiny_descriptive"]["stages"]) == {"init", "trained"}
    assert set(checks["EK_tiny_descriptive"]["stages"]["init"]) >= {"rel_E", "rel_K"}
    assert checks["EK_tiny_descriptive"]["gating"] is False


def test_cells_report_their_path_and_mask(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    cfg = dict(process_name="TTM-r43-test", device="cpu", workers=1, max_minutes=5,
               tf32=False, d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
               window=16, chunk=16, T=128, n_keys=4, batch=2, n_eval=4, n_val=4,
               n_quick_eval=4, eval_batch=2, quick_every=2, probe_every=2, n_probe=2,
               lr_total=4, warmup=1,
               checkpoint_steps=[1], checkpoint_steps_so=[2],
               cells=[dict(task="echo", arm=arm, seed=0, eta_base=10., steps=2, diag=False)
                      for arm in R43_ARMS])
    out = t.run_cells(cfg, t._settings(cfg), torch.device("cpu"))["cells"]
    assert len(out) == len(R43_ARMS)
    for arm in R43_ARMS:
        record = out[f"echo/s0/{arm}/eta10/st2"]
        assert "error" not in record, (arm, record.get("error"))
        assert record["create_graph"] is (arm in t.SECOND_ORDER_ARMS)
        assert record["factors"] == (list(t.FACTOR_ARMS[arm]) if arm in t.FACTOR_ARMS else None)
        assert record["mask"] == t.MASK_ARMS.get(arm, "none")
        assert record["masked_per_seq"] == (4 if arm in t.MASK_ARMS else 0)
        assert record["tf32"] is False and record["peak_cuda_mb"] == 0.0
        assert record["first_batch_sha1"]
        assert set(record["ablation"]) == {"all", "all_minus_kv", "kv_only"}
        assert record["ablation_ref_match"]
        assert [p["step"] for p in record["probes"]] == [2]
    captured = capfd.readouterr().out
    if "TF32 worker=" in captured:
        assert "TF32 worker=main matmul=False cudnn=False" in captured


# ---------------------------------------------------------------------- the configs
def test_r43_config_order_and_settings():
    with (ROOT / "configs/toy/r4-3.toml").open("rb") as f:
        cfg = tomllib.load(f)["ttt"]
    assert cfg["process_name"] == "TTM-r43-01" and cfg["workers"] == 6
    assert cfg["max_minutes"] == 180 and cfg["tf32"] is False and cfg["device"] == "auto"
    assert cfg["mps_memory_fraction"] == 0.20
    assert cfg["checkpoint_steps"] == [200, 300, 500, 1000]
    assert cfg["checkpoint_steps_so"] == [100]
    assert all(not c["diag"] and c["eta_base"] == 10 and c["task"] == "echo" for c in cfg["cells"])
    order = [(c["seed"], c["arm"]) for c in cfg["cells"]]
    expected = []
    for s in range(20):
        expected.append((s, "e2e2"))
        expected.append((s, "e2e2_kvm"))
        if s <= 9:
            expected += [(s, arm) for arm in ("e2e1emb", "e2e1emb_kvm", "e2e1e_emb",
                                              "e2e1k_emb", "e2e2emb")]
    assert order == expected
    assert len(order) == 90
    # every setting but the checkpoint lists, the worker count and tf32 is R4-2's
    keys = ("d_model", "n_layers", "n_heads", "backbone_mult", "fast_mult", "window", "chunk",
            "T", "steps", "batch", "n_keys", "n_eval", "n_val", "n_quick_eval", "eval_batch",
            "lr", "lr_total", "warmup", "lr_floor", "eta_lr", "grad_clip", "g_clip",
            "quick_every", "probe_every", "n_probe")
    old = tomllib.loads((ROOT / "configs/toy/r4-2.toml").read_text())["ttt"]
    assert all(cfg[k] == old[k] for k in keys), {k: (cfg[k], old[k]) for k in keys
                                                 if cfg[k] != old[k]}


def test_smoke_and_podsmoke_configs():
    with (ROOT / "configs/toy/smoke/r4-3.toml").open("rb") as f:
        smoke = tomllib.load(f)["ttt"]
    assert smoke["device"] == "cpu" and smoke["workers"] == 1 and smoke["steps"] <= 25
    assert smoke["probe_every"] and smoke["quick_every"]
    assert [(c["seed"], c["arm"]) for c in smoke["cells"]] == [
        (seed, arm) for seed in (0, 1) for arm in
        ("e2e2", "e2e2_kvm", "e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb", "e2e2emb")]
    with (ROOT / "configs/toy/smoke/r4-3-gpu.toml").open("rb") as f:
        pod = tomllib.load(f)["ttt"]
    assert pod["device"] == "auto" and pod["workers"] == 6 and pod["tf32"] is False
    assert pod["steps"] == 100 and pod["process_name"] == "TTM-r43-00-smoke"
    assert [c["arm"] for c in pod["cells"]] == ["e2e1emb", "e2e1e_emb", "e2e1k_emb",
                                                 "e2e1emb_kvm", "e2e2emb", "e2e2_kvm"]
    assert all(c["seed"] == 0 and c["eta_base"] == 10 and not c["diag"] for c in pod["cells"])
    assert len(pod["cells"]) == pod["workers"]      # one cell per worker: the real ms/step
    assert pod["quick_every"] and pod["probe_every"] == 25
    assert pod["checkpoint_steps"] == [50, 100] and pod["checkpoint_steps_so"] == [25, 50]
    # the per-cell summary line the smoke reads its numbers from
    for cell in pod["cells"]:
        assert cell["arm"] in t.ABLATION_ARMS | t.R43_ABLATION_EXTRA


# ------------------------------------------------------------------ the analysis
def cell(seed, arm, *, ever, final, acc, group, e_kv=0.02, p_kv=0.95, ablation=True,
         quick_max=0.9):
    """One synthetic CELL_RESULT line with the fields the analysis reads."""
    masked = 16 if arm in ("e2e1emb_kvm", "e2e2_kvm") else 0
    record = dict(
        cell_id=f"echo/s{seed}/{arm}/eta10/st1500", task="echo", seed=seed, arm=arm,
        first_batch_sha1=f"sha-{group}-{seed}", masked_per_seq=masked,
        mask=t.MASK_ARMS.get(arm, "none"), create_graph=arm in t.SECOND_ORDER_ARMS,
        factors=(list(t.FACTOR_ARMS[arm]) if arm in t.FACTOR_ARMS else None),
        tf32=False, peak_cuda_mb=0.0, bound_ever=ever, bound_final=final,
        collapse=bool(ever) and not final, degraded=False, partial=False,
        eval=dict(query_acc=acc, query_ce=1.0, ce=1.0, ce_late=1.0),
        quick=[[1500, 1.0, 1.0, acc, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]],
        probes=[dict(step=1500, addr_a=.9, pr_kv=8.0, e_kv_abs=e_kv, p_kv=p_kv,
                     kv_write_eff=.5, Wv_write_eff=.2, ce_kv=.1, ce_vk=.2, ce_vW=1.0,
                     eta_W1=10., eta_b1=1., eta_W2=10., eta_b2=1.)],
        eta_final=[10., 1., 10., 1.])
    if arm in ("e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb", "e2e2emb"):
        record["s5"] = dict(passed=True, lin_kv_max_diff=0.0, lin_kv_acc_diff=0.0)
    if ablation:
        record["ablation"] = dict(all=acc, all_minus_kv=0.1, kv_only=0.9)
        record["ablation_ref_match"] = True
    return "CELL_RESULT " + json.dumps(record)


def ek_share_line(share_E=0.40, share_K=0.20):
    """One R43_EK_SHARE record: the analyser reads the E/K gate from it, never a default."""
    per_seed = [dict(seed=s, e2e1e_emb=dict(share=share_E, cos=0.9, rel=0.05),
                     e2e1k_emb=dict(share=share_K, cos=0.8, rel=0.02)) for s in range(3)]
    return "R43_EK_SHARE " + json.dumps(dict(threshold=0.05, per_seed=per_seed,
                                            median_E_share=share_E, median_K_share=share_K))


def synthetic_log(share=True):
    """E retains, K does not, and the e2e2 anchor never loses the binding (S2-ret gate)."""
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32, ce=2.0), chance=dict(acc=.0625)))]
    if share:
        lines.append(ek_share_line())
    for s in range(10):
        lines.append(cell(s, "e2e1emb", ever=True, final=False, acc=0.20, group="fo"))
        lines.append(cell(s, "e2e1emb_kvm", ever=True, final=False, acc=0.20, group="fo"))
        lines.append(cell(s, "e2e1e_emb", ever=True, final=s < 8, acc=0.95 if s < 8 else 0.2,
                          group="fo", e_kv=0.02, p_kv=0.95))
        # K is starved at the write even though it acquires: MS1 must not fire for it
        lines.append(cell(s, "e2e1k_emb", ever=True, final=False, acc=0.20, group="fo",
                          e_kv=0.0005, p_kv=0.999))
        lines.append(cell(s, "e2e2emb", ever=True, final=True, acc=0.99, group="fo",
                          e_kv=0.02, p_kv=0.90))
    for s in range(20):
        kept = s < 14
        lines.append(cell(s, "e2e2", ever=kept, final=kept, acc=0.9 if kept else 0.2,
                          group="so", ablation=False))
        # the mask is predicted to lower p_kv in second order
        lines.append(cell(s, "e2e2_kvm", ever=kept, final=kept,
                          acc=(0.95 if s < 3 else 0.9) if kept else 0.2, group="so",
                          e_kv=0.02 if kept else 0.0005, p_kv=0.5 if kept else 0.95,
                          ablation=False))
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def analysis():
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_r43
    return analyze_r43


def test_exact_statistics(analysis):
    a = analysis
    # exact one-sided McNemar: 8-0 discordant is 0.5**8
    rows = [(s, dict(bound_final=True), dict(bound_final=False)) for s in range(8)]
    got = a.contrast(rows, "bound_final")
    assert got["mcnemar_one_sided"] == pytest.approx(2**-8) and got["n"] == 8
    assert a.contrast([], "bound_final")["mcnemar_one_sided"] == 1.0
    # exact 97.5% Clopper-Pearson bounds, projected into a >= 95% paired interval
    lo, hi = a.cp(8, 8)
    assert lo == pytest.approx(0.0125**(1/8), abs=1e-6) and hi == 1.0
    assert got["ci95"] == [lo - a.cp(0, 8)[1], 1.0 - 0.0]
    assert a.holm({"E": 0.5**8, "K": 1.0}) == {"E": pytest.approx(2**-7), "K": 1.0}
    # exact Wilcoxon against the enumerated null
    assert a.wilcoxon_one_sided([1., 1., 1., 1., 1.])["p_one_sided"] == pytest.approx(1/32)
    assert a.wilcoxon_one_sided([-1., -1., -1., -1., -1.])["p_one_sided"] == pytest.approx(1.0)
    assert a.wilcoxon_one_sided([0., 0.])["p_one_sided"] is None
    assert a.ratio(1., 0.) == math.inf and a.ratio(0., 0.) is None


def test_labels_on_the_synthetic_log(analysis, tmp_path):
    a = analysis
    log = tmp_path / "synthetic.log"
    log.write_text(synthetic_log())
    result = a.summarize(*a.records(log))
    labels = result["labels"]
    assert result["issues"] == []
    assert labels["V3a"] == "holds" and labels["V3b"] == "holds" and labels["V3c"] == "holds"
    assert labels["E"] == "E-retains"
    assert labels["K"] == "K-no retention gain shown"
    assert result["holm"]["E"] == pytest.approx(2**-7)
    assert labels["MS1_E"] is True and labels["MS2_E"] is True
    # K never ends bound, so MS1 has no population
    assert labels["MS1_K"] == "not evaluable (0 bound seeds)"
    assert result["details"]["MS1_K"]["bound_seeds"] == 0
    assert result["ek_manipulation"]["status"] == "computed"
    assert labels["MS_reading_E"].startswith("E resists starvation by keeping surprise")
    assert labels["EK_joint"].startswith("E-retains and not K-retains")
    assert labels["FM"] == "FM-no rescue shown"
    # the anchor acquired and never lost the binding: S2-ret is untestable at this n
    assert labels["S2-ret"] == "untestable at this n"
    assert labels["S2-acq"] == "no acquisition effect shown"
    assert labels["S2-overall"].startswith("reported")
    assert labels["FM_S2_joint"].startswith("FM-no rescue shown and S2-ret")
    assert result["details"]["S2"]["anchor_lost_after_acquiring"] == 0
    assert result["details"]["S2"]["n_common_acquired"] == 14
    assert 0. < result["details"]["S2_wilcoxon"]["p_one_sided"] < 1.
    assert result["n_complete"] == dict(first_order_emb=10, ek_E=10, ek_K=10, ms1_E=10,
                                        ms1_K=10, second_order=20)
    assert result["counts"]["e2e1e_emb"]["final"] == 8
    assert result["details"]["S2_mechanism"]["p_kv_paired"]["seeds_kvm_lower"] == 14
    assert len(result["eta_rows"]) == 40


def test_s2_ret_can_fire_and_pending_covers_short_runs(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    for s in range(10):
        lines += [cell(s, "e2e1emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e1emb_kvm", ever=True, final=True, acc=0.95, group="fo"),
                  cell(s, "e2e1e_emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e1k_emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e2emb", ever=True, final=True, acc=0.99, group="fo")]
    for s in range(20):
        lost_seed = s >= 6          # the anchor loses the binding in 6 of 20 seeds
        kept = s < 12
        lines += [cell(s, "e2e2", ever=kept, final=kept and not lost_seed,
                       acc=0.9 if (kept and not lost_seed) else 0.2, group="so", ablation=False),
                  cell(s, "e2e2_kvm", ever=kept, final=kept, acc=0.95 if kept else 0.2,
                       group="so", ablation=False)]
    log = tmp_path / "s2ret.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert labels["FM"] == "FM-rescue"
    assert labels["V3a"] == "holds"
    assert labels["S2-ret"] == "S2-ret rescue"
    assert labels["S2-overall"].startswith("label")
    assert labels["FM_S2_joint"].startswith("FM-rescue and S2-ret rescue")
    # fewer complete pairs than the group needs: every label is pending, never negative
    short = "\n".join(lines[:6]) + "\n"
    short_log = tmp_path / "short.log"
    short_log.write_text(short)
    pending = a.summarize(*a.records(short_log))["labels"]
    assert pending["V3a"] == "pending" and pending["E"] == "E-pending"
    assert pending["FM"] == "pending" and pending["S2-ret"] == "pending"
    assert pending["S2-acq"] == "pending" and pending["MS1_E"] == "pending"


def test_pairing_violations_exclude_the_seed(analysis, tmp_path):
    a = analysis
    log = tmp_path / "unpaired.log"
    lines = synthetic_log().splitlines()
    fixed = []
    for line in lines:
        if line.startswith("CELL_RESULT "):
            record = json.loads(line[len("CELL_RESULT "):])
            if (record["seed"], record["arm"]) == (3, "e2e1e_emb"):
                record["first_batch_sha1"] = "sha-other"
            if (record["seed"], record["arm"]) == (4, "e2e2_kvm"):
                record["masked_per_seq"] = 0
            if (record["seed"], record["arm"]) == (5, "e2e2_kvm"):
                record["create_graph"] = False
            line = "CELL_RESULT " + json.dumps(record)
        fixed.append(line)
    log.write_text("\n".join(fixed) + "\n")
    result = a.summarize(*a.records(log))
    joined = " ".join(result["issues"])
    assert "first_batch_sha1" in joined and "masked_per_seq" in joined and "create_graph" in joined
    assert result["n_complete"]["ek_E"] == 9 and result["n_complete"]["second_order"] == 18
    # the E arm's hash breaks only the comparisons that contain it, not FM's
    assert result["n_complete"]["first_order_emb"] == 10
    assert result["n_complete"]["ms1_E"] == 9
    assert result["labels"]["FM"] == "FM-no rescue shown"
    assert result["labels"]["E"] == "E-pending"


def test_tf32_and_a_wrong_factor_path_are_invalid(analysis, tmp_path):
    a = analysis
    fixed = []
    for line in synthetic_log().splitlines():
        if line.startswith("CELL_RESULT "):
            record = json.loads(line[len("CELL_RESULT "):])
            if (record["seed"], record["arm"]) == (6, "e2e1k_emb"):
                record["tf32"] = True
            if (record["seed"], record["arm"]) == (7, "e2e1e_emb"):
                record["factors"] = [True, False]      # the input-side switches, not error-side
            line = "CELL_RESULT " + json.dumps(record)
        fixed.append(line)
    log = tmp_path / "invalid.log"
    log.write_text("\n".join(fixed) + "\n")
    result = a.summarize(*a.records(log))
    joined = " ".join(result["issues"])
    assert "tf32 on" in joined and "factors [True, False]" in joined
    # an invalid cell is invalid for every comparison it is in
    assert result["n_complete"]["ek_K"] == 9 and result["n_complete"]["ms1_E"] == 9
    assert result["n_complete"]["first_order_emb"] == 10


def _at(value, seed):
    """A retention pattern: a constant for the whole group, or a per-seed predicate."""
    return bool(value(seed)) if callable(value) else bool(value)


def _fo_group(lines, *, anchor_final=0, kvm_final=0, e_final=0, e2eemb_final=10):
    """The five +emb cells of every seed, with the retention pattern the tests need."""
    for s in range(10):
        af, kf, ef, rf = (_at(v, s) for v in (anchor_final, kvm_final, e_final, e2eemb_final))
        lines += [cell(s, "e2e1emb", ever=True, final=af, acc=0.95 if af else 0.20, group="fo"),
                  cell(s, "e2e1emb_kvm", ever=True, final=kf, acc=0.95 if kf else 0.20,
                       group="fo"),
                  cell(s, "e2e1e_emb", ever=True, final=ef, acc=0.95 if ef else 0.20,
                       group="fo", e_kv=0.02, p_kv=0.95),
                  cell(s, "e2e1k_emb", ever=True, final=False, acc=0.20, group="fo"),
                  cell(s, "e2e2emb", ever=True, final=rf, acc=0.99 if rf else 0.20, group="fo",
                       e_kv=0.02, p_kv=0.90)]
    return lines


def _so_group(lines, *, e2e2_ever=lambda s: True, e2e2_final=lambda s: s < 14,
              kvm_ever=lambda s: True, kvm_final=lambda s: s < 14):
    for s in range(20):
        ae, af = _at(e2e2_ever, s), _at(e2e2_final, s)
        ke, kf = _at(kvm_ever, s), _at(kvm_final, s)
        lines += [cell(s, "e2e2", ever=ae, final=af, acc=0.9 if af else 0.2, group="so",
                       ablation=False),
                  cell(s, "e2e2_kvm", ever=ke, final=kf, acc=0.9 if kf else 0.2, group="so",
                       ablation=False)]
    return lines


def test_missing_ek_record_leaves_the_manipulation_unknown(analysis, tmp_path):
    a = analysis
    log = tmp_path / "no_share.log"
    log.write_text(synthetic_log(share=False))
    result = a.summarize(*a.records(log))
    assert result["ek_manipulation"]["status"] == "unknown"
    assert result["ek_manipulation"]["E"] is None and result["ek_manipulation"]["K"] is None
    assert result["labels"]["E"] == "E-untested (manipulation unknown)"
    assert result["labels"]["K"] == "K-untested (manipulation unknown)"
    assert result["labels"]["EK_joint"] == (
        "no joint reading (E: E-untested (manipulation unknown); "
        "K: K-untested (manipulation unknown))")


def test_a_failed_manipulation_gate_is_untested_failed(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line(0.01, 0.4)]
    log = tmp_path / "weak_share.log"
    _so_group(_fo_group(lines))
    log.write_text("\n".join(lines) + "\n")
    result = a.summarize(*a.records(log))
    assert result["ek_manipulation"]["median_share_E"] == 0.01
    # the cosine and the relative metric are reported with the gate, not used by it
    assert result["ek_manipulation"]["median_cos_E"] == 0.9
    assert result["ek_manipulation"]["median_rel_K"] == 0.02
    assert result["ek_manipulation"]["E"] is False and result["ek_manipulation"]["K"] is True
    assert result["labels"]["E"] == "E-untested (manipulation failed)"
    assert result["details"]["K"]["n"] == 10      # K itself still gets a reading


def test_ek_joint_needs_two_final_labels(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # E retains 8/10, K's cells are present but only 9 of the 10 pairs are complete
    for s in range(10):
        lines += [cell(s, "e2e1emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e1emb_kvm", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e1e_emb", ever=True, final=s < 8, acc=0.95 if s < 8 else 0.2,
                       group="fo", e_kv=0.02, p_kv=0.95),
                  cell(s, "e2e2emb", ever=True, final=True, acc=0.99, group="fo",
                       e_kv=0.02, p_kv=0.90)]
    lines += [cell(s, "e2e1k_emb", ever=True, final=False, acc=0.2, group="fo") for s in range(9)]
    _so_group(lines)
    log = tmp_path / "one_pending.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert labels["E"] == "E-retains" and labels["K"] == "K-pending"
    assert labels["EK_joint"] == "no joint reading (E: E-retains; K: K-pending)"
    # a final but unlisted pair also gets no reading
    assert a.ek_joint({"E": "E-partial", "K": "K-inconclusive"}) == (
        "No joint reading: see the per-half labels.")
    assert a.ek_joint({"E": "E-retains", "K": "K-blocked"}).startswith("E-retains and not K")
    assert a.ek_joint({"E": "E-no retention gain shown", "K": "K-no retention gain shown"}
                      ).startswith("Neither retains while both acquire")


def test_ek_joint_wording(analysis):
    """A retaining half is "sufficient alone (added to first order)"; the halves do not
    decompose e2e2emb's correction over a sequence, so nothing "carries the resistance"."""
    a = analysis
    for e, k in (("E-retains", "K-blocked"), ("E-inconclusive", "K-retains")):
        text = a.ek_joint({"E": e, "K": k})
        assert "is sufficient alone (added to first order)" in text and "carries" not in text
    neither = a.ek_joint({"E": "E-no retention gain shown", "K": "K-no retention gain shown"})
    assert "per-update sum or through the cross-update cross terms" in neither
    assert "multi-step dynamics" not in neither
    assert a.ek_joint({"E": "E-blocked", "K": "K-blocked"}).endswith("Descriptive.")


def test_a_failed_gate_leaves_pending_labels_pending(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # V3a fails (the anchor retains) while FM's seed-9 kvm cell is missing; V3c fails
    # (e2e2 retains 17/20) while one e2e2_kvm cell is missing
    _so_group(_fo_group(lines, anchor_final=True), e2e2_final=lambda s: s < 17,
              kvm_final=lambda s: s < 17)
    lines = [l for l in lines if not ('"seed": 9,' in l and '"arm": "e2e1emb_kvm"' in l)
             and not ('"seed": 19,' in l and '"arm": "e2e2_kvm"' in l)]
    log = tmp_path / "gate_pending.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert labels["V3a"] != "holds" and labels["V3c"] != "holds"
    assert labels["FM"] == "pending"
    assert labels["S2-acq"] == labels["S2-ret"] == labels["S2-overall"] == "pending"
    assert labels["FM_S2_joint"] == "descriptive"


def test_v3a_failure_makes_everything_it_governs_descriptive(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # the anchor keeps the binding in 8/10: V3a fails for want of a retention deficit
    _so_group(_fo_group(lines, anchor_final=True, e2eemb_final=True))
    log = tmp_path / "v3a_fail.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert not labels["V3a"] == "holds"
    assert labels["FM"] == "FM-no rescue shown (descriptive)"
    assert labels["E"] == "E-descriptive (V3a did not hold)"
    assert labels["K"] == "K-descriptive (V3a did not hold)"
    assert labels["MS1_E"] == "descriptive (V3a did not hold)"
    assert labels["MS2_K"] == "descriptive (V3a did not hold)"
    assert labels["EK_joint"] == "descriptive"
    assert labels["FM_S2_joint"] == "descriptive"
    # S2 does not depend on the first-order anchor, so it keeps its own reading
    assert labels["S2-ret"] == "no retention effect shown"


def test_v3b_failure_makes_ms1_descriptive_not_pending(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # e2e2emb is at 6/10, so V3b does not hold; E retains in all 10
    _so_group(_fo_group(lines, e_final=True, e2eemb_final=lambda s: s < 6))
    log = tmp_path / "v3b_fail.log"
    log.write_text("\n".join(lines) + "\n")
    result = a.summarize(*a.records(log))
    labels = result["labels"]
    assert labels["V3b"] == "not at ceiling"
    assert labels["MS1_E"] == "descriptive (V3b did not hold)"
    # a half that retains still gets an MS reading, and it carries the wording
    assert labels["E"] == "E-retains"
    assert labels["MS_reading_E"] == "descriptive (V3b did not hold)"
    assert result["details"]["MS1_E"] == dict(n=10)


def test_v3c_failure_makes_s2_descriptive(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # e2e2 keeps the binding in 17/20: no second-order deficit, so S2 cannot be a label
    _so_group(_fo_group(lines), e2e2_final=lambda s: s < 17, kvm_final=lambda s: s < 17)
    log = tmp_path / "v3c_fail.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert labels["V3c"] == "no second-order deficit on this device"
    assert labels["S2-acq"].endswith("(descriptive)")
    assert labels["S2-ret"] == "untestable at this n (descriptive)"
    assert labels["S2-overall"].endswith("(descriptive)")
    assert labels["FM_S2_joint"] == "descriptive"


def test_s2_ret_hurts_has_no_joint_reading(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    # the anchor acquires in 14 seeds and loses 4 of them, so S2-ret is testable; the mask
    # keeps the binding only where the anchor had already lost it, 3 against 10 (p = 0.046)
    _so_group(_fo_group(lines, kvm_final=True, e2eemb_final=True),
              e2e2_ever=lambda s: s < 14, e2e2_final=lambda s: s < 10,
              kvm_ever=lambda s: s < 14, kvm_final=lambda s: 10 <= s < 13)
    log = tmp_path / "s2_hurts.log"
    log.write_text("\n".join(lines) + "\n")
    labels = a.summarize(*a.records(log))["labels"]
    assert labels["S2-ret"] == "S2-ret hurts"
    assert labels["FM_S2_joint"] == "no pre-registered joint reading (S2-ret hurts); descriptive"
    # the same row with the other FM label, and with the mask helping instead
    assert a.fm_s2_joint("FM-rescue", dict(ret="S2-ret hurts")) == (
        "no pre-registered joint reading (S2-ret hurts); descriptive")
    assert a.fm_s2_joint("FM-no rescue shown", dict(ret="S2-ret hurts")) == (
        "no pre-registered joint reading (S2-ret hurts); descriptive")


def test_ms1_uses_the_bound_seeds_of_each_arm(analysis, tmp_path):
    a = analysis
    lines = ["REF echo " + json.dumps(dict(occ_minus=dict(acc=0.32))), ek_share_line()]
    for s in range(10):
        lines += [cell(s, "e2e1emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e1emb_kvm", ever=True, final=False, acc=0.2, group="fo"),
                  # E ends bound in 4 seeds with surprise, and unbound in 6 without: the
                  # all-seed median is starved, the bound-seed median is not
                  cell(s, "e2e1e_emb", ever=True, final=s < 4, acc=0.95 if s < 4 else 0.2,
                       group="fo", e_kv=0.02 if s < 4 else 0.0005,
                       p_kv=0.95 if s < 4 else 0.999),
                  cell(s, "e2e1k_emb", ever=True, final=False, acc=0.2, group="fo"),
                  cell(s, "e2e2emb", ever=True, final=True, acc=0.99, group="fo",
                       e_kv=0.02, p_kv=0.90)]
    _so_group(lines)
    log = tmp_path / "ms1_pop.log"
    log.write_text("\n".join(lines) + "\n")
    result = a.summarize(*a.records(log))
    labels, detail = result["labels"], result["details"]["MS1_E"]
    assert labels["MS1_E"] is True
    assert detail["bound_seeds"] == 4 and detail["ref_bound_seeds"] == 10
    assert detail["median_e_kv_abs"] == 0.02
    assert detail["all_seed"]["median_e_kv_abs"] == 0.0005
    # K ends bound in no seed, so MS1 says how few there were instead of judging
    cells, probes, refs = a.records(log)[:3]
    assert a.ms1(cells, probes, "K", "holds", "holds") == (
        "not evaluable (0 bound seeds)", dict(n=10, bound_seeds=0))


def test_ek_share_on_a_tiny_model():
    """The metric itself: finite shares, and zero when the half is the anchor's own arm."""
    st, task = tiny()
    got = t.ek_share(None, task, st, 0, n_seq=2)
    assert got["dtype"] == "float64" and got["device"] == "cpu" and got["n_seq"] == 2
    assert got["first_batch_sha1"] and got["correction_norm"] > 0
    for arm in ("e2e1e_emb", "e2e1k_emb"):
        for key in ("share", "cos", "rel"):
            assert math.isfinite(got[arm][key]), (arm, key)
    # the anchor's own arm carries nothing of the correction, so its share is exactly 0
    same = t.ek_share(None, task, st, 0, n_seq=2, halves=(t.EK_ANCHOR,))
    assert same[t.EK_ANCHOR]["share"] == 0. and same[t.EK_ANCHOR]["cos"] is None
    # the reference arm carries all of it
    ref = t.ek_share(None, task, st, 0, n_seq=2, halves=(t.EK_REF,))
    assert ref[t.EK_REF]["share"] == pytest.approx(1.0)
    assert ref[t.EK_REF]["cos"] == pytest.approx(1.0)
    # and the checkpoint path: the same weights as a state_dict instead of a fresh build,
    # which is what r43_ek_share_ckpt.py loads, and it must reproduce the in-run numbers
    state = t.build_model(task, st, torch.device("cpu"), 0, t.EK_ANCHOR).state_dict()
    loaded = t.ek_share(state, task, st, 0, n_seq=2)
    assert loaded["first_batch_sha1"] == got["first_batch_sha1"]
    for arm in ("e2e1e_emb", "e2e1k_emb"):
        for key in ("share", "cos", "rel"):
            assert loaded[arm][key] == got[arm][key], (arm, key)
    # the record the analyser reads carries the per-seed rows and the medians
    record = t.ek_share_record(st, task, seeds=(0, 1), n_seq=2)
    assert [r["seed"] for r in record["per_seed"]] == [0, 1]
    assert record["threshold"] == 0.05 and record["arms"] == list(t.EK_ARMS)
    assert math.isfinite(record["median_E_share"]) and math.isfinite(record["median_K_share"])


def test_cli_writes_json_and_csv(analysis, tmp_path):
    a = analysis
    log = tmp_path / "cli.log"
    log.write_text(synthetic_log())
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_r43.py"), str(log),
                           str(out)], capture_output=True, text=True, check=True)
    assert "E-retains" in done.stdout and "untestable at this n" in done.stdout
    summary = json.loads((out / "summary.json").read_text())
    assert summary["labels"]["E"] == "E-retains"
    for name in ("paired_seeds.csv", "eta_diagnostic.csv", "labels.csv"):
        assert (out / name).read_text().count("\n") > 1
    rows = (out / "eta_diagnostic.csv").read_text().splitlines()
    assert "final_eta_W2" in rows[0] and "window_eta_W2" in rows[0]