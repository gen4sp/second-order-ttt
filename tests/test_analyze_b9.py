"""Outcome fixtures only; unit ids are synthetic, never training seeds."""
import copy
import pytest

from tttmem.b9_analysis import analyze, manipulation
from test_analyze_frsb import payload, row


def b9(n=12, rescue=True, generic=False):
    base = payload(n)
    cells = {}
    for i, unit in enumerate(base["manifest"]["seeds"]):
        for arm in ("e2e1_lin_b9", "e2e1_lin_b9f", "e2e1_b9"):
            success = rescue and (arm == "e2e1_lin_b9" or generic and arm == "e2e1_lin_b9f")
            r = row(arm, unit, success, success)
            r["aux_log"] = [dict(step=s, sham_anchor_count=16, lag_ratio=.75, grad_ratio=1.,
                                 anchor_count=16, lag=255 if arm.endswith("b9f") else 340,
                                 grad_norm=1.) for s in range(1, 1501)]
            cells[f"{unit}-{arm}"] = r
    return dict(manifest=base["manifest"], cells=cells), base


@pytest.mark.parametrize("n", [12, 11, 10])
def test_b9_rescue_key_specific_scaled_gates(n):
    a, b = b9(n)
    s = analyze(a, b)
    assert s["verdict"] == "B9-RESCUE" and s["qualifier"] == "KEY-SPECIFIC"
    assert s["manipulation_gate"] == "OK"


def test_generic_identity_requires_matching():
    a, b = b9(generic=True)
    s = analyze(a, b)
    assert s["qualifier"] == "GENERIC-IDENTITY"
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9f":
            for q in r["aux_log"]:
                q["grad_norm"] = 3.
    s = analyze(a, b)
    assert s["verdict"] == "B9-RESCUE" and s["qualifier"] is None
    assert not s["sham_matching"]["passed"]


def test_manipulation_early_failure_precedes_late():
    a, b = b9(rescue=False)
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9":
            r["probes"][1]["addr_h"] = .79
            r["probes"][-1]["addr_h"] = .1
    s = analyze(a, b)
    assert s["verdict"] == "MANIPULATION FAILED"
    assert s["manipulation_counts"]["FAILED"] == 12


def test_final_five_collapse_even_with_above_ninety_percent_horizon():
    a, b = b9(rescue=False)
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9":
            for q in r["probes"][-4:]:
                q["addr_h"] = .79
    s = analyze(a, b)
    assert s["verdict"] == "LATE-COLLAPSE"
    assert s["reading"] == "manipulation not sustained"
    assert s["verdict"] != "B9-NO-RESCUE"


def test_final_five_collapse_does_not_suppress_rescue():
    a, b = b9()
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9":
            r["probes"][-5]["addr_h"] = .1
    scores = analyze(a, b)
    assert scores["verdict"] == "B9-RESCUE"
    assert scores["manipulation_gate"] == "LATE-COLLAPSE"
    assert scores["reading"] is None


def test_whole_horizon_ninety_percent_rule():
    a, b = b9(rescue=False)
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9":
            for q in r["probes"][12:17]:
                q["addr_h"] = .79
    assert analyze(a, b)["verdict"] == "LATE-COLLAPSE"


def test_divergence_does_not_invent_an_early_address_failure():
    r = row("e2e1_lin_b9", "synthetic", False, False)
    r.update(disposition="DIVERGED", probes=r["probes"][:20])
    assert manipulation(r) == "LATE-COLLAPSE"
    r["probes"][0]["addr_h"] = .79
    assert manipulation(r) == "FAILED"


def test_no_rescue_only_when_manipulation_ok_and_acquisition_low():
    a, b = b9(rescue=False)
    s = analyze(a, b)
    assert s["verdict"] == "B9-NO-RESCUE"
    assert all(r["address"] == "OK" and not r["bound_ever"] for r in s["joint"])


@pytest.mark.parametrize("n", [0, 9])
def test_not_evaluable_small_cohort(n):
    a, b = b9(n)
    assert analyze(a, b)["verdict"] == "NOT EVALUABLE"


def test_missing_probe_never_passes_manipulation():
    a, b = b9()
    next(r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9")["probes"].pop(4)
    assert analyze(a, b)["verdict"] == "B9-RESCUE"  # one missing cell => scaled n=11
    for r in [r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9"][1:3]:
        r["probes"].pop(4)
    assert analyze(a, b)["verdict"] == "NOT EVALUABLE"


@pytest.mark.parametrize("key, value", [("lag", 190), ("grad_norm", .49), ("anchor_count", 11)])
def test_sham_matching_gate_failure_suppresses_specificity(key, value):
    a, b = b9()
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9f":
            for q in r["aux_log"]:
                q[key] = value
    s = analyze(a, b)
    assert s["verdict"] == "B9-RESCUE" and s["qualifier"] is None


def test_sham_gate_uses_actual_arm_dose_not_counterfactual_reference():
    a, b = b9()
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9f":
            for q in r["aux_log"]:
                q["grad_ratio"] = 1.
                q["grad_norm"] = 3.
    s = analyze(a, b)
    assert not s["sham_matching"]["passed"]
    assert s["sham_matching"]["medians"]["grad_ratio"] == 3.


def test_matching_drops_incomplete_pairs_and_keeps_registered_minimum():
    a, b = b9()
    next(r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9f")["aux_log"].pop()
    scores = analyze(a, b)
    assert scores["sham_matching"]["passed"]
    assert scores["sham_matching"]["n_pairs"] == 11
    for r in [r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9f"][1:3]:
        r["aux_log"].pop()
    assert not analyze(a, b)["sham_matching"]["passed"]


def test_two_barrier_secondary_and_selectivity_portability():
    a, b = b9()
    for r in a["cells"].values():
        if r["arm"] == "e2e1_b9":
            r["quick"][0][3] = .95
        if r["arm"] == "e2e1_lin_b9":
            r["cross_eval"]["ce"]["query_acc"] = .1
    s = analyze(a, b)
    assert s["two_barrier"]
    assert s["selectivity"]["label"] == "training-selective"
    assert s["selectivity"]["ce_portability"]["k"] == 0


def test_secondary_ce_minimum_is_not_a_negative_result():
    a, b = b9()
    for r in [r for r in a["cells"].values() if r["arm"] == "e2e1_b9"][-3:]:
        r["error"] = "infrastructure retry exhausted"
    assert analyze(a, b)["two_barrier"] == "NOT EVALUABLE"


def test_smoke_mode_is_explicit():
    a, b = b9()
    a["manifest"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke-only"):
        analyze(a, b)
    assert analyze(a, b, True)["smoke"]


def test_late_collapse_does_not_relabel_inconclusive():
    a, b = b9(rescue=False)
    for r in a["cells"].values():
        if r["arm"] == "e2e1_lin_b9":
            r["quick"][0][3] = .95
            r["probes"][-1]["addr_h"] = .1
    s = analyze(a, b)
    assert s["verdict"] == "B9-INCONCLUSIVE"
    assert s["manipulation_gate"] == "LATE-COLLAPSE" and s["reading"] is None


def test_diverged_cell_is_adverse_without_voiding_eleven_ok_cells():
    a, b = b9()
    r = next(r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9")
    r.update(diverged=True, probes=r["probes"][:32], quick=r["quick"][:32])
    s = analyze(a, b)
    assert s["manipulation_counts"] == {"OK": 11, "FAILED": 0, "LATE-COLLAPSE": 1, "MISSING": 0}
    assert s["verdict"] == "B9-RESCUE"
    assert s["counts"]["lin_b9"]["bound_final"] == dict(k=11, n=12, ci95=s["counts"]["lin_b9"]["bound_final"]["ci95"])


@pytest.mark.parametrize("rescue", [False, True])
def test_sham_minimum_only_suppresses_qualifiers(rescue):
    a, b = b9(rescue=rescue)
    for r in [r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9f"][:3]:
        r["error"] = "infrastructure"
    s = analyze(a, b)
    assert s["verdict"] == ("B9-RESCUE" if rescue else "B9-NO-RESCUE")
    assert s["specificity"]["p"] == 1. and s["qualifier"] is None
    assert s["holm2"]["rescue"] == min(1., 2*s["rescue"]["p"])


def test_no_rescue_does_not_need_baseline_comparison():
    a, b = b9(rescue=False)
    for r in b["cells"].values():
        if r["arm"] == "e2e1_lin":
            r["error"] = "infrastructure"
    assert analyze(a, b)["verdict"] == "B9-NO-RESCUE"


def test_selectivity_uses_ce_auxiliary_reference():
    a, b = b9()
    for r in a["cells"].values():
        if r["arm"] == "e2e1_b9":
            r["cross_eval"]["lin"]["query_acc"] = .95
    s = analyze(a, b)
    assert s["selectivity"]["label"] == "eval-carried"
    assert s["selectivity"]["k_C"] == 12
    assert all("label" not in row for row in s["common_eval_descriptive"].values())


def test_failed_tied_for_most_common_is_explicitly_conservative():
    a, b = b9()
    treated = [r for r in a["cells"].values() if r["arm"] == "e2e1_lin_b9"]
    for r in treated[:6]:
        r["probes"][0]["addr_h"] = .1
    for r in treated[6:]:
        r["probes"][-1]["addr_h"] = .1
    assert analyze(a, b)["manipulation_gate"] == "MANIPULATION FAILED"


def test_production_cli_is_blinded_until_unblind(tmp_path, monkeypatch, capsys):
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    path = Path(__file__).resolve().parents[1]/"scripts/analyze_b9.py"
    spec = importlib.util.spec_from_file_location("b9_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    a, b = b9()
    aa, bb = tmp_path/"b9.json", tmp_path/"frsb.json"
    aa.write_text(json.dumps(a)); bb.write_text(json.dumps(b))
    args = [str(path), str(aa), str(bb), str(tmp_path/"scores"), "--process-name", os.environ["PROCESS_NAME"]]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    public = capsys.readouterr().out
    assert public.startswith("B9_ANALYSIS_BLINDED ") and "RESCUE" not in public
    monkeypatch.setattr(sys, "argv", args+["--unblind"])
    cli.main()
    assert "B9-RESCUE" in capsys.readouterr().out
