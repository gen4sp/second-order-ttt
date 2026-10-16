"""Synthetic outcomes only; unit labels are not RNG seeds."""
import copy
import pytest
from tttmem.frsb_analysis import analyze, cp, holm, selectivity


def row(arm, unit, ever=False, final=False, addr=50):
    steps = list(range(25, 1501, 25))
    quick = [[s, 1., 1., .95 if ever and s == 25 else (.95 if final and s == 1500 else .2)] for s in steps]
    return dict(arm=arm, seed=unit, block="core", trained=True, steps=1500,
                quick=quick, probes=[dict(step=s, addr_a=.2, addr_h=1.) for s in steps],
                eval=dict(query_acc=.95 if final else .2), t_addr_lost=None if ever else addr,
                cross_eval={k: dict(query_acc=.95 if final else .2) for k in ("ce", "lin")})


def payload(n=12, kind="code"):
    units = [f"unit-{i:02}" for i in range(n)]
    cells = {}
    for i, u in enumerate(units):
        for arm in ("e2e1", "e2e1_lin", "e2e1emb", "e2e1emb_lin", "e2e2"):
            good = arm in ("e2e1emb", "e2e1emb_lin", "e2e2") or arm == "e2e1_lin" and kind == "alone"
            cells[f"{u}-{arm}"] = row(arm, u, good, good)
    return dict(manifest=dict(smoke=False, seeds=units), cells=cells)


@pytest.mark.parametrize("n", [12, 11, 10])
def test_code_needed_address_trigger_and_scaled_gates(n):
    out = analyze(payload(n))
    assert out["scores"]["core"]["verdict"] == "CODE-NEEDED"
    assert out["scores"]["core"]["sublabel"] == "ADDRESS"
    assert out["b9_gate"]["run_b9"]


@pytest.mark.parametrize("n", [0, 1, 9])
def test_below_minimum_no_b9(n):
    out = analyze(payload(n))
    assert out["scores"]["core"]["verdict"] == "NOT EVALUABLE"
    assert not out["b9_gate"]["run_b9"]


def test_lin_alone_precedence_and_no_b9():
    out = analyze(payload(kind="alone"))
    assert out["scores"]["core"]["verdict"] == "LIN-ALONE"
    assert not out["b9_gate"]["run_b9"]


def test_nontrigger_gate_does_not_unblind_the_verdict():
    alone, inconclusive = payload(kind="alone"), payload()
    for r in inconclusive["cells"].values():
        if r["arm"] == "e2e1emb_lin":
            r["eval"]["query_acc"] = .1
    assert analyze(alone)["b9_gate"] == analyze(inconclusive)["b9_gate"] == \
        dict(run_b9=False, reason="registered trigger not met")


def test_indeterminate_gate_reports_its_reason_without_outcome_values():
    gate = analyze(payload(9))["b9_gate"]
    assert not gate["run_b9"] and "fewer than 10" in gate["reason"]
    assert set(gate) == {"run_b9", "reason"}


def test_production_cli_is_gate_only_until_explicit_unblind(tmp_path, monkeypatch, capsys):
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    script = Path(__file__).resolve().parents[1]/"scripts/analyze_frsb.py"
    spec = importlib.util.spec_from_file_location("frsb_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path/"input.json"
    path.write_text(json.dumps(payload()))
    args = [str(script), str(path), str(tmp_path/"out"), "--process-name",
            os.environ.get("PROCESS_NAME", "TTM-frb-76-tests")]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert capsys.readouterr().out.startswith("B9_GATE ")
    monkeypatch.setattr(sys, "argv", args+["--unblind"])
    cli.main()
    assert "FRSB_ANALYSIS " in capsys.readouterr().out


def test_retention_failure_denominator_is_acquired_cells():
    p = payload()
    rows = [r for r in p["cells"].values() if r["arm"] == "e2e1_lin"]
    for i, r in enumerate(rows):
        r["quick"][0][3] = .95 if i < 6 else .2
        r["eval"]["query_acc"] = .95 if i == 0 else .2
    out = analyze(p)["scores"]["core"]
    assert out["verdict"] == "ACQUIRES-NOT-RETAINS"
    assert (out["acquired"], out["acquired_retention_failures"]) == (6, 5)
    rows[5]["quick"][0][3] = .2
    assert analyze(p)["scores"]["core"]["verdict"] != "ACQUIRES-NOT-RETAINS"


def test_e_lin_competence_blocks_code_needed():
    p = payload()
    for i, r in enumerate(r for r in p["cells"].values() if r["arm"] == "e2e1emb_lin"):
        if i >= 8:
            r["eval"]["query_acc"] = .2
    s = analyze(p)
    assert s["scores"]["core"]["verdict"] == "INCONCLUSIVE"
    assert s["scores"]["core"]["reason"] == "no competent +emb-lin reference"
    assert not s["b9_gate"]["run_b9"]


def test_address_uses_unbound_lin_cells_and_other_no_trigger():
    p = payload()
    for i, r in enumerate(r for r in p["cells"].values() if r["arm"] == "e2e1_lin"):
        if i < 4:
            r["t_addr_lost"] = 225
    s = analyze(p)
    assert s["scores"]["core"]["sublabel"] == "OTHER"
    assert not s["b9_gate"]["run_b9"]


def test_missing_cells_are_missing_and_pairs_need_ten():
    p = payload()
    for i, r in enumerate(r for r in p["cells"].values() if r["arm"] == "e2e1_lin"):
        if i >= 9:
            r["error"] = "infrastructure retry exhausted"
    out = analyze(p)
    assert out["scores"]["core"]["counts"]["e2e1_lin"]["bound_ever"]["n"] == 9
    assert out["scores"]["core"]["verdict"] == "NOT EVALUABLE"
    assert len([u for u in out["units"] if u["disposition"] == "MISSING"]) == 3


def test_nine_ce_pairs_only_block_the_row_that_needs_ce():
    p = payload()
    for arm, missing in (("e2e1_lin", {"unit-10", "unit-11"}),
                         ("e2e1", {"unit-09", "unit-11"})):
        for r in p["cells"].values():
            if r["arm"] == arm and r["seed"] in missing:
                r["error"] = "infrastructure"
    s = analyze(p)["scores"]["core"]
    assert s["lin_vs_ce"]["n"] == 9 and s["verdict"] == "CODE-NEEDED"
    for r in p["cells"].values():
        if r["arm"] == "e2e1_lin":
            r["eval"]["query_acc"] = .95
            r["quick"][0][3] = .95
    assert analyze(p)["scores"]["core"]["verdict"] == "NOT EVALUABLE"


def test_divergence_is_adverse_observation():
    p = payload(kind="alone")
    r = next(r for r in p["cells"].values() if r["arm"] == "e2e1_lin")
    r.update(diverged=True, quick=r["quick"][:2], probes=r["probes"][:2])
    s = analyze(p)
    c = s["scores"]["core"]["counts"]["e2e1_lin"]["bound_final"]
    assert (c["n"], c["k"]) == (12, 11)


def test_truncated_grid_is_missing():
    p = payload()
    next(iter(p["cells"].values()))["probes"].pop(4)
    assert analyze(p)["units"][0]["disposition"] == "TRUNCATED"


def test_selectivity_common_lin_is_separate_from_ce_portability():
    p = payload(kind="alone")
    for r in p["cells"].values():
        if r["arm"] == "e2e1_lin":
            r["cross_eval"]["ce"]["query_acc"] = .1
    s = analyze(p)["scores"]["core"]["selectivity"]
    assert s["label"] == "training-selective"
    assert s["ce_portability"]["k"] == 0
    for r in p["cells"].values():
        if r["arm"] == "e2e1":
            r["cross_eval"]["lin"]["query_acc"] = .95
    assert analyze(p)["scores"]["core"]["selectivity"]["label"] == "eval-carried"


def test_missing_cross_evals_more_than_two_unknown():
    p = payload(kind="alone")
    plain = [r for r in p["cells"].values() if r["arm"] in ("e2e1", "e2e1_lin")]
    for r in plain[:3]:
        r["cross_eval"].pop("lin")
    s = analyze(p)["scores"]["core"]["selectivity"]
    assert s["label"] == "unknown" and s["missing_cross_evals"] == 3


def test_scaled_one_of_twelve_gate_is_zero_below_twelve():
    for n, expected in ((12, True), (11, False), (10, False)):
        p = payload(n)
        r = next(r for r in p["cells"].values() if r["arm"] == "e2e1")
        r["quick"][0][3] = .95
        assert analyze(p)["scores"]["core"]["P_CE"] == expected


def test_smoke_mode_never_triggers():
    p = payload()
    p["manifest"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke-only"):
        analyze(p)
    assert not analyze(p, True)["b9_gate"]["run_b9"]


def test_cp_and_holm_fixed_family():
    assert cp(0, 12)[1] == pytest.approx(1-.025**(1/12))
    assert holm({"a": .01, "b": .03}) == {"a": .02, "b": .03}


def test_duplicate_units_rejected():
    p = payload()
    p["cells"]["duplicate"] = copy.deepcopy(next(iter(p["cells"].values())))
    with pytest.raises(ValueError, match="duplicate"):
        analyze(p)


def test_lin_alone_does_not_need_emb_reference_minimum():
    p = payload(kind="alone")
    for r in [r for r in p["cells"].values() if r["arm"] == "e2e1emb_lin"][:3]:
        r["error"] = "infrastructure"
    assert analyze(p)["scores"]["core"]["verdict"] == "LIN-ALONE"


def test_divergence_preserves_preceding_observed_address_loss():
    p = payload()
    r = next(r for r in p["cells"].values() if r["arm"] == "e2e1_lin")
    r.update(diverged=True, probes=r["probes"][:8])
    assert analyze(p)["b9_gate"]["run_b9"]
    assert next(u for u in analyze(p)["units"] if u["arm"] == r["arm"] and u["seed"] == r["seed"])["t_addr_lost"] == 50


def test_diverged_nonfinite_accuracy_serialized_as_null_is_adverse():
    p = payload(kind="alone")
    r = next(r for r in p["cells"].values() if r["arm"] == "e2e1_lin")
    r.update(diverged=True, eval=dict(query_acc=None))
    r["quick"][-1][3] = None
    s = analyze(p)["scores"]["core"]
    assert s["counts"]["e2e1_lin"]["bound_final"]["k"] == 11


def test_family_writer_with_pinned_frs_a_score_reader(tmp_path):
    import json
    import math
    from tttmem.frsb_analysis import write_scores
    p = payload()
    # R-LIN uses final full eval; R-ADDR uses ever-bound on the quick grid.
    for r in p["cells"].values():
        if r["arm"] == "e2e1emb":
            r["eval"]["query_acc"] = .2
    scores = analyze(p)
    write_scores(tmp_path, scores)
    frsb = json.loads((tmp_path/"frsb_scores.json").read_text())
    # Minimal pinned copy of the FRS-A score() reader, plus
    # exact_mcnemar/paired_verdict from frs_cohorts.py.
    def score(frsb):
        family = {name: dict(p=1.) for name in ("R-SO", "R-NOISE", "R-RP")}
        for name in ("R-LIN", "R-ADDR"):
            source = (frsb or {}).get("family_inputs", {}).get(name)
            test, control = source["test"], source["comparator"]
            b = sum(t and not c for t, c in zip(test, control))
            c = sum(not t and c for t, c in zip(test, control))
            p = 1. if b+c == 0 else sum(math.comb(b+c, k) for k in range(b, b+c+1))/2**(b+c)
            family[name] = dict(n=len(test), test=sum(test), comparator=sum(control), p=p if len(test)>=10 else 1.)
        adjusted = holm({name: r["p"] for name, r in family.items()})
        for name in ("R-LIN", "R-ADDR"):
            r = family[name]
            r["verdict"] = "NOT EVALUABLE" if r["n"]<10 else "REPLICATED" if adjusted[name]<=.05 and r["test"]>=math.ceil(.75*r["n"]) and r["comparator"]<=math.floor(.25*r["n"]) else "INCONCLUSIVE"
        return family
    family = score(frsb)
    assert family["R-LIN"]["verdict"] == family["R-ADDR"]["verdict"] == "REPLICATED"
    for name in ("R-LIN", "R-ADDR"):
        source = scores["family_inputs"][name]
        assert source["seeds"] == sorted(p["manifest"]["seeds"])
        assert all(len(v) == 12 for v in source.values())
    r = next(r for r in p["cells"].values() if r["arm"] == "e2e1emb")
    r["diverged"] = True
    assert analyze(p)["family_inputs"]["R-ADDR"]["diverged"][0]


def test_result_json_contains_family_inputs(tmp_path, monkeypatch):
    import json
    from tttmem import exp_frsb as f
    p = payload()
    monkeypatch.setattr(f, "prepare", lambda cfg: (p["manifest"], tmp_path))
    p["manifest"]["plan_sha256"] = "fixture"
    monkeypatch.setattr(f, "execute", lambda *args: p["cells"])
    f.run({})
    assert json.loads((tmp_path/"result.json").read_text())["family_inputs"] == analyze(p)["family_inputs"]
