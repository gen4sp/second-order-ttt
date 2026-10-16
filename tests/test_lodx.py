import copy
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pytest

from tttmem import exp_lod as l
from tttmem import lodx_analysis as x


def cfg(**updates):
    c = tomllib.loads(Path("configs/toy/smoke/lod-x.toml").read_text())["lod"]
    c.update(updates)
    return c


def row(arm, load, seed, a, a1500=None, retained=True):
    c = dict(group="core", arm=arm, load=load, seed=seed, status="COMPLETE",
             composite={k: a for k in l.ENDPOINT_NAMES}, acquired=25, retained=retained,
             logs=[{"step": s, "quick": {"acc": .8 + .00001*s}} for s in range(25, 3001, 25)],
             layout_audit={"valid": True}, observed={"W0": .06}, B_nb=.2,
             completed_steps=3000, eval_sha256="same")
    c["cell_id"] = l.cell_id(c)
    if a1500 is not None:
        c.update(composite_1500={k: a1500 for k in l.ENDPOINT_NAMES}, retained_1500=retained,
                 acquired_1500=25, eval_step_1500=1500,
                 details_1500={"layout_audit": {"valid": True}, "eval_sha256": "same", "B_nb": .2,
                              "probes_sha256": "dummy"})
    return c


def test_x_manifest_is_exact_72_cells():
    cells = l.manifest_cells("X")
    assert len(cells) == 72
    assert {c["arm"] for c in cells} == {"lin", "so"}
    assert {c["load"] for c in cells} == {16, 32, 64}
    assert {c["seed"] for c in cells} == set(range(216, 228))
    assert all(c["group"] == "core" and c["layout"] == "main" for c in cells)


def test_lr_schedule_invariant_through_1500():
    short = cfg(steps=1500, lr_total=3000, full_eval_steps=[1500])
    long = cfg(steps=3000, lr_total=3000, full_eval_steps=[1500, 3000])
    for s in range(1, 1501):
        short_lr = l.t.lr_scale(s - 1, short["warmup"], short["lr_total"], short["lr_floor"])
        long_lr = l.t.lr_scale(s - 1, long["warmup"], long["lr_total"], long["lr_floor"])
        assert short_lr == long_lr
    assert l.full_eval_steps(short) == [1500]
    with pytest.raises(AssertionError):
        l.full_eval_steps(cfg(steps=3001, full_eval_steps=[1500, 3001]))


def test_x_admission_accounts_two_evaluations_and_cap():
    receipt = dict(cells=[dict(arm=a, load=n, layout="main", seed=0, ms_step=ms, peak_gb=2.)
                    for a in ("lin", "so") for n, ms in ((16, 32), (32, 40), (64, 50))],
                   workers=5, total_gb=24.)
    result = l.timing_admission_x(receipt, cfg(budget_usd=3., steps=3000, lr_total=3000,
                                                full_eval_steps=[1500, 3000], tier_order=["X"]))
    assert result["full_eval_steps"] == [1500, 3000]
    assert result["full_eval_equivalent_steps"] > 0
    assert result["tier"] == "X"
    assert l.timing_admission_x(receipt, cfg(budget_usd=0.0001, steps=3000, lr_total=3000,
                                             full_eval_steps=[1500, 3000], tier_order=["X"]))["tier"] is None


def test_x1_verdicts_and_plateau_window():
    cells = [row(a, n, s, .91 if a == "so" else .90, .89 if a == "lin" else .90)
             for n in (16, 32, 64) for s in range(216, 228) for a in ("lin", "so")]
    # A synthetic late rise in four cells is still plateau (budget-limited requires >3).
    for c in cells[:4]:
        c["logs"] = [{"step": s, "quick": {"acc": .85 if 2500 <= s <= 2700 else .90}}
                      for s in range(25, 3001, 25)]
    result = x.score(cells)
    assert result["X1"]["label"] == "FREE"
    assert result["Q_plat"]["core/lin/32"]["label"] == "plateau"


@pytest.mark.parametrize("values,label", [([-.01]*12, "FREE"), ([-.1]*12, "PRICED"),
                                           ([-.05]*12, "INCONCLUSIVE")])
def test_x1_verdict_logic(values, label):
    assert x.ni(values)["label"] == label


def test_diagnostic_analyzer_writes_contract_outputs(tmp_path):
    manifest = dict(config=cfg(), config_sha256=l.sha(cfg()),
                    source_hashes=l.source_hashes(), plan_sha256=None,
                    admission={"tier": "diagnostic", "addon": False}, created_before_production=True,
                    cells=[])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    # Empty diagnostic set exercises verdict shape without manufacturing checkpoints.
    result = x.analyze_directory(tmp_path, diagnostic=True)
    assert result["mode"].startswith("SMOKE ONLY") and result["X1"]["label"] == x.NE
    x.write_outputs(tmp_path, result, tmp_path / "analysis")
    assert all((tmp_path / "analysis" / name).exists() for name in
               ("lodx_scores.json", "lodx_scores.txt", "lodx_verdicts.csv"))


def test_verdict_bounds_follow_lod_convention():
    from tttmem import lod_analysis as lod
    values = [-.30, -.25, -.41, -.18, -.33, -.29, -.35, -.22, -.40, -.27, -.31, -.36]
    row, lod_row = x.ni(values), lod.interval(values)
    assert row["LB"] == pytest.approx(lod_row["LB"]) and row["UB"] == pytest.approx(lod_row["UB"])
    assert row["quantile"] == .95
    assert row["LB95_two_sided"] < row["LB"] and row["UB95_two_sided"] > row["UB"]
