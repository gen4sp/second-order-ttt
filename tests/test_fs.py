import math

import numpy as np

from tttmem.p2.mini import G0_READS
from tttmem.p2.fs_analysis import FO_READS, SO_READS, analyse, interpolate, main


def test_soq_fs_eta_names_are_registered_generically():
    assert G0_READS["eta500"] == (5.0, "none", None)
    assert G0_READS["eta600"] == (6.0, "none", None)


def test_fs_interpolate_rejects_extrapolation_and_interpolates_memory():
    points = [
        {"name": "ce", "eta": 1.0, "L": 1.0, "M": 0.0},
        {"name": "eta200", "eta": 2.0, "L": 2.0, "M": 0.4},
    ]
    out = interpolate(points, 1.5)
    assert out["eta"] == 1.5
    assert out["M"] == 0.2
    assert interpolate(points, 0.5) is None
    assert interpolate(points, 2.5) is None
    assert math.isfinite(out["eta"])


def test_fs_analysis_smoke_outputs(tmp_path):
    """The local analysis entry point consumes the GPU job's per-document NPZ shape."""
    for seed in range(3):
        for arm, names in (("e2e2", SO_READS), ("e2e1-ce", FO_READS)):
            cell = tmp_path / "mini-s" / arm / f"s{seed}"
            cell.mkdir(parents=True)
            for read in names:
                eta = float(G0_READS[read][0])
                # A monotone synthetic frontier puts SO's target inside the FO grid.
                l = 1.0 + 0.25 * eta if arm == "e2e1-ce" else 1.5
                m = (0.1 * eta) if arm == "e2e1-ce" else 1.0
                np.savez_compressed(
                    cell / f"g0_read_{read}.npz",
                    M_doc=np.full(1024, m),
                    L_adapt_sum=np.full(128, l),
                    L_adapt_count=np.ones(128),
                    L_noadapt_doc=np.full(128, 2.0 + 0.01 * seed),
                    L_noadapt=np.asarray(2.0 + 0.01 * seed),
                    eta=np.asarray(eta),
                )
    result = analyse(tmp_path)
    assert result["analysis_a"]["rho_match"]["n"] == 3
    assert result["analysis_b"]["fixed_rho"]["n"] == 3
    assert result["comparison"]["verdict"] in {"TREND-FLAT", "TREND-RISING", "TREND-FALLING"}
    out = tmp_path / "report"
    main([str(tmp_path), "--out", str(out)])
    assert (out / "fs_analysis.json").exists()
    assert (out / "data" / "mini-s" / "e2e2" / "s0" / "g0_read_ce.npz").exists()
