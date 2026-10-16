import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from tttmem.p2.mini_analysis import analyse
from tttmem.p2.mini_analysis import collect
from tttmem.p2.data import SEQ, open_split
from tttmem.p2.train import TrainCfg, checkpoint_steps


def _rows(delta, *, preset="mini-s", mix=False, code=False):
    so_arm = "e2e2-mix" if mix else "e2e2"
    fo_arm = "e2e1-ce-mix" if mix else "e2e1-ce"
    out = []
    for seed in range(3 if not mix else 2):
        so = .05
        fo = so - delta
        out += [{"preset": preset, "arm": so_arm, "seed": str(seed), "step": 1200,
                 "corpus": "code" if code else "dclm", "M": so, "G": .05, "strata": {}},
                {"preset": preset, "arm": fo_arm, "seed": str(seed), "step": 1200,
                 "corpus": "code" if code else "dclm", "M": fo, "G": .03, "strata": {}}]
    return out


def test_mini_analysis_changes_rules_and_pairs_seeds():
    rows = _rows(.02) + _rows(.08, mix=True) + _rows(.08, mix=True, code=True)
    r = analyse(rows)
    assert r["preflight_rules"]["R1"] in {"go", "ambiguous", "reconsider"}
    assert r["preflight_rules"]["R3"]["row"] in {"switch-to-mix", "mix-only-exposure", "stay-dclm"}
    assert r["t_critical"] == 1.886
    assert len(r["preset_stats"]["mini-s"]["pairs"]) == 3


def test_mini_analysis_reaches_r1_rows():
    go = analyse(_rows(.03))
    assert go["preflight_rules"]["R1"] == "go"
    null = _rows(.001)
    null[0]["M"] = null[2]["M"] = null[4]["M"] = .03
    null[1]["M"] = null[3]["M"] = null[5]["M"] = .029
    assert analyse(null)["preflight_rules"]["R1"] in {"reconsider", "ambiguous", "pending R3"}


def test_config_consistency_and_checkpoint_schedule():
    assert checkpoint_steps(TrainCfg(steps=1200), "so") == {600, 1200}
    assert checkpoint_steps(TrainCfg(steps=1200), "fo") == {1200}
    root = Path(__file__).parents[1] / "configs" / "mini"
    docs = []
    for path in sorted(root.glob("e2e*.toml")):
        with path.open("rb") as f:
            d = tomllib.load(f)["p2"]
        cfg = TrainCfg.from_dict(d)
        assert not cfg.assay_gate and not cfg.pilot_steps and not cfg.sizing_steps
        docs.append((cfg.lr_peak, cfg.warmup, cfg.steps, cfg.global_batch,
                     tuple(cfg.barriers), cfg.segment_len, cfg.fused_attn,
                     tuple(cfg.eval_steps), cfg.n_eval, cfg.ho_docs,
                     cfg.assay_gate, tuple(cfg.pilot_steps), tuple(cfg.sizing_steps)))
    assert len(set(docs)) == 1


def test_mini_analysis_synthetic_npz_reaches_rule_rows(tmp_path):
    # The on-disk layout is the one produced by train: preset/arm/seed/eval_step.
    def write_rows(root, rows):
        for row in rows:
            path = root / "mini-s" / row["arm"] / f"seed{row['seed']}"
            path.mkdir(parents=True, exist_ok=True)
            m = float(row["M"])
            np.savez(path / ("code_eval_step1200.npz" if row["corpus"] == "code"
                             else "eval_step1200.npz"),
                     doc_fP_focal=np.array([m + .1]), doc_fP_nonfocal=np.array([.1]),
                     doc_adapt=np.array([.2]), doc_noadapt=np.array([.2 + row["G"]]),
                     doc_heldout_noadapt=np.array([.3]), probe_stratum=np.array([0]),
                     tok_doc_adapt=np.array([.2]), tok_doc_noadapt=np.array([.3]))

    def core_rows(deltas):
        rows = []
        for seed, delta in enumerate(deltas):
            rows += [{"preset": "mini-s", "arm": "e2e2", "seed": str(seed), "step": 1200,
                      "corpus": "dclm", "M": .05, "G": .05, "strata": {}},
                     {"preset": "mini-s", "arm": "e2e1-ce", "seed": str(seed), "step": 1200,
                      "corpus": "dclm", "M": .05 - delta, "G": .03, "strata": {}}]
        return rows

    ambiguous = core_rows([-.02, .005, .02])
    cases = {
        "no-passing-presets": [{**r, "M": .01, "G": .01} for r in core_rows([.01, .01, .01])
                               if r["arm"] == "e2e2"],
        "go": core_rows([.03, .03, .03]),
        "reconsider": core_rows([.001, .001, .001]),
        "ambiguous": ambiguous,
        "stay-dclm": (core_rows([.001, .001, .001]) + _rows(.001, mix=True)
                      + _rows(.001, mix=True, code=True)),
        "switch-to-mix": (core_rows([.03, .03, .03]) + _rows(.08, mix=True)
                           + _rows(.08, mix=True, code=True)),
        "mix-only-exposure": (core_rows([.001, .001, .001]) + _rows(.001, mix=True)
                              + _rows(.08, mix=True, code=True)),
    }
    for expected, rows in cases.items():
        case_root = tmp_path / expected
        write_rows(case_root, rows)
        out = analyse(collect(case_root))
        assert {"R1", "R2", "R3"} <= set(out["preflight_rules"])
        if expected in {"go", "ambiguous"}:
            assert out["preflight_rules"]["R1"] == expected
        if expected == "reconsider":
            assert out["preflight_rules"]["R1"] in {"reconsider", "pending R3"}
        if expected == "no-passing-presets":
            assert out["preflight_rules"]["R1"] in {"no-passing-presets", "incomplete"}
        if expected == "reconsider":
            assert out["preflight_rules"]["R2"]["row"] == "drop-both"
        if expected == "go":
            assert out["preflight_rules"]["R2"]["row"] == "keep-both"
        if expected in {"stay-dclm", "switch-to-mix", "mix-only-exposure"}:
            assert out["preflight_rules"]["R3"]["row"] == expected


def test_mini_eval_diet_is_configured_and_defaults_stay_intact():
    root = Path(__file__).parents[1] / "configs" / "mini"
    for path in sorted(root.glob("e2e*.toml")):
        cfg = TrainCfg.from_dict(tomllib.loads(path.read_text())["p2"])
        assert cfg.eval_steps == [300, 600, 1200]
        assert cfg.full_eval_steps == [1200]
        assert cfg.trajectory_docs == 256 and cfg.ho_trajectory_docs == 64
        assert cfg.eval_modes == ["adapt", "noadapt", "fP_focal", "fP_nonfocal"]
        assert cfg.trajectory_modes == cfg.eval_modes
        assert cfg.full_extra_modes == []
        assert cfg.init_eval_docs == 256 and cfg.init_ho_docs == 64
        if "e2e2" in path.name:
            assert cfg.eval_at_init
        else:
            assert not cfg.eval_at_init
        if "mix" in path.name:
            assert cfg.code_eval_docs == 512 and cfg.code_eval_steps == [1200]
    default = TrainCfg()
    assert default.eval_at_init and default.eval_modes == ["adapt", "noadapt", "frozenP", "fP_focal", "fP_nonfocal"]
    assert default.full_eval_steps == [0, 400] and default.crossover_rules == ["ce", "lin", "aice"]


def test_resume_contract_excludes_process_and_storage_services():
    from dataclasses import replace
    from tttmem.p2.train import _resume_contract
    cfg = TrainCfg(process_name="TTM-min-32-s-e2e2-s0", measure_timing=True,
                   resume_dir="outputs/mini/resume/core-s/cell", resume=True,
                   stop_after=3, out_dir="/tmp/cell")
    contract = _resume_contract(cfg)
    assert "process_name" not in contract and "measure_timing" not in contract
    assert "resume_dir" not in contract and "resume" not in contract
    assert "stop_after" not in contract and "out_dir" not in contract


def test_matrix_budget_and_stale_helpers(tmp_path):
    from tttmem.p2.matrix import _can_start, _estimate_hours, _stale_cell
    out = tmp_path / "cell"
    out.mkdir(); (out / "steps.jsonl").write_text("partial\n")
    stale = _stale_cell(out, tmp_path / "resume")
    assert stale is not None and stale.name.startswith("cell.stale-") and not out.exists()
    cell = {"preset": "mini-s", "arm": "e2e1-rp", "init_seed": 0}
    cfg = {"max_hours": 1, "gpu_mem_budget_gb": 21,
           "podsmoke_hours": {"mini-s/e2e1-rp": .25},
           "memory_peaks_gb": {"mini-s/e2e1-rp": 10.67}}
    assert _estimate_hours(cell, cfg, [], workers=1) == .25
    assert _can_start(cell, cfg, [], 0., [6.], 2)
    assert not _can_start(cell, cfg, [], 0., [11.], 2)


def test_p2data_layout_is_read_only_and_mini_compatible(tmp_path):
    # P2-1 stores fixed-width uint32 rows plus row-aligned JSONL manifests.
    rng = np.random.default_rng(3)
    for split, n in (("A", 4), ("eval", 2), ("ho", 2), ("calib", 2)):
        rows = rng.integers(3, 97, size=(n, SEQ), dtype=np.uint32)
        rows[:, 0] = 1
        rows.tofile(tmp_path / f"{split}.bin")
        with (tmp_path / f"{split}.jsonl").open("w") as f:
            for i in range(n):
                f.write(json.dumps({"row": i, "sha1": f"{split}-{i}", "domain": "d"}) + "\n")
    np.save(tmp_path / "excluded_vocab.npy", np.zeros(97, dtype=np.bool_))
    before = (tmp_path / "A.bin").stat().st_mtime_ns
    rows, manifest = open_split(tmp_path, "A")
    assert rows.shape == (4, SEQ) and len(manifest) == 4
    assert not rows.flags.writeable
    assert (tmp_path / "A.bin").stat().st_mtime_ns == before


def test_mini_analysis_skips_stale_and_resume_folders(tmp_path):
    from tttmem.p2.mini_analysis import collect

    def write(path, m):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "eval_step1200.npz", doc_fP_focal=np.array([m + .1]),
                 doc_fP_nonfocal=np.array([.1]), doc_adapt=np.array([.2]),
                 doc_noadapt=np.array([.3]), doc_heldout_noadapt=np.array([.3]),
                 probe_stratum=np.array([0]))
    root = tmp_path / "matrix"
    write(root / "core-s" / "runs" / "mini-s" / "e2e2" / "s0", .05)
    write(root / "core-s" / "runs" / "mini-s" / "e2e2" / "s2.stale-20000101T000000.000000Z", .9)
    write(root / "resume" / "core-s" / "runs" / "mini-s" / "e2e2" / "s1", .9)
    rows = collect(root)
    assert [(r["arm"], r["seed"]) for r in rows] == [("e2e2", "0")]
    assert abs(rows[0]["M"] - .05) < 1e-9


def test_matrix_estimate_raises_without_podsmoke_hours():
    from tttmem.p2.matrix import _estimate_hours
    cell = {"preset": "mini-m", "arm": "e2e2", "steps": 1200}
    with pytest.raises(ValueError):
        _estimate_hours(cell, {}, [])
    assert _estimate_hours(cell, {"podsmoke_hours": {"mini-m/e2e2": 2.5}}, []) == 2.5
