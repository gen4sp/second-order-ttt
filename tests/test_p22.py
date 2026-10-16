"""P2-2 writer/reader swap: fast_init, diagonal reproduction, exact decomposition identities,
validity gate and write-once outputs."""

import json

import numpy as np
import pytest
import torch

from tttmem.p2.evaluate import eval_docs
from tttmem.p2.meta import InnerCfg, _fast_values, stream_eval
from tttmem.p2.model import PRESETS, build_model
from tttmem.p2.probe import find_probes
from tttmem.p2.swap import (SUPP, Cell, Tol, cells, classify, components, decompose, fast_for,
                            identical_arm_check, owen, quantity, reference_check, swap_eval,
                            write_states)

EXACT = Tol(floor=1e-5, cap=1e-4, mean_abs=1e-4, doc_max=1e-4, probe_q99=1e-4, probe_max=1e-4,
            total_budget=1e-5, total_cap=1e-4)


def _docs(preset, n, seed=0):
    cfg = PRESETS[preset]
    T = 8 * cfg.chunk
    rng = np.random.default_rng(seed)
    excluded = np.zeros(cfg.vocab_size, dtype=bool)
    excluded[[cfg.bos_id, cfg.eos_id]] = True
    docs, probes = [], []
    while len(docs) < n:
        d = rng.integers(3, 7, size=T + 1)
        d[0] = cfg.bos_id
        p = find_probes(d[:-1], d[1:], cfg.chunk, valid=d[1:] != cfg.bos_id, excluded=excluded,
                        seed=len(docs))
        if len(p.pos) and p.nf_ok and p.focal.any():
            docs.append(d)
            probes.append(p)
    return np.stack(docs), probes


def _models(preset, seeds=(0, 1)):
    return {a: build_model(preset, compute_dtype=torch.float32, seed=s).eval()
            for a, s in zip("sf", seeds)}


def test_fast_init_own_w0_is_identity():
    model = build_model("tiny2", compute_dtype=torch.float32, seed=0)
    docs, probes = _docs("tiny2", 2)
    t = torch.from_numpy(docs.astype(np.int64))
    ids, tgt = t[:, :-1], t[:, 1:]
    inner = InnerCfg()
    a = stream_eval(model, ids, tgt, tgt != 1, inner, barriers=[2, 4])["nll"]
    b = stream_eval(model, ids, tgt, tgt != 1, inner, barriers=[2, 4],
                    fast_init=_fast_values(model, 2))["nll"]
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        stream_eval(model, ids, tgt, tgt != 1, inner, fast_init=_fast_values(model, 1))
    # a crossed-start write from the writer's own W0 is the ordinary write
    f = torch.zeros(2, ids.shape[1], dtype=torch.bool)
    w1 = write_states(model, ids, tgt, tgt != 1, f, f, inner)
    w2 = write_states(model, ids, tgt, tgt != 1, f, f, inner, fast_init=_fast_values(model, 2))
    assert all(torch.equal(x, y) for k in SUPP for x, y in zip(w1[k], w2[k]))


def test_swap_diagonal_reproduces_saved_eval_modes():
    """Reader = writer cells equal the arm's own frozen-P / focal / non-focal / no-adapt NLL."""
    docs, probes = _docs("tiny2", 5)
    models = _models("tiny2")
    r = swap_eval(models, docs, probes, crossed=True, batch=2, device="cpu")
    assert len(r["cells"]) == len(cells(2, True)) == 2 * (2 * 7 + 2 * 3 + 3)
    refs, toks = {}, {}
    for a, m in models.items():
        e = eval_docs(m, docs, probes, "ce", modes=["noadapt", "frozenP", "fP_focal", "fP_nonfocal"],
                      batch=2, device="cpu")
        refs[a] = e["doc"]
        toks[a] = {k: np.concatenate(v) for k, v in e["tok"].items()}
        for k, mode in zip(SUPP, ("frozenP", "fP_focal", "fP_nonfocal")):
            np.testing.assert_allclose(r["doc"][f"r{a}x{a}y{a}_{k}"], refs[a][mode], atol=1e-5)
        np.testing.assert_allclose(r["doc"][f"r{a}x{a}y0_none"], refs[a]["noadapt"], atol=1e-5)
    chk = reference_check(r["doc"], refs, ("frozenP", "fP_focal", "fP_nonfocal", "noadapt"),
                          EXACT, tok=r["tok"], ref_tok=toks)
    assert chk["ok"] and {"total_M", "total_G", "total_L"} <= set(chk)
    assert chk["s_fP_focal"]["probe_abs_q50_q90_q99_max"][-1] < 1e-4
    # different models: crossing writer, reader or start changes the read
    assert not np.allclose(r["doc"]["rsxsys_none"], r["doc"]["rsxsyf_none"])
    assert not np.allclose(r["doc"]["rsxsys_none"], r["doc"]["rfxfys_none"])
    assert not np.allclose(r["doc"]["rsxsyf_none"], r["doc"]["rsxsyfc_none"])
    assert not np.allclose(r["doc"]["rsxsys+0_none"], r["doc"]["rsxsys_none"])
    assert np.isfinite(r["geo"]["dw_cos"]).all() and (r["geo"]["dw_norm_s"] > 0).all()


def test_reference_check_requires_every_mode():
    doc = {c.name: np.zeros(4) for c in cells(1)}
    ref = {a: {"noadapt": np.zeros(4)} for a in "sf"}
    chk = reference_check(doc, ref, ("frozenP", "noadapt"), Tol())
    assert not chk["ok"] and chk["missing"] == ["s_frozenP", "f_frozenP"]
    ref = {a: {"noadapt": np.zeros(4), "frozenP": np.full(4, 0.01)} for a in "sf"}
    chk = reference_check(doc, ref, ("frozenP", "noadapt"), Tol())
    assert not chk["ok"] and not chk["s_frozenP"]["ok"] and chk["s_noadapt"]["ok"]


def test_reference_check_rejects_rare_gross_errors():
    """Counterexample: one document off by +-5.12 nats inflates the SE; the tail bounds
    must still reject it (it would move the total M by 0.02)."""
    n = 1024
    modes = ("frozenP", "fP_focal", "fP_nonfocal", "noadapt")
    doc = {c.name: np.zeros(n) for c in cells(3)}
    ref = {a: {m: np.zeros(n) for m in modes} for a in "sf"}
    for a, sign in (("s", 1.0), ("f", -1.0)):
        ref[a]["fP_focal"][-1] = -5.12 * sign
        ref[a]["fP_nonfocal"][-1] = 5.12 * sign
    chk = reference_check(doc, ref, modes, Tol())
    assert not chk["ok"] and not chk["s_fP_focal"]["ok"] and chk["s_frozenP"]["ok"]
    assert abs(chk["total_M"]["mean"]) == pytest.approx(0.02, rel=1e-6) and not chk["total_M"]["ok"]
    # small, many-document noise of the calibrated size passes
    rng = np.random.default_rng(0)
    ref = {a: {m: rng.normal(0, 4e-3, n) for m in modes} for a in "sf"}
    assert reference_check(doc, ref, modes, Tol())["ok"]


def test_fast_for_routes_sentinel_tensors():
    """Distinct W0 / delta sentinels: every cell type gets exactly the registered fast weights."""
    W0b = {"s": [torch.full((1,), 1.0 + i) for i in range(6)],
           "f": [torch.full((1,), 10.0 + i) for i in range(6)]}
    k_off = {"none": 0.0, "F": 0.1, "NF": 0.2}
    W4 = {a: {k: [w + (100.0 if a == "s" else 1000.0) + o for w in W0b[a]] for k, o in k_off.items()}
          for a in "sf"}
    W4c = {("f", "s"): {k: [torch.full((1,), 7.0 + o)] * 6 for k, o in k_off.items()},
           ("s", "f"): {k: [torch.full((1,), 8.0 + o)] * 6 for k, o in k_off.items()}}
    val = lambda ts: [float(t) for t in ts]  # noqa: E731
    assert fast_for(Cell("x", "s", "s", "0", "none"), W0b, W4) is None
    assert val(fast_for(Cell("x", "s", "f", "0", "none"), W0b, W4)) == val(W0b["f"])
    assert val(fast_for(Cell("x", "f", "f", "f", "F"), W0b, W4)) == val(W4["f"]["F"])
    # reader s with its own W0 and FO's delta: W0_s + (W4_f - W0_f) = W0_s + 1000 + k
    assert val(fast_for(Cell("x", "s", "s", "f", "NF"), W0b, W4)) == pytest.approx(
        [1.0 + i + 1000.2 for i in range(6)])
    # reader f with SO's W0 and SO's delta: W0_s + 100 + k
    assert val(fast_for(Cell("x", "f", "s", "s", "none"), W0b, W4)) == pytest.approx(
        [1.0 + i + 100.0 for i in range(6)])
    # per-layer: reader s, own delta, layer 1 (tensors 3..5) from FO
    got = val(fast_for(Cell("x", "s", "s", "s", "F", 1), W0b, W4))
    assert got == pytest.approx([1.0 + i + 100.1 for i in range(3)] + [1.0 + i + 1000.1 for i in range(3, 6)])
    # crossed start: W4c is keyed (writer, reader); writer f started from reader s's W0
    assert val(fast_for(Cell("x", "s", "s", "f", "F", None, True), W0b, W4, W4c)) == pytest.approx([7.1] * 6)
    assert val(fast_for(Cell("x", "f", "f", "s", "NF", None, True), W0b, W4, W4c)) == pytest.approx([8.2] * 6)


def test_swap_with_identical_arms_is_a_no_op():
    docs, probes = _docs("tiny2", 3, seed=1)
    m = build_model("tiny2", compute_dtype=torch.float32, seed=0).eval()
    r = swap_eval({"s": m, "f": m}, docs, probes, crossed=True, batch=3, device="cpu")
    for c in cells(2, True):
        ref = r["doc"]["rsxsy0_none"] if c.Y == "0" else r["doc"][f"rsxsys_{c.k}"]
        np.testing.assert_allclose(r["doc"][c.name], ref, atol=1e-6)
    np.testing.assert_allclose(r["geo"]["dw_cos"], 1.0, atol=1e-9)
    e = eval_docs(m, docs, probes, "ce", modes=["noadapt", "frozenP", "fP_focal", "fP_nonfocal"],
                  batch=3, device="cpu")
    ref = {k: np.asarray(v) for k, v in e["doc"].items()}
    ref_tok = {k: np.concatenate(v) for k, v in e["tok"].items()}
    chk = identical_arm_check(r["doc"], r["tok"], ref, ref_tok, 2, EXACT, crossed=True)
    assert chk["ok"] and abs(chk["components_M"]["total"]) < 1e-6
    # a symmetric routing error (cross cells read the non-focal state) is caught per cell
    bad = dict(r["doc"], rsxsyf_F=r["doc"]["rsxsys_NF"], rfxfys_F=r["doc"]["rsxsys_NF"])
    bad_tok = dict(r["tok"], rsxsyf_F=r["tok"]["rsxsys_NF"], rfxfys_F=r["tok"]["rsxsys_NF"])
    chk = identical_arm_check(bad, bad_tok, ref, ref_tok, 2, EXACT, crossed=True)
    assert not chk["ok"] and not chk["rsxsyf_F"]["ok"] and chk["rsxsys_F"]["ok"]
    dec = decompose(r["doc"], np.array(["a", "b", "c"]), 2, n_boot=100, crossed=True)
    for kind in ("M", "G", "L"):
        assert abs(dec[kind]["total"]["est"]) < 1e-6 and abs(dec[kind]["write"]["est"]) < 1e-6
        assert dec[kind]["class"] == "no-deficit" and dec[kind]["write_share"]["est"] is None


def _random_doc(n=50, n_fast=3, seed=0):
    rng = np.random.default_rng(seed)
    return {c.name: rng.normal(2.0, 0.3, n) for c in cells(n_fast, True)}


def test_decomposition_identities():
    doc = _random_doc()
    for kind in ("M", "G", "L"):
        c = components(doc, kind, 3, crossed=True)
        np.testing.assert_allclose(c["write"] + c["read"], c["total"], atol=1e-12)
        np.testing.assert_allclose(c["owen_delta"], c["write"], atol=1e-12)
        np.testing.assert_allclose(c["owen_slow"] + c["owen_w0"], c["read"], atol=1e-12)
        np.testing.assert_allclose(c["write_c"] + c["read_c"], c["total"], atol=1e-12)
        np.testing.assert_allclose(c["write_at_reader_s"] - c["write_at_reader_f"],
                                   c["interaction"], atol=1e-12)
        total = quantity(doc, kind, "s", "s", "s") - quantity(doc, kind, "f", "f", "f")
        np.testing.assert_allclose(c["total"], total, atol=1e-12)
    # L' orientation: total = D = L_ff - L_ss on the frozen-P read
    c = components(doc, "L", 3)
    np.testing.assert_allclose(c["total"], doc["rfxfyf_none"] - doc["rsxsys_none"], atol=1e-12)
    # additive Q: each factor's own effect
    phi = owen(lambda R, X, Y: (R == "s") * 1.0 + (X == "s") * 2.0 + (Y == "s") * 4.0)
    assert (phi["slow"], phi["w0"], phi["delta"]) == pytest.approx((1.0, 2.0, 4.0))
    # pure three-way interaction: the grouped split is 1/2 : 1/2 and Owen keeps it
    phi = owen(lambda R, X, Y: float(R == X == Y == "s"))
    assert (phi["delta"], phi["slow"] + phi["w0"]) == pytest.approx((0.5, 0.5))
    assert phi["slow"] == pytest.approx(0.25) and phi["w0"] == pytest.approx(0.25)


def test_classify_rule():
    b = lambda lo, hi: {"lo": lo, "hi": hi}  # noqa: E731
    pos = b(0.01, 0.02)
    assert classify(pos, b(0.01, 0.02), b(-0.01, 0.01), b(0.001, 0.03)) == "write-dominant"
    assert classify(pos, b(-0.01, 0.01), b(0.01, 0.02), b(-0.03, -0.001)) == "read-dominant"
    assert classify(pos, b(0.01, 0.02), b(0.01, 0.02), b(-0.01, 0.01)) == "both"
    assert classify(pos, b(-0.01, 0.02), b(-0.01, 0.02), b(-0.01, 0.01)) == "unresolved"
    # FO writes more useful content but reads worse: the deficit is read-located
    assert classify(pos, b(-0.02, -0.01), b(0.02, 0.03), b(-0.05, -0.03)) == "read-dominant"
    assert classify(b(-0.001, 0.02), b(0.01, 0.02), b(-0.01, 0.0), b(0.001, 0.03)) == "no-deficit"


def test_decompose_detects_a_planted_read_deficit():
    n = 400
    rng = np.random.default_rng(3)
    doc = {c.name: rng.normal(2.0, 0.01, n) for c in cells(3, True)}
    # focal memory 0.1 for SO readers, 0.06 for FO readers, regardless of writer, W0 or start
    for c in cells(3, True):
        if c.k == "F" and c.Y != "0":
            doc[c.name] = doc[c.name] + (0.1 if c.R == "s" else 0.06)
    clusters = np.array([f"d{i % 97}" for i in range(n)])
    dec = decompose(doc, clusters, 3, n_boot=1000, crossed=True)
    M = dec["M"]
    assert M["class"] == "read-dominant" and M["class_c"] == "read-dominant"
    assert abs(M["total"]["est"] - 0.04) < 0.005
    assert abs(M["owen_slow"]["est"] - 0.04) < 0.005 and abs(M["owen_w0"]["est"]) < 0.005
    assert M["flags"]["read_pos"] and not M["flags"]["write_pos"]
    assert not M["flags"]["sign_reversal"] and abs(M["write_share"]["est"]) < 0.2
    assert dec["n_clusters"] == 97


def _two_tiny_arms(tmp_path):
    from test_p2 import _smoke_data
    from tttmem.p2.train import TrainCfg, train
    data, T = _smoke_data(tmp_path)
    for arm in ("e2e2", "e2e1-ce"):
        train(TrainCfg(arm=arm, preset="tiny", dtype="fp32", device="cpu", seq_len=T,
                       global_batch=4, micro_batch=2, steps=2, warmup=1, lr_peak=1e-1,
                       data_dir=str(data), out_dir=str(tmp_path / arm), n_eval=4, min_probes=1,
                       eval_steps=[0, 1, 2], full_eval_steps=[0, 2], trajectory_docs=2,
                       sizing_steps=[], assay_gate=False, ho_docs=3, ho_trajectory_docs=2,
                       eval_batch=2, pilot_steps=[1], pilot_docs=2, pilot_max_rise=1e9,
                       log_every=1, save_checkpoints=True, crossover_rules=["ce"]))
    n = len(np.load(tmp_path / "e2e2" / "eval_meta.npz")["eval_rows"])
    return data, T, n


def _cfg(tmp_path, data, T, n, **kw):
    from tttmem.p2.swap import SwapCfg
    return SwapCfg(so_dir=str(tmp_path / "e2e2"), fo_dir=str(tmp_path / "e2e1-ce"),
                   data_dir=str(data), out_dir=str(tmp_path / "swap"), preset="tiny", dtype="fp32",
                   device="cpu", seq_len=T, primary_step=2, primary_docs=n, trajectory_steps=[1],
                   trajectory_docs=2, ref_docs=2, batch=2, n_boot=200, **kw)


def test_swap_run_end_to_end(tmp_path, capsys):
    """Two tiny P2-1 arms with checkpoints -> swap run: validity, write-once outputs."""
    from tttmem.p2.swap import run
    data, T, n = _two_tiny_arms(tmp_path)
    cfg = _cfg(tmp_path, data, T, n)
    capsys.readouterr()
    res = run(cfg)
    out = capsys.readouterr().out
    assert res["status"] == "valid" and res["steps_done"] == [1, 2]
    assert res["class_M"] in ("no-deficit", "write-dominant", "read-dominant", "both", "unresolved")
    for step, k in ((2, n), (1, 2)):
        j = json.loads((tmp_path / "swap" / f"swap_step{step:04d}.json").read_text())
        v = j["validity"]
        assert j["n_docs"] == k and j["status"] == "valid" and v["ok"]
        assert {"s_frozenP", "f_noadapt", "total_L"} <= set(v["saved_reference"])
        assert {"s_fP_focal", "f_fP_nonfocal", "total_M"} <= set(v["same_gpu_reference"])
        assert (v["identical_arm_control"] is not None) == (step == 2)
        z = np.load(tmp_path / "swap" / f"swap_step{step:04d}.npz")
        assert len(z["tok_rsxfys_F"]) == z["probe_offset"][-1] == len(z["probe_stratum"])
        assert ("doc_rsxsyfc_F" in z) == (step == 2)
    # the step-1 saved evaluation has no focal modes: the saved check covers frozenP / noadapt
    j1 = json.loads((tmp_path / "swap" / "swap_step0001.json").read_text())
    assert "s_fP_focal" not in j1["validity"]["saved_reference"]
    decomp = [json.loads(l[11:]) for l in out.splitlines() if l.startswith("P22_DECOMP ")]
    assert len(decomp) == 6
    assert [d["role"] for d in decomp if d["step"] == 2] == ["confirmatory", "descriptive", "descriptive"]
    assert "class_c" in decomp[0] and "P22_RESULT" in out
    with pytest.raises(FileExistsError):
        run(cfg)


def test_swap_run_invalid_reference_publishes_no_decomposition(tmp_path, capsys):
    from tttmem.p2.swap import run
    data, T, n = _two_tiny_arms(tmp_path)
    p = tmp_path / "e2e2" / "eval_step0002.npz"
    z = dict(np.load(p))
    z["doc_fP_focal"] = z["doc_fP_focal"] + 0.05      # a reference the swap cannot reproduce
    np.savez(p, **z)
    capsys.readouterr()
    res = run(_cfg(tmp_path, data, T, n))
    out = capsys.readouterr().out
    assert res["status"] == "invalid" and "class_M" not in res and res["steps_done"] == [2]
    assert "P22_DECOMP" not in out and "P22_ABORT" in out
    j = json.loads((tmp_path / "swap" / "swap_step0002.json").read_text())
    assert j["status"] == "invalid" and "decomposition" not in j
    assert not j["validity"]["saved_reference"]["s_fP_focal"]["ok"]


def test_swap_preflight_rejects_partial_outputs_and_missing_modes(tmp_path):
    from tttmem.p2.swap import run
    data, T, n = _two_tiny_arms(tmp_path)
    (tmp_path / "swap").mkdir()
    (tmp_path / "swap" / "swap_step0001.npz").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        run(_cfg(tmp_path, data, T, n))
    (tmp_path / "swap" / "swap_step0001.npz").unlink()
    p = tmp_path / "e2e1-ce" / "eval_step0002.npz"
    z = {k: v for k, v in np.load(p).items() if k not in ("doc_noadapt", "tok_noadapt")}
    np.savez(p, **z)
    with pytest.raises(ValueError, match="lacks required modes"):
        run(_cfg(tmp_path, data, T, n))
    assert not (tmp_path / "swap" / "swap_step0002.json").exists()


def test_swap_preflight_checks_input_identities(tmp_path):
    from tttmem.p2.swap import run
    data, T, n = _two_tiny_arms(tmp_path)
    # saved rows reordered
    p = tmp_path / "e2e2" / "eval_step0001.npz"
    orig = dict(np.load(p))
    np.savez(p, **dict(orig, eval_rows=orig["eval_rows"][::-1].copy()))
    with pytest.raises(ValueError, match="misaligned"):
        run(_cfg(tmp_path, data, T, n))
    np.savez(p, **orig)
    # a checkpoint of the wrong arm under the right file name
    ck = tmp_path / "e2e1-ce" / "model_step0002.pt"
    good = torch.load(ck, weights_only=True)
    torch.save(dict(good, arm="e2e2"), ck)
    with pytest.raises(ValueError, match="is not e2e1-ce step 2"):
        run(_cfg(tmp_path, data, T, n))
    torch.save(good, ck)
    # the current data manifest no longer matches eval_meta
    man = (data / "eval.jsonl").read_text().splitlines()
    rec = json.loads(man[0])
    rec["sha1"] = "changed"
    (data / "eval.jsonl").write_text("\n".join([json.dumps(rec)] + man[1:]) + "\n")
    rows = np.load(tmp_path / "e2e2" / "eval_meta.npz")["eval_rows"]
    if 0 in rows:
        with pytest.raises(ValueError, match="manifest differs"):
            run(_cfg(tmp_path, data, T, n))
    assert not list((tmp_path / "swap").glob("swap_step*"))


def test_swap_invalid_trajectory_step_is_skipped_not_fatal(tmp_path, capsys):
    from tttmem.p2.swap import run
    data, T, n = _two_tiny_arms(tmp_path)
    p = tmp_path / "e2e1-ce" / "eval_step0001.npz"
    z = dict(np.load(p))
    np.savez(p, **dict(z, doc_frozenP=z["doc_frozenP"] + 0.05))
    capsys.readouterr()
    res = run(_cfg(tmp_path, data, T, n))
    out = capsys.readouterr().out
    assert res["status"] == "valid" and res["steps_invalid"] == [1] and "class_M" in res
    assert "P22_STEP_INVALID" in out
    assert [json.loads(l[11:])["step"] for l in out.splitlines() if l.startswith("P22_DECOMP ")] == [2, 2, 2]
    j = json.loads((tmp_path / "swap" / "swap_step0001.json").read_text())
    assert j["status"] == "invalid" and "decomposition" not in j
