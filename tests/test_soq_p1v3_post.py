from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tttmem.p2.echo_text import write_mask_v3
from tttmem.p2.soq_p1v3 import (build_echo_eval_set, build_echo_train_templates,
                                within_stratum_rank)
from tttmem.p2.soq_p1v3_post_analysis import analyse, v3x_labels


DATA = Path(__file__).resolve().parents[1] / "data/p2data"
needs_pack = pytest.mark.skipif(not (DATA / "excluded_vocab.npy").exists(),
                                reason="needs the prepared DCLM pack (data/p2data/excluded_vocab.npy)")


def _cfg(n=2):
    return SimpleNamespace(data_dir=str(DATA), echo_eval_split="ho", echo_eval_offset=256,
                           echo_eval_docs=n, echo_eval_p=.5, echo_lookup_share=.25,
                           echo_seed=0, seq_len=8192, segment_len=2048, echo_slots=16)


def _excluded():
    return np.load(DATA / "excluded_vocab.npy", allow_pickle=False)


@needs_pack
def test_z_and_p_a_masks_are_exact_unions():
    es = build_echo_eval_set(_cfg(), _excluded(), forward_masks=True)
    recs = []
    # The builder's metadata is deterministic; reconstruct the expected masks from spans.
    from tttmem.p2.echo_text import make_injector_v3
    inj = make_injector_v3(split="test", f=1, p=.5, lookup_share=.25, seed=0, T=8192,
                           S=2048, n_slots=16, excluded_vocab=_excluded())
    hosts, _ = __import__("tttmem.p2.data", fromlist=["open_split"]).open_split(DATA, "ho")
    _, meta = inj.eval_documents(np.asarray(hosts[256:258, :8193]), row_ids=np.arange(256, 258), query_mode="verbatim")
    _, meta_p = inj.eval_documents(np.asarray(hosts[256:258, :8193]), row_ids=np.arange(256, 258), query_mode="paraphrase")
    recs = meta["rows"]; recs_p = meta_p["rows"]
    expected_z = np.zeros_like(es.write_masks["Z"])
    expected_pa = np.zeros_like(es.write_masks["P-A"])
    for i, rec in enumerate(recs):
        expected_z[i, :2048] = True
        for s, e, kind, _ in rec["spans"]:
            if kind in ("cross_verbatim", "lookup_verbatim"):
                expected_z[i, max(0, s - 1):e - 1] = True
    for i, rec in enumerate(recs_p):
        for s, e, kind, _ in rec["spans"]:
            if kind in ("cross_paraphrase", "lookup_verbatim", "binding"):
                expected_pa[i, max(0, s - 1):e - 1] = True
    assert np.array_equal(~es.write_masks["Z"], expected_z)
    assert np.array_equal(~es.write_masks["P-A"], expected_pa)


@needs_pack
def test_train_templates_are_paired_and_training_layout_has_no_lookup():
    c = _cfg(2); ex = _excluded()
    a = build_echo_train_templates(c, ex, controls=True)
    b = build_echo_train_templates(c, ex, controls=True)
    t = build_echo_train_templates(c, ex, controls=False)
    assert np.array_equal(a.row_ids, t.row_ids) and np.array_equal(a.row_ids, np.arange(256, 258))
    assert np.array_equal(a.docs, b.docs) and a.docs_sha256 == b.docs_sha256
    assert "lookup_verbatim" in a.positions and "lookup_verbatim" not in t.positions
    assert np.array_equal(~t.write_masks["T"], np.zeros_like(t.write_masks["T"]))
    assert int((~t.write_masks["T-A"]).sum()) > 0


def test_within_stratum_rank_and_slot_order():
    av = np.asarray([[1, 1, 0, 0]], dtype=bool)
    c = np.asarray([[[4, 3, 2, 1], [0, 5, 4, 3], [9, 8, 7, 6], [6, 5, 4, 8]]], dtype=float)
    rank, top = within_stratum_rank(c, av)
    assert rank.tolist() == [[1, 1, 1, 1]]
    assert top.all()


def test_post_analysis_synthetic_and_v3x_label(tmp_path):
    av = np.tile(np.arange(16) < 8, (4, 1)); z = {"checkpoint_step": np.asarray(3200),
        "echo_test_available": av, "echo_train_available": av}
    for set_name, fwds, reads in (("test", ("N", "A", "Z", "P", "P-A"), ("cross_verbatim", "lookup_verbatim", "cross_paraphrase")),
                                  ("train", ("N", "A", "Z", "T", "T-A"), ("cross_verbatim",))):
        for fwd in fwds:
            for read in reads:
                if read == "cross_paraphrase" and fwd not in ("P", "P-A"): continue
                p = f"echo_{set_name}_{fwd}_{read}"; x = np.zeros((4, 16), bool); x[:, :8] = fwd in ("N", "T", "P")
                for suffix, value in (("R", x), ("top1_16", x), ("top1_8", x), ("first_nll", np.ones((4,16))),
                                      ("tokens23_nll", np.ones((4,16))*2), ("value_nll", np.ones((4,16)))): z[f"{p}_{suffix}"] = value
                z[f"{p}_segment"] = np.tile([2]*5+[3]*5+[4]*6, (4,1)); z[f"{p}_read_order"] = np.tile(np.arange(16), (4,1))
    path = tmp_path / "post_step3200.npz"; np.savez(path, **z)
    out = analyse([path]); assert "test_cross_R_N_minus_A" in out["checkpoints"]["3200"]["contrasts"]
    eval_keys = {"echo_available": av, "echo_N_cross_verbatim_em": np.ones((4,16)),
              "echo_A_cross_verbatim_em": np.zeros((4,16)), "echo_N_cross_verbatim_top1_16": np.ones((4,16)),
              "echo_N_cross_verbatim_within_stratum_top1_8": np.ones((4,16))*.2,
              "echo_A_cross_verbatim_within_stratum_top1_8": np.zeros((4,16))}
    assert v3x_labels(eval_keys)["label"] == "LATE-STORAGE"
