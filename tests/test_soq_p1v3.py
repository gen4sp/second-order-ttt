from pathlib import Path

import numpy as np
import pytest
import tomllib

from tttmem.p2.echo_text import EchoLayoutV3, make_injector_v3, write_mask_v3
from tttmem.p2.soq_p1v3_analysis import registered_rules


_EXCLUDED = Path(__file__).resolve().parents[1] / "data/p2data/excluded_vocab.npy"
if not _EXCLUDED.exists():
    pytest.skip("needs the prepared DCLM pack (data/p2data/excluded_vocab.npy)", allow_module_level=True)
EXCLUDED = np.load(_EXCLUDED, allow_pickle=False)


def _inj(**kw):
    return make_injector_v3(T=8192, S=2048, excluded_vocab=EXCLUDED, seed=17,
                            split="test", f=1.0, p=.5, lookup_share=.25, **kw)


def test_v3_determinism_layout_and_verbatim_reads():
    inj = _inj(); host = np.full(inj.layout.row_len, 100000, dtype=np.uint32)
    a, ma = inj.document(host, 23, controls=True)
    b, mb = inj.document(host, 23, controls=True)
    assert np.array_equal(a, b) and ma == mb
    assert inj.layout.n_slots == 16 and inj.layout.validate()
    assert set(ma["read_value_positions"]) == {"cross_verbatim", "lookup_verbatim"}
    assert ma["query_mode"] == "verbatim"
    assert all(not np.array_equal(np.asarray(ma["key_tokens"])[i], np.asarray(ma["key_tokens"])[j])
               for i in range(16) for j in range(i))
    toks = [tuple(x) for x in ma["key_tokens"] + ma["value_tokens"] + ma["distractor_tokens"]]
    flat = [x for row in toks for x in row]
    assert len(flat) == len(set(flat))


def test_v3_masks_block_reads_and_binding_tokens_for_a():
    inj = _inj(); host = np.full((3, inj.layout.row_len), 100000, dtype=np.uint32)
    _, meta = inj.eval_documents(host, row_ids=np.arange(3))
    n = write_mask_v3(meta["rows"], inj.layout.T, forward="N")
    a = write_mask_v3(meta["rows"], inj.layout.T, forward="A")
    expected = sum(e - s for rec in meta["rows"] for s, e, k, _ in rec["spans"]
                   if k in ("cross_verbatim", "lookup_verbatim"))
    assert (~n).sum() == expected
    for i, rec in enumerate(meta["rows"]):
        for s, e, k, _ in rec["spans"]:
            if k in ("cross_verbatim", "lookup_verbatim"):
                assert not n[i, s-1:e-1].any() and not a[i, s-1:e-1].any()
            if k == "binding":
                assert n[i, s-1:e-1].all() and not a[i, s-1:e-1].any()


def test_v3_memory_lookup_share_and_fallback_report():
    inj = make_injector_v3(T=8192, S=2048, excluded_vocab=EXCLUDED, seed=0,
                           f=.5, p=.5, lookup_share=.25, split="train")
    ids = np.arange(5000); sel = inj.selected(ids); kinds = [inj.kind(int(i)) for i in ids[sel]]
    assert .20 < kinds.count("lookup") / len(kinds) < .30
    small = EchoLayoutV3(T=8192, S=2048, chunk=512, n_slots=16)
    assert small.n_slots in (12, 16)
    if small.n_slots == 12:
        assert small.slot_fallback


def _arrays(n=80):
    av = np.tile(np.arange(16) < 8, (n, 1)); z = {"echo_available": av}
    for f, r in (("N", "cross_verbatim"), ("A", "cross_verbatim"), ("N", "lookup_verbatim")):
        em = np.zeros((n, 16), dtype=bool)
        if f == "N" and r == "lookup_verbatim": em[:] = True
        if f == "N" and r == "cross_verbatim": em[:, 8:] = True
        z[f"echo_{f}_{r}_em"] = em; z[f"echo_{f}_{r}_nll"] = np.ones((n, 16))
        z[f"echo_{f}_{r}_top1_16"] = em.copy()
        z[f"echo_{f}_{r}_first_token_rank"] = np.where(em, 1, 16)
        z[f"echo_{f}_{r}_segment"] = np.tile(np.array([2]*5+[3]*5+[4]*6), (n, 1))
    z["echo_version"] = np.asarray(3)
    return z


def test_v3_analysis_rules():
    z = _arrays(); r = registered_rules(z)
    assert r["C_plus"]["pass"] and r["V2_prime"]["pass"]
    assert r["verdict"] == "PASS"
    z["echo_N_cross_verbatim_em"][:, 8:] = False
    assert registered_rules(z)["verdict"] == "NOT-LEARNABLE"
    z = _arrays(); z["echo_N_lookup_verbatim_em"][:] = False
    assert registered_rules(z)["verdict"] == "LOOKUP-NOT-LEARNED"
    z = _arrays(); z["echo_N_cross_verbatim_top1_16"][:] = False
    assert registered_rules(z)["verdict"] == "PASS-UNQUALIFIED"


def test_v3_config_so_fo_diff():
    with open("configs/soq/p1v3-so.toml", "rb") as f: so = tomllib.load(f)["p2"]
    with open("configs/soq/p1v3-fo.toml", "rb") as f: fo = tomllib.load(f)["p2"]
    assert so["echo_version"] == fo["echo_version"] == 3
    assert so["arm"] == "e2e2" and fo["arm"] == "e2e1-ce"
    assert so["echo_manifest_sha256"] == fo["echo_manifest_sha256"]
