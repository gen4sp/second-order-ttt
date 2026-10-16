"""Echo-in-text generator v2: tables, layout, leakage, controls and exposure (NumPy only)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from tttmem.p2.echo_text import (CONTROL_KINDS, TABLE_PATH, EchoLayout, build_eval_documents,
                                 check_vocab, inject_document, load_tables, make_injector,
                                 rows_sha256, write_mask)

P2DATA = Path(__file__).resolve().parents[1] / "data/p2data"
SPLITS = ("train", "cal", "test")


def excluded():
    path = P2DATA / "excluded_vocab.npy"
    return np.load(path) if path.exists() else np.zeros(128256, dtype=bool)


def synthetic_rows(n=12, T=8192, seed=0):
    a = np.random.default_rng(seed).integers(1000, 30000, size=(n, T + 1)).astype(np.uint32)
    a[:, 0] = 128000
    return a


def host_rows(split, n, offset=0):
    """Real DCLM rows when the local data copy exists, else synthetic rows."""
    path = P2DATA / f"{split}.bin"
    if not path.exists():
        return synthetic_rows(n, seed=len(split))
    arr = np.memmap(path, dtype=np.uint32, mode="r").reshape(-1, 8193)
    return np.asarray(arr[offset:offset + n])


def occ(row, seq):
    seq = np.asarray(seq)
    return set(np.flatnonzero((sliding_window_view(row, len(seq)) == seq).all(1)).tolist())


# --------------------------------------------------------------------------- tables
def test_tables_pools_disjoint_sized_and_clean():
    t = load_tables()
    assert (t.key_len, t.value_len) == (4, 3)
    keys = {s: {tuple(x) for x in t.keys[s].tolist()} for s in SPLITS}
    vals = {s: {tuple(x) for x in t.values[s].tolist()} for s in SPLITS}
    for s in SPLITS:
        assert len(keys[s]) == len(t.keys[s]) >= 4096 and len(vals[s]) == len(t.values[s]) >= 4096
        assert len(t.wrapper_text[s]["binding"]) >= (6 if s == "train" else 3)
        assert len(t.wrapper_text[s]["mention"]) >= 3 and len(t.wrapper_text[s]["query"]) >= 3
        # first value tokens are spread: none starts more than 8 values of a split
        assert np.bincount(t.values[s][:, 0]).max() <= 8
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not keys[a] & keys[b] and not vals[a] & vals[b]
    texts = [x for s in SPLITS for k in ("binding", "mention", "query") for x in t.wrapper_text[s][k]]
    assert len(set(texts)) == len(texts)
    wrapper_tok = {int(x) for s in SPLITS for ws in t.wrappers[s].values() for w in ws for p in w for x in p}
    pool_tok = {int(x) for s in SPLITS for x in np.concatenate((t.keys[s].ravel(), t.values[s].ravel()))}
    assert not wrapper_tok & pool_tok
    check_vocab(t, excluded())
    for s in SPLITS:
        assert all(v == 1.0 for v in t.stats["joint_canonical_match_rate"][s].values())


def test_tables_detect_tampering(tmp_path):
    raw = TABLE_PATH.read_text(encoding="utf-8").replace('"value_len":3', '"value_len":4', 1)
    (tmp_path / "t.json").write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_tables(str(tmp_path / "t.json"))


def test_check_vocab_rejects_excluded_pool_token():
    t = load_tables()
    ex = np.zeros(128256, dtype=bool)
    ex[t.values["test"][5, 1]] = True
    with pytest.raises(ValueError, match="excluded"):
        check_vocab(t, ex)


# --------------------------------------------------------------------------- layout
def test_layout_no_overlap_segments_and_order():
    lay = EchoLayout()
    lay.validate()
    assert lay.row_len == 8193 and lay.chunk == 1024
    spans = sorted(lay.spans())
    assert all(s1 >= e0 for (_, e0, _, _), (s1, _, _, _) in zip(spans, spans[1:]))
    seg = lay.segment
    assert (seg(lay.mention_starts) == 1).all() and (seg(lay.binding_starts) == 1).all()
    assert lay.mention_starts.max() + lay.max_mention <= lay.binding_starts.min()
    assert lay.binding_starts.max() + lay.max_binding <= lay.chunk
    assert lay.binding_starts.max() + lay.max_binding <= lay.no_barrier_query_starts.min()
    assert (seg(lay.no_barrier_query_starts + lay.max_query) == 1).all()
    assert (seg(lay.query_starts[:4]) == 3).all() and (seg(lay.query_starts[4:]) == 4).all()
    assert (seg(lay.no_binding_query_starts[:4]) == 4).all()
    assert (seg(lay.no_binding_query_starts[4:]) == 3).all()
    assert (seg(lay.query_entity_starts) == 2).all()


@pytest.mark.parametrize("bad", [dict(max_query=40), dict(max_binding=60), dict(S=1024, T=4096),
                                 dict(max_mention=100)])
def test_layout_validate_rejects_overlap_or_misplacement(bad):
    with pytest.raises(ValueError):
        replace(EchoLayout(), **bad).validate()


# --------------------------------------------------------------------------- leakage property
def _check_doc(host, out, rec, lay, controls):
    vals, dis = np.asarray(rec["value_tokens"]), np.asarray(rec["distractor_tokens"])
    keys = np.asarray(rec["key_tokens"])
    av = np.asarray(rec["available"], bool)
    spans = rec["spans"]
    start = {(k, i): s for s, e, k, i in spans}
    # host text untouched outside the spans; spans start where registered
    inside = np.zeros(len(out), bool)
    for s, e, k, i in spans:
        assert s == lay.starts(k)[i] and e - s <= lay.max_len(k)
        inside[s:e] = True
    assert np.array_equal(out[~inside], host[~inside])
    kinds = {k for _, _, k, _ in spans}
    assert kinds == ({"mention", "binding", "query"} | (set(CONTROL_KINDS) if controls else set()))
    for k in kinds:  # one paraphrase per kind and document: equal lengths across slots
        assert len({e - s for s, e, kk, _ in spans if kk == k}) == 1
    toks = np.concatenate((keys.ravel(), vals.ravel(), dis.ravel()))
    assert len(np.unique(toks)) == len(toks)          # all key/value tokens distinct
    reads = ["source", "query"] + (["no_barrier", "no_binding", "query_entity"] if controls else [])
    koff = rec["query_key_positions"][0][0] - start[("query", 0)]   # key offset in a query
    for i in range(lay.n_slots):
        b0 = start[("binding", i)]
        o = occ(out, vals[i])
        before = {x for x in o if x < b0}
        if av[i]:          # exactly once before the binding: in the slot's own mention
            m0 = start[("mention", i)]
            assert len(before) == 1 and m0 < min(before) < m0 + lay.max_mention
        else:              # never before the binding; the distractor only in the mention
            assert not before
            assert occ(out, dis[i]) == {rec["mention_positions"][i][0]}
        want = {rec[f"{r}_value_positions"][i][0] for r in reads}
        if av[i]:
            want.add(rec["mention_positions"][i][0])
        assert o == want   # the value occurs exactly at its registered positions
        for j in range(2):  # and so do its bigrams (no partial host collision)
            assert len(occ(out, vals[i][j:j + 2])) == len(want)
        kw = {rec["source_key_positions"][i][0], rec["query_key_positions"][i][0]}
        if controls:       # + no-barrier query i + query-entity control of target i-1
            kw |= {start[("no_barrier", i)] + koff,
                   start[("query_entity", (i - 1) % lay.n_slots)] + koff}
        assert occ(out, keys[i]) == kw
    if controls:
        for i, k in enumerate(np.asarray(rec["no_binding_key_tokens"])):
            assert occ(out, k) == {start[("no_binding", i)] + koff}
        assert np.array_equal(rec["query_entity_key_tokens"], np.roll(keys, -1, axis=0))
        assert np.array_equal(rec["query_entity_target_tokens"], vals)


@pytest.mark.parametrize("split,host_split", [("train", "A"), ("cal", "calib"), ("test", "ho")])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_no_leakage_property_all_splits(split, host_split, p):
    lay = EchoLayout()
    hosts = host_rows(host_split, 48, offset=256 if host_split == "ho" else 0)
    inj = make_injector(split=split, f=1.0, p=p, seed=7, excluded_vocab=excluded())
    ids = np.arange(100, 100 + len(hosts))
    ev, meta = inj.eval_documents(hosts, row_ids=ids)
    tr, tmeta = inj(hosts, ids)
    for j in range(len(hosts)):
        rec, trec = meta["rows"][j], tmeta["rows"][j]
        assert sum(rec["available"]) == round(8 * p)
        _check_doc(hosts[j], ev[j], rec, lay, controls=True)
        _check_doc(hosts[j], tr[j], trec, lay, controls=False)
        assert "no_binding_value_positions" not in trec


def test_host_collision_is_redrawn():
    lay = EchoLayout()
    host = host_rows("calib", 1)[0].copy()
    _, rec0 = inject_document(host, split="cal", seed=3, p=.5, row_id=11)
    assert rec0["attempt"] == 0
    host[3000:3002] = rec0["value_tokens"][2][1:3]     # plant a value bigram in the host text
    out, rec = inject_document(host, split="cal", seed=3, p=.5, row_id=11)
    assert rec["attempt"] >= 1 and rec["paraphrase"] == rec0["paraphrase"]
    assert rec["available"] == rec0["available"]
    _check_doc(host, out, rec, lay, controls=True)


def test_host_collisions_are_rare_on_real_text():
    hosts = host_rows("calib", 64)
    _, meta = build_eval_documents(hosts, split="cal", p=.5, seed=20261010, excluded_vocab=excluded())
    assert sum(r["attempt"] > 0 for r in meta["rows"]) <= 2


# --------------------------------------------------------------------------- strata / controls
def test_strata_token_count_matched_and_differ_only_in_mention_values():
    host = host_rows("calib", 1)[0]
    a, ma = inject_document(host, split="cal", seed=3, p=1.0, row_id=9)
    b, mb = inject_document(host, split="cal", seed=3, p=0.0, row_id=9)
    assert ma["spans"] == mb["spans"] and ma["value_tokens"] == mb["value_tokens"]
    diff = set(np.flatnonzero(a != b).tolist())
    mention = {int(x) for row in ma["mention_positions"] for x in row}
    assert diff <= mention and len(diff) >= 20
    assert np.array_equal(a[np.asarray(ma["mention_positions"])], np.asarray(ma["value_tokens"]))
    assert np.array_equal(b[np.asarray(mb["mention_positions"])], np.asarray(mb["distractor_tokens"]))
    assert np.array_equal(ma["source_nll_positions"], np.asarray(ma["source_value_positions"]) - 1)


def test_available_slots_vary_across_documents():
    hosts = host_rows("calib", 32)
    _, meta = build_eval_documents(hosts, split="cal", p=.5, seed=1, excluded_vocab=excluded())
    av = np.asarray([r["available"] for r in meta["rows"]])
    assert (av.sum(1) == 4).all() and 0.2 < av.mean(0).min() and av.mean(0).max() < 0.8


def test_write_mask_blocks_only_control_tokens():
    hosts = host_rows("calib", 2)
    _, meta = build_eval_documents(hosts, split="cal", p=.5, seed=1, excluded_vocab=excluded())
    m = write_mask(meta["rows"], 8192)
    for j, rec in enumerate(meta["rows"]):
        want = np.ones(8192, bool)
        for s, e, k, _ in rec["spans"]:
            if k in CONTROL_KINDS:
                want[s - 1:e - 1] = False
        assert np.array_equal(m[j], want)
        assert m[j][np.asarray(rec["source_nll_positions"])].all()
        assert m[j][np.asarray(rec["query_nll_positions"])].all()
        assert not m[j][np.asarray(rec["no_binding_nll_positions"])].any()
    assert write_mask([{"spans": [[10, 20, "binding", 0]]}], 64).all()


# --------------------------------------------------------------------------- stream / exposure
def test_injector_determinism_and_global_row_selection():
    base = synthetic_rows()
    kw = dict(split="train", f=.5, p=.5, seed=11, T=8192, excluded_vocab=excluded())
    a = make_injector(**kw)
    out1, meta1 = a(base, np.arange(len(base), dtype=np.int64))
    out2, meta2 = a(base[[3, 0, 7]], np.array([3, 0, 7], dtype=np.int64))
    assert np.array_equal(out1[[3, 0, 7]], out2)
    assert meta1["selected"][3] == meta2["selected"][0]
    assert rows_sha256(out1) == rows_sha256(a(base, np.arange(len(base)))[0])
    assert a.manifest() == make_injector(**kw).manifest()
    assert a.manifest()["manifest_sha256"] != make_injector(**{**kw, "p": 1.0}).manifest()["manifest_sha256"]
    assert np.array_equal(a.selected(np.arange(12)), np.asarray(meta1["selected"]))
    assert np.array_equal((out1 != base).any(1), np.asarray(meta1["selected"]))


def test_exposure_accounting_is_exact():
    inj = make_injector(split="train", f=.25, p=.5, seed=0, excluded_vocab=excluded())
    ids = np.arange(400, dtype=np.int64)
    host = np.full((len(ids), 8193), 128000, dtype=np.uint32)   # no injected token equals BOS
    out, _ = inj(host, ids)
    per_row = (out != host).sum(1)
    assert np.array_equal(per_row, inj.injected_tokens(ids))
    e = inj.exposure_accounting(ids)
    assert e["injected_rows"] == int((per_row > 0).sum())
    assert e["injected_token_budget"] == int(per_row.sum())
    assert e["copy_available_source_writes"] == e["copy_blocked_source_writes"] == e["injected_rows"] * 4 * 3
    big = inj.exposure_accounting(np.arange(19200, dtype=np.int64))
    assert big["injected_rows"] == 4835 and big["token_share"] < .05      # same rows as v1
    assert big["copy_available_source_writes"] >= 1000


def test_bad_shape_and_fraction_fail_fast():
    with pytest.raises(ValueError):
        make_injector(f=1.2, excluded_vocab=excluded())
    inj = make_injector(f=1, excluded_vocab=excluded())
    with pytest.raises(ValueError):
        inj(np.zeros((2, 8192), np.uint32), np.array([0, 1]))
