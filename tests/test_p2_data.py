"""Offline, model-free checks of Phase 2 document preparation."""

import hashlib
import io
import json
from pathlib import Path
import sys
import types
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from tttmem.p2 import data as D


def char_tokenizer(texts):
    return [[ord(c) for c in text] for text in texts]


def whitespace_tokenizer(texts):
    return [[int(word) for word in text.split()] for text in texts]


def document(text, row=0, url="https://WWW.Example.COM:443/page"):
    return {"text": text, "url": url, "shard": "fake.parquet", "row": row}


def expected_split(text, quotas, seed=0):
    digest = hashlib.sha1(text.encode()).hexdigest()
    if seed:
        digest = hashlib.sha1(f"{seed}:{digest}".encode()).hexdigest()
    u = int(digest[:15], 16) / 16**15
    cumulative = 0
    for split in sorted(quotas):
        cumulative += quotas[split] / sum(quotas.values())
        if u < cumulative:
            return split


def for_split(tokens, split, quotas, seed=0):
    """Change only a trailing token, keeping planted overlaps and the stored prefix."""
    prefix = " ".join(map(str, tokens))
    for nonce in range(10000):
        text = prefix + f" {900000 + nonce}"
        if expected_split(text, quotas, seed) == split:
            return document(text)
    raise AssertionError("could not find a document in requested split")


def test_filters_exact_dedup_and_layout(tmp_path):
    calls = []

    def tokenize(texts):
        calls.extend(texts)
        return char_tokenizer(texts)

    short = document("tiny")
    too_few = document("a" * (D.SEQ - 2), row=1)
    exact = document("b" * (D.SEQ - 1), row=2)
    long = document("c" * (D.SEQ + 200), row=3)
    meta = D.prepare([short, too_few, exact, exact, long], tokenize, tmp_path,
                     {"A": 8}, min_chars=100, log_every=0)
    assert short["text"] not in calls
    assert len(calls) == 3
    assert meta["counts"] == {"A": 2}
    stats = meta["acceptance"]
    assert stats["docs_scanned"] == 5
    assert stats["passed_char_prefilter"] == 4
    assert stats["duplicates"] == 1
    assert stats["passed_token_length"] == 2
    assert stats["chars_per_token_median"] == 1.0
    assert stats["shards_used"] == ["fake.parquet"]
    assert json.loads((tmp_path / "meta.json").read_text()) == meta

    rows, manifest = D.open_split(tmp_path, "A")
    assert isinstance(rows, np.memmap)
    assert rows.dtype == np.uint32 and rows.shape == (2, D.SEQ)
    assert rows.flags.c_contiguous and not rows.flags.writeable
    np.testing.assert_array_equal(rows[:, 0], [D.BOS, D.BOS])
    np.testing.assert_array_equal(rows[0, 1:], np.full(D.SEQ - 1, ord("b")))
    np.testing.assert_array_equal(rows[1, 1:], np.full(D.SEQ - 1, ord("c")))
    assert (tmp_path / "A.bin").stat().st_size == 2 * D.SEQ * 4
    for i, doc in enumerate((exact, long)):
        record = manifest[i]
        assert record["row"] == i and record["source_row"] == doc["row"]
        assert record["sha1"] == hashlib.sha1(doc["text"].encode()).hexdigest()
        assert record["n_chars"] == len(doc["text"])
        assert record["n_tokens_full"] == len(doc["text"]) + 1
        assert record["domain"] == "example.com"
    assert not list(tmp_path.glob(".prepare-*"))


def test_whitespace_filter_and_optional_mask(tmp_path):
    few = document("1 " * (D.SEQ - 2))
    enough = document("2 " * (D.SEQ - 1))
    mask = np.array([False, True], dtype=np.bool_)
    meta = D.prepare([few, enough], whitespace_tokenizer, tmp_path, {"A": 2},
                     min_chars=100, exclude=mask, log_every=0)
    assert meta["acceptance"]["passed_token_length"] == 1
    assert meta["acceptance"]["chars_per_token_median"] == 2.0
    rows, manifest = D.open_split(tmp_path, "A")
    assert rows.shape == (1, D.SEQ)
    assert manifest[0]["n_tokens_full"] == D.SEQ
    np.testing.assert_array_equal(np.load(tmp_path / "excluded_vocab.npy"), mask)


@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_assignment_and_fixed_quotas(tmp_path, seed):
    quotas = {"eval": 2, "calib": 1, "A": 3, "B": 2}
    docs = []
    for split in quotas:
        for j in range(quotas[split] + 2):
            base = 10000 + len(docs) * 10000
            doc = for_split(range(base, base + D.SEQ), split, quotas, seed)
            doc["row"] = len(docs)
            docs.append(doc)
    outputs = [tmp_path / "one", tmp_path / "two"]
    for out, ordered_quotas in zip(outputs, (quotas, dict(reversed(list(quotas.items()))))):
        meta = D.prepare(iter(docs), whitespace_tokenizer, out, ordered_quotas,
                         min_chars=1, seed=seed, batch_size=2, log_every=0)
        assert meta["counts"] == quotas
        assert meta["acceptance"]["rejected_full_split"] > 0
        for split in quotas:
            rows, manifest = D.open_split(out, split)
            assert len(rows) == len(manifest) == quotas[split]
            expected = [d for d in docs if expected_split(d["text"], quotas, seed) == split]
            assert [m["sha1"] for m in manifest] == [
                hashlib.sha1(d["text"].encode()).hexdigest() for d in expected[:quotas[split]]]
    for split in quotas:
        for suffix in ("bin", "jsonl"):
            assert (outputs[0] / f"{split}.{suffix}").read_bytes() == \
                   (outputs[1] / f"{split}.{suffix}").read_bytes()


def overlap_docs(overlap, position=32, holdout="eval"):
    quotas = {holdout: 1, "A": 1}
    evaluation = list(range(10000, 10000 + D.SEQ - 1))
    training = list(range(40000, 40000 + D.SEQ - 1))
    # position is measured after prepending BOS; the eval window starts at 101.
    training[position - 1:position - 1 + overlap] = evaluation[100:100 + overlap]
    return quotas, for_split(evaluation, holdout, quotas), for_split(training, "A", quotas)


@pytest.mark.parametrize("holdout", ["eval", "calib"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("overlap,flag", [(64, True), (63, False)])
def test_near_duplicate_length_and_arrival_order(tmp_path, holdout, reverse, overlap, flag):
    quotas, evaluation, training = overlap_docs(overlap, holdout=holdout)
    docs = [evaluation, training]
    if reverse:
        docs.reverse()
    meta = D.prepare(docs, whitespace_tokenizer, tmp_path, quotas, min_chars=1,
                     near_dup_stride=1, log_every=0)
    _, manifest = D.open_split(tmp_path, holdout)
    assert manifest[0]["near_dup"] is flag
    assert meta["near_dup_counts"] == {holdout: int(flag)}


@pytest.mark.parametrize("position,stride,flag", [(32, 32, True), (33, 32, False), (33, 1, True)])
def test_training_stride_samples_bos_prefixed_window_starts(tmp_path, position, stride, flag):
    quotas, evaluation, training = overlap_docs(64, position=position)
    D.prepare([evaluation, training], whitespace_tokenizer, tmp_path, quotas,
              min_chars=1, near_dup_stride=stride, log_every=0)
    assert D.open_split(tmp_path, "eval")[1][0]["near_dup"] is flag


def test_all_holdouts_sharing_a_shingle_are_flagged(tmp_path):
    quotas = {"eval": 2, "calib": 1, "A": 1}
    shared = list(range(2000, 2064))
    docs = []
    for i, split in enumerate(["eval", "eval", "calib", "A"]):
        tokens = list(range(10000 + i * 20000, 10000 + i * 20000 + D.SEQ - 1))
        tokens[127:191] = shared
        docs.append(for_split(tokens, split, quotas))
    meta = D.prepare(docs, whitespace_tokenizer, tmp_path, quotas, min_chars=1, log_every=0)
    assert meta["near_dup_counts"] == {"calib": 1, "eval": 2}


def test_rolling_hash_matches_scalar_reference():
    tokens = np.arange(10, 180, dtype=np.uint32)
    actual = D._shingle_hashes(tokens, 64)
    expected = [sum((int(t) + 1) * pow(D._HASH_BASE, j, 2**64)
                    for j, t in enumerate(tokens[i:i + 64])) % 2**64
                for i in range(len(tokens) - 63)]
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.uint64))
    np.testing.assert_array_equal(D._shingle_hashes(tokens, 64, 7), actual[::7])
    assert len(D._shingle_hashes(tokens[:63])) == 0


def test_excluded_vocab_unicode_and_specials():
    decoded = ["", " \n\t", "...", "—«»", "$+🙂 \t", "a!", "猫", "42", "e\u0301"]

    class FakeTokenizer:
        def get_vocab_size(self):
            return len(decoded)

        def decode(self, ids):
            return decoded[ids[0]]

    mask = D.excluded_vocab(FakeTokenizer(), D.BOS + 2)
    assert mask.dtype == np.bool_ and mask.shape == (D.BOS + 2,)
    np.testing.assert_array_equal(mask[:len(decoded)],
                                  [True, True, True, True, True, False, False, False, False])
    assert mask[len(decoded):].all()
    assert D.excluded_vocab(FakeTokenizer(), 0).shape == (0,)


@pytest.mark.parametrize("url,expected", [
    ("https://WWW.Example.COM:443/a", "example.com"),
    ("http://user:password@WWW.Example.org/x", "example.org"),
    ("www.Example.net/path", "example.net"),
    ("//WWW.Example.io/path", "example.io"),
    ("https://sub.www.example.com", "sub.www.example.com"),
    ("https://[::1]:8080", "::1"), ("https://[broken", ""),
    ("", ""), (None, ""), ("https:///path", ""),
])
def test_domains(url, expected):
    assert D._domain(url) == expected


def test_zero_quotas_do_not_consume_source(tmp_path):
    def docs():
        raise AssertionError("source consumed despite zero quotas")
        yield

    meta = D.prepare(docs(), char_tokenizer, tmp_path, {"eval": 0, "A": 0}, log_every=0)
    assert meta["acceptance"]["docs_scanned"] == 0
    assert meta["acceptance"]["chars_per_token_median"] is None
    rows, manifest = D.open_split(tmp_path, "eval")
    assert isinstance(rows, np.memmap) and rows.shape == (0, D.SEQ) and manifest == []
    assert rows.mode == "r" and not rows.flags.writeable


def test_source_closes_at_quota_completion(tmp_path):
    closed = []

    def docs():
        try:
            yield document("a" * D.SEQ)
            raise AssertionError("source consumed after quota completion")
        finally:
            closed.append(True)

    D.prepare(docs(), char_tokenizer, tmp_path, {"A": 1}, batch_size=1,
              min_chars=1, log_every=0)
    assert closed == [True]


def test_open_split_rejects_misalignment(tmp_path):
    D.prepare([document("a" * D.SEQ)], char_tokenizer, tmp_path, {"A": 1}, log_every=0,
              min_chars=1)
    with (tmp_path / "A.bin").open("ab") as f:
        f.write(b"extra")
    with pytest.raises(ValueError, match="aligned"):
        D.open_split(tmp_path, "A")
    (tmp_path / "A.bin").write_bytes(bytes(D.SEQ * 4))
    manifest = json.loads((tmp_path / "A.jsonl").read_text())
    manifest["row"] = 7
    (tmp_path / "A.jsonl").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="aligned"):
        D.open_split(tmp_path, "A")


def test_rows_api_is_bounded_and_normalizes_metadata(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(json.dumps({"rows": [
            {"row_idx": 9, "row": {"text": "hello", "url": "https://example.com"}},
            {"row": {"text": None}},
        ]}).encode())

    monkeypatch.setattr(D, "urlopen", urlopen)
    docs = list(D.iter_rows_api("a/b", 9, 2))
    assert docs == [
        {"text": "hello", "url": "https://example.com", "shard": "rows-api:default/train", "row": 9},
        {"text": "", "url": None, "shard": "rows-api:default/train", "row": 10},
    ]
    query = parse_qs(urlsplit(calls[0][0]).query)
    assert query == {"dataset": ["a/b"], "config": ["default"], "split": ["train"],
                     "offset": ["9"], "length": ["2"]}
    assert list(D.iter_rows_api("a/b", 0, 0)) == []
    with pytest.raises(ValueError):
        list(D.iter_rows_api("a/b", 0, 101))
    with pytest.raises(ValueError):
        list(D.iter_rows_api("a/b", -1, 1))
    assert len(calls) == 1


def test_load_tokenizer_records_snapshot(monkeypatch, tmp_path):
    path = tmp_path / "snapshots" / ("a" * 40) / "tokenizer.json"
    calls = []

    def download(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(path)

    tok = types.SimpleNamespace()
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = types.SimpleNamespace(from_file=lambda filename: tok)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    assert D.load_tokenizer("fake/repo", "tag") is tok
    assert calls == [("fake/repo", "tokenizer.json", "tag")]
    assert D._TOKENIZERS[id(tok)] == {"repo": "fake/repo", "revision": "tag", "commit": "a" * 40}


@pytest.mark.parametrize("url_column", [False, True])
def test_parquet_shuffle_column_selection_and_cleanup(monkeypatch, tmp_path, url_column):
    calls, deleted = [], []
    shards = ["b.parquet", "README.md", "a.parquet"]

    class FakeParquet:
        def __init__(self, path):
            self.path = Path(path)
            self.num_row_groups = 2
            self.schema_arrow = types.SimpleNamespace(names=["text", "large-unused"] +
                                                      (["url"] if url_column else []))
            self.metadata = types.SimpleNamespace(row_group=lambda i: types.SimpleNamespace(num_rows=5))

        def iter_batches(self, batch_size, row_groups, columns, use_threads):
            assert columns == ["text"] + (["url"] if url_column else [])
            assert batch_size <= 128 and not use_threads
            group = row_groups[0]
            data = {"text": [f"doc {group * 5 + i}" for i in range(5)]}
            if url_column:
                data["url"] = ["https://example.com"] * 5
            yield types.SimpleNamespace(num_rows=5, to_pydict=lambda: data)

        def close(self):
            deleted.append(self.path)

    def download(repo, shard, **kwargs):
        assert all(not p.exists() for p in deleted)
        path = Path(kwargs["local_dir"]) / shard
        path.write_bytes(b"fake")
        calls.append((repo, shard, kwargs))
        return str(path)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda: types.SimpleNamespace(list_repo_files=lambda *args, **kwargs: shards)
    hub.hf_hub_download = download
    arrow = types.ModuleType("pyarrow")
    arrow.parquet = types.ModuleType("pyarrow.parquet")
    arrow.parquet.ParquetFile = FakeParquet
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", arrow.parquet)
    first = list(D.iter_parquet("fake", "rev", 9, tmp_path))
    second = list(D.iter_parquet("fake", "rev", 9, tmp_path))
    assert first == second and len(first) == 20
    for shard in ("a.parquet", "b.parquet"):
        selected = [d for d in first if d["shard"] == shard]
        assert sorted(d["row"] for d in selected) == list(range(10))
        assert [d["row"] for d in selected] != list(range(10))
        assert all(d["text"] == f"doc {d['row']}" for d in selected)
        assert all(d["url"] == ("https://example.com" if url_column else None) for d in selected)
    assert all(not p.exists() for p in deleted)
    assert all(c[2]["revision"] == "rev" and c[2]["repo_type"] == "dataset" for c in calls)
    assert len(list(D.iter_parquet("fake", "rev", 9, tmp_path, max_shards=1))) == 10
    source = D.iter_parquet("fake", "rev", 9, tmp_path)
    next(source)
    source.close()
    assert all(not p.exists() for p in deleted)


def test_progress_and_stats_cli(tmp_path, capsys):
    meta = D.prepare([document("a" * D.SEQ)], char_tokenizer, tmp_path, {"A": 1},
                     min_chars=1, log_every=1, batch_size=1)
    assert "DATA scanned=1" in capsys.readouterr().out
    D.main(["stats", "--out", str(tmp_path)])
    output = capsys.readouterr().out
    assert output.startswith("DATA_JSON ")
    assert json.loads(output.removeprefix("DATA_JSON ")) == meta


@pytest.mark.parametrize("kwargs", [
    {"quotas": {"A": -1}}, {"quotas": {}}, {"quotas": {"../A": 1}},
    {"quotas": {"A": 1}, "near_dup_stride": 0},
    {"quotas": {"A": 1}, "near_dup_n": D.SEQ + 1},
    {"quotas": {"A": 1}, "workers": 0},
])
def test_invalid_configuration_fails_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        D.prepare([], char_tokenizer, tmp_path, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_no_model_dependency_loaded():
    # fresh interpreter: other test modules in the same session import torch
    import subprocess
    code = "import sys, tttmem.p2.data; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
