"""Prepare document-local Phase 2 sequences without loading a model.

Each row is BOS plus the first 8192 text tokens; documents are never packed.
Split probabilities stay fixed when quotas fill. Near-duplicate checks cover the
retained tokens, including BOS, and flag holdouts after all splits are written.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass, field, fields
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import random
import sqlite3
import statistics
import resource
import sys
import tarfile
import tempfile
import time
import unicodedata
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import numpy as np

DCLM_REPO = "mlfoundations/dclm-baseline-1.0-parquet"
TOKENIZER_REPO = "NousResearch/Meta-Llama-3-8B"
TOKENIZER_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"
DCLM_REVISION = "817d6752765f6a41261085171dd546b104f60626"
CODE_REPO = "codeparrot/github-code-clean"
CODE_REVISION = "c48d40f9e70f0196f8236901ee35807f7d6c44c0"
CODE_EXCLUDED_LANGS = {"HTML", "Markdown", "CSS", "TeX", "SQL", "Makefile",
                       "CMake", "Dockerfile", "Batchfile"}
BOS = 128000
EOS = 128001
SEQ = 8193

_TOKENIZERS: dict[int, dict] = {}
_HOLDOUTS = {"eval", "calib", "ho"}
_HASH_BASE = 1000000007
_WORKER_TOKENIZE = None


def repeat_classes(rows: np.ndarray, S: int) -> np.ndarray:
    """Classify each training target under segment-blocked attention.

    The return value has shape ``rows.shape[0], rows.shape[1]-1`` and uses the
    registered codes: ``-1`` invalid boundary/BOS target, ``0`` key undefined,
    ``1`` non-repeat, ``2`` within-segment repeat, and ``3`` cross-segment repeat.
    It is deliberately vectorised over documents; the only loop is over the
    four key columns used to form the sortable 4-gram records.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or S < 1 or rows.shape[1] < 2:
        raise ValueError("rows must be [N,T+1] and S positive")
    n, t = rows.shape[0], rows.shape[1] - 1
    x, y = rows[:, :-1].astype(np.uint64), rows[:, 1:].astype(np.uint64)
    pos = np.arange(t, dtype=np.int64)
    seg_start = (pos // int(S)) * int(S)
    valid = y != np.uint64(BOS)
    crossing = (pos % int(S) == int(S) - 1) & (pos < t - 1)
    valid &= ~crossing[None, :]
    defined = valid & ((pos[None, :] - 2) >= seg_start[None, :])
    x1 = np.concatenate([np.zeros((n, 1), np.uint64), x[:, :-1]], axis=1)
    x2 = np.concatenate([np.zeros((n, 2), np.uint64), x[:, :-2]], axis=1)
    flat_ok = np.flatnonzero(defined.ravel())
    row = np.repeat(np.arange(n, dtype=np.int64), t)
    cols = (x2.ravel(), x1.ravel(), x.ravel(), y.ravel())
    if len(flat_ok):
        order = flat_ok[np.lexsort([np.tile(pos, n)[flat_ok]] +
                                   [c[flat_ok] for c in cols[::-1]] + [row[flat_ok]])]
        same = np.zeros(len(order), dtype=bool)
        if len(order) > 1:
            same[1:] = row[order[1:]] == row[order[:-1]]
            for c in cols:
                same[1:] &= c[order[1:]] == c[order[:-1]]
        prev = np.full(n * t, -1, dtype=np.int64)
        prev[order[1:][same[1:]]] = order[:-1][same[1:]]
        prev = prev.reshape(n, t)
        prev -= np.arange(n, dtype=np.int64)[:, None] * t
    else:
        prev = np.full((n, t), -1, dtype=np.int64)
    out = np.full((n, t), -1, dtype=np.int8)
    out[valid] = 0
    out[defined] = 1
    has = (prev >= 0) & defined
    within = has & (prev >= seg_start[None, :])
    out[within] = 2
    out[has & ~within] = 3
    return out


def hf_download(*args, **kwargs):
    """Three bounded HF download attempts; inherit a token without logging it."""
    from huggingface_hub import hf_hub_download
    if os.environ.get("HF_TOKEN"):
        kwargs.setdefault("token", os.environ["HF_TOKEN"])
    for attempt in range(3):
        try:
            return hf_hub_download(*args, **kwargs)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def load_tokenizer(repo=TOKENIZER_REPO, revision="main", path=None):
    """Load a Hub tokenizer or a local tokenizer.json.

    The local path is optional: omitting it keeps the Llama-3 tokenizer and its
    BOS/EOS constants.
    """
    from tokenizers import Tokenizer

    local = path is not None
    if path is None:
        path = hf_download(repo, "tokenizer.json", revision=revision)
        parts = Path(path).parts
        commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
        info = {"repo": repo, "revision": revision, "commit": commit}
    else:
        path = str(Path(path).expanduser().resolve())
        info = {"path": path, "sha256": sha256_file(path)}
    tok = Tokenizer.from_file(path)
    if local:
        info.update(tokenizer_special_ids(tok))
    _TOKENIZERS[id(tok)] = info
    return tok


def tokenizer_special_ids(tok):
    """Return vocabulary and BOS/EOS IDs, falling back to the Llama-3 defaults."""
    vocab = tok.get_vocab()
    def find(names, fallback):
        for name in names:
            if name in vocab:
                return int(vocab[name])
        return int(fallback)
    return {"vocab_size": int(tok.get_vocab_size()),
            "bos_id": find(("<|bos|>", "<|begin_of_text|>", "<s>"), BOS),
            "eos_id": find(("<|eos|>", "<|end_of_text|>", "</s>"), EOS)}


def excluded_vocab(tok, vocab_size=128256) -> np.ndarray:
    """Mask special, missing, blank, and punctuation/symbol-only token IDs."""
    if vocab_size < 0:
        raise ValueError("vocab_size must be nonnegative")
    excluded = np.ones(vocab_size, dtype=np.bool_)
    bos_id = (tokenizer_special_ids(tok)["bos_id"] if hasattr(tok, "get_vocab") else BOS)
    # The default Llama-3 tokenizer reserves IDs [BOS, vocab); a local tokenizer may place
    # BOS at zero, in which case every ordinary token must still be inspected.
    limit = min(bos_id, vocab_size, tok.get_vocab_size()) if bos_id >= BOS else min(vocab_size, tok.get_vocab_size())
    for i in range(limit):
        text = tok.decode([i])
        excluded[i] = all(c.isspace() or unicodedata.category(c)[0] in "PS" for c in text)
    return excluded


def iter_parquet(repo, revision, shard_seed, workdir, max_shards=None, *, shards=None):
    """Stream one disposable shard; shuffle row groups and rows in bounded batches.

    This block permutation avoids materializing an entire shard or row group.
    Source row numbers always refer to the original parquet row positions.
    """
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq

    if max_shards is not None and max_shards < 0:
        raise ValueError("max_shards must be nonnegative")
    if shards is None:
        files = sorted(f for f in HfApi().list_repo_files(
            repo, repo_type="dataset", revision=revision) if f.endswith(".parquet"))
        random.Random(shard_seed).shuffle(files)
    else:
        files = shards
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for shard in files[:max_shards]:
        path = Path(hf_download(repo, shard, repo_type="dataset", revision=revision,
                                    local_dir=str(workdir)))
        parquet = None
        try:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            if "text" not in names:
                raise ValueError(f"{shard}: no text column")
            columns = ["text"] + (["url"] if "url" in names else [])
            rng = np.random.default_rng(int.from_bytes(
                hashlib.sha1(shard.encode()).digest()[:8], "big"))
            sizes = [parquet.metadata.row_group(i).num_rows
                     for i in range(parquet.num_row_groups)]
            starts = np.cumsum([0] + sizes[:-1])
            for group in rng.permutation(parquet.num_row_groups):
                row = int(starts[group])
                for batch in parquet.iter_batches(batch_size=128, row_groups=[int(group)],
                                                  columns=columns, use_threads=False):
                    data = batch.to_pydict()
                    for i in rng.permutation(batch.num_rows):
                        yield {"text": data["text"][i] or "",
                               "url": data["url"][i] if "url" in data else None,
                               "shard": shard, "row": row + int(i)}
                    row += batch.num_rows
        finally:
            if parquet is not None:
                parquet.close()
            path.unlink(missing_ok=True)


def iter_codeparrot(revision, shards, excluded_langs, *, exclude_sha1=(), exclude_repos=()):
    """Stream CodeParrot shards in numeric order and sha1 order within each shard.

    Arrow batches are bounded; a SQLite spool sorts text without retaining the
    355 MB shard in Python. Local parquet paths are accepted for offline tests.
    """
    import pyarrow.parquet as pq
    excluded_langs, excluded_sha = set(excluded_langs), set(exclude_sha1)
    excluded_repos = set(exclude_repos)
    with tempfile.TemporaryDirectory(prefix="ttm-code-") as scratch:
        scratch = Path(scratch)
        for shard in sorted(shards, key=str):
            source = Path(shard)
            if not source.is_file():
                filename = str(shard)
                if str(shard).isdigit():
                    filename = f"data/train-{int(shard):05d}-of-00880.parquet"
                source = Path(hf_download(CODE_REPO, filename, repo_type="dataset",
                                             revision=revision, local_dir=str(scratch)))
            db = sqlite3.connect(scratch / "sort.sqlite")
            try:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA cache_size=-8192")
                db.execute("CREATE TABLE docs (sha TEXT, row INTEGER, text TEXT, repo TEXT, path TEXT, language TEXT)")
                row = 0
                parquet = pq.ParquetFile(source)
                try:
                    for batch in parquet.iter_batches(batch_size=64,
                            columns=["code", "repo_name", "path", "language"], use_threads=False):
                        data = batch.to_pydict()
                        for i in range(batch.num_rows):
                            text, repo = data["code"][i] or "", data["repo_name"][i] or ""
                            lang = data["language"][i] or ""
                            sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
                            if lang not in excluded_langs and sha not in excluded_sha and repo not in excluded_repos:
                                db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?)", (sha, row + i,
                                    text, repo, data["path"][i] or "", lang))
                        row += batch.num_rows
                finally:
                    parquet.close()
                db.commit()
                for sha, row, text, repo, path, lang in db.execute("SELECT * FROM docs ORDER BY sha, row"):
                    yield {"text": text, "url": f"github://{repo}/{path}", "shard": str(shard),
                           "row": row, "language": lang, "repo_name": repo, "path": path}
            finally:
                db.close()
                (scratch / "sort.sqlite").unlink(missing_ok=True)


def iter_rows_api(dataset, offset, length):
    """Read a single smoke-test slice of at most 100 rows, using urllib only."""
    if offset < 0 or not 0 <= length <= 100:
        raise ValueError("offset must be nonnegative and length must be in [0, 100]")
    if length == 0:
        return
    query = urlencode({"dataset": dataset, "config": "default", "split": "train",
                       "offset": offset, "length": length})
    with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=60) as response:
        payload = json.load(response)
    for i, entry in enumerate(payload["rows"]):
        row = entry["row"]
        yield {"text": row.get("text") or "", "url": row.get("url"),
               "shard": "rows-api:default/train", "row": entry.get("row_idx", offset + i)}


def _domain(url):
    if not url or not isinstance(url, str):
        return ""
    try:
        host = urlsplit(url if "://" in url or url.startswith("//") else "//" + url).hostname
    except ValueError:
        return ""
    return (host or "").lower().removeprefix("www.")


def _split_for(sha1, quotas, seed):
    # Canonical split order makes probabilities independent of dict insertion order.
    digest = sha1 if seed == 0 else hashlib.sha1(f"{seed}:{sha1}".encode()).hexdigest()
    position = int(digest[:15], 16) * sum(quotas.values())
    edge = 0
    for split in sorted(quotas):
        edge += quotas[split] * 16**15
        if position < edge:
            return split
    raise ValueError("at least one quota must be positive")


@lru_cache(maxsize=8)
def _hash_powers(length):
    powers = np.full(length, _HASH_BASE, dtype=np.uint64)
    inverse = np.full(length, pow(_HASH_BASE, -1, 2**64), dtype=np.uint64)
    powers[0] = inverse[0] = 1
    return np.multiply.accumulate(powers), np.multiply.accumulate(inverse)


def _shingle_hashes(tokens, n=64, stride=1):
    """Position-independent polynomial hashes modulo 2**64; no position loop.

    Stride samples window starts 0, stride, ... in the BOS-prefixed sequence.
    """
    tokens = np.asarray(tokens, dtype=np.uint64)
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    powers, inverse = _hash_powers(len(tokens))
    prefix = np.empty(len(tokens) + 1, dtype=np.uint64)
    prefix[0] = 0
    np.cumsum((tokens + np.uint64(1)) * powers, out=prefix[1:])
    hashes = (prefix[n:] - prefix[:-n]) * inverse[:len(tokens) - n + 1]
    return hashes[::stride]


def _near_duplicates(out_dir, counts, stride, n):
    """Bound the holdout index to 32 MiB; stream training rows for each chunk."""
    flags = {s: np.zeros(counts[s], dtype=np.bool_) for s in counts if s in _HOLDOUTS}
    holdouts = [(s, i) for s in sorted(flags) for i in range(counts[s])]
    if not holdouts or not any(counts[s] for s in counts if s not in _HOLDOUTS):
        return flags
    windows = SEQ - n + 1
    chunk_rows = max(1, (32 * 1024**2) // (windows * 8))
    with ExitStack() as stack:
        files = {s: stack.enter_context((out_dir / f"{s}.bin").open("rb"))
                 for s, count in counts.items() if count}

        def read_row(split, i):
            files[split].seek(i * SEQ * 4)
            return np.frombuffer(files[split].read(SEQ * 4), dtype=np.uint32)

        for start in range(0, len(holdouts), chunk_rows):
            chunk = holdouts[start:start + chunk_rows]
            hashes = np.empty(len(chunk) * windows, dtype=np.uint64)
            for j, (s, i) in enumerate(chunk):
                hashes[j * windows:(j + 1) * windows] = _shingle_hashes(read_row(s, i), n)
            index = np.unique(hashes)
            del hashes
            hits = np.zeros(len(index), dtype=np.bool_)
            for s in sorted(files):
                if s in _HOLDOUTS:
                    continue
                files[s].seek(0)
                for _ in range(counts[s]):
                    tokens = np.frombuffer(files[s].read(SEQ * 4), dtype=np.uint32)
                    sampled = _shingle_hashes(tokens, n, stride)
                    pos = np.searchsorted(index, sampled)
                    valid = pos < len(index)
                    pos, sampled = pos[valid], sampled[valid]
                    hits[pos[index[pos] == sampled]] = True
            for s, i in chunk:
                flags[s][i] = hits[np.searchsorted(index, _shingle_hashes(read_row(s, i), n))].any()
    return flags


def _set_process_name(name):
    if not name:
        return
    os.environ["PROCESS_NAME"] = name
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(name)


def _worker_init(tokenize, process_name):
    global _WORKER_TOKENIZE
    _WORKER_TOKENIZE = tokenize
    worker = mp.current_process()._identity[-1]
    _set_process_name(f"{process_name}-worker{worker}")


def _worker_batch(texts):
    return _WORKER_TOKENIZE(texts)


class _TokenizerBatch:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]


def prepare(docs_iter, tokenize, out_dir, quotas: dict[str, int], *, min_chars=30000,
            seed=0, near_dup_stride=32, near_dup_n=64, workers=1, log_every=1000,
            exclude=None, batch_size=16, tokenizer_repo=TOKENIZER_REPO,
            tokenizer_revision="main", tokenizer_commit=None, dataset_repo=DCLM_REPO,
            dataset_revision="main", known_sha1=(), reject_row=None,
            manifest_chars_per_token=True, bos_id=BOS, eos_id=EOS,
            eligible_chunk=None, eligible_seq=8192, min_probes=5) -> dict:
    """Write bounded tokenization batches and manifests, then flag contaminated holdouts.

    ``exclude`` is an optional precomputed vocabulary mask. Worker callables must
    be pickleable. At most one batch is read ahead of quota completion. Reported
    chars/token excludes the prepended BOS; n_tokens_full includes it.
    ``known_sha1`` (hex text digests, e.g. every split of an existing store) count as
    duplicates; ``reject_row(row) -> bool`` drops a finished row (``extend_store``'s
    holdout near-duplicate filter). Both default to no extra filtering.
    """
    if not quotas or any(not isinstance(q, int) or isinstance(q, bool) or q < 0
                         for q in quotas.values()):
        raise ValueError("quotas must be a nonempty mapping of nonnegative integers")
    if any(not s or s in {".", ".."} or "/" in s or "\\" in s for s in quotas):
        raise ValueError("split names must be plain filenames")
    if min_chars < 0 or workers < 1 or batch_size < 1 or near_dup_stride < 1:
        raise ValueError("invalid prefilter, workers, batch size, or shingle stride")
    if not 1 <= near_dup_n <= SEQ or log_every < 0:
        raise ValueError("invalid shingle length or logging interval")
    started = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {s: 0 for s in sorted(quotas)}
    acceptance = {"docs_scanned": 0, "passed_char_prefilter": 0, "duplicates": 0,
                  "tokenized": 0, "passed_token_length": 0, "rejected_full_split": 0}
    if reject_row is not None:
        acceptance["rejected_row_filter"] = 0
    shards, ratios = set(), []
    tokenized_chars = tokenized_tokens = 0
    docs = iter(docs_iter)

    def full():
        return all(counts[s] == quotas[s] for s in counts)

    def progress():
        elapsed = max(time.monotonic() - started, 1e-9)
        print(f"DATA scanned={acceptance['docs_scanned']} accepted={json.dumps(counts)} "
              f"rate={acceptance['docs_scanned'] / elapsed:.1f} docs/s", flush=True)

    with ExitStack() as stack:
        close = getattr(docs, "close", None)
        if close is not None:
            stack.callback(close)
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".prepare-", dir=out_dir)))
        seen = sqlite3.connect(tmp / "seen.sqlite")
        stack.callback(seen.close)
        seen.execute("PRAGMA journal_mode=OFF")
        seen.execute("PRAGMA cache_size=-4096")
        seen.execute("CREATE TABLE seen (sha1 BLOB PRIMARY KEY) WITHOUT ROWID")
        seen.executemany("INSERT OR IGNORE INTO seen VALUES (?)",
                         ((bytes.fromhex(h),) for h in known_sha1))
        bins = {}
        for s in counts:
            path = out_dir / f"{s}.bin"
            if quotas[s]:
                bins[s] = np.memmap(path, mode="w+", dtype=np.uint32, shape=(quotas[s], SEQ))
            else:
                path.write_bytes(b"")

        def close_bins():
            for split, mapping in bins.items():
                mapping.flush()
                mapping._mmap.close()
                with (out_dir / f"{split}.bin").open("r+b") as f:
                    f.truncate(counts[split] * SEQ * 4)

        stack.callback(close_bins)
        manifests = {s: stack.enter_context((out_dir / f"{s}.jsonl").open("w", encoding="utf-8"))
                     for s in counts}
        pool = None
        if workers > 1 and not full():
            pool = stack.enter_context(mp.get_context("spawn").Pool(
                workers, initializer=_worker_init,
                initargs=(tokenize, os.environ.get("PROCESS_NAME", "TTM-p20-01"))))
        exhausted = False
        while not full() and not exhausted:
            batch, chars = [], 0
            while len(batch) < batch_size and chars < 1024**2:
                try:
                    doc = next(docs)
                except StopIteration:
                    exhausted = True
                    break
                acceptance["docs_scanned"] += 1
                if doc.get("shard") is not None:
                    shards.add(str(doc["shard"]))
                if log_every and acceptance["docs_scanned"] % log_every == 0:
                    progress()
                text = doc["text"]
                if len(text) < min_chars:
                    continue
                acceptance["passed_char_prefilter"] += 1
                digest = hashlib.sha1(text.encode("utf-8")).digest()
                if not seen.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount:
                    acceptance["duplicates"] += 1
                    continue
                batch.append((doc, digest.hex()))
                chars += len(text)
            if not batch:
                continue
            texts = [doc["text"] for doc, _ in batch]
            if all("_ids" in doc for doc, _ in batch):
                encoded = [doc["_ids"] for doc, _ in batch]
            elif pool is None:
                encoded = tokenize(texts)
            else:
                size = max(1, (len(texts) + workers - 1) // workers)
                encoded = [ids for part in pool.map(_worker_batch,
                           [texts[i:i + size] for i in range(0, len(texts), size)]) for ids in part]
            if len(encoded) != len(batch):
                raise ValueError("tokenizer must return one token list per document")
            for (doc, sha1), ids in zip(batch, encoded):
                acceptance["tokenized"] += 1
                tokenized_chars += len(doc["text"])
                tokenized_tokens += doc.get("_n_tokens_full", len(ids))
                n_tokens = doc.get("_n_tokens_full", len(ids)) + 1
                if n_tokens < SEQ:
                    continue
                acceptance["passed_token_length"] += 1
                split = _split_for(sha1, quotas, seed)
                if counts[split] == quotas[split]:
                    acceptance["rejected_full_split"] += 1
                    continue
                tokens = np.asarray(ids[:SEQ - 1])
                if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0) \
                        or np.any(tokens > np.iinfo(np.uint32).max):
                    raise ValueError("token IDs must be uint32-compatible integers")
                row = np.empty(SEQ, dtype=np.uint32)
                row[0], row[1:] = bos_id, tokens
                if reject_row is not None and reject_row(row):
                    acceptance["rejected_row_filter"] += 1
                    continue
                if split == "eval" and eligible_chunk is not None:
                    from .probe import find_probes
                    p = find_probes(row[:eligible_seq], row[1:eligible_seq + 1],
                                    eligible_chunk, valid=row[1:eligible_seq + 1] != bos_id,
                                    excluded=exclude, barriers=(2, 4, 6), seed=counts[split])
                    if len(p.pos) < min_probes or not p.nf_ok:
                        acceptance["rejected_ineligible"] = acceptance.get("rejected_ineligible", 0) + 1
                        continue
                bins[split][counts[split]] = row
                manifest = {"row": counts[split], "sha1": sha1, "url": doc.get("url"),
                            "domain": _domain(doc.get("url")), "shard": doc.get("shard"),
                            "source_row": doc.get("row"), "n_chars": len(doc["text"]),
                            "n_tokens_full": n_tokens}
                for key in ("language", "repo_name", "path"):
                    if key in doc:
                        manifest[key] = doc[key]
                if manifest_chars_per_token:
                    manifest["chars_per_token"] = len(doc["text"]) / (n_tokens - 1)
                if split in _HOLDOUTS:
                    manifest["near_dup"] = False
                manifests[split].write(json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n")
                counts[split] += 1
                ratios.append(len(doc["text"]) / (n_tokens - 1))

    flags = _near_duplicates(out_dir, counts, near_dup_stride, near_dup_n)
    for split, values in flags.items():
        path = out_dir / f"{split}.jsonl"
        replacement = path.with_suffix(".jsonl.tmp")
        with path.open(encoding="utf-8") as source, replacement.open("w", encoding="utf-8") as dest:
            for i, line in enumerate(source):
                record = json.loads(line)
                record["near_dup"] = bool(values[i])
                dest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        replacement.replace(path)
    acceptance.update({"chars_per_token_median": statistics.median(ratios) if ratios else None,
                       "chars_per_token_quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"),
                           np.quantile(ratios, [.05, .25, .5, .75, .95]).tolist())) if ratios else {},
                       "tokenized_chars_per_token": tokenized_chars / tokenized_tokens
                       if tokenized_tokens else None, "shards_used": sorted(shards)})
    tokenizer_info = {"repo": tokenizer_repo, "revision": tokenizer_revision,
                      "commit": tokenizer_commit}
    tokenizer_info.update(_TOKENIZERS.get(id(getattr(tokenize, "tokenizer", None)), {}))
    meta = {"tokenizer": tokenizer_info,
            "dataset": {"repo": dataset_repo, "revision": dataset_revision},
            "quotas": dict(quotas), "counts": counts, "acceptance": acceptance,
            "near_dup_counts": {s: int(v.sum()) for s, v in flags.items()},
            "min_chars": min_chars, "seed": seed, "near_dup_stride": near_dup_stride,
            "near_dup_n": near_dup_n, "workers": workers, "batch_size": batch_size,
            "seq": SEQ, "dtype": "uint32", "wall_time": time.monotonic() - started}
    if bos_id != BOS or eos_id != EOS or eligible_chunk is not None:
        meta.update({"bos_id": int(bos_id), "eos_id": int(eos_id),
                     "vocab_size": int(len(exclude)) if exclude is not None else None,
                     "eligible_chunk": eligible_chunk, "eligible_seq": eligible_seq})
    if exclude is not None:
        np.save(out_dir / "excluded_vocab.npy", np.asarray(exclude, dtype=np.bool_), allow_pickle=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def open_split(out_dir, split, seq=SEQ):
    """Open read-only token rows and verify manifest alignment (including empty splits)."""
    out_dir = Path(out_dir)
    if not split or split in {".", ".."} or "/" in split or "\\" in split:
        raise ValueError("split must be a plain filename")
    with (out_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f]
    path = out_dir / f"{split}.bin"
    expected = len(manifest) * int(seq) * np.dtype(np.uint32).itemsize
    if path.stat().st_size != expected or any(m["row"] != i for i, m in enumerate(manifest)):
        raise ValueError("token file and manifest are not aligned")
    if manifest:
        tokens = np.memmap(path, mode="r", dtype=np.uint32, shape=(len(manifest), int(seq)))
    else:
        # NumPy cannot mmap an empty file; use a zero-length view of an anonymous map.
        import mmap

        mapping = mmap.mmap(-1, 1, access=mmap.ACCESS_READ)
        tokens = np.ndarray.__new__(np.memmap, shape=(0, int(seq)), dtype=np.uint32,
                                    buffer=mapping)
        tokens._mmap, tokens.filename, tokens.offset, tokens.mode = mapping, str(path), 0, "r"
    return tokens, manifest


def split_contamination(out_dir, split, against=("eval", "ho", "calib"), *,
                        prefix_tokens: int = 256) -> dict:
    """SOQ-N3B pre-flight: documents of ``split`` that overlap the ``against`` splits by exact
    document sha1 or by an identical first ``prefix_tokens`` tokens (a cheap near-duplicate
    check on the packed rows).  Returns the counts; the caller fails on any hit."""
    rows, man = open_split(out_dir, split)
    sha = {m["sha1"] for m in man}
    n = min(int(prefix_tokens), SEQ)

    def prefixes(arr):
        out = set()
        for b0 in range(0, len(arr), 4096):
            block = np.ascontiguousarray(arr[b0:b0 + 4096, :n])
            out.update(hashlib.sha1(r.tobytes()).digest() for r in block)
        return out

    mine = prefixes(rows)
    rec = {"split": split, "rows": len(man), "prefix_tokens": n, "against": {}}
    for other in against:
        orows, oman = open_split(out_dir, other)
        rec["against"][other] = {
            "rows": len(oman),
            "sha1_hits": sum(m["sha1"] in sha for m in oman),
            "prefix_hits": len(mine & prefixes(orows))}
    rec["hits"] = sum(v["sha1_hits"] + v["prefix_hits"] for v in rec["against"].values())
    return rec


def contamination_enforced(out_dir, split) -> bool:
    """Whether a ``split_contamination`` hit stops the run: only for splits added by
    ``extend_store``.  The splits of the original ``prepare`` (meta.json quotas: A, B, holdouts)
    are registered as they are -- the P2-1 store flags its holdout near-duplicates
    (``near_dup`` in the manifests) instead of removing training rows -- so their counts are
    logged only.  Without meta.json (fixtures) every split is enforced."""
    path = Path(out_dir) / "meta.json"
    if not path.exists():
        return True
    return split not in json.loads(path.read_text()).get("quotas", {})


def select_code_eval(root, *, quota=1024, train_refs=(), excluded=None,
                     chunk=1024, barriers=(2, 4, 6), min_probes=5):
    """Choose the first eligible pool rows after 64-gram near-duplicate removal.

    train_refs is a sequence of (store directory, split) pairs. Its absence is
    recorded, so a local eval-only rehearsal cannot be mistaken for final dedup.
    """
    from .probe import find_probes
    root = Path(root)
    rows, manifest = open_split(root, "code_eval_pool")
    index = set()
    for directory, split in train_refs:
        train, _ = open_split(directory, split)
        for row in train:
            index.update(int(x) for x in _shingle_hashes(row, 64, 32))
    selected, flags, eligible = [], 0, 0
    for i, row in enumerate(rows):
        hit = bool(index and any(int(x) in index for x in _shingle_hashes(row, 64)))
        flags += int(hit)
        if hit:
            continue
        p = find_probes(row[:-1], row[1:], chunk, valid=row[1:] != BOS,
                        excluded=excluded, barriers=barriers, seed=i)
        ok = len(p.pos) >= min_probes and p.nf_ok
        eligible += int(ok)
        if ok and len(selected) < quota:
            selected.append(i)
    with (root / "code_eval.bin").open("wb") as f, (root / "code_eval.jsonl").open("w") as m:
        for out_i, pool_i in enumerate(selected):
            np.asarray(rows[pool_i], dtype=np.uint32).tofile(f)
            rec = dict(manifest[pool_i], row=out_i, pool_row=pool_i, near_dup=False)
            m.write(json.dumps(rec, sort_keys=True) + "\n")
    langs = {}
    ratios = []
    for i in selected:
        rec = manifest[i]
        langs[rec.get("language", "")] = langs.get(rec.get("language", ""), 0) + 1
        ratios.append(rec["chars_per_token"])
    return {"pool_count": len(rows), "eval_count": len(selected), "eligible_count": eligible,
            "eligible_fraction": eligible / max(1, len(rows) - flags),
            "near_dup_count": flags, "near_dup_checked_against": [list(x) for x in train_refs],
            "language_counts": langs, "language_shares": {k: v / max(1, len(selected)) for k, v in langs.items()},
            "chars_per_token_quantiles": dict(zip(("q05", "q25", "q50", "q75", "q95"),
                np.quantile(ratios, [.05, .25, .5, .75, .95]).tolist())) if ratios else {}}


def prepare_code(cfg: dict):
    """Pinned CodeParrot preparation; pool/train are built before eval selection."""
    _set_process_name(cfg.get("process_name") or os.environ.get("PROCESS_NAME"))
    root = Path(cfg["out_dir"])
    split = cfg.get("split", "code_eval")
    splits = cfg.get("splits", [split])
    tok = load_tokenizer(TOKENIZER_REPO, TOKENIZER_REVISION)
    excluded = excluded_vocab(tok)
    pool_meta = root / "meta.json"
    if split in ("code_eval", "code_train") or set(splits) <= {"code_eval", "code_train"}:
        if not (root / "code_eval_pool.bin").exists():
            prepare(iter_codeparrot(CODE_REVISION, cfg.get("shards", ["00000"]), CODE_EXCLUDED_LANGS),
                _TokenizerBatch(tok), root, {"code_eval_pool": int(cfg.get("pool_quota", 10000))},
                min_chars=int(cfg.get("min_chars", 12000)), workers=int(cfg.get("workers", 1)),
                batch_size=int(cfg.get("batch_size", 8)), exclude=excluded,
                tokenizer_revision=TOKENIZER_REVISION, dataset_repo=CODE_REPO, dataset_revision=CODE_REVISION)
        results = {}
        if "code_train" in splits:
            _, pool_man = open_split(root, "code_eval_pool")
            pool_meta_text = pool_meta.read_text() if pool_meta.exists() else None
            results["code_train"] = prepare(iter_codeparrot(CODE_REVISION, cfg.get("train_shards",
                [f"{i:05d}" for i in range(1, 880)]), CODE_EXCLUDED_LANGS,
                exclude_sha1=[m["sha1"] for m in pool_man],
                exclude_repos=[m["repo_name"] for m in pool_man]), _TokenizerBatch(tok), root,
                {"code_train": int(cfg.get("train_quota", cfg.get("quota", 10080)))},
                min_chars=int(cfg.get("min_chars", 12000)), workers=int(cfg.get("workers", 1)),
                batch_size=int(cfg.get("batch_size", 8)), exclude=excluded,
                tokenizer_revision=TOKENIZER_REVISION,
                dataset_repo=CODE_REPO, dataset_revision=CODE_REVISION)
            if pool_meta_text is not None:
                pool_meta.write_text(pool_meta_text)
        if "code_eval" in splits:
            refs = [(x["data_dir"], x["split"]) for x in cfg.get("train_refs", [])]
            results["code_eval"] = select_code_eval(root, quota=int(cfg.get("quota", 1024)),
                                                     train_refs=refs, excluded=excluded)
        result = results.get(split) or results
    else:
        raise ValueError("split must be code_eval or code_train")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = int(rss * (1 if sys.platform == "darwin" else 1024))
    (root / f"{split}_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CODE_PREPARE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def cpu_quota(path="/sys/fs/cgroup/cpu.max") -> int:
    """Effective CPUs: cgroup quota and affinity, never the host's nproc alone."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path(path).read_text().split()
        if quota != "max":
            return max(1, min(affinity, int(quota) // int(period)))
    except FileNotFoundError:
        pass  # macOS / cgroup v1
    return max(1, affinity)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class PrepareCfg:
    out_dir: str = "data/p2data"
    quotas: dict = field(default_factory=lambda: dict(eval=3200, ho=512, calib=256, A=25600, B=25600))
    dataset_repo: str = DCLM_REPO
    dataset_revision: str = DCLM_REVISION
    tokenizer_repo: str = TOKENIZER_REPO
    tokenizer_revision: str = TOKENIZER_REVISION
    tokenizer_path: str = ""
    require_eligible_eval: bool = False
    eligible_seq: int = 8192
    exclude_shards: list = field(default_factory=list)
    min_chars: int = 30000
    seed: int = 0
    near_dup_stride: int = 32
    near_dup_n: int = 64
    workers: int = 0                 # auto, capped by cgroup quota
    batch_size: int = 16
    max_shards: int | None = None    # only smoke/tests may cap the source
    log_every: int = 1000
    pack_path: str = "data/p2data.tar"
    compute_eligible: bool = True
    min_probes: int = 5
    manifest_chars_per_token: bool = True  # S14 rebuild uses the registered P2-1 schema
    # A mounted prepared store can be reused without downloading HF shards.
    reuse_if_exists: bool = False
    # SOQ-N3B: ``extend`` appends new train splits (``extend_quotas``, e.g. {"C": 25600},
    # later {"D": n}) to an existing store without touching its files (``extend_store``).
    extend: bool = False
    extend_quotas: dict = field(default_factory=dict)
    extend_pack_path: str = ""       # "" = "<out_dir>-<splits>.tar" (new split files only)
    chunk: int = 1024

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"mode", "process_name"}
        if unknown:
            raise ValueError(f"unknown prepare keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in names})


def _shard_init(repo, revision, name, path=""):
    global _WORKER_TOKENIZE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _set_process_name(f"{name}-shard{mp.current_process()._identity[-1]}")
    _WORKER_TOKENIZE = _TokenizerBatch(load_tokenizer(repo, revision, path=path or None))


def _prepare_shard(task):
    """Download and stream a shard into a disposable disk spool; no large IPC payloads."""
    cfg, shard, scratch, index = task
    root = Path(scratch) / str(index)
    root.mkdir()
    spool = root / "tokens.pkl"
    counts = dict(scanned=0, char_prefilter=0, tokenized=0, long=0)
    with spool.open("wb") as f:
        docs = iter_parquet(cfg.dataset_repo, cfg.dataset_revision, cfg.seed, root, shards=[shard])
        batch, chars = [], 0

        def emit():
            enc = _WORKER_TOKENIZE([d["text"] for d in batch])
            counts["tokenized"] += len(batch)
            for d, ids in zip(batch, enc):
                # Retain short tokenized docs too: the parent's acceptance accounting is exact.
                counts["long"] += len(ids) + 1 >= SEQ
                d.update(_ids=np.asarray(ids[:SEQ - 1], dtype=np.uint32), _n_tokens_full=len(ids))
                pickle.dump(d, f, protocol=5)
            batch.clear()

        for doc in docs:
            counts["scanned"] += 1
            if len(doc["text"]) < cfg.min_chars:
                continue
            counts["char_prefilter"] += 1
            batch.append(doc)
            chars += len(doc["text"])
            if len(batch) >= cfg.batch_size or chars >= 1024**2:
                emit()
                chars = 0
        if batch:
            emit()
    return str(spool), counts


def iter_prepared_shards(cfg, files, scratch, workers, statistics_out):
    """Bounded per-shard process pool, consumed in the seeded order, independent of workers."""
    if workers == 1:
        # The local pack build must stay in one CPU/numpy process. Use exactly
        # the same shard writer and acceptance order as the parallel path.
        global _WORKER_TOKENIZE
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _WORKER_TOKENIZE = _TokenizerBatch(load_tokenizer(cfg.tokenizer_repo, cfg.tokenizer_revision))
        for i, shard in enumerate(files):
            path, counts = _prepare_shard((cfg, shard, str(scratch), i))
            statistics_out[shard] = counts
            with open(path, "rb") as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
            Path(path).unlink()
        return
    name = os.environ.get("PROCESS_NAME", "TTM-p21-02-prepare")
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                               initializer=_shard_init,
                               initargs=(cfg.tokenizer_repo, cfg.tokenizer_revision, name, cfg.tokenizer_path))
    pending = deque()
    tasks = iter(enumerate(files))

    def submit():
        item = next(tasks, None)
        if item is not None:
            i, shard = item
            pending.append((shard, pool.submit(_prepare_shard, (cfg, shard, str(scratch), i))))

    try:
        for _ in range(workers):
            submit()
        while pending:
            shard, future = pending.popleft()
            path, counts = future.result()
            statistics_out[shard] = counts
            with open(path, "rb") as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
            Path(path).unlink()
            submit()
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def pack_store(out_dir, pack_path):
    """One transferable archive and a standard sha256sum sidecar, published atomically."""
    root, dest = Path(out_dir), Path(pack_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for path in sorted(root.iterdir()):
            if path.is_file() and path != dest and path != tmp:
                tar.add(path, arcname=f"p2data/{path.name}", recursive=False)
    tmp.replace(dest)
    digest = sha256_file(dest)
    dest.with_name(dest.name + ".sha256").write_text(f"{digest}  {dest.name}\n")
    return digest


EXTENSIONS = "extensions.json"


class _HoldoutShingles:
    """``reject_row`` for an extension: a candidate train row that shares a sampled 64-gram
    with any holdout row is dropped (the same shingles as ``_near_duplicates``: every window of
    the holdouts, training windows at ``stride``)."""

    def __init__(self, root: Path, splits, n: int, stride: int):
        parts = [_shingle_hashes(row, n) for s in splits for row in open_split(root, s)[0]]
        self.index = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
        self.n, self.stride = n, stride

    def __call__(self, row) -> bool:
        if not len(self.index):
            return False
        sampled = _shingle_hashes(row, self.n, self.stride)
        pos = np.searchsorted(self.index, sampled)
        ok = pos < len(self.index)
        return bool((self.index[pos[ok]] == sampled[ok]).any())


def _store_splits(root: Path) -> list[str]:
    return sorted(p.stem for p in root.glob("*.jsonl") if (root / f"{p.stem}.bin").exists())


def extend_store(cfg: PrepareCfg, *, docs=None, tokenize=None, files=None) -> dict:
    """Append new train splits (``cfg.extend_quotas``) to the prepared store ``cfg.out_dir``.

    Every pre-existing file stays byte-identical: it is hashed before and after (and checked
    against ``prepare_summary.json`` and earlier extension records), and the new splits are
    written to a staging directory inside the store, verified, then published with
    ``os.replace``.  The filters are the store's own (``meta.json``: min_chars, seed, SEQ,
    dataset and tokenizer revisions).  Exact text-sha1 dedup covers every split of the store;
    a candidate sharing a sampled 64-gram with a holdout row (eval/calib/ho) is dropped, since
    the holdout manifests cannot be re-flagged without changing their bytes.  Source shards
    continue the store's seeded shard order after every shard already used (by ``prepare`` or
    an earlier extension), so C and a later D are disjoint and deterministic.  The record
    (counts, hashes, shards, acceptance) is appended to ``extensions.json``; the new files are
    also packed alone for transfer.  ``docs``/``tokenize``/``files`` inject a source (tests).
    """
    import shutil
    from datetime import datetime, timezone

    started = time.monotonic()
    root = Path(cfg.out_dir)
    quotas = {str(k): int(v) for k, v in dict(cfg.extend_quotas).items()}
    if not quotas or any(q <= 0 for q in quotas.values()):
        raise ValueError("extend=true requires extend_quotas with positive counts, e.g. {C = 25600}")
    if any(not s or s.startswith(".") or "/" in s or "\\" in s for s in quotas):
        raise ValueError("split names must be plain filenames")
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"extend needs an existing prepared store: {meta_path} is missing")
    meta = json.loads(meta_path.read_text())
    want = {"min_chars": cfg.min_chars, "seed": cfg.seed, "seq": SEQ,
            "dataset_revision": cfg.dataset_revision, "tokenizer_revision": cfg.tokenizer_revision}
    have = {"min_chars": meta.get("min_chars"), "seed": meta.get("seed"), "seq": meta.get("seq"),
            "dataset_revision": meta.get("dataset", {}).get("revision"),
            "tokenizer_revision": meta.get("tokenizer", {}).get("revision")}
    diff = sorted(k for k in want if have[k] != want[k])
    if diff:
        raise ValueError(f"extend filters differ from the store's meta.json: {diff}")
    tag = "-".join(sorted(quotas))
    ext_path = root / EXTENSIONS
    records = json.loads(ext_path.read_text()) if ext_path.exists() else []
    present = [s for s in quotas if (root / f"{s}.bin").exists() or (root / f"{s}.jsonl").exists()]
    if present:
        done = [r for r in records if r["quotas"] == quotas]
        if done and all((root / k).exists() and sha256_file(root / k) == v
                        for k, v in done[-1]["sha256"].items()):
            print("P2_EXTEND " + json.dumps({"reused": True, "splits": sorted(quotas),
                                             "counts": done[-1]["counts"]}, sort_keys=True), flush=True)
            return {**done[-1], "reused": True}
        raise FileExistsError(f"split(s) {present} already exist without a matching {EXTENSIONS} record")
    # Every existing file, before anything is written (extensions.json is the append-only log).
    before = {p.name: sha256_file(p) for p in sorted(root.iterdir())
              if p.is_file() and p.name != EXTENSIONS}
    registered = {}
    if (root / "prepare_summary.json").exists():
        registered.update(json.loads((root / "prepare_summary.json").read_text()).get("sha256", {}))
    for r in records:
        registered.update(r["sha256"])
    bad = sorted(k for k, v in registered.items() if before.get(k) != v)
    if bad:
        raise ValueError(f"store files differ from their registered sha256: {bad}")
    splits = _store_splits(root)
    known = []
    for s in splits:
        with (root / f"{s}.jsonl").open(encoding="utf-8") as f:
            known.extend(json.loads(line)["sha1"] for line in f)
    n_dup, stride = int(meta.get("near_dup_n", cfg.near_dup_n)), int(meta.get("near_dup_stride", cfg.near_dup_stride))
    holdouts = [s for s in splits if s in _HOLDOUTS]
    reject = _HoldoutShingles(root, holdouts, n_dup, stride)
    used = set(meta.get("acceptance", {}).get("shards_used", []))
    for r in records:
        used.update(r.get("shards_used", []))
    workers = min(cfg.workers or cpu_quota(), cpu_quota())
    if docs is None:
        from huggingface_hub import HfApi
        files = sorted(f for f in HfApi().list_repo_files(
            cfg.dataset_repo, repo_type="dataset", revision=cfg.dataset_revision) if f.endswith(".parquet"))
        random.Random(cfg.seed).shuffle(files)
    order = list(files or [])
    used_is_prefix = set(order[:len(used)]) == used
    fresh = [f for f in order if f not in used]
    if cfg.max_shards is not None:
        fresh = fresh[:cfg.max_shards]
    stage = root / f".extend-{tag}"
    if stage.exists():
        shutil.rmtree(stage)  # left by an interrupted extension; nothing of it was published
    stage.mkdir()
    shard_stats = {}
    with tempfile.TemporaryDirectory(prefix="p2-shards-") as scratch:
        if docs is None:
            tok = load_tokenizer(cfg.tokenizer_repo, cfg.tokenizer_revision)
            tokenize = _TokenizerBatch(tok)
            docs = iter_prepared_shards(cfg, fresh, scratch, workers, shard_stats)
        new_meta = prepare(docs, tokenize, stage, quotas, min_chars=cfg.min_chars, seed=cfg.seed,
                           near_dup_stride=stride, near_dup_n=n_dup, workers=1,
                           batch_size=cfg.batch_size, exclude=None, log_every=cfg.log_every,
                           tokenizer_repo=cfg.tokenizer_repo, tokenizer_revision=cfg.tokenizer_revision,
                           dataset_repo=cfg.dataset_repo, dataset_revision=cfg.dataset_revision,
                           known_sha1=known, reject_row=reject)
    if new_meta["counts"] != quotas:
        shutil.rmtree(stage)
        raise ValueError(f"source exhausted before extend quotas: {new_meta['counts']} of {quotas}")
    for s in sorted(quotas):
        open_split(stage, s)
        for ext in ("bin", "jsonl"):
            os.replace(stage / f"{s}.{ext}", root / f"{s}.{ext}")
        open_split(root, s)
    os.replace(stage / "meta.json", root / f"extend-{tag}.meta.json")
    shutil.rmtree(stage)
    after = {name: sha256_file(root / name) for name in before}
    changed = sorted(k for k in before if after[k] != before[k])
    if changed:
        raise RuntimeError(f"extend changed existing store files: {changed}")
    new_files = [f"{s}.{ext}" for s in sorted(quotas) for ext in ("bin", "jsonl")]
    hashes = {name: sha256_file(root / name) for name in new_files + [f"extend-{tag}.meta.json"]}
    record = {"splits": sorted(quotas), "quotas": quotas, "counts": new_meta["counts"],
              "sha256": hashes,
              "token_store_sha256": hashlib.sha256(json.dumps(
                  {k: v for k, v in hashes.items() if k.endswith(".bin")}, sort_keys=True).encode()).hexdigest(),
              "manifest_sha256": hashlib.sha256(json.dumps(
                  {k: v for k, v in hashes.items() if k.endswith(".jsonl")}, sort_keys=True).encode()).hexdigest(),
              "shards_used": new_meta["acceptance"]["shards_used"],
              "shards_skipped_as_used": len(used), "used_shards_are_seeded_prefix": used_is_prefix,
              "acceptance": new_meta["acceptance"], "dedup_known_sha1": len(known),
              "holdout_filter": {"splits": holdouts, "n": n_dup, "stride": stride,
                                 "index_size": int(len(reject.index))},
              "existing_unchanged": True, "existing_sha256": before,
              "workers": workers, "shard_statistics": shard_stats,
              "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = ext_path.with_name(ext_path.name + ".tmp")
    tmp.write_text(json.dumps(records + [record], indent=1, sort_keys=True) + "\n")
    tmp.replace(ext_path)
    pack = Path(cfg.extend_pack_path or f"{root}-{tag}.tar")
    pack.parent.mkdir(parents=True, exist_ok=True)
    ptmp = pack.with_name(pack.name + ".tmp")
    with tarfile.open(ptmp, "w") as tar:
        for name in new_files + [f"extend-{tag}.meta.json", EXTENSIONS]:
            tar.add(root / name, arcname=f"{root.name}/{name}", recursive=False)
    ptmp.replace(pack)
    digest = sha256_file(pack)
    pack.with_name(pack.name + ".sha256").write_text(f"{digest}  {pack.name}\n")
    out = {**record, "archive": str(pack), "archive_sha256": digest,
           "wall_s": time.monotonic() - started}
    print("P2_EXTEND " + json.dumps({k: v for k, v in out.items()
                                     if k not in ("shard_statistics", "existing_sha256")},
                                    sort_keys=True), flush=True)
    return out


def prepare_config(cfg: PrepareCfg):
    from huggingface_hub import HfApi

    started = time.monotonic()
    if cfg.workers < 0:
        raise ValueError("workers must be zero (auto) or positive")
    workers = min(cfg.workers or cpu_quota(), cpu_quota())
    root = Path(cfg.out_dir)
    if cfg.extend:
        return extend_store(cfg)
    if (root / "meta.json").exists():
        if cfg.reuse_if_exists:
            return json.loads((root / "meta.json").read_text())
        raise FileExistsError("prepared store already exists; use a fresh out_dir")
    files = sorted(f for f in HfApi().list_repo_files(
        cfg.dataset_repo, repo_type="dataset", revision=cfg.dataset_revision) if f.endswith(".parquet"))
    reserved = set(cfg.exclude_shards)
    if cfg.tokenizer_path:
        sidecar = Path(cfg.tokenizer_path + ".json")
        if sidecar.exists():
            reserved.update(json.loads(sidecar.read_text()).get("shards", []))
    files = [f for f in files if f not in reserved]
    random.Random(cfg.seed).shuffle(files)
    if cfg.max_shards is not None:
        if cfg.max_shards < 1:
            raise ValueError("max_shards must be positive")
        files = files[:cfg.max_shards]
    tok = load_tokenizer(cfg.tokenizer_repo, cfg.tokenizer_revision,
                         path=cfg.tokenizer_path or None)
    tok_info = tokenizer_special_ids(tok)
    excluded = excluded_vocab(tok, tok_info["vocab_size"])
    shard_stats = {}
    with tempfile.TemporaryDirectory(prefix="p2-shards-") as scratch:
        docs = iter_prepared_shards(cfg, files, scratch, workers, shard_stats)
        meta = prepare(docs, _TokenizerBatch(tok), root, cfg.quotas, min_chars=cfg.min_chars,
                       seed=cfg.seed, near_dup_stride=cfg.near_dup_stride, near_dup_n=cfg.near_dup_n,
                       workers=1, batch_size=cfg.batch_size, exclude=excluded, log_every=cfg.log_every,
                       tokenizer_repo=cfg.tokenizer_repo, tokenizer_revision=cfg.tokenizer_revision,
                       dataset_repo=cfg.dataset_repo, dataset_revision=cfg.dataset_revision,
                       manifest_chars_per_token=cfg.manifest_chars_per_token,
                       bos_id=tok_info["bos_id"], eos_id=tok_info["eos_id"],
                       eligible_chunk=cfg.chunk if cfg.require_eligible_eval else None,
                       eligible_seq=cfg.eligible_seq, min_probes=cfg.min_probes)
    eligible, probe_counts, stratum_shares = {}, {}, {}
    if cfg.compute_eligible:
        from .probe import BARRIERS, find_probes
        for split in ("eval", "calib", "ho"):
            if split not in cfg.quotas:
                continue
            rows, manifest = open_split(root, split)
            count, probes = 0, 0
            strata = np.zeros(5, dtype=np.int64)
            for row, m in zip(rows, manifest):
                p = find_probes(row[:cfg.eligible_seq], row[1:cfg.eligible_seq + 1], cfg.chunk,
                                valid=row[1:cfg.eligible_seq + 1] != tok_info["bos_id"],
                                excluded=excluded,
                                barriers=(2, 4, 6) if cfg.require_eligible_eval else BARRIERS)
                count += len(p.pos) >= cfg.min_probes and p.nf_ok
                probes += len(p.pos)
                strata += np.bincount(p.stratum, minlength=5)
            eligible[split] = int(count)
            probe_counts[split] = int(probes)
            stratum_shares[split] = dict(zip(("E1", "E2s", "E2c", "E2b", "E3"),
                                            (strata / max(probes, 1)).tolist()))
    used = meta["acceptance"]["shards_used"]
    hashes = {p.name: sha256_file(p) for p in sorted(root.iterdir())
              if p.suffix in (".bin", ".jsonl", ".npy")}
    aggregate = lambda suffix: hashlib.sha256(json.dumps(
        {k: v for k, v in hashes.items() if k.endswith(suffix)}, sort_keys=True).encode()).hexdigest()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    factor = 1 if sys.platform == "darwin" else 1024
    summary = dict(counts=meta["counts"], eligible_counts=eligible, sha256=hashes,
                   token_store_sha256=aggregate(".bin"), manifest_sha256=aggregate(".jsonl"),
                   shard_list_sha256=hashlib.sha256("\n".join(used).encode()).hexdigest(),
                   dataset_revision=cfg.dataset_revision, tokenizer_revision=cfg.tokenizer_revision,
                   workers=workers, cpu_quota=cpu_quota(), shard_statistics=shard_stats,
                   peak_rss_bytes=int(max(rss, child_rss) * factor),
                   peak_rss_scope="largest parent/worker process (not simultaneous sum)")
    if cfg.tokenizer_path or cfg.require_eligible_eval:
        summary.update(probe_counts=probe_counts, stratum_shares=stratum_shares,
                       eligible_seq=cfg.eligible_seq, tokenizer_path=cfg.tokenizer_path,
                       tokenizer_sha256=sha256_file(cfg.tokenizer_path) if cfg.tokenizer_path else None,
                       bos_id=tok_info["bos_id"], eos_id=tok_info["eos_id"],
                       vocab_size=tok_info["vocab_size"], chunk=cfg.chunk)
    summary["complete"] = meta["counts"] == cfg.quotas
    (root / "prepare_summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    summary["archive_sha256"] = pack_store(root, cfg.pack_path)
    summary["archive"] = cfg.pack_path
    summary["wall_s"] = time.monotonic() - started
    print("P2_PREPARE " + json.dumps({k: v for k, v in summary.items() if k != "shard_statistics"},
                                     sort_keys=True), flush=True)
    if not summary["complete"]:
        raise ValueError(f"source exhausted before quotas: {meta['counts']}")
    return summary


def _quotas(value):
    try:
        pairs = [part.split("=") for part in value.split(",")]
        result = {name: int(count) for name, count in pairs}
        if len(result) != len(pairs) or any(q < 0 for q in result.values()):
            raise ValueError
        return result
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected unique split=nonnegative_integer pairs") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME"))
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    stats = sub.add_parser("stats")
    for command in (prep, stats):
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--process-name", default=argparse.SUPPRESS)
    prep.add_argument("--source", choices=("parquet", "rows"), default="parquet")
    prep.add_argument("--revision", default="main")
    prep.add_argument("--tokenizer-revision", default="main")
    prep.add_argument("--tokenizer-path", default="")
    # eval: 3072 eligible documents at the measured 97.9 % probe eligibility, plus margin
    prep.add_argument("--quotas", type=_quotas, default=_quotas("eval=3200,ho=512,calib=256,A=25600,B=25600"))
    prep.add_argument("--min-chars", type=int, default=30000)
    prep.add_argument("--workers", type=int, default=1)
    prep.add_argument("--max-shards", type=int)
    prep.add_argument("--rows-offset", type=int, default=0)
    prep.add_argument("--rows-length", type=int, default=100)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--near-dup-stride", type=int, default=32)
    prep.add_argument("--near-dup-n", type=int, default=64)
    prep.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args(argv)
    _set_process_name(args.process_name)
    if args.command == "stats":
        meta = json.loads((args.out / "meta.json").read_text(encoding="utf-8"))
    else:
        # rows API exposes train at the server's current revision, not a pinned snapshot.
        if args.source == "rows" and args.revision != "main":
            parser.error("rows source supports only the server's current revision (main)")
        tok = load_tokenizer(revision=args.tokenizer_revision, path=args.tokenizer_path or None)
        info = tokenizer_special_ids(tok)
        with tempfile.TemporaryDirectory(prefix="tttmem-shards-") as workdir:
            docs = (iter_rows_api(DCLM_REPO, args.rows_offset, args.rows_length)
                    if args.source == "rows" else iter_parquet(
                        DCLM_REPO, args.revision, args.seed, workdir, args.max_shards))
            meta = prepare(docs, _TokenizerBatch(tok), args.out, args.quotas,
                           min_chars=args.min_chars, seed=args.seed,
                           near_dup_stride=args.near_dup_stride, near_dup_n=args.near_dup_n,
                           workers=args.workers, log_every=args.log_every,
                           exclude=excluded_vocab(tok, info["vocab_size"]), tokenizer_revision=args.tokenizer_revision,
                           dataset_revision=args.revision, bos_id=info["bos_id"], eos_id=info["eos_id"])
    print("DATA_JSON " + json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
