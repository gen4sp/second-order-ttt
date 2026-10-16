"""Deterministic natural-text binding injections for the SO-question P1 experiment.

Generator v2.  The module is NumPy-only and never calls a tokenizer: the wrappers and
the nonce pools are token tables committed in ``echo_text_tables.json`` (built offline
by ``scripts/make_echo_tables.py`` from the pinned Llama-3 tokenizer).  Per split
(train / CAL / TEST, all disjoint) the table holds paraphrased plain-English sentences::

    binding  "The access code for{K} is{V}."
    mention  "Earlier, someone scribbled{V} on a sticky note."
    query    "Q: What is the access code for{K}? A:{V}"

with nonce person-like keys (" Zemalm Craaza", 4 tokens) and nonce values
(" gluciriop", 3 tokens).  Each sentence is its own line.  The pieces are concatenated
token tables; the builder kept only keys and values whose concatenation equals the
tokenizer's encoding of every full sentence.

One document has eight slots.  All slots use the same binding / mention / query
paraphrase, so the two strata are token-count matched; they differ only in whether the
slot's mention carries its own value (copy-available) or an equal-length distractor
value (copy-blocked).  Which slots are available is drawn per document (exactly
``round(8 p)`` of them), so stratum is not confounded with position.  Everything is a
deterministic function of ``(split, seed, p, row id)``; streamed training and a
materialised replay are equivalent.

Layout (production T = 8192, S = 2048, chunk C = 1024; segment k = row positions
``[(k-1) S, k S)``, every segment boundary is a barrier):

- segment 1, chunk 0: the eight mentions, then the eight bindings (one TTT write);
- segment 1, end: the eval-only no-barrier queries, after every binding;
- segment 2, end: the eval-only query-entity controls (``key_{i+1} -> value_i``);
- segment 3: normal queries of slots 0-3, then (end) no-binding controls for targets 4-7;
- segment 4: normal queries of slots 4-7, then (end) no-binding controls for targets 0-3.

No control shares an attention segment with another sentence that carries its target
value, and every control follows all normal queries of its segment.  Training rows carry
only mentions, bindings and normal queries.  In eval reads the control tokens do not
write (:func:`write_mask`), so they cannot change the fast weights seen by later reads.

Row coordinates in the metadata are *row* coordinates (BOS is coordinate 0).  The TTT
evaluation arrays index the input that predicts a target, so ``*_nll_positions`` is
always ``*_value_positions - 1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


TABLE_PATH = Path(__file__).with_name("echo_text_tables.json")
SPECIAL_MIN = 128000                       # Llama-3 special tokens are >= 128000
SPLITS = ("train", "cal", "test")
CONTROL_KINDS = ("no_barrier", "no_binding", "query_entity")
SPAN_KINDS = ("mention", "binding", "query") + CONTROL_KINDS
_BIGRAM = 1 << 17                          # > vocab size (128256): exact bigram codes
_MAX_ATTEMPTS = 32


# ----------------------------------------------------------------------------- tables
def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@dataclass(frozen=True)
class EchoTables:
    """Token tables of the natural wrappers and nonce pools (read-only)."""

    sha256: str
    splits_sha256: str
    key_len: int
    value_len: int
    excluded_vocab_sha256: str
    # wrappers[split][kind] -> list of tuples of literal token pieces
    wrappers: dict
    wrapper_text: dict
    keys: dict          # split -> [n, key_len] int64
    values: dict        # split -> [n, value_len] int64
    key_text: dict
    value_text: dict
    stats: dict = field(default_factory=dict)

    def max_len(self, kind: str) -> int:
        """Longest sentence of ``kind`` over every split (wrapper + key + value tokens)."""
        slots = {"binding": self.key_len + self.value_len, "mention": self.value_len,
                 "query": self.key_len + self.value_len}[kind]
        return max(sum(len(p) for p in w) + slots
                   for split in SPLITS for w in self.wrappers[split][kind])


@functools.lru_cache(maxsize=4)
def load_tables(path: str | None = None) -> EchoTables:
    """Load and verify the committed token tables (sha256 over the canonical JSON)."""
    raw = json.loads(Path(path or TABLE_PATH).read_text(encoding="utf-8"))
    body = {k: v for k, v in raw.items() if k != "sha256"}
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    if digest != raw.get("sha256"):
        raise ValueError(f"echo tables sha256 mismatch: {digest} != {raw.get('sha256')}")
    if hashlib.sha256(_canonical(raw["splits"])).hexdigest() != raw["splits_sha256"]:
        raise ValueError("echo tables splits_sha256 mismatch")
    kl, vl = int(raw["key_len"]), int(raw["value_len"])
    wrappers, wtext, keys, values, ktext, vtext = {}, {}, {}, {}, {}, {}
    for split in SPLITS:
        s = raw["splits"][split]
        wrappers[split] = {kind: [tuple(np.asarray(p, dtype=np.int64) for p in w["ids"])
                                  for w in s[kind]] for kind in ("binding", "mention", "query")}
        wtext[split] = {kind: [w["text"] for w in s[kind]] for kind in ("binding", "mention", "query")}
        keys[split] = np.asarray(s["keys"]["ids"], dtype=np.int64).reshape(-1, kl)
        values[split] = np.asarray(s["values"]["ids"], dtype=np.int64).reshape(-1, vl)
        ktext[split], vtext[split] = list(s["keys"]["text"]), list(s["values"]["text"])
    return EchoTables(sha256=raw["sha256"], splits_sha256=raw["splits_sha256"], key_len=kl,
                      value_len=vl, excluded_vocab_sha256=raw["excluded_vocab_sha256"],
                      wrappers=wrappers, wrapper_text=wtext, keys=keys, values=values,
                      key_text=ktext, value_text=vtext, stats=raw.get("stats", {}))


def _excluded_ids(excluded_vocab: np.ndarray) -> np.ndarray:
    ex = np.asarray(excluded_vocab)
    if ex.ndim != 1:
        raise ValueError("excluded_vocab must be one-dimensional")
    if ex.dtype == np.bool_:
        return np.flatnonzero(ex).astype(np.int64)
    return ex.astype(np.int64, copy=False)


def check_vocab(tables: EchoTables, excluded_vocab: np.ndarray) -> None:
    """Pool tokens must avoid ``excluded_vocab`` and the special range; wrappers must
    avoid the special range (their punctuation is excluded-vocab by definition)."""
    ex = _excluded_ids(excluded_vocab)
    for split in SPLITS:
        pool = np.concatenate((tables.keys[split].ravel(), tables.values[split].ravel()))
        if (pool >= SPECIAL_MIN).any() or (pool < 0).any():
            raise ValueError(f"special or negative token in the {split} pools")
        if len(ex) and np.isin(pool, ex).any():
            raise ValueError(f"the {split} pools contain excluded_vocab tokens")
        for kind, ws in tables.wrappers[split].items():
            if any((p >= SPECIAL_MIN).any() for w in ws for p in w):
                raise ValueError(f"special token in a {split} {kind} wrapper")


# ----------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class EchoLayout:
    """Registered 8-slot layout (T = 8192, S = 2048, chunk 1024 in production).

    Start positions are fixed; a sentence occupies ``[start, start + len)`` with ``len``
    at most the longest paraphrase of its kind (``max_*``, from the tables)."""

    T: int = 8192
    S: int = 2048
    chunk: int = 1024
    n_slots: int = 8
    key_len: int = 0           # 0 = from the tables
    value_len: int = 0
    max_mention: int = 0
    max_binding: int = 0
    max_query: int = 0

    def __post_init__(self):
        if 0 in (self.key_len, self.value_len, self.max_mention, self.max_binding, self.max_query):
            t = load_tables()
            for name, val in (("key_len", t.key_len), ("value_len", t.value_len),
                              ("max_mention", t.max_len("mention")),
                              ("max_binding", t.max_len("binding")),
                              ("max_query", t.max_len("query"))):
                if getattr(self, name) == 0:
                    object.__setattr__(self, name, int(val))

    @property
    def row_len(self) -> int:
        return self.T + 1

    @property
    def half(self) -> int:
        return self.n_slots // 2

    def _i(self, n=None) -> np.ndarray:
        return np.arange(self.n_slots if n is None else n, dtype=np.int64)

    @property
    def mention_starts(self) -> np.ndarray:
        return 64 + 56 * self._i()

    @property
    def binding_starts(self) -> np.ndarray:
        return 64 + 56 * self.n_slots + 32 + 56 * self._i()

    @property
    def no_barrier_query_starts(self) -> np.ndarray:
        # End of segment 1, after every binding.
        return self.S - 16 - 32 * self.n_slots + 32 * self._i()

    @property
    def query_entity_starts(self) -> np.ndarray:
        # End of segment 2 (behind the first barrier, no normal query there).
        return 2 * self.S - 16 - 32 * self.n_slots + 32 * self._i()

    @property
    def query_starts(self) -> np.ndarray:
        # Slots 0..half-1 in segment 3, the rest in segment 4.
        h = self.half
        return np.concatenate((2 * self.S + 64 + 256 * self._i(h),
                               3 * self.S + 64 + 256 * self._i(self.n_slots - h)))

    @property
    def no_binding_query_starts(self) -> np.ndarray:
        # Target i sits at the end of the segment that does NOT hold query i.
        h, nb = self.half, self.n_slots - self.half
        first = 4 * self.S - 16 - 32 * h + 32 * self._i(h)       # targets 0..h-1: segment 4
        second = 3 * self.S - 16 - 32 * nb + 32 * self._i(nb)    # targets h..: segment 3
        return np.concatenate((first, second))

    def starts(self, kind: str) -> np.ndarray:
        return {"mention": self.mention_starts, "binding": self.binding_starts,
                "query": self.query_starts, "no_barrier": self.no_barrier_query_starts,
                "no_binding": self.no_binding_query_starts,
                "query_entity": self.query_entity_starts}[kind]

    def max_len(self, kind: str) -> int:
        return self.max_mention if kind == "mention" else (
            self.max_binding if kind == "binding" else self.max_query)

    def segment(self, pos) -> np.ndarray:
        """1-based segment of row position(s)."""
        return np.asarray(pos) // self.S + 1

    def spans(self, query_mode: str = "normal") -> list[tuple[int, int, str, int]]:
        """Worst-case spans (longest paraphrase of each kind)."""
        out = []
        for kind in SPAN_KINDS:
            for i, s in enumerate(self.starts(kind)):
                length = self.max_binding if kind == "query" and query_mode == "verbatim" else self.max_len(kind)
                out.append((int(s), int(s) + length, kind, i))
        return out

    def validate(self, query_mode: str = "normal") -> None:
        if query_mode not in ("normal", "verbatim"):
            raise ValueError("query_mode must be normal or verbatim")
        if self.T < 4 * self.S or self.chunk < 1 or self.S % self.chunk:
            raise ValueError("echo layout needs T >= 4 S and S a multiple of chunk")
        if self.n_slots < 2 or self.n_slots % 2:
            raise ValueError("n_slots must be even and >= 2")
        if min(self.key_len, self.value_len, self.max_mention, self.max_binding, self.max_query) < 1:
            raise ValueError("sentence lengths must be positive")
        spans = sorted(self.spans(query_mode))
        for (s0, e0, k0, i0), (s1, e1, k1, i1) in zip(spans, spans[1:]):
            if s1 < e0:
                raise ValueError(f"spans overlap: {k0}[{i0}] [{s0},{e0}) and {k1}[{i1}] [{s1},{e1})")
        want_seg = {"mention": 1, "binding": 1, "no_barrier": 1, "query_entity": 2}
        h = self.half
        for s, e, kind, i in spans:
            seg = int(self.segment(s))
            lo, hi = (seg - 1) * self.S, seg * self.S
            # Inside one segment, off its first token and the barrier-masked last target.
            if not (lo + 1 <= s and e <= hi - 2) or e > self.T:
                raise ValueError(f"{kind}[{i}] [{s},{e}) crosses a segment boundary")
            if kind in want_seg and seg != want_seg[kind]:
                raise ValueError(f"{kind}[{i}] must be in segment {want_seg[kind]}, is in {seg}")
            if kind == "query" and seg != (3 if i < h else 4):
                raise ValueError(f"normal query {i} must be in segment {3 if i < h else 4}")
            if kind == "no_binding" and seg != (4 if i < h else 3):
                raise ValueError(f"no-binding control {i} must be in segment {4 if i < h else 3}")
        ends = {k: max(e for _, e, kk, _ in spans if kk == k) for k in SPAN_KINDS}
        begins = {k: min(s for s, _, kk, _ in spans if kk == k) for k in SPAN_KINDS}
        if ends["mention"] > begins["binding"]:
            raise ValueError("every mention must precede every binding")
        if ends["binding"] > self.chunk:
            raise ValueError("mentions and bindings must lie in chunk 0 (one write, value mask)")
        if ends["binding"] > begins["no_barrier"]:
            raise ValueError("no-barrier queries must follow every binding")
        for seg in (3, 4):
            q_end = max(e for s, e, k, _ in spans if k == "query" and self.segment(s) == seg)
            c_beg = min(s for s, _, k, _ in spans if k == "no_binding" and self.segment(s) == seg)
            if c_beg < q_end:
                raise ValueError(f"controls must follow the normal queries of segment {seg}")
        # No control shares an attention segment with another carrier of its target.
        carriers: dict[int, dict[int, int]] = {}
        for s, _, kind, i in spans:
            if kind in ("query",) + CONTROL_KINDS:
                seg = int(self.segment(s))
                if kind != "no_barrier":
                    carriers.setdefault(seg, {})
                    carriers[seg][i] = carriers[seg].get(i, 0) + 1
        for seg, cnt in carriers.items():
            if any(c > 1 for c in cnt.values()):
                raise ValueError(f"segment {seg} carries one target value twice")

    def as_dict(self) -> dict:
        return {
            "T": int(self.T), "S": int(self.S), "chunk": int(self.chunk),
            "n_slots": int(self.n_slots), "key_len": int(self.key_len),
            "value_len": int(self.value_len), "max_mention": int(self.max_mention),
            "max_binding": int(self.max_binding), "max_query": int(self.max_query),
            **{f"{k}_starts": self.starts(k).tolist() for k in SPAN_KINDS},
        }


# ----------------------------------------------------------------------------- per document
def _u64(*parts) -> int:
    payload = "echo-v2|" + "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _u64_v1(seed: int, row_id: int, salt: int = 0) -> int:
    payload = f"echo-v1|{int(seed)}|{int(row_id)}|{int(salt)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _selected(f: float, seed: int, row_id: int) -> bool:
    """Row selection (generator-v1 hash rule, shared by every version: same injected rows)."""
    if f <= 0:
        return False
    if f >= 1:
        return True
    return _u64_v1(seed, row_id, 17) / float(1 << 64) < float(f)


@dataclass(frozen=True)
class _Plan:
    binding: int
    mention: int
    query: int
    available: np.ndarray


def _plan(tables: EchoTables, split: str, seed: int, row_id: int, p: float, n_slots: int) -> _Plan:
    """Host-independent part: paraphrases (independent of p) and the available slots."""
    rng = np.random.default_rng(_u64("plan", int(seed), split, int(row_id)))
    w = tables.wrappers[split]
    b, m, q = (int(rng.integers(len(w[k]))) for k in ("binding", "mention", "query"))
    order = rng.permutation(n_slots)
    available = np.zeros(n_slots, dtype=bool)
    available[order[:int(round(n_slots * float(p)))]] = True
    return _Plan(b, m, q, available)


def _disjoint_draw(pool: np.ndarray, k: int, rng: np.random.Generator, used: set) -> np.ndarray:
    """``k`` distinct pool rows whose tokens are pairwise disjoint and avoid ``used``."""
    chosen: list[int] = []
    for _ in range(64):
        for j in rng.integers(len(pool), size=4 * k).tolist():
            toks = pool[j].tolist()
            if j in chosen or used.intersection(toks):
                continue
            chosen.append(j)
            used.update(toks)
            if len(chosen) == k:
                return np.asarray(chosen, dtype=np.int64)
    raise RuntimeError("could not draw token-disjoint nonce items")


def _items(tables: EchoTables, split: str, seed: int, row_id: int, attempt: int, n: int):
    rng = np.random.default_rng(_u64("items", int(seed), split, int(row_id), int(attempt)))
    used: set = set()
    k = _disjoint_draw(tables.keys[split], 2 * n, rng, used)      # bound + unbound keys
    v = _disjoint_draw(tables.values[split], 2 * n, rng, used)    # values + distractors
    return k[:n], k[n:], v[:n], v[n:]


def _bigrams(seqs: np.ndarray) -> np.ndarray:
    s = np.asarray(seqs, dtype=np.int64)
    return (s[:, :-1] * _BIGRAM + s[:, 1:]).ravel()


def inject_document(host_row: np.ndarray, *, split: str, seed: int, p: float, row_id: int,
                    controls: bool = True, layout: EchoLayout | None = None,
                    tables: EchoTables | None = None,
                    query_mode: str = "normal") -> tuple[np.ndarray, dict]:
    """Inject one host row; return a copy and the complete slot metadata.

    ``controls=False`` is a training row (mentions, bindings, normal queries only).
    A row whose host text already contains a bigram of any drawn key or value is
    redrawn (``attempt`` in the metadata); the plan (paraphrases, strata) is kept."""
    tables = tables or load_tables()
    layout = layout or EchoLayout()
    if query_mode not in ("normal", "verbatim") or (query_mode != "normal" and not controls):
        raise ValueError("verbatim query_mode is eval-only (controls=true)")
    if query_mode == "verbatim":
        layout.validate(query_mode="verbatim")
    host = np.asarray(host_row)
    if host.ndim != 1 or len(host) != layout.row_len:
        raise ValueError(f"host row must have shape ({layout.row_len},)")
    if split not in SPLITS:
        raise ValueError("split must be train, cal or test")
    if not 0 <= float(p) <= 1:
        raise ValueError("p must be in [0, 1]")
    n = layout.n_slots
    plan = _plan(tables, split, seed, row_id, p, n)
    w = tables.wrappers[split]
    B, M, Q = w["binding"][plan.binding], w["mention"][plan.mention], w["query"][plan.query]
    kl, vl = tables.key_len, tables.value_len
    host32 = host.astype(np.uint32, copy=False)
    for attempt in range(_MAX_ATTEMPTS):
        kb, ku, vb, vd = _items(tables, split, seed, row_id, attempt, n)
        keys, unbound = tables.keys[split][kb], tables.keys[split][ku]
        values, distract = tables.values[split][vb], tables.values[split][vd]
        mention_vals = np.where(plan.available[:, None], values, distract)
        out = host32.copy()
        kv = np.zeros(layout.row_len, dtype=bool)
        spans: list[list] = []
        pos: dict[str, np.ndarray] = {}
        keypos: dict[str, np.ndarray] = {}

        def put(kind, starts, key_rows, value_rows, wrap):
            vpos = np.empty((n, vl), dtype=np.int64)
            kpos = np.empty((n, kl), dtype=np.int64) if key_rows is not None else None
            for i in range(n):
                s = int(starts[i])
                if key_rows is None:     # mention: pre V post
                    seq = np.concatenate((wrap[0], value_rows[i], wrap[1]))
                    v0 = s + len(wrap[0])
                else:                    # pre K mid V post
                    seq = np.concatenate((wrap[0], key_rows[i], wrap[1], value_rows[i], wrap[2]))
                    k0 = s + len(wrap[0])
                    kpos[i] = k0 + np.arange(kl)
                    kv[k0:k0 + kl] = True
                    v0 = k0 + kl + len(wrap[1])
                vpos[i] = v0 + np.arange(vl)
                kv[v0:v0 + vl] = True
                out[s:s + len(seq)] = seq
                spans.append([s, s + len(seq), kind, i])
            pos[kind] = vpos
            if kpos is not None:
                keypos[kind] = kpos

        put("mention", layout.mention_starts, None, mention_vals, M)
        put("binding", layout.binding_starts, keys, values, B)
        put("query", layout.query_starts, keys, values, B if query_mode == "verbatim" else Q)
        if controls:
            put("no_barrier", layout.no_barrier_query_starts, keys, values, Q)
            put("no_binding", layout.no_binding_query_starts, unbound, values, Q)
            put("query_entity", layout.query_entity_starts, np.roll(keys, -1, axis=0), values, Q)
        # Host collisions: no key/value bigram outside the written key/value tokens.
        drawn = np.concatenate((_bigrams(keys), _bigrams(unbound), _bigrams(values),
                                _bigrams(distract)))
        row_bi = out[:-1].astype(np.int64) * _BIGRAM + out[1:].astype(np.int64)
        hit = np.isin(row_bi, drawn)
        if not (hit & ~(kv[:-1] & kv[1:])).any():
            break
    else:
        raise RuntimeError(f"host collisions persisted for row {row_id} after {_MAX_ATTEMPTS} draws")
    spans.sort()

    def nll(x):
        return (x - 1).tolist()

    meta = {
        "version": 2, "split": split, "row_id": int(row_id), "seed": int(seed),
        "data_seed": int(seed), "p": float(p), "attempt": int(attempt), "controls": bool(controls),
        "paraphrase": {"binding": plan.binding, "mention": plan.mention, "query": plan.query},
        "available": plan.available.tolist(),
        "stratum": np.where(plan.available, "copy_available", "copy_blocked").tolist(),
        "key_tokens": keys.tolist(), "value_tokens": values.tolist(),
        "distractor_tokens": distract.tolist(), "mention_value_tokens": mention_vals.tolist(),
        "mention_positions": pos["mention"].tolist(),
        "source_key_positions": keypos["binding"].tolist(),
        "source_value_positions": pos["binding"].tolist(), "source_nll_positions": nll(pos["binding"]),
        "query_key_positions": keypos["query"].tolist(),
        "query_value_positions": pos["query"].tolist(), "query_nll_positions": nll(pos["query"]),
        "spans": spans, "injected": True,
    }
    if controls:
        meta.update({
            "query_mode": query_mode,
            "query_key_nll_positions": nll(keypos["query"]),
            "no_binding_key_nll_positions": nll(keypos["no_binding"]),
            "no_barrier_value_positions": pos["no_barrier"].tolist(),
            "no_barrier_nll_positions": nll(pos["no_barrier"]),
            "no_binding_key_positions": keypos["no_binding"].tolist(),
            "no_binding_key_tokens": unbound.tolist(),
            "no_binding_value_positions": pos["no_binding"].tolist(),
            "no_binding_nll_positions": nll(pos["no_binding"]),
            "query_entity_key_tokens": np.roll(keys, -1, axis=0).tolist(),
            "query_entity_target_tokens": values.tolist(),
            "query_entity_value_positions": pos["query_entity"].tolist(),
            "query_entity_nll_positions": nll(pos["query_entity"]),
        })
    return out, meta


def injected_length(tables: EchoTables, split: str, seed: int, row_id: int, n_slots: int = 8,
                    controls: bool = False) -> int:
    """Number of injected tokens in a row (host-independent: depends on the paraphrases)."""
    plan = _plan(tables, split, seed, row_id, 0.0, n_slots)
    w = tables.wrappers[split]
    kl, vl = tables.key_len, tables.value_len
    lb = sum(len(x) for x in w["binding"][plan.binding]) + kl + vl
    lm = sum(len(x) for x in w["mention"][plan.mention]) + vl
    lq = sum(len(x) for x in w["query"][plan.query]) + kl + vl
    return n_slots * (lm + lb + lq + (3 * lq if controls else 0))


def write_mask(records: list[Mapping], T: int) -> np.ndarray:
    """``[N, T]`` bool inner-loss (write) mask for eval reads: False at the positions that
    predict a control-sentence token, True elsewhere.  Pass as ``inner_mask``."""
    m = np.ones((len(records), int(T)), dtype=bool)
    for j, rec in enumerate(records):
        for s, e, kind, _ in rec.get("spans", ()):
            if kind in CONTROL_KINDS:
                m[j, max(0, int(s) - 1):min(int(T), int(e) - 1)] = False
    return m


# ----------------------------------------------------------------------------- injector
class EchoInjector:
    """Streaming training injector and eval-document builder with a stable manifest."""
    version = 2

    def __init__(self, *, split: str, f: float, p: float, seed: int, T: int,
                 excluded_vocab: np.ndarray, layout: EchoLayout | None = None,
                 tables: EchoTables | None = None):
        if split not in SPLITS:
            raise ValueError("split must be train, cal or test")
        if not 0 <= float(f) <= 1 or not 0 <= float(p) <= 1:
            raise ValueError("f and p must be in [0, 1]")
        self.split, self.f, self.p, self.seed = split, float(f), float(p), int(seed)
        self.tables = tables or load_tables()
        self.layout = layout or EchoLayout(T=int(T))
        if self.layout.T != int(T):
            raise ValueError("layout T and T argument disagree")
        if (self.layout.key_len, self.layout.value_len) != (self.tables.key_len, self.tables.value_len):
            raise ValueError("layout and tables disagree on key/value length")
        self.layout.validate()
        check_vocab(self.tables, excluded_vocab)
        self.excluded_vocab_sha256 = hashlib.sha256(
            np.ascontiguousarray(np.asarray(excluded_vocab)).tobytes()).hexdigest()

    # -- selection and exposure ------------------------------------------------------
    def selected(self, global_row_index) -> np.ndarray:
        ids = np.asarray(global_row_index, dtype=np.int64).reshape(-1)
        return np.fromiter((_selected(self.f, self.seed, int(i)) for i in ids), dtype=bool,
                           count=len(ids))

    def injected_tokens(self, global_row_index) -> np.ndarray:
        """Injected tokens per training row (0 where the row is not selected)."""
        ids = np.asarray(global_row_index, dtype=np.int64).reshape(-1)
        sel = self.selected(ids)
        out = np.zeros(len(ids), dtype=np.int64)
        for j in np.flatnonzero(sel):
            out[j] = injected_length(self.tables, self.split, self.seed, int(ids[j]),
                                     self.layout.n_slots, controls=False)
        return out

    def exposure_accounting(self, global_row_index: np.ndarray) -> dict:
        """Source/query token exposures and the exact injected-token budget of a stream."""
        ids = np.asarray(global_row_index, dtype=np.int64).reshape(-1)
        tok = self.injected_tokens(ids)
        docs = int((tok > 0).sum())
        n_available = int(round(self.layout.n_slots * self.p))
        n_blocked = self.layout.n_slots - n_available
        vl = self.layout.value_len
        return {
            "rows": int(len(ids)), "injected_rows": docs,
            "copy_available_slots": n_available, "copy_blocked_slots": n_blocked,
            "copy_available_source_writes": docs * n_available * vl,
            "copy_blocked_source_writes": docs * n_blocked * vl,
            "copy_available_query_exposures": docs * n_available * vl,
            "copy_blocked_query_exposures": docs * n_blocked * vl,
            "injected_token_budget": int(tok.sum()),
            "token_share": float(tok.sum()) / float(max(1, len(ids)) * self.layout.T),
        }

    # -- documents --------------------------------------------------------------------
    def document(self, host_row: np.ndarray, row_id: int, *, p: float | None = None,
                 controls: bool = False, query_mode: str = "normal") -> tuple[np.ndarray, dict]:
        return inject_document(host_row, split=self.split, seed=self.seed,
                               p=self.p if p is None else float(p), row_id=int(row_id),
                               controls=controls, layout=self.layout, tables=self.tables,
                               query_mode=query_mode)

    def __call__(self, rows_uint32: np.ndarray, global_row_index: np.ndarray) -> tuple[np.ndarray, dict]:
        """Training stream: inject the selected rows (no controls)."""
        rows = np.asarray(rows_uint32)
        ids = np.asarray(global_row_index)
        if rows.ndim != 2 or rows.shape[1] != self.layout.row_len:
            raise ValueError(f"rows must have shape [N, {self.layout.row_len}]")
        if ids.ndim != 1 or len(ids) != len(rows) or ids.dtype.kind not in "iu":
            raise ValueError("global_row_index must be a one-dimensional integer array")
        out = rows.astype(np.uint32, copy=True)
        selected = self.selected(ids)
        records: list[dict | None] = [None] * len(rows)
        for j in np.flatnonzero(selected):
            out[j], records[int(j)] = self.document(out[j], int(ids[j]), controls=False)
        meta = {"version": 2, "split": self.split, "f": self.f, "p": self.p,
                "seed": self.seed, "global_row_index": ids.astype(np.int64).tolist(),
                "selected": selected.tolist(), "rows": records,
                "rows_sha256": rows_sha256(out[selected]),
                "manifest_sha256": self.manifest()["manifest_sha256"]}
        return out, meta

    def eval_documents(self, host_rows: np.ndarray, *, row_ids: np.ndarray | None = None,
                       p: float | None = None, query_mode: str = "normal") -> tuple[np.ndarray, dict]:
        """Injected CAL/TEST documents with every control, plus metadata for V1/P1 reads."""
        rows = np.asarray(host_rows)
        if rows.ndim != 2 or rows.shape[1] != self.layout.row_len:
            raise ValueError(f"host_rows must have shape [N, {self.layout.row_len}]")
        ids = np.arange(len(rows), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
        if ids.shape != (len(rows),):
            raise ValueError("row_ids must have one id per host row")
        pp = self.p if p is None else float(p)
        out = np.empty(rows.shape, dtype=np.uint32)
        records = []
        for j, rid in enumerate(ids):
            out[j], rec = self.document(rows[j], int(rid), p=pp, controls=True, query_mode=query_mode)
            records.append(rec)
        return out, {"version": 2, "split": self.split, "p": pp,
                     "row_ids": ids.tolist(), "rows": records}

    # -- manifest ---------------------------------------------------------------------
    def _manifest_params(self) -> dict:
        return {
            "version": 2, "split": self.split, "f": self.f, "p": self.p, "seed": self.seed,
            "T": self.layout.T, "S": self.layout.S, "chunk": self.layout.chunk,
            "key_len": self.layout.key_len, "value_len": self.layout.value_len,
            "n_slots": self.layout.n_slots, "excluded_vocab_sha256": self.excluded_vocab_sha256,
            "tables_sha256": self.tables.sha256, "layout": self.layout.as_dict(),
        }

    def manifest(self) -> dict:
        params = self._manifest_params()
        result = {**params,
                  "params_sha256": hashlib.sha256(_canonical(params)).hexdigest(),
                  "pools_sha256": self.tables.splits_sha256}
        result["manifest_sha256"] = manifest_sha256(result)
        return result


def make_injector(*, split: str = "train", f: float = 0.0, p: float = 0.5,
                  seed: int = 0, T: int = 8192, excluded_vocab: np.ndarray,
                  S: int = 2048, tables: EchoTables | None = None, version: int = 2,
                  lookup_share: float = 0.25, n_slots: int = 8) -> EchoInjector:
    """Public training-side factory required by the P1 continuation trainer."""
    if int(version) == 3:
        return make_injector_v3(split=split, f=f, p=p, lookup_share=lookup_share, seed=seed,
                                T=T, S=S, excluded_vocab=excluded_vocab, n_slots=n_slots,
                                tables=tables)
    if int(version) != 2:
        raise ValueError("echo generator version must be 2 or 3")
    return EchoInjector(split=split, f=f, p=p, seed=seed, T=T, excluded_vocab=excluded_vocab,
                        tables=tables, layout=EchoLayout(T=T, S=S))


def build_eval_documents(host_rows: np.ndarray, *, split: str, p: float, seed: int,
                         excluded_vocab: np.ndarray, T: int = 8192, S: int = 2048,
                         row_ids: np.ndarray | None = None,
                         query_mode: str = "normal") -> tuple[np.ndarray, dict]:
    """Convenience eval API for CAL/TEST templates."""
    return make_injector(split=split, f=1.0, p=p, seed=seed, T=T, S=S,
                         excluded_vocab=excluded_vocab).eval_documents(host_rows, row_ids=row_ids,
                                                                       query_mode=query_mode)


def manifest_sha256(manifest: Mapping | Iterable[Mapping]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()


def write_manifest(path: str | Path, manifests: list[dict], *, rows_hash: str | None = None) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest_sha256": manifest_sha256(manifests), "rows_sha256": rows_hash,
               "count": len(manifests), "rows": manifests}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return payload["manifest_sha256"]


def _json_default(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


# ----------------------------------------------------------------------------- generator v3
#
# v3 is deliberately implemented beside the v2 objects above.  The v2 manifest and
# byte-level selection rule stay fixed so v2 documents remain reproducible.
V3_READ_KINDS = ("cross_verbatim", "lookup_verbatim", "cross_paraphrase")
V3_CONTROL_KINDS = ("cross_verbatim", "lookup_verbatim", "cross_paraphrase")


@dataclass(frozen=True)
class EchoLayoutV3:
    """Leak-free 16-slot layout for the registered verbatim generator.

    Starts are computed from the longest table entries rather than hard-coded offsets.
    This makes small CPU fixtures useful while retaining the production coordinates
    (T=8192, S=2048, chunk=1024).  If a caller requests 16 slots in a fixture whose
    chunk zero cannot hold them, the layout deterministically falls back to 12 slots.
    """

    T: int = 8192
    S: int = 2048
    chunk: int = 1024
    n_slots: int = 16
    key_len: int = 0
    value_len: int = 0
    max_mention: int = 0
    max_binding: int = 0
    max_query: int = 0
    gap: int = 4

    def __post_init__(self):
        t = load_tables()
        vals = (("key_len", t.key_len), ("value_len", t.value_len),
                ("max_mention", t.max_len("mention")),
                ("max_binding", t.max_len("binding")),
                ("max_query", t.max_len("query")))
        for name, val in vals:
            if getattr(self, name) == 0:
                object.__setattr__(self, name, int(val))
        if self.n_slots not in (12, 16):
            raise ValueError("EchoLayoutV3 supports 12 or 16 slots")
        if self.n_slots == 16 and not self._chunk0_fits(16):
            object.__setattr__(self, "n_slots", 12)

    @property
    def row_len(self):
        return self.T + 1

    @property
    def slot_fallback(self):
        return self.n_slots != 16

    @property
    def half(self):
        return self.n_slots // 2

    @property
    def chunk0_required(self):
        return self.n_slots * (self.max_mention + self.max_binding + 2 * self.gap)

    def _chunk0_fits(self, n):
        return 1 + n * (self.max_mention + self.max_binding + 2 * self.gap) <= self.chunk - 2

    def _starts_linear(self, n, first, length):
        return np.asarray([first + i * (length + self.gap) for i in range(n)], dtype=np.int64)

    @property
    def mention_starts(self):
        return self._starts_linear(self.n_slots, 8, self.max_mention)

    @property
    def binding_starts(self):
        first = int(self.mention_starts[-1] + self.max_mention + self.gap)
        return self._starts_linear(self.n_slots, first, self.max_binding)

    @property
    def lookup_starts(self):
        # Segment 1, chunk 1.  The gap is intentionally larger than the minimum so
        # host bigram redraws never create adjacent sentence boundaries.
        first = self.chunk + 16
        return self._starts_linear(self.n_slots, first, self.max_binding)

    def cross_starts(self, segment_slots=None):
        counts = (self.n_slots // 3, self.n_slots // 3,
                  self.n_slots - 2 * (self.n_slots // 3)) if segment_slots is None else tuple(segment_slots)
        expected = (self.n_slots // 3, self.n_slots // 3,
                    self.n_slots - 2 * (self.n_slots // 3))
        if tuple(counts) != expected:
            raise ValueError("cross segment counts must sum to n_slots")
        out = np.empty(self.n_slots, dtype=np.int64)
        k = 0
        for seg, count in zip((2, 3, 4), counts):
            first = (seg - 1) * self.S + 16
            vals = self._starts_linear(count, first, self.max_query)
            out[k:k + count] = vals
            k += count
        return out

    def starts(self, kind):
        return {"mention": self.mention_starts, "binding": self.binding_starts,
                "lookup_verbatim": self.lookup_starts,
                "cross_verbatim": self.cross_starts(),
                "cross_paraphrase": self.cross_starts()}[kind]

    def segment(self, pos):
        return np.asarray(pos, dtype=np.int64) // self.S + 1

    def validate(self, *, query_mode="verbatim", segment_slots=None):
        if self.T < 4 * self.S or self.S % self.chunk:
            raise ValueError("v3 layout needs T >= 4S and S divides chunk")
        if self.n_slots not in (12, 16):
            raise ValueError("v3 n_slots must be 12 or 16")
        if query_mode not in ("verbatim", "paraphrase"):
            raise ValueError("v3 query_mode must be verbatim or paraphrase")
        if self.n_slots == 16 and not self._chunk0_fits(16):
            raise ValueError("16-slot chunk 0 does not fit")
        spans = []
        for kind, starts in (("mention", self.mention_starts), ("binding", self.binding_starts),
                             ("lookup_verbatim", self.lookup_starts),
                             ("cross_verbatim", self.cross_starts(segment_slots)),
                             ("cross_paraphrase", self.cross_starts(segment_slots))):
            # The two cross read variants occupy the same coordinates in their respective
            # documents; they are not simultaneous spans and therefore are checked separately.
            if kind == "cross_paraphrase":
                continue
            length = self.max_query if kind == "cross_verbatim" and query_mode == "paraphrase" else (
                self.max_binding if kind in ("binding", "lookup_verbatim", "cross_verbatim") else self.max_mention)
            for i, s in enumerate(starts):
                e = int(s + length)
                seg = int(self.segment(s))
                lo, hi = (seg - 1) * self.S, seg * self.S
                if not (lo + 1 <= s < e <= hi - 2):
                    raise ValueError(f"{kind}[{i}] crosses segment boundary")
                spans.append((int(s), e, kind, i))
        for a, b in zip(sorted(spans), sorted(spans)[1:]):
            if a[1] > b[0]:
                raise ValueError(f"v3 spans overlap: {a} and {b}")
        if self.mention_starts[-1] + self.max_mention > self.binding_starts[0]:
            raise ValueError("mentions must precede bindings")
        if self.binding_starts[-1] + self.max_binding > self.chunk:
            raise ValueError("mentions and bindings must lie in chunk 0")
        if self.lookup_starts[0] < self.chunk:
            raise ValueError("lookup reads must be in chunk 1")
        if np.any(self.segment(self.cross_starts(segment_slots)) < 2) or \
                np.any(self.segment(self.cross_starts(segment_slots)) > 4):
            raise ValueError("cross-barrier reads must be in segments 2-4")
        return True

    def as_dict(self):
        return {"version": 3, "T": int(self.T), "S": int(self.S), "chunk": int(self.chunk),
                "n_slots": int(self.n_slots), "requested_slots": 16,
                "slot_fallback": bool(self.slot_fallback),
                "key_len": int(self.key_len), "value_len": int(self.value_len),
                "max_mention": int(self.max_mention), "max_binding": int(self.max_binding),
                "max_query": int(self.max_query), "gap": int(self.gap),
                "mention_starts": self.mention_starts.tolist(),
                "binding_starts": self.binding_starts.tolist(),
                "lookup_starts": self.lookup_starts.tolist(),
                "cross_starts": self.cross_starts().tolist(),
                "cross_segments": self.segment(self.cross_starts()).tolist()}


@dataclass(frozen=True)
class _V3Plan:
    binding: int
    mention: int
    query: int
    available: np.ndarray
    kind: str
    cross_order: np.ndarray
    cross_segments: np.ndarray
    lookup_order: np.ndarray


def _v3_plan(tables, split, seed, row_id, p, n_slots, *, force_kind=None):
    rng = np.random.default_rng(_u64("v3-plan", int(seed), split, int(row_id)))
    w = tables.wrappers[split]
    binding, mention, query = (int(rng.integers(len(w[k]))) for k in ("binding", "mention", "query"))
    order = rng.permutation(n_slots)
    available = np.zeros(n_slots, dtype=bool)
    available[order[:int(round(n_slots * float(p)))]] = True
    kind = force_kind or ("memory" if float(rng.random()) >= .25 else "lookup")
    cross_order = rng.permutation(n_slots)
    cross_segments = np.repeat((2, 3, 4), (n_slots // 3, n_slots // 3,
                                             n_slots - 2 * (n_slots // 3))).astype(np.int8)
    # Segment assignment is randomized independently of query order, then paired by slot.
    cross_segments = cross_segments[rng.permutation(n_slots)]
    return _V3Plan(binding, mention, query, available, kind, cross_order,
                   cross_segments, rng.permutation(n_slots))


def _v3_items(tables, split, seed, row_id, attempt, n):
    return _items(tables, split, seed ^ 0x53, row_id, attempt, n)


def _v3_put(out, kv, spans, positions, key_positions, kind, starts, keys, values, wrap, *, order=None):
    n, vl = len(starts), values.shape[1]
    kl = keys.shape[1] if keys is not None else 0
    vpos = np.empty((n, vl), dtype=np.int64)
    kpos = np.empty((n, kl), dtype=np.int64) if keys is not None else None
    for j in range(n):
        i = int(order[j]) if order is not None else j
        s = int(starts[j])
        if keys is None:
            seq = np.concatenate((wrap[0], values[i], wrap[1]))
            v0 = s + len(wrap[0])
        else:
            seq = np.concatenate((wrap[0], keys[i], wrap[1], values[i], wrap[2]))
            k0 = s + len(wrap[0]); kpos[i] = k0 + np.arange(kl)
            kv[k0:k0 + kl] = True; v0 = k0 + kl + len(wrap[1])
        vpos[i] = v0 + np.arange(vl); kv[v0:v0 + vl] = True
        out[s:s + len(seq)] = seq
        spans.append([s, s + len(seq), kind, i])
    positions[kind] = vpos
    if kpos is not None:
        key_positions[kind] = kpos


def inject_document_v3(host_row: np.ndarray, *, split: str, seed: int, p: float, row_id: int,
                       controls: bool = False, layout: EchoLayoutV3 | None = None,
                       tables: EchoTables | None = None, kind: str | None = None,
                       query_mode: str = "verbatim") -> tuple[np.ndarray, dict]:
    """Inject one v3 document.  ``kind`` is memory/lookup for training and memory for eval."""
    tables = tables or load_tables(); layout = layout or EchoLayoutV3()
    if split not in SPLITS or not 0 <= float(p) <= 1:
        raise ValueError("invalid v3 split or p")
    if query_mode not in ("verbatim", "paraphrase"):
        raise ValueError("invalid v3 query_mode")
    layout.validate(query_mode=query_mode)
    host = np.asarray(host_row)
    if host.ndim != 1 or len(host) != layout.row_len:
        raise ValueError(f"host row must have shape ({layout.row_len},)")
    n = layout.n_slots; plan = _v3_plan(tables, split, seed, row_id, p, n, force_kind=kind)
    w = tables.wrappers[split]; B, M, Q = w["binding"][plan.binding], w["mention"][plan.mention], w["query"][plan.query]
    for attempt in range(_MAX_ATTEMPTS):
        kb, ku, vb, vd = _v3_items(tables, split, seed, row_id, attempt, n)
        keys, unbound = tables.keys[split][kb], tables.keys[split][ku]
        values, distract = tables.values[split][vb], tables.values[split][vd]
        mention_vals = np.where(plan.available[:, None], values, distract)
        out = host.astype(np.uint32, copy=True); kv = np.zeros(layout.row_len, dtype=bool)
        spans=[]; pos={}; keypos={}
        _v3_put(out, kv, spans, pos, keypos, "mention", layout.mention_starts, None, mention_vals, M)
        _v3_put(out, kv, spans, pos, keypos, "binding", layout.binding_starts, keys, values, B)
        if plan.kind == "memory":
            starts = layout.cross_starts()
            # Put each slot in its assigned segment; segment counts are 5/5/6 (or 4/4/4).
            starts_by_seg = {s: starts[np.flatnonzero(layout.segment(starts) == s)] for s in (2, 3, 4)}
            curs = {s: 0 for s in starts_by_seg}
            ordered_starts = np.empty(n, dtype=np.int64)
            for slot in plan.cross_order:
                seg = int(plan.cross_segments[slot]); ordered_starts[slot] = starts_by_seg[seg][curs[seg]]; curs[seg] += 1
            wrap = B if query_mode == "verbatim" else Q
            _v3_put(out, kv, spans, pos, keypos, "cross_verbatim" if query_mode == "verbatim" else "cross_paraphrase",
                    ordered_starts, keys, values, wrap)
            read_segments = plan.cross_segments.copy()
            if controls:
                # Eval documents contain the in-segment lookup control in addition to
                # the primary cross-barrier read.  Training memory rows intentionally
                # omit it so the memory/lookup mixture is a real design variable.
                _v3_put(out, kv, spans, pos, keypos, "lookup_verbatim", layout.lookup_starts,
                        keys, values, B, order=plan.lookup_order)
        else:
            # Training lookup rows have all reads in segment 1/chunk 1 and no cross reads.
            _v3_put(out, kv, spans, pos, keypos, "lookup_verbatim", layout.lookup_starts,
                    keys, values, B, order=plan.lookup_order)
            read_segments = np.ones(n, dtype=np.int8)
        drawn = np.concatenate((_bigrams(keys), _bigrams(unbound), _bigrams(values), _bigrams(distract)))
        bi = out[:-1].astype(np.int64) * _BIGRAM + out[1:].astype(np.int64)
        if not (np.isin(bi, drawn) & ~(kv[:-1] & kv[1:])).any(): break
    else:
        raise RuntimeError(f"v3 host collisions persisted for row {row_id}")
    spans.sort()
    nll = lambda x: (np.asarray(x, dtype=np.int64) - 1).tolist()
    read_positions = {k: pos[k].tolist() for k in pos if k.startswith(("cross_", "lookup_"))}
    read_nll_positions = {k: nll(pos[k]) for k in pos if k.startswith(("cross_", "lookup_"))}
    read_seg = {k: (read_segments if k.startswith("cross_") else np.ones(n, dtype=np.int8)).tolist()
                for k in read_positions}
    read_stratum = {k: np.where(plan.available, "copy_available", "copy_blocked").tolist() for k in read_positions}
    rec = {"version": 3, "split": split, "row_id": int(row_id), "seed": int(seed), "p": float(p),
           "controls": bool(controls), "kind": plan.kind, "query_mode": query_mode,
           "n_slots": n, "slot_fallback": n != 16,
           "slot_fallback_report": ("requested 16, using 12 because chunk 0 does not fit" if n != 16 else None),
           "available": plan.available.tolist(),
           "stratum": np.where(plan.available, "copy_available", "copy_blocked").tolist(),
           "key_tokens": keys.tolist(), "value_tokens": values.tolist(), "distractor_tokens": distract.tolist(),
           "mention_value_tokens": mention_vals.tolist(), "mention_positions": pos["mention"].tolist(),
           "source_key_positions": keypos["binding"].tolist(), "source_value_positions": pos["binding"].tolist(),
           "source_nll_positions": nll(pos["binding"]), "spans": spans, "injected": True,
           "read_value_positions": read_positions, "read_nll_positions": read_nll_positions,
           "read_segments": read_seg, "read_strata": read_stratum,
           "cross_segments": plan.cross_segments.tolist(), "cross_order": plan.cross_order.tolist(),
           "lookup_order": plan.lookup_order.tolist()}
    # Aliases make metadata pleasant for callers and keep every position available by read kind.
    for k, v in read_positions.items(): rec[f"{k}_value_positions"] = v; rec[f"{k}_nll_positions"] = read_nll_positions[k]
    return out, rec


def write_mask_v3(records: list[Mapping], T: int, *, forward: str = "N",
                  segment_len: int = 2048) -> np.ndarray:
    """Inner-write mask for v3.

    ``N``/``Z`` mask reads; ``A``/``P-A`` additionally mask binding spans.  ``T``
    deliberately leaves every target writable, matching the training memory rows.
    """
    m = np.ones((len(records), int(T)), dtype=bool)
    fwd = forward.upper()
    if fwd == "Z":
        m[:, :min(int(T), int(segment_len))] = False
        kinds = set(V3_READ_KINDS)
    elif fwd in ("T", "T-A"):
        if fwd == "T-A":
            kinds = {"binding"}
        else:
            return m
    else:
        kinds = set(V3_READ_KINDS)
        if fwd in ("A", "P-A"):
            kinds.add("binding")
    for j, rec in enumerate(records):
        for s, e, kind, _ in rec.get("spans", ()):
            if kind in kinds: m[j, max(0, int(s)-1):min(int(T), int(e)-1)] = False
    return m


class EchoInjectorV3:
    """Streaming v3 injector with deterministic memory/lookup row kinds."""
    version = 3

    def __init__(self, *, split="train", f=0.5, p=0.5, lookup_share=0.25, seed=0, T=8192,
                 S=2048, excluded_vocab, n_slots=16, tables=None, layout=None):
        if split not in SPLITS or not 0 <= float(f) <= 1 or not 0 <= float(p) <= 1 or not 0 <= float(lookup_share) <= 1:
            raise ValueError("invalid v3 injector parameters")
        self.split, self.f, self.p, self.lookup_share, self.seed = split, float(f), float(p), float(lookup_share), int(seed)
        self.tables = tables or load_tables(); self.layout = layout or EchoLayoutV3(T=T, S=S, n_slots=n_slots)
        if self.layout.T != int(T): raise ValueError("layout T mismatch")
        check_vocab(self.tables, excluded_vocab); self.excluded_vocab_sha256 = hashlib.sha256(np.asarray(excluded_vocab).tobytes()).hexdigest()
        self.layout.validate()

    def selected(self, ids):
        ids=np.asarray(ids,dtype=np.int64).reshape(-1); return np.fromiter((_selected(self.f,self.seed,int(i)) for i in ids),dtype=bool,count=len(ids))

    def kind(self, row_id):
        return "lookup" if _u64("v3-kind", self.seed, self.split, int(row_id)) / float(1 << 64) < self.lookup_share else "memory"

    def injected_tokens(self, ids):
        ids=np.asarray(ids,dtype=np.int64).reshape(-1); sel=self.selected(ids); out=np.zeros(len(ids),dtype=np.int64)
        for j in np.flatnonzero(sel):
            rec = self.document(np.full(self.layout.row_len, 128000, dtype=np.uint32), int(ids[j]), controls=False)[1]
            out[j] = sum(int(e)-int(s) for s,e,k,_ in rec["spans"])
        return out

    def exposure_accounting(self, ids):
        ids=np.asarray(ids,dtype=np.int64).reshape(-1); sel=self.selected(ids); docs=int(sel.sum()); av=int(round(self.layout.n_slots*self.p)); vl=self.layout.value_len
        mem=sum(sel[j] and self.kind(int(i)) == "memory" for j,i in enumerate(ids)); look=docs-mem
        return {"version":3,"rows":len(ids),"injected_rows":docs,"memory_rows":int(mem),"lookup_rows":int(look),
                "memory_share":float(mem/max(1,docs)),"lookup_share":float(look/max(1,docs)),
                "copy_available_slots":av,"copy_blocked_slots":self.layout.n_slots-av,
                "source_writes":{"copy_available":docs*av*vl,"copy_blocked":docs*(self.layout.n_slots-av)*vl},
                "query_exposures":{"copy_available":docs*av*vl,"copy_blocked":docs*(self.layout.n_slots-av)*vl},
                "injected_token_budget":int(self.injected_tokens(ids).sum()),"token_share":float(self.injected_tokens(ids).sum())/max(1,len(ids)*self.layout.T)}

    def document(self, host_row, row_id, *, p=None, controls=False, kind=None, query_mode="verbatim"):
        if kind is None and not controls: kind=self.kind(row_id)
        return inject_document_v3(host_row, split=self.split, seed=self.seed, p=self.p if p is None else p,
                                  row_id=row_id, controls=controls, layout=self.layout, tables=self.tables,
                                  kind=kind, query_mode=query_mode)

    def __call__(self, rows_uint32, global_row_index):
        rows=np.asarray(rows_uint32); ids=np.asarray(global_row_index,dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.layout.row_len or ids.shape != (len(rows),): raise ValueError("v3 rows/ids shape mismatch")
        out=rows.astype(np.uint32,copy=True); sel=self.selected(ids); records=[None]*len(rows)
        for j in np.flatnonzero(sel): out[j],records[j]=self.document(out[j],int(ids[j]),controls=False)
        return out,{"version":3,"split":self.split,"f":self.f,"p":self.p,"lookup_share":self.lookup_share,"seed":self.seed,
                    "global_row_index":ids.tolist(),"selected":sel.tolist(),"rows":records,
                    "rows_sha256": rows_sha256(out[sel]), "manifest_sha256":self.manifest()["manifest_sha256"]}

    def eval_documents(self, host_rows, *, row_ids=None, p=None, query_mode="verbatim"):
        rows=np.asarray(host_rows); ids=np.arange(len(rows),dtype=np.int64) if row_ids is None else np.asarray(row_ids,dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != self.layout.row_len or ids.shape != (len(rows),): raise ValueError("v3 eval rows/ids shape mismatch")
        recs=[]; out=np.empty(rows.shape,dtype=np.uint32)
        for j,rid in enumerate(ids): out[j],rec=self.document(rows[j],int(rid),p=self.p if p is None else p,controls=True,kind="memory",query_mode=query_mode); recs.append(rec)
        return out,{"version":3,"split":self.split,"p":self.p if p is None else float(p),"row_ids":ids.tolist(),
                    "rows":recs,"manifest_sha256":self.manifest()["manifest_sha256"]}

    def _manifest_params(self):
        return {"version":3,"split":self.split,"f":self.f,"p":self.p,"lookup_share":self.lookup_share,"seed":self.seed,
                "T":self.layout.T,"S":self.layout.S,"chunk":self.layout.chunk,"n_slots":self.layout.n_slots,
                "excluded_vocab_sha256":self.excluded_vocab_sha256,"tables_sha256":self.tables.sha256,"layout":self.layout.as_dict()}

    def manifest(self):
        p=self._manifest_params(); p["params_sha256"]=hashlib.sha256(_canonical(p)).hexdigest(); p["pools_sha256"]=self.tables.splits_sha256
        p["manifest_sha256"]=manifest_sha256(p); return p


def make_injector_v3(*, split="train", f=0.5, p=0.5, lookup_share=0.25, seed=0, T=8192, S=2048,
                     excluded_vocab, n_slots=16, tables=None):
    return EchoInjectorV3(split=split, f=f, p=p, lookup_share=lookup_share, seed=seed, T=T, S=S,
                          excluded_vocab=excluded_vocab, n_slots=n_slots, tables=tables)


__all__ = ["EchoLayout", "EchoTables", "EchoInjector", "load_tables", "check_vocab",
           "make_injector", "inject_document", "injected_length", "write_mask",
           "build_eval_documents", "manifest_sha256", "rows_sha256", "write_manifest",
           "CONTROL_KINDS", "SPAN_KINDS", "TABLE_PATH"]
__all__ += ["EchoLayoutV3", "EchoInjectorV3", "inject_document_v3", "make_injector_v3",
            "write_mask_v3", "V3_READ_KINDS", "V3_CONTROL_KINDS"]
