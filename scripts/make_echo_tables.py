"""Build the committed token tables of the SOQ echo-in-text generator (generator v2).

The project runtime never imports a tokenizer: ``tttmem.p2.echo_text`` reads only the
JSON written here.  Run offline against the pinned, locally cached Llama-3 tokenizer::

    HF_HUB_OFFLINE=1 uv run --isolated --no-project \\
        --with tokenizers --with huggingface_hub --with numpy \\
        python scripts/make_echo_tables.py \\
        --excluded data/p2data/excluded_vocab.npy \\
        --freq data/p2data/eval.bin \\
        --out src/tttmem/p2/echo_text_tables.json

What the table holds, per split (train / cal / test, all disjoint):

- paraphrased natural wrappers for the binding, mention and query sentences, each
  stored as literal token pieces around the ``{K}``/``{V}`` slots;
- nonce person-like keys (two syllable words, 2 + 2 tokens) and nonce lowercase values
  (one syllable word, 3 tokens), built from syllable-like vocabulary tokens.

Filters (all recorded in ``stats``): exact token length; the composed ids equal the
tokenizer's encoding of the word; no excluded (``excluded_vocab.npy``) or special
token; no token shared with any wrapper; no dictionary word (``/usr/share/dict/words``);
no run of three consonants or vowels; no token repeated inside an item; the first value token is not among the 5000 most frequent DCLM tokens (counted on
``--freq``) and starts at most ``FIRST_CAP`` values per split (keys: ``KEY_FIRST_CAP``).  Every key and value is
also checked in every sentence template: concatenating the separately tokenized pieces
must equal the encoding of the full sentence.  A random sample of full sentences per
split re-checks this after filtering (the canonical-match rate).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

TOKENIZER_REPO = "NousResearch/Meta-Llama-3-8B"
TOKENIZER_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"
SEED = 20261010
KEY_LEN = 4          # " Vorlak Teshin"-like: two words of two tokens each
VALUE_LEN = 3        # one nonce word of three tokens
POOL = {"train": 8192, "cal": 4096, "test": 4096}
FIRST_CAP = 8        # max values sharing a first token, per split
KEY_FIRST_CAP = 16   # keys are never read targets; a looser cap on their first token
FIRST_RANK_MIN = 5000
KEY_RANK_MIN = 3000
N_CANDIDATES = 400_000
N_JOINT = 20_000     # sampled full sentences per split and sentence type
SPECIAL_MIN = 128000

# Short, plain English.  {K} and {V} carry their own leading space.  Each sentence is a
# line of its own (leading and trailing newline), which keeps the host text intact.
PARAPHRASES = {
    "train": {
        "binding": [
            "\nThe access code for{K} is{V}.\n",
            "\nThe door code for{K} is{V}.\n",
            "\nThe code assigned to{K} is{V}.\n",
            "\nFor{K}, the code is{V}.\n",
            "\nThe login code of{K} is{V}.\n",
            "\nWe gave{K} the code{V}.\n",
        ],
        "mention": [
            "\nEarlier, someone scribbled{V} on a sticky note.\n",
            "\nA note on the desk just said{V}.\n",
            "\nSomeone had written{V} on the whiteboard.\n",
            "\nThe word{V} was printed on a sticker.\n",
            "\nI once saw{V} written on a napkin.\n",
            "\nThe screen briefly showed{V}.\n",
        ],
        "query": [
            "\nQ: What is the access code for{K}? A:{V}\n",
            "\nQuestion: What code does{K} use? Answer:{V}\n",
            "\nQ: Which code belongs to{K}? A:{V}\n",
            "\nQ: What was the code for{K}? A:{V}\n",
            "\nQuestion: What is the code of{K}? Answer:{V}\n",
            "\nQ: Tell me the code for{K}. A:{V}\n",
        ],
    },
    "cal": {
        "binding": [
            "\nThe secret code for{K} is{V}.\n",
            "\nThe entry code given to{K} is{V}.\n",
            "\nRemember that the code for{K} is{V}.\n",
        ],
        "mention": [
            "\nA crumpled receipt read{V}.\n",
            "\nThe label on the box said{V}.\n",
            "\nSomebody had typed{V} into the search bar.\n",
        ],
        "query": [
            "\nQ: What is the code for{K}? A:{V}\n",
            "\nQuestion: Which code was given to{K}? Answer:{V}\n",
            "\nQ: Do you recall the code of{K}? A:{V}\n",
        ],
    },
    "test": {
        "binding": [
            "\nThe personal code of{K} is{V}.\n",
            "\nThe code that belongs to{K} is{V}.\n",
            "\nOur records list the code for{K} as{V}.\n",
        ],
        "mention": [
            "\nThe back of the card read{V}.\n",
            "\nA voice on the radio said{V}.\n",
            "\nThe old notebook mentioned{V}.\n",
        ],
        "query": [
            "\nQ: What code goes with{K}? A:{V}\n",
            "\nQuestion: What is{K} using as a code? Answer:{V}\n",
            "\nQ: Name the code for{K}. A:{V}\n",
        ],
    },
}
SPLITS = ("train", "cal", "test")
KINDS = ("binding", "mention", "query")
SLOTS = {"binding": ("K", "V"), "mention": ("V",), "query": ("K", "V")}

# Syllable-like vocabulary pieces: an English-legal onset, one or two vowels, a short coda.
_ONSETS = ("b|c|d|f|g|h|j|k|l|m|n|p|r|s|t|v|w|z|bl|br|ch|cl|cr|dr|fl|fr|gl|gr|kl|kr|pl|pr|"
           "sh|sk|sl|sm|sn|sp|st|sw|th|tr|tw|vr|zh")
_CODA = "bcdfghklmnprstvxz"
SYL_HEAD = re.compile(rf"^(?:{_ONSETS})[aeiou]{{1,2}}[{_CODA}]{{0,2}}$")
SYL_TAIL = re.compile(rf"^[aeiou]?(?:{_ONSETS})?[aeiou]{{1,2}}[{_CODA}]{{0,2}}$")
# Pronounceable words: no run of three consonants or three vowels.
CLUSTER = re.compile(r"[bcdfghjklmnpqrstvwxz]{3}|[aeiouy]{3}")
CODEISH = {"href", "http", "html", "json", "src", "var", "def", "img", "jpg", "png", "pdf",
           "php", "css", "str", "len", "url", "xml", "api", "sql", "dev", "env", "tmp",
           "lib", "obj", "msg", "nav", "btn", "col", "div", "pos", "num", "sub", "sup",
           "goog", "wget", "blog", "web", "app", "com", "org", "net"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def split_template(t: str, kind: str) -> list[str]:
    """Literal pieces around the slots: K/V templates -> [pre, mid, post]; V -> [pre, post]."""
    if kind == "mention":
        pre, post = t.split("{V}")
        if "{K}" in t:
            raise ValueError(t)
        return [pre, post]
    pre, rest = t.split("{K}")
    mid, post = rest.split("{V}")
    return [pre, mid, post]


def fill(pieces: list[str], kind: str, key: str = "", value: str = "") -> str:
    if kind == "mention":
        return pieces[0] + value + pieces[1]
    return pieces[0] + key + pieces[1] + value + pieces[2]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excluded", required=True)
    ap.add_argument("--freq", required=True, help="uint32 token file for DCLM token frequencies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", default="/usr/share/dict/words")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION)
    tok = Tokenizer.from_file(tok_path)
    tok_sha = sha256_bytes(Path(tok_path).read_bytes())
    excluded = np.load(args.excluded, allow_pickle=False)
    if excluded.dtype != np.bool_ or excluded.ndim != 1:
        raise ValueError("excluded_vocab.npy must be a 1-D bool mask")
    excluded_sha = sha256_bytes(np.ascontiguousarray(excluded).tobytes())
    freq_tokens = np.memmap(args.freq, dtype=np.uint32, mode="r")
    counts = np.bincount(np.asarray(freq_tokens), minlength=len(excluded))
    rank = np.empty(len(counts), dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(len(counts))
    words = {w.strip().lower() for w in open(args.words, encoding="utf-8")}

    def enc(texts: list[str]) -> list[list[int]]:
        return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]

    # ---------------------------------------------------------------- wrappers
    wrappers: dict = {}
    wrapper_tokens: set[int] = set()
    for split in SPLITS:
        wrappers[split] = {}
        for kind in KINDS:
            items = []
            for text in PARAPHRASES[split][kind]:
                pieces = split_template(text, kind)
                ids = enc(pieces)
                for p, i in zip(pieces, ids):
                    if not i or tok.decode(i) != p:
                        raise ValueError(f"wrapper piece does not round-trip: {p!r}")
                    if any(t >= SPECIAL_MIN for t in i):
                        raise ValueError(f"special token in wrapper {p!r}")
                wrapper_tokens.update(t for i in ids for t in i)
                items.append({"text": text, "pieces": pieces, "ids": ids})
            wrappers[split][kind] = items
    all_texts = [w["text"] for s in SPLITS for k in KINDS for w in wrappers[s][k]]
    if len(set(all_texts)) != len(all_texts):
        raise ValueError("paraphrases repeat across splits")

    # ---------------------------------------------------------------- vocabulary pieces
    vocab = [tok.decode([i]) for i in range(SPECIAL_MIN)]
    ok_tok = ~excluded[:SPECIAL_MIN]

    def usable(i: int) -> bool:
        return bool(ok_tok[i]) and i not in wrapper_tokens

    heads = [i for i, s in enumerate(vocab)
             if usable(i) and s.startswith(" ") and 3 <= len(s) - 1 <= 4 and SYL_HEAD.match(s[1:])
             and s[1:] not in words and s[1:] not in CODEISH and rank[i] >= FIRST_RANK_MIN]
    cap_heads = [i for i, s in enumerate(vocab)
                 if usable(i) and s.startswith(" ") and 3 <= len(s) - 1 <= 4 and s[1].isupper()
                 and s[2:].islower() and SYL_HEAD.match(s[1:].lower())
                 and s[1:].lower() not in words and s[1:].lower() not in CODEISH
                 and rank[i] >= KEY_RANK_MIN]
    tails = [i for i, s in enumerate(vocab)
             if usable(i) and 2 <= len(s) <= 3 and SYL_TAIL.match(s)
             and (len(s) < 3 or s not in words)]

    rng = random.Random(SEED)
    val_c = sorted({(rng.choice(heads), rng.choice(tails), rng.choice(tails))
                    for _ in range(N_CANDIDATES)})
    key_c = sorted({(rng.choice(cap_heads), rng.choice(tails), rng.choice(cap_heads), rng.choice(tails))
                    for _ in range(N_CANDIDATES)})
    stats: dict = {"candidates": {"values": len(val_c), "keys": len(key_c)},
                   "vocab_pieces": {"value_heads": len(heads), "key_heads": len(cap_heads),
                                    "tails": len(tails)}}

    def word_filter(cands, n_words):
        strs = ["".join(vocab[i] for i in c) for c in cands]
        ids = enc(strs)
        keep, fail_canon, fail_dict = [], 0, 0
        for s, c, e in zip(strs, cands, ids):
            if tuple(e) != tuple(c):
                fail_canon += 1
                continue
            if len(set(c)) != len(c):     # no token repeats inside one key or value
                fail_dict += 1
                continue
            ws = s.split()
            if len(ws) != n_words or any(w.lower() in words or CLUSTER.search(w.lower())
                                         for w in ws):
                fail_dict += 1
                continue
            keep.append((s, c))
        return keep, {"standalone_noncanonical": fail_canon, "repeat_dictionary_or_cluster": fail_dict,
                      "kept": len(keep)}

    vals, stats["value_word_filter"] = word_filter(val_c, 1)
    keys, stats["key_word_filter"] = word_filter(key_c, 2)

    # ---------------------------------------------------------------- context canonicality
    dummy_k = keys[0][0]
    dummy_v = vals[0][0]

    def context_ok(items, slot):
        """Keep items whose pieces concatenate to the full-sentence encoding in every
        template (of every split) that uses the slot."""
        ok = np.ones(len(items), dtype=bool)
        checks = 0
        for split in SPLITS:
            for kind in KINDS:
                if slot not in SLOTS[kind]:
                    continue
                for w in wrappers[split][kind]:
                    pieces, pids = w["pieces"], w["ids"]
                    texts, want = [], []
                    for s, c in items:
                        k, v = (s, dummy_v) if slot == "K" else (dummy_k, s)
                        texts.append(fill(pieces, kind, k, v))
                        kid = list(c) if slot == "K" else list(keys[0][1])
                        vid = list(c) if slot == "V" else list(vals[0][1])
                        if kind == "mention":
                            want.append(pids[0] + vid + pids[1])
                        else:
                            want.append(pids[0] + kid + pids[1] + vid + pids[2])
                    got = enc(texts)
                    ok &= np.asarray([g == x for g, x in zip(got, want)], dtype=bool)
                    checks += len(items)
        return ok, checks

    ok_v, nv = context_ok(vals, "V")
    ok_k, nk = context_ok(keys, "K")
    stats["context_canonical"] = {
        "values_pass": int(ok_v.sum()), "values_total": len(vals),
        "values_pass_rate": float(ok_v.mean()), "value_sentence_checks": nv,
        "keys_pass": int(ok_k.sum()), "keys_total": len(keys),
        "keys_pass_rate": float(ok_k.mean()), "key_sentence_checks": nk}
    vals = [x for x, o in zip(vals, ok_v) if o]
    keys = [x for x, o in zip(keys, ok_k) if o]

    # ---------------------------------------------------------------- split assignment
    def assign(items, name, cap):
        order = list(range(len(items)))
        random.Random(f"{SEED}-{name}").shuffle(order)
        used: dict[str, collections.Counter] = {s: collections.Counter() for s in SPLITS}
        out = {s: [] for s in SPLITS}
        taken = set()
        for split in SPLITS:
            for j in order:
                if len(out[split]) >= POOL[split]:
                    break
                if j in taken:
                    continue
                first = items[j][1][0]
                if used[split][first] >= cap:
                    continue
                used[split][first] += 1
                out[split].append(items[j])
                taken.add(j)
            if len(out[split]) < POOL[split]:
                raise RuntimeError(f"{name}: only {len(out[split])} items for {split}")
        return out, {s: {"size": len(out[s]), "distinct_first_tokens": len(used[s]),
                         "max_first_token_count": max(used[s].values())} for s in SPLITS}

    key_pools, stats["key_pools"] = assign(keys, "keys", KEY_FIRST_CAP)
    val_pools, stats["value_pools"] = assign(vals, "values", FIRST_CAP)

    # ---------------------------------------------------------------- joint sample check
    joint = {}
    jr = random.Random(f"{SEED}-joint")
    for split in SPLITS:
        joint[split] = {}
        for kind in KINDS:
            texts, want = [], []
            for _ in range(N_JOINT):
                w = jr.choice(wrappers[split][kind])
                ks, kid = jr.choice(key_pools[split])
                vs, vid = jr.choice(val_pools[split])
                texts.append(fill(w["pieces"], kind, ks, vs))
                p = w["ids"]
                want.append(p[0] + list(vid) + p[1] if kind == "mention"
                            else p[0] + list(kid) + p[1] + list(vid) + p[2])
            got = enc(texts)
            joint[split][kind] = float(np.mean([g == x for g, x in zip(got, want)]))
    stats["joint_canonical_match_rate"] = joint
    if min(v for s in joint.values() for v in s.values()) < 1.0:
        raise RuntimeError(f"joint canonical match below 1: {joint}")

    # ---------------------------------------------------------------- write
    splits = {}
    for split in SPLITS:
        splits[split] = {
            "binding": wrappers[split]["binding"], "mention": wrappers[split]["mention"],
            "query": wrappers[split]["query"],
            "keys": {"ids": [list(c) for _, c in key_pools[split]],
                     "text": [s for s, _ in key_pools[split]]},
            "values": {"ids": [list(c) for _, c in val_pools[split]],
                       "text": [s for s, _ in val_pools[split]]},
        }
    payload = {
        "version": 2, "key_len": KEY_LEN, "value_len": VALUE_LEN,
        "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION,
                      "tokenizer_json_sha256": tok_sha},
        "excluded_vocab_sha256": excluded_sha,
        "build": {"seed": SEED, "pool": POOL, "first_cap": FIRST_CAP, "key_first_cap": KEY_FIRST_CAP,
                  "first_rank_min": FIRST_RANK_MIN, "key_rank_min": KEY_RANK_MIN, "freq_source": Path(args.freq).name,
                  "freq_tokens": int(freq_tokens.size), "n_candidates": N_CANDIDATES,
                  "n_joint": N_JOINT, "words": args.words,
                  "command": "HF_HUB_OFFLINE=1 uv run --isolated --no-project --with tokenizers "
                             "--with huggingface_hub --with numpy python scripts/make_echo_tables.py "
                             "--excluded <p2data>/excluded_vocab.npy --freq <p2data>/eval.bin "
                             "--out src/tttmem/p2/echo_text_tables.json"},
        "stats": stats,
        "splits": splits,
    }
    payload["splits_sha256"] = sha256_bytes(canonical_json(splits))
    payload["sha256"] = sha256_bytes(canonical_json(payload))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(out), "sha256": payload["sha256"], "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
