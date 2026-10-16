"""XPO-0: exposure geometry of code vs DCLM under a segment-blocked layout (token-only, CPU).

Question: does a code evaluation set give more memory-relevant targets (cross-segment repeats,
barrier probes) per document than DCLM?
Layout: attention block-diagonal causal over segments of S = 2048 tokens; barriers after chunks
2, 4 and 6 (chunk = 1024 tokens). Classes of a loss-valid target t with a defined 4-gram key
(x_{t-2}, x_{t-1}, x_t -> y_t; defined only inside one segment):
- within-segment repeat: the most recent earlier occurrence of the key is in t's segment;
- cross-segment repeat: earlier occurrences exist, all in earlier segments;
- non-repeat: no earlier occurrence.
Probes (``tttmem.p2.probe.find_probes``) are first occurrences in chunks 4-7 of 4-grams seen in
chunks 0-3; a document is eligible with >= 5 probes and exact non-focal matching.

Sets (1024 documents each, first 8192 tokens after BOS):
- DCLM-end: the first N P2-1 endpoint documents (``eval_rows`` of a P2-1 ``eval_meta.npz``,
  rows of the P2 evaluation pool ``eval.bin``, uint32 [n, 8193]);
- code-file: one source file with >= 8192 tokens;
- code-repo: the files of one repository (within the shard), sorted by path, joined by a blank line.
Code comes from codeparrot/github-code-clean, shard 0 (downloaded through the Hugging Face hub);
tokenizer: Llama-3 (NousResearch/Meta-Llama-3-8B at a pinned revision, as in ``tttmem.p2.data``).
``--prog-only`` keeps programming languages only (drops EXCLUDED_LANGS); ``--min-chars`` is the
character prefilter for code files.
Output: a text table on stdout and in ``--out-dir`` (exposure_geometry.txt,
exposure_geometry_prog.txt or exposure_geometry_prog_min<N>.txt).
Needs pandas (``uv run --locked --with pandas ...``). Peak memory is 2.5-2.8 GB with one tokenizer
thread (``RAYON_NUM_THREADS=1``, set before the run); with the default four threads it exceeds 2.9 GB.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "exposure-geometry"))
ap.add_argument("--n-docs", type=int, default=1024)
ap.add_argument("--min-chars", type=int, default=24000, help="char prefilter for code files")
ap.add_argument("--prog-only", action="store_true",
                help="keep programming languages only (drop markup/config/data: EXCLUDED_LANGS)")
ap.add_argument("--eval-bin", default="data/p2data/eval.bin", help="P2 evaluation pool, uint32 [n, 8193]")
ap.add_argument("--excluded-vocab", default="data/p2data/excluded_vocab.npy")
ap.add_argument("--eval-meta", default="outputs/p21/e2e2/attempt0/eval_meta.npz",
                help="any P2-1 eval_meta.npz (eval_rows = endpoint documents)")
ap.add_argument("--out-dir", default="outputs/analysis/text-repeat-exposure")
args = ap.parse_args()
os.environ["PROCESS_NAME"] = args.process_name
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(k, "4")
try:
    import setproctitle
    setproctitle.setproctitle(args.process_name)
except ImportError:
    pass

import importlib.util
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from tttmem.p2.probe import find_probes  # noqa: E402

T1, C, BOS = 8193, 1024, 128000
T = T1 - 1


def classify(rows: np.ndarray, S: int):
    """rows [N, T+1] -> class [N, T]: -1 not loss-valid, 0 key undefined, 1 non-repeat,
    2 within-segment repeat, 3 cross-segment repeat."""
    N = rows.shape[0]
    x = rows[:, :-1].astype(np.uint64)
    y = rows[:, 1:].astype(np.uint64)
    pos = np.arange(T)
    seg_start = (pos // S) * S
    valid = rows[:, 1:] != BOS
    cross = (pos % S == S - 1) & (pos < T - 1)        # target crosses a segment start
    valid &= ~cross[None, :]
    keyok = valid & ((pos - 2) >= seg_start)[None, :]
    x1 = np.concatenate([np.zeros((N, 1), np.uint64), x[:, :-1]], 1)
    x2 = np.concatenate([np.zeros((N, 2), np.uint64), x[:, :-2]], 1)
    # exact 4-gram match: lexsort by (row, x2, x1, x, y, position), compare adjacent tuples
    row = np.repeat(np.arange(N), T)
    cols = [x2.reshape(-1), x1.reshape(-1), x.reshape(-1), y.reshape(-1)]
    ok = np.flatnonzero(keyok.reshape(-1))            # only defined keys take part
    order = ok[np.lexsort([np.tile(pos, N)[ok]] + [c[ok] for c in cols[::-1]] + [row[ok]])]
    same = np.zeros(len(order), bool)
    same[1:] = row[order[1:]] == row[order[:-1]]
    for c in cols:
        same[1:] &= c[order[1:]] == c[order[:-1]]
    prev = np.full(N * T, -1, np.int64)
    prev[order[1:][same[1:]]] = order[:-1][same[1:]]
    prev = prev.reshape(N, T) - (np.arange(N)[:, None] * T)   # back to in-row positions
    has_prev = (prev >= 0) & keyok
    cls = np.full((N, T), -1, np.int8)
    cls[valid] = 0
    cls[keyok] = 1
    within = has_prev & (prev >= seg_start[None, :])
    cls[has_prev & within] = 2
    cls[has_prev & ~within] = 3
    return cls


EXCL = args.excluded_vocab
TOK_REPO, TOK_REV = "NousResearch/Meta-Llama-3-8B", "315b20096dc791d381d514deb5f8bd9c8d6d3061"
N = args.n_docs
CPT_NOTE = []
EXCLUDED_LANGS = {"HTML", "Markdown", "CSS", "TeX", "SQL", "Makefile", "CMake", "Dockerfile", "Batchfile"}


def encode(tok, texts, sub=16):
    """``tok.encode_batch(texts, add_special_tokens=False)`` in sub-batches, yielded in order
    (memory: an Encoding keeps every token's string and offsets, and some files are megabytes)."""
    for j in range(0, len(texts), sub):
        yield from tok.encode_batch(texts[j:j + sub], add_special_tokens=False)


def build_docs(tok):
    path = hf_hub_download("codeparrot/github-code-clean", "data/train-00000-of-00880.parquet",
                           repo_type="dataset")
    # Memory: the shard is read one row group at a time, and the code text stays in the Arrow
    # column and is fetched row by row (``row`` = shard row); a one-shot read plus carrying the
    # text through the pandas filter/sort/groupby copies below would need more than 4 GB.
    pf = pq.ParquetFile(path)
    t = pa.concat_tables([pf.read_row_group(i, columns=["code", "repo_name", "path", "language"])
                          for i in range(pf.num_row_groups)])
    code = t.column("code")
    tab = t.drop_columns(["code"]).to_pandas()
    del t
    tab["row"] = np.arange(len(tab))
    if args.prog_only:
        tab = tab[~tab["language"].isin(EXCLUDED_LANGS)]
    tab["n"] = pc.utf8_length(code).to_numpy()[tab["row"].to_numpy()].astype(np.int64)  # characters
    out, langs = {}, {}
    # file-level: long single files
    big = tab[tab["n"] >= args.min_chars].sort_values(["repo_name", "path"])
    docs, lg, cpt = [], [], []
    for s in range(0, len(big), 256):
        part = big.iloc[s:s + 256]
        part_code = [code[int(i)].as_py() for i in part["row"]]
        for enc, lang, txt in zip(encode(tok, part_code), part["language"], part_code):
            if len(enc.ids) >= T:
                docs.append(np.asarray([BOS] + enc.ids[:T], np.uint32)); lg.append(lang)  # memory: no int lists
                cpt.append(len(txt) / len(enc.ids))
        if len(docs) >= N:
            break
    out["code-file"], langs["code-file"] = np.asarray(docs[:N], np.uint32), lg[:N]
    c = np.asarray(cpt[:N])
    CPT_NOTE.append(f"code-file chars/token (docs kept, prefilter {args.min_chars} chars): q01/q10/q50/q90 "
                    f"{np.quantile(c, [.01, .1, .5, .9]).round(2).tolist()}; share < {24000 / T:.2f} "
                    f"(dropped by a 24000-char prefilter) = {(c < 24000 / T).mean():.3f}")
    print(f"code-file: {len(docs)} files >= {T} tokens among the first {min(s + 256, len(big))} of "
          f"{len(big)} files >= 24000 chars ({len(tab)} files in the shard after filters)", flush=True)
    # repo-level: concatenation of a repository's files (only those in this shard)
    g = tab.groupby("repo_name")
    tot = g["n"].sum()
    repos = [r for r in tot[tot >= 24000].index]
    docs, lg = [], []
    for s in range(0, len(repos), 128):
        texts, ls = [], []
        for r in repos[s:s + 128]:
            sub = g.get_group(r).sort_values("path")
            texts.append("\n\n".join(code[int(i)].as_py() for i in sub["row"]))
            ls.append(sub["language"].mode().iloc[0])
        for enc, lang in zip(encode(tok, texts), ls):
            if len(enc.ids) >= T:
                docs.append(np.asarray([BOS] + enc.ids[:T], np.uint32)); lg.append(lang)  # memory: no int lists
        if len(docs) >= N:
            break
    out["code-repo"], langs["code-repo"] = np.asarray(docs[:N], np.uint32), lg[:N]
    return out, langs


def summarize(name, rows, excl, L):
    cls = classify(rows, 2048)
    nv = (cls >= 0).sum()
    sh = [(cls == k).sum() / nv for k in range(4)]
    inc = ~excl[rows[:, 1:].astype(np.int64)]          # target not whitespace/punct/symbol-only
    nvi = ((cls >= 0) & inc).sum()
    shi = [((cls == k) & inc).sum() / nvi for k in range(4)]
    per_doc = (cls == 3).sum(1)
    probes = [find_probes(d[:-1], d[1:], C, valid=d[1:] != BOS, excluded=excl, barriers=(2, 4, 6), seed=i)
              for i, d in enumerate(rows)]
    npb = np.array([len(p.pos) for p in probes])
    elig = np.array([len(p.pos) >= 5 and p.nf_ok for p in probes])
    P = [p for p, e in zip(probes, elig) if e]
    st = np.concatenate([p.stratum for p in P]) if P else np.zeros(0)
    dist = np.concatenate([p.dist for p in P]) if P else np.zeros(1)
    L.append(f"{name:<10} {len(rows):5d} | {sh[1]:.4f} {sh[2]:.4f} {sh[3]:.4f} | {np.median(per_doc):7.0f} "
             f"{(per_doc >= 20).mean():6.3f} | {elig.mean():6.3f} {np.median(npb[elig]) if elig.any() else 0:7.0f} "
             f"{npb[elig].sum():8d} | " + "/".join(f"{(st == k).mean():.2f}" for k in range(5))
             + f" | {np.quantile(dist, [0.1, 0.5, 0.9]).astype(int).tolist()}"
             + f" | excl-vocab targets removed: within {shi[2]:.4f} cross {shi[3]:.4f} (share kept {nvi / nv:.3f})")


def main():
    tok = Tokenizer.from_file(hf_hub_download(TOK_REPO, "tokenizer.json", revision=TOK_REV))
    excl = np.load(EXCL, allow_pickle=False)
    docs, langs = build_docs(tok)
    tag = " [programming languages only]" if args.prog_only else ""
    L = ["XPO-0: exposure geometry, segment-blocked layout S=2048, barriers (2,4,6); 4-gram repeat classes" + tag,
         "set        docs | non-rep  within  cross | med-cross/doc  >=20 | elig  probes/doc(med) probes(sum) | "
         "strata E1/E2s/E2c/E2b/E3 | probe dist q10/q50/q90"]
    meta = np.load(args.eval_meta, allow_pickle=False)
    pool = np.memmap(args.eval_bin, dtype=np.uint32, mode="r").reshape(-1, T + 1)
    dclm = np.asarray(pool[np.sort(meta["eval_rows"][:N])])
    summarize("DCLM-end", dclm, excl, L)
    for k, rows in docs.items():
        summarize(k, rows, excl, L)
        u, c = np.unique(langs[k], return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:6]
        L.append(f"   {k} languages: " + ", ".join(f"{l} {n}" for n, l in top))
    L.extend(CPT_NOTE)
    L.append("Notes: DCLM-end = the first N P2-1 endpoint docs (all eligible by construction); code sets are "
             "unfiltered (eligibility shown). Probes = first occurrences in chunks 4-7 of 4-grams seen in chunks 0-3.")
    txt = "\n".join(L)
    print(txt)
    os.makedirs(args.out_dir, exist_ok=True)
    open(os.path.join(args.out_dir, ("exposure_geometry_prog.txt" if args.min_chars == 24000 else f"exposure_geometry_prog_min{args.min_chars}.txt") if args.prog_only else "exposure_geometry.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
