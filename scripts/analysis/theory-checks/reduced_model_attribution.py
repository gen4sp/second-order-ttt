#!/usr/bin/env python3
"""Reduced model: registered linear-read attribution test on the SO final checkpoints.

For each SO (e2e2emb, eta 10, step 1500) final checkpoint of seeds 1-9 (set "S"; seed 0 was the
development unit), compares the exact chunk-0 kv write contribution to the query margin
(``i_Mkv_exact``: margin change when the kv write is added with later updates frozen, projected on the
base margin direction, averaged over queries and evaluation sequences) with its central
finite-difference linearisation at the base point (``i_Mkv_lin``). A unit passes if
|Mkv_lin - Mkv_exact| <= .25 Mkv_exact; the count rule is supported if >= 7 of 9 pass, falsified if <= 4.
Input (per-checkpoint probe summary of the toy checkpoints, not included in the repository): a JSON
with a ``checkpoints`` list; each entry has ``label`` (ending in -s<seed>-final), ``set``, ``arm``,
``n_eval``, ``i_Mkv_exact`` and ``i_Mkv_lin`` (point value first).
Writes OUT-DIR/theory-checks/reduced_model_attribution_test.csv.
"""
import argparse
import csv
import io
import json
import os
import re
from pathlib import Path

OUT = {}  # relpath -> text


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return buf.getvalue()



def item1(results_json):
    r = json.loads(Path(results_json).read_text())
    cp = {s["label"]: s for s in r["checkpoints"]}
    labels = sorted((l for l in cp if cp[l]["set"] == "S"), key=lambda l: int(re.search(r"-s(\d+)-", l).group(1)))
    assert len(labels) == 9
    rows = []
    for l in labels:
        s = cp[l]
        assert s["arm"] == "e2e2emb"
        seed = int(re.search(r"-s(\d+)-final$", l).group(1))
        ex, li = s["i_Mkv_exact"][0], s["i_Mkv_lin"][0]
        rel = abs(li - ex) / ex
        passes = bool(abs(li - ex) <= .25 * ex)  # the scoring rule, verbatim
        rows.append([s["arm"], 10, seed, 1500, s["n_eval"], ex, li, rel, 0.25, passes])
    errs = [x[7] for x in rows]
    print("item1 rel errors", min(errs), max(errs), "passes", sum(x[9] for x in rows), "/", len(rows))
    OUT["theory-checks/reduced_model_attribution_test.csv"] = csv_text(
        ["arm", "eta", "seed", "step", "n_eval_sequences", "Mkv_exact", "Mkv_lin", "rel_error", "tolerance", "passes"],
        rows)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_json", help="per-checkpoint probe summary (JSON with a checkpoints list)")
    ap.add_argument("--out-dir", default="outputs/analysis", help="written to OUT-DIR/<results path>")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "reduced-model-attribution"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    item1(args.results_json)
    for rel, text in OUT.items():
        p = Path(args.out_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print("WRITE", p, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
