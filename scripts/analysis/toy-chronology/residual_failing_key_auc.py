#!/usr/bin/env python3
"""Does the write residual predict which queries fail? AUC of the CE residual, post hoc and descriptive.

Units: six FO-CE (e2e1emb) final checkpoints of the fixed-eta runs (eta 40 seeds 0-3, eta 160 seeds
0 and 2). Each unit has 128 evaluation sequences x 16 queries; a query fails when its argmax is wrong.
The residual is the CE residual 1 - p_y at the key's chunk-0 write position. Per unit: number of
failing / correct queries, median residual of each, and AUC = P(residual on a correct query > residual
on a failing query) (ties count 1/2), computed by ranks (Mann-Whitney) and checked against a direct
pairwise count.
Inputs (per-key probe outputs of the toy checkpoints, not included in the repository):
PROBE_DIR/<label>.json with ``ckpt``, ``eta_base``, ``n_eval`` and PROBE_DIR/<label>.npz with
``nat_acc`` (0/1 per query) and ``w_resid`` (per query), labels FO-eta40-s0..s3, FO-eta160-s0, -s2.
Writes OUT-DIR/toy-chronology/residual_failing_key_auc.csv.
"""
import argparse
import csv
import io
import json
import os
import re
from pathlib import Path

import numpy as np

OUT = {}  # relpath -> text


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return buf.getvalue()



def ckpt_meta(path):
    b = os.path.basename(path)
    m = re.match(r"echo__s(\d+)__([a-z0-9_]+?)__eta(\d+)__st1500(?:__step(\d+))?\.pt$", b)
    assert m, b
    return int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4) or 1500)



def auc(pos, neg):
    """P(pos > neg) + .5 P(tie), via ranks (Mann-Whitney). Same as the analysis code."""
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = .5 * (i + j) + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_pairs(pos, neg):
    """Independent check: direct pairwise count."""
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + .5 * eq) / (len(pos) * len(neg))



def item3(probe_dir):
    labs = ["FO-eta40-s0", "FO-eta40-s1", "FO-eta40-s2", "FO-eta40-s3", "FO-eta160-s0", "FO-eta160-s2"]
    rows = []
    for l in labs:
        meta = json.loads((Path(probe_dir) / f"{l}.json").read_text())
        seed, arm, eta, step = ckpt_meta(meta["ckpt"])
        assert arm == "e2e1emb" and eta == meta["eta_base"]
        A = np.load(Path(probe_dir) / f"{l}.npz")
        nat = A["nat_acc"].reshape(-1)
        assert set(np.unique(nat)) <= {0.0, 1.0}
        C0 = nat > .5
        D0 = ~C0
        res = A["w_resid"].reshape(-1)
        a = float(auc(res[C0], res[D0]))
        a2 = float(auc_pairs(res[C0], res[D0]))
        assert abs(a - a2) < 1e-12, (a, a2)
        rows.append([arm, eta, seed, step, meta["n_eval"], int(nat.size), int(D0.sum()), int(C0.sum()),
                     float(np.median(res[D0])), float(np.median(res[C0])), a])
    print("item3 aucs", [(r[1], r[2], round(r[10], 4)) for r in rows], "range", min(r[10] for r in rows), max(r[10] for r in rows))
    print("item3 nD0/medians", [(r[6], f"{r[8]:.3g}", f"{r[9]:.3g}") for r in rows])
    OUT["toy-chronology/residual_failing_key_auc.csv"] = csv_text(
        ["arm", "eta", "seed", "step", "n_eval_sequences", "n_queries", "n_failing", "n_correct",
         "median_residual_failing", "median_residual_correct", "auc_residual_correct_above_failing"], rows)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_dir", help="per-key probe outputs (<label>.json and <label>.npz)")
    ap.add_argument("--out-dir", default="outputs/analysis", help="written to OUT-DIR/<results path>")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "residual-failing-key-auc"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    item3(args.probe_dir)
    for rel, text in OUT.items():
        p = Path(args.out_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print("WRITE", p, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
