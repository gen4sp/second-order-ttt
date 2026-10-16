#!/usr/bin/env python
"""Write factorisation: distribution of the per-key factor kappa, from saved per-key probe arrays (numpy, no torch).

1. Exact W2 write factor. For write key i the probe saved
     w_n_ce_T[..., 2]  = (eta_W2 s0 / 64) |e_i^CE| |a_i|,   e_i^CE  = J_i^T (p_i - e_y)
     w_n_lin_T[..., 2] = (eta_W2 s0 / 64) |e_i^lin| |a_i|,  e_i^lin = J_i^T (u - e_y)
   (J_i = d[U LN_f](o_i), the read Jacobian), and w_resid = r_i = 1 - p_{y,i}. Hence the exact per-key factor
     kappa_i = r_i * |delta_W2,lin^(i)| / |delta_W2,CE^(i)| = |J_i^T(u - e_y)| / |J_i^T(q_i - e_y)|.
   A logit-space bound gives kappa_i in [.71, 1] for every key; here: its distribution per unit and group.
2. Logit-space reference interval for V = 50: kappa in [sqrt((V-1)/V)/sqrt(2), (V-1)/V] when J = I.
Groups: R = FO responder, NR = FO non-responder, SO = SO bound, SOu = SO unbound, dev = development unit.
Inputs (per-key probe arrays of the toy checkpoints, not included in the repository): <label>.npz in
PROBE_DIR with ``w_resid`` [sequences, keys], ``w_n_ce_T`` and ``w_n_lin_T`` [sequences, keys, 3].
Output: UNIT and SET lines on stdout (results/theory-checks/write_factorisation_kappa.csv holds the same
numbers, one row per line, groups spelled out).
Usage: python write_factorisation_kappa.py PROBE_DIR > kappa.txt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

V = 50
SETS = {}
# unit label -> group (FO units: responder R / non-responder NR; SO units: bound SO / unbound SOu)
NR = [("e2e1emb", 40, s) for s in (0, 1, 2, 3)] + [("e2e1emb", 160, s) for s in (0, 2)]
R = [("e2e1emb", 10, s) for s in range(6)] + [("e2e1emb", 40, s) for s in (4, 5)] + \
    [("e2e1emb", 160, s) for s in (1, 3, 4)]
SO = [("e2e2emb", e, s) for e in (10, 40) for s in range(6)] + [("e2e2emb", 160, s) for s in (0, 2, 3)]
SOU = [("e2e2emb", 160, s) for s in (1, 4)]
TAG = {"e2e1emb": "FO", "e2e2emb": "SO"}
MANIFEST = [dict(set="dev", label="dev-FO-r42l-s2-st500"), dict(set="dev", label="dev-FO-eta160-s5")]
for st, units in (("NR", NR), ("R", R), ("SO", SO), ("SOu", SOU)):
    for arm, eta, seed in units:
        MANIFEST.append(dict(set=st, label=f"{TAG[arm]}-eta{eta}-s{seed}"))

SET_OF = {e["label"]: e["set"] for e in MANIFEST}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("probe_dir", help="per-key probe arrays (*.npz)")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "write-factorisation-kappa"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    od = args.probe_dir
    lo_ref, hi_ref = np.sqrt((V - 1) / V) / np.sqrt(2), (V - 1) / V
    print(f"LOGIT_SPACE_INTERVAL V={V} [{lo_ref:.4f}, {hi_ref:.4f}]  (logit-space bound [.71, 1])")
    pooled = {}
    for f in sorted(glob.glob(os.path.join(od, "*.npz"))):
        lab = os.path.basename(f)[:-4]
        st = SET_OF.get(lab, "dev")
        z = np.load(f)
        r = z["w_resid"].reshape(-1)
        nce, nlin = z["w_n_ce_T"][..., 2].reshape(-1), z["w_n_lin_T"][..., 2].reshape(-1)
        kap = r * nlin / nce
        ratio = nlin / nce
        out_bound = float(np.mean((kap < .71) | (kap > 1.0)))
        out_ref = float(np.mean((kap < lo_ref) | (kap > hi_ref)))
        q = np.quantile(kap, [0, .05, .5, .95, 1])
        pooled.setdefault(st, []).append(kap)
        # within-sequence residual heterogeneity
        rr = z["w_resid"]
        cv = np.median(rr.std(1) / rr.mean(1))
        print(f"UNIT {lab:22s} {st:4s} kappa min/q05/med/q95/max {q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f} {q[4]:.3f}"
              f" | share outside [.71,1] {out_bound:.3f} | outside logit interval {out_ref:.3f}"
              f" | med lin/CE W2 ratio {np.median(ratio):8.1f} vs 1/med r {1 / np.median(r):8.1f} | resid CV(seq) med {cv:.2f}")
    for st, v in pooled.items():
        k = np.concatenate(v)
        q = np.quantile(k, [0, .05, .5, .95, 1])
        print(f"SET {st:4s} keys {k.size:6d} kappa min/q05/med/q95/max {q[0]:.3f} {q[1]:.3f} {q[2]:.3f} {q[3]:.3f} {q[4]:.3f}"
              f" | share outside [.71,1] {np.mean((k < .71) | (k > 1)):.3f}")


if __name__ == "__main__":
    main()
