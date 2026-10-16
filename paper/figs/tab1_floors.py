"""Table 1 helper: adapted CAL floors and no-adapt H_LM of every arm in both waves (mini-m).

Inputs (repository-relative):
  results/text-original-study/analysis.json      original study, seeds 0-2: per-seed `floor_cal`
                                                 and `H_LM` per arm
  results/text-replication/floors_hlm_per_seed.csv
                                                 replication, seeds 3-5: per arm and seed the
                                                 adapted CAL floor (eta-grid minimum), its eta
                                                 multiplier and H_LM
  results/text-replication/analysis.json         replication analysis, used as a cross-check of
                                                 the SO floor (L_SO_min) and of H_LM

The CAL floor is the token-pooled adapted loss on the even-indexed (CAL) half of the 128
held-out documents, minimised over the arm's eta grid. The script checks the replication CSV
against the analysis JSON, prints the per-seed values and the seed means, and asserts that the
means round to the Floor / H_LM column of Table 1. It also prints SO-2/16's share of FO-CM's
no-adapt LM gain over SO, (H_SO - H_P) / (H_SO - H_CM), per seed.
Run from the repository root:
  uv run --isolated --no-project --with numpy python paper/figs/tab1_floors.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, os.pardir, os.pardir, "results")
ORIG = os.path.join(RESULTS, "text-original-study", "analysis.json")
REPL_CSV = os.path.join(RESULTS, "text-replication", "floors_hlm_per_seed.csv")
REPL_JSON = os.path.join(RESULTS, "text-replication", "analysis.json")

# Table 1, Floor / H_LM column (nats/token, two decimals), used as a check only.
TABLE1 = {
    "orig": {"so": (4.31, 4.47), "fo_ce": (4.27, 4.37), "fo_cm": (3.88, 3.94)},
    "repl": {"so": (4.31, 4.46), "fo-ce": (4.26, 4.36), "wsd-1200": (4.19, 4.28),
             "fo-cm": (3.88, 3.94), "wsd-3469": (3.91, 3.96), "mini-p": (3.95, 4.02)},
}
REPL_ARMS = ("so", "fo-ce", "wsd-1200", "fo-cm", "wsd-3469", "mini-p")
SEEDS = (3, 4, 5)


def original():
    """{arm: ([floor per seed], [H_LM per seed])} for seeds 0-2."""
    d = json.load(open(ORIG))
    return {a: ([s["floor_cal"][a] for s in d["seeds"]], [s["H_LM"][a] for s in d["seeds"]])
            for a in TABLE1["orig"]}


def replication():
    """{arm: ([floor per seed], [eta per seed], [H_LM per seed])} for seeds 3-5, from the CSV."""
    rows = list(csv.DictReader(open(REPL_CSV)))
    out = {}
    for a in REPL_ARMS:
        sel = {int(r["seed"]): r for r in rows if r["arm"] == a}
        out[a] = ([float(sel[s]["cal_floor"]) for s in SEEDS],
                  [float(sel[s]["cal_floor_eta_multiplier"]) for s in SEEDS],
                  [float(sel[s]["H_LM"]) for s in SEEDS])
    return out


def check_replication(rep):
    """The CSV rounds to 4 decimals; compare it with the full-precision analysis JSON."""
    cf = json.load(open(REPL_JSON))["versions"]["all"]["crossfit"]
    ok = True
    l_so_min = cf["aggregate"]["L_SO_min"]["per_seed"]
    for i, s in enumerate(SEEDS):
        if abs(rep["so"][0][i] - l_so_min[i]) > 5e-5:
            ok = False
            print(f"CHECK FAIL so floor s{s}: {rep['so'][0][i]:.4f} vs L_SO_min {l_so_min[i]:.6f}")
        for a in REPL_ARMS:
            h = cf["per_seed"][i]["H_LM"][a]["value"]
            if abs(rep[a][2][i] - h) > 5e-5:
                ok = False
                print(f"CHECK FAIL {a} H_LM s{s}: {rep[a][2][i]:.4f} vs {h:.6f}")
    return ok


def main():
    orig, rep = original(), replication()
    ok = check_replication(rep)
    print(f"{'wave':5s} {'arm':9s} {'CAL floor per seed':40s} mean   | {'H_LM per seed':24s} mean")
    for a, (fl, hl) in orig.items():
        fm, hm = np.mean(fl), np.mean(hl)
        ok &= (round(fm, 2), round(hm, 2)) == TABLE1["orig"][a]
        print(f"orig  {a:9s} {' / '.join(f'{v:.4f}' for v in fl):40s} {fm:.4f} | "
              f"{' / '.join(f'{v:.4f}' for v in hl):24s} {hm:.4f}")
    for a, (fl, eta, hl) in rep.items():
        fm, hm = np.mean(fl), np.mean(hl)
        ok &= (round(fm, 2), round(hm, 2)) == TABLE1["repl"][a]
        cells = ", ".join(f"{v:.4f} (x{e:g})" for v, e in zip(fl, eta))
        print(f"repl  {a:9s} {cells:40s} {fm:.4f} | "
              f"{' / '.join(f'{v:.4f}' for v in hl):24s} {hm:.4f}")
    h_so, h_p, h_cm = rep["so"][2], rep["mini-p"][2], rep["fo-cm"][2]
    share = [(h_so[i] - h_p[i]) / (h_so[i] - h_cm[i]) for i in range(3)]
    print("SO-2/16 share of FO-CM's no-adapt LM gain over SO:", " / ".join(f"{v:.3f}" for v in share))
    gap = [rep["so"][0][i] - rep["fo-cm"][0][i] for i in range(3)]
    print("replication SO - FO-CM CAL floor gap:", " / ".join(f"{v:.3f}" for v in gap),
          f"mean {np.mean(gap):.3f}")
    print("checks against the analysis JSON and Table 1:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
