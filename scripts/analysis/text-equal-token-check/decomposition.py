"""Decompose M = fP_focal - fP_nonfocal per arm and eta multiplier (mini-m, seeds 0-2).

Inputs: per-document reads ``<dir>/<arm>/s<seed>/g0_read_<name>.npz`` with the fields
``L_adapt_sum``, ``L_adapt_count``, ``fP_focal``, ``fP_nonfocal`` and ``M_doc``; ``<name>`` is
``ce`` for eta x1 and ``etaNNN`` (100 x multiplier) otherwise. The directories given to
``--mini-m-reads`` are searched in order and the first match is used.
Output: one line per arm and eta on stdout.
"""
import argparse
import numpy as np, glob, os
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--mini-m-reads", nargs="+",
                default=["outputs/mini/matrix/g0/mini-m", "outputs/mini/matrix/g0b/mini-m",
                         "outputs/mini/matrix/g0c/mini-m"],
                help="mini-m read directories, searched in order")
args = ap.parse_args()
MM = args.mini_m_reads
def find(arm,s,name):
    for d in MM:
        p=f'{d}/{arm}/s{s}/g0_read_{name}.npz'
        if os.path.exists(p): return np.load(p,allow_pickle=True)
ET={'e2e2':[.5,.75,1,1.5,2],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4]}
nm=lambda e:'ce' if e==1 else 'eta%03d'%round(e*100)
for s in range(3):
  print(f'--- seed {s}')
  for arm in ET:
    for e in ET[arm]:
      z=find(arm,s,nm(e)); L=z['L_adapt_sum'].sum()/z['L_adapt_count'].sum()
      ok=np.isfinite(z['fP_focal'])&np.isfinite(z['fP_nonfocal'])
      print(f"{arm:8s} x{e:<4} L {L:.4f}  fP_focal {z['fP_focal'][ok].mean():.4f}  fP_nonfocal {z['fP_nonfocal'][ok].mean():.4f}  M {z['M_doc'].mean():.4f}  n {ok.sum()}")
