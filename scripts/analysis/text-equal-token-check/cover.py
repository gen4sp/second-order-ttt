"""Cover quantities of the budgeted frontier (mini-s and mini-m, seeds 0-2).

Per seed, with B_SO = SO's adapted-loss floor + .01 and piecewise-linear curves in eta:
- R_B(.01): best M of SO minus best M of FO-CE, both at adapted loss <= B_SO;
- cover: the same, with FO-CE allowed an extra .01 of loss;
- SO's best M at B_SO minus FO-CE's largest M anywhere on its eta grid;
- own floor: SO's best M at its floor + .01 minus FO-CE's best M at its own floor + .01;
- the adapted-loss floor gap SO - FO-CE.

Inputs: per-document reads ``<dir>/<arm>/s<seed>/g0_read_<name>.npz`` with the fields
``L_adapt_sum``, ``L_adapt_count`` and ``M_doc``; ``<name>`` is ``ce`` for eta x1 and ``etaNNN``
(100 x multiplier) otherwise. mini-m directories are searched in the order given.
Output: stdout.
"""
import argparse
import numpy as np, os
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--mini-m-reads", nargs="+",
                default=["outputs/mini/matrix/g0/mini-m", "outputs/mini/matrix/g0b/mini-m",
                         "outputs/mini/matrix/g0c/mini-m"],
                help="mini-m read directories, searched in order")
ap.add_argument("--mini-s-reads", default="outputs/soq/fs/mini-s", help="mini-s read directory")
args = ap.parse_args()
MM = args.mini_m_reads
FS = args.mini_s_reads
GRID={'mini-m':{'e2e2':[.5,.75,1,1.5,2],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4]},'mini-s':{'e2e2':[1,1.5],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4,5,6]}}
nm=lambda e:'ce' if e==1 else 'eta%03d'%round(e*100)
def load(p,a,s,e):
    if p=='mini-s': return np.load(f'{FS}/{a}/s{s}/g0_read_{nm(e)}.npz')
    for d in MM:
        f=f'{d}/{a}/s{s}/g0_read_{nm(e)}.npz'
        if os.path.exists(f): return np.load(f)
def dense(p,a,s):
    e=np.array(GRID[p][a]); Z=[load(p,a,s,x) for x in e]
    L=[z['L_adapt_sum'].sum()/z['L_adapt_count'].sum() for z in Z]; M=[z['M_doc'].mean() for z in Z]
    g=np.linspace(e.min(),e.max(),4001); return np.interp(g,e,L),np.interp(g,e,M),max(M)
best=lambda L,M,B:(M[L<=B].max() if (L<=B).any() else np.nan)
for p in ('mini-s','mini-m'):
    rows=[]
    for s in range(3):
        Ls,Ms,_=dense(p,'e2e2',s); Lf,Mf,ceil=dense(p,'e2e1-ce',s)
        Bs=Ls.min()+.01; Bf=Lf.min()+.01
        rows.append((best(Ls,Ms,Bs)-best(Lf,Mf,Bs), best(Ls,Ms,Bs)-best(Lf,Mf,Bs+.01), best(Ls,Ms,Bs)-ceil, best(Ls,Ms,Bs)-best(Lf,Mf,Bf), Ls.min()-Lf.min()))
    r=np.array(rows)
    for j,lab in enumerate(['R_B(.01) same budget','cover: FO gets +.01 L','SO(floor+.01) - FO max M over whole grid','own-floor: M_SO(floor+.01)-M_FO(floor+.01)','adapted L floor gap SO-FO']):
        print(f'{p} {lab:45s} per seed {np.round(r[:,j],3)} mean {r[:,j].mean():+.3f}')
