"""Per-document breadth of the budgeted frontier gap R_B at b = .01 (mini-m and mini-s, seeds 0-2).

Each arm picks the eta with the largest mean M at adapted loss <= B (B = SO's loss floor + .01,
piecewise-linear curves in eta); the per-document M of that eta is interpolated between the two
neighbouring grid points, and d = M_SO - M_FO-CE per memory document. Reported: the chosen etas,
mean, median, 10% trimmed mean, P(d > 0) and the 10/25/75/90 percentiles of d.

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
for p in ('mini-m','mini-s'):
  for s in range(3):
    C={}
    for a in ('e2e2','e2e1-ce'):
        e=np.array(GRID[p][a]); Z=[load(p,a,s,x) for x in e]
        L=np.array([z['L_adapt_sum'].sum()/z['L_adapt_count'].sum() for z in Z]); Md=np.stack([z['M_doc'] for z in Z])
        C[a]=(e,L,Md)
    e,L,_=C['e2e2']; g=np.linspace(e.min(),e.max(),4001); B=np.interp(g,e,L).min()+.01
    pick={}
    for a,(e,L,Md) in C.items():
        g=np.linspace(e.min(),e.max(),4001); Lg=np.interp(g,e,L); Mg=np.interp(g,e,Md.mean(1))
        ok=Lg<=B; eta=g[ok][np.argmax(Mg[ok])]
        j=np.searchsorted(e,eta); j=min(max(j,1),len(e)-1); w=(eta-e[j-1])/(e[j]-e[j-1])
        pick[a]=(eta,(1-w)*Md[j-1]+w*Md[j])
    d=pick['e2e2'][1]-pick['e2e1-ce'][1]
    q=np.percentile(d,[10,25,50,75,90])
    tm=np.sort(d)[51:-51].mean()
    print(f"{p} s{s}: eta SO {pick['e2e2'][0]:.2f} FO {pick['e2e1-ce'][0]:.2f} | mean {d.mean():+.3f} median {q[2]:+.3f} 10% trimmed {tm:+.3f} | P(d>0) {np.mean(d>0):.2f} | deciles10/25/75/90 {q[0]:+.3f} {q[1]:+.3f} {q[3]:+.3f} {q[4]:+.3f}")
