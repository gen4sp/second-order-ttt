"""Symmetric budgeted frontier with a CAL/TEST split of the loss documents (mini-m and mini-s).

M*_a(B) = max M over arm a's eta curve (piecewise-linear in eta) with L_CAL <= B, where
B = min over SO's grid of L_CAL + b. R_B = M*_SO - M*_FO-CE. L is measured on 128 held-out
documents split into CAL/TEST halves; M on 1024 evaluation documents. Per seed and b: R on the
fixed even/odd split (with the chosen etas and the TEST loss difference at those etas) and the
median and 5-95% range over 300 random splits (seed 0). Over seeds: mean R, seed SD and the
one-sided 90% t bounds (3 seeds) on the fixed split.

Inputs: per-document reads ``<dir>/<arm>/s<seed>/g0_read_<name>.npz`` with the fields
``L_adapt_sum``, ``L_adapt_count`` and ``M_doc``; ``<name>`` is ``ce`` for eta x1 and ``etaNNN``
(100 x multiplier) otherwise. For mini-m, each eta is taken from the first directory in
``--mini-m-reads`` that holds it for seed 0.
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
def mm(arm,eta):
    name='ce' if eta==1 else 'eta%03d'%round(eta*100)
    for d in MM:
        p=f'{d}/{arm}/s{{s}}/g0_read_{name}.npz'
        if os.path.exists(p.format(s=0)): return p
    raise FileNotFoundError(name)
GRID={'mini-m':{'e2e2':[.5,.75,1,1.5,2],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4]},
      'mini-s':{'e2e2':[1,1.5],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4,5,6]}}
def path(preset,arm,eta):
    if preset=='mini-m': return mm(arm,eta)
    return f'{FS}/{arm}/s{{s}}/g0_read_'+('ce' if eta==1 else 'eta%03d'%round(eta*100))+'.npz'
def load(preset,arm,s):
    out=[]
    for e in GRID[preset][arm]:
        d=np.load(path(preset,arm,e).format(s=s),allow_pickle=True)
        out.append((e,d['L_adapt_sum'].astype(float),d['L_adapt_count'].astype(float),float(d['M_doc'].mean())))
    return out
def curve(pts,idx):
    e=np.array([p[0] for p in pts]); M=np.array([p[3] for p in pts])
    L=np.array([p[1][idx].sum()/p[2][idx].sum() for p in pts])
    g=np.linspace(e.min(),e.max(),2001); return g,np.interp(g,e,M),np.interp(g,e,L)
def best(pts,cal,test,B):
    g,M,Lc=curve(pts,cal); _,_,Lt=curve(pts,test); ok=Lc<=B
    if not ok.any(): return np.nan,np.nan,np.nan
    i=np.flatnonzero(ok)[np.argmax(M[ok])]; return M[i],g[i],Lt[i]
BUD=(0,.005,.01,.02); rng=np.random.default_rng(0)
for preset in ('mini-m','mini-s'):
    fixed=[]
    for s in range(3):
        so,fo=load(preset,'e2e2',s),load(preset,'e2e1-ce',s)
        splits=[(np.arange(0,128,2),np.arange(1,128,2))]+[tuple(np.split(rng.permutation(128),2)) for _ in range(300)]
        res=[]
        for cal,test in splits:
            _,_,Lso=curve(so,cal); Lmin=Lso.min(); row=[]
            for b in BUD:
                ms,es,lts=best(so,cal,test,Lmin+b); mf,ef,ltf=best(fo,cal,test,Lmin+b)
                row.append((ms-mf,es,ef,lts-ltf))
            res.append(row)
        res=np.array(res); fixed.append(res[0,:,0])
        print(f'{preset} s{s}:', ' | '.join(f'b+{b:.3f}: R {res[0,j,0]:+.3f} (eta SO {res[0,j,1]:.2f}, FO {res[0,j,2]:.2f}, TEST dL {res[0,j,3]:+.4f}); 300 splits med {np.nanmedian(res[1:,j,0]):+.3f} [{np.nanpercentile(res[1:,j,0],5):+.3f},{np.nanpercentile(res[1:,j,0],95):+.3f}]' for j,b in enumerate(BUD)))
    f=np.array(fixed)
    for j,b in enumerate(BUD):
        m,sd=f[:,j].mean(),f[:,j].std(ddof=1); print(f'  {preset} fixed split b+{b}: mean R {m:+.3f}, seed SD {sd:.3f}, one-sided 90% LB {m-1.886*sd/np.sqrt(3):+.3f}, UB {m+1.886*sd/np.sqrt(3):+.3f}')
