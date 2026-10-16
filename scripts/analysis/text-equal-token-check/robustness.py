"""Robustness of the budgeted frontier gap R_B (mini-m, seeds 0-2).

R_B = best M of SO minus best M of FO-CE over each arm's eta curve at adapted loss <= B, with
B = SO's loss floor + b, for b in (0, .005, .01, .02). Reported per seed and b:
- R with piecewise-linear interpolation in eta (the default);
- R with a local quadratic fit in log eta;
- R when each arm may use the better of its CE and inverse-sigma (isig4) read curves;
- a document bootstrap (400 draws, seed 1) that resamples the 128 loss documents and the 1024
  memory documents independently: SD and 5th percentile.

Inputs: per-document reads ``<dir>/<arm>/s<seed>/g0_read_<name>.npz`` with the fields
``L_adapt_sum``, ``L_adapt_count`` and ``M_doc``; ``<name>`` is ``ce``/``etaNNN`` for the CE
curve and ``isig4``/``isig4_etaNNN`` for the inverse-sigma curve. Directories are searched in the
order given.
Output: stdout.
"""
import argparse
import numpy as np, os
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
nm=lambda e:'ce' if e==1 else 'eta%03d'%round(e*100)
CE={'e2e2':[.5,.75,1,1.5,2],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4]}
IS={'e2e2':{1:'isig4',1.5:'isig4_eta150',2:'isig4_eta200'},'e2e1-ce':{1:'isig4',1.5:'isig4_eta150',2.5:'isig4_eta250',3:'isig4_eta300',4:'isig4_eta400'}}
def reads(arm,s,rule):
    names=[(e,nm(e)) for e in CE[arm]] if rule=='ce' else sorted(IS[arm].items())
    return [(e,find(arm,s,n)) for e,n in names]
def curve(R,lw=None,mw=None,kind='lin'):
    e=np.array([x[0] for x in R])
    if lw is None: L=np.array([z['L_adapt_sum'].sum()/z['L_adapt_count'].sum() for _,z in R])
    else: L=np.array([(z['L_adapt_sum']*lw).sum()/(z['L_adapt_count']*lw).sum() for _,z in R])
    M=np.array([z['M_doc'].mean() if mw is None else (z['M_doc']*mw).sum()/mw.sum() for _,z in R])
    g=np.linspace(e.min(),e.max(),2001)
    if kind=='lin': return np.interp(g,e,L),np.interp(g,e,M)
    # monotone-free quadratic-in-log-eta local fit (3-point) as alternative
    from numpy.polynomial import polynomial as P
    lg=np.log(e); Lg=np.empty_like(g); Mg=np.empty_like(g)
    for i,x in enumerate(np.log(g)):
        j=np.argsort(np.abs(lg-x))[:3]; Lg[i]=np.polyval(np.polyfit(lg[j],L[j],min(2,len(j)-1)),x); Mg[i]=np.polyval(np.polyfit(lg[j],M[j],min(2,len(j)-1)),x)
    return Lg,Mg
def best(L,M,B):
    ok=L<=B; return M[ok].max() if ok.any() else np.nan
def RB(so,fo,b,lw=None,mw=None,kind='lin'):
    Ls,Ms=curve(so,lw,mw,kind); Lf,Mf=curve(fo,lw,mw,kind); B=Ls.min()+b
    return best(Ls,Ms,B)-best(Lf,Mf,B)
rng=np.random.default_rng(1)
for s in range(3):
    so,fo=reads('e2e2',s,'ce'),reads('e2e1-ce',s,'ce')
    sol=so+[(e+1e-6,z) for e,z in reads('e2e2',s,'isig4')]  # not merged; separate curves below
    out=[]
    for b in (0,.005,.01,.02):
        base=RB(so,fo,b); quad=RB(so,fo,b,kind='quad')
        # FO allowed best of ce/isig4 curves; SO likewise
        fo_is=reads('e2e1-ce',s,'isig4'); so_is=reads('e2e2',s,'isig4')
        Ls,Ms=curve(so); B=Ls.min()+b
        Lsi,Msi=curve(so_is); Lf,Mf=curve(fo); Lfi,Mfi=curve(fo_is)
        so_best=np.nanmax([best(Ls,Ms,B),best(Lsi,Msi,B)]); fo_best=np.nanmax([best(Lf,Mf,B),best(Lfi,Mfi,B)])
        # doc bootstrap: resample 128 L docs and 1024 M docs independently
        bs=[]
        for _ in range(400):
            lw=np.bincount(rng.integers(0,128,128),minlength=128).astype(float)
            mw=np.bincount(rng.integers(0,1024,1024),minlength=1024).astype(float)
            bs.append(RB(so,fo,b,lw,mw))
        bs=np.array(bs)
        out.append(f"b+{b:.3f}: R {base:+.3f} (quad-logeta {quad:+.3f}; best-of-rules {so_best-fo_best:+.3f}) doc-boot SD {np.nanstd(bs):.3f} 5% {np.nanpercentile(bs,5):+.3f}")
    print(f"s{s}: "+' | '.join(out))
