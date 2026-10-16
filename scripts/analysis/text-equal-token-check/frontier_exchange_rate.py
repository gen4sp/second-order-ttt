"""Equal-token frontier of SO and FO-CE per model and seed: loss floors, exchange rate, matched-loss M.

For each preset (mini-s, mini-m) and seed 0-2:
- the adapted-loss floor of each arm over its eta grid and the floor gap, with the document-level
  SE of the gap at eta x1 (128 held-out documents);
- every eta point (L, M, fP_nonfocal, fP_focal);
- the exchange rate lam*: the smallest lam (nats of L per nat of M) at which
  min_eta (L - lam M) of SO falls below that of FO-CE, on piecewise-linear curves in eta, and the
  arg-min eta of each arm there;
- at each SO point inside FO-CE's loss range: FO-CE's best M and best (lowest) fP_nonfocal at
  adapted loss <= SO's.

Inputs: per-document reads ``<dir>/<arm>/s<seed>/g0_read_<name>.npz`` with the fields
``L_adapt_sum``, ``L_adapt_count``, ``M_doc``, ``fP_focal`` and ``fP_nonfocal``; ``<name>`` is
``ce`` for eta x1 and ``etaNNN`` (100 x multiplier) otherwise. mini-m directories are searched in
the order given; mini-s has one directory.
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
GRID={'mini-m':{'e2e2':[.5,.75,1,1.5,2],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4]},
      'mini-s':{'e2e2':[1,1.5],'e2e1-ce':[1,1.25,1.5,2,2.5,3,4,5,6]}}
nm=lambda e:'ce' if e==1 else 'eta%03d'%round(e*100)
def load(preset,arm,s,e):
    if preset=='mini-s': return np.load(f'{FS}/{arm}/s{s}/g0_read_{nm(e)}.npz',allow_pickle=True)
    for d in MM:
        p=f'{d}/{arm}/s{s}/g0_read_{nm(e)}.npz'
        if os.path.exists(p): return np.load(p,allow_pickle=True)
def pts(preset,arm,s):
    out=[]
    for e in GRID[preset][arm]:
        z=load(preset,arm,s,e)
        out.append(dict(e=e,L=z['L_adapt_sum'].sum()/z['L_adapt_count'].sum(),Ls=z['L_adapt_sum'],Lc=z['L_adapt_count'],
                        M=z['M_doc'].mean(),Md=z['M_doc'],ff=np.nanmean(z['fP_focal']),fn=np.nanmean(z['fP_nonfocal']),fnd=z['fP_nonfocal'],ffd=z['fP_focal']))
    return out
for preset in ('mini-s','mini-m'):
    print(f'================ {preset}')
    lam_all=[]
    for s in range(3):
        so,fo=pts(preset,'e2e2',s),pts(preset,'e2e1-ce',s)
        Lso_min=min(p['L'] for p in so); Lfo_min=min(p['L'] for p in fo)
        # paired SE of L floor gap (SO x1 vs FO x1), doc level on 128 ho docs
        d=so[[p['e'] for p in so].index(1)]['Ls']/so[[p['e'] for p in so].index(1)]['Lc']-fo[0]['Ls']/fo[0]['Lc']
        print(f's{s}: L floor SO {Lso_min:.4f} FO {Lfo_min:.4f} gap {Lso_min-Lfo_min:+.4f} (doc SE of x1 gap {d.std(ddof=1)/np.sqrt(len(d)):.4f})')
        print('   SO:', ' '.join(f"x{p['e']}:(L{p['L']:.4f},M{p['M']:.3f},fn{p['fn']:.3f},ff{p['ff']:.3f})" for p in so))
        print('   FO:', ' '.join(f"x{p['e']}:(L{p['L']:.4f},M{p['M']:.3f},fn{p['fn']:.3f},ff{p['ff']:.3f})" for p in fo))
        # exchange rate: J(lam)=min_eta L - lam*M on dense piecewise-linear curves; lam* where J_SO=J_FO
        def dense(P):
            e=np.array([p['e'] for p in P]); g=np.linspace(e.min(),e.max(),4001)
            return np.interp(g,e,[p['L'] for p in P]),np.interp(g,e,[p['M'] for p in P]),np.interp(g,e,[p['fn'] for p in P])
        Ls,Ms,fns=dense(so); Lf,Mf,fnf=dense(fo)
        lams=np.linspace(0,2,20001); Js=np.array([np.min(Ls-l*Ms) for l in lams]); Jf=np.array([np.min(Lf-l*Mf) for l in lams])
        i=np.flatnonzero(Js<Jf); lam=lams[i[0]] if len(i) else np.nan; lam_all.append(lam)
        # beware grid edge: at lam*, where are the argmins?
        if len(i):
            l=lam; es=np.linspace(so[0]['e'],so[-1]['e'],4001)[np.argmin(Ls-l*Ms)]; ef=np.linspace(fo[0]['e'],fo[-1]['e'],4001)[np.argmin(Lf-l*Mf)]
        else: es=ef=np.nan
        print(f'   exchange rate lam* (nats L per nat M at which SO becomes preferred): {lam:.3f}  argmin eta at lam*: SO x{es:.2f}, FO x{ef:.2f}')
        # absolute probe NLL with memory (fP_nonfocal) at matched L: SO at its points vs FO interpolated at same L
        for p in so:
            if p['L']<Lfo_min or p['L']>max(q['L'] for q in fo): continue
            # FO best (lowest) fn and best M at L<=p['L']
            ok=Lf<=p['L']
            print(f"   SO x{p['e']} L{p['L']:.4f}: M {p['M']:.3f} vs FO best M {Mf[ok].max():.3f} (R {p['M']-Mf[ok].max():+.3f}); abs probe NLL fn {p['fn']:.3f} vs FO best fn {fnf[ok].min():.3f} (diff {fnf[ok].min()-p['fn']:+.3f})")
    print(f'  lam* by seed {np.round(lam_all,3)}')
