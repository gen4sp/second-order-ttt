# /// script
# dependencies = ["numpy", "matplotlib", "setproctitle"]
# ///
"""Publication-size recovery curves from persisted analyzer inputs only."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import setproctitle
from figure_style import use_style,figure_grid,panel_labels,save,PALETTE,BASELINE


def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path)
    p.add_argument('--process-name',default='TTM-rdo-78');args=p.parse_args()
    setproctitle.setproctitle(args.process_name)
    units=json.loads((args.analysis/'units.json').read_text())['rdo']
    scores=json.loads((args.analysis/'rdo_scores.json').read_text())
    H=scores['provenance']['effective_config']['horizon']
    use_style();fig,axes=figure_grid(2,2,ratio=.8)
    a,b,c,d=axes.ravel();colors=[PALETTE[k] for k in ('blue','orange','green','purple')]
    counts={};rng=np.random.default_rng(0)
    def km(ax,branch,rule,color,style,label):
        events=[u['events'].get(branch,{}).get(rule,{}) for u in units.values()]
        events=[e for e in events if e.get('disposition') in ('COMPLETE','DIVERGED')]
        if not events:return
        n=len(events);counts[f'{branch}/{rule}']=n
        times=np.array([e['tau'] if e['reacquired'] else H+1 for e in events])
        grid=np.arange(0,H+1,5);unrestored=times[:,None]>grid
        means=unrestored.mean(0)
        ax.step(grid,means,where='post',color=color,linestyle=style,label=label)
        if n>1:
            boot=unrestored[rng.integers(0,n,(10000,n))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.fill_between(grid,lo,hi,color=color,alpha=.07,step='post',linewidth=0)
    for dose,col in zip(('01','03','10','30'),colors):
        km(a,'rpf'+dose,'rpf'+dose,col,'-',f'.{dose}')
        km(b,'ncf'+dose,'ncf'+dose,col,'--',f'.{dose}')
        km(b,'ce','rpf'+dose,col,':',None)
    km(a,'ce','ce',BASELINE,'-', 'CE')
    for ax in (a,b):
        ax.set_ylim(-.02,1.02);ax.set_xlabel('Updates since S1 (steps)');ax.set_ylabel('Not re-acquired (fraction)')
    a.legend(title='Floor',frameon=False,ncol=2,fontsize=6)
    b.legend(title='ncf / twin common eval',frameon=False,ncol=2,fontsize=6)
    rows=list(csv.DictReader((args.analysis/'rdo_series.csv').open()))
    for dose,col in zip(('01','03','10','30'),colors):
        for ax,key in ((c,'r_kv_med'),(d,'r_eff_med')):
            bystep=defaultdict(list)
            for row in rows:
                if row['branch']=='rpf'+dose and row.get(key) not in (None,'','None'):
                    bystep[int(row['step'])-int(row['start'])].append(float(row[key]))
            if bystep:
                x=np.array(sorted(bystep));y=np.array([np.median(bystep[t]) for t in x])
                ax.plot(x,y,color=col)
        for ax in (c,d):ax.set_xlabel('Updates since S1 (steps)');ax.set_yscale('log')
    c.set_ylabel('Raw residual (median)');d.set_ylabel('Effective coefficient (median)')
    panel_labels(axes)
    suffix=' Single legacy seed; smoke only.' if scores['smoke_only'] else ''
    caption=('Recovery over the registered horizon. Solid lines: rpf; dashed: ncf; dotted: CE twin under each floor; grey: CE. '
             'Shading: paired-unit bootstrap 95% pointwise intervals. Certified divergence remains a non-recovery outcome; '
             'missing jobs are excluded and n is listed below. Residual panels show medians, no smoothing.'+suffix)
    (args.analysis/'rdo_curves_caption.txt').write_text(caption+'\n'+json.dumps(counts,sort_keys=True)+'\n')
    save(fig,str(args.analysis/'rdo_curves'),formats=('pdf','svg','png'))


if __name__=='__main__':main()
