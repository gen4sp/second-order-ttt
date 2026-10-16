"""RDO decisions on endpoints produced exclusively by the shared episodes module."""
from __future__ import annotations

import math
import numpy as np

from .rdo_statistics import (paired, cp, page, worst_case_page, dose_shape, bootstrap_rmst,
                             holm)

DOSES=('01','03','10','30')


def observed(e):
    return e is not None and e.get('disposition') in ('COMPLETE','DIVERGED')


def value(e,key,default):
    return e.get(key,default) if observed(e) and e.get(key) is not None else default


def binary(e,key,default,*,control=False,adverse=True):
    if control and observed(e) and e['disposition']=='DIVERGED':
        return adverse
    return bool(value(e,key,default))


def component(t,c,baseline,kind,manipulation='OK',alpha=.01):
    p=paired(t,c); n=len(t); kt=sum(t); kc=sum(c)
    if n<8: verdict='NOT EVALUABLE'
    elif kind!='R1' and manipulation=='FAILED': verdict='MANIPULATION FAILED'
    elif sum(baseline)>=math.ceil(.3*n): verdict='INCONCLUSIVE (unstable baseline)'
    elif kind=='R3' and kt<math.ceil(.8*n): verdict='INCONCLUSIVE (not applicable)'
    elif kt>=math.ceil(.8*n) and kc<=math.floor(.2*n) and p['p']<=alpha: verdict='SUPPORTED'
    elif kind=='R1' and kt<=math.floor(.5*n): verdict='FALSIFIED'
    elif kind!='R1' and (kc>=math.ceil(.5*n) or kc>=kt): verdict='FALSIFIED'
    else: verdict='INCONCLUSIVE'
    return dict(verdict=verdict,manipulation=manipulation,**p)


def selectivity(t,tw,nc,immediate,manipulation='OK'):
    n=len(t)
    if manipulation=='FAILED': return 'unknown (manipulation failed)'
    if sum(tw)>=math.ceil(.5*n) or sum(immediate)>=math.ceil(.5*n): return 'eval-carried'
    if sum(a and not b and not c for a,b,c in zip(t,tw,nc))>=math.ceil(.8*n): return 'training-selective'
    return 'mixed'


def score(units,H=400):
    """Units are the frozen manifest cohort, including all missing jobs."""
    def e(u,branch,rule): return u.get('events',{}).get(branch,{}).get(rule)
    out=dict(frozen_n=len(units),doses={})
    for d in DOSES:
        rp,nc='rpf'+d,'ncf'+d
        needed=lambda u: all(observed(x) for x in (e(u,'ce','ce'),e(u,rp,rp),e(u,'ce',rp),e(u,nc,rp)))
        cohort=[u for u in units if needed(u)]
        t=[bool(e(u,rp,rp)['reacquired']) for u in cohort]
        own=[bool(e(u,'ce','ce')['reacquired']) for u in cohort]
        tw=[bool(e(u,'ce',rp)['reacquired']) for u in cohort]
        ctl=[bool(e(u,nc,rp)['reacquired']) for u in cohort]
        failures=[bool(u.get('manipulation',{}).get(d,True)) for u in cohort]
        n=len(cohort); nf=sum(failures)
        manipulation='FAILED' if n and nf>=math.ceil(.5*n) else 'LIMITED' if nf>=2 else 'OK'
        tests={name:component(t,c,own,name,manipulation) for name,c in (('R1',own),('R1-T',tw),('R3',ctl))}
        def complete(adverse):
            tt=[binary(e(u,rp,rp),'reacquired',not adverse) for u in units]
            base=[binary(e(u,'ce','ce'),'reacquired',adverse,control=True,adverse=adverse) for u in units]
            cc=[binary(e(u,'ce',rp),'reacquired',adverse,control=True,adverse=adverse) for u in units]
            nn=[binary(e(u,nc,rp),'reacquired',adverse,control=True,adverse=adverse) for u in units]
            return {name:component(tt,c,base,name,manipulation) for name,c in (('R1',base),('R1-T',cc),('R3',nn))}
        adverse,reverse=complete(True),complete(False)
        for name,rec in tests.items():
            rec['adverse']=adverse[name]; rec['reverse']=reverse[name]
            v=rec['verdict']
            if v=='SUPPORTED' and adverse[name]['verdict']!='SUPPORTED' or v=='FALSIFIED' and reverse[name]['verdict']!='FALSIFIED':
                rec['verdict']='INCONCLUSIVE (missing-sensitive)'
        immediate=[bool(e(u,rp,rp).get('eval_immediate',False)) for u in cohort]
        labels=[]
        for adverse_flag in (True,False):
            labels.append(selectivity(
                [binary(e(u,rp,rp),'reacquired',not adverse_flag) for u in units],
                [binary(e(u,'ce',rp),'reacquired',adverse_flag,control=True,adverse=adverse_flag) for u in units],
                [binary(e(u,nc,rp),'reacquired',adverse_flag,control=True,adverse=adverse_flag) for u in units],
                [bool(value(e(u,rp,rp),'eval_immediate',False)) for u in units],manipulation))
        label=labels[0] if labels[0]==labels[1] else 'unknown'
        rt=[bool(e(u,rp,rp)['retention']) for u in cohort]
        rc=[bool(e(u,'ce',rp)['retention']) for u in cohort]
        durable=n>=8 and sum(rt)>=math.ceil(.8*n) and sum(rc)<=math.floor(.2*n)
        durable &= sum(binary(e(u,rp,rp),'retention',False) for u in units)>=math.ceil(.8*len(units))
        durable &= sum(binary(e(u,'ce',rp),'retention',True,control=True) for u in units)<=math.floor(.2*len(units))
        out['doses'][d]=dict(n=n,seeds=[u['seed'] for u in cohort],tests=tests,manipulation=manipulation,
                              selectivity=label,durable=bool(durable),retention=dict(treated=sum(rt),control=sum(rc),
                              treated_ci=cp(sum(rt),n),control_ci=cp(sum(rc),n)),immediate=sum(immediate))
    h=out['doses']['03']
    out['headline']=dict(n=h['n'],supported=all(v['verdict']=='SUPPORTED' for v in h['tests'].values()),
                         durability='DURABLE' if h['durable'] else 'durability not shown')
    # Secondary specificity family excludes the registered .03 primary.
    secondary={d:out['doses'][d]['tests']['R3']['p'] for d in ('01','10','30')}
    passed=holm(secondary,.01)
    for d,ok in passed.items():
        r=out['doses'][d]['tests']['R3']; r['holm_pass']=ok
        if r['verdict']=='SUPPORTED' and not ok: r['verdict']='INCONCLUSIVE (Holm)'
    from .episodes import kaplan_meier
    out['recovery_KM']={}
    for d in DOSES:
        values=[e(u,'rpf'+d,'rpf'+d) for u in units if observed(e(u,'rpf'+d,'rpf'+d))]
        times=[v.get('tau',H) for v in values]
        recovered=[bool(v.get('reacquired')) for v in values]
        curve=kaplan_meier(times,recovered)
        median=next((r['time'] for r in curve if r['survival']<=.5),None)
        out['recovery_KM'][d]=dict(n=len(values),median=median,
            median_status='median not reached' if median is None else 'reached',curve=curve,descriptive=True)
    complete_units=[u for u in units if all(observed(e(u,'rpf'+d,'rpf'+d)) for d in DOSES)]
    taus=[[e(u,'rpf'+d,'rpf'+d)['tau'] for d in DOSES] for u in complete_units]
    frozen_taus=[[value(e(u,'rpf'+d,'rpf'+d),'tau',None) for d in DOSES] for u in units]
    n=len(taus)
    if n<8:
        out['R2']=dict(n=n,verdict='NOT EVALUABLE')
    else:
        p=page(taus); bound=worst_case_page(frozen_taus,H); rmst=bootstrap_rmst(taus)
        counts=[sum(observed(e(u,'rpf'+d,'rpf'+d)) and e(u,'rpf'+d,'rpf'+d)['reacquired'] for u in complete_units) for d in DOSES]
        weak=sum(out['doses'][d]['manipulation']!='OK' for d in DOSES)
        unstable=sum(e(u,'ce','ce')['reacquired'] for u in complete_units if observed(e(u,'ce','ce')))>=math.ceil(.3*n)
        if weak>=2 or unstable: verdict='INCONCLUSIVE'
        elif p['p']<=.01 and counts[-1]>=math.ceil(.8*n):
            verdict='GRADED TREND' if bound['p']<=.01 else 'INCONCLUSIVE (missing-sensitive)'
        elif counts[-1]<=math.floor(.5*n): verdict='NO RESCUE'
        elif min(counts)>=math.ceil(.8*n) and p['p']>.10: verdict='PLATEAU'
        else: verdict='INCONCLUSIVE'
        out['R2']=dict(n=n,verdict=verdict,**p,worst_case=bound,rmst=rmst,
                       shape=dose_shape(rmst['rmst']),non_monotone=[v<=.05 for v in rmst['reverse_p']])
    return out
