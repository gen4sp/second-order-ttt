"""Registered RPE tiers and PREVENTS conjunction; other labels are descriptive."""
from __future__ import annotations
import math
import numpy as np

from .rdo_statistics import paired,cp,mcnemar,holm
from .rdo_analysis import observed,value,binary


def dose_score(t,nc,tw,L,Lnc,Ltw,first,fine,*,manipulation='OK',write_gate=True,holm_pass=False):
    n=len(t); k=sum(t); kn=sum(nc); kw=sum(tw)
    pair_nc,pair_tw=paired(t,nc),paired(t,tw)
    tier='HIGH' if k>=math.ceil(.75*n) else 'LOW' if k<=math.floor(.5*n) else 'MIDDLE'
    b=sum(a<b for a,b in zip(L,Ltw)); c=sum(a>b for a,b in zip(L,Ltw)); sign=mcnemar(b,c)
    if tier=='HIGH':
        count='PREVENTS' if kn<=math.floor(.25*n) and holm_pass else 'PREVENTS-NONSPECIFIC' if kn>=math.ceil(.75*n) else 'PREVENTS-UNRESOLVED'
    elif tier=='LOW' and b>=math.ceil(.75*n) and sign<=.05: count='SHORTENS'
    elif tier=='LOW' and k<=math.floor(.25*n) and b<=math.floor(.5*n): count='NO-EFFECT'
    else: count='PARTIAL'
    if n<8: verdict='NOT EVALUABLE'
    elif manipulation=='FAILED': verdict='MANIPULATION FAILED'
    elif kw>math.floor(.25*n): verdict='BASELINE NOT MET'
    elif not write_gate: verdict='INCONCLUSIVE (write gate)'
    else: verdict=count
    lost=[v for ok,v in zip(t,first) if not ok and v is not None]
    fast='N/A' if not lost else 'FAST' if np.median(lost)<=50 else 'SLOW'
    bn=sum(a<b for a,b in zip(L,Lnc)); cn=sum(a>b for a,b in zip(L,Lnc))
    rapid=sum(ok and fi is False for ok,fi in zip(t,fine))
    missing=sum(ok and fi is None for ok,fi in zip(t,fine))
    return dict(n=n,k=k,tier=tier,verdict=verdict,count_label=count,ci=cp(k,n),ncf=kn,twin=kw,
                p=max(pair_nc['p'],pair_tw['p']),paired_ncf=pair_nc,paired_twin=pair_tw,
                shortens_sign_p_unadjusted=sign,shortens_count=b,speed=fast,
                kv_specific_descriptive=bn>=math.ceil(.75*n) and mcnemar(bn,cn)<=.05,
                rapid_reversal_compatible=tier=='HIGH' and rapid+missing>=math.ceil(.25*n),
                rapid_false=rapid,rapid_missing=missing,manipulation=manipulation,write_gate=write_gate)


def selectivity(t,tw,nc):
    n=len(t)
    if sum(tw)>=math.ceil(.5*n): return 'eval-carried'
    if sum(a and not b and not c for a,b,c in zip(t,tw,nc))>=math.ceil(.75*n): return 'training-selective'
    if sum(nc)>=math.ceil(.5*n): return 'not-kv-specific'
    return 'mixed'


def score(units,H=400):
    def e(u,b,r): return u.get('events',{}).get(b,{}).get(r)
    prepared={}
    for d in ('30','100'):
        rp,nc='rpf'+d,'ncf'+d
        cohort=[u for u in units if all(observed(v) and v.get('NL2') is not None for v in (e(u,'ce','ce'),e(u,rp,rp),e(u,nc,nc),e(u,'ce',rp),e(u,nc,rp)))]
        n=len(cohort)
        bad=sum(bool(u.get('manipulation',{}).get(d,True)) for u in cohort)
        manipulation='FAILED' if n and bad>=math.ceil(.5*n) else 'LIMITED' if bad>math.floor(.25*n) else 'OK'
        gate=sum(not u.get('write_gate',{}).get(d,False) for u in cohort)<=math.floor(.25*n)
        def vectors(us,completion=None,rp=rp,nc=nc):
            adverse=completion is not False
            t=[binary(e(u,rp,rp),'NL2',not adverse) for u in us]
            cc=[binary(e(u,nc,nc),'NL2',adverse,control=completion is not None,adverse=adverse) for u in us]
            tw=[binary(e(u,'ce','ce'),'NL2',adverse,control=completion is not None,adverse=adverse) for u in us]
            def loss(branch,control):
                return [0 if control and completion is not None and observed(e(u,branch,branch)) and e(u,branch,branch)['disposition']=='DIVERGED'
                        else value(e(u,branch,branch),'L',0 if control and adverse else H) for u in us]
            return (t,cc,tw,loss(rp,False),loss(nc,True),loss('ce',True),
                    [value(e(u,rp,rp),'first_outage',None) for u in us],
                    [value(e(u,rp,rp),'NL_fine',False if completion is not None else None) for u in us])
        vec=vectors(cohort)
        raw=dose_score(*vec,manipulation=manipulation,write_gate=gate)
        prepared[d]=(cohort,manipulation,gate,vec,vectors,raw)
    passed=holm({d:v[-1]['p'] for d,v in prepared.items()},.05)
    adverse_p={d:max(paired(v[4](units,True)[0],v[4](units,True)[1])['p'],
                     paired(v[4](units,True)[0],v[4](units,True)[2])['p']) for d,v in prepared.items()}
    adverse_holm=holm(adverse_p,.05)
    out=dict(frozen_n=len(units),doses={})
    for d,(cohort,manipulation,gate,vec,vectors,raw) in prepared.items():
        rp,nc='rpf'+d,'ncf'+d
        result=dose_score(*vec,manipulation=manipulation,write_gate=gate,holm_pass=passed[d])
        sensitivity=dose_score(*vectors(units,True),manipulation=manipulation,write_gate=gate,holm_pass=adverse_holm[d])
        result['adverse']=sensitivity; result['holm_pass']=passed[d]
        if result['verdict'] not in ('NOT EVALUABLE','MANIPULATION FAILED','BASELINE NOT MET','INCONCLUSIVE (write gate)','PARTIAL') and sensitivity['verdict']!=result['verdict']:
            result['verdict']='PARTIAL (missing-sensitive)'
        labels=[]
        for adverse in (True,False):
            labels.append(selectivity([binary(e(u,rp,rp),'NL2',not adverse) for u in units],
                                      [binary(e(u,'ce',rp),'NL2',adverse,control=True,adverse=adverse) for u in units],
                                      [binary(e(u,nc,rp),'NL2',adverse,control=True,adverse=adverse) for u in units]))
        result['selectivity']=labels[0] if labels[0]==labels[1] else 'unknown'
        result['eval_carried_overlap']=paired(vec[0],[e(u,'ce',rp)['NL2'] for u in cohort])
        leads=[(e(u,rp,rp).get('t_loss'),e(u,'ce','ce').get('t_loss')) for u in cohort]
        delta=[a-b for a,b in leads if a is not None and b is not None]
        result['timing']=dict(m=len(delta),status='insufficient failures' if len(delta)<6 else 'descriptive',
                              equal=sum(v==0 for v in delta),within10=sum(abs(v)<=10 for v in delta))
        result['seeds']=[u['seed'] for u in cohort]
        out['doses'][d]=result
    # Registered descriptive anchor and dose-order rows, outside Holm-2.
    anchor=[u for u in units if observed(e(u,'rp10','rp10')) and e(u,'rp10','rp10').get('NL2') is not None]
    outcome_counts={}
    for u in anchor:
        endpoint=e(u,'rp10','rp10')
        label='NL2' if endpoint['NL2'] else 'FAST LOSS' if endpoint.get('first_outage') is not None and endpoint['first_outage']<=50 else 'SLOW LOSS'
        outcome_counts[label]=outcome_counts.get(label,0)+1
    paired_anchor=[u for u in anchor if observed(e(u,'ce','ce')) and e(u,'ce','ce').get('NL2') is not None]
    t=[e(u,'rp10','rp10')['NL2'] for u in paired_anchor]
    tw=[e(u,'ce','ce')['NL2'] for u in paired_anchor]
    labels=dose_score(t,tw,tw,[value(e(u,'rp10','rp10'),'L',H) for u in paired_anchor],
        [value(e(u,'ce','ce'),'L',H) for u in paired_anchor],
        [value(e(u,'ce','ce'),'L',H) for u in paired_anchor],
        [value(e(u,'rp10','rp10'),'first_outage',None) for u in paired_anchor],
        [value(e(u,'rp10','rp10'),'NL_fine',None) for u in paired_anchor])
    count_label=labels['count_label']
    if count_label=='SHORTENS':count_label+='-'+labels['speed']
    out['R_replication']=dict(n=len(anchor),NL2=sum(e(u,'rp10','rp10')['NL2'] for u in anchor),
        outcome_counts=outcome_counts,classes=dict(n=len(paired_anchor),tier=labels['tier'],
            count_label=count_label if len(paired_anchor)>=8 else 'NOT EVALUABLE',
            shortens_sign_p_unadjusted=labels['shortens_sign_p_unadjusted']),
        CHR_NL2=2,CHR_n=9,descriptive=True)
    out['D_dose_order']={tag:dict(n=sum(observed(e(u,tag,tag)) and e(u,tag,tag).get('NL2') is not None for u in units),
        NL2=sum(bool(e(u,tag,tag).get('NL2')) for u in units if observed(e(u,tag,tag))),descriptive=True)
        for tag in ('rp10','rpf30','rpf100')}
    return out
