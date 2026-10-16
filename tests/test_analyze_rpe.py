import pytest
from tttmem import rpe_analysis as a


def score(k=12,nc=0,tw=0,n=12,**kw):
    t=[True]*k+[False]*(n-k); c=[True]*nc+[False]*(n-nc); w=[True]*tw+[False]*(n-tw)
    defaults=dict(manipulation='OK',write_gate=True,holm_pass=True)
    defaults.update(kw)
    return a.dose_score(t,c,w,[0 if x else 100 for x in t],[400]*n,[400]*n,[None if x else 20 for x in t],t,**defaults)


@pytest.mark.parametrize('k,nc,label',[(12,0,'PREVENTS'),(12,12,'PREVENTS-NONSPECIFIC'),
                                     (12,4,'PREVENTS-UNRESOLVED'),(8,0,'PARTIAL'),(6,0,'SHORTENS')])
def test_all_tiers(k,nc,label):
    r=score(k,nc); assert r['verdict']==label
    if label=='SHORTENS': assert r['speed']=='FAST'


def test_no_effect_slow_and_inference_unresolved():
    n=12
    r=a.dose_score([False]*n,[False]*n,[False]*n,[400]*n,[400]*n,[400]*n,[400]*n,[False]*n)
    assert r['verdict']=='NO-EFFECT'
    r=a.dose_score([False]*n,[False]*n,[False]*n,[100]*n,[400]*n,[400]*n,[100]*n,[False]*n)
    assert r['verdict']=='SHORTENS' and r['speed']=='SLOW'
    assert score(holm_pass=False)['verdict']=='PREVENTS-UNRESOLVED'


def test_precedence_round2_baseline_and_each_dose_gate():
    assert score(n=7,k=7,manipulation='FAILED')['verdict']=='NOT EVALUABLE'
    assert score(tw=4,manipulation='FAILED')['verdict']=='MANIPULATION FAILED'
    assert score(tw=4,write_gate=False)['verdict']=='BASELINE NOT MET'
    assert score(write_gate=False)['verdict']=='INCONCLUSIVE (write gate)'


def test_rapid_reversal_counts_only_nl2_true():
    t=[True]*9+[False]*3
    r=a.dose_score(t,[False]*12,[False]*12,[0]*9+[400]*3,[400]*12,[400]*12,
                   [None]*9+[400]*3,t,holm_pass=True)
    assert not r['rapid_reversal_compatible']
    fine=[False]*3+[True]*6+[False]*3
    r=a.dose_score(t,[False]*12,[False]*12,[0]*12,[400]*12,[400]*12,[None]*12,fine,holm_pass=True)
    assert r['rapid_reversal_compatible'] and r['rapid_false']==3


def event(nl=False,disp='COMPLETE'):
    return dict(disposition=disp,NL2=nl,L=0 if nl else 400,NL_fine=nl,
                first_outage=None if nl else 400,t_loss=None if nl else 5)


def units(n=12):
    return [dict(seed=0,fixture_unit=i,events={'ce':{'ce':event(), 'rpf30':event(),'rpf100':event()},
                               **{'rpf'+d:{'rpf'+d:event(True)} for d in ('30','100')},
                               **{'ncf'+d:{'ncf'+d:event(),'rpf'+d:event()} for d in ('30','100')}},
                 manipulation={'30':False,'100':False},write_gate={'30':True,'100':True}) for i in range(n)]


def test_two_conjunctions_holm_and_outcome_free_baseline():
    u=units(); r=a.score(u)
    assert all(d['verdict']=='PREVENTS' for d in r['doses'].values())
    for x in u[:4]:x['events']['ce']['ce']=event(True)
    assert all(d['verdict']=='BASELINE NOT MET' for d in a.score(u)['doses'].values())
    assert a.score(u)['frozen_n']==12


def test_dose_gate_independent():
    u=units()
    for x in u[:4]: x['write_gate']['30']=False
    r=a.score(u)
    assert r['doses']['30']['verdict']=='INCONCLUSIVE (write gate)'
    assert r['doses']['100']['verdict']=='PREVENTS'


def test_missing_completions_and_diverged_controls():
    u=units()
    for x in u[:4]: x['events']['ncf30']['ncf30']['disposition']='DIVERGED'
    r=a.score(u)['doses']['30']
    assert r['n']==12 and r['adverse']['ncf']==4 and r['verdict']=='PARTIAL (missing-sensitive)'
    u=units()
    for x in u[:6]:x['events']['ce'].pop('rpf30')
    assert a.score(u)['doses']['30']['selectivity']=='unknown'


@pytest.mark.parametrize('t,tw,nc,label',[([1]*12,[1]*6+[0]*6,[0]*12,'eval-carried'),
                                       ([1]*12,[0]*12,[0]*12,'training-selective'),
                                       ([1]*12,[0]*12,[1]*6+[0]*6,'not-kv-specific'),
                                       ([1]*6+[0]*6,[0]*12,[0]*12,'mixed')])
def test_common_selectivity(t,tw,nc,label):assert a.selectivity(t,tw,nc)==label


def test_descriptive_replication_classes_and_dose_order():
    u=units()
    for i,x in enumerate(u):
        endpoint=event(i<2);endpoint.update(L=0 if i<2 else 50,first_outage=None if i<2 else 20)
        x['events']['rp10']={'rp10':endpoint}
    r=a.score(u)
    assert r['R_replication']['NL2']==2 and r['R_replication']['CHR_NL2']==2
    assert r['R_replication']['classes']['count_label']=='SHORTENS-FAST'
    assert r['D_dose_order']['rp10']['NL2']==2
    assert r['D_dose_order']['rpf100']['NL2']==12
