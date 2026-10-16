import itertools
import numpy as np
import pytest

from tttmem import rdo_analysis as a
from tttmem import rdo_statistics as s


def event(ok=False,retained=False,disposition='COMPLETE',tau=None):
    return dict(disposition=disposition,reacquired=ok,tau=(5 if ok else 400) if tau is None else tau,
                retention=retained,eval_immediate=False)


def units(n=12):
    return [dict(seed=0,fixture_unit=i,events={**{'ce':{'ce':event(),**{'rpf'+d:event() for d in a.DOSES}}},
                                **{'rpf'+d:{'rpf'+d:event(True,True)} for d in a.DOSES},
                                **{'ncf'+d:{'rpf'+d:event()} for d in a.DOSES}},
                 manipulation={d:False for d in a.DOSES}) for i in range(n)]


@pytest.mark.parametrize('kind',['R1','R1-T','R3'])
def test_component_supported(kind):
    assert a.component([True]*12,[False]*12,[False]*12,kind)['verdict']=='SUPPORTED'


@pytest.mark.parametrize('kind',['R1','R1-T','R3'])
def test_precedence(kind):
    assert a.component([True]*7,[False]*7,[True]*7,kind,'FAILED')['verdict']=='NOT EVALUABLE'
    r=a.component([True]*12,[False]*12,[True]*12,kind,'FAILED')['verdict']
    assert r==('INCONCLUSIVE (unstable baseline)' if kind=='R1' else 'MANIPULATION FAILED')


def test_falsified_and_inconclusive_branches():
    assert a.component([False]*12,[False]*12,[False]*12,'R1')['verdict']=='FALSIFIED'
    assert a.component([True]*10+[False]*2,[True]*6+[False]*6,[False]*12,'R1-T')['verdict']=='FALSIFIED'
    assert a.component([True]*7+[False]*5,[False]*12,[False]*12,'R3')['verdict'].endswith('(not applicable)')
    assert a.component([True]*8+[False]*4,[False]*12,[False]*12,'R1')['verdict']=='INCONCLUSIVE'


def test_one_headline_cohort_and_frozen_sensitivity():
    u=units(); del u[-1]['events']['ncf03']
    r=a.score(u)
    assert r['frozen_n']==12 and r['headline']['n']==11
    assert {v['n'] for v in r['doses']['03']['tests'].values()}=={11}
    assert all(v['adverse']['n']==12 for v in r['doses']['03']['tests'].values())


def test_missing_and_diverged_control_adverse():
    u=units()
    for x in u[:4]: x['events']['ncf03']['rpf03']['disposition']='DIVERGED'
    r=a.score(u)['doses']['03']['tests']['R3']
    assert r['verdict']=='INCONCLUSIVE (missing-sensitive)'
    assert r['adverse']['control']==4


def test_falsified_reverse_completion():
    u=units()
    for x in u[:8]: x['events']['rpf03']['rpf03']=event()
    for x in u[8:]: x['events'].pop('ncf03'); x['events'].pop('rpf03')
    r=a.score(u)['doses']['03']['tests']['R1']
    assert r['verdict']=='FALSIFIED' and r['reverse']['treated']==4
    u=units()
    for x in u[:4]: x['events']['rpf03']['rpf03']=event()
    for x in u[8:]: x['events'].pop('ncf03'); x['events'].pop('rpf03')
    assert a.score(u)['doses']['03']['tests']['R1']['verdict']=='INCONCLUSIVE (missing-sensitive)'


def test_durability_is_separate_from_reacquisition():
    u=units()
    assert a.score(u)['headline']['durability']=='DURABLE'
    for x in u: x['events']['rpf03']['rpf03']['retention']=False
    r=a.score(u)['headline']
    assert r['supported'] and r['durability']=='durability not shown'


@pytest.mark.parametrize('tau',[[[400,400,100,20]],[[40,40,10,5],[400,100,100,20]],[[20,20,20,20]]*4])
def test_page_against_brute_force_with_ties(tau):
    ranks=np.array([s.midranks_speed(t) for t in tau]); observed=float(np.sum(ranks@s.J))
    counts=np.array([r[s.PERMS]@s.J for r in ranks])
    null=counts[0]
    for row in counts[1:]: null=(null[:,None]+row[None,:]).ravel()
    assert s.page(tau)['p']==pytest.approx(np.mean(null>=observed),abs=1e-14)


def test_audit_page_and_worst_case_bound():
    ranks=[[1,2,3,4]]*8
    assert s.page_from_ranks(ranks)['p']==pytest.approx(9.0846890375e-12,rel=1e-5)
    assert s.page_from_ranks(ranks+[[2,2,2,4]]*4)['p']==pytest.approx(3.549e-14,rel=1e-3)
    bound=s.worst_case_page([[200,100,50,20]]*8+[[None,None,None,20]]*4)
    assert bound['combinations']==4096 and bound['p']==pytest.approx(.02515,abs=1e-5)
    u=units()
    for x in u[:8]:
        for d,tau in zip(a.DOSES,[200,100,50,20]):x['events']['rpf'+d]['rpf'+d]['tau']=tau
    for x in u[8:]:
        for d in a.DOSES[:3]:x['events'].pop('rpf'+d)
        x['events']['rpf30']['rpf30']['tau']=20
    assert a.score(u)['R2']['verdict']=='INCONCLUSIVE (missing-sensitive)'


def test_newcombe_audit_intervals_and_cp():
    assert s.newcombe10(6,4,0,2)==pytest.approx([.054,.554],abs=.001)
    assert s.newcombe10(4,6,2,0)==pytest.approx([-.116,.652],abs=.001)
    assert s.cp(3,12)==pytest.approx([.0548606,.5718585],abs=1e-7)


@pytest.mark.parametrize('rmst,label',[([20,40,60,200],'REVERSED'),([40,40,40,40],'FLAT'),
                                      ([400,100,100,100],'THRESHOLD'),([400,300,200,100],'GRADED'),
                                      ([400,200,210,100],'IRREGULAR')])
def test_shapes(rmst,label): assert s.dose_shape(rmst)==label


def test_r2_verdicts_and_qualifiers():
    u=units(); assert a.score(u)['R2']['verdict']=='PLATEAU'
    for x in u:
        for d,tau in zip(a.DOSES,[200,100,50,20]):x['events']['rpf'+d]['rpf'+d]['tau']=tau
    assert a.score(u)['R2']['verdict']=='GRADED TREND'
    for x in u:x['events']['rpf30']['rpf30']=event()
    assert a.score(u)['R2']['verdict']=='NO RESCUE'
    for x in u:x['manipulation']['01']=True; x['manipulation']['10']=True
    assert a.score(u)['R2']['verdict']=='INCONCLUSIVE'


def test_selectivity_unknown_eval_and_selective():
    assert a.selectivity([1]*12,[0]*12,[0]*12,[0]*12)=='training-selective'
    assert a.selectivity([1]*12,[1]*6+[0]*6,[0]*12,[0]*12)=='eval-carried'
    u=units()
    for x in u[:6]: x['events']['ce'].pop('rpf03')
    assert a.score(u)['doses']['03']['selectivity']=='unknown'
