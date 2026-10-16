import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from tttmem import episodes as e,triggers


def rows(H=400,acc=.95,every=5):return [dict(step=t,acc=acc) for t in range(every,H+1,every)]


@pytest.mark.parametrize('common',[False,True])
def test_rdo_last_onset_395_confirmed_400(common):
    r=rows(acc=.8);r[-2]['acc']=r[-1]['acc']=.95
    s=e.endpoints(r,0,full_acc=.95)
    assert s['disposition']=='COMPLETE' and s['t_re']==395 and s['tau']==395 and s['retained']


def test_administrative_censoring_and_transient_retention():
    s=e.endpoints(rows(acc=.8),0,full_acc=.8)
    assert s['tau']==400 and s['censored']
    r=rows(acc=.8);r[0]['acc']=r[1]['acc']=.95
    s=e.endpoints(r,0,full_acc=.8)
    assert s['tau']==5 and not s['retained']


@pytest.mark.parametrize('recovery',[False,True])
def test_divergence_before_or_after_confirmed_recovery(recovery):
    r=rows(20,acc=.8)
    if recovery:r[0]['acc']=r[1]['acc']=.95
    cert=e.Divergence(20,20,2)
    s=e.endpoints(r,0,certificate=cert,full_acc=0)
    assert s['disposition']=='DIVERGED' and not s['censored'] and not s['retained']
    assert s['tau']==(5 if recovery else 400)
    assert e.adverse_completion(s,treatment=False)['nl2']


def test_uncertified_shortened_grid_and_nan_are_missing():
    assert e.endpoints(rows(20),0,full_acc=.95)['disposition']=='TRUNCATED'
    r=rows();r[2]['acc']=r[3]['acc']=np.nan
    assert e.endpoints(r,0,full_acc=.95)['nl2'] is None
    assert e.endpoints(rows(20),0,certificate=e.Divergence(20,20,1),full_acc=0)['disposition']=='TRUNCATED'


@pytest.mark.parametrize('horizon',[400,305])
def test_rpe_onset_300_confirmed_305_own_and_common(horizon):
    r=rows(horizon);r[59]['acc']=r[60]['acc']=.8
    s=e.endpoints(r,0,horizon,full_acc=.95,require_final=horizon==400)
    assert s['t_loss']==300 and s['nl2'] is False


def test_dense_onset_at_300_through_309_and_gap():
    r=rows(309,every=1)
    for x in r[299:309]:x['acc']=.8
    assert e.nl_fine(r,0) is False
    assert e.nl_fine(r[:-1],0) is None


def test_rpe_divergence_after_305_keeps_observed_nl2():
    s=e.endpoints(rows(310),0,certificate=e.Divergence(310,310,16),full_acc=0)
    assert s['nl2'] and s['lost_time']==90 and s['disposition']=='DIVERGED'
    r=rows(50);r[0]['acc']=r[1]['acc']=.8
    s=e.endpoints(r,0,certificate=e.Divergence(50,50,3),full_acc=0)
    assert s['t_loss']==5 and not s['nl2']


def fixture():
    series=defaultdict(list)
    for r in csv.DictReader((Path(__file__).parent/'fixtures/chr_rdo_rpe.csv').open()):
        series[(r['cell_id'],r['branch'])].append(dict(step=int(float(r['step'])),acc=float(r['a']),r_kv_med=float(r['r'])))
    return series


def test_s1_third_log_exact_chr_main_table():
    expected={0:595,2:540,3:650,4:610,5:575,6:570,7:585,8:770,9:670}
    for (cid,branch),r in fixture().items():
        if branch!='main':continue
        seed=int(cid.split('/')[1][1:]);found=triggers.detect(r)
        if seed==1:assert 'S1' not in found;continue
        assert found['S1'].state_step==expected[seed]
        at=found['S1'].state_step
        prefix=[x for x in r if x['step']<at]
        assert 'S1' not in triggers.detect(prefix)


def test_frozen_chr_rpe_dryrun_and_full_horizon_terminal_recovery():
    series=fixture();counts=defaultdict(int);better=defaultdict(int)
    for (cid,branch),main in series.items():
        if branch!='main' or '/s1/' in cid:continue
        acq=triggers.detect(main)['t_acq'].event_step
        def score(branch,h=400):
            r=[x for x in series[(cid,branch)] if acq<x['step']<=acq+h]
            return e.endpoints(r,acq,h,require_final=False)
        twin=score('fork-ce-twin')
        for tag in ('rp03','rp10','nc03','nc10'):
            s=score('fork-'+tag)
            assert s['disposition']=='COMPLETE'
            counts[tag]+=s['nl2'];better[tag]+=s['lost_time']<twin['lost_time']
        if '/s0/' in cid:
            assert score('fork-rp03')['episodes'][-1]['censored']
            assert not score('fork-rp03',1500-acq)['episodes'][-1]['censored']
            assert score('fork-rp03',1500-acq)['episodes'][-1]['outage']>50
    assert dict(counts)==dict(rp03=0,rp10=2,nc03=0,nc10=0)
    assert dict(better)==dict(rp03=9,rp10=9,nc03=0,nc10=1)
