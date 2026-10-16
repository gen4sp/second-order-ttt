"""Pure NumPy scoring fixtures; no torch process or training."""
import importlib.util
from pathlib import Path

import pytest

from tttmem import frs_cohorts as co
from tttmem import episodes as ep


@pytest.mark.parametrize("n,k,c,p,adjusted,expected",[
    (9,9,0,.002,.01,"NOT EVALUABLE"),
    (10,10,0,.001,.005,"REPLICATED"),
    (10,7,0,.001,.005,"PARTIAL"),
    (10,3,0,.1,.5,"NOT REPLICATED"),
    (10,7,3,.03,.15,"INCONCLUSIVE"),
])
def test_every_paired_verdict(n,k,c,p,adjusted,expected):
    assert co.paired_verdict(dict(n=n,test=k,comparator=c,p=p),adjusted)==expected


@pytest.mark.parametrize("k,n,expected",[(0,0,"NOT EVALUABLE"),(5,5,"NOT EVALUABLE"),
    (5,6,"REPLICATED"),(3,6,"NOT REPLICATED"),(4,6,"INCONCLUSIVE")])
def test_count_minima(k,n,expected):
    assert co.count_rule(k,n)["verdict"]==expected


def rows(loss=None,recovery=None,terminal=1500):
    return [dict(step=t,quick=dict(acc=.95 if loss is None or t<loss or
                                 (recovery is not None and t>=recovery) else .8,acc5=.95,acc6=.95),
                 write=dict(r_kv_med=.5,r_kv_mean=.5)) for t in range(5,terminal+1,5)]


def cell(seed,arm,loss=None,recovery=None):
    return dict(seed=seed,arm=arm,logs=rows(loss,recovery))


def test_transient_all_effective_denominators():
    cells=[cell(s,"e2e1emb",400) for s in range(6)]
    cells += [cell(s,"e2e2emb",400,420) for s in range(6)]
    cells += [cell(s,"e2e1emb_cn97",400,500) for s in range(6)]
    out=co.transient_adapter(cells)
    assert out["verdict"]=="REPLICATED"
    assert out["fo_no_recovery"]["n"]==6
    assert out["ordinal"]["n"]==out["ordinal"]["k"]==6
    assert all(r["n"]==6 for r in out["recovery"].values())
    assert out["km"]["e2e2emb"][-1]["survival"]==0


def test_transient_loss_too_rare_and_small_recovery_denominator():
    cells=[cell(s,"e2e2emb",400 if s<3 else None,420) for s in range(6)]
    out=co.transient_adapter(cells)
    assert out["incidence"]["e2e2emb"]["verdict"]=="REPLICATED"
    assert out["recovery"]["e2e2emb"]["n"]==3
    assert out["recovery"]["e2e2emb"]["verdict"]=="NOT EVALUABLE"
    cells=[cell(s,"e2e2emb") for s in range(6)]
    out=co.transient_adapter(cells)
    assert out["incidence"]["e2e2emb"]["reason"].startswith("loss too rare")


def test_fo_insufficient_followup_is_not_counted():
    out=co.transient_adapter([cell(s,"e2e1emb",1250) for s in range(6)])
    assert out["fo_no_recovery"]["n"]==0 and out["fo_no_recovery"]["verdict"]=="NOT EVALUABLE"


def test_divergence_and_log_gap_ineligible_for_transient():
    divergent=cell(0,"e2e1emb",400);divergent["diverged"]=True
    missing=cell(1,"e2e1emb",400);missing["logs"].pop(10)
    assert not co.transient_adapter([divergent,missing])["units"]["e2e1emb"]


def test_holm_ne_included_as_one_and_monotone():
    result=co.holm5({"R-SO":.001,"R-NOISE":.02,"R-LIN":.021})
    assert result["R-SO"]==.005
    assert result["R-NOISE"]==result["R-LIN"]==.08
    assert result["R-RP"]==result["R-ADDR"]==1


def test_adverse_reverse_bound_for_falsification():
    row=ep.adverse_completion(None,treatment=True,reverse=True)
    assert row["retained"] and row["lost_time"]==0
    row=ep.adverse_completion(None,treatment=False,reverse=True)
    assert not row["retained"] and row["lost_time"]==400


def test_analyzer_explicit_smoke_mode():
    path=Path(__file__).resolve().parents[1]/"scripts/analyze_frs.py"
    spec=importlib.util.spec_from_file_location("analyze_frs_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pool=dict(schema=1,smoke_only=True,complete=True,cells={},jobs={},
              provenance=dict(effective_config=dict(log_every=5,steps=80)))
    scores=module.score(pool,smoke=True)
    assert scores["status"]=="SMOKE ONLY — NOT SCORED"
    assert all(r["verdict"]=="SMOKE ONLY — NOT SCORED" for r in scores["family"].values())
    assert module.contract(pool,smoke=False)


def test_transient_aggregates_only_evaluable_parts():
    cells=[cell(s,co.FO,400) for s in range(6)]
    cells += [cell(s,co.SO,400,420) for s in range(6)]
    cells += [cell(s+1,co.NOISE,400,500) for s in range(6)]
    out=co.transient_adapter(cells)
    assert out['ordinal']['n']==5 and out['ordinal']['verdict']=='NOT EVALUABLE'
    assert out['verdict']=='REPLICATED'


def test_frsb_per_seed_contract_supplies_marginals_and_missing():
    import copy,json
    from tttmem.frs_family import validate_family_inputs,family_pair
    fixture=json.loads((Path(__file__).parent/'fixtures/frsb_family_inputs.json').read_text())
    validate_family_inputs(fixture,smoke=True)
    row=family_pair(fixture['family_inputs']['R-LIN'])
    assert row['n']==10 and row['test']==10 and row['comparator']==0 and row['p']==2**-10
    assert co.paired_verdict(row,5*row['p'])=='REPLICATED'
    missing=family_pair(fixture['family_inputs']['R-ADDR'])
    assert missing['missing']==list(range(1,10)) and missing['without_divergence']['n']==0
    with pytest.raises(ValueError):validate_family_inputs(None)
    bad=copy.deepcopy(fixture);bad['family_inputs']['R-LIN']['test'][0]=1
    with pytest.raises(ValueError):validate_family_inputs(bad,smoke=True)
    spec=importlib.util.spec_from_file_location('frs_contract_pipeline',Path(__file__).resolve().parents[1]/'scripts/analyze_frs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pool=dict(schema=1,smoke_only=True,complete=True,cells={},jobs={},
              provenance=dict(effective_config=dict(log_every=5,steps=80)))
    scored=module.score(pool,smoke=True,frsb=fixture)
    assert scored['family']['R-LIN']['n']==10 and scored['family']['R-LIN']['test']==10
    assert scored['family']['R-ADDR']['missing']==list(range(1,10))


def test_h4_registered_D_eligibility_excludes_small_gap():
    spec=importlib.util.spec_from_file_location('frs_h4_eligibility',Path(__file__).resolve().parents[1]/'scripts/analyze_frs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cells={}
    for seed,a6 in ((0,.65),(1,.8),(2,.75)):
        iv={variant:{str(c):v} for variant,c,v in (
            ('SLOW',6,.99),('RND_SLOW',6,a6),('SHAM5_SLOW',5,.95),
            ('BB',6,.99),('RND_BB',6,a6),('SHAM5_BB',5,.95))}
        cells[str(seed)]=dict(cell_id=str(seed),seed=seed,arm=co.FO,logs=rows(terminal=5),
            events={'t_acq':dict(event_step=5)},
            final={'ce':dict(acc=.1,acc5=.95,acc6=a6,decomp_gate_ok=True,intervention=iv)})
    pool=dict(schema=1,smoke_only=True,complete=True,cells=cells,jobs={},
              provenance=dict(effective_config=dict(log_every=5,steps=5)))
    out=module.score(pool,smoke=True)
    assert out['R_H4']['SLOW']['n']==2 and out['R_H4']['BB']['n']==2
    assert out['R_H4_excluded'][0]['seed']==1
