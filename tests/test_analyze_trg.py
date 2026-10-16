import os
import setproctitle
import numpy as np
import pytest
from tttmem import analyze_trg as a, episodes as e, trg_triggers as tr, trg_jobs as jobs
setproctitle.setproctitle(os.environ.get("PROCESS_NAME","TTM-trg-77-tests"))

def test_aggregated_mean_telescoping_median_counterexample():
    steps = np.arange(61)
    terms = np.zeros((61, 1, 3, 6))
    terms[35:46, 0, 0, 0] = 10
    terms[35:46, 0, 1, 1] = 10
    out = a.localisation(terms, terms.sum(-1), steps, [[6, 6, 6]], 0, 40)
    assert out["eligible"] and out["accounting_error"] < 1e-5
    assert sum(np.median(terms[35:46], axis=(0,1,2))) == 0
    assert np.median(terms[35:46].sum(-1)) == 10


def test_sham_hash_counts_no_drop_after600():
    drops = list(range(580, 598))
    sha = tr.realised_drop_hash(drops)
    expected = list(range(583, 601))
    assert tr.yoked_sham(drops, sha) == expected
    assert tr.yoked_sham(drops, sha) == expected
    with pytest.raises(ValueError, match="sha256"):
        tr.yoked_sham(drops, "bad")
    with pytest.raises(ValueError):
        tr.yoked_sham([598], tr.realised_drop_hash([598]))
    assert tr.yoked_sham([], tr.realised_drop_hash([])) == []


@pytest.mark.parametrize("kind", ["linear", "convex", "sigmoid", "audit-r", "audit-q"])
def test_onset_proto_scenarios(kind):
    s = np.arange(121)
    delta = np.maximum(s-40, 0)
    if kind == "linear": x = delta/30
    elif kind == "convex": x = (delta/30)**2
    elif kind == "sigmoid": x = 1/(1+np.exp(-(s-55)/3.75))-1/(1+np.exp(4))
    elif kind == "audit-r": x = .8-.004*delta
    else: x = .1+.0002*delta
    direction = -1 if kind == "audit-r" else 1
    out = a.fractional_crossing(s, x, 0, 100, direction=direction)
    assert out["label"] == "CROSSED" and out["confirmed"] == out["onset"]+4
    h = a.hinge_onset(s, x, 0, 100)
    assert 5 <= h <= 115
    if kind in ("linear", "audit-r", "audit-q"):
        assert h == 40


def test_flat_reversed_and_hinge_tie():
    s = np.arange(121)
    assert a.fractional_crossing(s, np.ones(121), 0, 100)["label"] == "FLAT"
    assert a.fractional_crossing(s, -np.maximum(s-40, 0), 0, 100)["label"] == "REVERSED"
    assert a.hinge_onset(s, np.ones(121), 0, 100) == 5
    assert a.fractional_crossing(s, np.arange(121), 0, 281)["label"] == "INELIGIBLE"


def test_kappa_deterministic_default_and_no_false_alarm_unit():
    assert tr.select_kappa([])["kappa"] == 5
    s = np.arange(0, 150)
    acc = np.ones(150)
    acc[30:33] = .5
    acc[80:83] = .5
    norms = np.ones(150)
    norms[[29, 79]] = 20
    unit = dict(steps=s, accuracy=acc, t_acq=0, telemetry_steps=s, norms=norms)
    result = tr.select_kappa([unit]*3)
    assert result["kappa"] == 10 and result["qualifying_units"] == 3
    assert result == tr.select_kappa([unit]*3)
    crowded = dict(unit, steps=np.arange(30), accuracy=np.resize([.5, 1., .5], 30),
                   telemetry_steps=np.arange(30), norms=np.ones(30))
    assert tr.select_kappa([crowded]*3)["no_data_default"]


def test_controller_warmup_history_and_end_boundary():
    c0 = tr.SpikeController(550)
    for s in range(551, 561):
        assert not c0.observe(s, 1.)["active"]
    assert c0.observe(561, 6.)["flag"]
    assert not c0.observe(598, 100.)["active"]
    before = c0.finite_count
    c0.observe(599, np.nan)
    assert c0.finite_count == before


def test_window_fail_open_and_t_l1_real_lookahead():
    window = tr.WindowController(10, "FO")
    assert window.ladder_due(310) and not window.ladder_due(311)
    for s in range(10, 30):
        window.observe(s, 1. if s < 20 else .8)
    assert window.onset == 20 and window.ladder_due(80) and not window.ladder_due(81)
    assert tr.dense_onset(range(10, 19), [.8]*9, 0) is None


def test_disposition_gap_nan_certified_divergence_and_nl2_boundary():
    s = np.arange(5, 306, 5)
    acc = np.ones(len(s))
    assert e.classify_branch(s, acc, start=0, end=305) == e.COMPLETE
    acc[-2:] = .8
    out = e.nl2_window(s, acc, 0)
    assert out["nl2"] is False and out["onset"] == 300
    assert e.nl2_window(s[:-1], acc[:-1], 0)["missing"]
    assert e.classify_branch(s[:-1], acc[:-1], start=0, end=305) == e.TRUNCATED
    assert e.classify_branch([5,10,15,20], [.8,.8,np.nan,np.nan], start=0, end=20) == e.TRUNCATED
    assert e.classify_branch([5,10,15], [1.,1.,1.], start=0, end=305, certified=True,
                             t_div=20, continuation_steps=20, nonfinite=2) == e.DIVERGED
    assert not e.nl2_window([5,10], [1.,1.], 0, disposition=e.DIVERGED, t_div=15)["nl2"]
    assert e.nl2_window(s[:-1], np.ones(len(s)-1), 0, disposition=e.DIVERGED, t_div=310)["missing"]
    assert e.certified_divergence(2,20) and not e.certified_divergence(1,20)


def test_burden_round2_witness_zero_tie_and_divergence():
    s = np.arange(661)
    skip, sham = np.ones(661), np.ones(661)
    skip[200:] = .5
    for k in range(10): sham[220+40*k:223+40*k] = .5
    us, uh = a.unbound_burden(s, skip, 70), a.unbound_burden(s, sham, 70)
    assert us["U"] == 461 and uh["U"] == 30
    assert a.burden_ratio(us["U"], uh["U"]) == pytest.approx(-.934924)
    assert a.burden_ratio(0,0) is None
    assert a.score_ht4([dict(skip=us, sham=uh, length=580)])['label'] == "NO REDUCTION"
    div = a.unbound_burden(np.arange(81, 200), np.ones(119), 70, disposition=e.DIVERGED, t_div=200)
    assert div["U"] == 461
    assert a.unbound_burden(np.arange(81,200), np.ones(119),70, disposition=e.TRUNCATED)["U"] is None


def test_ht3_holm2_and_worst_completion():
    units = [dict(family=f, kvm=dict(nl2=True), rnd=dict(nl2=False)) for f in ("FO","noise") for _ in range(8)]
    out = a.score_ht3(units)
    assert all(v["label"] == "SUPPORTED" for v in out.values())
    assert out["FO"]["holm_p"] == pytest.approx(2/256)
    units += [dict(family="FO",kvm=dict(nl2=None), rnd=dict(nl2=True)) for _ in range(8)]
    out = a.score_ht3(units)
    assert out["FO"]["label"] == "INCONCLUSIVE" and out["FO"]["missing_sensitive"]
    assert out["FO"]["n"] == 8 and out["FO"]["worst"]["n"] == 16
    units = [dict(family="FO",kvm=dict(nl2=False),rnd=dict(nl2=False)) for _ in range(8)]
    assert a.score_ht3(units)["FO"]["label"] == "FALSIFIED"
    assert a.score_ht3(units[:7])["FO"]["label"] == "NOT EVALUABLE"


def test_ht4_worst_missing_and_both_zero():
    units = [dict(skip=dict(U=10), sham=dict(U=100), length=200) for _ in range(6)]
    assert a.score_ht4(units)["label"] == "REDUCES"
    units.append(dict(skip=dict(U=None), sham=dict(U=0), length=200))
    assert a.score_ht4(units)["label"] == "UNCLEAR"
    assert a.score_ht4(units)["missing_sensitive"]
    zeros = a.score_ht4([dict(skip=dict(U=0), sham=dict(U=0), length=200)])
    assert zeros["ties"] == 1 and zeros["non_tied"] == 0


def test_harrell_conventions_and_event_censoring():
    assert a.harrell_c(np.arange(6), [True]*6, np.arange(6)[::-1])["C"] == 1.
    assert a.harrell_c(np.arange(6), [True]*6, np.ones(6))["C"] == .5
    assert a.harrell_c([1,1,1,1,1,1], [True]*6, np.arange(6))["C"] is None
    assert a.harrell_c([1,2,3], [True]*3, [3,2,1])["C"] is None
    assert a.harrell_c(np.arange(6), [False]*6, np.arange(6))["comparable"] == 0
    s = np.arange(31)
    m = np.ones((31, 1, 2))
    m[24:,0,0] = -1
    m[21:,0,1] = -1
    fail = a.event_failures(s,m,0)
    assert not fail["observed"][0,0] and fail["observed"][0,1]


def test_job_specs_outcome_free_horizon_dependency():
    meta = dict(family="FO", arm="e2e1emb", cell_id="echo/s0/e2e1emb/eta40/st1500",
                seed=0, valid=True, state_path="fork.pt", state_sha256="a"*64, trigger="t_acq",
                state_step=10, event_step=10, detected_step=10, completed_updates=10)
    built = jobs.make_job_specs(meta)
    specs = built["jobs"]
    assert built["disposition"] == "enrolled" and "RPE's enrolled units" in built["trg_rpe_intersection_rule"]
    assert len(specs) == 2 and all(s["end"] == 315 and s["reads"]["ladder_every"] == 0 for s in specs)
    assert all(s["arm"] == "e2e1emb" and s["state_sha256"] == "a"*64 for s in specs)
    assert specs[1]["intervention_rng"] == [0,4242,10]
    assert jobs.make_job_specs(dict(meta, valid=False))["disposition"] == "validation-failed"
    assert jobs.make_job_specs(dict(meta, detected_step=11))["disposition"] == "validation-failed"
    so = jobs.make_job_specs(dict(meta,family="SO",arm="e2e2emb"),
                             dict(kappa=3, addendum_sha256="b"*64))["jobs"]
    assert so[1]["after"] == [so[0]["job_id"]]
    assert all(s["kappa"] == 3 and s["addendum_sha256"] == "b"*64 for s in so)
    skip = dict(job_id=so[0]["job_id"], disposition="COMPLETE", drops=[20], drop_list_sha256=tr.realised_drop_hash([20]))
    assert jobs.resolve_yoked_job(so[1],skip)["schedule"] == [23]
    assert jobs.resolve_yoked_job(so[1],dict(skip,disposition="TRUNCATED"))["blocked"]


def test_frs_state_rows_and_all_unit_dispositions():
    row = dict(cell_id="echo/s0/e2e1emb/eta40/st1500", arm="e2e1emb", seed="0", trigger="t_acq",
               state_step="10", event_step="10", detected_step="10", completed_updates="10",
               file="fork.pt", sha256="a"*64, nonfinite=0, next_batch_sha1="b"*40,
               **{k:"c"*64 for k in ("config","source","rules","quick","eval","probe")})
    assert jobs.make_job_specs(row)["disposition"] == "enrolled"
    assert jobs.make_job_specs(dict(row, **{k:1101 for k in
                               ("state_step","event_step","detected_step","completed_updates")}))["disposition"] == "over-limit"
    none = dict(row, **{k:None for k in ("state_step","event_step","detected_step","completed_updates","file","sha256")})
    assert jobs.make_job_specs(none)["disposition"] == "no-acquisition"
    assert jobs.make_job_specs(dict(row, sha256="bad"))["disposition"] == "validation-failed"
    for field in ("cell_id", "arm", "seed", "trigger", "state_step", "event_step",
                  "detected_step", "completed_updates", "file", "sha256"):
        with pytest.raises(ValueError, match="missing required"):
            jobs.make_job_specs({k:v for k,v in row.items() if k != field})
    with pytest.raises(ValueError, match="trigger"):
        jobs.make_job_specs(dict(row, trigger="S1"))
    for cfg in ({}, {"kappa":5}, {"addendum_sha256":"b"*64}):
        with pytest.raises(ValueError, match="missing required"):
            jobs.make_job_specs(dict(row, arm="e2e2emb"), cfg)
    native = dict(row, state_path="fork.pt", state_sha256="a"*64)
    with pytest.raises(ValueError, match="valid"):
        jobs.make_job_specs(native)


def test_frs_drop_file_hash_convention_and_resolution(tmp_path):
    import hashlib
    import json
    drops = [20, 24]
    path = tmp_path/"drops.json"
    # Independently reproduce FRS-A frs_jobs.write_drop_list, not the TRG code path.
    path.write_text(json.dumps(dict(schema=1, drops=drops), sort_keys=True)+"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert tr.realised_drop_hash(drops) == digest
    assert tr.yoked_sham(drops, digest) == [23, 27]
    spec = dict(after=["skip"])
    skip = dict(job_id="skip", disposition="DIVERGED", drops=drops,
                drop_list_sha256=digest, drop_list_file=str(path))
    assert jobs.resolve_yoked_job(spec, skip)["schedule"] == [23, 27]
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="sha256"):
        jobs.resolve_yoked_job(spec, skip)


def test_active_false_alarm_domain_shared_with_calibration():
    at = 185
    steps = np.arange(661)
    acc = np.ones(661)
    for onset in (220, 300, 380, 460):
        acc[onset:onset+3] = .5
    norms = np.ones(661)
    norms[[219, 299, 379, 459, 500, 550, 600]] = 20
    controller = tr.SpikeController(at)
    telemetry = [dict(step=int(s), **controller.observe(int(s), norms[s])) for s in steps if s > at]
    onsets = tr.dip_onsets(steps, acc, at)
    co = tr.active_coincidence(telemetry, onsets, 5)
    active = [r for r in telemetry if r["active"]]
    far = [r for r in active if all(abs(r["step"]-on) > 10 for on in onsets)]
    assert co["hit"] == 1 and co["false_alarm"] == pytest.approx(2/len(far))
    padding = [dict(step=s, active=False, flags={str(k):False for k in tr.KAPPAS})
               for s in list(range(at+1))+list(range(598, 800))]
    assert tr.active_coincidence(padding+telemetry, onsets, 5) == co
    unit = dict(steps=steps, accuracy=acc, t_acq=at, telemetry_steps=steps, norms=norms)
    calibration = tr.select_kappa([unit]*3)
    assert calibration["median_j"]["5"] == pytest.approx(co["hit"]-co["false_alarm"])


def test_endpoints_frs_key_set_complete_and_truncated():
    rows = [dict(step=s,acc=1. if s < 100 or s >= 150 else .5) for s in range(5,306,5)]
    complete = e.endpoints(rows,0,305,require_final=False)
    assert set(complete) == {"disposition","nl2","nl_fine","lost_time","t_loss","t_re",
                            "tau","censored","diverged","retained","episodes","first_recovery",
                            "first_outage","t_div","min_acc"}
    assert complete["first_recovery"] == 150 and complete["first_outage"] == 50
    missing = e.endpoints(rows[:-1],0,305,require_final=False)
    assert set(missing) == set(complete)-{"diverged","min_acc"}
    assert missing["first_recovery"] is missing["first_outage"] is None


def test_reference_exclusive_boundaries():
    assert a.reference_category(74,100) == "EARLY"
    assert a.reference_category(75,100) == "CONCURRENT"
    assert a.reference_category(125,100) == "CONCURRENT"
    assert a.reference_category(126,100) == "DELAYED"
    assert a.reference_category(200,None) == "REF-NL"


def test_paired_onset_difference_censoring_and_missing():
    loss = dict(nl2=False,onset=100,missing=False)
    nl = dict(nl2=True,onset=None,missing=False)
    assert a.paired_onset_difference(loss,nl,10)["difference"] == -210
    assert a.paired_onset_difference(nl,loss,10)["difference"] == 210
    assert a.paired_onset_difference(nl,nl,10)["difference"] == 0
    assert a.paired_onset_difference(dict(nl2=None,missing=True),loss,10)["missing"]


def test_cp_intervals_and_statistical_edge_cases():
    assert a.clopper_pearson(0,8) == pytest.approx([0., 1-.025**(1/8)])
    assert a.clopper_pearson(8,8) == pytest.approx([.025**(1/8), 1.])
    assert a.exact_sign(0,0) == 1. and a.exact_sign(8,0) == 1/256
    assert a.holm2({"FO":.01,"noise":1.}) == {"FO":.02,"noise":1.}


def test_shared_row_api_and_array_adapter_agree():
    rows = [dict(step=s,acc=1. if s < 100 else .5) for s in range(5,306,5)]
    row_result = e.endpoints(rows,0,305,require_final=False)
    arr_result = e.nl2_window([r["step"] for r in rows],[r["acc"] for r in rows],0)
    assert row_result["nl2"] == arr_result["nl2"] == False
    assert row_result["t_loss"] == arr_result["onset"] == 100
    cert = e.Divergence(t_div=20, attempted=20, nonfinite=2)
    assert cert.valid(0)
    assert e.disposition(rows[:3],0,305,certificate=cert,require_final=False) == e.Disposition.DIVERGED
    assert e.classify_branch([5,10,15],[1.,1.,1.],start=0,end=305,t_div=20,
                             certified=True,continuation_steps=20,nonfinite=1) == "TRUNCATED"


def test_sequence_cluster_bootstrap_and_consistency():
    steps = np.arange(61)
    terms = np.zeros((61,3,2,6))
    terms[35:46,...,1] = 2.
    result = a.sequence_cluster_ci(terms, terms.sum(-1), steps, np.full((3,2),6),0,40,draws=32)
    assert result["clusters"] == 3
    assert result["total_ci95"] == pytest.approx([2.,2.])
    units = [dict(family=f,descriptive=dict(harrell=dict(residual=dict(C=.65))))
             for f in ("FO","noise") for _ in range(6)]
    out = a.event_consistency(units)
    assert out["residual"]["consistent"] and not out["write_norm"]["consistent"]
