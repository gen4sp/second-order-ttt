import json

import numpy as np
import pytest

from tttmem.p2 import s14_analysis as a
from tttmem.p2 import s14_checks as checks
from tttmem.p2.s14_protocol import N3B_FO_PILOT_S, PASSKEY_PROTOCOL
from pathlib import Path


def training_files(root, arm, seed, *, ratio=1, speed=1, micro_batch=4):
    d = root/'runs/mini-m'/arm/f's{seed}'
    d.mkdir(parents=True, exist_ok=True)
    n = 2800 if arm == 'mini-p' else 3469 if arm in ('fo-cm', 'wsd-3469') else 1200
    ta = a.N3B_SO_S*1200/n if arm == 'mini-p' else N3B_FO_PILOT_S[micro_batch]
    seconds = np.full(n, ta)
    seconds[60:] = (ratio*a.N3B_SO_S*1200-60*ta)/(n-60)
    seconds *= speed
    (d/'steps.jsonl').write_text('P2_STEP '+json.dumps(dict(step=n, loss=4., grad_norm=.1))+'\n')
    events = ['P2_GUARD '+json.dumps(dict(step=i, finite=True, value=4., trip=False))
              for i in range(50, n+1, 50)]
    for end in range(10, n+10, 10):
        end = min(end, n)
        start = (end-1)//10*10
        events.append('P2_TIME '+json.dumps(dict(step=end, step_s=seconds[start:end].tolist(),
                                                train_s=float(seconds[:end].sum()))))
    (d/'events.jsonl').write_text('\n'.join(events)+'\n')
    (d/'MINI_CELL_DONE.json').write_text(json.dumps(dict(status='done', final=dict(steps=n, aborted=False))))
    (d/'compute.json').write_text(json.dumps(dict(arm=arm, seed=seed, expected_steps=n,
        base_steps=1200, micro_batch=micro_batch, so_micro_batch=2 if arm == 'mini-p' else None,
        pilot_source='s14-pilot' if arm == 'mini-p' else 'n3b-50', pilot_micro_batch=micro_batch,
        pilot_so_micro_batch=2 if arm == 'mini-p' else None,
        t_so_pilot_s=a.N3B_SO_S, t_arm_pilot_s=ta, device_name='test CPU',
        torch_version='test', cuda_version=None, runpod_pod_id=None)))
    return d


def synthetic_store(root, seeds, arms=a.ARMS, *, legacy=False, passkeys=True):
    """Actual registered directories and required NPZ fields, not mock selectors."""
    offsets = {'so': .8, 'fo-ce': .5, 'fo-cm': .55, 'mini-p': .82,
               'wsd-1200': .6, 'wsd-3469': .75}
    rows = np.arange(1024)
    # Give folds opposite M preferences so selecting all rows is detectably wrong.
    fold = a.fold_ids(rows)
    for seed in seeds:
        for arm in arms:
            directory = root/'reads/mini-m'/arm/f's{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            grid = (a.OLD_GRID[arm] if legacy else a.SO_GRID if arm=='so' else a.FO_GRID)
            for e in grid:
                L = 4+.02*(e-1.25)**2 + (0 if arm=='so' else -.01 if arm in checks.EQUAL_COMPUTE else -.005)
                m = offsets[arm]+.02*e-.01*(e-2)**2 + .001*seed
                M = m + (fold*2-1)*.03*(e-1.5)
                N = np.full(len(rows), 3.0+(0 if arm=='so' else .01))
                rank_f = np.tile([1, 2], len(rows))
                rank_nf = np.tile([1, 1], len(rows))
                payload = dict(M_doc=M, fP_focal=M+N, fP_nonfocal=N,
                    L_adapt_sum=np.full(128, L*100), L_adapt_count=np.full(128, 100),
                    clip_bind_fP_focal=np.zeros(len(rows)), eval_rows=rows,
                    kz_rank_f=rank_f, kz_rank_nf=rank_nf,
                    kz_probe_offset=np.arange(len(rows)+1)*2)
                np.savez_compressed(directory/f'g0_read_{a.name(e)}.npz', **payload)
            np.savez(directory/'g0_hlm.npz', heldout_sum=np.full(256, 4.9 if arm in checks.EQUAL_COMPUTE else 5.),
                     heldout_count=np.ones(256))
            if not legacy:
                training_files(root, arm, seed)
            if passkeys:
                frozen = json.loads(Path(a.__file__).with_name('s14_passkey_hosts.json').read_text())
                for eta in a.PASSKEY_ETAS['so' if arm=='so' else 'fo']+[0]:
                    label = a.name(eta) if eta else 'noadapt'
                    values = dict(documents_sha256=np.asarray('a'*64), host_rows=frozen['host_rows'][:256],
                        eta=np.asarray(eta), seed=np.asarray(1014), key_token_count=np.full(256, 2),
                        host_set_sha256=np.asarray(frozen['documents_sha256']),
                        protocol=np.asarray(PASSKEY_PROTOCOL), leak_check=np.asarray('PASS'),
                        leak_check_max_abs_delta_nll=np.asarray(0.),
                        leak_check_per_doc_max_abs_delta_nll=np.zeros(8),
                        leak_check_n_docs=np.asarray(8), leak_check_tolerance=np.asarray(1e-5),
                        leak_check_dtype=np.asarray('fp32'),
                        key=np.full(256, '12345'), foil_keys=np.full((256, 8), '54321'))
                    for pos in ('in_window', 'beyond'):
                        lift = .4 if pos=='in_window' else (.3 if arm=='so' else .1) if eta else (seed%3-1)*.01
                        values.update({pos+'_lift': np.full(256, lift+eta*.001),
                                       pos+'_rank': np.ones(256, int), pos+'_exact': np.zeros(256, bool),
                                       pos+'_nll_true': np.ones(256),
                                       pos+'_nll_foils': np.ones((256,8)),
                                       pos+'_nll_token': np.ones((256,9,2)),
                                       pos+'_query_span': np.tile([10,20,23],(256,1))})
                    np.savez(directory/f'passkey_{label}.npz', **values)
    return a.ReadStore(root, legacy=legacy)


def test_hash_uses_document_id_not_position_and_is_frozen():
    ids = np.arange(12)
    assert a.fold_ids(ids).tolist() == [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    np.testing.assert_array_equal(a.fold_ids(ids[::-1])[::-1], a.fold_ids(ids))


def test_crossfit_reports_other_half_and_truncates_on_fit_half(tmp_path):
    store = synthetic_store(tmp_path, [0], passkeys=False)
    c = store.curve('fo-cm', 0)
    r = a.select_report(c, 4.02, crossfit=True, adaptive=True)
    assert len(r['parts']) == 2
    folds = a.fold_ids(c.rows)
    for f, p in enumerate(r['parts']):
        tr = c.mean_rows(folds==f)
        tr, _ = a.truncate(tr, 4.02)
        s = a.select(tr, 4.02)
        assert p['eta'] == s['eta']
        assert p['M'] == pytest.approx(a.interp(c.mean_rows(folds!=f), 'M', s['eta']))
    assert r['M'] < a.select_report(c, 4.02, crossfit=False, adaptive=True)['M']
    # Stop on the first extended point whose CAL budget is exceeded.
    rows = [dict(eta=e, M=e, L=0 if e<=6 else 1) for e in (1, 6, 7, 8, 10, 12)]
    truncated, stop = a.truncate(rows, .5)
    assert stop == 'budget' and truncated[-1]['eta'] == 7
    rows = [dict(eta=e, M=m, L=0) for e, m in [(1, 1), (6, 6), (7, 5), (8, 4), (10, 7)]]
    assert a.truncate(rows, 1)[0][-1]['eta'] == 8


def test_parity_with_unchanged_n3b_reference_for_synthetic_curves(tmp_path):
    from tttmem.p2 import s14_n3b_reference as reference
    store = synthetic_store(tmp_path, [0], arms=a.OLD_GRID, legacy=True, passkeys=False)
    curves = {arm: store.curve(arm, 0) for arm in a.OLD_GRID}
    # New files can have the dense grid; reference uses exactly their eta rows.
    for arm, c in curves.items():
        rows = c.mean_rows(np.ones(len(c.rows), bool))
        for b in (4.01, 4.02, 4.04):
            theirs = reference.mstar([{**r, 'clip': 0.} for r in rows], b)
            ours = a.select(rows, b)
            for key in ('M', 'eta', 'F', 'N', 'Lt', 'flag'):
                assert ours[key] == theirs[key]


def test_bounds_seed_unit_labels_and_fail_closed_missing_values():
    b = a.bounds([.171, .081, .156])
    assert b['lb90'] == pytest.approx(.08350711758723856, abs=1e-12)
    assert a.bounds(range(6))['t90'] == 1.4759
    assert a.cover_label(b, a.bounds([0, 0, 0])).startswith('NOT-COVERED')
    assert a.cover_label(a.bounds([.01]*3), a.bounds([.1]*3)) == 'COVERS-M-ONLY'
    assert a.cover_label(a.bounds([np.nan, .01, .01]), a.bounds([0]*3)) == 'NOT-EVALUABLE'


@pytest.mark.parametrize('layout', ['R', 'S'])
def test_exact_s14_layout_all_contrasts_pooled_and_old_seed_comparison(tmp_path, layout):
    old = synthetic_store(tmp_path/'old', [0, 1, 2], arms=a.OLD_GRID, legacy=True, passkeys=False)
    seeds = [3, 4, 5] if layout=='R' else [0, 1, 2]
    synthetic_store(tmp_path/'new', seeds,
                    arms=a.ARMS if layout=='R' else ('mini-p', 'wsd-1200', 'wsd-3469'))
    manifest = tmp_path/'eval.jsonl'
    manifest.write_text(''.join(json.dumps(dict(row=i, near_dup=i==9))+'\n' for i in range(1024)))
    result = a.analyze_s14(tmp_path/'new', tmp_path/'out', layout=layout, legacy_store=old, manifest=manifest)
    for variant, modes in result['versions'].items():
        for mode in ('in_sample', 'crossfit'):
            v = modes[mode]
            assert v['aggregate']['x_P']['n'] == 3
            assert v['decisions']['MINI_P_above_FO_CM']
            for b in v['per_seed']:
                s = b['selected']
                assert b['delta_steps'] == pytest.approx(s['wsd-3469']['M']-s['wsd-1200']['M'])
                assert b['delta_shape'] == pytest.approx(s['wsd-1200']['M']-s['fo-ce']['M'])
            if layout=='R':
                assert v['pooled_six']['R_cov_fo-cm']['n'] == 6
                assert v['pooled_six']['R_cov_fo-cm']['t90'] == 1.4759
                if variant=='all':
                    assert v['passkey']['valid']
                    assert v['passkey']['contrast']['mean'] == pytest.approx(.2, abs=.001)
                    assert v['passkey']['reading'] == 'SO higher (descriptive)'
            else:
                assert v['pooled_six'] is None
        assert modes['top1'][0]['top1_nf'] == 1
        assert modes['top1'][0]['top1_f'] == .5
    assert (tmp_path/'out/s14-analysis.md').exists()
    json.loads((tmp_path/'out/s14_analysis.json').read_text(), parse_constant=lambda x: pytest.fail(x))


def test_missing_grid_alignment_and_passkey_validity_fail_closed(tmp_path):
    store = synthetic_store(tmp_path, [3, 4, 5])
    blocks = [a.seed_block({arm: store.curve(arm, s) for arm in a.ARMS}, s, crossfit=True) for s in (3, 4, 5)]
    for b in blocks:
        for p in b['selected']['fo-cm']['parts']:
            p['eta'] = 11
    r = a.passkey_analysis(store, blocks)
    assert r['contrast'] is None and r['nearest_grid_proxy'] is not None
    assert r['reading'] == 'NOT-EVALUABLE'
    for s in (3, 4, 5):
        p = store.directory('fo-ce', s)/'passkey_ce.npz'
        with np.load(p) as z:
            payload = {k: z[k] for k in z.files}
        payload['in_window_lift'] = np.full(256, -100.)
        payload['in_window_rank'] = np.full(256, 9)
        np.savez(p, **payload)
    assert not a.passkey_analysis(store, blocks)['valid']
    p.unlink()
    store.path('so', 3, 1).unlink()
    with pytest.raises(FileNotFoundError):
        store.curve('so', 3)


def test_partial_document_set_and_cross_arm_misalignment_fail_closed(tmp_path):
    store = synthetic_store(tmp_path, [0], passkeys=False)
    p = store.path('so', 0, 1)
    with np.load(p) as z:
        payload = {k: z[k] for k in z.files}
    payload['eval_rows'] = payload['eval_rows'][::-1]
    np.savez(p, **payload)
    with pytest.raises(ValueError, match='alignment'):
        store.curve('so', 0)
    payload['eval_rows'] = np.arange(100)
    # Make the first point partial, so the explicit 1024-doc guard fires.
    np.savez(store.path('so', 0, .75), **payload)
    with pytest.raises(ValueError, match='1024'):
        store.curve('so', 0)


def test_weighted_folds_equal_in_sample_if_eta_agrees(tmp_path):
    store = synthetic_store(tmp_path, [0], arms=['so'], passkeys=False)
    c = store.curve('so', 0)
    # Both folds increase monotonically in eta, with distinct fold offsets.
    c.M[:] = c.eta[:, None] + (a.fold_ids(c.rows)*10)[None, :]
    c.F[:] = c.M+c.N
    b = 4.01
    cross = a.select_report(c, b, crossfit=True)
    full = a.select_report(c, b, crossfit=False)
    assert cross['parts'][0]['eta'] == cross['parts'][1]['eta']
    assert cross['parts'][0]['report_n'] != cross['parts'][1]['report_n']
    for k in ('M', 'F', 'N', 'Lt'):
        assert cross[k] == pytest.approx(full[k], abs=1e-12)
    np.testing.assert_allclose(a.utility_curve(c, 4, b, crossfit=True),
                               a.utility_curve(c, 4, b, crossfit=False), atol=1e-12)


def test_registered_grids_and_no_optional_fo16(tmp_path):
    assert a.SO_GRID == [.75,.9,1,1.1,1.25,1.4,1.5,1.75,2,2.25,2.5,3,3.5,4]
    assert a.FO_GRID == [.5,.75,1,1.25,1.5,2,2.5,2.75,3,3.5,4,4.5,5,5.5,6,7,8,10,12]
    store = synthetic_store(tmp_path, [0], arms=['fo-cm'], passkeys=False)
    np.savez(store.directory('fo-cm', 0)/'g0_read_eta1600.npz', garbage=[1])
    assert store.curve('fo-cm', 0).eta[-1] == 12
    store.path('fo-cm', 0, .5).unlink()
    with pytest.raises(FileNotFoundError):
        store.curve('fo-cm', 0)


def test_edges_open_top_and_lambda_censoring(tmp_path):
    st = synthetic_store(tmp_path, [0], arms=['so', 'fo-cm'], passkeys=False)
    cs = {arm: st.curve(arm, 0) for arm in ('so', 'fo-cm')}
    cs['so'].L[:] = 4
    cs['fo-cm'].L[:] = 3.99
    for c in cs.values():
        c.M[:] = c.eta[:, None]
    b = a.seed_block(cs, 0, crossfit=True)
    assert 'so:floor-edge:blocking' in b['flags']
    assert 'fo-cm:top-open:NOT-COVERED-blocked' in b['flags']
    assert b['lambda_star_fo-cm_label'] == 'upper bound'
    agg = a.aggregate([b, {**b, 'seed': 1}, {**b, 'seed': 2}])
    assert agg['labels']['fo-cm'].startswith('NOT-EVALUABLE')
    for c in cs.values():
        c.L[:] = (c.eta-.75)**2+4
    b = a.seed_block(cs, 0)
    assert 'fo-cm:floor-edge:resolved' in b['flags']
    cs['so'].L[:] = (cs['so'].eta-3)**2+4
    assert 'so:floor-edge:resolved' in a.seed_block(cs, 0)['flags']
    # An SO floor at either end of its actual grid is blocking.
    cs['so'].L[:] = (cs['so'].eta-4)**2+4
    assert 'so:floor-edge:blocking' in a.seed_block(cs, 0)['flags']
    for c in cs.values():
        c.L[:] = (c.eta-1)**2+4
    cs['fo-cm'].L -= .001
    cs['so'].M[:] = 0
    cs['fo-cm'].M[:] = 1
    b = a.seed_block(cs, 0)
    assert b['lambda_star_fo-cm'] == '>2'
    agg = a.aggregate([b]*3)
    assert agg['lambda_star_fo-cm']['mean'] == '>2'
    assert agg['lambda_star_fo-cm']['censored_seeds'] == 3
    assert '3 censored' in a.table(agg, ['lambda_star_fo-cm'])
    mixed = a.aggregate([b, b, {**b, 'lambda_star_fo-cm': 1.}])
    assert mixed['lambda_star_fo-cm']['mean'] == 'not estimated (censored)'
    assert mixed['lambda_star_fo-cm']['censored_seeds'] == 2


@pytest.mark.parametrize('ratio,notcovered,covers', [(.94, True, False), (.95, False, False),
                                                    (1.05, False, False), (1.06, False, True)])
def test_speed_normalized_compute_signed_bounds(tmp_path, ratio, notcovered, covers):
    training_files(tmp_path, 'fo-cm', 3, ratio=ratio)
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['status'] == 'OK', r
    assert r['compute']['r_act_over_r'] == pytest.approx(ratio)
    assert r['compute']['blocks_not_covered'] == notcovered
    assert r['compute']['blocks_covers'] == covers
    # Apply each signed block only to the corresponding label direction.
    for label, blocked in [('NOT-COVERED (large)', notcovered), ('COVERS', covers)]:
        agg = dict(labels={'fo-cm': label})
        checks.apply_checks(agg, [dict(checks={'so': dict(status='OK'), 'fo-ce': dict(status='OK'), 'fo-cm': r})])
        assert agg['labels']['fo-cm'].startswith('NOT-EVALUABLE') == blocked


@pytest.mark.parametrize('failure', ['missing_compute', 'missing_events', 'missing_guard', 'timing_gap',
                                    'bad_sum', 'trip', 'nonfinite', 'marker', 'gdiv', 'recomputed'])
def test_training_evidence_fails_closed(tmp_path, failure):
    d = training_files(tmp_path, 'fo-cm', 3)
    if failure.startswith('missing_') and failure != 'missing_guard':
        (d/('compute.json' if failure=='missing_compute' else 'events.jsonl')).unlink()
    elif failure == 'marker':
        (d/'guard_abort.json').write_text('{}')
    elif failure == 'gdiv':
        (d/'gdiv.json').write_text('[{"event": 1}]')
    else:
        path = d/'events.jsonl'
        content = path.read_text()
        if failure == 'missing_guard':
            content = '\n'.join(line for line in content.splitlines() if not line.startswith('P2_GUARD '))
        elif failure == 'timing_gap':
            content = '\n'.join(line for line in content.splitlines() if not line.startswith('P2_TIME {"step": 10,'))
        elif failure == 'bad_sum':
            content = content.replace('"train_s": ', '"train_s": 999999, "ignored": ')
        elif failure == 'trip':
            content = content.replace('"trip": false', '"trip": true', 1)
        elif failure == 'nonfinite':
            content = content.replace('"value": 4.0', '"value": "nan"', 1)
        else:
            lines = content.splitlines()
            for i in (1, 2):
                lines[i] = lines[i].replace('"value": 4.0', '"value": 4.2')
            content = '\n'.join(lines)
        path.write_text(content)
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['status'] == ('REPAIR' if failure in ('trip', 'nonfinite', 'marker', 'gdiv', 'recomputed') else 'NOT-EVALUABLE'), r


def test_pod_missing_compute_and_passkey_mismatch_still_writes_frontier(tmp_path):
    old = synthetic_store(tmp_path/'old', [0, 1, 2], arms=a.OLD_GRID, legacy=True, passkeys=False)
    st = synthetic_store(tmp_path/'new', [3, 4, 5])
    for arm in checks.EQUAL_COMPUTE:
        (tmp_path/'new/runs/mini-m'/arm/'s3/compute.json').unlink()
    p = st.directory('so', 3)/'passkey_ce.npz'
    with np.load(p) as z:
        payload = dict(z)
    payload['host_rows'] = np.arange(256)
    np.savez(p, **payload)
    manifest = tmp_path/'eval.jsonl'; manifest.write_text('{}\n'*1024)
    result = a.analyze_s14(tmp_path/'new', tmp_path/'out', legacy_store=old, manifest=manifest)
    v = result['versions']['all']['crossfit']
    assert all(v['aggregate']['labels'][arm].startswith('NOT-EVALUABLE') for arm in checks.EQUAL_COMPUTE)
    assert v['passkey']['reading'] == 'NOT-EVALUABLE'
    assert 'host ids/hash mismatch' in v['passkey']['error']
    assert (tmp_path/'out/s14_analysis.json').exists() and (tmp_path/'out/s14-analysis.md').exists()


@pytest.mark.parametrize('what', ['floor', 'hlm', 'missing_hlm'])
def test_schedule_regression_and_missing_hlm_no_reading(tmp_path, what):
    st = synthetic_store(tmp_path, [3], arms=['so', 'fo-ce', 'fo-cm'], passkeys=False)
    cs = {arm: st.curve(arm, 3) for arm in ('so', 'fo-ce', 'fo-cm')}
    if what == 'floor':
        cs['fo-cm'].L[:] = cs['fo-ce'].L
    elif what == 'hlm':
        np.savez(st.directory('fo-cm', 3)/'g0_hlm.npz', heldout_sum=np.full(256, 5.), heldout_count=np.ones(256))
    else:
        (st.directory('fo-cm', 3)/'g0_hlm.npz').unlink()
    b = a.seed_block(cs, 3)
    a.cell_checks(b, cs, {arm: st for arm in cs})
    assert b['checks']['fo-cm']['status'] == ('NOT-EVALUABLE' if what == 'missing_hlm' else 'REPAIR')
    if what == 'missing_hlm':
        assert b['H_LM']['fo-cm']['status'] == 'NOT-EVALUABLE'


def test_passkey_primary_subtracts_paired_noadapt_and_gate_is_eta1(tmp_path):
    st = synthetic_store(tmp_path, [3, 4, 5])
    blocks = [a.seed_block({arm: st.curve(arm, s) for arm in a.ARMS}, s, crossfit=True) for s in (3, 4, 5)]
    for b in blocks:
        for arm, etas in (('so', [1.25, 2.25]), ('fo-cm', [3.5, 7.5])):
            for p, eta, count in zip(b['selected'][arm]['parts'], etas, [100, 924]):
                p.update(eta=eta, report_n=count)
    r = a.passkey_analysis(st, blocks)
    expected = .2+.001*(np.average([1.25, 2.25], weights=[100, 924])
                        -np.average([3.5, 7.5], weights=[100, 924]))
    assert r['contrast']['mean'] == pytest.approx(expected, abs=1e-12)
    # Validity is at x1, regardless of other eta's in-window results; FO-CM is descriptive.
    for s in (3, 4, 5):
        for arm, tag in [('so', 'eta200'), ('fo-cm', 'ce')]:
            p = st.directory(arm, s)/f'passkey_{tag}.npz'
            with np.load(p) as z:
                payload = dict(z)
            payload['in_window_lift'] = np.full(256, -100.)
            payload['in_window_rank'] = np.full(256, 9)
            np.savez(p, **payload)
    assert a.passkey_analysis(st, blocks)['valid']
    # SO has higher raw lift, but that difference is entirely noadapt, not adaptation.
    for s in (3, 4, 5):
        for p in st.directory('so', s).glob('passkey_*.npz'):
            with np.load(p) as z:
                payload = dict(z)
            payload['beyond_lift'] += .2
            np.savez(p, **payload)
    r = a.passkey_analysis(st, blocks)
    assert r['raw_lift_contrast']['mean']-r['contrast']['mean'] == pytest.approx(.2)
    assert r['validity']['so']['eta'] == 1
    assert r['no_leak']['so']['status'] == 'PASS'
    assert r['no_leak']['so']['descriptive_noadapt_lift']['lb95'] > 0
    assert r['no_leak']['so']['descriptive_noadapt_lift']['unit'] == 'per-doc mean over seeds'
    assert r['reading'] == 'SO higher (descriptive)'
    # Removing a registered eta is missing evidence, even if working point is inside remaining reads.
    (st.directory('fo-cm', 3)/'passkey_eta800.npz').unlink()
    assert a.passkey_analysis(st, blocks)['reading'] == 'NOT-EVALUABLE'


def test_compute_v2_uniform_pod_speed_and_actual_microbatch_reference(tmp_path):
    training_files(tmp_path, 'fo-cm', 3, speed=2.7)
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['status'] == 'OK' and r['compute']['status'] == 'PASS', r
    assert r['compute']['speed_factor'] == pytest.approx(2.7)
    assert r['compute']['r_act_over_r'] == pytest.approx(1.)
    assert not r['compute']['blocks_covers']
    training_files(tmp_path, 'wsd-3469', 3, micro_batch=2)
    r = checks.training_check(tmp_path, 'wsd-3469', 3)
    assert r['status'] == 'OK', r
    assert r['compute']['planned_ratio'] == pytest.approx(1.065061207162312, abs=1e-12)
    assert r['compute']['blocks_covers'] and not r['compute']['blocks_not_covered']
    for label, blocked in [('COVERS', True), ('NOT-COVERED', False)]:
        agg = dict(labels={'wsd-3469': label})
        checks.apply_checks(agg, [dict(checks={'so': {'status': 'OK'}, 'fo-ce': {'status': 'OK'}, 'wsd-3469': r})])
        assert agg['labels']['wsd-3469'].startswith('NOT-EVALUABLE') == blocked


@pytest.mark.parametrize('arm,field,value', [('fo-cm', 'micro_batch', 2),
    ('mini-p', 'so_micro_batch', 1), ('fo-cm', 'pilot_source', 's14-pilot'),
    ('wsd-3469', 'pilot_micro_batch', 2), ('fo-cm', 'device_name', None)])
def test_compute_v2_mismatch_blocks_both_directions(tmp_path, arm, field, value):
    d = training_files(tmp_path, arm, 3)
    p = d/'compute.json'
    c = json.loads(p.read_text()); c[field] = value; p.write_text(json.dumps(c))
    r = checks.training_check(tmp_path, arm, 3)
    assert r['status'] == 'NOT-EVALUABLE' and r['compute']['status'] == 'FAIL'
    for label in ('COVERS', 'NOT-COVERED'):
        agg = dict(labels={arm: label})
        checks.apply_checks(agg, [dict(checks={'so': {'status': 'OK'}, 'fo-ce': {'status': 'OK'}, arm: r})])
        assert agg['labels'][arm].startswith('NOT-EVALUABLE')


@pytest.mark.parametrize('gdiv,abort,status', [([{'event': 1, 'step': 100}], False, 'OK'),
    ([{'event': 1, 'step': 100}, {'event': 2, 'step': 150}], False, 'REPAIR'),
    ([{'event': 1, 'step': 100}], True, 'REPAIR')])
def test_one_recovered_gdiv_flag_second_or_unrecovered_repairs(tmp_path, gdiv, abort, status):
    d = training_files(tmp_path, 'fo-cm', 3)
    (d/'gdiv.json').write_text(json.dumps(gdiv))
    # The same event in the marker and log counts only once.
    with (d/'events.jsonl').open('a') as f:
        f.write('P2_GDIV '+json.dumps(dict(**gdiv[0], action='reload_last_resume_state'))+'\n')
    if abort:
        p = d/'MINI_CELL_DONE.json'; c = json.loads(p.read_text())
        c['final']['aborted'] = True; p.write_text(json.dumps(c))
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['status'] == status, r
    assert r['gdiv']['status'] == ('FLAG' if status == 'OK' else 'REPAIR')
    if status == 'OK':
        assert r['flags']


def test_unrecovered_gdiv_without_done_is_repair(tmp_path):
    d = training_files(tmp_path, 'fo-cm', 3)
    (d/'gdiv.json').write_text(json.dumps([dict(event=1, step=100)]))
    (d/'MINI_CELL_DONE.json').unlink()
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['status'] == 'REPAIR' and not r['gdiv']['recovered']


def test_old_compute_sidecar_is_rejected_by_v2(tmp_path):
    d = training_files(tmp_path, 'fo-cm', 3)
    (d/'compute.json').write_text(json.dumps(dict(arm='fo-cm', seed=3, base_steps=1200,
        expected_steps=3469, t_so_pilot_s=a.N3B_SO_S, t_arm_pilot_s=a.N3B_FO_S)))
    r = checks.training_check(tmp_path, 'fo-cm', 3)
    assert r['compute']['status'] == 'FAIL' and r['status'] == 'NOT-EVALUABLE'
    assert 'v2 missing fields' in r['error']


@pytest.mark.parametrize('missing_arm', ['wsd-1200', 'fo-cm', 'so'])
def test_missing_cell_keeps_outputs_and_unrelated_decisions(tmp_path, missing_arm):
    old = synthetic_store(tmp_path/'old', [0, 1, 2], arms=a.OLD_GRID, legacy=True, passkeys=False)
    st = synthetic_store(tmp_path/'new', [3, 4, 5], passkeys=False)
    st.path(missing_arm, 3, 1).unlink()
    manifest = tmp_path/'eval.jsonl'; manifest.write_text('{}\n'*1024)
    r = a.analyze_s14(tmp_path/'new', tmp_path/'out', legacy_store=old, manifest=manifest)
    v = r['versions']['all']['crossfit']
    assert v['per_seed'][0]['cells'][missing_arm]['status'] == 'NOT-EVALUABLE'
    assert (tmp_path/'out/s14-analysis.md').is_file() and (tmp_path/'out/s14_analysis.json').is_file()
    if missing_arm == 'wsd-1200':
        assert v['decisions']['R1'] == 'REPLICATES'
        assert v['decisions']['contrast_status']['x'] == 'OK'
        assert v['decisions']['contrast_status']['delta_steps'].startswith('NOT-EVALUABLE')
        assert v['aggregate']['labels']['wsd-3469'] == 'NOT-COVERED'
    else:
        assert v['decisions']['R1'] == 'NOT-EVALUABLE'
        assert v['pooled_six']['R_cov_fo-cm']['status'] == 'NOT-EVALUABLE'
    assert {c['arm'] for c in v['decisions']['required_cells']['R1']} == {'so', 'fo-ce', 'fo-cm'}


def test_descriptive_passkey_arms_and_foce_support_cannot_void_primary(tmp_path):
    st = synthetic_store(tmp_path, [3, 4, 5])
    blocks = [a.seed_block({arm: st.curve(arm, s) for arm in a.ARMS}, s, crossfit=True) for s in (3, 4, 5)]
    for b in blocks:
        for arm in ('mini-p', 'wsd-1200', 'wsd-3469', 'fo-ce'):
            for p in b['selected'][arm]['parts']:
                p['eta'] = 11
    # Missing/descriptive failures, including a detected leak, cannot void SO-CM.
    (st.directory('mini-p', 3)/'passkey_ce.npz').unlink()
    (st.directory('fo-ce', 3)/'passkey_eta700.npz').unlink()
    p = st.directory('wsd-1200', 3)/'passkey_noadapt.npz'
    with np.load(p) as z:
        payload = dict(z)
    payload.update(leak_check=np.asarray('FAIL'), leak_check_max_abs_delta_nll=np.asarray(.1),
                   leak_check_per_doc_max_abs_delta_nll=np.full(8, .1))
    np.savez(p, **payload)
    r = a.passkey_analysis(st, blocks)
    assert r['reading'] == 'SO higher (descriptive)' and r['valid']
    assert r['no_leak']['wsd-1200']['status'] == 'LEAK'
    # Required checkpoint FAIL overrides its passkey reading, with finite NLL.
    p = st.directory('fo-cm', 3)/'passkey_noadapt.npz'
    with np.load(p) as z:
        payload = dict(z)
    payload.update(leak_check=np.asarray('FAIL'), leak_check_max_abs_delta_nll=np.asarray(.1),
                   leak_check_per_doc_max_abs_delta_nll=np.full(8, .1))
    np.savez(p, **payload)
    assert a.passkey_analysis(st, blocks)['reading'] == 'LEAK'


def test_so_open_top_does_not_block_contrasts_and_mini_objective_is_descriptive(tmp_path):
    st = synthetic_store(tmp_path, [3], passkeys=False)
    cs = {arm: st.curve(arm, 3) for arm in a.ARMS}
    cs['mini-p'].L[:] = cs['fo-ce'].L+1
    b = a.seed_block(cs, 3)
    a.cell_checks(b, cs, {arm: st for arm in a.ARMS})
    assert b['checks']['mini-p']['status'] == 'OK'
    assert b['checks']['mini-p']['regression']['descriptive_only']
    assert b['checks']['mini-p']['regression']['CAL_floor_not_lower']
    b['flags'] = ['so:top-open:COVERS-blocked']
    for arms in a.CONTRAST_ARMS.values():
        assert a.ready([b], arms)
    assert not a.ready([b], ('so', 'fo-cm'), uses_so_max=True)


def test_fo_open_top_blocks_only_not_covered(tmp_path):
    st = synthetic_store(tmp_path, [0], arms=['so', 'fo-cm'], passkeys=False)
    cs = {arm: st.curve(arm, 0) for arm in ('so', 'fo-cm')}
    cs['fo-cm'].L[:] = 3.99
    cs['fo-cm'].M[:] = 2+cs['fo-cm'].eta[:, None]
    b = a.seed_block(cs, 0)
    assert 'fo-cm:top-open:NOT-COVERED-blocked' in b['flags']
    assert a.aggregate([b]*3)['labels']['fo-cm'] == 'COVERS'
    cs['so'].M[:] = 20
    b = a.seed_block(cs, 0)
    assert a.aggregate([b]*3)['labels']['fo-cm'] == 'NOT-EVALUABLE (FO top)'


def test_passkey_registered_fo_grid_has_seven():
    assert a.PASSKEY_ETAS == {'so': [1, 1.5, 2, 2.5], 'fo': [1, 2, 3, 4, 5, 6, 7, 8, 10]}
