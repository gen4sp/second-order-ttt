"""Regression tests: dead workers, time-budget boundaries, and frozen enrollment."""
import copy
import json
from pathlib import Path
import time

import pytest
import torch

from tttmem import exp_frs as f, exp_chr as c, frs_state as s
from tttmem.frs_jobs import JobSpec, run_queue, job_record, _infrastructure
from tttmem.pool_manifest import freeze
from tttmem.pool_runtime import BudgetExhausted
from test_frs import tiny, setup, start_file
from pool_worker_probe import worker as probe_worker


@pytest.mark.parametrize('mode', ['main', 'job'])
@pytest.mark.parametrize('crash_twice', [False, True])
def test_dead_worker_exit9_retries_respawns_under_30s(tiny, tmp_path, monkeypatch, mode, crash_twice):
    # Exercise the real supervisor and POOL ownership/retry handlers with tiny
    # stdlib-only children, so no extra torch process is started.
    monkeypatch.setattr(f, '_worker', probe_worker)
    cell = tiny['cells'][0]
    key = f.ttt._cell_id(cell, tiny['steps']) if mode == 'main' else 'crash'
    if mode == 'main':
        monkeypatch.setattr(f.ttt, '_cell_id', lambda *a: 'main-crash')
        key = 'main-crash'
    items = ([(key, cell), ('healthy', dict(cell, seed=1))] if mode == 'main' else
             [(key, JobSpec(key, 'unused', 'x', c.CE, 5)),
              ('healthy', JobSpec('healthy', 'unused', 'x', c.CE, 5)),
              ('dependent', JobSpec('dependent', 'unused', 'x', c.CE, 5, after=(key,)))])
    cfg = {**tiny, 'process_name': 'TTM-pool-74', 'workers': 2,
           '_deadline': time.time()+60, 'crash_twice': crash_twice}
    before = time.monotonic()
    out = f._dispatch_units(cfg, mode, items, lambda _: None)
    assert time.monotonic()-before < 30
    assert out[key]['attempts'] == 2
    assert out[key]['status'] == ('MISSING' if crash_twice else 'DONE')
    assert out['healthy']['status'] == 'DONE'
    if mode == 'job':
        assert out['dependent']['status'] == ('MISSING' if crash_twice else 'DONE')
    base = Path(cfg['output_dir'])/('mains' if mode == 'main' else 'jobs/retry')/key
    assert (base/('cell.json' if mode == 'main' else 'result.json')).is_file()


def test_budget_session_1_2_job_resumes_without_retry(tiny, tmp_path):
    ctx, model, opt, rng, row = start_file(tiny, tmp_path)
    class Cut(f.ChrBackend):
        def loss(self, *args, **kwargs):
            result = super().loss(*args, **kwargs)
            ctx._frs_cfg['_deadline'] = 0  # expires between attempted updates
            return result
    job = JobSpec('budget', row['file'], row['sha256'], c.CE, 25)
    assert not _infrastructure(BudgetExhausted('budget exhausted'))
    first = run_queue([job], lambda _: ctx, tmp_path/'jobs', backend=Cut())['budget']
    assert first['resumable'] and first['reason'] == 'budget exhausted' and first['attempts'] == 0
    assert not list((tmp_path/'jobs').glob('**/result.json'))
    assert not (tmp_path/'jobs/retry').exists()
    ctx._frs_cfg['_deadline'] = time.time()+60
    second = run_queue([job], lambda _: ctx, tmp_path/'jobs', resume=True)['budget']
    assert second['status'] == 'DONE' and second['attempts'] == 1


def test_budget_main_resumes_without_partial_or_retry(tiny, tmp_path, monkeypatch):
    original = f.frs_segment
    def cut(*args, **kwargs):
        original(*args, **{**kwargs, 'stop': 5})
        raise BudgetExhausted('pool segment deadline exhausted')
    monkeypatch.setattr(f, 'frs_segment', cut)
    cfg = {**tiny, 'resume': True}
    first = f.run_frs_cell(cfg, cfg['cells'][0], torch.device('cpu'))
    assert first['resumable'] and first['attempts'] == 0
    mains = Path(cfg['output_dir'])/'mains'
    assert not list(mains.glob('**/cell.json')) and not list(mains.glob('*.partial-*'))
    monkeypatch.setattr(f, 'frs_segment', original)
    second = f.run_frs_cell(cfg, cfg['cells'][0], torch.device('cpu'))
    assert second['status'] == 'DONE' and second['attempts'] == 1
    assert not list(mains.glob('*.partial-*'))


def test_budget_cut_on_retry_preserves_remaining_attempt(tiny, tmp_path):
    ctx, model, opt, rng, row = start_file(tiny, tmp_path)
    class CutRetry(f.ChrBackend):
        calls = 0
        def loss(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError('worker I/O error')
            loss = super().loss(*args, **kwargs)
            ctx._frs_cfg['_deadline'] = 0
            return loss
    job = JobSpec('retry-budget', row['file'], row['sha256'], c.CE, 25)
    first = run_queue([job], lambda _: ctx, tmp_path/'jobs', backend=CutRetry())[job.job_id]
    assert first['resumable'] and first['attempts'] == 1
    assert not list((tmp_path/'jobs').glob('**/result.json'))
    ctx._frs_cfg['_deadline'] = time.time()+60
    second = run_queue([job], lambda _: ctx, tmp_path/'jobs', resume=True)[job.job_id]
    assert second['status'] == 'DONE' and second['attempts'] == 2


def manifest(jobs, states=(), cohort=(0,), route=None):
    return dict(schema=1, jobs=[job_record(j) for j in jobs], states=list(states),
                h5_cohort=list(cohort), route_cohorts=route, trg_dispositions=[])


def test_frozen_manifest_recovered_main_cannot_join(tiny, tmp_path):
    root = tmp_path/'pool'; root.mkdir()
    frozen = manifest([], cohort=(0,), route={'t_acq': [dict(seed=0, missing=True)]})
    freeze(root, frozen, [], resume=False)
    original = (root/'job_manifest.json').read_bytes()
    job = JobSpec('recovered', 'unused', 'new-sha', c.CE, 5)
    rebuilt = manifest([job], route={'t_acq': [dict(seed=0, missing=False)]})
    kept, runnable, prior, events = freeze(root, rebuilt, [job], resume=True)
    assert kept == frozen and runnable == [] and prior == {}
    assert events and (root/'job_manifest.json').read_bytes() == original
    assert 'MANIFEST_DRIFT' in (root/'manifest_drift.jsonl').read_text()


def test_frozen_manifest_corrupt_state_keeps_completed_jobs(tiny, tmp_path):
    ctx, model, opt, rng, row = start_file(tiny, tmp_path)
    root = tmp_path/'pool'; root.mkdir()
    job = JobSpec('kept', row['file'], row['sha256'], c.CE, 25)
    initial = manifest([job], [row])
    freeze(root, initial, [job], resume=False)
    before = (root/'job_manifest.json').read_bytes()
    complete = run_queue([job], lambda _: ctx, root/'jobs')['kept']
    Path(row['file']).write_bytes(b'truncated')
    kept, runnable, prior, events = freeze(root, manifest([], [], cohort=()), [], resume=True)
    assert kept == initial and not runnable and events
    assert prior['kept']['final_sha256'] == complete['final_sha256'] and prior['kept']['status'] == 'DONE'
    assert (root/'job_manifest.json').read_bytes() == before


def test_frozen_manifest_comparison_ignores_relocation(tiny, tmp_path):
    from dataclasses import replace
    root = tmp_path/'pool'; root.mkdir()
    job = JobSpec('kept', '/old/state.pt', 'digest', c.CE, 5)
    initial = manifest([job], [dict(file='/old/state.pt', sha256='digest')])
    freeze(root, initial, [job], resume=False)
    moved = replace(job, state_file='/new/state.pt')
    _, jobs, prior, events = freeze(root, manifest([moved], [dict(file='/new/state.pt', sha256='digest')]),
                                    [moved], resume=True)
    assert jobs == [moved] and not prior and not events


def test_pool_resume_corrupt_main_keeps_frozen_job_results(tiny, tmp_path):
    cfg = {**tiny, 'max_minutes': 2, 'trigger_bind': 0., 'trigger_residual': 2.,
           'smoke_force_loss': True, 'h5_limit': 1, 'job_modules': []}
    first = f.run(cfg, 0)
    assert first['complete'] and first['jobs'] == 2
    root = Path(cfg['output_dir'])
    old = json.loads((root/'pool.json').read_text())
    frozen = (root/'job_manifest.json').read_bytes()
    Path(old['states'][0]['file']).write_bytes(b'truncated')
    second = f.run({**cfg, 'resume': True}, 0)
    current = json.loads((root/'pool.json').read_text())
    assert not second['complete'] and current['manifest_drift']
    assert second['manifest_drift'] == len(current['manifest_drift']) > 0
    assert (root/'job_manifest.json').read_bytes() == frozen
    assert current['jobs'].keys() == old['jobs'].keys()
    assert all(r['status'] == 'DONE' for r in current['jobs'].values())


def test_h5_scoring_keeps_recovered_slot_missing():
    from tttmem.frs_cohorts import h5_adapter
    recovered = dict(seed=0, arm='e2e1emb', events={'t_acq': {'event_step':5}}, final={'ce': {'acc':1.}})
    result = h5_adapter([recovered], [], smoke=True, frozen_slots=[dict(seed=0, missing=True)])
    assert result['cohort'] == [0] and result['h5_units'][0]['main_retained'] is None


def test_frozen_manifest_reuses_sham_generated_controller(tiny, tmp_path):
    from test_trg_pool import fixture
    from tttmem.trg_pool import TrgBackend
    cfg, ctx, row, jobs = fixture(tmp_path, 'e2e2emb', 20)
    root = tmp_path/'pool'; root.mkdir()
    frozen = manifest(jobs, [row])
    freeze(root, frozen, jobs, resume=False)
    run_queue(jobs, lambda _: ctx, root/'jobs', backend=TrgBackend())
    Path(row['file']).write_bytes(b'truncated')
    _, runnable, prior, drift = freeze(root, manifest([], [], cohort=()), [], resume=True)
    assert not runnable and drift and all(r['status'] == 'DONE' for r in prior.values())


def test_scheduled_sham_drop_counts_nonfinite_attempt(tiny, tmp_path):
    from tttmem.frs_jobs import ScheduledSkip
    ctx, model, opt, rng = setup(tiny)
    class NonfiniteFirst(f.ChrBackend):
        calls = 0
        def loss(self, *args, **kwargs):
            self.calls += 1
            loss = super().loss(*args, **kwargs)
            return loss*float('nan') if self.calls == 1 else loss
    record = f.frs_segment(model, opt, rng, ctx, c.CE, 'sham', 0,
                           backend=NonfiniteFirst(), skip_controller=ScheduledSkip(None, frozenset({1})),
                           output_dir=tmp_path/'sham')
    assert record['nonfinite'] == 1 and record['drops'] == [1] and not record['diverged']


def test_output_dir_symlink_preserved_through_main_and_resume(tiny, tmp_path, monkeypatch):
    real = tmp_path/'real'; real.mkdir()
    link = tmp_path/'workspace'; link.symlink_to(real, target_is_directory=True)
    cfg = {**tiny, 'output_dir': str(link/'pool'), 'max_minutes': 2, 'trigger_bind': 0.,
           'trigger_residual': 2., 'smoke_force_loss': True, 'h5_limit': 1, 'job_modules': []}
    guard = f.execution_guard
    def checked(value):
        assert value['output_dir'] == cfg['output_dir']
        return guard(value)
    monkeypatch.setattr(f, 'execution_guard', checked)
    assert f.run(cfg, 0)['complete']
    assert f.run({**cfg, 'resume': True}, 0)['complete']
    assert not json.loads((link/'pool/pool.json').read_text())['manifest_drift']


def test_budget_mains_two_sessions_defer_manifest_and_run_recovered_jobs(tiny, tmp_path, monkeypatch, capsys):
    cfg = {**tiny, 'T': 128, 'n_layers': 4, 'window': 16, 'steps': 65, 'chunk_early': 5, 'chunk_late': 6,
           'resume': True, 'max_minutes': 15, 'trigger_bind': 0., 'trigger_residual': 2.,
           'smoke_force_loss': True, 'h5_limit': 1, 'trg_kappa': 3,
           'trg_smoke_horizon': 20, 'trg_smoke_so_stop': 40,
           'coverage_factory': 'tttmem.trg_pool:create_logger',
           'backend_factory': 'tttmem.trg_pool:create_backend',
           'job_modules': ['tttmem.trg_pool', 'tttmem.exp_rdo'],
           'rdo_rpe': dict(candidate_seeds=[0], limit=1, horizon=20, late_horizon=20, late=True),
           'cells': [dict(tiny['cells'][0], arm=arm)
                     for arm in ('e2e1emb', 'e2e2emb', 'e2e1emb_cn97')]}
    original = f.frs_segment
    def cut(*args, **kwargs):
        if args[3].arm == 'e2e1emb' and args[5] == 'main':
            original(*args, **{**kwargs, 'stop': 10})
            args[3]._frs_cfg['_deadline'] = 0
            raise BudgetExhausted('forced session-1 mains deadline')
        return original(*args, **kwargs)
    monkeypatch.setattr(f, 'frs_segment', cut)
    first = f.run(cfg, 0)
    root = Path(cfg['output_dir'])
    pool = json.loads((root/'pool.json').read_text())
    assert not first['complete'] and first['resume_required'] and first['manifest_drift'] == 0
    assert pool['complete'] is False and pool['resume_required'] is True and not pool['jobs']
    assert 'POOL_RESUME_REQUIRED' in capsys.readouterr().out
    assert not (root/'job_manifest.json').exists() and not (root/'jobs').exists()
    fo = next(r for r in pool['cells'].values() if r['arm'] == 'e2e1emb')
    assert fo['resumable'] and fo['status'] == 'MISSING' and not fo.get('error')
    monkeypatch.setattr(f, 'frs_segment', original)
    second = f.run(cfg, 0)
    pool = json.loads((root/'pool.json').read_text())
    manifest = json.loads((root/'job_manifest.json').read_text())
    assert second['complete'] and not second['resume_required'] and second['manifest_drift'] == 0
    assert manifest['h5_slots'] == [dict(seed=0, cell_id=fo['cell_id'], missing=False)]
    expected = {'trg-FO-s0-kvm', 'trg-FO-s0-rnd', 'h5-s0-rp03', 'h5-s0-nc03'}
    assert expected <= pool['jobs'].keys()
    assert all(pool['jobs'][key]['status'] == 'DONE' for key in expected)
    assert all(r['status'] == 'DONE' for r in pool['jobs'].values())
    assert all(not slot.get('missing') for group in manifest['route_cohorts'].values()
               for slot in group if isinstance(slot, dict))
    recovered = pool['cells'][fo['cell_id']]
    assert recovered['status'] == 'DONE' and recovered['attempts'] == 1
    assert not pool['manifest_drift'] and not (root/'manifest_drift.jsonl').exists()


def test_trg_missing_main_disposition_without_error():
    from tttmem.trg_pool import build_jobs
    cfg = dict(smoke_only=True, trg_kappa=3)
    cell = dict(cell_id='missing', seed=0, arm='e2e1emb', status='MISSING', resumable=True)
    assert not build_jobs([], [cell], cfg)
    assert cfg['_trg_dispositions'][0]['disposition'] == 'missing-main'


@pytest.mark.parametrize('mode', ['main', 'job'])
def test_preempted_then_undispatched_keeps_one_retry(tiny, tmp_path, monkeypatch, mode):
    cfg = {**tiny, 'resume': True}
    if mode == 'main':
        cell = cfg['cells'][0]; key = f.ttt._cell_id(cell, cfg['steps'])
        folder = Path(cfg['output_dir'])/'mains'/key.replace('/', '__')
        original = f.frs_segment
        def preempt(*args, **kwargs):
            original(*args, **{**kwargs, 'stop': 5})
            raise KeyboardInterrupt('session preempted')
        monkeypatch.setattr(f, 'frs_segment', preempt)
        with pytest.raises(KeyboardInterrupt):
            f.run_frs_cell(cfg, cell, torch.device('cpu'))
        item = cell
    else:
        ctx, _, _, _, row = start_file(tiny, tmp_path)
        item = JobSpec('preempted', row['file'], row['sha256'], c.CE, 20)
        key = item.job_id; folder = Path(cfg['output_dir'])/'jobs'/key
        class Preempt(f.ChrBackend):
            def loss(self, *args, **kwargs):
                super().loss(*args, **kwargs)
                raise KeyboardInterrupt('session preempted')
        with pytest.raises(KeyboardInterrupt):
            run_queue([item], lambda _: ctx, folder.parent, backend=Preempt())
    before = {p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    assert before and not (folder/'budget.json').exists()
    cut = f._dispatch_units({**cfg, 'workers': 2, '_deadline': 0}, mode, [(key, item)], None)[key]
    assert cut['resumable'] and cut['reason'] == 'budget exhausted'
    assert {p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()} == before
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError('retry fails')
    if mode == 'main':
        monkeypatch.setattr(f, 'frs_segment', fail)
        resume = lambda: f.run_frs_cell(cfg, item, torch.device('cpu'))
    else:
        class Fail(f.ChrBackend):
            loss = staticmethod(fail)
        resume = lambda: run_queue([item], lambda _: ctx, folder.parent, backend=Fail(), resume=True)[key]
    last = resume()
    assert last['status'] == 'MISSING' and last['attempts'] == 2 and calls == 1
    assert resume()['status'] == 'MISSING' and calls == 1
    assert not list(folder.parent.glob('*.budget-*'))
