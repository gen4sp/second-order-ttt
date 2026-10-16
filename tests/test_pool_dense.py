import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tttmem import dense_log as dense
from tttmem.pool_dense import PoolDenseLogger
from test_frs import tiny


@pytest.fixture(autouse=True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # job continuations default to ./checkpoints


def row(value):
    return dict(ladder=True, summary=dict(acc=1.), events=dict(memory_norm=np.array([[value]]),
                write_norm=np.array([[value]]), sigma=np.array([[value]]),
                query_ce=np.array([[value]]), margin=np.array([[value]]),
                probability=np.array([[.9]]), correct=np.array([[1]], dtype=np.int8)))


def test_pool_dense_large_finite_norm_float32_readers(tmp_path):
    logger = PoolDenseLogger(tmp_path/'new', emit=False)
    logger.append(1, row(1e6)); manifest = logger.close()
    dense.validate_shards(tmp_path/'new')
    with np.load(tmp_path/'new'/manifest['shards'][0]['file']) as data:
        for key in ('memory_norm', 'write_norm', 'sigma', 'query_ce', 'margin'):
            assert data[key].dtype == np.float32 and data[key].item() == 1e6
        assert data['probability'].dtype == np.float16
    # The calibrated analyzer's own loader must accept both native and POOL NPZ.
    path = Path(__file__).resolve().parents[1]/'scripts/analyze_trg.py'
    spec = importlib.util.spec_from_file_location('native_analyzer_probe', path)
    analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
    assert analyzer.d.validate_shards(tmp_path/'new')[1] == [1]
    old = dense.DenseLogger(tmp_path/'old', emit=False)
    old.append(1, row(1.)); old.close()
    assert analyzer.d.validate_shards(tmp_path/'old')[1] == [1]


@pytest.mark.parametrize('value', [np.inf, np.nan])
def test_pool_dense_nonfinite_event_is_flagged_and_persisted(tmp_path, value):
    logger = PoolDenseLogger(tmp_path, emit=False)
    read = row(value); read['summary']['ce'] = value
    logger.append(1, read)
    manifest = logger.close()
    summary = json.loads((tmp_path/'summaries.jsonl').read_text())
    assert summary['telemetry_nonfinite'] and summary['ce'] is None
    with np.load(tmp_path/manifest['shards'][0]['file']) as data:
        assert data['telemetry_nonfinite'].item()
        assert data['memory_norm__nonfinite'].item() and np.isnan(data['memory_norm']).all()
    dense.validate_shards(tmp_path)


def force_nonfinite_reads(monkeypatch):
    for name in ('light_read', 'ladder_log'):
        original = getattr(dense, name)
        def bad(*args, _original=original, **kwargs):
            read = _original(*args, **kwargs)
            read['events']['memory_norm'] = np.full_like(read['events']['query_ce'], np.inf)
            read['summary']['ce'] = float('nan')
            return read
        monkeypatch.setattr(dense, name, bad)


@pytest.mark.parametrize('training_nonfinite', [False, True])
def test_main_nonfinite_event_keeps_training_and_divergence_rule(tiny, tmp_path, monkeypatch, training_nonfinite):
    import torch
    from tttmem import exp_frs as f
    force_nonfinite_reads(monkeypatch)
    if training_nonfinite:
        original = f.ChrBackend.loss
        calls = 0
        def loss(*args, **kwargs):
            nonlocal calls
            calls += 1
            value = original(*args, **kwargs)
            return value*float('nan') if calls in (8, 9) else value
        monkeypatch.setattr(f.ChrBackend, 'loss', loss)
    cfg = {**tiny, 'T': 128, 'n_layers': 4, 'window': 16, 'chunk_early': 5, 'chunk_late': 6,
           'trigger_bind': 0., 'trg_kappa': 3,
           'coverage_factory': 'tttmem.trg_pool:create_logger'}
    record = f.run_frs_cell(cfg, cfg['cells'][0], torch.device('cpu'))
    assert record['status'] == 'DONE' and not record.get('error') and record['attempts'] == 1
    assert record['disposition'] == ('DIVERGED' if training_nonfinite else 'COMPLETE')
    assert record['nonfinite'] == (2 if training_nonfinite else 0)
    assert record['attempted'] == (20 if training_nonfinite else cfg['steps'])
    summary = Path(record['logger_final']['shard_manifest']).with_name('summaries.jsonl')
    rows = [json.loads(line) for line in summary.read_text().splitlines()]
    assert rows and all(r['telemetry_nonfinite'] and r['ce'] is None for r in rows)
    dense.validate_shards(summary.parent)


def test_trg_job_nonfinite_event_keeps_training(tmp_path, monkeypatch):
    from test_trg_pool import fixture
    from tttmem.trg_pool import TrgBackend
    from tttmem.frs_jobs import run_queue
    cfg, ctx, _, jobs = fixture(tmp_path, 'e2e1emb', 20)
    force_nonfinite_reads(monkeypatch)
    records = run_queue(jobs, lambda _: ctx, tmp_path/'jobs', backend=TrgBackend())
    assert all(r['status'] == 'DONE' and not r.get('error') and r['attempts'] == 1
               and r['nonfinite'] == 0 for r in records.values())
    for rec in records.values():
        folder = Path(rec['logger_final']['shard_manifest']).parent
        assert all(json.loads(line)['telemetry_nonfinite']
                   for line in (folder/'summaries.jsonl').read_text().splitlines())
        dense.validate_shards(folder)
