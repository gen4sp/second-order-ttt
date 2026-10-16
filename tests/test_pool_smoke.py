import copy
from pathlib import Path
import tomllib

import pytest
import torch
from tttmem.pool_determinism import compare, probe
from test_frs import tiny

ROOT = Path(__file__).resolve().parents[1]


def test_determinism_compare_reports_first_loss_difference():
    assert compare([1.,2.,3.],[1.,2.5,3.])==dict(identical=False,first_diff_step=2,max_abs_loss_diff=.5)
    assert compare([1.,2.],[1.,2.])['identical']
    assert not compare([1.,2.],[1.,2.],models_equal=False)['identical']


def test_descriptive_so_seed0_probe_runs_twice(tiny,tmp_path,capsys):
    cfg={**tiny,'determinism_horizon':10,'trigger_bind':0.}
    result=probe(cfg,torch.device('cpu'),tmp_path/'probe')
    assert result['completed'] and result['identical'] and result['first_diff_step'] is None
    assert result['mode'] == 'sequential-single-process'
    import json
    assert json.loads((tmp_path/'probe/result.json').read_text())['mode'] == result['mode']
    assert len(list((tmp_path/'probe').glob('*/mains/*/probe-final.pt')))==2
    assert 'POOL_DETERMINISM ' in capsys.readouterr().out


def test_dispatch_priority_is_ledger_rate_times_horizon():
    from tttmem.exp_frs import _job_cost
    from tttmem.exp_chr import CE
    from tttmem.frs_jobs import JobSpec
    h5=JobSpec('h5','unused','sha',CE,100,metadata=dict(pack='H5',arm='e2e1emb'))
    so=JobSpec('so','unused','sha',CE,100,metadata=dict(pack='TRG',arm='e2e2emb'))
    assert _job_cost(so)>_job_cost(h5)


def test_concurrency_uses_overlapping_training_intervals():
    from tttmem.pool_dispatch import overlap_peak
    def span(a,b):return dict(training_started_at=a,training_finished_at=b)
    assert overlap_peak([span(1,2),span(2,3)])==1
    assert overlap_peak([span(1,3),span(2,4)])==2
    assert overlap_peak([span(i,10) for i in range(5)])==5
