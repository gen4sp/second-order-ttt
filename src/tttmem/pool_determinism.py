"""Descriptive SO repeat probe; it never gates a smoke or changes its cohorts."""
import json
from pathlib import Path

import torch
from . import exp_frs as frs, exp_chr as chr
from . import frs_state as state


def compare(losses_a, losses_b, *, models_equal=True):
    pairs = list(zip(losses_a, losses_b))
    diffs = [abs(a-b) for a,b in pairs]
    first = next((i+1 for i,(a,b) in enumerate(pairs) if a != b), None)
    if first is None and (len(losses_a) != len(losses_b) or not models_equal):
        first = min(len(losses_a), len(losses_b))+1 if len(losses_a) != len(losses_b) else len(pairs)
    return dict(identical=first is None, first_diff_step=first, max_abs_loss_diff=max(diffs, default=0.))


def probe(cfg, dev, output):
    horizon = int(cfg.get('determinism_horizon', 20))
    if not cfg.get('smoke_only') or not 5 <= horizon <= min(80, cfg['steps']) or horizon % 5:
        raise ValueError('determinism probe requires an aligned legacy smoke horizon <=80')
    cell = dict(task='echo', seed=0, arm='e2e2emb', eta_base=40)
    records, losses = [], []
    # All global RNGs are restored, so this descriptive probe cannot perturb jobs.
    with state.preserve_rngs():
        for attempt in ('first', 'second'):
            local = {**cfg, 'cells': [cell], 'steps': horizon, 'resume': False,
                     'output_dir': str(Path(output)/attempt), '_determinism_probe': True}
            record = frs.run_frs_cell(local, cell, dev)
            records.append(record)
            if record.get('status') != 'DONE':
                result = dict(identical=False, first_diff_step=None, max_abs_loss_diff=None,
                              completed=False, reason=record.get('reason', record.get('error')))
                break
            folder = Path(record['probe_final']).parent
            losses.append([r['loss'] for r in json.loads((folder/'telemetry.json').read_text())])
        else:
            a, b = [torch.load(r['probe_final'], map_location='cpu', weights_only=True) for r in records]
            result = dict(compare(*losses, models_equal=chr._deep_equal(a,b)), completed=True, horizon=horizon)
    result['mode'] = 'sequential-single-process'
    state.atomic_json(Path(output)/'result.json', result)
    print('POOL_DETERMINISM '+json.dumps(result), flush=True)
    return result
