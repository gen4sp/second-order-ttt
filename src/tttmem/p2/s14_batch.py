"""Registered per-checkpoint batch checks; NumPy only, shared with analysis."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


FRONTIER_LIMITS = dict(mean_M=5e-4, mean_L=2e-4, H_LM=2e-4, kappa_top1_agreement=.99)
PASSKEY_LIMITS = dict(rank_agreement=.97, mean_lift=1e-3)


def frontier_comparison(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    with np.load(reference/'g0_read_ce.npz', allow_pickle=False) as a, np.load(candidate/'g0_read_ce.npz', allow_pickle=False) as b:
        if (not np.array_equal(a['eval_rows'], b['eval_rows'])
                or not np.array_equal(a['L_adapt_count'], b['L_adapt_count'])
                or not np.array_equal(a['kz_probe_offset'], b['kz_probe_offset'])):
            raise ValueError('batch comparison document/probe alignment changed')
        delta = np.asarray(a['M_doc'], float)-b['M_doc']
        values = dict(mean_M=float(abs(delta.mean())),
                      mean_L=float(abs(np.mean(a['L_adapt_doc'])-np.mean(b['L_adapt_doc']))),
                      max_abs_M_doc=float(np.abs(delta).max()),
                      kappa_top1_agreement=float(min(np.mean((a['kz_rank_'+tag]==1)==(b['kz_rank_'+tag]==1))
                                                     for tag in ('f','nf'))))
        counts = dict(n_docs=len(delta), n_ho=len(a['L_adapt_doc']))
    with np.load(reference/'g0_hlm.npz', allow_pickle=False) as a, np.load(candidate/'g0_hlm.npz', allow_pickle=False) as b:
        if not np.array_equal(a['heldout_count'], b['heldout_count']):
            raise ValueError('batch comparison H_LM alignment changed')
        values['H_LM'] = float(abs(a['heldout_sum'].sum()/a['heldout_count'].sum()
                                  -b['heldout_sum'].sum()/b['heldout_count'].sum()))
        counts['n_hlm'] = len(a['heldout_count'])
    passed = all(np.isfinite(v) for v in values.values()) and all(
        values[k] >= bound if k.endswith('agreement') else values[k] <= bound
        for k, bound in FRONTIER_LIMITS.items())
    return dict(status='PASS' if passed else 'FALLBACK', limits=FRONTIER_LIMITS,
                values=values, **counts, max_abs_M_doc_flag=values['max_abs_M_doc'] > 5e-4)


def passkey_comparison(reference, candidate):
    values = {pos: dict(rank_agreement=float(np.mean([a['rank']==b['rank'] for a,b in zip(reference[pos],candidate[pos])])),
                       mean_lift=float(abs(np.mean([a['lift'] for a in reference[pos]])
                                           -np.mean([b['lift'] for b in candidate[pos]]))))
              for pos in ('in_window','beyond')}
    passed = all(np.isfinite(list(v.values())).all() and v['rank_agreement'] >= .97
                 and v['mean_lift'] <= 1e-3 for v in values.values())
    return dict(status='PASS' if passed else 'FALLBACK', limits=PASSKEY_LIMITS,
                values=values, n_docs=len(reference['beyond']))


def save_check(out, kind, check):
    path = Path(out)/'batch_check.json'
    record = json.loads(path.read_text()) if path.exists() else {}
    record[kind] = check
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)
    return record


def validate_check(check, kind, requested, checkpoint_sha256, cell_config_sha256):
    """Missing/malformed evidence cannot authorize a larger batch, even on fallback."""
    limits = FRONTIER_LIMITS if kind == 'frontier' else PASSKEY_LIMITS
    baseline = 2 if kind == 'frontier' else 1
    if not isinstance(check,dict) or not isinstance(check.get('values'),dict):
        raise ValueError(f'malformed {kind} batch check')
    if (check['requested_batch'] != requested or check['reference_batch'] != baseline
            or check['checkpoint_sha256'] != checkpoint_sha256
            or check['cell_config_sha256'] != cell_config_sha256
            or check['status'] not in ('PASS','FALLBACK') or check['limits'] != limits
            or check['effective_batch'] != (requested if check['status']=='PASS' else baseline)):
        raise ValueError(f'invalid {kind} batch check identity/configuration')
    values = [check['values']] if kind == 'frontier' else list(check['values'].values())
    if any(not isinstance(v,dict) for v in values):
        raise ValueError(f'malformed {kind} batch check values')
    if kind == 'passkey' and set(check['values']) != {'in_window','beyond'}:
        raise ValueError('passkey batch check requires both positions')
    passed = all(np.isfinite([v[k] for k in limits]).all() and all(
        v[k] >= bound if k.endswith('agreement') else 0 <= v[k] <= bound
        for k,bound in limits.items()) for v in values)
    if passed != (check['status']=='PASS'):
        raise ValueError(f'{kind} batch check status disagrees with measured bounds')
    return check
