"""S14 CPU-only frontier selection and analysis of the job outputs.

No torch import. ``a0`` audits existing N3B reads; ``s14`` consumes the registered
reads/mini-m/<arm>/s<seed> layout. The unchanged N3B analysis is vendored in
s14_n3b_reference.py for a numerical regression gate, never used as new evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from .s14_protocol import SO_GRID, FO_GRID, PASSKEY_ETAS, N3B_SO_S, N3B_FO_S, PASSKEY_PROTOCOL
from .s14_checks import training_check, apply_checks
from .s14_evidence import PullIntegrity, read_evidence

OUTPUTS = Path('outputs')
DATA = Path('data')
ROPE = .04
T90 = {2: 3.0777, 3: 1.8856, 4: 1.6377, 5: 1.5332, 6: 1.4759}
OLD_GRID = {
    'so': [.75, .9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3],
    'fo-ce': [.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8],
    'fo-cm': [.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12],
}
ARMS = ('so', 'fo-ce', 'fo-cm', 'mini-p', 'wsd-1200', 'wsd-3469')
OVERLAP = {17, 162, 328, 693, 939, 110, 301, 361, 531, 540, 593, 984, 986}
HO_OVERLAP = {44, 45, 102}
SPLIT_RULE = 'int(sha1(str(int(eval_row)).encode("ascii")).hexdigest()[-1], 16) % 2'


def name(eta):
    return 'ce' if eta == 1 else 'eta%03d' % round(eta * 100)


def fold_ids(rows):
    return np.array([int(hashlib.sha1(str(int(r)).encode('ascii')).hexdigest()[-1], 16) % 2
                     for r in rows])


def bounds(values):
    """Seed, not document, uncertainty. Never silently discard missing seeds."""
    x = np.asarray(values, float)
    n = len(x)
    if n not in T90:
        raise ValueError(f'need 2..6 seeds for registered t bounds, got {n}')
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = T90[n] * sd / np.sqrt(n)
    return dict(mean=m, sd=sd, lb90=m-h, ub90=m+h, per_seed=x.tolist(), n=n,
                t90=T90[n], complete=bool(np.isfinite(x).all()))


def cover_label(r, dfn):
    if not r['complete'] or not dfn['complete']:
        return 'NOT-EVALUABLE'
    if r['ub90'] <= ROPE:
        return 'COVERS' if dfn['ub90'] <= ROPE else 'COVERS-M-ONLY'
    if r['lb90'] > ROPE:
        return 'NOT-COVERED' + (' (large)' if r['mean'] >= .08 else '')
    return 'UNDERPOWERED'


@dataclass
class Curve:
    eta: np.ndarray
    rows: np.ndarray
    L: np.ndarray
    Lt: np.ndarray
    M: np.ndarray
    F: np.ndarray
    N: np.ndarray
    files: list[Path]

    def mean_rows(self, mask):
        if not mask.any():
            raise ValueError('empty selection/reporting half')
        return [dict(eta=float(e), L=float(l), Lt=float(t),
                     M=float(m[mask].mean()), F=float(f[mask].mean()),
                     N=float(n[mask].mean()))
                for e, l, t, m, f, n in zip(self.eta, self.L, self.Lt, self.M, self.F, self.N)]


class ReadStore:
    def __init__(self, root=None, *, legacy=False, outputs=OUTPUTS, data=DATA, test_contract=None):
        self.root = Path(root) if root is not None else None
        self.legacy, self.outputs, self.data = legacy, Path(outputs), Path(data)
        self.test_contract = test_contract
        self.integrity = PullIntegrity(self.root) if self.root is not None else None

    def evidence(self,arm,seed,kind='frontier',required=False):
        return read_evidence(self,arm,seed,kind=kind,required=required)

    def verify(self,path):
        return self.integrity.verify(path) if self.integrity else path

    def directory(self, arm, seed):
        if self.root is None:
            raise ValueError('legacy reads have multiple fallback directories')
        return self.root / 'reads' / (self.test_contract.preset if self.test_contract else 'mini-m') / arm / f's{seed}'

    def path(self, arm, seed, eta):
        if self.root is not None:
            p = self.directory(arm, seed) / f'g0_read_{name(eta)}.npz'
            if p.exists():
                return p
            raise FileNotFoundError(p)
        candidates = []
        for base in (self.outputs / 'mini/matrix',):
            p = base / 'n3b-reads' / 'mini-m' / arm / f's{seed}'
            candidates.append(p / ('ncm' if arm == 'fo-cm' else '') / f'g0_read_{name(eta)}.npz')
        if arm in ('so', 'fo-ce'):
            for folder in ('data', 'data-g0b', 'data-g0c'):
                candidates.append(self.outputs / 'g0' / folder / 'mini-m'
                                  / {'so': 'e2e2', 'fo-ce': 'e2e1-ce'}[arm]
                                  / f's{seed}' / f'g0_read_{name(eta)}.npz')
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f'{arm} s{seed} eta={eta}: {candidates}')

    def curve(self, arm, seed, *, drop=(), hodrop=()):
        if self.root is not None:
            self.evidence(arm,seed)
        grid = OLD_GRID[arm] if self.legacy else (SO_GRID if arm == 'so' else FO_GRID)
        if self.test_contract:
            grid = self.test_contract.grids.get('so' if arm == 'so' else 'fo', grid)
        paths = [self.path(arm, seed, e) for e in grid]
        # All extensions are unconditional; FO x16 is outside this registration.
        L, Lt, M, F, N = [], [], [], [], []
        row_ids = None
        for p in paths:
            self.verify(p)
            with np.load(p, allow_pickle=False) as z:
                rows = np.asarray(z['eval_rows'], np.int64)
                if row_ids is None:
                    row_ids = rows.copy()
                    if len(rows) != (self.test_contract.n_docs if self.test_contract else 1024):
                        raise ValueError(f'expected 1024 memory documents before overlap drop: {p}')
                if not np.array_equal(rows, row_ids) or len(np.unique(rows)) != len(rows):
                    raise ValueError(f'document alignment/identity failure: {p}')
                if not np.allclose(z['M_doc'], z['fP_focal']-z['fP_nonfocal'], atol=1e-6):
                    raise ValueError(f'M decomposition failure: {p}')
                n = len(z['L_adapt_sum'])
                if n != (self.test_contract.n_ho_adapt if self.test_contract else 128):
                    raise ValueError(f'CAL/TEST require the registered 128 ho docs: {p}')
                ix = np.arange(n)
                cal = (ix % 2 == 0) & ~np.isin(ix, list(hodrop))
                tst = (ix % 2 == 1) & ~np.isin(ix, list(hodrop))
                lfun = lambda mask: float(z['L_adapt_sum'][mask].sum()/z['L_adapt_count'][mask].sum())
                L.append(lfun(cal)); Lt.append(lfun(tst))
                for out, key in ((M, 'M_doc'), (F, 'fP_focal'), (N, 'fP_nonfocal')):
                    out.append(np.asarray(z[key], float))
        keep = ~np.isin(row_ids, list(drop))
        arrays = [np.asarray(v)[:, keep] for v in (M, F, N)]
        if not all(np.isfinite(v).all() for v in [L, Lt, *arrays]):
            raise ValueError(f'non-finite read: {arm} s{seed}')
        return Curve(np.asarray(grid), row_ids[keep], np.array(L), np.array(Lt), *arrays, paths)


def interp(rows, key, eta):
    return np.interp(eta, [r['eta'] for r in rows], [r[key] for r in rows])


def truncate(rows, budget):
    """The N3B sequential top rule; M here is selection-half M in cross-fit."""
    out = [r for r in rows if r['eta'] <= 6]
    stop = None
    for r in [r for r in rows if r['eta'] > 6]:
        out.append(r)
        if r['L'] >= budget:
            stop = 'budget'; break
        if len(out) >= 3 and out[-1]['M'] < out[-2]['M'] < out[-3]['M']:
            stop = 'M-falls'; break
    return out, stop


def select(rows, budget):
    # Preserve N3B's 40001-point approximation exactly, including tie order.
    es = np.array([r['eta'] for r in rows])
    g = np.union1d(np.linspace(es.min(), es.max(), 40001), es)
    valid = interp(rows, 'L', g) <= budget
    if not valid.any():
        return dict(M=np.nan, F=np.nan, N=np.nan, Lt=np.nan, eta=np.nan, flag='unreachable')
    m = interp(rows, 'M', g)
    j = np.flatnonzero(valid)[np.argmax(m[valid])]
    return {**{k: float(interp(rows, k, g[j])) for k in ('M', 'F', 'N', 'Lt')},
            'eta': float(g[j]), 'flag': 'top-not-binding' if valid[-1] and j == len(g)-1 else ''}


def select_report(curve, budget, *, crossfit, adaptive=False):
    folds = fold_ids(curve.rows)
    parts = []
    for f in ([0, 1] if crossfit else [None]):
        fit = np.ones(len(folds), bool) if f is None else folds == f
        report = fit if f is None else ~fit
        train = curve.mean_rows(fit)
        train, stop = truncate(train, budget) if adaptive else (train, None)
        sel = select(train, budget)
        test = curve.mean_rows(report)
        eta = sel['eta']
        parts.append({**sel, **{k: float(interp(test, k, eta)) for k in ('M', 'F', 'N', 'Lt')},
                      'selected_M': sel['M'], 'fit_fold': f, 'fit_n': int(fit.sum()),
                      'report_n': int(report.sum()), 'top': train[-1]['eta'], 'stop': stop})
    return {**{k: float(np.average([p[k] for p in parts], weights=[p['report_n'] for p in parts]))
               for k in ('M', 'F', 'N', 'Lt')},
            'parts': parts}


def utility_curve(curve, floor, budget, *, crossfit, adaptive=False):
    """Held-out utility of each half's L-wM minimizer; average, then cross.

    L is always CAL, independent of the M split. No claim that held-out utility
    is an optimized frontier; lambda is the cross-fitted policy exchange rate.
    """
    ws = np.linspace(0, 2, 2001)
    fold = fold_ids(curve.rows)
    vals, weights = [], []
    for f in ([0, 1] if crossfit else [None]):
        fit = np.ones(len(fold), bool) if f is None else fold == f
        train = curve.mean_rows(fit)
        train = truncate(train, budget)[0] if adaptive else train
        test = curve.mean_rows(fit if f is None else ~fit)
        e = np.array([r['eta'] for r in train])
        g = np.union1d(np.linspace(e.min(), e.max(), 4001), e)
        g = g[interp(train, 'L', g) <= floor+.05]
        if not len(g):
            return np.full(len(ws), np.nan)
        l, m = interp(train, 'L', g), interp(train, 'M', g)
        mt = interp(test, 'M', g)
        vv = []
        for batch in np.array_split(ws, 20):
            j = np.argmin(l[None, :] - batch[:, None]*m[None, :], axis=1)
            vv.extend(l[j] - batch*mt[j])
        vals.append(vv)
        weights.append(int((fit if f is None else ~fit).sum()))
    return np.average(vals, axis=0, weights=weights)


def seed_block(curves, seed, *, crossfit=False, legacy=False):
    if 'so' not in curves:
        return dict(seed=seed, L_SO_min=np.nan, B_SO=np.nan, B_FO=np.nan,
                    selected={}, fold_sizes=[], flags=['so:unavailable'])
    ids = next(iter(curves.values())).rows
    if any(not np.array_equal(c.rows, ids) for c in curves.values()):
        raise ValueError('paired arms must have identical document ids')
    floor = float(curves['so'].L.min())
    bso, bfo = floor+.01, floor+.02
    out = dict(seed=seed, L_SO_min=floor, B_SO=bso, B_FO=bfo, selected={},
               fold_sizes=[int((fold_ids(ids)==i).sum()) for i in (0, 1)])
    utilities = {}
    for arm, c in curves.items():
        adaptive = arm == 'fo-cm' if legacy else arm != 'so'
        b = bso if arm == 'so' else bfo
        out['selected'][arm] = select_report(c, b, crossfit=crossfit, adaptive=adaptive)
        utilities[arm] = utility_curve(c, floor, b, crossfit=crossfit, adaptive=adaptive)
    sels = out['selected']
    for arm in curves:
        if arm == 'so':
            continue
        out[f'R_cov_{arm}'] = sels['so']['M']-sels[arm]['M']
        out[f'dfn_{arm}'] = sels[arm]['N']-sels['so']['N']
        diff = utilities['so']-utilities[arm]
        crossing = np.flatnonzero(diff <= 0)
        out[f'lambda_star_{arm}'] = (None if not np.isfinite(diff).all() else
                                   float(crossing[0]/1000) if len(crossing) else '>2')
        out[f'lambda_star_{arm}_label'] = ('upper bound' if any(
            p['flag'] == 'top-not-binding' for p in sels['so']['parts']) else 'estimate')
    for key, a, b in (('x', 'fo-cm', 'fo-ce'), ('x_P', 'mini-p', 'fo-cm'),
                      ('delta_steps', 'wsd-3469', 'wsd-1200'), ('delta_shape', 'wsd-1200', 'fo-ce')):
        if a in sels and b in sels:
            out[key] = sels[a]['M']-sels[b]['M']
    out['flags'] = []
    for arm, c in curves.items():
        floor_eta = float(c.eta[np.argmin(c.L)])
        blockers = ((.75, 3) if legacy else (float(c.eta[0]), float(c.eta[-1]))) if arm == 'so' else (.75,)
        if floor_eta in blockers:
            # The unconditional read beyond that edge resolves the N3B grid boundary.
            unresolved = ((floor_eta == c.eta[0]) or (floor_eta == c.eta[-1]))
            out['flags'].append(f'{arm}:floor-edge'+(':blocking' if unresolved else ':resolved'))
        elif arm == 'so' and floor_eta == 3 and not legacy:
            out['flags'].append('so:floor-edge:resolved')
        elif int(np.argmin(c.L)) in (0, len(c.L)-1):
            out['flags'].append(f'{arm}:floor-edge:flag-only')
        for p in sels[arm]['parts']:
            if not np.isfinite(p['eta']):
                out['flags'].append(f'{arm}:unreachable')
            elif arm == 'so' and p['flag'] == 'top-not-binding':
                out['flags'].append('so:top-open:COVERS-blocked')
            elif arm != 'so' and p['stop'] is None and c.L[-1] < bfo:
                out['flags'].append(f'{arm}:top-open:NOT-COVERED-blocked')
    return out


def aggregate(blocks, arms=None):
    keys = list(dict.fromkeys(k for b in blocks for k, v in b.items()
                             if isinstance(v, (int, float)) and k != 'seed'))
    keys = [k for k in keys if not k.startswith('lambda_star_')]
    if arms is not None:
        keys = list(dict.fromkeys(keys+['R_cov_'+a for a in arms if a != 'so']
                                 +['dfn_'+a for a in arms if a != 'so']
                                 +['x', 'x_P', 'delta_steps', 'delta_shape']))
    out = {k: bounds([b.get(k, np.nan) for b in blocks]) for k in keys}
    lambdas = list(dict.fromkeys(k for b in blocks for k in b
                                if k.startswith('lambda_star_') and not k.endswith('_label')))
    if arms is not None:
        lambdas = list(dict.fromkeys(lambdas+['lambda_star_'+a for a in arms if a != 'so']))
    for key in lambdas:
        values = [b.get(key) for b in blocks]
        censored = sum(v == '>2' for v in values)
        label = 'upper bound' if any(b.get(key+'_label') == 'upper bound' for b in blocks) else 'estimate'
        out[key] = (dict(per_seed=values, censored_seeds=censored, n=len(values),
                         mean=('>2' if censored == len(values) else 'not estimated (censored)') if censored else None,
                         status='NOT-EVALUABLE' if any(v is None for v in values) else 'censored',
                         label=label) if censored or any(v is None for v in values)
                    else dict(**bounds(values), censored_seeds=0, label=label))
    out['labels'] = {}
    for k in keys:
        if k.startswith('R_cov_'):
            arm = k[6:]
            label = cover_label(out[k], out['dfn_'+arm])
            flags = [f for b in blocks for f in b['flags']]
            relevant = [f for f in flags if f.split(':')[0] in ('so', arm)]
            if any('floor-edge:blocking' in f or 'unreachable' in f for f in relevant):
                label = 'NOT-EVALUABLE (grid/floor)'
            elif label.startswith('COVERS') and any('COVERS-blocked' in f for f in flags):
                label = 'NOT-EVALUABLE (SO top)'
            elif any(f'{arm}:top-open:blocking' == f for f in flags) or (
                    label.startswith('NOT-COVERED') and any(f'{arm}:top-open' in f for f in flags)):
                label = 'NOT-EVALUABLE (FO top)'
            out['labels'][arm] = label
    return out


def top1(curves_by_seed):
    out = []
    for seed, curves in curves_by_seed.items():
        for arm, c in curves.items():
            for eta, path in zip(c.eta, c.files):
                with np.load(path, allow_pickle=False) as z:
                    if 'kz_rank_f' not in z.files:
                        continue
                    keep = np.isin(z['eval_rows'], c.rows)
                    off = z['kz_probe_offset']
                    pmask = np.repeat(keep, np.diff(off))
                    rec = dict(seed=seed, arm=arm, eta=float(eta), n_probes=int(pmask.sum()))
                    for tag in ('f', 'nf'):
                        rank = z['kz_rank_'+tag][pmask]
                        rec['top1_'+tag] = float(np.mean(rank == 1))
                        rec['mean_rank_'+tag] = float(rank.mean())
                    out.append(rec)
    return out


def overlap_rows(manifest):
    near = {i for i, line in enumerate(Path(manifest).open()) if json.loads(line).get('near_dup')}
    return OVERLAP | near


def parity_gate(blocks, store):
    from . import s14_n3b_reference as ref
    ref.path = lambda arm, seed, eta: str(store.path(arm, seed, eta))
    reference = [ref.seed_block(s) for s in range(3)]
    comparisons = {}
    mapping = {'R_cov_cm': 'R_cov_fo-cm', 'R_cov_ce': 'R_cov_fo-ce',
               'dfn_cm': 'dfn_fo-cm', 'dfn_ce': 'dfn_fo-ce', 'x': 'x',
               'lambda_star_cm': 'lambda_star_fo-cm', 'lambda_star_ce': 'lambda_star_fo-ce'}
    for old, new in mapping.items():
        comparisons[new] = bool(np.allclose([b[new] for b in blocks], [b[old] for b in reference],
                                           rtol=0, atol=1e-12))
    agg = aggregate(blocks)
    rounded = [round(b['R_cov_fo-cm'], 3) for b in blocks]
    comparisons['published_rounding'] = (rounded == [.171, .081, .156]
        and round(agg['R_cov_fo-cm']['mean'], 3) == .136
        and round(agg['R_cov_fo-cm']['lb90'], 3) == .084
        and round(agg['R_cov_fo-ce']['mean'], 3) == .101 and round(agg['x']['mean'], 3) == -.035)
    passed = all(comparisons.values())
    line = 'PARITY '+('PASS' if passed else 'FAIL')+': N3B R_cov_CM .171/.081/.156; mean .136; LB90 .084; R_cov_CE .101; x -.035'
    print(line, flush=True)
    if not passed:
        raise AssertionError(comparisons)
    return dict(line=line, checks=comparisons, reference_seed_blocks=reference)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n')


def table(agg, keys):
    lines = ['| quantity | per seed | mean | LB90 | UB90 |', '|---|---|---:|---:|---:|']
    for key in keys:
        a = agg[key]
        if (a.get('censored_seeds', 0) or a.get('status') == 'NOT-EVALUABLE'
                or a.get('complete') is False):
            lines.append(f"| {key} ({a.get('status', a.get('label', 'NOT-EVALUABLE'))}) | "
                         f"{' / '.join(map(str, a['per_seed']))} | "
                         f"{a.get('mean')} ({a.get('censored_seeds', 0)} censored) | — | — |")
            continue
        per = ' / '.join(f'{v:+.6f}' for v in a['per_seed'])
        suffix = ' (upper bound)' if a.get('label') == 'upper bound' else ''
        lines.append(f"| {key}{suffix} | {per} | {a['mean']:+.6f} | {a['lb90']:+.6f} | {a['ub90']:+.6f} |")
    return '\n'.join(lines)


def a0(out_dir, *, store=None, manifest=None):
    store = store or ReadStore(legacy=True)
    manifest = manifest or DATA / 'p2data/eval.jsonl'
    result = {'split_rule': SPLIT_RULE, 't90': 1.8856, 'seeds': [0, 1, 2], 'versions': {},
              'fold_weighting': 'report-fold document count'}
    for variant, drop, hodrop in (('all', (), ()), ('overlap_drop', overlap_rows(manifest), HO_OVERLAP)):
        curves = {s: {a: store.curve(a, s, drop=drop, hodrop=hodrop) for a in OLD_GRID} for s in range(3)}
        versions = {}
        for cf in (False, True):
            blocks = [seed_block(curves[s], s, crossfit=cf, legacy=True) for s in range(3)]
            versions['crossfit' if cf else 'in_sample'] = dict(per_seed=blocks, aggregate=aggregate(blocks))
            if variant == 'all' and not cf:
                result['parity'] = parity_gate(blocks, store)
        optimism = {}
        for arm in OLD_GRID:
            optimism['M_'+arm] = bounds([versions['in_sample']['per_seed'][s]['selected'][arm]['M']
                                        - versions['crossfit']['per_seed'][s]['selected'][arm]['M'] for s in range(3)])
        for key in ('R_cov_fo-cm', 'R_cov_fo-ce', 'x', 'dfn_fo-cm'):
            optimism[key] = bounds([versions['in_sample']['per_seed'][s][key]
                                    - versions['crossfit']['per_seed'][s][key] for s in range(3)])
        versions['selection_optimism'] = optimism
        versions['top1'] = top1(curves)
        versions['source_files'] = [str(p) for cs in curves.values() for c in cs.values() for p in c.files]
        result['versions'][variant] = versions
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'a0_crossfit.json', result)
    keys = ['R_cov_fo-cm', 'R_cov_fo-ce', 'x', 'dfn_fo-cm', 'dfn_fo-ce',
            'lambda_star_fo-cm', 'lambda_star_fo-ce']
    text = ['# A0: cross-fit selection audit', '', result['parity']['line'], '',
            '[Observation] Recomputed from real per-document N3B/G0 reads; no new model training.',
            '', '## Fixed method', '', f'Split: `{SPLIT_RULE}`. This is a hash partition, not forced equal counts.',
            'Select on one half and report on the other; swap, then weight by report-fold size. '
            'CAL even ho ids define budgets; TEST odd ho ids remain diagnostic. '
            'The 40001-point eta interpolation and 4001-point utility grid reproduce N3B. '
            'Adaptive stopping uses selection-half M only. T90=1.8856 over three seeds.',
            'For lambda*, select L-wM on each training half within L_SO,min+.05; report held-out utility; '
            'take the first crossing of the two averaged held-out utilities on w=0:.001:2. '
            'This is a cross-fitted policy exchange rate, not a newly optimized held-out frontier.', '']
    for variant, v in result['versions'].items():
        text += ['## '+variant, '']
        for mode in ('in_sample', 'crossfit'):
            text += ['### '+mode, '', table(v[mode]['aggregate'], keys), '',
                     '[Observation] Labels: '+str(v[mode]['aggregate']['labels'])+'.', '']
        text += ['### Selection optimism = in-sample minus cross-fit', '',
                 table(v['selection_optimism'], list(v['selection_optimism'])), '']
    v = result['versions']['all']
    sizes = v['crossfit']['per_seed'][0]['fold_sizes']
    text += [f'[Observation] The fixed hash gives fold sizes {sizes[0]} and {sizes[1]}. '
             'Report-fold size weighting makes cross-fit equal in-sample whenever both folds '
             'select the same eta. A realized selection-optimism estimate may still be negative.', '']
    text += ['## Selected operating points [observation]', '',
             '| seed | arm | in-sample eta | cross-fit eta (fit 0 / fit 1) | fold sizes |', '|---|---|---:|---|---|']
    for b, c in zip(v['in_sample']['per_seed'], v['crossfit']['per_seed']):
        for arm in OLD_GRID:
            text.append(f"| {b['seed']} | {arm} | {b['selected'][arm]['parts'][0]['eta']:.5f} | "
                        + ' / '.join(f"{p['eta']:.5f}" for p in c['selected'][arm]['parts'])
                        + f" | {c['fold_sizes']} |")
    text += ['', '## Stored candidate ranks [observation]', '',
             '`_nf` = fP_nonfocal: suppress matched nonfocal writes, retain the focal source; '
             '`_f` = fP_focal: suppress focal source writes. Both freeze P/read-segment writes. '
             'These are candidate-token ranks among distinct probe targets in the document, '
             'not full-vocabulary accuracy and not passkey exact recall. Rank 1 counts ties as first '
             '(the stored definition uses strictly greater competing scores). Probe-weighted fractions.', '',
             '| seed | arm | eta | probes | top1_nf (source retained) | top1_f (source suppressed) |', '|---|---|---:|---:|---:|---:|']
    for r in v['top1']:
        text.append(f"| {r['seed']} | {r['arm']} | {r['eta']:g} | {r['n_probes']} | {r['top1_nf']:.6f} | {r['top1_f']:.6f} |")
    text += ['', '## Reading [interpretation]', '',
             'Cross-fit '+v['crossfit']['aggregate']['labels']['fo-cm']+'; overlap-drop '
             +result['versions']['overlap_drop']['crossfit']['aggregate']['labels']['fo-cm']+'. '
             'Compare these with the in-sample label above. The numerical correction addresses reuse '
             'of M documents for selection and reporting; it is not an independent replication.',
             'Keep the paper wording “no closer than FO-CE.” Cross-fit does not supply the missing '
             'midpoint refinement, and the old-grid x bound alone does not license “further.” '
             'The registered upgrade requires new seeds 3–5 on the dense grid with UB90(x)<0.',
             'All values, per-fold stops, source paths and the unchanged N3B reference blocks are in '
             '[a0_crossfit.json](a0_crossfit.json). No document bootstrap is substituted for seed uncertainty.', '']
    (out / 'a0-crossfit-results.md').write_text('\n'.join(text))
    print(table(v['crossfit']['aggregate'], keys), flush=True)
    return result


def _passkey_point(path, frozen, eta, *, test_contract=None):
    n = test_contract.passkey_docs if test_contract else 256
    audit_n = min(8, n)
    digits = test_contract.passkey_digit_tokens if test_contract else 2
    with np.load(path, allow_pickle=False) as z:
        if str(z['protocol']) != PASSKEY_PROTOCOL:
            raise ValueError(f'old/unknown passkey scoring protocol: {path}')
        ids = z['host_rows']
        if float(z['eta']) != eta or int(z['seed']) != frozen['seed']:
            raise ValueError(f'passkey eta/seed mismatch: {path}')
        if (len(ids) != n or not np.array_equal(ids, frozen['host_rows'][:n])
                or str(z['host_set_sha256']) != frozen['documents_sha256']):
            raise ValueError(f'frozen passkey host ids/hash mismatch: {path}')
        rec = dict(file=str(path), document_hash=str(z['documents_sha256']),
                   host_rows=ids.tolist(), key=z['key'].tolist(), foil_keys=z['foil_keys'].tolist())
        deltas = np.asarray(z['leak_check_per_doc_max_abs_delta_nll'], float)
        maximum = float(z['leak_check_max_abs_delta_nll'])
        leak = str(z['leak_check'])
        if (str(z['leak_check_dtype']) != 'fp32' or int(z['leak_check_n_docs']) != audit_n
                or deltas.shape != (audit_n,) or not np.isfinite(deltas).all() or np.any(deltas < 0)
                or maximum != float(deltas.max()) or float(z['leak_check_tolerance']) != 1e-5
                or leak not in ('PASS', 'FAIL') or (leak == 'PASS' and maximum > 1e-5)):
            raise ValueError(f'invalid deterministic leak check: {path}')
        rec['leak_check'] = dict(status=leak, max_abs_delta_nll=maximum,
                                 per_doc_max_abs_delta_nll=deltas.tolist(), n_docs=audit_n)
        if len(rec['document_hash']) != 64 or len(rec['key']) != len(ids) or np.shape(z['foil_keys']) != (n, 8):
            raise ValueError(f'invalid passkey document/key identity: {path}')
        if z['key_token_count'].shape != (n,) or not np.all(z['key_token_count'] == digits):
            raise ValueError(f'expected two scored Llama-3 digit tokens, space excluded: {path}')
        for pos in ('in_window', 'beyond'):
            arrays = {k: np.asarray(z[pos+'_'+k]) for k in ('lift', 'rank', 'exact')}
            if any(v.shape != (n,) or not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError(f'invalid passkey arrays: {path}')
            for key, shape in (('nll_true', (n,)), ('nll_foils', (n,8)),
                               ('nll_token', (n,9,digits)), ('query_span', (n,3))):
                value = np.asarray(z[pos+'_'+key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'invalid passkey {pos}_{key}: {path}')
            if (np.any((arrays['rank'] < 1) | (arrays['rank'] > 9))
                    or not np.all(arrays['rank'] == arrays['rank'].astype(int))
                    or not np.isin(arrays['exact'], [0, 1]).all()):
                raise ValueError(f'invalid passkey ranks: {path}')
            rec[pos] = dict(lift=arrays['lift'], top1=(arrays['rank']==1).astype(float),
                            exact_match=arrays['exact'].astype(float))
        return rec


def passkey_analysis(store, blocks):
    """Digit-only paired adaptation lift, interpolated in eta without extrapolation."""
    frozen = json.loads(Path(__file__).with_name('s14_passkey_hosts.json').read_text())
    records, missing = [], []
    noadapt_docs = {a: [] for a in ARMS}
    identity = None
    for block in blocks:
        seed, per_arm = block['seed'], {}
        for arm in ARMS:
            directory = store.directory(arm, seed)
            points, errors = [], []
            noadapt = None
            try:
                batch_evidence = store.evidence(arm,seed,'passkey')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                batch_evidence = dict(status='NOT-EVALUABLE',error=str(exc))
            for eta in PASSKEY_ETAS['so' if arm == 'so' else 'fo']+[0]:
                path = directory/f'passkey_{name(eta) if eta else "noadapt"}.npz'
                try:
                    if batch_evidence['status'] != 'OK':
                        raise ValueError(batch_evidence['error'])
                    store.verify(path)
                    p = dict(eta=eta, **_passkey_point(path, frozen, eta))
                    ident = (p['document_hash'], p['host_rows'], p['key'], p['foil_keys'])
                    if identity is None and arm in ('so', 'fo-ce', 'fo-cm'):
                        identity = ident
                    if identity is not None and ident != identity:
                        raise ValueError(f'passkey document/host/key/foil mismatch: {path}')
                    if eta:
                        points.append(p)
                    else:
                        noadapt = p
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{path}: {type(exc).__name__}: {exc}')
            missing.extend(errors)
            def summary(point):
                return {pos: {k: float(v.mean()) for k, v in point[pos].items()}
                        for pos in ('in_window', 'beyond')}
            etas = [p['eta'] for p in points]
            op = block['selected'].get(arm, {}).get('parts', [])
            def evaluate(nearest=False):
                if noadapt is None or len(points) != len(PASSKEY_ETAS['so' if arm == 'so' else 'fo']) or not op:
                    return None
                values, weights, used = [], [], []
                for part in op:
                    e = part['eta']
                    if not np.isfinite(e) or (not nearest and not etas[0] <= e <= etas[-1]):
                        return None
                    if nearest:
                        e = min(etas, key=lambda t: abs(np.log(t)-np.log(e)))
                    # Interpolate each document before subtracting its paired noadapt.
                    doc = {}
                    for pos in ('in_window', 'beyond'):
                        doc[pos] = {}
                        for k in ('lift', 'top1', 'exact_match'):
                            arrays = np.stack([p[pos][k] for p in points])
                            j = min(max(int(np.searchsorted(etas, e, side='right'))-1, 0), len(etas)-2)
                            t = (e-etas[j])/(etas[j+1]-etas[j])
                            v = arrays[j]*(1-t)+arrays[j+1]*t
                            doc[pos][k] = float(v.mean())
                            if k == 'lift':
                                doc[pos]['adaptation_lift'] = float((v-noadapt[pos][k]).mean())
                    values.append(doc); weights.append(part['report_n']); used.append(e)
                return dict(etas=used, status='OK', **{
                    pos: {k: float(np.average([v[pos][k] for v in values], weights=weights))
                          for k in values[0][pos]} for pos in ('in_window', 'beyond')})
            selected = evaluate()
            checked = [p['leak_check'] for p in points+([noadapt] if noadapt else [])]
            leak_status = ('FAIL' if any(p['status'] == 'FAIL' for p in checked) else
                           'PASS' if checked else 'NOT-EVALUABLE')
            if checked and any(p != checked[0] for p in checked):
                errors.append(f'{arm}/s{seed}: inconsistent checkpoint leak checks')
                selected = None
                if leak_status != 'FAIL':
                    leak_status = 'NOT-EVALUABLE'
            if noadapt is not None:
                noadapt_docs[arm].append(noadapt['beyond']['lift'])
            per_arm[arm] = dict(points=[dict(eta=p['eta'], **summary(p)) for p in points],
                batch_check=batch_evidence,
                selected=selected, selected_status='OK' if selected else 'NOT-EVALUABLE (eta outside reads)',
                nearest_proxy=evaluate(True), noadapt=summary(noadapt) if noadapt else None,
                eta1=summary(points[0]) if points and points[0]['eta'] == 1 else None,
                leak_check=dict(status=leak_status, evidence=next((p for p in checked if p['status'] == 'FAIL'),
                                                                 checked[0] if checked else None)),
                errors=errors, status='LEAK' if leak_status == 'FAIL' else 'OK' if not errors else 'NOT-EVALUABLE')
        records.append(dict(seed=seed, arms=per_arm))
    gate, leak = {}, {}
    for arm in ARMS:
        arms = [r['arms'].get(arm) for r in records]
        checks = [p['leak_check'] for p in arms]
        leak[arm] = dict(status='LEAK' if any(p['status'] == 'FAIL' for p in checks) else
                        'PASS' if all(p['status'] == 'PASS' for p in checks) else 'NOT-EVALUABLE',
                        per_checkpoint=[dict(seed=r['seed'], **p) for r, p in zip(records, checks)])
        # Shared keys/docs across seeds: average over seeds first, then use docs
        # as units for this two-sided 95% descriptive interval. It never gates.
        if len(noadapt_docs[arm]) == len(blocks):
            docs = np.stack(noadapt_docs[arm]).mean(axis=0)
            mean, sd = float(docs.mean()), float(docs.std(ddof=1))
            h = 1.969310569*sd/np.sqrt(len(docs))  # t_.975, df=255
            leak[arm]['descriptive_noadapt_lift'] = dict(mean=mean, sd=sd, lb95=mean-h, ub95=mean+h,
                n_docs=len(docs), n_seeds=len(blocks), unit='per-doc mean over seeds', t95=1.969310569,
                overrides_reading=False)
        if arm in ('so', 'fo-ce'):
            if any(p['eta1'] is None for p in arms):
                gate[arm] = dict(valid=False, status='NOT-EVALUABLE (missing eta1 passkey)')
                continue
            lift = bounds([p['eta1']['in_window']['lift'] for p in arms])
            top = float(np.mean([p['eta1']['in_window']['top1'] for p in arms]))
            valid = bool(lift['lb90'] > 0 and top >= .5)
            gate[arm] = dict(lift=lift, top1=top, eta=1, valid=valid,
                             status='VALID' if valid else 'INVALID')
    contrasts = {}
    for kind in ('selected', 'nearest_proxy'):
        for measure in ('adaptation_lift', 'lift'):
            vals = []
            for r in records:
                so = r['arms'].get('so', {}).get(kind)
                cm = r['arms'].get('fo-cm', {}).get(kind)
                if so is None or cm is None:
                    break
                vals.append(so['beyond'][measure]-cm['beyond'][measure])
            contrasts[kind+'_'+measure] = bounds(vals) if len(vals) == len(blocks) else None
    contrast = contrasts['selected_adaptation_lift']
    required = ('so', 'fo-ce', 'fo-cm')
    suspect = any(leak[a]['status'] == 'LEAK' for a in required)
    complete = (contrast is not None and all(g['status'] != 'NOT-EVALUABLE (missing eta1 passkey)' for g in gate.values())
                and all(leak[a]['status'] in ('PASS', 'LEAK') for a in required)
                and all(b.get('checks', {}).get(a, {'status': 'OK'})['status'] == 'OK'
                        for b in blocks for a in required))
    valid = complete and all(g['valid'] for g in gate.values()) and not suspect
    reading = ('LEAK' if suspect else 'NOT-EVALUABLE' if not complete else
               'INVALID' if not valid else 'SO higher (descriptive)' if contrast['lb90'] > 0 else 'interval')
    return dict(per_seed=records, validity=gate, no_leak=leak, valid=valid, contrast=contrast,
                raw_lift_contrast=contrasts['selected_lift'],
                nearest_grid_proxy=contrasts['nearest_proxy_adaptation_lift'], missing=missing,
                required_cells=[dict(arm=a, seed=b['seed'], role='eta1 gate' if a == 'fo-ce' else 'primary working point')
                                for b in blocks for a in required],
                error='; '.join(missing) if not complete and missing else None,
                reading=reading, primary='per-document lift(eta) - lift(noadapt)',
                interpolation='piecewise linear in eta; report-fold size weights; no extrapolation',
                fo_cm_in_window='descriptive only; validity uses SO and FO-CE at eta x1')


def safe_passkey_analysis(store, blocks):
    try:
        return passkey_analysis(store, blocks)
    except Exception as exc:
        # Independent diagnostic must never prevent the frontier from being persisted.
        return dict(reading='NOT-EVALUABLE', valid=False,
                    error=f'{type(exc).__name__}: {exc}')


def _hlm(store, arm, seed):
    if store.legacy and store.root is None:
        p = store.path(arm, seed, 1).parent/'g0_hlm.npz'
        if not p.exists():
            # The ce read can come from a G0 folder; take H_LM from the N3B reads.
            p = store.outputs/'mini/matrix/n3b-reads/mini-m'/arm/f's{seed}'
            p = p/('ncm' if arm == 'fo-cm' else '')/'g0_hlm.npz'
    else:
        p = store.directory(arm, seed)/'g0_hlm.npz'
    if not p.exists():
        return dict(status='NOT-EVALUABLE', file=str(p), error='missing H_LM file')
    try:
        if store.root is not None:
            store.evidence(arm,seed)
        store.verify(p)
        with np.load(p, allow_pickle=False) as z:
            sums, counts = z['heldout_sum'], z['heldout_count']
            tc = getattr(store, 'test_contract', None)
            n = tc.ho_docs if tc else 256
            if (sums.shape != (n,) or counts.shape != (n,) or not np.isfinite(sums).all()
                    or not np.isfinite(counts).all() or np.any(counts <= 0)):
                raise ValueError('H_LM requires 256 finite heldout sums and positive counts')
            return dict(status='OK', value=float(sums.sum()/counts.sum()), file=str(p))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(status='NOT-EVALUABLE', file=str(p), error=str(exc))


def cell_checks(block, curves, stores):
    seed = block['seed']
    hlm = {a: _hlm(st, a, seed) for a, st in stores.items()}
    checks = {a: (dict(status='OK', legacy=True, compute=None) if st.legacy else
                  training_check(st.root, a, seed, verify=st.verify)) for a, st in stores.items()}
    for arm, check in checks.items():
        if not stores[arm].legacy:
            try:
                check['READS_DONE'] = stores[arm].evidence(arm,seed)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                check['READS_DONE'] = dict(status='NOT-EVALUABLE',error=str(exc))
                check['status'] = 'NOT-EVALUABLE'
        check['H_LM'] = hlm[arm]
        check['read'] = block.get('cells', {}).get(arm, dict(status='OK' if arm in curves else 'NOT-EVALUABLE'))
        if check['read']['status'] != 'OK' and check['status'] != 'REPAIR':
            check['status'] = 'NOT-EVALUABLE'
        if hlm[arm]['status'] != 'OK' and check['status'] != 'REPAIR':
            check['status'] = 'NOT-EVALUABLE'
        if not stores[arm].legacy and check.get('steps', 0) > 1200:
            if arm not in curves or 'fo-ce' not in curves:
                if arm != 'mini-p' and check['status'] != 'REPAIR':
                    check['status'] = 'NOT-EVALUABLE'
                check['regression'] = dict(status='NOT-EVALUABLE', descriptive_only=arm == 'mini-p',
                                           error='missing arm/comparator read')
                continue
            floor_bad = float(curves[arm].L.min()) >= float(curves['fo-ce'].L.min())
            if hlm['fo-ce']['status'] != 'OK' or checks['fo-ce']['status'] != 'OK':
                if arm != 'mini-p':
                    check['status'] = 'NOT-EVALUABLE' if check['status'] != 'REPAIR' else 'REPAIR'
                check['regression'] = dict(status='NOT-EVALUABLE', descriptive_only=arm == 'mini-p')
            elif hlm[arm]['status'] == 'OK':
                lm_bad = hlm[arm]['value'] >= hlm['fo-ce']['value']
                check['regression'] = dict(CAL_floor_not_lower=floor_bad, H_LM_not_lower=lm_bad,
                                           comparator='fo-ce', comparator_seed=seed,
                                           descriptive_only=arm == 'mini-p')
                if (floor_bad or lm_bad) and arm != 'mini-p':
                    check['status'] = 'REPAIR'
                    check.setdefault('repair', []).append('schedule regression: no reading')
    block.update(H_LM=hlm, checks=checks)


CONTRAST_ARMS = {'x': ('fo-cm', 'fo-ce'), 'x_P': ('mini-p', 'fo-cm'),
                 'delta_steps': ('wsd-3469', 'wsd-1200'), 'delta_shape': ('wsd-1200', 'fo-ce')}


def ready(blocks, arms, *, uses_so_max=False):
    """Every point uses SO's CAL floor, but only R_cov uses SO's M maximum."""
    arms = set(arms) | {'so'}
    if arms & {'fo-cm', 'wsd-3469'}:
        arms.add('fo-ce')  # schedule comparison evidence
    for b in blocks:
        for arm in arms:
            check = b['checks'].get(arm, dict(status='NOT-EVALUABLE'))
            if check['status'] != 'OK':
                return False
            compute = check.get('compute')
            if compute and (compute.get('status') != 'PASS' or compute['blocks_not_covered'] or compute['blocks_covers']):
                return False
        for flag in b['flags']:
            arm = flag.split(':')[0]
            if arm not in arms:
                continue
            if arm == 'so' and ':top-open:' in flag and not uses_so_max:
                continue
            if any(t in flag for t in ('blocking', 'blocked', 'unreachable', 'unavailable')):
                return False
    return True


def analyze_contract(root, out_dir, contract):
    """Exercise the same C2 readers on writer outputs, without scientific claims."""
    from .s14_test_contract import TestContract
    from .data import sha256_file
    if not isinstance(contract, TestContract):
        raise TypeError('test override must be an explicit TestContract')
    store = ReadStore(root, test_contract=contract)
    hosts_path = Path(contract.passkey_hosts_file) if contract.passkey_hosts_file else Path(__file__).with_name('s14_passkey_hosts.json')
    frozen = json.loads(store.verify(hosts_path).read_text())
    cells = []
    for seed in contract.seeds:
        for arm in contract.expected_steps:
            checks = dict(training=training_check(root, arm, seed, test_contract=contract,verify=store.verify))
            training = checks['training']
            checks['timing'] = training.get('timing', dict(status='NOT-EVALUABLE'))
            checks['compute_v2'] = training.get('compute') or dict(status='NOT-EVALUABLE')
            checks['guard'] = dict(status='OK' if training['status']=='OK' else training['status'],
                                   probes=training.get('guard_probes'))
            checks['gdiv'] = training.get('gdiv', dict(status='OK', events=[]))
            checks['MINI_CELL_DONE'] = dict(status=training['status'])
            directory = store.directory(arm, seed)
            try:
                curve = store.curve(arm, seed)
                checks['reads'] = dict(status='OK', grid=curve.eta.tolist())
                # Validate the kappa evidence at every requested kappa point.
                for eta, path in zip(curve.eta, curve.files):
                    if eta in (1,2,4):
                        with np.load(path, allow_pickle=False) as z:
                            for key in ('kz_rank_f','kz_rank_nf','kz_probe_offset'):
                                if key not in z or not np.isfinite(z[key]).all():
                                    raise ValueError(f'missing/invalid {key}: {path}')
                checks['kappa'] = dict(status='OK')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                checks['reads'] = checks['kappa'] = dict(status='NOT-EVALUABLE', error=str(exc))
            checks['H_LM'] = _hlm(store, arm, seed)
            try:
                etas = contract.passkey_etas.get('so' if arm=='so' else 'fo', PASSKEY_ETAS['so' if arm=='so' else 'fo'])
                points = [_passkey_point(directory/f'passkey_{name(e) if e else "noadapt"}.npz',
                          frozen, e, test_contract=contract) for e in [*etas,0]]
                store.evidence(arm,seed,'passkey',required=True)
                identities = [(p['document_hash'],p['host_rows'],p['key'],p['foil_keys']) for p in points]
                if any(p != identities[0] for p in identities):
                    raise ValueError('passkey identity mismatch across conditions')
                checks['passkey'] = dict(status='OK', conditions=len(points))
                checks['leak'] = dict(status='OK' if all(p['leak_check']['status']=='PASS' for p in points) else 'FAIL')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                checks['passkey'] = checks['leak'] = dict(status='NOT-EVALUABLE',error=str(exc))
            try:
                marker = json.loads((directory/'READS_DONE.json').read_text())
                checks['batch_check'] = store.evidence(arm,seed,required=True)
                expected = [f'g0_read_{name(e)}.npz' for e in (contract.grids.get('so' if arm=='so' else 'fo') or
                           (SO_GRID if arm=='so' else FO_GRID))]+['g0_hlm.npz']+[
                           f'passkey_{name(e) if e else "noadapt"}.npz' for e in [*etas,0]]
                if (marker.get('complete') is not True or set(marker['read_list']) != set(expected)
                        or set(marker['npz_sha256']) != set(expected)
                        or any(sha256_file(directory/n)!=marker['npz_sha256'][n] for n in expected)
                        or len(marker['checkpoint_sha256'])!=64 or len(marker['config_sha256'])!=64):
                    raise ValueError('incomplete/changed READS_DONE evidence')
                checks['READS_DONE'] = dict(status='OK')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                checks['READS_DONE'] = dict(status='NOT-EVALUABLE',error=str(exc))
            ok = all(c.get('status') in ('OK','PASS','FLAG') for c in checks.values())
            cells.append(dict(arm=arm,seed=seed,status='OK' if ok else 'FAIL',checks=checks))
    result = dict(contract_only=True, scientific_labels=False,
                  status='OK' if all(c['status']=='OK' for c in cells) else 'FAIL', cells=cells)
    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'s14_analysis.json',result)
    (out/'s14-analysis.md').write_text('S14 test-only contract analysis: '+result['status']+'\nNo scientific labels.\n')
    print('S14_CONTRACT '+result['status'],flush=True)
    return result


def analyze_s14(root, out_dir, *, layout='R', legacy_store=None, manifest=None, test_contract=None):
    if test_contract is not None:
        return analyze_contract(root, out_dir, test_contract)
    if layout not in ('R', 'S'):
        raise ValueError('layout must be R or S')
    store = ReadStore(root)
    old = legacy_store or ReadStore(legacy=True)
    seeds = [3, 4, 5] if layout == 'R' else [0, 1, 2]
    result = dict(layout=layout, seeds=seeds, split_rule=SPLIT_RULE,
                  primary='crossfit', versions={})
    manifest = Path(manifest or DATA/'p2data/eval.jsonl')
    variants = [('all', (), ())]
    if manifest.exists():
        try:
            variants.append(('overlap_drop', overlap_rows(store.verify(manifest)), HO_OVERLAP))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            store.integrity.error = str(exc)
            result['overlap_status'] = dict(status='NOT-EVALUABLE',error=str(exc))
    else:
        result['overlap_status'] = dict(status='NOT-EVALUABLE', error=f'missing eval manifest: {manifest}')
    for variant, drop, hodrop in variants:
        curves, stores, cells = {}, {}, {}
        for s in seeds:
            stores[s] = {arm: old if layout == 'S' and arm in OLD_GRID else store for arm in ARMS}
            curves[s], cells[s] = {}, {}
            for arm, st in stores[s].items():
                try:
                    curves[s][arm] = st.curve(arm, s, drop=drop, hodrop=hodrop)
                    cells[s][arm] = dict(status='OK', files=[str(p) for p in curves[s][arm].files])
                except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                    cells[s][arm] = dict(status='NOT-EVALUABLE', error=f'{type(exc).__name__}: {exc}')
            if 'so' in curves[s]:
                for arm in list(curves[s]):
                    if not np.array_equal(curves[s][arm].rows, curves[s]['so'].rows):
                        cells[s][arm] = dict(status='NOT-EVALUABLE', error='paired arm ids differ from SO')
                        del curves[s][arm]
        versions = {}
        for cf in (False, True):
            mode = 'crossfit' if cf else 'in_sample'
            # In S, legacy FO-CE retains its N3B grid, without new adaptive stopping.
            blocks = []
            for s in seeds:
                if layout == 'S':
                    base = seed_block({a: curves[s][a] for a in OLD_GRID if a in curves[s]}, s, crossfit=cf, legacy=True)
                    new = seed_block(curves[s], s, crossfit=cf)
                    for a in base['selected']:
                        new['selected'][a] = base['selected'][a]
                    for a in ('fo-ce', 'fo-cm'):
                        for prefix in ('R_cov_', 'dfn_', 'lambda_star_'):
                            new[prefix+a] = base.get(prefix+a, None if prefix == 'lambda_star_' else np.nan)
                    new['x'] = base.get('x', np.nan)
                    for key, (left, right) in CONTRAST_ARMS.items():
                        if left in new['selected'] and right in new['selected']:
                            new[key] = new['selected'][left]['M']-new['selected'][right]['M']
                    # Remove flags for the replaced legacy selections.
                    new['flags'] = [f for f in new['flags'] if f.split(':')[0] not in OLD_GRID]+base['flags']
                    blocks.append(new)
                else:
                    blocks.append(seed_block(curves[s], s, crossfit=cf))
            for b in blocks:
                b['cells'] = cells[b['seed']]
                cell_checks(b, curves[b['seed']], stores[b['seed']])
            agg = aggregate(blocks, ARMS)
            apply_checks(agg, blocks)
            decision = dict(MINI_P=agg['labels']['mini-p'], WSD_3469=agg['labels']['wsd-3469'],
                            MINI_P_above_FO_CM=bool(agg['x_P']['lb90']>0 and ready(blocks, ('mini-p', 'fo-cm'))),
                            WSD_changes_paper=agg['labels']['wsd-3469']=='COVERS')
            decision['contrast_status'] = {key: 'OK' if ready(blocks, arms) and agg[key]['complete']
                                          else 'NOT-EVALUABLE (no reading)' for key, arms in CONTRAST_ARMS.items()}
            decision['required_cells'] = {key: [dict(arm=a, seed=s) for s in seeds
                for a in sorted(set(arms) | {'so'} | ({'fo-ce'} if set(arms) & {'fo-cm', 'wsd-3469'} else set()))]
                for key, arms in {**CONTRAST_ARMS, 'R1': ('so', 'fo-ce', 'fo-cm')}.items()}
            for key, status in decision['contrast_status'].items():
                agg[key]['status'] = 'OK' if status == 'OK' else 'NOT-EVALUABLE'
            if layout == 'R':
                decision['R1'] = ('REPLICATES' if agg['R_cov_fo-cm']['lb90']>ROPE
                                  and agg['labels']['fo-cm'].startswith('NOT-COVERED') else
                                  'NOT-EVALUABLE' if agg['labels']['fo-cm'].startswith('NOT-EVALUABLE') else 'INTERVAL')
                decision['x_wording'] = ('further' if agg['x']['ub90']<0 and ready(blocks, tuple(OLD_GRID))
                                         else 'no closer (no upgrade)')
                old_blocks = []
                for s in range(3):
                    old_curves = {}
                    for arm in OLD_GRID:
                        try:
                            old_curves[arm] = old.curve(arm, s, drop=drop, hodrop=hodrop)
                        except (OSError, ValueError, KeyError, TypeError, IndexError):
                            continue
                    ob = seed_block(old_curves, s, crossfit=cf, legacy=True)
                    cell_checks(ob, old_curves, {a: old for a in OLD_GRID})
                    old_blocks.append(ob)
                pooled = {}
                for key, arms in {'R_cov_fo-cm': ('so', 'fo-cm'), 'R_cov_fo-ce': ('so', 'fo-ce'),
                                  'x': ('fo-cm', 'fo-ce'), 'dfn_fo-cm': ('so', 'fo-cm')}.items():
                    pooled[key] = bounds([b.get(key, np.nan) for b in old_blocks+blocks])
                    pooled[key]['status'] = ('OK' if pooled[key]['complete'] and ready(
                        old_blocks+blocks, arms, uses_so_max=key != 'x') else 'NOT-EVALUABLE')
                    pooled[key]['required_cells'] = [dict(arm=a, seed=b['seed']) for b in old_blocks+blocks
                                                     for a in sorted(set(arms) | {'so', 'fo-ce'})]
            else:
                pooled = None
                decision.update(R1='NOT-APPLICABLE (layout S)', x_wording='no new-seed wording upgrade')
            versions[mode] = dict(per_seed=blocks, aggregate=agg, decisions=decision, pooled_six=pooled)
            if variant == 'all':
                versions[mode]['passkey'] = safe_passkey_analysis(store, blocks) if layout=='R' else dict(
                    reading='NOT-EVALUABLE (legacy checkpoints unavailable in layout S)', valid=False)
        versions['selection_optimism'] = {
            'M_'+a: bounds([b['selected'].get(a, {}).get('M', np.nan)-c['selected'].get(a, {}).get('M', np.nan)
                           for b, c in zip(versions['in_sample']['per_seed'], versions['crossfit']['per_seed'])])
            for a in ARMS}
        versions['top1'] = top1(curves)
        result['versions'][variant] = versions
    for mode in ('in_sample', 'crossfit'):
        a = result['versions']['all'][mode]
        if 'overlap_drop' in result['versions']:
            b = result['versions']['overlap_drop'][mode]
            a['overlap_sensitive'] = a['aggregate']['labels'] != b['aggregate']['labels']
        else:
            a['overlap_sensitive'] = None
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out/'s14_analysis.json', result)
    text = ['# S14 pulled-output analysis', '', f'Layout {layout}. Cross-fit primary; in-sample alongside.',
            'One-sided 90% t bounds over training seeds; pooled six uses T90=1.4759, not 1.8856.', '']
    keys = ['R_cov_fo-cm', 'x', 'R_cov_mini-p', 'dfn_mini-p', 'x_P',
            'R_cov_wsd-3469', 'dfn_wsd-3469', 'delta_steps', 'delta_shape',
            'lambda_star_fo-cm', 'lambda_star_mini-p', 'lambda_star_wsd-3469']
    for variant, versions in result['versions'].items():
        text += ['## '+variant, '']
        for mode in ('crossfit', 'in_sample'):
            v = versions[mode]
            text += ['### '+mode, '', table(v['aggregate'], keys), '',
                     '[Observation → registered rules] '+json.dumps(v['decisions']), '']
            if mode=='crossfit' and variant=='all':
                for block in v['per_seed']:
                    for arm,check in block['checks'].items():
                        batch=check.get('READS_DONE',{}).get('batch_check')
                        if batch:
                            text += [f"Batch check (descriptive): {arm} s{block['seed']}, "
                                     f"requested {batch['requested_batch']}, effective {batch['effective_batch']}, "
                                     f"{batch['status']}; {json.dumps(batch['values'],sort_keys=True)}."]
                if 'passkey' in v and 'per_seed' in v['passkey']:
                    for record in v['passkey']['per_seed']:
                        for arm,point in record['arms'].items():
                            batch=point.get('batch_check',{}).get('batch_check')
                            if batch:
                                text += [f"Passkey batch check (descriptive): {arm} s{record['seed']}, "
                                         f"requested {batch['requested_batch']}, effective {batch['effective_batch']}, "
                                         f"{batch['status']}; {json.dumps(batch['values'],sort_keys=True)}."]
            if v['pooled_six']:
                text += ['Pooled old sparse-grid plus new dense-grid seeds (descriptive heterogeneity caveat):', '',
                         table(v['pooled_six'], list(v['pooled_six'])), '']
            if 'passkey' in v:
                text += ['Passkey: '+v['passkey']['reading']+'. Descriptive; never changes frontier labels.', '']
    text += ['[Interpretation] WSD-3469 COVERS would make schedule dependence a main paper finding. '
             'MINI-P COVERS would qualify any broad claim against inexpensive partial-SO alternatives. '
             'Neither contrast identifies a general causal benefit of steps independently of tokens or schedule.',
             'Arrays, flags, passkey validity, lambda, ranks, missingness and source points: '
             '[s14_analysis.json](s14_analysis.json).', '']
    (out/'s14-analysis.md').write_text('\n'.join(text))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['a0', 's14'])
    p.add_argument('--root')
    p.add_argument('--layout', choices=['R', 'S'], default='R')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--process-name', default=os.environ.get('PROCESS_NAME', 'TTM-s14-31'))
    p.add_argument('--test-contract-json', help='explicit test-only counts/horizons; suppresses scientific labels')
    args = p.parse_args(argv)
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    if args.mode == 'a0':
        a0(args.out_dir)
    else:
        from .s14_test_contract import TestContract
        contract = TestContract(**json.loads(Path(args.test_contract_json).read_text())) if args.test_contract_json else None
        result = analyze_s14(args.root, args.out_dir, layout=args.layout, test_contract=contract)
        if contract and result['status'] != 'OK':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
