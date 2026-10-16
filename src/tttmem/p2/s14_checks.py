"""Fail-closed training evidence checks for S14 job outputs (no torch)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .s14_protocol import N3B_SO_S, N3B_FO_PILOT_S, COMPUTE_FIELDS
from .guard import RunningMinGuard, running_min_rise

EQUAL_COMPUTE = ('fo-cm', 'mini-p', 'wsd-3469')


def _records(path):
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        # Trainer's *.jsonl contains TAG {JSON}, not bare JSON objects.
        if line.startswith('{'):
            rec = json.loads(line)
            tag = rec.pop('event')
        else:
            tag, raw = line.split(' ', 1)
            rec = json.loads(raw)
        records.append((tag, rec))
    return records


def timing_check(records, n, *, test_only=False):
    seconds, cumulative = {}, {}
    for tag, rec in records:
        if tag != 'P2_TIME':
            continue
        end, window = rec['step'], rec['step_s']
        if type(end) is not int or not window or len(window) > end or end > n:
            raise ValueError('invalid P2_TIME step window')
        for i, value in enumerate(window, end-len(window)+1):
            value = float(value)
            if not np.isfinite(value) or value <= 0 or (i in seconds and seconds[i] != value):
                raise ValueError(f'invalid/overlapping P2_TIME at step {i}')
            seconds[i] = value
        if end in cumulative and cumulative[end] != rec['train_s']:
            raise ValueError('conflicting P2_TIME cumulative train_s')
        cumulative[end] = float(rec['train_s'])
    if set(seconds) != set(range(1, n+1)):
        raise ValueError('P2_TIME seconds for every step 1..N (including final tail) required')
    for end, train_s in cumulative.items():
        if not np.isfinite(train_s) or not np.isclose(train_s,
                sum(seconds[i] for i in range(1,end+1)), rtol=1e-7, atol=1e-6):
            raise ValueError('P2_TIME train_s disagrees with step_s sum')
    if n < 60 and not test_only:
        raise ValueError('production timing requires steps 11..60')
    median = float(np.median([seconds[i] for i in (range(1,n+1) if n < 60 else range(11,61))]))
    return dict(status='OK', train_s=float(sum(seconds.values())), median_11_60_s=median,
                covered_steps=n, test_only=test_only)


def training_check(root, arm, seed, *, test_contract=None, verify=None):
    preset = test_contract.preset if test_contract else 'mini-m'
    directory = Path(root)/'runs'/preset/arm/f's{seed}'
    result = dict(status='OK', directory=str(directory), missing=[], repair=[], flags=[], compute=None)
    gdiv = []
    try:
        if verify:
            # These are exactly the training files consumed below, including all
            # failure markers; a corrupt marker is never silently skipped.
            paths = [directory/f for f in ('steps.jsonl','events.jsonl','MINI_CELL_DONE.json','compute.json')
                     if (directory/f).exists()]
            paths += [*directory.glob('guard_*.json'),*directory.glob('*abort.json')]
            if (directory/'gdiv.json').exists():
                paths.append(directory/'gdiv.json')
            for path in paths:
                verify(path)
        # Trip markers take precedence over missing success files.
        markers = [*directory.glob('guard_*.json'), *directory.glob('*abort.json')]
        if (directory/'gdiv.json').exists():
            markers.append(directory/'gdiv.json')
        for p in markers:
            content = json.loads(p.read_text())
            if p.name == 'gdiv.json':
                if not isinstance(content, list):
                    result['repair'].append('malformed gdiv.json')
                else:
                    gdiv.extend(content)
                    if len(content) > 1:
                        result['repair'].append('second G-div marker event')
            elif p.name != 'guard_state.json' or content.get('trip') or content.get('label'):
                result['repair'].append(str(p))
        required = ('steps.jsonl', 'events.jsonl', 'MINI_CELL_DONE.json')
        result['missing'] = [str(directory/f) for f in required if not (directory/f).is_file()]
        if result['missing']:
            if gdiv:
                result['repair'].append('G-div recovery not demonstrated (missing completion evidence)')
                result['gdiv'] = dict(status='REPAIR', recovered=False, events=gdiv)
            result['status'] = 'REPAIR' if result['repair'] else 'NOT-EVALUABLE'
            return result
        done = json.loads((directory/'MINI_CELL_DONE.json').read_text())
        final = done['final']
        if type(final['steps']) is not int or type(final['aborted']) is not bool:
            raise ValueError('completion steps/aborted must be integer/boolean')
        n = final['steps']
        expected_n = 1200 if arm in ('so', 'fo-ce', 'wsd-1200') else 3469 if arm in ('fo-cm', 'wsd-3469') else n
        if test_contract:
            expected_n = test_contract.expected_steps[arm]
        if (done.get('status') != 'done' or final['aborted']
                or final.get('guard_label') or final.get('label') or n != expected_n):
            result['repair'].append('incomplete/aborted/guard-labelled cell or wrong horizon')
        result['steps'] = n
        records = _records(directory/'steps.jsonl') + _records(directory/'events.jsonl')
        guards, losses = {}, {}
        for tag, rec in records:
            if tag in ('P2_ABORT', 'P2_GUARD_LABEL'):
                result['repair'].append(f'{tag} step {rec.get("step")}')
            if tag == 'P2_GDIV':
                gdiv.append(rec)
                if rec.get('action') != 'reload_last_resume_state':
                    result['repair'].append('G-div did not reload the last resume state')
            if tag == 'P2_STEP':
                losses[int(rec['step'])] = float(rec['loss'])
                if not all(np.isfinite(float(rec[k])) for k in ('loss', 'grad_norm')):
                    result['repair'].append(f'nonfinite training step {rec.get("step")}')
            if tag == 'P2_GUARD':
                if (type(rec['step']) is not int or type(rec['trip']) is not bool
                        or type(rec['finite']) is not bool):
                    raise ValueError('guard step/trip/finite must be integer/booleans')
                step = int(rec['step'])
                if step in guards and guards[step] != rec:
                    raise ValueError(f'conflicting guard records at {step}')
                guards[step] = rec
                if rec['trip'] or not rec['finite'] or not np.isfinite(float(rec['value'])):
                    result['repair'].append(f'guard trip/nonfinite at {step}')
        # N3B cadence: every 50 steps from 50, no dense phase.
        cadence = test_contract.guard_every if test_contract else 50
        if set(guards) != set(range(cadence, n+1, cadence)):
            result['missing'].append('complete P2_GUARD cadence (50,100,...,floor(N/50)*50)')
        result['guard_probes'] = len(guards)
        guard = RunningMinGuard(max_rise=.10, consecutive=2)
        for step, rec in sorted(guards.items()):
            if guard.update(step, float(rec['value']))['trip']:
                result['repair'].append(f'recomputed guard trip at {step}')
        if not losses:
            result['missing'].append('P2_STEP loss/grad_norm evidence')
        history = []
        for step, loss in sorted(losses.items()):
            if running_min_rise(history, loss) > .5:
                result['repair'].append(f'recomputed train-loss guard at {step}')
            history.append(loss)
        result['timing'] = timing_check(records, n, test_only=test_contract is not None)
        # SOQ-N3B stops on guard trips. G-div recovery is specified in the N3B
        # config/runner (one reload) but not classified there: S14 registers
        # one recovered reload as FLAG; a second or unrecovered one as REPAIR.
        events = {}
        for rec in gdiv:
            if (not isinstance(rec, dict) or type(rec.get('event')) is not int
                    or type(rec.get('step')) is not int or rec['event'] < 1
                    or not 1 <= rec['step'] <= n):
                result['repair'].append('malformed G-div evidence')
                continue
            event = rec['event']
            if event in events and events[event]['step'] != rec['step']:
                result['repair'].append('conflicting G-div evidence')
            events[event] = rec
        if events:
            recovered = (set(events) == {1} and not result['repair'] and not result['missing']
                         and done.get('status') == 'done' and final['aborted'] is False)
            result['gdiv'] = dict(events=list(events.values()), recovered=recovered,
                                  status='FLAG' if recovered else 'REPAIR')
            if recovered:
                result['flags'].append('one recovered G-div reload (reported)')
            else:
                result['repair'].append('second or unrecovered G-div')
        if arm in EQUAL_COMPUTE or (directory/'compute.json').exists() or test_contract:
            cp = directory/'compute.json'
            if not cp.is_file():
                result['missing'].append(str(cp))
            else:
                c = json.loads(cp.read_text())
                result['compute'] = dict(status='FAIL', source=str(cp))
                if not set(COMPUTE_FIELDS) <= c.keys():
                    raise ValueError('compute.json v2 missing fields: '+str(sorted(set(COMPUTE_FIELDS)-c.keys())))
                for key in ('seed', 'base_steps', 'expected_steps', 'micro_batch', 'pilot_micro_batch'):
                    if type(c[key]) is not int or c[key] < (0 if key == 'seed' else 1):
                        raise ValueError(f'compute.json {key} must be an integer')
                for key in ('so_micro_batch', 'pilot_so_micro_batch'):
                    if c[key] is not None and (type(c[key]) is not int or c[key] < 1):
                        raise ValueError(f'compute.json {key} must be a positive integer or null')
                for key in ('device_name', 'torch_version'):
                    if not isinstance(c[key], str) or not c[key].strip():
                        raise ValueError(f'compute.json {key} must be a nonempty string')
                for key in ('cuda_version', 'runpod_pod_id'):
                    if c[key] is not None and (not isinstance(c[key], str) or not c[key].strip()):
                        raise ValueError(f'compute.json {key} must be a nonempty string or null')
                if c['arm'] != arm or c['seed'] != seed or c['base_steps'] != 1200 or c['expected_steps'] != n:
                    raise ValueError('compute.json cell identity/horizon mismatch')
                if (c['micro_batch'] != c['pilot_micro_batch'] or (arm == 'mini-p' and
                        (c['so_micro_batch'] is None or c['so_micro_batch'] != c['pilot_so_micro_batch']))):
                    raise ValueError('compute.json micro-batch does not match timing basis')
                if any(type(c[k]) not in (int, float) for k in ('t_so_pilot_s', 't_arm_pilot_s')):
                    raise ValueError('compute pilot seconds must be JSON numbers')
                tso, ta = float(c['t_so_pilot_s']), float(c['t_arm_pilot_s'])
                if not np.isfinite([tso, ta]).all() or min(tso, ta) <= 0:
                    raise ValueError('invalid compute pilot seconds')
                if arm == 'mini-p':
                    if c['pilot_source'] != 's14-pilot' or (not test_contract and round(1200*tso/ta) != n):
                        raise ValueError('MINI-P requires its registered S14 pilot horizon')
                elif arm in EQUAL_COMPUTE and not test_contract and (c['pilot_source'] != 'n3b-50' or c['pilot_micro_batch'] not in N3B_FO_PILOT_S
                      or not np.allclose([tso, ta], [N3B_SO_S, N3B_FO_PILOT_S[c['pilot_micro_batch']]],
                                         rtol=0, atol=1e-12)):
                    raise ValueError('FO equal-compute reference must match N3B pilot for actual micro-batch')
                planned = n*ta/(1200*tso)
                seconds, cumulative = {}, {}
                for tag, rec in records:
                    if tag != 'P2_TIME':
                        continue
                    end, window = int(rec['step']), rec['step_s']
                    if not window or len(window) > end or end > n:
                        raise ValueError('invalid P2_TIME step window')
                    for i, value in enumerate(window, end-len(window)+1):
                        value = float(value)
                        if not np.isfinite(value) or value <= 0 or (i in seconds and seconds[i] != value):
                            raise ValueError(f'invalid/overlapping P2_TIME at step {i}')
                        seconds[i] = value
                    if end in cumulative and cumulative[end] != rec['train_s']:
                        raise ValueError('conflicting P2_TIME cumulative train_s')
                    cumulative[end] = float(rec['train_s'])
                if set(seconds) != set(range(1, n+1)):
                    result['missing'].append('P2_TIME seconds for every step 1..N (including final tail)')
                else:
                    total = float(sum(seconds.values()))
                    for end, train_s in cumulative.items():
                        if not np.isfinite(train_s) or not np.isclose(train_s, sum(seconds[i] for i in range(1, end+1)), rtol=1e-7, atol=1e-6):
                            raise ValueError('P2_TIME train_s disagrees with step_s sum')
                    median = result['timing']['median_11_60_s']
                    speed = median/ta
                    raw = total/(tso*1200)
                    ratio = raw/speed
                    result['compute'] = dict(status='PASS', planned_ratio=planned,
                                             r_act_over_r=ratio, raw_ratio=raw, speed_factor=speed,
                                             median_11_60_s=median, train_s=total, target_s=tso*1200,
                                             pilot_ratio=tso/ta, source=str(cp),
                                             record=c,
                                             blocks_not_covered=arm in EQUAL_COMPUTE and min(round(ratio, 12), round(planned, 12)) < .95,
                                             blocks_covers=arm in EQUAL_COMPUTE and max(round(ratio, 12), round(planned, 12)) > 1.05,
                                             actual_ratio_wording=max(abs(ratio-1), abs(planned-1)) > .02)
        result['status'] = 'REPAIR' if result['repair'] else 'NOT-EVALUABLE' if result['missing'] else 'OK'
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        if gdiv and not result.get('gdiv', {}).get('recovered', False):
            result['repair'].append('G-div recovery not demonstrated (malformed evidence)')
        result.update(status='REPAIR' if result['repair'] else 'NOT-EVALUABLE',
                      error=f'{type(exc).__name__}: {exc}')
    return result


def apply_checks(aggregate, blocks):
    """Preserve numerical diagnostics; prevent claims when their evidence is blocked."""
    for arm, label in aggregate['labels'].items():
        # FO-CE is a required comparator for schedule/replication evidence.
        required = list(dict.fromkeys(('so', arm)+(('fo-ce',) if arm in ('fo-cm', 'wsd-3469') else ())))
        aggregate.setdefault('required_cells', {})[arm] = [
            dict(arm=a, seed=b.get('seed')) for b in blocks for a in required]
        checks = [b['checks'].get(a, dict(status='NOT-EVALUABLE')) for b in blocks for a in required]
        if any(c['status'] == 'REPAIR' for c in checks):
            aggregate['labels'][arm] = 'NOT-EVALUABLE (repair; no reading)'
        elif any(c['status'] != 'OK' for c in checks):
            aggregate['labels'][arm] = 'NOT-EVALUABLE (missing training/H_LM evidence)'
        elif label.startswith('NOT-COVERED') and any(c.get('compute', {}).get('blocks_not_covered') for c in checks if c.get('compute')):
            aggregate['labels'][arm] = 'NOT-EVALUABLE (compute <.95; NOT-COVERED blocked)'
        elif label.startswith('COVERS') and any(c.get('compute', {}).get('blocks_covers') for c in checks if c.get('compute')):
            aggregate['labels'][arm] = 'NOT-EVALUABLE (compute >1.05; COVERS blocked)'
