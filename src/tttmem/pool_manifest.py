"""Freeze enrollment before any continuation outcome; preserve it across resumes."""
import json
from pathlib import Path

from . import frs_state as state
from .frs_jobs import job_record


def path_independent(value):
    if isinstance(value, list):
        return [path_independent(v) for v in value]
    if isinstance(value, dict):
        return {k: path_independent(v) for k,v in value.items()
                if k not in ('file', 'state_file', 'output_dir', 'provenance')}
    return value


def _reused_result(root, spec):
    """Completed continuation evidence can survive loss of its source checkpoint."""
    for folder in (root/'jobs'/spec['job_id'], root/'jobs'/'retry'/spec['job_id']):
        try:
            rec = json.loads((folder/'result.json').read_text())
            old = json.loads((folder/'spec.json').read_text())
            if spec.get('sham_from') and old.get('skip_controller') == 'tttmem.frs_jobs:ScheduledSkip':
                old['skip_controller'] = spec['skip_controller']  # generated from the verified yoked list
            final = folder/Path(rec['final_checkpoint']).name
            drops = folder/Path(rec['drop_file']).name
            if (rec['status'] != 'DONE' or path_independent(old) != path_independent(spec)
                    or rec['state_sha256'] != spec['state_sha256']
                    or state.file_sha256(final) != rec['final_sha256']
                    or state.file_sha256(drops) != rec['drop_sha256']):
                continue
            rec.update(final_checkpoint=str(final), drop_file=str(drops))
            for row in rec.get('states', []):
                row['file'] = str(folder/Path(row['file']).name)
                if state.file_sha256(row['file']) != row['sha256']:
                    raise ValueError('completed saved-state digest mismatch')
            return rec
        except (OSError, ValueError, KeyError):
            continue
    return None


def freeze(root, rebuilt, jobs, *, resume):
    root = Path(root)
    path = root/'job_manifest.json'
    if not path.exists():
        state.atomic_json(path, rebuilt, exclusive=True)
        return rebuilt, jobs, {}, []
    if not resume:
        raise FileExistsError('job manifest already frozen; enable resume')
    frozen = json.loads(path.read_text())
    events = []
    for key in ('h5_cohort', 'h5_slots', 'route_cohorts', 'states', 'jobs', 'trg_dispositions'):
        if path_independent(frozen.get(key)) != path_independent(rebuilt.get(key)):
            events.append(dict(event='MANIFEST_DRIFT', field=key,
                               frozen=path_independent(frozen.get(key)),
                               rebuilt=path_independent(rebuilt.get(key))))
    byid = {j.job_id: j for j in jobs}
    runnable, prior = [], {}
    for spec in frozen['jobs']:
        job = byid.get(spec['job_id'])
        if job is not None and path_independent(job_record(job)) == path_independent(spec):
            runnable.append(job)
            continue
        rec = _reused_result(root, spec)
        prior[spec['job_id']] = rec or dict(job_id=spec['job_id'], cell_id=spec['cell_id'],
            seed=spec['metadata'].get('seed'), arm=spec['metadata'].get('arm'), rule=spec['rule'],
            metadata=spec['metadata'], status='MISSING', disposition='TRUNCATED',
            reason='MANIFEST_DRIFT: frozen source unavailable or changed')
        events.append(dict(event='MANIFEST_DRIFT', job_id=spec['job_id'],
                           action='reuse verified completed result' if rec else 'keep frozen job missing'))
    if events:
        with (root/'manifest_drift.jsonl').open('a') as stream:
            for event in events:
                line = json.dumps(event, sort_keys=True)
                stream.write(line+'\n')
                print('MANIFEST_DRIFT '+line, flush=True)
    return frozen, runnable, prior, events
