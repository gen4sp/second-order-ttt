"""Range-safe POOL storage of the calibrated TRG logical event schema.

Native readers inspect hashes/coverage and load NPZ dtypes without coercing them.
Keep that logical schema for interoperability; record physical dtype overrides
in additive manifest metadata. The calibrated writer remains byte-locked.
"""
import json

import numpy as np
from . import dense_log as dense
from .exp_chr import _finite

FLAGS = {'correct', 'duplicate', 'chunks', 'query_order'}
# Only these mathematically bounded floats use the native compact storage.
# Every other float (norms, CE, margins, rung/term contributions and future keys)
# is unbounded and gets float32. Telemetry JSON already retains Python floats.
BOUNDED = {'probability', 'residual', 'maximum_key_cosine', 'address_margin', 'inner_clip_scale'}


def nonfinite_fields(value, prefix=''):
    """Find missing telemetry without changing training-update finite counters."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_fields(v, f'{prefix}.{k}' if prefix else k)]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in nonfinite_fields(v, f'{prefix}[{i}]')]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return [prefix]
    return []


class PoolDenseLogger(dense.DenseLogger):
    def append(self, step, read):
        fields = [key for key, values in read['events'].items() if not np.isfinite(values).all()]
        fields += nonfinite_fields(read['summary'], 'summary')
        summary = _finite(read['summary'])
        if fields:
            summary = dict(summary, telemetry_nonfinite=True, nonfinite_fields=fields)
        super().append(step, dict(read, summary=summary))
        return summary

    def flush(self):
        if not self.rows:
            return
        out = dict(steps=np.array([s for s,_ in self.rows], np.int32),
                   ladder_steps=np.array([s for s,r in self.rows if r['ladder']], np.int32))
        out['telemetry_nonfinite'] = np.array([r['summary'].get('telemetry_nonfinite', False)
                                             for _, r in self.rows], dtype=np.bool_)
        for key in sorted(set().union(*(r['events'] for _,r in self.rows))):
            data = np.stack([r['events'][key] for _,r in self.rows if key in r['events']])
            invalid = ~np.isfinite(data)
            if invalid.any():
                # NPZ floats represent missing values as NaN plus an explicit
                # element mask; JSON summary values are null. Never retry a main
                # or job because its measured event/telemetry value is non-finite.
                out[key+'__nonfinite'] = invalid
                data = np.where(invalid, np.nan, data)
            dtype = np.int8 if key in FLAGS and not invalid.any() else np.float16 if key in BOUNDED else np.float32
            if dtype != np.int8 and (np.abs(data) > np.finfo(dtype).max).any():
                # Range alone never creates an outcome-correlated failure.
                dtype = np.float32 if ((np.abs(data) <= np.finfo(np.float32).max) | invalid).all() else np.float64
            out[key] = data.astype(dtype)
        name = f'dense-{self.rows[0][0]:06d}-{self.rows[-1][0]:06d}.npz'
        path = self.folder/name
        np.savez_compressed(path, **out)
        self.shards.append(dict(file=name, sha256=dense.sha256_file(path), steps=out['steps'].tolist(),
            ladder_steps=out['ladder_steps'].tolist(), fields=sorted(out),
            dtypes={k:str(v.dtype) for k,v in out.items()}))
        self.rows.clear()

    def close(self):
        manifest = super().close()
        manifest['storage_policy'] = dict(unbounded_float='float32', bounded_float='float16',
                                         bounded_keys=sorted(BOUNDED), finite_range_fallback='float64',
                                         nonfinite_event='nan-with-mask', nonfinite_summary='null')
        (self.folder/'shards.json').write_text(json.dumps(manifest, indent=2))
        return manifest
