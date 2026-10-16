"""Pulled-file integrity and checkpoint/config bindings, without torch imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .s14_batch import validate_check
from .s14_protocol import SO_GRID, FO_GRID, PASSKEY_ETAS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class PullIntegrity:
    def __init__(self,root):
        self.root = Path(root)
        self.files, self.error = None, None
        self._digests = {}
        path = self.root/'manifest.json'
        if path.exists():
            try:
                manifest = json.loads(path.read_text())
                self.files = {}
                for rec in manifest['files']:
                    name = rec['path']
                    if (Path(name).is_absolute() or '..' in Path(name).parts or name in self.files
                            or len(rec['sha256']) != 64):
                        raise ValueError('invalid/duplicate pull manifest entry')
                    self.files[name] = rec['sha256']
                done = self.root/'DONE'
                if done.exists() and json.loads(done.read_text())['manifest_sha256'] != sha256(path):
                    raise ValueError('DONE manifest hash mismatch')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                self.error = str(exc)

    @property
    def present(self):
        return self.files is not None or self.error is not None

    def digest(self,path):
        path=Path(path)
        stat=path.stat()
        identity=(stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        old=self._digests.get(path)
        if old is None or old[0] != identity:
            self._digests[path]=(identity,sha256(path))
        return self._digests[path][1]

    def verify(self,path):
        if self.error:
            raise ValueError(self.error)
        if self.files is None:
            return Path(path)
        path = Path(path)
        # Only pulled inputs are governed by this manifest; external frozen hosts
        # and original data manifests retain their own protocol checks.
        try:
            name = str(path.relative_to(self.root))
        except ValueError:
            return path
        if name not in self.files or self.digest(path) != self.files[name]:
            raise ValueError(f'pull manifest missing/hash mismatch: {name}')
        return path


def read_evidence(store,arm,seed,*,kind='frontier',required=False):
    directory = store.directory(arm,seed)
    marker_path = directory/'READS_DONE.json'
    verifier = store.integrity
    preset = store.test_contract.preset if store.test_contract else 'mini-m'
    cell = store.root/'.cell-configs'/f'{preset}-{arm}-s{seed}.json'
    if not marker_path.exists() and cell.exists():
        base = json.loads(verifier.verify(cell).read_text())
        if not isinstance(base,dict):
            raise ValueError('malformed cell configuration')
        suffix = 'reads' if kind=='frontier' else 'passkey'
        job = store.root/'.job-configs'/f"{base.get('process_name','')}-{suffix}.json"
        if job.exists():
            config = json.loads(verifier.verify(job).read_text())
            if config.get('read_batch' if kind=='frontier' else 'batch',1) > (2 if kind=='frontier' else 1):
                required = True
    if not marker_path.exists() and not (required or verifier.present):
        return dict(status='OK',legacy_without_manifest=True)
    marker = json.loads(verifier.verify(marker_path).read_text())
    config = marker['config']
    item = config['item']
    run = store.root/'runs'/preset/arm/f's{seed}'
    base = json.loads(verifier.verify(cell).read_text())
    checkpoint = run/f"model_step{base['steps']:04d}.pt"
    digest = verifier.digest(verifier.verify(checkpoint))
    if (marker['checkpoint_sha256'] != digest or marker['cell_config_sha256'] != verifier.digest(cell)
            or marker['config_sha256'] != fingerprint(config) or item['overrides'] != base
            or item['step'] != base['steps'] or item['init_seed'] != seed
            or item['cell'] != f'{preset}/{arm}/s{seed}'
            or Path(item['ckpt']).name != checkpoint.name or Path(item['cell_config']).name != cell.name
            or marker.get('complete') is not True or marker['read_list'] != config['reads']
            or set(marker['npz_sha256']) != set(marker['read_list'])):
        raise ValueError('incomplete/changed READS_DONE checkpoint/config evidence')
    if base.get('process_name'):
        job=store.root/'.job-configs'/f"{base['process_name']}-reads.json"
        if job.exists():
            actual=json.loads(verifier.verify(job).read_text())
            if actual['items'] != [item] or actual['read_batch'] != config['read_batch']:
                raise ValueError('READS_DONE configuration differs from the read worker configuration')
        job=store.root/'.job-configs'/f"{base['process_name']}-passkey.json"
        if kind=='passkey' and job.exists():
            actual=json.loads(verifier.verify(job).read_text())
            if (actual['batch'] != config['passkey_batch'] or actual['n_docs'] != config['passkey_docs']
                    or actual['seed'] != config['passkey_seed']):
                raise ValueError('READS_DONE configuration differs from the passkey worker configuration')
    group = 'so' if arm=='so' else 'fo'
    tc = store.test_contract
    grid = tc.grids.get(group,SO_GRID if arm=='so' else FO_GRID) if tc else SO_GRID if arm=='so' else FO_GRID
    etas = tc.passkey_etas.get(group,PASSKEY_ETAS[group]) if tc else PASSKEY_ETAS[group]
    name = lambda e:'ce' if e==1 else f'eta{round(100*e):03d}'
    expected = ([f'g0_read_{name(e)}.npz' for e in grid]+['g0_hlm.npz'] if kind=='frontier' else
                [f'passkey_{name(e)}.npz' for e in etas]+['passkey_noadapt.npz'])
    if not set(expected) <= set(marker['read_list']):
        raise ValueError('READS_DONE missing registered outputs')
    prefix = 'g0_' if kind=='frontier' else 'passkey_'
    for name in marker['read_list']:
        if Path(name).name != name:
            raise ValueError('invalid read filename')
        if name.startswith(prefix):
            path = verifier.verify(directory/name)
            if verifier.digest(path) != marker['npz_sha256'][name]:
                raise ValueError(f'READS_DONE output hash mismatch: {name}')
            with np.load(path,allow_pickle=False) as z:
                claimed = str(z['ckpt_sha256']) if kind=='frontier' else json.loads(str(z['metadata']))['model_sha256']
                if claimed != digest:
                    raise ValueError(f'read checkpoint hash mismatch: {name}')
    batch = config['read_batch' if kind=='frontier' else 'passkey_batch']
    baseline = 2 if kind=='frontier' else 1
    comparison = None
    if batch > baseline:
        path = verifier.verify(directory/'batch_check.json')
        checks = json.loads(path.read_text())
        if checks != marker['batch_check'] or verifier.digest(path) != marker['batch_check_sha256']:
            raise ValueError('batch_check disagrees with READS_DONE')
        comparison = validate_check(checks[kind],kind,batch,digest,verifier.digest(cell))
        tc = store.test_contract
        counts = (dict(n_docs=tc.n_docs if tc else 1024,n_ho=tc.n_ho_adapt if tc else 128,
                       n_hlm=tc.ho_docs if tc else 256) if kind=='frontier' else
                  dict(n_docs=min(32,tc.passkey_docs if tc else 256)))
        if any(comparison[k] != v for k,v in counts.items()):
            raise ValueError(f'{kind} batch check document counts changed')
    return dict(status='OK',requested_batch=batch,batch_check=comparison)
