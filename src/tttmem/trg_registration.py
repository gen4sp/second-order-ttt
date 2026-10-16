"""Read the native dated stage-(i) addendum and its calibration sidecar."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]


def read_addendum(cfg,root=ROOT):
    path=root/cfg['trg_addendum_file']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not cfg.get('trg_addendum_sha256') or digest!=cfg['trg_addendum_sha256']:
        raise ValueError('TRG requires the exact stage-(i) addendum sha256')
    text=path.read_text()
    match=re.search(r'κ = (3|5|10);',text)
    calibration=json.loads((root/cfg['trg_calibration_file']).read_text())
    if match is None or int(match[1])!=cfg['trg_kappa'] or calibration.get('kappa')!=cfg['trg_kappa']:
        raise ValueError('configured kappa differs from stage-(i) addendum/calibration')
    if calibration.get('stage')!='i' or calibration.get('smoke_only') is not False or 'smoke only' in text:
        raise ValueError('smoke calibration cannot seal production')
    hashes=calibration.get('source_sha256',{})
    if not hashes:raise ValueError('stage-(i) calibration lacks source hashes')
    for name,expected in hashes.items():
        source=root/name if name.startswith('scripts/') else root/'src/tttmem'/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('stage-(i) calibration source differs from final pool source')
    return calibration
