import hashlib
import json

import pytest
from tttmem.trg_registration import read_addendum


def test_native_addendum_and_calibration_are_bound(tmp_path):
    source=tmp_path/'src/tttmem/exp_trg.py';source.parent.mkdir(parents=True)
    source.write_text('registered source')
    addendum=tmp_path/'addendum.md';addendum.write_text('Stage (i), legacy seeds; exploratory.\nκ = 5; qualifying units = 0.\n')
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps(dict(stage='i',smoke_only=False,kappa=5,
        source_sha256={'exp_trg.py':hashlib.sha256(source.read_bytes()).hexdigest()})))
    cfg=dict(trg_addendum_file='addendum.md',trg_calibration_file='calibration.json',trg_kappa=5,
             trg_addendum_sha256=hashlib.sha256(addendum.read_bytes()).hexdigest())
    assert read_addendum(cfg,tmp_path)['kappa']==5
    with pytest.raises(ValueError,match='sha256'):read_addendum({**cfg,'trg_addendum_sha256':''},tmp_path)
    with pytest.raises(ValueError,match='kappa'):read_addendum({**cfg,'trg_kappa':3},tmp_path)
    source.write_text('modified source')
    with pytest.raises(ValueError,match='source'):read_addendum(cfg,tmp_path)


def test_smoke_addendum_cannot_seal(tmp_path):
    (tmp_path/'a.md').write_text('κ = 5; smoke only; cannot seal')
    (tmp_path/'c.json').write_text(json.dumps(dict(stage='i',kappa=5,smoke_only=True)))
    cfg=dict(trg_addendum_file='a.md',trg_calibration_file='c.json',trg_kappa=5,
             trg_addendum_sha256=hashlib.sha256((tmp_path/'a.md').read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='smoke'):read_addendum(cfg,tmp_path)
