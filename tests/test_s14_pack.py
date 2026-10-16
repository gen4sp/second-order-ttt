"""External C pins and the single-process production shard preparation path."""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from tttmem.p2 import data as D, s14_data

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from s14_test_support import tiny_store


def test_one_worker_uses_same_shard_spool_without_a_child(tmp_path,monkeypatch):
    monkeypatch.setattr(D,'load_tokenizer',lambda *a:object())
    monkeypatch.setattr(D,'ProcessPoolExecutor',lambda *a,**k:pytest.fail('single-process path spawned a child'))
    order=[]
    def shard(task):
        cfg,name,scratch,index=task
        order.append((name,index))
        spool=tmp_path/f'{index}.pkl'
        spool.write_bytes(pickle.dumps(dict(shard=name,_ids=np.array([index],dtype=np.uint32))))
        return str(spool),dict(scanned=index+1)
    monkeypatch.setattr(D,'_prepare_shard',shard)
    stats={};rows=list(D.iter_prepared_shards(D.PrepareCfg(),['b','a'],tmp_path,1,stats))
    assert order==[('b',0),('a',1)] and [r['shard'] for r in rows]==['b','a']
    assert stats=={'b':{'scanned':1},'a':{'scanned':2}}
    assert not list(tmp_path.glob('*.pkl'))


def _c_store(tmp_path):
    root=tmp_path/'p2data';expected=tiny_store(root)
    np.full((2,8193),99,dtype=np.uint32).tofile(root/'C.bin')
    (root/'C.jsonl').write_text(''.join(json.dumps(dict(row=i,sha1=f'C-{i}'))+'\n' for i in range(2)))
    (root/'extend-C.meta.json').write_text(json.dumps(dict(clock='original')))
    record=dict(counts={'C':2},sha256={n:D.sha256_file(root/n) for n in ('C.bin','C.jsonl','extend-C.meta.json')})
    (root/'extensions.json').write_text(json.dumps([record]))
    pin=tmp_path/'expected-C.json';pin.write_text(json.dumps(dict(counts={'C':2},sha256={
        n:D.sha256_file(root/n) for n in ('C.bin','C.jsonl','extend-C.meta.json','extensions.json')})))
    report=tmp_path/'report';report.mkdir()
    cfg=dict(data_dir=str(root),data=dict(mode='uploaded',rebuild_c=False,c_rows=2,
        expected_hashes=str(expected),expected_c_hashes=str(pin),audit_shingles=True))
    return root,report,cfg


def test_uploaded_metadata_is_pinned_and_rebuild_checks_fresh_record(tmp_path,monkeypatch):
    root,report,cfg=_c_store(tmp_path)
    result=s14_data.prepare_data(cfg,report,need_c=True)
    assert result['verification']['C']['contamination']['sampled_64gram_hits']==0
    # Self-consistent changed metadata cannot pass the uploaded external pin.
    (root/'extend-C.meta.json').write_text(json.dumps(dict(clock='fresh')))
    record=json.loads((root/'extensions.json').read_text())[0]
    record['sha256']['extend-C.meta.json']=D.sha256_file(root/'extend-C.meta.json')
    (root/'extensions.json').write_text(json.dumps([record]))
    with pytest.raises(ValueError,match='pinned split-C'):
        s14_data.prepare_data(cfg,report,need_c=True)
    # A rebuild keeps exact immutable C bytes while validating its own runtime metadata.
    cfg['data']['mode']='rebuild'
    monkeypatch.setattr(s14_data,'prepare_config',lambda *a:pytest.fail('verified base unnecessarily rebuilt'))
    assert s14_data.prepare_data(cfg,report,need_c=True)['verification']['verified']


@pytest.mark.parametrize('mode',['uploaded','rebuild'])
def test_external_c_pin_rejects_self_consistent_token_corruption(tmp_path,mode):
    root,report,cfg=_c_store(tmp_path)
    s14_data.prepare_data(cfg,report,need_c=True)
    cfg['data']['mode']=mode
    with (root/'C.bin').open('r+b') as f:f.write(np.array([98],dtype=np.uint32).tobytes())
    record=json.loads((root/'extensions.json').read_text())[0]
    record['sha256']['C.bin']=D.sha256_file(root/'C.bin')
    (root/'extensions.json').write_text(json.dumps([record]))
    with pytest.raises(ValueError,match='pinned split-C'):
        s14_data.prepare_data(cfg,report,need_c=True)
