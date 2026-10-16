"""Rebuild a pinned first-shard slice with real tokenizer; compare local binary prefixes.

Uses production quotas (small quotas would change _split_for), but at most 64 eligible source
documents. The temporary sparse row files are removed after the check. No torch import.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

from tttmem.p2.data import (_set_process_name, _TokenizerBatch, excluded_vocab, iter_parquet,
                            load_tokenizer, prepare, open_split)


def check(local, out, n_docs=64):
    local=Path(local)
    meta=json.loads((local/'meta.json').read_text())
    first=json.loads((local/'A.jsonl').open().readline())
    started=time.monotonic()
    tok=load_tokenizer(meta['tokenizer']['repo'],meta['tokenizer']['revision'])
    with tempfile.TemporaryDirectory(prefix='TTM-s14-slice-') as tmp:
        root=Path(tmp)
        source=iter_parquet(meta['dataset']['repo'],meta['dataset']['revision'],meta['seed'],
                            root/'shard',shards=[first['shard']])
        docs=itertools.islice((d for d in source if len(d['text'])>=meta['min_chars']),n_docs)
        actual=prepare(docs,_TokenizerBatch(tok),root/'rebuilt',meta['quotas'],
                       min_chars=meta['min_chars'],seed=meta['seed'],
                       near_dup_stride=meta['near_dup_stride'],near_dup_n=meta['near_dup_n'],
                       workers=1,batch_size=meta['batch_size'],exclude=excluded_vocab(tok),log_every=0,
                       tokenizer_repo=meta['tokenizer']['repo'],tokenizer_revision=meta['tokenizer']['revision'],
                       dataset_repo=meta['dataset']['repo'],dataset_revision=meta['dataset']['revision'],
                       manifest_chars_per_token=False)
        source.close()
        checks=[]
        for split,count in actual['counts'].items():
            if not count:continue
            nbytes=count*8193*4
            with (local/f'{split}.bin').open('rb') as f:expected=f.read(nbytes)
            with (root/'rebuilt'/f'{split}.bin').open('rb') as f:observed=f.read(nbytes)
            checks.append(dict(split=split,rows=count,bytes=nbytes,byte_match=observed==expected,
                               local_sha256=hashlib.sha256(expected).hexdigest(),
                               rebuilt_sha256=hashlib.sha256(observed).hexdigest()))
        assert any(c['split']=='A' and c['rows']>0 for c in checks)
        if not all(c['byte_match'] for c in checks):
            raise ValueError(f'first-shard data determinism mismatch: {checks}')
        a_man=[json.loads(s) for s in (root/'rebuilt'/'A.jsonl').read_text().splitlines()]
        with (local/'A.jsonl').open() as f:registered=[json.loads(next(f)) for _ in a_man]
        if a_man!=registered:raise ValueError('registered first A manifests differ')
    factor=1 if sys.platform=='darwin' else 1024
    result=dict(status='byte-match',shard=first['shard'],dataset_revision=meta['dataset']['revision'],
                tokenizer_revision=meta['tokenizer']['revision'],source_eligible_docs=n_docs,
                checks=checks,A_manifest_match=True,wall_s=time.monotonic()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor,
                scope='First shard slice only; full-store 13-file hash verification occurs in every S14 job.')
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    print('S14_DATA_SLICE '+json.dumps(result),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--local',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--process-name',default=os.environ.get('PROCESS_NAME'))
    args=ap.parse_args();_set_process_name(args.process_name);check(args.local,args.out)


if __name__=='__main__':main()
