"""Synthetic store, hosts and tokenizer for CPU tests; all scoring and file writers are real.

Never imported by the production package or production configs.
"""
import json
import re
from pathlib import Path

import numpy as np


def tiny_store(root, quotas=None):
    """Write a synthetic uploaded store (all splits, metadata, hashes) and its expected-hash file."""
    from tttmem.p2.data import sha256_file
    root.mkdir(parents=True)
    rng = np.random.default_rng(17)
    quotas = quotas or dict(A=48, B=32, eval=4, ho=2, calib=2)
    for split, n in quotas.items():
        rows = rng.integers(3, 97, size=(n,8193),dtype=np.uint32)
        rows[:,0] = 1
        rows[:,3:7] = [11,12,13,14]
        rows[:,35:39] = [11,12,13,14]
        rows[:,9:13] = [21,22,23,24]
        rows[:,41:45] = [21,22,23,24]
        rows[:,131:135] = [11,12,13,14]
        rows[:,137:141] = [21,22,23,24]
        rows.tofile(root/f'{split}.bin')
        (root/f'{split}.jsonl').write_text(''.join(json.dumps(dict(row=i,sha1=f'{split}-{i}',
                                  domain=f'{split}{i}.example'))+'\n' for i in range(n)))
    np.save(root/'excluded_vocab.npy',np.zeros(97,dtype=bool))
    (root/'meta.json').write_text(json.dumps(dict(quotas=quotas,counts=quotas,seq=8193)))
    hashes={p.name:sha256_file(p) for p in root.iterdir() if p.suffix in ('.bin','.jsonl','.npy')}
    (root/'prepare_summary.json').write_text(json.dumps(dict(complete=True,counts=quotas,sha256=hashes)))
    hashes.update({n:sha256_file(root/n) for n in ('meta.json','prepare_summary.json')})
    expected=root.parent/'tiny-expected.json'
    expected.write_text(json.dumps(dict(sha256=hashes,counts=quotas)))
    return expected


class TinyDigitTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in re.findall(r'\d+|[A-Za-z]+|[^\w\s]', text):
            if word.isdigit() and len(word)==5:
                if text.endswith(' '+word):
                    ids.append(96)
                ids.extend([3+int(word[:3])%40, 43+int(word[3:])%40])
            else:
                ids.append(3+sum(map(ord,word))%80)
        if text == ' ':
            ids = [96]
        return type('Encoded', (), dict(ids=ids))()


def install(data, hosts_file):
    """Test-only content substitution. Preserve the real public passkey module."""
    from tttmem.p2 import passkey
    from tttmem.p2.model import P2Config, PRESETS
    PRESETS['tiny'] = P2Config(vocab_size=97, hidden_size=16, intermediate_size=24,
        num_layers=3,num_heads=2,chunk=32,window=256,suffix_len=1,bos_id=1,eos_id=2,max_len=256)
    passkey.HOSTS_FILE = Path(hosts_file)
    if not passkey.HOSTS_FILE.exists():
        passkey.freeze_hosts(data,passkey.HOSTS_FILE)
    passkey._tokenizer = lambda *args: TinyDigitTokenizer()


def overrides():
    return dict(seq_len=256,segment_len=64,barriers=[2,4,6],global_batch=16,
        micro_batch=4,n_eval=8,ho_docs=8,ho_trajectory_docs=2,trajectory_docs=8,pilot_docs=2,
        warmup=1,inner_warmup=1,lr_peak=.001,eval_at_init=False,
        eval_modes=['adapt','noadapt','fP_focal','fP_nonfocal'],
        trajectory_modes=['adapt','noadapt','fP_focal','fP_nonfocal'],
        full_extra_modes=[],log_every=3,resume_every=3,guard_every=2,guard_first=2)


def contract(hosts_file, steps=4):
    from tttmem.p2.s14_test_contract import TestContract
    from tttmem.p2.s14 import ARMDIRS
    return TestContract(test_only=True,expected_steps={a:steps for a in ARMDIRS},preset='tiny',
        seeds=(3,),guard_every=2,n_docs=8,n_ho_adapt=8,ho_docs=8,passkey_docs=4,
        passkey_hosts_file=str(hosts_file))


def force_batch_fallback():
    """Test-only forced failures with coherent reported measurements."""
    from tttmem.p2 import s14_batch
    frontier, passkey = s14_batch.frontier_comparison, s14_batch.passkey_comparison
    def fail_frontier(*args):
        result=frontier(*args)
        result['values']['mean_M']=.001
        result['status']='FALLBACK'
        return result
    def fail_passkey(*args):
        result=passkey(*args)
        result['values']['beyond']['mean_lift']=.002
        result['status']='FALLBACK'
        return result
    s14_batch.frontier_comparison=fail_frontier
    s14_batch.passkey_comparison=fail_passkey
