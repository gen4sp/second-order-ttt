"""Real sequential S14 job on a tiny CPU model, with test-only content providers."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from tttmem.p2.data import _set_process_name, sha256_file
from tttmem.p2 import s14, s14_analysis
import s14_test_support as support
from s14_test_support import tiny_store


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--process-name',required=True)
    p.add_argument('--canary',action='store_true',help='rehearse canary stages with tiny CPU counts')
    p.add_argument('--read-batch',type=int,default=2)
    p.add_argument('--passkey-batch',type=int,default=1)
    p.add_argument('--force-batch-fallback',action='store_true',help='test-only comparison perturbation')
    args=p.parse_args();_set_process_name(args.process_name)
    root=Path(args.out)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    expected=tiny_store(root/'p2data',quotas=dict(A=96,B=32,eval=12,ho=8,calib=8))
    hosts=root/'tiny-hosts.json'
    injection=root/'test-content';injection.mkdir()
    (injection/'sitecustomize.py').write_text(
        f'import s14_test_support as s\ns.install({str(root/"p2data")!r},{str(hosts)!r})\n'
        + ('s.force_batch_fallback()\n' if args.force_batch_fallback else ''))
    cfg=root/'config.toml'
    text=f'''experiment = "p2"
seed = 3
[p2]
mode = "s14_pod_job"
process_name = "{args.process_name}"
process_nn = 51
preset = "tiny"
device = "cpu"
max_hours = 0.25
steps = 4
n_p = 4
p_micro_batch = 4
p_so_micro_batch = 1
t_so_pilot_s = 0.02
t_arm_pilot_s = 0.01
out_dir = "{root/'s14'}"
data_dir = "{root/'p2data'}"
passkey = true
passkey_docs = 4
read_batch = {args.read_batch}
passkey_batch = {args.passkey_batch}
[p2.data]
mode = "uploaded"
expected_hashes = "{expected}"
[p2.tiny_overrides]
'''
    for k,v in support.overrides().items():text+=k+' = '+json.dumps(v)+'\n'
    if args.canary:
        text+=f'''[p2.canary]
enabled = true
steps = 4
guard_every = 2
micro_batch = 4
so_micro_batch = 1
n_docs = 8
ho_docs = 8
read_etas = [1, 2, 4]
passkey_etas = [1, 2]
benchmark_docs = 8
benchmark_ho_docs = 8
read_batches = [2, 4, 8, 16]
passkey_batches = [1, 8, 16]
test_hosts_file = "{hosts}"
'''
    for arm in s14.ARMDIRS:text+=f'[[p2.cells]]\narmdir = "{arm}"\nseed = 3\n'
    cfg.write_text(text)
    env={**os.environ,'PROCESS_NAME':args.process_name,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'4','PYTHONPATH':str(injection)+os.pathsep+str(Path(__file__).parent.resolve())}
    started=time.monotonic()
    with (root/'smoke.log').open('w') as log:
        subprocess.run([sys.executable,'-m','tttmem.run',str(cfg),'--process-name',args.process_name],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    contract=support.contract(hosts)
    if args.canary:
        from dataclasses import replace
        contract=replace(contract,grids={k:[1,2,4] for k in ('so','fo')},
                         passkey_etas={k:[1,2] for k in ('so','fo')})
    (root/'test-contract.json').write_text(json.dumps(asdict(contract),indent=2)+'\n')
    result=s14_analysis.analyze_s14(root/'s14',root/'analysis',test_contract=contract)
    assert result['status']=='OK',result
    manifest=json.loads((root/'s14/manifest.json').read_text())
    assert (root/'s14/DONE').is_file()
    assert all(sha256_file(root/'s14'/r['path'])==r['sha256'] for r in manifest['files'])
    peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    summary=dict(status='OK',contract_e2e='OK',cells=6,passkey='real writer/scorer; test-only hosts/tokenizer',
                 wall_s=time.monotonic()-started,peak_child_rss_bytes=peak,threads=4,
                 canary_rehearsal=args.canary)
    summary.update(read_batch=args.read_batch,passkey_batch=args.passkey_batch,
                   forced_fallback=args.force_batch_fallback)
    if args.force_batch_fallback:
        for arm in s14.ARMDIRS:
            check=json.loads((root/f's14/reads/tiny/{arm}/s3/batch_check.json').read_text())
            assert check['frontier']['status']=='FALLBACK' and check['frontier']['effective_batch']==2
            if args.passkey_batch>1:
                assert check['passkey']['status']=='FALLBACK' and check['passkey']['effective_batch']==1
        logs='\n'.join(p.read_text() for p in (root/'s14/logs').glob('*.log'))
        assert 'BATCH_FALLBACK' in logs and 'PASSKEY_BATCH_FALLBACK' in logs
    assert peak<6_000_000_000
    (root/'smoke-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
