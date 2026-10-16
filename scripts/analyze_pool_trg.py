"""Portable R2-ii H-T3 and descriptive H-T4 analysis of a downloaded pool."""
import argparse,json,os
from pathlib import Path
import setproctitle
from tttmem.trg_pool import analyze_pool


def main():
    p=argparse.ArgumentParser()
    p.add_argument('pool',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--process-name',default=os.environ.get('PROCESS_NAME','TTM-pool-76-trg-analysis'))
    args=p.parse_args();setproctitle.setproctitle(args.process_name)
    pool=json.loads(args.pool.read_text())
    if bool(pool.get('smoke_only'))!=args.smoke_only:raise ValueError('explicit analyzer mode mismatch')
    manifest=json.loads((args.pool.parent/'job_manifest.json').read_text())
    analyze_pool(pool,manifest,args.out,smoke=args.smoke_only,pool_root=args.pool.parent)


if __name__=='__main__':main()
