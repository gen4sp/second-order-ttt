"""Analyze persisted merged jobs. Smoke evidence requires --smoke-only."""
import argparse,json,os
from pathlib import Path
import setproctitle
from tttmem.rdo_evidence import analyze


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--process-name',default='TTM-rdo-78')
    args=p.parse_args();setproctitle.setproctitle(args.process_name)
    analyze(json.loads(args.results.read_text()),args.out,smoke=args.smoke_only)


if __name__=='__main__':main()
