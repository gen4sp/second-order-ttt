from __future__ import annotations
import argparse, os
from tttmem.p2.data import _set_process_name
from tttmem.p2.tbb_analysis import tbb_analysis

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--out-json'); ap.add_argument('--out-md'); ap.add_argument('--process-name',default=os.environ.get('PROCESS_NAME','TTM-tbb-05-analysis')); a=ap.parse_args(); _set_process_name(a.process_name); r=tbb_analysis(a.root,a.out_json,a.out_md); print('TBB_ANALYSIS',r['label'],r['failed'])
