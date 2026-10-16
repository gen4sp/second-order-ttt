"""B9 analyzer; always reads the paired FRS-B result."""
import argparse
import json
import os
from pathlib import Path
import setproctitle

from tttmem.b9_analysis import analyze
from tttmem.frsb_analysis import write_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("frsb", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--unblind", action="store_true")
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-b9-79-analyze"))
    args = parser.parse_args()
    setproctitle.setproctitle(args.process_name)
    scores = analyze(json.loads(args.input.read_text()), json.loads(args.frsb.read_text()), args.smoke_only)
    write_scores(args.out, scores, prefix="b9")
    if args.smoke_only or args.unblind:
        print("B9_ANALYSIS " + json.dumps({k: v for k, v in scores.items() if k != "units"}))
    else:
        print("B9_ANALYSIS_BLINDED " + json.dumps(dict(smoke=False, scored=True)))


if __name__ == "__main__":
    main()
