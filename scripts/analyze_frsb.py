"""Analyze result.json; smoke-only mode never authorizes B9."""
import argparse
import json
from pathlib import Path
import os
import setproctitle

from tttmem.frsb_analysis import analyze, write_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--unblind", action="store_true", help="print full scores after the session ends")
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-frb-79-analyze"))
    args = parser.parse_args()
    setproctitle.setproctitle(args.process_name)
    scores = analyze(json.loads(args.input.read_text()), args.smoke_only)
    write_scores(args.out, scores)
    print("B9_GATE " + json.dumps(scores["b9_gate"]))
    if not args.gate_only and (args.smoke_only or args.unblind):
        print("FRSB_ANALYSIS " + json.dumps({k: v for k, v in scores.items() if k != "units"}))


if __name__ == "__main__":
    main()
