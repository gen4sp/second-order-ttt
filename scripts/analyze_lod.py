#!/usr/bin/env python3
"""Analyze sealed LOD receipts. Tiny driver output requires --diagnostic."""
import argparse
import os
from pathlib import Path
from tttmem.lod_analysis import analyze_directory, write_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-lod-74-analyze"))
    args = parser.parse_args()
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    result = analyze_directory(args.input, args.diagnostic)
    write_outputs(args.input, result, args.out or args.input / "analysis")
    print("LOD_ANALYSIS", result["mode"])
    print("LOD_CLAIMS", result["claims"])
    for h in ("H2", "H3a", "H1d", "H1i"):
        print(h, result[h]["label"], "n=", result[h]["n"])


if __name__ == "__main__":
    main()
