#!/usr/bin/env python3
"""Analyze LOD-X receipts; --lod-ref reads LOD JSON only, never checkpoints."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--lod-ref", type=Path)
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "TTM-lodx-82-analyze"))
    args = parser.parse_args()
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    from tttmem.lodx_analysis import analyze_directory, write_outputs
    result = analyze_directory(args.input, args.diagnostic, args.lod_ref)
    write_outputs(args.input, result, args.out or args.input / "analysis")
    print("LOD_X_ANALYSIS", result["mode"])
    print("LOD_X_CLAIMS", result["claims"])
    print("X1", result["X1"])
    print("R", {k: v for k, v in result["R"].items() if k != "cells"})


if __name__ == "__main__":
    main()
