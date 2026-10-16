#!/usr/bin/env python3
"""Convert a locally downloaded official Orbax model_weights checkpoint; no GCS access.

Run it in the isolated JAX/Orbax/torch environment given in README.md (App. D).
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--process-name", required=True)
    parser.add_argument("--preset", choices=("125m", "tiny2"), default="125m")
    args = parser.parse_args()
    os.environ["PROCESS_NAME"] = args.process_name
    import setproctitle
    setproctitle.setproctitle(args.process_name)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tttmem.p2.convert import convert
    from tttmem.p2.model import PRESETS
    convert(args.checkpoint, args.out, PRESETS[args.preset])


if __name__ == "__main__":
    main()
