"""Local P3-a analysis of the run outputs (CPU, numpy only; never imports torch).

    uv run --locked python scripts/analyze_p3a.py --root outputs/soq/p3a \
        [--out outputs/soq/p3a/analysis]

Writes ``analysis.json`` and ``analysis.md`` (per-cell table, FO-SO gaps with 90 %
seed-level CIs, DiD, verdict D, qualifiers, multipliers) and prints the verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_analysis():
    # Load the module file directly: importing the package would be fine too, but this
    # keeps the script independent of the installed environment's torch.
    path = Path(__file__).resolve().parents[1] / "src" / "tttmem" / "p3a_analysis.py"
    spec = importlib.util.spec_from_file_location("p3a_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p3a_analysis"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="P3-a output root (holds main/ and cal/)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default <root>/analysis-local)")
    args = p.parse_args(argv)
    mod = _load_analysis()
    res = mod.analyze(args.root)
    out = args.out or args.root / "analysis-local"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, sort_keys=True, indent=1) + "\n")
    (out / "analysis.md").write_text(mod.render_md(res))
    print("P3A_VERDICT " + json.dumps(res["verdict"], sort_keys=True))
    print(f"wrote {out / 'analysis.json'} and {out / 'analysis.md'}")
    return res


if __name__ == "__main__":
    main()
