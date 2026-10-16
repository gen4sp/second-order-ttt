"""Parameter counts of the text models (mini-s, mini-m, 125m) from the model presets.

Builds each preset with ``tttmem.p2.model.build_model`` on CPU and counts with ``param_count``:
total, tied embedding (``wte``), non-embedding (total - embedding) and fast weights (the prime MLPs
of the suffix blocks), in millions with two decimals. Writes the CSV and prints it.
Peak memory is about 1.5 GB (the 125m preset).

Usage: uv run --locked python scripts/analysis/text-param-counts/parameter_counts.py [--out CSV]
"""
import argparse
import os
from pathlib import Path

PRESETS = ("mini-s", "mini-m", "125m")
HEADER = "preset,total_params_millions,embedding_tied_millions,non_embedding_millions,fast_weights_millions"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="outputs/analysis/text-param-counts/parameter_counts.csv")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "parameter-counts"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    from tttmem.p2.model import build_model, param_count

    lines = [HEADER]
    for preset in PRESETS:
        c = param_count(build_model(preset))
        lines.append(",".join([preset] + [f"{c[k] / 1e6:.2f}" for k in ("total", "embedding", "non_embedding", "fast")]))
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
