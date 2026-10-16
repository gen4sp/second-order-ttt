"""Entry point: ``python -m tttmem.run config.toml``.

The ``experiment`` key selects the measurement; its table holds the settings.
Prints the effective configuration, per-step results and a final
``RESULT_JSON`` line.
"""

from __future__ import annotations

import importlib
import argparse
import json
import os
import platform
import sys
import time
import tomllib

import numpy as np

EXPERIMENTS = {
    "lod": "tttmem.exp_lod",
    "ttt": "tttmem.exp_ttt",
    "wab": "tttmem.exp_wab",
    "kag": "tttmem.exp_kag",
    "p01": "tttmem.exp_p01",
    "chr": "tttmem.exp_chr",
    "frs": "tttmem.exp_frs",
    "rdo_rpe": "tttmem.exp_rdo",
    "trg": "tttmem.exp_trg",
    "p3a": "tttmem.exp_p3a",
    "c1": "tttmem.exp_c1",
    "p2": "tttmem.exp_p2",
    "frsb": "tttmem.exp_frsb",
    "b9": "tttmem.exp_b9",
    "rtw": "tttmem.exp_rtw",
    "ve": "tttmem.exp_ve",
    "ve_diag": "tttmem.exp_ve_diag",
    "ve_budget": "tttmem.exp_ve_budget",
    "ve_write": "tttmem.exp_ve_write",
    "ve_gain": "tttmem.exp_ve_gain",
    "rnk": "tttmem.exp_rnk",
}


def _main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.toml')
    parser.add_argument('--process-name', default=os.environ.get('PROCESS_NAME'))
    args = parser.parse_args(argv[1:])
    if args.process_name:
        import setproctitle
        os.environ["PROCESS_NAME"] = args.process_name
        setproctitle.setproctitle(args.process_name)
    path = args.config
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    name = cfg["experiment"]
    if args.process_name:
        cfg[name]["process_name"] = args.process_name
    seed = int(cfg.get("seed", 0))
    # Data preparation must also work on a CPU-only machine without importing torch/CUDA.
    if name == "p2" and cfg[name].get("mode") in ("prepare", "prepare_code"):
        torch_version = "not-loaded (prepare)"
    elif name == "p2" and cfg[name].get("mode") in ("soq_n3b_fo_cm", "s14_pod_job"):
        # Coordinator only: every torch process is a worker subprocess.
        torch_version = "not-loaded (coordinator)"
    else:
        import torch
        torch_version = torch.__version__
    print(f"tttmem | experiment={name} seed={seed} | python {platform.python_version()} "
          f"numpy {np.__version__} torch {torch_version}")
    print("config:", json.dumps(cfg, sort_keys=True))
    module = importlib.import_module(EXPERIMENTS[name])
    t0 = time.time()
    try:
        from .frs_registry import execution_guard
        settings = cfg[name]
        seeds = [seed, *settings.get('seeds', []), *settings.get('candidate_seeds', []),
                 *[c['seed'] for c in settings.get('cells', [])],
                 *[c['seed'] for c in settings.get('extra_jobs', [])]]
        # The sealed-pool seed protocol is opt-in (require_seal = true).
        if settings.get('require_seal', False) and any(200 <= int(s) <= 239 for s in seeds):
            if name not in ('frs','rdo_rpe'):
                raise ValueError('fresh seeds require the sealed combined pool entrypoint')
            if name=='frs':execution_guard(settings)
        result = module.run(settings, seed)
    except BaseException as exc:
        result = dict(complete=False, error=repr(exc))
        pool_path = settings.get('output_dir') if 'settings' in locals() else None
        if name == 'frs' and pool_path:
            from pathlib import Path
            file = Path(pool_path) / 'pool.json'
            if file.exists():
                pool = json.loads(file.read_text())
                result.update(errored_mains=sum(bool(r.get('error')) for r in pool.get('cells', {}).values()),
                    missing_jobs=sum(r.get('status') == 'MISSING' for r in pool.get('jobs', {}).values()),
                    fail_open_cells=sum(bool(r.get('coverage_fail_open')) for r in pool.get('cells', {}).values()))
        print('RESULT_JSON ' + json.dumps(dict(experiment=name, seed=seed,
              elapsed_s=round(time.time()-t0, 1), **result)), flush=True)
        exc._result_emitted=True
        raise
    elapsed = time.time() - t0
    print(f"elapsed_s={elapsed:.1f}")
    print("RESULT_JSON " + json.dumps({"experiment": name, "seed": seed,
                                       "elapsed_s": round(elapsed, 1), **result}))


def main(argv):
    try:
        return _main(argv)
    except BaseException as exc:
        if not getattr(exc,'_result_emitted',False):
            print('RESULT_JSON '+json.dumps(dict(complete=False,error=repr(exc),
                errored_mains=0,missing_jobs=0,fail_open_cells=0)),flush=True)
        raise


if __name__ == "__main__":
    main(sys.argv)
