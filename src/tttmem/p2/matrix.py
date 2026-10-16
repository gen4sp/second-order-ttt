"""Restartable, memory- and time-budgeted MINI matrix coordinator."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from .mini_analysis import write as write_analysis

PARTIAL_SO_ESTIMATES = {
    'podsmoke_hours': {'mini-m/e2e12-p125': 1.10, 'mini-s/e2e12-p125': .82},
    'memory_peaks_gb': {'mini-m/e2e12-p125': 11.5, 'mini-s/e2e12-p125': 6.9},
}


def _load(path):
    with Path(path).open('rb') as f:
        return tomllib.load(f).get('p2', {})


def _cells(cfg):
    top = cfg.get('preset', 'mini-s')
    if top == 'TBD':
        raise ValueError('mini_matrix refuses preset = TBD')
    cells = cfg.get('cells')
    if cells is None:
        cells = [{'arm': a, 'init_seed': s, 'data_seed': s}
                 for s in range(3) for a in ('e2e2', 'e2e1-ce')]
        cells += [{'arm': a, 'init_seed': 0} for a in ('e2e1-rp', 'e2e1-nc')]
    result = []
    for raw in cells:
        cell = dict(raw)
        cell['preset'] = cell.get('preset') or top
        if cell['preset'] == 'TBD':
            raise ValueError('mini_matrix refuses a cell with preset = TBD')
        cell.setdefault('data_seed', cell.get('init_seed', 0))
        result.append(cell)
    return result


def _base_config(cfg, cell, repo):
    base = _preset_base(cfg, cell, repo)
    # SOQ-N3B: per-cell keys on top of the production config (e.g. the FO-CM schedule).
    base.update(cell.get('overrides', {}))
    return base


def _preset_base(cfg, cell, repo):
    arm, preset = cell['arm'], cell['preset']
    if preset == 'tiny':
        data = cfg.get('data_dir', '')
        return dict(arm=arm.removesuffix('-mix'), preset='tiny', dtype='fp32', device=cfg.get('device', 'cpu'),
                    seq_len=64, global_batch=16 if arm.endswith('-mix') else 2,
                    micro_batch=4 if arm.endswith('-mix') else 1, steps=int(cfg.get('steps', 2)),
                    warmup=2, lr_peak=1e-2, barriers=[2], segment_len=16, fused_attn=False,
                    data_dir=data, code_data_dir=data, mix=arm.endswith('-mix'),
                    code_train_split='code_train', code_eval_split='code_eval', code_eval_steps=[int(cfg.get('steps', 2))],
                    eval_steps=[int(cfg.get('steps', 2))], full_eval_steps=[int(cfg.get('steps', 2))],
                    n_eval=4, ho_docs=2, ho_trajectory_docs=2, trajectory_docs=4, pilot_docs=2,
                    min_probes=1, eval_batch=2, crossover_rules=[], assay_gate=False,
                    pilot_steps=[], sizing_steps=[], save_checkpoints=False)
    mapping = cfg.get('production_configs', {})
    path = mapping.get(f'{preset}/{arm}') or mapping.get(arm)
    if path is None:
        path = 'configs/mini/' + arm + ('-m' if preset == 'mini-m' and not arm.endswith('-mix') else '') + '.toml'
    path = Path(path)
    if not path.is_absolute():
        path = repo / path
    base = _load(path)
    base.update(preset=preset, arm=arm.removesuffix('-mix'),
                init_seed=int(cell.get('init_seed', 0)), data_seed=int(cell['data_seed']))
    if arm.endswith('-mix'):
        base['mix'] = True
    return base


def _worker(cfg_path, out_log, process_name):
    env = dict(os.environ, PROCESS_NAME=process_name, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    log = out_log.open('a')
    proc = subprocess.Popen([sys.executable, '-m', 'tttmem.p2.matrix', '--cell-json', str(cfg_path),
                             '--process-name', process_name], stdout=log, stderr=subprocess.STDOUT, env=env)
    proc._ttm_log_handle = log
    return proc


def _run_cell_json(path, process_name):
    # Keep the CPU coordinator free of torch; workers are the only torch processes.
    import torch
    from .train import TrainCfg, train
    torch.set_num_threads(4)
    d = json.loads(Path(path).read_text())
    d['process_name'] = process_name
    result = train(TrainCfg.from_dict(d))
    peak = torch.cuda.max_memory_allocated() / 2**30 if d.get('device') == 'cuda' else 0.
    times = result.get('step_ms', [])
    print('MINI_CELL_DONE ' + json.dumps(dict(wall_s=result.get('wall_s'),
          ms_per_step=sum(times)/max(1, len(times)), peak_gb=peak, final=result)), flush=True)
    return result


def _resume_dir(cell_out, configured='', matrix_root=None, resume_root=None):
    if configured:
        return Path(configured)
    if resume_root:
        return Path(resume_root) / cell_out.relative_to(matrix_root)
    if str(cell_out).startswith('outputs/mini/'):
        # One rolling state per cell in the shared resume directory.
        return Path('outputs/mini/resume') / Path(matrix_root).name / cell_out.relative_to(matrix_root)
    return cell_out.parent / '.resume' / cell_out.name


def _stale_cell(cell_out, state_root):
    if (cell_out / 'MINI_CELL_DONE.json').exists() or any(state_root.glob('resume_step*.pt')) or not cell_out.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    stale = cell_out.with_name(cell_out.name + '.stale-' + stamp)
    cell_out.rename(stale)
    return stale


def _lookup(cfg, field, cell, default=0.):
    values = cfg.get(field, {})
    key = f"{cell['preset']}/{cell['arm']}"
    fallback = PARTIAL_SO_ESTIMATES.get(field, {}).get(key, default)
    return float(values.get(key, values.get(cell['arm'], fallback)))


def _estimate_hours(cell, cfg, finished, workers=1):
    same = [x for x in finished if x.get('preset') == cell['preset'] and x.get('arm') == cell['arm']
            and x.get('status') == 'done' and x.get('wall_s') is not None]
    if same:
        # Already measured under this root's concurrency.
        return float(same[-1]['wall_s']) / 3600.
    hours = _lookup(cfg, 'podsmoke_hours', cell)
    if not hours:
        if cell.get('preset') in ('mini-s', 'mini-m') and not cfg.get('estimates_ms_step'):
            # Without an estimate the max_hours guard would be inert.
            raise ValueError(f"no podsmoke_hours estimate for {cell['preset']}/{cell['arm']}")
        hours = _lookup(cfg, 'estimates_ms_step', cell, 1.) * int(cell.get('steps', cfg.get('steps', 1200))) / 3.6e6
    return hours * (float(cfg.get('concurrency_slowdown', 1.)) if workers > 1 else 1.)


def _can_start(cell, cfg, finished, elapsed, running_peaks, workers):
    wall = _estimate_hours(cell, cfg, finished, workers)
    peak = _lookup(cfg, 'memory_peaks_gb', cell)
    return (elapsed + wall * 3600 <= float(cfg.get('max_hours', float('inf'))) * 3600
            and sum(running_peaks) + peak <= float(cfg.get('gpu_mem_budget_gb', 21.)))


def _analyse_root(root, label):
    try:
        result = write_analysis(root, root / 'mini_analysis.json', root / 'mini_analysis.md')
    except FileNotFoundError:
        result = None
    print('MINI_ANALYSIS ' + json.dumps(dict(root=str(root), label=label,
          rules=result['preflight_rules'] if result else 'incomplete'), sort_keys=True), flush=True)
    return result


def run(cfg, repo_root=None, clock=None):
    clock = clock or time
    # ``screen_cells`` lets the repository root keep the MINI-P fixture
    # used by its regression test while the registered TBB screen runs
    # only the two explicitly listed SO/FO cells.
    cell_cfg = dict(cfg)
    if cfg.get('screen_cells') is not None:
        cell_cfg['cells'] = cfg['screen_cells']
    cells = _cells(cell_cfg)  # Validate every preset before any output or worker starts.
    repo = Path(repo_root or cfg.get('repo_root', Path(__file__).resolve().parents[3]))
    out = Path(cfg.get('out_dir', 'outputs/mini/matrix'))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    configs_dir = out / '.cell-configs'
    configs_dir.mkdir(exist_ok=True)
    # Explicit registration order completes SO/CE pairs early.
    workers = max(1, int(cfg.get('workers', 1)))
    process_nn = int(cfg.get('process_nn', 30))
    start = clock.monotonic()
    pending, running, finished, failed = list(cells), {}, [], []
    started_count = 0
    kill_after = float(cfg.get('kill_after_s', 0.))
    kill_target = int(cfg.get('kill_cell_index', 0))
    killed = False
    while pending or running:
        # Reconcile completed markers before applying either budget.
        for cell in list(pending):
            cell_out = _cell_out(out, cell)
            marker = cell_out / 'MINI_CELL_DONE.json'
            if marker.exists():
                item = json.loads(marker.read_text())
                item.update(preset=cell['preset'], arm=cell['arm'])
                finished.append(item)
                pending.remove(cell)
        while pending and len(running) < workers:
            cell = next((c for c in pending if _can_start(c, cfg, finished, clock.monotonic()-start,
                         [x['peak'] for x in running.values()], workers)), None)
            if cell is None:
                break
            key = f"{cell['preset']}-{cell['arm']}-s{cell.get('init_seed', 0)}"
            cell_out = _cell_out(out, cell)
            state_root = _resume_dir(cell_out, cell.get('resume_dir', ''), out, cfg.get('resume_root'))
            stale = _stale_cell(cell_out, state_root)
            if stale:
                print('MINI_CELL_STALE ' + json.dumps(dict(cell=key, stale=str(stale))), flush=True)
            base = _base_config(cfg, cell, repo)
            base.update(out_dir=str(cell_out), resume=any(state_root.glob('resume_step*.pt')),
                        resume_dir=str(state_root), assay_gate=False, pilot_steps=[], sizing_steps=[])
            for field in ('steps', 'resume_every', 'device'):
                if field in cfg: base[field] = cfg[field]
                if field in cell: base[field] = cell[field]
            if 'resume_every' not in cfg and 'resume_every' not in cell:
                base['resume_every'] = 75  # <10 min at measured slowest SO (~7.2 s/step)
            suffix = {'mini-s': 's', 'mini-m': 'm'}.get(cell['preset'], cell['preset'])
            prefix = cfg.get('name_prefix', cfg.get('process_prefix', 'TTM-min'))
            name = f"{prefix}-{process_nn:02d}-{suffix}-{cell['arm']}-s{cell.get('init_seed', 0)}"
            base['process_name'] = name
            path = configs_dir / (key + '.json')
            path.write_text(json.dumps(base, sort_keys=True) + '\n')
            print('MINI_CELL_START ' + json.dumps(dict(cell=key, process_name=name,
                  est_wall_h=_estimate_hours(cell, cfg, finished, workers))), flush=True)
            proc = _worker(path, out / 'logs' / (key + '.log'), name)
            started_count += 1
            running[proc.pid] = dict(proc=proc, cell=cell, key=key, out=cell_out,
                                    started=clock.monotonic(), peak=_lookup(cfg, 'memory_peaks_gb', cell),
                                    ordinal=started_count)
            pending.remove(cell)
        if not running:
            break
        for pid, info in list(running.items()):
            proc = info['proc']
            if (kill_after > 0 and not killed and clock.monotonic() - info['started'] >= kill_after
                    and (kill_target == 0 or info.get('ordinal') == kill_target)):
                proc.kill(); killed = True
                print('MINI_CELL_FAIL ' + json.dumps(dict(cell=info['key'], reason='intentional-kill-for-resume-test')), flush=True)
                try: proc._ttm_log_handle.close()
                except Exception: pass
                del running[pid]
                continue
            if proc.poll() is None:
                continue
            proc._ttm_log_handle.close()
            if proc.returncode == 0:
                child = {}
                for line in reversed((out / 'logs' / (info['key'] + '.log')).read_text().splitlines()):
                    if line.startswith('MINI_CELL_DONE '):
                        child = json.loads(line.split(' ', 1)[1]); break
                result = dict(cell=info['key'], preset=info['cell']['preset'], arm=info['cell']['arm'],
                              status='done', wall_s=clock.monotonic()-info['started'],
                              ms_per_step=child.get('ms_per_step'), peak_gb=child.get('peak_gb'),
                              final=child.get('final', {}), bytes_written=_bytes_written(info['out']))
                info['out'].mkdir(parents=True, exist_ok=True)
                (info['out'] / 'MINI_CELL_DONE.json').write_text(json.dumps(result) + '\n')
                finished.append(result)
                print('MINI_CELL_DONE ' + json.dumps(result), flush=True)
            else:
                failed.append(info['cell'])
                print('MINI_CELL_FAIL ' + json.dumps(dict(cell=info['key'], returncode=proc.returncode)), flush=True)
            del running[pid]
        clock.sleep(.05)
    print('MINI_MATRIX_DONE ' + json.dumps(dict(finished=len(finished), remaining=len(pending), failed=len(failed))), flush=True)
    analysis = registered_analysis = None
    if cfg.get('analyse', True):  # SOQ-N3B roots hold no MINI pairs
        analysis = _analyse_root(out, 'partial')
        registered = Path(cfg.get('registered_root', 'outputs/mini/matrix'))
        registered_analysis = _analyse_root(registered, 'registered')
    return dict(finished=finished, remaining=pending, failed=failed, analysis=analysis,
                registered_analysis=registered_analysis)


def _cell_out(out, cell):
    return Path(cell.get('out_dir', out / 'runs' / cell['preset'] / cell['arm'] / f"s{cell.get('init_seed', 0)}"))


def _bytes_written(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell-json')
    ap.add_argument('--process-name', default=os.environ.get('PROCESS_NAME', 'TTM-min-30-cell'))
    ap.add_argument('config', nargs='?')
    args = ap.parse_args(argv)
    from .data import _set_process_name
    _set_process_name(args.process_name)
    if args.cell_json:
        _run_cell_json(args.cell_json, args.process_name)
    elif args.config:
        run(_load(args.config))
    else:
        raise SystemExit('matrix config required')


if __name__ == '__main__':
    main()
