"""Pure-stdlib worker probe: child processes never import torch."""
import os
from pathlib import Path
import json


def worker(index, cfg, mode, inbox, done, lock):
    import setproctitle
    name = cfg['process_name']+f'-w{index}'
    os.environ['PROCESS_NAME'] = name
    setproctitle.setproctitle(name)
    while True:
        item = inbox.get()
        if item is None:
            return
        key, payload, attempt = item
        done.put((index, key, dict(_pool_event='start')))
        done.put((index, None, dict(_pool_event='log', stream='stdout', line='POOL_PROBE '+key+'\n')))
        folder = Path(cfg['output_dir'])/('mains' if mode == 'main' else 'jobs')
        if mode != 'main' and attempt == 2:
            folder = folder/'retry'
        folder = folder/key.replace('/', '__')
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'attempt.json').write_text(json.dumps(dict(attempt=attempt)))
        if key.endswith('crash') and (attempt == 1 or cfg.get('crash_twice')):
            os._exit(9)
        rec = dict(job_id=key, status='DONE', attempts=attempt)
        (folder/('cell.json' if mode == 'main' else 'result.json')).write_text(json.dumps(rec))
        done.put((index, key, rec))
