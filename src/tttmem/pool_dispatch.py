"""Supervise a dependency queue with explicit ownership of every dispatched unit.

Per-worker inboxes let the parent know the held unit even if the child dies before
its first message. Only the held unit is retried; survivors retain their work.
"""
import multiprocessing as mp
import time
import sys
from multiprocessing.connection import wait


def overlap_peak(records):
    """Reconcile concurrency from worker timestamps, independent of message delay."""
    events = [(r[key], delta) for r in records
              for key,delta in [('training_started_at', 1), ('training_finished_at', -1)]
              if key in r]
    active = peak = 0
    for _,delta in sorted(events):  # finish precedes start at an equal timestamp
        active += delta
        peak = max(peak, active)
    return peak


class SendResults:
    """Synchronous per-worker pipe: no feeder thread or shared writer lock."""
    def __init__(self, connection):
        self.connection = connection

    def put(self, value):
        self.connection.send(value)


class PipeLines:
    """Forward complete worker log lines through its isolated result pipe."""
    def __init__(self, results, index, stream):
        self.results, self.index, self.stream = results, index, stream
        self.buffer = ''

    def write(self, value):
        self.buffer += value
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.results.put((self.index, None, dict(_pool_event='log', stream=self.stream, line=line+'\n')))
        return len(value)

    def flush(self):
        if self.buffer:
            self.results.put((self.index, None, dict(_pool_event='log', stream=self.stream, line=self.buffer)))
            self.buffer = ''


def dispatch(cfg, items, *, mode, worker, dependencies, payload_for, missing,
             worker_lost, priority, progress=None, prior=None):
    pending = dict(items)
    prior = prior or {}
    if len(pending) != len(items) or any(set(dependencies(k))-pending.keys()-prior.keys() for k in pending):
        raise ValueError('duplicate unit or unknown global dependency')
    available = set(prior)
    while pending.keys() - available:
        batch = {k for k in pending if k not in available and set(dependencies(k)) <= available}
        if not batch:
            raise ValueError('global dependency cycle')
        available |= batch
    context = mp.get_context('spawn')
    inboxes, receivers, processes, held, attempts = {}, {}, {}, {}, {}
    retrying, out = set(), dict(prior)
    active, peak = set(), 0
    hard = cfg['_deadline'] + cfg.get('hard_grace_minutes', 60)*60

    def spawn(index):
        inboxes[index] = context.Queue()
        receiver, sender = context.Pipe(duplex=False)
        receivers[index] = receiver
        p = context.Process(target=worker,
                            args=(index, cfg, mode, inboxes[index], SendResults(sender), None),
                            name=f"{cfg['process_name']}-w{index}")
        processes[index] = p
        p.start()
        sender.close()  # child alone owns the writer; death produces EOF, even mid-message

    def emit():
        if progress:
            progress(out)

    def receive(index, key, result):
        nonlocal peak
        if result.get('_pool_event') == 'log':
            stream = sys.stderr if result['stream'] == 'stderr' else sys.stdout
            stream.write(result['line']); stream.flush()
            return
        if held.get(index) != key:
            return
        if result.get('_pool_event') == 'start':
            active.add(index)
            peak = max(peak, len(active))
            cfg['_peak_'+mode] = peak
            return
        active.discard(index)
        del held[index]
        out[key] = result
        emit()

    def drain():
        for index, connection in list(receivers.items()):
            while connection.poll():
                try:
                    worker_index, key, result = connection.recv()
                    if worker_index != index:
                        raise ValueError('worker result ownership mismatch')
                    receive(index, key, result)
                except (EOFError, OSError):
                    connection.close()
                    del receivers[index]
                    break

    def close_inbox(index):
        inboxes[index].cancel_join_thread()  # a dead receiver cannot stall parent teardown
        inboxes[index].close()

    try:
        while pending or held:
            if time.time() > hard:
                for key in pending:
                    out[key] = missing(key, 'budget exhausted', True)
                for key in held.values():
                    out[key] = missing(key, 'budget exhausted', True, dispatched=True)
                pending.clear(); held.clear(); emit()
                break
            drain()  # accept a completed message before classifying an exited child
            for index, p in list(processes.items()):
                if p.is_alive():
                    continue
                p.join(timeout=.1)
                key = held.pop(index, None)
                active.discard(index)
                if key is not None:
                    attempt, terminal = worker_lost(key, p.exitcode, attempts.pop(index, 1))
                    if attempt < 2 and key not in retrying and time.time() < cfg['_deadline']:
                        retrying.add(key)
                        pending[key] = dict(items)[key]
                    else:
                        out[key] = terminal
                        emit()
                close_inbox(index)
                if index in receivers:
                    receivers.pop(index).close()
                del processes[index]
            ready = sorted((k for k in pending if set(dependencies(k)) <= out.keys()), key=priority)
            for key in ready:
                deps = dependencies(key)
                failed = any(out[d].get('status') != 'DONE' for d in deps)
                if failed or time.time() >= cfg['_deadline']:
                    out[key] = missing(key, 'failed prerequisite' if failed else 'budget exhausted',
                                       not failed or any(out[d].get('resumable', False) for d in deps))
                    del pending[key]; emit()
                    continue
                index = next((i for i in range(cfg['workers']) if i not in held), None)
                if index is None:
                    break
                if index not in processes:
                    spawn(index)
                held[index] = key
                attempts[index] = 2 if key in retrying else 1
                inboxes[index].put((key, payload_for(key, out), attempts[index]))
                del pending[key]
            if held and receivers:
                wait(list(receivers.values()), timeout=.2)
    finally:
        for index, p in processes.items():
            inboxes[index].put(None)
            p.join(timeout=2)
            if p.is_alive():
                p.terminate(); p.join(timeout=2)
            close_inbox(index)
        for connection in receivers.values():
            connection.close()
    print(f'POOL_CONCURRENCY mode={mode} peak={peak}', flush=True)
    return out
