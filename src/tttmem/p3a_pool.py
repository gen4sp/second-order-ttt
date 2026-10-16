"""Torch-free process pool for the P3-a cell waves.

The pool imports its worker function by name inside each spawned child, so this module
itself never imports torch.  That keeps the orchestration testable on a CPU-only machine
with a stdlib-only probe job (``probe_job``), while on a CUDA machine every child owns its
own model, allocator and CUDA context.

``ProcessPoolExecutor`` is used instead of ``multiprocessing.Pool``: an abrupt worker
death raises ``BrokenProcessPool`` on the pending futures instead of hanging the parent
forever.  Python exceptions inside a job are caught in the child and returned with their
traceback, so one failing cell never stops the wave.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib
import multiprocessing as mp
import os
import time
import traceback
from typing import Any, Callable, Iterable, Iterator

_WORKER: Callable[[dict], Any] | None = None


def resolve(path: str) -> Callable:
    """Import ``"package.module:function"`` and return the function."""
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(f"expected 'module:function', got {path!r}")
    return getattr(importlib.import_module(module), name)


def worker_index() -> int:
    """1-based index of a pool child; 0 in the parent / inline process."""
    ident = getattr(mp.current_process(), "_identity", ())
    return int(ident[0]) if ident else 0


def _init(func_path: str, init_path: str | None, init_args: tuple) -> None:
    global _WORKER
    if init_path:
        resolve(init_path)(*init_args)
    _WORKER = resolve(func_path)


def _call(job: dict) -> tuple[str, Any, str | None]:
    try:
        assert _WORKER is not None, "pool child was not initialised"
        return job["key"], _WORKER(job), None
    except BaseException:  # noqa: BLE001 - the parent records every failure
        return job["key"], None, traceback.format_exc()


class CellPool:
    """``workers > 1``: spawned children; ``workers == 1``: inline, same process."""

    def __init__(self, workers: int, func_path: str, init_path: str | None = None,
                 init_args: tuple = ()) -> None:
        self.workers = max(1, int(workers))
        self._executor: cf.ProcessPoolExecutor | None = None
        if self.workers > 1:
            self._executor = cf.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=_init, initargs=(func_path, init_path, tuple(init_args)))
        else:
            _init(func_path, init_path, tuple(init_args))

    def run(self, jobs: Iterable[dict], stall_s: float | None = None
            ) -> Iterator[tuple[str, Any, str | None]]:
        """Yield ``(key, result, error_traceback)`` as jobs finish.

        Jobs are submitted in order, so a FIFO executor starts them in order.  If no job
        finishes within ``stall_s`` seconds the generator raises ``TimeoutError``.
        """
        jobs = list(jobs)
        if self._executor is None:
            for job in jobs:
                yield _call(job)
            return
        futures = [self._executor.submit(_call, job) for job in jobs]
        pending = set(futures)
        while pending:
            done, pending = cf.wait(pending, timeout=stall_s, return_when=cf.FIRST_COMPLETED)
            if not done:
                for f in pending:
                    f.cancel()
                raise TimeoutError(f"no P3-a job finished within {stall_s} s")
            for f in done:
                yield f.result()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "CellPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def probe_init(tag: str) -> None:
    """Stdlib-only initialiser used by the pool tests."""
    os.environ["P3A_PROBE_TAG"] = f"{tag}-w{worker_index()}"


def probe_job(job: dict) -> dict:
    """Stdlib-only job used by the pool tests: optional sleep, failure or hard exit."""
    if job.get("sleep"):
        time.sleep(float(job["sleep"]))
    if job.get("fail"):
        raise RuntimeError(f"probe failure {job['key']}")
    if job.get("exit"):
        os._exit(7)
    return dict(key=job["key"], pid=os.getpid(), tag=os.environ.get("P3A_PROBE_TAG"),
                index=worker_index())
