"""Generic continuation jobs, dependency queue and sealed manifest providers."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import torch

from . import exp_chr as chr
from . import exp_ttt as ttt
from . import frs_state as state
from .frs_backend import ChrBackend, TimedBackend
from .pool_runtime import BudgetExhausted, budget_record, mark_budget, archive_budget


@dataclass(frozen=True)
class ReadSchedule:
    rule: Any
    every: int = 5
    start: int | None = None
    stop: int | None = None
    full: bool = False

    def active(self, step: int, fork: int) -> bool:
        first = fork + self.every if self.start is None else self.start
        return (self.every > 0 and step >= first and
                (self.stop is None or step <= self.stop) and (step - first) % self.every == 0)


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    state_file: str
    state_sha256: str
    rule: Any
    horizon: int
    after: tuple[str, ...] = ()
    reads: tuple[ReadSchedule, ...] = ()
    final_rules: tuple[Any, ...] = ()
    save_at: tuple[int, ...] = ()
    outer_mask: Callable | None = None
    skip_controller: Callable | None = None
    sham_from: str | None = None
    pre_rules: tuple[Any, ...] = ()
    log_every: int = 5
    expected_hashes: dict | None = None
    cell_id: str = ""
    metadata: dict = field(default_factory=dict)
    hooks_factory: Callable | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.job_id):
            raise ValueError("job_id must be a safe directory component")
        if self.horizon <= 0 or self.log_every <= 0 or self.horizon % self.log_every:
            raise ValueError("job horizon must align with log grid")
        if self.sham_from and self.sham_from not in self.after:
            raise ValueError("sham_from must be listed in after")
        if any(r.every <= 0 for r in self.reads):
            raise ValueError("read cadence must be positive")


def job_record(job: JobSpec) -> dict:
    def name(value):
        return None if value is None else getattr(value, "__module__", type(value).__module__) + ":" + \
            getattr(value, "__qualname__", type(value).__qualname__)
    return dict(job_id=job.job_id, state_file=job.state_file, state_sha256=job.state_sha256,
                rule=job.rule.tag, rule_spec=vars(job.rule), horizon=job.horizon,
                after=list(job.after), reads=[dict(rule=r.rule.tag, rule_spec=vars(r.rule),
                every=r.every, start=r.start, stop=r.stop, full=r.full) for r in job.reads],
                final_rules=[r.tag for r in job.final_rules], pre_rules=[r.tag for r in job.pre_rules],
                save_at=list(job.save_at), outer_mask=name(job.outer_mask),
                skip_controller=name(job.skip_controller), sham_from=job.sham_from,
                log_every=job.log_every, expected_hashes=job.expected_hashes,
                cell_id=job.cell_id, metadata=job.metadata, hooks_factory=name(job.hooks_factory))


def write_drop_list(path: Path, drops: list[int]) -> tuple[str, str]:
    if drops != sorted(set(drops)):
        raise ValueError("drop steps must be unique and sorted")
    path.write_text(json.dumps(dict(schema=1, drops=drops), sort_keys=True) + "\n")
    return str(path), state.file_sha256(path)


def yoked_schedule(path: str | Path, sha256: str, *, start: int, stop: int) -> list[int]:
    if state.file_sha256(path) != sha256:
        raise ValueError("drop list sha256 mismatch")
    payload = json.loads(Path(path).read_text())
    drops = payload.get("drops")
    if payload.get("schema") != 1 or not isinstance(drops, list) or drops != sorted(set(drops)):
        raise ValueError("invalid realised drop list")
    if any(type(d) is not int or not start < d <= stop for d in drops):
        raise ValueError("drop outside branch horizon")
    shifted: list[int] = []
    used: set[int] = set()
    for drop in drops:
        candidate = drop + 3
        while candidate in used:
            candidate += 1
        if candidate > stop:
            raise ValueError("yoked sham overflow: equal counts cannot fit in horizon")
        used.add(candidate)
        shifted.append(candidate)
    if len(shifted) != len(drops):
        raise ValueError("unequal skip/sham drop counts")
    return shifted


@dataclass
class ScheduledSkip:
    base: Any
    scheduled: frozenset
    def __call__(self,step,telemetry):
        if self.base is not None:self.base(step,telemetry)
        return step in self.scheduled


def run_job(job: JobSpec, ctx, output_dir, *, backend=None) -> dict:
    import time
    wall_start=time.perf_counter()
    from .exp_frs import frs_segment
    if not 0 <= ctx.seed <= 9 and not getattr(ctx, "_frs_authorized", False):
        raise ValueError("fresh job requires a verified execution seal before model construction")
    backend = TimedBackend(backend or ChrBackend())
    folder = Path(output_dir) / job.job_id
    folder.mkdir(parents=True, exist_ok=False)
    state.atomic_json(folder/'attempt.json', dict(attempt=getattr(ctx, '_pool_attempt', 1)))
    with state.preserve_rngs():
        model = ttt.build_model(ctx.task, ctx.st, ctx.dev, ctx.seed, ctx.arm)
        opt = ttt.build_optim(model, ctx.st)
        rng = np.random.default_rng()
        payload = state.load_state(job.state_file, model, opt, rng, ctx,
                                   sha256=job.state_sha256, expected_hashes=job.expected_hashes)
        for name, controller in (("outer_mask", job.outer_mask), ("skip_controller", job.skip_controller)):
            if name in payload["extra"]:
                if not hasattr(controller, "load_state_dict"):
                    raise ValueError(f"saved {name} requires a stateful controller")
                controller.load_state_dict(payload["extra"][name])
        start = payload["state_step"]
        if hasattr(backend, "begin_job"):
            backend.begin_job(job, payload, ctx, model)
        if start + job.horizon > ctx.task.steps:
            raise ValueError("job continuation extends past the registered main horizon")
        (folder / "spec.json").write_text(json.dumps(job_record(job), sort_keys=True, indent=2))
        hooks = None if job.hooks_factory is None else job.hooks_factory(ctx, folder, job)
        pre = {}
        for r in {r.tag: r for r in (job.rule, *job.pre_rules)}.values():
            pre[r.tag] = dict(quick=backend.quick(model, ctx, r),
                              full=backend.final(model, ctx, r, [r])[r.tag])
            if hasattr(backend, "pre_write"):
                pre[r.tag]["write"] = backend.pre_write(model, ctx, r)
        result = frs_segment(model, opt, rng, ctx, job.rule, job.job_id, start,
                             stop=start + job.horizon, nonfinite0=payload["nonfinite"],
                             reads=job.reads, backend=backend, outer_mask=job.outer_mask,
                             skip_controller=job.skip_controller, save_at=job.save_at,
                             output_dir=folder, hashes=payload["hashes"], log_every=job.log_every,
                             hooks=hooks, expected_first_sha=payload['next_batch_sha1'])
        model.eval()
        result["final"] = (backend.final(model, ctx, job.rule, job.final_rules)
                           if not result["diverged"] else
                           {r.tag: dict(acc=0.0, divergence_coded=True)
                            for r in (job.rule, *job.final_rules, chr.CE)})
        result["disposition"] = "DIVERGED" if result["diverged"] else "COMPLETE"
        result['read_costs']=backend.costs
        result.update(status="DONE", job_id=job.job_id, rule=job.rule.tag, fork_step=start,
                      cell_id=ctx.ident, seed=ctx.seed, arm=ctx.arm,
                      horizon=job.horizon, pre=pre, state_sha256=job.state_sha256,
                      metadata=job.metadata,
                      pre_write=pre[job.rule.tag].get("write", {}),
                      drop_file=None, drop_sha256=None,
                      next_batch_verified=result["first_batch_sha1"] == payload["next_batch_sha1"])
        if not result["next_batch_verified"]:
            raise ValueError("actual first batch differs from verified source state")
        if hooks is not None and hasattr(hooks, 'finalize'):
            with state.preserve_rngs(rng):
                result['logger_final'] = hooks.finalize()
        result["drop_file"], result["drop_sha256"] = write_drop_list(folder / "drops.json", result["drops"])
        result["final_checkpoint"] = str(folder / "final.pt")
        torch.save(state._cpu(model.state_dict()), result["final_checkpoint"])
        result["final_sha256"] = state.file_sha256(result["final_checkpoint"])
        result['job_wall_s']=round(time.perf_counter()-wall_start,3)
        (folder / "result.json").write_text(json.dumps(chr._finite(result), sort_keys=True, indent=2))
        chr._emit("FRS_JOB", chr._finite(result))
        return result


def _infrastructure(exc: Exception) -> bool:
    if isinstance(exc, BudgetExhausted):
        return False
    return isinstance(exc, (OSError, TimeoutError, MemoryError)) or bool(
        re.search(r"out of memory|worker died|device-side|CUDA error|NCCL|watchdog", str(exc), re.I))


def run_queue(jobs, ctx_factory, output_dir, *, backend=None, resume=False, prior=None,
              first_attempt=1) -> dict[str, dict]:
    jobs = list(jobs)
    ids = {j.job_id for j in jobs}
    if len(ids) != len(jobs):
        raise ValueError("duplicate job IDs")
    prior = prior or {}
    if any(set(j.after) - ids - prior.keys() for j in jobs):
        raise ValueError("unknown job dependency")
    # Preflight cycle detection before running anything.
    graph = {j.job_id: set(j.after) for j in jobs}
    ready = set(prior)
    while graph:
        batch = {i for i, deps in graph.items() if deps <= ready}
        if not batch:
            raise ValueError("job dependency cycle")
        ready |= batch
        graph = {i: deps for i, deps in graph.items() if i not in batch}
    out: dict[str, dict] = dict(prior)
    pending = {j.job_id: j for j in jobs}
    route_backends = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries=output/'queue-summaries'
    summaries.mkdir(exist_ok=True)
    digest=state.json_hash(sorted(ids))[:12]
    number=len(list(summaries.glob(f'{digest}-*.json')))+1
    summary_file=summaries/f'{digest}-{number:02d}.json'
    while pending:
        for ident, original in list(pending.items()):
            if not set(original.after) <= out.keys():
                continue
            del pending[ident]
            if any(out[d]["status"] != "DONE" for d in original.after):
                out[ident] = dict(job_id=ident, status="MISSING", disposition="TRUNCATED",
                                  cell_id=original.cell_id, seed=original.metadata.get('seed'),
                                  arm=original.metadata.get('arm'), rule=original.rule.tag,
                                  reason="failed prerequisite", attempts=0,
                                  resumable=any(out[d].get('resumable', False) for d in original.after))
                continue
            job = original
            identity = dict(cell_id=job.cell_id, arm=job.metadata.get('arm'),
                            seed=job.metadata.get('seed'), rule=job.rule.tag, metadata=job.metadata)
            try:
                if job.state_file.startswith("job:"):
                    _, parent, step = job.state_file.split(":")
                    if parent not in job.after:
                        raise ValueError("state parent must be listed in after")
                    row = next(s for s in out[parent]["states"] if s["state_step"] == int(step))
                    job = replace(job, state_file=row["file"], state_sha256=row["sha256"])
                else:
                    pool_root=output.parent if output.name=='jobs' else output
                    job=replace(job,state_file=state.relocated_file(job.state_file,pool_root))
                ctx = ctx_factory(job)
                if __import__('time').time() >= getattr(ctx, '_frs_cfg', {}).get('_deadline', float('inf')):
                    raise BudgetExhausted('pool job deadline exhausted')
                identity.update(cell_id=ctx.ident, arm=ctx.arm, seed=ctx.seed)
                job_backend = backend
                if job.metadata.get("backend_factory"):
                    factory = job.metadata["backend_factory"]
                    if not 0 <= ctx.seed <= 9 and ctx._frs_cfg.get('require_seal', False):
                        from .frs_registry import verify_seal
                        seal = verify_seal(ctx._frs_cfg.get('seal_file', 'frs_seal.json'))
                        if factory not in seal.get('backend_factories', []):
                            raise ValueError('per-job backend factory is not sealed')
                    if factory not in route_backends:
                        module, name = factory.split(":")
                        route_backends[factory] = getattr(importlib.import_module(module), name)()
                    job_backend = route_backends[factory]
                if job.sham_from:
                    source = out[job.sham_from]
                    drops = yoked_schedule(source["drop_file"], source["drop_sha256"],
                                           start=source["fork_step"],
                                           stop=source["fork_step"] + source["horizon"])
                    scheduled = frozenset(drops)
                    job = replace(job, skip_controller=ScheduledSkip(job.skip_controller,scheduled))
                candidates = [output / ident / 'result.json', output / 'retry' / ident / 'result.json']
                for result_file in candidates if resume else []:
                    if not result_file.is_file():
                        continue
                    rec = json.loads(result_file.read_text())
                    old_spec = json.loads((result_file.parent / "spec.json").read_text())
                    # Function identities may vary for the internally generated sham closure.
                    current = job_record(job)
                    for field in ("skip_controller",):
                        if job.sham_from:
                            current[field] = old_spec[field]
                    # State identity is its bytes, not the original machine's absolute path.
                    current.pop('state_file',None)
                    old_spec.pop('state_file',None)
                    if current != old_spec or state.file_sha256(job.state_file) != job.state_sha256:
                        raise ValueError("resume job spec or state mismatch")
                    if rec.get('status')=='MISSING':
                        out[ident]=rec
                        break
                    final=result_file.parent/Path(rec.get('final_checkpoint','final.pt')).name
                    if rec.get("status") == "DONE" and final.is_file() and \
                            state.file_sha256(final) == rec.get("final_sha256"):
                        rec['final_checkpoint']=str(final)
                        for row in rec.get('states',[]):
                            row['file']=str(result_file.parent/Path(row['file']).name)
                            state.verify_state(row['file'],ctx,sha256=row['sha256'],expected_hashes=job.expected_hashes)
                        if rec.get('drop_file'):
                            rec['drop_file']=str(result_file.parent/Path(rec['drop_file']).name)
                            if state.file_sha256(rec['drop_file'])!=rec['drop_sha256']:
                                raise ValueError('resume drop-list checksum mismatch')
                        out[ident] = rec
                        break
                if ident in out:
                    continue
                cut = next((p for p in (output/'retry'/ident, output/ident)
                            if (p/'budget.json').is_file()), None) if resume else None
                interrupted = resume and (output/ident).exists()
                initial = first_attempt
                if cut is not None:
                    initial = json.loads((cut/'budget.json').read_text())['attempt']
                    archive_budget(cut)
                    interrupted = False
                elif resume and (output/'retry'/ident).exists():
                    raise RuntimeError('interrupted retry exhausted; frozen job remains missing')
                initial = max(initial, 2 if interrupted else 1)
                for attempt in range(initial, 3):
                    attempt_output = output if attempt == 1 else output / "retry"
                    try:
                        folder = attempt_output / ident
                        if resume and folder.exists():
                            old = sorted(folder.parent.glob(ident + '.partial-*'))
                            folder.rename(folder.with_name(ident + f'.partial-{len(old)+1}'))
                        import copy
                        ctx._pool_attempt = attempt
                        rec = run_job(copy.deepcopy(job), ctx, attempt_output, backend=job_backend)
                        rec["attempts"] = attempt
                        if job.sham_from:
                            rec.update(scheduled_drops=sorted(scheduled), realised_drops=rec['drops'],
                                       equal_realised_count=len(rec['drops']) == len(scheduled))
                        state.atomic_json(attempt_output / ident / 'result.json', chr._finite(rec))
                        out[ident] = rec
                        break
                    except Exception as exc:
                        if isinstance(exc, BudgetExhausted):
                            mark_budget(attempt_output/ident, attempt)
                            out[ident] = budget_record(**identity, job_id=ident, attempts=attempt-1)
                            break
                        if attempt == 1 and _infrastructure(exc):
                            continue
                        out[ident] = dict(**identity, job_id=ident, status="MISSING", disposition="TRUNCATED",
                                          error=repr(exc), attempts=attempt)
                        folder=attempt_output/ident
                        folder.mkdir(parents=True,exist_ok=True)
                        state.atomic_json(folder/'spec.json',job_record(job))
                        state.atomic_json(folder/'result.json',out[ident])
                        break
            except Exception as exc:
                out[ident] = (budget_record(**identity, job_id=ident, attempts=0)
                              if isinstance(exc, BudgetExhausted) else
                              dict(**identity, job_id=ident, status="MISSING", disposition="TRUNCATED",
                                   error=repr(exc), attempts=0))
            state.atomic_json(summary_file, chr._finite({k:v for k,v in out.items() if k in ids}))
    state.atomic_json(summary_file, chr._finite({k:v for k,v in out.items() if k in ids}))
    return {key: out[key] for key in ids}


def build_manifest(states, cells, *, horizon=1500, limit=12, job_modules=(), config=None,
                   dispositions=None, provider_errors=None):
    from .frs_cohorts import first_acquired
    config = config or {}
    cohort = first_acquired(cells, limit, smoke=bool(config.get("smoke_only")))
    enrolled = {int(r["seed"]) for r in cohort}
    acquired = sorted([r for r in states if r["arm"] == "e2e1emb" and r["trigger"] == "t_acq"
                       and int(r["seed"]) in enrolled], key=lambda r: int(r["seed"]))
    jobs = []
    for row in acquired:
        if horizon - int(row['state_step']) < 5:
            if dispositions is not None:
                dispositions.append(dict(pack='H5', seed=int(row['seed']),
                                         disposition='no continuation', state_step=int(row['state_step'])))
            continue
        for tag in ("rp03", "nc03"):
            jobs.append(JobSpec(f"h5-s{row['seed']}-{tag}", row["file"], row["sha256"],
                                chr.parse_rule(tag, 1000.), horizon - int(row["state_step"]),
                                cell_id=row["cell_id"], final_rules=(chr.CE, chr.Rule("rp", .03),
                                                                  chr.Rule("rp", .10)),
                                expected_hashes={k: row[k] for k in state.HASH_KEYS},
                                metadata=dict(pack="H5", seed=int(row["seed"]),arm=row['arm'])))
    for name in job_modules:
        try:
            provider = importlib.import_module(name)
            supplied = provider.build_jobs(states, cells, config or {})
            if any(not isinstance(j, JobSpec) for j in supplied):
                raise TypeError(f"{name}.build_jobs must return JobSpec objects")
            jobs.extend(supplied)
        except Exception as exc:
            if provider_errors is None:
                raise
            provider_errors.append(dict(provider=name, status='MISSING', error=repr(exc)))
    if len({j.job_id for j in jobs}) != len(jobs):
        raise ValueError("duplicate imported job IDs")
    return jobs
