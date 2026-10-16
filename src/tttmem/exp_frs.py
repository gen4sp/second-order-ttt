"""Fresh CHR pool with additive observers and verified file-based continuations."""
from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import re
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback

import numpy as np
import torch

from . import exp_chr as chr
from . import exp_ttt as ttt
from . import frs_state as state
from .frs_backend import ChrBackend, TimedBackend
from .frs_registry import execution_guard
from . import triggers
from .pool_runtime import BudgetExhausted, budget_record, mark_budget, archive_budget, saved_attempt

ROOT = Path(__file__).resolve().parents[2]


def _backend(cfg):
    if not cfg.get("backend_factory"):
        return ChrBackend()
    module, factory = cfg["backend_factory"].split(":")
    return getattr(importlib.import_module(module), factory)(cfg)


def make_context(cfg: dict, cell: dict, dev: torch.device):
    """Same fixed sets, corruption and mediation construction as CHR."""
    execution_guard(cfg)
    if cfg.get("smoke_only") and not 0 <= int(cell["seed"]) <= 9:
        raise ValueError("pre-seal context requires a legacy cell seed")
    if not cfg.get("smoke_only") and cell not in cfg["cells"]:
        raise ValueError("cell is outside the sealed pool manifest")
    st = ttt._settings(cfg, cell["task"])
    cs = chr.chr_settings(cfg)
    task = ttt._task(st, cell["task"])
    arm, seed, eta = cell["arm"], int(cell["seed"]), float(cell["eta_base"])
    if arm not in chr.CHR_ARMS or st.eta_lr != 0 or st.spike_skip:
        raise ValueError("FRS pool needs CHR arms, fixed eta, and no main spike skip")
    ev_tokens, ev_meta = ttt.fixed_set(task, st, "eval")
    ev = ttt.eval_batches(task, ev_tokens, ev_meta, dev, st.eval_batch)
    quick = ttt.eval_batches(task, *ttt.fixed_set(task, st, "quick"), dev, st.eval_batch)
    probe = ttt.eval_batches(task, *task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]), st.n_probe),
                             dev, st.eval_batch)
    probe_c = med = None
    if arm in ttt.NOISE_ARMS:
        raw, pm = task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]), st.n_probe)
        raw, pm = ttt.corrupt_echo(raw, pm, arm, ttt.CORRUPTION_EVAL_SEED, 2,
                                   rf_probability=st.rf_probability)
        probe_c = ttt.eval_batches(task, raw, pm, dev, st.eval_batch)
    if cs.mediation_p > 0:
        raw, meta = chr.corrupt_events(ev_tokens, ev_meta, cs.mediation_p,
                                       ttt.CORRUPTION_EVAL_SEED, chr.MEDIATION_STEP)
        med = list(zip(ttt.eval_batches(task, raw, meta, dev, st.eval_batch), ev))
    ctx = chr.Ctx(task, arm, eta, st, cs, dev, seed, quick, probe, probe_c, ev, ev_meta,
                  ttt._cell_id(cell, task.steps), med)
    ctx._frs_authorized = True
    ctx._frs_cfg = cfg
    return ctx


def _observer(hooks, method, *args, rng=None):
    if hooks is not None and hasattr(hooks, method):
        with state.preserve_rngs(rng):
            return getattr(hooks, method)(*args)


def preclip_norm(parameters):
    norms = [p.grad.detach().norm().square() for p in parameters if p.grad is not None]
    return torch.stack(norms).sum().sqrt() if norms else torch.tensor(float('nan'))


def group_norms(opt):
    return {str(g.get("name", i)): float(preclip_norm(g["params"]))
            for i, g in enumerate(opt.param_groups) if any(p.grad is not None for p in g["params"])}


def _diag_row(diag):
    if not diag.get("added"):
        return {}
    clipped = torch.cat(diag["clipped"]).float()
    req = torch.cat(diag["added"]).detach().cpu().double()
    act = torch.cat(diag["actual"]).detach().cpu().double()
    out = dict(train_added_req_med=round(float(req.median()), 8),
               train_added_req_sum=round(float(req.sum()), 8),
               train_added_actual_sum=round(float(act.sum()), 8),
               train_clip_c0=round(float(clipped.mean()), 6), train_n=int(clipped.numel()),
               train_clip_count=int(clipped.sum()),
               train_match_fail_count=int(torch.cat(diag["match_fail"]).sum()))
    if diag.get("cap_bound"):
        cb = torch.cat(diag["cap_bound"]).float()
        out.update(train_cap_bound=round(float(cb.mean()), 6), train_cap_bound_count=int(cb.sum()))
    if diag.get("cap_hit"):
        out["train_cap_hit_mean"] = round(float(torch.cat(diag["cap_hit"]).mean()), 6)
    diag.clear()
    return out


def frs_segment(model, opt, rng, ctx, rule, branch, start, *, stop=None, nonfinite0=0,
                hooks=None, reads=(), backend=None, outer_mask=None, skip_controller=None,
                save_at=(), output_dir=None, hashes=None, log_every=None,
                expected_first_sha=None) -> dict:
    """CHR update arithmetic, plus observers; a branch's divergence count is local."""
    if not 0 <= ctx.seed <= 9 and not getattr(ctx, "_frs_authorized", False):
        raise ValueError("fresh continuation lacks a verified execution seal")
    backend = backend or ChrBackend()
    st, cs, task, arm, dev = ctx.st, ctx.cs, ctx.task, ctx.arm, ctx.dev
    end = task.steps if stop is None else int(stop)
    cadence = cs.log_every if log_every is None else log_every
    if not 0 <= start < end <= task.steps:
        raise ValueError("invalid continuation bounds")
    warmup = ttt.arm_warmup(arm, st)
    total = st.lr_total or task.steps
    if total < end or st.spike_skip:
        raise ValueError("FRS needs lr_total >= horizon and the generic skip hook")
    output = Path(output_dir or "checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    stats, diag = ttt.GradStats(), {}
    logs, probes, saved, read_rows, states, drops = [], [], [], [], [], []
    nonfinite, diverged, t_div = 0, False, None
    first_sha = None
    times = []
    wall0, log_s = time.time(), 0.0
    _observer(hooks,'at_start',start,model,opt,rng,ctx,rng=rng)
    cross_rules = [r for r in [chr.CE, *cs.rules("cross_rules", arm)] if r != rule]
    for step in range(start, end):
        if time.time() >= getattr(ctx, '_frs_cfg', {}).get('_deadline',float('inf')):
            raise BudgetExhausted('pool segment deadline exhausted')
        ttt.set_kamin_regime(model, arm, step, st)
        if skip_controller is None:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * ttt.lr_scale(step, warmup, total, st.lr_floor)
        raw, meta = task.raw(rng, st.batch)
        ctx._frs_batch_sha=chr._sha1(raw)
        if first_sha is None:
            first_sha = chr._sha1(raw)
            if expected_first_sha is not None and first_sha != expected_first_sha:
                raise ValueError("actual first batch differs from verified source state")
        raw, meta = ttt.corrupt_echo(raw, meta, arm, ctx.seed, step, rf_probability=st.rf_probability)
        idx, y, meta = task.tensors(raw, meta, dev)
        if outer_mask is not None:
            idx, y, meta = outer_mask(idx, y, meta, step=step, ctx=ctx)
        ctx._frs_update_inputs = dict(idx=idx, y=y, meta=meta)
        ttt._sync(dev)
        t0 = time.perf_counter()
        observer = _observer(hooks, "loss_observer", step + 1, model, ctx, rng=rng)
        if observer is None:
            loss = backend.loss(model, idx, y, meta, ctx, rule, stats, diag)
        else:
            loss = backend.loss(model, idx, y, meta, ctx, rule, stats, diag, observer=observer)
        opt.zero_grad(set_to_none=True)
        done = step + 1
        telemetry = dict(loss=float(loss.detach()), finite_loss=bool(torch.isfinite(loss).item()),
                         grad_norm=None, skipped=False, nonfinite=False)
        if hooks is not None:
            n_chunk=task.T//task.chunk
            telemetry['inner_chunks']={str(c):dict(norm_mean=float(n.detach().mean()),
                norm_median=float(n.detach().median()),clip_share=float(cl.float().mean()))
                for c,n,cl in zip(stats.indices[-n_chunk:],stats.norms[-n_chunk:],stats.clips[-n_chunk:])}
        finite = telemetry["finite_loss"]
        skip = False
        if finite:
            loss.backward()
            if skip_controller or hooks is not None:
                telemetry["grad_norm"] = float(preclip_norm(ttt.clipped_params(model)))
                rho_finite=model.rho.grad is None or bool(torch.isfinite(model.rho.grad).all())
                if not np.isfinite(telemetry['grad_norm']) or not rho_finite:
                    telemetry['grad_norm']=float('nan')
            if hooks is not None and hasattr(hooks, "pre_clip"):
                _observer(hooks, "pre_clip", done, group_norms(opt), model, opt, ctx, rng=rng)
            if skip_controller:
                skip = bool(skip_controller(done, telemetry))
            norm = torch.nn.utils.clip_grad_norm_(ttt.clipped_params(model), st.grad_clip)
            rho_norm = (model.rho.grad.norm() if model.rho.grad is not None
                        else torch.zeros((), device=norm.device)).to(norm.device)
            finite = bool(torch.isfinite(torch.stack([norm.float(), rho_norm.float()])).all())
            if telemetry['grad_norm'] is None:
                telemetry["grad_norm"] = float(norm)
            if finite and not skip:
                _observer(hooks, "before_update", done, model, opt, rng, ctx, rng=rng)
                if skip_controller is not None:
                    for group in opt.param_groups:
                        group['lr']=group['base_lr']*ttt.lr_scale(step,warmup,total,st.lr_floor)
                opt.step()
        elif skip_controller:
            telemetry["grad_norm"] = float("nan")
            skip = bool(skip_controller(done, telemetry))
        if not finite:
            nonfinite += 1
            opt.zero_grad(set_to_none=True)
        if skip:
            drops.append(done)
        telemetry.update(skipped=skip, nonfinite=not finite, nonfinite_local=nonfinite,
                         nonfinite_cumulative=nonfinite0 + nonfinite)
        ttt._sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
        _observer(hooks, "after_update", done, telemetry, model, opt, rng, ctx, rng=rng)
        cache = {}
        if done % cadence == 0 or done == end:
            t_log = time.perf_counter()
            row = dict(cell_id=ctx.ident, branch=branch, rule=rule.tag, step=done,
                       loss=round(float(loss.detach()), 6),
                       quick=backend.quick(model, ctx, rule, geometry_on=True),
                       write=backend.write(model, ctx, rule), **_diag_row(diag))
            cache[(rule.tag, False)] = row["quick"]
            if done % cs.cross_every == 0 or done == end:
                row["cross"] = {r.tag: backend.quick(model, ctx, r) for r in cross_rules}
                cache.update({(r, False): v for r, v in row["cross"].items()})
                if rule != chr.CE:
                    row["write_ce"] = backend.write(model, ctx, chr.CE)
            if not getattr(backend, "drop_r4_probe", False) and (done % cs.probe_every == 0 or done == end):
                if ctx.probe_corrupted is not None:
                    row["write_corrupted"] = chr.write_log(model, ctx.probe_corrupted, ctx.eta,
                                                           st.g_clip, task.chunk, rule)
                p = ttt.r4_probe(model, task, ctx.probe, ctx.eta, st.g_clip, arm=arm)
                if ctx.probe_corrupted is not None:
                    p.update(ttt.kv_w0_stats(model, task, ctx.probe_corrupted, arm))
                p = dict(cell_id=ctx.ident, branch=branch, step=done, train_rule=rule.tag,
                         measurement_rule="ce", **p)
                probes.append(p)
                chr._emit("CHR_PROBE", chr._finite(p))
            logs.append(row)
            chr._emit("CHR_LOG", chr._finite(row))
            _observer(hooks, "after_log", row, model, opt, rng, ctx, rng=rng)
            log_s += time.perf_counter() - t_log
        for schedule in reads:
            if not schedule.active(done, start):
                continue
            key = (schedule.rule.tag, schedule.full)
            if key not in cache:
                cache[key] = (backend.final(model, ctx, schedule.rule, [schedule.rule])[schedule.rule.tag]
                              if schedule.full else backend.quick(model, ctx, schedule.rule))
            if not any(r["step"] == done and r["rule"] == key[0] and r["set"] ==
                       ("full" if key[1] else "quick") for r in read_rows[-len(reads):]):
                rec = dict(cell_id=ctx.ident, branch=branch, step=done, rule=key[0],
                           set="full" if key[1] else "quick", quick=cache[key])
                read_rows.append(rec)
                chr._emit("FRS_READ", chr._finite(rec))
        _observer(hooks, "after_step", done, model, opt, rng, ctx, cache, rng=rng)
        if done in save_at:
            if hashes is None:
                raise ValueError("route save_at needs provenance hashes")
            extra = {}
            for name, controller in (("outer_mask", outer_mask), ("skip_controller", skip_controller)):
                if hasattr(controller, "state_dict"):
                    extra[name] = controller.state_dict()
            states.append(state.save_state(output / f"state-S2-step{done}.pt", model, opt, rng, ctx,
                                           trigger="S2", event_step=done, detected_step=done,
                                           state_step=done, nonfinite=nonfinite0 + nonfinite,
                                           hashes=hashes, detector_tail=logs, extra=extra))
        full_at_step = done in save_at or (hooks is not None and hasattr(hooks, "has_state") and hooks.has_state(done))
        if (done != end or branch == "main") and not full_at_step and chr._save_here(cs, arm in cs.ckpt_dense_arms, branch, rule,
                                             start, done, end):
            file = output / f"weights-step{done}.pt"
            torch.save(state._cpu(model.state_dict()), file)
            saved.append(str(file))
        attempted = done - start
        if attempted >= 20 and nonfinite / attempted > .05:
            diverged, t_div = True, done
            break
    return dict(logs=logs, probes=probes, reads=read_rows, saved=saved, states=states, drops=drops,
                nonfinite=nonfinite, nonfinite_local=nonfinite,
                nonfinite_cumulative=nonfinite0 + nonfinite, diverged=diverged, t_div=t_div,
                divergence_certified=diverged, attempted=done - start, first_batch_sha1=first_sha,
                steps_done=done, segment_wall_s=round(time.time() - wall0, 3),
                log_wall_s=round(log_s, 3), fork_wall_s=0.0,
                ms_step=round(float(np.median(times[min(100, len(times) // 2):] or times or [0])), 3),
                grad=stats.summary())


class SmokeLogger:
    """Legacy-seed smoke adapter; the TRG per-event ladder is an external module."""
    def telemetry(self, step, data):
        pass

    def light(self, model, ctx, step):
        return ChrBackend().quick(model, ctx, chr.CE)

    def ladder(self, model, ctx, step):
        return dict(smoke_adapter=True, **self.light(model, ctx, step))


class PoolHooks:
    def __init__(self, ctx, cfg, folder, hashes):
        self.cfg, self.folder, self.hashes = cfg, Path(folder), hashes
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rows, self.states, self.coverage, self.telemetry_rows = [], [], [], []
        self.events = {}
        self.window = None
        self.candidate = None
        self.pending_weights = []
        self.logger_errors = {}
        self.bind = float(cfg.get("trigger_bind", .90))
        self.loss_bind = self.bind
        self.residual = float(cfg.get("trigger_residual", .01))
        if cfg.get("coverage_factory"):
            module, name = cfg["coverage_factory"].split(":")
            self.logger = getattr(importlib.import_module(module), name)(ctx, self.folder, cfg)
        elif cfg.get("smoke_only"):
            self.logger = SmokeLogger()
        else:
            raise ValueError("production requires the sealed TRG logger")
        if any(not callable(getattr(self.logger, name, None)) for name in ("telemetry", "light", "ladder")):
            raise TypeError("coverage logger must implement telemetry, light and ladder")

    def logger_call(self, method, *args):
        if method in self.logger_errors:
            return None
        try:
            return getattr(self.logger, method)(*args)
        except Exception as exc:
            self.logger_errors[method] = repr(exc)
            chr._emit("FRS_OBSERVER_FAILED", dict(method=method, error=repr(exc)))
            if self.window:
                self.window.fail_open = True
                self.window.stop = self.window.cap
            return None

    def loss_observer(self, step, model, ctx):
        if not hasattr(self.logger, "loss_observer"):
            return None
        observer = self.logger_call("loss_observer", step, model, ctx)
        if observer is None:
            return None
        def guarded(*args, **kwargs):
            try:
                with state.preserve_rngs():
                    return observer(*args, **kwargs)
            except Exception as exc:
                if "loss_observer" not in self.logger_errors:
                    self.logger_errors["loss_observer"] = repr(exc)
                    chr._emit("FRS_OBSERVER_FAILED", dict(method="loss_observer", error=repr(exc)))
        return guarded

    def pre_clip(self, step, norms, model, opt, ctx):
        if hasattr(self.logger, "pre_clip"):
            self.logger_call("pre_clip", step, norms, model, opt, ctx)

    def after_update(self, step, telemetry, model, opt, rng, ctx):
        self.nonfinite = telemetry["nonfinite_cumulative"]
        self.telemetry_rows.append(dict(step=step, **chr._finite(telemetry)))
        self.logger_call("telemetry", step, dict(telemetry, model=model, opt=opt, ctx=ctx,
                                                 **getattr(ctx, "_frs_update_inputs", {})))

    def before_update(self, step, model, opt, rng, ctx):
        if hasattr(self.logger, "before_update"):
            self.logger_call("before_update", step, model, opt, ctx)

    def persist_states(self):
        # Candidate insurance is not committed until the next log confirms loss.
        with open(self.folder / "states.jsonl", "w") as f:
            for row in self.states:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def _save(self, trigger, step, model, opt, rng, ctx, *, candidate=False):
        file = self.folder / f"state-{trigger}-step{step}.pt"
        row = state.save_state(file, model, opt, rng, ctx, trigger=trigger,
                               event_step=step, detected_step=step, state_step=step,
                               nonfinite=self.nonfinite, hashes=self.hashes, detector_tail=self.rows)
        self.states.append(row)
        if not candidate:
            self.persist_states()
        chr._emit("FRS_STATE_CANDIDATE" if candidate else "FRS_STATE", row)
        return row

    def discard_candidate(self, ctx):
        if self.candidate is None:
            return
        at = self.candidate["state_step"]
        if chr._save_here(ctx.cs, ctx.arm in ctx.cs.ckpt_dense_arms, "main", chr.CE, 0, at, ctx.task.steps):
            payload = torch.load(self.candidate["file"], map_location="cpu", weights_only=False)
            file = self.folder / f"weights-step{at}.pt"
            torch.save(payload["model"], file)
            self.pending_weights.append(str(file))
        Path(self.candidate["file"]).unlink()
        self.candidate = None

    def has_state(self, step):
        return any(r["state_step"] == step for r in self.states) or (
            self.candidate is not None and self.candidate["state_step"] == step)

    def after_log(self, row, model, opt, rng, ctx):
        self.rows.append(row)
        with open(self.folder / "logs.jsonl", "a") as f:
            f.write(json.dumps(chr._finite(row), sort_keys=True) + "\n")
        found = triggers.detect(self.rows, bind=self.bind, residual=self.residual, grid=ctx.cs.log_every)
        # Smoke-only relaxed loss criterion fires without fabricating accuracy logs.
        if self.cfg.get("smoke_only") and self.cfg.get("smoke_force_loss") and "t_acq" in found:
            at = found["t_acq"].event_step
            if row["step"] >= at + 2 * ctx.cs.log_every:
                found["t_loss"] = triggers.Trigger("t_loss", at + ctx.cs.log_every,
                                                   at + 2 * ctx.cs.log_every, None)
                low = [r for r in self.rows if r["step"] > at + ctx.cs.log_every and
                       r["write"]["r_kv_med"] < self.residual]
                if len(low) >= 3:
                    step = low[2]["step"]
                    found["S1"] = triggers.Trigger("S1", step, step, step)
        for name, event in found.items():
            if name in self.events:
                continue
            self.events[name] = dataclasses.asdict(event)
            chr._emit("FRS_TRIGGER", dict(cell_id=ctx.ident, **self.events[name]))
            if name == "t_acq":
                self._save(name, event.state_step, model, opt, rng, ctx)
                if hasattr(self.logger,'acquisition'):
                    self.logger_call('acquisition',event.state_step)
                self.window = triggers.CoverageWindow(ctx.arm, event.event_step, bind=self.bind,
                    span=int(self.cfg.get("coverage_span", 300)), tail=int(self.cfg.get("coverage_tail", 60)),
                    so_stop=int(self.cfg.get("so_coverage_stop", 660)))
            elif name == "S1" and ctx.arm == "e2e1emb":
                if event.state_step != row["step"]:
                    raise ValueError("S1 must save the third-log detection state")
                self._save(name, event.state_step, model, opt, rng, ctx)
        if self.candidate and row["step"] > self.candidate["state_step"]:
            loss = self.events.get("t_loss")
            if loss and loss["event_step"] <= self.candidate["state_step"]:
                self.states.append(self.candidate)
                self.persist_states()
                chr._emit("FRS_STATE", self.candidate)
            else:
                self.discard_candidate(ctx)
            self.candidate = None

    def after_step(self, step, model, opt, rng, ctx, cache):
        if self.window and self.window.active(step):
            ladder = ctx.arm != "e2e2emb" or step % 5 == 0
            light = self.logger_call("ladder" if ladder else "light", model, ctx, step)
            if light is not None and (not isinstance(light,dict) or 'acc' not in light):
                raise ValueError('coverage adapter must return flat acc')
            try:
                if not isinstance(light, dict) or "acc" not in light:
                    raise ValueError("coverage adapter must return flat acc")
                self.window.observe(dict(step=step, **light))
            except Exception as exc:
                self.window.fail_open = True
                self.window.stop = self.window.cap
                self.logger_errors.setdefault("coverage", repr(exc))
            scalars = {k: v for k, v in (light or {}).items()
                       if isinstance(v, (int, float, bool, str)) or v is None}
            row = dict(cell_id=ctx.ident, step=step, light=scalars, ladder=ladder,
                       fail_open=self.window.fail_open)
            self.coverage.append(row)
            chr._emit("FRS_COVERAGE", dict(cell_id=ctx.ident, step=step,
                                           acc=scalars.get("acc"), fail_open=self.window.fail_open))
        if ctx.arm == "e2e1emb" and step % 50 == 0 and step <= min(1000, ctx.task.steps):
            loss = self.events.get("t_loss")
            if loss and step >= loss["event_step"]:
                if not self.has_state(step):
                    self._save("insurance", step, model, opt, rng, ctx)
            elif "t_acq" in self.events and step > self.events["t_acq"]["event_step"] and self.rows and \
                    self.rows[-1]["step"] == step and self.rows[-1]["quick"]["acc"] < self.bind:
                # Sparse candidate at a global-50 first low log, never a t_loss state.
                row = self._save("insurance", step, model, opt, rng, ctx, candidate=True)
                self.states.pop()
                self.candidate = row


def _write_csv(path: Path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _main_failure(cfg, cell, reason):
    ident = ttt._cell_id(cell, cfg['steps'])
    folder = Path(cfg['output_dir']) / 'mains' / ident.replace('/', '__')
    def rows(base, name):
        file = base / name
        if not file.exists():
            return []
        out = []
        for line in file.read_text().splitlines():
            try:
                row = json.loads(line)
                if name == 'states.jsonl':
                    row['file'] = str(base / Path(row['file']).name)
                out.append(row)
            except json.JSONDecodeError:
                break  # a killed writer may leave a partial last row
        return out
    attempts = [*sorted(folder.parent.glob(folder.name + '.partial-*')), folder]
    saved = {(r['trigger'], r['state_step']): r for base in attempts for r in rows(base, 'states.jsonl')
             if Path(r['file']).is_file() and state.file_sha256(r['file']) == r['sha256']}
    logs = max((rows(base, 'logs.jsonl') for base in attempts), key=len, default=[])
    return dict(cell_id=ident, arm=cell['arm'], seed=int(cell['seed']), task=cell['task'],
                eta_base=cell['eta_base'], status='MISSING', disposition='TRUNCATED',
                retryable=bool(re.search(r'OSError|TimeoutError|MemoryError|watchdog|worker lost|interrupted|out of memory',str(reason))),
                error=str(reason), states=list(saved.values()),
                logs=logs, events={k: dataclasses.asdict(v) for k, v in triggers.detect(logs).items()})


def run_frs_cell(cfg, cell, dev):
    from .frs_jobs import _infrastructure
    ident = ttt._cell_id(cell, cfg['steps'])
    folder = Path(cfg['output_dir'])/'mains'/ident.replace('/', '__')
    cut = folder/'budget.json'
    initial = (json.loads(cut.read_text())['attempt'] if cfg.get('resume') and cut.is_file()
               else int(cfg.get('_first_attempt', 1)))
    if cfg.get('resume') and folder.with_name(folder.name+'.partial-1').exists():
        initial = max(initial, 2)
    if cfg.get('resume') and folder.exists() and any(folder.iterdir()) and not cut.is_file():
        record_file = folder/'cell.json'
        previous = json.loads(record_file.read_text()) if record_file.is_file() else {}
        if previous.get('status') != 'DONE':
            initial = max(initial, 2)  # interrupted/lost attempt 1 already spent its slot
    for attempt in range(initial, 3):
        try:
            record = _run_frs_cell({**cfg, '_attempt': attempt,
                                    'resume': bool(cfg.get('resume') or attempt > 1)}, cell, dev)
            record.setdefault('attempts', attempt)
            return record
        except Exception as exc:
            if isinstance(exc, BudgetExhausted):
                mark_budget(folder, attempt)
                record = _main_failure(cfg, cell, 'budget exhausted')
                record.update(budget_record(attempts=attempt-1))
                record.pop('error', None)
                return record
            if attempt == 1 and _infrastructure(exc):
                continue
            record = _main_failure(cfg, cell, repr(exc))
            record['retryable']=_infrastructure(exc) or 'interrupted' in str(exc).lower()
            record['attempts'] = attempt
            folder = Path(cfg['output_dir']) / 'mains' / record['cell_id'].replace('/', '__')
            folder.mkdir(parents=True, exist_ok=True)
            state.atomic_json(folder / 'cell.json', record)
            return record


def _run_frs_cell(cfg, cell, dev):
    ctx = make_context(cfg, cell, dev)
    folder = Path(cfg["output_dir"]) / "mains" / ctx.ident.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    hashes = state.context_hashes(ctx, cfg)
    record_file = folder / "cell.json"
    if cfg.get("resume") and record_file.exists():
        record = json.loads(record_file.read_text())
        if record.get("hashes", hashes) != hashes:
            raise ValueError("resume main provenance mismatch")
        for row in record["states"]:
            row['file']=state.relocated_file(row['file'],cfg['output_dir'],folder=folder)
            state.verify_state(row["file"], ctx, sha256=row["sha256"], expected_hashes=hashes)
        for weight in record.get("weights", []):
            weight['file']=state.relocated_file(weight['file'],cfg['output_dir'],folder=folder)
            if state.file_sha256(weight["file"]) != weight["sha256"]:
                raise ValueError("resume main weights checksum mismatch")
        if record.get("steps_done") == ctx.task.steps or record.get("diverged"):
            record['saved']=[state.relocated_file(p,cfg['output_dir'],folder=folder) for p in record.get('saved',[])]
            for key in ('shard_manifest','telemetry_file'):
                if (record.get('logger_final') or {}).get(key):
                    record['logger_final'][key]=state.relocated_file(record['logger_final'][key],cfg['output_dir'])
            chr._emit("FRS_CELL_RESUMED", dict(cell_id=ctx.ident, states=len(record["states"])))
            return record
        if record.get('error') and not record.get('retryable',True):
            return record # Scientific/code failures are terminal; only infrastructure retries.
    if any(folder.iterdir()):
        if not cfg.get('resume'):
            raise FileExistsError("partial main output exists; enable resume")
        if (folder/'budget.json').is_file():
            archive_budget(folder)
        else:
            previous = sorted(folder.parent.glob(folder.name + '.partial-*'))
            if previous:
                raise RuntimeError("main retry exhausted; seed slot stays missing")
            # Same seed/protocol, restarted from zero; no outcome-adaptive replacement.
            folder.rename(folder.with_name(folder.name + '.partial-1'))
        folder.mkdir()
    if time.time() >= cfg.get('_deadline', float('inf')):
        raise BudgetExhausted('pool main deadline exhausted')
    state.atomic_json(folder/'attempt.json', dict(attempt=cfg.get('_attempt', 1)))
    model = ttt.build_model(ctx.task, ctx.st, dev, ctx.seed, ctx.arm)
    opt = ttt.build_optim(model, ctx.st)
    rng = np.random.default_rng([ctx.seed, ttt.TRAIN_SEED])
    hooks = PoolHooks(ctx, cfg, folder, hashes)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    if cfg.get('_unit_started'):
        cfg['_unit_started']()  # training-ready, after model/fixed-set initialization
    start = time.time()
    backend=TimedBackend(_backend(cfg))
    res = frs_segment(model, opt, rng, ctx, chr.CE, "main", 0, hooks=hooks,
                      backend=backend, output_dir=folder)
    res['read_costs']=backend.costs
    model.eval()
    final_start=time.perf_counter()
    res["final"] = chr.final_eval(model, ctx, chr.CE, "main") if not res["diverged"] else {
        r.tag: dict(acc=0., divergence_coded=True) for r in [chr.CE, *ctx.cs.rules("final_rules", ctx.arm)]}
    res["mediation"] = chr.mediation_eval(model, ctx) if ctx.med and not res["diverged"] else None
    ttt._sync(dev)
    res['read_costs']={**backend.costs,'final_and_mediation':dict(calls=1,wall_s=time.perf_counter()-final_start)}
    hooks.discard_candidate(ctx)
    res["saved"].extend(hooks.pending_weights)
    res["states"] = hooks.states
    with state.preserve_rngs(rng):
        logger_final = hooks.logger_call("finalize") if hasattr(hooks.logger, "finalize") else None
    record = dict(cell_id=ctx.ident, task=ctx.task.name, arm=ctx.arm, seed=ctx.seed,
                  eta_base=ctx.eta, device=str(dev), steps=ctx.task.steps, hashes=hashes,
                  attempts=cfg.get('_attempt', 1),
                  status="DONE", disposition="DIVERGED" if res['diverged'] else "COMPLETE",
                  events=hooks.events, logger_final=logger_final, logger_errors=hooks.logger_errors,
                  telemetry_failed=bool(hooks.logger_errors), wall_s=round(time.time()-start, 3),
                  training_started_at=start, training_finished_at=time.time(),
                  coverage_fail_open=bool(hooks.window and hooks.window.fail_open),
                  coverage_stop=None if hooks.window is None else hooks.window.stop,
                  memory={} if dev.type != "cuda" else dict(
                    cuda_peak_alloc_mib=torch.cuda.max_memory_allocated(dev)/2**20,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(dev)/2**20),
                  **res)
    record["weights"] = [dict(file=p, sha256=state.file_sha256(p)) for p in res["saved"]]
    if cfg.get('_determinism_probe'):
        torch.save(state._cpu(model.state_dict()), folder/'probe-final.pt')
        record['probe_final'] = str(folder/'probe-final.pt')
    (folder / "series.json").write_text(json.dumps(chr._finite(res["logs"])))
    (folder / "telemetry.json").write_text(json.dumps(hooks.telemetry_rows))
    (folder / "coverage.json").write_text(json.dumps(chr._finite(hooks.coverage)))
    # Per-event payloads belong to the TRG shards, referenced by logger_final.
    np.savez_compressed(folder / "coverage.npz", step=np.asarray([r["step"] for r in hooks.coverage]),
                        acc=np.asarray([r["light"].get("acc", np.nan) for r in hooks.coverage]))
    state.atomic_json(record_file, chr._finite(record))
    chr._emit("FRS_CELL", {k: v for k, v in chr._finite(record).items() if k not in ("logs", "probes", "reads")})
    return record


def _worker(index, cfg, mode, jobs, results, lock):
    sys = importlib.import_module("sys")
    from .pool_dispatch import PipeLines
    sys.stdout = PipeLines(results, index, 'stdout')
    sys.stderr = PipeLines(results, index, 'stderr')
    ttt._mark_process(f"{cfg['process_name']}-w{index}")
    torch.set_num_threads(int(cfg.get("threads_per_worker", 2)))
    dev = ttt._device(cfg["device"])
    ttt.configure_tf32(False, str(index))
    while True:
        item = jobs.get()
        if item is None:
            break
        key, payload, attempt = item
        unit_cfg = {**cfg, **({'resume': True, '_first_attempt': 2} if attempt == 2 else {})}
        unit_cfg['_unit_started'] = lambda: results.put((index, key, dict(_pool_event='start')))
        try:
            if time.time() >= cfg["_deadline"]:
                raise BudgetExhausted("worker budget exhausted")
            if mode == "main":
                record = run_frs_cell(unit_cfg, payload, dev)
            else:
                unit_cfg['_unit_started']()
                from .frs_jobs import run_queue
                cells = {ttt._cell_id(c, cfg['steps']): c for c in cfg["cells"]}
                factory = lambda j: make_context(unit_cfg, cells[j.cell_id], dev)
                record = run_queue(payload['jobs'], factory, Path(cfg["output_dir"]) / "jobs",
                                   backend=_backend(unit_cfg), resume=bool(unit_cfg.get("resume")), prior=payload['prior'],
                                   first_attempt=attempt)
            results.put((index, key, record if mode == 'main' else record[key]))
        except Exception as exc:
            traceback.print_exc()
            if mode == 'main':
                rec = _main_failure(cfg, payload, repr(exc))
            else:
                job = payload['jobs'][0]
                rec = dict(job_id=key, cell_id=job.cell_id, metadata=job.metadata,
                           seed=job.metadata.get('seed'), arm=job.metadata.get('arm'), rule=job.rule.tag,
                           status='MISSING', disposition='TRUNCATED', error=repr(exc))
            if isinstance(exc, BudgetExhausted):
                rec.update(budget_record())
                rec.pop('error', None)
            results.put((index, key, rec))
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()


def _parallel(cfg, mode, items, progress=None):
    if int(cfg["workers"]) == 1:
        dev = ttt._device(cfg["device"])
        out = {}
        for key, payload in items:
            if time.time() >= cfg["_deadline"]:
                out[key] = _main_failure(cfg, payload, 'budget exhausted')
                out[key].update(budget_record())
                out[key].pop('error', None)
                if progress:
                    progress(out)
                continue
            try:
                if mode == "main":
                    out[key] = run_frs_cell(cfg, payload, dev)
                else:
                    from .frs_jobs import run_queue
                    cells = {ttt._cell_id(c, cfg['steps']): c for c in cfg["cells"]}
                    out[key] = run_queue(payload, lambda j: make_context(cfg, cells[j.cell_id], dev),
                                         Path(cfg["output_dir"]) / "jobs" / key.replace("/", "__"),
                                         backend=_backend(cfg), resume=bool(cfg.get("resume")))
            except Exception as exc:
                traceback.print_exc()
                out[key] = dict(error=repr(exc))
            if mode == 'main' and out[key].get('error'):
                out[key] = _main_failure(cfg, payload, out[key]['error'])
            if progress:
                progress(out)
        return out
    return _dispatch_units(cfg, mode, items, progress)


def _job_cost(job):
    # Registered update rates, including the rule/arm cost rather than horizon alone.
    arm = job.metadata.get('arm', '')
    rates = {'H5': 8424/27600, 'TRG': 9600/28480, 'rdo': 14869.1/43200,
             'rpe': 12691.2/28800, 'late': 3612.4/10800}
    rate = rates.get(job.metadata.get('pack'), 1.0)
    rate *= 1.35 if arm == 'e2e2emb' else 1.1 if arm == 'e2e1emb_cn97' else 1.0
    return rate * job.horizon


def _dispatch_units(cfg, mode, items, progress, prior=None):
    from .pool_dispatch import dispatch
    lookup = dict(items)
    order = {key: i for i,(key,_) in enumerate(items)}
    def missing(key, reason, resumable, *, dispatched=False):
        if mode == 'main':
            rec = _main_failure(cfg, lookup[key], reason)
            if resumable:
                rec.update(budget_record())
                rec.pop('error', None)
                folder = Path(cfg['output_dir'])/'mains'/key.replace('/', '__')
                if dispatched:
                    mark_budget(folder, saved_attempt(folder))
            return rec
        job = lookup[key]
        if reason == 'budget exhausted' and dispatched:
            folder = Path(cfg['output_dir'])/'jobs'/key
            retry = Path(cfg['output_dir'])/'jobs'/'retry'/key
            if retry.exists():
                mark_budget(retry, saved_attempt(retry, 2))
            elif folder.exists():
                mark_budget(folder, saved_attempt(folder))
        return dict(job_id=key, cell_id=job.cell_id, arm=job.metadata.get('arm'),
                    seed=job.metadata.get('seed'), rule=job.rule.tag, metadata=job.metadata,
                    status='MISSING', disposition='TRUNCATED', reason=reason, resumable=resumable)
    def worker_lost(key, exitcode, scheduled_attempt):
        root = Path(cfg['output_dir'])
        if mode == 'main':
            folder = root/'mains'/key.replace('/', '__')
            attempt = max(scheduled_attempt, saved_attempt(folder))
            terminal = _main_failure(cfg, lookup[key], f'worker lost (exit {exitcode})')
            terminal.update(attempts=attempt, retryable=attempt < 2)
            if attempt >= 2:
                state.atomic_json(folder/'cell.json', terminal)
        else:
            folder = root/'jobs'/key
            retry = root/'jobs'/'retry'/key
            attempt = max(scheduled_attempt, saved_attempt(retry, 2) if retry.exists() else saved_attempt(folder))
            terminal = missing(key, f'worker lost (exit {exitcode})', False)
            terminal.update(attempts=attempt, error=f'worker died (exit {exitcode})')
            if attempt >= 2:
                from .frs_jobs import job_record
                state.atomic_json(retry/'spec.json', job_record(lookup[key]))
                state.atomic_json(retry/'result.json', terminal)
        chr._emit('POOL_WORKER_LOST', dict(unit=key, attempt=attempt, exitcode=exitcode,
                                          retry=attempt < 2))
        return attempt, terminal
    return dispatch(cfg, items, mode=mode, worker=_worker,
                    dependencies=lambda k: () if mode == 'main' else lookup[k].after,
                    payload_for=lambda k, out: lookup[k] if mode == 'main' else
                        dict(jobs=[lookup[k]], prior={d: out[d] for d in lookup[k].after}),
                    missing=missing, worker_lost=worker_lost,
                    priority=lambda k: (order[k], k) if mode == 'main' else (-_job_cost(lookup[k]), k),
                    progress=progress, prior=prior)


def _global_jobs(cfg, jobs, progress, prior=None):
    """One dependency DAG, dispatching longest registered cost first."""
    from .frs_jobs import run_queue
    cells = {ttt._cell_id(c, cfg['steps']): c for c in cfg['cells']}
    if int(cfg['workers']) == 1:
        dev = ttt._device(cfg['device'])
        result = run_queue(jobs, lambda j: make_context(cfg, cells[j.cell_id], dev),
                           Path(cfg['output_dir'])/'jobs', backend=_backend(cfg), resume=cfg.get('resume',False),
                           prior=prior)
        result = {**(prior or {}), **result}
        progress(result)
        return result
    return _dispatch_units(cfg, 'job', [(j.job_id, j) for j in jobs], progress, prior)


def run(cfg, seed):
    from .frs_jobs import build_manifest, job_record
    seal = execution_guard(cfg)
    cfg = {**cfg, "_deadline": time.time() + float(cfg.get("max_minutes", 360)) * 60}
    if cfg.get('smoke_attempt_subdir'):
        if not cfg.get('smoke_only'):
            raise ValueError('attempt subdirectories are smoke-only')
        base = Path(cfg['output_dir']); number = 1
        while (base/f'attempt-{number:02d}').exists():
            number += 1
        cfg['output_dir'] = str(base/f'attempt-{number:02d}')
    st = ttt._settings(cfg)
    ttt._mark_process(cfg["process_name"])
    dev = ttt._device(st.device)
    if cfg.get("require_cuda") and dev.type != "cuda":
        raise ValueError("production FRS protocol requires CUDA")
    if dev.type == "mps" or (dev.type == "cpu" and int(cfg["workers"]) != 1):
        raise ValueError("local FRS smoke uses exactly one CPU process; MPS is disabled")
    torch.set_num_threads(min(2, int(cfg.get("threads_per_worker", 2))))
    ttt.configure_tf32(False, "main")
    output = Path(cfg["output_dir"]).absolute()
    cfg['output_dir'] = str(output)  # absolute without resolving symlinks
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(protocol=cfg["protocol"], smoke_only=bool(cfg.get("smoke_only")),
                      plan_sha256=state.file_sha256(ROOT / cfg["plan_file"]) if cfg.get("plan_file") else None,
                      registry_sha256=state.file_sha256(ROOT / "configs/toy/pool/seed-registry.toml"),
                      source_sha256=state.source_hash(ROOT), device=str(dev),
                      effective_config={k: v for k, v in cfg.items() if not k.startswith("_")},
                      seal_sha256=None if seal is None else state.file_sha256(cfg.get("seal_file", "frs_seal.json")))
    chr._emit("FRS_PROTOCOL", provenance)
    sanity = chr.chr_sanity()
    chr._emit("FRS_SANITY", sanity)
    if not all(s["passed"] for s in sanity.values()):
        raise RuntimeError("CHR sanity failed")
    cells = list(cfg["cells"])
    ids = [ttt._cell_id(c, st.steps) for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pool cells")
    costs = {"e2e2emb": 1.35, "e2e1emb_cn97": 1.1, "e2e1emb": 1.0}
    ordered = sorted(cells, key=lambda c: (-costs[c["arm"]], int(c["seed"])))
    if cfg.get('smoke_only') and int(cfg['workers']) >= 5:
        ordered = sorted(cells, key=lambda c: (c['arm'] != 'e2e1emb', int(c['seed'])))
    result = dict(schema=1, complete=False, protocol=cfg['protocol'], smoke_only=bool(cfg.get('smoke_only')),
                  provenance=provenance, cells={}, jobs={}, states=[], output_dir=str(output),
                  plan_sha256=provenance['plan_sha256'], provider_errors=[], dispositions=[])
    def progress(mains):
        result['cells'] = dict(mains)
        result['states'] = [row for rec in mains.values() for row in rec.get('states', [])]
        state.atomic_json(output / 'pool.json', chr._finite(result))
    progress({})
    phase0 = time.time()
    mains = _parallel(cfg, "main", [(ttt._cell_id(c, st.steps), c) for c in ordered], progress)
    from .pool_dispatch import overlap_peak
    result['concurrency'] = dict(mains=overlap_peak(mains.values()),
                                 evidence='overlapping worker training timestamps')
    chr._emit('POOL_CONCURRENCY', result['concurrency'])
    chr._emit('POOL_PHASE_RATE', dict(phase='mains', wall_s=time.time()-phase0,
                                     worker_s=sum(r.get('wall_s',0) for r in mains.values()),
                                     attempted=sum(r.get('attempted', 0) for r in mains.values())))
    state.atomic_json(output / 'pool_mains.json', chr._finite(result))
    for ident, record in mains.items():
        chr._emit('FRS_SUMMARY', dict(cell_id=ident, phase='mains', status=record.get('status'),
            steps=record.get('steps_done'), states=len(record.get('states', [])), error=record.get('error')))
    if any(r.get('resumable') for r in mains.values()):
        result['resume_required'] = True
        state.atomic_json(output / 'pool.json', chr._finite(result))
        chr._emit('POOL_RESUME_REQUIRED', dict(complete=False, resume_required=True,
            cells=[key for key, r in mains.items() if r.get('resumable')]))
        return _pool_summary(result)
    states = [row for rec in mains.values() for row in rec.get("states", [])]
    states.sort(key=lambda r: (r["arm"], int(r["seed"]), int(r["state_step"]), r["trigger"]))
    _write_csv(output / "states_manifest.csv",
               [{**r, "file":str(Path(r["file"]).resolve().relative_to(output.resolve()))} for r in states],
               state.MANIFEST_FIELDS)
    jobs = build_manifest(states, list(mains.values()), horizon=st.steps,
                          limit=int(cfg.get("h5_limit", 12)), job_modules=cfg.get("job_modules", ()), config=cfg,
                          dispositions=result['dispositions'], provider_errors=result['provider_errors'])
    state.atomic_json(output / 'pool.json', chr._finite(result))
    from .frs_cohorts import first_acquired
    h5_slots = [dict(seed=int(r['seed']), cell_id=r.get('cell_id'),
                     missing=r.get('status') == 'MISSING' or bool(r.get('error')))
                for r in first_acquired(list(mains.values()), int(cfg.get('h5_limit',12)),
                                        smoke=bool(cfg.get('smoke_only')))]
    manifest = dict(schema=1, protocol=cfg["protocol"], jobs=[job_record(j) for j in jobs],
                    h5_cohort=[int(r['seed']) for r in first_acquired(list(mains.values()),
                        int(cfg.get('h5_limit',12)),smoke=bool(cfg.get('smoke_only')))],
                    h5_slots=h5_slots, source_modules=cfg.get("job_modules", []), states=states, provenance=provenance,
                    route_cohorts=cfg.get('_rdo_cohorts'),trg_dispositions=cfg.get('_trg_dispositions',[]))
    from .pool_manifest import freeze
    manifest, jobs, prior, drift = freeze(output, manifest, jobs, resume=bool(cfg.get('resume')))
    result['manifest_drift'] = drift
    result['frozen_h5_slots'] = manifest.get('h5_slots')
    chr._emit("FRS_MANIFEST", dict(jobs=len(jobs), states=len(states),
                                  h5_cohort=manifest["h5_cohort"],
                                  sha256=state.file_sha256(output / "job_manifest.json")))
    phase0 = time.time()
    def job_progress(branches):
        result['jobs'] = dict(branches)
        state.atomic_json(output/'pool.json', chr._finite(result))
    branches = _global_jobs(cfg, jobs, job_progress, prior=prior) if jobs else prior
    for job in jobs:
        branches.setdefault(job.job_id, dict(job_id=job.job_id, cell_id=job.cell_id,
            seed=job.metadata.get('seed'), arm=next(c['arm'] for c in cells if
            ttt._cell_id(c, st.steps) == job.cell_id), rule=job.rule.tag,
            status='MISSING', disposition='TRUNCATED', error='worker lost or budget exhausted'))
    for pack in sorted({j.metadata.get('pack', 'TRG') for j in jobs}):
        records = [branches[j.job_id] for j in jobs if j.metadata.get('pack', 'TRG') == pack]
        chr._emit('POOL_PHASE_RATE', dict(phase=pack, worker_s=sum(r.get('job_wall_s',r.get('segment_wall_s',0)) for r in records),
                segment_s=sum(r.get('segment_wall_s',0) for r in records),
                attempted=sum(r.get('attempted', 0) for r in records), jobs=len(records)))
    complete = (not drift and all(not r.get("error") and r.get("steps_done") == st.steps or r.get("diverged")
                    for r in mains.values()) and len(branches) == len(manifest['jobs']) and
                all(r["status"] == "DONE" for r in branches.values()))
    result = dict(schema=1, complete=bool(complete), protocol=cfg["protocol"],
                  smoke_only=bool(cfg.get("smoke_only")), provenance=provenance,
                  cells=mains, jobs=branches, states=states,
                  plan_sha256=provenance["plan_sha256"], output_dir=str(output),
                  provider_errors=result['provider_errors'], dispositions=result['dispositions'],
                  manifest_drift=drift, concurrency=result['concurrency'],
                  frozen_h5_slots=manifest.get('h5_slots'))
    if cfg.get('determinism_probe'):
        from .pool_determinism import probe
        result['determinism'] = probe(cfg, dev, output/'determinism')
    result['complete'] = result['complete'] and not result['provider_errors']
    state.atomic_json(output / 'pool.json', chr._finite(result))
    if 'tttmem.exp_rdo' in cfg.get('job_modules', []):
        from .exp_rdo import reanalyze_pool
        reanalyze_pool(output, output/'analysis'/'rdo-rpe')
    if 'tttmem.trg_pool' in cfg.get('job_modules', []):
        from .trg_pool import analyze_pool
        analyze_pool(result, manifest, output/'analysis'/'trg', smoke=bool(cfg.get('smoke_only')))
    for ident, record in mains.items():
        chr._emit("FRS_SUMMARY", dict(cell_id=ident, steps=record.get("steps_done"),
                                      states=len(record.get("states", [])), diverged=record.get("diverged"),
                                      error=record.get("error")))
    print(f"FRS_DONE cells={len(mains)} states={len(states)} jobs={len(branches)} complete={result['complete']}", flush=True)
    return _pool_summary(result)


def _pool_summary(result):
    mains, branches = result['cells'], result['jobs']
    return dict(complete=result['complete'], resume_required=bool(result.get('resume_required')),
                manifest_drift=len(result.get('manifest_drift', [])),
                cells=len(mains), states=len(result['states']), jobs=len(branches),
                errored_mains=sum(bool(r.get('error')) for r in mains.values()),
                missing_jobs=sum(r['status'] == 'MISSING' for r in branches.values()),
                fail_open_cells=sum(bool(r.get('coverage_fail_open')) for r in mains.values()),
                provider_errors=len(result['provider_errors']),
                plan_sha256=result["plan_sha256"], output_dir=result['output_dir'], smoke_only=result["smoke_only"])
