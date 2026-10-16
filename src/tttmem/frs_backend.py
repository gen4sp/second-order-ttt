"""Rule dispatch boundary: route modules may supply another backend."""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol, Any
import time
from statistics import median

from . import exp_chr as chr
from . import exp_ttt as ttt


class RuleBackend(Protocol):
    def loss(self, model, idx, y, meta, ctx, rule, stats, diag, *, observer=None): ...
    def quick(self, model, ctx, rule, *, geometry_on: bool = False): ...
    def write(self, model, ctx, rule): ...
    def final(self, model, ctx, rule, rules): ...


class ChrBackend:
    """Unmodified CHR arithmetic; unsupported rules fail explicitly."""
    def loss(self, model, idx, y, meta, ctx, rule, stats, diag, *, observer=None):
        if not isinstance(rule, chr.Rule):
            raise TypeError(f"rule {rule.tag} needs its route RuleBackend")
        args=(model, ctx.arm, idx, y, ctx.eta, ctx.st.g_clip,ctx.task.chunk, stats, meta, rule, diag)
        if observer is not None and rule.name == 'ce':
            # Keep the calibrated CHR source byte-locked; mirror its CE branch here.
            keep = ttt.outer_keep(meta, tuple(y.shape), rule.outer, None, idx.device)
            return ttt.arm_loss(model, ctx.arm, idx, y, ctx.eta, ctx.st.g_clip,
                                ctx.task.chunk, stats, keep=keep, meta=meta, observer=observer)
        return chr.chr_arm_loss(*args)

    def quick(self, model, ctx, rule, *, geometry_on=False):
        return chr.quick_log(model, ctx.quick, ctx.eta, ctx.st.g_clip, ctx.task.chunk,
                             rule, (ctx.cs.chunk_early, ctx.cs.chunk_late), geometry_on=geometry_on)

    def write(self, model, ctx, rule):
        return chr.write_log(model, ctx.probe, ctx.eta, ctx.st.g_clip, ctx.task.chunk, rule)

    def final(self, model, ctx, rule, rules):
        cs = replace(ctx.cs, final_rules={ctx.arm: list(rules)})
        return chr.final_eval(model, replace(ctx, cs=cs), rule, "job")


class TimedBackend:
    """Read-cost accounting, with the wrapped arithmetic and return values intact."""
    def __init__(self, backend):
        self.backend, self.costs = backend, {}

    def __getattr__(self, name):
        method=getattr(self.backend,name)
        if name not in ('quick','write','final'):
            return method
        def measured(*args, **kwargs):
            from . import exp_ttt as ttt
            ctx=args[1]
            ttt._sync(ctx.dev)
            before=time.perf_counter()
            result=method(*args,**kwargs)
            ttt._sync(ctx.dev)
            entry=self.costs.setdefault(name,dict(calls=0,wall_s=0.,samples_s=[]))
            elapsed=time.perf_counter()-before
            entry['calls']+=1;entry['wall_s']+=elapsed
            entry['samples_s'].append(elapsed);entry['median_s']=median(entry['samples_s'])
            return result
        return measured
