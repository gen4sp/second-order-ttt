"""From-scratch instability rule (P2-3, SOQ-N3B): pure Python, no torch.

The calibration probe trips when it exceeds its **running minimum** (over earlier probes) by
more than ``max_rise`` nats at ``consecutive`` consecutive probes.  A trip at step
<= ``restart_until`` restarts the run at attempt 1 (outer peak 1e-3; attempt 1 tripping again
stops the run); a later trip labels the arm ``SO-UNSTABLE`` (SO stops) or
``FO-UNSTABLE`` (descriptive: the FO arm continues).  A non-finite probe is a G-div event,
handled by the trainer.  ``train.py`` and the dry-run script share this code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def probe_steps(start: int, horizon: int, *, every: int, first: int = 50,
                dense_every: int = 0, dense_until: int = 200) -> list[int]:
    """Probe steps in (start, horizon]: ``first``, then every ``dense_every`` up to
    ``dense_until`` (0 = no dense phase), then every ``every``.  The registered P2-3 cadence is
    first=50, dense_every=10, dense_until=200, every=50; the SOQ-N3B default is every 50."""
    if every <= 0:
        return []
    steps = set()
    if dense_every > 0:
        steps.update(range(first, dense_until + 1, dense_every))
        nxt = dense_until + every
    else:
        nxt = first
    steps.update(range(nxt, horizon + 1, every))
    return sorted(s for s in steps if start < s <= horizon)


@dataclass
class RunningMinGuard:
    """Running-minimum rule; ``history`` round-trips through the resume state."""

    max_rise: float = 0.10
    consecutive: int = 2
    history: list = field(default_factory=list)

    def update(self, step: int, value: float) -> dict:
        prior = [h["value"] for h in self.history if math.isfinite(h["value"])]
        run_min = min(prior) if prior else None
        finite = math.isfinite(value)
        rise = (value - run_min) if (finite and run_min is not None) else None
        exceed = bool(rise is not None and rise > self.max_rise)
        streak = (self.history[-1]["streak"] + 1 if exceed and self.history else int(exceed))
        rec = {"step": int(step), "value": float(value), "running_min": run_min, "rise": rise,
               "exceed": exceed, "streak": streak, "finite": finite,
               "trip": bool(exceed and streak >= self.consecutive)}
        self.history.append(rec)
        return rec


def trip_action(step: int, kind: str, attempt: int, restart_until: int = 200) -> dict:
    """What a trip at ``step`` means for an arm of ``kind`` ("so"/"fo") at ``attempt``."""
    if step <= restart_until:
        if attempt == 0:
            return {"label": "RESTART", "stop": True,
                    "next": "restart_from_scratch_attempt_1_lr_peak_1e-3"}
        return {"label": "UNSTABLE_AT_ATTEMPT_1", "stop": True, "next": "stop_replan", "final": True}
    if kind == "so":
        return {"label": "SO-UNSTABLE", "stop": True, "next": "stop_user_decides"}
    return {"label": "FO-UNSTABLE", "stop": False, "next": "continue_descriptive"}


def running_min_rise(prior: list[float], loss: float) -> float:
    """From-scratch train-loss guard: rise of a logged loss over the running minimum of the
    logged losses so far, this one included (the first log has rise 0, as the start-relative
    rule; a negative threshold therefore still trips at once, the test hook)."""
    if not math.isfinite(loss):
        return math.nan
    base = min([h for h in prior if math.isfinite(h)] + [loss])
    return loss - base
