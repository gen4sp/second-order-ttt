"""Pure CHR-grid detectors, shared by online pool hooks and offline analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Trigger:
    name: str
    event_step: int
    detected_step: int
    state_step: int | None


def _columns(rows: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray([r["step"] for r in rows], dtype=np.int64)
    acc = np.asarray([r.get("quick", r).get("acc", np.nan) for r in rows], dtype=float)
    resid = np.asarray([r.get("write", r).get("r_kv_med", np.nan) for r in rows], dtype=float)
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("detector rows must have unique increasing steps")
    return steps, acc, resid


def detect(rows: Sequence[dict], *, bind: float = .90, residual: float = .01,
           grid: int = 5) -> dict[str, Trigger]:
    """The exact prefix detector; a gap breaks every consecutive-log criterion."""
    steps, acc, resid = _columns(rows)
    out: dict[str, Trigger] = {}
    acq = np.flatnonzero(acc >= bind)
    if not len(acq):
        return out
    t_acq = int(steps[acq[0]])
    out["t_acq"] = Trigger("t_acq", t_acq, t_acq, t_acq)
    if len(steps) < 2:
        return out
    low = (acc < bind) & np.isfinite(acc)
    loss = np.flatnonzero((steps[:-1] > t_acq) & low[:-1] & low[1:]
                          & (np.diff(steps) == grid))
    if not len(loss):
        return out
    i = int(loss[0])
    t_loss = int(steps[i])
    out["t_loss"] = Trigger("t_loss", t_loss, int(steps[i + 1]), None)
    if len(steps) >= 3:
        below = (resid < residual) & np.isfinite(resid) & (steps > t_loss)
        s1 = np.flatnonzero(below[:-2] & below[1:-1] & below[2:]
                            & (np.diff(steps)[:-1] == grid) & (np.diff(steps)[1:] == grid))
        if len(s1):
            t = int(steps[int(s1[0]) + 2])
            out["S1"] = Trigger("S1", t, t, t)
    return out


def detect_l1(rows: Sequence[dict], t_acq: int, *, bind: float = .90) -> Trigger | None:
    steps, acc, _ = _columns(rows)
    if len(steps) < 10:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(acc, 10)
    sw = np.lib.stride_tricks.sliding_window_view(steps, 10)
    ok = ((steps[:-9] > t_acq) & (acc[:-9] < bind) & np.isfinite(windows).all(1)
          & (windows.mean(1) < bind) & (np.diff(sw, axis=1) == 1).all(1))
    ix = np.flatnonzero(ok)
    if not len(ix):
        return None
    t = int(steps[ix[0]])
    return Trigger("t_L1", t, t + 9, None)


def insurance_steps(t_loss: int, *, last: int = 1000) -> list[int]:
    return list(range(((t_loss + 49) // 50) * 50, last + 1, 50))


class CoverageWindow:
    """Fail-open coverage controller; never reconstructs a pre-acquisition read."""
    def __init__(self, arm: str, t_acq: int, *, bind: float = .90,
                 span: int = 300, tail: int = 60, so_stop: int = 660):
        self.arm, self.t_acq, self.bind = arm, t_acq, bind
        self.tail = tail
        self.cap = so_stop if arm == "e2e2emb" else t_acq + span
        self.stop = self.cap
        self.rows: list[dict] = []
        self.t_l1: Trigger | None = None
        self.fail_open = False

    def active(self, step: int) -> bool:
        return self.t_acq <= step <= self.stop

    def observe(self, row: dict) -> None:
        if 'acc' not in row:
            raise ValueError('coverage adapter must return flat acc')
        if self.fail_open or self.t_l1 is not None:
            return
        try:
            if self.rows and row["step"] != self.rows[-1]["step"] + 1:
                raise ValueError("coverage gap")
            if not np.isfinite(row["acc"]):
                raise ValueError("non-finite coverage read")
            self.rows.append(row)
            self.t_l1 = detect_l1(self.rows, self.t_acq, bind=self.bind)
            if self.t_l1 and self.arm != "e2e2emb":
                self.stop = min(self.cap, self.t_l1.event_step + self.tail)
        except Exception:
            self.fail_open = True
            self.stop = self.cap
