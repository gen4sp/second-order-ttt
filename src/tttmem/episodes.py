"""Shared RDO/RPE disposition and episode endpoints."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import numpy as np


class Disposition(StrEnum):
    COMPLETE = "COMPLETE"
    DIVERGED = "DIVERGED"
    TRUNCATED = "TRUNCATED"


@dataclass(frozen=True)
class Divergence:
    t_div: int
    attempted: int
    nonfinite: int
    certified: bool = True

    def valid(self, start: int) -> bool:
        return (self.certified and self.attempted >= 20 and
                self.t_div == start + self.attempted and
                0 <= self.nonfinite <= self.attempted and
                self.nonfinite / self.attempted > .05)


def _series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    step = np.asarray([r["step"] for r in rows], dtype=np.int64)
    acc = np.asarray([r.get("quick", r).get("acc", np.nan) for r in rows], dtype=float)
    return step, acc


def _grid(rows, expected, t_div=None) -> bool:
    steps, acc = _series(rows)
    if len(np.unique(steps)) != len(steps) or (len(steps) > 1 and (np.diff(steps) <= 0).any()):
        return False
    expected = np.asarray(expected, dtype=np.int64)
    if t_div is not None:
        expected = expected[expected < t_div]
        keep = steps < t_div
        steps, acc = steps[keep], acc[keep]
    return np.array_equal(steps, expected) and bool(np.isfinite(acc).all())


def disposition(rows: list[dict], start: int, horizon: int, *, every: int = 5,
                certificate: Divergence | None = None, needed: tuple = (),
                finals: dict | None = None, require_final: bool = True) -> Disposition:
    """`needed`: (rows, exact expected steps) pairs; missing common reads are missing."""
    if horizon <= 0 or horizon % every or start % every:
        raise ValueError("episode horizon/start must align with the registered grid")
    expected = np.arange(start + every, start + horizon + 1, every)
    div = certificate is not None and certificate.valid(start)
    t_div = certificate.t_div if div else None
    if certificate is not None and not div:
        return Disposition.TRUNCATED
    if not _grid(rows, expected, t_div):
        return Disposition.TRUNCATED
    if any(not _grid(series, grid, t_div) for series, grid in needed):
        return Disposition.TRUNCATED
    if require_final and not div:
        if not finals or any(not np.isfinite(v.get("acc", np.nan)) for v in finals.values()):
            return Disposition.TRUNCATED
    return Disposition.DIVERGED if div else Disposition.COMPLETE


def coded_series(rows: list[dict], start: int, horizon: int, *, every: int = 5,
                 certificate: Divergence | None = None) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(start + every, start + horizon + 1, every, dtype=np.int64)
    have_steps, have_acc = _series(rows)
    values = dict(zip(have_steps.tolist(), have_acc.tolist()))
    acc = np.asarray([values.get(int(t), np.nan) for t in steps], dtype=float)
    if certificate is not None and certificate.valid(start):
        acc[steps >= certificate.t_div] = 0.0
    return steps, acc


def _first_pair(steps, acc, bound: bool, *, after: int, last: int, bind: float, every: int):
    mask = acc >= bind if bound else acc < bind
    candidates = np.flatnonzero((steps[:-1] > after) & (steps[:-1] <= last)
                                & mask[:-1] & mask[1:] & (np.diff(steps) == every))
    return int(steps[candidates[0]]) if len(candidates) else None


def nl_fine(rows: list[dict], start: int, *, window: int = 300, bind: float = .90,
            certificate: Divergence | None = None) -> bool | None:
    if certificate is not None and certificate.valid(start) and certificate.t_div <= start + window:
        return False
    if not _grid(rows, np.arange(start + 1, start + window + 10),
                 certificate.t_div if certificate is not None and certificate.valid(start) else None):
        return None
    steps, acc = coded_series(rows, start, window + 9, every=1, certificate=certificate)
    failed = np.lib.stride_tricks.sliding_window_view(acc < bind, 10).all(axis=1)
    return not bool(failed.any())


def endpoints(rows: list[dict], start: int, horizon: int = 400, *, window: int = 300,
              bind: float = .90, every: int = 5, certificate: Divergence | None = None,
              full_acc: float | None = None, dense: list[dict] | None = None,
              needed: tuple = (), require_final: bool = True) -> dict:
    final = None if full_acc is None else {"own": {"acc": full_acc}}
    state = disposition(rows, start, horizon, every=every, certificate=certificate,
                        needed=needed, finals=final, require_final=require_final)
    if state == Disposition.TRUNCATED:
        return dict(disposition=state.value, nl2=None, nl_fine=None, lost_time=None,
                    t_loss=None, t_re=None, tau=None, censored=None, retained=None,
                    episodes=[], t_div=None, first_recovery=None, first_outage=None)
    steps, acc = coded_series(rows, start, horizon, every=every, certificate=certificate)
    first_loss = _first_pair(steps, acc, False, after=start, last=start + window,
                            bind=bind, every=every)
    t_re = _first_pair(steps, acc, True, after=start, last=start + horizon - every,
                       bind=bind, every=every)
    lost = np.zeros(len(steps), dtype=bool)
    intervals = []
    cursor = start
    while True:
        onset = _first_pair(steps, acc, False, after=cursor, last=start + horizon - every,
                           bind=bind, every=every)
        if onset is None:
            break
        recovery = _first_pair(steps, acc, True, after=onset, last=start + horizon - every,
                               bind=bind, every=every)
        lost |= (steps >= onset) & (steps < (recovery if recovery is not None else start + horizon + every))
        intervals.append(dict(onset=onset, recovery=recovery,
                              outage=(recovery - onset) if recovery is not None else start + horizon - onset,
                              censored=recovery is None and state != Disposition.DIVERGED))
        if recovery is None:
            break
        cursor = recovery
    if state == Disposition.DIVERGED:
        lost[steps == certificate.t_div] = False  # registered tail is (t_div, start+H]
        lost[steps > certificate.t_div] = True
    retained = bool(state != Disposition.DIVERGED and len(acc) >= 2 and
                    (acc[-2:] >= bind).all() and full_acc is not None and full_acc >= bind)
    first_outage = next((e for e in intervals if e["onset"] == first_loss), None)
    return dict(disposition=state.value, nl2=first_loss is None,
                nl_fine=None if dense is None else nl_fine(dense, start, window=window,
                                                          bind=bind, certificate=certificate),
                lost_time=int(lost.sum() * every), t_loss=first_loss, t_re=t_re,
                tau=(t_re - start) if t_re is not None else horizon,
                censored=t_re is None and state != Disposition.DIVERGED,
                diverged=state == Disposition.DIVERGED, retained=retained, episodes=intervals,
                first_recovery=None if first_outage is None else first_outage["recovery"],
                first_outage=None if first_outage is None else first_outage["outage"],
                t_div=certificate.t_div if state == Disposition.DIVERGED else None,
                min_acc=float(np.nanmin(acc[(steps > start) & (steps <= start + window)])))


def adverse_completion(value: dict | None, *, treatment: bool, horizon: int = 400,
                       reverse: bool = False) -> dict:
    missing = value is None or value.get("disposition") == Disposition.TRUNCATED
    control_diverged = not treatment and value and value.get("disposition") == Disposition.DIVERGED
    if not missing and not control_diverged:
        return dict(value)
    success = bool(treatment) if reverse else not treatment
    return dict(disposition="IMPUTED", nl2=success, nl_fine=success,
                lost_time=0 if success else horizon, retained=success,
                t_re=5 if success else None, tau=5 if success else horizon,
                censored=False, adverse=True)


def common_label(completions: tuple, label: Callable) -> str:
    first, second = (label(c) for c in completions)
    return first if first == second else "unknown"


def kaplan_meier(times: list[int], observed: list[bool]) -> list[dict]:
    """Administrative right censoring, with events processed before tied censors."""
    t, event = np.asarray(times), np.asarray(observed, dtype=bool)
    survival = 1.0
    out = []
    for point in np.unique(t):
        risk = int((t >= point).sum())
        deaths = int(((t == point) & event).sum())
        survival *= 1 - deaths / risk
        out.append(dict(time=int(point), at_risk=risk, events=deaths,
                        censored=int(((t == point) & ~event).sum()), survival=survival))
    return out


# TRG array adapters. Core disposition/certificate semantics match the shared
# row API above.
COMPLETE, DIVERGED, TRUNCATED = (d.value for d in Disposition)


def certified_divergence(nonfinite, continuation_steps):
    return continuation_steps >= 20 and nonfinite / continuation_steps > .05


def classify_branch(steps, accuracy, *, start, end, grid=5, t_div=None,
                    certified=False, state_valid=True, code_fault=False,
                    continuation_steps=None, nonfinite=None, certificate_start=None, **_):
    if not state_valid or code_fault:
        return TRUNCATED
    s, acc = np.asarray(steps, int), np.asarray(accuracy, float)
    if len(s) != len(acc) or (len(s) > 1 and (np.diff(s) <= 0).any()):
        return TRUNCATED
    cert = None
    if t_div is not None:
        if continuation_steps is None or nonfinite is None:
            return TRUNCATED
        cert = Divergence(int(t_div), int(continuation_steps), int(nonfinite), bool(certified))
    # Dense acquisition read is inclusive; the episode grid itself is exclusive.
    rows = [dict(step=int(t), acc=float(v)) for t, v in zip(s, acc) if start < t <= end]
    if cert is not None and certificate_start is not None:
        # SO main trains from zero, while its dense reads start at acquisition.
        # Certify on the training segment, check coverage on the dense segment.
        valid = cert.valid(certificate_start) and _grid(rows, np.arange(start+grid, end+1, grid), t_div)
        return DIVERGED if valid else TRUNCATED
    return disposition(rows, start, end-start, every=grid, certificate=cert,
                       require_final=False).value if end > start else COMPLETE


def coded_dense_series(steps, accuracy, *, start, end, grid=5, disposition=COMPLETE, t_div=None):
    if disposition == TRUNCATED:
        return None, None
    s, a = np.asarray(steps, int), np.asarray(accuracy, float)
    req = np.arange(start+grid, end+1, grid)
    vals = np.full(req.shape, np.nan)
    loc = np.searchsorted(s, req)
    valid = loc < len(s)
    match = np.zeros(len(req), bool)
    match[valid] = s[loc[valid]] == req[valid]
    vals[match] = a[loc[match]]
    if disposition == DIVERGED:
        if t_div is None:
            raise ValueError("DIVERGED requires certified t_div")
        vals[req >= t_div] = 0.
    return (req, vals) if np.isfinite(vals).all() else (None, None)


def nl2_window(steps, accuracy, t_acq, *, window=300, grid=5,
               disposition=COMPLETE, t_div=None):
    s, a = coded_dense_series(steps, accuracy, start=t_acq, end=t_acq+window+grid,
                             grid=grid, disposition=disposition, t_div=t_div)
    if s is None:
        return dict(nl2=None, onset=None, missing=True)
    hit = np.flatnonzero((a[:-1] < .90) & (a[1:] < .90) & (s[:-1] <= t_acq+window))
    return dict(nl2=not bool(len(hit)), onset=int(s[hit[0]]) if len(hit) else None, missing=False)
