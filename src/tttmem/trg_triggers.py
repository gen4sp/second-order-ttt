"""Frozen TRG triggers, coverage, calibration and yoked schedules."""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np

KAPPAS = (3, 5, 10)
ACTIVE_STEP_DOMAIN = "finite post-fork observations after ten finite warm-up observations, step <= 597"


class SpikeController:
    def __init__(self, fork_step, kappa=5):
        self.fork_step, self.kappa = int(fork_step), int(kappa)
        self.history = deque(maxlen=50)
        self.finite_count = 0

    def observe(self, step, norm):
        """Consume the current pre-clip norm after backward, including dropped steps."""
        if step <= self.fork_step:
            return dict(flags={str(k): False for k in KAPPAS}, flag=False, active=False, baseline=None)
        finite = bool(np.isfinite(norm))
        active = finite and self.finite_count >= 10 and self.fork_step < step <= 597
        base = float(np.median(self.history)) if self.history else None
        flags = {str(k): bool(active and norm > k * base) for k in KAPPAS}
        if finite:
            self.history.append(float(norm))
            self.finite_count += 1
        return dict(flags=flags, flag=flags[str(self.kappa)], active=active, baseline=base)


def drop_list_bytes(drops):
    """FRS-A write_drop_list bytes, including its spaces and terminal newline."""
    return (json.dumps(dict(schema=1, drops=list(map(int, drops))), sort_keys=True)+"\n").encode()


def realised_drop_hash(drops):
    return hashlib.sha256(drop_list_bytes(drops)).hexdigest()


def yoked_sham(drops, sha256):
    drops = list(map(int, drops))
    if realised_drop_hash(drops) != sha256:
        raise ValueError("realised drop-list sha256 mismatch")
    if drops != sorted(set(drops)) or any(d < 0 or d > 597 for d in drops):
        raise ValueError("invalid realised drop list")
    out = []
    for d in drops:
        s = max(d + 3, out[-1] + 1 if out else 0)
        if s > 600:
            raise ValueError("yoked schedule exceeds 600")
        out.append(s)
    assert len(set(out)) == len(drops)
    return out


def dense_onset(steps, a, t_acq):
    steps, a = np.asarray(steps, int), np.asarray(a, float)
    if len(a) < 10:
        return None
    win = np.lib.stride_tricks.sliding_window_view(a, 10)
    grids = np.lib.stride_tricks.sliding_window_view(steps, 10)
    valid = (a[:-9] < .90) & (win.mean(-1) < .90) & (steps[:-9] > t_acq)
    valid &= np.all(np.diff(grids, axis=1) == 1, axis=1) & np.isfinite(win).all(1)
    hit = np.flatnonzero(valid)
    return int(steps[hit[0]]) if len(hit) else None


class WindowController:
    """Inclusive main ladder; unknown onset fails open to the registered cap."""
    def __init__(self, t_acq, family, *, cap=300, so_end=660, so_every=5):
        self.t_acq, self.family, self.cap = t_acq, family, cap
        self.so_end, self.so_every = so_end, so_every
        self.steps, self.accuracy, self.onset = [], [], None

    def ladder_due(self, step):
        if self.family == "SO":
            return self.t_acq <= step <= self.so_end and step % self.so_every == 0
        end = min(self.t_acq+self.cap, self.onset+60) if self.onset is not None else self.t_acq+self.cap
        return self.t_acq <= step <= end

    def observe(self, step, accuracy):
        self.steps.append(step)
        self.accuracy.append(accuracy)
        if self.onset is None:
            # Only the new ten-row window can first confirm an onset.
            self.onset = dense_onset(self.steps[-10:], self.accuracy[-10:], self.t_acq)


def dip_onsets(steps, accuracy, t_acq, *, last=600):
    s, a = np.asarray(steps, int), np.asarray(accuracy, float)
    if len(s) < 4:
        return np.array([], int)
    windows = np.lib.stride_tricks.sliding_window_view(a, 4)
    grids = np.lib.stride_tricks.sliding_window_view(s, 4)
    valid = (windows[:, 0] >= .90) & (windows[:, 1] < .90) & (windows[:, 1:].mean(1) < .90)
    valid &= np.all(np.diff(grids, axis=1) == 1, axis=1)
    valid &= (s[1:-2] >= t_acq+11) & (s[1:-2] <= last)
    return s[1:-2][valid]


def coincidence(steps, flags, onsets):
    s, f, on = np.asarray(steps, int), np.asarray(flags, bool), np.asarray(onsets, int)
    if not len(on):
        return dict(hit=None, false_alarm=None, ratio=None, associated=False)
    lag = s[:, None] - on[None, :]
    near = (lag >= -3) & (lag <= 0)
    far = (np.abs(lag) > 10).all(1)
    hit = float((near & f[:, None]).any(0).mean())
    fa = float(f[far].mean()) if far.any() else None
    rate = float(f[near.any(1)].mean()) if near.any() else 0.
    ratio = None if fa is None else (rate/fa if fa > 0 else (float("inf") if rate else 0.))
    return dict(hit=hit, false_alarm=fa, ratio=ratio,
                associated=hit >= 2/3 and ratio is not None and ratio >= 3)


def active_coincidence(telemetry, onsets, kappa):
    """H-T2 and calibration share exactly the controller-active near/far domain."""
    rows = [r for r in telemetry if r["active"]]
    return coincidence([r["step"] for r in rows],
                       [r["flags"][str(kappa)] for r in rows], onsets)


def select_kappa(units):
    """units: {steps, accuracy, t_acq, telemetry_steps, norms}; no-data default 5."""
    scores = {k: [] for k in KAPPAS}
    for unit in units:
        on = dip_onsets(unit["steps"], unit["accuracy"], unit["t_acq"])
        if len(on) < 2:
            continue
        controller = SpikeController(unit["t_acq"])
        telemetry = []
        for s, g in zip(unit["telemetry_steps"], unit["norms"]):
            if s <= unit["t_acq"]:
                continue
            row = controller.observe(int(s), float(g) if g is not None else float("nan"))
            telemetry.append(dict(step=int(s), **row))
        for k in KAPPAS:
            row = active_coincidence(telemetry, on, k)
            if row["false_alarm"] is not None:
                scores[k].append(row["hit"]-row["false_alarm"])
    n = min(map(len, scores.values()))
    med = {str(k): float(np.median(v)) if v else None for k, v in scores.items()}
    choice = 5 if n < 3 else max(KAPPAS, key=lambda k: (med[str(k)], k))
    return dict(kappa=choice, qualifying_units=n, median_j=med, no_data_default=n < 3,
                active_step_domain=ACTIVE_STEP_DOMAIN)
