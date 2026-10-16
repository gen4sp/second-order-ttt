"""P2-1 continuation loop: fresh AdamW from a checkpoint, re-warmed outer LR,
gradient accumulation, inline barrier evaluation, stdout evidence.

Stdout lines: ``P2_CONFIG`` (echo), ``P2_ASSAY`` (step-0 gate), ``P2_STEP``, ``P2_EVAL``
(per evaluated checkpoint), ``P2_ABORT`` (pilot rule), ``P2_STATE`` / ``P2_RESUME``,
``P2_PERF`` (timing separated from repeatable metrics), and the final summary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import queue
import random
import shutil
from pathlib import Path
import threading
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import torch

from .meta import InnerCfg, fo_backward, so_backward
from .model import PRESETS, build_model, load_state, param_count

ARMS = {  # arm -> (outer engine, inner rule)
    "e2e2": ("so", "ce"),
    "e2e12-p125": ("somix", "ce"),
    "wsd": ("fo", "ce"),
    "e2e1-ce": ("fo", "ce"),
    "e2e1-lin": ("fo", "lin"),
    "e2e1-aice": ("fo", "aice"),
    "e2e1-rp": ("fo", "rp"),
    "e2e1-nc": ("fo", "nc"),
}


@dataclass
class TrainCfg:
    arm: str = "e2e2"
    preset: str = "125m"
    init: str = ""                    # checkpoint path ("" = random init; tests only)
    init_label: str = ""              # provenance label of ``init``, fixed before launch
    init_seed: int = 0
    dtype: str = "bf16"
    device: str = "cuda"
    seq_len: int = 8192
    global_batch: int = 64
    micro_batch: int = 2
    so_micro_batch: int = 1
    so_fraction: float = 0.125
    steps: int = 400
    lr_peak: float = 3e-3
    attempt: int = 0                  # one fallback, at peak 1e-3
    lr_floor: float = 1e-5
    warmup: int = 30
    lr_schedule: str = "cosine"
    wsd_decay: int = 240
    final_model_only: bool = False
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    inner_lr: float = 1.0
    inner_clip: float = 1.0
    r_ref: float = 0.03
    barriers: list = field(default_factory=lambda: [2, 4])
    segment_len: int = 2048
    # P2-1/A6: per-rule inner-lr multiplier, applied wherever that rule runs (training and every
    # evaluation incl. the crossover). CE stays at 1; values come from mode="calibrate".
    rule_lr_scale: dict = field(default_factory=dict)
    trunc_kv: bool = False
    fused_attn: bool = False          # MINI/P2-3 requires math attention for FO exactness
    so_fused_forward: bool = False    # SO forward sweep backend (default: math, = reverse)
    prefix_checkpoint: bool = False
    data_dir: str = ""
    code_data_dir: str = ""
    train_split: str = "A"
    mix: bool = False
    code_train_split: str = "code_train"
    code_eval_split: str = "code_eval"
    code_eval_steps: list = field(default_factory=lambda: [300, 600, 1200])
    data_seed: int = 0
    data_start_step: int = 0          # position in the original data_seed stream
    eval_split: str = "eval"
    ho_split: str = "ho"
    ho_docs: int = 512
    ho_trajectory_docs: int = 128
    # Evaluation is deliberately configuration-driven.  The defaults below
    # serve P2-1/P2-3; MINI production configs override them.
    eval_at_init: bool = True
    init_eval_docs: int = 0
    init_ho_docs: int = 0
    init_eval_modes: list = field(default_factory=lambda: ["adapt", "noadapt", "fP_focal", "fP_nonfocal"])
    init_crossover_rules: list = field(default_factory=lambda: ["ce", "lin", "aice"])
    full_extra_modes: list = field(default_factory=lambda: ["fP_focal_refclip"])
    sizing_modes: list = field(default_factory=lambda: ["fP_focal", "fP_nonfocal"])
    code_eval_docs: int = 1024
    calib_split: str = "calib"
    n_eval: int = 3072                # eligible pool; coordinator freezes endpoint N
    assay_gate: bool = True
    eval_steps: list = field(default_factory=lambda: [0, 30, 60, 100, 200, 400])
    full_eval_steps: list = field(default_factory=lambda: [0, 400])
    sizing_steps: list = field(default_factory=lambda: [60])
    crossover_rules: list = field(default_factory=lambda: ["ce", "lin", "aice"])
    trajectory_docs: int = 256
    eval_batch: int = 8
    eval_modes: list = field(default_factory=lambda: ["adapt", "noadapt", "frozenP",
                                                      "fP_focal", "fP_nonfocal"])
    trajectory_modes: list = field(default_factory=lambda: ["adapt", "noadapt", "frozenP"])
    min_probes: int = 5
    pilot_steps: list = field(default_factory=lambda: [10, 20, 30, 40, 50, 60])
    pilot_docs: int = 64
    pilot_max_rise: float = 0.10
    # Divergence guard for every logged step (0 = off; negative trips at once, for tests): abort when the logged train loss is
    # non-finite or exceeds the run's first logged train loss by more than this.
    train_loss_max_rise: float = 0.0
    inner_warmup: int = 120
    log_every: int = 10
    out_dir: str = "p2_out"
    save_checkpoints: bool = True
    resume_every: int = 25
    resume_dir: str = ""
    measure_timing: bool = False
    resume: bool = False
    stop_after: int | None = None     # graceful interruption; never changes LR horizon
    process_name: str = "TTM-min-05"
    max_hours: float = 0.0            # wall-clock cap: save state and stop (0 = none)
    estimate_wall_s: float = 0.0      # informational (launcher metadata), never read by the loop
    # Continuation: keep the endpoint state; a longer horizon extends it by continuation only.
    keep_final_state: bool = False    # keep the final resume_stepNNNN.pt (+ final_state.json)
    continue_from: str = ""           # kept final state of the parent run ("" = no continuation)
    continue_weights_from: str = ""   # deviation: model-only parent, fresh Adam
    continue_parent_run_id: str = ""
    continue_lost_state_sha256: str = ""
    continue_steps: int = 0           # new total horizon, e.g. 2400 (``steps`` = parent horizon)
    continue_lr_peak: float = 0.0     # re-warm peak; 0 = parent's lr_peak; fallback 1e-3
    # Splits appended to the stream after the full ``train_split`` permutation (rows beyond
    # len(train_split) for horizons the train split cannot cover, e.g. 2400 x 16 > 25600).
    extra_train_splits: list = field(default_factory=list)
    # SOQ P1 echo injection (generator ``echo_text.make_injector``); 0 = plain DCLM.
    echo_f: float = 0.0
    echo_version: int = 2
    echo_p: float = 0.5
    echo_lookup_share: float = 0.25
    echo_slots: int = 8
    echo_seed: int = 0
    echo_manifest_sha256: str = ""    # registered injector manifest sha256 ("" = not checked)
    # Echo eval read at every eval step (0 docs = off): TEST templates in clean host rows.
    echo_eval_docs: int = 0
    echo_eval_split: str = "ho"
    echo_eval_offset: int = 256       # ho[:256] is the production held-out LM read
    echo_eval_p: float = 0.5          # the test is balanced over both strata
    echo_eval_steps: list = field(default_factory=list)  # extra echo-only reads (e.g. loaded start)
    echo_diagnostics: bool = False
    echo_train_eval_split: str = "eval"
    echo_train_eval_offset: int = 2048
    echo_train_eval_seed: int = 1000003  # different pairings, same train pools/templates
    # Parity check (V0): compare doc_adapt/doc_noadapt on the first ``parity_docs`` documents
    # with a reference eval npz (fresh run: the step-0 eval; continuation: a re-read at start).
    parity_ref: str = ""
    parity_docs: int = 64
    parity_tol: float = 0.01
    # Post-hoc SOQ-P1 v3 keys are accepted by the shared config parser.  The
    # training loop never reads them; exp_p2 dispatches post mode before train().
    post_checkpoints: list = field(default_factory=list)
    post_out_dir: str = ""
    # SOQ-N3B from-scratch guard (P2-3, ``guard.py``): the adapted calibration loss
    # (own rule, training layout, all loss-valid targets, ``pilot_docs`` calib documents) trips
    # at ``guard_consecutive`` consecutive probes above its running minimum + ``guard_max_rise``.
    # 0 = off.  Registered P2-3 cadence: first 50, dense_every 10 to 200, then every 50.
    guard_every: int = 0
    guard_first: int = 50
    guard_dense_every: int = 0
    guard_dense_until: int = 200
    guard_max_rise: float = 0.10
    guard_consecutive: int = 2
    guard_restart_until: int = 200    # trip at <= this step: attempt 1; later: SO-/FO-UNSTABLE
    guard_fo_stop: bool = False       # SOQ-N3B: any FO trip stops the run (no descriptive FO-UNSTABLE)
    # Train-loss guard baseline: "auto" = first logged loss for continuations (P1),
    # running minimum from scratch; "start" / "running_min" force one.
    train_loss_guard: str = "auto"
    # G-div: a non-finite step loss or grad norm reloads the last resume state once (a step-start
    # state is saved first); a second one stops the run as DIVERGED.
    gdiv: bool = False
    save_model_steps: list = field(default_factory=list)  # extra model_stepNNNN.pt (eval steps)
    # P2_TIME evidence: per-step training seconds and the run's cumulative total (resumed runs
    # carry it in the state), checked against ``train_seconds_target`` (equal GPU-seconds).
    log_train_seconds: bool = False
    train_seconds_target: float = 0.0
    # Resume states also saved at these steps and kept as resume_keepNNNN.pt (never rotated or
    # cleaned up; copy one back to resume_stepNNNN.pt to resume there): SOQ-N3B keeps 3200,
    # the last step before the stream enters split C.
    keep_resume_steps: list = field(default_factory=list)
    # Before step 1: every extra train split must share no document (sha1, or the first 256
    # tokens) with the eval / ho / calib splits; any hit stops the run (SOQ-N3B, split C).
    contamination_check: bool = False
    mps_memory_fraction: float = 0.20

    @classmethod
    def from_dict(cls, d: dict) -> "TrainCfg":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names - {"mode", "process_name"}
        if unknown:
            raise ValueError(f"unknown p2 train keys: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in d.items() if k in names})
        cfg.validate_layout()
        return cfg

    def validate_layout(self) -> None:
        if self.arm == "e2e12-p125" and (not 0 <= self.so_fraction <= 1 or
                min(self.global_batch, self.micro_batch, self.so_micro_batch) < 1):
            raise ValueError("partial SO requires so_fraction in [0, 1] and positive batch sizes")
        if self.lr_schedule not in ("cosine", "wsd"):
            raise ValueError("lr_schedule must be cosine or wsd")
        if self.arm == "wsd" and self.lr_schedule != "wsd":
            raise ValueError("wsd arm requires lr_schedule=wsd")
        if self.lr_schedule == "wsd" and (self.warmup < 1 or self.wsd_decay < 1 or
                self.horizon() - self.start() < self.warmup + self.wsd_decay):
            raise ValueError("WSD requires positive W/D with W+D <= horizon")
        if self.lr_schedule == "wsd" and (not all(math.isfinite(v) for v in (
                self.lr_floor, self.peak(), self.inner_lr)) or
                not 0 <= self.lr_floor < min(self.peak(), self.inner_lr)):
            raise ValueError("WSD requires finite peak/inner LR above a nonnegative lr_floor")
        if self.echo_version not in (2, 3):
            raise ValueError("echo_version must be 2 or 3")
        if self.echo_version == 3 and self.echo_slots not in (12, 16):
            raise ValueError("echo_slots for v3 must be 12 or 16")
        if self.preset in {"mini-s", "mini-m", "micro16k", "125m"} and self.preset in PRESETS and len(self.barriers) >= 2:
            gaps = np.diff(sorted(self.barriers))
            if np.any(gaps <= 0) or self.segment_len != int(gaps[0]) * PRESETS[self.preset].chunk \
                    or not np.all(gaps == gaps[0]):
                raise ValueError("segment_len must equal uniform barrier spacing times model chunk")
        if not 0 < float(self.mps_memory_fraction) <= 1:
            raise ValueError("mps_memory_fraction must be in (0, 1]")

    def rule_lr(self, rule: str) -> float:
        return self.inner_lr * float(self.rule_lr_scale.get(rule, 1.0))

    @property
    def continuing(self) -> bool:
        return bool(self.continue_from or self.continue_weights_from)

    def horizon(self) -> int:
        """Total optimizer steps of this run's schedule (the LR reaches its floor here)."""
        return int(self.continue_steps) if self.continuing else int(self.steps)

    def start(self) -> int:
        """First optimizer step of this schedule (the continuation start, else 0)."""
        return int(self.steps) if self.continuing else 0

    def peak(self) -> float:
        return float(self.continue_lr_peak or self.lr_peak) if self.continuing else float(self.lr_peak)


def check_rule_lr_scale(scale: dict) -> None:
    for k, v in scale.items():
        if k not in ("lin", "aice"):
            raise ValueError(f"rule_lr_scale: only lin/aice may be rescaled, got {k!r}")
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"rule_lr_scale[{k!r}] must be a positive finite number")


def lr_at(step: int, cfg: TrainCfg, *, start: int = 0, horizon: int | None = None,
          peak: float | None = None) -> float:
    """Re-warm floor -> peak over ``warmup`` steps, cosine back to floor at ``steps``.

    A continuation uses the same form shifted by its ``start`` step: re-warm
    over ``warmup`` steps from ``start``, cosine to the floor at ``horizon``.  The defaults
    reproduce the single-phase schedule exactly."""
    horizon = cfg.steps if horizon is None else horizon
    peak = cfg.lr_peak if peak is None else peak
    s = step - start
    if s < cfg.warmup:
        return cfg.lr_floor + (peak - cfg.lr_floor) * s / cfg.warmup
    if cfg.lr_schedule == "wsd":
        decay_start = horizon - start - cfg.wsd_decay
        t = min(1.0, max(0.0, (s - decay_start) / cfg.wsd_decay))
        return cfg.lr_floor + (peak - cfg.lr_floor) * (1.0 - t)
    t = min(1.0, (s - cfg.warmup) / max(1, horizon - start - cfg.warmup))
    return cfg.lr_floor + 0.5 * (peak - cfg.lr_floor) * (1 + math.cos(math.pi * t))


def schedule_lr(step: int, cfg: TrainCfg) -> float:
    """The outer LR this run applies at completed-step coordinate ``step``."""
    return lr_at(step, cfg, start=cfg.start(), horizon=cfg.horizon(), peak=cfg.peak())


def inner_lr_multiplier(step: int, cfg: TrainCfg) -> float:
    """MINI's 0.1→1 warmup for every schedule; WSD changes only the outer LR."""
    return 0.1 + 0.9 * min(step, cfg.inner_warmup) / max(1, cfg.inner_warmup)


class StreamRows:
    """Read-only row view over ``train_split`` followed by ``extra_train_splits``.

    Index ``i`` addresses the concatenation [train; extra_1; ...] without materialising it;
    ``arr[rows, :width]`` mirrors the memmap indexing the Prefetcher uses."""

    def __init__(self, parts: list):
        self.parts = list(parts)
        self.offsets = np.cumsum([0] + [len(a) for a in self.parts])
        self.shape = (int(self.offsets[-1]), self.parts[0].shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        rows, cols = key
        rows = np.asarray(rows, dtype=np.int64)
        part = np.searchsorted(self.offsets, rows, side="right") - 1
        out = np.empty((len(rows), len(range(*cols.indices(self.shape[1])))),
                       dtype=self.parts[0].dtype)
        for k in np.unique(part):
            m = part == k
            out[m] = self.parts[k][rows[m] - self.offsets[k], cols]
        return out


def stream_order(sizes: list[int], data_seed: int, n: int) -> np.ndarray:
    """The training row stream: the ``data_seed`` permutation of the train split, then a
    permutation of each extra split (offset into :class:`StreamRows`).

    With no extra split this is exactly ``permutation(len(train))[:n]``; every horizon's
    order is a prefix of every longer horizon's order."""
    parts = [np.random.default_rng(int(data_seed)).permutation(int(sizes[0]))]
    off = int(sizes[0])
    for k, size in enumerate(sizes[1:], start=1):
        parts.append(off + np.random.default_rng([int(data_seed), k]).permutation(int(size)))
        off += int(size)
    order = np.concatenate(parts)
    if len(order) < n:
        raise ValueError(f"training stream has {len(order)} rows, needs {n}")
    return order[:n]


class Prefetcher:
    """Background thread: rows of a uint32 [N, T + 1] array -> pinned int64 -> device.

    ``injector`` (SOQ P1) rewrites the host rows of each micro-batch before the copy to the
    device; its global row index is the stream position ``base + i`` of the row after the
    in-micro-batch sort, i.e. exactly the inner ``row_ids`` of that row.  The order of rows
    is never changed.  ``last_injected`` is the number of rows injected in the micro-batch
    most recently yielded."""

    def __init__(self, arr: np.ndarray, order: np.ndarray, micro: int, device, depth: int = 3,
                 width: int | None = None, injector=None, base: int = 0):
        self.arr, self.order, self.micro, self.device = arr, order, micro, torch.device(device)
        self.width = width or arr.shape[1]
        self.injector, self.base = injector, int(base)
        self.last_injected = 0
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.pin = self.device.type == "cuda"
        self.closed = threading.Event()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def host_rows(self, i: int) -> tuple[np.ndarray, int]:
        """uint32 rows of the micro-batch starting at order position ``i`` (injected)."""
        rows = np.sort(self.order[i:i + self.micro])
        x = np.asarray(self.arr[rows, : self.width])
        if self.injector is None:
            return x, 0
        idx = self.base + i + np.arange(len(rows), dtype=np.int64)
        y, _ = self.injector(x, idx)
        return y, int((y != x).any(1).sum())

    def _put(self, value):
        while not self.closed.is_set():
            try:
                self.q.put(value, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _work(self):
        for i in range(0, len(self.order), self.micro):
            x, n_inj = self.host_rows(i)
            t = torch.from_numpy(x.astype(np.int64))
            if not self._put((t.pin_memory() if self.pin else t, n_inj)):
                return
        self._put(None)

    def close(self):
        self.closed.set()
        self.worker.join(timeout=1)

    def __iter__(self):
        while (item := self.q.get()) is not None:
            t, self.last_injected = item
            yield t.to(self.device, non_blocking=True)


def mix_order(dclm_rows: int, code_rows: int, *, steps: int, batch: int,
              data_seed: int) -> np.ndarray:
    """Return deterministic (source, row) pairs for an 8+8 mixed stream.

    The DCLM half is exactly the prefix of the core DCLM permutation for the
    same seed; code rows use an independent counter-derived permutation.  The
    returned order is batch-major and keeps DCLM first in every batch.
    """
    if batch < 2 or batch % 2 or steps < 1 or dclm_rows < steps * (batch // 2):
        raise ValueError("mix requires an even batch and enough DCLM rows")
    half = batch // 2
    rng_d = np.random.default_rng(int(data_seed))
    rng_c = np.random.default_rng(int(data_seed) ^ 0x9E3779B9)
    d = rng_d.permutation(dclm_rows)[:steps * half]
    c = rng_c.permutation(code_rows)
    if len(c) < steps * half:
        c = np.resize(c, steps * half)
    c = c[:steps * half]
    out = np.empty((steps * batch, 2), dtype=np.int64)
    for step in range(steps):
        a, b = step * half, (step + 1) * half
        q = step * batch
        out[q:q + half] = np.stack([np.zeros(half, dtype=np.int64), d[a:b]], 1)
        out[q + half:q + batch] = np.stack([np.ones(half, dtype=np.int64), c[a:b]], 1)
    return out


def training_order(n_rows: int, *, steps: int, batch: int, data_seed: int,
                   data_start_step: int = 0) -> np.ndarray:
    """Return the contiguous slice of the seeded training stream for a run."""
    if data_start_step < 0 or steps < 1 or batch < 1:
        raise ValueError("steps, batch and data_start_step must be valid")
    full = np.random.default_rng(int(data_seed)).permutation(int(n_rows))
    lo, hi = data_start_step * batch, (data_start_step + steps) * batch
    if hi > len(full):
        raise ValueError("training stream is shorter than requested continuation")
    return full[lo:hi]


class MixedPrefetcher(Prefetcher):
    """Prefetch rows from DCLM/code arrays without materialising a mixed store."""

    def __init__(self, dclm: np.ndarray, code: np.ndarray, order: np.ndarray,
                 micro: int, device, depth: int = 3, width: int | None = None):
        self.dclm, self.code = dclm, code
        super().__init__(dclm, order, micro, device, depth=depth, width=width)

    def _work(self):
        for i in range(0, len(self.order), self.micro):
            pairs = self.order[i:i + self.micro]
            rows = np.stack([self.dclm[int(j)] if int(src) == 0 else self.code[int(j)]
                             for src, j in pairs], axis=0)[:, :self.width]
            t = torch.from_numpy(np.asarray(rows, dtype=np.int64))
            if not self._put((t.pin_memory() if self.pin else t, 0)):
                return
        self._put(None)


def _batch(t: torch.Tensor, bos: int):
    ids, tgt = t[:, :-1], t[:, 1:]
    return ids, tgt, tgt != bos


def build(cfg: TrainCfg):
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[cfg.dtype]
    if torch.device(cfg.device).type == "mps" and dt == torch.bfloat16:
        try:
            torch.empty(2, device="mps", dtype=torch.bfloat16).mul_(1)
        except (RuntimeError, TypeError):
            dt = torch.float32
            print("P2_MPS_BF16 fallback=fp32", flush=True)
    model_cfg = PRESETS[cfg.preset]
    if cfg.preset == "micro16k" and cfg.data_dir:
        meta_path = Path(cfg.data_dir) / "meta.json"
        if meta_path.exists():
            info = json.loads(meta_path.read_text()).get("tokenizer", {})
            model_cfg = replace(model_cfg, vocab_size=int(info.get("vocab_size", model_cfg.vocab_size)),
                                bos_id=int(info.get("bos_id", model_cfg.bos_id)),
                                eos_id=int(info.get("eos_id", model_cfg.eos_id)))
    model = build_model(model_cfg, compute_dtype=dt, device="cpu", seed=cfg.init_seed)
    if dt == torch.float64:
        model = model.double()
    meta = {}
    if cfg.init:
        meta = load_state(model, cfg.init)
    model.prefix_checkpoint = cfg.prefix_checkpoint
    return model.to(cfg.device), meta


def _sync(device):
    typ = torch.device(device).type
    if typ == "cuda":
        torch.cuda.synchronize()
    elif typ == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _configure_mps(cfg):
    if torch.device(cfg.device).type != "mps":
        return
    if hasattr(torch, "mps") and hasattr(torch.mps, "set_per_process_memory_fraction"):
        # PyTorch's fraction is relative to the recommended working set. Clamp
        # the configured fraction so the process never requests over 4 GB.
        recommended = torch.mps.recommended_max_memory()
        fraction = min(float(cfg.mps_memory_fraction), 4_000_000_000 / recommended)
        torch.mps.set_per_process_memory_fraction(fraction)


def make_optimizer(model, cfg: TrainCfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr_floor, betas=(cfg.beta1, cfg.beta2),
                             weight_decay=cfg.weight_decay,
                             fused=torch.device(cfg.device).type == "cuda")


def outer_microstep(model, t, cfg: TrainCfg, kind: str, inner: InnerCfg):
    ids, tgt, mask = _batch(t, model.cfg.bos_id)
    scale = 1.0 / cfg.global_batch
    if kind == "so":
        return so_backward(model, ids, tgt, mask, inner, scale,
                           barriers=cfg.barriers, fused_forward=cfg.so_fused_forward)
    return fo_backward(model, ids, tgt, mask, inner, scale, trunc_kv=cfg.trunc_kv,
                       barriers=cfg.barriers, fused_attn=cfg.fused_attn)


def somix_mask(data_seed: int, step: int, global_batch: int, p: float,
               device: torch.device | str = "cpu") -> torch.Tensor:
    """Counter-based subset of SO rows, independent of mutable RNG/resume state."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    perm = np.random.default_rng([int(data_seed), int(step), 0x534F4D49]).permutation(global_batch)
    mask = np.zeros(global_batch, dtype=bool)
    mask[perm[:round(p * global_batch)]] = True
    return torch.from_numpy(mask).to(device)


def partial_so_step(model, batch, cfg: TrainCfg, inner: InnerCfg):
    """Partition the unchanged loader batch; every microstep has row weight 1/B."""
    mask = somix_mask(cfg.data_seed, inner.step, cfg.global_batch, cfg.so_fraction).numpy()
    stats = []
    for kind, rows, micro in ((ARMS["e2e1-ce"][0], np.flatnonzero(~mask), cfg.micro_batch),
                              ("so", np.flatnonzero(mask), cfg.so_micro_batch)):
        indices = torch.as_tensor(rows, device=batch.device)
        for start in range(0, len(rows), micro):
            idx = indices[start:start + micro]
            inner.row_ids = inner.step * cfg.global_batch + idx
            stats.append(outer_microstep(model, batch.index_select(0, idx), cfg, kind, inner))
    return stats, int(mask.sum())


# SOQ-N3B keys: left out of the resume contract at their defaults, so resume states that
# lack them (P1 v3 / v3x) still match.
_N3B_KEYS = ("guard_every", "guard_first", "guard_dense_every", "guard_dense_until",
             "guard_max_rise", "guard_consecutive", "guard_restart_until", "train_loss_guard",
             "gdiv", "save_model_steps", "so_micro_batch", "so_fraction", "lr_schedule",
             "wsd_decay", "final_model_only", "data_start_step")


def _resume_contract(cfg):
    """Everything that defines the run's trajectory (continuation keys included)."""
    defaults = asdict(TrainCfg())
    return {k: v for k, v in asdict(cfg).items()
            if k not in {"resume", "stop_after", "out_dir", "resume_every",
                         "process_name", "measure_timing", "resume_dir",
                         "max_hours", "estimate_wall_s", "log_train_seconds",
                         "train_seconds_target", "keep_resume_steps", "contamination_check",
                         "mps_memory_fraction"}
            and not (k in _N3B_KEYS and v == defaults[k])}


def missing_train_split_error(cfg: TrainCfg, have: int | None = None, need: int | None = None,
                              split: str | None = None) -> ValueError:
    """The trainer refuses to start without the rows its horizon needs, naming the split."""
    names = [cfg.train_split, *cfg.extra_train_splits]
    if split is not None:
        return ValueError(f"training split {split!r} (extra_train_splits) is missing from "
                          f"{cfg.data_dir}: prepare it first (mode='prepare', extend=true, "
                          f"extend_quotas={{{split} = N}})")
    last = names[-1]
    nxt = chr(ord(last) + 1) if len(last) == 1 and "A" <= last < "Z" else "<next>"
    return ValueError(f"split {cfg.train_split} (+{cfg.extra_train_splits}) has {have} rows, "
                      f"needs {need}: rows {have}..{need - 1} need a further train split "
                      f"(split {nxt}: extend the store, then add it to extra_train_splits)")


# Keys a continuation must share with its parent: they define the trajectory up to the
# parent's endpoint and the data/injection stream after it.  Evaluation,
# logging, checkpointing and the continuation's own re-warm keys may differ.
CONTINUATION_IDENTITY = (
    "arm", "preset", "init", "init_label", "init_seed", "dtype", "seq_len", "global_batch",
    "micro_batch", "steps", "lr_peak", "lr_floor", "warmup", "beta1", "beta2", "weight_decay",
    "grad_clip", "inner_lr", "inner_clip", "r_ref", "barriers", "segment_len", "rule_lr_scale",
    "trunc_kv", "fused_attn", "so_fused_forward", "prefix_checkpoint", "train_split",
    "extra_train_splits", "mix", "code_train_split", "data_seed", "inner_warmup",
    "echo_f", "echo_version", "echo_p", "echo_lookup_share", "echo_slots", "echo_seed")


def check_continuation_parent(parent: dict, cfg: TrainCfg, order: np.ndarray,
                              data_hashes: dict) -> dict:
    """Validate a kept final state as the start of ``cfg``'s continuation; return provenance."""
    pcfg = parent.get("config")
    if not isinstance(pcfg, dict):
        raise ValueError("continue_from is not a trainer resume state")
    defaults = asdict(TrainCfg())
    mine = asdict(cfg)
    # A parent that was itself a continuation (e.g. P1 v3 1200 -> 2400) ends at its own
    # horizon; this run's ``steps`` is that endpoint, so only ``steps`` may differ.
    parent_continued = bool(pcfg.get("continue_from") or pcfg.get("continue_weights_from"))
    endpoint = int(pcfg.get("continue_steps", 0)) if parent_continued else pcfg.get("steps")
    keys = [k for k in CONTINUATION_IDENTITY if not (parent_continued and k == "steps")]
    changed = [k for k in keys if pcfg.get(k, defaults[k]) != mine[k]]
    if changed:
        raise ValueError(f"continuation changes parent training keys: {changed}")
    if parent["step"] != cfg.steps or endpoint != cfg.steps:
        raise ValueError(f"continue_from holds step {parent['step']}, not the parent endpoint "
                         f"steps={cfg.steps}")
    n0 = cfg.steps * cfg.global_batch
    if parent["cursor"] != n0 or not np.array_equal(parent["order"], order[:n0]):
        raise ValueError("continuation order prefix differs from the parent's stored order")
    by_name = lambda h: {(k if k in ("init", "echo_manifest") else Path(k).name): v  # noqa: E731
                         for k, v in h.items() if k not in ("continue_from", "continue_weights_from")}
    ph, mh = by_name(parent["data_sha256"]), by_name(data_hashes)
    diff = sorted(k for k in ph if ph[k] != mh.get(k))
    if diff:
        raise ValueError(f"continuation data or injection changed vs parent: {diff}")
    return {"parent_state": str(cfg.continue_from),
            "parent_state_sha256": data_hashes["continue_from"],
            "parent_step": int(parent["step"]), "parent_config": pcfg,
            "start": cfg.start(), "horizon": cfg.horizon(), "lr_peak": cfg.peak(),
            "warmup": cfg.warmup, "lr_floor": cfg.lr_floor,
            "rows": [cfg.start() * cfg.global_batch, cfg.horizon() * cfg.global_batch]}


def weights_continuation_provenance(cfg: TrainCfg, weights_sha256: str) -> dict:
    """The missing Adam/RNG state is a declared deviation, never a purported exact resume.

    CE/SO training has no dropout or global-RNG draws after model construction. Data order
    uses data_seed; injection uses (echo_seed, global row); other inner rules use explicit
    counters (seed, step, row, chunk). Global RNGs are seeded at the continuation boundary
    and then saved/restored by the ordinary resume machinery.
    """
    return {"parent_weights": cfg.continue_weights_from,
            "parent_weights_sha256": weights_sha256,
            "parent_run_id": cfg.continue_parent_run_id,
            "lost_state_sha256": cfg.continue_lost_state_sha256,
            "optimizer": "fresh (parent Adam state lost)",
            "rng": "boundary seeded from (data_seed, step); CE/SO training uses no global RNG",
            "parent_step": cfg.steps, "start": cfg.start(), "horizon": cfg.horizon(),
            "lr_peak": cfg.peak(), "warmup": cfg.warmup, "lr_floor": cfg.lr_floor,
            "rows": [cfg.start() * cfg.global_batch, cfg.horizon() * cfg.global_batch]}


def _rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else [])


def _restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload, path):
    """Publish after fsync; a killed writer leaves the preceding complete state usable."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _latest_state(root):
    paths = sorted(Path(root).glob("resume_step[0-9][0-9][0-9][0-9].pt"), reverse=True)
    if not paths:
        raise FileNotFoundError("resume=true but no resumable state exists")
    failures = []
    for path in paths:
        try:
            return torch.load(path, map_location="cpu", weights_only=False), path
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            failures.append(f"{path.name}: {exc}")
    raise ValueError("no readable resumable state: " + "; ".join(failures))


def _resume_root(cfg: TrainCfg, output_root: Path) -> Path:
    """Keep MINI's rolling restart state on the durable volume."""
    if cfg.resume_dir:
        path = Path(cfg.resume_dir)
    elif str(output_root).startswith("outputs/mini/"):
        path = Path("outputs/mini/resume") / output_root.relative_to("outputs/mini")
    else:
        path = output_root
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_steps(cfg: TrainCfg, kind: str) -> set[int]:
    """Volume checkpoint schedule: final for every arm, step 600 for SO.

    A continuation saves its own endpoint (the parent already holds 600 and its final)."""
    lo, hi = cfg.start(), cfg.horizon()
    if cfg.final_model_only:
        return {hi}
    steps = {hi}
    if kind == "so" and lo < 600 <= hi:
        steps.add(600)
    steps.update(int(x) for x in cfg.save_model_steps if lo < int(x) <= hi)
    # Tiny/P2-1 fixtures use intermediate checkpoints as swap inputs;
    # the mounted MINI production presets use the bounded schedule above.
    if cfg.preset not in {"mini-s", "mini-m", "125m"}:
        steps.update(int(x) for x in cfg.eval_steps if lo <= int(x) <= hi)
    return steps


@dataclass
class EvalSetup:
    """The immutable document/probe selection shared by training and MINI-E."""

    eval_docs_arr: np.ndarray
    probes: list
    eval_man: list
    elig: list
    ho_docs: np.ndarray
    ho_man: list
    calib_docs: np.ndarray
    calib_probes: list
    eval_meta: dict
    code_root: Path
    T1: int
    pool_n: int
    eligible_n: int


def prepare_eval_setup(cfg: TrainCfg, model, excluded, *, write_meta: bool = True) -> EvalSetup:
    """Prepare exactly the endpoint, held-out and calibration inputs used by ``train``.

    Keeping this selection in one function is important for MINI-E: a checkpoint
    read must use the same eligible rows, probe seeds, barriers and class labels as
    the in-training step-1200 read.
    """
    from .data import open_split
    from .evaluate import prepare_probes

    T1 = cfg.seq_len + 1
    eval_arr, eval_man = open_split(cfg.data_dir, cfg.eval_split)
    calib_arr, _ = open_split(cfg.data_dir, cfg.calib_split)
    excluded = np.asarray(excluded)
    pool = np.asarray(eval_arr[:, :T1])
    pool_probes = prepare_probes(
        pool, model.cfg.chunk, excluded, bos=model.cfg.bos_id,
        barriers=cfg.barriers,
        seeds=[int(m.get("row", i)) for i, m in enumerate(eval_man)],
    )
    eligible = [i for i, p in enumerate(pool_probes)
                if len(p.pos) >= cfg.min_probes and p.nf_ok]
    elig = eligible[:cfg.n_eval]
    if not elig:
        raise ValueError("no eligible evaluation documents")
    eval_docs_arr = pool[elig]
    probes = [pool_probes[i] for i in elig]
    ho_arr, ho_man = open_split(cfg.data_dir, cfg.ho_split)
    ho_docs = np.asarray(ho_arr[: cfg.ho_docs, :T1])
    calib_docs = np.asarray(calib_arr[: cfg.pilot_docs, :T1])
    for name, have, want in (("ho", len(ho_docs), cfg.ho_docs),
                             ("calib", len(calib_docs), cfg.pilot_docs)):
        if have < want:
            raise ValueError(f"split {name} has {have} documents, needs {want}")
    off = np.cumsum([0] + [len(p.pos) for p in probes])
    cat = lambda a: np.concatenate([getattr(p, a) for p in probes]) if probes else np.zeros(0)  # noqa: E731
    eval_meta = dict(
        init_label=np.asarray(cfg.init_label),
        eval_rows=np.asarray(elig),
        eval_sha1=np.asarray([eval_man[i]["sha1"] for i in elig]),
        eval_domain=np.asarray([eval_man[i]["domain"] for i in elig]),
        ho_sha1=np.asarray([m["sha1"] for m in ho_man[: len(ho_docs)]]),
        ho_domain=np.asarray([m["domain"] for m in ho_man[: len(ho_docs)]]),
        probe_offset=off,
        probe_pos=cat("pos"),
        probe_stratum=cat("stratum"),
        probe_n_src=cat("n_src"),
        probe_dist=cat("dist"),
        probe_match8=cat("match8"),
        focal_offset=np.cumsum([0] + [int(p.focal.sum()) for p in probes]),
        focal_pos=np.concatenate([np.flatnonzero(p.focal) for p in probes]),
    )
    if write_meta:
        root = Path(cfg.out_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "eval_meta.npz", **eval_meta)
    calib_probes = prepare_probes(
        calib_docs, model.cfg.chunk, excluded, bos=model.cfg.bos_id,
        barriers=cfg.barriers,
    )
    return EvalSetup(eval_docs_arr, probes, eval_man, elig, ho_docs, ho_man,
                     calib_docs, calib_probes, eval_meta,
                     Path(cfg.code_data_dir or cfg.data_dir), T1,
                     len(pool), len(eligible))


def parity_check(doc: dict, sha1: np.ndarray, ref_path: str, n: int, tol: float) -> dict:
    """V0 parity: mean |doc_m - ref doc_m| over the first ``n`` documents (m = adapt, noadapt).

    The documents must be the reference's first ``n`` (``eval_sha1``)."""
    modes = ["adapt", "noadapt"]
    with np.load(ref_path, allow_pickle=False) as z:
        if any(m not in doc or f"doc_{m}" not in z.files for m in modes):
            return {"ref": str(ref_path), "n_docs": 0, "same_documents": False, "tol": tol,
                    "mean_abs_delta": {}, "pass": False}
        n = int(min([n, len(sha1), len(z["eval_sha1"])] + [len(doc[m]) for m in modes]))
        same = bool(n > 0 and np.array_equal(np.asarray(z["eval_sha1"][:n]), np.asarray(sha1[:n])))
        deltas = {m: float(np.mean(np.abs(np.asarray(doc[m][:n], dtype=np.float64)
                                          - z[f"doc_{m}"][:n]))) for m in modes} if same else {}
    ok = bool(same and deltas and all(math.isfinite(d) and d <= tol for d in deltas.values()))
    deltas = {m: (d if math.isfinite(d) else None) for m, d in deltas.items()}  # JSON-safe
    return {"ref": str(ref_path), "n_docs": n, "same_documents": same, "tol": tol,
            "mean_abs_delta": deltas, "pass": ok}


def train(cfg: TrainCfg) -> dict:
    from . import probe, soq_p1, soq_p1v3
    from .data import open_split, repeat_classes, sha256_file  # numpy-only module
    from .evaluate import adapted_loss, eval_docs, prepare_probes
    from .guard import RunningMinGuard, probe_steps, running_min_rise, trip_action

    from .data import _set_process_name
    t_launch = time.time()
    _set_process_name(cfg.process_name or os.environ.get("PROCESS_NAME"))
    torch.set_num_threads(4)
    cfg.validate_layout()
    aborts = sorted(Path(cfg.out_dir).glob("*_abort.json"))
    if aborts:
        label = aborts[0].name.removesuffix("_abort.json")
        raise ValueError(f"{label}-stopped attempt cannot resume or restart: {[p.name for p in aborts]}")
    if cfg.continue_from and cfg.continue_weights_from:
        raise ValueError("continue_from and continue_weights_from are mutually exclusive")
    _configure_mps(cfg)
    kind, rule = ARMS[cfg.arm]
    H, start0 = cfg.horizon(), cfg.start()
    if cfg.resume_every < 1 or cfg.log_every < 1 or cfg.steps < 1:
        raise ValueError("resume_every, log_every and steps must be positive")
    if cfg.continuing:
        if cfg.continue_steps <= cfg.steps:
            raise ValueError("continue_steps must exceed the parent horizon (steps)")
        if cfg.continue_lr_peak not in (0.0, cfg.lr_peak, 1e-3):
            raise ValueError("continue_lr_peak must be 0 (same peak), lr_peak or the fallback 1e-3")
    elif cfg.continue_steps or cfg.continue_lr_peak:
        raise ValueError("continue_steps / continue_lr_peak require continue_from or continue_weights_from")
    if cfg.data_start_step < 0:
        raise ValueError("data_start_step must be nonnegative")
    if cfg.stop_after is not None and not start0 + 1 <= cfg.stop_after <= H:
        raise ValueError("stop_after must be within the unchanged steps horizon")
    if cfg.attempt not in (0, 1) or (cfg.attempt == 1 and cfg.peak() != 1e-3):
        raise ValueError("attempt must be 0 or 1; attempt 1 requires lr_peak=1e-3")
    if cfg.mix and (cfg.echo_f > 0 or cfg.extra_train_splits):
        raise ValueError("echo injection and extra_train_splits are defined for mix=false only")
    if cfg.train_loss_guard not in ("auto", "start", "running_min"):
        raise ValueError("train_loss_guard must be auto, start or running_min")
    if cfg.guard_every < 0 or cfg.guard_consecutive < 1:
        raise ValueError("guard_every must be >= 0 and guard_consecutive >= 1")
    unsaved = sorted(set(int(s) for s in cfg.save_model_steps)
                     - set(int(s) for s in cfg.eval_steps + cfg.full_eval_steps))
    if unsaved:
        raise ValueError(f"save_model_steps {unsaved} must be evaluation steps (eval_steps/full_eval_steps)")
    for split in cfg.extra_train_splits:  # hard precondition before step 1: present, non-empty
        if not all((Path(cfg.data_dir) / f"{split}.{ext}").exists() for ext in ("bin", "jsonl")) \
                or (Path(cfg.data_dir) / f"{split}.bin").stat().st_size == 0:
            raise missing_train_split_error(cfg, split=split)
    if cfg.contamination_check and cfg.extra_train_splits and not cfg.resume:
        from .data import contamination_enforced, split_contamination
        for split in cfg.extra_train_splits:
            rec = split_contamination(cfg.data_dir, split,
                                      (cfg.eval_split, cfg.ho_split, cfg.calib_split))
            rec["enforced"] = contamination_enforced(cfg.data_dir, split)
            print("P2_CONTAMINATION " + json.dumps(rec, sort_keys=True), flush=True)
            if rec["hits"] and rec["enforced"]:
                raise ValueError(f"split {split!r} overlaps the eval/ho/calib documents: {rec}")
    excl_path = os.path.join(cfg.data_dir, "excluded_vocab.npy")
    if not os.path.exists(excl_path):
        raise ValueError(f"required exclusions missing: {excl_path}")
    excluded = np.load(excl_path, allow_pickle=False)
    check_rule_lr_scale(cfg.rule_lr_scale)
    echo_api = soq_p1v3 if int(cfg.echo_version) == 3 else soq_p1
    injector = echo_api.make_train_injector(cfg, excluded)
    echo_manifest = echo_api.compact_manifest(injector) if injector is not None else None
    inner = InnerCfg(rule=rule, lr=cfg.rule_lr(rule), clip=cfg.inner_clip,
                     r_ref=cfg.r_ref, seed=cfg.data_seed)
    random.seed(cfg.data_seed)
    np.random.seed(cfg.data_seed)
    torch.manual_seed(cfg.data_seed)
    model, meta = build(cfg)
    if cfg.init and cfg.data_start_step:
        meta["sha256"] = sha256_file(cfg.init)
    train_config = asdict(cfg)
    if cfg.data_start_step == 0:
        train_config.pop("data_start_step", None)
    config_rec = {"train": train_config, "model": model.cfg.as_dict(),
                  "params": param_count(model), "init_meta": meta}
    if echo_manifest is not None:
        config_rec["echo"] = echo_manifest
    if cfg.continuing:
        config_rec["schedule"] = {"start": start0, "horizon": H, "lr_peak": cfg.peak()}
    if cfg.continue_weights_from:
        config_rec["continuation"] = weights_continuation_provenance(
            cfg, sha256_file(cfg.continue_weights_from))
    print("P2_CONFIG " + json.dumps(config_rec), flush=True)
    if (echo_manifest is not None and cfg.echo_manifest_sha256
            and echo_api.manifest_id(echo_manifest) != cfg.echo_manifest_sha256):
        rec = {"reason": "echo_manifest", "registered": cfg.echo_manifest_sha256,
               "actual": echo_api.manifest_id(echo_manifest)}
        print("P2_ABORT " + json.dumps(rec, sort_keys=True), flush=True)
        print("ABORT echo injector manifest differs from the registered echo_manifest_sha256", flush=True)
        return {"aborted": True, **rec}
    os.makedirs(cfg.out_dir, exist_ok=True)
    root = Path(cfg.out_dir)
    state_root = _resume_root(cfg, root)
    if cfg.continuing and Path(cfg.continue_from or cfg.continue_weights_from).resolve().parent == state_root.resolve():
        raise ValueError("a continuation needs its own out_dir/resume_dir, not the parent's")
    state, resume_path = _latest_state(state_root) if cfg.resume else (None, None)
    if state is None and (root / "steps.jsonl").exists():
        raise FileExistsError("fresh training requires a new out_dir (or resume=true)")
    data_splits = [cfg.train_split, *cfg.extra_train_splits, cfg.eval_split, cfg.ho_split,
                   cfg.calib_split]
    if cfg.echo_eval_docs:
        data_splits.append(cfg.echo_eval_split)
        if cfg.echo_diagnostics:
            data_splits.append(cfg.echo_train_eval_split)
    if cfg.mix:
        data_splits += [cfg.code_train_split, cfg.code_eval_split]
    code_root = Path(cfg.code_data_dir or cfg.data_dir)
    data_hashes = {str(p): sha256_file(p) for p in [Path(excl_path)] + [
        (code_root if cfg.mix and s in {cfg.code_train_split, cfg.code_eval_split}
         else Path(cfg.data_dir)) / f"{s}.{ext}" for s in dict.fromkeys(data_splits)
        for ext in ("bin", "jsonl")]}
    if cfg.init:
        data_hashes["init"] = sha256_file(cfg.init)
    if echo_manifest is not None:
        data_hashes["echo_manifest"] = echo_api.manifest_id(echo_manifest)
    if cfg.continue_from:
        data_hashes["continue_from"] = sha256_file(cfg.continue_from)
    if cfg.continue_weights_from:
        data_hashes["continue_weights_from"] = config_rec["continuation"]["parent_weights_sha256"]
    if state is not None:
        if state["config"] != _resume_contract(cfg) or state["data_sha256"] != data_hashes:
            raise ValueError("resume config, checkpoint or data changed")
        model.load_state_dict(state["model"], strict=True)
        for name, offset in state["log_offsets"].items():
            with (root / name).open("r+b") as f:
                if f.seek(0, os.SEEK_END) < offset:
                    raise ValueError("resume log is shorter than its committed offset")
                f.truncate(offset)
        # Discard any artifacts ahead of the last durable optimizer step.
        for pattern in ("eval_step*.npz", "model_step*.pt"):
            for p in root.glob(pattern):
                if int(p.stem.split("step")[-1]) > state["step"]:
                    p.unlink()
        print(f"P2_RESUME step={state['step']} cursor={state['cursor']} state={resume_path.name}", flush=True)

    def evidence(prefix, rec, log="events.jsonl"):
        line = prefix + " " + json.dumps(rec, sort_keys=True, allow_nan=False)
        with (root / log).open("a") as f:
            f.write(line + "\n")
            f.flush()
        print(line, flush=True)
    train_arr, _ = open_split(cfg.data_dir, cfg.train_split)
    offset = cfg.data_start_step * cfg.global_batch
    need = H * cfg.global_batch
    stream = train_arr
    if cfg.mix:
        if len(train_arr) < need // 2:
            raise ValueError(f"split {cfg.train_split} has {len(train_arr)} rows, needs {need // 2}")
        code_arr, _ = open_split(code_root, cfg.code_train_split)
        if len(code_arr) < need // 2:
            raise ValueError(f"split {cfg.code_train_split} has {len(code_arr)} rows, needs {need // 2}")
        order = mix_order(len(train_arr), len(code_arr), steps=cfg.data_start_step + H,
                          batch=cfg.global_batch, data_seed=cfg.data_seed)
        order = order[offset:]
    else:
        code_arr = None
        parts = [train_arr] + [open_split(cfg.data_dir, s)[0] for s in cfg.extra_train_splits]
        if sum(len(a) for a in parts) < offset + need:
            raise missing_train_split_error(cfg, sum(len(a) for a in parts), offset + need)
        order = stream_order([len(a) for a in parts], cfg.data_seed, offset + need)[offset:]
        if len(parts) > 1:
            stream = StreamRows(parts)
    if state is not None and (not np.array_equal(order, state["order"])
                              or state["cursor"] != state["step"] * cfg.global_batch):
        raise ValueError("resume data order or cursor changed")
    parent, provenance = None, None
    if cfg.continuing and state is None:
        if cfg.continue_from:
            parent = torch.load(cfg.continue_from, map_location="cpu", weights_only=False)
            provenance = check_continuation_parent(parent, cfg, order, data_hashes)
            model.load_state_dict(parent["model"], strict=True)
        else:
            checkpoint = torch.load(cfg.continue_weights_from, map_location="cpu", weights_only=True)
            if "model" in checkpoint:
                for key, expected in (("step", cfg.steps), ("arm", cfg.arm), ("preset", cfg.preset),
                                      ("init_seed", cfg.init_seed), ("data_seed", cfg.data_seed)):
                    if key in checkpoint and checkpoint[key] != expected:
                        raise ValueError(f"weights checkpoint {key} differs from parent config")
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
            del checkpoint
            provenance = config_rec["continuation"]
            boundary_seed = int.from_bytes(hashlib.sha256(
                f"{cfg.data_seed}:{start0}".encode()).digest()[:4], "little")
            random.seed(boundary_seed)
            np.random.seed(boundary_seed)
            torch.manual_seed(boundary_seed)
        (root / "continuation.json").write_text(json.dumps(provenance, sort_keys=True, indent=1) + "\n")
        evidence("P2_CONTINUE", {k: v for k, v in provenance.items() if k != "parent_config"})
    print(f"P2_DATA order_sha1={hashlib.sha1(order.tobytes()).hexdigest()[:12]} rows={need}"
          f" mix={cfg.mix}",
          flush=True)
    if injector is not None and state is None:
        if echo_api is soq_p1v3:  # v3 counts only the rows this run trains on
            ledger = echo_api.exposure(injector, need, cfg.seq_len + 1, start_row=start0 * cfg.global_batch)
        else:
            ledger = echo_api.exposure(injector, need, cfg.seq_len + 1)
        evidence("P2_ECHO_EXPOSURE", ledger)
        if cfg.preset in {"mini-s", "mini-m", "125m"}:
            echo_api.check_exposure(ledger)
    setup = prepare_eval_setup(cfg, model, excluded, write_meta=state is None)
    T1 = setup.T1
    eval_docs_arr, probes, eval_man, elig = (setup.eval_docs_arr, setup.probes,
                                             setup.eval_man, setup.elig)
    ho_docs, ho_man = setup.ho_docs, setup.ho_man
    calib_docs, calib_probes = setup.calib_docs, setup.calib_probes
    code_root = setup.code_root
    domains = len({eval_man[i]["domain"] for i in elig})
    print(f"P2_ELIGIBLE pool={setup.pool_n} eligible={setup.eligible_n} eligible_used={len(elig)} "
          f"domains={domains} "
          f"probes_per_doc_median={np.median([len(p.pos) for p in probes]):.0f}", flush=True)
    if state is not None:
        with np.load(root / "eval_meta.npz", allow_pickle=False) as old:
            if set(old.files) != set(setup.eval_meta) or any(
                    not np.array_equal(old[k], v) for k, v in setup.eval_meta.items()):
                raise ValueError("resume evaluation metadata changed")
    accum = cfg.global_batch // cfg.micro_batch
    if accum * cfg.micro_batch != cfg.global_batch:
        raise ValueError("global_batch must be a multiple of micro_batch")
    params = [p for p in model.parameters()]
    echo_set = echo_api.build_echo_eval_set(cfg, excluded) if cfg.echo_eval_docs else None
    echo_train_set = (soq_p1.build_train_echo_eval_set(cfg, excluded)
                      if cfg.echo_eval_docs and cfg.echo_diagnostics and cfg.echo_version != 3 else None)
    echo_verbatim_set = (soq_p1.build_echo_eval_set(cfg, excluded, query_mode="verbatim")
                         if cfg.echo_eval_docs and cfg.echo_diagnostics and cfg.echo_version != 3 else None)
    if echo_set is not None:
        print("P2_ECHO_EVAL_SET " + json.dumps({
            "n_docs": len(echo_set.docs), "split": cfg.echo_eval_split,
            "rows": [int(echo_set.row_ids[0]), int(echo_set.row_ids[-1]) + 1],
            "p": cfg.echo_eval_p, "docs_sha256": echo_set.docs_sha256,
            "params_sha256": echo_set.manifest.get("params_sha256")}), flush=True)
    pilot0 = state["pilot"]["baseline"] if state is not None else None
    pilot_history = state["pilot"]["history"] if state is not None else []
    own = state if state is not None else {}  # SOQ-N3B keys are absent from older states
    prior_peak_gb = float(own.get("peak_gb", 0.0))

    def peak_gb():
        return max(prior_peak_gb, torch.cuda.max_memory_allocated()/2**30
                   if torch.device(cfg.device).type == "cuda" else 0.0)
    guard = RunningMinGuard(cfg.guard_max_rise, cfg.guard_consecutive,
                            list(own.get("guard", {}).get("history", [])))
    guard_label = own.get("guard", {}).get("label")
    guard_at = set(probe_steps(start0, H, every=cfg.guard_every, first=cfg.guard_first,
                               dense_every=cfg.guard_dense_every, dense_until=cfg.guard_dense_until))
    running_loss_guard = (cfg.train_loss_guard == "running_min"
                          or (cfg.train_loss_guard == "auto" and not cfg.continuing))
    loss_hist = list(own.get("train_loss_hist", []))
    keep_steps = {int(x) for x in cfg.keep_resume_steps}
    train_s = float(own.get("train_s", 0.0))
    window_s = []

    def run_eval(step: int):
        eval_t0 = time.perf_counter()
        _sync(cfg.device)
        if step == 0:
            n_docs = cfg.init_eval_docs or cfg.n_eval
            n_ho = cfg.init_ho_docs or cfg.ho_docs
            full = not bool(cfg.init_eval_docs)
            modes = list(cfg.init_eval_modes if cfg.init_eval_docs else
                         dict.fromkeys(cfg.eval_modes + cfg.full_extra_modes + cfg.init_eval_modes))
            crossover = list(cfg.init_crossover_rules)
        else:
            full = step == H or step in cfg.full_eval_steps
            n_docs = cfg.n_eval if full else cfg.trajectory_docs
            n_ho = cfg.ho_docs if full else cfg.ho_trajectory_docs
            modes = list(cfg.eval_modes if full else cfg.trajectory_modes)
            if full:
                modes += cfg.full_extra_modes
            crossover = list(cfg.crossover_rules if full else [])
        if step in cfg.sizing_steps:
            modes += cfg.sizing_modes
        n_docs = min(len(eval_docs_arr), int(n_docs))
        n_ho = min(len(ho_docs), int(n_ho))
        sel = slice(0, n_docs)
        modes = list(dict.fromkeys(modes))
        model.eval()
        kw = dict(batch=cfg.eval_batch, inner_clip=cfg.inner_clip,
                  device=cfg.device, fused_attn=cfg.fused_attn,
                  barriers=cfg.barriers, r_ref=cfg.r_ref, data_seed=cfg.data_seed)
        own = dict(kw, inner_lr=cfg.rule_lr(rule))
        eval_cls = repeat_classes(eval_docs_arr[sel], cfg.segment_len)
        r = eval_docs(model, eval_docs_arr[sel], probes[sel], rule, modes=modes,
                      classes=eval_cls, **own)
        ho_cls = repeat_classes(ho_docs[:n_ho], cfg.segment_len)
        rh = eval_docs(model, ho_docs[:n_ho], [None] * n_ho, rule, modes=["heldout"],
                       classes=ho_cls, **own)
        r["doc"]["heldout"] = rh["doc"]["heldout"]
        r["doc"]["heldout_noadapt"] = rh["doc"]["heldout_noadapt"]
        extra = {}
        if crossover and (step == H or step == 0 or rule not in ("rp", "nc")):
            for other in dict.fromkeys(crossover):
                if other == rule:
                    continue
                xkw = dict(kw, inner_lr=cfg.rule_lr(other))
                x_modes = ["fP_focal", "fP_nonfocal"] if rule in ("rp", "nc") else ["adapt"]
                xr = eval_docs(model, eval_docs_arr[sel], probes[sel], other,
                               modes=x_modes, classes=eval_cls, **xkw)
                xh = eval_docs(model, ho_docs[:n_ho], [None] * n_ho, other,
                               modes=["heldout"], **xkw)
                if rule in ("rp", "nc"):
                    extra[f"x_{other}_M"] = xr["doc"]["fP_focal"] - xr["doc"]["fP_nonfocal"]
                else:
                    extra[f"x_{other}_adapt"] = xr["doc"]["adapt"]
                extra[f"x_{other}_heldout"] = xh["doc"]["heldout"]
        extra.update({f"src_nll_{m}": v for m, v in r["src_nll"].items()})
        for mode_name, by_class in rh.get("class_means", {}).items():
            for cls, vals in by_class.items():
                extra[f"ho_class_{mode_name}_{cls}"] = vals
        # per-source write-time NLL and cotangent norm (offsets: focal_offset in eval_meta.npz)
        for key in ("src_tok", "src_cot"):
            extra.update({f"{key}_{m}": np.concatenate(v).astype(np.float32)
                          for m, v in r[key].items() if v})
        extra.update({k: r[k] for k in ("clip_frac", "doc_clip_frac", "clip_scale", "update_norm")
                      if k in r})
        extra.update(eval_rows=np.asarray(elig)[sel],
                     init_label=np.asarray(cfg.init_label),
                     eval_sha1=np.asarray([eval_man[i]["sha1"] for i in elig])[sel],
                     eval_domain=np.asarray([eval_man[i]["domain"] for i in elig])[sel],
                     ho_sha1=np.asarray([m["sha1"] for m in ho_man[:n_ho]]),
                     ho_domain=np.asarray([m["domain"] for m in ho_man[:n_ho]]))
        # per-probe NLL concatenated in document order (offsets in eval_meta.npz)
        toks = {f"tok_{m}": np.concatenate(v).astype(np.float32)
                for m, v in r["tok"].items() if v and all(x is not None for x in v)}
        echo_summary = None
        if echo_set is not None:  # SOQ P1 echo read on TEST templates (clean evals unchanged)
            if cfg.echo_version == 3:
                echo_arrays, echo_summary = echo_api.echo_read(
                    model, echo_set, rule=rule, inner_lr=cfg.rule_lr(rule),
                    inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=cfg.device,
                    fused_attn=cfg.fused_attn, batch=cfg.eval_batch, r_ref=cfg.r_ref,
                    data_seed=cfg.data_seed, step=step, include_p=(step == H))
                print("P2_ECHO3 " + json.dumps({"step": step, **echo_summary}, sort_keys=True), flush=True)
            else:
                echo_arrays, echo_summary = soq_p1.echo_read(
                    model, echo_set, rule=rule, inner_lr=cfg.rule_lr(rule),
                    inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=cfg.device,
                    fused_attn=cfg.fused_attn, batch=cfg.eval_batch, r_ref=cfg.r_ref,
                    data_seed=cfg.data_seed, diagnostics=cfg.echo_diagnostics)
            extra.update(echo_arrays)
            if cfg.echo_diagnostics and cfg.echo_version != 3:
                diag_summaries = {}
                for label, eset in (("train", echo_train_set), ("verbatim", echo_verbatim_set)):
                    ea, esum = soq_p1.echo_read(model, eset, rule=rule, inner_lr=cfg.rule_lr(rule),
                        inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=cfg.device,
                        fused_attn=cfg.fused_attn, batch=cfg.eval_batch, r_ref=cfg.r_ref,
                        data_seed=cfg.data_seed, diagnostics=True)
                    extra.update({f"echo_{label}_{k[5:] if k.startswith('echo_') else k}": v
                                  for k, v in ea.items()})
                    extra[f"echo_{label}_summary_json"] = np.asarray(json.dumps(esum, sort_keys=True))
                    diag_summaries[label] = esum
                evidence("P2_ECHO_DIAG", {"step": step, "reads": diag_summaries})
        np.savez_compressed(os.path.join(cfg.out_dir, f"eval_step{step:04d}.npz"),
                            **{f"doc_{m}": v for m, v in r["doc"].items()}, **toks, **extra)
        if echo_summary is not None:
            evidence("P2_ECHO", {"step": step, **echo_summary})
        if cfg.mix and step in cfg.code_eval_steps:
            code_eval_arr, code_man = open_split(code_root, cfg.code_eval_split)
            code_docs = np.asarray(code_eval_arr[:cfg.code_eval_docs, :T1])
            cp = prepare_probes(code_docs, model.cfg.chunk, excluded, bos=model.cfg.bos_id,
                                barriers=cfg.barriers,
                                seeds=[int(m["pool_row"]) for m in code_man[:len(code_docs)]])
            cr = eval_docs(model, code_docs, cp, rule, modes=modes,
                           classes=repeat_classes(code_docs, cfg.segment_len), **own)
            ccat = lambda key: np.concatenate([getattr(p, key) for p in cp])
            np.savez_compressed(root / f"code_eval_step{step:04d}.npz",
                **{f"doc_{m}": v for m, v in cr["doc"].items()},
                **{f"tok_{m}": np.concatenate(v).astype(np.float32)
                   for m, v in cr["tok"].items() if v and all(x is not None for x in v)},
                probe_offset=np.cumsum([0] + [len(p.pos) for p in cp]),
                probe_stratum=ccat("stratum"),
                eval_sha1=np.asarray([m["sha1"] for m in code_man[:len(code_docs)]]))
            evidence("P2_CODE_EVAL", {"step": step, "n_docs": len(code_docs),
                     "mean": {m: float(np.nanmean(v)) for m, v in cr["doc"].items()}})
        summ = {m: float(np.nanmean(v)) for m, v in r["doc"].items()}
        evidence("P2_EVAL", {"step": step, "full": full, "n_docs": len(eval_docs_arr[sel]),
                                       "mean": summ,
                                       "clip_frac": r.get("clip_frac", np.zeros(0)).tolist(),
                                       "update_norm": r.get("update_norm", np.zeros(0)).tolist()})
        if cfg.save_checkpoints and step in checkpoint_steps(cfg, kind):
            torch.save({"model": model.state_dict(), "step": step, "arm": cfg.arm,
                        "preset": cfg.preset, "init_seed": cfg.init_seed,
                        "data_seed": cfg.data_seed, "init_label": cfg.init_label},
                       os.path.join(cfg.out_dir, f"model_step{step:04d}.pt"))
        _sync(cfg.device)
        eval_times.append((step, (time.perf_counter() - eval_t0) * 1000.0))
        return r

    def pilot_loss() -> float:  # SO arm's ordinary LM loss on calibration docs
        # SOQ-N3B: in the run's own barrier layout (not the probe default (2, 4)).
        r = eval_docs(model, calib_docs, calib_probes, rule, modes=["heldout"],
                      batch=cfg.eval_batch, device=cfg.device, fused_attn=cfg.fused_attn,
                      barriers=cfg.barriers)
        return float(np.mean(r["doc"]["heldout"]))

    def guard_loss() -> float:  # P2-3 guard probe: adapted NLL, own rule, all loss-valid targets
        model.eval()
        s, n = adapted_loss(model, calib_docs, rule, batch=cfg.eval_batch,
                            inner_lr=cfg.rule_lr(rule), inner_clip=cfg.inner_clip,
                            device=cfg.device, fused_attn=cfg.fused_attn, barriers=cfg.barriers,
                            r_ref=cfg.r_ref, data_seed=cfg.data_seed,
                            classes=(repeat_classes(calib_docs, cfg.segment_len)
                                     if rule in ("rp", "nc") else None))
        return float(s.sum() / max(1.0, n.sum()))

    def parity_abort(rec):
        evidence("P2_PARITY", rec)
        if rec["pass"]:
            return None
        abort = {"reason": "parity", **rec}
        evidence("P2_ABORT", abort)
        print(f"ABORT parity: mean |delta| {rec['mean_abs_delta']} > tol {rec['tol']} "
              f"(same_documents={rec['same_documents']}) vs {rec['ref']}", flush=True)
        (root / "parity_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
        return {"aborted": True, **abort}

    eval_times = []
    if state is None and not cfg.continuing and cfg.eval_at_init:
        r0 = run_eval(0)
        if cfg.parity_ref:  # V0: the step-0 read against the reference run's step-0 read
            sha = np.asarray([eval_man[i]["sha1"] for i in elig])
            out = parity_abort(parity_check(r0["doc"], sha, cfg.parity_ref, cfg.parity_docs,
                                            cfg.parity_tol))
            if out is not None:
                return out
        a = probe.assay(r0["doc"]["fP_focal"] - r0["doc"]["fP_nonfocal"],
                        r0["doc"]["noadapt"] - r0["doc"]["adapt"],
                        np.asarray([eval_man[i]["domain"] for i in elig[:min(len(elig), cfg.init_eval_docs or cfg.n_eval)]]))
        evidence("P2_ASSAY", {"rule": rule, "descriptive": rule != "ce", **a})
        if rule == "ce" and cfg.assay_gate and not a["pass"]:
            return {"aborted": True, "assay": a}
    elif state is None and not cfg.continuing:
        evidence("P2_ASSAY", {"rule": rule, "skipped": True})
    elif state is None and cfg.parity_ref:
        # V0 for a continuation: the loaded parent weights re-read the parent's start-step eval.
        n = min(len(eval_docs_arr), int(cfg.parity_docs))
        model.eval()
        rp = eval_docs(model, eval_docs_arr[:n], probes[:n], rule, modes=["adapt", "noadapt"],
                       classes=repeat_classes(eval_docs_arr[:n], cfg.segment_len),
                       batch=cfg.eval_batch, inner_lr=cfg.rule_lr(rule), inner_clip=cfg.inner_clip,
                       device=cfg.device, fused_attn=cfg.fused_attn, barriers=cfg.barriers,
                       r_ref=cfg.r_ref, data_seed=cfg.data_seed)
        sha = np.asarray([eval_man[i]["sha1"] for i in elig])
        out = parity_abort({"step": start0, **parity_check(rp["doc"], sha, cfg.parity_ref, n,
                                                          cfg.parity_tol)})
        if out is not None:
            return out
    if state is None and cfg.continuing and cfg.echo_diagnostics and start0 in cfg.echo_eval_steps:
        # Echo-only endpoint read at the loaded parent weights.  The parity check above
        # remains the production V0 guard and aborts before this read on mismatch.
        def run_echo_only(step):
            if cfg.echo_version == 3:
                payload, esum = echo_api.echo_read(
                    model, echo_set, rule=rule, inner_lr=cfg.rule_lr(rule),
                    inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=cfg.device,
                    fused_attn=cfg.fused_attn, batch=cfg.eval_batch, r_ref=cfg.r_ref,
                    data_seed=cfg.data_seed, step=step, include_p=(step == H))
                evidence("P2_ECHO3", {"step": step, **esum})
                np.savez_compressed(root / f"eval_step{step:04d}.npz", **payload)
                return
            payload = {}
            diag_summaries = {}
            for label, eset in (("", echo_set), ("train", echo_train_set),
                                ("verbatim", echo_verbatim_set)):
                ea, esum = soq_p1.echo_read(model, eset, rule=rule, inner_lr=cfg.rule_lr(rule),
                    inner_clip=cfg.inner_clip, barriers=cfg.barriers, device=cfg.device,
                    fused_attn=cfg.fused_attn, batch=cfg.eval_batch, r_ref=cfg.r_ref,
                    data_seed=cfg.data_seed, diagnostics=True)
                payload.update(({k: v for k, v in ea.items()} if not label else
                                {f"echo_{label}_{k[5:] if k.startswith('echo_') else k}": v
                                 for k, v in ea.items()}))
                if label:
                    diag_summaries[label] = esum
                else:
                    evidence("P2_ECHO", {"step": step, **esum})
            evidence("P2_ECHO_DIAG", {"step": step, "reads": diag_summaries})
            np.savez_compressed(root / f"eval_step{step:04d}.npz", **payload)
        run_echo_only(start0)
    opt = make_optimizer(model, cfg)
    src = state if state is not None else parent
    start_step = src["step"] if src is not None else start0
    if src is not None:
        opt.load_state_dict(src["optimizer"])
        _restore_rng(src["rng"], cfg.device)
    prefetch = (MixedPrefetcher(train_arr, code_arr, order[start_step * cfg.global_batch:],
                                cfg.micro_batch, cfg.device, width=T1)
                if cfg.mix else
                Prefetcher(stream, order[start_step * cfg.global_batch:], cfg.micro_batch,
                           cfg.device, width=T1, injector=injector,
                           base=start_step * cfg.global_batch))
    data = iter(prefetch)
    if state is None and cfg.arm == "e2e2" and cfg.pilot_steps:
        pilot0 = pilot_loss()
        evidence("P2_PILOT", dict(step=start0, calib_heldout=pilot0))
    t0, tok = time.time(), start_step * cfg.global_batch * cfg.seq_len
    session_tokens = tok
    loss_acc = src["loss_acc"].to(cfg.device) if src is not None else torch.zeros((), device=cfg.device)
    loss_count = src["loss_count"] if src is not None else 0
    # Injected rows since the last P2_STEP record (persisted so resumed logs are identical).
    echo_rows = int(src.get("echo_rows", 0)) if src is not None else 0
    echo_rows_total = 0

    def save_resume(done):
        # Commit the non-log-interval tail before persisting its log offsets and train_s.
        # A restart then restores exactly the same complete timing prefix.
        if cfg.log_train_seconds and window_s:
            evidence("P2_TIME", {"step": done, "step_s": list(window_s), "train_s": train_s})
            window_s.clear()
        for log in ("steps.jsonl", "events.jsonl"):
            with (root / log).open("a") as f:
                f.flush()
                os.fsync(f.fileno())
        payload = dict(version=1, model=model.state_dict(), optimizer=opt.state_dict(), step=done,
                       cursor=done * cfg.global_batch, order=order, rng=_rng_state(cfg.device),
                       pilot=dict(baseline=pilot0, history=pilot_history), attempt=cfg.attempt,
                       guard=dict(history=guard.history, label=guard_label),
                       train_loss_hist=loss_hist, train_s=train_s, peak_gb=peak_gb(),
                       loss_acc=loss_acc, loss_count=loss_count, echo_rows=echo_rows,
                       config=_resume_contract(cfg),
                       data_sha256=data_hashes,
                       log_offsets={n: (root / n).stat().st_size for n in ("steps.jsonl", "events.jsonl")})
        began = time.monotonic()
        path = state_root / f"resume_step{done:04d}.pt"
        _atomic_save(payload, path)
        if done in keep_steps:
            kept = state_root / f"resume_keep{done:04d}.pt"
            shutil.copyfile(path, kept)
            print("P2_STATE_KEEP " + json.dumps(dict(step=done, path=str(kept))), flush=True)
        keep = 1 if (str(state_root).startswith("outputs/mini/")
                     or str(root).startswith("outputs/mini/")) else 2
        for old in sorted(state_root.glob("resume_step[0-9][0-9][0-9][0-9].pt"))[:-keep]:
            old.unlink()
        print("P2_STATE " + json.dumps(dict(step=done, bytes=path.stat().st_size,
                                             wall_s=time.monotonic() - began)), flush=True)

    def gdiv_event(done, loss_v, gnorm_v):
        """P2-3 G-div: reload the last resume state once and replay; a second event stops."""
        prefetch.close()
        marker = root / "gdiv.json"
        prior = json.loads(marker.read_text()) if marker.exists() else []
        rec = {"step": done, "loss": repr(loss_v), "grad_norm": repr(gnorm_v), "event": len(prior) + 1}
        marker.write_text(json.dumps(prior+[rec], sort_keys=True) + "\n")
        if prior:
            abort = {"reason": "gdiv_diverged", "label": "DIVERGED", **rec, "previous": prior}
            evidence("P2_ABORT", abort)
            (root / "divergence_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
            return {"aborted": True, **abort}
        print("P2_GDIV " + json.dumps({**rec, "action": "reload_last_resume_state"}, sort_keys=True),
              flush=True)
        return train(replace(cfg, resume=True))

    if cfg.gdiv and state is None:
        save_resume(start_step)  # G-div needs a state to reload before the first periodic one
    step_times = []
    train_loss0 = None
    final_step = cfg.stop_after or H
    for step in range(start_step, final_step):
        step_t0 = time.perf_counter()
        _sync(cfg.device)
        model.train()
        inner.step = step
        # Registered η warmup applies to every inner rule.  The object is
        # mutated once per optimizer step, so no host synchronisation enters
        # the microstep itself.
        inner.lr = cfg.rule_lr(rule) * inner_lr_multiplier(step + 1, cfg)
        # Completed optimizer steps are the preregistered time coordinate:
        # peak at step 30, floor at step 400 (step 0 is the initialization assay).
        lr = schedule_lr(step + 1, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        stats = []
        so_rows = cfg.global_batch if kind == "so" else 0
        if kind == "somix":
            batch = torch.cat([next(data) for _ in range(accum)])
            stats, so_rows = partial_so_step(model, batch, cfg, inner)
            tok += sum(s.n_seq for s in stats) * cfg.seq_len
        for micro_idx in range(accum if kind != "somix" else 0):
            batch = next(data)
            inner.row_ids = torch.arange(offset + step * cfg.global_batch + micro_idx * cfg.micro_batch,
                                         offset + step * cfg.global_batch + (micro_idx + 1) * cfg.micro_batch,
                                         device=batch.device)
            if rule == "nc":
                cls = repeat_classes(batch.detach().cpu().numpy(), cfg.segment_len)
                inner.classes = torch.from_numpy(cls).to(batch.device)
            echo_rows += prefetch.last_injected
            st = outer_microstep(model, batch, cfg, kind, inner)
            stats.append(st)
            tok += st.n_seq * cfg.seq_len
        loss = sum(s.loss_sum for s in stats) / cfg.global_batch
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, foreach=True)
        if cfg.gdiv and not bool(torch.isfinite(loss) & torch.isfinite(gnorm)):
            opt.zero_grad(set_to_none=True)
            return gdiv_event(step + 1, float(loss), float(gnorm))
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_acc += loss.detach()
        loss_count += 1
        done = step + 1
        _sync(cfg.device)
        step_times.append((time.perf_counter() - step_t0) * 1000.0)
        if cfg.log_train_seconds:
            window_s.append(step_times[-1] / 1000.0)
            train_s += window_s[-1]
        if done % cfg.log_every == 0 or done == H:
            norms = torch.cat([s.inner_norm.flatten() for s in stats])
            rec = {"step": done, "lr": lr, "loss": float(loss_acc) / loss_count,
                   "loss_last": float(loss), "grad_norm": float(gnorm),
                   "inner_norm_med": float(norms.median()) if norms.numel() else None,
                   "inner_lr_mult": inner_lr_multiplier(done, cfg),
                   "inner_clip_frac": float((norms > cfg.inner_clip).float().mean())
                   if norms.numel() else None,
                   "chunk_loss": (sum(s.chunk_loss * s.n_seq for s in stats) /
                                  cfg.global_batch).tolist()}
            if kind == "somix":
                rec.update(so_rows=so_rows, global_batch=cfg.global_batch)
                if cfg.measure_timing:
                    rec["ms_per_step"] = step_times[-1]
            rs = [s.rule_stats or {} for s in stats]
            active = [x for d in rs for x in d.get("rp_floor_active", [])]
            if active:
                aa = torch.cat([x.reshape(-1) for x in active]).to(torch.float32)
                rec["rp_floor_active_share"] = float(aa.mean())
            ex = [x for d in rs for x in d.get("extra_norm", []) if x is not None]
            ce = [x for d in rs for x in d.get("ce_norm", []) if x is not None]
            if ex and ce:
                exn, cen = torch.cat(ex), torch.cat(ce)
                rec["rp_extra_norm"] = exn.tolist()
                rec["rp_ce_norm"] = cen.tolist()
                rec["rp_median_ratio"] = float(torch.median(exn / cen.clamp_min(1e-12)))
                binds = [x for d in rs for x in d.get("induced_binding", []) if x is not None]
                rec["rp_induced_binding"] = float(torch.cat(binds).float().mean()) if binds else 0.0
            counts = {}
            code0 = 0
            nc_fail = 0
            for d in rs:
                for key, value in d.get("nc_counts", {}).items():
                    if isinstance(key, tuple) and len(key) == 3:
                        key = (key[1], key[2])
                    if key in counts:
                        counts[key] = (counts[key][0] + value[0], counts[key][1] + value[1])
                    else:
                        counts[key] = tuple(value)
                code0 += int(d.get("nc_code0_count", 0))
                nc_fail += int(torch.cat(d.get("nc_fail", [])).sum().item()) if d.get("nc_fail") else 0
            if counts:
                values = torch.stack([torch.stack([torch.as_tensor(x, device=cfg.device) for x in v])
                                      for v in counts.values()]).cpu().tolist()
                rec["nc_counts"] = {str(k): v for k, v in zip(counts, values)}
                rec["nc_code0_count"] = code0
                rec["nc_fail"] = nc_fail
            if injector is not None:
                rec["echo_rows"] = echo_rows
                echo_rows_total += echo_rows
                echo_rows = 0
            if cfg.log_train_seconds:
                rec.update(micro_batch=cfg.micro_batch, so_micro_batch=cfg.so_micro_batch,
                           train_seconds=train_s, train_seconds_target=cfg.train_seconds_target,
                           ms_per_step=step_times[-1])
            evidence("P2_STEP", rec, "steps.jsonl")
            if cfg.log_train_seconds:
                evidence("P2_TIME", {"step": done, "step_s": window_s, "train_s": train_s})
                window_s = []
            if cfg.train_loss_max_rise and running_loss_guard:
                # From scratch the start-relative rule is inert: running minimum.
                rise = running_min_rise(loss_hist, rec["loss"])
                loss_hist.append(rec["loss"])
                if not math.isfinite(rec["loss"]) or rise > cfg.train_loss_max_rise:
                    abort = {"reason": "train_loss_divergence", "step": done, "loss": rec["loss"],
                             "baseline": rec["loss"] - rise if math.isfinite(rise) else None,
                             "baseline_kind": "running_min", "max_rise": cfg.train_loss_max_rise}
                    print("P2_ABORT " + json.dumps(abort, sort_keys=True), flush=True)
                    (root / "divergence_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
                    prefetch.close()
                    return {"aborted": True, **abort}
            elif cfg.train_loss_max_rise:  # a negative value trips at the first log (test hook)
                if train_loss0 is None:
                    train_loss0 = rec["loss"]
                if not math.isfinite(rec["loss"]) or rec["loss"] - train_loss0 > cfg.train_loss_max_rise:
                    abort = {"reason": "train_loss_divergence", "step": done, "loss": rec["loss"],
                             "baseline": train_loss0, "max_rise": cfg.train_loss_max_rise}
                    print("P2_ABORT " + json.dumps(abort, sort_keys=True), flush=True)
                    (root / "divergence_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
                    prefetch.close()
                    return {"aborted": True, **abort}
            print("P2_PERF " + json.dumps({"step": done, "tok_per_s": (tok - session_tokens) / (time.time() - t0),
                   "peak_gb": (torch.cuda.max_memory_allocated() / 2**30
                              if torch.device(cfg.device).type == "cuda" else
                              (torch.mps.current_allocated_memory() / 2**30
                               if torch.device(cfg.device).type == "mps" and hasattr(torch, "mps") and
                               hasattr(torch.mps, "current_allocated_memory") else 0.0))
                  }), flush=True)
            loss_acc.zero_()
            loss_count = 0
        if pilot0 is not None and done in cfg.pilot_steps:
            pl = pilot_loss()
            pilot_history.append(dict(step=done, calib_heldout=pl, rise=pl - pilot0))
            evidence("P2_PILOT", pilot_history[-1])
            # Plain searchable line for external monitors (besides the JSON record).
            print(f"P2_PILOT step={done} calib_heldout={pl:.5f} rise={pl - pilot0:+.5f}", flush=True)
            if not math.isfinite(pl) or pl - pilot0 > cfg.pilot_max_rise:
                retry = ("rerun_continuation_from_parent_lr_1e-3_attempt_1" if cfg.continuing
                         else "restart_so_lr_1e-3_attempt_1_then_launch_fo")
                abort = {"step": done, "rise": pl - pilot0, "attempt": cfg.attempt,
                         "next": retry if cfg.attempt == 0 else "stop_replan"}
                if cfg.attempt == 1:
                    abort["final"] = True
                print("P2_ABORT " + json.dumps(abort), flush=True)
                (root / "pilot_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
                save_resume(done)
                prefetch.close()
                return {"aborted": True, **abort}
        if done in guard_at:
            pl = guard_loss()
            if not math.isfinite(pl) and cfg.gdiv:
                return gdiv_event(done, pl, float("nan"))
            g = guard.update(done, pl)
            evidence("P2_GUARD", {k: (repr(v) if isinstance(v, float) and not math.isfinite(v) else v)
                                  for k, v in g.items()})
            act = None
            if not g["finite"]:
                act = {"label": "NONFINITE_PROBE", "stop": True, "next": "stop_replan"}
            elif g["trip"]:
                act = trip_action(done, kind, cfg.attempt, cfg.guard_restart_until)
                if cfg.guard_fo_stop and not act["stop"]:
                    act = dict(act, stop=True, next="stop_user_decides")
            if act is not None and not act["stop"]:
                if guard_label is None:  # FO-UNSTABLE is descriptive: record it and continue
                    guard_label = act["label"]
                    (root / "guard_label.json").write_text(json.dumps(
                        {"step": done, **act}, sort_keys=True) + "\n")
                evidence("P2_GUARD_LABEL", {"step": done, **act})
            elif act is not None:
                abort = {"reason": "guard", "step": done, "kind": kind, "attempt": cfg.attempt,
                         "value": repr(pl), "rise": g["rise"], **act}
                evidence("P2_ABORT", abort)
                (root / "guard_abort.json").write_text(json.dumps(abort, sort_keys=True) + "\n")
                save_resume(done)
                prefetch.close()
                return {"aborted": True, **abort}
        if done in set(cfg.eval_steps + cfg.full_eval_steps + cfg.sizing_steps):
            run_eval(done)
        if done % cfg.resume_every == 0 or done == final_step or done in keep_steps:
            save_resume(done)
        if cfg.max_hours and done < final_step and time.time() - t_launch > 3600.0 * cfg.max_hours:
            if done % cfg.resume_every:
                save_resume(done)
            prefetch.close()
            abort = {"reason": "max_hours", "step": done, "max_hours": cfg.max_hours}
            print("P2_ABORT " + json.dumps(abort, sort_keys=True), flush=True)
            print(f"ABORT max_hours={cfg.max_hours} reached at step {done}; resume=true continues",
                  flush=True)
            return {"aborted": True, **abort}
    prefetch.close()
    completed = final_step >= H
    final_state = None
    if completed and cfg.keep_final_state:
        # The endpoint state (model, Adam moments, RNG, cursor, order) is kept
        # for a later continuation; older rolling states are dropped.
        final = state_root / f"resume_step{H:04d}.pt"
        if not final.exists():
            raise RuntimeError(f"keep_final_state: {final} was not written")
        for p in state_root.glob("resume_step[0-9][0-9][0-9][0-9].pt"):
            if p != final:
                p.unlink()
        final_state = {"path": str(final), "step": H, "bytes": final.stat().st_size,
                       "sha256": sha256_file(final)}
        (root / "final_state.json").write_text(json.dumps(final_state, sort_keys=True) + "\n")
        print("P2_FINAL_STATE " + json.dumps(final_state, sort_keys=True), flush=True)
    elif completed and (str(state_root).startswith("outputs/mini/")
                        or str(root).startswith("outputs/mini/")):
        for p in state_root.glob("resume_step*.pt"):
            p.unlink(missing_ok=True)
    out = {"aborted": False, "steps": final_step, "tokens": tok,
           "init_label": cfg.init_label,
           "wall_s": time.time() - t0, "tok_per_s": (tok - session_tokens) / (time.time() - t0),
           "step_ms": step_times, "eval_ms": eval_times, "peak_gb": peak_gb()}
    if cfg.continuing:
        out.update(start=start0, horizon=H)
    if injector is not None:
        out["echo_rows"] = echo_rows_total + echo_rows
    if final_state is not None:
        out["final_state"] = final_state
    if guard_at:
        out["guard_label"] = guard_label
    if cfg.log_train_seconds:
        target = float(cfg.train_seconds_target)
        ratio = train_s / target if target > 0 else None
        out["train_seconds"] = {"steps": final_step, "complete": completed, "train_s": train_s,
                                "target_s": target or None, "ratio": ratio,
                                "within_2pct": (abs(ratio - 1.0) <= 0.02) if ratio is not None else None}
        evidence("P2_TRAIN_SECONDS", out["train_seconds"])
    return out
