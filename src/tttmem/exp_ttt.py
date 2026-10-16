"""TTT-E2E toy kill-test: second-order vs first-order vs parallel-Jacobi.

One architecture, several arms. ``fa`` and ``swa`` are static models with full or
sliding-window attention. The TTT arms turn the last positionwise block
``h' = h + W2 gelu(W1 LN(h) + b1) + b2`` into per-sequence fast weights ``W``
initialised from slow params ``W0`` and updated by gradient steps on the
next-token loss over chunks of the context:

- ``e2e2``: sequential chunks, inner gradient kept in the graph (second order);
- ``e2e1``: sequential chunks, inner gradient detached (first order; ``W0`` and
  ``eta`` still receive outer gradients);
- ``jac1``/``jac2``: parallel Jacobi iteration of the same chunk recurrence;
- ``naive``: no training-time TTT, first-order TTT applied at eval only.

Two tasks bound the claims. ``icl_mc`` (in-context Markov chains) has an exact
Bayesian ceiling, which makes "how much of the SWA-to-full-attention gap does
TTT close" measurable. ``echo`` / ``echo_ctrl`` are built so that the inner
gradient vanishes at write time whenever attention can already copy the answer,
which is where first-order TTT should fail by construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import multiprocessing as mp
import queue
import sys
import traceback
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FAST_KEYS = ("W1", "b1", "W2", "b2")
MATRIX_NDIM = {"W1": 2, "b1": 1, "W2": 2, "b2": 1}
STATIC_ARMS = ("fa", "swa")
from .aux_b9 import B9_BASE
# R4-2L arms: first order, frozen code, no outer mask. The four lin arms
# replace the inner CE at the selector's positions with the prediction-independent
# linear token loss; w600 only moves the LR peak; ctrl is the anchor model on echo_ctrl.
R42L_ARMS = ("e2e1emb_linkv", "e2e1emb_lin", "e2e1emb_linvW", "e2e1emb_linxkv",
             "e2e1emb_w600", "e2e1emb_ctrl")
# P0.0 arm: plain first order, no frozen code, the linear token loss at every
# position. It needs no position metadata, so it runs on icl_mc.
P00_ARMS = ("e2e1_lin",)
AICE_ARMS = ("e2e1emb_aice", "e2e1_aice")
LIN_ARMS = {"e2e1emb_linkv": "kv", "e2e1emb_lin": "all", "e2e1emb_linvW": "vW",
            "e2e1emb_linxkv": "xkv", "e2e1_lin": "all"}
ARM_INNER_LOSS = {"e2e1emb_linkv": "linkv", "e2e1emb_lin": "lin",
                  "e2e1emb_linvW": "linvW", "e2e1emb_linxkv": "linxkv", "e2e1_lin": "lin"}
ARM_WARMUP = {"e2e1emb_w600": 600}
ARM_TASKS = {"e2e1emb_ctrl": "echo_ctrl"}
ARM_TASKS.update({a: "echo" for a in B9_BASE})
COPY_Q = {"e2e1emb_cn99": .99, "e2e1emb_cn97": .97,
          "e2e1emb_cn95": .95, "e2e1emb_cn90": .90}
NOISE_ARMS = (*COPY_Q, "e2e1emb_rf95")
KAMIN_ARMS = ("e2e1emb_kamin", "e2e1emb_kaminlin", "e2e1emb_kaminctl")
KVETA_ARMS = ("e2e1emb_kveta", "e2e1emb_kveta10")
R44A_ARMS = ("e2e1emb", "e2e1emb_edge", *NOISE_ARMS, *KAMIN_ARMS, *KVETA_ARMS)
ARM_TASKS.update({a: "echo" for a in R44A_ARMS if a != "e2e1emb"})
CORRUPTION_SEED = 4406
CORRUPTION_EVAL_SEED = 20250308
SEQUENTIAL_ARMS = ("e2e1", "e2e2", "e2e1k", "e2e1e", "e2e1emb", "e2e2emb", "e2e1embP",
                   "e2e1emb_kvm", "e2e1emb_rnd", "e2e1_kvm", *R42L_ARMS, *R44A_ARMS[1:],
                   *P00_ARMS, *AICE_ARMS, "e2e1_code", *B9_BASE,
                   # R4-3: the two factor halves with the frozen code, and the kv outer
                   # mask on plain second order.
                   "e2e1e_emb", "e2e1k_emb", "e2e2_kvm")
JACOBI_ARMS = {"jac1": 1, "jac2": 2, "jac2so": 2}
SECOND_ORDER_ARMS = {"e2e2", "e2e2emb", "e2e2_kvm", "jac2so"}
FACTOR_ARMS = {"e2e1k": (True, False), "e2e1e": (False, True),
               "e2e1k_emb": (True, False), "e2e1e_emb": (False, True)}
EMB_ARMS = {"e2e1emb", "e2e2emb", "e2e1embP", "e2e1emb_kvm", "e2e1emb_rnd", *R42L_ARMS,
            *R44A_ARMS, "e2e1emb_aice", "e2e1k_emb", "e2e1e_emb"}
MASK_ARMS = {"e2e1emb_kvm": "kv", "e2e1emb_rnd": "rnd", "e2e1_kvm": "kv",
             "e2e2_kvm": "kv"}
# Arms introduced by R4-3; their presence switches on r43_sanity.
R43_ARMS = {"e2e1e_emb", "e2e1k_emb", "e2e2_kvm"}
# The four arms the E/K manipulation metric is built from, and the half -> arm map it
# reports. All four are +emb sequential arms, so one init and one frozen code serve them
# all.
EK_ANCHOR, EK_REF = "e2e1emb", "e2e2emb"
EK_HALVES = {"E": "e2e1e_emb", "K": "e2e1k_emb"}
EK_ARMS = (EK_ANCHOR, EK_HALVES["E"], EK_HALVES["K"], EK_REF)
# the pre-registered gate: the median share over seeds 0-2, at init, on the run's own
# model size, first 8 sequences of each seed's first training batch
EK_SHARE_SEEDS = (0, 1, 2)
EK_SHARE_NSEQ = 8
EK_SHARE_MIN = 0.05
# cells that end with R4-2's fixed write ablation: R4-2's three arms plus every R4-2L arm.
# e2e1embP and e2e2emb are deliberately not in it, as in R4-2.
ABLATION_ARMS = {"e2e1emb", "e2e1emb_kvm", "e2e1emb_rnd", *R42L_ARMS, "e2e1emb_edge", "e2e1emb_aice",
                 *R43_ARMS}
# In a run that contains an R4-3 arm every cell ends with the write ablation, so the
# plain and +emb second-order references are ablated too; the plain arms take the plain
# input path inside r42_ablation, which dispatches on frozen_emb.
R43_ABLATION_EXTRA = {"e2e2emb", "e2e2"}
ARM_TASKS.update({"e2e1emb_aice": "echo", "e2e1_aice": "icl_mc"})
MASK_SEED = 4242
ECHO_MODES = {"echo", "echo_ctrl", "echo_off", "echo_ctrl_nor"}
# tasks whose cells get the R4 probes, the r4_flags and the fixed write ablation
PROBE_MODES = ("echo", "echo_ctrl")
TTT_ARMS = SEQUENTIAL_ARMS + tuple(JACOBI_ARMS)
TRAIN_SEED = 20240101
EVAL_SEED = 991
PROBE_SEED = 271828
H1_RATIO = 0.8

DEFAULTS: dict[str, Any] = {
    "process_name": "RQS-ttt-01",
    "device": "auto",
    "max_minutes": 110.0,
    "tasks": ["icl_mc", "echo", "echo_ctrl"],
    "arms": ["fa", "swa", "naive", "e2e2", "e2e1", "jac1", "jac2"],
    "seeds": [0],
    "extra_jobs": [],
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "backbone_mult": 4,
    "fast_mult": 4,
    "window": 64,
    "chunk": 64,
    "n_vocab": 32,
    "alpha": 0.1,
    "T": 1024,
    "n_eval": 512,
    "n_val": 128,
    "n_quick_eval": 64,
    "eval_batch": 32,
    "n_bins": 16,
    "g_clip": 10.0,
    "pilot_etas": [0.3, 1.0, 3.0, 10.0],
    "pilot_steps": 300,
    "pilot": True,
    "eta_lr": 0.01,
    "gamma_lr": 0.01,
    "naive_etas": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "steps": 4000,
    "batch": 32,
    "lr": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "weight_decay": 0.01,
    "warmup": 200,
    "lr_floor": 0.1,
    "grad_clip": 1.0,
    "quick_every": 500,
    "probe_every": 0,
    "n_probe": 32,
    "workers": 1,
    "mps_memory_fraction": None,
    "worker_devices": [],
    "tf32": True,
    "checkpoint_steps": [],
    "checkpoint_steps_so": [],
    "checkpoint_dense": [],
    "checkpoint_dense_arms": [],
    "checkpoint_dense_max_seed": None,
    "checkpoint_arm_steps": [],
    "late_frac": 0.25,
    "n_keys": 16,
}


@dataclass
class Settings:
    """Resolved ``[ttt]`` settings for one task: defaults <- table <- task table."""

    process_name: str = "RQS-ttt-01"
    device: str = "auto"
    max_minutes: float = 110.0
    tasks: list[str] = field(default_factory=lambda: list(DEFAULTS["tasks"]))
    arms: list[str] = field(default_factory=lambda: list(DEFAULTS["arms"]))
    seeds: list[int] = field(default_factory=lambda: list(DEFAULTS["seeds"]))
    extra_jobs: list[dict] = field(default_factory=list)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    backbone_mult: int = 4
    fast_mult: int = 4
    window: int = 64
    chunk: int = 64
    n_vocab: int = 32
    alpha: float = 0.1
    T: int = 1024
    n_eval: int = 512
    n_val: int = 128
    n_quick_eval: int = 64
    eval_batch: int = 32
    n_bins: int = 16
    g_clip: float = 10.0
    pilot_etas: list[float] = field(default_factory=lambda: list(DEFAULTS["pilot_etas"]))
    pilot_steps: int = 300
    pilot: bool = True
    eta_lr: float = 0.01
    gamma_lr: float = 0.01
    kv_eta_lr: float = 0.01
    kv_eta10_lr: float = 0.1
    kamin_switch: int = 500
    copy_noise_q: list[float] = field(default_factory=lambda: [.99, .97, .95, .90])
    rf_probability: float = .05
    d1_n_eval: int = 256
    naive_etas: list[float] = field(default_factory=lambda: list(DEFAULTS["naive_etas"]))
    steps: int = 4000
    batch: int = 32
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    warmup: int = 200
    lr_floor: float = 0.1
    grad_clip: float = 1.0
    # length of the cosine lr schedule; 0 means the run's own step count. A shorter
    # run with lr_total = 3000 follows the first steps of a 3000-step schedule.
    lr_total: int = 0
    # skip the optimizer step when the outer-gradient norm exceeds this multiple of
    # the median of the previous 50 finite post-warmup norms; 0 disables it
    spike_skip: float = 0.0
    # Jacobi arms also get a quick eval under their own parallel forward (R3); the rows
    # [step, query_ce, query_acc] go to ``quick_par``, apart from the quick-log layout
    quick_par: bool = False
    quick_every: int = 500
    probe_every: int = 0
    n_probe: int = 32
    workers: int = 1
    mps_memory_fraction: float | None = None
    worker_devices: list[str] = field(default_factory=list)
    # matmul/cuDNN TF32. True (the default) leaves TF32 allowed; False is asserted
    # and logged per worker so the in-run S5 check holds at 1e-4.
    tf32: bool = True
    checkpoint_steps: list[int] = field(default_factory=list)
    checkpoint_steps_so: list[int] = field(default_factory=list)
    # dense window: steps added for checkpoint_dense_arms when the cell's
    # seed is at most checkpoint_dense_max_seed, plus per-arm extras as
    # [{arm = ..., steps = [...]}] (a list of tables, so it stays a scalar setting)
    checkpoint_dense: list[int] = field(default_factory=list)
    checkpoint_dense_arms: list[str] = field(default_factory=list)
    checkpoint_dense_max_seed: int | None = None
    checkpoint_arm_steps: list[dict] = field(default_factory=list)
    late_frac: float = 0.25
    n_keys: int = 16
    prefix_min: int = 64
    prefix_max: int = 192
    tables: dict = field(default_factory=dict)

    def task_table(self, name: str) -> dict:
        """The ``[ttt.<name>]`` sub-table, empty when the task has none."""
        return self.tables.get(name, {})


def _settings(cfg: dict, task: str | None = None) -> Settings:
    """Merge defaults, the ``[ttt]`` table and ``[ttt.<task>]`` into settings."""
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
    tables = {k: dict(v) for k, v in cfg.items() if isinstance(v, dict)}
    if task is not None:
        merged.update(tables.get(task, {}))
    known = {f.name for f in dataclasses.fields(Settings)}
    return Settings(tables=tables, **{k: v for k, v in merged.items() if k in known})


def _mark_process(name: str) -> None:
    """Rename the process so ``ps``/``pgrep`` find the run, and print the name."""
    os.environ["PROCESS_NAME"] = name
    try:
        import setproctitle

        setproctitle.setproctitle(name)
    except Exception:  # pragma: no cover - naming is best effort
        pass
    print(f"ttt | process_name={name}", flush=True)


def _device(requested: str) -> torch.device:
    """``cuda`` > ``mps`` > ``cpu``, or the backend named in the config."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_tf32(enabled: bool, worker: str = "main") -> dict:
    """Set, assert and log the TF32 flags before any device allocation.

    The key is ``tf32``: ``True`` (the default) leaves TF32 allowed, ``False``
    forces both the matmul and the cuDNN flag off plus the highest float32 matmul
    precision, so the in-run S5 identity check is not floored by TF32 rounding.
    The observed state is returned and logged per worker.
    """
    enabled = bool(enabled)
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    if not enabled:
        torch.set_float32_matmul_precision("highest")
    matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn = bool(torch.backends.cudnn.allow_tf32)
    assert matmul == enabled and cudnn == enabled, f"TF32 state {matmul}/{cudnn} != {enabled}"
    print(f"TF32 worker={worker} matmul={matmul} cudnn={cudnn}", flush=True)
    return dict(tf32=enabled, tf32_matmul=matmul, tf32_cudnn=cudnn,
                float32_matmul_precision=torch.get_float32_matmul_precision())


def _reset_peak(dev: torch.device) -> None:
    """Zero the backend's peak-memory counter before an arm is timed."""
    try:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
    except Exception:  # pragma: no cover - backend dependent
        pass


def _sync(dev: torch.device) -> None:
    """Block until queued device work is done, so wall-clock step timings are real."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _peak_mem(dev: torch.device) -> float:
    """Peak allocated device memory in MiB, 0 when the backend cannot report it."""
    try:
        if dev.type == "cuda":
            return round(float(torch.cuda.max_memory_allocated(dev)) / 2**20, 1)
        if dev.type == "mps":
            return round(float(torch.mps.current_allocated_memory()) / 2**20, 1)
    except Exception:  # pragma: no cover - backend dependent
        pass
    return 0.0


def _tf32_state(tf32: bool) -> dict:
    """Set the CUDA TF32 flags to ``tf32`` and return the state that took effect.

    With ``tf32`` false the float32 matmul precision is also pinned to
    "highest", because that precision setting is what actually selects TF32 for
    matmuls and would otherwise re-enable it. Call this before the first CUDA
    allocation, then log the result: the pre-registered in-run S5 and the R4-3
    mask identities are only meaningful in fp32.
    """
    if not tf32:
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    state = dict(matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                 cudnn=bool(torch.backends.cudnn.allow_tf32),
                 precision=torch.get_float32_matmul_precision())
    if state["matmul"] is not bool(tf32) or state["cudnn"] is not bool(tf32):
        raise RuntimeError(f"TF32 flags did not take: {state}, tf32={tf32}")
    return state


def _log_tf32(tf32: bool, worker: str = "main") -> dict:
    """Apply the configured TF32 state in this process and log it."""
    state = _tf32_state(tf32)
    print(f"TF32 worker={worker} matmul={state['matmul']} cudnn={state['cudnn']} "
          f"precision={state['precision']} configured={bool(tf32)}", flush=True)
    return state


# --------------------------------------------------------------------------- data


def mc_batch(rng: np.random.Generator, n_seq: int, T: int, V: int, alpha: float) -> np.ndarray:
    """``n_seq`` Markov chains of length ``T + 1``.

    Each chain has its own transition matrix whose rows are ``Dirichlet(alpha *
    1_V)``; the first token is uniform and the rest follows the chain, so all
    ``T`` input positions are supervised by the ``T + 1`` tokens.
    """
    prior = np.full(V, alpha)
    x = np.empty((n_seq, T + 1), dtype=np.int64)
    x[:, 0] = rng.integers(V, size=n_seq)
    u = rng.random((n_seq, T))
    cdf = rng.dirichlet(prior, size=(n_seq, V)).cumsum(axis=2)
    rows = np.arange(n_seq)
    for t in range(T):
        # inverse-CDF step against the cumulative masses of the current state's row:
        # the number of masses below u is the next token (clamped, since a float sum
        # can leave the last cumulative mass a hair below 1)
        cum = cdf[rows, x[:, t], :]
        x[:, t + 1] = np.minimum((u[:, t, None] > cum).sum(axis=1), V - 1)
    return x


@dataclass(frozen=True)
class EchoVocab:
    """Token ids of the echo tasks: two markers, keys, values, fillers.

    With ``n_keys`` keys there are ``2 + 3 * n_keys`` tokens: ``W`` and ``Q``,
    then the keys, then the values, then as many filler tokens.
    """

    n_keys: int = 16

    @property
    def size(self) -> int:
        """Vocabulary size."""
        return 2 + 3 * self.n_keys

    @property
    def write(self) -> int:
        """Id of the write marker."""
        return 0

    @property
    def query(self) -> int:
        """Id of the query marker."""
        return 1

    @property
    def key(self) -> np.ndarray:
        """Key token ids."""
        return np.arange(2, 2 + self.n_keys)

    @property
    def value(self) -> np.ndarray:
        """Value token ids."""
        return np.arange(2 + self.n_keys, 2 + 2 * self.n_keys)

    @property
    def filler(self) -> np.ndarray:
        """Filler token ids."""
        return np.arange(2 + 2 * self.n_keys, 2 + 3 * self.n_keys)


def echo_batch(rng: np.random.Generator, n_seq: int, T: int, window: int, mode: str,
               n_keys: int = 16, n_layers: int = 1, prefix_min: int = 64,
               prefix_max: int = 192) -> tuple[np.ndarray, dict]:
    """Write / filler / query sequences for the four echo modes.

    Every sequence samples its own map ``f`` from keys to values (uniform, with
    replacement) and writes each key once, in random order, as the four-token
    event ``W f(k) k f(k)`` (``echo``) or ``W r k f(k)`` with ``r`` a uniformly
    random value token (``echo_ctrl``). ``echo_off`` offsets the write phase by
    a random filler prefix; ``echo_ctrl_nor`` writes a filler in place of ``r``.
    A run of at least ``n_layers * (window - 1) + n_keys`` filler tokens separates
    the write phase from the query phase, where
    the keys reappear in a new random order as ``Q k f(k)``. Stacked sliding-window
    attention sees ``n_layers * (window - 1) + 1`` tokens back, so a filler of
    ``window + n_keys`` would leave the first queries inside the receptive field
    and copyable by attention alone. In ``echo`` the value after ``k`` is copyable
    two tokens back, so attention already knows it and the inner gradient
    may be small; in ``echo_ctrl`` the value is not input-side copyable.

    Returns tokens of length ``T + 1`` and metadata with the position of the
    first query of every key (the position whose target is that query's value).
    """
    if mode not in ECHO_MODES:
        raise ValueError(f"unknown echo mode {mode!r}")
    if prefix_min < 0 or prefix_max < prefix_min:
        raise ValueError("invalid echo prefix range")
    vocab = EchoVocab(n_keys)
    write_len, filler_len = 4 * n_keys, n_layers * (window - 1) + n_keys
    query_len = 3 * n_keys + 4 * (n_keys - 1)
    if (prefix_max if mode == "echo_off" else 0) + write_len + filler_len + query_len + 2 > T + 1:
        raise ValueError(f"T={T} too short for the echo layout of mode {mode}")
    tokens = np.empty((n_seq, T + 1), dtype=np.int64)
    query_pos = np.empty((n_seq, n_keys), dtype=np.int64)
    query_tok = np.empty((n_seq, n_keys), dtype=np.int64)
    write_pos = np.empty((n_seq, n_keys), dtype=np.int64)
    write_start = np.zeros(n_seq, dtype=np.int64)
    for i in range(n_seq):
        row = tokens[i]
        row[:] = rng.choice(vocab.filler, size=T + 1)
        f = rng.choice(vocab.value, size=n_keys)
        pos = int(rng.integers(prefix_min, prefix_max + 1)) if mode == "echo_off" else 0
        write_start[i] = pos
        for k in rng.permutation(n_keys):
            row[pos] = vocab.write
            row[pos + 1] = (f[k] if mode in ("echo", "echo_off") else
                            rng.choice(vocab.filler if mode == "echo_ctrl_nor" else vocab.value))
            row[pos + 2] = vocab.key[k]
            row[pos + 3] = f[k]
            write_pos[i, k] = pos + 2
            pos += 4
        pos += filler_len
        for j, k in enumerate(rng.permutation(n_keys)):
            row[pos] = vocab.query
            row[pos + 1] = vocab.key[k]
            row[pos + 2] = f[k]
            query_pos[i, k] = pos + 1
            query_tok[i, k] = f[k]
            pos += 3 + (int(rng.integers(1, 5)) if j < n_keys - 1 else 0)
    meta = dict(query_pos=query_pos, query_tok=query_tok, write_pos=write_pos,
                write_start=write_start, mode=mode)
    return tokens, meta


def bayes_ce(tokens: np.ndarray, V: int, alpha: float, window: int | None = None
             ) -> tuple[np.ndarray, np.ndarray | None]:
    """Cross-entropy of the Dirichlet posterior predictive on Markov chains.

    Position ``t`` predicts ``tokens[:, t + 1]`` with
    ``(n_ij + alpha) / (n_i + V * alpha)``, where ``n_ij`` counts the transitions
    ``i -> j`` seen inside ``tokens[:, :t + 1]`` with ``i = tokens[:, t]``. With
    ``window`` given, only the transitions fully inside the last ``window``
    tokens (``x[t - window + 1..t]``) are counted, which bounds a pure
    sliding-window model. Returns the full-context curve and the windowed one,
    both of shape ``(n_seq, T)``.
    """
    n_seq, length = tokens.shape
    rows = np.arange(n_seq)
    cols = np.arange(V)
    full = np.zeros((n_seq, V * V), dtype=np.int32)
    win = np.zeros((n_seq, V * V), dtype=np.int32) if window else None
    ce_full = np.empty((n_seq, length - 1))
    ce_win = np.empty((n_seq, length - 1)) if window else None
    for t in range(length - 1):
        src, dst = tokens[:, t], tokens[:, t + 1]
        edge = (src * V)[:, None]
        counts = full[rows[:, None], edge + cols[None, :]]
        p = (counts + alpha) / (counts.sum(1, keepdims=True) + V * alpha)
        ce_full[:, t] = -np.log(p[rows, dst])
        full[rows, src * V + dst] += 1
        if window:
            counts = win[rows[:, None], edge + cols[None, :]]
            p = (counts + alpha) / (counts.sum(1, keepdims=True) + V * alpha)
            ce_win[:, t] = -np.log(p[rows, dst])
            win[rows, src * V + dst] += 1
            drop = t - window + 1
            if drop >= 0:
                win[rows, tokens[:, drop] * V + tokens[:, drop + 1]] -= 1
    return ce_full, ce_win


@dataclass
class Task:
    """One benchmark: token stream, targets, and where the query metric lives."""

    name: str
    V: int
    T: int
    window: int
    chunk: int
    steps: int
    n_eval: int
    n_val: int
    mode: str = "mc"
    alpha: float = 0.1
    n_keys: int = 16
    n_layers: int = 1
    prefix_min: int = 64
    prefix_max: int = 192

    def raw(self, rng: np.random.Generator, n_seq: int) -> tuple[np.ndarray, dict]:
        """Fresh raw sequences of length ``T + 1`` plus their metadata."""
        if self.mode == "mc":
            return mc_batch(rng, n_seq, self.T, self.V, self.alpha), {}
        return echo_batch(rng, n_seq, self.T, self.window, self.mode, self.n_keys,
                          self.n_layers, self.prefix_min, self.prefix_max)

    @property
    def receptive_field(self) -> int:
        """Tokens of context a stacked ``n_layers`` sliding-window model can see."""
        return self.n_layers * (self.window - 1) + 1

    def tensors(self, tokens: np.ndarray, meta: dict, dev: torch.device
                ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Split raw sequences into model inputs and next-token targets."""
        idx = torch.from_numpy(tokens[:, : self.T]).to(dev)
        y = torch.from_numpy(tokens[:, 1: self.T + 1]).to(dev)
        return idx, y, meta


def _slice_meta(meta: dict, lo: int, hi: int) -> dict:
    """Restrict echo metadata to sequences ``[lo, hi)``."""
    out = dict(meta)
    for key in ("query_pos", "query_tok", "write_pos", "write_start", "corrupted", "corruption_pos"):
        if key in meta:
            out[key] = meta[key][lo:hi]
    return out


def fixed_set(task: Task, st: Settings, which: str) -> tuple[np.ndarray, dict]:
    """Fixed evaluation split: eval, disjoint val, or quick-eval set.

    The stream is seeded by ``EVAL_SEED`` alone, never by the training seed, so
    every arm and every seed is scored on exactly the same sequences. A
    seed-dependent eval set would make the shared ``fa`` and Bayes references
    unusable for the seed-1 extras and turn the paired arm comparisons into
    between-sample comparisons. Training streams stay seed-dependent.
    """
    n = {"eval": task.n_eval, "val": task.n_val,
         "quick": min(st.n_quick_eval, task.n_eval)}[which]
    rng = np.random.default_rng([EVAL_SEED, {"eval": 0, "val": 1, "quick": 2}[which]])
    return task.raw(rng, n)


def copy_edge_mask(write_pos, T, device=torch.device("cpu")):
    """True blocks t -> t-1. Reusable at training time; no labels required."""
    wp = torch.as_tensor(write_pos, device=device, dtype=torch.long)
    if wp.ndim != 2 or bool(((wp < 1) | (wp >= T)).any()):
        raise ValueError("copy destinations must be B x K positions in [1,T)")
    blocked = torch.zeros(len(wp), 1, T, T, device=device, dtype=torch.bool)
    rows = torch.arange(len(wp), device=device)[:, None]
    blocked[rows, 0, wp, wp - 1] = True
    return blocked


def arm_edge_mask(model, arm, idx, meta):
    """The a5 mask, or Kamin-control's stage-1 mask, in every backbone layer."""
    active = arm == "e2e1emb_edge" or (arm == "e2e1emb_kaminctl" and
                                      model.inner_multiplier == 0)
    if not active:
        return None
    if meta is None or "write_pos" not in meta:
        raise ValueError(f"{arm} needs write_pos metadata for the copy-edge mask")
    return copy_edge_mask(meta["write_pos"], idx.shape[1], idx.device)


def regime_etas(model, eta_base):
    """Kamin stage 1 has zero effective eta, with no graph to rho."""
    etas = model.etas(eta_base)
    return ([e.new_zeros(()) for e in etas] if model.inner_multiplier == 0 else etas)


def set_kamin_regime(model, arm, completed_steps, st):
    if arm in KAMIN_ARMS:
        model.inner_multiplier = float(completed_steps >= st.kamin_switch)


def corrupt_echo(tokens, meta, arm, seed, step, *, rf_probability=.05):
    """Per write event, independently with probability 1−q, replace the copy-source
    token at pos+1 by a value drawn uniformly from the other 15 values (r ≠ v).
    The kv label (next token of write_pos) stays f(k). Consequence (accepted):
    the W-marker position's label is the corrupted r. For 16 values the channel
    eigenvalue is lambda = (16q-1)/15. Tiny tests use the analogous K-value channel.

    rf95: per write event, independently with probability 0.05, replace one
    uniformly random filler token strictly between the end of the write phase
    and the first query by a uniformly random value token. Its event flags mark
    filler-corruption draws, not corruption of that key's copy source.
    Filler positions are sampled without replacement within each sequence, so
    the number of altered tokens equals the number of independent event flags.
    A separate RNG seeded by cell seed and step never advances the clean stream.
    """
    if arm not in NOISE_ARMS:
        return tokens, meta
    wp = meta["write_pos"]
    n_keys = wp.shape[1]
    if n_keys < 2:
        raise ValueError("copy corruption needs at least two values")
    rng = np.random.default_rng([seed, CORRUPTION_SEED, step])
    probability = 1 - COPY_Q[arm] if arm in COPY_Q else rf_probability
    flags = rng.random(wp.shape) < probability
    out = tokens.copy()
    positions = np.full(wp.shape, -1, dtype=np.int64)
    rows, keys = np.nonzero(flags)
    value0 = 2 + n_keys
    if arm in COPY_Q:
        positions[rows, keys] = wp[rows, keys] - 1
        original = tokens[rows, wp[rows, keys] + 1]
        out[rows, positions[rows, keys]] = value0 + (
            original - value0 + rng.integers(1, n_keys, len(rows))) % n_keys
    else:
        for row in np.unique(rows):
            selected_keys = np.flatnonzero(flags[row])
            lo, hi = int(wp[row].max()) + 2, int(meta["query_pos"][row].min()) - 1
            if hi-lo < len(selected_keys):
                raise ValueError("rf95 needs enough distinct filler tokens before the first query")
            pos = rng.choice(np.arange(lo, hi), len(selected_keys), replace=False)
            positions[row, selected_keys] = pos
            out[row, pos] = rng.integers(value0, value0 + n_keys, len(selected_keys))
    return out, dict(meta, corrupted=flags, corruption_pos=positions)


def corrupted_batches(task, st, arm, dev, which):
    raw, meta = fixed_set(task, st, which)
    raw, meta = corrupt_echo(raw, meta, arm, CORRUPTION_EVAL_SEED,
                             {"eval": 0, "quick": 1}[which], rf_probability=st.rf_probability)
    return eval_batches(task, raw, meta, dev, st.eval_batch), meta


def corruption_recall(parts, meta):
    """Query recall grouped by the corresponding key's write-event flag."""
    preds = torch.cat([p for _, p in parts]).cpu().numpy()
    rows = np.arange(len(preds))[:, None]
    hit = preds[rows, meta["query_pos"]] == meta["query_tok"]
    flags = meta["corrupted"]
    out = {}
    for name, selected in (("corrupted", flags), ("clean", ~flags)):
        count = int(selected.sum())
        out[f"n_{name}_keys"] = count
        out[f"acc_{name}_keys"] = float(hit[selected].mean()) if count else None
    return out


@torch.no_grad()
def kv_w0_stats(model, task, data, arm):
    """Mean target probability at W0, also split by corruption event when present."""
    values, flags = [], []
    for idx, y, meta in data:
        h = model.backbone(idx, edge_mask=arm_edge_mask(model, arm, idx, meta))
        z = model.read(model.fast_block(h, idx, model.slow()))
        wp = torch.as_tensor(meta["write_pos"], device=idx.device)
        pv = (-token_ce(z, y)).exp().gather(1, wp).flatten().cpu().numpy()
        values.extend(pv.tolist())
        if "corrupted" in meta:
            flags.extend(meta["corrupted"].flatten().tolist())
    out = dict(p_kv_mean=float(np.mean(values)))
    if flags:
        v, f = np.asarray(values), np.asarray(flags, dtype=bool)
        for name, selected in (("corrupted", f), ("clean", ~f)):
            out[f"p_kv_{name}"] = float(v[selected].mean()) if selected.any() else None
            out[f"n_{name}_events"] = int(selected.sum())
    return out


@torch.no_grad()
def d1_probe(model, task, st, arm, dev):
    """P0.1 reassign layout at W0: write-1 vs write-2 (descriptive only)."""
    from .exp_p01 import copy_competitor, gather, reassign_set

    conditions, layout = reassign_set(st.d1_n_eval, task.T, task.window, task.n_layers,
                                      task.n_keys, task.chunk, CORRUPTION_EVAL_SEED)
    raw, meta = conditions["reassign"]
    rtask = dataclasses.replace(task, T=layout["T"])
    pooled = {name: {k: [] for k in ("p_v", "p_v_edge_on", "p_v_edge_off",
                                    "competitor_norm", "competitor_target_logit")}
              for name in ("write1", "write2")}
    for idx, y, m in eval_batches(rtask, raw, meta, dev, st.eval_batch):
        c, on, off = copy_competitor(model, idx, m["write_pos"])
        primary = off if arm == "e2e1emb_edge" else on
        for j, name in enumerate(("write1", "write2")):
            wp = m["write_pos"][:, j*task.n_keys:(j+1)*task.n_keys]
            target = gather(y, wp)
            comp = gather(c, wp)
            for key, val in (("p_v", gather((-token_ce(primary, y)).exp(), wp)),
                             ("p_v_edge_on", gather((-token_ce(on, y)).exp(), wp)),
                             ("p_v_edge_off", gather((-token_ce(off, y)).exp(), wp)),
                             ("competitor_norm", comp.norm(dim=-1)),
                             ("competitor_target_logit", comp.gather(-1, target[..., None]).squeeze(-1))):
                pooled[name][key].extend(val.flatten().tolist())
    return dict(layout=layout, n_seq=len(raw), seed=CORRUPTION_EVAL_SEED,
                regime="edge_off" if arm == "e2e1emb_edge" else "edge_on",
                **{name: {key: float(np.mean(v)) for key, v in row.items()}
                   for name, row in pooled.items()})


def echo_references(meta: dict, n_keys: int) -> dict:
    """Exact unbound-value predictive CE and tie-randomised accuracy per sequence."""
    values, positions = np.asarray(meta["query_tok"]), np.asarray(meta["query_pos"])
    n_seq = len(values)
    scores = {key: {"ce": [], "acc": []} for key in ("chance", "occ", "occ_minus")}
    for row, pos in zip(values, positions):
        order = np.argsort(pos)
        counts = np.bincount(row, minlength=int(values.max()) + 1)
        remaining = counts.copy()
        for j, k in enumerate(order):
            v = row[k]
            for key, distribution in (("occ", counts / n_keys),
                                      ("occ_minus", remaining / (n_keys - j))):
                p = distribution[v]
                scores[key]["ce"].append(-math.log(p))
                scores[key]["acc"].append(float(p == distribution.max()) /
                                           np.count_nonzero(distribution == distribution.max()))
            scores["chance"]["ce"].append(math.log(n_keys))
            scores["chance"]["acc"].append(1 / n_keys)
            remaining[v] -= 1
    out = {}
    for key, entries in scores.items():
        ce = np.asarray(entries["ce"]).reshape(n_seq, n_keys).mean(1)
        acc = np.asarray(entries["acc"]).reshape(n_seq, n_keys).mean(1)
        out[key] = dict(ce=float(ce.mean()), acc=float(acc.mean()),
                        ce_se=float(ce.std(ddof=1) / math.sqrt(n_seq)) if n_seq > 1 else 0.0,
                        acc_se=float(acc.std(ddof=1) / math.sqrt(n_seq)) if n_seq > 1 else 0.0)
        if key == "occ_minus":
            out[key]["acc_per_seq"] = acc.tolist()
    return out


def outer_keep(meta: dict, shape: tuple[int, int], mask: str,
               rng: np.random.Generator | None, dev: torch.device) -> torch.Tensor | None:
    """R4-2 oracle mask on CE indices (the target at write_pos is the value)."""
    if mask == "none":
        return None
    wp = np.asarray(meta["write_pos"])
    if mask == "kv":
        positions = wp
    elif mask == "rnd":
        assert rng is not None
        candidates = np.concatenate((wp - 1, wp + 1), axis=1)
        positions = np.stack([rng.choice(row, size=wp.shape[1], replace=False)
                              for row in candidates])
    else:
        raise ValueError(f"unknown outer mask {mask}")
    if (positions.shape != (shape[0], wp.shape[1]) or np.any(positions < 0) or
            np.any(positions >= shape[1])):
        raise ValueError("mask positions outside the training batch")
    keep = torch.ones(shape, dtype=torch.float32, device=dev)
    keep.scatter_(1, torch.as_tensor(positions, device=dev), 0.)
    return keep


@torch.no_grad()
def r4_probe(model: TTTModel, task: Task, data: list, eta_base: float,
             g_clip: float, *, arm: str,
             capture: dict | None = None) -> dict:
    """Address geometry and the clipped kv-only linear W2 readout at slow W0.

    ``arm`` is the cell's arm: e is then ``d(inner token loss)/d(out)`` and the
    clip scale the arm's own chunk-0 inner gradient, so every write
    field describes the write the model actually makes. ``capture``, when given,
    receives a detached copy of that chunk-0 inner gradient (sanity L8).
    """
    from . import exp_kag as kag
    arm = B9_BASE.get(arm, arm)

    parts: dict[str, list] = {k: [] for k in (
        "addr_a", "addr_h", "addr_margin", "pr_kv", "common_kv", "amp_ratio",
        "err_ratio", "kv_write", "Wv_write", "lin_kv", "emb_ln_norm",
        "ce_kv", "ce_vk", "ce_vW", "p_kv", "e_kv_abs", "p_wv", "e_wv_abs",
        "kv_write_eff", "Wv_write_eff", "e_kv_ce_abs", "e_vW_abs", "vW_write_eff",
        "upd_W1", "upd_b1", "upd_W2", "upd_b2",
        "emb_e_norm", "emb_dot", "fast_input_norm", "code_norm", "emb_addr")}
    max_diff = 0.0
    direct_hits: list[float] = []
    c0_clipped: list[float] = []
    c0_norm: list[float] = []
    for idx, y, meta in data:
        h = model.backbone(idx, edge_mask=arm_edge_mask(model, arm, idx, meta)).detach()
        code = kag._code(model, h, idx=idx)
        wp = meta["write_pos"]
        write, query, wv = (kag._gather(code, pos) for pos in (wp, meta["query_pos"], wp - 2))
        ln = model.head.ln(h)
        hw, hq = kag._gather(ln, wp), kag._gather(ln, meta["query_pos"])
        n = write.shape[1]
        dot = torch.bmm(query, write.transpose(1, 2))
        hdot = torch.bmm(hq, hw.transpose(1, 2))
        target = torch.arange(n, device=idx.device)[None]
        parts["addr_a"].extend((dot.argmax(-1) == target).float().flatten().tolist())
        parts["addr_h"].extend((hdot.argmax(-1) == target).float().flatten().tolist())
        matched = dot.diagonal(dim1=1, dim2=2)
        wrong = dot.masked_fill(torch.eye(n, device=idx.device, dtype=torch.bool)[None], -torch.inf)
        parts["addr_margin"].extend((matched - wrong.max(-1).values).flatten().tolist())
        parts.setdefault("abs_dot", []).extend(dot.abs().flatten().tolist())
        centred = write - write.mean(1, keepdim=True)
        sv2 = torch.linalg.svdvals(centred).square()
        parts["pr_kv"].extend((sv2.sum(-1).square() / sv2.square().sum(-1).clamp_min(1e-24)).tolist())
        parts["common_kv"].extend((write.mean(1).square().sum(-1) /
                                    write.square().sum(-1).mean(-1).clamp_min(1e-24)).tolist())
        parts["amp_ratio"].extend(write.norm(dim=-1).flatten().tolist())
        parts.setdefault("wv_amp", []).extend(wv.norm(dim=-1).flatten().tolist())
        sel = lin_selector(arm, wp, len(idx), task.T, idx.device)
        with torch.enable_grad():
            weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
            out0 = model.fast_block(h[:, :task.chunk], idx[:, :task.chunk], weights)
            logits0 = model.read(out0)
            ce0 = token_ce(logits0, y[:, :task.chunk])
            inner0 = inner_token_loss(arm_inner_loss(arm), logits0, y[:, :task.chunk], ce=ce0,
                                     sel=None if sel is None else sel[:, :task.chunk],
                                     reference=logits0.detach() if arm in AICE_ARMS else None)
            loss0 = inner0.mean(1)
            grads = torch.autograd.grad(loss0.sum(), weights, retain_graph=True)
            clipped_grads, norm, clipped = _clip_fast(list(grads), g_clip)
            scale = torch.where(norm > g_clip, g_clip / norm, torch.ones_like(norm))
            delta = torch.autograd.grad(inner0.sum(), out0, retain_graph=True)[0]
            dce = torch.autograd.grad(ce0.sum(), out0)[0]
        if capture is not None:
            capture["inner_grads"] = [g.detach().clone() for g in grads]
        ek, ew = kag._gather(delta, wp), kag._gather(delta, wp - 2)
        ev, cv = kag._gather(delta, wp + 1), kag._gather(code, wp + 1)
        prob = (-ce0).exp()
        for name, pos in (("ce_kv", wp), ("ce_vk", wp - 1), ("ce_vW", wp + 1)):
            parts[name].extend(kag._gather(ce0, pos).flatten().tolist())
        parts["p_kv"].extend(kag._gather(prob, wp).flatten().tolist())
        parts["p_wv"].extend(kag._gather(prob, wp - 2).flatten().tolist())
        parts["e_kv_abs"].extend(ek.norm(dim=-1).flatten().tolist())
        parts["e_wv_abs"].extend(ew.norm(dim=-1).flatten().tolist())
        parts["e_kv_ce_abs"].extend(kag._gather(dce, wp).norm(dim=-1).flatten().tolist())
        parts["e_vW_abs"].extend(ev.norm(dim=-1).flatten().tolist())
        etas = [e.detach() for e in regime_etas(model, eta_base)]
        eta = (model.eta_kvW2(eta_base).detach() if arm in KVETA_ARMS else etas[2])
        parts["err_ratio"].extend(ew.norm(dim=-1).flatten().tolist())
        parts.setdefault("kv_err", []).extend(ek.norm(dim=-1).flatten().tolist())
        parts["kv_write"].extend((eta.abs() * ek.norm(dim=-1) * write.norm(dim=-1)).flatten().tolist())
        parts["Wv_write"].extend((etas[2].abs() * ew.norm(dim=-1) * wv.norm(dim=-1)).flatten().tolist())
        parts["kv_write_eff"].extend((eta.abs() * ek.norm(dim=-1) * write.norm(dim=-1) *
                                      scale[:, None]).flatten().tolist())
        parts["Wv_write_eff"].extend((etas[2].abs() * ew.norm(dim=-1) * wv.norm(dim=-1) *
                                      scale[:, None]).flatten().tolist())
        parts["vW_write_eff"].extend((etas[2].abs() * ev.norm(dim=-1) * cv.norm(dim=-1) *
                                      scale[:, None]).flatten().tolist())
        c0_clipped.extend(clipped.float().tolist())
        c0_norm.extend(norm.tolist())
        for name, g, step in zip(("upd_W1", "upd_b1", "upd_W2", "upd_b2"),
                                  clipped_grads, etas):
            if name == "upd_W2" and arm in KVETA_ARMS:
                # delta is d(sum token losses)/d(out), so include the chunk mean.
                gkv = torch.einsum("bkd,bkm->bdm", ek, write) / task.chunk
                update_g = step * g + (eta - step) * scale[:, None, None] * gkv
                parts[name].extend(update_g.reshape(g.shape[0], -1).norm(dim=-1).tolist())
            else:
                # Preserve the parent probe's arithmetic for every existing arm.
                parts[name].extend((step.abs() * g.reshape(g.shape[0], -1).norm(dim=-1)).tolist())
        terms = kag._linear_terms(model, h, meta, delta, eta, scale, task.chunk, idx)
        linear = model.read(terms["o0"] + terms["factor"] * terms["memory"])
        update = torch.einsum("bkd,bkm->bdm", ek, write) / task.chunk
        W2 = model.slow()[2].unsqueeze(0) - eta * scale[:, None, None] * update
        qh = kag._gather(h, meta["query_pos"])
        # Query codes are gathered from the complete sequence, so Q at a chunk
        # boundary is handled identically in the streaming path and this probe.
        direct = model.read(qh + torch.einsum("bkm,bdm->bkd", query, W2) + model.slow()[3])
        max_diff = max(max_diff, float((linear - direct).abs().max()))
        tgt = torch.as_tensor(meta["query_tok"], device=idx.device)
        parts["lin_kv"].extend((linear.argmax(-1) == tgt).float().flatten().tolist())
        direct_hits.extend((direct.argmax(-1) == tgt).float().flatten().tolist())
        if model.frozen_emb is not None:
            w1 = model.slow()[0]
            emb = F.embedding(idx, model.frozen_emb)
            embedded = model.fast_input(h, idx)
            parts["fast_input_norm"].extend(kag._gather(embedded, wp).norm(dim=-1).flatten().tolist())
            parts["code_norm"].extend(write.norm(dim=-1).flatten().tolist())
            we = kag._gather(F.embedding(idx, model.frozen_emb), wp)
            gated = model.code_gate is not None
            qe = kag._gather(embedded - ln if gated else embedded * math.sqrt(2) - ln,
                             meta["query_pos"])
            emb_dot = torch.bmm(qe, we.transpose(1, 2))
            parts["emb_addr"].extend((emb_dot.argmax(-1) == target).float().flatten().tolist())
            left = kag._gather(F.linear(ln if gated else ln / math.sqrt(2), w1), wp)
            right = kag._gather(F.linear(emb * model.code_gate if gated else emb / math.sqrt(2), w1), wp)
            parts["emb_ln_norm"].extend(left.norm(dim=-1).flatten().tolist())
            parts["emb_e_norm"].extend(right.norm(dim=-1).flatten().tolist())
            parts["emb_dot"].extend((left * right).sum(-1).flatten().tolist())
    out = {k: float(np.mean(v)) if k in ("addr_a", "addr_h", "lin_kv", "emb_addr") else
           float(np.median(v)) for k, v in parts.items()
           if v and k not in ("wv_amp", "kv_err", "abs_dot")}
    out["addr_margin"] /= max(float(np.median(parts["abs_dot"])), 1e-12)
    out["amp_ratio"] /= max(float(np.median(parts["wv_amp"])), 1e-12)
    out["err_ratio"] /= max(float(np.median(parts["kv_err"])), 1e-12)
    out["clipped_frac_c0"] = round(float(np.mean(c0_clipped)), 6) if c0_clipped else None
    out["inner_norm_c0"] = round(float(np.median(c0_norm)), 6) if c0_norm else None
    for name, e in zip(("eta_W1", "eta_b1", "eta_W2", "eta_b2"), model.etas(eta_base)):
        out[name] = float(e.detach())
    if arm in KVETA_ARMS:
        out["eta_kvW2"] = float(model.eta_kvW2(eta_base).detach())
    if arm in KAMIN_ARMS:
        out["inner_multiplier"] = model.inner_multiplier
    out["lin_kv_max_diff"] = max_diff
    out["lin_kv_acc_diff"] = abs(out["lin_kv"] - float(np.mean(direct_hits)))
    return out


def r4_flags(quick: list[list], probes: list[dict], final_acc: float, occ_minus: float) -> dict:
    """Derived acquisition, retention and drop episodes from aligned quick/probe rows."""
    accuracy = {int(r[0]): r[3] for r in quick if r[3] is not None}
    geometry = {int(r["step"]): r for r in probes}
    bound = [s for s, a in accuracy.items() if a >= .90]
    episodes = []
    steps = sorted(accuracy)
    for i, step in enumerate(steps[:-1]):
        if accuracy[step] < .90 or accuracy[steps[i + 1]] >= .90:
            continue
        later = steps[i + 1:]
        recovered = next((s for s in later if accuracy[s] >= .90), None)
        interval = [s for s in later if recovered is None or s < recovered]
        geo_interval = [s for s in sorted(geometry) if step <= s and
                        (recovered is None or s < recovered)]
        episodes.append(dict(t_bound_last=step,
                             t_acc_drop=next((s for s in interval if accuracy[s] < .50), None),
                             t_addr_drop=next((s for s in geo_interval
                                               if geometry[s]["addr_a"] < .50), None),
                             min_acc=min(accuracy[s] for s in interval),
                             recovered=recovered is not None))
    first = min(bound) if bound else None
    return dict(bound_final=final_acc >= .90, bound_ever=bool(bound),
                first_bound_step=first, collapse=bool(bound) and final_acc < .50,
                degraded=bool(bound) and .50 <= final_acc < .90,
                partial=not bound and bool(accuracy) and max(accuracy.values()) >= occ_minus + .10,
                episodes=episodes,
                t_addr_lost=(next((s for s in sorted(geometry)
                                   if geometry[s]["addr_a"] <= .30), None) if not bound else None),
                probe_at_first_bound=geometry.get(first))


def r4_sanity() -> dict:
    """Small CPU algebra checks plus the random-init echo address check."""
    from . import exp_kag as kag

    dev = torch.device("cpu")
    st = _settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                        fast_mult=2, window=8, chunk=4, n_keys=4))
    task = Task("icl_mc", 8, 16, 8, 4, 5, 4, 2)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_model(task, st, dev, 0, "e2e2")
    finally:
        torch.set_default_dtype(old_dtype)
    rng = np.random.default_rng(17)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    results = {}
    for stage in ("init", "trained"):
        idx, y, _ = task.tensors(*task.raw(rng, 2), dev)
        h = model.backbone(idx)
        W = model.per_sequence(2)
        sl = slice(0, task.chunk)
        loss_ref = _ce_tokens(model, fast_block(h[:, sl], model.head.ln, W), y[:, sl]).mean(1)
        grad_ref = torch.autograd.grad(loss_ref.sum(), W, retain_graph=True)
        x = model.fast_input(h[:, sl], idx[:, sl])
        losses, values = [], []
        for switches in ((True, False), (False, True)):
            loss, grads = factor_grads(model, h[:, sl], y[:, sl], x, W, switches)
            losses.append(float(((loss - loss_ref).abs().max() /
                                 loss_ref.abs().max().clamp_min(1e-12)).detach()))
            values.append(max(float(((a - b).norm() / b.norm().clamp_min(1e-12)).detach())
                              for a, b in zip(grads, grad_ref)))
        results[f"S1_{stage}"] = dict(passed=max(losses + values) < 1e-5,
                                     loss_rel=max(losses), gradient_rel=max(values))
        for clip in ((10., .001) if stage == "trained" else (10.,)):
            vectors, clip_stats = {}, GradStats()
            # Full sequence for both-on/both-off; the first two chunks (one inner update,
            # where the product-rule split is exact) for the input/error additivity.
            two = slice(0, 2 * task.chunk)
            for arm, switches, sl in (("e2e2", None, slice(None)), ("e2e1", None, slice(None)),
                                      ("both_on", (True, True), slice(None)),
                                      ("both_off", (False, False), slice(None)),
                                      ("e2e2_two", None, two), ("e2e1_two", None, two),
                                      ("input_only", (True, False), two),
                                      ("error_only", (False, True), two)):
                model.zero_grad(set_to_none=True)
                stats = clip_stats if arm == "e2e2" else GradStats()
                loss = sequential_inner(model, model.backbone(idx[:, sl]), y[:, sl],
                                        model.per_sequence(2), model.etas(1.), clip,
                                        arm.startswith("e2e2"), stats, task.chunk, idx[:, sl],
                                        switches)
                loss.backward()
                vectors[arm] = torch.cat([p.grad.flatten() if p.grad is not None else
                                          torch.zeros_like(p).flatten() for p in model.parameters()])
            on = float((vectors["both_on"] - vectors["e2e2"]).norm() /
                       vectors["e2e2"].norm().clamp_min(1e-12))
            off = float((vectors["both_off"] - vectors["e2e1"]).norm() /
                        vectors["e2e1"].norm().clamp_min(1e-12))
            second = vectors["e2e2_two"] - vectors["e2e1_two"]
            split = ((vectors["input_only"] - vectors["e2e1_two"]) +
                     (vectors["error_only"] - vectors["e2e1_two"]))
            add = float((split - second).norm() / second.norm().clamp_min(1e-12))
            clipped = clip_stats.summary()["clipped_frac"]
            results[f"S2_{stage}_clip{clip:g}"] = dict(
                passed=max(on, off, add) < 1e-4 and (clip > 1 or clipped > 0),
                both_on_rel=on, both_off_rel=off, additivity_rel=add, clipped_frac=clipped)
        if stage == "init":
            for _ in range(5):
                ix, yy, _ = task.tensors(*task.raw(rng, 2), dev)
                opt.zero_grad(set_to_none=True)
                arm_loss(model, "e2e1", ix, yy, 1., 10., 4, GradStats()).backward()
                opt.step()
    emb_st = _settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                            fast_mult=2, window=16, chunk=16, n_keys=4, T=128))
    emb_task = _task(emb_st, "echo")
    plain = build_model(emb_task, emb_st, dev, 0, "e2e1")
    emb = build_model(emb_task, emb_st, dev, 0, "e2e1emb")
    idx, _, _ = emb_task.tensors(*emb_task.raw(rng, 2), dev)
    with torch.no_grad():
        ph, eh = plain.backbone(idx), emb.backbone(idx)
        p = plain.fast_block(ph, idx, plain.slow()).norm()
        e = emb.fast_block(eh, idx, emb.slow()).norm()
        ratio = float(e / p)
    results["S3"] = dict(passed=.8 <= ratio <= 1.2, norm_ratio=ratio)
    echo_st = _settings(dict(d_model=128, n_layers=4, n_heads=4, window=64,
                             chunk=64, n_keys=16, T=512))
    echo = _task(echo_st, "echo")
    address_model = build_model(echo, echo_st, dev, 0, "e2e1")
    tokens, meta = echo.raw(np.random.default_rng([PROBE_SEED, 0]), 32)
    with torch.no_grad():
        ix, _, _ = echo.tensors(tokens, meta, dev)
        h = address_model.backbone(ix)
        a = kag._code(address_model, h, idx=ix)
        dot = torch.bmm(kag._gather(a, meta["query_pos"]),
                        kag._gather(a, meta["write_pos"]).transpose(1, 2))
        addr = float((dot.argmax(-1) == torch.arange(16)[None]).float().mean())
    results["S4"] = dict(passed=addr >= .95, addr_a=addr)
    return results


def r42_sanity() -> dict:
    """CPU fp32 mask identities and independent training-stream checks (N1–N4)."""
    dev = torch.device("cpu")
    st = _settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                        fast_mult=2, window=16, chunk=16, n_keys=4, T=128))
    task = _task(st, "echo")
    tokens, meta = task.raw(np.random.default_rng(17), 2)
    idx, y, _ = task.tensors(tokens, meta, dev)
    idx, y = idx[:, :32], y[:, :32]
    model = build_model(task, st, dev, 0, "e2e1emb")

    def gradient(keep: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, list]:
        model.zero_grad(set_to_none=True)
        loss = arm_loss(model, "e2e1emb", idx, y, 10., 10., 16, GradStats(), keep)
        grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                            for g, p in zip(grads, model.parameters())])
        return loss.detach(), vector.detach(), grads
    base, bg, _ = gradient(None)
    ones, og, _ = gradient(torch.ones_like(y, dtype=torch.float32))
    loss_rel = float((base - ones).abs() / base.abs().clamp_min(1e-12))
    grad_rel = float((bg - og).norm() / bg.norm().clamp_min(1e-12))
    n1 = dict(passed=loss_rel <= 1e-6 and grad_rel <= 1e-6,
              loss_rel=loss_rel, gradient_rel=grad_rel)
    keep = outer_keep(meta, y.shape, "kv", None, dev)
    masked, _, _ = gradient(keep)
    # Rebuild both recurrences independently and compare the actual fast gradients
    # at every chunk, even after the first fast update.
    h = model.backbone(idx)

    def fast_grads(mask: torch.Tensor | None) -> list[list[torch.Tensor]]:
        weights = model.per_sequence(len(idx))
        history = []
        for c in range(2):
            sl = slice(c * 16, (c + 1) * 16)
            ce_c = token_ce(model.read(model.fast_block(h[:, sl], idx[:, sl], weights)), y[:, sl])
            inner_c = ce_c.mean(1)
            outer_c = inner_c if mask is None else (ce_c * mask[:, sl]).sum(1) / 16
            # The outer branch is deliberately present, but is not differentiated
            # to form the fast update.
            assert outer_c.shape == inner_c.shape
            g = torch.autograd.grad(inner_c.sum(), weights, retain_graph=True)
            history.append([v.detach() for v in g])
            clipped, _, _ = _clip_fast(list(g), 10.)
            weights = [w - eta * v for w, eta, v in zip(weights, model.etas(10.), clipped)]
        return history
    inner_equal = all(torch.equal(a, b) for ga, gb in zip(fast_grads(None), fast_grads(keep))
                      for a, b in zip(ga, gb))
    weights = model.per_sequence(len(idx))
    logits = model.read(model.fast_block(h[:, :16], idx[:, :16], weights))
    ce = token_ce(logits, y[:, :16])
    wp = torch.as_tensor(meta["write_pos"])
    expected = base - ce.gather(1, wp).sum() / (16 * len(idx) * 2)
    masked_logits_grad = torch.autograd.grad((ce * keep[:, :16]).sum(), logits,
                                             retain_graph=True)[0]
    zero = bool((masked_logits_grad.gather(1, wp[..., None].expand(-1, -1, task.V)) == 0).all())
    n2_rel = float(((masked - expected).abs() / base.abs().clamp_min(1e-12)).detach())
    n2 = dict(passed=inner_equal and zero and n2_rel <= 1e-6,
              inner_equal=inner_equal, kv_outer_zero=zero, loss_rel=n2_rel)
    streams = {}
    for arm in ("e2e1emb", "e2e1emb_kvm", "e2e1emb_rnd", "e2e1_kvm"):
        rng = np.random.default_rng([0, TRAIN_SEED])
        masks = np.random.default_rng([0, MASK_SEED])
        hashes = []
        for _ in range(5):
            tok, m = task.raw(rng, 2)
            hashes.append(hashlib.sha1(tok[:, :task.T].tobytes()).hexdigest())
            outer_keep(m, (2, task.T), MASK_ARMS.get(arm, "none"), masks, dev)
        streams[arm] = hashes
    n3 = dict(passed=all(v == streams["e2e1emb"] for v in streams.values()),
              hashes=streams["e2e1emb"])
    rnd = outer_keep(meta, (2, task.T), "rnd", np.random.default_rng([0, MASK_SEED]), dev)
    kv = outer_keep(meta, (2, task.T), "kv", None, dev)
    rnd_pos = [set(torch.where(row == 0)[0].tolist()) for row in rnd]
    kv_pos = [set(torch.where(row == 0)[0].tolist()) for row in kv]
    valid = all(kv_pos[i] == set(meta["write_pos"][i]) and
                bool(np.array_equal(y[i, meta["write_pos"][i]].numpy(),
                                    tokens[i, meta["write_pos"][i] - 1])) and
                len(rnd_pos[i]) == task.n_keys and
                rnd_pos[i] <= set(np.concatenate((meta["write_pos"][i] - 1,
                                                  meta["write_pos"][i] + 1))) and
                not rnd_pos[i] & (kv_pos[i] | set(meta["write_pos"][i] - 2))
                for i in range(2))
    n4 = dict(passed=valid, kv_counts=[len(p) for p in kv_pos],
              rnd_counts=[len(p) for p in rnd_pos])
    return dict(N1=n1, N2=n2, N3=n3, N4=n4)


def _inner_grad_of(loss: torch.Tensor, params: tuple) -> torch.Tensor:
    """Flattened parameter gradient of one loss, ``None`` entries as zeros."""
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                      for g, p in zip(grads, params)])


def _rel(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Largest relative distance between two per-tensor gradient lists."""
    return max(float((x - y).norm() / y.norm().clamp_min(1e-12)) for x, y in zip(a, b))


def _capture_inner_grads(fn) -> list[torch.Tensor]:
    """Run ``fn`` and return the first per-sequence inner gradient it clips.

    The training and eval paths do not expose their fast gradients, so the shared
    ``_clip_fast`` step they both call is observed instead: sanity L8 then compares
    the real paths, not reimplementations of them.
    """
    calls: list[list[torch.Tensor]] = []
    original = _clip_fast

    def spy(grads, g_clip):
        calls.append([g.detach().clone() for g in grads])
        return original(grads, g_clip)

    globals()["_clip_fast"] = spy
    try:
        fn()
    finally:
        globals()["_clip_fast"] = original
    return calls[0] if calls else []


def r42l_sanity() -> dict:
    """CPU fp32 identities of the linear inner loss, the one selector and w600 (L1-L8).

    L1 keeps the CE arms bit-identical, L2-L4 the token loss and the
    selectors, L5 the unchanged outer loss, L6 the schedule override, L7 the
    echo_ctrl batch and L8 that all four code paths build the same chunk-0 inner
    gradient from the one selector.
    """
    dev = torch.device("cpu")
    st = _settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2,
                        fast_mult=2, window=16, chunk=64, n_keys=16, T=256,
                        n_eval=4, eval_batch=2, n_probe=4, n_quick_eval=4))
    task = _task(st, "echo")
    tokens, meta = task.raw(np.random.default_rng(17), 2)
    idx, y, _ = task.tensors(tokens, meta, dev)
    full_idx, full_y = idx, y
    idx, y = idx[:, :2 * task.chunk], y[:, :2 * task.chunk]
    idx_i, y_i = idx[:, :task.chunk], y[:, :task.chunk]
    model = build_model(task, st, dev, 0, "e2e1emb")
    params = tuple(model.parameters())

    # ---- L2, prediction independence of the linear token loss
    yy = torch.randint(0, task.V, (2, 5), generator=torch.Generator().manual_seed(3))

    def z_grad(seed: int, fn) -> torch.Tensor:
        z = torch.randn(2, 5, task.V, generator=torch.Generator().manual_seed(seed))
        z = z.detach().requires_grad_(True)
        return torch.autograd.grad(fn(z).sum(), z)[0]

    ref = 1.0 / task.V - F.one_hot(yy, task.V).to(torch.float32)
    g1, g2 = z_grad(3, lambda z: lin_token(z, yy)), z_grad(11, lambda z: lin_token(z, yy))
    l2 = dict(passed=bool(torch.equal(g1, g2) and (g1 - ref).abs().max() <= 1e-7 and
                           (g2 - ref).abs().max() <= 1e-7),
              max_abs=float(max((g1 - ref).abs().max(), (g2 - ref).abs().max())))

    # ---- L3, at a uniform prediction lin is CE
    zero = torch.zeros(2, 5, task.V, requires_grad=True)
    g_ce = torch.autograd.grad(token_ce(zero, yy).sum(), zero, retain_graph=True)[0]
    g_lin = torch.autograd.grad(lin_token(zero, yy).sum(), zero, retain_graph=True)[0]
    l3 = dict(passed=bool((g_ce - g_lin).abs().max() <= 1e-7),
              max_abs=float((g_ce - g_lin).abs().max()))

    # ---- L1, the existing path and an all-CE selector through the new path
    def original_recurrence(m: TTTModel) -> torch.Tensor:
        hh = m.backbone(idx)
        xx = m.fast_input(hh, idx)
        weights = m.per_sequence(len(idx))
        total = 0.0
        for c in range(idx.shape[1] // task.chunk):
            sl = slice(c * task.chunk, (c + 1) * task.chunk)
            loss_c = _ce_tokens(m, fast_block(hh[:, sl], m.head.ln, weights, xx[:, sl]),
                                y[:, sl]).mean(1)
            g = torch.autograd.grad(loss_c.sum(), weights, retain_graph=True)
            g, _, _ = _clip_fast(list(g), 10.)
            total = total + loss_c.mean()
            weights = [w - eta * v for w, eta, v in zip(weights, m.etas(10.), g)]
        return total / (idx.shape[1] // task.chunk)

    ref_loss = original_recurrence(model)
    arm_ref = arm_loss(model, "e2e1emb", idx, y, 10., 10., task.chunk, GradStats())
    existing_equal = bool(torch.equal(ref_loss.detach(), arm_ref.detach()) and
                          torch.equal(_inner_grad_of(ref_loss, params),
                                      _inner_grad_of(arm_ref, params)))
    plain = sequential_inner(model, model.backbone(idx), y, model.per_sequence(len(idx)),
                             model.etas(10.), 10., False, GradStats(), task.chunk, idx)
    all_ce = sequential_inner(model, model.backbone(idx), y, model.per_sequence(len(idx)),
                              model.etas(10.), 10., False, GradStats(), task.chunk, idx,
                              sel=torch.zeros_like(y, dtype=torch.bool))
    selector_equal = bool(torch.equal(plain.detach(), all_ce.detach()) and
                          torch.equal(_inner_grad_of(plain, params),
                                      _inner_grad_of(all_ce, params)))
    l1 = dict(passed=existing_equal and selector_equal,
              existing_path_equal=existing_equal, all_ce_selector_equal=selector_equal,
              loss=float(plain.detach()), n_chunks=idx.shape[1] // task.chunk)

    # ---- L4, the selectors on real echo batches
    full_tokens, full_meta = task.raw(np.random.default_rng(19), 4)
    _, fy, _ = task.tensors(full_tokens, full_meta, dev)
    positions = np.asarray(full_meta["write_pos"])
    rows = np.arange(4)[:, None]
    written = bool(np.array_equal(fy.numpy()[rows, positions],
                                  full_tokens[rows, positions - 1]))
    l4: dict[str, dict] = {}
    for arm, kind in LIN_ARMS.items():
        sel = lin_selector(arm, full_meta["write_pos"], 4, task.T, dev)
        mask = np.zeros((4, task.T), dtype=bool)
        if kind in ("all", "xkv"):
            mask[:] = True
        if kind == "kv":
            mask[rows, positions] = True
        elif kind == "vW":
            mask[rows, positions + 1] = True
        elif kind == "xkv":
            mask[rows, positions] = False
        counts = sel.sum(1).tolist()
        l4[arm] = dict(passed=bool(torch.equal(sel, torch.from_numpy(mask)) and
                                    len(set(counts)) == 1 and
                                    counts[0] == lin_count(arm, task.T, task.n_keys)),
                       per_seq=counts, want=lin_count(arm, task.T, task.n_keys))
    l4["kv_target_is_written_value"] = dict(passed=written)
    l4 = dict(passed=all(v["passed"] for v in l4.values()), **l4)

    # ---- L5, the outer loss at W0 is the anchor's; the inner gradient splits additively
    def outer_c0(arm: str) -> torch.Tensor:
        sel = lin_selector(arm, meta["write_pos"], len(idx), y.shape[1], dev)
        return arm_loss(model, arm, idx_i, y_i, 10., 10., task.chunk, GradStats(), None,
                        None if sel is None else sel[:, :task.chunk]).detach()

    anchor_c0 = outer_c0("e2e1emb")
    l5: dict[str, dict] = {arm: dict(passed=bool(torch.equal(outer_c0(arm), anchor_c0)),
                                     ce=float(anchor_c0)) for arm in LIN_ARMS}
    hh = model.backbone(idx_i)
    W0 = model.per_sequence(len(idx))
    z0 = model.read(fast_block(hh, model.head.ln, W0,
                               model.fast_input(hh, idx_i)))
    ce0 = token_ce(z0, y_i)
    sel_linkv = lin_selector("e2e1emb_linkv", meta["write_pos"], len(idx), y.shape[1],
                             dev)[:, :task.chunk]
    lin0 = lin_token(z0, y_i)
    g_arm = torch.autograd.grad(token_loss(sel_linkv, lin0, ce0).mean(1).sum(), W0,
                                retain_graph=True)
    g_anchor = torch.autograd.grad(ce0.mean(1).sum(), W0, retain_graph=True)
    g_diff = torch.autograd.grad(((lin0 - ce0) * sel_linkv).sum() / task.chunk, W0)
    # Compare the reconstructed full gradient: subtraction of two nearly equal
    # fp32 gradients amplifies rounding relative to the tiny intervention itself.
    split_rel = _rel(list(g_arm), [b + d for b, d in zip(g_anchor, g_diff)])
    cancellation_rel = _rel([a - b for a, b in zip(g_arm, g_anchor)], list(g_diff))
    l5["linkv_inner_gradient_split"] = dict(passed=bool(split_rel <= 1e-6), rel=split_rel,
                                            difference_relative=cancellation_rel,
                                            normalization="full reconstructed gradient")
    l5 = dict(passed=all(v["passed"] for v in l5.values()), **l5)

    # ---- L6, the schedule override
    sched = _settings(dict(lr_total=3000, warmup=200, lr_floor=0.1))
    def scale(arm: str, step: int) -> float:
        """The LR scale this cell uses at ``step`` (0-indexed)."""
        return lr_scale(step, arm_warmup(arm, sched), sched.lr_total, sched.lr_floor)

    peak, peak_end = scale("e2e1emb_w600", 199), scale("e2e1emb_w600", 599)
    others = {arm: scale(arm, 199) for arm in ("e2e1emb", *R42L_ARMS)
              if arm != "e2e1emb_w600"}
    l6 = dict(passed=bool(abs(peak - 200 / 600) <= 1e-12 and abs(peak_end - 1.0) <= 1e-12
                           and all(v == 1.0 for v in others.values())),
              w600_step199=peak, w600_step599=peak_end, other_step199=others)

    # ---- L7, echo_ctrl has no copyable value at the kv target
    ctrl = _task(st, "echo_ctrl")
    c_tokens, c_meta = ctrl.raw(np.random.default_rng(23), 64)
    _, cy, _ = ctrl.tensors(c_tokens, c_meta, dev)
    cwp = np.asarray(c_meta["write_pos"])
    crow = np.arange(64)[:, None]
    unequal = float((c_tokens[crow, cwp - 1] != cy.numpy()[crow, cwp]).mean())
    l7 = dict(passed=bool(.85 <= unequal <= .99), r_ne_f_frac=unequal,
              n_events=int(64 * ctrl.n_keys), sequences=64)

    # ---- L8, one selector behind the training, eval, probe and ablation paths
    idx, y = full_idx, full_y
    l8: dict[str, dict] = {}
    for arm in LIN_ARMS:
        sel = lin_selector(arm, meta["write_pos"], len(idx), y.shape[1], dev)
        train_g = _capture_inner_grads(
            lambda: arm_loss(model, arm, idx, y, 10., 10., task.chunk, GradStats(), None, sel))
        eval_g = _capture_inner_grads(
            lambda: eval_batch_logits(model, task, arm, idx, y, 10., 10., task.chunk,
                                      GradStats(), meta=meta))
        capture: dict = {}
        r4_probe(model, task, [(idx, y, meta)], 10., 10., arm=arm, capture=capture)
        ablation_g = _capture_inner_grads(
            lambda: r42_ablation(model, task, [(idx, y, meta)], 10., 10., arm=arm))
        paths = dict(eval=eval_g, probe=capture["inner_grads"], ablation=list(ablation_g))
        rels = {name: (_rel(g, train_g) if train_g and g else math.inf)
                for name, g in paths.items()}
        worst = max(rels.values())
        l8[arm] = dict(passed=bool(worst <= 1e-6), rel=worst, rels=rels, n_grads=len(train_g))
    l8 = dict(passed=all(v["passed"] for v in l8.values()), **l8)
    return dict(L1=l1, L2=l2, L3=l3, L4=l4, L5=l5, L6=l6, L7=l7, L8=l8)

def r43_sanity() -> dict:
    """R4-3 checks on CPU: factor algebra on the +emb path, the E/K manipulation
    check, the second-order mask identities and the new arms' batch streams.

    S1/S2 are R4-1's checks repeated with ``x = model.fast_input(h, idx)`` on a tiny
    ``e2e2emb`` model in fp64 (R4-1's precision). A plain model would make
    ``fast_input`` silently return ``LN(h)`` and check nothing, hence the assert on
    ``frozen_emb``. N2 is R4-2's mask identity on ``e2e2_kvm`` plus the second-order
    linearity in ``keep`` on the real ``arm_loss`` path. Everything except the E/K
    manipulation check aborts the run on failure.
    """
    dev = torch.device("cpu")
    st = _settings(dict(d_model=16, n_layers=1, n_heads=2, backbone_mult=2, fast_mult=2,
                        window=16, chunk=16, n_keys=4, T=128))
    task = _task(st, "echo")
    b, n_chunk = task.chunk, task.T // task.chunk
    results: dict[str, dict] = {}
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        emb_model = build_model(task, st, dev, 0, "e2e2emb")
    finally:
        torch.set_default_dtype(old_dtype)
    if emb_model.frozen_emb is None:
        raise AssertionError("S1/S2 need an +emb model, built with e2e2emb")

    # ------------------------------------------------------------- S1 / S2 (+emb)
    rng = np.random.default_rng(17)
    opt = torch.optim.AdamW(emb_model.parameters(), lr=1e-2)
    for stage in ("init", "trained"):
        idx, y, _ = task.tensors(*task.raw(rng, 2), dev)
        h = emb_model.backbone(idx)
        x = emb_model.fast_input(h, idx)
        # S1 on every chunk, so later chunks are checked on the weights the inner
        # update actually produced rather than only on W0
        W, losses, values = emb_model.per_sequence(2), [], []
        for c in range(n_chunk):
            sl = slice(c * b, (c + 1) * b)
            loss_ref = _ce_tokens(emb_model, emb_model.fast_block(h[:, sl], idx[:, sl], W),
                                  y[:, sl]).mean(1)
            grad_ref = torch.autograd.grad(loss_ref.sum(), W, create_graph=True,
                                           retain_graph=True)
            for switches in ((True, False), (False, True)):
                loss, grads = factor_grads(emb_model, h[:, sl], y[:, sl], x[:, sl], W, switches)
                losses.append(float(((loss - loss_ref).abs().max() /
                                     loss_ref.abs().max().clamp_min(1e-12)).detach()))
                values.append(max(float(((u - v).norm() / v.norm().clamp_min(1e-12)).detach())
                                  for u, v in zip(grads, grad_ref)))
            clipped, _, _ = _clip_fast(list(grad_ref), 10.)
            W = [w - eta * g for w, eta, g in zip(W, emb_model.etas(1.), clipped)]
        results[f"S1_{stage}"] = dict(passed=max(losses + values) < 1e-5,
                                     loss_rel=max(losses), gradient_rel=max(values))
        for clip in ((10., .001) if stage == "trained" else (10.,)):
            vectors, clip_stats = {}, GradStats()
            # the full sequence for both-on/both-off; the first two chunks (one inner
            # update, where the product-rule split is exact) for the additivity
            two = slice(0, 2 * b)
            for arm, switches, sl in (("e2e2", None, slice(None)), ("e2e1", None, slice(None)),
                                      ("both_on", (True, True), slice(None)),
                                      ("both_off", (False, False), slice(None)),
                                      ("e2e2_two", None, two), ("e2e1_two", None, two),
                                      ("input_only", (True, False), two),
                                      ("error_only", (False, True), two)):
                emb_model.zero_grad(set_to_none=True)
                stats = clip_stats if arm == "e2e2" else GradStats()
                loss = sequential_inner(emb_model, emb_model.backbone(idx[:, sl]), y[:, sl],
                                        emb_model.per_sequence(2), emb_model.etas(1.), clip,
                                        arm.startswith("e2e2"), stats, b, idx[:, sl], switches)
                loss.backward()
                vectors[arm] = torch.cat([p.grad.flatten() if p.grad is not None else
                                          torch.zeros_like(p).flatten()
                                          for p in emb_model.parameters()])
            on = float((vectors["both_on"] - vectors["e2e2"]).norm() /
                       vectors["e2e2"].norm().clamp_min(1e-12))
            off = float((vectors["both_off"] - vectors["e2e1"]).norm() /
                        vectors["e2e1"].norm().clamp_min(1e-12))
            second = vectors["e2e2_two"] - vectors["e2e1_two"]
            split = ((vectors["input_only"] - vectors["e2e1_two"]) +
                     (vectors["error_only"] - vectors["e2e1_two"]))
            add = float((split - second).norm() / second.norm().clamp_min(1e-12))
            clipped = clip_stats.summary()["clipped_frac"]
            results[f"S2_{stage}_clip{clip:g}"] = dict(
                passed=max(on, off, add) < 1e-4 and (clip > 1 or clipped > 0),
                both_on_rel=on, both_off_rel=off, additivity_rel=add, clipped_frac=clipped)
        if stage == "init":
            # the anchor trajectory the E/K manipulation check below reuses
            for _ in range(25):
                ix, yy, _ = task.tensors(*task.raw(rng, 2), dev)
                opt.zero_grad(set_to_none=True)
                arm_loss(emb_model, "e2e1emb", ix, yy, 1., 10., b, GradStats()).backward()
                opt.step()

    # ------------------------------------------------- E/K share, tiny model (descriptive)
    # The gate is ek_share_record at the run's own size (logged separately as
    # R43_EK_SHARE); these tiny-model numbers are the relative metric, logged because
    # they are cheap and because they show the correction is small at init. They never gate
    # and never abort.
    ek_model = build_model(task, st, dev, 0, EK_ANCHOR)
    rng = np.random.default_rng(23)
    ek_opt = torch.optim.AdamW(ek_model.parameters(), lr=1e-2)
    ek_stages: dict[str, dict] = {}
    for stage in ("init", "trained"):
        ix, yy, _ = task.tensors(*task.raw(rng, 2), dev)
        vectors = {arm: _outer_grad_vector(ek_model, arm, ix, yy, 1., 10., b)
                   for arm in (EK_ANCHOR, EK_HALVES["E"], EK_HALVES["K"])}
        anchor = vectors[EK_ANCHOR].norm().clamp_min(1e-12)
        rel = {arm: float((vectors[arm] - vectors[EK_ANCHOR]).norm() / anchor)
               for arm in EK_HALVES.values()}
        ek_stages[stage] = dict(rel_E=rel[EK_HALVES["E"]], rel_K=rel[EK_HALVES["K"]],
                                note="descriptive only: the gate is the real-size share")
        if stage == "init":
            for _ in range(25):
                ix, yy, _ = task.tensors(*task.raw(rng, 2), dev)
                ek_opt.zero_grad(set_to_none=True)
                arm_loss(ek_model, EK_ANCHOR, ix, yy, 1., 10., b, GradStats()).backward()
                ek_opt.step()
    results["EK_tiny_descriptive"] = dict(stages=ek_stages, threshold=EK_SHARE_MIN,
                                          gating=False, passed=True)

    # -------------------------------------- N2, second order, on the real path
    # e2e2_kvm carries no frozen code, so it runs the plain input path
    so_model = build_model(task, st, dev, 0, "e2e2_kvm")
    if so_model.frozen_emb is not None:
        raise AssertionError("e2e2_kvm is a plain arm")
    tokens, meta = task.raw(np.random.default_rng(29), 2)
    idx, y, _ = task.tensors(tokens, meta, dev)
    keep = outer_keep(meta, y.shape, "kv", None, dev)
    ones = torch.ones_like(y, dtype=torch.float32)

    def grads_of(mask: torch.Tensor | None,
                 arm: str = "e2e2_kvm") -> tuple[torch.Tensor, torch.Tensor]:
        so_model.zero_grad(set_to_none=True)
        loss = arm_loss(so_model, arm, idx, y, 10., 10., b, GradStats(), mask)
        grads = torch.autograd.grad(loss, tuple(so_model.parameters()), allow_unused=True)
        vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                            for g, p in zip(grads, so_model.parameters())])
        return loss.detach(), vector.detach()
    base, base_g = grads_of(None)
    on, on_g = grads_of(ones)
    masked, masked_g = grads_of(keep)
    inverse, inverse_g = grads_of(1. - keep)
    ones_rel = float((base - on).abs() / base.abs().clamp_min(1e-12))
    ones_grad_rel = float((base_g - on_g).norm() / base_g.norm().clamp_min(1e-12))
    # linearity in keep on the real path: the inner loss does not depend on keep, so
    # the masked split of the outer loss adds back to the unmasked one, losses and
    # all parameter gradients alike
    lin_loss_rel = float((masked + inverse - on).abs() / on.abs().clamp_min(1e-12))
    lin_grad_rel = float((masked_g + inverse_g - on_g).norm() / on_g.norm().clamp_min(1e-12))

    # the inner fast gradients never see the mask, now with create_graph on
    h = so_model.backbone(idx)

    def fast_grads_so(mask: torch.Tensor | None) -> list[list[torch.Tensor]]:
        weights = so_model.per_sequence(len(idx))
        history = []
        for c in range(2):
            sl = slice(c * b, (c + 1) * b)
            ce = token_ce(so_model.read(so_model.fast_block(h[:, sl], idx[:, sl], weights)),
                          y[:, sl])
            inner_c = ce.mean(1)
            outer_c = inner_c if mask is None else (ce * mask[:, sl]).sum(1) / b
            assert outer_c.shape == inner_c.shape
            g = torch.autograd.grad(inner_c.sum(), weights, create_graph=True, retain_graph=True)
            history.append([v.detach() for v in g])
            clipped, _, _ = _clip_fast(list(g), 10.)
            weights = [w - eta * v for w, eta, v in zip(weights, so_model.etas(10.), clipped)]
        return history
    inner_equal = all(torch.equal(u, v) for gu, gv in zip(fast_grads_so(None), fast_grads_so(keep))
                      for u, v in zip(gu, gv))

    # the mask removes exactly the kv-token CE sum (the loss identity)
    weights = so_model.per_sequence(len(idx))
    logits0 = so_model.read(so_model.fast_block(h[:, :b], idx[:, :b], weights))
    ce0 = token_ce(logits0, y[:, :b])
    wp = torch.as_tensor(meta["write_pos"])
    expected = base - ce0.gather(1, wp).sum() / (b * len(idx) * n_chunk)

    # Real-path kv-logit gradient. The chunk-0 logits are captured on the
    # real arm_loss path through a hook on the unembedding, and the gradient of the outer
    # loss is taken at those logits: zero at the kv targets means the mask removed the
    # direct CE term and nothing else reaches it, so exactly 0 is the first-order answer and
    # > 0 is the proof that create_graph really runs through e on the masked path. A check
    # of the unmasked gradient against itself, or on a hand-built graph, could not fail.
    emb_kv_model = build_model(task, st, dev, 0, "e2e1emb_kvm")
    if emb_kv_model.frozen_emb is None:
        raise AssertionError("e2e1emb_kvm must carry the frozen code")
    kv_rows = (keep[:, :b] == 0)

    def kv_logits_grad(model: TTTModel, arm: str, mask: torch.Tensor | None) -> float:
        """L2 norm of dL/d(chunk-0 logits) at the masked positions of chunk 0."""
        captured: list[torch.Tensor] = []
        handle = model.unembed.register_forward_hook(lambda m, i, o: captured.append(o))
        try:
            loss = arm_loss(model, arm, idx, y, 10., 10., b, GradStats(), mask)
            grad = torch.autograd.grad(loss, captured[0], retain_graph=True)[0]
        finally:
            handle.remove()
        assert len(captured) == n_chunk, len(captured)
        rows = kv_rows[..., None].expand_as(grad)
        return float(grad[rows].norm())

    kv_grad = dict(e2e1_kvm=kv_logits_grad(so_model, "e2e1_kvm", keep),
                   e2e1emb_kvm=kv_logits_grad(emb_kv_model, "e2e1emb_kvm", keep),
                   e2e2_kvm=kv_logits_grad(so_model, "e2e2_kvm", keep),
                   e2e2_unmasked=kv_logits_grad(so_model, "e2e2", None))
    kv_ok = (kv_grad["e2e1_kvm"] == 0. and kv_grad["e2e1emb_kvm"] == 0.
             and kv_grad["e2e2_kvm"] > 0.)
    n2_rel = float(((masked - expected).abs() / base.abs().clamp_min(1e-12)).detach())
    results["N2_so"] = dict(
        passed=(ones_rel <= 1e-6 and ones_grad_rel <= 1e-6 and inner_equal and kv_ok
                and n2_rel <= 1e-6 and lin_loss_rel <= 1e-6 and lin_grad_rel <= 1e-6),
        inner_equal=inner_equal, kv_logits_grad=kv_grad, kv_logits_grad_ok=kv_ok,
        kv_targets_per_seq=int(kv_rows[0].sum()),
        loss_rel=n2_rel, ones_loss_rel=ones_rel, ones_gradient_rel=ones_grad_rel,
        linearity_loss_rel=lin_loss_rel, linearity_gradient_rel=lin_grad_rel,
        create_graph="e2e2_kvm" in SECOND_ORDER_ARMS)

    # ------------------------------------------------------------------- N3, N4
    # the first five training batches are the stream; the mask draws from its own
    # stream, so the hash cannot see the mask at all
    streams = {}
    for arm in ("e2e2", "e2e2_kvm", "e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb",
                "e2e2emb"):
        rng = np.random.default_rng([0, TRAIN_SEED])
        masks = np.random.default_rng([0, MASK_SEED])
        hashes = []
        for _ in range(5):
            tok, m = task.raw(rng, 2)
            hashes.append(hashlib.sha1(tok[:, :task.T].tobytes()).hexdigest())
            outer_keep(m, (2, task.T), MASK_ARMS.get(arm, "none"), masks, dev)
        streams[arm] = hashes
    groups = {"second_order": ("e2e2", "e2e2_kvm"),
              "emb": ("e2e1emb", "e2e1emb_kvm", "e2e1e_emb", "e2e1k_emb", "e2e2emb")}
    group_ok = {name: len({tuple(streams[a]) for a in arms}) == 1 for name, arms in groups.items()}
    results["N3"] = dict(passed=all(group_ok.values()), groups_equal=group_ok,
                         hashes=streams["e2e2emb"])
    # N4: the second-order mask is the same kv mask, on the same batch convention
    kv_pos = [set(torch.where(row == 0)[0].tolist()) for row in keep]
    n4_valid = all(kv_pos[i] == set(meta["write_pos"][i]) and
                   bool(np.array_equal(y[i, meta["write_pos"][i]].numpy(),
                                       tokens[i, meta["write_pos"][i] - 1]))
                   for i in range(len(idx)))
    results["N4"] = dict(passed=n4_valid, kv_counts=[len(p) for p in kv_pos])

    # ------------------------------------------------------------- registration
    # arm-table traps that would otherwise fail silently
    so_model.zero_grad(set_to_none=True)
    try:
        arm_loss(so_model, "e2e1e_emb", idx, y, 10., 10., b, GradStats(), keep)
        factor_mask_raises = False
    except ValueError:
        factor_mask_raises = True
    registration = dict(
        passed=all((("e2e1e_emb" in EMB_ARMS and "e2e1k_emb" in EMB_ARMS and
                     FACTOR_ARMS["e2e1e_emb"] == FACTOR_ARMS["e2e1e"] and
                     FACTOR_ARMS["e2e1k_emb"] == FACTOR_ARMS["e2e1k"]),
                    ("e2e2_kvm" in SECOND_ORDER_ARMS),
                    (MASK_ARMS.get("e2e2_kvm") == "kv"),
                    all(a in SEQUENTIAL_ARMS for a in R43_ARMS),
                    factor_mask_raises)))
    registration.update(e2e1e_emb_switches=list(FACTOR_ARMS["e2e1e_emb"]),
                        e2e1k_emb_switches=list(FACTOR_ARMS["e2e1k_emb"]),
                        e2e2_kvm_mask=MASK_ARMS["e2e2_kvm"],
                        second_order_arms=sorted(SECOND_ORDER_ARMS),
                        factor_mask_raises=factor_mask_raises)
    results["registration"] = registration
    return results

def eval_batches(task: Task, tokens: np.ndarray, meta: dict, dev: torch.device,
                 batch: int) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    """Split a fixed sequence set into evaluation batches of inputs and targets."""
    n = tokens.shape[0]
    return [task.tensors(tokens[lo:lo + batch], _slice_meta(meta, lo, min(lo + batch, n)), dev)
            for lo in range(0, n, batch)]


@torch.no_grad()
def r42_ablation(model: TTTModel, task: Task, data: list, eta_base: float,
                 g_clip: float, *, arm: str) -> dict[str, float]:
    """R2-4 fixed write-class ablation with the model's token-aware fast input.

    The per-class gradients are those of the arm's own token loss and the later
    chunks stream with that same inner loss, so "all" reproduces the
    streaming eval of the arm.
    """
    from . import exp_wab as wab

    hits = {name: [] for name in ("all", "all_minus_kv", "kv_only")}
    subsets = {"all": wab.ALL, "all_minus_kv": wab.ALL - {"kv"},
               "kv_only": frozenset({"kv"})}
    etas = [e.detach() for e in model.etas(eta_base)]
    for idx, y, meta in data:
        h = model.backbone(idx, edge_mask=arm_edge_mask(model, arm, idx, meta)).detach()
        masks = wab.class_masks(meta, task.chunk, idx.device)
        sel = lin_selector(arm, meta["write_pos"], len(idx), task.T, idx.device)
        weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
        logits0, _, grads = wab._class_grads(model, h[:, :task.chunk], y[:, :task.chunk],
                                             weights, masks, idx[:, :task.chunk],
                                             None if sel is None else sel[:, :task.chunk],
                                             rule=arm_inner_loss(arm))
        pos = torch.as_tensor(meta["query_pos"], device=idx.device)
        target = torch.as_tensor(meta["query_tok"], device=idx.device)
        for name, subset in subsets.items():
            logits = wab.eval_condition(model, h, y, logits0, grads, etas, g_clip,
                                        task.chunk, "fixed", subset, idx=idx, sel=sel,
                                        rule=arm_inner_loss(arm))
            hits[name].extend((logits.gather(1, pos[..., None].expand(-1, -1, task.V))
                               .argmax(-1) == target).flatten().tolist())
    return {name: float(np.mean(values)) for name, values in hits.items()}


# --------------------------------------------------------------------------- model


class Block(nn.Module):
    """Pre-LN block: causal (optionally windowed) self-attention then an MLP."""

    def __init__(self, d: int, n_heads: int, mult: int, depth: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.resid_scale = 1.0 / math.sqrt(2 * depth)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor | None) -> torch.Tensor:
        """Run the block on ``(B, T, d)`` activations."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        q, k = _rope(q, rope), _rope(k, rope)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        attn = attn.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(attn)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def _rope(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Rotate channel pairs of ``(B, H, T, hd)`` queries or keys."""
    cos, sin = (t[None, None] for t in rope)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class FastHead(nn.Module):
    """The positionwise block whose tensors become the per-sequence fast weights.

    ``h' = h + W2 gelu(W1 LN(h) + b1) + b2``. Its ``LayerNorm`` is never touched
    by the inner loop; only ``(W1, b1, W2, b2)`` is.
    """

    def __init__(self, d: int, m: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.W1 = nn.Parameter(torch.empty(m, d))
        self.b1 = nn.Parameter(torch.zeros(m))
        self.W2 = nn.Parameter(torch.empty(d, m))
        self.b2 = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.W1, std=0.02)
        nn.init.normal_(self.W2, std=0.02)


def fast_block(h: torch.Tensor, ln: nn.LayerNorm,
               params: tuple[torch.Tensor, ...] | list[torch.Tensor],
               x: torch.Tensor | None = None) -> torch.Tensor:
    """Apply the fast block with weights that broadcast against ``h``.

    ``params`` holds ``(W1, b1, W2, b2)``; their leading dimensions must match
    the batch dimensions of ``h``, so one call serves static weights,
    per-sequence weights ``(B, ...)`` and per-chunk weights ``(B, C, ...)``.
    """
    W1, b1, W2, b2 = params
    u = torch.einsum("...bd,...md->...bm", ln(h) if x is None else x, W1) + _bias(b1)
    return h + torch.einsum("...bm,...dm->...bd", F.gelu(u), W2) + _bias(b2)


def _bias(v: torch.Tensor) -> torch.Tensor:
    """Insert a length-1 axis so a fast bias broadcasts against the hidden axis."""
    return v.reshape(*v.shape[:-1], 1, v.shape[-1])


class TTTModel(nn.Module):
    """Sliding-window transformer whose last block holds the fast weights.

    ``window=None`` gives full causal attention (arm ``fa``). The backbone is
    computed once for the whole sequence; only the positionwise fast block is
    replayed per chunk, which is what makes the inner loop cheap to express.
    """

    def __init__(self, n_vocab: int, d: int, n_layers: int, n_heads: int, backbone_mult: int,
                 fast_mult: int, window: int | None) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.window = window
        self.head_dim = d // n_heads
        self.tok = nn.Embedding(n_vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, backbone_mult, n_layers)
                                    for _ in range(n_layers))
        self.head = FastHead(d, fast_mult * d)
        self.ln_f = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, n_vocab, bias=False)
        self.rho = nn.Parameter(torch.zeros(len(FAST_KEYS)))
        self.inner_sign = 1.0
        self.inner_multiplier = 1.0
        self.register_buffer("frozen_emb", None)
        self.register_buffer("key_permutation", None)
        # None adds no state-dict entry: checkpoints without a key code still load strictly.
        self.register_parameter("code_gate", None)
        self.apply(_init_module)
        for blk in self.blocks:
            with torch.no_grad():
                blk.proj.weight.mul_(blk.resid_scale)
                blk.fc2.weight.mul_(blk.resid_scale)
        self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor | None] = {}

    def slow(self) -> tuple[torch.Tensor, ...]:
        """Slow head weights ``(W1, b1, W2, b2)``."""
        return tuple(getattr(self.head, k) for k in FAST_KEYS)

    def etas(self, eta_base: float) -> list[torch.Tensor]:
        """Signed learnable step size per fast tensor, ``sign * eta_base * exp(rho)``."""
        return [self.inner_sign * eta_base * torch.exp(self.rho[i]) for i in range(len(FAST_KEYS))]

    def eta_kvW2(self, eta_base: float) -> torch.Tensor:
        return self.inner_sign * eta_base * self.rho_kvW2.exp()

    def per_sequence(self, batch: int) -> list[torch.Tensor]:
        """Slow head weights expanded into per-sequence fast copies."""
        return [p.unsqueeze(0).expand(batch, *p.shape) for p in self.slow()]

    def backbone(self, idx: torch.Tensor, edge_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Static trunk output; optional bool mask marks BLOCKED attention edges.

        ``edge_mask`` has shape (T,T), (B,T,T), or (B,1,T,T), and is combined
        with causality/windowing in every layer. None preserves the original path.
        """
        T = idx.shape[1]
        x = self.tok(idx)
        rope, mask = self._rope(T, idx.device), self._mask(T, idx.device)
        if edge_mask is not None:
            if edge_mask.dtype != torch.bool or edge_mask.device != idx.device:
                raise ValueError("edge_mask must be bool and on the input device")
            if edge_mask.ndim == 3:
                edge_mask = edge_mask[:, None]
            if (edge_mask.ndim not in (2, 4) or edge_mask.shape[-2:] != (T, T)
                    or (edge_mask.ndim == 4 and
                        edge_mask.shape[:2] not in ((1, 1), (len(idx), 1)))):
                raise ValueError("edge_mask shape must be (T,T), (B,T,T), or (B,1,T,T)")
            if mask is None:
                mask = torch.ones(T, T, device=idx.device, dtype=torch.bool).tril()
            mask = mask & ~edge_mask
        for blk in self.blocks:
            x = blk(x, rope, mask)
        return x

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """Final ``LayerNorm`` and untied unembedding to vocabulary logits."""
        return self.unembed(self.ln_f(h))

    def fast_input(self, h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Single token-aware input to the fast block for all R4 paths."""
        x = self.head.ln(h)
        if self.frozen_emb is None:
            return x
        tokens = idx
        if self.key_permutation is not None:
            keys = (idx >= 2) & (idx < 2 + len(self.key_permutation))
            after_q = torch.zeros_like(keys)
            after_q[..., 1:] = idx[..., :-1] == 1
            tokens = torch.where(keys & after_q, self.key_permutation[(idx - 2).clamp(
                0, len(self.key_permutation) - 1)], idx)
        code = F.embedding(tokens, self.frozen_emb)
        if self.code_gate is not None:
            return x + self.code_gate * code
        return (x + code) / math.sqrt(2)

    def fast_block(self, h: torch.Tensor, idx: torch.Tensor,
                   weights: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        """Token-aware block; plain models retain the original fast_block path."""
        if self.frozen_emb is None:
            return fast_block(h, self.head.ln, weights)
        return fast_block(h, self.head.ln, weights, self.fast_input(h, idx))

    def _rope(self, T: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._rope_cache.get(T)
        if cached is None:
            freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=dev)
                                      / self.head_dim))
            ang = torch.outer(torch.arange(T, device=dev, dtype=torch.float32), freqs)
            cached = (ang.cos(), ang.sin())
            self._rope_cache[T] = cached
        return cached

    def _mask(self, T: int, dev: torch.device) -> torch.Tensor | None:
        """Full causal mask when ``window`` is None, else the last ``window`` tokens."""
        key = (T, dev)
        if key not in self._mask_cache:
            if self.window is None:
                self._mask_cache[key] = None
            else:
                pos = torch.arange(T, device=dev)
                lag = pos[:, None] - pos[None, :]
                self._mask_cache[key] = (lag >= 0) & (lag < self.window)
        return self._mask_cache[key]


def _init_module(m: nn.Module) -> None:
    """Small GPT-style initialisation for every weight matrix and embedding."""
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


def frozen_code(n_vocab: int, d_model: int, seed: int) -> tuple[torch.Tensor, torch.Generator]:
    """The frozen RMS-normalised key code and its advanced CPU generator."""
    gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
    emb = torch.randn(n_vocab, d_model, generator=gen)
    return emb / emb.square().mean(-1, keepdim=True).sqrt(), gen


def install_key_code(model: TTTModel, seed: int) -> None:
    """Function-preserving R4-4c install; call after strict plain checkpoint load."""
    if model.frozen_emb is not None or model.code_gate is not None:
        raise ValueError("key-code installation requires a plain checkpoint")
    code, _ = frozen_code(model.n_vocab, model.tok.weight.shape[1], seed)
    model.frozen_emb = code.to(model.tok.weight)
    model.code_gate = nn.Parameter(model.rho.new_zeros(()))


def build_model(task: Task, st: Settings, dev: torch.device, seed: int,
                arm: str) -> TTTModel:
    """Fresh model whose initialisation depends on ``seed`` only, not on the arm.

    ``fa`` gets full causal attention, every other arm a window of ``w`` tokens.
    """
    torch.manual_seed(seed)
    window = None if arm == "fa" else task.window
    model = TTTModel(task.V, st.d_model, st.n_layers, st.n_heads, st.backbone_mult, st.fast_mult,
                     window)
    if arm in EMB_ARMS:
        model.frozen_emb, gen = frozen_code(task.V, st.d_model, seed)
        if arm == "e2e1embP":
            perm = torch.randperm(task.n_keys, generator=gen)
            # A cyclic shift of a seeded ordering is a derangement for every n > 1.
            mapping = torch.empty_like(perm)
            mapping[perm] = perm.roll(1)
            model.key_permutation = mapping + 2
    if arm == "e2e1_code":
        install_key_code(model, seed)
    if arm in KVETA_ARMS:
        model.rho_kvW2 = nn.Parameter(model.rho[2].detach().clone())
        model.kv_eta_lr = st.kv_eta10_lr if arm == "e2e1emb_kveta10" else st.kv_eta_lr
    set_kamin_regime(model, arm, 0, st)
    return model.to(dev)


# --------------------------------------------------------------------------- inner loops


class GradStats:
    """Pre-clip norms and clip hits of the inner gradients, kept off the hot path."""

    def __init__(self) -> None:
        self.norms: list[torch.Tensor] = []
        self.clips: list[torch.Tensor] = []
        self.indices: list[int] = []

    def add(self, norm: torch.Tensor, clipped: torch.Tensor, index: int = -1) -> None:
        """Record one chunk (or one parallel sweep) of per-sequence gradients."""
        self.norms.append(norm.detach())
        self.clips.append(clipped.detach())
        self.indices.append(index)

    def summary(self) -> dict:
        """Median pre-clip norm and clipped fraction over all recorded pairs."""
        if not self.norms:
            return {}
        norms = torch.cat([n.reshape(-1) for n in self.norms]).cpu().to(torch.float64)
        clips = torch.cat([c.reshape(-1) for c in self.clips]).cpu().to(torch.float64)
        return dict(norm_median=round(float(norms.median()), 6),
                    clipped_frac=round(float(clips.mean()), 6), n_pairs=int(norms.numel()))


def _clip_fast(grads: list[torch.Tensor], g_clip: float
               ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Scale each per-sequence (or per-sequence-chunk) gradient to a global norm.

    The four fast gradients are flattened over their matrix or vector axes and
    reduced jointly, so one scale comes out per leading index. Returns the
    scaled gradients, the pre-clip norms and the clipped mask.
    """
    sq = None
    for g, key in zip(grads, FAST_KEYS):
        tail = MATRIX_NDIM[key]
        part = (g.reshape(*g.shape[:g.ndim - tail], -1) ** 2).sum(-1)
        sq = part if sq is None else sq + part
    norm = sq.clamp_min(1e-24).sqrt().detach()
    scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
    out = []
    for g, key in zip(grads, FAST_KEYS):
        tail = MATRIX_NDIM[key]
        out.append(g * scale.detach().view(*g.shape[:g.ndim - tail], *([1] * tail)))
    return out, norm, norm > g_clip


def _ce_tokens(model: TTTModel, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy of the unembedding of ``out`` against targets ``y``."""
    return token_ce(model.read(out), y)


def token_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token cross-entropy of ``(..., V)`` logits against integer targets."""
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return flat.view(y.shape)


def lin_token(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The R4-2L linear token loss ``-(z[y_t] - mean_j z[t, j])``.

    Its gradient with respect to the logits is ``1/V - onehot(y_t)``, the same for
    every ``z``: it equals the CE gradient at a uniform prediction, and unlike CE it
    does not shrink as the model learns to predict the token.
    """
    return logits.mean(-1) - logits.gather(-1, y.reshape(*y.shape, 1)).squeeze(-1)


def token_loss(sel: torch.Tensor | None, lin: torch.Tensor, ce: torch.Tensor) -> torch.Tensor:
    """Per-token inner loss: the linear loss at the selector's positions, CE elsewhere.

    ``sel is None`` (every CE arm) returns ``ce`` itself, so those paths stay
    bit-identical to the pre-R4-2L code.
    """
    return ce if sel is None else torch.where(sel, lin, ce)


def lin_selector(arm: str, write_pos: Any, batch: int, T: int,
                 dev: torch.device) -> torch.Tensor | None:
    """The one (arm -> lin positions) map every path uses.

    ``kv`` is ``write_pos``, ``vW`` is ``write_pos + 1``, ``xkv`` is everything else
    in ``[0, T)`` and ``all`` is the whole sequence. Returns ``None`` for a CE arm,
    which keeps its code path unchanged.
    """
    arm = B9_BASE.get(arm, arm)
    arm_inner_loss(arm)  # reject unknown arms instead of silently selecting CE
    kind = LIN_ARMS.get("e2e1emb_linkv" if arm == "e2e1emb_kaminlin" else arm)
    if kind is None:
        return None
    if kind == "all":
        return torch.ones(batch, T, dtype=torch.bool, device=dev)
    if write_pos is None:
        raise ValueError(f"{arm} needs the task's write_pos metadata")
    wp = torch.as_tensor(write_pos, device=dev, dtype=torch.long)
    if wp.ndim != 2 or wp.shape[0] != batch:
        raise ValueError(f"{arm}: write_pos {tuple(wp.shape)} does not match batch {batch}")
    sel = torch.zeros(batch, T, dtype=torch.bool, device=dev)
    if bool((wp < 0).any()) or bool((wp >= T).any()):
        raise ValueError(f"{arm}: selector positions outside [0, {T})")
    if kind == "xkv":
        sel[:] = True
        sel.scatter_(1, wp, False)
    else:
        offset = 0 if kind == "kv" else 1
        positions = wp + offset
        if bool((positions < 0).any()) or bool((positions >= T).any()):
            raise ValueError(f"{arm}: selector positions outside [0, {T})")
        sel.scatter_(1, positions, True)
    return sel


def lin_count(arm: str, T: int, n_keys: int) -> int:
    """How many lin positions one sequence must have for ``arm`` (0 for CE arms)."""
    arm = B9_BASE.get(arm, arm)
    kind = LIN_ARMS.get("e2e1emb_linkv" if arm == "e2e1emb_kaminlin" else arm)
    if kind is None:
        return 0
    return T if kind == "all" else T - n_keys if kind == "xkv" else n_keys


def arm_inner_loss(arm: str) -> str:
    """The cell's ``inner_loss`` label: ``ce``, or the arm's selector name."""
    arm = B9_BASE.get(arm, arm)
    if arm not in (*STATIC_ARMS, *TTT_ARMS, "naive"):
        raise ValueError(f"unknown inner-rule arm: {arm}")
    if arm in AICE_ARMS:
        return "aice"
    return "linkv" if arm == "e2e1emb_kaminlin" else ARM_INNER_LOSS.get(arm, "ce")


def inner_token_loss(rule, logits, y, *, ce=None, sel=None, reference=None):
    """Explicit loss dispatcher. CE/selector arithmetic is the inherited path."""
    if rule not in ("ce", "lin", "linkv", "linvW", "linxkv", "aice"):
        raise ValueError(f"unknown inner rule: {rule}")
    if rule == "aice":
        if reference is None or reference.requires_grad:
            raise ValueError("AICE requires completely detached reference logits")
        return token_ce(logits - reference, y)
    ce = token_ce(logits, y) if ce is None else ce
    if rule == "ce" and sel is None:
        return ce
    if sel is None:
        if rule != "lin":
            raise ValueError(f"{rule} needs a selector")
        sel = torch.ones_like(y, dtype=torch.bool)
    return token_loss(sel, lin_token(logits, y), ce)


@torch.no_grad()
def reference_logits(model, h, idx, weights):
    """One batched reference pass, immutable W_ref, entire output stop-gradient."""
    x = model.fast_input(h, idx)
    return model.read(fast_block(h, model.head.ln, weights, x)).detach()


def arm_warmup(arm: str, st: Settings) -> int:
    """The cell's warmup steps: the run's, unless the arm overrides it (w600)."""
    return ARM_WARMUP.get(arm, st.warmup)


def checkpoint_plan(st: Settings, arm: str, seed: int | None) -> set[int]:
    """Step numbers whose checkpoints this cell saves.

    ``checkpoint_steps`` for every arm; ``checkpoint_dense`` added for
    ``checkpoint_dense_arms`` when the seed is at most ``checkpoint_dense_max_seed``;
    ``checkpoint_arm_steps`` per arm; ``checkpoint_steps_so`` for the second-order arms.
    """
    steps = set(st.checkpoint_steps)
    dense_ok = (st.checkpoint_dense_max_seed is None or
                (seed is not None and seed <= st.checkpoint_dense_max_seed))
    if arm in st.checkpoint_dense_arms and dense_ok:
        steps.update(st.checkpoint_dense)
    for entry in st.checkpoint_arm_steps:
        if entry.get("arm") == arm and ("max_seed" not in entry or
                                        (seed is not None and seed <= entry["max_seed"])):
            steps.update(int(s) for s in entry.get("steps", ()))
    if arm in ("e2e2", "e2e2emb", "e2e2_kvm"):
        steps.update(st.checkpoint_steps_so)
    return steps


def kv_w2_step(w, clipped_full, raw_kv, eta_rest, eta_kv, norm, g_clip):
    """s*(eta_kv*g_kv + eta_rest*g_rest), using the original full-list clip.

    Expressed as the shared step plus a kv correction to preserve exact equality
    with the shared-eta path when the two etas are equal. Inner gradients and the
    scalar clipping scale are detached; both log-etas keep their outer gradient.
    """
    scale = torch.where(norm > g_clip, g_clip / norm.clamp_min(1e-24), torch.ones_like(norm))
    return w - eta_rest * clipped_full - (eta_kv - eta_rest) * scale[:, None, None] * raw_kv.detach()


def sequential_inner(model: TTTModel, h: torch.Tensor, y: torch.Tensor,
                      fast: list[torch.Tensor], etas: list[torch.Tensor], g_clip: float,
                      create_graph: bool, stats: GradStats, chunk: int,
                      idx: torch.Tensor | None = None,
                      factors: tuple[bool, bool] | None = None,
                      keep: torch.Tensor | None = None,
                      sel: torch.Tensor | None = None,
                      gradient_transform: Callable | None = None,
                      kv_sel: torch.Tensor | None = None,
                      eta_kv: torch.Tensor | None = None,
                      rule: str = "ce", observer: Callable | None = None) -> torch.Tensor:
    """Sequential chunk recurrence over per-sequence fast weights.

    Chunk ``c`` is predicted with ``W_c``, the weights from *before* its own
    targets are used, and only then ``W_{c+1} = W_c - eta * g_c`` with ``g_c`` the
    gradient of the chunk loss summed over the batch. ``create_graph`` keeps the
    inner graph, which is what makes the arm second order; with it off, ``W0``
    and ``eta`` still receive outer gradients because ``W_c`` stays a
    differentiable function of them.

    ``sel`` is the R4-2L inner-loss selector: the fast gradient comes from the
    linear token loss at those positions and CE elsewhere, while the outer loss
    stays the per-chunk CE at ``W_c`` over every token. ``keep`` is
    R4-2's outer mask. With neither, the code path is unchanged bit-for-bit.

    ``gradient_transform(c, inner_tokens, W, raw_grads, clipped_grads)`` is an
    optional first-order probe hook after scalar clipping. None uses the original
    gradients; no gradient of the clip is introduced by the default path.
    """
    T, b = h.shape[1], chunk
    if factors is not None and keep is not None:
        # the factor branch below returns before the mask is ever read, so a factor
        # arm with a mask would silently train unmasked; refuse the combination
        # instead of dropping the mask
        raise ValueError("a factor arm cannot take an outer mask")
    W, total, n_chunk = fast, 0.0, T // b
    if rule not in ("ce", "lin", "linkv", "linvW", "linxkv", "aice"):
        raise ValueError(f"unknown inner rule: {rule}")
    reference = reference_logits(model, h, idx, fast) if rule == "aice" else None
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    for c in range(n_chunk):
        sl = slice(c * b, (c + 1) * b)
        if factors is not None:
            loss_c, grads = factor_grads(model, h[:, sl], y[:, sl],
                                         model.head.ln(h[:, sl]) if x is None else x[:, sl],
                                         W, factors)
            outer = loss_c
        elif keep is None and sel is None and rule == "ce":
            ce = _ce_tokens(model, fast_block(h[:, sl], model.head.ln, W,
                                             None if x is None else x[:, sl]), y[:, sl])
            inner_tokens = ce
            loss_c = ce.mean(1)
            # The outer loss contains these chunk graphs, so retain them for backward.
            grads = torch.autograd.grad(loss_c.sum(), W, create_graph=create_graph,
                                        retain_graph=True)
            outer = loss_c
        else:
            z = model.read(fast_block(h[:, sl], model.head.ln, W,
                                      None if x is None else x[:, sl]))
            ce = token_ce(z, y[:, sl])
            inner_tokens = inner_token_loss(
                rule, z, y[:, sl], ce=ce, sel=None if sel is None else sel[:, sl],
                reference=None if reference is None else reference[:, sl])
            inner_c = inner_tokens.mean(1)
            grads = torch.autograd.grad(inner_c.sum(), W, create_graph=create_graph,
                                        retain_graph=True)
            # outer loss: the unmasked CE over every token, as in e2e1emb
            outer = (ce * keep[:, sl]).sum(1) / b if keep is not None else ce.mean(1)
        if observer is not None:
            observer(c, ce, h[:, sl], W, model, None if x is None else x[:, sl],
                     y[:, sl], None if keep is None else keep[:, sl])
        raw_grads = list(grads)
        if kv_sel is not None:
            if create_graph or factors is not None or gradient_transform is not None:
                raise ValueError("kv eta requires ordinary first-order gradients")
            gkv = torch.autograd.grad((inner_tokens * kv_sel[:, sl]).mean(1).sum(),
                                      W[2], retain_graph=True)[0]
        grads, norm, clipped = _clip_fast(raw_grads, g_clip)
        if gradient_transform is not None:
            # Eval-only first-order attribution, after the shared scalar clip.
            if create_graph or factors is not None:
                raise ValueError("gradient_transform requires plain first-order gradients")
            grads = gradient_transform(c, inner_tokens, W, raw_grads, grads)
        stats.add(norm, clipped, c)
        total = total + outer.mean()
        previous_W2 = W[2]
        W = [w - eta * g for w, eta, g in zip(W, etas, grads)]
        if kv_sel is not None:
            W[2] = kv_w2_step(previous_W2, grads[2], gkv, etas[2], eta_kv, norm, g_clip)
    return total / n_chunk


def factor_grads(model: TTTModel, h: torch.Tensor, y: torch.Tensor, x: torch.Tensor,
                 W: list[torch.Tensor], factors: tuple[bool, bool]
                 ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Manual per-sequence gradients with independent input/error factor graphs."""
    input_on, error_on = factors
    W1, b1, W2, b2 = W
    u = torch.einsum("btd,bmd->btm", x, W1) + b1[:, None]
    a = F.gelu(u)
    out = h + torch.einsum("btm,bdm->btd", a, W2) + b2[:, None]
    loss = _ce_tokens(model, out, y).mean(1)
    # The error factor keeps its full graph, including e's dependence on a through
    # out; a, s, x are cut only as outer-product factors. By the product rule the
    # input-only and error-only corrections then sum exactly to e2e2's per update.
    if error_on:
        e = torch.autograd.grad(loss.sum(), out, create_graph=True, retain_graph=True)[0]
        eps = torch.einsum("btd,bdm->btm", e, W2)
    else:
        e = torch.autograd.grad(loss.sum(), out, retain_graph=True)[0].detach()
        eps = torch.einsum("btd,bdm->btm", e, W2).detach()
    s = (0.5 * (1 + torch.erf(u / math.sqrt(2))) +
         u * torch.exp(-u.square() / 2) / math.sqrt(2 * math.pi))
    if not input_on:
        x, a, s = x.detach(), a.detach(), s.detach()
    r = eps * s
    grads = [torch.einsum("btm,btd->bmd", r, x), r.sum(1),
             torch.einsum("btd,btm->bdm", e, a), e.sum(1)]
    return loss, grads


def jacobi_inner(model: TTTModel, h: torch.Tensor, y: torch.Tensor,
                 slow: tuple[torch.Tensor, ...], etas: list[torch.Tensor], g_clip: float,
                 sweeps: int, stats: GradStats, chunk: int,
                 create_graph: bool = False) -> torch.Tensor:
    """Parallel Jacobi iteration of the chunk recurrence.

    Sweep ``j`` evaluates every chunk at once with its own weights ``W^(j)_c``,
    takes per-chunk gradients (detached for first-order arms), and rebuilds weights as
    ``W^(j+1)_c = W0 - eta * sum_{c' < c} g^(j)_{c'}`` (an exclusive cumulative
    sum over chunks). ``sweeps >= C - 1`` therefore reproduces the sequential
    first-order loop exactly, while one or two sweeps cost one or two passes.
    """
    B, T, d = h.shape
    C = T // chunk
    hv = h.reshape(B, C, chunk, d)
    yv = y.reshape(B, C, chunk)
    W = [p.unsqueeze(0).unsqueeze(0).expand(B, C, *p.shape) for p in slow]
    for _ in range(sweeps):
        leaves = W if create_graph else [w.detach().requires_grad_(True) for w in W]
        loss_c = _ce_tokens(model, fast_block(hv, model.head.ln, leaves), yv).mean(2)
        grads = torch.autograd.grad(loss_c.sum(), leaves, create_graph=create_graph)
        grads, norm, clipped = _clip_fast(list(grads), g_clip)
        stats.add(norm, clipped, -1)
        W = [p - eta * (g.cumsum(1) - g) for p, eta, g in zip(slow, etas, grads)]
    out = fast_block(hv, model.head.ln, W).reshape(B, T, d)
    return _ce_tokens(model, out, y).mean()


def arm_loss(model: TTTModel, arm: str, idx: torch.Tensor, y: torch.Tensor, eta_base: float,
             g_clip: float, chunk: int, stats: GradStats,
             keep: torch.Tensor | None = None,
             sel: torch.Tensor | None = None, *, meta: dict | None = None,
             observer: Callable | None = None) -> torch.Tensor:
    """Outer training loss of one arm on a fresh batch."""
    rule = arm_inner_loss(arm)
    if (B9_BASE.get(arm, arm) in LIN_ARMS or arm == "e2e1emb_kaminlin") and sel is None:
        raise ValueError(f"{arm} needs its inner-loss selector")
    h = model.backbone(idx, edge_mask=arm_edge_mask(model, arm, idx, meta))
    if arm in B9_BASE:
        from .aux_b9 import auxiliary, LAMBDA
        if meta is None:
            raise ValueError("B9 needs echo key/write/query metadata")
        outer = sequential_inner(model, h, y, model.per_sequence(idx.shape[0]), etas=regime_etas(model, eta_base),
                                 g_clip=g_clip, create_graph=False, stats=stats, chunk=chunk, idx=idx,
                                 keep=keep, sel=sel, rule=rule)
        return outer + LAMBDA*auxiliary(model, h, idx, meta, chunk,
                                       sham=arm == "e2e1_lin_b9f", log_grad=True)
    if arm in STATIC_ARMS:
        if keep is not None:
            raise ValueError("outer mask requires a sequential arm")
        return _ce_tokens(model, fast_block(h, model.head.ln, model.slow()), y).mean()
    etas = regime_etas(model, eta_base)
    if arm in SEQUENTIAL_ARMS:
        return sequential_inner(model, h, y, model.per_sequence(idx.shape[0]), etas, g_clip,
                                 arm in SECOND_ORDER_ARMS, stats, chunk, idx,
                                 FACTOR_ARMS.get(arm), keep, sel,
                                 kv_sel=(lin_selector("e2e1emb_linkv", (meta or {}).get("write_pos"),
                                                       len(idx), idx.shape[1], idx.device)
                                         if arm in KVETA_ARMS else None),
                                 eta_kv=model.eta_kvW2(eta_base) if arm in KVETA_ARMS else None,
                                 rule=rule, observer=observer)
    if keep is not None:
        raise ValueError("outer mask requires a sequential arm")
    return jacobi_inner(model, h, y, model.slow(), etas, g_clip, JACOBI_ARMS[arm], stats,
                        chunk, arm in SECOND_ORDER_ARMS)


def _outer_grad_vector(model: TTTModel, arm: str, idx: torch.Tensor, y: torch.Tensor,
                       eta_base: float, g_clip: float, chunk: int) -> torch.Tensor:
    """Flat outer gradient of one arm on one batch, over every trainable slow parameter."""
    model.zero_grad(set_to_none=True)
    grads = torch.autograd.grad(arm_loss(model, arm, idx, y, eta_base, g_clip, chunk, GradStats()),
                                tuple(model.parameters()), allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(grads, model.parameters())]).detach()


def ek_share(model_state: dict | None, task: Task, st: Settings, seed: int,
             n_seq: int = EK_SHARE_NSEQ, eta_base: float = 10.0,
             halves: tuple[str, ...] = tuple(EK_HALVES.values())
             ) -> dict:
    """Share of the second-order correction that one factor half carries.

    ``share_X = ||grad L_X - grad L_e2e1emb|| / ||grad L_e2e2emb - grad L_e2e1emb||`` over
    all trainable slow parameters, plus the cosine of the two differences and the
    relative metric ``||grad L_X - grad L_e2e1emb|| / ||grad L_e2e1emb||``. The batch is the
    first ``n_seq`` sequences of seed ``seed``'s first training batch, drawn from the same
    stream the cells consume (``rng([seed, TRAIN_SEED])``); no cell's stream is touched.

    CPU and fp64, with the run's own model size. The weights are built (or loaded) in the
    cells' own dtype and then cast to fp64, so the in-run gate measures exactly the weights
    a cell starts from and ``scripts/r43_ek_share_ckpt.py`` measures exactly the weights it
    saved. ``model_state`` of ``None`` means a fresh init, which is the in-run gate; a saved
    ``state_dict`` is loaded instead. At init the four arms are asserted to share the weights
    and the frozen code, so a difference in the numbers can only come from the arm. The gate
    (median share over seeds 0-2 >= ``EK_SHARE_MIN``) is judged by the analyser from the
    logged record, not here.
    """
    dev = torch.device("cpu")
    model = build_model(task, st, dev, seed, EK_ANCHOR)
    if model_state is None:
        for arm in EK_ARMS:
            other = build_model(task, st, dev, seed, arm)
            assert all(torch.equal(a, b) for a, b in
                       zip(model.state_dict().values(), other.state_dict().values())), arm
    else:
        model.load_state_dict(model_state)
    model = model.to(torch.float64)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        rng = np.random.default_rng([seed, TRAIN_SEED])
        idx, y, _ = task.tensors(*task.raw(rng, st.batch), dev)
        idx, y = idx[:n_seq], y[:n_seq]
        vectors = {arm: _outer_grad_vector(model, arm, idx, y, eta_base, st.g_clip, task.chunk)
                   for arm in EK_ARMS}
    finally:
        torch.set_default_dtype(old_dtype)
    first, correction = vectors[EK_ANCHOR], vectors[EK_REF] - vectors[EK_ANCHOR]
    first_norm, denom = float(first.norm()), float(correction.norm())
    out = dict(seed=seed, n_seq=n_seq, eta_base=eta_base, dtype="float64", device="cpu",
               first_order_norm=first_norm, correction_norm=denom,
               first_batch_sha1=hashlib.sha1(idx.cpu().numpy().tobytes()).hexdigest())
    for arm in halves:
        delta = vectors[arm] - first
        norm = float(delta.norm())
        out[arm] = dict(
            share=norm / denom if denom > 0 else None,
            cos=float(delta @ correction / (norm * denom)) if denom > 0 and norm > 0 else None,
            rel=norm / first_norm if first_norm > 0 else None)
    return out


def ek_share_record(st: Settings, task: Task, seeds: tuple[int, ...] = EK_SHARE_SEEDS,
                    n_seq: int = EK_SHARE_NSEQ, eta_base: float = 10.0) -> dict:
    """The in-run EK share record: one ``ek_share`` row per seed at init, plus the medians.

    Logged as a single ``R43_EK_SHARE`` line by ``run`` and read back by
    ``scripts/analyze_r43.py``; the per-seed values are in it so the analyser computes the
    gate itself. Never aborts a run.
    """
    per_seed = [ek_share(None, task, st, seed, n_seq, eta_base) for seed in seeds]
    row = dict(metric="||grad X - grad e2e1emb|| / ||grad e2e2emb - grad e2e1emb||",
               threshold=EK_SHARE_MIN, n_seq=n_seq, eta_base=eta_base, dtype="float64",
               arms=list(EK_ARMS), half_arms=dict(EK_HALVES), weights="trainable slow params",
               batch="first n_seq sequences of rng([seed, TRAIN_SEED]) at init",
               per_seed=per_seed)
    for half, arm in EK_HALVES.items():
        for key in ("share", "cos", "rel"):
            values = [r[arm][key] for r in per_seed if r[arm][key] is not None]
            row[f"median_{half}_{key}"] = float(np.median(values)) if values else None
    return row


def _fixed_etas(model: TTTModel, eta: float) -> list[torch.Tensor]:
    """Step sizes held fixed at ``eta`` (eval-time-only TTT, no learned ``rho``)."""
    return [model.rho.new_tensor(eta) for _ in FAST_KEYS]


@torch.no_grad()
def eval_sequential(model: TTTModel, idx: torch.Tensor, y: torch.Tensor,
                    etas: list[torch.Tensor], g_clip: float, chunk: int,
                    stats: GradStats, sel: torch.Tensor | None = None, *,
                    h: torch.Tensor | None = None,
                    inner_transform: Callable | None = None,
                    kv_sel: torch.Tensor | None = None,
                    eta_kv: torch.Tensor | None = None,
                    rule: str = "ce", freeze_from: int | None = None,
                    snapshot_chunk: int | None = None, observer: Callable | None = None) -> torch.Tensor:
    """Logits under the streaming inner loop: sequential chunks, first order.

    This is what test-time inference does, and what every TTT arm is evaluated
    with; chunk ``c`` is again predicted with the weights it had on arrival. With
    ``sel`` the arm's own token loss drives the inner writes.

    ``h`` can reuse detached backbone features. ``inner_transform(c, logits,
    targets, inner_tokens)`` optionally changes the inner loss, keeping the
    shared recurrence, pre-update logits, learned eta and clipping rule intact.
    """
    h = model.backbone(idx).detach() if h is None else h.detach()
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    T, b = h.shape[1], chunk
    W = [w.detach().requires_grad_(True) for w in model.per_sequence(idx.shape[0])]
    if rule not in ("ce", "lin", "linkv", "linvW", "linxkv", "aice"):
        raise ValueError(f"unknown inner rule: {rule}")
    reference = reference_logits(model, h, idx, W) if rule == "aice" else None
    snapshot = None
    out = []
    with torch.enable_grad():
        for c in range(T // b):
            sl = slice(c * b, (c + 1) * b)
            if snapshot is not None:
                W = snapshot
            logits = model.read(fast_block(h[:, sl], model.head.ln, W,
                                           None if x is None else x[:, sl]))
            out.append(logits.detach())
            ce = token_ce(logits, y[:, sl])
            if (isinstance(freeze_from, int) and c >= freeze_from) or snapshot is not None:
                continue
            inner = inner_token_loss(rule, logits, y[:, sl], ce=ce,
                                 sel=None if sel is None else sel[:, sl],
                                 reference=None if reference is None else reference[:, sl])
            if inner_transform is not None:
                inner = inner_transform(c, logits, y[:, sl], inner)
            grads = torch.autograd.grad(inner.mean(1).sum(), W, retain_graph=kv_sel is not None)
            if kv_sel is not None:
                gkv = torch.autograd.grad((inner * kv_sel[:, sl]).mean(1).sum(), W[2])[0]
            raw_grads = list(grads)
            if torch.is_tensor(freeze_from):
                active = (c < freeze_from).to(logits.dtype)
                raw_grads = [g * active.reshape(-1, *([1]*(g.ndim-1))) for g in raw_grads]
            grads, norm, clipped = _clip_fast(raw_grads, g_clip)
            if observer is not None:
                observer(c, logits.detach(), reference, W, raw_grads, grads, norm, clipped)
            stats.add(norm, clipped, c)
            previous_W2 = W[2]
            W = [(w - eta * g).detach().requires_grad_(True) for w, eta, g in zip(W, etas, grads)]
            if kv_sel is not None:
                W[2] = kv_w2_step(previous_W2, grads[2], gkv, etas[2], eta_kv,
                                  norm, g_clip).detach().requires_grad_(True)
            if snapshot_chunk is not None and c + 1 == snapshot_chunk:
                snapshot = W
    return torch.cat(out, dim=1)


@torch.no_grad()
def eval_jacobi(model: TTTModel, idx: torch.Tensor, y: torch.Tensor,
                etas: list[torch.Tensor], g_clip: float, chunk: int, sweeps: int,
                stats: GradStats) -> torch.Tensor:
    """Logits under the train-consistent parallel inner loop of a Jacobi arm."""
    h = model.backbone(idx).detach()
    B, T, d = h.shape
    C = T // chunk
    hv = h.reshape(B, C, chunk, d)
    yv = y.reshape(B, C, chunk)
    slow = model.slow()
    W = [p.unsqueeze(0).unsqueeze(0).expand(B, C, *p.shape) for p in slow]
    for _ in range(sweeps):
        leaves = [w.detach().requires_grad_(True) for w in W]
        with torch.enable_grad():
            loss_c = _ce_tokens(model, fast_block(hv, model.head.ln, leaves), yv).mean(2)
            grads = torch.autograd.grad(loss_c.sum(), leaves)
        grads, norm, clipped = _clip_fast(list(grads), g_clip)
        stats.add(norm, clipped)
        W = [p - eta * (g.cumsum(1) - g) for p, eta, g in zip(slow, etas, grads)]
    out = fast_block(hv, model.head.ln, W).reshape(B, T, d)
    return model.read(out)


def eval_batch_logits(model: TTTModel, task: Task, arm: str, idx: torch.Tensor, y: torch.Tensor,
                      eta: float, g_clip: float, chunk: int, stats: GradStats,
                      learned: bool = True, jacobi: bool = False,
                      meta: dict | None = None) -> torch.Tensor:
    """Logits of one batch under an arm's inference-time forward pass.

    ``fa``/``swa`` use the static head, ``e2e1``/``e2e2``/``naive`` the sequential
    first-order inner loop, and a Jacobi arm its own parallel loop when ``jacobi``
    is set. ``learned=False`` holds the step size fixed instead of reading ``rho``.
    The site-selective R4-2L arms need ``meta``; ``lin`` needs no position metadata.
    The streaming loop uses that arm's token loss at eval time too.
    """
    rule = arm_inner_loss(arm)
    if arm in STATIC_ARMS:
        h = model.backbone(idx)
        return model.read(model.fast_block(h, idx, model.slow()))
    etas = regime_etas(model, eta) if learned else _fixed_etas(model, eta)
    if jacobi:
        return eval_jacobi(model, idx, y, etas, g_clip, chunk, JACOBI_ARMS[arm], stats)
    sel = None
    if B9_BASE.get(arm, arm) in LIN_ARMS or arm == "e2e1emb_kaminlin":
        sel = lin_selector(arm, (meta or {}).get("write_pos"), idx.shape[0],
                           idx.shape[1], idx.device)
    mask = arm_edge_mask(model, arm, idx, meta)
    h = model.backbone(idx, edge_mask=mask).detach() if mask is not None else None
    kv_sel = (lin_selector("e2e1emb_linkv", (meta or {}).get("write_pos"), len(idx),
                            idx.shape[1], idx.device) if arm in KVETA_ARMS else None)
    return eval_sequential(model, idx, y, etas, g_clip, chunk, stats, sel, h=h,
                           kv_sel=kv_sel, eta_kv=model.eta_kvW2(eta).detach()
                           if learned and arm in KVETA_ARMS else
                           etas[2] if arm in KVETA_ARMS else None, rule=rule)


@torch.no_grad()
def picard_diagnostics(model: TTTModel, idx: torch.Tensor, y: torch.Tensor,
                       eta_base: float, g_clip: float, chunk: int) -> dict:
    """Compare three first-order Picard iterates to streaming fast weights."""
    h = model.backbone(idx).detach()
    B, T, d = h.shape
    C = T // chunk
    hv, yv = h.reshape(B, C, chunk, d), y.reshape(B, C, chunk)
    slow, etas = model.slow(), model.etas(eta_base)
    base = [p.detach().unsqueeze(0).unsqueeze(0).expand(B, C, *p.shape) for p in slow]
    W = [p[:, 0].detach() for p in base]
    seq = []
    with torch.enable_grad():
        for c in range(C):
            W = [w.detach().requires_grad_(True) for w in W]
            seq.append([w.detach() for w in W])
            loss = _ce_tokens(model, fast_block(hv[:, c], model.head.ln, W), yv[:, c]).mean(1)
            grads = torch.autograd.grad(loss.sum(), W)
            grads, _, _ = _clip_fast(list(grads), g_clip)
            W = [w - eta * g for w, eta, g in zip(W, etas, grads)]
        iters = [base]
        for _ in range(3):
            leaves = [w.detach().requires_grad_(True) for w in iters[-1]]
            loss = _ce_tokens(model, fast_block(hv, model.head.ln, leaves), yv).mean(2)
            grads = torch.autograd.grad(loss.sum(), leaves)
            grads, _, _ = _clip_fast(list(grads), g_clip)
            iters.append([(p - eta * (g.cumsum(1) - g)).detach()
                           for p, eta, g in zip(slow, etas, grads)])
    def flat(weights: list[torch.Tensor]) -> torch.Tensor:
        """Flatten the four fast tensors per batch item and chunk."""
        return torch.cat([w.reshape(B, C, -1) for w in weights], dim=-1)
    ref = flat([torch.stack([seq[c][i] for c in range(C)], 1) for i in range(4)])
    arrays = [flat(w) for w in iters]
    denom = (ref - arrays[0]).norm(dim=-1).clamp_min(1e-12)
    out = {}
    for k in (1, 2, 3):
        err = (arrays[k] - ref).norm(dim=-1)
        out[f"abs_err_{k}"] = err.mean(0).tolist()
        out[f"rel_err_{k}"] = (err / denom).mean(0).tolist()
    for k in (1, 2):
        # chunks c <= k are exact after k sweeps, so both differences there are
        # rounding noise and their ratio means nothing: report None, and the
        # median rate over the chunks c > k where the iteration is still moving
        delta = (arrays[k + 1] - arrays[k]).norm(dim=-1)
        prev = (arrays[k] - arrays[k - 1]).norm(dim=-1).clamp_min(1e-12)
        ratio = (delta / prev).mean(0).tolist()
        out[f"contraction_{k}"] = [None if c <= k else r for c, r in enumerate(ratio)]
        tail = ratio[k + 1:]
        out[f"contraction_{k}_median"] = float(np.median(tail)) if tail else None
    return out


def outer_grads(model: TTTModel, inner: str, sweeps: int, create_graph: bool,
                idx: torch.Tensor, y: torch.Tensor, eta_base: float, g_clip: float,
                chunk: int) -> torch.Tensor:
    """Flatten outer gradients over all parameters for either inner recurrence."""
    model.zero_grad(set_to_none=True)
    h = model.backbone(idx)
    etas = model.etas(eta_base)
    if inner == "sequential":
        loss = sequential_inner(model, h, y, model.per_sequence(len(idx)), etas, g_clip,
                                create_graph, GradStats(), chunk)
    elif inner == "jacobi":
        loss = jacobi_inner(model, h, y, model.slow(), etas, g_clip, sweeps,
                            GradStats(), chunk, create_graph)
    else:
        raise ValueError(f"unknown inner loop {inner}")
    loss.backward()
    flat = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else
                      torch.zeros_like(p).reshape(-1) for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return flat


def metagrad_fidelity(model: TTTModel, idx: torch.Tensor, y: torch.Tensor,
                      eta_base: float, g_clip: float, chunk: int) -> dict:
    """Compare Picard and first-order meta-gradients with sequential second order."""
    idx, y = idx[:8], y[:8]
    shapes = [(n, p.numel()) for n, p in model.named_parameters()]
    reference = outer_grads(model, "sequential", 0, True, idx, y, eta_base, g_clip, chunk)
    vectors = {"e2e2": reference,
               "e2e1": outer_grads(model, "sequential", 0, False, idx, y, eta_base, g_clip, chunk),
               "jac2": outer_grads(model, "jacobi", 2, False, idx, y, eta_base, g_clip, chunk)}
    # k = C - 1 is exact Picard, so its error is the numerical floor of the comparison
    n_chunks = idx.shape[1] // chunk
    errors = {}
    for k in sorted({1, 2, 3, 4, 8, n_chunks - 1} & set(range(1, n_chunks))):
        # one k running out of memory (k = C - 1 is the largest graph) must not
        # take the others down with it
        try:
            vectors[f"jac{k}so"] = outer_grads(model, "jacobi", k, True, idx, y,
                                                eta_base, g_clip, chunk)
        except Exception as exc:  # pragma: no cover - hardware dependent
            model.zero_grad(set_to_none=True)
            errors[f"jac{k}so"] = {"error": repr(exc)}
            if idx.device.type == "cuda":
                torch.cuda.empty_cache()
    offsets = np.cumsum([0] + [size for _, size in shapes])
    subsets = {"all": torch.arange(reference.numel(), device=reference.device)}
    for key, predicate in (("head", lambda n: n in {f"head.{k}" for k in FAST_KEYS}),
                           ("rho", lambda n: n == "rho")):
        subsets[key] = torch.cat([torch.arange(offsets[i], offsets[i + 1], device=reference.device)
                                  for i, (name, _) in enumerate(shapes) if predicate(name)])
    out = {}
    for name, vector in vectors.items():
        out[name] = {}
        for part, indices in subsets.items():
            a, b = vector[indices], reference[indices]
            out[name][part] = dict(rel_err=float((a - b).norm() / b.norm().clamp_min(1e-12)),
                                   cos=float(torch.dot(a, b) /
                                             (a.norm() * b.norm()).clamp_min(1e-12)))
    out.update(errors)
    return out


# --------------------------------------------------------------------------- metrics


def summarize_ce(ce: torch.Tensor, n_bins: int, late_frac: float, meta: dict | None = None,
                 preds: torch.Tensor | None = None, chunk: int = 64) -> dict:
    """Overall, late and per-bin CE of shape ``(n_seq, T)``, with standard errors.

    Bins split the ``T`` supervised positions into ``n_bins`` equal parts. Echo
    metadata adds ``query_ce`` and ``query_acc`` over the value token of each
    key's first query; ``preds`` holds the arg-max token per position and is only
    needed for that accuracy.
    """
    x = ce.detach().cpu().to(torch.float64).numpy()
    n_seq, T = x.shape
    edges = [round(i * T / n_bins) for i in range(n_bins + 1)]
    bins, se = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        per_seq = x[:, lo:hi].mean(1)
        bins.append(round(float(per_seq.mean()), 6))
        se.append(round(float(per_seq.std(ddof=1) / math.sqrt(n_seq)), 6))
    late = x[:, int(T * (1 - late_frac)):]
    late_per_seq = late.mean(1)
    out = dict(ce=round(float(x.mean()), 6), ce_late=round(float(late.mean()), 6),
               ce_late_se=round(float(late_per_seq.std(ddof=1) / math.sqrt(n_seq)), 6),
               bins=bins, bin_se=se, late_per_seq=[round(float(v), 6) for v in late_per_seq])
    if meta and preds is not None:
        pos = torch.from_numpy(np.asarray(meta["query_pos"]))
        tok = torch.from_numpy(np.asarray(meta["query_tok"]))
        cpu = ce.detach().to("cpu")
        q = torch.gather(cpu, 1, pos).to(torch.float64)
        out["query_ce"] = round(float(q.mean()), 6)
        out["query_ce_se"] = round(float(q.mean(1).std(0) / math.sqrt(n_seq)), 6)
        hit = preds.to("cpu").gather(1, pos) == tok
        out["query_acc"] = round(float(hit.to(torch.float64).mean()), 6)
        accuracy = hit.to(torch.float64).numpy()
        out["query_acc_per_seq"] = accuracy.mean(1).tolist()
        out["query_acc_se"] = (float(accuracy.mean(1).std(ddof=1) / math.sqrt(n_seq))
                               if n_seq > 1 else 0.0)
        qp = np.asarray(meta["query_pos"])
        wp = np.asarray(meta.get("write_pos", np.zeros_like(qp)))
        qce = q.numpy()

        def split(groups: np.ndarray) -> dict:
            """Aggregate already-scored queries by a metadata-derived label."""
            return {str(label): dict(acc=float(accuracy[groups == label].mean()),
                                     ce=float(qce[groups == label].mean()),
                                     n=int((groups == label).sum())) for label in np.unique(groups)}

        out["query_by_chunk"] = split(qp // chunk)
        if "write_pos" in meta:
            positions = np.stack([wp - 2, wp - 1, wp], axis=-1)
            out["write_ce"] = [float(x[np.arange(n_seq)[:, None], positions[:, :, i]].mean())
                               for i in range(3)]
            crosses = (wp - 2) // chunk != (wp + 1) // chunk
            out["query_by_crossing"] = split(crosses)
            out["query_by_write_chunk"] = split(wp // chunk)
    return out


def _gap(value: float | None, base: float | None, base_se: float, ref: float | None,
         ref_se: float, k: float = 2.0) -> float | None:
    """Share of the ``base - ref`` gap closed by ``value``, or ``None``.

    ``None`` when the reference gap itself is not separated from zero by ``k``
    standard errors: a ratio against a zero-width gap is noise, not a finding.
    """
    if value is None or base is None or ref is None:
        return None
    denom = base - ref
    if abs(denom) < 1e-9 or abs(denom) <= k * math.hypot(base_se, ref_se):
        return None
    return round((base - value) / denom, 6)


# --------------------------------------------------------------------------- training


def lr_scale(step: int, warmup: int, total: int, floor: float) -> float:
    """Linear warmup, then cosine decay from 1 down to ``floor``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def clipped_params(model: TTTModel) -> list[nn.Parameter]:
    """Every parameter except ``rho``, the step-size logits.

    ``rho`` is a hyperparameter, not a network weight: clipping its gradient
    together with the trunk would let a large trunk gradient shrink the step size
    update, which is exactly the update the pilot and the eta grid control.
    """
    return [p for n, p in model.named_parameters() if n not in ("rho", "rho_kvW2")]


def build_optim(model: TTTModel, st: Settings) -> torch.optim.AdamW:
    """AdamW: matrix decay, separate no-decay gamma/eta learning rates.

    Gamma remains in ``clipped_params`` with the network; rho is excluded.
    Without a kv step-size parameter, rho is the last group (eta-LR logging reads
    it there); the kveta arms append their kv step-size group after rho.
    """
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and n != "rho"]
    plain = [p for n, p in model.named_parameters()
             if p.ndim < 2 and n not in ("rho", "code_gate", "rho_kvW2")]
    groups = [{"params": decay, "weight_decay": st.weight_decay, "base_lr": st.lr},
              {"params": plain, "weight_decay": 0.0, "base_lr": st.lr}]
    if model.code_gate is not None:
        groups.append({"params": [model.code_gate], "weight_decay": 0.0,
                       "base_lr": st.gamma_lr})
    groups.append({"params": [model.rho], "weight_decay": 0.0, "base_lr": st.eta_lr})
    if hasattr(model, "rho_kvW2"):
        groups.append({"params": [model.rho_kvW2], "weight_decay": 0., "base_lr": model.kv_eta_lr})
    return torch.optim.AdamW(groups, lr=st.lr, betas=(st.beta1, st.beta2))


@torch.no_grad()
def quick_summary(model: TTTModel, task: Task, arm: str, eta_base: float, st: Settings,
                  quick: list[tuple[torch.Tensor, torch.Tensor, dict]],
                  jacobi: bool = False) -> dict:
    """Quick-eval metrics under the arm's own forward: overall CE and query CE.

    Both numbers are returned because they answer different questions: overall CE
    is what the model spends capacity on, and ``query_ce`` is the echo task's
    actual target. A step size can lower overall CE while leaving the queries
    untouched, which is why the pilot ranks the echo arms on ``query_ce``.
    ``jacobi`` evaluates a Jacobi arm under its own parallel forward instead.
    """
    was_training = model.training
    model.eval()
    parts = _eval_parts(model, task, arm, quick, eta_base, st, GradStats(), jacobi=jacobi)
    model.train(was_training)
    meta = _concat_meta([m for _, _, m in quick])
    summ = _summarize_parts(parts, st, meta, task.chunk)
    return dict(ce=summ["ce"], query_ce=summ.get("query_ce"),
                query_acc=summ.get("query_acc"))


def quick_loss(model: TTTModel, task: Task, arm: str, eta_base: float, st: Settings,
               quick: list[tuple[torch.Tensor, torch.Tensor, dict]]) -> float:
    """Token-weighted mean CE on the quick-eval set under the arm's own forward."""
    return quick_summary(model, task, arm, eta_base, st, quick)["ce"]


def _concat_meta(metas: list[dict]) -> dict:
    """Stack per-batch echo metadata back into one whole-set metadata dict."""
    keys = ("query_pos", "query_tok", "write_pos", "write_start", "corrupted", "corruption_pos")
    out = {k: np.concatenate([m[k] for m in metas if k in m]) for k in keys
           if all(k in m for m in metas)}
    return out if out else {}


def train_arm(model: TTTModel, task: Task, arm: str, eta_base: float, st: Settings, dev: torch.device,
              rng: np.random.Generator, quick: list[tuple[torch.Tensor, torch.Tensor, dict]] | None,
              tag: str, steps: int | None = None, seed: int | None = None) -> dict:
    """Train one arm; report timing, the loss curve and inner-gradient health."""
    check_arm_task(arm, task.mode)
    n_steps = int(task.steps if steps is None else steps)
    opt = build_optim(model, st)
    stats = GradStats()
    times: list[float] = []
    quick_log: list[list[float]] = []
    quick_par: list[list[float]] = []
    probe_log: list[dict] = []
    saved_steps: list[str] = []
    probe_data = (eval_batches(task, *task.raw(np.random.default_rng([PROBE_SEED, 0]), st.n_probe),
                               dev, st.eval_batch) if st.probe_every and
                  task.mode in PROBE_MODES else None)
    outer_norms: list[float] = []
    outer_steps: list[int] = []
    rho_norms: list[float] = []
    quick_norm_start = 0
    nonfinite_steps = 0
    first_nonfinite = None
    diverged = False
    spike_steps: list[int] = []
    step_losses: list[float] = []
    loss_val = float("nan")
    mask = MASK_ARMS.get(arm, "none")
    mask_rng = np.random.default_rng([seed, MASK_SEED]) if mask == "rnd" and seed is not None else None
    if mask == "rnd" and mask_rng is None:
        raise ValueError("random mask requires the cell seed")
    first_batch_sha1 = None
    masked_per_seq = None
    lin_per_seq = None
    peak_mps_mb = 0.0
    warmup = arm_warmup(arm, st)
    save_steps = checkpoint_plan(st, arm, seed)
    if arm in NOISE_ARMS and seed is None:
        raise ValueError("corruption requires a cell seed")
    if arm in COPY_Q and st.copy_noise_q != list(COPY_Q.values()):
        raise ValueError("copy_noise_q must match the four binding arm names")
    if arm in KAMIN_ARMS and st.kamin_switch < 1:
        raise ValueError("kamin_switch must be positive")
    rho_init = model.rho.detach().clone() if arm in KAMIN_ARMS else None
    switch_record = None
    first_clean_batch_sha1 = None
    corrupted_quick_log, kv_eta_log = [], []
    aux_log = []
    corrupted_quick = (corrupted_batches(task, st, arm, dev, "quick")
                       if arm in NOISE_ARMS and quick else None)
    corrupted_probe = None
    if arm in NOISE_ARMS and probe_data:
        raw, pm = task.raw(np.random.default_rng([PROBE_SEED, 0]), st.n_probe)
        raw, pm = corrupt_echo(raw, pm, arm, CORRUPTION_EVAL_SEED, 2,
                               rf_probability=st.rf_probability)
        corrupted_probe = eval_batches(task, raw, pm, dev, st.eval_batch)

    def probe_now():
        p = r4_probe(model, task, probe_data, eta_base, st.g_clip, arm=arm)
        if corrupted_probe is not None:
            p.update(kv_w0_stats(model, task, corrupted_probe, arm))
        return p

    def extra_quick(completed):
        if arm in KVETA_ARMS:
            kv_eta_log.append(dict(step=completed, eta_kvW2=float(model.eta_kvW2(eta_base).detach()),
                                   eta_W2=float(model.etas(eta_base)[2].detach())))
        if corrupted_quick is not None:
            data, cm = corrupted_quick
            with torch.no_grad():
                parts = _eval_parts(model, task, arm, data, eta_base, st, GradStats())
                row = dict(step=completed, **corruption_recall(parts, cm))
            corrupted_quick_log.append(row)
            print("CORRUPTED_QUICK " + json.dumps(dict(cell_id=tag, **row)), flush=True)

    def sample_memory() -> None:
        nonlocal peak_mps_mb
        if dev.type == "mps":
            peak_mps_mb = max(peak_mps_mb, torch.mps.driver_allocated_memory() / 2**20)
    every = max(1, st.quick_every)
    total = st.lr_total or n_steps
    if total < n_steps:
        raise ValueError(f"lr_total {total} is shorter than the run ({n_steps} steps)")
    for step in range(n_steps):
        set_kamin_regime(model, arm, step, st)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * lr_scale(step, warmup, total, st.lr_floor)
        raw, meta = task.raw(rng, st.batch)
        if first_clean_batch_sha1 is None:
            first_clean_batch_sha1 = hashlib.sha1(raw[:, :task.T].tobytes()).hexdigest()
        raw, meta = corrupt_echo(raw, meta, arm, seed, step, rf_probability=st.rf_probability)
        idx, y, meta = task.tensors(raw, meta, dev)
        keep = outer_keep(meta, tuple(y.shape), mask, mask_rng, dev)
        sel = lin_selector(arm, meta.get("write_pos"), len(idx), y.shape[1], dev)
        if first_batch_sha1 is None:
            first_batch_sha1 = hashlib.sha1(idx.cpu().numpy().tobytes()).hexdigest()
            counts = (keep == 0).sum(1).tolist() if keep is not None else [0] * len(idx)
            if len(set(counts)) != 1 or counts[0] != (task.n_keys if keep is not None else 0):
                raise AssertionError(f"wrong masked count: {counts}")
            masked_per_seq = counts[0]
            lin_counts = sel.sum(1).tolist() if sel is not None else [0] * len(idx)
            want = lin_count(arm, y.shape[1], task.n_keys)
            if len(set(lin_counts)) != 1 or lin_counts[0] != want:
                raise AssertionError(f"wrong lin count for {arm}: {lin_counts}, want {want}")
            lin_per_seq = lin_counts[0]
        _sync(dev)
        t0 = time.perf_counter()
        loss = arm_loss(model, arm, idx, y, eta_base, st.g_clip, task.chunk, stats, keep, sel, meta=meta)
        if arm in B9_BASE:
            aux_row = dict(step=step+1, **model.b9_last_metrics)
            aux_log.append(aux_row)
            if (step+1) % 25 == 0 or step+1 == n_steps:
                print("AUX_B9 " + json.dumps(dict(cell_id=tag, **aux_row)), flush=True)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss).item():
            loss.backward()
            if arm in KAMIN_ARMS and step < st.kamin_switch:
                # None prevents AdamW moments or weight decay from touching rho.
                model.rho.grad = None
            # one sync per step: the clip's total norm is non-finite iff some clipped
            # gradient is, and rho (outside the clip) is checked alongside it; a
            # non-finite clip rescales the gradients, but they are dropped below
            norm = torch.nn.utils.clip_grad_norm_(clipped_params(model), st.grad_clip)
            rho_norm = (model.rho.grad.norm() if model.rho.grad is not None
                        else torch.zeros((), device=norm.device)).to(norm.device)
            if arm in KVETA_ARMS and model.rho_kvW2.grad is not None:
                rho_norm = (rho_norm.square() + model.rho_kvW2.grad.square()).sqrt()
            finite = bool(torch.isfinite(torch.stack([norm.float(), rho_norm.float()])).all())
            if finite:
                value = float(norm.detach())
                spike = False
                if st.spike_skip > 0 and step >= warmup:
                    recent = [v for s, v in zip(outer_steps[-50:], outer_norms[-50:])
                              if s >= warmup]
                    spike = len(recent) >= 10 and value > st.spike_skip * float(np.median(recent))
                outer_norms.append(value)
                outer_steps.append(step)
                rho_norms.append(float(rho_norm.detach()))
                if spike:
                    # the spike probe: drop this update entirely (Adam moments untouched)
                    spike_steps.append(step)
                    opt.zero_grad(set_to_none=True)
                else:
                    opt.step()
        else:
            finite = False
        if not finite:
            nonfinite_steps += 1
            first_nonfinite = first_nonfinite or step + 1
            opt.zero_grad(set_to_none=True)
            print(f"  WARNING {tag}: non-finite step {step + 1}", flush=True)
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
        loss_val = float(loss.detach())
        step_losses.append(loss_val)
        if arm in KAMIN_ARMS and step < st.kamin_switch:
            assert torch.equal(model.rho.detach(), rho_init), "rho changed in Kamin stage 1"
        if arm in KAMIN_ARMS and step + 1 == st.kamin_switch:
            # The W0-only manipulation check is before the first stage-2 update.
            assert torch.equal(model.rho.detach(), rho_init), "rho(switch) != rho(0)"
            switch_data = probe_data or quick
            if not switch_data:
                raise ValueError("Kamin switch needs a fixed probe or quick set")
            on = kv_w0_stats(model, task, switch_data, "e2e1emb")["p_kv_mean"]
            switch_record = dict(step=step + 1, p_kv_at_switch=on, passed=on >= .99,
                                  rho_unchanged=True, rho_init=rho_init.tolist(),
                                  rho_at_switch=model.rho.detach().tolist())
            if arm == "e2e1emb_kaminctl":
                switch_record["p_kv_masked_at_switch"] = kv_w0_stats(
                    model, task, switch_data, arm)["p_kv_mean"]
            print("KAMIN_SWITCH " + json.dumps(dict(cell_id=tag, **switch_record)), flush=True)
        # Checkpoints/quick evals labelled switch are at the stage-2 boundary;
        # switch_record above still measures the completed stage-1 model at W0.
        set_kamin_regime(model, arm, step + 1, st)
        if quick and (step + 1) % every == 0:
            summary = quick_summary(model, task, arm, eta_base, st, quick)
            sample_memory()
            etas = [float(e.detach()) for e in model.etas(eta_base)]
            window_norms = outer_norms[quick_norm_start:]
            row = [step + 1, summary["ce"], summary["query_ce"], summary["query_acc"],
                   loss_val, float(np.median(window_norms)) if window_norms else None, *etas]
            quick_log.append(row)
            extra_quick(step + 1)
            quick_norm_start = len(outer_norms)
            if st.quick_par and arm in JACOBI_ARMS:
                par = quick_summary(model, task, arm, eta_base, st, quick, jacobi=True)
                quick_par.append([step + 1, par["query_ce"], par["query_acc"]])
            print(f"  {tag} step {step + 1}/{n_steps} train={loss_val:.4f} "
                  f"quick={summary['ce']:.4f} q={summary['query_ce']} acc={summary['query_acc']} "
                  f"gnorm={row[5]} eta={etas}", flush=True)
        if probe_data and (step + 1) % st.probe_every == 0:
            probe = dict(step=step + 1, lr=st.lr * lr_scale(step, warmup, total, st.lr_floor),
                         **probe_now())
            probe_log.append(probe)
            print("PROBE " + json.dumps(dict(cell_id=tag, **probe)), flush=True)
        if step + 1 in save_steps:
            saved_steps.append(save_checkpoint(model, f"{tag}__step{step + 1}"))
        if step + 1 >= 20 and nonfinite_steps / (step + 1) > 0.05:
            diverged = True
            print(f"  WARNING {tag}: diverged after {step + 1} steps", flush=True)
            break
    final = quick_summary(model, task, arm, eta_base, st, quick) if quick else {}
    if quick:
        sample_memory()
    if probe_data and (not probe_log or probe_log[-1]["step"] != step + 1):
        probe = dict(step=step + 1, lr=st.lr * lr_scale(step, warmup, total, st.lr_floor),
                     **probe_now())
        probe_log.append(probe)
        print("PROBE " + json.dumps(dict(cell_id=tag, **probe)), flush=True)
    if quick and (not quick_log or quick_log[-1][0] != step + 1):
        etas = [float(e.detach()) for e in model.etas(eta_base)]
        quick_log.append([step + 1, final["ce"], final["query_ce"], final["query_acc"],
                          loss_val, float(np.median(outer_norms[quick_norm_start:]))
                          if outer_norms[quick_norm_start:] else None,
                          *etas])
        extra_quick(step + 1)
        print(f"  {tag} step {step + 1}/{n_steps} train={loss_val:.4f} "
              f"quick={final['ce']:.4f} q={final['query_ce']} acc={final['query_acc']} "
              f"gnorm={quick_log[-1][5]} eta={etas}", flush=True)
    if (st.quick_par and quick and arm in JACOBI_ARMS and
            (not quick_par or quick_par[-1][0] != step + 1)):
        par = quick_summary(model, task, arm, eta_base, st, quick, jacobi=True)
        quick_par.append([step + 1, par["query_ce"], par["query_acc"]])
    final_quick = final.get("ce", float("nan"))
    lo = min(100, max(1, n_steps // 2))
    window = times[lo:] or times
    # outer-gradient statistics cover the finite steps after warmup (all of them
    # when a short run never leaves warmup)
    post = [i for i, s in enumerate(outer_steps) if s >= warmup] or list(range(len(outer_steps)))
    norms_post = np.asarray([outer_norms[i] for i in post], dtype=np.float64)
    rho_post = np.asarray([rho_norms[i] for i in post], dtype=np.float64)
    out = dict(steps=n_steps, ms_step=round(float(np.median(window)), 3), ms_step_from=lo,
                 mask=mask, mask_seed=MASK_SEED if mask == "rnd" else None,
                 masked_per_seq=masked_per_seq, first_batch_sha1=first_batch_sha1,
                 inner_loss=arm_inner_loss(arm), lin_per_seq=lin_per_seq, warmup=warmup,
                 peak_mps_mb=round(peak_mps_mb, 1),
                 # which second-order / factor path actually ran (an unregistered
                 # arm would otherwise train as something else without a word)
                 create_graph=arm in SECOND_ORDER_ARMS,
                 factors=(list(FACTOR_ARMS[arm]) if arm in FACTOR_ARMS else None),
                 tf32=bool(st.tf32),
                 peak_cuda_mb=round(_peak_mem(dev), 1) if dev.type == "cuda" else 0.0,
                 train_loss_last=round(loss_val, 6), quick=quick_log,
                 probes=probe_log, step_checkpoints=saved_steps,
                 quick_final=round(final_quick, 6), grad=stats.summary(),
                 nonfinite_steps=nonfinite_steps, first_nonfinite=first_nonfinite,
                 diverged=diverged,
                 outer_grad=dict(median=float(np.median(norms_post)) if post else None,
                                 max=float(norms_post.max()) if post else None,
                                 clip_fraction=float(np.mean(norms_post > st.grad_clip))
                                 if post else None,
                                 rho_median=float(np.median(rho_post)) if post else None,
                                 from_step=warmup,
                                 # the ten largest finite post-warmup steps (0-indexed step,
                                 # pre-clip norm); quick-log rows count completed steps
                                 top=[[outer_steps[i], outer_norms[i]]
                                      for i in sorted(post, key=lambda i: -outer_norms[i])[:10]],
                                 rho_top=[[outer_steps[i], rho_norms[i]]
                                          for i in sorted(post, key=lambda i: -rho_norms[i])[:10]]),
                 spike_skips=len(spike_steps), spike_skip_steps=spike_steps[:100],
                 lr_total=total)
    if st.quick_par:
        out["quick_par"] = quick_par
    if arm in NOISE_ARMS:
        out.update(first_clean_batch_sha1=first_clean_batch_sha1,
                   copy_q=COPY_Q.get(arm), corruption_eval_seed=CORRUPTION_EVAL_SEED,
                   corrupted_quick=corrupted_quick_log)
    if arm in B9_BASE:
        out["aux_log"] = aux_log
    if arm in KVETA_ARMS:
        out.update(eta_kvW2_trajectory=kv_eta_log, kv_eta_lr=model.kv_eta_lr,
                   eta_kvW2_final=float(model.eta_kvW2(eta_base).detach()))
    if arm in KAMIN_ARMS:
        bound = [r[0] for r in quick_log if r[0] > st.kamin_switch and
                 r[3] is not None and r[3] >= .90]
        out.update(kamin_switch=st.kamin_switch, switch_check=switch_record,
                   stage2_ever_bound=bool(bound), stage2_first_bound=min(bound) if bound else None,
                   rho_stage1_unchanged=torch.equal(model.rho.detach(), rho_init)
                   if n_steps <= st.kamin_switch else bool(switch_record and switch_record["rho_unchanged"]))
    # per-step series for the cells mode to save beside the checkpoint; callers pop it
    # so it never reaches the JSON lines
    out["_trace"] = dict(step=np.asarray(outer_steps, dtype=np.int32),
                         outer_norm=np.asarray(outer_norms, dtype=np.float32),
                         rho_norm=np.asarray(rho_norms, dtype=np.float32),
                         loss=np.asarray(step_losses, dtype=np.float32),
                         spike_step=np.asarray(spike_steps, dtype=np.int32))
    if final.get("query_ce") is not None:
        out["quick_query_ce"] = final["query_ce"]
    _clip_warning(tag, out["grad"], out)
    return out


def _clip_warning(tag: str, grad: dict | None, out: dict | None = None) -> None:
    """Warn when at least 10% of inner gradients hit the clip, at train or eval time.

    ``g_clip`` stops the inner step from exploding, so a high fraction means the
    clipped steps are not the updates the algebra describes.
    """
    clipped = (grad or {}).get("clipped_frac")
    if clipped is not None and clipped >= 0.1:
        print(f"  WARNING {tag}: inner-gradient clip fraction {clipped:.3f} >= 0.10", flush=True)
        if out is not None:
            out["clip_warning"] = True


def pilot_eta_base(task: Task, st: Settings, dev: torch.device, seed: int,
                   quick: list[tuple[torch.Tensor, torch.Tensor, dict]]) -> tuple[float, dict]:
    """Pick ``eta_base`` by training ``e2e1`` briefly at each candidate value.

    The winner is used by every TTT arm of this task and seed. ``icl_mc`` is
    ranked by overall CE, but the echo tasks are ranked by ``query_ce``: queries
    are a few percent of the tokens, so a step size that buys filler-token CE
    while leaving the queries where they were can win on overall CE and lose the
    experiment. Both numbers are printed either way.
    """
    if not st.pilot or not st.pilot_etas:
        return 1.0, {}
    by_query = task.mode in ECHO_MODES
    scores: dict[float, dict] = {}
    for eta in st.pilot_etas:
        model = build_model(task, st, dev, seed, "e2e1")
        res = train_arm(model, task, "e2e1", eta, dataclasses.replace(st, lr_total=0), dev,
                        np.random.default_rng([seed, TRAIN_SEED]), quick, f"pilot eta={eta:g}",
                        steps=st.pilot_steps)
        res.pop("_trace", None)
        scores[float(eta)] = dict(ce=res["quick_final"],
                                  query_ce=res.get("quick_query_ce"))
    key = "query_ce" if by_query else "ce"
    ranked = [k for k in scores if scores[k].get(key) is not None] or list(scores)
    best = float(min(ranked, key=lambda k: scores[k][key]))
    detail = " ".join(
        f"{k:g}:ce={v['ce']:.4f}" + (f",q={v['query_ce']:.4f}" if v["query_ce"] is not None else "")
        for k, v in scores.items())
    print(f"  pilot eta_base [rank by {key}] {detail} -> {best:g}", flush=True)
    return best, {f"{k:g}": v for k, v in scores.items()}


# --------------------------------------------------------------------------- orchestration


def _jobs(st: Settings) -> list[tuple[str, int, list[str]]]:
    """Execution order: every task at every seed, then the extra seed jobs.

    The shipped order (icl_mc seed 0, then the echo tasks, then the icl_mc seed 1
    extras) means a budget stop costs the least important runs.
    """
    jobs: list[tuple[str, int, list[str]]] = []
    for name in st.tasks:
        arms = list(st.task_table(name).get("arms", st.arms))
        for seed in st.seeds:
            jobs.append((name, int(seed), arms))
    for job in st.extra_jobs:
        jobs.append((str(job["task"]), int(job["seed"]), list(job["arms"])))
    return jobs


def _task(st: Settings, name: str) -> Task:
    """Build one task from the merged settings and its sub-table."""
    table = st.task_table(name)
    mode = {"icl_mc": "mc", **{mode: mode for mode in ECHO_MODES}}[name]
    T, chunk = int(table.get("T", st.T)), int(table.get("chunk", st.chunk))
    window = int(table.get("window", st.window))
    if T % chunk:
        raise ValueError(f"task {name}: T={T} is not a multiple of chunk={chunk}")
    if chunk > window and mode == "mc":
        raise ValueError(f"task {name}: chunk={chunk} exceeds window={window}")
    n_keys = int(table.get("n_keys", st.n_keys))
    V = int(table.get("n_vocab", st.n_vocab)) if mode == "mc" else EchoVocab(n_keys).size
    return Task(name=name, V=V, T=T, window=window, chunk=chunk,
                steps=int(table.get("steps", st.steps)),
                n_eval=int(table.get("n_eval", st.n_eval)), n_val=int(table.get("n_val", st.n_val)),
                mode=mode, alpha=float(table.get("alpha", st.alpha)), n_keys=n_keys,
                n_layers=int(table.get("n_layers", st.n_layers)),
                prefix_min=int(table.get("prefix_min", st.prefix_min)),
                prefix_max=int(table.get("prefix_max", st.prefix_max)))


def _ceilings(task: Task, st: Settings) -> dict:
    """Exact Bayes ceilings on the fixed eval set.

    Three curves: full context (``bayes``), the window a single attention layer
    sees (``bayes_w64``, ``w = 64``) and the window the whole stack actually sees
    (``bayes_weff``, ``n_layers * (window - 1) + 1``). ``bayes_weff`` is the
    ceiling a sliding-window arm can really approach, so it is the one the
    ``gap_closed_bayes_late`` ratios use.
    """
    tokens, _ = fixed_set(task, st, "eval")
    full, w64, weff = [], [], []
    for lo in range(0, tokens.shape[0], 64):
        f, w = bayes_ce(tokens[lo:lo + 64], task.V, st.alpha, task.window)
        full.append(f)
        w64.append(w)
        if task.receptive_field == task.window:
            weff.append(w)
        else:
            weff.append(bayes_ce(tokens[lo:lo + 64], task.V, st.alpha, task.receptive_field)[1])
    return {"bayes": summarize_ce(torch.from_numpy(np.concatenate(full)), st.n_bins, st.late_frac),
            "bayes_w64": summarize_ce(torch.from_numpy(np.concatenate(w64)), st.n_bins,
                                     st.late_frac),
            "bayes_weff": summarize_ce(torch.from_numpy(np.concatenate(weff)), st.n_bins,
                                      st.late_frac),
            "w64": task.window, "weff": task.receptive_field}


@torch.no_grad()
def evaluate(model: TTTModel, task: Task, arm: str, data: list, meta: dict, eta_base: float,
             st: Settings, learned: bool = True) -> dict:
    """Evaluate an arm on fixed batches: CE summary, query metric, gradient health.

    No autograd graph is kept: the ``fa``/``swa`` arms would otherwise retain a
    graph per batch, and holding sixteen batches of ``(B, T, V)`` logits with
    graph is what OOMs a GPU at the full-scale config. The inner loops are
    unaffected because ``eval_sequential`` and ``eval_jacobi`` re-enable grad
    internally for their own gradients.
    """
    stats = GradStats()
    out = {"eval": _summarize_parts(_eval_parts(model, task, arm, data, eta_base, st, stats,
                                                learned), st, meta, task.chunk),
           "eval_grad": stats.summary()}
    if stats.norms:
        by_chunk = {}
        write, other = [], []
        for batch_index, (_, _, batch_meta) in enumerate(data):
            for c in range(task.T // task.chunk):
                norm = stats.norms[batch_index * (task.T // task.chunk) + c].cpu().numpy()
                by_chunk.setdefault(c, []).extend(norm.tolist())
                if "write_pos" in batch_meta:
                    mask = (batch_meta["write_pos"] // task.chunk == c).any(1)
                    write.extend(norm[mask].tolist())
                    other.extend(norm[~mask].tolist())
        out["eval_grad_by_chunk"] = {str(k): float(np.median(v)) for k, v in by_chunk.items()}
        if "write_pos" in meta:
            out["eval_grad_write"] = float(np.median(write)) if write else None
            out["eval_grad_other"] = float(np.median(other)) if other else None
    if arm in JACOBI_ARMS:
        stats_j = GradStats()
        out["eval_jacobi"] = _summarize_parts(
            _eval_parts(model, task, arm, data, eta_base, st, stats_j, learned, jacobi=True),
            st, meta, task.chunk)
        out["eval_jacobi_grad"] = stats_j.summary()
    _clip_warning(f"eval {task.name}/{arm}", out.get("eval_grad"))
    return out


def _eval_parts(model: TTTModel, task: Task, arm: str, data: list, eta_base: float,
                st: Settings, stats: GradStats, learned: bool = True,
                jacobi: bool = False) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Detached per-batch ``(ce, argmax)`` pairs, logits freed batch by batch.

    Lazy evaluation matters for memory: only the ``(B, T)`` cross-entropy and the
    ``(B, T)`` predicted-token id survive a batch, never the ``(B, T, V)`` logits.
    """
    parts = []
    for idx, y, meta in data:
        logits = eval_batch_logits(model, task, arm, idx, y, eta_base, st.g_clip, task.chunk,
                                   stats, learned, jacobi, meta)
        parts.append((token_ce(logits, y).detach(), logits.detach().argmax(-1)))
        del logits
    return parts


def _summarize_parts(parts: list[tuple[torch.Tensor, torch.Tensor]], st: Settings,
                     meta: dict, chunk: int = 64) -> dict:
    """Concatenate evaluated batches and summarise them with the task metric."""
    ce, preds = zip(*parts, strict=True)
    return summarize_ce(torch.cat(ce, dim=0), st.n_bins, st.late_frac, meta,
                        torch.cat(preds, dim=0), chunk)


def _task_gaps(tres: dict) -> None:
    """Fill gap-closed fractions for every arm of a task.

    ``gap_closed_late`` normalises by the sliding-window arm of the same seed and
    by the full-attention arm. A seed without ``fa`` (the seed-1 extras) reuses
    the nearest available ``fa`` as its reference, recorded in ``fa_ref_seed``.
    Every ratio is ``None`` unless its denominator clears two standard errors,
    so an unseparated reference cannot produce a meaningless number.
    """
    runs = tres.get("runs", {})
    ceiling = tres.get("ceiling")
    fa_late = {key: (r["fa"]["eval"]["ce_late"], r["fa"]["eval"]["ce_late_se"])
               for key, r in runs.items()
               if "fa" in r and not r["fa"].get("skipped") and "eval" in r["fa"]}
    if not fa_late:
        return
    bayes_late = bayes_weff_late = None
    if ceiling:
        bayes_late = ceiling["bayes"]["ce_late"]
        bayes_weff_late = ceiling["bayes_weff"]["ce_late"]
    for key, results in sorted(runs.items()):
        swa = results.get("swa")
        if not swa or swa.get("skipped") or "eval" not in swa:
            continue
        swa_late, swa_se = swa["eval"]["ce_late"], swa["eval"]["ce_late_se"]
        ref_key = key if key in fa_late else min(fa_late)
        fa_ref, fa_ref_se = fa_late[ref_key]
        for arm, res in results.items():
            if res.get("skipped") or "eval" not in res:
                continue
            res["fa_ref_seed"] = int(ref_key.replace("seed", ""))
            res["gap_closed_late_deprecated"] = _gap(res["eval"]["ce_late"], swa_late, swa_se,
                                                      fa_ref, fa_ref_se)
            res["gap_closed_bayes_late"] = (
                _gap(res["eval"]["ce_late"], bayes_weff_late,
                     ceiling["bayes_weff"]["ce_late_se"], bayes_late,
                     ceiling["bayes"]["ce_late_se"]) if ceiling else None)
            if "eval_jacobi" in res:
                res["gap_closed_late_jacobi_deprecated"] = _gap(
                    res["eval_jacobi"]["ce_late"], swa_late, swa_se, fa_ref, fa_ref_se)


def _arm_summary(res: dict) -> str:
    """One human-readable line for an arm, printed before its JSON record."""
    ev = res.get("eval")
    if not ev:
        return f"    {res['arm']}: skipped ({res.get('reason', '?')})"
    bits = [f"ce={ev['ce']:.4f}", f"late={ev['ce_late']:.4f}"]
    for key in ("query_ce", "query_acc"):
        if key in ev:
            bits.append(f"{key}={ev[key]:.4f}")
    if res.get("gap_closed_bayes_late") is not None:
        bits.append(f"gap_bayes={res['gap_closed_bayes_late']:.3f}")
    bits.append(f"ms/step={res['ms_step']:.1f}")
    grad = res.get("grad") or {}
    if grad:
        bits.append(f"gnorm={grad['norm_median']:.2f} clip={grad['clipped_frac']:.3f}")
    if "eval_jacobi" in res:
        bits.append(f"late_jac={res['eval_jacobi']['ce_late']:.4f}")
    return f"    {res['arm']}: " + " ".join(bits)


def _naive_arm(task: Task, st: Settings, dev: torch.device, seed: int, swa_state: dict,
               val: list, eval_data: list, meta: dict) -> dict:
    """TTT-naive: the trained sliding-window model plus eval-time first-order TTT.

    The step size is picked on the disjoint validation set by late-position CE,
    the regime where test-time learning has anything to contribute.
    """
    model = build_model(task, st, dev, seed, "swa")
    model.load_state_dict(swa_state)
    model.eval()
    _reset_peak(dev)
    grid: dict[str, dict] = {}
    with torch.no_grad():
        for eta in st.naive_etas:
            summ = _summarize_parts(
                _eval_parts(model, task, "naive", val, eta, st, GradStats(), learned=False),
                st, {})
            grid[f"{eta:g}"] = dict(ce=summ["ce"], ce_late=summ["ce_late"])
            print(f"  naive eta={eta:g}: val ce={summ['ce']:.4f} late={summ['ce_late']:.4f}",
                  flush=True)
    best = float(min(grid, key=lambda k: grid[k]["ce_late"]))
    out = evaluate(model, task, "naive", eval_data, meta, best, st, learned=False)
    out.update(arm="naive", seed=seed, task=task.name, steps=0, trained=False,
               params=sum(p.numel() for p in model.parameters()),
               eta_base=best, eta_chosen=best, eta_grid=grid,
               ms_step=0.0, ms_step_from=0, peak_mem_mb=_peak_mem(dev),
               eta_final=[best] * len(FAST_KEYS), gap_closed_late_deprecated=None,
               gap_closed_bayes_late=None)
    return out


def _seed_gap(tres: dict, arm: str, field: str = "gap_closed_late_deprecated") -> list[float]:
    """Per-seed values of one metric for one arm, in seed order."""
    vals = []
    for _, results in sorted(tres.get("runs", {}).items()):
        res = results.get(arm)
        if res and not res.get("skipped") and res.get(field) is not None:
            vals.append(res[field])
    return vals


def paired_vs_swa(tres: dict, arm: str, k: float = 2.0) -> dict:
    """Paired late-position CE test of ``arm`` against ``swa`` on the same eval set.

    The eval set is fixed independently of the training seed (see ``fixed_set``),
    so for every seed both arms score the same sequences in the same order and the
    per-sequence late-mean difference ``swa - arm`` is a paired observation. Its
    mean must clear ``k`` standard errors of its own spread to count as a win;
    comparing two unpaired means with their own standard errors would double the
    variance and, at these sample sizes, hide a real difference inside the noise.
    Per-seed differences are pooled into one sample.
    """
    diffs: list[float] = []
    per_seed: dict[str, dict] = {}
    for key, results in sorted(tres.get("runs", {}).items()):
        swa, res = results.get("swa"), results.get(arm)
        if not swa or not res or swa.get("skipped") or res.get("skipped"):
            continue
        ev_swa, ev_arm = swa.get("eval", {}), res.get("eval", {})
        a, b = ev_swa.get("late_per_seq"), ev_arm.get("late_per_seq")
        if not a or not b or len(a) != len(b):
            continue
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        diffs.extend(d.tolist())
        per_seed[key] = dict(mean_diff=round(float(d.mean()), 6),
                             se=round(float(d.std(ddof=1) / math.sqrt(d.size)), 6)
                             if d.size > 1 else None)
    if not diffs:
        return dict(rule=f"paired late CE (swa - {arm}) > {k:g} SE", passed=None, n=0)
    d = np.asarray(diffs, dtype=np.float64)
    mean_diff = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    return dict(rule=f"paired late CE (swa - {arm}) > {k:g} SE", mean_diff=round(mean_diff, 6),
                se=round(se, 6), n=int(d.size), per_seed=per_seed,
                passed=bool(mean_diff > k * se))


def _verdicts(out: dict) -> dict:
    """Pre-registered decision rules, evaluated on the collected arm results."""
    tasks = out.get("tasks", {})
    mc, echo, ctrl = tasks.get("icl_mc", {}), tasks.get("echo", {}), tasks.get("echo_ctrl", {})
    verdicts: dict[str, Any] = {}

    m1 = paired_vs_swa(mc, "e2e1").get("mean_diff")
    m2 = paired_vs_swa(mc, "e2e2").get("mean_diff")
    verdicts["H1_first_order_keeps_gain"] = dict(
        rule="paired gain(e2e1) >= 0.8 * paired gain(e2e2)",
        gain_e2e1=m1, gain_e2e2=m2,
        passed=None if m1 is None or m2 is None else bool(m1 >= H1_RATIO * m2))
    mj = paired_vs_swa(mc, "jac2").get("mean_diff")
    verdicts["H2_parallel_matches_sequential"] = dict(
        rule="paired gain(jac2, sequential eval) >= 0.8 * paired gain(e2e1)",
        gain_jac2=mj, gain_e2e1=m1,
        passed=None if mj is None or m1 is None else bool(mj >= H1_RATIO * m1))

    def query_ce(tres: dict, arm: str) -> float | None:
        for _, results in sorted(tres.get("runs", {}).items()):
            res = results.get(arm)
            if res and not res.get("skipped"):
                return res["eval"].get("query_ce")
        return None

    q_echo, q_ctrl = query_ce(echo, "e2e1"), query_ce(ctrl, "e2e1")
    verdicts["H1_by_construction"] = dict(
        rule="echo query_ce(e2e1) > 2.0 while echo_ctrl query_ce(e2e1) < 1.0",
        echo_e2e1=q_echo, echo_ctrl_e2e1=q_ctrl,
        passed=None if q_echo is None or q_ctrl is None
        else bool(q_echo > 2.0 and q_ctrl < 1.0))
    verdicts["control_ttt_beats_swa"] = dict(gap_e2e2=m2, **paired_vs_swa(mc, "e2e2"))
    verdicts["beats_swa_e2e1"] = paired_vs_swa(mc, "e2e1")
    verdicts["beats_swa_jac2"] = paired_vs_swa(mc, "jac2")
    naive, e2e2 = mc.get("runs", {}).get("seed0", {}), None
    if naive:
        e2e2 = naive.get("e2e2", {}).get("eval", {}).get("ce_late")
        nv = naive.get("naive", {}).get("eval", {}).get("ce_late")
    else:
        nv = None
    verdicts["control_trained_beats_naive"] = dict(
        rule="icl_mc seed 0: late CE of TTT training < late CE of TTT-naive",
        late_e2e2=e2e2, late_naive=nv,
        passed=None if e2e2 is None or nv is None else bool(e2e2 < nv))
    return verdicts


def _sanity(out: dict) -> dict:
    """Bin-0 agreement, Bayes leak check and the SWA-vs-windowed-Bayes gap."""
    flags: dict[str, Any] = {"bin0_agreement": {}, "leak_flag": [], "swa_minus_bayesweff_late": {}}
    for tname, tres in out.get("tasks", {}).items():
        ceiling = tres.get("ceiling")
        for key, results in sorted(tres.get("runs", {}).items()):
            swa = results.get("swa")
            if not swa or swa.get("skipped") or "eval" not in swa:
                continue
            bin0 = swa["eval"]["bins"][0]
            if ceiling:
                flags["swa_minus_bayesweff_late"][f"{tname}/{key}"] = round(
                    swa["eval"]["ce_late"] - ceiling["bayes_weff"]["ce_late"], 6)
            for arm, res in sorted(results.items()):
                if res.get("skipped") or "eval" not in res:
                    continue
                tag = f"{tname}/{key}/{arm}"
                flags["bin0_agreement"][tag] = round(abs(res["eval"]["bins"][0] - bin0), 6)
                if ceiling:
                    bayes = ceiling["bayes"]
                    for i, (b, se) in enumerate(zip(res["eval"]["bins"], bayes["bin_se"])):
                        if b < bayes["bins"][i] - 2 * se:
                            flags["leak_flag"].append(f"{tag}/bin{i}")
    return flags


def run(cfg: dict, seed: int) -> dict:
    """Run every configured arm of every task and return the full result block."""
    if "wave_b" in cfg or str(cfg.get("process_name", "")).startswith("TTM-r44b-"):
        from . import exp_r44b
        return exp_r44b.run(cfg, seed)
    st = _settings(cfg)
    _mark_process(st.process_name)
    configure_tf32(st.tf32, "main")
    dev = _device(st.device)
    if "cells" in cfg:
        if st.probe_every:
            sanity = r4_sanity()
            print("R4_SANITY " + json.dumps(sanity), flush=True)
            if not all(v["passed"] for v in sanity.values()):
                raise RuntimeError("R4 sanity failed")
        if any(c["arm"] in MASK_ARMS for c in cfg["cells"]):
            checks = r42_sanity()
            print("R42_SANITY " + json.dumps(checks), flush=True)
            if not all(v["passed"] for v in checks.values()):
                raise RuntimeError("R42 sanity failed")
        if any(c["arm"] in (*R42L_ARMS, "e2e1emb_kaminlin") for c in cfg["cells"]):
            checks = r42l_sanity()
            print("R42L_SANITY " + json.dumps(checks), flush=True)
            if not all(v["passed"] for v in checks.values()):
                raise RuntimeError("R42L sanity failed")
        if any(c["arm"] in R43_ARMS for c in cfg["cells"]):
            checks = r43_sanity()
            print("R43_SANITY " + json.dumps(checks), flush=True)
            if not all(v["passed"] for v in checks.values()):
                raise RuntimeError("R43 sanity failed")
            # the E/K manipulation gate on the run's own model size, first R4-3 cell's
            # task and eta. Reported, never an abort: the analyser turns it into the
            # "untested" labels when it is missing or below threshold
            try:
                cell = next(c for c in cfg["cells"] if c["arm"] in R43_ARMS)
                cst = _settings(cfg, cell["task"])
                share = ek_share_record(cst, _task(cst, cell["task"]),
                                        eta_base=float(cell.get("eta_base", 10.0)))
                print("R43_EK_SHARE " + json.dumps(share), flush=True)
            except Exception as exc:
                traceback.print_exc()
                print("R43_EK_SHARE_ERROR " + repr(exc), flush=True)
        return run_cells(cfg, st, dev)
    print(f"ttt | device={dev} torch={torch.__version__} jobs={len(_jobs(st))} "
          f"max_minutes={st.max_minutes:g}", flush=True)
    out: dict[str, Any] = {"device": str(dev), "torch": torch.__version__, "tasks": {},
                           "sanity": {}, "verdicts": {}}
    deadline = time.time() + st.max_minutes * 60.0
    swa_state: dict[tuple[str, int], dict] = {}
    skipped: list[dict] = []
    for tname, jseed, arms in _jobs(st):
        st = _settings(cfg, tname)
        task = _task(st, tname)
        tres = out["tasks"].setdefault(tname, {})
        tres.update(T=task.T, V=task.V, window=task.window, chunk=task.chunk, arms=list(arms))
        if task.mode == "mc" and "ceiling" not in tres:
            tres["ceiling"] = _ceilings(task, st)
            print(f"  ceiling {tname}: bayes late={tres['ceiling']['bayes']['ce_late']:.4f} "
                  f"bayes_weff late={tres['ceiling']['bayes_weff']['ce_late']:.4f} "
                  f"(w_eff={tres['ceiling']['weff']}) "
                  f"bayes_w64 late={tres['ceiling']['bayes_w64']['ce_late']:.4f}", flush=True)
        print(f"== {tname} seed {jseed} V={task.V} T={task.T} arms={arms} ==", flush=True)
        ev_tokens, ev_meta = fixed_set(task, st, "eval")
        if task.mode in ECHO_MODES and "refs" not in tres:
            tres["refs"] = echo_references(ev_meta, task.n_keys)
            print(f"REF {tname} {json.dumps(tres['refs'])}", flush=True)
        eval_data = eval_batches(task, ev_tokens, ev_meta, dev, st.eval_batch)
        quick = eval_batches(task, *fixed_set(task, st, "quick"), dev, st.eval_batch)
        results = tres.setdefault("runs", {}).setdefault(f"seed{jseed}", {})
        eta_pool: float | None = None
        for arm in arms:
            if deadline - time.time() <= 0:
                results[arm] = dict(arm=arm, seed=jseed, task=tname, skipped=True,
                                    reason="budget")
                skipped.append(dict(task=tname, seed=jseed, arm=arm))
                print(f"!! budget guard: skipping {tname} seed {jseed} {arm}", flush=True)
                continue
            if arm == "naive":
                if (tname, jseed) not in swa_state:
                    results[arm] = dict(arm=arm, seed=jseed, task=tname, skipped=True,
                                        reason="no swa model")
                    print("!! naive needs a trained swa model first; skipped", flush=True)
                    continue
                val_tokens, _ = fixed_set(task, st, "val")
                val_data = eval_batches(task, val_tokens, {}, dev, st.eval_batch)
                res = _naive_arm(task, st, dev, jseed, swa_state[(tname, jseed)], val_data,
                                 eval_data, ev_meta)
            else:
                if arm in TTT_ARMS and eta_pool is None:
                    eta_pool, pilot = pilot_eta_base(task, st, dev, jseed, quick)
                    tres.setdefault("pilot", {})[f"seed{jseed}"] = dict(eta_base=eta_pool,
                                                                        losses=pilot)
                model = build_model(task, st, dev, jseed, arm)
                _reset_peak(dev)
                rng = np.random.default_rng([jseed, TRAIN_SEED])
                t0 = time.time()
                res = train_arm(model, task, arm, 1.0 if eta_pool is None else eta_pool, st, dev,
                                rng, quick, f"{tname} s{jseed} {arm}")
                res.pop("_trace", None)
                if arm == "swa":
                    swa_state[(tname, jseed)] = {k: v.detach().clone()
                                                 for k, v in model.state_dict().items()}
                eta = 1.0 if eta_pool is None else eta_pool
                res.update(evaluate(model, task, arm, eval_data, ev_meta, eta, st))
                if arm in TTT_ARMS:
                    res["picard_diag"] = picard_diagnostics(model, *eval_data[0][:2], eta,
                                                             st.g_clip, task.chunk)
                    res["metagrad_fidelity"] = metagrad_fidelity(model, *quick[0][:2], eta,
                                                                   st.g_clip, task.chunk)
                res["checkpoint"] = save_checkpoint(model, f"{tname}__s{jseed}__{arm}")
                res.update(arm=arm, seed=jseed, task=tname, trained=True,
                           params=sum(p.numel() for p in model.parameters()),
                           fast_params=sum(p.numel() for p in model.head.parameters()),
                           eta_base=eta_pool if arm in TTT_ARMS else None,
                           eta_final=[round(float(e.detach()), 6) for e in model.etas(eta)],
                           peak_mem_mb=_peak_mem(dev), train_s=round(time.time() - t0, 1),
                           skipped=False)
            results[arm] = res
            _task_gaps(tres)
            print(_arm_summary(res), flush=True)
            print("ARM_RESULT " + json.dumps(res), flush=True)
    for tres in out["tasks"].values():
        _task_gaps(tres)
    out["skipped"] = skipped
    out["budget_stopped"] = bool(skipped)
    out["sanity"] = _sanity(out)
    out["verdicts"] = _verdicts(out)
    print("\n== SANITY ==", flush=True)
    print(f"bin-0 agreement |CE_arm(bin0) - CE_swa(bin0)|: "
          f"max {max(out['sanity']['bin0_agreement'].values(), default=0.0):.4f}", flush=True)
    print(f"leak flag: {out['sanity']['leak_flag'] or 'none'}", flush=True)
    print(f"swa_minus_bayesweff_late: {out['sanity']['swa_minus_bayesweff_late']}", flush=True)
    print("\n== VERDICTS ==", flush=True)
    for name, v in out["verdicts"].items():
        print(f"{name}: {'PASS' if v['passed'] else 'FAIL' if v['passed'] is not None else 'n/a'}"
              f"  [{v['rule']}]", flush=True)
        print(f"    {json.dumps({k: v[k] for k in v if k not in ('rule', 'passed')})}", flush=True)
    print("\nms/step is reported, not a speed claim: this toy is overhead-dominated.",
          flush=True)
    return out


class _no_tf32:
    """Context that turns TF32 matmuls off, so diagnostics compare at fp32 precision.

    The diagnostics set sequential and Picard weights side by side and read
    differences of a few percent; TF32 rounding would put a noise floor right
    under the pre-registered 5% threshold.
    """

    def __enter__(self) -> None:
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *exc: object) -> None:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved


def save_trace(trace: dict, name: str) -> str:
    """Persist the per-step outer-gradient, rho-gradient and loss series beside the checkpoint."""
    folder = Path("checkpoints")
    folder.mkdir(exist_ok=True)
    path = folder / f"{name.replace('/', '__')}__trace.npz"
    np.savez_compressed(path, **trace)
    return str(path)


def save_checkpoint(model: TTTModel, name: str) -> str:
    """Persist a CPU snapshot under the working directory."""
    folder = Path("checkpoints")
    folder.mkdir(exist_ok=True)
    path = folder / f"{name.replace('/', '__')}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    print(f"CHECKPOINT {path}", flush=True)
    return str(path)


def _cell_id(cell: dict, steps: int) -> str:
    """Stable unique identifier for a pinned-eta cell."""
    prefix = f"{cell['task']}/s{cell['seed']}/{cell['arm']}"
    if cell["arm"] in TTT_ARMS:
        prefix += f"/eta{cell['eta_base']:g}"
    suffix = "/sgn-1" if cell.get("inner_sign", 1.0) < 0 else ""
    if "grad_clip" in cell:
        suffix += f"/clip{float(cell['grad_clip']):g}"
    if "spike_skip" in cell:
        suffix += f"/skip{float(cell['spike_skip']):g}"
    if "lr" in cell:
        suffix += f"/lr{float(cell['lr']):g}"
    return f"{prefix}/st{steps}{suffix}"


def check_arm_task(arm: str, tname: str) -> None:
    """Refuse an arm on a task it is not defined for (``e2e1emb_ctrl`` is echo_ctrl)."""
    want = ARM_TASKS.get(arm)
    if want is not None and want != tname:
        raise ValueError(f"arm {arm} must run on task {want}, not {tname}")


def run_cells(cfg: dict, st: Settings, dev: torch.device) -> dict:
    """Execute independently trained cells in priority order with deadline and gate."""
    if st.workers > 1:
        return run_cells_parallel(cfg, st, dev)
    entries = []
    seen = set()
    for cell in cfg["cells"]:
        if cell["arm"] not in STATIC_ARMS + TTT_ARMS:
            raise ValueError(f"unknown cell arm: {cell['arm']}")
        check_arm_task(cell["arm"], cell["task"])
        if cell["arm"] in (*LIN_ARMS, *R44A_ARMS[1:]) and cell.get("diag", True):
            raise ValueError(f"arm {cell['arm']} requires diag=false: diag paths use CE without R4-4a interventions")
        if cell["arm"] in TTT_ARMS and "eta_base" not in cell:
            raise ValueError("TTT cell requires eta_base")
        task = _task(st, cell["task"])
        steps = int(cell.get("steps", task.steps))
        ident = _cell_id(cell, steps)
        if ident in seen:
            raise ValueError(f"duplicate cell id: {ident}")
        seen.add(ident)
        entries.append((cell, task, steps, ident))
    deadline = cfg.get("_deadline", time.time() + st.max_minutes * 60)
    r43_layout = cfg.get("_r43_layout", any(c["arm"] in R43_ARMS for c in cfg["cells"]))
    out: dict = dict(device=str(dev), torch=torch.__version__, tasks={}, cells={})
    gate_failed = False
    for cell, task, steps, ident in entries:
        tname, arm, jseed = task.name, cell["arm"], int(cell["seed"])
        record = dict(cell_id=ident, tag=cell.get("tag", ""), task=tname, arm=arm,
                      seed=jseed, steps=steps, eta_base=cell.get("eta_base"), device=str(dev))
        out["cells"][ident] = record
        if gate_failed and not cell.get("run_if_gate_fails", False):
            record.update(skipped=True, reason="gate")
            print(f"!! gate: skipping {ident}", flush=True)
            continue
        if time.time() >= deadline:
            record.update(skipped=True, reason="budget")
            print(f"!! budget guard: skipping {ident}", flush=True)
            continue
        try:
            tres = out["tasks"].setdefault(tname, dict(T=task.T, V=task.V,
                                                         window=task.window, chunk=task.chunk))
            ev_tokens, ev_meta = fixed_set(task, st, "eval")
            if task.mode in ECHO_MODES and "refs" not in tres:
                tres["refs"] = echo_references(ev_meta, task.n_keys)
                print(f"REF {tname} {json.dumps(tres['refs'])}", flush=True)
            ev = eval_batches(task, ev_tokens, ev_meta, dev, st.eval_batch)
            quick = eval_batches(task, *fixed_set(task, st, "quick"), dev, st.eval_batch)
            model = build_model(task, st, dev, jseed, arm)
            model.inner_sign = float(cell.get("inner_sign", 1.0))
            _reset_peak(dev)
            start = time.time()
            eta = float(cell.get("eta_base", 1.0))
            cst = _settings(cfg, tname)
            for key, cast in (("grad_clip", float), ("spike_skip", float), ("lr", float),
                              ("lr_total", int)):
                # per-cell optimiser probes; everything else unchanged
                if key in cell:
                    cst = dataclasses.replace(cst, **{key: cast(cell[key])})
            result = train_arm(model, task, arm, eta, cst, dev,
                               np.random.default_rng([jseed, TRAIN_SEED]), quick, ident, steps,
                               seed=jseed)
            trace = result.pop("_trace")
            model.eval()
            result.update(evaluate(model, task, arm, ev, ev_meta, eta, cst))
            if cst.probe_every and task.mode in PROBE_MODES:
                if arm in EMB_ARMS:
                    last = result["probes"][-1]
                    result["s5"] = dict(passed=last["lin_kv_max_diff"] <= 1e-4 and
                                        last["lin_kv_acc_diff"] <= .002,
                                        lin_kv_max_diff=last["lin_kv_max_diff"],
                                        lin_kv_acc_diff=last["lin_kv_acc_diff"])
                    if not result["s5"]["passed"]:
                        print(f"S5_FAIL {ident} {json.dumps(result['s5'])}", flush=True)
                result.update(r4_flags(result["quick"], result["probes"],
                                       result["eval"]["query_acc"], tres["refs"]["occ_minus"]["acc"]))
            # train_s and peak memory cover training plus eval, before the
            # diagnostics; the checkpoint and the record are secured before anything
            # optional can fail
            result.update(trained=True, skipped=False, params=sum(p.numel() for p in model.parameters()),
                          fast_params=sum(p.numel() for p in model.head.parameters()),
                          eta_final=[float(e.detach()) for e in model.etas(eta)],
                          inner_sign=model.inner_sign, grad_clip=cst.grad_clip,
                          spike_skip=cst.spike_skip, lr=cst.lr, eta_lr=cst.eta_lr,
                          tf32=bool(st.tf32),
                          tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                          tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                          peak_mem_mb=_peak_mem(dev),
                          peak_cuda_mb=_peak_mem(dev) if dev.type == "cuda" else 0.0,
                          train_s=round(time.time() - start, 1),
                          checkpoint=save_checkpoint(model, ident),
                          trace=save_trace(trace, ident))
            if arm in ABLATION_ARMS or (r43_layout and arm in R43_ABLATION_EXTRA):
                # after the checkpoint: a failed or mismatched ablation never loses the
                # trained cell; the analysis counts it as an ML3 failure for that seed
                try:
                    result["ablation"] = r42_ablation(model, task, ev, eta, cst.g_clip, arm=arm)
                    diff = abs(result["ablation"]["all"] - result["eval"]["query_acc"])
                    result["ablation_ref_match"] = diff <= .01
                    if not result["ablation_ref_match"]:
                        print(f"ABLATION_MISMATCH {ident} diff={diff}", flush=True)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"ABLATION_ERROR {ident}", flush=True)
                    result["ablation_error"] = repr(exc)
                    if dev.type == "mps":
                        torch.mps.empty_cache()
            # R4-4a diagnostics run after the checkpoint and never lose the trained cell:
            # an error is logged and recorded instead.
            if arm in NOISE_ARMS:
                try:
                    corrupted_ev, cm = corrupted_batches(task, cst, arm, dev, "eval")
                    with torch.no_grad():
                        parts = _eval_parts(model, task, arm, corrupted_ev, eta, cst, GradStats())
                        result["corrupted_eval"] = dict(
                            **_summarize_parts(parts, cst, cm, task.chunk), **corruption_recall(parts, cm))
                except Exception as exc:
                    traceback.print_exc()
                    print(f"CORRUPTED_EVAL_ERROR {ident}", flush=True)
                    result["corrupted_eval_error"] = repr(exc)
            if arm == "e2e1emb_edge":
                try:
                    with torch.no_grad():
                        off_eval = evaluate(model, task, "e2e1emb", ev, ev_meta, eta, cst)["eval"]
                    result["route_checks"] = dict(
                        p_v_kv_masked=kv_w0_stats(model, task, ev, arm)["p_kv_mean"],
                        ablation=result.get("ablation"),
                        mask_off=dict(query_acc=off_eval["query_acc"],
                                      p_kv=kv_w0_stats(model, task, ev, "e2e1emb")["p_kv_mean"]))
                except Exception as exc:
                    traceback.print_exc()
                    print(f"ROUTE_CHECKS_ERROR {ident}", flush=True)
                    result["route_checks_error"] = repr(exc)
            if arm in ("e2e1emb", "e2e1emb_edge") and "kamin_switch" in cfg:
                try:
                    result["d1"] = d1_probe(model, task, cst, arm, dev)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"D1_ERROR {ident}", flush=True)
                    result["d1_error"] = repr(exc)
            if dev.type == "mps":
                # the training-time samples miss the final eval and the ablation
                result["peak_mps_mb"] = max(result.get("peak_mps_mb", 0.0),
                                            round(torch.mps.driver_allocated_memory() / 2**20, 1))
            # Include the full-set write ablation in the per-cell CUDA peak.
            result["peak_cuda_mb"] = _peak_mem(dev) if dev.type == "cuda" else 0.0
            record.update(result)
            if "gate_min_acc" in cell:
                acc = record["eval"].get("query_acc")
                if acc is None or not acc >= cell["gate_min_acc"]:
                    gate_failed = True
            # the eval numbers reach the log before any diagnostic can hang or fail
            print(_arm_summary(record), flush=True)
            print("CELL_RESULT " + json.dumps(record), flush=True)
            if arm in TTT_ARMS and cell.get("diag", True):
                try:
                    with _no_tf32():
                        record["picard_diag"] = picard_diagnostics(model, *ev[0][:2], eta,
                                                                    st.g_clip, task.chunk)
                        record["metagrad_fidelity"] = metagrad_fidelity(model, *quick[0][:2], eta,
                                                                          st.g_clip, task.chunk)
                except Exception as exc:
                    traceback.print_exc()
                    print(f"DIAG_ERROR {ident}", flush=True)
                    record["diag_error"] = repr(exc)
                    model.zero_grad(set_to_none=True)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                diag = {k: record[k] for k in ("picard_diag", "metagrad_fidelity", "diag_error")
                        if k in record}
                print("CELL_DIAG " + json.dumps(dict(cell_id=ident, **diag)), flush=True)
        except Exception as exc:
            traceback.print_exc()
            print(f"CELL_ERROR {ident}", flush=True)
            record["error"] = repr(exc)
            if "gate_min_acc" in cell and "eval" not in record:
                # a gate cell that never produced its metric cannot vouch for the harness
                gate_failed = True
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print("\n== CELLS ==", flush=True)
    for ident, res in out["cells"].items():
        ev = res.get("eval", {})
        grad = res.get("outer_grad", {})
        status = "skipped" if res.get("skipped") else "error" if res.get("error") else (
            "diverged" if res.get("diverged") else "ok")
        print(f"{ident} | {res['tag']} | {status} | qacc={ev.get('query_acc')} "
              f"±{ev.get('query_acc_se')} qce={ev.get('query_ce')} "
              f"late={ev.get('ce_late')} ±{ev.get('ce_late_se')} "
              f"ms/step={res.get('ms_step')} train_s={res.get('train_s')} "
              f"peak_cuda_mb={res.get('peak_cuda_mb')} "
              f"eta={res.get('eta_final')} gnorm={grad.get('median')} "
              f"clip={grad.get('clip_fraction')} nonfinite={res.get('nonfinite_steps')}", flush=True)
    for name, tres in out["tasks"].items():
        if "refs" in tres:
            print(f"REF {name} {json.dumps(tres['refs'])}", flush=True)
    return out


class _LockedLines:
    """Write whole lines under a lock shared by all workers (lines exceed PIPE_BUF)."""

    def __init__(self, stream, lock) -> None:
        self.stream, self.lock, self.buf = stream, lock, ""

    def write(self, text: str) -> int:
        self.buf += text
        if "\n" in self.buf:
            head, _, self.buf = self.buf.rpartition("\n")
            with self.lock:
                self.stream.write(head + "\n")
                self.stream.flush()
        return len(text)

    def flush(self) -> None:
        if self.buf:
            with self.lock:
                self.stream.write(self.buf)
                self.stream.flush()
            self.buf = ""


def _cell_worker(index: int, cfg: dict, dev_name: str, jobs: mp.Queue,
                 results: mp.Queue, lock=None) -> None:
    """Isolated spawned process, continuing after a cell-level error."""
    if lock is not None:
        sys.stdout = _LockedLines(sys.stdout, lock)
        sys.stderr = _LockedLines(sys.stderr, lock)
    # before any device allocation: the tf32 key is asserted and logged here
    configure_tf32(bool(_settings(cfg).tf32), str(index))
    _mark_process(f"{cfg.get('process_name', 'TTM-r41-01')}-w{index}")
    dev = _device(dev_name)
    if dev.type == "mps":
        torch.set_num_threads(2)
        if cfg.get("mps_memory_fraction") is not None:
            fraction = float(cfg["mps_memory_fraction"])
            torch.mps.set_per_process_memory_fraction(fraction)
            recommended = torch.mps.recommended_max_memory()
            print(f"MPS_CAP worker={index} recommended_gb={recommended / 2**30:.2f} "
                  f"fraction={fraction} cap_gb={fraction * recommended / 2**30:.2f}", flush=True)
    if dev.type == "cpu":
        torch.set_num_threads(3)
    if dev.type == "cuda":
        # os.cpu_count() in a container reports the host; the GPU path needs few CPU threads.
        torch.set_num_threads(2)
    while True:
        cell = jobs.get()
        if cell is None:
            break
        try:
            local = {**cfg, "cells": [cell], "workers": 1}
            result = run_cells(local, _settings(local), dev)
            results.put((cell, result))
        except Exception as exc:
            traceback.print_exc()
            ident = _cell_id(cell, _cell_steps(_settings(cfg), cell))
            record = dict(cell_id=ident, task=cell["task"], seed=cell["seed"], arm=cell["arm"],
                          device=str(dev), error=repr(exc))
            print("CELL_ERROR " + json.dumps(record), flush=True)
            results.put((cell, {"cells": {ident: record}, "tasks": {}}))


def _cell_steps(st: Settings, cell: dict) -> int:
    """Resolve steps without constructing a task when the cell pins its length."""
    return int(cell["steps"]) if "steps" in cell else _task(st, cell["task"]).steps


def run_cells_parallel(cfg: dict, st: Settings, dev: torch.device) -> dict:
    """Spawn independent workers; mixed devices keep every seed on one device."""
    if any("gate_min_acc" in cell for cell in cfg["cells"]):
        raise ValueError("gated cells require workers=1")
    ids = []
    for cell in cfg["cells"]:
        if cell["arm"] not in STATIC_ARMS + TTT_ARMS:
            raise ValueError(f"unknown cell arm: {cell['arm']}")
        check_arm_task(cell["arm"], cell["task"])
        if cell["arm"] in (*LIN_ARMS, *R44A_ARMS[1:]) and cell.get("diag", True):
            raise ValueError(f"arm {cell['arm']} requires diag=false: diag paths use CE without R4-4a interventions")
        if cell["arm"] in TTT_ARMS and "eta_base" not in cell:
            raise ValueError("TTT cell requires eta_base")
        ids.append(_cell_id(cell, _cell_steps(st, cell)))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cell ids")
    devices = st.worker_devices or [str(dev)] * st.workers
    if len(devices) != st.workers:
        raise ValueError("worker_devices must have one entry per worker")
    ctx = mp.get_context("spawn")
    types = {torch.device(d).type for d in devices}
    if len(types) > 1:
        raise ValueError(f"R4 runs use one device type, got {sorted(types)}")
    queues = [ctx.Queue() for _ in range(st.workers)] if len(types) > 1 else [ctx.Queue()]
    ordered = list(cfg["cells"])
    for cell in ordered:
        if len(queues) > 1:
            queues[int(cell["seed"]) % st.workers].put(cell)
        else:
            queues[0].put(cell)
    for q in queues:
        for _ in range(st.workers if len(queues) == 1 else 1):
            q.put(None)
    results = ctx.Queue()
    lock = ctx.Lock()
    shared = {**cfg, "_deadline": time.time() + st.max_minutes * 60,
              "_r43_layout": any(c["arm"] in R43_ARMS for c in cfg["cells"])}
    processes = [ctx.Process(target=_cell_worker, args=(i, shared, devices[i],
                 queues[0] if len(queues) == 1 else queues[i], results, lock))
                 for i in range(st.workers)]
    for proc in processes:
        proc.start()
    collected = {}
    while len(collected) < len(ordered):
        try:
            cell, result = results.get(timeout=30)
        except queue.Empty:
            # a hard worker crash (driver abort) must not hang the run
            if not any(proc.is_alive() for proc in processes):
                break
            continue
        collected[_cell_id(cell, _cell_steps(st, cell))] = result
    for proc in processes:
        proc.join(timeout=60)
    if any(proc.exitcode != 0 for proc in processes):
        print(f"!! worker exit codes {[proc.exitcode for proc in processes]}", flush=True)
    out = dict(device=str(dev), torch=torch.__version__, tasks={}, cells={})
    for cell in ordered:
        ident = _cell_id(cell, _cell_steps(st, cell))
        result = collected.get(ident, {"cells": {ident: dict(
            cell_id=ident, task=cell["task"], seed=cell["seed"], arm=cell["arm"],
            error="worker died before returning this cell")}, "tasks": {}})
        out["cells"].update(result["cells"])
        out["tasks"].update(result["tasks"])
    return out
