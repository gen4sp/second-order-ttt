"""Atomic, integrity-checked full start states."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from . import exp_chr as chr
from . import exp_ttt as ttt

SCHEMA = 1
HASH_KEYS = ("config", "source", "rules", "quick", "eval", "probe")
MANIFEST_FIELDS = ("cell_id", "arm", "seed", "trigger", "event_step", "detected_step",
                   "state_step", "completed_updates", "nonfinite", "file", "sha256",
                   "next_batch_sha1", *HASH_KEYS)


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if exclusive:
            os.link(name, path)  # atomic first write, never replace a frozen manifest
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def resolve_manifest(rows: list[dict], root: str | Path) -> list[dict]:
    """CSV state paths are relative to the downloaded/worker output root."""
    root = Path(root).resolve()
    resolved = []
    for row in rows:
        file = Path(row["file"])
        if not file.is_absolute():
            file = root / file
        resolved.append({**row, "file": str(file.resolve()),
                         **{k: int(row[k]) for k in ("seed", "event_step", "detected_step",
                            "state_step", "completed_updates", "nonfinite") if k in row}})
    return resolved


def relocated_file(file, root, *, folder=None):
    """Prefer the current output tree; verify identity separately with its digest."""
    path=Path(file);root=Path(root)
    candidates=[]
    if not path.is_absolute():candidates.append(root/path)
    for kind in ('mains','jobs'):
        if kind in path.parts:
            candidates.append(root/Path(*path.parts[path.parts.index(kind):]))
    if folder is not None:candidates.append(Path(folder)/path.name)
    candidates.append(path)
    return str(next((p.resolve() for p in candidates if p.is_file()),path))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode()).hexdigest()


def source_hash(root: Path) -> str:
    h = hashlib.sha256()
    for base in ("src", "scripts"):
        for file in sorted((root / base).rglob("*.py")):
            h.update(file.relative_to(root).as_posix().encode() + b"\0" + file.read_bytes() + b"\0")
    return h.hexdigest()


def context_hashes(ctx, cfg: dict, *, root: Path | None = None) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2] if root is None else root
    admin = {*chr.ADMIN_KEYS, "output_dir", "resume", "seal_file"}
    science = {k: v for k, v in cfg.items() if k not in admin and not k.startswith("_")}
    return dict(config=json_hash(science), source=source_hash(root),
                rules=json_hash({k: cfg.get(k, chr.CHR_DEFAULTS.get(k))
                                 for k in ("rp_cap", "cross_rules", "final_rules", "forks")}),
                quick=chr._sha1(ttt.fixed_set(ctx.task, ctx.st, "quick")[0]),
                eval=chr._sha1(ttt.fixed_set(ctx.task, ctx.st, "eval")[0]),
                probe=chr._sha1(ctx.task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]),
                                            ctx.st.n_probe)[0]))


def capture_rngs() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=[s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])


def restore_rngs(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    if states["torch_cuda"]:
        if not torch.cuda.is_available() or len(states["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device-count mismatch")
        torch.cuda.set_rng_state_all(states["torch_cuda"])


@contextmanager
def preserve_rngs(rng: np.random.Generator | None = None):
    states = capture_rngs()
    data = copy.deepcopy(rng.bit_generator.state) if rng is not None else None
    try:
        yield
    finally:
        restore_rngs(states)
        if rng is not None:
            rng.bit_generator.state = data


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return copy.deepcopy(value)


def save_state(path, model, opt, rng, ctx, *, trigger: str, event_step: int,
               detected_step: int, state_step: int, nonfinite: int, hashes: dict,
               detector_tail: list[dict], extra: dict | None = None) -> dict:
    path = Path(path)
    if set(HASH_KEYS) - hashes.keys():
        raise ValueError("incomplete provenance hashes")
    if not 0 <= event_step <= detected_step <= state_step or nonfinite > state_step:
        raise ValueError("invalid state chronology")
    payload = dict(schema=SCHEMA, cell_id=ctx.ident, arm=ctx.arm, seed=ctx.seed, eta_base=ctx.eta,
                   trigger=trigger, event_step=event_step, detected_step=detected_step,
                   state_step=state_step, completed_updates=state_step,
                   model=_cpu(model.state_dict()), opt=_cpu(opt.state_dict()),
                   rng=copy.deepcopy(rng.bit_generator.state), rngs=capture_rngs(),
                   copy_noise=dict(rng="numpy", identity=[ctx.seed, 4406, state_step],
                                   stateless=True, next_update=state_step),
                   nonfinite=nonfinite, next_batch_sha1=chr.rng_peek(rng.bit_generator.state,
                                                                  ctx.task, ctx.st.batch),
                   hashes=dict(hashes), detector_tail=_cpu(detector_tail[-20:]), extra=extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable state already exists: {path}")
    fd, name = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)  # atomic, cannot overwrite another writer
    finally:
        Path(name).unlink(missing_ok=True)
    return {k: payload[k] for k in MANIFEST_FIELDS if k in payload} | dict(
        file=str(path), sha256=file_sha256(path), **hashes)


def verify_state(path, ctx, *, sha256: str, expected_hashes: dict | None = None) -> dict:
    if not sha256 or file_sha256(path) != sha256:
        raise ValueError("state sha256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema", "model", "opt", "rng", "rngs", "copy_noise", "hashes",
                "detector_tail", "extra", "next_batch_sha1", "nonfinite", "cell_id", "seed",
                "arm", "eta_base", "trigger", "event_step", "detected_step", "state_step",
                "completed_updates"}
    if required - payload.keys() or payload["schema"] != SCHEMA:
        raise ValueError("invalid state schema")
    if (payload["cell_id"], payload["arm"], payload["seed"], payload["eta_base"]) != (
            ctx.ident, ctx.arm, ctx.seed, ctx.eta):
        raise ValueError("state cell identity mismatch")
    at = payload["state_step"]
    if not (payload["completed_updates"] == at and 0 <= payload["event_step"]
            <= payload["detected_step"] <= at and 0 <= payload["nonfinite"] <= at):
        raise ValueError("invalid completed-update state count")
    if chr.rng_peek(payload["rng"], ctx.task, ctx.st.batch) != payload["next_batch_sha1"]:
        raise ValueError("state next-batch SHA1 mismatch")
    if set(HASH_KEYS) - payload["hashes"].keys():
        raise ValueError("state provenance missing")
    fixed = dict(quick=chr._sha1(ttt.fixed_set(ctx.task, ctx.st, "quick")[0]),
                 eval=chr._sha1(ttt.fixed_set(ctx.task, ctx.st, "eval")[0]),
                 probe=chr._sha1(ctx.task.raw(np.random.default_rng([ttt.PROBE_SEED, 0]),
                                             ctx.st.n_probe)[0]))
    for key, value in {**fixed, **(expected_hashes or {})}.items():
        if payload["hashes"].get(key) != value:
            raise ValueError(f"state {key} identity mismatch")
    if payload["copy_noise"]["identity"] != [ctx.seed, 4406, at]:
        raise ValueError("copy-noise stream position mismatch")
    return payload


def load_state(path, model, opt, rng, ctx, *, sha256: str,
               expected_hashes: dict | None = None, restore_rng: bool = True) -> dict:
    payload = verify_state(path, ctx, sha256=sha256, expected_hashes=expected_hashes)
    current = model.state_dict()
    if current.keys() != payload["model"].keys() or any(
            current[k].shape != payload["model"][k].shape or
            current[k].dtype != payload["model"][k].dtype for k in current):
        raise ValueError("model layout mismatch before state reload")
    groups = opt.state_dict()["param_groups"]
    if len(groups) != len(payload["opt"]["param_groups"]) or any(
            len(a["params"]) != len(b["params"]) for a, b in zip(groups, payload["opt"]["param_groups"])):
        raise ValueError("optimizer layout mismatch before state reload")
    if restore_rng and payload["rngs"]["torch_cuda"] and (
            not torch.cuda.is_available() or len(payload["rngs"]["torch_cuda"]) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG layout mismatch before state reload")
    model.load_state_dict(payload["model"], strict=True)
    opt.load_state_dict(copy.deepcopy(payload["opt"]))
    rng.bit_generator.state = copy.deepcopy(payload["rng"])
    if restore_rng:
        restore_rngs(payload["rngs"])
    if not chr._state_equal(_cpu(model.state_dict()), payload["model"]) or not chr._deep_equal(
            _cpu(opt.state_dict()), payload["opt"]):
        raise ValueError("state reload equality failed")
    return payload
