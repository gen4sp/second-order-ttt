"""Official Orbax model_weights -> the ouroboros-compatible P2 tensor layout.

The exhaustive layer slicing matches the official-JAX reference mapping in
tests/jax_parity/mapping.json: matrices already have [in,out] layout.
JAX and Orbax are imported only by the optional converter, never the run path.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os

import numpy as np
import torch

from .data import sha256_file
from .model import P2Config, PRESETS, TTTE2E
from .train import _atomic_save


def expected_shapes(cfg: P2Config):
    # Meta allocation validates the real model's names/shapes without allocating 125M weights.
    with torch.device("meta"):
        model = TTTE2E(cfg, compute_dtype=torch.float32)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def _leaves(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from _leaves(value, f"{prefix}.{i}")
    elif tree is not None:
        yield prefix, tree


def map_weights(tree, cfg: P2Config):
    """No unused parameter, guessed transpose, or silently initialized missing tensor."""
    expected = expected_shapes(cfg)
    out, mapping = {}, []
    source_count = 0
    for source, leaf in _leaves(tree):
        a = np.asarray(leaf)
        if a.dtype.name not in ("float32", "bfloat16"):
            raise ValueError(f"unexpected official parameter dtype: {source}: {a.dtype}")
        if not np.isfinite(a).all():
            raise ValueError(f"nonfinite official parameter: {source}")
        source_count += a.size
        base = "language_model.model."
        if not source.startswith(base):
            raise ValueError(f"unmapped official leaf: {source}")
        relative = source[len(base):]
        if relative in ("wte.weight", "ln_f.weight"):
            destinations = [(relative, a)]
        elif relative.startswith("h.blocks."):
            tail = relative[len("h.blocks."):].replace("seq_modeling_block.", "attn.")
            if a.shape[0] != cfg.num_layers:
                raise ValueError(f"incorrect official layer stack: {source}: {a.shape}")
            destinations = [(f"layers.{i}.{tail}", a[i]) for i in range(cfg.num_layers)]
        elif relative.startswith("h.prime_storage."):
            tail = relative[len("h.prime_storage."):]
            if a.shape[0] != cfg.suffix_len:
                raise ValueError(f"incorrect official prime stack: {source}: {a.shape}")
            start = cfg.num_layers - cfg.suffix_len
            destinations = [(f"layers.{start + i}.{tail}", a[i]) for i in range(cfg.suffix_len)]
        else:
            raise ValueError(f"unmapped official leaf: {source}")
        for name, tensor in destinations:
            if name not in expected or name in out or tensor.shape != expected[name]:
                raise ValueError(f"incorrect/duplicate official tensor: {source} -> {name}: {tensor.shape}")
            out[name] = torch.from_numpy(np.array(tensor, dtype=np.float32, copy=True))
            mapping.append(dict(official=source, torch=name, shape=list(tensor.shape)))
    if set(out) != set(expected):
        raise ValueError(f"missing official tensors: {sorted(set(expected) - set(out))}")
    count = sum(t.numel() for t in out.values())
    if count != source_count or count != sum(int(np.prod(s)) for s in expected.values()):
        raise ValueError("official/port parameter counts differ")
    return out, mapping, count


def model_weights_dir(path):
    if "://" in str(path):
        raise ValueError("converter accepts local downloaded checkpoints only; no requester-pays download")
    root = Path(path).resolve()
    if (root / "model_weights").is_dir():
        return root / "model_weights"
    if (root / "_METADATA").is_file():
        return root
    candidates = sorted((p / "model_weights" for p in root.iterdir()
                         if p.name.isdecimal() and (p / "model_weights").is_dir()),
                        key=lambda p: int(p.parent.name)) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("specify one local Orbax model_weights directory or an unambiguous step directory")


def convert(path, out, cfg=PRESETS["125m"]):
    import jax
    import orbax.checkpoint as ocp

    weights = model_weights_dir(path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"checkpoint already exists: {out}")
    # Read the official StandardSave format without loading optimizer/dataset state.
    with ocp.StandardCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(str(weights))
        # Explicit single-CPU sharding overrides the release's multi-GPU mesh.
        # A ShapeDtypeStruct with sharding=None can still inherit saved sharding.
        cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
        target = jax.tree.map(lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=cpu), metadata.tree)
        tree = checkpointer.restore(str(weights), target)
    state, mapping, count = map_weights(tree, cfg)
    source_hashes = {str(p.relative_to(weights)): sha256_file(p)
                     for p in sorted(weights.rglob("*")) if p.is_file()}
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=state, cfg=asdict(cfg), source="official-local-orbax",
                   source_path=str(weights), source_sha256=source_hashes,
                   mapping=mapping, parameter_count=count)
    _atomic_save(payload, out)
    digest = sha256_file(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    summary = dict(output=str(out), sha256=digest, parameters=count,
                   tensors=len(state), source=str(weights), source_sha256=source_hashes)
    print("P2_CONVERT " + json.dumps(summary, sort_keys=True), flush=True)
    return summary
