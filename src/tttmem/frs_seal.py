"""Schema-1 combined seal verification for FRS-B/B9, before model creation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
OPERATIONAL = {"process_name", "output_dir", "workers", "max_minutes", "require_seal"}


def source_paths(root=ROOT):
    return (sorted((root / "src/tttmem").glob("*.py")) +
            sorted((root / "scripts").glob("*.py")) +
            [root / "pyproject.toml", root / "uv.lock"])


def protocol_hash(cfg):
    protocol = {k: v for k, v in cfg.items() if k not in OPERATIONAL}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def execution_guard(cfg, *, b9=False, root=ROOT):
    # Seal verification is opt-in: set require_seal = true to enforce it.
    if cfg.get("smoke", False) or not cfg.get("require_seal", False):
        return None
    path = Path(cfg.get("seal_file", "frs_seal.json"))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise ValueError("production requires the combined seal file")
    seal = json.loads(path.read_text())
    if seal.get("schema") != 1 or seal.get("phase") != "sealed":
        raise ValueError("production requires a schema-1 sealed manifest")
    node, stem = ("b9", "b9") if b9 else ("frsb", "frb")
    plan = Path(cfg["plan_path"])
    plan = plan if plan.is_absolute() else root / plan
    name = "b9" if b9 else "frs-b"
    required = source_paths(root) + [plan, root / f"configs/toy/{name}.toml",
                                    root / f"configs/toy/smoke/{name}.toml",
                                    root / f"configs/toy/smoke/{name}-gpu.toml"]
    # The top-level files list belongs to FRS-A. Per-node lists cover branch
    # snapshots whose identical relative paths have different hashes; FRS-A
    # ignores this additive metadata and verifies its own files.
    scoped = seal.get("nodes", {}).get(node)
    entries = (scoped.get("files", []) if scoped is not None else
               [e for e in seal.get("files", []) if e.get("node", node) == node])
    sealed = {}
    for entry in entries:
        file = (root / entry["path"]).resolve()
        if file in sealed and sealed[file]["sha256"] != entry["sha256"]:
            raise ValueError("conflicting seal entries")
        sealed[file] = entry
    for file in required:
        file = file.resolve()
        entry = sealed.get(file)
        if entry is None:
            raise ValueError(f"seal lacks this node's file: {file.relative_to(root)}")
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"seal sha256 mismatch: {entry['role']}")
    if sealed[plan.resolve()]["role"] != f"{node}_plan":
        raise ValueError("seal registration role mismatch")
    archived = tomllib.loads((root / f"configs/toy/{name}.toml").read_text())[node]
    runtime = {k: v for k, v in cfg.items() if k != "_frsb_result_sha256"}
    if protocol_hash(runtime) != protocol_hash(archived):
        raise ValueError("runtime protocol differs from sealed config")
    return hashlib.sha256(path.read_bytes()).hexdigest()
