"""Typed RNG identities and a hash-verified phase boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def registry(path: str | Path = ROOT / "configs/toy/pool/seed-registry.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def streams(seed: int, *, step: int = 0, state_step: int = 0) -> dict[str, list[tuple[int, ...]]]:
    return dict(init=[(seed,)], frozen_code=[(seed + 7919,)],
                data=[(seed, 20240101)], copy_noise=[(seed, 4406, step)],
                chr_directions=[(31337 + seed + 7919 * j,) for j in range(4)],
                rnd_mask=[(seed, 4242, state_step)])


def guard(stream: str, identity: tuple[int, ...], reg: dict | None = None) -> None:
    """Compare identities only within the same RNG and purpose; arity matters."""
    reg = registry() if reg is None else reg
    if stream not in reg["streams"]:
        raise ValueError(f"unknown stream {stream}; B9 sham has no stream")
    historical = [s for a, b in reg["streams"][stream]["historical_seed_ranges"]
                  for s in range(a, b + 1)]
    identity = tuple(identity)
    if stream == "copy_noise":
        collision = len(identity) == 3 and identity[:2] in [(s, 4406) for s in historical]
    elif stream == "rnd_mask":
        collision = identity in [(s, 4242) for s in historical]
    else:
        collision = any(identity in streams(s)[stream] for s in historical)
    if collision:
        raise ValueError(f"historical {stream} RNG identity: {identity}")


def guard_seed(seed: int, reg: dict | None = None) -> None:
    reg = registry() if reg is None else reg
    if not reg["fresh_family"][0] <= seed <= reg["fresh_family"][1]:
        raise ValueError(f"seed {seed} is not in the registered fresh family")
    for stream, identities in streams(seed).items():
        for identity in identities:
            guard(stream, identity, reg)


SEAL_ROLES = {"frs_plan", "frsb_plan", "b9_plan", "rdo_plan", "rpe_plan", "trg_plan",
              "interfaces", "registry", "state", "triggers", "episodes", "floor_rules",
              "trg_logger", "trg_calibration", "job_manifest", "cohort_manifest", "frs_driver",
              "frs_jobs", "frs_analyzer", "frs_cohorts", "frs_backend", "seed_guard",
              "run_entrypoint", "base_model", "base_chr", "locked_deps", "production_config"}

ROLE_PATHS = dict(
    registry='configs/toy/pool/seed-registry.toml', state='src/tttmem/frs_state.py', triggers='src/tttmem/triggers.py',
    episodes='src/tttmem/episodes.py', floor_rules='src/tttmem/floor_rules.py',
    frs_driver='src/tttmem/exp_frs.py', frs_jobs='src/tttmem/frs_jobs.py',
    frs_analyzer='scripts/analyze_frs.py', frs_cohorts='src/tttmem/frs_cohorts.py',
    frs_backend='src/tttmem/frs_backend.py', seed_guard='src/tttmem/frs_registry.py',
    run_entrypoint='src/tttmem/run.py', base_model='src/tttmem/exp_ttt.py',
    base_chr='src/tttmem/exp_chr.py', locked_deps='uv.lock', production_config='configs/toy/pool.toml',
    trg_logger='src/tttmem/trg_pool.py')


def effective_config(cfg):
    # Process naming is administrative; all scientific and scheduling knobs are bound.
    return {k: v for k, v in cfg.items() if not k.startswith('_') and k != 'process_name'}


def verify_seal(path: str | Path, *, root: Path = ROOT) -> dict:
    seal = json.loads(Path(path).read_text())
    if seal.get("schema") != 1 or seal.get("phase") != "sealed":
        raise ValueError("production requires a schema-1 sealed manifest")
    entries = seal.get("files", [])
    if len({e['role'] for e in entries}) != len(entries):
        raise ValueError('duplicate seal roles')
    if SEAL_ROLES - {e["role"] for e in entries}:
        raise ValueError(f"seal lacks roles: {sorted(SEAL_ROLES - {e['role'] for e in entries})}")
    for entry in entries:
        if entry['role'] in ROLE_PATHS and entry['path'] != ROLE_PATHS[entry['role']]:
            raise ValueError(f"seal role path mismatch: {entry['role']}")
        file = root / entry["path"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"seal sha256 mismatch: {entry['role']}")
    from .frs_state import source_hash
    if seal.get('source_sha256') != source_hash(root):
        raise ValueError('sealed source-tree hash mismatch')
    if not seal.get("job_modules") or not seal.get("coverage_factory"):
        raise ValueError("seal must declare route job modules and TRG logger factory")
    import importlib.util
    sealed_paths = {(root / e["path"]).resolve() for e in entries}
    for module in [*seal["job_modules"], *seal.get('imported_modules', []),
                   *[f.split(':')[0] for f in seal.get('backend_factories', [])],
                   seal["coverage_factory"].split(":")[0],
                   *([seal["backend_factory"].split(":")[0]] if seal.get("backend_factory") else [])]:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() not in sealed_paths:
            raise ValueError(f"unsealed module source: {module}")
    return seal


def execution_guard(cfg: dict) -> dict | None:
    """Called before model construction; a no-op unless cfg sets require_seal = true."""
    if not cfg.get("require_seal", False):
        return None
    seeds = [int(c["seed"]) for c in cfg["cells"]]
    smoke = bool(cfg.get("smoke_only"))
    if smoke:
        if not all(0 <= s <= 9 for s in seeds):
            raise ValueError("pre-seal smokes permit only legacy seeds 0–9")
        if int(cfg.get("steps", 0)) > 80:
            raise ValueError("smoke step cap is 80")
        return None
    seal = verify_seal(cfg.get("seal_file", "frs_seal.json"))
    from .frs_state import json_hash
    production = tomllib.loads((ROOT / 'configs/toy/pool.toml').read_text())['frs']
    if json_hash(effective_config(cfg)) != seal.get('effective_config_sha256') or \
            effective_config(cfg) != effective_config(production):
        raise ValueError('running effective config differs from sealed production config')
    if not cfg.get('trg_addendum_sha256') or cfg['trg_addendum_sha256'] != seal.get('trg_addendum_sha256'):
        raise ValueError('production requires sealed stage-(i) kappa addendum sha256')
    if cfg.get('trg_calibration_file'):
        from .trg_registration import read_addendum
        read_addendum(cfg)
    for seed in seeds:
        guard_seed(seed)
    if cfg.get("coverage_factory") != seal["coverage_factory"]:
        raise ValueError("coverage factory differs from seal")
    if cfg.get("job_modules", []) != seal["job_modules"]:
        raise ValueError("job modules differ from seal")
    if cfg.get("backend_factory", "") != seal.get("backend_factory", ""):
        raise ValueError("rule backend factory differs from seal")
    return seal
