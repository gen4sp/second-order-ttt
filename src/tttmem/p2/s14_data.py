"""Deterministic per-machine data preparation (no shared storage). This module does not import torch."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from .data import (PrepareCfg, extend_store, prepare_config, sha256_file, _HoldoutShingles,
                   split_contamination, open_split)


def _read(path):
    return json.loads(Path(path).read_text())


def verify_store(data_dir, expected_path) -> dict:
    root, expected = Path(data_dir), _read(expected_path)
    hashes = expected["sha256"]
    if not hashes or not {"meta.json", "prepare_summary.json"} <= set(hashes):
        raise ValueError("expected hashes must include metadata and prepare_summary.json")
    bad = []
    for name, digest in hashes.items():
        if Path(name).name != name:
            raise ValueError(f"invalid expected filename: {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            bad.append(name)
    if bad:
        raise ValueError(f"data sha256 mismatch: {bad}")
    summary = _read(root / "prepare_summary.json")
    if not summary.get("complete") or summary.get("counts") != expected["counts"]:
        raise ValueError("incomplete or wrong registered data counts")
    for name, digest in summary.get("sha256", {}).items():
        if hashes.get(name) != digest:
            raise ValueError(f"summary hash disagrees with shipped reference: {name}")
    for split, count in expected["counts"].items():
        rows, _ = open_split(root, split)
        if len(rows) != count:
            raise ValueError(f"wrong row count for {split}: {len(rows)} != {count}")
    return {"verified": True, "sha256": hashes, "counts": expected["counts"]}


def _contains(actual, expected, label="meta"):
    """Registered deterministic metadata must match; extra diagnostic fields may differ."""
    for key, value in expected.items():
        if key == "wall_time":
            continue
        if isinstance(value, dict):
            _contains(actual.get(key, {}), value, f"{label}.{key}")
        elif actual.get(key) != value:
            raise ValueError(f"rebuild provenance mismatch: {label}.{key}")


def prepare_data(cfg: dict, root: Path, *, need_c: bool) -> dict:
    d = cfg.get("data", {})
    data = Path(cfg.get("data_dir", "data/p2data"))
    repo = Path(cfg.get("repo_root", Path(__file__).resolve().parents[3]))
    expected_path = Path(d.get("expected_hashes", "configs/s14/data/expected-sha256.json"))
    if not expected_path.is_absolute():
        expected_path = repo / expected_path
    mode = d.get("mode", "rebuild")
    started = time.monotonic()
    if mode not in ("rebuild", "uploaded"):
        raise ValueError("data.mode must be rebuild or uploaded")
    prep_result = None
    marker = data / ".s14-base-verified.json"
    # Archives are disposable transfer intermediates, never pullable S14 outputs.
    scratch = root.parent / "tmp" / f"{root.name}-data"
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("data-base.tar", "data-C.tar"):
        (root / name).unlink(missing_ok=True)
        (root / (name + ".sha256")).unlink(missing_ok=True)
    if mode == "rebuild" and not marker.exists():
        # meta.json precedes prepare_summary and verification. Its presence alone
        # cannot distinguish a good base from an interrupted preparation.
        resolved = data.resolve()
        if resolved == Path(resolved.anchor) or resolved in (root.resolve(), root.resolve().parent):
            raise ValueError("data_dir must be a dedicated store directory")
        if data.exists():
            shutil.rmtree(data)
        reference = _read(expected_path.parent / "meta.json")
        p = PrepareCfg(out_dir=str(data), quotas=reference["quotas"],
                       dataset_repo=reference["dataset"]["repo"],
                       dataset_revision=reference["dataset"]["revision"],
                       tokenizer_repo=reference["tokenizer"]["repo"],
                       tokenizer_revision=reference["tokenizer"]["revision"],
                       min_chars=reference["min_chars"], seed=reference["seed"],
                       near_dup_stride=reference["near_dup_stride"], near_dup_n=reference["near_dup_n"],
                       batch_size=reference["batch_size"], workers=int(d.get("workers", 4)),
                       manifest_chars_per_token=False, pack_path=str(scratch / "data-base.tar"))
        prep_result = prepare_config(p)
        # Runtime timings and host worker count are not portable bytes. Keep them in the
        # job log; only after every immutable data hash AND deterministic provenance match
        # publish the original registered metadata bytes. Never overwrite a mismatched bin.
        _contains(_read(data / "meta.json"), reference)
        expected = _read(expected_path)
        bad = [n for n, h in expected["sha256"].items()
               if n not in ("meta.json", "prepare_summary.json") and sha256_file(data / n) != h]
        if bad:
            raise ValueError(f"rebuilt data differs from P2-1: {bad}")
        for name in ("meta.json", "prepare_summary.json"):
            (data / name).write_bytes((expected_path.parent / name).read_bytes())
        (scratch / "data-base.tar").unlink(missing_ok=True)
        (scratch / "data-base.tar.sha256").unlink(missing_ok=True)
    verified = verify_store(data, expected_path)
    # Publish only after all 13 hashes and row counts pass, atomically and last.
    tmp_marker = marker.with_suffix(".tmp")
    tmp_marker.write_text(json.dumps(verified, sort_keys=True) + "\n")
    tmp_marker.replace(marker)
    extension = None
    if need_c:
        records = _read(data / "extensions.json") if (data / "extensions.json").exists() else []
        records = [r for r in records if "C" in r.get("counts", {})]
        if not records and (mode == "rebuild" or d.get("rebuild_c", False)):
            # extend_store publishes C files before its extension record. Remove
            # that incomplete C only; a verified uploaded base stays untouched.
            for name in ("C.bin", "C.jsonl", "extend-C.meta.json"):
                (data / name).unlink(missing_ok=True)
            extension = extend_store(PrepareCfg(out_dir=str(data), extend=True,
                    extend_quotas={"C": int(d.get("c_rows", 25600))},
                    workers=int(d.get("workers", 4)),
                    extend_pack_path=str(scratch / "data-C.tar")))
            (scratch / "data-C.tar").unlink(missing_ok=True)
            (scratch / "data-C.tar.sha256").unlink(missing_ok=True)
        if not (data / "C.bin").exists() or not (data / "extensions.json").exists():
            raise ValueError("split C and its extension hash record are required; upload or rebuild C")
        records = _read(data / "extensions.json")
        records = [r for r in records if "C" in r.get("counts", {})]
        if not records:
            raise ValueError("no registered split-C extension hashes")
        for name, digest in records[-1]["sha256"].items():
            if Path(name).name != name or not (data / name).is_file() or sha256_file(data / name) != digest:
                raise ValueError(f"split-C sha256 mismatch: {name}")
        if not {"C.bin", "C.jsonl"} <= records[-1]["sha256"].keys():
            raise ValueError("split-C hashes must include tokens and manifest")
        if d.get("expected_c_hashes"):
            pinned = Path(d["expected_c_hashes"])
            if not pinned.is_absolute():
                pinned = repo / pinned
            expected_c = _read(pinned)
            for name, digest in expected_c["sha256"].items():
                if (mode=="rebuild" or extension is not None) and not name.endswith((".bin",".jsonl")):
                    # Fresh runtime metadata has different clocks/worker diagnostics.
                    # Immutable C tokens/manifests must still match the local build.
                    continue
                if Path(name).name != name or not (data/name).is_file() or sha256_file(data/name) != digest:
                    raise ValueError(f"pinned split-C sha256 mismatch: {name}")
            if expected_c["counts"] != records[-1]["counts"]:
                raise ValueError("pinned split-C count mismatch")
        rows, _ = open_split(data, "C")
        if len(rows) != int(records[-1]["counts"]["C"]) or len(rows) < int(d.get("c_rows", 25600)):
            raise ValueError("split-C row coverage is insufficient")
        contam = split_contamination(data, "C")
        if d.get("audit_shingles",False):
            meta = _read(data/"meta.json")
            audit = _HoldoutShingles(data,("eval","ho","calib"),
                                    int(meta.get("near_dup_n",64)),int(meta.get("near_dup_stride",32)))
            contam["sampled_64gram_hits"] = sum(audit(row) for row in rows)
            contam["hits"] += contam["sampled_64gram_hits"]
        if contam["hits"]:
            raise ValueError(f"split C contamination: {contam}")
        verified = verify_store(data, expected_path)  # extension must preserve all base bytes
        verified["C"] = {"hashes": records[-1]["sha256"], "contamination": contam}
    result = {"mode": mode, "wall_s": time.monotonic() - started,
              "verification": verified, "prep": prep_result, "extension": extension}
    (root / "data-verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
