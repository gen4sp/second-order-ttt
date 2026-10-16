"""Outcome-free TRG manifests; the execution adapters live in ``trg_pool``."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .trg_triggers import realised_drop_hash, yoked_sham

TRG_RPE_INTERSECTION_RULE = (
    "RPE's first 12 enrolled FO units in seed order with a valid t_acq state and "
    "t_acq <= 1100 are a subset of TRG's all eligible FO units; the intersection "
    "is RPE's enrolled units, reported before descriptive cross-route comparisons. "
    "Noise is TRG-only; TRG references the main, RPE its CE twin."
)
ARM_FAMILIES = {"e2e1emb": "FO", "e2e1emb_cn97": "noise", "e2e2emb": "SO"}


def _required(row, keys):
    missing = set(keys)-row.keys()
    if missing:
        raise ValueError("missing required fields: "+", ".join(sorted(missing)))


def adapt_state_row(row):
    """Accept FRS-A state rows or explicit TRG native rows.

    FRS-A file/sha256 rows are producer-validated records, without `valid`.
    Native state_path/state_sha256 rows require an explicit boolean `valid`.
    This normalizes metadata only: the caller must verify bytes/provenance
    with frs_state.load_state before execution. No-acquisition rows use None for
    all four step fields, path and hash; valid=False records validation failure.
    """
    _required(row, ("cell_id", "arm", "seed", "trigger", "event_step", "detected_step",
                    "state_step", "completed_updates"))
    if row["trigger"] != "t_acq":
        raise ValueError("TRG jobs require trigger == t_acq")
    if row["arm"] not in ARM_FAMILIES:
        raise ValueError("unknown TRG arm")
    family = ARM_FAMILIES[row["arm"]]
    if row.get("family", family) != family:
        raise ValueError("family/arm mismatch")
    native = "state_path" in row or "state_sha256" in row
    _required(row, ("state_path", "state_sha256", "valid") if native else ("file", "sha256"))
    valid = row["valid"] if "valid" in row else True
    if type(valid) is not bool:
        raise ValueError("valid must be an explicit boolean")
    return dict(row, family=family, seed=int(row["seed"]), valid=valid,
                state_path=row["state_path"] if native else row["file"],
                state_sha256=row["state_sha256"] if native else row["sha256"])


def make_job_specs(state_meta, cfg=None):
    """Return {disposition, jobs, ...}, including the registered intersection rule.

    Dispositions: no-acquisition, over-limit, validation-failed, enrolled.
    SO enrolment requires cfg['kappa'] and cfg['addendum_sha256']; neither defaults.
    `jobs` are local dictionaries, not FRS-A JobSpec objects.
    """
    cfg = cfg or {}
    m = adapt_state_row(state_meta)
    family, seed, at = m["family"], m["seed"], m["state_step"]
    result = dict(cell_id=m["cell_id"], family=family, arm=m["arm"], seed=seed,
                  disposition=None, jobs=[], trg_rpe_intersection_rule=TRG_RPE_INTERSECTION_RULE)
    if at is None:
        if any(m[k] is not None for k in ("event_step", "detected_step", "completed_updates")):
            raise ValueError("inconsistent no-acquisition step fields")
        return dict(result, disposition="no-acquisition")
    at = int(at)
    chronology = all(m[k] is not None and int(m[k]) == at
                     for k in ("event_step", "detected_step", "completed_updates"))
    digest = m["state_sha256"]
    if (not m["valid"] or not chronology or at <= 0 or not m["state_path"]
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        return dict(result, disposition="validation-failed")
    if at > (450 if family == "SO" else 1100):
        return dict(result, disposition="over-limit")
    base = dict(family=family, arm=m["arm"], cell_id=m["cell_id"], seed=seed,
                trigger="t_acq", state_path=m["state_path"], state_sha256=digest, state_step=at,
                start=at, end=660 if family == "SO" else at+305,
                reads={"light_every": 1, "ladder_every": 0}, telemetry_every=1,
                retry_limit=1, enrolment="outcome-free")
    prefix = f"trg-{family}-s{seed}"
    if family != "SO":
        specs = [dict(base, job_id=f"{prefix}-{tag}", branch=tag, outer=mask,
                     after=[], gradient_diagnostic_every=25,
                     intervention_rng=[seed, 4242, at] if tag == "rnd" else None)
                for tag, mask in (("kvm", "kv"), ("rnd", "rnd"))]
    else:
        _required(cfg, ("kappa", "addendum_sha256"))
        if cfg["kappa"] not in (3, 5, 10) or type(cfg["kappa"]) is not int:
            raise ValueError("kappa must be a sealed value in {3,5,10}")
        if not isinstance(cfg["addendum_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", cfg["addendum_sha256"]):
            raise ValueError("invalid addendum sha256")
        base.update(kappa=cfg["kappa"], addendum_sha256=cfg["addendum_sha256"])
        skip_id = f"{prefix}-skip"
        specs = [dict(base, job_id=skip_id, branch="skip", after=[],
                      drop_list_sha256=None, writes_realised_drops=True),
                 dict(base, job_id=f"{prefix}-sham", branch="sham", after=[skip_id],
                      drop_list_sha256=None, requires_realised_drop_hash=True, schedule=None)]
    return dict(result, disposition="enrolled", jobs=specs)


def resolve_yoked_job(spec, skip_record):
    if spec["after"] != [skip_record["job_id"]]:
        raise ValueError("wrong skip dependency")
    if skip_record.get("disposition") not in ("COMPLETE", "DIVERGED"):
        return dict(spec, blocked=True, disposition="TRUNCATED", reason="skip failed")
    drops, digest = skip_record["drops"], skip_record["drop_list_sha256"]
    if skip_record.get("drop_list_file"):
        data = Path(skip_record["drop_list_file"]).read_bytes()
        payload = json.loads(data)
        if hashlib.sha256(data).hexdigest() != digest or payload != dict(schema=1, drops=drops):
            raise ValueError("realised drop-list file sha256/payload mismatch")
    if realised_drop_hash(drops) != digest:
        raise ValueError("realised drop-list sha256 mismatch")
    return dict(spec, schedule=yoked_sham(drops, digest), drop_list_sha256=digest,
                realised_drops=list(drops), blocked=False)
