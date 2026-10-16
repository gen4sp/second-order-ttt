"""Score CHR run logs against the registered CHR rules and production contract.

Usage: python scripts/analyze_chr.py [--diagnostic] <out_dir> <run.log> [<retry.log>]

Reads the CHR_* lines of the run log and of at most one infrastructure-retry log,
checks the registered contract (production mode: protocol, 1500 steps, 5-step grid, CUDA,
the sealed plan/analyzer hashes, one scientific fingerprint across logs, the full sanity
gate set; ``--diagnostic`` reports violations instead of aborting and labels the output
as not a registered score), selects one attempt per planned cell, applies the eligibility
and evidence-validity predicates, derives the per-run events on the exact grid, applies
the H1, H2 (flag), H4, H5, H6 and H8 rules and the pre-declared sensitivities, and writes
``chr_scores.json``, ``chr_scores.txt``, ``chr_events.csv``, ``chr_series.csv`` and
``chr_disposition.csv``. Missing evidence is never read as 0 or as "not retained".
Pure Python + NumPy; no torch.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

BIND = 0.90
R_STAR = 0.02
R_SENS = (0.01, 0.03)
FO, SO, FN = "e2e1emb", "e2e2emb", "e2e1emb_cn97"
N_UNITS = 6
GRID = 5
STARVED_STRETCH = 20  # consecutive post-acquisition logs (spanning 95 steps)
BB_RATIO = 3.0
MATCH_FAIL_MAX = 0.01  # NC/NS norm-match failures tolerated over a branch's training sequences
NS_ATTAINED_MIN = 0.8  # NS: applied / requested removal over training below this -> under-matched
MED_INVALID_MAX = 0.05  # H8: share of manipulated events invalid or capped, per condition
MED_COUPLING_MAX = 0.02  # H8: |change of clean-event recall under CLAMP| above this -> coupled
MAX_LOGS = 2  # the run log and at most one retry log
PRODUCTION = dict(protocol="chr-full-v2", steps=1500, log_every=5, device="cuda")
EXPECTED_SANITY = (
    "cn97_ce_equals_e2e1emb", "decomposition_identity", "event_weights_ones_equal_ce",
    "event_weights_reproduce_rp", "inner_ce_equals_ttt", "nc0_equals_ce", "nc_norm_match",
    "ns100_equals_ce", "ns_changes_gradient", "ns_norm_match", "rp0_equals_ce", "rp_changes_gradient",
    "rs100_equals_ce", "rs_changes_gradient", "stream_ce_equals_eval_sequential", "r4_S1_init",
    "r4_S1_trained", "r4_S2_init_clip10", "r4_S2_trained_clip0.001", "r4_S2_trained_clip10", "r4_S3", "r4_S4")
INFRA_RE = re.compile(r"out of memory|outofmemory|cuda|nccl|cudnn|device-side|ecc|worker died|watchdog|"
                      r"killed|broken pipe|eoferror|connection", re.I)
KINDS = ("CHR_LOG", "CHR_FORK", "CHR_CELL", "CHR_CELL_ERROR", "CHR_SANITY", "CHR_SETTINGS",
         "CHR_SUMMARY", "CHR_MANIFEST", "CHR_PROTOCOL", "CHR_PROBE", "CHR_BRANCH")
TEXT_KINDS = ("CHR_BRANCH_ERROR", "CHR_DONE")  # plain-text lines, inventoried only
REQUIRED = {"CHR_LOG": ("cell_id", "branch", "step", "quick", "write"), "CHR_FORK": ("cell_id", "step"),
            "CHR_CELL": ("cell_id",), "CHR_CELL_ERROR": ("cell_id",), "CHR_SUMMARY": ("cell_id", "status"),
            "CHR_MANIFEST": ("cells", "steps"), "CHR_PROTOCOL": ("protocol",), "CHR_PROBE": ("cell_id", "step"),
            "CHR_BRANCH": ("branch",)}
NONCE_TRAIN_KEYS = ("train_n", "train_clip_count", "train_match_fail_count", "train_added_req_sum",
                    "train_added_actual_sum")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


ANALYZER_SHA256 = _sha256(Path(__file__).resolve())
PLAN_SHA256 = None  # the driver records the sha256 of an optional plan_file; none is configured


# --------------------------------------------------------------------------- parsing


def parse(path: str | Path) -> dict:
    """One log file. Malformed or schema-incomplete CHR_* records are listed with line
    numbers and reasons, never silently dropped; unknown CHR_ kinds are inventoried."""
    out = dict(path=str(path), logs={}, forks={}, cells={}, errors={}, summaries={}, sanity=None,
               settings=None, manifest=None, protocol=None, malformed=[], unknown_kinds={}, text={})
    with open(path, errors="replace") as fh:
        for no, line in enumerate(fh, 1):
            kind, _, rest = line.rstrip("\n").partition(" ")
            if not kind.startswith("CHR_"):
                continue
            if kind in TEXT_KINDS:
                out["text"].setdefault(kind, []).append(rest)
                continue
            if kind not in KINDS:
                out["unknown_kinds"][kind] = out["unknown_kinds"].get(kind, 0) + 1
                continue
            try:
                rec = json.loads(rest)
            except json.JSONDecodeError:
                out["malformed"].append((no, kind, "json"))
                continue
            missing = [k for k in REQUIRED.get(kind, ()) if not isinstance(rec, dict) or k not in rec]
            if missing:
                out["malformed"].append((no, kind, f"missing {missing}"))
                continue
            if kind == "CHR_LOG":
                out["logs"].setdefault((rec["cell_id"], rec["branch"]), []).append(rec)
            elif kind == "CHR_FORK":
                out["forks"][rec["cell_id"]] = rec
            elif kind == "CHR_CELL":
                out["cells"][rec["cell_id"]] = rec
            elif kind == "CHR_CELL_ERROR":
                out["errors"][rec["cell_id"]] = rec
            elif kind == "CHR_SUMMARY":
                out["summaries"][rec["cell_id"]] = rec
            elif kind in ("CHR_PROBE", "CHR_BRANCH"):
                pass  # syntax/schema-checked; their content is in CHR_LOG / CHR_CELL
            else:
                out[kind[4:].lower()] = rec
    for key, rows in out["logs"].items():
        rows.sort(key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{path}: duplicate log steps in {key}")
    return out


def sanity_ok(p: dict, production: bool) -> tuple[bool, str]:
    s = p["sanity"]
    if not s:
        return False, "CHR_SANITY missing"
    failed = [k for k, v in s.items() if not (isinstance(v, dict) and v.get("passed"))]
    if failed:
        return False, f"sanity failed: {failed}"
    absent = [k for k in EXPECTED_SANITY if k not in s]
    if production and absent:
        return False, f"sanity gates absent: {absent}"
    return True, "ok"


def contract_violations(parsed: list[dict]) -> list[str]:
    """Production contract: the sealed protocol on every log, one scientific fingerprint."""
    out = []
    for i, p in enumerate(parsed):
        pr, st, man = p["protocol"] or {}, p["settings"] or {}, p["manifest"] or {}
        if pr.get("protocol") != PRODUCTION["protocol"]:
            out.append(f"log {i}: protocol {pr.get('protocol')!r} != {PRODUCTION['protocol']!r}")
        if not str(pr.get("device", "")).startswith(PRODUCTION["device"]):
            out.append(f"log {i}: device {pr.get('device')!r} is not CUDA")
        if pr.get("analyzer_sha256") != ANALYZER_SHA256:
            out.append(f"log {i}: analyzer hash {pr.get('analyzer_sha256')} != this analyzer {ANALYZER_SHA256}")
        if pr.get("plan_sha256") != PLAN_SHA256:
            out.append(f"log {i}: plan hash {pr.get('plan_sha256')} != {PLAN_SHA256}")
        if man.get("steps") != PRODUCTION["steps"]:
            out.append(f"log {i}: manifest steps {man.get('steps')} != {PRODUCTION['steps']}")
        if st.get("log_every") != PRODUCTION["log_every"]:
            out.append(f"log {i}: log_every {st.get('log_every')} != {PRODUCTION['log_every']}")
    keys = ("science_sha256", "exp_chr_sha256", "exp_ttt_sha256", "gpu")
    ref = parsed[0]["protocol"] or {}
    for i, p in enumerate(parsed[1:], 1):
        pr = p["protocol"] or {}
        for k in keys:
            if pr.get(k) != ref.get(k):
                out.append(f"log {i}: {k} {pr.get(k)!r} differs from log 0 {ref.get(k)!r}")
    return out


def attempt_state(p: dict, cid: str) -> tuple[str, str | None]:
    """complete / branch_error (main record ok, a branch errored) / error / missing."""
    rec = p["cells"].get(cid)
    if rec is not None and "error" not in rec:
        berr = [b for b in rec.get("forks") or [] if "error" in b]
        if not berr:
            return "complete", None
        return "branch_error", "; ".join(f"{b.get('branch')}: {b.get('error')}" for b in berr)
    if rec is not None:
        return "error", str(rec.get("error"))
    if cid in p["errors"]:
        return "error", str(p["errors"][cid].get("error"))
    summ = p["summaries"].get(cid)
    return "missing", f"summary status {summ.get('status')}" if summ else "no record"


def infrastructure(state: str, reason: str | None) -> bool:
    return state == "missing" or bool(reason and INFRA_RE.search(reason))


def consolidate(parsed: list[dict], production: bool = True) -> dict:
    """One attempt per planned cell (first log's manifest): the first log's if complete; else,
    for an infrastructure failure only, the retry log's if complete there; else the attempt
    that has a main record (its errored branches are ineligible). Never replaces a complete
    attempt or a divergence (a result)."""
    if len(parsed) > MAX_LOGS:
        raise SystemExit(f"{len(parsed)} logs given; the protocol allows the run log and one retry log")
    for p in parsed:
        ok, why = sanity_ok(p, production)
        if not ok:
            raise SystemExit(f"{p['path']}: {why}; nothing is scored")
    violations = contract_violations(parsed)
    if production and violations:
        raise SystemExit("contract violations (use --diagnostic for a non-registered look):\n  "
                         + "\n  ".join(violations))
    first = parsed[0]
    planned = list((first["manifest"] or {}).get("cells") or sorted(set(first["cells"]) | set(first["errors"])))
    out = dict(logs={}, forks={}, cells={}, attempt={}, status={}, attempts_detail={}, planned=planned,
               settings=first["settings"] or {}, protocol=first["protocol"] or {},
               protocols=sorted(str((p["protocol"] or {}).get("protocol")) for p in parsed),
               production=production, contract_violations=violations,
               malformed={p["path"]: p["malformed"] for p in parsed if p["malformed"]},
               unknown_kinds={p["path"]: p["unknown_kinds"] for p in parsed if p["unknown_kinds"]},
               retry_ignored=[], steps=int((first["manifest"] or {}).get("steps") or 0) or None)
    for cid in planned:
        states = [attempt_state(p, cid) for p in parsed]
        s0, r0 = states[0]
        chosen, note = None, ""
        if s0 == "complete":
            chosen = 0
            if len(parsed) > 1 and states[1][0] != "missing":
                out["retry_ignored"].append(cid)
        elif len(parsed) > 1 and infrastructure(s0, r0) and states[1][0] == "complete":
            chosen, note = 1, f"retry (first attempt {s0}: {r0})"
        elif s0 == "branch_error":
            chosen, note = 0, f"branch error kept ({'infrastructure' if infrastructure(s0, r0) else 'non-infrastructure'}): {r0}"
        elif len(parsed) > 1 and infrastructure(s0, r0) and states[1][0] == "branch_error":
            chosen, note = 1, f"retry with branch error (first attempt {s0}: {r0}; retry: {states[1][1]})"
        out["attempts_detail"][cid] = dict(states=states, chosen=chosen, note=note)
        if chosen is None:
            out["status"][cid] = f"{s0}{'' if infrastructure(s0, r0) else ' (non-infrastructure)'}: {r0}"
            continue
        p = parsed[chosen]
        out["attempt"][cid] = chosen
        out["status"][cid] = states[chosen][0] + (f" [{note}]" if note else "")
        out["cells"][cid] = p["cells"][cid]
        if cid in p["forks"]:
            out["forks"][cid] = p["forks"][cid]
        for (c, b), rows in p["logs"].items():
            if c == cid:
                out["logs"][(c, b)] = rows
    return out


def seed_of(cell_id: str) -> int:
    return int(re.search(r"/s(\d+)/", cell_id).group(1))


def arm_of(cell_id: str) -> str:
    return cell_id.split("/")[2]


def chunks_of(data: dict) -> tuple[int, int]:
    st = data.get("settings") or {}
    return int(st.get("chunk_early", 5)), int(st.get("chunk_late", 6))


def grid_of(data: dict) -> int:
    return int((data.get("settings") or {}).get("log_every", GRID))


# --------------------------------------------------------------------------- series and events


def series(rows: list[dict], early: int = 5, late: int = 6) -> dict[str, np.ndarray]:
    e, l = str(early), str(late)

    def get(fn):
        vals = []
        for r in rows:
            try:
                v = fn(r)
            except (KeyError, TypeError, ZeroDivisionError):
                v = None
            vals.append(np.nan if v is None else float(v))
        return np.asarray(vals, dtype=np.float64)

    q, g, w = (lambda r: r["quick"]), (lambda r, c, k: r["quick"]["geo"][c][k]), (lambda r: r["write"])
    s = dict(step=get(lambda r: r["step"]), a=get(lambda r: q(r)["acc"]),
             a5=get(lambda r: q(r).get(f"acc{e}")), a6=get(lambda r: q(r).get(f"acc{l}")),
             r=get(lambda r: w(r)["r_kv_med"]), r_mean=get(lambda r: w(r)["r_kv_mean"]),
             kv_write=get(lambda r: w(r)["kv_write_med"]), g0=get(lambda r: w(r)["g0_med"]),
             clip=get(lambda r: w(r)["clip_c0"]), train_clip=get(lambda r: r.get("train_clip_c0")),
             kv_write_ce=get(lambda r: r["write_ce"]["kv_write_med"]),
             r_corrupted=get(lambda r: r["write_corrupted"]["r_kv_med"]))
    for name, key in (("bb", "bb"), ("slow", "slow"), ("mem", "mem"), ("cos", "cos_bb_head0"),
                      ("sig", "sigma")):
        s[f"{name}5"] = get(lambda r, k=key: g(r, e, k))
        s[f"{name}6"] = get(lambda r, k=key: g(r, l, k))
    with np.errstate(divide="ignore", invalid="ignore"):
        s["B6"] = s["bb6"] / s["bb5"]
    return s


def grid_check(rows: list[dict], first: int, last: int, every: int = GRID, early: int = 5,
               late: int = 6) -> dict:
    """Exact grid {first, first + every, …, last}: no missing or extra step, and finite
    quick accuracy (overall and per chunk) and kv residual at every step."""
    s = series(rows, early, late)
    expected = set(range(first, last + 1, every))
    have = set(int(x) for x in s["step"])
    missing, extra = sorted(expected - have), sorted(have - expected)
    bad = sorted(int(t) for t, *v in zip(s["step"], s["a"], s["a6"], s["r"], s["r_mean"])
                 if not all(np.isfinite(v)))
    return dict(ok=not missing and not extra and not bad, missing=missing[:20], n_missing=len(missing),
                extra=extra[:20], n_extra=len(extra), nonfinite=bad[:20], n_nonfinite=len(bad))


def _first(mask: np.ndarray, steps: np.ndarray, after: float = -math.inf, sustained: bool = True,
           gap: int = GRID, terminal: int | None = None) -> int | None:
    """First step > ``after`` where ``mask`` holds and (sustained) also holds at exactly
    ``step + gap``; at the registered terminal step the condition uses that step alone."""
    for i in range(len(steps)):
        if steps[i] <= after or not mask[i]:
            continue
        if not sustained:
            return int(steps[i])
        nxt = i + 1 < len(steps) and steps[i + 1] == steps[i] + gap and mask[i + 1]
        if nxt or (terminal is not None and int(steps[i]) == terminal):
            return int(steps[i])
    return None


def subgrid(s: dict, every: int) -> dict:
    keep = (s["step"] % every == 0)
    return {k: v[keep] for k, v in s.items()}


def episode_onset(below: np.ndarray, steps: np.ndarray, t: int, gap: int = GRID) -> int | None:
    """Start of the contiguous (exact-grid) run of ``below`` that contains step ``t``."""
    idx = np.flatnonzero(steps == t)
    if not len(idx) or not below[idx[0]]:
        return None
    i = int(idx[0])
    while i > 0 and below[i - 1] and steps[i - 1] == steps[i] - gap:
        i -= 1
    return int(steps[i])


def events(s: dict, r_star: float = R_STAR, sustained: bool = True, resid: str = "r",
           gap: int = GRID, terminal: int | None = None, acq_at: int | None = None) -> dict:
    """Chronology events; ``acq_at`` gives an inherited acquisition step (branches). A
    category needs a finite residual at the loss (else ``unknown``)."""
    st = s["step"]
    kw = dict(sustained=sustained, gap=gap, terminal=terminal)
    t_acq = acq_at if acq_at is not None else _first(s["a"] >= BIND, st, sustained=False)
    t_loss = None if t_acq is None else _first(s["a"] < BIND, st, after=t_acq, **kw)
    below = s[resid] < r_star
    t_starve = _first(below, st, **kw)
    t_acq6 = _first(s["a6"] >= BIND, st, sustained=False)
    t_loss6 = None if t_acq6 is None else _first(s["a6"] < BIND, st, after=t_acq6, **kw)
    t_bb = None if t_acq is None else _first(s["B6"] >= BB_RATIO, st, after=t_acq, **kw)
    t_on = None if t_loss is None else episode_onset(below, st, t_loss, gap)
    r_at_loss = None
    if t_loss is not None:
        r_at_loss = float(s[resid][np.flatnonzero(st == t_loss)[0]])
    if t_loss is None:
        category = None
    elif not math.isfinite(r_at_loss):
        category = "unknown"
    elif t_on is not None and t_on < t_loss:
        category = "before_persistent"
    elif t_on is not None:
        category = "simultaneous"
    elif t_starve is not None and t_starve < t_loss:
        category = "resolved_earlier_episode"
    elif t_starve is not None:
        category = "after_loss"
    else:
        category = "never_starved"
    return dict(t_acq=t_acq, t_loss=t_loss, t_starve=t_starve, t_onset_at_loss=t_on,
                r_at_loss=r_at_loss if r_at_loss is None or math.isfinite(r_at_loss) else None,
                category=category, t_acq6=t_acq6, t_loss6=t_loss6, t_bb=t_bb)


# --------------------------------------------------------------------------- verdict rules


def count_verdict(k: int, n: int) -> str:
    """H1/H4 rule: n = 6 → S ≥ 5, F ≤ 3; n < 6 → S needs n ≥ 5 and ≥ n−1, F ≤ n/2; n < 4 → NE."""
    if n < 4:
        return "NOT EVALUABLE"
    if n >= 6:
        return "SUPPORTED" if k >= 5 else "FALSIFIED" if k <= 3 else "INCONCLUSIVE"
    if n >= 5 and k >= n - 1:
        return "SUPPORTED"
    if k <= n / 2:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def h5_verdict(rp: int, nc: int, twin: int, n: int) -> str:
    if n < 4:
        return "NOT EVALUABLE"
    if twin >= 2:
        return "INCONCLUSIVE"
    if n >= 6:
        s_ok, f_low = rp >= 5, rp <= 2
    else:  # registered H5 rule: n = 4–5 → S needs n = 5 and rp ≥ 4; F if rp ≤ n/3
        s_ok, f_low = n >= 5 and rp >= n - 1, rp <= n / 3
    if s_ok and nc <= 1 and twin <= 1:
        return "SUPPORTED"
    if f_low or nc >= rp - 1:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def h6_verdict(rs: int, ns: int, twin: int, n: int) -> str:
    """Registered H6 rule: NE if n < 4; INCONCLUSIVE if the CE twin fails to retain in ≥ 2 units (no
    stable reference); F if rs01 retains in ≥ ⌈2n/3⌉; INCONCLUSIVE (non-specific) if ns01
    retains in ≤ n/2; S if n ≥ 5, rs01 ≤ 1 and ns01 ≥ n − 1; else INCONCLUSIVE."""
    if n < 4:
        return "NOT EVALUABLE"
    if n - twin >= 2:
        return "INCONCLUSIVE"
    if rs >= math.ceil(2 * n / 3):
        return "FALSIFIED"
    if ns <= n / 2:
        return "INCONCLUSIVE"
    if n >= 5 and rs <= 1 and ns >= n - 1:
        return "SUPPORTED"
    return "INCONCLUSIVE"


def selectivity(k_eval: int, k_sel: int, n: int, verdict: str, known: bool) -> str:
    """Registered common-eval qualifier of a SUPPORTED H5/H6: ``eval-carried`` if the eval rule
    alone carries the outcome in ≥ n − 1 units; ``training-selective`` if the treatment-trained
    state is selectively different from the CE- and control-trained states under the common
    eval rule in ≥ n − 1 units; else ``mixed``; ``unknown`` if a needed evaluation is missing."""
    if verdict != "SUPPORTED":
        return "n/a"
    if not known:
        return "unknown"
    if k_eval >= n - 1:
        return "eval-carried"
    if k_sel >= n - 1:
        return "training-selective"
    return "mixed"


def manipulation_qualifier(flags: list[bool]) -> str:
    """Registered rule: OK if at most one unit is flagged (weak / clip / cap / missing check), else LIMITED."""
    return "OK" if sum(flags) <= 1 else "LIMITED"


def exact_ci(k: int, n: int, level: float = 0.95) -> list[float] | None:
    """Clopper–Pearson interval for k of n (bisection on the binomial tail; no SciPy)."""
    if n <= 0:
        return None
    a = (1 - level) / 2

    def tail_ge(p, j=k):  # P(X >= j), increasing in p
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(j, n + 1))

    def solve(f, target):  # f increasing in p
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(tail_ge, a)
    high = 1.0 if k == n else solve(lambda p: tail_ge(p, k + 1), 1 - a)  # P(X <= k) = a
    return [round(low, 4), round(high, 4)]


def label_count(k: int, n: int) -> bool:
    """A unit-count label holds with ≥ n − 1 of n (n ≥ 5) or 4 of 4."""
    return (n >= 5 and k >= n - 1) or (n == 4 and k == 4)


def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def final_acc(rec: dict, tag: str = "ce") -> float | None:
    return _num(((rec.get("final") or {}).get(tag) or {}).get("acc"))


def retained(rec: dict, tag: str) -> bool | None:
    a = final_acc(rec, tag)
    return None if a is None else a >= BIND


def _ge(v: float | None) -> bool:
    return v is not None and v >= BIND


def _lt(v: float | None) -> bool:
    return v is not None and v < BIND


# --------------------------------------------------------------------------- eligibility


def main_eligible(rec: dict | None, steps: int | None) -> tuple[bool, str]:
    if rec is None:
        return False, "missing"
    if "error" in rec:
        return False, "error"
    if rec.get("diverged"):
        return False, "diverged"
    n = steps or rec.get("steps")
    if n is None or rec.get("steps_done") != n:
        return False, f"incomplete (steps_done {rec.get('steps_done')} of {n})"
    if final_acc(rec) is None:
        return False, "no finite final CE accuracy"
    return True, "ok"


def _sum_rows(rows: list[dict], key: str) -> float | None:
    """Sum of a per-window counter; None if any window lacks it (missing is not 0)."""
    if not rows:
        return None
    vals = [_num(r.get(key)) for r in rows]
    return None if any(v is None for v in vals) else float(sum(vals))


def branch_eligible(b: dict | None, steps: int | None, rows: list[dict], grid: int = GRID,
                    early: int = 5, late: int = 6) -> tuple[bool, str]:
    """Registered branch eligibility: as a main cell (incl. finite own-rule and CE finals), the
    three in-run identities, a complete exact grid from fork + grid, and for non-CE rules the
    training counters on every window (constant window volume); NC/NS norm-match failures
    ≤ 1 % of the training sequences."""
    if b is None:
        return False, "missing"
    if "error" in b:
        return False, f"error: {b.get('error')}"
    if b.get("diverged"):
        return False, "diverged"
    if steps is not None and b.get("steps_done") != steps:
        return False, f"incomplete (steps_done {b.get('steps_done')} of {steps})"
    if not (b.get("state_identical") and b.get("batch_identical") and b.get("main_batch_identical")):
        return False, "identity"
    rule = b.get("rule", "ce")
    if final_acc(b, rule) is None or final_acc(b, "ce") is None:
        return False, "no finite own-rule and CE finals"
    fork = b.get("fork_step")
    if fork is None or steps is None:
        return False, "no fork step / steps"
    g = grid_check(rows, fork + grid, steps, grid, early, late)
    if not g["ok"]:
        return False, f"grid ({g['n_missing']} missing, {g['n_extra']} extra, {g['n_nonfinite']} non-finite)"
    if rule != "ce":
        keys = NONCE_TRAIN_KEYS + (("train_cap_bound_count",) if rule.startswith("ns") else ())
        if any(_sum_rows(rows, k) is None for k in keys):
            return False, "training counters missing"
        if len({r["train_n"] for r in rows}) != 1 or not rows[0]["train_n"]:
            return False, "training windows of unequal volume"
        if rule[:2] in ("nc", "ns") and _sum_rows(rows, "train_match_fail_count") / _sum_rows(rows, "train_n") > MATCH_FAIL_MAX:
            return False, "norm match failed"
    return True, "ok"


def train_fraction(rows: list[dict], key: str) -> float | None:
    n, k = _sum_rows(rows, "train_n"), _sum_rows(rows, key)
    return None if not n or k is None else k / n


def train_write_ratio(rows: list[dict]) -> float | None:
    """Median over the branch's logged windows of kv_write(rule) / kv_write(CE), same state."""
    vals = [r["write"]["kv_write_med"] / r["write_ce"]["kv_write_med"] for r in rows
            if r.get("write_ce") and _num(r["write_ce"].get("kv_write_med"))
            and _num(r["write"].get("kv_write_med")) is not None]
    return float(np.median(vals)) if vals else None


def attained_ratio(rows: list[dict]) -> float | None:
    req, act = _sum_rows(rows, "train_added_req_sum"), _sum_rows(rows, "train_added_actual_sum")
    return None if not req or act is None else act / req


# --------------------------------------------------------------------------- scoring


def score(data: dict) -> dict:
    logs, cells, forks = data["logs"], data["cells"], data["forks"]
    steps = data.get("steps")
    early, late = chunks_of(data)
    grid = grid_of(data)
    ser = lambda rows: series(rows, early, late)  # noqa: E731
    e, l = str(early), str(late)
    elig = {cid: main_eligible(cells.get(cid), steps) for cid in data["planned"]}
    out: dict = dict(registered=bool(data.get("production")), contract_violations=data.get("contract_violations"),
                     chunks=[early, late], protocol=data.get("protocol"), protocols=data.get("protocols"),
                     malformed=data.get("malformed"), unknown_kinds=data.get("unknown_kinds"),
                     planned=len(data["planned"]), status=data["status"], attempts=data["attempt"],
                     retry_ignored=data.get("retry_ignored"),
                     ineligible={c: why for c, (ok, why) in elig.items() if not ok})
    by_arm: dict[str, list[str]] = {}
    for cid in data["planned"]:
        by_arm.setdefault(arm_of(cid), []).append(cid)
    for arm in by_arm:
        by_arm[arm].sort(key=seed_of)
    ok_cells = {arm: [c for c in ids if elig[c][0]] for arm, ids in by_arm.items()}
    fo = ok_cells.get(FO, [])
    terminal = lambda cid: steps or cells[cid].get("steps")  # noqa: E731
    grids = {cid: grid_check(logs.get((cid, "main"), []), grid, terminal(cid), grid, early, late)
             for ids in ok_cells.values() for cid in ids}
    out["grid"] = {c: g for c, g in grids.items() if not g["ok"]}

    def fo_events(cid, every=None, **kw):
        s = ser(logs.get((cid, "main"), []))
        if every:
            s = subgrid(s, every)
        return events(s, gap=every or grid, terminal=terminal(cid), **kw)

    # ---- H1 (units frozen by the primary definition; sensitivities on the same units)
    primary = []
    for cid in fo:
        if not grids[cid]["ok"]:
            continue
        ev = fo_events(cid)
        if ev["t_acq"] is not None and ev["t_loss"] is not None:
            primary.append(dict(cell_id=cid, seed=seed_of(cid), **ev,
                                supports=ev["category"] == "before_persistent",
                                lead=None if ev["t_onset_at_loss"] is None else ev["t_loss"] - ev["t_onset_at_loss"],
                                starve_before_acq=ev["t_starve"] is not None and ev["t_starve"] < ev["t_acq"]))
    used = primary[:N_UNITS]
    k = sum(u["supports"] for u in used)
    cats = {}
    for u in used:
        cats[u["category"]] = cats.get(u["category"], 0) + 1
    out["H1"] = dict(verdict=count_verdict(k, len(used)), k=k, n=len(used), ci95=exact_ci(k, len(used)),
                     n_evaluable_total=len(primary), categories=cats, units=used)

    def sens(support, **kw):
        """Frozen: the primary units (undefined under the variant = not supporting, listed);
        reselected (labeled, secondary): the first 6 clean-grid FO cells on which it is defined."""
        frozen, undefined = 0, []
        for u in used:
            ok = support(fo_events(u["cell_id"], **kw))
            if ok is None:
                undefined.append(u["cell_id"])
            frozen += bool(ok)
        res = [(c, ok) for c, ok in ((c, support(fo_events(c, **kw))) for c in fo if grids[c]["ok"])
               if ok is not None][:N_UNITS]
        kr = sum(ok for _, ok in res)
        return dict(frozen=dict(k=frozen, n=len(used), verdict=count_verdict(frozen, len(used)),
                                undefined=undefined),
                    reselected=dict(k=kr, n=len(res), verdict=count_verdict(kr, len(res)),
                                    units=[c for c, _ in res]))

    def persist(ev):
        if ev["t_loss"] is None or ev["category"] == "unknown":
            return None
        return ev["category"] == "before_persistent"

    def first_onset(ev):
        return None if ev["t_loss"] is None else (ev["t_starve"] is not None and ev["t_starve"] < ev["t_loss"])

    def loss6(ev):
        if ev["t_loss6"] is None:
            return None
        return ev["t_starve"] is not None and ev["t_starve"] < ev["t_loss6"]

    out["H1_sensitivity"] = {
        **{f"r_star_{r:g}": sens(persist, r_star=r) for r in R_SENS},
        "first_onset_v1": sens(first_onset),
        "single_log": sens(persist, sustained=False),
        "grid25": sens(persist, every=25),
        "r_mean": sens(persist, resid="r_mean"),
        "chunk6_loss_first_onset": sens(loss6),
    }

    # ---- H2 (descriptive + SUFFICIENCY-COUNTER flag)
    h2 = {}
    for arm in (SO, FN, FO):
        rows_out = []
        for cid in ok_cells.get(arm, []):
            s = ser(logs.get((cid, "main"), []))
            ev = events(s, gap=grid, terminal=terminal(cid))
            post = s["step"] > (ev["t_acq"] if ev["t_acq"] is not None else math.inf)
            starved = s["r"] < R_STAR
            stretch, best = 0, 0
            for i in range(len(starved)):
                contiguous = i > 0 and s["step"][i] == s["step"][i - 1] + grid
                if post[i] and starved[i] and s["a"][i] >= BIND:
                    stretch = stretch + 1 if (contiguous and stretch) else 1
                    best = max(best, stretch)
                else:
                    stretch = 0
            rows_out.append(dict(cell_id=cid, seed=seed_of(cid), retained=retained(cells[cid], "ce"),
                                 final_acc=final_acc(cells[cid]), t_acq=ev["t_acq"], grid_ok=grids[cid]["ok"],
                                 post_acq_starved_share=(float(starved[post].mean()) if post.any() else None),
                                 longest_starved_bound_logs=best, starved_but_bound=best >= STARVED_STRETCH))
        h2[arm] = rows_out
    so_sbb = sum(r["starved_but_bound"] for r in h2.get(SO, []))
    out["H2"] = dict(cells=h2, SUFFICIENCY_COUNTER=so_sbb >= 2, so_starved_but_bound=so_sbb,
                     retained={arm: sum(bool(r["retained"]) for r in rows) for arm, rows in h2.items()},
                     n={arm: len(rows) for arm, rows in h2.items()})

    # ---- H3 (descriptive order)
    out["H3"] = [dict(cell_id=u["cell_id"], t_acq=u["t_acq"], t_starve=u["t_starve"],
                      t_onset_at_loss=u["t_onset_at_loss"], t_bb=u["t_bb"], t_loss6=u["t_loss6"],
                      t_loss=u["t_loss"]) for u in used]

    # ---- H4 (D and NAT from the streamed accuracies; the gate needs prediction agreement).
    # Selection order: eligible FO mains in seed order with D ≥ .20 → drop gate failures and
    # incomplete intervention records → first 6 (a dropped unit is replaced by the next).
    def h4(variant, descriptive=False):
        units = []
        for cid in fo:
            fin = (cells[cid].get("final") or {}).get("ce") or {}
            iv = fin.get("intervention") or {}
            a5, a6 = _num(fin.get(f"acc{e}")), _num(fin.get(f"acc{l}"))
            if a5 is None or a6 is None or a5 - a6 < 0.20:
                continue
            vals = [_num((iv.get(k) or {}).get(c)) for k, c in ((variant, l), (f"RND_{variant}", l),
                                                                 (f"SHAM5_{variant}", e))]
            valid = bool(fin.get("decomp_gate_ok")) and all(v is not None for v in vals)
            d = a5 - a6
            delta = None if vals[0] is None else vals[0] - a6
            ctrl = None if vals[1] is None else vals[1] - a6
            units.append(dict(cell_id=cid, seed=seed_of(cid), valid=valid, D=d, delta6=delta, delta6_rnd=ctrl,
                              sham5_delta5=None if vals[2] is None else vals[2] - a5,
                              repaired=valid and delta >= d / 2 and delta - ctrl >= 0.05,
                              alpha=iv.get(f"alpha_{variant.lower()}_median")))
        used4 = [u for u in units if u["valid"]][:N_UNITS]
        k4 = sum(u["repaired"] for u in used4)
        res = dict(k=k4, n=len(used4), ci95=exact_ci(k4, len(used4)), n_invalid=sum(not u["valid"] for u in units),
                   invalid=[u["cell_id"] for u in units if not u["valid"]], units=used4)
        res["verdict"] = "descriptive (no registered rule)" if descriptive else count_verdict(k4, len(used4))
        return res

    out["H4a_SLOW"], out["H4b_BB"] = h4("SLOW"), h4("BB")
    out["H4c_HEAD_descriptive"] = h4("HEAD", descriptive=True)

    def harm(cid, branch, fin):
        iv = (fin or {}).get("intervention") or {}
        nat = _num((iv.get("NAT") or {}).get(l))
        if nat is None:
            return None
        d = {v: _num((iv.get(v) or {}).get(l)) for v in ("SLOW", "BB", "HEAD") if v in iv}
        d = {k: (None if x is None else x - nat) for k, x in d.items()}
        return dict(cell_id=cid, branch=branch, **d,
                    harmed=any(x is not None and x < -0.02 for k, x in d.items() if k != "HEAD"))

    no_harm = []
    for cid in sorted(c for ids in ok_cells.values() for c in ids):
        cell = cells[cid]
        if arm_of(cid) != FO:
            no_harm.append(harm(cid, "main", (cell.get("final") or {}).get("ce")))
        for b in cell.get("forks") or []:
            no_harm.append(harm(cid, b.get("branch"), (b.get("final") or {}).get(b.get("rule"))))
    out["H4_no_harm"] = [x for x in no_harm if x]

    # ---- branch helpers
    def branches_of(cid):
        return {b.get("rule"): b for b in cells[cid].get("forks") or []}

    def brows(cid, b):
        return logs.get((cid, b.get("branch")), []) if b else []

    def bvalid(cid, tag):
        b = branches_of(cid).get(tag)
        return branch_eligible(b, steps or cells[cid].get("steps"), brows(cid, b), grid, early, late)[1]

    def unit_valid(cid, tags):
        why = {t: bvalid(cid, t) for t in tags}
        return all(v == "ok" for v in why.values()), why

    # ---- H5 (pair-specific validity: twin + the dose's rp and nc). Selection order: eligible
    # FO mains with a fork, not retained, in seed order → drop invalid pairs → first 6.
    def h5(rp_tag, nc_tag):
        units = []
        for cid in fo:
            bs = branches_of(cid)
            if not bs or retained(cells[cid], "ce") is not False:
                continue
            ok, why = unit_valid(cid, ("ce", rp_tag, nc_tag))
            fk = forks.get(cid, {})
            w = fk.get("write_at_fork", {})
            ce_w, rp_w = _num((w.get("ce") or {}).get("kv_write_med")), _num((w.get(rp_tag) or {}).get("kv_write_med"))
            m1_fork = None if not ce_w or rp_w is None else rp_w / ce_w
            rp_rows = brows(cid, bs.get(rp_tag))
            m1_train = train_write_ratio(rp_rows)
            clip_fork = _num((w.get(rp_tag) or {}).get("clip_c0"))
            clip_train = train_fraction(rp_rows, "train_clip_count")
            cap_hits = [r["train_cap_hit_mean"] for r in rp_rows if _num(r.get("train_cap_hit_mean")) is not None]
            weak = m1_train is None or m1_train < 2
            clip_conf = clip_fork is None or clip_fork > .5 or clip_train is None or clip_train > .5
            twin, rpb, ncb = bs.get("ce") or {}, bs.get(rp_tag) or {}, bs.get(nc_tag) or {}
            units.append(dict(
                cell_id=cid, seed=seed_of(cid), valid=ok, why=why,
                fork_step=next(iter(bs.values())).get("fork_step"),
                retained_twin=retained(twin, "ce"), retained_rp=retained(rpb, rp_tag),
                retained_nc=retained(ncb, nc_tag),
                M1_fork=m1_fork, M1_train=m1_train, weak_manipulation=weak, clip_fork=clip_fork,
                clip_train=clip_train, clip_confounded=clip_conf,
                nc_attained=attained_ratio(brows(cid, bs.get(nc_tag))),
                rp_cap_hit_median=float(np.median(cap_hits)) if cap_hits else None,
                ce_main_under_rp_eval=final_acc(cells[cid], rp_tag),
                twin_under_rp_eval=final_acc(twin, rp_tag), nc_under_rp_eval=final_acc(ncb, rp_tag),
                rp_under_ce_eval=final_acc(rpb, "ce"), nc_under_ce_eval=final_acc(ncb, "ce"),
                quick_at_fork={t: (fk.get("quick_at_fork", {}).get(t) or {}).get("acc") for t in ("ce", rp_tag, nc_tag)}))
        used5 = [u for u in units if u["valid"]][:N_UNITS]
        n = len(used5)
        rp = sum(bool(u["retained_rp"]) for u in used5)
        nc = sum(bool(u["retained_nc"]) for u in used5)
        twin = sum(bool(u["retained_twin"]) for u in used5)
        verdict = h5_verdict(rp, nc, twin, n)
        known = all(u[k] is not None for u in used5
                    for k in ("ce_main_under_rp_eval", "twin_under_rp_eval", "nc_under_rp_eval"))
        k_eval = sum(_ge(u["ce_main_under_rp_eval"]) for u in used5)
        k_sel = sum(bool(u["retained_rp"]) and _lt(u["twin_under_rp_eval"]) and _lt(u["nc_under_rp_eval"])
                    and _lt(u["ce_main_under_rp_eval"]) for u in used5)
        return dict(verdict=verdict, rp=rp, nc=nc, twin=twin, n=n,
                    ci95=dict(rp=exact_ci(rp, n), nc=exact_ci(nc, n), twin=exact_ci(twin, n)),
                    manipulation=manipulation_qualifier([u["weak_manipulation"] or u["clip_confounded"]
                                                         for u in used5]),
                    selectivity=selectivity(k_eval, k_sel, n, verdict, known),
                    ce_main_retained_under_rp_eval=k_eval, selective_under_rp_eval=k_sel,
                    twin_retained_under_rp_eval=sum(_ge(u["twin_under_rp_eval"]) for u in used5),
                    nc_retained_under_rp_eval=sum(_ge(u["nc_under_rp_eval"]) for u in used5),
                    rp_retained_under_ce_eval=sum(_ge(u["rp_under_ce_eval"]) for u in used5),
                    nc_retained_under_ce_eval=sum(_ge(u["nc_under_ce_eval"]) for u in used5),
                    n_invalid=sum(not u["valid"] for u in units),
                    invalid=[(u["cell_id"], u["why"]) for u in units if not u["valid"]], units=used5)

    out["H5_primary_r03"] = h5("rp03", "nc03")
    out["H5_secondary_r10"] = h5("rp10", "nc10")

    # ---- H6 (necessity of the kv residual for the noise rescue). Selection order: eligible
    # noise mains with a fork, retained, in seed order → drop invalid (twin, rs, ns) → first 6.
    def h6(rs_tag, ns_tag):
        units = []
        for cid in ok_cells.get(FN, []):
            bs = branches_of(cid)
            if not bs or retained(cells[cid], "ce") is not True:
                continue
            ok, why = unit_valid(cid, ("ce", rs_tag, ns_tag))
            fk = forks.get(cid, {})
            ratio, capf = {}, {}
            for probe in ("write_at_fork", "write_at_fork_corrupted"):
                w = fk.get(probe, {})
                ce_w, rs_w = _num((w.get("ce") or {}).get("kv_write_med")), _num((w.get(rs_tag) or {}).get("kv_write_med"))
                ratio[probe] = None if not ce_w or rs_w is None else rs_w / ce_w
                capf[probe] = _num((w.get(ns_tag) or {}).get("cap_bound"))
            rs_rows, ns_rows = brows(cid, bs.get(rs_tag)), brows(cid, bs.get(ns_tag))
            ratio_train = train_write_ratio(rs_rows)
            ns_att = attained_ratio(ns_rows)
            cap_train = train_fraction(ns_rows, "train_cap_bound_count")
            weak = ratio_train is None or ratio_train > 0.5
            cap_flag = any(v is None or v > .5 for v in (*capf.values(), cap_train))  # missing = flagged
            under = ns_att is None or ns_att < NS_ATTAINED_MIN
            twin, rsb, nsb = bs.get("ce") or {}, bs.get(rs_tag) or {}, bs.get(ns_tag) or {}
            units.append(dict(
                cell_id=cid, seed=seed_of(cid), valid=ok, why=why,
                fork_step=next(iter(bs.values())).get("fork_step"),
                retained_twin=retained(twin, "ce"), retained_rs=retained(rsb, rs_tag),
                retained_ns=retained(nsb, ns_tag),
                rs_under_ce_eval=final_acc(rsb, "ce"), ns_under_ce_eval=final_acc(nsb, "ce"),
                twin_under_rs_eval=final_acc(twin, rs_tag), ns_under_rs_eval=final_acc(nsb, rs_tag),
                ce_main_under_rs_eval=final_acc(cells[cid], rs_tag),
                kv_write_ratio_fork_clean=ratio["write_at_fork"],
                kv_write_ratio_fork_corrupted=ratio["write_at_fork_corrupted"],
                kv_write_ratio_train=ratio_train, weak_manipulation=weak,
                ns_cap_bound_fork_clean=capf["write_at_fork"], ns_cap_bound_fork_corrupted=capf["write_at_fork_corrupted"],
                ns_cap_bound_train=cap_train, ns_cap_flag=cap_flag,
                ns_attained_train=ns_att, ns_under_matched=under))
        used6 = [u for u in units if u["valid"]][:N_UNITS]
        n = len(used6)
        rs = sum(bool(u["retained_rs"]) for u in used6)
        ns = sum(bool(u["retained_ns"]) for u in used6)
        twin = sum(bool(u["retained_twin"]) for u in used6)
        verdict = h6_verdict(rs, ns, twin, n)
        known = all(u[k] is not None for u in used6 for k in ("rs_under_ce_eval", "ns_under_ce_eval"))
        k_eval = sum(_ge(u["rs_under_ce_eval"]) for u in used6)
        k_sel = sum(_lt(u["rs_under_ce_eval"]) and _ge(u["ns_under_ce_eval"]) and bool(u["retained_twin"])
                    for u in used6)
        return dict(verdict=verdict, rs=rs, ns=ns, twin=twin, n=n,
                    ci95=dict(rs=exact_ci(rs, n), ns=exact_ci(ns, n), twin=exact_ci(twin, n)),
                    manipulation=manipulation_qualifier([u["weak_manipulation"] or u["ns_cap_flag"]
                                                         or u["ns_under_matched"] for u in used6]),
                    selectivity=selectivity(k_eval, k_sel, n, verdict, known),
                    rs_retained_under_ce_eval=k_eval, selective_under_ce_eval=k_sel,
                    ns_retained_under_ce_eval=sum(_ge(u["ns_under_ce_eval"]) for u in used6),
                    ce_main_retained_under_rs_eval=sum(_ge(u["ce_main_under_rs_eval"]) for u in used6),
                    ns_retained_under_rs_eval=sum(_ge(u["ns_under_rs_eval"]) for u in used6),
                    n_invalid=sum(not u["valid"] for u in units),
                    invalid=[(u["cell_id"], u["why"]) for u in units if not u["valid"]], units=used6)

    out["H6_r01"] = h6("rs01", "ns01")

    # ---- H8 (CLAMP sensitivity of the corrupted-event advantage at fixed weights)
    def med_row(cid):
        m = cells[cid].get("mediation") or {}
        conds = ("NAT", "CLAMP", "FLOOR")
        if not m or not all(isinstance(m.get(c), dict) for c in conds):
            return None
        d = {c: _num(m[c].get("d")) for c in conds}
        n_c, n_k = m.get("n_corrupted") or 0, m.get("n_clean") or 0
        why = []
        if any(v is None for v in d.values()):
            why.append("non-finite d")
        for c, den in (("CLAMP", n_c), ("FLOOR", n_k)):
            bad = (m[c].get("n_invalid_events"), m[c].get("n_capped_events"))
            if any(x is None for x in bad) or not den or sum(bad) / den > MED_INVALID_MAX:
                why.append(f"{c} invalid/capped events {bad} of {den}")
        if any((m[c].get("n_nonfinite_norms") is None or m[c]["n_nonfinite_norms"] > 0) for c in conds):
            why.append("non-finite inner norms")
        d0 = d["NAT"]
        acc = {f"{g}_{c}": _num(m[c].get(f"acc_{g}")) for g in ("corr", "clean") for c in conds}
        row = dict(cell_id=cid, seed=seed_of(cid), arm=arm_of(cid), valid=not why, why=why,
                   d_NAT=d0, d_CLAMP=d["CLAMP"], d_FLOOR=d["FLOOR"], n_corrupted=n_c, n_clean=n_k,
                   **{f"clip_{c}": _num(m[c].get("clip_c0")) for c in conds},
                   **{f"drive_{g}_{c}": _num(m[c].get(f"drive_{g}_med")) for g in ("corr", "clean") for c in conds},
                   r_clean_med=m.get("r_clean_med"), r_corr_med=m.get("r_corr_med"), **acc,
                   **{f"d{c}_{k}": m[k].get(f"d{c}") for c in (e, l) for k in conds})
        ok = d0 is not None and abs(d0) >= 1e-12
        row["m_CLAMP"] = 1 - d["CLAMP"] / d0 if ok and d["CLAMP"] is not None else None
        row["m_FLOOR"] = 1 - d["FLOOR"] / d0 if ok and d["FLOOR"] is not None else None
        dc = (None if acc["corr_CLAMP"] is None or acc["corr_NAT"] is None else acc["corr_CLAMP"] - acc["corr_NAT"])
        dk = (None if acc["clean_CLAMP"] is None or acc["clean_NAT"] is None else acc["clean_CLAMP"] - acc["clean_NAT"])
        row.update(dacc_corr_CLAMP=dc, dacc_clean_CLAMP=dk,
                   coupled=dk is None or abs(dk) > MED_COUPLING_MAX,
                   out_of_range=row["m_CLAMP"] is not None and not (-.25 <= row["m_CLAMP"] <= 1.25))
        return row

    # Selection order: eligible FO mains in seed order → drop missing / invalid mediation records
    # (listed) → those with d_NAT ≥ .05 → first 6.
    rows8, missing8 = [], []
    for cid in fo:
        r = med_row(cid)
        if r is None:
            missing8.append((cid, ["no mediation record"]))
        elif r["d_NAT"] is None or r["d_NAT"] >= 0.05:
            rows8.append(r)
    units8 = [r for r in rows8 if r["valid"]][:N_UNITS]
    n8 = len(units8)
    hi = sum(u["m_CLAMP"] is not None and u["m_CLAMP"] >= .75 for u in units8)
    lo = sum(u["m_CLAMP"] is not None and -.25 <= u["m_CLAMP"] <= .25 for u in units8)
    verdict8 = ("NOT EVALUABLE" if n8 < 4 else "CLAMP-SENSITIVE" if label_count(hi, n8) else
                "CLAMP-RESISTANT" if label_count(lo, n8) else "MIXED")
    out["H8"] = dict(verdict=verdict8, n=n8, k_sensitive=hi, k_resistant=lo,
                     n_coupled=sum(u["coupled"] for u in units8), n_out_of_range=sum(u["out_of_range"] for u in units8),
                     ci95=dict(sensitive=exact_ci(hi, n8), resistant=exact_ci(lo, n8)), units=units8,
                     invalid=missing8 + [(r["cell_id"], r["why"]) for r in rows8 if not r["valid"]],
                     descriptive=[r for r in (med_row(c) for c in sorted(c for ids in ok_cells.values() for c in ids))
                                  if r and r["arm"] != FO])

    # ---- disposition of every planned cell
    disp = []
    unit_keys = ("H1", "H4a_SLOW", "H4b_BB", "H5_primary_r03", "H5_secondary_r10", "H6_r01", "H8")
    units_of = {key: {u["cell_id"] for u in out[key]["units"]} for key in unit_keys}
    for cid in data["planned"]:
        ok, why = elig[cid]
        rec = cells.get(cid) or {}
        ev = fo_events(cid) if ok and arm_of(cid) == FO and grids[cid]["ok"] else {}
        tags = [b.get("rule") for b in rec.get("forks") or []]
        disp.append(dict(cell_id=cid, arm=arm_of(cid), seed=seed_of(cid), attempt=data["attempt"].get(cid),
                         status=data["status"].get(cid), eligible=ok, reason=why,
                         grid_ok=grids[cid]["ok"] if cid in grids else None,
                         final_acc=final_acc(rec) if rec else None, forked=rec.get("forked"),
                         t_acq=ev.get("t_acq"), t_loss=ev.get("t_loss"),
                         branches=json.dumps({t: bvalid(cid, t) for t in tags}) if ok and tags else "",
                         **{f"unit_{k}": cid in v for k, v in units_of.items()}))
    out["disposition"] = disp
    return out


# --------------------------------------------------------------------------- outputs


def write_outputs(data: dict, scores: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    early, late = chunks_of(data)
    grid = grid_of(data)
    (out / "chr_scores.json").write_text(json.dumps(scores, indent=1, default=str))
    with open(out / "chr_series.csv", "w", newline="") as fh:
        w = None
        for (cid, branch), rows in sorted(data["logs"].items()):
            s = series(rows, early, late)
            keys = list(s)
            if w is None:
                w = csv.writer(fh)
                w.writerow(["cell_id", "branch", *keys])
            for i in range(len(rows)):
                w.writerow([cid, branch, *[("" if not np.isfinite(s[k][i]) else f"{s[k][i]:.8g}") for k in keys]])
    with open(out / "chr_events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["t_acq", "t_starve", "t_onset_at_loss", "category", "r_at_loss", "t_bb", "t_acq6", "t_loss6",
                "t_loss"]
        w.writerow(["cell_id", "branch", *cols, "final_acc_own"])
        steps = data.get("steps")
        for (cid, branch), rows in sorted(data["logs"].items()):
            cell = data["cells"].get(cid, {})
            if branch == "main":
                ev = events(series(rows, early, late), gap=grid, terminal=steps or cell.get("steps"))
                fa = final_acc(cell)
            else:
                b = next((x for x in cell.get("forks") or [] if x.get("branch") == branch), {})
                q = ((data["forks"].get(cid) or {}).get("quick_at_fork") or {}).get(b.get("rule"), {})
                inherited = b.get("fork_step") if (q.get("acc") or 0) >= BIND else None
                ev = events(series(rows, early, late), gap=grid, terminal=steps or cell.get("steps"),
                            acq_at=inherited)
                fa = final_acc(b, b.get("rule", "ce")) if b else None
            w.writerow([cid, branch, *[ev[c] for c in cols], fa])
    with open(out / "chr_disposition.csv", "w", newline="") as fh:
        rows = scores["disposition"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["cell_id"])
        w.writeheader()
        w.writerows(rows)
    head = ("REGISTERED SCORE" if scores["registered"] else
            "DIAGNOSTIC — NOT A REGISTERED SCORE; contract violations: " + str(scores["contract_violations"] or "none"))
    lines = [head,
             f"protocol {scores['protocol'].get('protocol')} ({scores['protocols']}); planned {scores['planned']}; "
             f"chunks {scores['chunks']}; malformed {scores['malformed'] or 'none'}; unknown kinds "
             f"{scores['unknown_kinds'] or 'none'}; retry ignored {scores['retry_ignored'] or 'none'}",
             f"ineligible: {scores['ineligible'] or 'none'}",
             f"grid issues: {scores['grid'] or 'none'}"]
    h1 = scores["H1"]
    lines.append(f"H1 (r* .02, persistent episode strictly before the sustained loss): {h1['verdict']} "
                 f"k={h1['k']}/{h1['n']} CI95 {h1['ci95']} (evaluable total {h1['n_evaluable_total']}); "
                 f"categories {h1['categories']}")
    for u in h1["units"]:
        lines.append(f"  {u['cell_id']}: acq {u['t_acq']} first-starve {u['t_starve']} onset@loss "
                     f"{u['t_onset_at_loss']} bb {u['t_bb']} loss6 {u['t_loss6']} loss {u['t_loss']} "
                     f"r@loss {u['r_at_loss']} -> {u['category']}")
    for key, v in scores["H1_sensitivity"].items():
        f, r = v["frozen"], v["reselected"]
        lines.append(f"  sens {key}: frozen {f['verdict']} k={f['k']}/{f['n']} (undefined {len(f['undefined'])}); "
                     f"reselected {r['verdict']} k={r['k']}/{r['n']}")
    h2 = scores["H2"]
    lines.append(f"H2 retained {h2['retained']} of {h2['n']}; SO starved-but-bound {h2['so_starved_but_bound']}"
                 f" -> SUFFICIENCY_COUNTER={h2['SUFFICIENCY_COUNTER']}")
    for key in ("H4a_SLOW", "H4b_BB", "H4c_HEAD_descriptive"):
        v = scores[key]
        lines.append(f"{key}: {v['verdict']} k={v['k']}/{v['n']} CI95 {v['ci95']} (invalid {v['n_invalid']})")
        for u in v["units"]:
            lines.append(f"  {u['cell_id']}: D {u['D']:.3f} d6 {u['delta6']:+.3f} rnd {u['delta6_rnd']:+.3f} "
                         f"sham5 {u['sham5_delta5']:+.3f} alpha {u['alpha']} -> {'R' if u['repaired'] else '-'}")
    harmed = [f"{x['cell_id']}|{x['branch']}" for x in scores["H4_no_harm"] if x["harmed"]]
    lines.append(f"H4 no-harm (non-FO mains + all branches, Δ6 ≥ −.02): {len(scores['H4_no_harm'])} finals, "
                 f"harmed {harmed}")
    for key in ("H5_primary_r03", "H5_secondary_r10"):
        v = scores[key]
        lines.append(f"{key}: {v['verdict']} [manipulation {v['manipulation']}; selectivity {v['selectivity']}] "
                     f"rp={v['rp']} nc={v['nc']} twin={v['twin']} n={v['n']} CI95 {v['ci95']} (invalid "
                     f"{v['n_invalid']}); under rp eval: CE main {v['ce_main_retained_under_rp_eval']}, twin "
                     f"{v['twin_retained_under_rp_eval']}, nc {v['nc_retained_under_rp_eval']}, selective "
                     f"{v['selective_under_rp_eval']}; under CE eval: rp {v['rp_retained_under_ce_eval']}, nc "
                     f"{v['nc_retained_under_ce_eval']}")
        for u in v["units"]:
            lines.append(f"  {u['cell_id']}: fork {u['fork_step']} twin {u['retained_twin']} rp {u['retained_rp']}"
                         f" nc {u['retained_nc']} M1 fork/train {u['M1_fork']}/{u['M1_train']} clip fork/train "
                         f"{u['clip_fork']}/{u['clip_train']} cap-hit {u['rp_cap_hit_median']} nc-attained "
                         f"{u['nc_attained']}")
        for cid, why in v["invalid"]:
            lines.append(f"  invalid {cid}: {why}")
    v = scores["H6_r01"]
    lines.append(f"H6_r01: {v['verdict']} [manipulation {v['manipulation']}; selectivity {v['selectivity']}] "
                 f"rs={v['rs']} ns={v['ns']} twin={v['twin']} n={v['n']} CI95 {v['ci95']} (invalid "
                 f"{v['n_invalid']}); under CE eval: rs {v['rs_retained_under_ce_eval']}, ns "
                 f"{v['ns_retained_under_ce_eval']}, selective {v['selective_under_ce_eval']}; under rs eval: CE "
                 f"main {v['ce_main_retained_under_rs_eval']}, ns {v['ns_retained_under_rs_eval']}")
    for u in v["units"]:
        lines.append(f"  {u['cell_id']}: fork {u['fork_step']} twin {u['retained_twin']} rs {u['retained_rs']}/"
                     f"{u['rs_under_ce_eval']} ns {u['retained_ns']}/{u['ns_under_ce_eval']} CE@rs "
                     f"{u['ce_main_under_rs_eval']} ratio fork clean/corr/train {u['kv_write_ratio_fork_clean']}/"
                     f"{u['kv_write_ratio_fork_corrupted']}/{u['kv_write_ratio_train']} ns cap fork clean/corr/train "
                     f"{u['ns_cap_bound_fork_clean']}/{u['ns_cap_bound_fork_corrupted']}/{u['ns_cap_bound_train']} "
                     f"ns attained {u['ns_attained_train']}")
    for cid, why in v["invalid"]:
        lines.append(f"  invalid {cid}: {why}")
    v = scores["H8"]
    lines.append(f"H8: {v['verdict']} n={v['n']} sensitive={v['k_sensitive']} resistant={v['k_resistant']} "
                 f"coupled={v['n_coupled']} out-of-range={v['n_out_of_range']} CI95 {v['ci95']}")
    for u in v["units"] + v["descriptive"]:
        lines.append(f"  {u['cell_id']}: d NAT {u['d_NAT']} CLAMP {u['d_CLAMP']} FLOOR {u['d_FLOOR']}"
                     f" m_CLAMP {u['m_CLAMP']} m_FLOOR {u['m_FLOOR']} Δacc corr/clean under CLAMP "
                     f"{u['dacc_corr_CLAMP']}/{u['dacc_clean_CLAMP']} clip {u['clip_NAT']}/{u['clip_CLAMP']}/"
                     f"{u['clip_FLOOR']} valid {u['valid']} {u['why'] or ''}")
    for cid, why in v["invalid"]:
        lines.append(f"  invalid {cid}: {why}")
    (out / "chr_scores.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv: list[str]) -> None:
    args = argv[1:]
    production = "--diagnostic" not in args
    args = [a for a in args if a != "--diagnostic"]
    if len(args) < 2:
        raise SystemExit(__doc__)
    data = consolidate([parse(p) for p in args[1:]], production=production)
    scores = score(data)
    write_outputs(data, scores, args[0])


if __name__ == "__main__":
    main(sys.argv)
