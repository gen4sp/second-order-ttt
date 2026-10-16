"""NumPy coordinator for the step-0 assay, step-60 sizing and endpoint G2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from .probe import assay, g2_decision, joint_power, boot_mean, bootstrap_weights, one_sided


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(arm_dir, step: int):
    root = Path(arm_dir)
    paths = [root / "eval_meta.npz", root / f"eval_step{step:04d}.npz"]
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            arrays.append({k: z[k] for k in z.files})
    return *arrays, {str(p.resolve()): _sha1(p) for p in paths}


def _values(z: dict, keys, n: int) -> dict:
    out = {}
    for k in keys:
        if k not in z:
            raise ValueError(f"missing evaluation array: {k}")
        v = np.asarray(z[k], dtype=np.float64)
        if v.shape != (n,) or not n or not np.isfinite(v).all():
            raise ValueError(f"unaligned or non-finite evaluation array: {k}")
        out[k] = v
    return out


def _ids(meta: dict, z: dict, split: str, n: int):
    out = []
    for suffix in ("sha1", "domain"):
        key = f"{split}_{suffix}"
        if key not in meta or meta[key].ndim != 1 or len(meta[key]) < n:
            raise ValueError(f"missing or short document identities: {key}")
        ids = meta[key][:n]
        if key in z and not np.array_equal(z[key], ids):
            raise ValueError(f"step document order differs from metadata: {key}")
        out.append(ids)
    return out


def assay_decision(ce_arm_dir, *, n_boot: int = 10000, seed: int = 0,
                   min_clusters: int = 20) -> dict:
    """Released CE assay from the full step-0 eligible set."""
    meta, z, inputs = _load(ce_arm_dir, 0)
    n = len(meta["eval_sha1"])
    v = _values(z, ("doc_adapt", "doc_noadapt", "doc_fP_focal", "doc_fP_nonfocal"), n)
    _, clusters = _ids(meta, z, "eval", n)
    r = assay(v["doc_fP_focal"] - v["doc_fP_nonfocal"],
              v["doc_noadapt"] - v["doc_adapt"], clusters,
              n_boot=n_boot, seed=seed, min_clusters=min_clusters)
    return {**r, "inputs": inputs}


def _paired(arm_dirs: dict, step: int, rule: str):
    names = {"so": "e2e2", "fo": "e2e1-ce"}
    if rule is not None:
        names[rule] = f"e2e1-{rule}"
    if not set(names.values()).issubset(arm_dirs):
        raise ValueError(f"required arms: {sorted(set(names.values()))}")
    L, G, M, Lho, inputs, metas = {}, {}, {}, {}, {}, {}
    ref, ref_ho, ref_pool = None, None, None
    for a, name in names.items():
        meta, z, hashes = _load(arm_dirs[name], step)
        if "doc_adapt" not in z or z["doc_adapt"].ndim != 1:
            raise ValueError("missing document losses")
        n = len(z["doc_adapt"])
        v = _values(z, ("doc_adapt", "doc_noadapt", "doc_fP_focal", "doc_fP_nonfocal"), n)
        nh = len(z.get("doc_heldout", []))
        h = _values(z, ("doc_heldout",), nh)
        ids, ids_ho = _ids(meta, z, "eval", n), _ids(meta, z, "ho", nh)
        pool = _ids(meta, meta, "eval", len(meta["eval_sha1"]))
        if ref is not None and any(not np.array_equal(x, y)
                                   for x, y in zip(ref + ref_ho + ref_pool, ids + ids_ho + pool)):
            raise ValueError("arm documents or domain clusters are not aligned")
        ref, ref_ho, ref_pool = ids, ids_ho, pool
        ho_pool = meta["ho_domain"]
        L[a] = v["doc_adapt"]
        G[a] = v["doc_noadapt"] - v["doc_adapt"]
        M[a] = v["doc_fP_focal"] - v["doc_fP_nonfocal"]
        Lho[a] = h["doc_heldout"]
        inputs.update(hashes)
        metas[name] = _sha1(Path(arm_dirs[name]) / "eval_meta.npz")
    return L, G, M, ref[1], Lho, ref_ho[1], ref_pool[1], ho_pool, inputs, metas


def _destination(decision_dir, filename: str) -> Path:
    path = Path(decision_dir) / filename
    if path.exists():
        raise FileExistsError(f"decision already exists: {path}")
    return path


def _write(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def futility(so_dir, fo_dir, decision_dir, *, expected_docs=256,
             n_boot=10000, seed=0) -> dict:
    """A1: stop rule arms only when the one-sided upper bound at step 100 is < .05."""
    path = _destination(decision_dir, "futility.json")
    inputs = {}
    hash_errors = []
    for root in (so_dir, fo_dir):
        for name in ("eval_meta.npz", "eval_step0100.npz"):
            p = Path(root) / name
            if p.is_file():
                try:
                    inputs[str(p.resolve())] = _sha1(p)
                except OSError as exc:
                    hash_errors.append(str(exc))
    r = dict(step=100, n_boot=n_boot, seed=seed, threshold=0.05,
             launch_rule_arms=True, inputs=inputs)
    try:
        if hash_errors:
            raise OSError("; ".join(hash_errors))
        pairs = []
        for root in (so_dir, fo_dir):
            meta, z, _ = _load(root, 100)
            values = _values(z, ("doc_adapt",), expected_docs)["doc_adapt"]
            identities = _ids(meta, z, "eval", expected_docs)
            # Training always writes explicit step identities. Their absence is invalid.
            if any(f"eval_{suffix}" not in z for suffix in ("sha1", "domain")):
                raise ValueError("missing step document identities")
            if "eval_rows" in meta:
                if "eval_rows" not in z or not np.array_equal(z["eval_rows"], meta["eval_rows"][:expected_docs]):
                    raise ValueError("step evaluation rows differ from metadata")
            pairs.append((values, identities))
        if any(not np.array_equal(a, b) for a, b in zip(pairs[0][1], pairs[1][1])):
            raise ValueError("arm documents or domain clusters are not aligned")
        if len(np.unique(pairs[0][1][0])) != expected_docs:
            raise ValueError("duplicate document identities")
        contrast = pairs[1][0] - pairs[0][0]
        bound = one_sided(*boot_mean(contrast, bootstrap_weights(pairs[0][1][1], n_boot, seed)))
        r.update(D=bound, U_100=bound["hi"], n_docs=expected_docs,
                 domains=int(len(np.unique(pairs[0][1][1]))),
                 launch_rule_arms=not (bound["hi"] < 0.05),
                 reason="upper_bound_below_0.05" if bound["hi"] < 0.05 else "not_futile")
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, TypeError, IndexError) as exc:
        r.update(U_100=None, reason=f"missing_or_invalid: {type(exc).__name__}: {exc}")
    _write(path, r)
    return r


def _rule_present(arm_dirs, rule, decision_dir):
    if f"e2e1-{rule}" in arm_dirs:
        return True
    path = Path(decision_dir) / "futility.json"
    if not path.is_file() or json.loads(path.read_text()).get("launch_rule_arms") is not False:
        raise ValueError("missing rule arm requires a recorded futility stop")
    return False


def sizing(arm_dirs: dict, step: int = 60, Ns=(1024, 1536, 2048, 2560, 3072),
           rule: str = "lin", *, decision_dir=None, n_sim: int = 2000,
           alpha: float = 0.05, seed: int = 0, ho_n: int = 512, smoke: bool = False) -> dict:
    """Freeze endpoint N from paired trajectory contrasts and the released CE assay (the SO
    arm's step-0 evaluation, which gated the launch). The power simulation uses the frozen
    endpoint design: the first N eligible pool documents and the first ``ho_n`` held-out
    documents, with their domain clusters (``probe.joint_power``).

    Confirmation (stream B) evaluates the same documents, so its pass is not
    independent of this endpoint's; ``two_stage_bounds`` reports the Fréchet bounds
    [2p - 1, p] on passing both stages, assuming equal marginal power p."""
    if decision_dir is None:
        decision_dir = Path(arm_dirs["e2e2"]).parent / "decisions"
    path = _destination(decision_dir, "sizing.json")
    a = assay_decision(arm_dirs["e2e2"], seed=seed, min_clusters=2 if smoke else 20)
    if not a["pass"] and not smoke:
        raise ValueError("released CE assay failed")
    has_rule = _rule_present(arm_dirs, rule, decision_dir)
    L, _, M, clusters, Lho, ho_clusters, pool, ho_pool, inputs, metas = _paired(
        arm_dirs, step, rule if has_rule else None)
    if has_rule:
        for other in ("lin", "aice"):
            if other != rule and f"e2e1-{other}" in arm_dirs:
                *_, hashes, extra_metas = _paired(arm_dirs, step, other)
                inputs.update(hashes)
                metas.update(extra_metas)
    inputs.update(a["inputs"])
    if not has_rule:
        stop_path = Path(decision_dir) / "futility.json"
        inputs[str(stop_path.resolve())] = _sha1(stop_path)
    available = len(pool)
    if len(ho_pool) < ho_n:
        raise ValueError(f"held-out split has {len(ho_pool)} documents, endpoint needs {ho_n}")
    D = L["fo"] - L["so"]
    dM = M["so"] - M["fo"]
    contrasts = {"D": D, "dM": dM}
    if has_rule:
        contrasts.update(CmD=L["fo"] - L[rule] - 0.5 * D,
                         RM=M[rule] - M["fo"] - 0.5 * dM, M_rule=M[rule])
    M0 = a["M0"]["est"]
    alternatives = {"D": 0.10, "CmD": 0.025, "dM": 0.5 * M0,
                    "RM": 0.25 * M0, "M_rule": M0, "ho": 0.0}
    Ns = list(Ns)
    if not Ns or any(N < 1 or int(N) != N for N in Ns):
        raise ValueError("Ns must contain positive document counts")
    candidates = sorted({min(int(N), available) for N in Ns})
    if not has_rule:
        alternatives = {k: alternatives[k] for k in ("D", "dM")}
    powers = joint_power(contrasts, clusters, Lho[rule if has_rule else "fo"] - Lho["so"], ho_clusters,
                         alternatives, candidates, pool_clusters=pool,
                         ho_pool_clusters=ho_pool[:ho_n], n_sim=n_sim, alpha=alpha, seed=seed,
                         min_clusters=2 if smoke else 20)
    N = next((n for n in candidates if powers[n] >= 0.8), max(candidates))
    r = {"step": step, "rule": rule, "N_eval": N, "eligible_docs": available,
         "launch_rule_arms": has_rule, "conjuncts": [1, 2, 3, 4, 5, 6] if has_rule else [1, 4],
         "requested_Ns": Ns, "powers": powers, "alternatives": alternatives,
         "domains": {int(n): int(len(np.unique(pool[:n]))) for n in candidates},
         "ho_domains": int(len(np.unique(ho_pool[:ho_n]))),
         "two_stage_bounds": {n: [max(0.0, 2 * p - 1), p] for n, p in powers.items()},
         "n_sim": n_sim, "alpha": alpha, "seed": seed, "ho_n": ho_n, "assay": a,
         "smoke": smoke,
         "inputs": inputs, "meta_sha1": metas}
    _write(path, r)
    return r


def endpoint(arm_dirs: dict, decision_dir, step: int = 400, rule: str = "lin", *,
             n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
             min_clusters: int = 20) -> dict:
    """G2 on the first frozen N eligible documents and the fixed held-out split."""
    path = _destination(decision_dir, "g2.json")
    size_path = Path(decision_dir) / "sizing.json"
    if not size_path.exists():
        raise ValueError("endpoint requires sizing.json")
    size = json.loads(size_path.read_text())
    if size.get("smoke") and min_clusters >= 20:
        raise ValueError("smoke sizing cannot be used for a scientific endpoint")
    if size["rule"] != rule or step <= size["step"]:
        raise ValueError("endpoint rule must match sizing and its step must be later")
    if alpha != size["alpha"]:
        raise ValueError("endpoint alpha must match the frozen sizing alpha")
    has_rule = size.get("launch_rule_arms", True)
    if has_rule != _rule_present(arm_dirs, rule, decision_dir):
        raise ValueError("launched arms differ from sizing")
    L, G, M, clusters, Lho, ho_clusters, _, _, inputs, metas = _paired(
        arm_dirs, step, rule if has_rule else None)
    if has_rule:
        for other in ("lin", "aice"):
            if other != rule and f"e2e1-{other}" in arm_dirs:
                *_, hashes, extra_metas = _paired(arm_dirs, step, other)
                inputs.update(hashes)
                metas.update(extra_metas)
    if metas != size["meta_sha1"]:
        raise ValueError("eligible pool changed after sizing")
    if len(ho_clusters) != size["ho_n"]:
        raise ValueError(f"endpoint has {len(ho_clusters)} held-out documents, "
                         f"sizing froze {size['ho_n']}")
    N = size["N_eval"]
    if not isinstance(N, int) or N < 1 or N > len(clusters):
        raise ValueError("endpoint does not contain the frozen N documents")
    L, G, M = [{k: v[:N] for k, v in a.items()} for a in (L, G, M)]
    if has_rule:
        r = g2_decision(L, G, M, clusters[:N], Lho, ho_clusters, rule=rule,
                        n_boot=n_boot, alpha=alpha, seed=seed, min_clusters=min_clusters)
    else:
        L0, _, M0, c0, _, _, _, _, i0, m0 = _paired(arm_dirs, 0, None)
        if m0 != metas or not np.array_equal(c0[:N], clusters[:N]):
            raise ValueError("step-0 pool differs from endpoint")
        w = bootstrap_weights(clusters[:N], n_boot, seed)
        D, dM = L["fo"] - L["so"], M["so"] - M["fo"]
        describe = lambda v: one_sided(*boot_mean(v, w), alpha)
        r = dict(G2=None, evaluable=False,
                 reason="not evaluable (rule arms stopped for futility at step 100)",
                 D=describe(D), delta_M=describe(dM),
                 paired_change_D=describe(D - (L0["fo"] - L0["so"])[:N]),
                 paired_change_delta_M=describe(dM - (M0["so"] - M0["fo"])[:N]),
                 n_docs=N, n_ho=len(ho_clusters))
        inputs.update(i0)
    inputs[str(size_path.resolve())] = _sha1(size_path)
    r.update(step=step, rule=rule, N_eval=N, inputs=inputs, smoke=size.get("smoke", False))
    _write(path, r)
    return r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-name", default=os.environ.get("PROCESS_NAME"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("assay")
    p.add_argument("ce_arm_dir")
    p.add_argument("--process-name", default=argparse.SUPPRESS)
    p = sub.add_parser("futility")
    p.add_argument("--so", required=True)
    p.add_argument("--fo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--process-name", default=argparse.SUPPRESS)
    for command, step in (("size", 60), ("endpoint", 400)):
        p = sub.add_parser(command, aliases=["sizing"] if command == "size" else [])
        p.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
        p.add_argument("--decision-dir", required=True)
        p.add_argument("--step", type=int, default=step)
        p.add_argument("--rule", default="lin")
        p.add_argument("--alpha", type=float, default=.05)
        p.add_argument("--process-name", default=argparse.SUPPRESS)
        if command == "size":
            p.add_argument("--Ns", type=int, nargs="+", default=[1024, 1536, 2048, 2560, 3072])
            p.add_argument("--ho-n", type=int, default=512)
    args = parser.parse_args(argv)
    if args.process_name:
        try:
            import setproctitle
            setproctitle.setproctitle(args.process_name)
        except ImportError:
            pass
    if args.command == "assay":
        result = assay_decision(args.ce_arm_dir)
    elif args.command == "futility":
        result = futility(args.so, args.fo, args.out)
    else:
        arms = {}
        for item in args.arm:
            name, sep, directory = item.partition("=")
            if not sep or not directory or name in arms:
                parser.error("each --arm must be a distinct NAME=DIR")
            arms[name] = directory
        kw = dict(step=args.step, rule=args.rule, decision_dir=args.decision_dir, alpha=args.alpha)
        result = sizing(arms, Ns=args.Ns, ho_n=args.ho_n, **kw) if args.command in ("size", "sizing") \
            else endpoint(arms, **kw)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
