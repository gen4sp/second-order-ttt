"""Eval-only write-code geometry and W2-only readouts at slow weights W0."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import exp_kag as kag
from . import exp_ttt as ttt

QUANTILES = (5, 25, 50, 75, 95)
STRATA = ("bound", "collapsed_e2e2", "e2e1")


def _emit(kind: str, record: dict) -> None:
    print(kind + " " + json.dumps(record, sort_keys=True), flush=True)


def _geometry(a: torch.Tensor) -> dict[str, np.ndarray]:
    """One row per sequence; centred singular energies define PR and r99."""
    centred = a.double() - a.double().mean(1, keepdim=True)
    energy = torch.linalg.svdvals(centred).square()
    total = energy.sum(-1)
    pr = total.square() / energy.square().sum(-1).clamp_min(1e-30)
    pr = torch.where(total > 1e-20, pr, torch.zeros_like(pr))
    r99 = ((energy.cumsum(-1) < .99 * total[:, None]).sum(-1) + 1)
    r99 = torch.where(total > 1e-20, r99, torch.zeros_like(r99))
    # A centred set of n codes has rank <= n - 1, whatever the head width.
    assert float(pr.max()) <= a.shape[1] - 1 + 1e-6 and int(r99.max()) <= a.shape[1] - 1
    common = a.mean(1).square().sum(-1) / a.square().sum(-1).mean(-1).clamp_min(1e-20)
    unit = torch.nn.functional.normalize(a, dim=-1)
    cos = torch.bmm(unit, unit.transpose(1, 2))
    n = a.shape[1]
    offdiag = (cos.sum((1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(-1)) / (n * (n - 1))
    return {key: value.cpu().numpy() for key, value in
            {"pr": pr, "r99": r99, "common": common, "offdiag_cos": offdiag}.items()}


def _margin(dot: torch.Tensor) -> torch.Tensor:
    """Median absolute score is the per-sequence normaliser for all queries."""
    n = dot.shape[-1]
    matched = dot.diagonal(dim1=1, dim2=2)
    other = dot.masked_fill(torch.eye(n, device=dot.device, dtype=torch.bool)[None], -torch.inf)
    return ((matched - other.max(-1).values) /
            dot.abs().flatten(1).median(-1).values[:, None].clamp_min(1e-12)).median(-1).values


def _overshoot(a: torch.Tensor, eta: float, chunk: int) -> np.ndarray:
    # The uncentred code Gram includes the common mode (and the W2 update does too).
    largest = torch.linalg.svdvals(a.double())[:, 0].square()
    return (eta * largest / chunk).cpu().numpy()


def _verdicts(models: list[dict], refs: dict) -> dict:
    groups = {name: [m for m in models if m["stratum"] == name] for name in STRATA}
    bound = groups["bound"]
    if not bound:
        return {"verdict": "insufficient bound reference"}
    baseline = float(np.median([m["pr_kv"] for m in bound]))
    reverse = sum(m["acc_kv_only"] >= .90 for m in bound)
    result = {"bound": {"n": len(bound), "pr_kv_median": baseline,
                        "kv_reverse_count": reverse, "kv_reverse_pass": reverse >= 8,
                        "kv_reverse_verdict": "Pass" if reverse >= 8 else "Fail"},
              "thresholds": {"collapsed_pr": "<= 0.5 x bound median", "scrambled_pr": ">= 0.8 x bound median",
                             "addr_chance_tol": .05, "share": .75, "wv_joint_min": 6,
                             "kv_reverse_min": 8, "occ_tol": .05}}
    for group in ("collapsed_e2e2", "e2e1"):
        rows = groups[group]
        low = sum(m["pr_kv"] <= .5 * baseline for m in rows)
        high = sum(m["pr_kv"] >= .8 * baseline and
                   abs(m["addr_a"] - refs["chance"]["acc"]) <= .05 for m in rows)
        result[group] = {"n": len(rows), "collapsed_count": low, "scrambled_count": high,
                         "kv_verdict": ("Collapsed" if rows and low / len(rows) >= .75 else
                                        "Scrambled" if rows and high / len(rows) >= .75 else "Mixed")}
    rows = groups["collapsed_e2e2"]
    limits = {"acc_Wv_only": ("within_occ", lambda m: abs(m["acc_Wv_only"] - refs["occ"]["acc"]) <= .05),
              "occ_hit_Wv_only": ("occ_hit", lambda m: m["occ_hit_Wv_only"] >= .8),
              "acc_kv_only": ("kv_below_occ_minus", lambda m: m["acc_kv_only"] <= refs["occ_minus"]["acc"] + .05)}
    counts = {name: sum(pred(m) for m in rows) for name, (_, pred) in limits.items()}
    passes = sum(all(pred(m) for _, pred in limits.values()) for m in rows)
    failing = [label for name, (label, _) in limits.items() if counts[name] < 6]
    if passes < 6 and not failing:
        failing.append("joint_cooccurrence")
    result["collapsed_e2e2"].update(
        wv_verdict="Wv-supplies" if passes >= 6 else "Not Wv-supplies",
        wv_joint_count=passes, wv_condition_counts=counts,
        wv_failing_conditions=failing,
        wv_per_model={m["label"]: [label for _, (label, pred) in limits.items() if not pred(m)]
                      for m in rows})
    return result


def _reference(path: str) -> dict[str, dict[str, float]]:
    """Versioned R2-4b echo-only excerpt (address_dot, linear.1.0)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["label"]: {"addr_a": float(r["addr_a"]),
                          "acc_kv_only": float(r["acc_kv_only"])} for r in rows}


def _batch(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor, meta: dict,
           eta: torch.Tensor, chunk: int, g_clip: float) -> dict[str, np.ndarray]:
    with torch.no_grad():
        code = kag._code(model, h)
        codes = {"kv": kag._gather(code, meta["write_pos"]),
                 "Wv": kag._gather(code, np.asarray(meta["write_pos"]) - 2),
                 "q": kag._gather(code, meta["query_pos"])}
        o = ttt.fast_block(h[:, :chunk], model.head.ln, model.slow())
    delta = kag._delta(model, o, y[:, :chunk])  # derivative of SUM of token CE
    weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(y))]
    with torch.enable_grad():
        logits = model.read(ttt.fast_block(h[:, :chunk], model.head.ln, weights))
        grads = torch.autograd.grad(ttt.token_ce(logits, y[:, :chunk]).mean(1).sum(), weights)
    _, norm, clipped = ttt._clip_fast(list(grads), g_clip)
    scale = torch.where(clipped, g_clip / norm, torch.ones_like(norm))
    with torch.no_grad():
        x = kag._linear_terms(model, h, meta, delta, eta, scale, chunk)
        dot = x["dot"]
        n = dot.shape[-1]
        nonmatch = dot.masked_select(~torch.eye(n, device=h.device, dtype=torch.bool)[None]).reshape(len(y), -1)
        record = {}
        for name, a in codes.items():
            for metric, values in _geometry(a).items():
                record[f"{metric}_{name}"] = values
        record["addr_a"] = (dot.argmax(-1) == torch.arange(n, device=h.device)[None]).cpu().numpy()
        record["addr_margin"] = _margin(dot).cpu().numpy()
        for name, values in (("matched", dot.diagonal(dim1=1, dim2=2)),
                             ("mismatched", nonmatch)):
            record[f"{name}_dot_q"] = torch.quantile(values.float(),
                                                       values.new_tensor(QUANTILES).float() / 100,
                                                       dim=-1).transpose(0, 1).cpu().numpy()
        q, wv = codes["q"], codes["Wv"]
        record["qWv_cos"] = torch.bmm(torch.nn.functional.normalize(q, dim=-1),
                                      torch.nn.functional.normalize(wv, dim=-1).transpose(1, 2)).mean((1, 2)).cpu().numpy()
        for name, pos in (("kv", np.asarray(meta["write_pos"])),
                          ("Wv", np.asarray(meta["write_pos"]) - 2)):
            d = kag._gather(delta, pos)
            # eta_W2 * |e| * |a|, as in R2-4b's write_norm (no 1/chunk).
            record[f"{name}_write"] = (eta.abs() * d.norm(dim=-1) *
                                        codes[name].norm(dim=-1)).median(-1).values.cpu().numpy()
            record[f"overshoot_{name}"] = _overshoot(codes[name], eta.item(), chunk)
            memory = -torch.bmm(torch.bmm(q, codes[name].transpose(1, 2)), d)
            pred = model.read(x["o0"] + x["factor"] * memory).argmax(-1)
            record[f"hit_{name}"] = (pred == x["target"]).cpu().numpy()
            if name == "Wv":
                values = torch.as_tensor(meta["query_tok"], device=h.device)
                record["occ_hit_Wv"] = (pred[..., None] == values[:, None, :]).any(-1).cpu().numpy()
        return record


def run(cfg: dict, seed: int) -> dict:
    ttt._mark_process(cfg["process_name"])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    dev = ttt._device(cfg["device"])
    assert dev.type == "cpu", "R4-0 is CPU-only"
    st = ttt._settings(cfg, "echo")
    task = ttt._task(st, "echo")
    assert task.T == 512 and task.chunk == 64 and task.n_keys == 16
    tokens, meta = ttt.fixed_set(task, st, "eval")
    data = ttt.eval_batches(task, tokens, meta, dev, st.eval_batch)
    refs = ttt.echo_references(meta, task.n_keys)
    refs = {k: {name: value for name, value in v.items() if name != "acc_per_seq"}
            for k, v in refs.items()}
    baseline = _reference(cfg["kag_reference"])
    entries = cfg["checkpoints"]
    if not cfg.get("smoke", False):
        assert st.n_eval == 256 and len(entries) == 31
        assert {e["label"] for e in entries if e["label"] in baseline} == set(baseline)
        assert {name: sum(e["stratum"] == name for e in entries) for name in STRATA} == {
            "bound": 10, "collapsed_e2e2": 8, "e2e1": 12}
    models = []
    for entry in entries:
        model = ttt.build_model(task, st, dev, seed, entry["arm"])
        if "path" in entry:
            model.load_state_dict(torch.load(Path(cfg["ckpt_root"]) / entry["path"],
                                             map_location=dev, weights_only=True), strict=True)
        model.eval()
        eta = model.etas(float(entry["eta_base"]))[2].detach()
        parts = []
        for idx, y, bm in data:
            with torch.no_grad():
                h = model.backbone(idx).detach()
            parts.append(_batch(model, h, y, bm, eta, task.chunk, st.g_clip))
        def med(key: str) -> float:
            return float(np.median(np.concatenate([p[key] for p in parts])))
        record = {k: entry[k] for k in ("label", "arm", "stratum", "eta_base")}
        record["eta_W2"] = float(eta)
        record.update({f"{key}_{name}": med(f"{key}_{name}")
                       for name in ("kv", "Wv", "q")
                       for key in ("pr", "r99", "common", "offdiag_cos")})
        record["addr_a"] = float(np.concatenate([p["addr_a"].reshape(-1) for p in parts]).mean())
        record["addr_margin"] = med("addr_margin")
        for name in ("matched", "mismatched"):
            all_q = np.concatenate([p[f"{name}_dot_q"] for p in parts])
            record[f"{name}_dot_quantiles"] = {str(q): float(np.median(all_q[:, i]))
                                                for i, q in enumerate(QUANTILES)}
        for key in ("qWv_cos", "kv_write", "Wv_write", "overshoot_kv", "overshoot_Wv"):
            record[key] = med(key)
        record["write_ratio_Wv_kv"] = record["Wv_write"] / max(record["kv_write"], 1e-20)
        for name in ("kv", "Wv"):
            record[f"acc_{name}_only"] = float(np.concatenate(
                [p[f"hit_{name}"].reshape(-1) for p in parts]).mean())
        record["occ_hit_Wv_only"] = float(np.concatenate(
            [p["occ_hit_Wv"].reshape(-1) for p in parts]).mean())
        # Query accuracy defines the stratum; check it against the run that saved the model.
        record["query_acc"] = float(ttt.evaluate(model, task, entry["arm"], data, meta,
                                                 float(entry["eta_base"]), st)["eval"]["query_acc"])
        models.append(record)
        _emit("RNK", record)
        if "ref" in entry and not cfg.get("smoke", False):
            diff = abs(record["query_acc"] - float(entry["ref"]))
            _emit("RNK_ACC", {"label": entry["label"], "query_acc": record["query_acc"],
                              "ref": float(entry["ref"]), "diff": diff, "pass": diff <= .02})
            if diff > .02:
                raise AssertionError(f"query accuracy of {entry['label']} does not match its run")
        if entry["label"] in baseline and st.n_eval == 256:
            expected = baseline[entry["label"]]
            diffs = {key: abs(record[key] - expected[key]) for key in expected}
            sanity = {"label": entry["label"], "reference": expected, "differences": diffs,
                      "pass": all(d <= .01 for d in diffs.values())}
            _emit("RNK_SANITY", sanity)
            if not sanity["pass"]:
                raise AssertionError(f"R2-4b reproduction failed: {sanity}")
        elif cfg.get("smoke", False):
            _emit("RNK_SANITY", {"label": entry["label"], "pass": None,
                                 "reason": f"{st.n_eval}-sequence smoke; 256-sequence reference"})
    scalars = [k for k, v in models[0].items() if isinstance(v, float)]
    for name in sorted({m["stratum"] for m in models}):
        rows = [m for m in models if m["stratum"] == name]
        _emit("RNK_STRATUM", {"stratum": name, "n": len(rows), **{
            k: {"median": float(np.median([m[k] for m in rows])),
                "min": float(min(m[k] for m in rows)), "max": float(max(m[k] for m in rows))}
            for k in scalars}})
    return {"refs": refs, "models": models, "verdicts": (
        {"verdict": "not evaluated (smoke subset)"} if cfg.get("smoke", False)
        else _verdicts(models, refs)), "smoke": bool(cfg.get("smoke", False))}
