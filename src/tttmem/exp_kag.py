"""Eval-only key-addressability and copy-gate probe at slow weights W0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as ttt
from . import exp_wab as wab

MODES = ("echo", "echo_ctrl", "echo_ctrl_nor", "echo_off")
K_MODES = MODES[:3]
KV = frozenset({"kv"})
GAINS = (0.01, 0.1, 1., 10., 100.)


def _emit(kind: str, record: dict) -> None:
    print(kind + " " + json.dumps(record, sort_keys=True), flush=True)


def _gather(x: torch.Tensor, positions: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Gather B x K positions, preserving the trailing feature axes."""
    pos = torch.as_tensor(np.asarray(positions) if not isinstance(positions, torch.Tensor)
                          else positions, device=x.device)
    index = pos.reshape(*pos.shape, *([1] * (x.ndim - 2))).expand(-1, -1, *x.shape[2:])
    return x.gather(1, index)


def _code(model: ttt.TTTModel, h: torch.Tensor,
          other: ttt.TTTModel | None = None,
          idx: torch.Tensor | None = None) -> torch.Tensor:
    w1, b1 = (other or model).slow()[:2]
    x = model.fast_input(h, idx) if idx is not None else model.head.ln(h)
    return F.gelu(F.linear(x, w1, b1))


def _gate_positions(a: torch.Tensor, write_pos: np.ndarray | torch.Tensor,
                    chunk: int) -> dict[str, np.ndarray]:
    """W0 codes, including off-chunk writes; no chunk-0 partition needed."""
    wp = torch.as_tensor(write_pos, device=a.device)
    write = _gather(a, wp)
    return {"write_norm": write.norm(dim=-1).cpu().numpy(),
            "wv_norm": _gather(a, wp - 2).norm(dim=-1).cpu().numpy(),
            "centre": (write - a[:, :chunk].mean(dim=1, keepdim=True)).norm(dim=-1).cpu().numpy(),
            "event": wp.argsort(dim=1).argsort(dim=1).cpu().numpy()}


def _delta(model: ttt.TTTModel, o: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token derivative in residual space, including ln_f and unembedding."""
    with torch.enable_grad():
        leaf = o.detach().requires_grad_(True)
        return torch.autograd.grad(ttt.token_ce(model.read(leaf), y).sum(), leaf)[0].detach()


def _address(write: torch.Tensor, query: torch.Tensor, delta: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dot = torch.bmm(query, write.transpose(1, 2))
    cos = dot / (query.norm(dim=-1, keepdim=True).clamp_min(1e-12) *
                 write.norm(dim=-1).unsqueeze(1).clamp_min(1e-12))
    strength = dot * delta.norm(dim=-1).unsqueeze(1)
    match = dot.diagonal(dim1=1, dim2=2)
    share = (match * delta.norm(dim=-1) /
             strength.abs().sum(-1).clamp_min(1e-12))
    margin = (match - dot.masked_fill(torch.eye(dot.shape[-1], device=dot.device,
                                                 dtype=torch.bool)[None], -torch.inf).max(-1).values)
    margin = margin / query.norm(dim=-1).clamp_min(1e-12)
    return dot, cos, strength, share, margin


def _linear_terms(model: ttt.TTTModel, h: torch.Tensor, meta: dict,
                  delta: torch.Tensor, eta: torch.Tensor, scale: torch.Tensor,
                  chunk: int = 64, idx: torch.Tensor | None = None
                  ) -> dict[str, torch.Tensor]:
    """Query residuals for the exact kv W2 intervention and oracle controls."""
    code = _code(model, h, idx=idx)
    write, query = _gather(code, meta["write_pos"]), _gather(code, meta["query_pos"])
    dq = _gather(delta, meta["write_pos"])
    pos = torch.as_tensor(meta["query_pos"], device=h.device)
    target = torch.as_tensor(meta["query_tok"], device=h.device)
    o0 = _gather(model.fast_block(h, idx, model.slow()) if idx is not None else
                 ttt.fast_block(h, model.head.ln, model.slow()), pos)
    dot, cos, strength, share, margin = _address(write, query, dq)
    memory = -torch.bmm(dot, dq)  # queries and writes both ordered by key
    factor = (eta * scale[:, None, None] / h.new_tensor(chunk))
    matched = dot.diagonal(dim1=1, dim2=2)
    oracle = write.norm(dim=-1) * query.norm(dim=-1)
    return dict(o0=o0, memory=memory, factor=factor, write=write, query=query,
                delta=dq, target=target, dot=dot, cos=cos, strength=strength,
                share=share, margin=margin, matched=matched, oracle=oracle)


def _hits(model: ttt.TTTModel, residual: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (model.read(residual).argmax(-1) == target).cpu().numpy()


def _oracle_content(o0: torch.Tensor, factor: torch.Tensor, similarity: torch.Tensor,
                    delta: torch.Tensor, gain: float = 1.) -> torch.Tensor:
    """Read one write per key; similarity is its oracle norm product or actual dot."""
    return o0 - gain * factor * similarity[..., None] * delta


def _query_record(logits: torch.Tensor, meta: dict) -> dict[str, np.ndarray]:
    q = _gather(logits, meta["query_pos"])
    target = torch.as_tensor(meta["query_tok"], device=q.device)
    return {"hit": (q.argmax(-1) == target).cpu().numpy(),
            "ce": ttt.token_ce(q, target).cpu().numpy()}


def _scaled_grads(grads: list[list[torch.Tensor]], scale: torch.Tensor,
                  kv_factor: torch.Tensor) -> list[list[torch.Tensor]]:
    """Preapply the original full-chunk clip, then rescale only the kv update."""
    return [[g * (scale * (kv_factor if c == 2 else 1)).reshape(
        -1, *([1] * (g.ndim - 1))) for g in row] for c, row in enumerate(grads)]


def _norm_match(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor,
                logits0: torch.Tensor, grads: list[list[torch.Tensor]],
                etas: list[torch.Tensor], scale: torch.Tensor,
                target_norm: float, chunk: int, g_clip: float) -> torch.Tensor:
    own = wab._joint_norm(wab._weighted(grads[2], etas, scale))
    factor = target_norm / own.clamp_min(1e-20)
    adjusted = _scaled_grads(grads, scale, factor)
    # eval_condition clips its newly supplied chunk-0 gradients again; doing
    # that would undo the per-sequence norm match. Keep its fixed/all streaming
    # recurrence and the *original* g_clip on every subsequent chunk.
    batch = len(y)
    updates = wab._weighted(wab._sum_grads(adjusted, wab.ALL), etas,
                            torch.ones_like(scale))
    weights = [(p.detach().unsqueeze(0).expand(batch, *p.shape) - update)
               .detach().requires_grad_(True) for p, update in zip(model.slow(), updates)]
    out = [logits0]
    with torch.enable_grad():
        for c in range(1, y.shape[1] // chunk):
            sl = slice(c * chunk, (c + 1) * chunk)
            logits = model.read(ttt.fast_block(h[:, sl], model.head.ln, weights))
            out.append(logits.detach())
            grads_c = torch.autograd.grad(ttt.token_ce(logits, y[:, sl]).mean(1).sum(), weights)
            clipped, _, _ = ttt._clip_fast(list(grads_c), g_clip)
            weights = [(w - eta * g).detach().requires_grad_(True)
                       for w, eta, g in zip(weights, etas, clipped)]
    return torch.cat(out, dim=1)


def _med(parts: list[np.ndarray]) -> float:
    return float(np.median(np.concatenate([x.reshape(-1) for x in parts])))


def _k_metrics(parts: list[dict]) -> dict:
    def acc(key: str) -> float:
        return float(np.concatenate([p[key].reshape(-1) for p in parts]).mean())
    return {"linear": {str(g): acc(f"linear_{g}") for g in GAINS},
            **{key: acc(key) for key in ("memory_inf", "content_inf", "address_dot",
                                          "address_cos", "effective_address", "address_ln",
                                          "address_random_w1", "oracle_content_1",
                                          "oracle_content_16", "matched", "direct")},
            "matched_share": float(np.concatenate([p["share"].reshape(-1) for p in parts]).mean()),
            "address_margin": _med([p["margin"] for p in parts]),
            "write_code_norm": _med([p["write_norm"] for p in parts]),
            "write_delta_norm": _med([p["delta_norm"] for p in parts]),
            "query_code_norm": _med([p["query_norm"] for p in parts]),
            "max_logit_diff": max(float(p["max_diff"]) for p in parts)}


def _k_batch(model: ttt.TTTModel, random_model: ttt.TTTModel, h: torch.Tensor,
             meta: dict, delta: torch.Tensor, eta: torch.Tensor, scale: torch.Tensor,
             direct: torch.Tensor) -> dict:
    with torch.no_grad():
        x = _linear_terms(model, h, meta, delta, eta, scale)
        o0, memory, factor, target = (x[k] for k in ("o0", "memory", "factor", "target"))
        linear = o0 + factor * memory
        qlogits = model.read(linear)
        dlogits = _gather(direct, meta["query_pos"])
        n = x["target"].shape[-1]
        expected = torch.arange(n, device=h.device)[None, :]
        ln = model.head.ln(h)
        ldot = _address(_gather(ln, meta["write_pos"]),
                        _gather(ln, meta["query_pos"]), x["delta"])[0]
        rcode = _code(model, h, random_model)
        rdot = _address(_gather(rcode, meta["write_pos"]),
                        _gather(rcode, meta["query_pos"]), x["delta"])[0]
        record = {f"linear_{g}": _hits(model, o0 + g * factor * memory, target)
                  for g in GAINS}
        record.update(memory_inf=_hits(model, memory, target),
                      content_inf=_hits(model, -x["delta"], target),
                      address_dot=(x["dot"].argmax(-1) == expected).cpu().numpy(),
                      address_cos=(x["cos"].argmax(-1) == expected).cpu().numpy(),
                      effective_address=(x["strength"].argmax(-1) == expected).cpu().numpy(),
                      address_ln=(ldot.argmax(-1) == expected).cpu().numpy(),
                      address_random_w1=(rdot.argmax(-1) == expected).cpu().numpy(),
                      direct=(dlogits.argmax(-1) == target).cpu().numpy(),
                      share=x["share"].cpu().numpy(), margin=x["margin"].cpu().numpy(),
                      write_norm=x["write"].norm(dim=-1).cpu().numpy(),
                      delta_norm=x["delta"].norm(dim=-1).cpu().numpy(),
                      query_norm=x["query"].norm(dim=-1).cpu().numpy(),
                      max_diff=float((qlogits - dlogits).abs().max()))
        for gain in (1, 16):
            record[f"oracle_content_{gain}"] = _hits(
                model, _oracle_content(o0, factor, x["oracle"], x["delta"], gain), target)
        record["matched"] = _hits(model, _oracle_content(o0, factor, x["matched"],
                                                         x["delta"]), target)
        return record


def _verdicts(models: list[dict], refs: dict) -> dict:
    bound = [m for m in models if m["group"] == "bound"]
    first = [m for m in models if m["group"] == "e2e1"]
    nb = len(bound)
    k = lambda m, key: m["K"]["echo"][key]
    valid = nb > 0 and sum(k(m, "linear")["1.0"] >= .90 for m in bound) >= nb - 1
    address = nb > 0 and sum(k(m, "address_dot") >= .80 for m in bound) >= nb - 1
    pre = len(first) == 6 and sum(k(m, "linear")["1.0"] <=
                                   refs["echo"]["occ_minus"]["acc"] + .05 for m in first) >= 5
    counts = {"gain": sum(k(m, "memory_inf") >= .80 for m in first),
              "address": sum(k(m, "memory_inf") <= .30 and k(m, "content_inf") >= .80
                             for m in first),
              "content": sum(k(m, "content_inf") <= .30 for m in first),
              "mis_addressed": sum(k(m, "memory_inf") <= .30 and k(m, "content_inf") >= .80
                                     and k(m, "address_dot") <= .30 for m in first),
              "non_selective": sum(k(m, "memory_inf") <= .30 and k(m, "content_inf") >= .80
                                     and k(m, "address_dot") >= .80 for m in first)}
    if len(first) != 6:
        kv = "insufficient e2e1 models"
    elif not valid:
        kv = "descriptive (K-invalid)"
    elif not pre:
        kv = "Downstream"
    else:
        kv = next((name.capitalize() for name in ("gain", "address", "content")
                   if counts[name] >= 5), "Mixed")
        if kv == "Address":
            kv += (" / mis-addressed" if counts["mis_addressed"] >= 5 else
                   " / non-selective" if counts["non_selective"] >= 5 else "")
    g = {}
    for mode in K_MODES[1:]:
        floor = refs[mode]["occ_minus"]["acc"] + .05
        labels = []
        for m in bound:
            zero = m["G"][mode]["zero_shot"]["query_acc"]
            ratio = m["G"][mode]["gate_ratio_events_1_15"]
            linear = m["K"][mode]["linear"]["1.0"]
            if zero >= .80:
                labels.append("Generic")
            elif ratio <= .5 and zero <= floor:
                labels.append("Gated")
            elif linear >= .80 and zero <= floor:
                labels.append("Downstream / overdrive" if
                              m["G"][mode]["norm_matched"]["query_acc"] >= .80 else "Downstream")
            elif ratio > .8 and linear < .80 and zero <= floor:
                labels.append("Fails-ungated")
            else:
                labels.append("Mixed")
        counts_g = {name: labels.count(name) for name in set(labels)}
        # Overdrive is a sub-label of Downstream, not a separate group threshold.
        downstream = sum(label.startswith("Downstream") for label in labels)
        winner = next((name for name in ("Generic", "Gated", "Downstream", "Fails-ungated")
                       if (downstream if name == "Downstream" else counts_g.get(name, 0)) >= nb - 1),
                      "Mixed") if nb else "insufficient bound models"
        g[mode] = {"verdict": winner, "counts": counts_g,
                   "per_model": {m["label"]: label for m, label in zip(bound, labels)},
                   "overdrive_count": labels.count("Downstream / overdrive"), "n_bound": nb}
    return {"K": {"valid": valid, "bound_addressable": address,
                  "bound_address_verdict": "addressable" if address else "superposed address",
                  "e2e1_precondition": pre, "e2e1_counts": counts, "verdict": kv,
                  "n_bound": nb, "n_e2e1": len(first)}, "G": g}


def _summary(model: dict) -> dict:
    return {"label": model["label"], "arm": model["arm"], "group": model["group"],
            "K": {mode: {"linear": k["linear"], **{key: k[key] for key in
                  ("memory_inf", "content_inf", "address_dot", "address_cos",
                   "effective_address", "address_ln", "address_random_w1",
                   "matched_share", "direct", "max_logit_diff")}}
                  for mode, k in model["K"].items()},
            "G": {mode: {key: g[key] for key in
                  ("zero_shot", "gate_ratio_event0", "gate_ratio_events_1_15",
                   "gate_norm", "write_norm", "chunk0_clipped_frac", "chunk0_scale_median") +
                  (("frozen_kv", "fixed_kv", "norm_matched") if mode in K_MODES else ())}
                  for mode, g in model["G"].items()},
            "sanity": model["sanity"]}


def run(cfg: dict, seed: int) -> dict:
    ttt._mark_process(cfg["process_name"])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    dev = ttt._device(cfg["device"])
    datasets = {}
    refs = {}
    for mode in MODES:
        st = ttt._settings(cfg, mode)
        task = ttt._task(st, mode)
        tokens, meta = ttt.fixed_set(task, st, "eval")
        datasets[mode] = (st, task, meta, ttt.eval_batches(task, tokens, meta, dev, st.eval_batch))
        refs[mode] = {k: {name: value for name, value in v.items() if name != "acc_per_seq"}
                      for k, v in ttt.echo_references(meta, task.n_keys).items()}
        _emit("KAG_REFS", {"mode": mode, **refs[mode]})
    # The published fingerprint is for 256 sequences even in the short smoke split.
    st, task, _, _ = datasets["echo"]
    ref_task = ttt._task(ttt._settings({**cfg, "echo": {**cfg["echo"], "n_eval": 256}},
                                            "echo"), "echo")
    fingerprint = ttt.echo_references(ttt.fixed_set(ref_task, st, "eval")[1], task.n_keys)
    assert abs(fingerprint["occ"]["acc"] - .1931) <= .001
    assert abs(fingerprint["occ_minus"]["acc"] - .3168) <= .001
    random_model = ttt.build_model(task, st, dev, 0, "e2e2").eval()
    models = []
    for entry in [*cfg["checkpoints"], {"label": "random-init", "arm": "e2e2",
                                          "eta_base": 10}]:
        model = (random_model if entry["label"] == "random-init" else
                 ttt.build_model(task, st, dev, 0, entry["arm"]))
        if "path" in entry:
            model.load_state_dict(torch.load(Path(cfg["ckpt_root"]) / entry["path"],
                                             map_location=dev, weights_only=True), strict=True)
        model.eval()
        etas = [e.detach() for e in model.etas(float(entry["eta_base"]))]
        k_results, g_results = {}, {}
        for mode in MODES:
            st, task, meta, data = datasets[mode]
            rows = []
            for idx, y, bm in data:
                with torch.no_grad():
                    h = model.backbone(idx).detach()
                    a = _code(model, h)
                    wp = torch.as_tensor(bm["write_pos"], device=dev)
                    gates = _gate_positions(a, wp, task.chunk)
                row = dict(h=h, y=y, meta=bm, **gates)
                if mode in K_MODES:
                    masks = wab.class_masks(bm, task.chunk, dev)
                    weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
                    z, _, grads = wab._class_grads(model, h[:, :task.chunk], y[:, :task.chunk],
                                                   weights, masks)
                    full = wab._sum_grads(grads, wab.ALL)
                    _, norm, clipped = ttt._clip_fast(full, st.g_clip)
                    scale = torch.where(clipped, st.g_clip / norm, torch.ones_like(norm))
                    with torch.no_grad():
                        o0 = ttt.fast_block(h[:, :task.chunk], model.head.ln, model.slow())
                    delta = _delta(model, o0, y[:, :task.chunk])
                    row.update(z=z, grads=grads, scale=scale, delta=delta,
                               clipped=clipped.cpu().numpy(),
                               kv_update_norm=wab._joint_norm(wab._weighted(grads[2], etas,
                                                                              scale)).cpu().numpy(),
                               write_delta=_gather(delta, wp).norm(dim=-1).cpu().numpy())
                else:
                    with torch.no_grad():
                        o_write = _gather(ttt.fast_block(h, model.head.ln, model.slow()), wp)
                    row["write_delta"] = _delta(model, o_write, _gather(y, wp)).norm(dim=-1).cpu().numpy()
                    weights = [w.detach().requires_grad_(True) for w in model.per_sequence(len(idx))]
                    with torch.enable_grad():
                        logits0 = model.read(ttt.fast_block(h[:, :task.chunk], model.head.ln, weights))
                        grad = torch.autograd.grad(ttt.token_ce(logits0, y[:, :task.chunk])
                                                   .mean(1).sum(), weights)
                    _, norm, clipped = ttt._clip_fast(list(grad), st.g_clip)
                    row["clipped"] = clipped.cpu().numpy()
                    row["scale"] = torch.where(clipped, st.g_clip / norm, torch.ones_like(norm))
                rows.append(row)
            if mode == "echo":
                echo_norm = _med([r["write_norm"] for r in rows])
                echo_event0 = _med([r["write_norm"][r["event"] == 0] for r in rows])
                echo_later = _med([r["write_norm"][r["event"] > 0] for r in rows])
                echo_kv_update = _med([r["kv_update_norm"] for r in rows])
            # Reuse the same fixed batches for a direct exp_ttt.evaluate zero-shot score.
            zero = ttt.evaluate(model, task, entry["arm"], data, meta,
                                float(entry["eta_base"]), st)["eval"]
            zero = {k: zero[k] for k in ("query_acc", "query_acc_se", "query_ce", "query_ce_se")}
            norms = [r["write_norm"] for r in rows]
            ev0 = _med([r["write_norm"][r["event"] == 0] for r in rows])
            later = _med([r["write_norm"][r["event"] > 0] for r in rows])
            g = {"zero_shot": zero, "gate_norm": _med(norms),
                 "gate_ratio": _med(norms) / echo_norm,
                 "gate_norm_event0": ev0, "gate_norm_events_1_15": later,
                 "gate_ratio_event0": ev0 / echo_event0,
                 "gate_ratio_events_1_15": later / echo_later,
                 "centred_gate_norm": _med([r["centre"] for r in rows]),
                 "write_norm": _med([etas[2].abs().item() * r["write_delta"] *
                                      r["write_norm"] for r in rows]),
                 "gate_state": "W0 code, not the streaming state" if mode == "echo_off" else "W0",
                 "kv_wv_gate_ratio": _med(norms) / _med([r["wv_norm"] for r in rows]),
                 "chunk0_clipped_frac": float(np.concatenate([r["clipped"] for r in rows]).mean()),
                 "chunk0_scale_median": _med([r["scale"].cpu().numpy() for r in rows])}
            if mode in K_MODES:
                k_parts = []
                conditions = {name: [] for name in ("frozen_kv", "fixed_kv", "norm_matched")}
                for r in rows:
                    h, y, bm = r["h"], r["y"], r["meta"]
                    direct = wab.eval_condition(model, h, y, r["z"], r["grads"], etas,
                                                st.g_clip, task.chunk, "frozen", KV,
                                                tensors=("W2",))
                    k_parts.append(_k_batch(model, random_model, h, bm, r["delta"],
                                            etas[2], r["scale"], direct))
                    for name, family in (("frozen_kv", "frozen"), ("fixed_kv", "fixed")):
                        logits = wab.eval_condition(model, h, y, r["z"], r["grads"], etas,
                                                    st.g_clip, task.chunk, family, KV)
                        conditions[name].append(_query_record(logits, bm))
                    matched = _norm_match(model, h, y, r["z"], r["grads"], etas,
                                          r["scale"], echo_kv_update, task.chunk, st.g_clip)
                    conditions["norm_matched"].append(_query_record(matched, bm))
                k = _k_metrics(k_parts)
                k_results[mode] = k
                _emit("KAG_K", {"label": entry["label"], "mode": mode, **k})
                for name, records in conditions.items():
                    g[name] = wab._metrics(records)
            g_results[mode] = g
            _emit("KAG_G", {"label": entry["label"], "mode": mode, **g})
        sanity = {"label": entry["label"], "fingerprint": True,
                  "echo_fingerprint_256": {key: fingerprint[key]["acc"]
                                            for key in ("occ", "occ_minus")},
                  "echo_ref_match": (None if "ref" not in entry else
                                     abs(g_results["echo"]["zero_shot"]["query_acc"] -
                                         entry["ref"]) <= .01),
                  "echo_ctrl_event0_ratio": g_results["echo_ctrl"]["gate_ratio_event0"],
                  "echo_ctrl_event0_in_range": .9 <= g_results["echo_ctrl"]["gate_ratio_event0"] <= 1.1,
                  "K0": {mode: {"accuracy_diff": abs(k["linear"]["1.0"] - k["direct"]),
                                "max_logit_diff": k["max_logit_diff"]} for mode, k in k_results.items()}}
        _emit("KAG_SANITY", sanity)
        if not cfg.get("smoke", False):
            assert sanity["echo_ref_match"] is not False, entry["label"]
        assert all(v["accuracy_diff"] <= .002 and v["max_logit_diff"] <= 1e-4
                   for v in sanity["K0"].values()), entry["label"]
        group = ("random-init" if entry["label"] == "random-init" else
                 "e2e1" if entry["arm"] == "e2e1" else
                 "bound" if g_results["echo"]["zero_shot"]["query_acc"] >= .90 else
                 "e2e2 contrast")
        models.append({"label": entry["label"], "arm": entry["arm"], "group": group,
                       "K": k_results, "G": g_results, "sanity": sanity})
    verdicts = _verdicts(models, refs) if not cfg.get("smoke", False) else {
        "K": {"verdict": "not evaluated (smoke subset)"},
        "G": {mode: {"verdict": "not evaluated (smoke subset)"} for mode in K_MODES[1:]}}
    return {"refs": refs, "models": [_summary(m) for m in models], "verdicts": verdicts,
            "smoke": bool(cfg.get("smoke", False))}
