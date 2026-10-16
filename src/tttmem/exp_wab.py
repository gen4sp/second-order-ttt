"""Eval-only write-position ablations of saved echo fast-weight checkpoints."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as ttt

CLASSES = ("Wv", "vk", "kv", "vW")
ALL = frozenset(CLASSES)
SCALES = (-1, 0.03, 0.1, 0.3, 3, 10, 30, 100, 300)


def _emit(kind: str, record: dict) -> None:
    print(kind + " " + json.dumps(record, sort_keys=True), flush=True)


def class_masks(meta: dict, chunk: int, device: torch.device) -> torch.Tensor:
    """(4, B, chunk) boolean partition, indexed by the key's write position."""
    wp = torch.as_tensor(np.asarray(meta["write_pos"]), device=device)
    assert wp.ndim == 2 and wp.shape[1] * 4 == chunk
    assert bool(((wp >= 2) & (wp + 1 < chunk)).all()), "writes must lie in chunk 0"
    masks = torch.zeros(4, wp.shape[0], chunk, dtype=torch.bool, device=device)
    for c, offset in enumerate((-2, -1, 0, 1)):
        masks[c].scatter_(1, wp + offset, True)
    assert bool((masks.sum(0) == 1).all()), "write classes must partition chunk 0"
    return masks


def _class_grads(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor,
                 weights: list[torch.Tensor], masks: torch.Tensor,
                 idx: torch.Tensor | None = None,
                 sel: torch.Tensor | None = None, *, rule: str = "ce"
                 ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
    """Four per-sequence mean-chunk gradients, reusing one chunk-0 graph.

    ``sel`` replaces the token CE by the caller's own token loss before the class
    masks are applied, so a lin arm's class gradients are its own (as in R4-2L).
    """
    with torch.enable_grad():
        logits = model.read(model.fast_block(h, idx, weights) if idx is not None else
                            ttt.fast_block(h, model.head.ln, weights))
        ce = ttt.token_ce(logits, y)
        loss = ttt.inner_token_loss(rule, logits, y, ce=ce, sel=sel,
                                    reference=logits.detach() if rule == "aice" else None)
        grads = [list(torch.autograd.grad((loss * masks[c]).sum() / y.shape[1], weights,
                                         retain_graph=c < 3)) for c in range(4)]
    return logits.detach(), ce.detach(), grads


def _sum_grads(grads: list[list[torch.Tensor]], subset: frozenset[str]
               ) -> list[torch.Tensor]:
    return [sum((grads[c][t] for c, name in enumerate(CLASSES) if name in subset),
                torch.zeros_like(grads[0][t])) for t in range(4)]


def _joint_norm(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([p.flatten(1).square().sum(1) for p in parts]).sum(0).sqrt()


def _weighted(parts: list[torch.Tensor], etas: list[torch.Tensor],
              scale: torch.Tensor) -> list[torch.Tensor]:
    return [eta * g * scale.reshape(-1, *([1] * (g.ndim - 1)))
            for eta, g in zip(etas, parts)]


def _permutations(n: int, n_keys: int, vocab: int, generator: torch.Generator,
                  device: torch.device) -> torch.Tensor:
    """One independent permutation per sequence, drawn in eval sequence order."""
    first = 2 + n_keys
    pi = torch.arange(vocab).repeat(n, 1)
    for row in pi:
        row[first:first + n_keys] = first + torch.randperm(n_keys, generator=generator)
    return pi.to(device)


def _swapped_kv(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor,
                weights: list[torch.Tensor], mask: torch.Tensor,
                pi: torch.Tensor) -> list[torch.Tensor]:
    """Backpropagate a permuted output error, not CE of a permuted target."""
    with torch.enable_grad():
        z = model.read(ttt.fast_block(h, model.head.ln, weights))
        delta = (z.softmax(-1) - F.one_hot(y, z.shape[-1])).detach()
        py = pi.gather(1, y)
        swapped = delta.clone()
        swapped.scatter_(2, y[..., None], delta.gather(2, py[..., None]))
        swapped.scatter_(2, py[..., None], delta.gather(2, y[..., None]))
        loss = ((z * swapped).sum(-1) * mask).sum() / y.shape[1]
        return list(torch.autograd.grad(loss, weights))


def _conditions(smoke: bool, clipped: bool) -> list[tuple[str, str, frozenset[str], float, tuple[str, ...]]]:
    subsets = [frozenset(s) for n in range(5) for s in itertools.combinations(CLASSES, n)]
    if smoke:
        subsets = [frozenset(), frozenset(("kv",)), ALL - {"kv"}, ALL]
    out = [("fixed", "all" if s == ALL else "+".join(n for n in CLASSES if n in s) or "none",
            s, 1., ttt.FAST_KEYS)
           for s in subsets]
    if not smoke:
        if clipped:
            out += [("reclip", label, s, 1., ttt.FAST_KEYS)
                    for _, label, s, _, _ in out[:16]]
        out += [("frozen", label, s, 1., ttt.FAST_KEYS)
                for label, s in (("all", ALL), ("none", frozenset()), ("kv", frozenset(("kv",))),
                                 ("all-kv", ALL - {"kv"}))]
        out += [("tensor", "+".join(keys) + "/" + label, s, 1., keys)
                for keys in (("W2",), ("W1", "b1"), ("b2",))
                for label, s in (("all", ALL), ("kv", frozenset(("kv",))))]
        out += [("transplant", "kv-donor", ALL, 1., ttt.FAST_KEYS),
                ("dswap", "kv-delta-swap", ALL, 1., ttt.FAST_KEYS)]
        out += [("scale", str(a), ALL, float(a), ttt.FAST_KEYS) for a in SCALES]
    return out


@torch.no_grad()
def eval_condition(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor,
                   logits0: torch.Tensor, grads: list[list[torch.Tensor]],
                   etas: list[torch.Tensor], g_clip: float, chunk: int,
                   family: str, subset: frozenset[str], alpha: float = 1.,
                   tensors: tuple[str, ...] = ttt.FAST_KEYS,
                   swapped: list[torch.Tensor] | None = None,
                   idx: torch.Tensor | None = None,
                   sel: torch.Tensor | None = None, *, rule: str = "ce") -> torch.Tensor:
    """Streaming logits given a changed chunk-0 update; later chunks are standard.

    With ``sel`` the later chunks stream with the caller's own inner token loss
    (as in R4-2L), which is what makes "all" reproduce the arm's streaming eval.
    """
    batch = y.shape[0]
    reference = (ttt.reference_logits(model, h, idx, model.per_sequence(batch))
                 if rule == "aice" else None)
    full = _sum_grads(grads, ALL)
    _, norm, _ = ttt._clip_fast(full, g_clip)
    scale = torch.where(norm > g_clip, g_clip / norm, torch.ones_like(norm))
    chosen = _sum_grads(grads, subset)
    if family == "transplant":
        chosen = [g - own + own.roll(1, 0) for g, own in zip(chosen, grads[2])]
    elif family == "dswap":
        assert swapped is not None
        chosen = [g - own + replacement for g, own, replacement in zip(chosen, grads[2], swapped)]
    elif family == "scale":
        chosen = [g + (alpha - 1) * own for g, own in zip(chosen, grads[2])]
    if family == "reclip":
        chosen, _, _ = ttt._clip_fast(chosen, g_clip)
        scale = torch.ones_like(scale)
    updates = _weighted(chosen, etas, scale)
    weights = [(p.detach().unsqueeze(0).expand(batch, *p.shape) - u * (key in tensors))
               .detach().requires_grad_(True)
               for p, u, key in zip(model.slow(), updates, ttt.FAST_KEYS)]
    out = [logits0]
    with torch.enable_grad():
        for c in range(1, y.shape[1] // chunk):
            sl = slice(c * chunk, (c + 1) * chunk)
            logits = model.read(model.fast_block(h[:, sl], idx[:, sl], weights) if idx is not None
                                else ttt.fast_block(h[:, sl], model.head.ln, weights))
            out.append(logits.detach())
            if family == "frozen":
                continue
            ce = ttt.token_ce(logits, y[:, sl])
            inner = ttt.inner_token_loss(rule, logits, y[:, sl], ce=ce,
                                        sel=None if sel is None else sel[:, sl],
                                        reference=None if reference is None else reference[:, sl])
            g = list(torch.autograd.grad(inner.mean(1).sum(), weights))
            g, _, _ = ttt._clip_fast(g, g_clip)
            weights = [(w - eta * d).detach().requires_grad_(True)
                       for w, eta, d in zip(weights, etas, g)]
    return torch.cat(out, dim=1)


def _metrics(chunks: list[dict[str, np.ndarray]], by_event: bool = False) -> dict:
    hit = np.concatenate([x["hit"] for x in chunks])
    ce = np.concatenate([x["ce"] for x in chunks])
    n = len(hit)
    per_hit, per_ce = hit.mean(1), ce.mean(1)
    result = dict(query_acc=float(per_hit.mean()),
                  query_acc_se=float(per_hit.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.,
                  query_ce=float(per_ce.mean()),
                  query_ce_se=float(per_ce.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.)
    if "alt" in chunks[0]:
        result["alternative_map_acc"] = float(np.concatenate([x["alt"] for x in chunks]).mean())
        result["chance_agreement"] = float(np.concatenate([x["agreement"] for x in chunks]).mean())
    if by_event:
        order = np.concatenate([x["event"] for x in chunks])
        result["query_acc_by_write_event"] = [float(hit[order == j].mean())
                                               for j in range(hit.shape[1])]
    return result


def _diagnostics(model: ttt.TTTModel, parts: list[dict], etas: list[torch.Tensor]) -> dict:
    out = {"eta": {k: float(e) for k, e in zip(ttt.FAST_KEYS, etas)},
           "clipped_frac": float(np.concatenate([p["clipped"] for p in parts]).mean()),
           "classes": {}, "norm_share_note": "ill-posed: cancellation; shares need not sum to 1"}
    for name in CLASSES:
        items = [p[name] for p in parts]
        out["classes"][name] = {
            "ce_w0": float(np.concatenate([v["ce"] for v in items]).mean()),
            "update_norm_median": float(np.median(np.concatenate([v["norm"] for v in items]))),
            "norm_share_median": float(np.median(np.concatenate([v["share"] for v in items]))),
            "projection_share_median": float(np.median(np.concatenate([v["projection"] for v in items]))),
            "logit_error_norm_median": float(np.median(np.concatenate([v["delta"] for v in items]))),
            "activation_norm_median": float(np.median(np.concatenate([v["activation"] for v in items]))),
            "tensor_update_norm_median": {k: float(np.median(np.concatenate([v[k] for v in items])))
                                          for k in ttt.FAST_KEYS}}
    return out


def _batch_diagnostics(model: ttt.TTTModel, h: torch.Tensor, y: torch.Tensor,
                       logits: torch.Tensor, ce: torch.Tensor, masks: torch.Tensor,
                       grads: list[list[torch.Tensor]], etas: list[torch.Tensor],
                       g_clip: float) -> dict:
    full = _sum_grads(grads, ALL)
    _, norm, clipped = ttt._clip_fast(full, g_clip)
    scale = torch.where(clipped, g_clip / norm, torch.ones_like(norm))
    all_update = _weighted(full, etas, scale)
    total = _joint_norm(all_update).clamp_min(1e-20)
    denom = total.square()
    delta = (logits.softmax(-1) - F.one_hot(y, logits.shape[-1])).norm(dim=-1)
    with torch.no_grad():
        w1, b1 = model.slow()[:2]
        activation = F.gelu(F.linear(model.head.ln(h), w1, b1)).norm(dim=-1)
    record = {"clipped": clipped.cpu().numpy()}
    for c, name in enumerate(CLASSES):
        u = _weighted(grads[c], etas, scale)
        norm_c = _joint_norm(u)
        select = masks[c]
        record[name] = {
            "ce": ce[select].cpu().numpy(),
            "norm": norm_c.cpu().numpy(),
            "share": (norm_c / total).cpu().numpy(),
            "projection": (sum((a * b).flatten(1).sum(1) for a, b in zip(u, all_update))
                           / denom).cpu().numpy(),
            "delta": delta[select].cpu().numpy(),
            "activation": activation[select].cpu().numpy(),
            **{key: v.flatten(1).norm(dim=1).cpu().numpy()
               for key, v in zip(ttt.FAST_KEYS, u)}}
    return record


def _verdict(models: list[dict], refs: dict, family: str) -> dict:
    bound = [m for m in models if m["bound"]]
    n = len(bound)
    if not n:
        return {"n_bound": 0, "verdict": "no bound models"}
    def acc(model: dict) -> dict:
        # Unclipped models have no reclip lines; their fixed results stand in.
        return model["accuracy"].get(family, model["accuracy"]["fixed"])

    counts = {c: sum(acc(m)["+".join(k for k in CLASSES if k != c)]
                     <= refs["occ_minus"]["acc"] + 0.05 for m in bound) for c in CLASSES}
    sufficient = {c: sum(acc(m)[c] >= .90 for m in bound) for c in CLASSES}
    relative = {c: sum(acc(m)[c] >= acc(m)["all"] - .05
                       for m in bound) for c in CLASSES}
    fired = "".join(letter for c, letter in zip(CLASSES, ("W", "R", "F", "S"))
                    if counts[c] >= n - 1)
    minima = {}
    per_model = {}
    def members(label: str) -> frozenset[str]:
        return ALL if label == "all" else frozenset() if label == "none" else frozenset(label.split("+"))

    for m in bound:
        good = [s for s in acc(m) if acc(m)[s] >= .90]
        minima[m["label"]] = [s for s in good if not any(
            members(other) < members(s) for other in good)]
        per_model[m["label"]] = {
            c: {"necessary": acc(m)["+".join(k for k in CLASSES if k != c)]
                 <= refs["occ_minus"]["acc"] + .05,
                "sufficient": acc(m)[c] >= .90,
                "relative_sufficient": acc(m)[c] >= acc(m)["all"] - .05,
                "partial": refs["occ_minus"]["acc"] + .05 < acc(m)[c] < .90}
            for c in CLASSES}
    return dict(n_bound=n, necessary_counts=counts, sufficient_counts=sufficient,
                relative_sufficient_counts=relative, fired=list(fired),
                verdict=" & ".join(fired) if fired else "Distributed",
                per_model=per_model, minimal_sufficient_subsets=minima)


def _post_hoc_verdict(models: list[dict], pre: dict) -> dict:
    bound = [m for m in models if m["bound"]]
    n = len(bound)
    if n < 3:
        return {"n_bound": n, "verdict": "no verdict (n_bound < 3)"}
    counts = pre["necessary_counts"]
    kv_suff = pre["sufficient_counts"]["kv"]
    f = ("F-strong" if counts["kv"] >= n - 1 and kv_suff >= n - 1 else
         "F-weak" if counts["kv"] >= n - 1 else None)
    fired = [x for x in pre["fired"] if x != "F"] + ([f] if f else [])
    heterogeneous = all(any(m["accuracy"]["fixed"]["+".join(k for k in CLASSES if k != c)]
                            <= m["occ_minus_threshold"] for c in CLASSES) for m in bound)
    if not fired and heterogeneous and max(counts.values()) < n - 1:
        fired = ["Heterogeneous"]
    return {"n_bound": n, "verdict": " & ".join(fired) if fired else "Distributed",
            "fired": fired, "relative_sufficient_counts": pre["relative_sufficient_counts"]}


def run(cfg: dict, seed: int) -> dict:
    ttt._mark_process(cfg["process_name"])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    dev = ttt._device(cfg["device"])
    st = ttt._settings(cfg, "echo")
    task = ttt._task(st, "echo")
    assert task.chunk == 4 * task.n_keys and task.T == 512 and task.n_keys == 16
    tokens, meta = ttt.fixed_set(task, st, "eval")
    # The fingerprint is defined on 256 sequences even for the short smoke split.
    ref_task = ttt._task(ttt._settings({**cfg, "echo": {**cfg["echo"], "n_eval": 256}},
                                           "echo"), "echo")
    _, ref_meta = ttt.fixed_set(ref_task, st, "eval")
    refs = ttt.echo_references(ref_meta, task.n_keys)
    assert abs(refs["occ"]["acc"] - .1931) < .001
    assert abs(refs["occ_minus"]["acc"] - .3168) < .001
    refs = {key: {k: v for k, v in value.items() if k != "acc_per_seq"}
            for key, value in refs.items()}
    _emit("WAB_REFS", refs)
    data = ttt.eval_batches(task, tokens, meta, dev, st.eval_batch)
    results = []
    for entry in cfg["checkpoints"]:
        path = Path(cfg["ckpt_root"]) / entry["path"]
        assert "sgn-1" not in entry["path"] and "sgn-1" not in entry["label"]
        model = ttt.build_model(task, st, dev, 0, entry["arm"])
        model.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            etas = [e.detach() for e in model.etas(float(entry["eta_base"]))]
        records: dict[tuple[str, str], list[dict]] = {}
        diagnostics = []
        generator = torch.Generator().manual_seed(0)
        clipped_any = False
        for idx, y, batch_meta in data:
            B = len(idx)
            with torch.no_grad():
                h = model.backbone(idx).detach()
            masks = class_masks(batch_meta, task.chunk, dev)
            weights = [w.detach().requires_grad_(True) for w in model.per_sequence(B)]
            logits0, ce0, grads = _class_grads(model, h[:, :task.chunk], y[:, :task.chunk],
                                               weights, masks)
            diagnostic = _batch_diagnostics(model, h[:, :task.chunk], y[:, :task.chunk],
                                            logits0, ce0, masks, grads, etas, st.g_clip)
            diagnostics.append(diagnostic)
            clipped_any |= bool(diagnostic["clipped"].any())
            # Draw in sequence order even when a batch has no clipped gradients.
            pi = _permutations(B, task.n_keys, task.V, generator, dev)
            swap = None
            if not cfg.get("smoke", False):
                swap = _swapped_kv(model, h[:, :task.chunk], y[:, :task.chunk], weights,
                                   masks[2], pi)
            pos = torch.as_tensor(batch_meta["query_pos"], device=dev)
            target = torch.as_tensor(batch_meta["query_tok"], device=dev)
            event = np.argsort(np.argsort(batch_meta["write_pos"], axis=1), axis=1)
            for family, label, subset, alpha, tensors in _conditions(bool(cfg.get("smoke", False)),
                                                                     True):
                logits = eval_condition(model, h, y, logits0, grads, etas, st.g_clip, task.chunk,
                                        family, subset, alpha, tensors, swap)
                with torch.no_grad():
                    query = logits.gather(1, pos[..., None].expand(-1, -1, task.V))
                    record = {"hit": (query.argmax(-1) == target).cpu().numpy(),
                              "ce": ttt.token_ce(query, target).cpu().numpy()}
                    if family in ("transplant", "dswap"):
                        alt = target.roll(1, 0) if family == "transplant" else pi.gather(1, target)
                        record["alt"] = (query.argmax(-1) == alt).cpu().numpy()
                        record["agreement"] = (target == alt).cpu().numpy()
                    if family == "fixed" and label in ("all", "kv"):
                        record["event"] = event
                    records.setdefault((family, label), []).append(record)
        if not clipped_any:
            records = {key: value for key, value in records.items() if key[0] != "reclip"}
        accuracy = {}
        for (family, label), chunks in records.items():
            metrics = _metrics(chunks, by_event=family == "fixed" and label in ("all", "kv"))
            _emit("WAB", {"label": entry["label"], "arm": entry["arm"], "family": family,
                          "subset": label, **metrics})
            accuracy.setdefault(family, {})[label] = metrics["query_acc"]
            if family == "fixed" and label in ("all", "kv"):
                accuracy.setdefault("by_write_event", {})[label] = metrics["query_acc_by_write_event"]
        all_acc = accuracy["fixed"]["all"]
        sanity = {"label": entry["label"], "ref_match": abs(all_acc - entry["ref"]) <= .01,
                  "none_above_occ_minus": accuracy["fixed"]["none"] > refs["occ_minus"]["acc"] + .05,
                  "occ_fingerprint": True, "masks_partition": True}
        _emit("WAB_SANITY", sanity)
        if not cfg.get("smoke", False):
            assert sanity["ref_match"], f"{entry['label']}: all={all_acc}, ref={entry['ref']}"
        info = {"label": entry["label"], "arm": entry["arm"], "path": entry["path"],
                "ref": entry["ref"], "all_acc": all_acc, "bound": all_acc >= .90,
                "diagnostics": _diagnostics(model, diagnostics, etas)}
        if not clipped_any:
            info["reclip"] = "identical by construction"
        _emit("WAB_MODEL", info)
        results.append({"label": entry["label"], "arm": entry["arm"], "bound": info["bound"],
                        "accuracy": accuracy, "sanity": sanity,
                        "occ_minus_threshold": refs["occ_minus"]["acc"] + .05})
    if cfg.get("smoke", False):
        return {"refs": refs, "models": results, "smoke": True}
    pre = _verdict(results, refs, "fixed")
    if any("reclip" in m["accuracy"] for m in results):
        other = _verdict(results, refs, "reclip")
        pre["clip_sensitive"] = other["verdict"] != pre["verdict"]
        pre["reclip_verdict"] = other
    else:
        pre["clip_sensitive"] = False
    return {"refs": refs, "models": results, "pre_registered": pre,
            "post_hoc_verdict": _post_hoc_verdict(results, pre)}
