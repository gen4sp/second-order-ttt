"""SOQ-P1 training plumbing: echo injection hooks and the echo eval read.

The generator lives in :mod:`tttmem.p2.echo_text`.  This module only
adapts it to the trainer:

- :func:`make_train_injector` builds the streaming training injector from ``TrainCfg``;
  the Prefetcher calls it on host rows before they go to the device, keyed by the
  global stream position (the same index as the inner ``row_ids``).
- :func:`build_echo_eval_set` builds the TEST-template documents once per run.
- :func:`echo_read` runs the production adapted stream on them and returns, per slot,
  the value NLL and greedy exact match at the source (binding) and at every query read.
  Control sentences (no-barrier, no-binding, query-entity) are removed from the inner
  (write) loss with the generator's ``write_mask``, so they cannot change the fast
  weights that later reads see.  Training rows never contain controls.

Strata: copy-available / copy-blocked (the two binding strata, read at the
barrier query), no-binding, no-barrier and the query-entity counterfactual.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

# Generator metadata key -> short read name used in the npz/evidence keys.
READS = {
    "source": "source_nll_positions",
    "query": "query_nll_positions",
    "no_binding": "no_binding_nll_positions",
    "no_barrier": "no_barrier_nll_positions",
    "query_entity": "query_entity_nll_positions",
}
STRATA = ("copy_available", "copy_blocked", "no_binding", "no_barrier", "query_entity")


def make_train_injector(cfg, excluded):
    """The training injector, or ``None`` when ``echo_f == 0`` (plain MINI training)."""
    if cfg.echo_f <= 0:
        return None
    from .echo_text import make_injector
    return make_injector(split="train", f=cfg.echo_f, p=cfg.echo_p, seed=cfg.echo_seed,
                         T=cfg.seq_len, S=cfg.segment_len, excluded_vocab=np.asarray(excluded))


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compact_manifest(injector) -> dict:
    """``injector.manifest()`` without the (large) pool lists, plus their sha256."""
    man = injector.manifest()
    out = {k: v for k, v in man.items() if k != "pools"}
    if "pools" in man and "pools_sha256" not in man:
        out["pools_sha256"] = _sha256_json(man["pools"])
    return out


def manifest_id(manifest: dict) -> str:
    """The sha256 that pins the generator: ``manifest_sha256`` (params + pools) when the
    generator provides it, else ``params_sha256``."""
    return manifest.get("manifest_sha256") or manifest["params_sha256"]


def selected_mask(injector, n_rows: int) -> np.ndarray:
    """Which global stream positions ``[0, n_rows)`` the injector selects (exposure ledger).

    Uses the generator's selection rule directly (no row data needed)."""
    from . import echo_text
    sel = getattr(injector, "selected", None)
    if callable(sel):
        return np.asarray(sel(np.arange(n_rows, dtype=np.int64)), dtype=bool)
    return np.fromiter((echo_text._selected(injector.f, injector.seed, i) for i in range(n_rows)),
                       dtype=bool, count=n_rows)


def tokens_per_injection(injector, row_len: int) -> int:
    """Tokens overwritten in one injected row (host filled with an id no pool can draw).
    Fallback for injectors without ``injected_tokens`` (test stubs)."""
    host = np.full((1, row_len), 128000, dtype=np.uint32)
    out, _ = injector.eval_documents(host, row_ids=np.zeros(1, dtype=np.int64))
    return int((out[0] != host[0]).sum())


def exposure(injector, n_rows: int, row_len: int) -> dict:
    """Training exposure ledger over the stream positions ``[0, n_rows)``.

    With the real generator the injected-token count is exact per training row (the
    paraphrase lengths vary by row; training rows carry no controls)."""
    sel = selected_mask(injector, n_rows)
    n_slots = injector.layout.n_slots
    n_avail = int(round(n_slots * injector.p))
    n_inj = int(sel.sum())
    per_row = getattr(injector, "injected_tokens", None)
    if callable(per_row):
        tok_rows = np.asarray(per_row(np.arange(n_rows, dtype=np.int64)), dtype=np.int64)
        if not np.array_equal(tok_rows > 0, sel):
            raise ValueError("echo exposure: injected-token rows differ from the selected rows")
        total = int(tok_rows.sum())
    else:
        total = n_inj * tokens_per_injection(injector, row_len)
    return {"rows": int(n_rows), "injected_rows": n_inj,
            "source_writes": {"copy_available": n_inj * n_avail,
                              "copy_blocked": n_inj * (n_slots - n_avail)},
            "query_exposures": {"copy_available": n_inj * n_avail,
                                "copy_blocked": n_inj * (n_slots - n_avail)},
            "injected_tokens": total,
            "tokens_per_injected_row": total / max(1, n_inj),
            "token_share": total / float(n_rows * (row_len - 1))}


def check_exposure(ledger: dict, *, min_per_stratum: int = 1000, max_token_share: float = .05):
    """Assert >= 1000 writes per trained stratum and injected tokens <= 5 % (registered exposure gate)."""
    for stratum, n in ledger["source_writes"].items():
        if 0 < n < min_per_stratum:
            raise ValueError(f"echo exposure: {stratum} has {n} source writes < {min_per_stratum}")
    if ledger["token_share"] > max_token_share:
        raise ValueError(f"echo exposure: injected token share {ledger['token_share']:.4f} "
                         f"> {max_token_share}")


@dataclass
class EchoEvalSet:
    docs: np.ndarray            # [N, T + 1] uint32, TEST templates in clean host rows
    available: np.ndarray       # [N, n_slots] bool
    positions: dict             # read -> [N, n_slots, L] int64 input positions (target = +1)
    row_ids: np.ndarray         # host row ids in the eval split
    docs_sha256: str
    manifest: dict
    write_mask: np.ndarray | None = None   # [N, T] bool inner mask (False = control token)
    value_tokens: np.ndarray | None = None # [N, n_slots, L], candidate first tokens
    key_positions: dict | None = None
    query_mode: str = "normal"


def build_echo_eval_set(cfg, excluded, query_mode: str = "normal") -> EchoEvalSet:
    """TEST templates (generator split ``test``) in host rows of a held-out split.

    The pools come from the same ``echo_seed`` as the training injector, so the train/CAL/TEST
    pools are the generator's disjoint partition."""
    from .data import open_split
    from .echo_text import make_injector, write_mask
    T1 = cfg.seq_len + 1
    arr, _ = open_split(cfg.data_dir, cfg.echo_eval_split)
    lo, n = int(cfg.echo_eval_offset), int(cfg.echo_eval_docs)
    if lo < 0 or n < 1 or lo + n > len(arr):
        raise ValueError(f"echo eval needs rows [{lo}, {lo + n}) of split {cfg.echo_eval_split} "
                         f"({len(arr)} rows)")
    hosts = np.asarray(arr[lo:lo + n, :T1])
    row_ids = np.arange(lo, lo + n, dtype=np.int64)
    inj = make_injector(split="test", f=1.0, p=cfg.echo_eval_p, seed=cfg.echo_seed,
                        T=cfg.seq_len, S=cfg.segment_len, excluded_vocab=np.asarray(excluded))
    docs, meta = inj.eval_documents(hosts, row_ids=row_ids, p=cfg.echo_eval_p, query_mode=query_mode)
    recs = meta["rows"]
    available = np.asarray([r["available"] for r in recs], dtype=bool)
    positions = {name: np.asarray([r[key] for r in recs], dtype=np.int64) for name, key in READS.items()}
    for name, pos in positions.items():
        if pos.shape[:2] != available.shape or pos.min() < 0 or pos.max() >= cfg.seq_len:
            raise ValueError(f"echo eval positions for {name} out of range: {pos.shape}")
    return EchoEvalSet(docs=docs, available=available, positions=positions, row_ids=row_ids,
                       docs_sha256=hashlib.sha256(np.ascontiguousarray(docs).tobytes()).hexdigest(),
                       manifest=compact_manifest(inj), write_mask=write_mask(recs, cfg.seq_len),
                       value_tokens=np.asarray([r["value_tokens"] for r in recs], dtype=np.int64),
                       key_positions={k: np.asarray([r[k] for r in recs], dtype=np.int64)
                                      for k in ("query_key_positions", "no_binding_key_positions")
                                      if k in recs[0]}, query_mode=query_mode)


def build_train_echo_eval_set(cfg, excluded) -> EchoEvalSet:
    """Train templates/pools, independently re-paired, in LM-held-out host rows."""
    from .data import open_split
    from .echo_text import make_injector, write_mask
    arr, _ = open_split(cfg.data_dir, cfg.echo_train_eval_split)
    lo, n = int(cfg.echo_train_eval_offset), int(cfg.echo_eval_docs)
    if lo < 0 or lo + n > len(arr):
        raise ValueError(f"train echo eval needs rows [{lo}, {lo+n}) of {cfg.echo_train_eval_split}")
    hosts = np.asarray(arr[lo:lo+n, :cfg.seq_len + 1])
    row_ids = np.arange(lo, lo+n, dtype=np.int64)
    inj = make_injector(split="train", f=1.0, p=cfg.echo_eval_p,
                        seed=cfg.echo_train_eval_seed, T=cfg.seq_len, S=cfg.segment_len,
                        excluded_vocab=np.asarray(excluded))
    docs, meta = inj.eval_documents(hosts, row_ids=row_ids, p=cfg.echo_eval_p)
    recs = meta["rows"]
    available = np.asarray([r["available"] for r in recs], dtype=bool)
    positions = {name: np.asarray([r[key] for r in recs], dtype=np.int64) for name, key in READS.items()}
    return EchoEvalSet(docs=docs, available=available, positions=positions, row_ids=row_ids,
                       docs_sha256=hashlib.sha256(np.ascontiguousarray(docs).tobytes()).hexdigest(),
                       manifest=compact_manifest(inj), write_mask=write_mask(recs, cfg.seq_len),
                       value_tokens=np.asarray([r["value_tokens"] for r in recs], dtype=np.int64),
                       key_positions={k: np.asarray([r[k] for r in recs], dtype=np.int64)
                                      for k in ("query_key_positions", "no_binding_key_positions")
                                      if k in recs[0]}, query_mode="normal")


def echo_read(model, es: EchoEvalSet, *, rule: str, inner_lr: float, inner_clip: float,
              barriers, device, fused_attn: bool, batch: int, r_ref: float = .03,
              data_seed: int = 0, diagnostics: bool = False) -> tuple[dict, dict]:
    """Adapted production stream on the echo documents.

    Returns ``(arrays, summary)``.  ``arrays`` holds ``echo_<read>_nll`` / ``echo_<read>_em``
    [N, n_slots] (value NLL mean over the value tokens; greedy exact match of all value
    tokens under teacher forcing, which equals greedy-decoding exact match) and the slot
    labels.  ``summary`` holds the per-stratum means (recall at the query reads, source NLL
    per binding stratum)."""
    import torch
    from .meta import InnerCfg, stream_eval
    from .probe import boundary_valid

    C, bos = model.cfg.chunk, model.cfg.bos_id
    N, T1 = es.docs.shape
    T = T1 - 1
    bars = list(barriers)
    bvalid = torch.from_numpy(boundary_valid(T, C, bars)).to(device)
    nll = {k: np.full(es.available.shape, np.nan) for k in READS}
    em = {k: np.zeros(es.available.shape, dtype=bool) for k in READS}
    first_nll = {k: np.full(es.available.shape, np.nan) for k in READS}
    rest_nll = {k: np.full(es.available.shape, np.nan) for k in READS}
    key_nll = {k: np.full(es.available.shape, np.nan) for k in ("query", "no_binding")}
    candidate = {k: np.full((N, es.available.shape[1], es.available.shape[1]), np.nan, dtype=np.float32)
                 for k in ("query", "no_binding")} if diagnostics and es.value_tokens is not None else None
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        arr = torch.from_numpy(es.docs[sl].astype(np.int64)).to(device)
        ids, tgt = arr[:, :-1], arr[:, 1:]
        bmask = (tgt != bos) & bvalid
        B = ids.shape[0]
        sel = torch.zeros(B, T, dtype=torch.bool, device=device)
        for pos in es.positions.values():
            p = torch.from_numpy(pos[sl].reshape(B, -1)).to(device)
            sel.scatter_(1, p, True)
        if es.key_positions:
            for pos in es.key_positions.values():
                p = torch.from_numpy(pos[sl].reshape(B, -1)).to(device)
                sel.scatter_(1, p, True)
        inner = InnerCfg(rule=rule, lr=inner_lr, clip=inner_clip, r_ref=r_ref, seed=data_seed,
                         row_ids=torch.arange(b0, b0 + B, device=device))
        wmask = (torch.from_numpy(es.write_mask[sl]).to(device)
                 if es.write_mask is not None else None)
        r = stream_eval(model, ids, tgt, bmask, inner, barriers=bars, nll_positions=sel,
                        inner_mask=wmask, fused_attn=fused_attn, return_greedy=True,
                        candidate_positions=(torch.from_numpy(np.concatenate(
                            [es.positions["query"][sl, :, 0],
                             es.positions["no_binding"][sl, :, 0]], axis=1)).to(device)
                                             if candidate is not None else None),
                        candidate_ids=(torch.from_numpy(np.array(np.broadcast_to(
                            es.value_tokens[sl, :, 0][:, None, :],
                            (B, 2 * es.available.shape[1], es.available.shape[1])), copy=True)).to(device)
                                       if candidate is not None else None))
        tok_nll = r["nll"].double().cpu().numpy()
        tok_ok = r["greedy"].cpu().numpy()
        rows = np.arange(B)[:, None, None]
        for k, pos in es.positions.items():
            q = pos[sl]                                   # [B, n_slots, L]
            nll[k][sl] = tok_nll[rows, q].mean(-1)
            em[k][sl] = tok_ok[rows, q].all(-1)
            first_nll[k][sl] = tok_nll[rows, q[:, :, :1]].squeeze(-1)
            rest_nll[k][sl] = tok_nll[rows, q[:, :, 1:]].mean(-1)
        if es.key_positions:
            for k, pos in es.key_positions.items():
                short = "query" if k.startswith("query_") else "no_binding"
                q = pos[sl]
                key_nll[short][sl] = tok_nll[rows, q[:, :, 1:]].mean(-1)
        if candidate is not None:
            cp = r["candidate_log_probs"].cpu().numpy()
            candidate["query"][sl] = cp[:, :es.available.shape[1]]
            candidate["no_binding"][sl] = cp[:, es.available.shape[1]:]
    arrays = {"echo_available": es.available, "echo_row_ids": es.row_ids,
              "echo_docs_sha256": np.asarray(es.docs_sha256)}
    for k in READS:
        arrays[f"echo_{k}_nll"] = nll[k].astype(np.float32)
        arrays[f"echo_{k}_em"] = em[k]
        if diagnostics:
            arrays[f"echo_{k}_first_nll"] = first_nll[k].astype(np.float32)
            arrays[f"echo_{k}_rest_nll"] = rest_nll[k].astype(np.float32)
    if candidate is not None:
        vals = es.value_tokens[:, :, 0]
        collision = np.zeros(vals.shape, dtype=bool)
        for i in range(vals.shape[1]):
            collision[:, i] = (vals == vals[:, i:i+1]).sum(1) > 1
        for read, cand in candidate.items():
            own = np.take_along_axis(cand, np.arange(cand.shape[1])[None, :, None], axis=2)
            order = (cand > own).sum(2) + 1
            arrays.update({f"echo_{read}_first_token_logprobs": cand,
                           f"echo_{read}_first_token_rank": order.astype(np.int16),
                           f"echo_{read}_first_token_collision": collision})
    if diagnostics:
        for k, v in key_nll.items():
            arrays[f"echo_{k}_key_nll"] = v.astype(np.float32)
        arrays["echo_query_segment"] = np.broadcast_to(np.arange(es.available.shape[1]) <
                                                         es.available.shape[1] // 2,
                                                         es.available.shape).copy()
    return arrays, summarize(arrays)


def summarize(arrays: dict) -> dict:
    """Per-stratum means: recall (exact match) and value NLL at the reads; source NLL per stratum."""
    av = np.asarray(arrays["echo_available"], dtype=bool)

    def mean(x, m=None):
        x = np.asarray(x, dtype=np.float64)
        x = x.ravel() if m is None else x[m]
        return float(x.mean()) if x.size else None

    out = {"n_docs": int(av.shape[0]), "recall": {}, "query_nll": {}, "source_nll": {}}
    for stratum, read, mask in (("copy_available", "query", av), ("copy_blocked", "query", ~av),
                                ("no_binding", "no_binding", None), ("no_barrier", "no_barrier", None),
                                ("query_entity", "query_entity", None)):
        out["recall"][stratum] = mean(arrays[f"echo_{read}_em"], mask)
        out["query_nll"][stratum] = mean(arrays[f"echo_{read}_nll"], mask)
    out["source_nll"]["copy_available"] = mean(arrays["echo_source_nll"], av)
    out["source_nll"]["copy_blocked"] = mean(arrays["echo_source_nll"], ~av)
    if "echo_query_first_nll" in arrays:
        out["first_nll"] = {k: mean(arrays[f"echo_{k}_first_nll"]) for k in READS}
        out["rest_nll"] = {k: mean(arrays[f"echo_{k}_rest_nll"]) for k in READS}
    if "echo_query_first_token_rank" in arrays:
        out["first_token_discrimination"] = {}
        for read in ("query", "no_binding"):
            valid = ~np.asarray(arrays[f"echo_{read}_first_token_collision"], dtype=bool)
            rank = np.asarray(arrays[f"echo_{read}_first_token_rank"])
            out["first_token_discrimination"][read] = {
                "accuracy": mean(rank == 1, valid), "mean_rank": mean(rank, valid),
                "excluded_fraction": float((~valid).mean())}
        rank = np.asarray(arrays["echo_query_first_token_rank"])
        valid = ~np.asarray(arrays["echo_query_first_token_collision"], dtype=bool)
        out["first_token_discrimination"].update({
            "copy_available": {"accuracy": mean(rank == 1, valid & av),
                "mean_rank": mean(rank, valid & av), "excluded_fraction": float((~valid & av).sum() / max(1, av.sum()))},
            "copy_blocked": {"accuracy": mean(rank == 1, valid & ~av),
                "mean_rank": mean(rank, valid & ~av), "excluded_fraction": float((~valid & ~av).sum() / max(1, (~av).sum()))}})
    return out


__all__ = ["READS", "STRATA", "make_train_injector", "compact_manifest", "manifest_id", "selected_mask",
           "tokens_per_injection", "exposure", "check_exposure", "EchoEvalSet",
           "build_echo_eval_set", "build_train_echo_eval_set", "echo_read", "summarize"]
