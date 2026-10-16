"""SOQ-P1-v3 trainer adapter and registered read measures.

The generator itself is in :mod:`echo_text`.  This module leaves the SOQ-P1 adapter in
:mod:`soq_p1` as is and uses explicit N/A/P forward labels so an eval npz is self-describing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

READS = ("source", "cross_verbatim", "lookup_verbatim", "cross_paraphrase")
FORWARDS = ("N", "A", "P")


@dataclass
class EchoEvalSetV3:
    docs: np.ndarray
    docs_p: np.ndarray
    available: np.ndarray
    positions: dict
    positions_p: dict
    row_ids: np.ndarray
    docs_sha256: str
    docs_p_sha256: str
    manifest: dict
    write_masks: dict
    segment: dict
    strata: dict
    value_tokens: np.ndarray
    key_positions: dict | None = None
    query_mode: str = "verbatim"
    read_order: dict | None = None
    slot_index: dict | None = None


def _compact_manifest(inj):
    from .soq_p1 import compact_manifest
    return compact_manifest(inj)


def make_train_injector(cfg, excluded):
    from .echo_text import make_injector_v3
    return make_injector_v3(split="train", f=cfg.echo_f, p=cfg.echo_p,
                            lookup_share=cfg.echo_lookup_share, seed=cfg.echo_seed,
                            T=cfg.seq_len, S=cfg.segment_len, n_slots=cfg.echo_slots,
                            excluded_vocab=np.asarray(excluded)) if cfg.echo_f > 0 else None


def compact_manifest(inj):
    return _compact_manifest(inj)


def manifest_id(manifest):
    return manifest.get("manifest_sha256") or manifest["params_sha256"]


def selected_mask(inj, n_rows):
    return np.asarray(inj.selected(np.arange(n_rows, dtype=np.int64)), dtype=bool)


def exposure(inj, n_rows, row_len, start_row=0):
    """Ledger over the stream rows ``[start_row, n_rows)`` this run trains on (a continuation
    starts at ``start * global_batch``)."""
    ids = np.arange(start_row, n_rows, dtype=np.int64)
    ledger = inj.exposure_accounting(ids)
    ledger["rows"] = [int(start_row), int(n_rows)]
    ledger["tokens_per_injected_row"] = ledger["injected_token_budget"] / max(1, ledger["injected_rows"])
    ledger["token_share"] = ledger["injected_token_budget"] / float(max(1, len(ids) * (row_len - 1)))
    return ledger


def check_exposure(ledger, *, min_per_stratum=1000, max_token_share=.05):
    for key, val in ledger.get("source_writes", {}).items():
        if 0 < val < min_per_stratum:
            raise ValueError(f"echo exposure: {key} has {val} source writes < {min_per_stratum}")
    if ledger.get("token_share", 0) > max_token_share:
        raise ValueError("echo exposure token share exceeds limit")


def _from_records(docs, docs_p, recs, recs_p, row_ids, manifest, masks, *, include_metadata=False):
    n = len(recs); slots = len(recs[0]["available"])
    avail = np.asarray([r["available"] for r in recs], dtype=bool)
    pos = {"source": np.asarray([r["source_nll_positions"] for r in recs], dtype=np.int64)}
    pos_p = {"source": np.asarray([r["source_nll_positions"] for r in recs_p], dtype=np.int64)}
    seg = {}; strata = {}; order = {}; slots_idx = {}
    slot_idx = np.broadcast_to(np.arange(slots, dtype=np.int16), (n, slots)).copy()
    for key in ("cross_verbatim", "lookup_verbatim"):
        if key in recs[0]["read_nll_positions"]:
            pos[key] = np.asarray([r["read_nll_positions"][key] for r in recs], dtype=np.int64)
            seg[key] = np.asarray([r["read_segments"][key] for r in recs], dtype=np.int8)
            strata[key] = np.asarray([r["read_strata"][key] for r in recs])
            if include_metadata:
                starts = np.asarray([r["read_value_positions"][key] for r in recs], dtype=np.int64)[:, :, 0]
                order[key] = np.empty_like(starts, dtype=np.int16)
                for i in range(n):
                    for s in np.unique(seg[key][i]):
                        idx = np.flatnonzero(seg[key][i] == s)
                        order[key][i, idx[np.argsort(starts[i, idx])]] = np.arange(len(idx), dtype=np.int16)
                slots_idx[key] = slot_idx.copy()
    if "cross_paraphrase" in recs_p[0]["read_nll_positions"]:
        pos_p["cross_paraphrase"] = np.asarray([r["read_nll_positions"]["cross_paraphrase"] for r in recs_p], dtype=np.int64)
        seg["cross_paraphrase"] = np.asarray([r["read_segments"]["cross_paraphrase"] for r in recs_p], dtype=np.int8)
        strata["cross_paraphrase"] = np.asarray([r["read_strata"]["cross_paraphrase"] for r in recs_p])
        if include_metadata:
            starts = np.asarray([r["read_value_positions"]["cross_paraphrase"] for r in recs_p], dtype=np.int64)[:, :, 0]
            order["cross_paraphrase"] = np.empty_like(starts, dtype=np.int16)
            for i in range(n):
                for s in np.unique(seg["cross_paraphrase"][i]):
                    idx = np.flatnonzero(seg["cross_paraphrase"][i] == s)
                    order["cross_paraphrase"][i, idx[np.argsort(starts[i, idx])]] = np.arange(len(idx), dtype=np.int16)
            slots_idx["cross_paraphrase"] = slot_idx.copy()
    if include_metadata:
        # Source bindings live in segment 1 and are in binding slot order.
        source_seg = np.ones((n, slots), dtype=np.int8)
        source_order = np.broadcast_to(np.arange(slots, dtype=np.int16), (n, slots)).copy()
        seg["source"] = source_seg; order["source"] = source_order; slots_idx["source"] = slot_idx.copy()
    else:
        order = {}; slots_idx = {}
    return EchoEvalSetV3(
        docs=np.asarray(docs, dtype=np.uint32), docs_p=np.asarray(docs_p, dtype=np.uint32),
        available=avail, positions=pos, positions_p=pos_p, row_ids=np.asarray(row_ids, dtype=np.int64),
        docs_sha256=hashlib.sha256(np.ascontiguousarray(docs).tobytes()).hexdigest(),
        docs_p_sha256=hashlib.sha256(np.ascontiguousarray(docs_p).tobytes()).hexdigest(),
        manifest=manifest, write_masks=masks, segment=seg, strata=strata,
        value_tokens=np.asarray([r["value_tokens"] for r in recs], dtype=np.int64),
        key_positions=None, read_order=order, slot_index=slots_idx)


def build_echo_eval_set(cfg, excluded, *, split="test", controls=True, kind="memory",
                        forward_masks=False):
    from .data import open_split
    from .echo_text import make_injector_v3, write_mask_v3
    arr, _ = open_split(cfg.data_dir, cfg.echo_eval_split)
    lo, n = int(cfg.echo_eval_offset), int(cfg.echo_eval_docs)
    if lo < 0 or n < 1 or lo + n > len(arr):
        raise ValueError(f"v3 eval needs rows [{lo},{lo+n}) of {cfg.echo_eval_split}")
    hosts = np.asarray(arr[lo:lo+n, :cfg.seq_len+1])
    ids = np.arange(lo, lo+n, dtype=np.int64)
    # The default arguments are the registered TEST builder, byte-for-byte
    # equivalent to the SOQ-P1-v3 production read; post-hoc diagnostics use
    # the optional TRAIN split and controls=False below.
    inj = make_injector_v3(split=split, f=1.0, p=cfg.echo_eval_p,
                           lookup_share=cfg.echo_lookup_share, seed=cfg.echo_seed,
                           T=cfg.seq_len, S=cfg.segment_len, n_slots=cfg.echo_slots,
                           excluded_vocab=np.asarray(excluded))
    if split == "test" and controls and kind == "memory":
        docs, meta = inj.eval_documents(hosts, row_ids=ids, query_mode="verbatim")
        docs_p, meta_p = inj.eval_documents(hosts, row_ids=ids, query_mode="paraphrase")
    else:
        docs = np.empty_like(hosts); docs_p = np.empty_like(hosts); recs=[]; recs_p=[]
        for j, rid in enumerate(ids):
            docs[j], rec = inj.document(hosts[j], int(rid), controls=controls, kind=kind, query_mode="verbatim")
            docs_p[j], rec_p = inj.document(hosts[j], int(rid), controls=controls, kind=kind, query_mode="paraphrase")
            recs.append(rec); recs_p.append(rec_p)
        meta = {"rows": recs}; meta_p = {"rows": recs_p}
    recs, recs_p = meta["rows"], meta_p["rows"]
    masks = {"N": write_mask_v3(recs, cfg.seq_len, forward="N", segment_len=cfg.segment_len),
             "A": write_mask_v3(recs, cfg.seq_len, forward="A", segment_len=cfg.segment_len),
             "P": write_mask_v3(recs_p, cfg.seq_len, forward="P", segment_len=cfg.segment_len)}
    if forward_masks:
        masks["Z"] = write_mask_v3(recs, cfg.seq_len, forward="Z", segment_len=cfg.segment_len)
        masks["P-A"] = write_mask_v3(recs_p, cfg.seq_len, forward="P-A", segment_len=cfg.segment_len)
    if not controls and forward_masks:
        masks["T"] = np.ones_like(masks["N"], dtype=bool)
        masks["T-A"] = write_mask_v3(recs, cfg.seq_len, forward="T-A", segment_len=cfg.segment_len)
    return _from_records(docs, docs_p, recs, recs_p, ids, _compact_manifest(inj), masks,
                         include_metadata=forward_masks)


def build_echo_train_templates(cfg, excluded, *, controls=True):
    """Build paired TRAIN-pool memory documents on the TEST host rows.

    ``controls=True`` is the eval layout (N/A/Z); ``False`` is the exact training
    memory layout, with no lookup block and a fully writable T forward.
    """
    return build_echo_eval_set(cfg, excluded, split="train", controls=controls,
                               kind="memory", forward_masks=True)


def build_train_echo_eval_set(cfg, excluded):
    # Retained as a compatibility helper for callers that construct diagnostics sets.
    return build_echo_eval_set(cfg, excluded)


def _positions_for(es, forward):
    if forward == "P":
        return es.positions_p
    return es.positions


def within_stratum_rank(cand_logp: np.ndarray, available: np.ndarray):
    """Rank each slot among the candidates in its own availability stratum."""
    c = np.asarray(cand_logp)
    av = np.asarray(available, dtype=bool)
    if c.ndim != 3 or c.shape[:2] != av.shape or c.shape[1] != c.shape[2]:
        raise ValueError("cand_logp must be [N, slots, slots] and match available")
    own = np.take_along_axis(c, np.arange(c.shape[1])[None, :, None], axis=2)
    same = av[:, :, None] == av[:, None, :]
    rank = 1 + ((c > own) & same).sum(axis=2)
    return rank.astype(np.int16), rank == 1


def _read_forward(model, docs, positions, write_mask, reads, es, *, rule, inner_lr, inner_clip,
                  barriers, device, fused_attn, batch, r_ref, data_seed):
    import torch
    from .meta import InnerCfg, stream_eval
    from .probe import boundary_valid

    N, T1 = docs.shape; T = T1 - 1; slots = es.available.shape[1]
    nll = {r: np.full((N, slots), np.nan, dtype=np.float32) for r in reads}
    em = {r: np.zeros((N, slots), dtype=bool) for r in reads}
    first = {r: np.full((N, slots), np.nan, dtype=np.float32) for r in reads}
    rank = {r: np.full((N, slots), np.nan, dtype=np.float32) for r in reads}
    top1 = {r: np.zeros((N, slots), dtype=bool) for r in reads}
    greedy_pos = {r: np.zeros((N, slots, 3), dtype=bool) for r in reads}
    tok23 = {r: np.full((N, slots), np.nan, dtype=np.float32) for r in reads}
    within_rank = {r: np.full((N, slots), np.nan, dtype=np.float32) for r in reads}
    within_top1 = {r: np.zeros((N, slots), dtype=bool) for r in reads}
    # first-token log-probs of all 16 bound values at each read (v3x: within-stratum ranks post hoc)
    cand = {r: np.full((N, slots, slots), np.nan, dtype=np.float32) for r in reads}
    bars = list(barriers)
    bvalid = torch.from_numpy(boundary_valid(T, model.cfg.chunk, bars)).to(device)
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch)); B = sl.stop - sl.start
        arr = torch.from_numpy(docs[sl].astype(np.int64)).to(device)
        ids, tgt = arr[:, :-1], arr[:, 1:]; valid = (tgt != model.cfg.bos_id) & bvalid
        selected = torch.zeros(B, T, dtype=torch.bool, device=device)
        qnames = [r for r in reads]
        for name in qnames:
            selected.scatter_(1, torch.from_numpy(positions[name][sl].reshape(B, -1)).to(device), True)
        cand_pos = np.concatenate([positions[r][sl, :, 0] for r in qnames], axis=1)
        cand_ids = np.broadcast_to(es.value_tokens[sl, :, 0][:, None, :], (B, len(qnames)*slots, slots)).copy()
        inner = InnerCfg(rule=rule, lr=inner_lr, clip=inner_clip, r_ref=r_ref, seed=data_seed,
                         row_ids=torch.arange(b0, b0+B, device=device))
        rr = stream_eval(model, ids, tgt, valid, inner, barriers=bars,
                         nll_positions=selected, inner_mask=torch.from_numpy(write_mask[sl]).to(device),
                         fused_attn=fused_attn, return_greedy=True,
                         candidate_positions=torch.from_numpy(cand_pos.astype(np.int64)).to(device),
                         candidate_ids=torch.from_numpy(cand_ids.astype(np.int64)).to(device))
        tn, tg = rr["nll"].cpu().numpy(), rr["greedy"].cpu().numpy()
        cp = rr["candidate_log_probs"].cpu().numpy()
        rows = np.arange(B)[:, None, None]
        for j, name in enumerate(qnames):
            q = positions[name][sl]; n = q.shape[1]
            nll[name][sl] = tn[rows, q].mean(-1)
            em[name][sl] = tg[rows, q].all(-1)
            greedy_pos[name][sl] = tg[rows, q]
            first[name][sl] = tn[rows, q[:, :, :1]].squeeze(-1)
            tok23[name][sl] = tn[rows, q[:, :, 1:]].mean(-1)
            c = cp[:, j*n:(j+1)*n]
            cand[name][sl] = c
            own = np.take_along_axis(c, np.arange(n)[None, :, None], axis=2)
            rank[name][sl] = (1 + (c > own).sum(2)).astype(np.float32)
            top1[name][sl] = rank[name][sl] == 1
            wr, wt = within_stratum_rank(c, es.available[sl])
            within_rank[name][sl] = wr.astype(np.float32)
            within_top1[name][sl] = wt
    return {"nll": nll, "em": em, "greedy_pos": greedy_pos, "first_nll": first,
            "tokens23_nll": tok23, "rank": rank, "top1": top1, "cand": cand,
            "within_rank": within_rank, "within_top1": within_top1}


def _summary(es, results, *, extended=False):
    out = {"version": 3, "n_docs": int(len(es.available)), "forwards": {}}
    for fwd, rr in results.items():
        out["forwards"][fwd] = {}
        for read in rr["em"]:
            out["forwards"][fwd][read] = {}
            masks = {"pooled": np.ones_like(es.available, dtype=bool),
                     "copy_available": es.available, "copy_blocked": ~es.available}
            seg = es.segment.get(read)
            if seg is not None:
                masks.update({f"segment{int(s)}": seg == s for s in np.unique(seg)})
            for label, mask in masks.items():
                def mean(a):
                    x = np.asarray(a)[mask]; return float(x.mean()) if x.size else None
                row = {
                    "R": mean(rr["em"][read]), "value_nll": mean(rr["nll"][read]),
                    "first_nll": mean(rr["first_nll"][read]),
                    "first_rank_16": mean(rr["rank"][read]), "top1_16": mean(rr["top1"][read]),
                }
                if extended:
                    row.update({"tokens23_nll": mean(rr["tokens23_nll"][read]),
                                "first_rank_8": mean(rr["within_rank"][read]),
                                "top1_8": mean(rr["within_top1"][read])})
                out["forwards"][fwd][read][label] = row
    return out


def echo_read(model, es: EchoEvalSetV3, *, rule, inner_lr, inner_clip, barriers, device,
              fused_attn, batch, r_ref=.03, data_seed=0, step=None, include_p=False):
    """Run N and A, plus P only when ``include_p`` is true (registered final step)."""
    results = {}
    nreads = tuple(r for r in ("source", "cross_verbatim", "lookup_verbatim") if r in es.positions)
    # Source is evaluated in both forwards so the arrays are easy to align; the registered
    # decisions use only the corresponding cross/lookup read.
    results["N"] = _read_forward(model, es.docs, es.positions, es.write_masks["N"], nreads, es,
                                  rule=rule, inner_lr=inner_lr, inner_clip=inner_clip, barriers=barriers,
                                  device=device, fused_attn=fused_attn, batch=batch, r_ref=r_ref, data_seed=data_seed)
    results["A"] = _read_forward(model, es.docs, es.positions, es.write_masks["A"], nreads, es,
                                  rule=rule, inner_lr=inner_lr, inner_clip=inner_clip, barriers=barriers,
                                  device=device, fused_attn=fused_attn, batch=batch, r_ref=r_ref, data_seed=data_seed)
    if include_p:
        preads = tuple(r for r in ("source", "cross_paraphrase") if r in es.positions_p)
        results["P"] = _read_forward(model, es.docs_p, es.positions_p, es.write_masks["P"], preads, es,
                                      rule=rule, inner_lr=inner_lr, inner_clip=inner_clip, barriers=barriers,
                                      device=device, fused_attn=fused_attn, batch=batch, r_ref=r_ref, data_seed=data_seed)
    arrays = {"echo_version": np.asarray(3), "echo_available": es.available, "echo_row_ids": es.row_ids,
              "echo_docs_sha256": np.asarray(es.docs_sha256), "echo_docs_p_sha256": np.asarray(es.docs_p_sha256)}
    for fwd, rr in results.items():
        for read in rr["em"]:
            prefix = f"echo_{fwd}_{read}"
            arrays[f"{prefix}_nll"] = rr["nll"][read]; arrays[f"{prefix}_value_nll"] = rr["nll"][read]
            arrays[f"{prefix}_em"] = rr["em"][read]; arrays[f"{prefix}_R"] = rr["em"][read]
            arrays[f"{prefix}_first_nll"] = rr["first_nll"][read]
            arrays[f"{prefix}_first_token_rank"] = rr["rank"][read].astype(np.int16)
            arrays[f"{prefix}_top1_16"] = rr["top1"][read]
            arrays[f"{prefix}_cand_logp"] = rr["cand"][read]
            if read in es.segment: arrays[f"{prefix}_segment"] = es.segment[read]
            arrays[f"{prefix}_stratum"] = np.where(es.available, "copy_available", "copy_blocked")
    summary = _summary(es, results)
    return arrays, summary


__all__ = ["EchoEvalSetV3", "make_train_injector", "build_echo_eval_set", "build_echo_train_templates",
           "build_train_echo_eval_set",
           "echo_read", "within_stratum_rank", "compact_manifest", "manifest_id", "selected_mask", "exposure", "check_exposure"]
