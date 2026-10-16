"""E1-VE-D4: what sets the read-out gain of a write. Eval only.

E1-VE-D3 found that a write's size differs between chance-level
writes (E writes; A and N writes without a copy path) and confident writes
(retrieved X and S, copied A and N) almost only through the read-out gain
``||delta|| / ||g||``. The read-out is ``z = U ln_f(o)``, so with
``o_hat = (o - mean) / sigma`` and ``P = I - 11^T/d - o_hat o_hat^T/d``

    delta = (1 / sigma) P (gamma * U^T g)

and the gain splits exactly into four factors:

    ||delta|| / ||g|| = ||U^T g|| / ||g||                      (unembedding pull-back)
                      * ||gamma * U^T g|| / ||U^T g||          (ln_f gain)
                      * ||P w|| / ||w||, w = gamma * U^T g     (LN projection retention)
                      * 1 / sigma(o)                           (LN scale)

For every E write and C write this module records the four factors, ``sigma``
of the backbone output ``h`` and of the fast block's added output, and the
cosine of ``w`` with ``o_hat`` (the part the projection removes).
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import exp_ttt as tt
from . import exp_ve as ve
from . import exp_ve_budget as vb
from . import exp_ve_diag as dg
from . import exp_ve_write as vw

FACTORS = ("gain", "fU", "fgamma", "fP", "fsigma")
EXTRA = ("sigma_o", "sigma_h", "sigma_fast", "cos_w_ohat", "mean_frac", "p_maxval")
LN_PART = ("fgamma", "fP", "fsigma")


@dataclass
class GainSettings:
    process_name: str = "TTM-vd4-01"
    device: str = "cpu"
    threads: int = 8
    ckpt_dir: str = ""
    reference_csv: str = ""
    n_eval: int = 256
    eval_batch: int = 32
    ve: dict = field(default_factory=dict)
    cells: list = field(default_factory=list)


def settings(cfg: dict) -> GainSettings:
    known = {f.name for f in dataclasses.fields(GainSettings)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown [ve_gain] keys: {unknown}")
    return GainSettings(**cfg)


def _sigma(x: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.var(-1, unbiased=False) + eps).sqrt()


def gain_factors(model: tt.TTTModel, o: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
    """The four factors of ``||delta|| / ||g||`` at outputs ``o`` (..., d) with logit gradients ``g``."""
    ln = model.ln_f
    d = o.shape[-1]
    mu = o.mean(-1, keepdim=True)
    sigma = _sigma(o, ln.eps)[..., None]
    o_hat = (o - mu) / sigma
    v = g @ model.unembed.weight                      # U^T g
    w = v * ln.weight
    mean_part = w.mean(-1, keepdim=True)
    self_part = o_hat * (o_hat * w).sum(-1, keepdim=True) / d
    pw = w - mean_part - self_part
    delta = pw / sigma
    gn, vn, wn, pn = (t.norm(dim=-1) for t in (g, v, w, pw))
    return dict(gain=delta.norm(dim=-1) / gn.clamp_min(1e-12), fU=vn / gn.clamp_min(1e-12),
                fgamma=wn / vn.clamp_min(1e-12), fP=pn / wn.clamp_min(1e-12),
                fsigma=1 / sigma.squeeze(-1), sigma_o=sigma.squeeze(-1),
                cos_w_ohat=F.cosine_similarity(w, o_hat, dim=-1),
                mean_frac=(mean_part.squeeze(-1).abs() * math.sqrt(d)) / wn.clamp_min(1e-12),
                delta=delta)


def gain_stream(model: tt.TTTModel, lay: ve.Layout, vst: ve.VeSettings, cell: dict,
                tok: torch.Tensor, meta: dict) -> dict:
    idx, y, sel, h, etas = vb._setup(model, lay, cell, tok, meta)
    rule = ve.RULES[cell["rule"]]
    cc = dg.c_chunk(lay)
    arrive: dict[int, list[torch.Tensor]] = {}

    def observer(c, logits, reference, W, raw_grads, grads, norm, clipped):
        if c in (0, cc):
            arrive[c] = [w.detach() for w in W]

    logits = tt.eval_sequential(model, idx, y, etas, vst.g_clip, lay.chunk, tt.GradStats(), sel,
                                h=h, rule=rule, observer=observer)
    zq = ve._gather_last(logits, meta["query_pos"])
    out: dict[str, Any] = dict(q=dg._queries(zq, meta))
    x = model.fast_input(h, idx) if model.frozen_emb is not None else None
    xin = (model.head.ln(h) if x is None else x).detach()
    reference = tt.reference_logits(model, h, idx, arrive[0]) if rule == "aice" else None
    vals = slice(lay.val0, lay.val0 + lay.n_val)
    lo = cc * lay.chunk
    recon = 0.0
    per: dict[str, dict[str, torch.Tensor]] = {}
    for c, pos, tag in ((0, meta["e_write"], "E"), (cc, meta["c_write"], "C")):
        sl = slice(c * lay.chunk, (c + 1) * lay.chunk)
        W1, b1, W2, b2 = arrive[c]
        with torch.enable_grad():
            a = F.gelu(torch.einsum("btd,bmd->btm", xin[:, sl], W1) + b1[:, None, :])
            fast = torch.einsum("btm,bdm->btd", a, W2) + b2[:, None, :]
            o = (h[:, sl] + fast).detach().requires_grad_(True)
            z = model.read(o)
            inner = tt.inner_token_loss(rule, z, y[:, sl],
                                        sel=None if sel is None else sel[:, sl],
                                        reference=None if reference is None else reference[:, sl])
            g, delta = torch.autograd.grad(inner.sum(), [z, o])
        loc = pos - (lo if c == cc else 0)
        take = lambda t3: t3.gather(1, loc[..., None].expand(-1, -1, t3.shape[-1]))
        with torch.no_grad():
            f = gain_factors(model, take(o.detach()), take(g))
            recon = max(recon, float((f.pop("delta") - take(delta)).abs().max()))
            f["sigma_h"] = _sigma(take(h[:, sl]), model.ln_f.eps)
            f["sigma_fast"] = _sigma(take(fast.detach()), model.ln_f.eps)
            f["p_maxval"] = take(z.detach().softmax(-1))[..., vals].max(-1).values
        per[tag] = f
    out["delta_recon_maxdiff"] = recon

    etype = vw.e_types(meta, idx)
    for tag, types in (("E", etype), ("C", meta["c_type"])):
        for ti, t in enumerate(ve.TYPES):
            m = types == ti
            if not bool(m.any()):
                continue
            for k in FACTORS + EXTRA:
                out[f"{tag}_{t}.{k}"] = per[tag][k][m].tolist()
    return out


def _gm(vals: list[float]) -> float | None:
    return float(np.exp(np.mean(np.log(np.clip(vals, 1e-30, None))))) if vals else None


def summarize(batches: list[dict]) -> dict:
    out: dict[str, Any] = {}
    qs = [b["q"] for b in batches]
    tot = {k: sum(q[k] for q in qs) for k in qs[0]}
    for t in ve.TYPES:
        out[f"base.acc_{t}"] = tot[f"hit_{t}"] / (tot[f"n_{t}"] or 1.0)
    out["delta_recon_maxdiff"] = max(b["delta_recon_maxdiff"] for b in batches)
    keys = sorted({k for b in batches for k in b if "." in k})
    for k in keys:
        vals = sum((b.get(k, []) for b in batches), [])
        if not vals:
            continue
        name = k.split(".", 1)[1]
        # factors multiply exactly, so their geometric means multiply too
        out[k] = _gm(vals) if name in FACTORS or name.startswith("sigma") else float(np.median(vals))
    return out


def run_cell(cell: dict, st: GainSettings, vst: ve.VeSettings, lay: ve.Layout,
             dev: torch.device) -> dict:
    t0 = time.time()
    model = dg.load_model(vst, lay, cell, st.ckpt_dir, dev)
    data = ve.fixed_set(lay, st.n_eval, ve.EVAL_SEED, dev, st.eval_batch)
    rec: dict[str, Any] = dict(cell_id=ve.cell_id(cell), **cell)
    rec.update(summarize([gain_stream(model, lay, vst, cell, tok, meta) for tok, meta in data]))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def ln_share(rec: dict, hi: str, lo: str) -> float | None:
    """Share of log(gain_hi / gain_lo) carried by the three LayerNorm factors."""
    try:
        total = math.log(rec[f"{hi}.gain"] / rec[f"{lo}.gain"])
        ln_part = sum(math.log(rec[f"{hi}.{k}"] / rec[f"{lo}.{k}"]) for k in LN_PART)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return ln_part / total if abs(total) > 1e-9 else None


def score(recs: list[dict]) -> dict:
    """Registered L1: in lesion cells the LN part carries >= half of log(gain X_C / gain A_C)."""
    out: dict[str, Any] = {}
    for eta in sorted({float(r["eta"]) for r in recs}):
        les = [r for r in recs if r["rule"] == "ce" and r.get("edge") and float(r["eta"]) == eta
               and r.get("g0") is not False and r["base.acc_U"] >= .90 and r["base.acc_S"] >= .90]
        shares = [ln_share(r, "C_X", "C_A") for r in les]
        n = sum(s is not None and s >= .5 for s in shares)
        out[f"L1_eta{eta:g}"] = dict(verdict=dg.count_verdict(n, len(les)), n=n, k=len(les),
                                     shares=[None if s is None else round(s, 3) for s in shares])
    return out


def run(cfg: dict, seed: int) -> dict:
    st = settings(cfg)
    tt._mark_process(st.process_name)
    torch.set_num_threads(max(1, int(st.threads)))
    dev = tt._device(st.device)
    tt.configure_tf32(False, st.process_name)
    vst = ve.VeSettings(**{**st.ve, "n_eval": st.n_eval, "eval_batch": st.eval_batch,
                           "device": st.device})
    lay = vst.layout()
    if not st.cells:
        raise ValueError("no cells")
    for cell in st.cells:
        ve.check_cell(cell)
    ref = dg.load_reference(st.reference_csv)
    print("VE_GAIN_CONFIG " + json.dumps(dict(dataclasses.asdict(st), cells=len(st.cells),
                                             device_resolved=str(dev))), flush=True)
    recs = []
    for i, cell in enumerate(st.cells):
        rec = run_cell(cell, st, vst, lay, dev)
        rec["g0"], rec["g0_maxdiff"] = dg.gate_g0(rec, ref)
        recs.append(rec)
        print("CELL_GAIN " + json.dumps(rec), flush=True)
        print(f"  [{i + 1}/{len(st.cells)}] {rec['cell_id']} g0={rec['g0']} "
              f"recon={rec['delta_recon_maxdiff']:.1e} {rec['wall_s']}s", flush=True)
    bad = [r["cell_id"] for r in recs if r["g0"] is False]
    print(f"G0 reproduction: {len(recs) - len(bad)}/{len(recs)} within {dg.G0_TOL}; "
          f"failing: {bad or 'none'}", flush=True)
    print(f"delta reconstruction max diff {max(r['delta_recon_maxdiff'] for r in recs):.2e}",
          flush=True)
    res = score(recs)
    for key, val in res.items():
        print(f"{key}: {json.dumps(val)}", flush=True)
    print("VE_GAIN_SUMMARY " + json.dumps(dict(cells=len(recs), g0_fail=bad, scores=res)),
          flush=True)
    return dict(cells=len(recs), g0_fail=len(bad), scores=res)
