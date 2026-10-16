"""E1-VE-D3: what sets the size of a write. Eval only.

E1-VE-D2 found that, with the copy edge blocked, the C write of
an A location is born at about 0.8 times its E write, while the C write of an
X location in the same models is 11 to 26 times its E write. At one position
``t`` the fast ``W2`` gradient is the outer product ``delta_t a_t^T``, so the
write's size factors exactly:

    ||dW2_t|| = ||g_t|| * (||delta_t|| / ||g_t||) * ||a_t||

* ``g_t``: the gradient of the inner token loss in logit space (``p - e_y`` for
  CE), the prediction error the write corrects;
* ``||delta_t|| / ||g_t||``: the read-out gain, the gradient carried back
  through ``ln_f`` and the unembedding to the fast block's output;
* ``a_t = gelu(W1 x_t + b1)``: the fast block's activation under the streaming
  weights (``a0_t`` under ``W0``).

For every E write (chunk 0) and C write (chunk C) this module records the three
factors, the value probabilities at the write, and the **key response** of a C
input to the same location's E write: a single write updates ``W1`` by a rank-1
term along its input ``x_E``, so it moves the activation at another input ``x``
by ``<x_E, x> / ||x_E||^2`` times what it moves it at ``x_E``.
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

C_TYPES = ("S", "A", "X", "N")     # the C events
PAIRED = ("S", "A", "X")           # C events whose location also has an E write
FACTORS = ("size", "g", "gain", "a", "a0", "G")


@dataclass
class WriteSettings:
    process_name: str = "TTM-vd3-01"
    device: str = "cpu"
    threads: int = 8
    ckpt_dir: str = ""
    reference_csv: str = ""     # wave-1 per-cell finals (gate G0)
    n_eval: int = 256
    eval_batch: int = 32
    ve: dict = field(default_factory=dict)
    cells: list = field(default_factory=list)


def settings(cfg: dict) -> WriteSettings:
    known = {f.name for f in dataclasses.fields(WriteSettings)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown [ve_write] keys: {unknown}")
    return WriteSettings(**cfg)


def chunk_factors(model: tt.TTTModel, rule: str, h: torch.Tensor, xin: torch.Tensor,
                  y: torch.Tensor, sel, reference, W, W0, sl: slice) -> dict[str, torch.Tensor]:
    """Per-position write factors in one chunk under the arrival weights ``W``.

    Returns (B, chunk) tensors: ``g``, ``delta``, ``a``, ``a0``, and the softmax
    ``p`` (B, chunk, V). ``logits`` equals the stream's pre-update logits.
    """
    W1, b1, W2, b2 = W
    with torch.enable_grad():
        u = torch.einsum("btd,bmd->btm", xin[:, sl], W1) + b1[:, None, :]
        a = F.gelu(u)
        o = h[:, sl] + torch.einsum("btm,bdm->btd", a.detach(), W2) + b2[:, None, :]
        o = o.detach().requires_grad_(True)
        z = model.read(o)
        inner = tt.inner_token_loss(rule, z, y[:, sl],
                                    sel=None if sel is None else sel[:, sl],
                                    reference=None if reference is None else reference[:, sl])
        g, delta = torch.autograd.grad(inner.sum(), [z, o])
    with torch.no_grad():
        a0 = F.gelu(torch.einsum("btd,bmd->btm", xin[:, sl], W0[0]) + W0[1][:, None, :])
    return dict(g=g.norm(dim=-1), delta=delta.norm(dim=-1), a=a.detach().norm(dim=-1),
                a0=a0.norm(dim=-1), p=z.detach().softmax(-1), logits=z.detach(),
                x=xin[:, sl])


def e_types(meta: dict, idx: torch.Tensor) -> torch.Tensor:
    """Type of every E write (B, n_e): the C type of the same location, U if none."""
    loc_e = idx.gather(1, meta["e_write"])
    loc_c = idx.gather(1, meta["c_write"])
    eq = loc_e[:, :, None] == loc_c[:, None, :]
    j = eq.float().argmax(-1)
    return torch.where(eq.any(-1), meta["c_type"].gather(1, j), torch.full_like(j, ve.U))


def write_stream(model: tt.TTTModel, lay: ve.Layout, vst: ve.VeSettings, cell: dict,
                 tok: torch.Tensor, meta: dict) -> dict:
    """Factors at every E and C write of one batch, and the per-type summaries."""
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
    lo = cc * lay.chunk
    fac = {c: chunk_factors(model, rule, h, xin, y, sel, reference, arrive[c], arrive[0],
                            slice(c * lay.chunk, (c + 1) * lay.chunk)) for c in (0, cc)}
    out["logit_maxdiff"] = max(float((fac[c]["logits"] - logits[:, c * lay.chunk:(c + 1) * lay.chunk])
                                     .abs().max()) for c in (0, cc))

    vals = slice(lay.val0, lay.val0 + lay.n_val)

    def at(c: int, pos: torch.Tensor) -> dict[str, torch.Tensor]:
        f = fac[c]
        loc = pos - (lo if c == cc else 0)
        r = {k: f[k].gather(1, loc) for k in ("g", "delta", "a", "a0")}
        p = f["p"].gather(1, loc[..., None].expand(-1, -1, f["p"].shape[-1]))
        r["p_target"] = p.gather(-1, y.gather(1, pos)[..., None]).squeeze(-1)
        r["p_maxval"] = p[..., vals].max(-1).values
        r["p"] = p
        r["x"] = f["x"].gather(1, loc[..., None].expand(-1, -1, f["x"].shape[-1]))
        r["size"] = r["delta"] * r["a"]
        r["gain"] = r["delta"] / r["g"].clamp_min(1e-12)
        r["G"] = r["gain"] * r["a"]
        return r

    e = at(0, meta["e_write"])
    c = at(cc, meta["c_write"])
    etype = e_types(meta, idx)
    old = meta["c_old"].clamp_min(0)
    c["p_old"] = c["p"].gather(-1, old[..., None]).squeeze(-1)

    # per-sequence reference: the mean of each factor over that sequence's E writes
    e_mean = {k: e[k].mean(1, keepdim=True) for k in FACTORS}
    # pairs: the E write of each C event's location
    loc_e, loc_c = idx.gather(1, meta["e_write"]), idx.gather(1, meta["c_write"])
    eq = loc_c[:, :, None] == loc_e[:, None, :]                       # (B, n_c, n_e)
    has = eq.any(-1)
    j = eq.float().argmax(-1)
    paired = {k: e[k].gather(1, j) for k in FACTORS}
    xe = e["x"]
    xe_sq = (xe * xe).sum(-1)                                         # (B, n_e)
    dots = torch.einsum("bcd,bed->bce", c["x"], xe) / xe_sq[:, None, :]
    kappa_own = dots.gather(2, j[..., None]).squeeze(-1)
    n_other = (~eq).sum(-1).clamp_min(1)
    kappa_other = (dots * (~eq)).sum(-1) / n_other

    for ti, t in enumerate(ve.TYPES):
        m_e = etype == ti
        m_c = meta["c_type"] == ti
        for k in FACTORS + ("p_target", "p_maxval"):
            if bool(m_e.any()):
                out[f"E_{t}.{k}"] = e[k][m_e].tolist()
            if bool(m_c.any()):
                out[f"C_{t}.{k}"] = c[k][m_c].tolist()
        if not bool(m_c.any()):
            continue
        if t in ("A", "X", "S"):
            out[f"C_{t}.p_old"] = c["p_old"][m_c].tolist()
        for k in FACTORS:
            out[f"C_{t}.{k}_over_Emean"] = (c[k] / e_mean[k])[m_c].tolist()
        if t in PAIRED:
            mp = m_c & has
            for k in FACTORS:
                out[f"C_{t}.{k}_over_Eown"] = (c[k] / paired[k])[mp].tolist()
            out[f"C_{t}.kappa_own"] = kappa_own[mp].tolist()
            out[f"C_{t}.kappa_other"] = kappa_other[mp].tolist()
    return out


def summarize(batches: list[dict]) -> dict:
    out: dict[str, Any] = {}
    qs = [b["q"] for b in batches]
    tot = {k: sum(q[k] for q in qs) for k in qs[0]}
    for t in ve.TYPES:
        out[f"base.acc_{t}"] = tot[f"hit_{t}"] / (tot[f"n_{t}"] or 1.0)
    out["logit_maxdiff"] = max(b["logit_maxdiff"] for b in batches)
    keys = sorted({k for b in batches for k in b if "." in k and k != "q"})
    for k in keys:
        vals = sum((b.get(k, []) for b in batches), [])
        out[k] = float(np.median(vals)) if vals else None
        if "_over_" in k and vals:
            # geometric mean: log size = log g + log gain + log a holds exactly for it
            out[k + ".gm"] = float(np.exp(np.mean(np.log(np.clip(vals, 1e-30, None)))))
    return out


def run_cell(cell: dict, st: WriteSettings, vst: ve.VeSettings, lay: ve.Layout,
             dev: torch.device) -> dict:
    t0 = time.time()
    model = dg.load_model(vst, lay, cell, st.ckpt_dir, dev)
    data = ve.fixed_set(lay, st.n_eval, ve.EVAL_SEED, dev, st.eval_batch)
    rec: dict[str, Any] = dict(cell_id=ve.cell_id(cell), **cell)
    rec.update(summarize([write_stream(model, lay, vst, cell, tok, meta) for tok, meta in data]))
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------- scoring


def score(recs: list[dict]) -> dict:
    """Registered V1, V2 and N0 over cells that pass G0 and U, S >= .90.

    V1 and N0 (per eta, lesion CE cells): ``C_A.kappa_own`` is within 10 % of
    ``C_X.kappa_own``; ``C_N.G_over_Emean`` lies in [1/1.5, 1.5]. V2 (all cells):
    Spearman rho of ``p_maxval`` vs log ``G_over_Emean`` over the C types; S at
    rho >= .5, F at rho <= .2, NE below 8 points.
    """
    out: dict[str, Any] = {}
    ok = [r for r in recs if r.get("g0") is not False
          and r["base.acc_U"] >= .90 and r["base.acc_S"] >= .90]
    for eta in sorted({float(r["eta"]) for r in recs}):
        les = [r for r in ok if r["rule"] == "ce" and r.get("edge") and float(r["eta"]) == eta]
        k = len(les)
        tag = f"eta{eta:g}"
        v1 = sum(abs(r["C_A.kappa_own"] - r["C_X.kappa_own"]) <= .1 * abs(r["C_X.kappa_own"])
                 for r in les)
        n0 = sum(1 / 1.5 <= r["C_N.G_over_Emean"] <= 1.5 for r in les)
        out[f"V1_{tag}"] = dict(verdict=dg.count_verdict(v1, k), n=v1, k=k)
        out[f"N0_{tag}"] = dict(verdict=dg.count_verdict(n0, k), n=n0, k=k)
    xs, ys = [], []
    for r in ok:
        for t in C_TYPES:
            g, p = r.get(f"C_{t}.G_over_Emean"), r.get(f"C_{t}.p_maxval")
            if g is not None and p is not None and g > 0:
                xs.append(p)
                ys.append(math.log(g))
    rho = vb.spearman(xs, ys) if len(xs) >= 8 else float("nan")
    out["V2"] = dict(verdict="NE" if len(xs) < 8 else
                     ("S" if rho >= .5 else "F" if rho <= .2 else "I"), rho=rho, k=len(xs))
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
    print("VE_WRITE_CONFIG " + json.dumps(dict(dataclasses.asdict(st), cells=len(st.cells),
                                              device_resolved=str(dev))), flush=True)
    recs = []
    for i, cell in enumerate(st.cells):
        rec = run_cell(cell, st, vst, lay, dev)
        rec["g0"], rec["g0_maxdiff"] = dg.gate_g0(rec, ref)
        recs.append(rec)
        print("CELL_WRITE " + json.dumps(rec), flush=True)
        print(f"  [{i + 1}/{len(st.cells)}] {rec['cell_id']} g0={rec['g0']} "
              f"logit_diff={rec['logit_maxdiff']:.1e} {rec['wall_s']}s", flush=True)
    bad = [r["cell_id"] for r in recs if r["g0"] is False]
    print(f"G0 reproduction: {len(recs) - len(bad)}/{len(recs)} within {dg.G0_TOL}; "
          f"failing: {bad or 'none'}", flush=True)
    print(f"logit max diff {max(r['logit_maxdiff'] for r in recs):.2e}", flush=True)
    res = score(recs)
    for key, val in res.items():
        print(f"{key}: {json.dumps(val)}", flush=True)
    print("VE_WRITE_SUMMARY " + json.dumps(dict(cells=len(recs), g0_fail=bad, scores=res)),
          flush=True)
    return dict(cells=len(recs), g0_fail=len(bad), scores=res)
