"""Performative correspondence: the FO meta-gradient is the repeated-gradient-descent (RGD)
direction; the SO meta-gradient is the gradient of the performative risk.

Numerical check on a small nonlinear model with the structure of the TTT-E2E echo toy
(``tttmem.exp_ttt``):
  * a static "backbone" h_t = E[x_t] + a*E[x_{t-2}] + tanh(P E[x_{t-1}])   (a = copy-edge gain)
  * fast head  out = h + W2 gelu(W1 LN(h) + b1) + b2 ; logits = U LN_f(out)
  * chunked inner loop W_{c+1} = W_c - eta_i * clip(g_c), eta_i = eta_base*exp(rho_i),
    g_c = grad of chunk-c mean inner loss at W_c, per sequence; clip scale detached (as in the toy).
  * outer loss = mean over chunks of chunk-c CE at W_c.
FO (toy e2e1): g_c = stop_gradient(...); SO (toy e2e2): create_graph, i.e. no stop_gradient.

Objects:
  write map   G_theta(s) = (g_0..g_{C-1}) along the true trajectory at theta
  replay      W_c(theta; gamma) = W0 - eta(theta) * sum_{c'<c} gamma_{c'}
  loss        l((s,gamma); theta) = mean_c CE_c(W_c(theta; gamma); theta, s)
  DR(td, t)   = E_s l((s, G_td(s)); t)          (decoupled performative risk)
  PR(t)       = DR(t, t)
Checks (central finite differences, float64):
  C1  PR(theta) == L(theta)
  C2  FO autodiff gradient == d/dt DR(theta0, t) at t = theta0      (FO = RGD direction)
  C3  SO autodiff gradient == d/dt L(t)                              (SO = grad PR)
  C4  SO - FO == d/dtd DR(td, theta0) at td = theta0                 (dropped performative term)
      (C3, C4 are required to hold only with the clip inactive; with an active clip the toy's
       SO detaches the clip scale, so it is not the exact gradient of PR -- see C5.)
  C5  with an ACTIVE clip, SO (detached clip scale, toy convention) != grad L, i.e. the toy's SO
      equals grad PR only where the clip is inactive.
  C6  AICE / lin inner rules: FO still equals the RGD direction (C2 holds for any rule).
      For AICE the rule itself detaches z_ref (in SO as well), so SO-with-AICE is NOT grad L;
      C3/C4 are expected to fail for AICE and are reported, not required.
Writes the JSON to ``--out``; exit code 0 iff C1/C2 pass for every rule and clip setting and C3/C4
pass with the clip inactive (CE, lin).

Run:  uv run --locked --with jax python scripts/analysis/theory-checks/performative_correspondence.py
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--out", default="outputs/analysis/theory-checks/performative_correspondence_check.json")
ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "performative-correspondence"))
args = ap.parse_args()
try:
    import setproctitle
    setproctitle.setproctitle(args.process_name)
except Exception:
    pass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

V, D, M, T, CH, B = 6, 5, 7, 12, 4, 3       # vocab, width, fast hidden, length, chunk, batch
C = T // CH
ETA_BASE = 0.7
OUT = args.out


def ln(x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5)


def init(key):
    ks = jax.random.split(key, 8)
    return dict(
        E=0.5 * jax.random.normal(ks[0], (V, D)),
        P=0.5 * jax.random.normal(ks[1], (D, D)),
        a=jnp.array(0.8),
        U=0.5 * jax.random.normal(ks[2], (V, D)),
        W1=0.3 * jax.random.normal(ks[3], (M, D)),
        b1=0.1 * jax.random.normal(ks[4], (M,)),
        W2=0.3 * jax.random.normal(ks[5], (D, M)),
        b2=0.1 * jax.random.normal(ks[6], (D,)),
        rho=0.2 * jax.random.normal(ks[7], (4,)),
    )


def backbone(th, x):                       # x: (T,) ints -> (T, D)
    e = th["E"][x]
    prev2 = jnp.concatenate([jnp.zeros((2, D)), e[:-2]], 0)
    prev1 = jnp.concatenate([jnp.zeros((1, D)), e[:-1]], 0)
    return e + th["a"] * prev2 + jnp.tanh(prev1 @ th["P"].T)


def head(th, W, h):
    W1, b1, W2, b2 = W
    u = ln(h) @ W1.T + b1
    out = h + jax.nn.gelu(u) @ W2.T + b2
    return ln(out) @ th["U"].T


def ce(z, y):
    return jax.nn.logsumexp(z, -1) - jnp.take_along_axis(z, y[:, None], -1)[:, 0]


def inner_tok(rule, z, y, zref):
    if rule == "ce":
        return ce(z, y)
    if rule == "lin":
        return z.mean(-1) - jnp.take_along_axis(z, y[:, None], -1)[:, 0]
    if rule == "aice":
        return ce(z - zref, y)
    raise ValueError(rule)


def etas(th):
    return ETA_BASE * jnp.exp(th["rho"])


def clip(gs, g_clip):
    norm = jnp.sqrt(sum((g ** 2).sum() for g in gs))
    scale = jax.lax.stop_gradient(jnp.where(norm > g_clip, g_clip / norm, 1.0))
    return [g * scale for g in gs]


def W0_of(th):
    return [th["W1"], th["b1"], th["W2"], th["b2"]]


def one_seq_forward(th, x, y, mode, rule, g_clip):
    """Returns (outer loss, write record). mode in {'fo','so'}."""
    h = backbone(th, x)
    W = W0_of(th)
    zref = jax.lax.stop_gradient(head(th, W, h))          # AICE reference: W0, detached (as toy)
    eta = etas(th)
    total, writes = 0.0, []
    for c in range(C):
        sl = slice(c * CH, (c + 1) * CH)
        z = head(th, W, h[sl])
        outer = ce(z, y[sl]).mean()

        def inner(Wl):
            return inner_tok(rule, head(th, Wl, h[sl]), y[sl], zref[sl]).mean()
        g = jax.grad(inner)(W)
        g = clip(g, g_clip)
        if mode == "fo":
            g = [jax.lax.stop_gradient(gi) for gi in g]
        writes.append(g)
        total = total + outer
        W = [w - e * gi for w, e, gi in zip(W, eta, g)]
    return total / C, writes


def L(th, X, Y, mode="so", rule="ce", g_clip=1e9):
    return jnp.mean(jnp.stack([one_seq_forward(th, X[i], Y[i], mode, rule, g_clip)[0]
                               for i in range(X.shape[0])]))


def write_record(th, X, Y, rule="ce", g_clip=1e9):
    return [one_seq_forward(th, X[i], Y[i], "so", rule, g_clip)[1] for i in range(X.shape[0])]


def replay_loss(th, X, Y, record):
    """l((s, gamma); theta) with gamma = record (data), averaged over the batch."""
    vals = []
    for i in range(X.shape[0]):
        h = backbone(th, X[i])
        W, eta, tot = W0_of(th), etas(th), 0.0
        for c in range(C):
            sl = slice(c * CH, (c + 1) * CH)
            tot = tot + ce(head(th, W, h[sl]), Y[i][sl]).mean()
            W = [w - e * gi for w, e, gi in zip(W, eta, record[i][c])]
        vals.append(tot / C)
    return jnp.mean(jnp.stack(vals))


def DR(th_d, th, X, Y, rule="ce", g_clip=1e9):
    return replay_loss(th, X, Y, write_record(th_d, X, Y, rule, g_clip))


# ---- flat parameter helpers -------------------------------------------------------------
def flatten(th):
    leaves, tree = jax.tree_util.tree_flatten(th)
    return jnp.concatenate([jnp.ravel(l) for l in leaves]), (tree, [l.shape for l in leaves])


def unflatten(v, meta):
    tree, shapes = meta
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s)) if len(s) else 1
        out.append(v[i:i + n].reshape(s))
        i += n
    return jax.tree_util.tree_unflatten(tree, out)


def fd_dir(fun, v0, d, h=1e-5):
    return (fun(v0 + h * d) - fun(v0 - h * d)) / (2 * h)


def main():
    key = jax.random.PRNGKey(0)
    th0 = init(key)
    rng = np.random.default_rng(1)
    X = jnp.array(rng.integers(0, V, size=(B, T)))
    Y = jnp.array(np.concatenate([np.array(X[:, 1:]), rng.integers(0, V, size=(B, 1))], 1))
    v0, meta = flatten(th0)
    P = v0.shape[0]
    dirs = [jnp.array(rng.standard_normal(P)) for _ in range(6)]
    dirs = [d / jnp.linalg.norm(d) for d in dirs]
    # also coordinate directions for a, rho_W2 and one W1 entry
    names = list(jax.tree_util.tree_flatten(th0)[0])
    res = {"P_params": int(P), "checks": {}}

    for rule in ("ce", "lin", "aice"):
        for g_clip in (1e9, 0.05):
            tag = f"rule={rule},clip={'off' if g_clip > 1e6 else g_clip}"
            Lf = jax.jit(lambda v: L(unflatten(v, meta), X, Y, "so", rule, g_clip))
            g_so = jax.jit(jax.grad(lambda v: L(unflatten(v, meta), X, Y, "so", rule, g_clip)))(v0)
            g_fo = jax.jit(jax.grad(lambda v: L(unflatten(v, meta), X, Y, "fo", rule, g_clip)))(v0)
            dr2 = jax.jit(lambda v: DR(th0, unflatten(v, meta), X, Y, rule, g_clip))  # 2nd argument
            dr1 = jax.jit(lambda v: DR(unflatten(v, meta), th0, X, Y, rule, g_clip))  # 1st argument
            pr = float(DR(th0, th0, X, Y, rule, g_clip))
            lv = float(Lf(v0))
            c2, c3, c4, c5 = [], [], [], []
            for d in dirs:
                fo_d = float(g_fo @ d)
                so_d = float(g_so @ d)
                c2.append(abs(fo_d - float(fd_dir(dr2, v0, d))) / (abs(fo_d) + 1e-12))
                fd_L = float(fd_dir(Lf, v0, d))
                c3.append(abs(so_d - fd_L) / (abs(so_d) + 1e-12))
                c4.append(abs((so_d - fo_d) - float(fd_dir(dr1, v0, d))) / (abs(so_d - fo_d) + 1e-12))
                # PR with a FROZEN clip scale: our L already detaches the scale, so
                # FD of L differentiates the scale too only through the norm; record mismatch.
                c5.append(abs(so_d - fd_L))
            res["checks"][tag] = dict(
                C1_PR_minus_L=abs(pr - lv),
                C2_FO_vs_FD_DR2_maxrel=max(c2),
                C3_SO_vs_FD_L_maxrel=max(c3),
                C4_SOminusFO_vs_FD_DR1_maxrel=max(c4),
                C5_SO_vs_FD_L_maxabs=max(c5),
                norm_FO=float(jnp.linalg.norm(g_fo)), norm_SO=float(jnp.linalg.norm(g_so)),
                norm_SO_minus_FO=float(jnp.linalg.norm(g_so - g_fo)),
            )
            print(tag, json.dumps(res["checks"][tag], indent=None), flush=True)

    # Which parameters does FO still differentiate?  FO gradient w.r.t. W0, rho, U, E, P, a.
    th_g_fo = jax.grad(lambda t: L(t, X, Y, "fo", "ce"))(th0)
    th_g_so = jax.grad(lambda t: L(t, X, Y, "so", "ce"))(th0)
    res["fo_differentiates"] = {k: float(jnp.linalg.norm(th_g_fo[k])) for k in th_g_fo}
    res["so_minus_fo_by_group"] = {k: float(jnp.linalg.norm(th_g_so[k] - th_g_fo[k])) for k in th_g_so}
    print("FO grad norms by group:", res["fo_differentiates"])
    print("|SO-FO| by group:", res["so_minus_fo_by_group"])

    ok = all(v["C1_PR_minus_L"] < 1e-12 and v["C2_FO_vs_FD_DR2_maxrel"] < 1e-6
             for v in res["checks"].values())
    ok_noclip = all(v["C3_SO_vs_FD_L_maxrel"] < 1e-6 and v["C4_SOminusFO_vs_FD_DR1_maxrel"] < 1e-5
                    for k, v in res["checks"].items() if "clip=off" in k and "aice" not in k)
    # AICE detaches its reference logits inside the rule (also in SO), so its "SO" is the
    # gradient of PR with the reference path removed; C3/C4 are expected to FAIL for it.
    aice_c3 = res["checks"]["rule=aice,clip=off"]["C3_SO_vs_FD_L_maxrel"]
    clip_mismatch = [v["C3_SO_vs_FD_L_maxrel"] for k, v in res["checks"].items() if "clip=0.05" in k]
    res["verdict"] = dict(C1_C2_pass_all_rules_and_clip=bool(ok),
                          C3_C4_pass_when_clip_inactive_ce_lin=bool(ok_noclip),
                          C6_aice_SO_is_not_gradL_expected=bool(aice_c3 > 1e-3),
                          C5_clip_active_SO_minus_gradL_rel=clip_mismatch)
    print("VERDICT", res["verdict"])
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    return 0 if (ok and ok_noclip) else 1


if __name__ == "__main__":
    sys.exit(main())
