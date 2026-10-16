"""SOQ-N3B analysis: registered quantities, labels, branches and cross-over kappa. NumPy only.

Inputs: N3B reads (SO/FO-CE and FO-CM), stored G0/G0b/G0c reads for SO/FO-CE grid points
that were not re-read, FO-CM run summaries.
Output: n3b-results/n3b_analysis.json and a printed report (per-seed table, labels, branches).
"""
import json, os, sys
import numpy as np

os.environ.setdefault("PROCESS_NAME", "TTM-n3b-65-analysis")
try:
    import setproctitle; setproctitle.setproctitle(os.environ["PROCESS_NAME"])
except ImportError:
    pass

G0 = "outputs/g0"
PP = "outputs/mini/matrix"
RP = f"{PP}/n3b-reads"          # SO / FO-CE reads
FP = f"{PP}/n3b-reads"          # FO-CM reads
RUNS = f"{PP}/n3b/runs/mini-m/e2e1-ce"
OUT = "outputs/n3b-results"
EVAL_MANIFEST = "data/p2data/eval.jsonl"
GRID = {"so": [.75, .9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3],
        "fo-ce": [.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8],
        "fo-cm": [.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12]}
ARMDIR = {"so": "e2e2", "fo-ce": "e2e1-ce"}
ROPE, T90 = .04, 1.8856
T_SO_PILOT, T_FO_PILOT, N_CM = 6607.372670434415, 2285.4128843173385, 3469
OVERLAP_ROWS = {17, 162, 328, 693, 939, 110, 301, 361, 531, 540, 593, 984, 986}   # eval rows dropped in the overlap sensitivity read
HO_OVERLAP = {44, 45, 102}
rng = np.random.default_rng(0)


def name(e):
    return "ce" if e == 1 else "eta%03d" % round(e * 100)


def path(arm, s, e):
    if arm == "fo-cm":
        f = f"{FP}/mini-m/fo-cm/s{s}/ncm/g0_read_{name(e)}.npz"
        return f if os.path.exists(f) else None
    f = f"{RP}/mini-m/{arm}/s{s}/g0_read_{name(e)}.npz"
    if os.path.exists(f):
        return f
    for d in ("data", "data-g0b", "data-g0c"):
        f = f"{G0}/{d}/mini-m/{ARMDIR[arm]}/s{s}/g0_read_{name(e)}.npz"
        if os.path.exists(f):
            return f
    return None


def load_curve(arm, s, mdrop=None, hodrop=None):
    rows = []
    for e in GRID[arm]:
        f = path(arm, s, e)
        if f is None:
            raise FileNotFoundError(f"{arm} s{s} x{e}")
        z = np.load(f)
        keep = np.ones(len(z["M_doc"]), bool)
        if mdrop is not None:
            keep &= ~np.isin(z["eval_rows"], list(mdrop))
        idx = np.arange(128)
        cal = idx[(idx % 2 == 0) & ~np.isin(idx, list(hodrop or []))]
        tst = idx[(idx % 2 == 1) & ~np.isin(idx, list(hodrop or []))]
        L = lambda ii: z["L_adapt_sum"][ii].sum() / z["L_adapt_count"][ii].sum()
        rows.append(dict(eta=e, L=L(cal), Lt=L(tst), M=z["M_doc"][keep].mean(), F=z["fP_focal"][keep].mean(),
                         N=z["fP_nonfocal"][keep].mean(),
                         clip=float(np.mean(z["clip_bind_fP_focal"][keep])) if "clip_bind_fP_focal" in z.files else np.nan,
                         kappa="kz_doc_pair_S" in z.files, file=f))
    return rows


def truncate_fo_cm(rows, B):
    """Registered adaptive top, applied post hoc: base grid to x6 always; then x8, x10, x12 in turn,
    stopping after the first point with L_CAL >= B or after M has fallen at two consecutive points."""
    base = [r for r in rows if r["eta"] <= 6]
    top = [r for r in rows if r["eta"] > 6]
    out = list(base); stop = None
    for r in top:
        out.append(r)
        if r["L"] >= B:
            stop = "budget"; break
        if len(out) >= 3 and out[-1]["M"] < out[-2]["M"] < out[-3]["M"]:
            stop = "M-falls"; break
    return out, stop


def interp(rows, key, g):
    e = np.array([r["eta"] for r in rows]); v = np.array([r[key] for r in rows])
    return np.interp(g, e, v)


def mstar(rows, B, shift=0.0):
    e = np.array([r["eta"] for r in rows])
    g = np.union1d(np.linspace(e.min(), e.max(), 40001), e)
    Lg = interp(rows, "L", g) - shift; Mg = interp(rows, "M", g)
    ok = Lg <= B
    if not ok.any():
        return dict(M=np.nan, eta=np.nan, flag="unreachable")
    i = np.flatnonzero(ok)[np.argmax(Mg[ok])]
    flag = "top-not-binding" if (ok[-1] and i == len(g) - 1) else ""
    return dict(M=Mg[i], eta=g[i], F=interp(rows, "F", g[i]), N=interp(rows, "N", g[i]),
                Lt=interp(rows, "Lt", g[i]), clip=interp(rows, "clip", g[i]), flag=flag)


def bounds(x):
    x = np.asarray(x, float); m = x.mean(); sd = x.std(ddof=1); h = T90 * sd / np.sqrt(len(x))
    return dict(mean=m, sd=sd, lb90=m - h, ub90=m + h, per_seed=[float(v) for v in x])


def kappa_at(arm, s, eta, nboot=2000):
    """Registered cross-over kappa: sum_doc S / sum_doc diag at the kappa read nearest to eta."""
    cands = [e for e in GRID[arm] if path(arm, s, e) and "kz_doc_pair_S" in np.load(path(arm, s, e)).files]
    e = min(cands, key=lambda c: abs(np.log(c) - np.log(eta)))
    z = np.load(path(arm, s, e))
    S, D = z["kz_doc_pair_S"].astype(float), z["kz_doc_pair_diag"].astype(float)
    ok = np.isfinite(S) & np.isfinite(D)
    S, D = S[ok], D[ok]
    bs = []
    for _ in range(nboot):
        j = rng.integers(0, len(S), len(S)); bs.append(S[j].sum() / D[j].sum() if D[j].sum() > 0 else np.nan)
    return dict(eta=e, S=S.sum(), D=D.sum(), kappa=S.sum() / D.sum() if D.sum() > 0 else np.nan,
                lb_doc=float(np.nanpercentile(bs, 10)), diag_lb=float(np.percentile(
                    [D[rng.integers(0, len(D), len(D))].sum() for _ in range(500)], 10)))


def seed_block(s, mdrop=None, hodrop=None):
    so = load_curve("so", s, mdrop, hodrop); ce = load_curve("fo-ce", s, mdrop, hodrop)
    cm_all = load_curve("fo-cm", s, mdrop, hodrop)
    Lmin_i = int(np.argmin([r["L"] for r in so])); Lmin = so[Lmin_i]["L"]
    B_SO, B_FO = Lmin + .01, Lmin + .02
    cm, stop = truncate_fo_cm(cm_all, B_FO)
    o = dict(seed=s, L_SO_min=Lmin, eta_SO_min=so[Lmin_i]["eta"], cm_top_stop=stop, cm_top=cm[-1]["eta"])
    o["so"] = mstar(so, B_SO); o["fo_ce"] = mstar(ce, B_FO); o["fo_cm"] = mstar(cm, B_FO)
    o["R_cov_cm"] = o["so"]["M"] - o["fo_cm"]["M"]; o["R_cov_ce"] = o["so"]["M"] - o["fo_ce"]["M"]
    o["dfn_cm"] = o["fo_cm"]["N"] - o["so"]["N"]; o["dfn_ce"] = o["fo_ce"]["N"] - o["so"]["N"]
    o["x"] = o["fo_cm"]["M"] - o["fo_ce"]["M"]
    o["delta_needed"] = o["R_cov_ce"] - ROPE
    # LM channel: FO-CE's L shifted by FO-CM's CAL floor gain
    Lce = min(r["L"] for r in ce); Lcm = min(r["L"] for r in cm_all)
    o["delta_hat"] = Lce - Lcm
    o["x_lm"] = mstar(ce, B_FO, shift=o["delta_hat"])["M"] - o["fo_ce"]["M"]
    o["x_mcurve"] = o["x"] - o["x_lm"]
    # floors
    o["floor_cal"] = {"so": Lmin, "fo_ce": Lce, "fo_cm": Lcm}
    o["floor_test"] = {k: min(r["Lt"] for r in v) for k, v in (("so", so), ("fo_ce", ce), ("fo_cm", cm_all))}
    o["edge"] = {"so": so[Lmin_i]["eta"] in (.75, 3), "fo_ce": min(ce, key=lambda r: r["L"])["eta"] == .75,
                 "fo_cm": min(cm_all, key=lambda r: r["L"])["eta"] == .75}
    # damage + memory-only sensitivity
    for k, rows in (("fo_cm", cm), ("fo_ce", ce)):
        sel = o[k]; F1 = next(r["F"] for r in rows if r["eta"] == 1)
        o[f"damage_{k}"] = sel["F"] - min(r["F"] for r in rows)
        o[f"R_cov_mem_{k[3:]}"] = o["so"]["M"] - (sel["M"] - max(0.0, sel["F"] - F1))
    # descriptives
    o["R_cov_b"] = {b: o_so - mstar(cm, Lmin + b + .01)["M"] for b, o_so in
                    ((b, mstar(so, Lmin + b)["M"]) for b in (.005, .01, .02))}
    o["R_B"] = {b: mstar(so, Lmin + b)["M"] - mstar(cm, Lmin + b)["M"] for b in (0, .005, .01, .02)}
    o["M_fixed"] = {arm: {e: next((r["M"] for r in rows if r["eta"] == e), np.nan) for e in (1, 2, 4)}
                    for arm, rows in (("so", so), ("fo_ce", ce), ("fo_cm", cm_all))}
    o["M_own_floor"] = {arm: {d: mstar(rows, min(r["L"] for r in rows) + d)["M"] for d in (.01, .05)}
                        for arm, rows in (("so", so), ("fo_ce", ce), ("fo_cm", cm_all))}
    o["rho_B"] = o["R_cov_cm"] / (o["M_fixed"]["so"][1] - o["M_fixed"]["fo_cm"][1])
    # refinement not read: worst case for the label = FO-CM's M* replaced by the larger bracketing grid M
    e_cm = [r["eta"] for r in cm]; j = int(np.searchsorted(e_cm, o["fo_cm"]["eta"]))
    br = [cm[i]["M"] for i in (j - 1, j) if 0 <= i < len(cm)]
    o["R_cov_cm_worst"] = o["so"]["M"] - max(br + [o["fo_cm"]["M"]])
    # lambda*: first w in [0, 2] where SO's best (L - w M) over points with L <= L_SO,min + .05 matches FO-CM's
    def best(rows, w):
        e = np.array([r["eta"] for r in rows]); g = np.union1d(np.linspace(e.min(), e.max(), 4001), e)
        Lg, Mg = interp(rows, "L", g), interp(rows, "M", g); ok = Lg <= Lmin + .05
        return (Lg[ok] - w * Mg[ok]).min()
    ws = np.linspace(0, 2, 2001); diff = np.array([best(so, w) - best(cm, w) for w in ws])
    cross = np.flatnonzero(diff <= 0)
    o["lambda_star_cm"] = float(ws[cross[0]]) if len(cross) else None
    diffc = np.array([best(so, w) - best(ce, w) for w in ws]); crc = np.flatnonzero(diffc <= 0)
    o["lambda_star_ce"] = float(ws[crc[0]]) if len(crc) else None
    o["so_at_top"] = o["so"]["eta"] >= 3 - 1e-9
    o["cm_top_open"] = cm[-1]["eta"] == 12 and cm[-1]["L"] < B_FO and stop is None
    o["grids"] = {"so": [(r["eta"], r["L"], r["M"]) for r in so], "fo_ce": [(r["eta"], r["L"], r["M"]) for r in ce],
                  "fo_cm": [(r["eta"], r["L"], r["M"]) for r in cm_all]}
    return o


def hlm(s):
    out = {}
    for arm, f in (("so", f"{RP}/mini-m/so/s{s}/g0_hlm.npz"), ("fo_ce", f"{RP}/mini-m/fo-ce/s{s}/g0_hlm.npz"),
                   ("fo_cm", f"{FP}/mini-m/fo-cm/s{s}/ncm/g0_hlm.npz")):
        z = np.load(f); out[arm] = float(z["heldout_sum"].sum() / z["heldout_count"].sum())
    return out


def compute(s):
    d = json.load(open(f"{RUNS}/s{s}/MINI_CELL_DONE.json"))["final"]
    st = np.array(d["step_ms"], float)
    raw = d["train_seconds"]["ratio"]
    norm = (st.sum() * T_FO_PILOT / np.median(st[10:60])) / (T_SO_PILOT * 1200)
    return dict(raw_ratio=raw, r_act_over_r=norm, speed_factor=float(np.median(st[10:60]) / T_FO_PILOT),
                steps=d["steps"], aborted=d["aborted"], guard=d["guard_label"])


def main():
    res = {"seeds": [seed_block(s) for s in range(3)]}
    near = [i for i, l in enumerate(open(EVAL_MANIFEST)) if json.loads(l).get("near_dup")]
    res["sens"] = [seed_block(s, mdrop=OVERLAP_ROWS | set(near), hodrop=HO_OVERLAP) for s in range(3)]
    res["n_near_dup_rows"] = len(near)
    for s, o in enumerate(res["seeds"]):
        o["H_LM"] = hlm(s); o["compute"] = compute(s)
        so_k = kappa_at("so", s, o["so"]["eta"]); cm_k = kappa_at("fo-cm", s, o["fo_cm"]["eta"])
        ce_k = kappa_at("fo-ce", s, o["fo_ce"]["eta"])
        so1, cm1 = kappa_at("so", s, 1.0), kappa_at("fo-cm", s, 1.0)
        o["kappa"] = {"so_op": so_k, "fo_cm_cover": cm_k, "fo_ce_cover": ce_k, "so_x1": so1, "fo_cm_x1": cm1,
                      "kappa_delta_matched": (so_k["S"] - cm_k["S"]) / (so_k["D"] - cm_k["D"]) if so_k["D"] != cm_k["D"] else np.nan,
                      "kappa_delta_x1": (so1["S"] - cm1["S"]) / (so1["D"] - cm1["D"]) if so1["D"] != cm1["D"] else np.nan}
        for tag, a, b in (("matched", ("so", so_k["eta"]), ("fo-cm", cm_k["eta"])), ("x1", ("so", 1.0), ("fo-cm", 1.0))):
            za, zb = np.load(path(a[0], s, a[1])), np.load(path(b[0], s, b[1]))
            Sa, Da, Sb, Db = (np.nan_to_num(z[k].astype(float)) for z in (za, zb) for k in ("kz_doc_pair_S", "kz_doc_pair_diag"))
            bd = []; bk = []
            for _ in range(2000):
                j = rng.integers(0, len(Sa), len(Sa)); dd = Da[j].sum() - Db[j].sum(); bd.append(dd)
                bk.append((Sa[j].sum() - Sb[j].sum()) / dd if dd > 0 else np.nan)
            o["kappa"][f"kd_{tag}_denLB"] = float(np.percentile(bd, 10))
            o["kappa"][f"kd_{tag}_docLB"] = float(np.nanpercentile(bk, 10))
        # M(x1) trajectory of FO-CM (descriptive): reads at 2400 and N_CM
        o["M_x1_traj"] = {st: float(np.load(f"{FP}/mini-m/fo-cm/s{s}/{d}/g0_read_ce.npz")["M_doc"].mean())
                          for st, d in ((2400, "step2400"), (N_CM, "ncm"))}
    S = res["seeds"]
    agg = {k: bounds([o[k] for o in S]) for k in ("R_cov_cm", "R_cov_ce", "dfn_cm", "dfn_ce", "x", "delta_needed",
                                                   "x_lm", "x_mcurve", "delta_hat", "R_cov_mem_cm", "R_cov_mem_ce",
                                                   "damage_fo_cm", "rho_B", "R_cov_cm_worst")}
    agg["R_cov_b"] = {b: bounds([o["R_cov_b"][b] for o in S]) for b in (.005, .01, .02)}
    agg["sens_R_cov_cm"] = bounds([o["R_cov_cm"] for o in res["sens"]])
    agg["sens_dfn_cm"] = bounds([o["dfn_cm"] for o in res["sens"]])
    agg["sens_x"] = bounds([o["x"] for o in res["sens"]])
    agg["test_gap"] = bounds([o["fo_cm"]["Lt"] - o["so"]["Lt"] for o in S])
    agg["kappa_x_so_op"] = bounds([o["kappa"]["so_op"]["kappa"] for o in S])
    agg["kappa_x_cm_cover"] = bounds([o["kappa"]["fo_cm_cover"]["kappa"] for o in S])
    agg["kappa_delta_matched"] = bounds([o["kappa"]["kappa_delta_matched"] for o in S])
    agg["kappa_delta_x1"] = bounds([o["kappa"]["kappa_delta_x1"] for o in S])
    R = agg["R_cov_cm"]; Dn = agg["dfn_cm"]
    if R["ub90"] <= ROPE and Dn["ub90"] <= ROPE:
        lab = "COVERS"
    elif R["ub90"] <= ROPE:
        lab = "COVERS-M-ONLY"
    elif R["lb90"] > ROPE:
        lab = "NOT-COVERED" + (" (large)" if R["mean"] >= .08 else "")
    else:
        lab = "UNDERPOWERED"
    Rs = agg["sens_R_cov_cm"]
    lab_s = ("NOT-COVERED" if Rs["lb90"] > ROPE else "COVERS" if (Rs["ub90"] <= ROPE and agg["sens_dfn_cm"]["ub90"] <= ROPE)
             else "COVERS-M-ONLY" if Rs["ub90"] <= ROPE else "UNDERPOWERED")
    res["agg"] = agg; res["label"] = lab; res["label_sens"] = lab_s
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/n3b_analysis.json", "w"), indent=1, default=lambda v: float(v) if np.isscalar(v) else str(v))
    # report
    f3 = lambda v: f"{v:+.3f}"
    print("LABEL:", lab, "| overlap-sensitivity label:", lab_s)
    for k in ("R_cov_cm", "R_cov_ce", "dfn_cm", "x", "delta_needed", "x_lm", "x_mcurve", "delta_hat", "R_cov_mem_cm", "damage_fo_cm"):
        a = agg[k]; print(f"{k:14s} seeds {' '.join(f3(v) for v in a['per_seed'])} mean {f3(a['mean'])} SD {a['sd']:.3f} LB90 {f3(a['lb90'])} UB90 {f3(a['ub90'])}")
    for b, a in agg["R_cov_b"].items():
        print(f"R_cov(b={b}) mean {f3(a['mean'])} UB90 {f3(a['ub90'])} LB90 {f3(a['lb90'])}")
    print("sens R_cov", f3(Rs["mean"]), "LB90", f3(Rs["lb90"]), "| sens dfn", f3(agg["sens_dfn_cm"]["mean"]), "| sens x", f3(agg["sens_x"]["mean"]), "| near_dup rows", len(near))
    print("TEST gap (FO-CM cover - SO op)", f3(agg["test_gap"]["mean"]), "SD", round(agg["test_gap"]["sd"], 4))
    for o in S:
        c = o["compute"]
        print(f"s{o['seed']}: L_SO,min {o['L_SO_min']:.4f} (x{o['eta_SO_min']}) | SO op x{o['so']['eta']:.2f} M {o['so']['M']:.4f} N {o['so']['N']:.3f}"
              f" | FO-CE x{o['fo_ce']['eta']:.2f} M {o['fo_ce']['M']:.4f} | FO-CM x{o['fo_cm']['eta']:.2f} M {o['fo_cm']['M']:.4f} N {o['fo_cm']['N']:.3f} {o['fo_cm']['flag']}"
              f" | top {o['cm_top']} stop={o['cm_top_stop']} open={o['cm_top_open']} so_top={o['so_at_top']} edge={o['edge']}")
        print(f"    floors CAL so {o['floor_cal']['so']:.4f} ce {o['floor_cal']['fo_ce']:.4f} cm {o['floor_cal']['fo_cm']:.4f} | TEST so {o['floor_test']['so']:.4f} ce {o['floor_test']['fo_ce']:.4f} cm {o['floor_test']['fo_cm']:.4f}"
              f" | H_LM so {o['H_LM']['so']:.4f} ce {o['H_LM']['fo_ce']:.4f} cm {o['H_LM']['fo_cm']:.4f}")
        print(f"    M@x1 so {o['M_fixed']['so'][1]:.3f} ce {o['M_fixed']['fo_ce'][1]:.3f} cm {o['M_fixed']['fo_cm'][1]:.3f} | M@x2 so {o['M_fixed']['so'][2]:.3f} ce {o['M_fixed']['fo_ce'][2]:.3f} cm {o['M_fixed']['fo_cm'][2]:.3f}"
              f" | M@x4 ce {o['M_fixed']['fo_ce'][4]:.3f} cm {o['M_fixed']['fo_cm'][4]:.3f} | R_B(.01) {o['R_B'][.01]:+.3f} R_B(0) {o['R_B'][0]:+.3f}")
        print(f"    compute raw {c['raw_ratio']:.4f} r_act/r(norm) {c['r_act_over_r']:.4f} speed {c['speed_factor']:.4f} steps {c['steps']} aborted {c['aborted']} guard {c['guard']}")
        k = o["kappa"]
        print(f"    kappa x: SO op (x{k['so_op']['eta']}) {k['so_op']['kappa']:.3f} [docLB {k['so_op']['lb_doc']:.3f}] | FO-CM cover (x{k['fo_cm_cover']['eta']}) {k['fo_cm_cover']['kappa']:.3f}"
              f" | FO-CE cover (x{k['fo_ce_cover']['eta']}) {k['fo_ce_cover']['kappa']:.3f} | x1 SO {k['so_x1']['kappa']:.3f} CM {k['fo_cm_x1']['kappa']:.3f}"
              f" | kappa_delta matched {k['kappa_delta_matched']:.3f} x1 {k['kappa_delta_x1']:.3f}")
    print("R_cov worst-case (no refinement):", ' '.join(f3(o['R_cov_cm_worst']) for o in S), "LB90", f3(agg['R_cov_cm_worst']['lb90']))
    print("lambda* SO vs FO-CM:", [o['lambda_star_cm'] for o in S], "| SO vs FO-CE:", [o['lambda_star_ce'] for o in S])
    print("FO-CM M(x1) at 2400 / N_CM:", [(round(o['M_x1_traj'][2400], 3), round(o['M_x1_traj'][N_CM], 3)) for o in S])
    print("R_B(0):", [round(o['R_B'][0], 3) for o in S], "rho_B", [round(o['rho_B'], 3) for o in S])
    print("kappa_delta denominators doc-LB10 (matched, x1):", [(round(o['kappa']['kd_matched_denLB'], 1), round(o['kappa']['kd_x1_denLB'], 1)) for o in S],
          "| kappa_delta doc-LB10:", [(round(o['kappa']['kd_matched_docLB'], 3), round(o['kappa']['kd_x1_docLB'], 3)) for o in S])
    for k in ("kappa_x_so_op", "kappa_x_cm_cover", "kappa_delta_matched", "kappa_delta_x1"):
        a = agg[k]; print(f"{k}: mean {a['mean']:.3f} LB90 {a['lb90']:.3f} UB90 {a['ub90']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
