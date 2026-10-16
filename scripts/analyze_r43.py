"""Pre-registered R4-3 seed-paired analysis.

Usage: python scripts/analyze_r43.py RUN.log OUTPUT_DIR [--s5-fp32 S5.json]
       [--ek-manipulation EK_SHARE.json]

Two comparison groups, each paired by seed with its own full pair count:

- first-order / +emb group (``seeds 0-9``): ``e2e1emb`` (anchor, also FM's anchor),
  ``e2e1emb_kvm`` (FM), ``e2e1e_emb`` (E), ``e2e1k_emb`` (K), ``e2e2emb`` (the MS1
  reference);
- second-order group (``seeds 0-19``): ``e2e2`` (anchor) and ``e2e2_kvm``.

Endpoints are R4-1/R4-2's ``r4_flags`` per seed. Binary paired differences get
exact one-sided McNemar tests and conservative exact >= 95% intervals (Bonferroni
projection of two exact 97.5% Clopper-Pearson intervals). A label with fewer
complete pairs than its group needs is ``pending``, never negative.

Every label name below is the registered wording, and the function that emits it quotes
the rule it implements. ``--s5-fp32`` applies the pre-registered fallback: an +emb
cell whose in-run S5 failed is judged by an fp32 CPU recomputation on its final
checkpoint (a list of ``{cell_id, passed}`` records). The E/K manipulation gate is
read from the run log's own ``R43_EK_SHARE`` record (median share of the second-order
correction over seeds 0-2, per a registered amendment); there is no default pass, and a
missing record leaves the gate "unknown" and both halves "untested". The optional
``--ek-manipulation`` replaces that record (same schema, or a bare ``{"E", "K"}``).
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

ANCHOR = "e2e1emb"
KVM_FO = "e2e1emb_kvm"
REF_SO_EMB = "e2e2emb"
E_ARM, K_ARM = "e2e1e_emb", "e2e1k_emb"
SO_ANCHOR, SO_KVM = "e2e2", "e2e2_kvm"
FO_ARMS = (ANCHOR, KVM_FO, E_ARM, K_ARM, REF_SO_EMB)
SO_ARMS = (SO_ANCHOR, SO_KVM)
ARMS = FO_ARMS + SO_ARMS
HALVES = {"E": E_ARM, "K": K_ARM}
FO_SEEDS, SO_SEEDS = 10, 20
MASKED_ARMS = (KVM_FO, SO_KVM)
SECOND_ORDER = (SO_ANCHOR, REF_SO_EMB, SO_KVM)
EMB_ARMS = FO_ARMS
# every registered comparison, with the full pair count it needs. Pairing is judged
# per comparison: a first_batch_sha1 mismatch unpairs the comparisons that contain the
# mismatched cell and nothing else (registered amendment).
COMPARISONS = (("FM", KVM_FO, ANCHOR, FO_SEEDS), ("EK_E", E_ARM, ANCHOR, FO_SEEDS),
               ("EK_K", K_ARM, ANCHOR, FO_SEEDS), ("MS1_E", E_ARM, REF_SO_EMB, FO_SEEDS),
               ("MS1_K", K_ARM, REF_SO_EMB, FO_SEEDS),
               ("S2", SO_KVM, SO_ANCHOR, SO_SEEDS))
# the factor switches each arm must have logged in CELL_RESULT; None for every other arm
FACTOR_ARMS = {E_ARM: [False, True], K_ARM: [True, False]}
# R4-1's bound/unbound separator, and the registered surprise thresholds of MS1
P_KV_SEPARATOR = 0.993
E_KV_ABSOLUTE = 0.0069
EK_MANIPULATION_MIN = 0.05
# MS1's surprise floor: median e_kv_abs(X) >= this times the e2e2emb median, over the seeds
# where each arm ends bound. Fewer than this many bound X seeds and MS1 is not evaluable.
MS1_RATIO = 0.5
MS1_MIN_BOUND = 4
# the EK labels that are final, in the sense that a joint reading may use them
EK_FINAL = ("blocked", "retains", "partial", "no retention gain shown", "inconclusive")
DESCRIPTIVE = " (descriptive)"


# ----------------------------------------------------------------------- records
def ek_manipulation_gate(record):
    """The E/K manipulation gate from the run's ``R43_EK_SHARE`` record.

    The pre-registered rule is the median ``share_X`` over seeds 0-2 of
    ``||grad X - grad e2e1emb|| / ||grad e2e2emb - grad e2e1emb||`` at or above 0.05, read
    from the log itself. ``None`` for a half means the gate is unknown, which is reported
    as "untested (manipulation unknown)"; there is no default pass.
    """
    out = {"threshold": EK_MANIPULATION_MIN, "source": "R43_EK_SHARE"}
    per_seed = (record or {}).get("per_seed") or []
    if not record or not per_seed:
        out.update(status="unknown", reason="no R43_EK_SHARE record in the log")
        out.update({half: None for half in HALVES})
        return out
    out.update(status="computed", n_seeds=len(per_seed))
    for half, arm in HALVES.items():
        key = f"share_{half}"
        values = [r.get(arm, {}).get("share") for r in per_seed]
        values = [v for v in values if v is not None]
        med = median_of(values)
        out[f"per_seed_{key}"] = values
        out[f"median_{key}"] = med
        out[half] = None if med is None else bool(med >= EK_MANIPULATION_MIN)
        # reported next to the labels, never gating (registered amendment): the cosine with the
        # full correction and the relative metric
        for extra in ("cos", "rel"):
            got = [r.get(arm, {}).get(extra) for r in per_seed]
            out[f"per_seed_{extra}_{half}"] = got
            out[f"median_{extra}_{half}"] = median_of(got)
    return out


def records(path, s5_override=None, n_keys=16):
    """Cells, probe series, echo references and the EK gate from a run log, plus issues.

    ``n_keys`` is the number of masked targets a kv mask must produce per sequence
    (16 for the pre-registered run; a smaller smoke config passes its own value).
    """
    cells, probes, refs, share = {}, {}, {}, None
    for line in Path(path).open():
        if line.startswith("CELL_RESULT "):
            c = json.loads(line[len("CELL_RESULT "):])
            cells[(c["seed"], c["arm"])] = c
        elif line.startswith("PROBE "):
            p = json.loads(line[len("PROBE "):])
            probes.setdefault(p["cell_id"], {})[p["step"]] = p
        elif line.startswith("R43_EK_SHARE "):
            share = json.loads(line[len("R43_EK_SHARE "):])
        elif line.startswith("REF echo "):
            refs = json.loads(line[len("REF echo "):])
    for c in cells.values():
        probes.setdefault(c["cell_id"], {}).update({p["step"]: p for p in c.get("probes", [])})
        if s5_override and c["cell_id"] in s5_override and "s5" in c:
            c["s5_in_run"] = dict(c["s5"])
            c["s5"] = dict(c["s5"], passed=bool(s5_override[c["cell_id"]]), source="fp32-cpu")
    issues = []
    for c in cells.values():
        # a cell whose path or numerics are wrong is invalid for every comparison it is in
        want = n_keys if c["arm"] in MASKED_ARMS else 0
        if c.get("masked_per_seq") is not None and c["masked_per_seq"] != want:
            issues.append(f"{c['cell_id']}: masked_per_seq {c['masked_per_seq']} != {want}")
            c["invalid"] = True
        if "create_graph" in c:
            want_so = c["arm"] in SECOND_ORDER
            if bool(c["create_graph"]) is not want_so:
                issues.append(f"{c['cell_id']}: create_graph {c['create_graph']} but the arm is "
                              f"{'second' if want_so else 'first'} order")
                c["invalid"] = True
        if c.get("tf32") is True:
            issues.append(f"{c['cell_id']}: tf32 on; the in-run S5 needs fp32 matmuls")
            c["invalid"] = True
        if c.get("factors") != FACTOR_ARMS.get(c["arm"]):
            issues.append(f"{c['cell_id']}: factors {c.get('factors')} but the registry says "
                          f"{FACTOR_ARMS.get(c['arm'])} for {c['arm']}")
            c["invalid"] = True
    for name, x_arm, a_arm, seeds in COMPARISONS:
        for s in range(seeds):
            x, a = cells.get((s, x_arm)), cells.get((s, a_arm))
            if not x or not a:
                continue
            hx, ha = x.get("first_batch_sha1"), a.get("first_batch_sha1")
            if hx and ha and hx != ha:
                issues.append(f"{name} seed {s}: first_batch_sha1 differs, "
                              f"{x['cell_id']} {hx} vs {a['cell_id']} {ha}; this comparison "
                              f"is unpaired")
                x.setdefault("sha_mismatch", {})[a_arm] = True
                a.setdefault("sha_mismatch", {})[x_arm] = True
    return cells, probes, refs, issues, ek_manipulation_gate(share)


def _sha_mismatch(cell, other_arm):
    """True when ``cell``'s first batch hash differs from ``other_arm``'s in that seed."""
    return bool((cell.get("sha_mismatch") or {}).get(other_arm))


def valid(c):
    """Complete cell: trained, unpaired with this comparison, evaluated, S5-clean for +emb."""
    return bool(c and not c.get("error") and not c.get("skipped") and not c.get("invalid") and
                "eval" in c and (c["arm"] not in EMB_ARMS or
                                 c.get("s5", {}).get("passed") is True))


def paired(cells, a, b, seeds):
    """Seeds where both cells of the pair are complete and share their first batch."""
    return [(s, cells[s, a], cells[s, b]) for s in range(seeds)
            if valid(cells.get((s, a))) and valid(cells.get((s, b)))
            and not _sha_mismatch(cells[s, a], b) and not _sha_mismatch(cells[s, b], a)]


def complete(cells, arm, seeds):
    """Complete cells of one arm."""
    return [cells[s, arm] for s in range(seeds) if valid(cells.get((s, arm)))]


# --------------------------------------------------------------------- statistics
def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))


def cp(k, n, alpha=.025):
    """Exact Clopper-Pearson bounds of the two discordance probabilities."""
    if not n:
        return (0., 1.)
    lo, hi = 0., 1.
    if k:
        a, b = 0., k/n
        for _ in range(60):
            mid = (a+b)/2
            if 1-binom_cdf(k-1, n, mid) > alpha/2:
                b = mid
            else:
                a = mid
        lo = (a+b)/2
    if k < n:
        a, b = k/n, 1.
        for _ in range(60):
            mid = (a+b)/2
            if binom_cdf(k, n, mid) > alpha/2:
                a = mid
            else:
                b = mid
        hi = (a+b)/2
    return lo, hi


def contrast(rows, field):
    """Paired binary contrast: counts, exact >= 95% interval, one-sided McNemar both ways."""
    positive = sum(bool(a[field]) and not bool(b[field]) for _, a, b in rows)
    negative = sum(bool(b[field]) and not bool(a[field]) for _, a, b in rows)
    n = len(rows)
    pl, ph = cp(positive, n)
    nl, nh = cp(negative, n)
    both = positive + negative
    return dict(n=n, positive=positive, negative=negative,
                difference=(positive-negative)/n if n else None,
                ci95=[pl-nh, ph-nl] if n else None,
                mcnemar_one_sided=binom_cdf(negative, both, .5) if both else 1.,
                mcnemar_reverse=binom_cdf(positive, both, .5) if both else 1.)


def holm(pvals):
    """Holm step-down adjusted p-values over a family of one-sided tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    total = len(items)
    out, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        adjusted = min(1.0, max(running, p * (total - i)))
        out[key] = adjusted
        running = adjusted
    return out


def midranks(values):
    """Doubled 1-based midranks of |differences|, so ties stay integral."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j+1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = i + 1 + (j + 1)
        i = j + 1
    return ranks


def wilcoxon_one_sided(diffs):
    """Exact one-sided Wilcoxon signed-rank test (H0: a symmetric difference).

    Doubled midranks are integers, so the null distribution over all 2^n sign
    assignments comes from a small subset-sum dynamic programme: no normal
    approximation, no ties correction beyond the midranks.
    """
    d = [float(x) for x in diffs if x is not None and float(x) != 0.]
    if not d:
        return dict(n=len(diffs or []), n_nonzero=0, w_plus=None, p_one_sided=None,
                    median_diff=None, mean_diff=None)
    ranks = midranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    counts = {0: 1}
    for r in ranks:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = sum(counts.values())
    return dict(n=len(diffs or []), n_nonzero=len(d), w_plus=w_plus/2,
                p_one_sided=sum(c for s, c in counts.items() if s >= w_plus)/total,
                median_diff=median(d), mean_diff=sum(d)/len(d))


def ratio(a, b):
    """Ratio with R4-2's inf/None convention for a starved or missing denominator."""
    if a is None or b is None:
        return None
    return a/b if b > 0 else (math.inf if a > 0 else None)


def probe(probes, cell, step=None):
    """One probe row of a cell: at ``step``, else the latest one."""
    series = probes.get(cell["cell_id"], {}) if cell else {}
    if not series:
        return None
    return series[step] if step is not None else series[max(series)]


def median_of(values):
    values = [v for v in values if v is not None]
    return float(median(values)) if values else None


def _strip_descriptive(label):
    """A label without the "(descriptive)" suffix the gates append."""
    return (label or "").removesuffix(DESCRIPTIVE)


def last_bound(c):
    """Latest step whose quick eval accuracy is at least 0.90 (R4-1's ``t_s``)."""
    if not c:
        return None
    bound = [r[0] for r in c.get("quick", []) if r[3] is not None and r[3] >= .90]
    return max(bound) if bound else None


def lost(c):
    """Acquired the binding and no longer holds it at the end."""
    return bool(c["bound_ever"]) and not bool(c["bound_final"])


# ------------------------------------------------------------------------ labels
def v3a(cells):
    """V3a: e2e1emb acquisition >= 7/10 and bound_final <= 3/10."""
    good = complete(cells, ANCHOR, FO_SEEDS)
    if len(good) < FO_SEEDS:
        return "pending"
    acq = sum(c["bound_ever"] for c in good)
    ret = sum(c["bound_final"] for c in good)
    if acq >= 7 and ret <= 3:
        return "holds"
    if acq < 7:
        return "anchor not reproduced"
    return "no retention deficit on this device"


def v3b(cells):
    """V3b: e2e2emb bound_final >= 8/10, the reference MS1 is measured against."""
    good = complete(cells, REF_SO_EMB, FO_SEEDS)
    if len(good) < FO_SEEDS:
        return "pending"
    return "holds" if sum(c["bound_final"] for c in good) >= 8 else "not at ceiling"


def v3c(cells):
    """V3c: e2e2 bound_final <= 16/20, i.e. a second-order deficit this round can explain."""
    good = complete(cells, SO_ANCHOR, SO_SEEDS)
    if len(good) < SO_SEEDS:
        return "pending"
    return "holds" if sum(c["bound_final"] for c in good) <= 16 else (
        "no second-order deficit on this device")


def fm(cells):
    """FM: McNemar bound_final e2e1emb_kvm > e2e1emb at p <= .05 and >= 5/10 retained.

    Otherwise FM-no rescue shown. Pending below 10 complete pairs.
    """
    rows = paired(cells, KVM_FO, ANCHOR, FO_SEEDS)
    test = contrast(rows, "bound_final")
    if len(rows) < FO_SEEDS:
        return "pending", dict(test=test, kvm_bound_final=sum(x["bound_final"] for _, x, _ in rows))
    retained = sum(x["bound_final"] for _, x, _ in rows)
    label = "FM-rescue" if (test["mcnemar_one_sided"] <= .05 and retained >= 5) else (
        "FM-no rescue shown")
    return label, dict(test=test, kvm_bound_final=retained)


def ek_half(cells, half, holm_p, anchor_label, gate):
    """EK for one half, in the registered order of precedence:

    1. X-blocked: acquisition <= 3/10 (retention untested);
    2. X-retains: Holm-adjusted McNemar bound_final vs e2e1emb at p <= .05 and >= 5/10;
    3. X-partial: bound_final >= 4/10 and >= 4 more seeds ending not collapsed among
       those that acquired in both arms (the floor is the >= 4/10);
    4. X-no retention gain shown: acquisition >= 7/10 and bound_final <= 2/10;
    5. otherwise X-inconclusive.

    The gates come first, in the order they can fire: an incomplete pair set is "pending",
    an unreproduced anchor makes the half descriptive (registered amendment), and the
    manipulation gate ("untested") follows it, because a half that does not carry the
    correction cannot be read as a null result.
    """
    arm = HALVES[half]
    rows = paired(cells, arm, ANCHOR, FO_SEEDS)
    prefix = f"{half}-"
    if len(rows) < FO_SEEDS:
        return prefix + "pending", dict(n=len(rows))
    if anchor_label != "holds":
        return prefix + "descriptive (V3a did not hold)", dict(n=len(rows))
    if gate.get(half) is not True:
        reason = "manipulation unknown" if gate.get(half) is None else "manipulation failed"
        return prefix + f"untested ({reason})", dict(n=len(rows), gate=gate.get(half))
    acq = sum(x["bound_ever"] for _, x, _ in rows)
    ret = sum(x["bound_final"] for _, x, _ in rows)
    test = contrast(rows, "bound_final")
    common = [(x, a) for _, x, a in rows if x["bound_ever"] and a["bound_ever"]]
    partial = (sum(x["eval"]["query_acc"] >= .50 and a["eval"]["query_acc"] < .50 for x, a in common) -
               sum(a["eval"]["query_acc"] >= .50 and x["eval"]["query_acc"] < .50 for x, a in common))
    detail = dict(n=len(rows), acquisition=acq, bound_final=ret,
                  common_acquired=len(common), x_partial_difference=partial,
                  partial_floor=4, p_raw=test["mcnemar_one_sided"], p_holm=holm_p,
                  ret_contrast=test)
    if acq <= 3:
        return prefix + "blocked", detail
    if holm_p is not None and holm_p <= .05 and ret >= 5:
        return prefix + "retains", detail
    if ret >= 4 and partial >= 4:
        return prefix + "partial", detail
    if acq >= 7 and ret <= 2:
        return prefix + "no retention gain shown", detail
    return prefix + "inconclusive", detail


def ms1(cells, probes, half, a_label, b_label):
    """MS1, anchor-free: median e_kv_abs(X) >= 0.5 x the e2e2emb median and median
    p_kv(X) <= 0.993, each median over the seeds where that arm ends ``bound_final`` (registered
    amendment). All-seed medians are reported descriptively. The R4-1 absolute separator
    e_kv_abs >= 0.0069 is reported descriptively too.

    Not evaluable below ``MS1_MIN_BOUND`` bound X seeds; descriptive when a gate fails.
    """
    rows = paired(cells, HALVES[half], REF_SO_EMB, FO_SEEDS)
    if a_label not in ("holds", "pending"):
        return "descriptive (V3a did not hold)", dict(n=len(rows))
    if len(rows) < FO_SEEDS:
        return "pending", dict(n=len(rows), reason="incomplete pairs")
    if b_label != "holds":
        return "descriptive (V3b did not hold)", dict(n=len(rows))
    bound = [(x, r) for _, x, r in rows if x["bound_final"]]
    if len(bound) < MS1_MIN_BOUND:
        return f"not evaluable ({len(bound)} bound seeds)", dict(n=len(rows), bound_seeds=len(bound))
    ref_bound = [(x, r) for _, x, r in rows if r["bound_final"]]
    med_e = median_of([probe(probes, x)["e_kv_abs"] for x, _ in bound])
    med_ref = median_of([probe(probes, r)["e_kv_abs"] for _, r in ref_bound])
    med_p = median_of([probe(probes, x)["p_kv"] for x, _ in bound])
    surprise = med_e is not None and med_ref is not None and med_e >= MS1_RATIO * med_ref
    not_confident = med_p is not None and med_p <= P_KV_SEPARATOR
    return (surprise and not_confident), dict(
        n=len(rows), bound_seeds=len(bound), ref_bound_seeds=len(ref_bound),
        median_e_kv_abs=med_e, median_e_kv_abs_ref=med_ref, median_p_kv=med_p,
        surprise_kept=surprise, not_confident=not_confident, ratio_to_ref=ratio(med_e, med_ref),
        absolute_separator_0069=med_e is not None and med_e >= E_KV_ABSOLUTE,
        all_seed=dict(
            median_e_kv_abs=median_of([probe(probes, x)["e_kv_abs"] for _, x, _ in rows]),
            median_e_kv_abs_ref=median_of([probe(probes, r)["e_kv_abs"] for _, _, r in rows]),
            median_p_kv=median_of([probe(probes, x)["p_kv"] for _, x, _ in rows])))


def ms2(cells, half, floor, a_label):
    """MS2: R4-2's joint write ablation, in >= 2/3 of the seeds X retains.

    Both conditions must hold in the same retained seed; a missing, failed or
    mismatched ablation fails that seed; no retained seed means not evaluable. A failed
    V3a makes the whole reading descriptive.
    """
    rows = paired(cells, HALVES[half], ANCHOR, FO_SEEDS)
    if a_label not in ("holds", "pending"):
        return "descriptive (V3a did not hold)", dict(n=len(rows))
    if len(rows) < FO_SEEDS:
        return "pending", dict(n=len(rows))
    retained = [x for _, x, _ in rows if x["bound_final"]]

    def holds(c):
        ab = c.get("ablation") or {}
        return bool(c.get("ablation_ref_match") is True and floor is not None and
                    ab.get("all_minus_kv", math.inf) <= floor + .10 and
                    ab.get("kv_only", -math.inf) >= .80)
    passed = sum(holds(x) for x in retained)
    if not retained:
        return "not evaluable", dict(retained=0, passed=0)
    return passed * 3 >= 2 * len(retained), dict(retained=len(retained), passed=passed)


def s2(cells, c_label="holds"):
    """S2, the kv mask in plain second order.

    - S2-acq: McNemar bound_ever at p <= .05, in either direction.
    - S2-ret: untestable at this n when the e2e2 anchor has <= 3 seeds that acquired
      and then lost the binding; otherwise, among the seeds that acquired in both
      arms, McNemar bound_final at p <= .05 in either direction.
    - S2-overall: McNemar bound_final over all pairs, a label only if S2-acq or
      S2-ret fires.

    A failed V3c means there is no second-order deficit on this device, so all three
    readings carry "(descriptive)" (registered amendment).
    """
    rows = paired(cells, SO_KVM, SO_ANCHOR, SO_SEEDS)
    acq_test = contrast(rows, "bound_ever")
    overall = contrast(rows, "bound_final")
    if len(rows) < SO_SEEDS:
        acq_label, ret_label = "pending", "pending"
    else:
        acq_label = ("the mask helps second order acquire" if acq_test["mcnemar_one_sided"] <= .05
                     else "the mask hurts acquisition" if acq_test["mcnemar_reverse"] <= .05
                     else "no acquisition effect shown")
        lost_anchor = sum(lost(c) for _, _, c in rows)
        common = [(s, x, a) for s, x, a in rows if x["bound_ever"] and a["bound_ever"]]
        if lost_anchor <= 3:
            ret_label = "untestable at this n"
        else:
            test = contrast(common, "bound_final")
            ret_label = ("S2-ret rescue" if test["mcnemar_one_sided"] <= .05 else
                         "S2-ret hurts" if test["mcnemar_reverse"] <= .05
                         else "no retention effect shown")
    lost_anchor = sum(lost(c) for _, _, c in rows)
    common = [(s, x, a) for s, x, a in rows if x["bound_ever"] and a["bound_ever"]]
    fires = acq_label in ("the mask helps second order acquire", "the mask hurts acquisition") \
        or ret_label in ("S2-ret rescue", "S2-ret hurts")
    overall_label = ("pending" if len(rows) < SO_SEEDS else
                     "label (S2-acq or S2-ret fired)" if fires else
                     "reported; not a label (neither S2-acq nor S2-ret fired)")
    # a pending reading stays "pending": the gate qualifies readings, not missing pairs
    suffix = "" if c_label in ("holds", "pending") or len(rows) < SO_SEEDS else DESCRIPTIVE
    return dict(acq=acq_label + suffix, ret=ret_label + suffix, overall=overall_label + suffix,
                n_pairs=len(rows),
                n_common_acquired=len(common), anchor_lost_after_acquiring=lost_anchor,
                acq_contrast=acq_test, overall_contrast=overall,
                ret_contrast=contrast(common, "bound_final"))


def s2_mechanism(cells, probes):
    """Descriptive: starved-vs-bound e2e2 geometry, and p_kv under each mask."""
    rows = paired(cells, SO_ANCHOR, SO_ANCHOR, SO_SEEDS)
    bound = [probe(probes, c) for _, c, _ in rows if c["bound_final"]]
    unbound = [probe(probes, c) for _, c, _ in rows if not c["bound_final"]]
    pairs = [(probe(probes, x), probe(probes, a))
             for _, x, a in paired(cells, SO_KVM, SO_ANCHOR, SO_SEEDS)
             if probe(probes, x) and probe(probes, a)]
    return dict(
        e2e2_bound=dict(n=len(bound),
                        median_kv_write_eff=median_of([p["kv_write_eff"] for p in bound]),
                        median_p_kv=median_of([p["p_kv"] for p in bound])),
        e2e2_unbound=dict(n=len(unbound),
                          median_kv_write_eff=median_of([p["kv_write_eff"] for p in unbound]),
                          median_p_kv=median_of([p["p_kv"] for p in unbound])),
        p_kv_paired=dict(n=len(pairs),
                         median_kvm_p_kv=median_of([x["p_kv"] for x, _ in pairs]),
                         median_e2e2_p_kv=median_of([a["p_kv"] for _, a in pairs]),
                         median_difference=median_of([x["p_kv"] - a["p_kv"] for x, a in pairs]),
                         seeds_kvm_lower=sum(x["p_kv"] < a["p_kv"] for x, a in pairs)))


def ek_kind(label):
    """The final EK reading of a half label, or None when it is pending/untested/descriptive."""
    body = (label or "").partition("-")[2]
    for suffix in EK_FINAL:
        if body.startswith(suffix):
            return suffix
    return None


def ek_joint(labels):
    """The EK joint reading, written before any result.

    It needs both half labels final (registered amendment): blocked, retains, partial, no
    retention gain shown or inconclusive. Pending, untested and descriptive give no joint
    reading, and so does any pair of final labels that the pre-registered rows do not name.
    Over a sequence the halves do not decompose e2e2emb's correction (registered amendment), so
    a retaining half reads as "sufficient alone (added to first order)", not as carrying
    the resistance.
    """
    e, k = labels.get("E", ""), labels.get("K", "")
    ke, kk = ek_kind(e), ek_kind(k)
    if ke is None or kk is None:
        return f"no joint reading (E: {e}; K: {k})"
    e_retains, k_retains = ke == "retains", kk == "retains"
    if e_retains and not k_retains:
        return ("E-retains and not K-retains: the error-side meta-gradient is sufficient alone "
                "(added to first order) to resist starvation.")
    if k_retains and not e_retains:
        return ("K-retains and not E-retains: the forward-map meta-gradient is sufficient alone "
                "(added to first order). Read the code geometry (addr_a, pr_kv; qWv_cos is not "
                "in the R4-2 probe).")
    if e_retains and k_retains:
        return "Both retain: either half suffices."
    if ke == "blocked" and kk == "blocked":
        return ("Both blocked: the halves do not pass the first barrier even with the code. "
                "Descriptive.")
    if ke == "no retention gain shown" and kk == "no retention gain shown":
        return ("Neither retains while both acquire: retention needs both paths together, "
                "either through their per-update sum or through the cross-update cross terms.")
    return "No joint reading: see the per-half labels."


def fm_s2_joint(fm_label, s2_labels, a_label="holds", c_label="holds"):
    """The FM x S2-ret joint reading, written before any result.

    The gates come first: a failed V3a or V3c leaves no pre-registered reading
    (registered amendment). A hurting S2-ret has its own registered row; it has no reading
    with either FM label, and says so.
    """
    if a_label not in ("holds", "pending") or c_label not in ("holds", "pending"):
        return "descriptive"
    fm_label = _strip_descriptive(fm_label)
    ret = _strip_descriptive(s2_labels.get("ret"))
    if fm_label == "pending" or ret == "pending":
        return "pending"
    if ret == "S2-ret hurts":
        return "no pre-registered joint reading (S2-ret hurts); descriptive"
    if fm_label == "FM-rescue":
        if ret == "S2-ret rescue":
            return ("FM-rescue and S2-ret rescue: starvation is a failure mode shared by both "
                    "orders, and second order resists it only partly.")
        if ret == "no retention effect shown":
            return ("FM-rescue and S2-ret no effect: second order's own retention failures have "
                    "another cause, for example R2's post-spike collapse.")
        return "FM-rescue and S2-ret untestable: no reading for second-order retention."
    if ret in ("no retention effect shown", "untestable at this n"):
        return ("FM-no rescue shown and S2-ret no effect or untestable: the mask is not shown to "
                "matter in either order on this device. Read with R4-2 and R4-2L, descriptively.")
    if ret == "S2-ret rescue":
        return ("FM-no rescue shown and S2-ret rescue: descriptive; check p_kv in both mask arms.")
    return "pending"


# ----------------------------------------------------------------------- per-seed
def _per_seed(cells, probes, floor):
    """One row per seed: endpoints, pairing fields and the final probe of each arm."""
    rows = []
    for s in range(max(FO_SEEDS, SO_SEEDS)):
        row = {"seed": s}
        for arm in ARMS:
            c = cells.get((s, arm))
            ok = valid(c)
            row.update({f"{arm}_{key}": (c.get(key) if ok else None) for key in (
                "bound_ever", "bound_final", "collapse", "degraded", "partial", "mask",
                "masked_per_seq", "create_graph", "factors", "tf32", "peak_cuda_mb",
                "first_batch_sha1")})
            for condition in ("all", "all_minus_kv", "kv_only"):
                row[f"{arm}_ablation_{condition}"] = (c.get("ablation", {}).get(condition)
                                                        if ok else None)
            row[f"{arm}_ablation_ref_match"] = c.get("ablation_ref_match") if ok else None
            row[f"{arm}_last_bound_step"] = last_bound(c) if ok else None
            row[f"{arm}_lost_after_acquiring"] = lost(c) if ok else None
            if ok:
                p = probe(probes, c)
                row.update({f"{arm}_{k}": (p.get(k) if p else None) for k in (
                    "e_kv_abs", "p_kv", "kv_write_eff", "Wv_write_eff", "addr_a", "pr_kv",
                    "ce_kv", "ce_vk", "ce_vW", "eta_W1", "eta_b1", "eta_W2", "eta_b2")})
        for arm in (E_ARM, K_ARM):
            other = cells.get((s, ANCHOR))
            if valid(cells.get((s, arm))) and valid(other):
                row[f"{arm}_not_collapsed_where_anchor_is"] = (
                    cells[s, arm]["eval"]["query_acc"] >= .50 and
                    other["eval"]["query_acc"] < .50)
        rows.append(row)
    return rows


def _eta_rows(cells, probes):
    """Eta diagnostic (reported, not gating): X against the anchors, window and final."""
    anchors = complete(cells, ANCHOR, FO_SEEDS)
    acquired = [last_bound(c) for c in anchors if c["bound_ever"] and last_bound(c) is not None]
    # the fallback t_s is snapped to the probe grid, as in R4-2
    fallback = 25 * round(median(acquired) / 25) if acquired else None
    rows = []
    for s in range(FO_SEEDS):
        anchor = cells.get((s, ANCHOR))
        t_s = (last_bound(anchor) if valid(anchor) and anchor["bound_ever"] else None)
        if t_s is None:
            t_s = fallback
        anchor_steps = sorted(probes.get(anchor["cell_id"], {})) if valid(anchor) else []
        steps = [step for step in anchor_steps if t_s is not None and t_s <= step <= t_s + 150]
        for arm in (ANCHOR, E_ARM, K_ARM, REF_SO_EMB):
            c = cells.get((s, arm))
            if not valid(c):
                rows.append(dict(seed=s, arm=arm, t_s=t_s, window_probes=0))
                continue
            series = probes.get(c["cell_id"], {})
            window = [series[step] for step in steps if step in series]
            final = probe(probes, c)
            rows.append(dict(
                seed=s, arm=arm, t_s=t_s, window_probes=len(window),
                window_eta_W1=median_of([p.get("eta_W1") for p in window]),
                window_eta_b1=median_of([p.get("eta_b1") for p in window]),
                window_eta_W2=median_of([p.get("eta_W2") for p in window]),
                window_eta_b2=median_of([p.get("eta_b2") for p in window]),
                final_eta_W1=(final or {}).get("eta_W1"),
                final_eta_b1=(final or {}).get("eta_b1"),
                final_eta_W2=(final or {}).get("eta_W2"),
                final_eta_b2=(final or {}).get("eta_b2"),
                eta_final=c.get("eta_final")))
    return rows


# ----------------------------------------------------------------------- summarize
def summarize(cells, probes, refs, issues=(), gate=None):
    """All pre-registered R4-3 labels, contrasts, per-seed rows and diagnostics.

    ``gate`` is the E/K manipulation gate from ``records``; without it the gate is
    "unknown" and both halves are untested, which is what a log without an
    ``R43_EK_SHARE`` record must report.
    """
    gate = gate if gate is not None else ek_manipulation_gate(None)
    floor = refs.get("occ_minus", {}).get("acc")
    labels, details = {}, {}
    a_failed = c_failed = False

    labels["V3a"] = a_label = v3a(cells)
    labels["V3b"] = b_label = v3b(cells)
    labels["V3c"] = c_label = v3c(cells)
    if a_label not in ("holds", "pending"):
        a_failed = True
        labels["V3a_note"] = ("V3a did not hold: EK, FM, MS and both joint readings are "
                              "descriptive.")
    if c_label not in ("holds", "pending"):
        c_failed = True
        labels["V3c_note"] = ("V3c did not hold: there is no second-order deficit on this device, "
                              "S2 and FM_S2_joint are descriptive.")

    labels["FM"], details["FM"] = fm(cells)
    if a_failed and labels["FM"] != "pending":
        labels["FM"] += DESCRIPTIVE
    # Holm runs over the family {E, K} whatever the individual labels turn out to be
    raw = {half: (contrast(paired(cells, arm, ANCHOR, FO_SEEDS), "bound_final")["mcnemar_one_sided"]
                  if paired(cells, arm, ANCHOR, FO_SEEDS) else 1.0)
           for half, arm in HALVES.items()}
    adjusted = holm(raw)
    for half in HALVES:
        labels[half], details[half] = ek_half(cells, half, adjusted[half], a_label, gate)
        labels[f"MS1_{half}"], details[f"MS1_{half}"] = ms1(cells, probes, half, a_label, b_label)
        labels[f"MS2_{half}"], details[f"MS2_{half}"] = ms2(cells, half, floor, a_label)
        if labels[half].endswith("retains"):
            ms1_label = labels[f"MS1_{half}"]
            labels[f"MS_reading_{half}"] = (
                f"{half} resists starvation by keeping surprise at the write"
                if ms1_label is True else
                f"{half} keeps the binding by another route" if ms1_label is False else
                "pending" if ms1_label == "pending" else ms1_label)
    labels["EK_joint"] = "descriptive" if a_failed else ek_joint(labels)

    s2_labels = s2(cells, c_label)
    labels["S2-acq"] = s2_labels["acq"]
    labels["S2-ret"] = s2_labels["ret"]
    labels["S2-overall"] = s2_labels["overall"]
    details["S2"] = s2_labels
    details["S2_mechanism"] = s2_mechanism(cells, probes)
    accuracy = [x["eval"]["query_acc"] - a["eval"]["query_acc"]
                for _, x, a in paired(cells, SO_KVM, SO_ANCHOR, SO_SEEDS)]
    details["S2_wilcoxon"] = wilcoxon_one_sided(accuracy)
    labels["S2_wilcoxon"] = (f"descriptive; paired Wilcoxon on final query accuracy over "
                             f"{len(accuracy)} pairs, one-sided p="
                             f"{details['S2_wilcoxon']['p_one_sided']}")
    labels["FM_S2_joint"] = fm_s2_joint(labels["FM"], s2_labels, a_label, c_label)
    if _strip_descriptive(s2_labels["acq"]) in ("the mask helps second order acquire",
                                                "the mask hurts acquisition"):
        labels["S2_acq_note"] = ("S2-acq fired: the mask also acts on the first barrier (address) "
                                 "in second order, a different mechanism from starvation; "
                                 "reported separately.")
    counts = {arm: dict(n=len(good), ever=sum(bool(c["bound_ever"]) for c in good),
                        final=sum(bool(c["bound_final"]) for c in good),
                        collapsed=sum(bool(c["collapse"]) for c in good),
                        degraded=sum(bool(c["degraded"]) for c in good),
                        partial=sum(bool(c["partial"]) for c in good))
              for arm, good in ((arm, complete(cells, arm, seeds))
                                for arm, seeds in ((ANCHOR, FO_SEEDS), (KVM_FO, FO_SEEDS),
                                                   (E_ARM, FO_SEEDS), (K_ARM, FO_SEEDS),
                                                   (REF_SO_EMB, FO_SEEDS),
                                                   (SO_ANCHOR, SO_SEEDS), (SO_KVM, SO_SEEDS)))}
    return dict(labels=labels, details=details, issues=list(issues), counts=counts,
                n_complete=dict(first_order_emb=len(paired(cells, ANCHOR, KVM_FO, FO_SEEDS)),
                                ek_E=len(paired(cells, E_ARM, ANCHOR, FO_SEEDS)),
                                ek_K=len(paired(cells, K_ARM, ANCHOR, FO_SEEDS)),
                                ms1_E=len(paired(cells, E_ARM, REF_SO_EMB, FO_SEEDS)),
                                ms1_K=len(paired(cells, K_ARM, REF_SO_EMB, FO_SEEDS)),
                                second_order=len(paired(cells, SO_KVM, SO_ANCHOR, SO_SEEDS))),
                s2_labels=s2_labels, holm=adjusted, ek_manipulation=gate,
                floor_occ_minus=floor,
                per_seed=_per_seed(cells, probes, floor),
                eta_rows=_eta_rows(cells, probes))


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--s5-fp32", type=Path, default=None)
    parser.add_argument("--n-keys", type=int, default=16,
                        help="masked targets per sequence that a kv mask must produce")
    parser.add_argument("--ek-manipulation", type=Path, default=None,
                        help="replace the log's R43_EK_SHARE record with this json (an "
                             "R43_EK_SHARE record or a bare {\"E\", \"K\"} gate)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    override = ({r["cell_id"]: r["passed"] for r in json.loads(args.s5_fp32.read_text())}
                if args.s5_fp32 else None)
    cells, probes, refs, issues, gate = records(args.log, override, args.n_keys)
    if args.ek_manipulation:
        given = json.loads(args.ek_manipulation.read_text())
        gate = ek_manipulation_gate(given) if "per_seed" in given else dict(
            gate, source=str(args.ek_manipulation), status="override",
            **{half: given.get(half) for half in HALVES})
    result = summarize(cells, probes, refs, issues, gate)
    (args.out / "summary.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k not in ("per_seed", "eta_rows")},
        indent=2, sort_keys=True, default=str) + "\n")
    write_csv(args.out / "paired_seeds.csv", result["per_seed"])
    write_csv(args.out / "eta_diagnostic.csv", result["eta_rows"])
    write_csv(args.out / "labels.csv", [dict(label=k, value=v)
                                        for k, v in result["labels"].items()])
    print(json.dumps({"labels": result["labels"], "n_complete": result["n_complete"],
                      "issues": result["issues"], "counts": result["counts"],
                      "holm": result["holm"], "ek_manipulation": result["ek_manipulation"]},
                     sort_keys=True, default=str))


if __name__ == "__main__":
    main()