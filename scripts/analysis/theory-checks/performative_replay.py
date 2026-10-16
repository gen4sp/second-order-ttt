#!/usr/bin/env python3
"""Performative replay agreement on the toy itself (float64).

The probe's manual replay of the inner loop (writes recorded, then W_c = W0 - eta * sum of earlier
writes) recomputes the toy's own first-order and second-order training losses on the same batch;
this collects, per checkpoint, the relative differences |L_toy - L_replay| / |L_toy| and the largest
difference between the probe's query logits and the toy's own evaluation, for the 45 registered
checkpoints (sets fo_pre_loss, fo_early_post_loss, fo_final, fo_linkv_final, so_final) and the three
compatibility-gate (development) checkpoints, with their maxima.
Inputs (probe outputs of the toy checkpoints, not included in the repository):
- PROBE_DIR/*.json, one per registered checkpoint: ``label`` (first letter = set code E/P/S/F/L),
  ``ckpt`` (checkpoint file name echo__s<seed>__<arm>__eta<eta>__st1500[__step<step>].pt), ``arm``,
  ``checks`` with ``gamma1_fo``, ``gamma1_so``, ``logits_vs_eval``;
- SANITY_DIR/sanity_{r42l_anchor_s2,r41_so_s0,r42l_linkv_s2}.json with ``ckpt`` and ``G3_checks``
  (same three fields);
- RESULTS_JSON: the per-checkpoint probe summary, whose ``identities`` maxima are asserted to match.
Writes OUT-DIR/theory-checks/performative_replay_agreement.json.
"""
import argparse
import json
import os
import re
from pathlib import Path

OUT = {}  # relpath -> text


def json_text(obj):
    return json.dumps(obj, indent=1, allow_nan=False) + "\n"



SET_NAMES = {"E": "fo_early_post_loss", "P": "fo_pre_loss", "S": "so_final", "F": "fo_final", "L": "fo_linkv_final"}


def ckpt_meta(path):
    b = os.path.basename(path)
    m = re.match(r"echo__s(\d+)__([a-z0-9_]+?)__eta(\d+)__st1500(?:__step(\d+))?\.pt$", b)
    assert m, b
    return int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4) or 1500)


def item2(results_json, probe_dir, sanity_dir):
    r = json.loads(Path(results_json).read_text())
    reg = []
    for f in sorted(Path(probe_dir).glob("*.json")):
        d = json.loads(f.read_text())
        seed, arm, eta, step = ckpt_meta(d["ckpt"])
        assert arm == d["arm"]
        c = d["checks"]
        reg.append({"set": SET_NAMES[d["label"][0]], "arm": arm, "eta": eta, "seed": seed, "step": step,
                    "fo_loss_rel_diff": c["gamma1_fo"], "so_loss_rel_diff": c["gamma1_so"],
                    "eval_logits_max_abs_diff": c["logits_vs_eval"]})
    order = list(SET_NAMES.values())
    reg.sort(key=lambda x: (order.index(x["set"]), x["arm"], x["seed"]))
    assert len(reg) == 45
    gate = []
    for name in ("sanity_r42l_anchor_s2.json", "sanity_r41_so_s0.json", "sanity_r42l_linkv_s2.json"):
        d = json.loads((Path(sanity_dir) / name).read_text())
        seed, arm, eta, step = ckpt_meta(d["ckpt"])
        g = d["G3_checks"]
        gate.append({"arm": arm, "eta": eta, "seed": seed, "step": step,
                     "fo_loss_rel_diff": g["gamma1_fo"], "so_loss_rel_diff": g["gamma1_so"],
                     "eval_logits_max_abs_diff": g["logits_vs_eval"]})
    ident = r["identities"]
    assert ident["gamma1_fo"]["max"] == max(x["fo_loss_rel_diff"] for x in reg)
    assert ident["gamma1_so"]["max"] == max(x["so_loss_rel_diff"] for x in reg)
    out = {
        "description": ("Replay check on the toy itself, float64. The probe's manual replay of the inner loop "
                        "(writes recorded, then W_c = W0 - eta * sum of earlier writes) recomputes the toy's own "
                        "first-order and second-order training losses on the same batch (the first evaluation "
                        "batch of each checkpoint, 16 sequences). Values are relative differences "
                        "|L_toy - L_replay| / |L_toy|. "
                        "eval_logits_max_abs_diff compares the probe's query logits with the toy's own evaluation."),
        "dtype": "float64",
        "summary": {
            "registered_checkpoints": {"n": len(reg),
                                       "fo_loss_rel_diff_max": ident["gamma1_fo"]["max"],
                                       "so_loss_rel_diff_max": ident["gamma1_so"]["max"],
                                       "eval_logits_max_abs_diff_max": ident["logits_vs_eval"]["max"]},
            "compatibility_gate_checkpoints": {"n": len(gate),
                                               "fo_loss_rel_diff_max": max(x["fo_loss_rel_diff"] for x in gate),
                                               "so_loss_rel_diff_max": max(x["so_loss_rel_diff"] for x in gate),
                                               "eval_logits_max_abs_diff_max": max(x["eval_logits_max_abs_diff"] for x in gate)},
        },
        "registered_checkpoints": reg,
        "compatibility_gate_checkpoints": gate,
    }
    print("item2 reg max", out["summary"]["registered_checkpoints"], "gate max", out["summary"]["compatibility_gate_checkpoints"])
    print("item2 reg rows >2e-16:", [(x["set"], x["arm"], x["seed"], x["fo_loss_rel_diff"]) for x in reg if x["fo_loss_rel_diff"] > 2e-16])
    OUT["theory-checks/performative_replay_agreement.json"] = json_text(out)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_json", help="per-checkpoint probe summary (JSON with identities)")
    ap.add_argument("probe_dir", help="per-checkpoint probe outputs (*.json)")
    ap.add_argument("sanity_dir", help="compatibility-gate probe outputs (sanity_*.json)")
    ap.add_argument("--out-dir", default="outputs/analysis", help="written to OUT-DIR/<results path>")
    ap.add_argument("--process-name", default=os.environ.get("PROCESS_NAME", "performative-replay"))
    args = ap.parse_args()
    try:
        import setproctitle
        setproctitle.setproctitle(args.process_name)
    except ImportError:
        pass
    item2(args.results_json, args.probe_dir, args.sanity_dir)
    for rel, text in OUT.items():
        p = Path(args.out_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print("WRITE", p, len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
