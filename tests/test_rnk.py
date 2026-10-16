"""R4-0 geometry, decisions and tiny CPU integration."""

import numpy as np
import torch

from tttmem import exp_rnk as rnk


def test_geometry_orthogonal_and_constant():
    eye = torch.eye(4).unsqueeze(0)
    g = rnk._geometry(eye)
    np.testing.assert_allclose(g["pr"], [3], atol=1e-10)
    np.testing.assert_allclose(g["r99"], [3])
    np.testing.assert_allclose(g["common"], [.25])
    np.testing.assert_allclose(g["offdiag_cos"], [0])
    flat = rnk._geometry(torch.ones(1, 4, 4))
    assert flat["pr"][0] == flat["r99"][0] == 0
    assert flat["common"][0] == flat["offdiag_cos"][0] == 1


def test_margin_uses_sequence_median_absolute_dot():
    dot = torch.tensor([[[4., 1., 0.], [0., 3., 2.], [1., 0., 5.]]])
    # median abs(dot)=1; query margins are 3, 1, 4.
    torch.testing.assert_close(rnk._margin(dot), torch.tensor([3.]))


def test_verdict_thresholds_and_joint_wv_condition():
    refs = {"chance": {"acc": .0625}, "occ": {"acc": .19},
            "occ_minus": {"acc": .32}}
    bound = [dict(label=f"b{i}", stratum="bound", pr_kv=10., acc_kv_only=.95)
             for i in range(10)]
    collapsed = [dict(label=f"c{i}", stratum="collapsed_e2e2", pr_kv=4.,
                      addr_a=.06, acc_Wv_only=.19, occ_hit_Wv_only=.9, acc_kv_only=.1)
                 for i in range(8)]
    first = [dict(label=f"f{i}", stratum="e2e1", pr_kv=9., addr_a=.06)
             for i in range(12)]
    verdict = rnk._verdicts(bound + collapsed + first, refs)
    assert verdict["bound"]["kv_reverse_pass"]
    assert verdict["collapsed_e2e2"]["kv_verdict"] == "Collapsed"
    assert verdict["e2e1"]["kv_verdict"] == "Scrambled"
    assert verdict["collapsed_e2e2"]["wv_verdict"] == "Wv-supplies"
    for i in range(3):
        collapsed[i]["acc_Wv_only"] = .5
    verdict = rnk._verdicts(bound + collapsed + first, refs)["collapsed_e2e2"]
    assert verdict["wv_verdict"] == "Not Wv-supplies"
    assert "within_occ" in verdict["wv_failing_conditions"]


def test_tiny_random_run(tmp_path, monkeypatch):
    from tttmem import exp_ttt as ttt
    # Keep run()'s fixed echo layout but shrink only the network and eval count.
    monkeypatch.setattr(rnk, "_reference", lambda path: {})
    cfg = {"process_name": "TTM-r40-03-test", "device": "cpu", "smoke": True,
           "ckpt_root": str(tmp_path), "kag_reference": "unused",
           "d_model": 16, "n_layers": 1, "n_heads": 2, "fast_mult": 2,
           "backbone_mult": 2, "echo": {"T": 512, "n_keys": 16, "n_eval": 2, "eval_batch": 2},
           "checkpoints": [{"label": "random-init", "stratum": "random", "arm": "e2e2",
                            "eta_base": 10}]}
    result = rnk.run(cfg, 0)
    assert result["smoke"]
    row = result["models"][0]
    assert 0 <= row["pr_kv"] <= 15
    assert 0 <= row["addr_a"] <= 1
    assert np.isfinite(row["acc_kv_only"])
