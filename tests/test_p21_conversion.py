"""Offline exhaustive official tensor mapping; Orbax/JAX are optional isolated tools."""
from pathlib import Path

import numpy as np
import pytest

from tttmem.p2.convert import map_weights, expected_shapes, model_weights_dir
from tttmem.p2.model import PRESETS


def _official_tree():
    path = Path(__file__).parent / "jax_parity" / "official_tiny.npz"
    with np.load(path, allow_pickle=False) as z:
        flat = {}
        cfg = PRESETS["tiny2"]
        flat["language_model.model.wte.weight"] = z["param/wte.weight"].copy()
        flat["language_model.model.ln_f.weight"] = z["param/ln_f.weight"].copy()
        for name in expected_shapes(cfg):
            if not name.startswith("layers.0."):
                continue
            tail = name[len("layers.0."):]
            source = tail.replace("attn.", "seq_modeling_block.")
            flat["language_model.model.h.blocks." + source] = np.stack([
                z[f"param/layers.{i}.{tail}"] for i in range(cfg.num_layers)])
        start = cfg.num_layers - cfg.suffix_len
        for name in expected_shapes(cfg):
            if not name.startswith(f"layers.{start}."):
                continue
            tail = name[len(f"layers.{start}."):]
            if "prime" in tail:
                flat["language_model.model.h.prime_storage." + tail] = np.stack([
                    z[f"param/layers.{i}.{tail}"] for i in range(start, cfg.num_layers)])
    root = {}
    for name, value in flat.items():
        node = root
        parts = name.split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value
    return root


def test_official_mapping_is_exhaustive_and_has_no_transposes():
    state, mapping, count = map_weights(_official_tree(), PRESETS["tiny2"])
    with np.load(Path(__file__).parent / "jax_parity" / "official_tiny.npz") as z:
        for name, tensor in state.items():
            np.testing.assert_array_equal(tensor.numpy(), z["param/" + name])
    assert count == sum(t.numel() for t in state.values())
    assert len(mapping) == len(state)


@pytest.mark.parametrize("bad", ["missing", "shape", "extra", "nonfinite"])
def test_converter_rejects_invalid_parameter_tree(bad):
    tree = _official_tree()
    m = tree["language_model"]["model"]
    if bad == "missing":
        del m["wte"]
    elif bad == "shape":
        m["wte"]["weight"] = m["wte"]["weight"][:-1]
    elif bad == "extra":
        m["lm_head"] = np.ones(3, dtype=np.float32)
    else:
        m["wte"]["weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        map_weights(tree, PRESETS["tiny2"])


def test_official_converter_never_downloads_and_shape_count_is_125m():
    with pytest.raises(ValueError, match="local downloaded"):
        model_weights_dir("gs://ttt-e2e-checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc")
    assert sum(int(np.prod(s)) for s in expected_shapes(PRESETS["125m"]).values()) == 184363776
