# Official TTT-E2E reference fixture

`official_tiny.npz` is a numerical reference produced with the official TTT-E2E JAX code
([test-time-training/e2e](https://github.com/test-time-training/e2e) at commit
`a4fc4788ace38e29b5067916d4f4be33da894085`). A tiny, randomly initialised official `MetaModel`
(4 layers, width 32, 4 heads, vocabulary 101; layers 2–3 carry the fast `feed_forward_prime`
weights; inner chunk 8) was run on two random 32-token sequences through the official inner loop
(SGD, learning rate 1, global-norm clip 1) on one CPU in fp32, with JAX/jaxlib 0.5.3, Equinox
0.12.1 and Optax 0.2.4. On CPU the prefix attention uses the XLA implementation without CUDA
sharding, and the RoPE table holds 128 positions.

The file stores, named and laid out for `src/tttmem/p2/model.py`:

- `param/*`: the 64 parameter tensors;
- `input_ids`, `target_ids`, `loss_mask`;
- `loss_so`, `loss_fo`, `seq_loss_so`, `chunk_loss_so`, `token_nll` (unmasked per-token NLL);
- `grad_so/*`: outer gradients through the unmodified inner loop (second order);
- `grad_fo/*`: outer gradients with a stop-gradient on every inner update (first order);
- `inner_norms`, `inner_norms_all`, `inner_update_delta_norms`, `fast_so/*`: pre-clip inner
  gradient norms, update sizes and the fast weights after each update;
- `config_json`: every setting, the package versions and the SHA-256 of each upstream source
  file read and of the port module at generation time.

`mapping.json` lists the 64 official-to-port parameter pairs (slices of stacked layer leaves, no
transposes). `verification.json` records a central finite-difference check of the second-order
gradient against the unmodified official inner loop, along two directions (all parameters; the
fast-weight matrices only) at two step sizes. All four agree with the second-order prediction
within 1% and differ clearly from the first-order one. It also holds the fixture's SHA-256.
`summary.txt` is the printed summary of the generating run, with every tensor's SO and FO
gradient norm.

Tests that read the fixture:

- `tests/test_p2.py::test_matches_official_jax`: the port's token NLL, loss and SO and FO
  gradients against the fixture;
- `tests/test_p21_conversion.py`: the checkpoint converter's official-to-port mapping.

The generator is not distributed, because it adapts code from the upstream repository, which
states no licence.
