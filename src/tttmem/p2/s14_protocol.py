"""Frozen S14 read protocol; NumPy analysis may import this without torch."""

PASSKEY_ETAS = {'so': [1, 1.5, 2, 2.5], 'fo': [1, 2, 3, 4, 5, 6, 7, 8, 10]}
SO_GRID = [.75, .9, 1, 1.1, 1.25, 1.4, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4]
FO_GRID = [.5, .75, 1, 1.25, 1.5, 2, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12]

# SOQ-N3B n_cm.json: fastest fitting warm-cache pilot settings.
N3B_SO_PILOT_S = {1: 7.19176319334656, 2: 6.607372670434415}
N3B_SO_S = N3B_SO_PILOT_S[2]
N3B_FO_PILOT_S = {2: 2.434334844350815, 4: 2.2854128843173385}
N3B_FO_S = N3B_FO_PILOT_S[4]  # Fastest setting; retained for A0 compatibility.

# compute.json v2: seconds per complete global-batch step, never rounded.
COMPUTE_FIELDS = ('arm', 'seed', 'base_steps', 'expected_steps', 'micro_batch',
                  'so_micro_batch', 'pilot_source', 'pilot_micro_batch',
                  'pilot_so_micro_batch', 't_so_pilot_s', 't_arm_pilot_s',
                  'device_name', 'torch_version', 'cuda_version', 'runpod_pod_id')
PASSKEY_PROTOCOL = 's14-passkey-v3-digits-source-ablation'
