"""Phase 2 entry: prepare, profile, smoke or one continuation train arm.

``profile`` is the P2-0 GPU measurement; ``train`` is one P2-1 arm; ``swap`` is the
P2-2 writer/reader decomposition on saved P2-1 checkpoints (eval only).
"""

from __future__ import annotations

import os

from .p2.data import _set_process_name


def run(cfg: dict, seed: int) -> dict:
    if cfg.get("mode") == "prepare_code":
        from .p2.data import prepare_code
        return prepare_code(cfg)
    name = cfg.get("process_name") or os.environ.get("PROCESS_NAME")
    if name:
        _set_process_name(name)
    mode = cfg.get("mode", "profile")
    if mode == "prepare":
        from .p2.data import PrepareCfg, prepare_config
        return prepare_config(PrepareCfg.from_dict(cfg))
    if mode == "smoke":
        from .p2.smoke import smoke
        return smoke(cfg)
    if mode == "profile":
        from .p2.profile import ProfileCfg, profile
        return profile(ProfileCfg.from_dict(cfg))
    if mode == "calibrate":
        from .p2.calibrate import CalibCfg, calibrate
        return calibrate(CalibCfg.from_dict(cfg))
    if mode == "train":
        from .p2.train import TrainCfg, train
        return train(TrainCfg.from_dict(cfg))
    if mode == "swap":
        from .p2.swap import run_config
        return run_config(cfg)
    if mode == "mini_smoke":
        from .p2.mini import mini_smoke
        return mini_smoke(cfg)
    if mode == "mini_profile":
        from .p2.mini import mini_profile
        return mini_profile(cfg)
    if mode == "mini_podsmoke":
        from .p2.podsmoke import run
        return run(cfg)
    if mode == "mini_matrix":
        from .p2.matrix import run
        return run(cfg)
    if mode == "mini_ckpt_eval":
        from .p2.mini import mini_ckpt_eval
        return mini_ckpt_eval(cfg)
    if mode == "mini_align":
        from .p2.mini import mini_align
        return mini_align(cfg)
    if mode == "mini_g0":
        from .p2.mini import mini_g0
        return mini_g0(cfg)
    if mode == "soq_n3b_fo_cm":
        from .p2.soq_n3b import soq_n3b_fo_cm
        return soq_n3b_fo_cm(cfg)
    if mode == "s14_pod_job":
        from .p2.s14 import pod_job
        return pod_job(cfg)
    if mode == "soq_n3b_reads":
        from .p2.soq_n3b import soq_n3b_reads
        return soq_n3b_reads(cfg)
    if mode == "soq_v1":
        from .p2.v1 import run as v1_run
        return v1_run(cfg)
    if mode == "soq_p1v3_post":
        # Evaluation-only dispatch.  Training keys in the copied v3x config are
        # deliberately inert because this branch never imports/calls train().
        from .p2.soq_p1v3_post import run as post_run
        return post_run(cfg)
    if mode == "mini_late":
        from .p2.mini import mini_late
        return mini_late(cfg)
    raise ValueError(f"unknown p2 mode {mode!r}")
