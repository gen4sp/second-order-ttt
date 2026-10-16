"""Config dataclasses mirroring the official JAX TTT-E2E implementation."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    # Inner-loop chunk size in tokens (TTT "mini batch")
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    # Number of trailing "suffix" layers that carry fast weights and run chunk-wise
    suffix_len: int = 0
    # Whether suffix layers get a feed_forward_prime fast-weight MLP
    prime: bool = False
    # "self_attention" (full causal) | "SWA" (prefix: full sliding window, suffix: chunked)
    seq_modeling_block: str = "self_attention"


@dataclass
class InnerOptConfig:
    lr: float = 1.0
    clip_gradient: float = 1.0


@dataclass
class OuterOptConfig:
    lr: float = 3e-3
    init_lr: float = 0.0
    end_lr: float = 1e-5
    lr_warmup_steps: int = 480
    lr_decay_steps: int = 4800
    b1: float = 0.9
    b2: float = 0.95
    clip_gradient: float = 1.0
    weight_decay: float = 0.1


@dataclass
class TrainingConfig:
    train_mode: str = "meta"  # "pretrain" (plain LM) | "meta" (TTT-E2E)
    seq_length: int = 8192
    global_batch_size: int = 64
    accum_steps: int = 1
    total_steps: int = 4800
    # Inner-LR warmup: multiplier ramps ilr_init -> 1.0 over ilr_warmup_steps outer steps
    ilr_warmup_steps: int = 480
    ilr_init: float = 0.1
    model_seed: int = 0
    data_seed: int = 0
    save_freq: int = 1000
    log_freq: int = 10
    dataset_path: str = ""  # flat token .npy memmap; empty => dummy random tokens
    exp_dir: str = "./experiments"
    exp_name: str = "dev"
    optimizer_inner: InnerOptConfig = field(default_factory=InnerOptConfig)
    optimizer_outer: OuterOptConfig = field(default_factory=OuterOptConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def preset_125m_e2e() -> Config:
    """configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml"""
    cfg = Config()
    cfg.model = ModelConfig(
        num_hidden_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=1664,
        seq_modeling_block="SWA",
        mini_batch_size=1024,
        sliding_window_size=8192,
        rope_theta=500000.0,
        prime=True,
        suffix_len=3,
        tie_word_embeddings=True,
    )
    cfg.training = TrainingConfig(
        train_mode="meta",
        seq_length=8192,
        global_batch_size=64,
        total_steps=4800,
        ilr_warmup_steps=480,
        ilr_init=0.1,
        optimizer_inner=InnerOptConfig(lr=1.0, clip_gradient=1.0),
        optimizer_outer=OuterOptConfig(
            lr=3e-3, lr_warmup_steps=480, lr_decay_steps=4800, end_lr=1e-5
        ),
        exp_name="pretrain-125m-e2e-pt",
    )
    return cfg


def preset_125m_fa() -> Config:
    """Full-attention baseline (pretrain-125m-fa): plain transformer, no TTT."""
    cfg = preset_125m_e2e()
    cfg.model.seq_modeling_block = "self_attention"
    cfg.model.prime = False
    cfg.model.suffix_len = 0
    cfg.model.intermediate_size = 2048
    cfg.training.train_mode = "pretrain"
    cfg.training.exp_name = "pretrain-125m-fa-pt"
    return cfg


def preset_125m_e2e_ext32k() -> Config:
    """configs/experiment/125m/extension/ext-125m-e2e-32K.yaml (Books -> PG-19)."""
    cfg = preset_125m_e2e()
    cfg.training.seq_length = 32768
    cfg.training.global_batch_size = 32
    cfg.training.total_steps = 120
    cfg.training.ilr_warmup_steps = 0
    cfg.training.ilr_init = 1.0
    cfg.training.optimizer_outer = OuterOptConfig(
        lr=4e-4, lr_warmup_steps=12, lr_decay_steps=120, end_lr=1e-5
    )
    cfg.training.exp_name = "ext-125m-e2e-32k-pt"
    return cfg


def preset_760m_e2e() -> Config:
    """Official 760m e2e config, reduced budget: 8000 steps (4.2B tokens)
    instead of 29000 (1x Chinchilla) — scale-trend check, not headline runs."""
    cfg = Config()
    cfg.model = ModelConfig(
        num_hidden_layers=24,
        hidden_size=1536,
        num_attention_heads=16,
        intermediate_size=3328,
        seq_modeling_block="SWA",
        mini_batch_size=1024,
        sliding_window_size=8192,
        rope_theta=500000.0,
        prime=True,
        suffix_len=6,
        tie_word_embeddings=True,
    )
    cfg.training = TrainingConfig(
        train_mode="meta",
        seq_length=8192,
        global_batch_size=64,
        total_steps=8000,
        ilr_warmup_steps=800,
        ilr_init=0.1,
        optimizer_inner=InnerOptConfig(lr=1.0, clip_gradient=1.0),
        optimizer_outer=OuterOptConfig(
            lr=1.25e-3, lr_warmup_steps=800, lr_decay_steps=8000, end_lr=1e-5
        ),
        exp_name="pretrain-760m-e2e-pt",
    )
    return cfg


def preset_760m_e2e_ext32k() -> Config:
    """Official ext-760m-e2e-32K (Books -> PG-19)."""
    cfg = preset_760m_e2e()
    cfg.training.seq_length = 32768
    cfg.training.global_batch_size = 32
    cfg.training.total_steps = 725
    cfg.training.ilr_warmup_steps = 0
    cfg.training.ilr_init = 1.0
    cfg.training.optimizer_outer = OuterOptConfig(
        lr=4e-4, lr_warmup_steps=72, lr_decay_steps=725, end_lr=1e-5
    )
    cfg.training.exp_name = "ext-760m-e2e-32k-pt"
    return cfg


def preset_3b_e2e_ext128k() -> Config:
    """Official ext-3b-e2e-128K (released checkpoint
    gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc)."""
    cfg = Config()
    cfg.model = ModelConfig(
        num_hidden_layers=32,
        hidden_size=2560,
        num_attention_heads=32,
        intermediate_size=5632,
        seq_modeling_block="SWA",
        mini_batch_size=1024,
        sliding_window_size=8192,
        rope_theta=500000.0,
        prime=True,
        suffix_len=8,
        tie_word_embeddings=True,
    )
    cfg.training = TrainingConfig(
        train_mode="meta",
        seq_length=131072,
        global_batch_size=16,
        total_steps=1300,
        ilr_warmup_steps=0,
        ilr_init=1.0,
        optimizer_inner=InnerOptConfig(lr=1.0, clip_gradient=1.0),
        optimizer_outer=OuterOptConfig(
            lr=4e-4, lr_warmup_steps=130, lr_decay_steps=1300, end_lr=1e-5
        ),
        exp_name="official-3b-e2e-128k",
    )
    return cfg


def preset_1b_e2e_books8k() -> Config:
    """Official 1b_ttt_e2e_finetune_books_8k_1x_cc.

    intermediate_size is 4224, read off the checkpoint. The official YAML says
    4352; the model-construction code evidently rounds it, and the value that
    matters is the one in the weights. With 4352 every FFN tensor has the wrong
    shape; PyTorch rejects same-name shape mismatches even with strict=False.
    The conversion-time model-load check therefore guards this preset.
    """
    cfg = preset_3b_e2e_ext128k()
    cfg.model.num_hidden_layers = 24
    cfg.model.hidden_size = 2048
    cfg.model.num_attention_heads = 32
    cfg.model.intermediate_size = 4224
    cfg.model.suffix_len = 6
    cfg.training.seq_length = 8192
    cfg.training.exp_name = "official-1b-e2e-books8k"
    return cfg


def preset_1b_e2e_dclm8k() -> Config:
    """Official 1b_ttt_e2e_pretrain_dclm_8k_1x_cc.

    Same architecture as the books finetune; the point of the separate entry is
    the weights. DCLM is web-crawl data and can contain public-domain book
    mirrors, so PG-19 must be checked for contamination per checkpoint rather
    than assumed to be held out from the dataset name.
    """
    cfg = preset_1b_e2e_books8k()
    cfg.training.exp_name = "official-1b-e2e-dclm8k"
    return cfg


def preset_3b_e2e_dclm8k() -> Config:
    """Official 3b_ttt_e2e_pretrain_dclm_8k_3x_cc: the 8K pretrain, not the
    128K books extension. Architecture is identical to the latter."""
    cfg = preset_3b_e2e_ext128k()
    cfg.training.seq_length = 8192
    cfg.training.exp_name = "official-3b-e2e-dclm8k"
    return cfg


def preset_760m_official() -> Config:
    """Official 760m_ttt_e2e_pretrain_dclm_8k_1x_cc."""
    cfg = preset_3b_e2e_ext128k()
    cfg.model.num_hidden_layers = 24
    cfg.model.hidden_size = 1536
    cfg.model.num_attention_heads = 16
    cfg.model.intermediate_size = 3328
    cfg.model.suffix_len = 6
    cfg.training.seq_length = 8192
    cfg.training.exp_name = "official-760m-e2e"
    return cfg


PRESETS = {
    "125m-e2e": preset_125m_e2e,
    "125m-fa": preset_125m_fa,
    "125m-e2e-ext32k": preset_125m_e2e_ext32k,
    "760m-e2e": preset_760m_e2e,
    "760m-e2e-ext32k": preset_760m_e2e_ext32k,
    "official-3b-ext128k": preset_3b_e2e_ext128k,
    "official-1b-books8k": preset_1b_e2e_books8k,
    "official-1b-dclm8k": preset_1b_e2e_dclm8k,
    "official-3b-dclm8k": preset_3b_e2e_dclm8k,
    "official-760m": preset_760m_official,
}
