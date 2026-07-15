from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config128:
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset_root: str = "data_128"
    image_size: int = 128
    max_samples: Optional[int] = 20000

    # ------------------------------------------------------------------ #
    # Conditioner MLP
    # ------------------------------------------------------------------ #
    cond_input_dim: int = 10                  # 3 one-hot + 7 scalars
    cond_hidden_dim: int = 10
    cond_output_dim: int = 10                # must equal cross_attention_dim

    # ------------------------------------------------------------------ #
    # UNet  (pixel-space — operates directly on 128x128 RGB)
    # ------------------------------------------------------------------ #
    unet_in_channels: int = 3
    unet_out_channels: int = 3
    unet_block_out_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 128]
    )
    unet_layers_per_block: int = 2
    unet_attention_head_dim: int = 2
    cross_attention_dim: int = 10
    unet_down_block_types: List[str] = field(
        default_factory=lambda: [
            "DownBlock2D",            # 128x128 → 64x64   no attention
            "DownBlock2D",            # 64x64   → 32x32   no attention
            "CrossAttnDownBlock2D",   # 32x32   → 16x16   attention
            "CrossAttnDownBlock2D",   # 16x16   → 8x8     attention
        ]
    )
    unet_up_block_types: List[str] = field(
        default_factory=lambda: [
            "CrossAttnUpBlock2D",     # 8x8     → 16x16   attention
            "CrossAttnUpBlock2D",     # 16x16   → 32x32   attention
            "UpBlock2D",              # 32x32   → 64x64   no attention
            "UpBlock2D",              # 64x64   → 128x128 no attention
        ]
    )

    # ------------------------------------------------------------------ #
    # Diffusion scheduler
    # ------------------------------------------------------------------ #
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    batch_size: int = 64                      # 128x128 is small, can afford larger batch
    gradient_accumulation_steps: int = 1
    num_epochs: int = 200
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    use_torch_compile: bool = False
    num_workers: int = 0
    cfg_dropout_prob: float = 0.05

    # ------------------------------------------------------------------ #
    # EMA
    # ------------------------------------------------------------------ #
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100

    # ------------------------------------------------------------------ #
    # Logging & checkpointing
    # ------------------------------------------------------------------ #
    output_dir: str = "outputs_128_pixel_20k"
    wandb_project: str = "shape-diffusion"
    wandb_run_name: str = "run-128"
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    num_validation_samples: int = 4