from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConfigMLP:
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset_root: str = "data"
    image_size: int = 256
    max_samples: Optional[int] = None

    # ------------------------------------------------------------------ #
    # VAE  (frozen — used only for encode / decode)
    # ------------------------------------------------------------------ #
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    vae_scale_factor: int = 8                 # 256 → 32 latents
    latent_channels: int = 4

    @property
    def latent_size(self) -> int:
        return self.image_size // self.vae_scale_factor   # 32

    # ------------------------------------------------------------------ #
    # Conditioner MLP
    # LayerNorm → Linear(10→128) → SiLU → Linear(128→128) → SiLU → Linear(128→128)
    # output: (B, 1, 128) — single token for cross-attention
    # ------------------------------------------------------------------ #
    cond_input_dim: int = 10                  # 3 one-hot + 7 scalars
    cond_hidden_dim: int = 64
    cond_output_dim: int = 64                 # must equal cross_attention_dim

    # ------------------------------------------------------------------ #
    # UNet  (latent-space — operates on 32x32 latents)
    # ------------------------------------------------------------------ #
    unet_in_channels: int = 4                 # VAE latent channels
    unet_out_channels: int = 4
    unet_block_out_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 128]
    )
    unet_layers_per_block: int = 2
    unet_attention_head_dim: int = 8
    cross_attention_dim: int = 64             # must equal cond_output_dim
    unet_down_block_types: List[str] = field(
        default_factory=lambda: [
            "DownBlock2D",            # 32x32 → 16x16  no attention
            "CrossAttnDownBlock2D",   # 16x16 → 8x8    attention
            "CrossAttnDownBlock2D",   # 8x8   → 4x4    attention
            "DownBlock2D",            # bottleneck
        ]
    )
    unet_up_block_types: List[str] = field(
        default_factory=lambda: [
            "UpBlock2D",              # bottleneck
            "CrossAttnUpBlock2D",     # 4x4   → 8x8    attention
            "CrossAttnUpBlock2D",     # 8x8   → 16x16  attention
            "UpBlock2D",              # 16x16 → 32x32  no attention
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
    batch_size: int = 32
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
    output_dir: str = "outputs_mlp"
    wandb_project: str = "shape-diffusion"
    wandb_run_name: str = "run-mlp"
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    num_validation_samples: int = 4