from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConfigUncond:
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset_root: str = "data"
    image_size: int = 256
    max_samples: Optional[int] = None

    # ------------------------------------------------------------------ #
    # VAE
    # ------------------------------------------------------------------ #
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    vae_scale_factor: int = 8
    latent_channels: int = 4

    @property
    def latent_size(self) -> int:
        return self.image_size // self.vae_scale_factor  # 32

    # ------------------------------------------------------------------ #
    # UNet (no conditioning — UNet2DModel, not UNet2DConditionModel)
    # ------------------------------------------------------------------ #
    unet_block_out_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 128]
    )
    unet_layers_per_block: int = 2

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
    num_epochs: int = 50
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    num_workers: int = 0

    # ------------------------------------------------------------------ #
    # EMA
    # ------------------------------------------------------------------ #
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_after_step: int = 100

    # ------------------------------------------------------------------ #
    # Logging & checkpointing
    # ------------------------------------------------------------------ #
    output_dir: str = "outputs_uncond"
    wandb_project: str = "shape-diffusion"
    wandb_run_name: str = "uncond"
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10
