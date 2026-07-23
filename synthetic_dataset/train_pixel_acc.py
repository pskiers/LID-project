"""
train_pixel_64.py
-----------------
Pixel-space diffusion training for 64x64 shape images with full Accelerate.
No VAE, no CFG.
Conditioning: linear projection of 10-float vector to single 64-dim token.

Aligned with accelerate version:
  - Full Accelerator wrapping (mixed precision, gradient accumulation, device)
  - Beta schedule: "linear"
  - EMA: diffusers EMAModel with warmup
  - AdamW: beta1=0.95, beta2=0.999, weight_decay=1e-6, eps=1e-8
  - LR scheduler: get_scheduler("cosine")
  - Train/val split (5%)
  - tqdm progress bar with ema_decay logging

Run:
    python train_pixel_64.py
    accelerate launch train_pixel_64.py   # for multi-GPU
"""

import logging
import math
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version

import wandb
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder
from models import build_unet
from dataset_64_update import ShapeDataset64


logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, epoch, accelerator, unet, conditioner, optimizer, ema=None):
    ckpt_dir = Path(output_dir) / f"checkpoint-epoch-{epoch:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_unet = accelerator.unwrap_model(unet)

    if ema is not None:
        ema.store(raw_unet.parameters())
        ema.copy_to(raw_unet.parameters())
        raw_unet.save_pretrained(ckpt_dir / "unet_ema")
        ema.restore(raw_unet.parameters())

    raw_unet.save_pretrained(ckpt_dir / "unet")
    torch.save(accelerator.unwrap_model(conditioner).state_dict(), ckpt_dir / "conditioner.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    logger.info(f"  Saved checkpoint → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = Config()

    logging_dir = Path(cfg.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir,
        logging_dir=str(logging_dir),
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset — 95% train / 5% val
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        print("Loading dataset into RAM...")
    t0 = time.time()
    full_dataset = ShapeDataset64(cfg.dataset_root, max_samples=cfg.max_samples)
    n_val   = max(1, int(len(full_dataset) * 0.05))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    if accelerator.is_main_process:
        print(f"Dataset loaded in {time.time()-t0:.1f}s | train: {n_train} | val: {n_val}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=0, drop_last=False,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    unet = build_unet(cfg)

    conditioner = ShapeConditioningEncoder(
        input_dim=cfg.cond_input_dim,
        hidden_dim=cfg.cond_hidden_dim,
        output_dim=cfg.cond_output_dim,
    )

    # EMAModel — diffusers version with warmup schedule
    if cfg.use_ema:
        ema = EMAModel(
            unet.parameters(),
            decay=cfg.ema_decay,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=0.75,
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )
        ema.to(accelerator.device)
    else:
        ema = None

    # ------------------------------------------------------------------
    # Noise scheduler
    # ------------------------------------------------------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="linear",
        prediction_type=cfg.prediction_type,
    )

    # ------------------------------------------------------------------
    # Optimizer + LR scheduler
    # ------------------------------------------------------------------
    trainable_params = list(unet.parameters()) + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * cfg.num_epochs,
    )

    # ------------------------------------------------------------------
    # Prepare with accelerator
    # ------------------------------------------------------------------
    unet, conditioner, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, conditioner, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.wandb_project,
            config=cfg.__dict__,
            init_kwargs={"wandb": {"name": cfg.wandb_run_name}},
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    total_params = sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in conditioner.parameters())
    logger.info(f"Trainable parameters: {total_params:,}")
    logger.info(f"Epochs: {cfg.num_epochs} | Steps/epoch: {num_update_steps_per_epoch}")

    for epoch in range(cfg.num_epochs):
        unet.train()
        conditioner.train()
        epoch_loss = 0.0
        t_epoch_start = time.time()

        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{cfg.num_epochs}",
        )

        for step, (images, cond_vectors) in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                noise     = torch.randn_like(images)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=images.device,
                ).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                encoder_hidden_states = conditioner(cond_vectors)
                noise_pred = unet(
                    noisy_images, timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                target = noise if cfg.prediction_type == "epsilon" else images
                loss   = F.mse_loss(noise_pred.float(), target.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if ema is not None:
                    ema.step(unet.parameters())

                global_step += 1
                loss_val = loss.detach().item()
                epoch_loss += loss_val

                logs = {
                    "loss": f"{loss_val:.4f}",
                    "lr":   f"{lr_scheduler.get_last_lr()[0]:.2e}",
                }
                if ema is not None:
                    logs["ema_decay"] = f"{ema.cur_decay_value:.4f}"
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)

                if global_step % cfg.log_every_n_steps == 0:
                    wandb_logs = {
                        "train/loss":        loss_val,
                        "train/lr":          lr_scheduler.get_last_lr()[0],
                        "train/epoch":       epoch,
                        "train/global_step": global_step,
                    }
                    if ema is not None:
                        wandb_logs["train/ema_decay"] = ema.cur_decay_value
                    accelerator.log(wandb_logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # ── validation ────────────────────────────────────────────────
        unet.eval()
        conditioner.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, cond_vectors in val_dataloader:
                noise     = torch.randn_like(images)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=images.device,
                ).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                encoder_hidden_states = conditioner(cond_vectors)
                noise_pred = unet(noisy_images, timesteps,
                                  encoder_hidden_states=encoder_hidden_states).sample
                target = noise if cfg.prediction_type == "epsilon" else images
                val_loss += F.mse_loss(noise_pred.float(), target.float()).item()
        val_loss /= len(val_dataloader)

        # ── epoch summary ─────────────────────────────────────────────
        if accelerator.is_main_process:
            epoch_time = time.time() - t_epoch_start
            avg_loss   = epoch_loss / num_update_steps_per_epoch
            eta_hours  = epoch_time * (cfg.num_epochs - epoch - 1) / 3600
            print(
                f"Epoch {epoch+1}/{cfg.num_epochs} | "
                f"train_loss {avg_loss:.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"epoch {epoch_time:.1f}s | "
                f"ETA {eta_hours:.1f}h"
            )
            accelerator.log({
                "epoch/train_loss": avg_loss,
                "epoch/val_loss":   val_loss,
                "epoch/time_s":     epoch_time,
                "epoch/eta_hours":  eta_hours,
                "epoch":            epoch + 1,
            }, step=global_step)

            # ── checkpoint ────────────────────────────────────────────
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                save_checkpoint(
                    cfg.output_dir, epoch + 1,
                    accelerator, unet, conditioner, optimizer, ema,
                )

        unet.train()
        conditioner.train()

    # final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            cfg.output_dir, cfg.num_epochs,
            accelerator, unet, conditioner, optimizer, ema,
        )

    accelerator.end_training()
    print("Training complete.")


if __name__ == "__main__":
    main()