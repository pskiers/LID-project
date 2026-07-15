"""
train_uncond_accelerate.py
--------------------------
Unconditional diffusion training for 64x64 shape images using Accelerate.
Adapted from the diffusers example script — replaces HuggingFace datasets
with ShapeDataset64 and adjusts UNet architecture for small shapes.

Run:
    python train_uncond_accelerate.py --output_dir outputs_uncond_accel
    python train_uncond_accelerate.py --output_dir outputs_uncond_accel --use_ema
"""

import argparse
import inspect
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datetime import timedelta
from tqdm.auto import tqdm
from packaging import version

#import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_wandb_available

import wandb

from dataset_64 import ShapeDataset64


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",    type=str,   default="data_64")
    parser.add_argument("--max_samples",     type=int,   default=None)
    parser.add_argument("--output_dir",      type=str,   default="outputs/checkpoints/outputs_acc_50k")
    parser.add_argument("--train_batch_size",type=int,   default=64)
    parser.add_argument("--eval_batch_size", type=int,   default=16)
    parser.add_argument("--num_epochs",      type=int,   default=200)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs",  type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate",   type=float, default=1e-4)
    parser.add_argument("--lr_scheduler",    type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int,   default=500)
    parser.add_argument("--adam_beta1",      type=float, default=0.95)
    parser.add_argument("--adam_beta2",      type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon",    type=float, default=1e-8)
    parser.add_argument("--use_ema",         action="store_true")
    parser.add_argument("--ema_inv_gamma",   type=float, default=1.0)
    parser.add_argument("--ema_power",       type=float, default=0.75)
    parser.add_argument("--ema_max_decay",   type=float, default=0.9999)
    parser.add_argument("--mixed_precision", type=str,   default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--prediction_type", type=str,   default="epsilon",
                        choices=["epsilon", "sample"])
    parser.add_argument("--ddpm_num_steps",  type=int,   default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--val_split",       type=float, default=0.05,
                        help="Fraction of dataset to use for validation (default 5%%)")
    parser.add_argument("--wandb_project",   type=str,   default="shape-diffusion")
    parser.add_argument("--wandb_run_name",  type=str,   default="uncond-accel")
    return parser.parse_args()


def main(args):
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
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
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset — split into train/val
    # ------------------------------------------------------------------
    full_dataset = ShapeDataset64(args.dataset_root, max_samples=args.max_samples)
    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {n_train} | Val: {n_val}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0,
    )

    # ------------------------------------------------------------------
    # UNet — small architecture for 64x64
    # ------------------------------------------------------------------
    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 128),
        norm_num_groups=16,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"UNet parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    # ------------------------------------------------------------------
    # Scheduler + optimizer
    # ------------------------------------------------------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # ------------------------------------------------------------------
    # Prepare with accelerator
    # ------------------------------------------------------------------
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    for epoch in range(args.num_epochs):
        model.train()
        t_epoch_start = time.time()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )

        for step, (images, _) in enumerate(train_dataloader):  # ignore cond vectors
            images = images.to(weight_dtype)
            noise  = torch.randn_like(images)
            bsz    = images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps).sample
                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())
                else:
                    loss = F.mse_loss(model_output.float(), images.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "train/loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

        progress_bar.close()

        # ── validation ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        for images, _ in val_dataloader:
            images = images.to(weight_dtype)
            noise  = torch.randn_like(images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (images.shape[0],), device=images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            with torch.no_grad():
                model_output = model(noisy_images, timesteps).sample
            if args.prediction_type == "epsilon":
                val_loss += F.mse_loss(model_output.float(), noise.float()).item()
            else:
                val_loss += F.mse_loss(model_output.float(), images.float()).item()
        val_loss /= len(val_dataloader)
        accelerator.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        epoch_time = time.time() - t_epoch_start
        eta_hours  = epoch_time * (args.num_epochs - epoch - 1) / 3600
        train_loss = loss.detach().item()
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"epoch: {epoch_time:.1f}s | "
            f"ETA: {eta_hours:.1f}h"
        )

        # ── sample images ─────────────────────────────────────────────
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                images_out = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="np",
                ).images
                if args.use_ema:
                    ema_model.restore(unet.parameters())
                images_processed = (images_out * 255).round().astype("uint8")
                accelerator.get_tracker("wandb").log(
                    {"samples": [wandb.Image(img) for img in images_processed],
                     "epoch": epoch},
                    step=global_step,
                )

            # ── save checkpoint ───────────────────────────────────────
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                pipeline.save_pretrained(args.output_dir)
                if args.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)