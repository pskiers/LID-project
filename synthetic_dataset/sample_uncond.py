"""
sample_uncond.py
----------------
Generate images from unconditional latent diffusion checkpoint.

Usage:
    python sample_uncond.py --checkpoint outputs_uncond/uncond-epoch-0050
"""

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DModel
from torchvision.transforms.functional import to_pil_image

from configs.config_uncond import ConfigUncond as Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--num_steps",   type=int, default=50)
    parser.add_argument("--n_samples",   type=int, default=8)
    parser.add_argument("--out_dir",     type=str, default="samples_uncond")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.bfloat16 if cfg.mixed_precision == "bf16"
        else torch.float16 if cfg.mixed_precision == "fp16"
        else torch.float32
    )

    ckpt = Path(args.checkpoint)

    # VAE
    print(f"Loading VAE from {cfg.vae_model_id} ...")
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_id).to(device, dtype=weight_dtype)
    vae.eval()

    # UNet
    unet = UNet2DModel.from_pretrained(ckpt / "unet").to(device, dtype=weight_dtype)
    unet.eval()

    # Scheduler
    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    # Sample
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents = torch.randn(
        args.n_samples, cfg.latent_channels, cfg.latent_size, cfg.latent_size,
        device=device, dtype=weight_dtype,
    )

    with torch.no_grad():
        for t in ddim.timesteps:
            noise_pred = unet(latents, t).sample
            latents    = ddim.step(noise_pred, t, latents).prev_sample

        latents = latents / vae.config.scaling_factor
        images  = vae.decode(latents).sample
        images  = (images.clamp(-1, 1) + 1) / 2

    for i, img in enumerate(images):
        path = out_dir / f"sample_{i:02d}.png"
        to_pil_image(img.float().cpu()).save(path)
        print(f"  Saved {path}")

    print(f"\nDone — {args.n_samples} images saved to {out_dir}/")


if __name__ == "__main__":
    main()
