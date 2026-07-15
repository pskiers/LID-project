"""
sample_prompt.py
----------------
Generate N images from a specific conditioning vector and save as individual PNG files.

Usage:
    # default prompt: circle, all other attributes at 0.5
    python sample_prompt.py --checkpoint outputs_64/checkpoint-epoch-0200

    # custom prompt
    python sample_prompt.py --checkpoint outputs_64/checkpoint-epoch-0200 \\
        --prompt 0 1 0  0.9 0.1 0.1  0.8  0.0 0.0 0.0

    # [is_triangle, is_square, is_circle, r, g, b, size, h_stripe, v_stripe, grain]
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/outputs_64_acc_random/checkpoint-epoch-0200")
    parser.add_argument("--num_steps",  type=int, default=10)
    parser.add_argument("--n",          type=int, default=5000,
                        help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir",    type=str, default="outputs/samples_for_grads/circle_red_big")
    parser.add_argument("--prompt",     type=float, nargs=10,
                        default=[0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],
                        metavar=("is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                                 "h_stripe", "v_stripe", "grain"),
                        help="10 conditioning values: [is_triangle, is_square, is_circle, "
                             "r, g, b, size, h_stripe, v_stripe, grain] (default: circle, all others 0.5)")
    return parser.parse_args()


@torch.no_grad()
def generate_batch(unet, conditioner, ddim, cond_tensor, cfg, device):
    images = torch.randn(
        len(cond_tensor), cfg.unet_in_channels, cfg.image_size, cfg.image_size,
        device=device,
    )
    encoder_hidden_states = conditioner(cond_tensor.to(device))
    for t in ddim.timesteps:
        noise_pred = unet(images, t, encoder_hidden_states=encoder_hidden_states).sample
        images = ddim.step(noise_pred, t, images).prev_sample
    images = (images.clamp(-1, 1) + 1) / 2
    return [to_pil_image(img.float().cpu()) for img in images]


def main():
    args    = parse_args()
    cfg     = Config()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {ckpt} ...")
    unet = UNet2DConditionModel.from_pretrained(ckpt / "unet_ema").to(device)
    unet.eval()

    conditioner = ShapeConditioningEncoder(
        cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim,
    ).to(device)
    conditioner.load_state_dict(
        torch.load(ckpt / "conditioner.pt", map_location=device, weights_only=True)
    )
    conditioner.eval()

    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    cond_vec = torch.tensor(args.prompt, dtype=torch.float32)
    print(f"Prompt: {args.prompt}")
    print(f"        [is_tri, is_sq, is_circ, r, g, b, size, h_stripe, v_stripe, grain]")

    # repeat same conditioning vector for all images
    all_conds = cond_vec.unsqueeze(0).repeat(args.n, 1)  # (N, 10)
    n_batches = (args.n + args.batch_size - 1) // args.batch_size

    print(f"Generating {args.n} images | {n_batches} batches of {args.batch_size} | {args.num_steps} steps")

    saved = 0
    for batch_idx in range(n_batches):
        start       = batch_idx * args.batch_size
        end         = min(start + args.batch_size, args.n)
        batch_conds = all_conds[start:end]

        t0         = time.time()
        pil_images = generate_batch(unet, conditioner, ddim, batch_conds, cfg, device)
        batch_time = time.time() - t0

        for img in pil_images:
            img.save(out_dir / f"{saved:06d}.png")
            saved += 1

        if batch_idx == 0:
            eta_min = batch_time * (n_batches - 1) / 60
            print(f"  First batch: {batch_time:.1f}s | ETA: {eta_min:.1f} min")
        else:
            print(f"  Batch {batch_idx+1}/{n_batches} | {saved}/{args.n} saved")

    print(f"\nDone — {saved} images saved to {out_dir}/")


if __name__ == "__main__":
    main()