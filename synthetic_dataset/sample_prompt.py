"""
sample_prompt.py
----------------
Generate N images from a specific conditioning vector and save as individual PNG files.

Conditioning dimensionality is FLEXIBLE via --cond_input_dim (default 10,
matching the original shape+rgb+size+stripe+grain dataset). Set it to 7 for
the no-texture variant (shape+rgb+size, stripe/grain columns dropped
entirely), or any other value that follows the same feature ordering
convention:
    [is_tri, is_sq, is_circ,  r, g, b, size,  h_stripe, v_stripe, grain]
                                                (first N of these, N=cond_input_dim)
--prompt takes exactly --cond_input_dim values; if omitted, a sensible
default is built automatically for whichever --cond_input_dim you pass.

Usage:
    # default prompt: circle, all other attributes at 0.5 (10-dim)
    python sample_prompt.py --checkpoint outputs_64/checkpoint-epoch-0200

    # custom prompt (10-dim)
    python sample_prompt.py --checkpoint outputs_64/checkpoint-epoch-0200 \\
        --prompt 0 1 0  0.9 0.1 0.1  0.8  0.0 0.0 0.0

    # [is_triangle, is_square, is_circle, r, g, b, size, h_stripe, v_stripe, grain]

    # 7-dim no-texture variant (shape+rgb+size)
    python sample_prompt.py --checkpoint outputs_64_notexture/checkpoint-epoch-0200 \\
        --cond_input_dim 7 --prompt 0 1 0  0.9 0.1 0.1  0.8
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


# Canonical feature ordering. --cond_input_dim N takes the first N of these —
# so 10 is the full original set, 7 is shape+rgb+size with stripe/grain
# columns dropped entirely, and any other N follows the same prefix rule if
# your dataset variant matches this ordering convention.
FEATURE_NAMES_FULL = ["is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                       "h_stripe", "v_stripe", "grain"]
N_SHAPE_DIMS = 3   # is_tri, is_sq, is_circ — always the first 3, one-hot

# Default value for each continuous feature, used only when --prompt is
# omitted. Matches this script's original default exactly (r/g/b/size at
# 0.5, stripe/grain at 0 — note this differs from collect_gradients.py's
# defaults, which use 1.0 / 0.5; each script keeps its own convention).
CONTINUOUS_DEFAULTS = {
    "r": 0.5, "g": 0.5, "b": 0.5, "size": 0.5,
    "h_stripe": 0.0, "v_stripe": 0.0, "grain": 0.0,
}


def feature_names_for(cond_input_dim: int) -> list:
    if cond_input_dim < N_SHAPE_DIMS or cond_input_dim > len(FEATURE_NAMES_FULL):
        raise ValueError(
            f"--cond_input_dim={cond_input_dim} is outside the supported range "
            f"[{N_SHAPE_DIMS}, {len(FEATURE_NAMES_FULL)}] for this script's feature "
            f"ordering convention. If your dataset uses a different feature layout "
            f"entirely (not a prefix of {FEATURE_NAMES_FULL}), FEATURE_NAMES_FULL "
            f"above needs to be edited to match, not just this dimension count."
        )
    return FEATURE_NAMES_FULL[:cond_input_dim]


def default_prompt_for(cond_input_dim: int) -> list:
    """Shape defaults to circle ([0,0,1]); continuous features default per
    CONTINUOUS_DEFAULTS. Matches the original script's default exactly when
    cond_input_dim=10."""
    names = feature_names_for(cond_input_dim)
    shape_default = [0.0, 0.0, 1.0][:N_SHAPE_DIMS]
    vec = list(shape_default)
    for name in names[N_SHAPE_DIMS:]:
        vec.append(CONTINUOUS_DEFAULTS[name])
    return vec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc_mixed_dz0.3/checkpoint-epoch-0200/")
    parser.add_argument("--num_steps",  type=int, default=10)
    parser.add_argument("--n",          type=int, default=5000,
                        help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir",    type=str, default="outputs/samples_for_grads/square_mixed")
    parser.add_argument("--cond_input_dim", type=int, default=10,
                        help="Conditioner input dimensionality. Default 10 matches the "
                             "original shape+rgb+size+stripe+grain dataset. Set to 7 for "
                             "the no-texture variant (shape+rgb+size, stripe/grain columns "
                             "dropped entirely). Determines --prompt's expected length and "
                             "the ShapeConditioningEncoder's input layer shape — must match "
                             "whatever --checkpoint was actually trained with, or "
                             "load_state_dict will raise a shape mismatch.")
    parser.add_argument("--prompt",     type=float, nargs="+", default=None,
                        help="Conditioning values, length must equal --cond_input_dim. "
                             "If omitted, a sensible default is built automatically "
                             "(circle shape, mid-value color/size, zero stripe/grain for "
                             "any of those dims present). Feature order: "
                             + ", ".join(FEATURE_NAMES_FULL) + " (first --cond_input_dim of these).")
    args = parser.parse_args()

    feature_names = feature_names_for(args.cond_input_dim)
    if args.prompt is None:
        args.prompt = default_prompt_for(args.cond_input_dim)
    elif len(args.prompt) != args.cond_input_dim:
        raise SystemExit(
            f"--prompt has {len(args.prompt)} values but --cond_input_dim={args.cond_input_dim} "
            f"expects {args.cond_input_dim}. Feature order: {feature_names}"
        )
    args._feature_names = feature_names   # stashed for logging below
    return args


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
    cfg     = Config(cond_input_dim=args.cond_input_dim)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt    = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Conditioning dim: {args.cond_input_dim}  |  feature order: {args._feature_names}")

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
    print(f"        {args._feature_names}")

    # repeat same conditioning vector for all images
    all_conds = cond_vec.unsqueeze(0).repeat(args.n, 1)  # (N, cond_input_dim)
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