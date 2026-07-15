"""
visualize_x0.py
---------------
Generates N images conditioned on a given vector and visualizes the predicted
clean image (x0) at K equally spaced timesteps during the diffusion process.

For each generated image:
  - Samples a random noise z_T
  - Runs forward DDIM loop up to each timestep t_k
  - At each t_k: computes predicted x0 = (z_t - sigma_t * eps_theta) / alpha_t
  - Saves a grid: rows = images, cols = timesteps (left=high noise, right=clean)

This shows what the model "thinks" the final image will be at each denoising step,
which reveals at which timestep different semantic features (color, shape, texture)
are committed to.

Usage:
    python visualize_x0.py
    python visualize_x0.py --n_images 6 --n_timesteps 12 --prompt 0 0 1 0.9 0.1 0.1 0.5 0 0 0
    python visualize_x0.py --n_images 8 --n_timesteps 8 --num_steps 20 --out_dir outputs/x0_vis
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str,
                        default="outputs/checkpoints/outputs_64_acc_random/checkpoint-epoch-0200")
    parser.add_argument("--out_dir",      type=str, default="outputs/x0_visualization")
    parser.add_argument("--prompt",       type=float, nargs=10,
                        default=[0.0, 0.0, 1.0,  1, 0, 0,  1,  0, 0.2, 0.5],
                        metavar=("is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                                 "h_stripe", "v_stripe", "grain"))
    parser.add_argument("--n_images",     type=int, default=3,
                        help="Number of images to generate (rows in grid)")
    parser.add_argument("--n_timesteps",  type=int, default=20,
                        help="Number of equally spaced timesteps to visualize (cols in grid)")
    parser.add_argument("--num_steps",    type=int, default=50,
                        help="Total DDIM denoising steps")
    parser.add_argument("--scale",        type=int, default=3,
                        help="Scale factor for each cell (default: 3 → 192x192px per cell)")
    parser.add_argument("--seed",         type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def predict_x0(unet, z_t, t, cond_out, scheduler):
    """
    Compute predicted x0 from noisy latent z_t at timestep t.
    x0_pred = (z_t - sigma_t * eps_theta(z_t, t, c)) / alpha_t
    """
    t_tensor   = torch.tensor([t], device=z_t.device)
    noise_pred = unet(z_t, t_tensor, encoder_hidden_states=cond_out).sample

    # get alpha and sigma for this timestep
    alpha_prod = scheduler.alphas_cumprod[t]
    alpha_t    = alpha_prod ** 0.5
    sigma_t    = (1 - alpha_prod) ** 0.5

    x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
    x0_pred = x0_pred.clamp(-1, 1)
    return x0_pred


def tensor_to_pil(t):
    img = (t.squeeze().float().cpu().clamp(-1, 1) + 1) / 2
    return to_pil_image(img)


def make_grid(images_2d, timestep_labels, scale):
    """
    images_2d: list[row][col] → PIL image (64x64)
    rows = images, cols = timesteps
    """
    n_rows = len(images_2d)
    n_cols = len(images_2d[0])
    cell   = 64 * scale
    pad    = 4
    label_h = 28
    label_w = 20

    grid_w = label_w + n_cols * (cell + pad) + pad
    grid_h = label_h + n_rows * (cell + pad) + pad
    grid   = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw   = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    # column headers: timestep labels
    for c, label in enumerate(timestep_labels):
        x = label_w + pad + c * (cell + pad) + cell // 2
        draw.text((x, 6), label, fill=(60, 60, 60), font=font, anchor="mt")

    # paste images
    for r, row in enumerate(images_2d):
        y = label_h + pad + r * (cell + pad)
        for c, img in enumerate(row):
            x = label_w + pad + c * (cell + pad)
            grid.paste(img.resize((cell, cell), Image.NEAREST), (x, y))

    return grid


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt   = Path(args.checkpoint)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
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

    # conditioning
    cond_vec = torch.tensor(args.prompt, dtype=torch.float32).unsqueeze(0).to(device)
    cond_out = conditioner(cond_vec)   # (1, 1, 64)
    print(f"Prompt: {args.prompt}")

    # which DDIM timesteps to visualize — equally spaced
    # ddim.timesteps goes from high t to low t (e.g. 999 → 0)
    all_steps     = ddim.timesteps.tolist()   # descending
    n_vis         = args.n_timesteps
    vis_indices   = np.linspace(0, len(all_steps) - 1, n_vis, dtype=int).tolist()
    vis_timesteps = [all_steps[i] for i in vis_indices]   # subset to visualize
    timestep_labels = [f"t={t}" for t in vis_timesteps]
    print(f"Visualizing timesteps: {vis_timesteps}")

    # ------------------------------------------------------------------
    # Generate images
    # ------------------------------------------------------------------
    images_2d = []   # [row=image][col=timestep]

    for img_idx in range(args.n_images):
        torch.manual_seed(args.seed + img_idx)
        z = torch.randn(
            1, cfg.unet_in_channels, cfg.image_size, cfg.image_size, device=device
        )
        row_images = []
        vis_set    = set(vis_timesteps)

        # run DDIM step by step, snapshot x0_pred at each vis timestep
        image = z.clone()
        vis_ptr = 0

        for step_idx, t in enumerate(all_steps):
            # snapshot x0_pred before the denoising step
            if t in vis_set and vis_ptr < n_vis:
                x0_pred = predict_x0(unet, image, t, cond_out, ddpm)
                row_images.append(tensor_to_pil(x0_pred))
                vis_ptr += 1

            # DDIM step
            noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
            image      = ddim.step(noise_pred, t, image).prev_sample

        # if we haven't captured all vis timesteps (e.g. t=0 not in schedule)
        # append the final fully denoised image for the last slot
        while len(row_images) < n_vis:
            row_images.append(tensor_to_pil(image.clamp(-1, 1)))

        images_2d.append(row_images)
        print(f"  Image {img_idx+1}/{args.n_images} done")

    # ------------------------------------------------------------------
    # Save grid
    # ------------------------------------------------------------------
    grid      = make_grid(images_2d, timestep_labels, args.scale)
    grid_path = out_dir / "x0_timesteps.png"
    grid.save(grid_path)
    print(f"\nSaved grid → {grid_path}")
    print(f"Grid size: {grid.size}  ({args.n_images} rows × {args.n_timesteps} cols)")


if __name__ == "__main__":
    main()