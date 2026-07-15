"""
sample_conditioning_test.py
---------------------------
Tests whether the model has learned to follow conditioning by generating
multiple images from the same conditioning vector. If conditioning works,
images in the same row should share the specified attributes despite
different noise seeds.

Grid layout:
  rows = conditioning vectors (concepts)
  cols = repeated samples (different noise seeds, same conditioning)

Usage:
    python sample_conditioning_test.py --checkpoint outputs_64/checkpoint-epoch-0200
    python sample_conditioning_test.py --checkpoint outputs_64/checkpoint-epoch-0200 --num_steps 100 --repeats 6
"""

import argparse
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps",  type=int, default=1000)
    parser.add_argument("--repeats",    type=int, default=6,
                        help="Number of repeated samples per conditioning vector")
    parser.add_argument("--out_dir",    type=str, default="outputs/samples/samples_64_acc")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Conditioning vectors grouped by concept being tested
# [is_triangle, is_square, is_circle, r, g, b, size, h_stripe, v_stripe, grain]
# ---------------------------------------------------------------------------

# (vector, label) pairs — each will be repeated `repeats` times
TEST_CONDITIONS = [
    # ── shape only ──────────────────────────────────────────────────────
    ([0.0, 1.0, 0.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],  "square"),
    ([1.0, 0.0, 0.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],  "triangle"),
    ([0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],  "circle"),
    # ── shape + color ───────────────────────────────────────────────────
    ([0.0, 1.0, 0.0,  0.9, 0.1, 0.1,  0.5,  0.5, 0.5, 0.5],  "red square"),
    ([1.0, 0.0, 0.0,  0.1, 0.2, 0.9,  0.5,  0.5, 0.5, 0.5],  "blue triangle"),
    ([0.0, 0.0, 1.0,  0.1, 0.8, 0.1,  0.5,  0.5, 0.5, 0.5],  "green circle"),
    # ── shape + size ────────────────────────────────────────────────────
    ([0.0, 1.0, 0.0,  0.5, 0.5, 0.5,  0.1,  0.5, 0.5, 0.5],  "small square"),
    ([0.0, 1.0, 0.0,  0.5, 0.5, 0.5,  0.9,  0.5, 0.5, 0.5],  "large square"),
    ([0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.1,  0.5, 0.5, 0.5],  "small circle"),
    ([0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.9,  0.5, 0.5, 0.5],  "large circle"),
    # ── shape + texture ─────────────────────────────────────────────────
    ([0.0, 1.0, 0.0,  0.1, 0.4, 0.8,  0.5,  0.9, 0.5, 0.5],  "blue square\nh-stripes"),
    ([0.0, 1.0, 0.0,  0.1, 0.4, 0.8,  0.5,  0.5, 0.9, 0.5],  "blue square\nv-stripes"),
    ([1.0, 0.0, 0.0,  0.8, 0.5, 0.0,  0.5,  0.8, 0.8, 0.5],  "orange triangle\ngrid"),
    ([0.0, 0.0, 1.0,  0.7, 0.1, 0.7,  0.5,  0.5, 0.5, 0.9],  "purple circle\ngrain"),
    # ── full specification ──────────────────────────────────────────────
    ([0.0, 1.0, 0.0,  0.9, 0.1, 0.1,  0.8,  0.8, 0.8, 0.5],  "large red square\nall textures"),
    ([1.0, 0.0, 0.0,  0.1, 0.7, 0.1,  0.2,  0.5, 0.5, 0.5],  "small green\ntriangle plain"),
]


@torch.no_grad()
def generate_repeated(unet, conditioner, scheduler, cond_vec, cfg, device, repeats, base_seed):
    """Generate `repeats` images from the same conditioning vector with different seeds."""
    results = []
    cond_out = conditioner(cond_vec.unsqueeze(0).to(device))  # (1, 1, 64)
    for i in range(repeats):
        torch.manual_seed(base_seed + i)
        image = torch.randn(1, cfg.unet_in_channels, cfg.image_size, cfg.image_size, device=device)
        for t in scheduler.timesteps:
            noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
            image = scheduler.step(noise_pred, t, image).prev_sample
        image = (image.clamp(-1, 1) + 1) / 2
        results.append(to_pil_image(image.squeeze().float().cpu()))
    return results


def make_grid(rows_images, labels, scale=4, label_w=120):
    """
    rows_images: list of lists — [row][col] → PIL image
    labels: one label per row
    """
    cols   = len(rows_images[0])
    rows   = len(rows_images)
    W, H   = rows_images[0][0].size
    cell_w = W * scale
    cell_h = H * scale
    pad    = 4
    grid_w = label_w + cols * (cell_w + pad) + pad
    grid_h = rows * (cell_h + pad) + pad
    grid   = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw   = ImageDraw.Draw(grid)

    for r, (imgs, label) in enumerate(zip(rows_images, labels)):
        y = pad + r * (cell_h + pad)
        # row label on the left
        for i, line in enumerate(label.split("\n")):
            draw.text((4, y + cell_h // 2 - 7 + i * 14), line, fill=(30, 30, 30))
        for c, img in enumerate(imgs):
            x = label_w + pad + c * (cell_w + pad)
            grid.paste(img.resize((cell_w, cell_h), Image.NEAREST), (x, y))

    return grid


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = Path(args.checkpoint)
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

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    scheduler.set_timesteps(args.num_steps)

    print(f"Generating {len(TEST_CONDITIONS)} conditions × {args.repeats} repeats ({args.num_steps} steps)...")

    rows_images = []
    labels      = []
    for vec, label in TEST_CONDITIONS:
        cond_vec = torch.tensor(vec, dtype=torch.float32)
        imgs     = generate_repeated(unet, conditioner, scheduler, cond_vec, cfg, device, args.repeats, args.seed)
        rows_images.append(imgs)
        labels.append(label)
        print(f"  Done: {label.replace(chr(10), ' ')}")

    grid      = make_grid(rows_images, labels)
    grid_path = out_dir / "conditioning_test.png"
    grid.save(grid_path)
    print(f"\nSaved → {grid_path}")


if __name__ == "__main__":
    main()