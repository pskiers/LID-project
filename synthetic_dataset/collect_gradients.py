"""
collect_gradients.py
--------------------
Computes gradients of pairwise image losses with respect to conditioner
output embeddings. Saves gradient matrix for subsequent PCA analysis.

For each pair (image_A, vector_A) and (image_B, vector_B):
  1. Noise image_A to timestep t
  2. Denoise with UNet conditioned on vector_A — grad tracked on conditioner output
  3. Compute MSE loss against image_B
  4. Backprop → gradient w.r.t. conditioner output (64-dim)

Runs for each specified timestep independently.
Different timesteps capture different levels of abstraction:
  - high t (e.g. 700): coarse structure, shape
  - mid  t (e.g. 500): mid-level features
  - low  t (e.g. 300): fine details, texture

Conditioning dimensionality is FLEXIBLE via --cond_input_dim (default 10,
matching the original shape+rgb+size+stripe+grain dataset). Set it to 7 for
the no-texture variant (shape+rgb+size, stripe/grain columns dropped
entirely from SampleVector64), or any other value that follows the same
feature ordering convention:
    [is_tri, is_sq, is_circ,  r, g, b, size,  h_stripe, v_stripe, grain]
                                                (first N of these, N=cond_input_dim)
--prompt takes exactly --cond_input_dim values; if omitted, a sensible
default is built automatically for whichever --cond_input_dim you pass.

Usage:
    # single timestep, default 10-dim conditioning
    python collect_gradients.py 
        --checkpoint /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc/checkpoint-epoch-0200 \\
        --data_dir /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/samples/samples_64_random_acc \\
        --timesteps 500

    # multiple timesteps
    python collect_gradients.py --checkpoint outputs_64/checkpoint-epoch-0200 \\
        --data_dir outputs/samples/samples_64_random --timesteps 100 300 500 700

    # 7-dim no-texture variant (shape+rgb+size, stripe/grain columns dropped)
    python collect_gradients.py --checkpoint outputs_64_notexture/checkpoint-epoch-0200 \\
        --data_dir outputs/samples/samples_64_notexture --cond_input_dim 7
"""

import argparse
import inspect
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision.transforms.functional import to_tensor

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder
from dataset_64 import SampleVector64, ImageGenerator64


# Canonical feature ordering. --cond_input_dim N takes the first N of these —
# so 10 is the full original set, 7 is shape+rgb+size with stripe/grain
# columns dropped entirely (matching how SampleVector64 was slimmed down for
# the no-texture variant), and any other N follows the same prefix rule if
# your dataset variant matches this ordering convention.
FEATURE_NAMES_FULL = ["is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                       "h_stripe", "v_stripe", "grain"]
N_SHAPE_DIMS = 3   # is_tri, is_sq, is_circ — always the first 3, one-hot

# Default value for each continuous feature, used only when --prompt is
# omitted and a default conditioning vector needs to be built for whichever
# --cond_input_dim is in effect.
CONTINUOUS_DEFAULTS = {
    "r": 1.0, "g": 1.0, "b": 1.0, "size": 1.0,
    "h_stripe": 0.5, "v_stripe": 0.5, "grain": 0.5,
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
    """Shape defaults to circle ([0,0,1]) truncated/padded to however many
    shape dims are in play (always 3 here); continuous features default per
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
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/outputs_64_acc_update1/checkpoint-epoch-0200")
    parser.add_argument("--data_dir",   type=str, default="outputs/samples_for_grads/circle_only_text_update1",
                        help="Folder with 000000.png ... and vectors.npy")
    parser.add_argument("--out_dir",    type=str, default="outputs/gradients/t_mult_only_text_5k_update1")
    parser.add_argument("--num_steps",  type=int, default=10,
                        help="DDIM steps for denoising")
    parser.add_argument("--n_pairs",    type=int, default=5000,
                        help="Number of pairs to compute gradients for")
    parser.add_argument("--timesteps",  type=int, nargs="+", default=[999,900,800,700,600,500, 50],
                        help="Noise timestep(s) to run, e.g. --timesteps 500 or --timesteps 100 300 500 700")
    parser.add_argument("--cond_input_dim", type=int, default=10,
                        help="Conditioner input dimensionality. Default 10 matches the "
                             "original shape+rgb+size+stripe+grain dataset. Set to 7 for "
                             "the no-texture variant (shape+rgb+size, stripe/grain columns "
                             "dropped entirely). Determines both --prompt's expected length "
                             "and which SampleVector64 kwargs get passed in --average_anchor mode.")
    parser.add_argument("--prompt",     type=float, nargs="+", default=None,
                        help="Conditioning vector for anchor image. Length must equal "
                             "--cond_input_dim. If omitted, a sensible default is built "
                             "automatically (circle shape, full-value color/size, neutral "
                             "0.5 for any stripe/grain dims present). Feature order: "
                             + ", ".join(FEATURE_NAMES_FULL) + " (first --cond_input_dim of these).")
    parser.add_argument("--average_anchor", action="store_true", default=True,
                        help="If set, anchor image is a deterministically rendered average image "
                             "(all continuous features set to 0.5, shape taken from --prompt)")
    parser.add_argument("--multiple_anchors", action="store_true", default=True,
                        help="If set, compute gradients between random pairs — no fixed anchor. "
                             "Overrides --average_anchor.")
    parser.add_argument("--seed",       type=int, default=42)
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


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir):
    data_dir  = Path(data_dir)
    img_files = sorted(data_dir.glob("*.png"))

    dataset = []
    for img_path in img_files:
        img   = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img) * 2.0 - 1.0   # [0,1] → [-1,1]
        dataset.append(img_t)

    print(f"Loaded {len(dataset)} images from {data_dir}")
    return dataset


# ---------------------------------------------------------------------------
# Noising + denoising
# ---------------------------------------------------------------------------

def noise_image(image, timestep, scheduler, device, seed):
    t     = torch.tensor([timestep], device=device)
    torch.manual_seed(seed)
    noise = torch.randn_like(image)
    return scheduler.add_noise(image, noise, t)


def denoise_with_grad(unet, cond_out, noisy_image, ddim):
    image = noisy_image.clone()
    for t in ddim.timesteps:
        noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
        image      = ddim.step(noise_pred, t, image).prev_sample
    return image


def build_average_anchor_vector(prompt, feature_names):
    """Builds the deterministic average-render SampleVector64 for whichever
    continuous features are actually present, given --cond_input_dim.

    Only passes kwargs that SampleVector64.__init__ actually accepts —
    inspected at runtime rather than hardcoded, so this works whether your
    slimmed-down (e.g. 7-dim) SampleVector64 dropped stripe/grain fields
    entirely, or kept them with fixed neutral defaults. Any continuous
    feature both present in `feature_names` and accepted by SampleVector64
    gets set to 0.5 by default, then overridden from `prompt` if the prompt
    explicitly fixed it away from 0.5.
    """
    valid_params = set(inspect.signature(SampleVector64.__init__).parameters) - {"self"}
    continuous_names = feature_names[N_SHAPE_DIMS:]

    kwargs = {"shape_id": int(np.argmax(prompt[:N_SHAPE_DIMS]))}
    for name in continuous_names:
        if name in valid_params:
            kwargs[name] = 0.5
        # else: this SampleVector64 variant doesn't have this field at all
        # (e.g. stripe/grain dropped for the no-texture dataset) — skip it
        # silently rather than passing an unexpected kwarg.

    avg_vec = SampleVector64(**kwargs)

    # override only features that are not at their default (0.5) in the prompt
    # i.e. keep any explicitly fixed continuous values from prompt
    for feat_idx, name in enumerate(continuous_names, start=N_SHAPE_DIMS):
        if name not in valid_params:
            continue
        val = prompt[feat_idx]
        if val != 0.5:
            setattr(avg_vec, name, val)

    return avg_vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    cfg     = Config(cond_input_dim=args.cond_input_dim)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt    = Path(args.checkpoint)

    print(f"Conditioning dim: {args.cond_input_dim}  |  feature order: {args._feature_names}")

    # save run config
    config_txt = "\n".join(f"{k}: {v}" for k, v in vars(args).items() if not k.startswith("_"))
    (out_dir / "run_config.txt").write_text(config_txt)
    print(f"Saved run config → {out_dir / 'run_config.txt'}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from {ckpt} ...")
    unet = UNet2DConditionModel.from_pretrained(ckpt / "unet_ema").to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    conditioner = ShapeConditioningEncoder(
        cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim,
    ).to(device)
    conditioner.load_state_dict(
        torch.load(ckpt / "conditioner.pt", map_location=device, weights_only=True)
    )
    conditioner.eval()
    for p in conditioner.parameters():
        p.requires_grad_(False)

    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    # ------------------------------------------------------------------
    # Load dataset + sample pairs (same pairs for all timesteps)
    # ------------------------------------------------------------------
    dataset = load_dataset(args.data_dir)
    assert len(dataset) >= 2, "Need at least 2 images"

    rng = random.Random(args.seed)

    # anchor conditioning vector from --prompt argument
    vec_a = torch.tensor(args.prompt, dtype=torch.float32)
    print(f"Anchor prompt ({args.cond_input_dim}-dim): {args.prompt}")

    if args.multiple_anchors:
        # random pairs — different image_A each time, vec_a taken from dataset not used
        print("Mode: multiple anchors — random pairs from dataset")
        pairs = [tuple(rng.sample(range(len(dataset)), 2)) for _ in range(args.n_pairs)]
        img_a = None   # will be set per pair
    elif args.average_anchor:
        avg_vec = build_average_anchor_vector(args.prompt, args._feature_names)

        pil_img = ImageGenerator64().generate(avg_vec, seed=0)
        img_t   = torch.from_numpy(
            np.array(pil_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1) * 2.0 - 1.0   # (3, H, W) in [-1, 1]
        img_a   = img_t.unsqueeze(0).to(device)
        pil_img.save(out_dir / "anchor_image.png")
        print(f"Anchor image: deterministic average render of {avg_vec}")
        print(f"  Saved anchor → {out_dir / 'anchor_image.png'}")
        pairs = [rng.randint(0, len(dataset) - 1) for _ in range(args.n_pairs)]
    else:
        # anchor image — random image from dataset
        idx_a   = rng.randint(0, len(dataset) - 1)
        img_a   = dataset[idx_a].unsqueeze(0).to(device)
        print(f"Anchor image: index {idx_a}")
        pairs = [rng.randint(0, len(dataset) - 1) for _ in range(args.n_pairs)]

    # ------------------------------------------------------------------
    # Run for each timestep
    # ------------------------------------------------------------------
    job_start = time.time()
    for ts_idx, noise_t in enumerate(args.timesteps):
        print(f"\n{'='*60}")
        print(f"Timestep t={noise_t} | {args.n_pairs} pairs  ({ts_idx+1}/{len(args.timesteps)} timesteps)")
        print(f"{'='*60}")

        all_grads = []
        t_start   = time.time()

        for pair_idx, pair in enumerate(pairs):
            if args.multiple_anchors:
                idx_a, idx_b = pair
                img_a_p = dataset[idx_a].unsqueeze(0).to(device)
                img_b   = dataset[idx_b].unsqueeze(0).to(device)
                # use the conditioning vector of image_a from the dataset
                # since we have no vectors.npy, fall back to the prompt vec
                vec_a_p = vec_a
            else:
                img_a_p = img_a
                img_b   = dataset[pair].unsqueeze(0).to(device)
                vec_a_p = vec_a

            noisy_a  = noise_image(img_a_p, noise_t, ddpm, device, seed=args.seed + pair_idx)
            cond_out = conditioner(vec_a_p.unsqueeze(0).to(device)).detach().requires_grad_(True)

            denoised_a = denoise_with_grad(unet, cond_out, noisy_a, ddim)
            loss       = F.mse_loss(denoised_a.float(), img_b.float())
            loss.backward()

            if cond_out.grad is not None:
                all_grads.append(cond_out.grad.squeeze().detach().cpu().numpy())

            del noisy_a, cond_out, denoised_a, loss
            if args.multiple_anchors:
                del img_a_p
            torch.cuda.empty_cache()

            if (pair_idx + 1) % 200 == 0:
                elapsed   = time.time() - t_start
                rate      = (pair_idx + 1) / elapsed
                remaining = (args.n_pairs - pair_idx - 1) / rate / 60
                job_elapsed = time.time() - job_start
                job_rate    = (ts_idx * args.n_pairs + pair_idx + 1) / job_elapsed
                job_remaining = (len(args.timesteps) * args.n_pairs - (ts_idx * args.n_pairs + pair_idx + 1)) / job_rate / 60
                print(f"  [{pair_idx+1}/{args.n_pairs}] | {rate:.1f} pairs/s | "
                      f"this timestep ETA: {remaining:.1f} min | overall ETA: {job_remaining:.1f} min")
            elif pair_idx == 0:
                elapsed = time.time() - t_start
                eta_min = elapsed * (args.n_pairs - 1) / 60
                job_eta_min = eta_min * len(args.timesteps)
                print(f"  First pair: {elapsed:.1f}s | this timestep ETA: {eta_min:.1f} min | "
                      f"overall ETA: ~{job_eta_min:.1f} min")

        grads_matrix = np.stack(all_grads, axis=0)   # (N, cond_output_dim)
        grads_path   = out_dir / f"grads_t{noise_t}.npy"
        np.save(grads_path, grads_matrix)
        print(f"  Saved {grads_matrix.shape} → {grads_path}")

    print(f"\nAll timesteps complete")


if __name__ == "__main__":
    main()