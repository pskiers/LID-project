"""
feature_mse.py
--------------
For each of the 9 continuous features (r, g, b, size, h_stripe, v_stripe, grain,
and the two shape-transition pairs), generates pairs of images that differ only
in that one feature and measures the mean MSE between them.

This quantifies how much each feature contributes to pixel-level MSE —
which directly determines how much that feature's gradient will dominate PCA.

For each feature:
  - Sample a base conditioning vector (other features fixed at 0.5)
  - Generate n_pairs images at value=0.0 and n_pairs images at value=1.0
    (maximum contrast to get an upper bound on MSE contribution)
  - Also sample random pairs within [0,1] for a more realistic estimate
  - Measure mean pixel MSE between the paired images

Usage:
    python feature_mse.py --checkpoint outputs/checkpoints/outputs_64_acc_random/checkpoint-epoch-0200
    python feature_mse.py --checkpoint ... --n_pairs 50 --out_dir outputs/feature_mse
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/outputs_64_acc_random/checkpoint-epoch-0200")
    parser.add_argument("--out_dir",    type=str, default="outputs/feature_mse")
    parser.add_argument("--n_pairs",    type=int, default=50,
                        help="Number of pairs per feature per contrast level")
    parser.add_argument("--num_steps",  type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def generate_image(unet, conditioner, ddim, cond_vec, cfg, device, seed):
    torch.manual_seed(seed)
    image = torch.randn(1, cfg.unet_in_channels, cfg.image_size, cfg.image_size, device=device)
    cond_out = conditioner(cond_vec.unsqueeze(0).to(device))
    for t in ddim.timesteps:
        noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
        image = ddim.step(noise_pred, t, image).prev_sample
    return image.squeeze(0)   # (C, H, W) in [-1, 1]


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def main():
    args    = parse_args()
    cfg     = Config()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt    = Path(args.checkpoint)

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


    # ------------------------------------------------------------------
    # Feature definitions
    # index in 10-dim vector, name, and what values to contrast
    # For continuous features: contrast 0.0 vs 1.0 (max), and random pairs
    # For shape (one-hot): contrast each shape against each other
    # ------------------------------------------------------------------
    # Base vector: circle, all continuous features fixed at 0.0 (no stochastic variation)
    # For each tested feature: set it to 0.5 (max stochastic variation)
    # and compare two images generated with the same seed but different noise realizations
    # → MSE is purely from that feature's stochastic rendering variation
    base_fixed = [0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0]

    continuous_features = [
        (3, "r"),
        (4, "g"),
        (5, "b"),
        (6, "size"),
        (7, "h_stripe"),
        (8, "v_stripe"),
        (9, "grain"),
    ]

    shape_pairs = [
        ((0, "triangle"), (2, "circle")),
        ((1, "square"),   (2, "circle")),
        ((0, "triangle"), (1, "square")),
    ]

    results = {}

    # ------------------------------------------------------------------
    # Part 1: Progressive feature fixing
    # Start from fully random (all at 0.5) and fix one feature at a time
    # to see how much MSE drops — i.e. which feature contributes most
    # to inter-image variation
    # ------------------------------------------------------------------
    print(f"\n--- Part 1: MSE reduction when fixing each feature ---")

    def measure_mse_vec(vec, n_pairs, label=""):
        """Generate n_pairs pairs with given vector, return mean MSE."""
        mse_vals = []
        for pair_idx in range(n_pairs):
            seed_a = args.seed + pair_idx * 2
            seed_b = args.seed + pair_idx * 2 + 1
            img_a = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec, dtype=torch.float32),
                                   cfg, device, seed_a)
            img_b = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec, dtype=torch.float32),
                                   cfg, device, seed_b)
            mse_vals.append(mse(img_a, img_b))
        mean = float(np.mean(mse_vals))
        std  = float(np.std(mse_vals))
        if label:
            print(f"  {label:<35} MSE = {mean:.5f} ± {std:.5f}")
        return mean, std

    # baseline: all features at 0.5 (fully random circles)
    vec_all_random = [0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5]
    base_mean, base_std = measure_mse_vec(vec_all_random, args.n_pairs,
                                          "all random (baseline)")
    results["all_random"] = {"mean_mse": base_mean, "std_mse": base_std,
                             "reduction": 0.0, "reduction_pct": 0.0}

    # fix each continuous feature to 0 (extreme → no stochastic variation in that dim)
    # and measure MSE reduction vs baseline
    print()
    fixing_results = {}
    for feat_idx, feat_name in continuous_features:
        vec = vec_all_random.copy()
        vec[feat_idx] = 0.0   # fix to extreme
        mean, std = measure_mse_vec(vec, args.n_pairs, f"fix {feat_name}=0")
        reduction = base_mean - mean
        reduction_pct = 100 * reduction / base_mean
        fixing_results[feat_name] = {
            "mean_mse": mean, "std_mse": std,
            "reduction": reduction, "reduction_pct": reduction_pct,
        }

    # also try fixing to 1
    print()
    for feat_idx, feat_name in continuous_features:
        vec = vec_all_random.copy()
        vec[feat_idx] = 1.0
        mean, std = measure_mse_vec(vec, args.n_pairs, f"fix {feat_name}=1")
        # keep whichever extreme gives larger reduction
        if fixing_results[feat_name]["reduction"] < base_mean - mean:
            fixing_results[feat_name] = {
                "mean_mse": mean, "std_mse": std,
                "reduction": base_mean - mean,
                "reduction_pct": 100 * (base_mean - mean) / base_mean,
                "fixed_at": 1,
            }
        else:
            fixing_results[feat_name]["fixed_at"] = 0

    # all features fixed (pure network noise floor)
    vec_all_fixed = [0.0, 0.0, 1.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0]
    floor_mean, floor_std = measure_mse_vec(vec_all_fixed, args.n_pairs,
                                            "\nall fixed (noise floor)")
    results["all_fixed"] = {"mean_mse": floor_mean, "std_mse": floor_std,
                            "reduction": base_mean - floor_mean,
                            "reduction_pct": 100 * (base_mean - floor_mean) / base_mean}

    # sort by reduction
    sorted_feats = sorted(fixing_results.items(),
                          key=lambda x: x[1]["reduction"], reverse=True)

    print(f"\n  {'Feature':<12} {'Fixed at':>9}  {'MSE':>10}  {'Reduction':>12}  {'% reduction':>12}")
    print(f"  {'-'*60}")
    print(f"  {'all_random':<12} {'':>9}  {base_mean:>10.5f}  {'':>12}  {'':>12}")
    for feat_name, vals in sorted_feats:
        print(f"  {feat_name:<12} {'='+str(vals.get('fixed_at',0)):>9}  "
              f"{vals['mean_mse']:>10.5f}  "
              f"{vals['reduction']:>+12.5f}  "
              f"{vals['reduction_pct']:>11.1f}%")
    print(f"  {'all_fixed':<12} {'':>9}  {floor_mean:>10.5f}  "
          f"{results['all_fixed']['reduction']:>+12.5f}  "
          f"{results['all_fixed']['reduction_pct']:>11.1f}%")

    results.update(fixing_results)

    # ------------------------------------------------------------------
    # Continuous features
    # ------------------------------------------------------------------
    for feat_idx, feat_name in continuous_features:
        print(f"\n--- Feature: {feat_name} (index {feat_idx}) ---")
        mse_vals = []

        # set tested feature to 0.5 (max stochastic variation), all others fixed at 0.0
        vec = base_fixed.copy()
        vec[feat_idx] = 0.5

        for pair_idx in range(args.n_pairs):
            # same conditioning vector, different seeds → variation comes only from
            # stochastic rendering of the tested feature
            seed_a = args.seed + pair_idx * 2
            seed_b = args.seed + pair_idx * 2 + 1
            img_a = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec, dtype=torch.float32),
                                   cfg, device, seed_a)
            img_b = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec, dtype=torch.float32),
                                   cfg, device, seed_b)
            mse_vals.append(mse(img_a, img_b))

            if (pair_idx + 1) % 10 == 0:
                print(f"  [{pair_idx+1}/{args.n_pairs}] mean MSE: {np.mean(mse_vals):.5f}")

        results[feat_name] = {
            "mean_mse": float(np.mean(mse_vals)),
            "std_mse":  float(np.std(mse_vals)),
        }
        print(f"  {feat_name}: mean MSE = {results[feat_name]['mean_mse']:.5f} "
              f"± {results[feat_name]['std_mse']:.5f}")

    # ------------------------------------------------------------------
    # Shape pairs (one-hot features)
    # ------------------------------------------------------------------
    for (idx_a, name_a), (idx_b, name_b) in shape_pairs:
        feat_name = f"{name_a}_vs_{name_b}"
        print(f"\n--- Feature: {feat_name} ---")
        mse_vals = []

        vec_a_base = base_fixed.copy(); vec_a_base[idx_a] = 1.0; vec_a_base[idx_b] = 0.0
        vec_b_base = base_fixed.copy(); vec_b_base[idx_a] = 0.0; vec_b_base[idx_b] = 1.0

        for pair_idx in range(args.n_pairs):
            seed_pair = args.seed + pair_idx * 2
            img_a = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec_a_base, dtype=torch.float32),
                                   cfg, device, seed_pair)
            img_b = generate_image(unet, conditioner, ddim,
                                   torch.tensor(vec_b_base, dtype=torch.float32),
                                   cfg, device, seed_pair)
            mse_vals.append(mse(img_a, img_b))

            if (pair_idx + 1) % 10 == 0:
                print(f"  [{pair_idx+1}/{args.n_pairs}] mean MSE: {np.mean(mse_vals):.5f}")

        results[feat_name] = {
            "mean_mse": float(np.mean(mse_vals)),
            "std_mse":  float(np.std(mse_vals)),
        }
        print(f"  {feat_name}: mean MSE = {results[feat_name]['mean_mse']:.5f} "
              f"± {results[feat_name]['std_mse']:.5f}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"{'Feature':<30}  {'Mean MSE':>10}  {'Std':>10}")
    print(f"{'-'*55}")
    for name, vals in results.items():
        print(f"  {name:<28}  {vals['mean_mse']:>10.5f}  {vals['std_mse']:>10.5f}")
    print(f"{'='*55}")

    # ------------------------------------------------------------------
    # Save text summary
    # ------------------------------------------------------------------
    txt = "Per-feature MSE contribution\n"
    txt += f"n_pairs={args.n_pairs}, seed={args.seed}\n"
    txt += "Method: all features fixed at 0.0, tested feature set to 0.5\n"
    txt += "        (stochastic rendering varies only the tested feature)\n\n"
    txt += f"{'Feature':<30}  {'Mean MSE':>12}  {'Std':>12}\n"
    txt += "-" * 58 + "\n"
    for name, vals in results.items():
        txt += f"{name:<30}  {vals['mean_mse']:>12.5f}  {vals['std_mse']:>12.5f}\n"
    (out_dir / "feature_mse.txt").write_text(txt)
    print(f"\nSaved summary → {out_dir / 'feature_mse.txt'}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    all_names = list(results.keys())
    mean_vals = [results[n]["mean_mse"] for n in all_names]
    std_vals  = [results[n]["std_mse"]  for n in all_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(all_names))
    ax.bar(x, mean_vals, yerr=std_vals, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Mean MSE between image pairs")
    ax.set_title("Per-feature MSE contribution\n"
                 "(all other features fixed at 0, tested feature at 0.5 → max stochastic variation)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_mse.png", dpi=150)
    plt.close()
    print(f"Saved plot → {out_dir / 'feature_mse.png'}")


if __name__ == "__main__":
    main()