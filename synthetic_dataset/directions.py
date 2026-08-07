"""
pca_analysis.py
---------------
Runs PCA on collected gradients and visualizes what each principal direction
controls by intervening on the conditioner output at inference time.

For each PCA direction:
  - Takes a base conditioning vector
  - Adds the direction at various strengths (both + and -)
  - Generates images at each strength
  - Saves a grid showing the effect

Analogous to the text-to-image intervention script but for your conditioner.

Usage:
    # single grads file
    python directions.py --checkpoint /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc/checkpoint-epoch-0200 --grads outputs/gradients/t_500/grads_t500.npy

    # compare two grads files (e.g. different timesteps)
    python pca_analysis.py --checkpoint outputs_64/checkpoint-epoch-0200 \\
        --grads outputs/gradients/grads_t500.npy outputs/gradients/grads_t300.npy

    # a directory: ALL .npy files in it are used (override with --glob_pattern)
    python pca_analysis.py --checkpoint outputs_64/checkpoint-epoch-0200 \\
        --grads outputs/gradients/

    # gradients from a conditioner with a different cond_input_dim (e.g. the
    # no-texture 7-dim variant: shape[3] + color[3] + size[1]) need a matching
    # --base_prompt of the same length:
    python pca_analysis.py --cond_input_dim 7 \\
        --base_prompt 0 0 1  0.5 0.5 0.5  0.5 \\
        --checkpoint outputs_64_no_text/checkpoint-epoch-0200 \\
        --grads outputs/gradients_no_text/
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder


# Canonical, ordered full feature list. Follows dataset_64.py's FEATURE_ORDER
# convention (shape, color, size, stripes, grain): any --cond_input_dim <= 10
# is assumed to be the leading prefix of this list (e.g. 7 = shape+color+size,
# dropping stripes/grain) -- matches how the dataset generator's --features
# flag composes subsets. If your conditioner used some other, non-prefix
# subset, pass --feature_names explicitly to override this assumption.
FULL_FEATURE_NAMES = ["is_tri", "is_sq", "is_circ", "r", "g", "b", "size", "h_stripe", "v_stripe", "grain"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",         type=str, default="outputs/checkpoints/outputs_64_acc_update1/checkpoint-epoch-0200")
    parser.add_argument("--grads",              type=str, nargs="+", default=["outputs/gradients/t_mult_only_text_5k_update1"],
                        help="One or more gradient files (.npy), OR a single directory "
                             "containing them -- ALL files matching --glob_pattern in that "
                             "directory are used, sorted numerically by any digits in the "
                             "filename (falling back to alphabetical for files with no "
                             "digits, e.g. non-timestep-named files).")
    parser.add_argument("--glob_pattern",       type=str, default="*.npy",
                        help="Glob pattern used when --grads is a single directory. "
                             "Default '*.npy' takes every .npy file in it. Narrow this "
                             "(e.g. 'grads_t*.npy') if the directory also contains "
                             "unrelated .npy files you want to exclude.")
    parser.add_argument("--out_dir",            type=str, default="outputs/interventions/t_mult_retrained")
    parser.add_argument("--num_steps",          type=int, default=10)
    parser.add_argument("--n_directions",       type=int, default=64,
                        help="Number of top PCA directions to visualize")
    parser.add_argument("--num_interventions",  type=int, default=10,
                        help="How many directions to visualize per dataset")
    parser.add_argument("--intervention_strengths", type=float, nargs="+",
                        default=[ 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5 , -0.6],
                        help="Strengths to apply each direction at")
    parser.add_argument("--imgs_per_strength",  type=int, default=4,
                        help="Images per intervention strength (columns)")
    parser.add_argument("--cond_input_dim",     type=int, default=10,
                        help="Dimensionality of the raw conditioning vector your "
                             "gradients/conditioner were trained with. Default 10 is "
                             "the full shape+color+size+stripes+grain vector; e.g. use "
                             "7 for the no-texture (shape+color+size) variant. "
                             "--base_prompt and --feature_names must match this length.")
    parser.add_argument("--base_prompt",        type=float, nargs="+",
                        default=[0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],
                        metavar="V",
                        help="Base conditioning vector for intervention images, length "
                             "must equal --cond_input_dim. Fixed values stay fixed; 0.5 "
                             "values are randomised across seeds. Default (10 values) "
                             "matches the default --cond_input_dim=10 ordering: "
                             f"{FULL_FEATURE_NAMES}. E.g. for --cond_input_dim 7: "
                             "--base_prompt 0 0 1  0.9 0.1 0.1  0.8 gives a large red "
                             "circle (shape+color+size only, no texture dims to set).")
    parser.add_argument("--feature_names",      type=str, nargs="+", default=None,
                        help="Override the feature name labels used in output tables "
                             "(length must equal --cond_input_dim). Default: the first "
                             f"--cond_input_dim entries of {FULL_FEATURE_NAMES}, matching "
                             "dataset_64.py's --features prefix convention. Only need "
                             "this if your conditioner used some other, non-prefix subset.")
    parser.add_argument("--qr_drop_last",       type=int, default=1,
                        help="How many trailing QR columns to drop when projecting onto "
                             "the orthogonalized conditioner space, for rank deficiency. "
                             "Default 1 assumes the standard one-hot shape encoding "
                             "(is_tri/is_sq/is_circ sum to 1, so it spans rank 2 not 3, "
                             "i.e. rank(W) = cond_input_dim - 1). Set 0 if your W is "
                             "already full rank (e.g. no one-hot shape encoding at all).")
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--no_interventions",   action="store_true", default=True,
                        help="Skip intervention grid generation (dot products and cross-space analysis only)")
    parser.add_argument("--normalize_grads",    action="store_true", default=True,
                        help="L2-normalize each gradient to unit length before PCA. "
                             "Removes magnitude weighting so all pairs contribute equally "
                             "regardless of MSE loss size. More stable across seeds.")
    args = parser.parse_args()

    if len(args.base_prompt) != args.cond_input_dim:
        raise SystemExit(
            f"--base_prompt has {len(args.base_prompt)} values but --cond_input_dim="
            f"{args.cond_input_dim}. Pass exactly {args.cond_input_dim} values, e.g. for "
            f"the shape+color+size (7-dim) case: --base_prompt 0 0 1  0.5 0.5 0.5  0.5"
        )
    if args.feature_names is not None and len(args.feature_names) != args.cond_input_dim:
        raise SystemExit(
            f"--feature_names has {len(args.feature_names)} entries but --cond_input_dim="
            f"{args.cond_input_dim}. Pass exactly {args.cond_input_dim} names, or omit "
            f"--feature_names to use the default prefix-of-{FULL_FEATURE_NAMES} convention."
        )
    return args


def resolve_feature_names(cond_input_dim: int, override: list = None) -> list:
    """Default: the leading `cond_input_dim` entries of FULL_FEATURE_NAMES,
    matching dataset_64.py's prefix-subset convention (e.g. cond_input_dim=7
    -> shape+color+size). Falls back to generic f"f{i}" labels for any
    dims beyond the known 10 (e.g. a conditioner with extra custom inputs).
    An explicit --feature_names always wins if provided."""
    if override is not None:
        return list(override)
    if cond_input_dim <= len(FULL_FEATURE_NAMES):
        return FULL_FEATURE_NAMES[:cond_input_dim]
    return FULL_FEATURE_NAMES + [f"f{i}" for i in range(len(FULL_FEATURE_NAMES), cond_input_dim)]


def expand_grad_paths(grads: list, glob_pattern: str) -> list:
    """If `grads` is a single directory, expand it to every file matching
    `glob_pattern` inside it, sorted numerically by any digits in the
    filename (grads_t500 sorts after grads_t50, not alphabetically before
    it); files with no digits at all sort after all numbered ones,
    alphabetically among themselves, instead of raising on the digit
    extraction (the previous behavior would crash on a non-numeric
    filename). Otherwise returns the explicit list of paths as-is."""
    if len(grads) == 1 and Path(grads[0]).is_dir():
        grads_dir = Path(grads[0])
        def sort_key(p):
            digits = ''.join(filter(str.isdigit, p.stem))
            return (int(digits), p.stem) if digits else (float("inf"), p.stem)
        found = sorted(grads_dir.glob(glob_pattern), key=sort_key)
        if not found:
            raise FileNotFoundError(f"No files matching '{glob_pattern}' found in {grads_dir}")
        print(f"Expanded directory {grads_dir} -> {len(found)} files:")
        for p in found:
            print(f"  {p}")
        return [str(p) for p in found]
    return grads


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def find_significant_directions(grads: np.ndarray, n_directions: int):
    """
    PCA on gradient matrix.
    grads: (N, D)
    Returns:
        directions: (k, D) top principal directions
        variances:  (k,)  explained variance ratios
        eigvals:    (k,)  raw eigenvalues
        cov_matrix: (D, D) covariance matrix
    """
    centered = grads - grads.mean(axis=0, keepdims=True)
    N, D = centered.shape

    if N < D:
        C_small  = (1.0 / (N - 1)) * centered @ centered.T
        eigvals, eigvecs_small = np.linalg.eigh(C_small)
        idx      = np.argsort(eigvals)[::-1]
        eigvals  = eigvals[idx]
        eigvecs_small = eigvecs_small[:, idx]
        valid    = eigvals > 1e-9
        eigvals  = eigvals[valid]
        eigvecs_small = eigvecs_small[:, valid]
        scale    = 1.0 / np.sqrt(eigvals * (N - 1))
        eigvecs  = centered.T @ eigvecs_small * scale[np.newaxis, :]
        cov      = (1.0 / (N - 1)) * centered.T @ centered
    else:
        cov      = (1.0 / (N - 1)) * centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx      = np.argsort(eigvals)[::-1]
        eigvals  = eigvals[idx]
        eigvecs  = eigvecs[:, idx]

    total_var  = eigvals.sum()
    explained  = eigvals / total_var if total_var > 0 else eigvals * 0
    k          = min(n_directions, eigvecs.shape[1])
    directions = eigvecs[:, :k].T       # (k, D)
    variances  = explained[:k]

    return (
        torch.from_numpy(directions.astype(np.float32)),
        torch.from_numpy(variances.astype(np.float32)),
        torch.tensor(total_var, dtype=torch.float32),
        torch.from_numpy(cov.astype(np.float32)),
    )


# ---------------------------------------------------------------------------
# Intervention generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_intervention(
    unet, conditioner, ddim, base_cond_vec, direction, strength, cfg, device, seed
):
    """Generate one image with cond_out shifted by direction * strength."""
    torch.manual_seed(seed)
    image    = torch.randn(1, cfg.unet_in_channels, cfg.image_size, cfg.image_size, device=device)
    cond_out = conditioner(base_cond_vec.unsqueeze(0).to(device))   # (1, 1, 64)
    # intervene: shift the embedding
    cond_out = cond_out + strength * direction.to(device).view(1, 1, -1)
    for t in ddim.timesteps:
        noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
        image      = ddim.step(noise_pred, t, image).prev_sample
    image = (image.clamp(-1, 1) + 1) / 2
    return to_pil_image(image.squeeze().float().cpu())


def make_intervention_grid(images_2d, strengths, direction_idx, variance, scale=4):
    """
    images_2d: list of lists [strength][img_idx] → PIL image
    rows = starting images (6), cols = intervention strengths
    """
    n_strengths = len(strengths)
    n_images    = len(images_2d[0])
    W, H        = images_2d[0][0].size
    label_h     = 20
    label_w     = 60
    cell_w      = W * scale
    cell_h      = H * scale
    pad         = 4
    grid_w      = label_w + n_strengths * (cell_w + pad) + pad
    grid_h      = label_h + n_images * (cell_h + pad) + pad
    grid        = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw        = ImageDraw.Draw(grid)

    draw.text((4, 4), f"Dir {direction_idx+1}  var={variance*100:.1f}%", fill=(0, 0, 0))

    # column headers — intervention strengths
    for c, strength in enumerate(strengths):
        x = label_w + pad + c * (cell_w + pad)
        draw.text((x + 2, label_h - 14), f"{strength:+g}", fill=(0, 0, 0))

    # rows = starting images, cols = strengths
    for img_idx in range(n_images):
        y = label_h + pad + img_idx * (cell_h + pad)
        for c, strength_imgs in enumerate(images_2d):
            x   = label_w + pad + c * (cell_w + pad)
            img = strength_imgs[img_idx]
            grid.paste(img.resize((cell_w, cell_h), Image.NEAREST), (x, y))

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    args.grads = expand_grad_paths(args.grads, args.glob_pattern)

    cfg     = Config(cond_input_dim=args.cond_input_dim)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt    = Path(args.checkpoint)
    feature_names = resolve_feature_names(args.cond_input_dim, args.feature_names)
    qr_rank = max(1, args.cond_input_dim - args.qr_drop_last)

    # ------------------------------------------------------------------
    # Load models (only needed for intervention grids)
    # ------------------------------------------------------------------
    if not args.no_interventions:
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
    else:
        print("Skipping model loading (--no_interventions set)")
        unet        = None
        conditioner = ShapeConditioningEncoder(
            cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim,
        )
        conditioner.load_state_dict(
            torch.load(ckpt / "conditioner.pt", map_location="cpu", weights_only=True)
        )
        conditioner.eval()

    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    # base conditioning vectors — only needed for intervention grids
    if not args.no_interventions:
        torch.manual_seed(args.seed)
        rng_base = torch.Generator()
        rng_base.manual_seed(args.seed)
        base_prompt   = args.base_prompt
        imgs_per_base = 6
        base_conds    = []
        for _ in range(imgs_per_base):
            vec = []
            for v in base_prompt:
                if v == 0.5:
                    vec.append(torch.rand(1, generator=rng_base).item())
                else:
                    vec.append(v)
            base_conds.append(torch.tensor(vec, dtype=torch.float32))
        print(f"Base prompt: {base_prompt}")
        print(f"Generated {len(base_conds)} base conditioning vectors (0.5 dims randomised)")

    # ------------------------------------------------------------------
    # PCA per grads file
    # ------------------------------------------------------------------
    pca_results = {}
    for grads_path in args.grads:
        grads_path = Path(grads_path)
        label      = grads_path.stem   # e.g. "grads_t500"
        grads      = np.load(grads_path).astype(np.float32)
        if args.normalize_grads:
            norms = np.linalg.norm(grads, axis=1, keepdims=True)
            grads = grads / (norms + 1e-8)
            print(f"  Gradients L2-normalized (mean norm before: {norms.mean():.4f})")
        print(f"\nPCA on {label} | shape: {grads.shape}")

        directions, variances, total_var, cov = find_significant_directions(
            grads, n_directions=args.n_directions
        )
        pca_results[label] = (directions, variances, total_var, cov)

        print(f"  Top {len(variances)} directions explained variance:")
        for i, (v, ev) in enumerate(zip(variances, variances * total_var)):
            print(f"    Dir {i+1:2d}: {v*100:.3f}%")

        # scree plot
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(variances)), variances.numpy() * 100)
        plt.xlabel("Component"); plt.ylabel("Explained variance (%)")
        plt.title(f"PCA scree — {label}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scree_{label}.png", dpi=150)
        plt.close()
        print(f"  Saved scree plot → {out_dir / f'scree_{label}.png'}")

    # ------------------------------------------------------------------
    # Dot products between gradient directions and conditioner weight columns
    # ------------------------------------------------------------------
    print(f"\n--- Dot products: gradient dirs vs conditioner input features ---")
    W = conditioner.proj.weight.detach().cpu()   # (64, cond_input_dim)
    W_normalized = F.normalize(W, dim=0)   # normalize each column (feature direction)

    for label, (directions, variances, _, _) in pca_results.items():
        n_meaningful = min(args.num_interventions, len(directions))
        grad_dirs = F.normalize(directions[:n_meaningful], dim=1)   # (k, 64)
        dots = grad_dirs @ W_normalized   # (k, cond_input_dim)

        row_norms = torch.norm(dots, dim=1)   # (k,) — projection ratio (||d_proj||/||d||, d is unit)
        col_norms = torch.norm(dots, dim=0)   # (cond_input_dim,)

        print(f"\n  {label} — top {n_meaningful} gradient directions:")
        header = "         " + "".join(f"{n:>10}" for n in feature_names) + "  proj_ratio"
        print(header)
        for i in range(n_meaningful):
            row = f"  Dir {i+1:2d} " + "".join(f"{dots[i,j].item():>+10.3f}" for j in range(len(feature_names)))
            row += f"  {row_norms[i].item():>10.3f}"
            print(row)
        col_row = "  col_norm" + "".join(f"{col_norms[j].item():>+10.3f}" for j in range(len(feature_names)))
        print(col_row)

        txt = header + "\n"
        for i in range(n_meaningful):
            txt += f"  Dir {i+1:2d} " + "".join(f"{dots[i,j].item():>+10.3f}" for j in range(len(feature_names)))
            txt += f"  {row_norms[i].item():>10.3f}\n"
        txt += col_row + "\n"
        (out_dir / f"dot_products_{label}.txt").write_text(txt)
        print(f"  Saved → {out_dir / f'dot_products_{label}.txt'}")

        # ------------------------------------------------------------------
        # Second table: projection onto orthogonalized (QR) ground truth space
        # ------------------------------------------------------------------
        Q, _        = torch.linalg.qr(W_normalized)   # Q: (64, cond_input_dim), orthonormal columns
        Qk          = Q[:, :qr_rank]                   # drop trailing cols for rank deficiency (one-hot constraint)
        dots_q      = grad_dirs @ Qk                    # (k, qr_rank)
        proj_ratios = torch.norm(dots_q, dim=1)         # (k,) — guaranteed <= 1
        col_norms_q = torch.norm(dots_q, dim=0)         # (qr_rank,)

        print(f"\n  {label} — projected onto orthogonalized conditioner space (QR, {qr_rank} dims):")
        header_q = "         " + "".join(f"{f'Q{j+1}':>10}" for j in range(qr_rank)) + "  proj_ratio"
        print(header_q)
        for i in range(n_meaningful):
            row = f"  Dir {i+1:2d} " + "".join(f"{dots_q[i,j].item():>+10.3f}" for j in range(qr_rank))
            row += f"  {proj_ratios[i].item():>10.3f}"
            print(row)
        col_row_q = "  col_norm" + "".join(f"{col_norms_q[j].item():>+10.3f}" for j in range(qr_rank))
        print(col_row_q)

        txt_q = header_q + "\n"
        for i in range(n_meaningful):
            txt_q += f"  Dir {i+1:2d} " + "".join(f"{dots_q[i,j].item():>+10.3f}" for j in range(qr_rank))
            txt_q += f"  {proj_ratios[i].item():>10.3f}\n"
        txt_q += col_row_q + "\n"
        (out_dir / f"dot_products_qr_{label}.txt").write_text(txt_q)
        print(f"  Saved → {out_dir / f'dot_products_qr_{label}.txt'}")

    # ------------------------------------------------------------------
    # Pairwise cross-space dot product tables
    # ------------------------------------------------------------------
    labels = list(pca_results.keys())
    if len(labels) > 1:
        threshold = 2.5 / 64   # significance threshold (same as ELROND paper)
        print(f"\n--- Pairwise dot products between gradient spaces (natural cutoff, threshold={threshold:.4f}) ---")
        for i, label_a in enumerate(labels):
            for j, label_b in enumerate(labels):
                if j <= i:
                    continue
                # select only directions above variance threshold for each dataset
                dirs_a, vars_a = pca_results[label_a][0], pca_results[label_a][1]
                dirs_b, vars_b = pca_results[label_b][0], pca_results[label_b][1]
                mask_a = vars_a > threshold
                mask_b = vars_b > threshold
                ka = max(1, mask_a.sum().item())
                kb = max(1, mask_b.sum().item())
                D_A = F.normalize(dirs_a[:ka], dim=1)   # (ka, 64)
                D_B = F.normalize(dirs_b[:kb], dim=1)   # (kb, 64)
                cross = D_A @ D_B.T   # (ka, kb)

                print(f"\n  {label_a} ({ka} dirs) vs {label_b} ({kb} dirs):")
                header = "         " + "".join(f"{f'B{c+1}':>8}" for c in range(cross.shape[1]))
                print(header)
                for r in range(cross.shape[0]):
                    row = f"  A{r+1:2d}    " + "".join(f"{cross[r,c].item():>+8.3f}" for c in range(cross.shape[1]))
                    print(row)

                txt = f"Pairwise dot products: {label_a} ({ka} dirs, rows) vs {label_b} ({kb} dirs, cols)\n"
                txt += f"Variance threshold: {threshold:.4f}\n"
                txt += header + "\n"
                for r in range(cross.shape[0]):
                    txt += f"  A{r+1:2d}    " + "".join(f"{cross[r,c].item():>+8.3f}" for c in range(cross.shape[1])) + "\n"

                # summary metrics — now potentially asymmetric when ka != kb
                cross_sq   = cross ** 2
                avg_A_in_B = cross_sq.sum(dim=1).mean().item()   # mean ||D[i,:]||^2 over A dirs
                avg_B_in_A = cross_sq.sum(dim=0).mean().item()   # mean ||D[:,j]||^2 over B dirs
                symmetric  = 0.5 * (avg_A_in_B + avg_B_in_A)
                summary = (
                    f"\n  k_A={ka}, k_B={kb}  (asymmetric when k_A != k_B)\n"
                    f"  A explained by B  (mean ||D[i,:]||^2 over A dirs): {avg_A_in_B:.4f}\n"
                    f"  B explained by A  (mean ||D[:,j]||^2 over B dirs): {avg_B_in_A:.4f}\n"
                    f"  Symmetric avg                                     : {symmetric:.4f}\n"
                )
                print(summary)
                txt += summary
                (out_dir / f"cross_{label_a}_vs_{label_b}.txt").write_text(txt)
                print(f"  Saved → {out_dir / f'cross_{label_a}_vs_{label_b}.txt'}")

    # ------------------------------------------------------------------
    # Shared gradient space: PCA on concatenated gradients from all timesteps
    # ------------------------------------------------------------------
    if len(args.grads) > 1:
        print(f"\n--- PCA on shared (concatenated) gradient space ---")
        all_grads_list = []
        for grads_path in args.grads:
            g = np.load(grads_path).astype(np.float32)
            if args.normalize_grads:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                g = g / (norms + 1e-8)
            all_grads_list.append(g)
        shared_grads = np.concatenate(all_grads_list, axis=0)   # (N_total, 64)
        print(f"  Shared grads matrix: {shared_grads.shape}")

        shared_dirs, shared_vars, shared_tv, _ = find_significant_directions(
            shared_grads, n_directions=args.n_directions
        )
        pca_results["shared"] = (shared_dirs, shared_vars, shared_tv, None)

        print(f"  Top {args.num_interventions} shared directions explained variance:")
        for i in range(min(args.num_interventions, len(shared_vars))):
            print(f"    Dir {i+1:2d}: {shared_vars[i]*100:.3f}%")

        # scree plot
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(shared_vars)), shared_vars.numpy() * 100)
        plt.xlabel("Component"); plt.ylabel("Explained variance (%)")
        plt.title("PCA scree — shared gradient space")
        plt.tight_layout()
        plt.savefig(out_dir / "scree_shared.png", dpi=150)
        plt.close()
        print(f"  Saved scree plot → {out_dir / 'scree_shared.png'}")

        # dot product table vs ground truth
        n_meaningful = min(args.num_interventions, len(shared_dirs))
        shared_dirs_norm = F.normalize(shared_dirs[:n_meaningful], dim=1)
        dots_s = shared_dirs_norm @ W_normalized
        row_norms_s = torch.norm(dots_s, dim=1)
        col_norms_s = torch.norm(dots_s, dim=0)

        print(f"\n  shared — top {n_meaningful} directions vs conditioner features:")
        header_s = "         " + "".join(f"{n:>10}" for n in feature_names) + "  proj_ratio"
        print(header_s)
        for i in range(n_meaningful):
            row = f"  Dir {i+1:2d} " + "".join(f"{dots_s[i,j].item():>+10.3f}" for j in range(len(feature_names)))
            row += f"  {row_norms_s[i].item():>10.3f}"
            print(row)
        col_row_s = "  col_norm" + "".join(f"{col_norms_s[j].item():>+10.3f}" for j in range(len(feature_names)))
        print(col_row_s)
        txt_s = header_s + "\n"
        for i in range(n_meaningful):
            txt_s += f"  Dir {i+1:2d} " + "".join(f"{dots_s[i,j].item():>+10.3f}" for j in range(len(feature_names)))
            txt_s += f"  {row_norms_s[i].item():>10.3f}\n"
        txt_s += col_row_s + "\n"
        (out_dir / "dot_products_shared.txt").write_text(txt_s)

        # QR table
        Q, _          = torch.linalg.qr(W_normalized)
        Qk            = Q[:, :qr_rank]
        dots_sq       = shared_dirs_norm @ Qk
        proj_ratios_s = torch.norm(dots_sq, dim=1)
        col_norms_sq  = torch.norm(dots_sq, dim=0)
        header_sq = "         " + "".join(f"{f'Q{j+1}':>10}" for j in range(qr_rank)) + "  proj_ratio"
        txt_sq = header_sq + "\n"
        for i in range(n_meaningful):
            txt_sq += f"  Dir {i+1:2d} " + "".join(f"{dots_sq[i,j].item():>+10.3f}" for j in range(qr_rank))
            txt_sq += f"  {proj_ratios_s[i].item():>10.3f}\n"
        txt_sq += "  col_norm" + "".join(f"{col_norms_sq[j].item():>+10.3f}" for j in range(qr_rank)) + "\n"
        (out_dir / "dot_products_qr_shared.txt").write_text(txt_sq)
        print(f"  Saved dot product tables → {out_dir}/dot_products_shared.txt, dot_products_qr_shared.txt")

        # interventions for shared directions
        if not args.no_interventions:
            ddim_local = DDIMScheduler.from_config(ddpm.config)
            ddim_local.set_timesteps(args.num_steps)
            torch.manual_seed(args.seed)
            seeds_shared = [torch.randint(0, 2**31, (1,)).item() for _ in base_conds]
            print(f"\n--- Interventions for shared gradient directions ---")
            for dir_idx in range(n_meaningful):
                direction = F.normalize(shared_dirs[dir_idx], dim=0)
                variance  = shared_vars[dir_idx].item()
                images_2d = []
                for strength in args.intervention_strengths:
                    row_imgs = []
                    for base_cond, seed in zip(base_conds, seeds_shared):
                        img = generate_with_intervention(
                            unet, conditioner, ddim_local,
                            base_cond, direction, strength,
                            cfg, device, seed,
                        )
                        row_imgs.append(img)
                    images_2d.append(row_imgs)
                grid      = make_intervention_grid(images_2d, args.intervention_strengths, dir_idx, variance)
                grid_path = out_dir / f"shared_dir{dir_idx+1:02d}.png"
                grid.save(grid_path)
                print(f"  Saved shared dir {dir_idx+1} → {grid_path}")

    # ------------------------------------------------------------------
    # Intervention grids per individual gradient dataset
    # ------------------------------------------------------------------
    if not args.no_interventions:
        torch.manual_seed(args.seed)
        seeds_per_shape = [torch.randint(0, 2**31, (1,)).item() for _ in base_conds]

        for label, (directions, variances, _, _) in pca_results.items():
            if label == "shared":
                continue
            print(f"\n--- Interventions for {label} ---")
            n_to_show = min(args.num_interventions, len(directions))

            for dir_idx in range(n_to_show):
                direction = F.normalize(directions[dir_idx], dim=0)
                variance  = variances[dir_idx].item()
                images_2d = []

                for strength in args.intervention_strengths:
                    row_imgs = []
                    for base_cond, seed in zip(base_conds, seeds_per_shape):
                        img = generate_with_intervention(
                            unet, conditioner, ddim,
                            base_cond, direction, strength,
                            cfg, device, seed,
                        )
                        row_imgs.append(img)
                    images_2d.append(row_imgs)

                grid      = make_intervention_grid(images_2d, args.intervention_strengths, dir_idx, variance)
                grid_path = out_dir / f"{label}_dir{dir_idx+1:02d}.png"
                grid.save(grid_path)
                print(f"  Saved dir {dir_idx+1} → {grid_path}")

    print(f"\nDone. All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()