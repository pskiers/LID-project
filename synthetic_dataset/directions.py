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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",         type=str, default="outputs/checkpoints/outputs_64_acc_update1/checkpoint-epoch-0200")
    parser.add_argument("--grads",              type=str, nargs="+", default=["outputs/gradients/t_mult_only_text_5k_update1"],
                        help="One or more grads_tXXX.npy files, or a single directory "
                             "containing them (all grads_t*.npy files will be used, sorted by timestep)")
    parser.add_argument("--out_dir",            type=str, default="outputs/interventions/t_mult_only_text_5k_update1")
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
    parser.add_argument("--base_prompt",   type=float, nargs=10,
                        default=[0.0, 0.0, 1.0,  0.5, 0.5, 0.5,  0.5,  0.5, 0.5, 0.5],
                        metavar=("is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                                 "h_stripe", "v_stripe", "grain"),
                        help="Base conditioning vector for intervention images. "
                             "Fixed values stay fixed; 0.5 values are randomised across seeds. "
                             "E.g. --base_prompt 0 0 1  0.9 0.1 0.1  0.8  0.5 0.5 0.5 "
                             "gives large red circles with random textures.")
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--no_interventions",   action="store_true", default=True,
                        help="Skip intervention grid generation (dot products and cross-space analysis only)")
    parser.add_argument("--normalize_grads",    action="store_true", default=True,
                        help="L2-normalize each gradient to unit length before PCA. "
                             "Removes magnitude weighting so all pairs contribute equally "
                             "regardless of MSE loss size. More stable across seeds.")
    return parser.parse_args()


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

    # expand directory to sorted list of grads_t*.npy files
    if len(args.grads) == 1 and Path(args.grads[0]).is_dir():
        grads_dir = Path(args.grads[0])
        found = sorted(grads_dir.glob("grads_t*.npy"),
                       key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
        if not found:
            raise FileNotFoundError(f"No grads_t*.npy files found in {grads_dir}")
        args.grads = [str(p) for p in found]
        print(f"Expanded directory {grads_dir} → {len(args.grads)} files:")
        for p in args.grads:
            print(f"  {p}")
    cfg     = Config(cond_input_dim=10)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt    = Path(args.checkpoint)

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
    W = conditioner.proj.weight.detach().cpu()   # (64, 10)
    W_normalized = F.normalize(W, dim=0)   # normalize each column (feature direction)
    feature_names = ["is_tri", "is_sq", "is_circ", "r", "g", "b", "size", "h_stripe", "v_stripe", "grain"]

    for label, (directions, variances, _, _) in pca_results.items():
        n_meaningful = min(args.num_interventions, len(directions))
        grad_dirs = F.normalize(directions[:n_meaningful], dim=1)   # (k, 64)
        dots = grad_dirs @ W_normalized   # (k, 10)

        row_norms = torch.norm(dots, dim=1)   # (k,) — projection ratio (||d_proj||/||d||, d is unit)
        col_norms = torch.norm(dots, dim=0)   # (10,)

        print(f"\n  {label} — top {n_meaningful} gradient directions:")
        header = "         " + "".join(f"{n:>10}" for n in feature_names) + "  proj_ratio"
        print(header)
        for i in range(n_meaningful):
            row = f"  Dir {i+1:2d} " + "".join(f"{dots[i,j].item():>+10.3f}" for j in range(10))
            row += f"  {row_norms[i].item():>10.3f}"
            print(row)
        col_row = "  col_norm" + "".join(f"{col_norms[j].item():>+10.3f}" for j in range(10))
        print(col_row)

        txt = header + "\n"
        for i in range(n_meaningful):
            txt += f"  Dir {i+1:2d} " + "".join(f"{dots[i,j].item():>+10.3f}" for j in range(10))
            txt += f"  {row_norms[i].item():>10.3f}\n"
        txt += col_row + "\n"
        (out_dir / f"dot_products_{label}.txt").write_text(txt)
        print(f"  Saved → {out_dir / f'dot_products_{label}.txt'}")

        # ------------------------------------------------------------------
        # Second table: projection onto orthogonalized (QR) ground truth space
        # ------------------------------------------------------------------
        Q, _ = torch.linalg.qr(W_normalized)   # Q: (64, 10), orthonormal columns
        Q9          = Q[:, :9]                  # drop Q10 — rank(W)=9 due to one-hot constraint
        dots_q      = grad_dirs @ Q9            # (k, 9)
        proj_ratios = torch.norm(dots_q, dim=1) # (k,) — guaranteed <= 1
        col_norms_q = torch.norm(dots_q, dim=0) # (9,)

        print(f"\n  {label} — projected onto orthogonalized conditioner space (QR, 9 dims):")
        header_q = "         " + "".join(f"{f'Q{j+1}':>10}" for j in range(9)) + "  proj_ratio"
        print(header_q)
        for i in range(n_meaningful):
            row = f"  Dir {i+1:2d} " + "".join(f"{dots_q[i,j].item():>+10.3f}" for j in range(9))
            row += f"  {proj_ratios[i].item():>10.3f}"
            print(row)
        col_row_q = "  col_norm" + "".join(f"{col_norms_q[j].item():>+10.3f}" for j in range(9))
        print(col_row_q)

        txt_q = header_q + "\n"
        for i in range(n_meaningful):
            txt_q += f"  Dir {i+1:2d} " + "".join(f"{dots_q[i,j].item():>+10.3f}" for j in range(9))
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
            row = f"  Dir {i+1:2d} " + "".join(f"{dots_s[i,j].item():>+10.3f}" for j in range(10))
            row += f"  {row_norms_s[i].item():>10.3f}"
            print(row)
        col_row_s = "  col_norm" + "".join(f"{col_norms_s[j].item():>+10.3f}" for j in range(10))
        print(col_row_s)
        txt_s = header_s + "\n"
        for i in range(n_meaningful):
            txt_s += f"  Dir {i+1:2d} " + "".join(f"{dots_s[i,j].item():>+10.3f}" for j in range(10))
            txt_s += f"  {row_norms_s[i].item():>10.3f}\n"
        txt_s += col_row_s + "\n"
        (out_dir / "dot_products_shared.txt").write_text(txt_s)

        # QR table
        Q, _ = torch.linalg.qr(W_normalized)
        Q9            = Q[:, :9]
        dots_sq       = shared_dirs_norm @ Q9
        proj_ratios_s = torch.norm(dots_sq, dim=1)
        col_norms_sq  = torch.norm(dots_sq, dim=0)
        header_sq = "         " + "".join(f"{f'Q{j+1}':>10}" for j in range(9)) + "  proj_ratio"
        txt_sq = header_sq + "\n"
        for i in range(n_meaningful):
            txt_sq += f"  Dir {i+1:2d} " + "".join(f"{dots_sq[i,j].item():>+10.3f}" for j in range(9))
            txt_sq += f"  {proj_ratios_s[i].item():>10.3f}\n"
        txt_sq += "  col_norm" + "".join(f"{col_norms_sq[j].item():>+10.3f}" for j in range(9)) + "\n"
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