#!/usr/bin/env python
"""
residual_directions.py -- what PCA found that the ground truth does not contain
===============================================================================
The subspace comparison says the 6-dim ground-truth conditioning subspace sits
almost entirely inside the 8-dim PCA subspace (recall 0.97, precision 0.73).
That leaves roughly two directions of the PCA span unaccounted for. This
script isolates them, generates intervention grids so you can see what they
do, and measures how much they actually move the image against three
reference arms.

THE CONSTRUCTION
----------------
Take any orthonormal basis V of the k_pca-dimensional PCA space and the
orthonormal basis A of the k_gt-dimensional ground-truth space. Subtract from
each row of V its projection onto the GT space:

    R = V - (V A^T) A

The k_pca residual rows span the part of the PCA space orthogonal to GT. They
are linearly dependent by construction -- k_pca of them spanning at most
k_pca - k_gt dimensions -- so an SVD of R gives the basis.

WHAT THE SINGULAR VALUES MEAN
-----------------------------
R R^T = I - (V A^T)(A V^T), so the singular values of R are exactly
sin(theta_i) for the principal angles theta_i between the two subspaces. That
makes the spectrum readable rather than arbitrary:

  - a value at 1.0 is a direction of the PCA space with NO counterpart in GT
    (angle 90 degrees). There are k_pca - k_gt of these when GT is contained
    in the PCA span.
  - the rest are sin of the actual principal angles. At 10-13 degrees that is
    0.18-0.23, so the gap at the boundary is about 4x and the cut is clean.

If the gap is NOT clean, GT is not well contained in the PCA span and the
"leftover directions" framing does not hold. The script prints the ratio so
this is visible rather than assumed.

THE MSE COMPARISON (--measure_mse)
----------------------------------
Four arms, every direction unit-norm so a given strength means the same
displacement in every arm -- without that the comparison measures nothing:

  residual     the k_pca - k_gt directions above
  gt           the ground-truth basis rows, i.e. the labelled factors
  random_orth  random directions orthogonal to the WHOLE PCA span. The
               control that matters: if these move the image as much as the
               residual ones, the residual directions are not special, they
               are just perturbations of a given size.
  random       random directions with no constraint. The paper's Table 2
               baseline; expect it to sit between the two, since an
               unconstrained random vector has some component inside the
               PCA span.

For every (direction, base conditioning vector, strength) it samples with a
FIXED seed and compares against the strength-0 image from the same seed, then
reports mean pixel MSE. Seed matching is the whole point -- without it the
difference between two samples is dominated by the noise draw rather than by
the intervention.

Read it as: if residual > random_orth at matched strength, those two leftover
directions carry real structure the diffusion model responds to, and the PCA
space is finding something the labelled factors miss. If residual is at
random_orth level, they are estimation noise and the PCA dimension is
overstated.

Usage
-----
  python residual_directions.py \\
      --grads outputs/gradients/<run> \\
      --checkpoint outputs/checkpoints/<ckpt> \\
      --shape circle --k_pca 8 \\
      --output_dir runs/residual_circle \\
      --measure_mse
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subspace_comparison import (pca_on_gradients, variance_cut, apply_norm,
                                 principal_angles, assert_orthonormal)

SHAPE_PROMPTS = {
    # [is_tri, is_sq, is_circ, r, g, b, size, pos_x, pos_y]; 0.5 = swept
    "square":   [0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],   # K=3
    "triangle": [1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],   # K=4
    "circle":   [0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # K=6
}


# ---------------------------------------------------------------------------
# Subspace algebra
# ---------------------------------------------------------------------------

def residual_space(V_pca: torch.Tensor, A_gt: torch.Tensor, n_keep=None,
                   rtol=0.5):
    """Basis of span(V_pca) orthogonal to span(A_gt)."""
    R = V_pca - (V_pca @ A_gt.T) @ A_gt
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)

    k_pca, k_gt = V_pca.shape[0], A_gt.shape[0]
    if n_keep is None:
        # k_pca - k_gt is the dimension left over when GT sits inside the PCA
        # span, which is the case this script is for. The threshold count is
        # only a fallback for k_pca <= k_gt, where that difference says
        # nothing -- taking the max of the two would let a poorly-contained
        # GT space inflate the count to all of k_pca.
        n_keep = (k_pca - k_gt) if k_pca > k_gt else int((S > rtol).sum())
    n_keep = max(1, min(n_keep, int((S > 1e-8).sum())))

    B = Vh[:n_keep]
    gap = float(S[n_keep - 1] / S[n_keep]) if n_keep < len(S) else float("inf")
    return B, S, n_keep, gap


def random_orthogonal_to(V: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """n orthonormal directions orthogonal to span(V).

    Sampled, projected out of span(V), then QR'd -- so they are orthonormal
    among themselves as well as orthogonal to V, matching how the residual
    and GT bases are constructed. A set of merely unit-norm random vectors
    would have a different expected pairwise structure and would not be a
    like-for-like control.
    """
    d = V.shape[1]
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, d, generator=g)
    R = R - (R @ V.T) @ V
    Q, _ = torch.linalg.qr(R.T)
    return Q.T[:n].contiguous()


def random_unconstrained(d: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    R = torch.randn(n, d, generator=g)
    Q, _ = torch.linalg.qr(R.T)
    return Q.T[:n].contiguous()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_image(unet, conditioner, ddim, device, cond_vec, seed,
                 direction=None, strength=0.0):
    """One image, fully determined by (cond_vec, seed, direction, strength).

    Shapes come from unet.config rather than the project config, so this does
    not depend on attribute names that differ between setups.
    """
    ch = unet.config.in_channels
    size = 64
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(1, ch, size, size, generator=g).to(device)

    c = conditioner(cond_vec.unsqueeze(0).to(device))
    if c.dim() == 2:                       # (B, D) -> (B, 1, D)
        c = c.unsqueeze(1)
    if direction is not None and strength != 0.0:
        c = c + float(strength) * direction.view(1, 1, -1).to(device, c.dtype)

    for t in ddim.timesteps:
        eps = unet(x, t, encoder_hidden_states=c).sample
        x = ddim.step(eps, t, x).prev_sample
    # the model works in [-1, 1]; map to [0, 1] so the MSE numbers are on a
    # scale that reads directly as squared pixel error
    return (x.clamp(-1, 1) + 1.0) / 2.0


def measure_mse(arms, unet, conditioner, ddim, device, base_conds, seeds,
                strengths, max_dirs_per_arm):
    """Mean pixel MSE against the strength-0 image, per arm and strength.

    Baselines are computed once per (base, seed) and reused across every arm
    and direction -- they do not depend on the direction, and recomputing
    them would be both wasteful and a source of drift if anything about the
    sampler were nondeterministic.
    """
    print("\ncomputing seed-matched baselines ...")
    baselines = [sample_image(unet, conditioner, ddim, device, cv, sd)
                 for cv, sd in zip(base_conds, seeds)]

    results = {}
    for arm, D in arms.items():
        n_d = min(max_dirs_per_arm, D.shape[0]) if max_dirs_per_arm else D.shape[0]
        per_strength = {s: [] for s in strengths}
        print(f"  {arm}: {n_d} direction(s) x {len(base_conds)} base(s) "
              f"x {len(strengths)} strength(s)")
        for di in range(n_d):
            for cv, sd, base in zip(base_conds, seeds, baselines):
                for s in strengths:
                    img = sample_image(unet, conditioner, ddim, device, cv, sd,
                                       D[di], s)
                    per_strength[s].append(
                        float(F.mse_loss(img, base).item()))
        results[arm] = {
            "n_directions": n_d,
            "per_strength": {str(s): {"mean": float(np.mean(v)),
                                      "std": float(np.std(v)),
                                      "n": len(v)}
                             for s, v in per_strength.items()},
        }
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_spectrum(S, angles, n_keep, out_path, title):
    S = S.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(S)), S,
                color=["tab:red" if i < n_keep else "tab:blue" for i in range(len(S))],
                edgecolor="black", linewidth=0.4)
    axes[0].axhline(1.0, ls=":", c="gray", lw=1)
    axes[0].set_xlabel("residual component")
    axes[0].set_ylabel(r"singular value = $\sin\theta$")
    axes[0].set_title(f"Residual spectrum (red = kept, {n_keep} direction(s))")
    axes[0].set_ylim(0, 1.08)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    axes[1].bar(range(len(angles)), angles, color="tab:green",
                edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("principal angle index")
    axes[1].set_ylabel("degrees")
    axes[1].set_title("Principal angles, PCA space vs GT space")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mse(results, strengths, out_path, title, skip_arms=("random",)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"residual": ("tab:red", "o", "-"),
              "ground truth": ("tab:green", "s", "-"),
              "random_orthogonal": ("tab:blue", "^", "--"),
              "random": ("tab:gray", "v", ":")}
    for arm, r in results.items():
        if arm in skip_arms:
            continue
        c, m, ls = styles.get(arm, ("black", ".", "-"))
        means = [r["per_strength"][str(s)]["mean"] for s in strengths]
        stds = [r["per_strength"][str(s)]["std"] for s in strengths]
        ax.errorbar(strengths, means, yerr=stds, marker=m, color=c, ls=ls,
                    capsize=3, label=f"{arm} (n={r['n_directions']})")
    ax.set_xlabel("intervention strength")
    ax.set_ylabel("pixel MSE")
    #ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grads", nargs="+", default="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/gradients/t_mult_circle_mixed_dz0.1_pos1")
    p.add_argument("--glob_pattern", default="grads_t*.npy")
    p.add_argument("--checkpoint", default="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc_mixed_dz0.1_pos1/checkpoint-epoch-0200",
                   help="diffusion checkpoint dir (unet_ema/, conditioner.pt)")
    p.add_argument("--output_dir", required="./outputs/")
    p.add_argument("--cond_input_dim", type=int, default=9)
    p.add_argument("--shape", default="circle",
                   choices=["square", "triangle", "circle"])
    p.add_argument("--base_prompt", type=float, nargs="+", default=None)
    p.add_argument("--pca_norm", default="l2",
                   choices=["none", "l2", "standardize", "l2_then_standardize"])
    p.add_argument("--pca_variance_frac", type=float, default=0.95)
    p.add_argument("--k_pca", type=int, default=8,
                   help="fix the PCA dimension instead of the variance cut. "
                        "Set it to the k_sel your subspace_comparison run "
                        "reported, so both analyses use the same space.")
    p.add_argument("--n_residual", type=int, default=0,
                   help="residual directions to keep; 0 uses k_pca - k_gt")
    p.add_argument("--no_center", action="store_true")
    # grids
    p.add_argument("--num_interventions", type=int, default=4)
    p.add_argument("--intervention_strengths", type=float, nargs="+",
                   default=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0,
                            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--imgs_per_base", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--skip_interventions", action="store_true",
                   help="skip the image grids")
    # mse comparison
    p.add_argument("--measure_mse", action="store_true",
                   help="seed-matched pixel MSE across the four direction arms")
    p.add_argument("--mse_strengths", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                   help="strengths to measure; 0 is the implicit baseline")
    p.add_argument("--mse_bases", type=int, default=100,
                   help="base conditioning vectors (and seeds) per measurement")
    p.add_argument("--mse_max_dirs", type=int, default=6,
                   help="cap on directions per arm, so gt (6) does not cost "
                        "three times what residual (2) does")
    p.add_argument("--n_random", type=int, default=2,
                   help="how many random directions in each random arm")
    args = p.parse_args()

    if args.base_prompt is None:
        if args.shape is None:
            p.error("pass --shape or --base_prompt; k_gt is shape-dependent (3/4/6)")
        if args.cond_input_dim != 9:
            p.error("--shape assumes the 9-dim schema; pass --base_prompt")
        args.base_prompt = SHAPE_PROMPTS[args.shape]
        print(f"shape={args.shape} -> base_prompt={args.base_prompt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from vaease import load_gradients, build_alignment_tables, run_intervention_grids
    from configs.config_64 import Config64 as Config
    from models.conditioner import ShapeConditioningEncoder
    from directions import build_valid_subspace_basis, resolve_feature_names
    from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel

    # ---- PCA space -------------------------------------------------------
    data = load_gradients(args.grads, glob_pattern=args.glob_pattern)
    pca_data = apply_norm(data, args.pca_norm)
    V_full, evr, _ = pca_on_gradients(pca_data, center=not args.no_center)
    k_pca = args.k_pca or variance_cut(evr, args.pca_variance_frac)
    k_pca = max(1, min(k_pca, V_full.shape[0]))
    V_pca = V_full[:k_pca]
    assert_orthonormal(V_pca, "V_pca")
    print(f"PCA: N={len(data)} d={data.shape[1]} norm={args.pca_norm} -> k_pca={k_pca}"
          + ("" if args.k_pca else f" ({args.pca_variance_frac:.0%} variance)"))

    # ---- GT space --------------------------------------------------------
    cfg = Config(cond_input_dim=args.cond_input_dim)
    ck = Path(args.checkpoint)
    conditioner = ShapeConditioningEncoder(
        cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim).to(device)
    conditioner.load_state_dict(
        torch.load(ck / "conditioner.pt", map_location=device, weights_only=True))
    conditioner.eval()

    raw_basis, names = build_valid_subspace_basis(
        conditioner, args.base_prompt, resolve_feature_names(args.cond_input_dim),
        args.cond_input_dim, device, subtract_baseline=True)
    basis = torch.from_numpy(raw_basis.astype(np.float32))
    basis_norm = F.normalize(basis, dim=0)
    basis_ortho, _ = torch.linalg.qr(basis_norm)      # (d, k_gt) columns
    A_gt = basis_ortho.T                              # (k_gt, d) rows
    assert_orthonormal(A_gt, "A_gt")
    print(f"GT: k_gt={A_gt.shape[0]} {list(names)}")

    # ---- residual space --------------------------------------------------
    B, S, n_keep, gap = residual_space(V_pca, A_gt, n_keep=args.n_residual or None)
    ang = principal_angles(V_pca, A_gt)

    print("\nresidual singular values (= sin of the principal angles):")
    print("  " + "  ".join(f"{s:.4f}" for s in S.tolist()))
    print("principal angles (deg): " + "  ".join(f"{a:.2f}" for a in ang.tolist()))
    print(f"\nkeeping {n_keep} residual direction(s); boundary "
          f"{S[n_keep-1]:.4f} vs "
          + (f"{S[n_keep]:.4f} (ratio {gap:.1f}x)" if n_keep < len(S) else "--"))
    if gap < 2.0:
        print("  WARNING: small gap, so the ground-truth space is not cleanly "
              "contained in the PCA space. Treat these as 'least explained by "
              "GT' rather than 'orthogonal to GT'.")

    leak = float((B @ A_gt.T).abs().max())
    inside = float((B - (B @ V_pca.T) @ V_pca).abs().max())
    print(f"  max |B . A_gt| = {leak:.2e}   (0 = exactly orthogonal to GT)")
    print(f"  distance from span(V_pca) = {inside:.2e}")

    plot_spectrum(S, ang, n_keep, out_dir / "residual_spectrum.png",
                  f"k_pca={k_pca}, k_gt={A_gt.shape[0]}, "
                  f"norm={args.pca_norm}, shape={args.shape}")
    (out_dir / "residual_summary.json").write_text(json.dumps({
        "k_pca": int(k_pca), "k_gt": int(A_gt.shape[0]),
        "n_residual": int(n_keep),
        "singular_values": S.tolist(),
        "principal_angles_deg": ang.tolist(),
        "boundary_gap_ratio": gap,
        "max_overlap_with_gt": leak,
        "pca_norm": args.pca_norm, "shape": args.shape, "gt_names": list(names),
    }, indent=2))
    build_alignment_tables(B, basis_norm, basis_ortho, list(names),
                           "residual", out_dir)

    # ---- the four arms ---------------------------------------------------
    d = V_pca.shape[1]
    R_orth = random_orthogonal_to(V_pca, args.n_random, args.seed)
    R_free = random_unconstrained(d, args.n_random, args.seed)
    arms = {"residual": B, "gt": A_gt, "random_orth": R_orth, "random": R_free}

    print("\narms (all rows unit-norm, so strength means the same in each):")
    for k, D in arms.items():
        n = D.norm(dim=1)
        print(f"  {k:<12} {D.shape[0]:>2} direction(s)   norms "
              f"{n.min():.4f}-{n.max():.4f}   "
              f"max |. V_pca| = {float((D @ V_pca.T).abs().max()):.2e}")
    torch.save({**arms, "V_pca": V_pca, "A_gt": A_gt,
                "singular_values": S, "principal_angles_deg": ang},
               out_dir / "residual_directions.pt")

    if args.skip_interventions and not args.measure_mse:
        print(f"\nDone (no image generation). Outputs under {out_dir}/")
        return

    # ---- diffusion model -------------------------------------------------
    print(f"\nLoading UNet from {ck} ...")
    unet = UNet2DConditionModel.from_pretrained(ck / "unet_ema").to(device).eval()
    ddpm = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,
                         beta_schedule=cfg.beta_schedule,
                         prediction_type=cfg.prediction_type)
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    base_conds = [
        torch.tensor([torch.rand(1, generator=g).item() if v == 0.5 else v
                      for v in args.base_prompt], dtype=torch.float32)
        for _ in range(args.imgs_per_base)
    ]
    seeds = [torch.randint(0, 2 ** 31, (1,)).item() for _ in base_conds]

    # ---- grids -----------------------------------------------------------
    if not args.skip_interventions:
        run_intervention_grids(B, None, "residual", unet, conditioner, ddim, cfg,
                               device, base_conds, seeds,
                               args.intervention_strengths,
                               out_dir / "interventions_residual",
                               min(args.num_interventions, B.shape[0]))
        run_intervention_grids(R_orth, None, "random_orth", unet, conditioner,
                               ddim, cfg, device, base_conds, seeds,
                               args.intervention_strengths,
                               out_dir / "interventions_random_orth",
                               min(args.num_interventions, R_orth.shape[0]))

    # ---- MSE -------------------------------------------------------------
    if args.measure_mse:
        nb = min(args.mse_bases, len(base_conds))
        strengths = list(args.mse_strengths)
        total = sum(min(args.mse_max_dirs, D.shape[0]) for D in arms.values())
        print(f"\n=== pixel MSE, seed-matched ===")
        print(f"{total} direction(s) x {nb} base(s) x {len(strengths)} strength(s) "
              f"= {total * nb * len(strengths)} samples + {nb} baselines")
        res = measure_mse(arms, unet, conditioner, ddim, device,
                          base_conds[:nb], seeds[:nb], strengths,
                          args.mse_max_dirs)

        hdr = f"{'strength':>9}" + "".join(f"{a:>16}" for a in arms)
        print("\n" + hdr)
        print("-" * len(hdr))
        for s in strengths:
            row = f"{s:>9.2f}"
            for a in arms:
                st = res[a]["per_strength"][str(s)]
                row += f"{st['mean']:>10.5f}+-{st['std']:<4.3f}"
            print(row)

        ratios = []
        for s in strengths:
            r = res["residual"]["per_strength"][str(s)]["mean"]
            o = res["random_orth"]["per_strength"][str(s)]["mean"]
            ratios.append(r / o if o > 0 else float("nan"))
        print(f"\nresidual / random_orth: "
              + "  ".join(f"{x:.2f}" for x in ratios))
        print("  > 1 means the leftover PCA directions move the image more than "
              "an arbitrary\n  perturbation of the same size does -- i.e. they "
              "carry real structure.\n  ~ 1 means they are indistinguishable "
              "from noise at this strength.")

        (out_dir / "mse_comparison.json").write_text(json.dumps(
            {"arms": res, "strengths": strengths,
             "residual_over_random_orth": ratios,
             "n_bases": nb, "num_steps": args.num_steps}, indent=2))
        plot_mse(res, strengths, out_dir / "mse_comparison.png",
                 f"How far each direction moves the image "
                 f"({args.shape}, k_pca={k_pca}, k_gt={A_gt.shape[0]})")

    print(f"\nDone. Outputs under {out_dir}/")


if __name__ == "__main__":
    main()