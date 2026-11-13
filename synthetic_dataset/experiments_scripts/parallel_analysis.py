"""
parallel_analysis.py
--------------------
Horn's Parallel Analysis for determining which PCA directions are significant
vs noise, applied to gradient matrices from collect_gradients.py.

The key insight over a fixed threshold (e.g. 1/D):
  - A fixed threshold asks "is this direction above the average noise level?"
  - Parallel Analysis asks per-position: "is the k-th eigenvalue larger than
    the k-th eigenvalue you'd get from pure noise with the same N and D?"
  - This matters when the first few directions dominate: the remaining real
    directions capture a smaller share of explained variance but are still
    above their position-specific noise floor.

Method:
  1. Run PCA on real gradients → real_eigenvalues (sorted descending)
  2. For each simulation:
       - Shuffle each of the 64 columns independently (destroys correlations,
         preserves per-dimension distribution)
       - Run PCA → simulated_eigenvalues[sim, :]
  3. threshold[k] = percentile(simulated_eigenvalues[:, k], 95)
  4. Keep directions where real_eigenvalues[k] > threshold[k]

Usage:
    python parallel_analysis.py --grads outputs/gradients/t_500_no_anch/grads_t500.npy
    python parallel_analysis.py --grads outputs/gradients/t_mult/grads_t50.npy outputs/gradients/t_mult/grads_t500.npy outputs/gradients/t_mult/grads_t950.npy
    python parallel_analysis.py --grads outputs/gradients/t_500/grads_t500.npy --n_simulations 200 --percentile 99
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grads",          type=str, nargs="+", required=True,
                        help="One or more grads_tXXX.npy files to analyse, OR a single "
                             "directory containing them -- ALL files matching "
                             "--glob_pattern in that directory are used, and (unlike "
                             "passing explicit files) are COMBINED into one dataset for "
                             "a single Parallel Analysis, not analysed individually.")
    parser.add_argument("--glob_pattern",   type=str, default="*.npy",
                        help="Glob pattern used when --grads is a single directory. "
                             "Default '*.npy' takes every .npy file in it. Narrow this "
                             "(e.g. 'grads_t*.npy') if the directory also contains "
                             "unrelated .npy files you want to exclude.")
    parser.add_argument("--out_dir",        type=str, default="outputs/parralel_analysis/",
                        help="Where to save plots and results. Defaults to same dir as first grads file.")
    parser.add_argument("--n_simulations",  type=int, default=100,
                        help="Number of random permutation simulations (more = more stable threshold, slower)")
    parser.add_argument("--percentile",     type=float, default=95,
                        help="Percentile of noise distribution to use as threshold (default: 95)")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--normalize_grads", action="store_true", default=False,
                        help="L2-normalize each gradient to unit length before PCA.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PCA (eigenvalue decomposition on covariance matrix)
# ---------------------------------------------------------------------------

def expand_grad_paths(grads: list, glob_pattern: str) -> tuple:
    """If `grads` is a single directory, expands it to every file matching
    `glob_pattern` inside it, sorted numerically by any digits in the
    filename (grads_t500 sorts after grads_t50, not alphabetically before
    it); files with no digits at all sort after all numbered ones,
    alphabetically among themselves. Otherwise returns the explicit list
    of paths as-is.

    Returns (expanded_paths, combine_all): combine_all is True only when
    a directory was given -- signaling that these files should be
    CONCATENATED into one combined dataset and analysed as a single
    Parallel Analysis, not individually. An explicit file list (however
    many files) always gets combine_all=False and keeps the existing
    per-file behavior, unchanged.
    """
    if len(grads) == 1 and Path(grads[0]).is_dir():
        grads_dir = Path(grads[0])
        def sort_key(p):
            digits = ''.join(filter(str.isdigit, p.stem))
            return (int(digits), p.stem) if digits else (float("inf"), p.stem)
        found = sorted(grads_dir.glob(glob_pattern), key=sort_key)
        if not found:
            raise FileNotFoundError(f"No files matching '{glob_pattern}' found in {grads_dir}")
        print(f"Expanded directory {grads_dir} -> {len(found)} files "
              f"(will be COMBINED into one dataset, not analysed individually):")
        for p in found:
            print(f"  {p}")
        return [str(p) for p in found], True
    return grads, False


def pca_eigenvalues(data: np.ndarray) -> np.ndarray:
    """
    Returns eigenvalues of the covariance matrix, sorted descending.
    data: (N, D)
    Returns: (D,) eigenvalues (not explained variance ratios — raw eigenvalues)
    """
    centered = data - data.mean(axis=0, keepdims=True)
    N, D = centered.shape
    cov = (1.0 / (N - 1)) * centered.T @ centered   # (D, D)
    eigvals = np.linalg.eigvalsh(cov)                # ascending
    return eigvals[::-1].copy()                      # descending


# ---------------------------------------------------------------------------
# Parallel Analysis
# ---------------------------------------------------------------------------

def parallel_analysis(
    grads: np.ndarray,
    n_simulations: int = 100,
    percentile: float = 95,
    seed: int = 42,
) -> dict:
    """
    Horn's Parallel Analysis on gradient matrix.

    grads: (N, D) — gradient matrix
    Returns dict with:
        real_eigenvalues:    (D,)   eigenvalues of real data, sorted descending
        thresholds:          (D,)   noise percentile threshold per position
        real_var_ratios:     (D,)   explained variance ratios of real data
        threshold_var_ratios:(D,)   thresholds expressed as variance ratios
        significant_mask:    (D,)   bool — True where real > threshold
        n_significant:       int    — number of significant directions
        simulated_eigenvalues: (n_simulations, D) — full noise distribution
    """
    rng = np.random.RandomState(seed)
    N, D = grads.shape

    print(f"  Running PCA on real data ({N} samples, {D} dims) ...")
    real_eigvals = pca_eigenvalues(grads)
    total_var = real_eigvals.sum()
    real_var_ratios = real_eigvals / total_var

    print(f"  Running {n_simulations} noise simulations ...")
    simulated_eigvals = np.zeros((n_simulations, D), dtype=np.float32)

    for i in range(n_simulations):
        noise_data = np.zeros_like(grads)
        for col in range(D):
            noise_data[:, col] = rng.permutation(grads[:, col])
        simulated_eigvals[i, :] = pca_eigenvalues(noise_data)
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n_simulations}]")

    thresholds = np.percentile(simulated_eigvals, percentile, axis=0)   # (D,)
    threshold_var_ratios = thresholds / total_var

    significant_mask = real_eigvals > thresholds
    # find first position where real <= threshold — everything from there on is noise
    # (we want a contiguous block of significant directions from the top)
    n_significant = int(significant_mask.argmin()) if not significant_mask.all() else D
    # if argmin returns 0 but mask[0] is True, all are significant
    if significant_mask[0] and n_significant == 0:
        n_significant = D

    return {
        "real_eigenvalues":      real_eigvals,
        "thresholds":            thresholds,
        "real_var_ratios":       real_var_ratios,
        "threshold_var_ratios":  threshold_var_ratios,
        "significant_mask":      significant_mask,
        "n_significant":         n_significant,
        "simulated_eigenvalues": simulated_eigvals,
        "total_var":             total_var,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_parallel_analysis(result: dict, label: str, out_path: Path, percentile: float):
    D = len(result["real_eigenvalues"])
    x = np.arange(1, D + 1)

    real   = result["real_var_ratios"] * 100
    thresh = result["threshold_var_ratios"] * 100
    sims   = result["simulated_eigenvalues"] / result["total_var"] * 100
    n_sig  = result["n_significant"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: full scree with threshold ----
    ax = axes[0]
    # noise distribution band (5th–95th percentile)
    noise_lo = np.percentile(sims, 5,  axis=0)
    noise_hi = np.percentile(sims, 95, axis=0)
    ax.fill_between(x, noise_lo, noise_hi, alpha=0.25, color="gray", label="noise 5–95th pct")
    ax.plot(x, thresh, color="gray", linestyle="--", linewidth=1.5,
            label=f"noise {percentile:.0f}th pct (threshold)")
    ax.plot(x, real, color="steelblue", linewidth=2, marker="o", markersize=4, label="real data")
    if n_sig > 0:
        ax.axvline(n_sig + 0.5, color="red", linestyle=":", linewidth=1.5,
                   label=f"cutoff (k={n_sig})")
        ax.scatter(x[:n_sig], real[:n_sig], color="steelblue", zorder=5, s=40)
        ax.scatter(x[n_sig:], real[n_sig:], color="lightgray", zorder=5, s=40)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Parallel Analysis — {label}\n{n_sig} significant directions")
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, min(D, 30) + 0.5)   # show first 30 components

    # ---- Right: zoom on first n_sig + 5 components ----
    ax2 = axes[1]
    zoom = min(n_sig + 8, D)
    ax2.fill_between(x[:zoom], noise_lo[:zoom], noise_hi[:zoom],
                     alpha=0.25, color="gray", label="noise 5–95th pct")
    ax2.plot(x[:zoom], thresh[:zoom], color="gray", linestyle="--", linewidth=1.5,
             label=f"noise {percentile:.0f}th pct")
    ax2.plot(x[:zoom], real[:zoom], color="steelblue", linewidth=2,
             marker="o", markersize=6, label="real data")
    for i in range(zoom):
        color = "steelblue" if i < n_sig else "salmon"
        ax2.annotate(f"{real[i]:.1f}%", (x[i], real[i]),
                     textcoords="offset points", xytext=(0, 7),
                     fontsize=8, ha="center", color=color)
    if n_sig > 0:
        ax2.axvline(n_sig + 0.5, color="red", linestyle=":", linewidth=1.5,
                    label=f"cutoff (k={n_sig})")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Explained variance (%)")
    ax2.set_title(f"Zoom — first {zoom} components")
    ax2.legend(fontsize=9)

    plt.suptitle(f"{label}  |  N={len(result['real_eigenvalues'])} dirs, {percentile:.0f}th pct threshold",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.grads, combine_all = expand_grad_paths(args.grads, args.glob_pattern)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.grads[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all grads files and verify they share the same N and D
    # ------------------------------------------------------------------
    all_grads = {}
    for grads_path in args.grads:
        grads_path = Path(grads_path)
        g = np.load(grads_path).astype(np.float32)
        if args.normalize_grads:
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g / (norms + 1e-8)
        all_grads[grads_path.stem] = g

    if combine_all:
        # A directory was given -- concatenate every loaded file's pairs
        # into ONE combined dataset (matching pca_analysis.py's own
        # "shared gradient space" concept) and analyse THAT single
        # dataset, rather than each file individually. Normalization (if
        # requested) has already been applied per-file above, before
        # concatenation -- exactly as pca_analysis.py does it for its own
        # shared-space PCA.
        D_vals_pre = {g.shape[1] for g in all_grads.values()}
        if len(D_vals_pre) > 1:
            dims = {label: g.shape[1] for label, g in all_grads.items()}
            raise SystemExit(
                f"--grads directory contains files with mismatched dimensionality, "
                f"can't be combined into one dataset: {dims}. Combine only works "
                f"across files that share the same D (e.g. all from the same "
                f"--cond_input_dim run)."
            )
        combined = np.concatenate(list(all_grads.values()), axis=0)
        print(f"\nCombining {len(all_grads)} files into ONE dataset for Parallel "
              f"Analysis: total shape {combined.shape} "
              f"({sum(g.shape[0] for g in all_grads.values())} pairs pooled from "
              f"{len(all_grads)} files, {combined.shape[1]} dims)")
        all_grads = {"combined": combined}

    # ------------------------------------------------------------------
    # Run proper (column-permutation) Parallel Analysis per entry in
    # all_grads. Each entry gets its OWN independent permutation
    # simulation -- unlike the old scaled-Gaussian-noise approach, a
    # permutation simulation is built from that entry's OWN actual data
    # values, so there's no meaningful way to "share" one entry's
    # simulation with another's (even if they happen to share N and D).
    # When --grads was a directory, all_grads has exactly ONE entry here
    # ("combined") -- the concatenated data from every file in it,
    # already treated as a single unified dataset by this point, not
    # analysed as separate files.
    # ------------------------------------------------------------------
    all_results = {}

    for label, grads in all_grads.items():
        N, D = grads.shape
        print(f"\n{'='*60}")
        print(f"Parallel Analysis: {label}  |  shape: ({N}, {D})")
        print(f"{'='*60}")

        result = parallel_analysis(
            grads, n_simulations=args.n_simulations,
            percentile=args.percentile, seed=args.seed,
        )
        all_results[label] = result

        n_sig = result["n_significant"]
        print(f"\n  → {n_sig} significant directions (above {args.percentile:.0f}th pct noise threshold)")
        print(f"\n  Component  Real var%   Threshold%   Significant")
        print(f"  ---------  ---------   ----------   -----------")
        for k in range(min(20, D)):
            real_pct = result["real_var_ratios"][k] * 100
            thr_pct  = result["threshold_var_ratios"][k] * 100
            sig      = "YES" if k < n_sig else "no"
            marker   = " ←" if k == n_sig else ""
            print(f"  {k+1:>5}      {real_pct:>7.3f}%    {thr_pct:>7.3f}%    {sig}{marker}")

        txt  = f"Parallel Analysis: {label}\n"
        txt += f"N={N}, D={D}, n_simulations={args.n_simulations}, percentile={args.percentile}"
        txt += f", method=column_permutation\n"
        txt += f"Significant directions: {n_sig}\n\n"
        txt += f"{'Component':>10}  {'Real var%':>10}  {'Threshold%':>10}  {'Significant':>12}\n"
        for k in range(D):
            real_pct = result["real_var_ratios"][k] * 100
            thr_pct  = result["threshold_var_ratios"][k] * 100
            sig      = "YES" if k < n_sig else "no"
            txt += f"{k+1:>10}  {real_pct:>10.4f}  {thr_pct:>10.4f}  {sig:>12}\n"
        (out_dir / f"parallel_analysis_{label}.txt").write_text(txt)
        print(f"  Saved summary → {out_dir / f'parallel_analysis_{label}.txt'}")

        plot_parallel_analysis(
            result, label,
            out_dir / f"parallel_analysis_{label}.png",
            args.percentile,
        )

    # if multiple files: print comparison table of n_significant
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Summary: significant directions per timestep")
        print(f"{'='*60}")
        for label, result in all_results.items():
            print(f"  {label:<25}  k = {result['n_significant']}")

    print(f"\nDone. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()