"""
subspace_comparison.py — raw-gradient PCA vs VAEase subspaces
=============================================================
Standalone entry point. Loads a trained VAEase checkpoint + the gradients
it was trained on, optionally the diffusion checkpoint (for the
ground-truth conditioning subspace), and writes every subspace statistic
into the same --output_dir as the VAEase run.

Moved out of vaease.py: the `orthogonal_directions_pca` block that used
to live in train_vaease is superseded by this file (same effective-rank
numbers, reported per arm). Delete that block from vaease.py -- see the
note at the bottom of this docstring. Nothing here is imported by
vaease.py, so the dependency runs one way only (this -> vaease) and
there is no import cycle.

WHAT IT COMPUTES
----------------
1. PCA directly on the gradient matrix, cut at the 95% cumulative-variance
   point of its own spectrum -- the same criterion `eff95` applies to the
   VAEase arms. Both arms are then cut by what they actually found,
   rather than one of them being handed the ground-truth dimension, which
   would make the GT containment partly circular. --n_top caps the count
   when you want every arm matched at a fixed size instead.
   The VAEase arms are still not truncated: their active-set selection
   followed by PCA-of-columns produces its own count, which is the
   property under test.

2. Containment in both directions against the ground-truth subspace:
       recall    = ||A_gt B^T||_F^2 / k_gt     how much of GT is covered
       precision = ||A_gt B^T||_F^2 / k_disc   how much of B lies in GT
   One overlap, two normalisers -- equal iff k_disc == k_gt, and only
   informative as a pair when the discovered count differs from the true
   one. Plus principal angles, which say *how* two subspaces differ
   rather than only how much.

3. The same pair between each VAEase arm and the PCA arm.

PREPROCESSING (--pca_norm / VAEase side)
----------------------------------------
Default is deliberately asymmetric: **PCA on L2-normalized gradients,
VAEase on raw**. The reasoning is that PCA on unnormalized gradients is
dominated by whichever samples happen to have the largest loss scale --
magnitude variance masquerading as direction variance -- whereas VAEase's
single learned gamma already absorbs a global scale, and its per-sample
gate is what you want to see respond to raw magnitude.

The cost is that the two arms no longer see identical inputs, so a
difference between them is a difference in method *and* preprocessing.
--pca_norm match_vae removes that confound when you want the strict
comparison; run both if the arms disagree.

ORTHONORMALITY IS REQUIRED, NOT ASSUMED
---------------------------------------
Frobenius overlap against a non-orthogonal basis can exceed
min(k_a, k_b), so the "fraction" stops being one -- measured: three
near-duplicate unit rows against a 6-dim GT gave a naive recall of 1.47.
PCA components are orthonormal by construction; VAEase decoder columns
are unit-norm but NOT orthogonal, and vaease.py's own
subspace_basis_normalized is not orthogonal either (see its docstring).
So every VAEase selector -- group_averaged, frequency, importance, union
-- goes through orthonormal_span() first, and assert_orthonormal() raises
rather than warns.

USAGE
-----
  # full run: PCA (l2) vs VAEase (raw), against GT subspace
  python subspace_comparison.py \\
      --load_checkpoint runs/vaease_run1/vaease_checkpoint.pt \\
      --grads /path/to/grads_dir \\
      --checkpoint /path/to/diffusion_ckpt \\
      --output_dir runs/vaease_run1

  # no diffusion checkpoint: skips GT, still does PCA vs VAEase pairwise
  python subspace_comparison.py \\
      --load_checkpoint runs/vaease_run1/vaease_checkpoint.pt \\
      --grads /path/to/grads_dir --output_dir runs/vaease_run1

  # strict comparison, both arms on identical inputs
  python subspace_comparison.py ... --pca_norm match_vae

  # cap every arm at a fixed dimension instead of the 95% cut
  python subspace_comparison.py ... --n_top 64

vaease.py EDIT
--------------
In train_vaease, delete the block that starts

    # --- NEW: estimate how many effectively ORTHOGONAL directions the

and ends

    elif verbose:
        print("\\n(Skipping orthogonal-direction PCA estimate: requires linear_decoder=True)")

That is the only removal. pca_effective_rank / plot_pca_scree become
unused there; leave them or delete them. Everything else in vaease.py
(per-sample AD, group-averaged AD, neuron activation counts/plots,
alignment tables, interventions) stays as is -- this script reads the
checkpoint, not those outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Preprocessing (local copies so this module has no import-time dependency
# on vaease.py -- kept behaviourally identical to l2_normalize_rows there)
# ---------------------------------------------------------------------------

def l2_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def apply_norm(data: np.ndarray, mode: str,
               mean: Optional[np.ndarray] = None,
               std: Optional[np.ndarray] = None) -> np.ndarray:
    """mode in {none, l2, standardize, l2_then_standardize}. If mean/std
    are given they are used (the checkpoint's saved normalizer); if not,
    standardize fits fresh on this data."""
    out = np.asarray(data, dtype=np.float32)
    if mode in ("l2", "l2_then_standardize"):
        out = l2_rows(out)
    if mode in ("standardize", "l2_then_standardize"):
        if mean is None or std is None:
            mean = out.mean(axis=0)
            std = out.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
        out = (out - mean) / std
    return out


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_on_gradients(data: np.ndarray, center: bool = True, rtol: float = 1e-8):
    """PCA of the gradient matrix. Returns orthonormal components as ROWS,
    matching the (k, d) "rows = directions" layout used throughout
    vaease.py -- the opposite of the eigenvector-as-column convention, and
    an easy source of silently-transposed products if mixed up.

    center=True is standard PCA (variance about the mean). center=False
    decomposes the second moment about the origin, keeping whatever mean
    direction the gradients share inside the subspace. Neither is
    obviously right: the mean gradient may be a real semantic direction
    rather than a nuisance offset, which is why vaease.py's
    pca_effective_rank deliberately does not centre. Hence a flag.

    SVD rather than eigh of the covariance: better conditioned, and one
    code path for N < d and N >= d.
    """
    X = torch.as_tensor(np.asarray(data, dtype=np.float32))
    n, d = X.shape
    if n < 2:
        raise ValueError(f"need >= 2 gradients for PCA, got {n}")
    if center:
        X = X - X.mean(dim=0, keepdim=True)

    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    eigvals = (S ** 2) / max(n - 1, 1)
    evr = eigvals / eigvals.sum().clamp_min(1e-30)
    keep = S > rtol * S[0].clamp_min(1e-30)
    return Vh[keep], evr[keep], eigvals[keep]


def variance_cut(evr: torch.Tensor, frac: float = 0.95) -> int:
    """How many leading components reach `frac` of the total variance.

    The PCA arm's counterpart of effective_rank_stats' n_components_95pct,
    so the two arms are cut by one criterion instead of the PCA arm being
    given the ground-truth dimension.
    """
    if evr.numel() == 0:
        return 0
    cum = np.cumsum(evr.detach().cpu().numpy())
    return int(np.searchsorted(cum, frac) + 1)


def orthonormal_span(directions: torch.Tensor, rtol: float = 1e-6):
    """Orthonormal basis for the span of a set of (generally
    non-orthogonal) unit-norm directions, via SVD -- the "+ PCA" step in
    "group average + PCA", keeping every non-zero singular direction.
    Applied to EVERY selector, not just group_averaged.

    Deliberately UNCENTERED, same reasoning as vaease.py's
    pca_effective_rank: these rows are directions from the origin, not
    samples around a mean. Centring would subtract the average direction
    before decomposing, which is the wrong operation for a span.

    Returns (Q, S, rank) with Q of shape (rank, d), orthonormal rows.
    """
    A = torch.as_tensor(directions, dtype=torch.float32)
    if A.shape[0] == 0:
        return A.new_zeros((0, A.shape[1])), A.new_zeros(0), 0
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = S > rtol * S[0].clamp_min(1e-30)
    return Vh[keep], S[keep], int(keep.sum().item())


def effective_rank_stats(S: torch.Tensor,
                         thresholds: Sequence[float] = (0.90, 0.95, 0.99)) -> dict:
    """How many of a direction set's singular directions are needed to
    reach each cumulative-variance threshold. Replaces vaease.py's
    pca_effective_rank block: distinguishes "k_disc > k_gt because the
    selector kept near-duplicate columns" (effective rank << k_disc)
    from "k_disc > k_gt because it found real structure outside GT"
    (effective rank ~= k_disc). That distinction is what makes a low
    precision number interpretable."""
    if S.numel() == 0:
        return {}
    v = (S ** 2).cpu().numpy()
    evr = v / v.sum()
    cum = np.cumsum(evr)
    out = {"explained_variance_ratio": evr.tolist(),
           "cumulative_variance_ratio": cum.tolist()}
    for t in thresholds:
        out[f"n_components_{int(round(t * 100))}pct"] = int(np.searchsorted(cum, t) + 1)
    return out


def assert_orthonormal(V: torch.Tensor, name: str, atol: float = 1e-3):
    """Fail loudly rather than silently reporting a 'fraction' above 1."""
    k = V.shape[0]
    if k == 0:
        return
    G = V @ V.T
    err = (G - torch.eye(k, dtype=G.dtype, device=G.device)).abs().max().item()
    if err > atol:
        raise ValueError(
            f"{name} is not orthonormal (max |V V^T - I| = {err:.2e}). "
            "Containment ratios are only fractions for orthonormal bases -- "
            "pass this through orthonormal_span() first."
        )


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

def overlap_sq(A: torch.Tensor, B: torch.Tensor) -> float:
    """||A B^T||_F^2 for orthonormal row bases -- the sum of squared
    cosines of the principal angles between span(A) and span(B), in
    [0, min(k_a, k_b)]. Symmetric: recall and precision below differ only
    in which dimension divides it."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return 0.0
    return float((A @ B.T).pow(2).sum())


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """Principal angles in degrees, ascending. A leading 0 means one
    exactly shared direction; angles near 90 mean directions in one
    subspace with no counterpart in the other. Distinguishes 'mostly
    aligned but rotated' from 'a few matched, rest unrelated', which a
    single containment scalar cannot."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([])
    s = torch.linalg.svdvals(A @ B.T).clamp(-1.0, 1.0)
    return np.degrees(np.arccos(s.cpu().numpy()))


def containment(A_ref: torch.Tensor, B_disc: torch.Tensor) -> dict:
    k_ref, k_disc = A_ref.shape[0], B_disc.shape[0]
    ov = overlap_sq(A_ref, B_disc)
    ang = principal_angles(A_ref, B_disc)
    return {
        "k_ref": int(k_ref),
        "k_disc": int(k_disc),
        "overlap_sq": ov,
        "recall": ov / k_ref if k_ref else float("nan"),
        "precision": ov / k_disc if k_disc else float("nan"),
        "principal_angles_deg": ang.tolist(),
        "mean_angle_deg": float(ang.mean()) if len(ang) else float("nan"),
        "max_angle_deg": float(ang.max()) if len(ang) else float("nan"),
    }


def containment_curve(A_ref: torch.Tensor, B_ordered: torch.Tensor,
                      k_max: Optional[int] = None) -> dict:
    """recall(k) and precision(k) as the first k rows of B_ordered are
    admitted. B_ordered must be meaningfully ordered (PCA: descending
    eigenvalue; VAEase: descending importance) and orthonormal, so that
    truncating rows leaves an orthonormal basis.

    This is what makes any single truncation choice inspectable rather
    than assumed: recall rises and saturates, precision falls, and where
    recall flattens is the honest answer to how many directions are
    actually needed to cover GT."""
    k_ref = A_ref.shape[0]
    k_max = min(k_max or B_ordered.shape[0], B_ordered.shape[0])
    ks, rec, prec = [], [], []
    for k in range(1, k_max + 1):
        ov = overlap_sq(A_ref, B_ordered[:k])
        ks.append(k)
        rec.append(ov / k_ref if k_ref else float("nan"))
        prec.append(ov / k)
    return {"k": ks, "recall": rec, "precision": prec}


# ---------------------------------------------------------------------------
# Neuron selectors
# ---------------------------------------------------------------------------

def select_neurons(sigma_z: np.ndarray, mode: str,
                   group_active_indices: Sequence[int],
                   activation_counts: np.ndarray,
                   importance_all: np.ndarray,
                   n_top: int = 16,
                   freq_frac: float = 0.05) -> np.ndarray:
    """Which latent dims form "the discovered set".

    group_averaged -- the paper's Table 2/3 protocol: average sigma^2
        across samples per latent dim, then threshold that averaged
        profile once. Because it averages BEFORE thresholding, a dim that
        fires strongly on a small subset of samples averages out to
        ~inactive and is dropped -- so the selection is biased toward
        broadly-firing directions and against narrow, concept-specific
        ones. Indices come from vaease.py's group_averaged_active_dims so
        that cutoff has a single source of truth.
    frequency_topn -- the n dims active on the most samples. A ranking,
        not a threshold, so "top n" needs no cutoff; and it does not
        penalise a dim for being inactive on most samples, so narrow
        features survive.
    frequency_thresh -- dims active on more than freq_frac of samples
        (the active_over_5pct_samples selection vaease.py used).
    importance_topn -- top n by Var(z_tilde_j) * ||W_dec[:,j]||^2. NOT a
        sum of weights: it is the gated code's variance times the SQUARED
        column norm, i.e. how much that dim moves the reconstruction.
        Closest analogue to ranking PCA components by eigenvalue, though
        not an exact variance decomposition (ignores cross-covariance
        between dims).
    union -- every dim active for at least one sample. Opposite bias to
        group_averaged: catches narrow features, admits noise.
    """
    kappa = sigma_z.shape[1]
    if mode == "group_averaged":
        return np.asarray(sorted(group_active_indices), dtype=int)
    if mode == "frequency_topn":
        n = min(n_top, kappa)
        return np.sort(np.argsort(activation_counts)[::-1][:n]).astype(int)
    if mode == "frequency_thresh":
        return np.flatnonzero(activation_counts > freq_frac * sigma_z.shape[0]).astype(int)
    if mode == "importance_topn":
        n = min(n_top, kappa)
        return np.sort(np.argsort(importance_all)[::-1][:n]).astype(int)
    if mode == "union":
        return np.flatnonzero(activation_counts > 0).astype(int)
    raise ValueError(f"unknown selector: {mode}")


def vaease_subspace(W_dec: np.ndarray, indices: np.ndarray,
                    importance_all: np.ndarray, rtol: float = 1e-6) -> dict:
    """Active decoder columns -> ordered unit directions -> orthonormal
    span. W_dec is (d, kappa); column j is the fixed vector latent dim j
    writes when its gate opens (linear decoder only -- with an MLP there
    is no single global direction per dim, only an input-dependent
    Jacobian).

    Ordered by importance before orthonormalising, so truncating the
    resulting basis stays meaningful for the containment curve.
    """
    d = W_dec.shape[0]
    if len(indices) == 0:
        return {"k_selected": 0, "rank": 0, "indices": [],
                "raw": torch.zeros((0, d)), "ortho": torch.zeros((0, d)),
                "singular_values": [], "effective_rank": {}}

    order = np.argsort(importance_all[indices])[::-1]
    idx_sorted = np.asarray(indices)[order]

    cols = W_dec[:, idx_sorted].T                       # (k, d)
    norms = np.linalg.norm(cols, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    raw = torch.from_numpy((cols / norms).astype(np.float32))

    ortho, S, rank = orthonormal_span(raw, rtol=rtol)
    return {
        "k_selected": int(len(idx_sorted)),
        "rank": int(rank),
        "indices": idx_sorted.tolist(),
        "raw": raw,
        "ortho": ortho,
        "singular_values": S.cpu().numpy().tolist(),
        "effective_rank": effective_rank_stats(S),
    }


# ---------------------------------------------------------------------------
# Plots / tables
# ---------------------------------------------------------------------------

def plot_containment_curves(curves: dict, k_gt: int, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, c in curves.items():
        axes[0].plot(c["k"], c["recall"], marker="o", ms=3, label=label)
        axes[1].plot(c["k"], c["precision"], marker="s", ms=3, label=label)
    for ax, name, ylab in ((axes[0], "Recall", "fraction of GT subspace covered"),
                           (axes[1], "Precision", "fraction of directions inside GT")):
        ax.axvline(k_gt, ls="--", c="gray", lw=1, label=f"$k_{{gt}}$={k_gt}")
        ax.set_xlabel("Number of directions admitted ($k$)")
        ax.set_ylabel(ylab)
        ax.set_title(name)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_spectrum(evr: torch.Tensor, k_cut: Optional[int], out_path: Path,
                      title: str, frac: float = 0.95):
    v = (evr.cpu().numpy() * 100)[:min(len(evr), 64)]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(v)), v)
    if k_cut:
        plt.axvline(k_cut - 0.5, ls="--", c="red", lw=1,
                    label=f"cut at {frac:.0%} cum. var: k={k_cut}")
        plt.legend()
    plt.xlabel("Component")
    plt.ylabel("Explained variance (%)")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary_table(payload: dict, out_path: Path):
    L = []
    L.append(f"gradients: N={payload['n_gradients']}, d={payload['d']}")
    L.append(f"PCA arm    : norm={payload['pca_norm']}, centered={payload['pca_centered']}"
             f", cut at {payload.get('pca_variance_frac', 0.95):.0%} cumulative "
             f"variance -> k={payload.get('k_pca', '?')}"
             + (f" (capped by n_top={payload['n_top']})"
                if payload.get("pca_capped_by_n_top") else ""))
    L.append(f"VAEase arms: norm={payload['vae_norm']} (checkpoint's own), not truncated")
    if payload["k_gt"]:
        L.append(f"GT subspace: k_gt={payload['k_gt']}  ({', '.join(payload['gt_names'])})")
    L.append("")

    hdr = (f"{'arm':<26}{'k_sel':>6}{'rank':>6}{'eff95':>7}"
           f"{'recall':>9}{'precis':>9}{'ang_mu':>8}{'ang_max':>9}")

    if payload["vs_gt"]:
        L.append("=== Containment in ground-truth subspace ===")
        L.append(hdr)
        L.append("-" * len(hdr))
        for name, c in payload["vs_gt"].items():
            L.append(f"{name:<26}{c.get('k_selected', c['k_disc']):>6}{c.get('rank', c['k_disc']):>6}"
                     f"{c.get('eff95', 0) or 0:>7}"
                     f"{c['recall']:>9.4f}{c['precision']:>9.4f}"
                     f"{c['mean_angle_deg']:>8.2f}{c['max_angle_deg']:>9.2f}")
        L.append("")

    L.append("=== VAEase arms vs the PCA arm (PCA is the reference) ===")
    L.append(hdr.replace("recall", "pca_cov").replace("precis", "in_pca"))
    L.append("-" * len(hdr))
    for name, c in payload["vs_pca"].items():
        L.append(f"{name:<26}{c.get('k_selected', c['k_disc']):>6}{c.get('rank', c['k_disc']):>6}"
                 f"{c.get('eff95', 0) or 0:>7}"
                 f"{c['recall']:>9.4f}{c['precision']:>9.4f}"
                 f"{c['mean_angle_deg']:>8.2f}{c['max_angle_deg']:>9.2f}")
    L.append("")
    L.append("recall  = ||A_ref B^T||_F^2 / k_ref    how much of the reference is covered")
    L.append("precis  = ||A_ref B^T||_F^2 / k_disc   how much of B lies in the reference")
    L.append("          equal iff k_disc == k_ref")
    L.append("eff95   = singular directions needed for 95% of the set's variance;")
    L.append("          eff95 << rank means the extra directions are near-duplicates,")
    L.append("          not new structure -- which is what makes a low precision readable.")
    L.append("          the PCA arm is cut by this same criterion, so k_sel == eff95 there")
    L.append("ang_*   = principal angles in degrees; 0 = an exactly shared direction")
    txt = "\n".join(L)
    Path(out_path).write_text(txt)
    print("\n" + txt)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_subspace_comparison(
    pca_data: np.ndarray,
    sigma_z: np.ndarray,
    mu_z: np.ndarray,
    W_dec: np.ndarray,
    out_dir: Path,
    gt_basis_orthonormal: Optional[torch.Tensor] = None,
    gt_names: Sequence[str] = (),
    group_active_indices: Sequence[int] = (),
    activation_counts: Optional[np.ndarray] = None,
    selectors: Sequence[str] = ("group_averaged", "frequency_topn", "importance_topn"),
    n_top: int = 16,
    freq_frac: float = 0.05,
    pca_centered: bool = True,
    pca_norm: str = "l2",
    vae_norm: str = "none",
    pca_variance_frac: float = 0.95,
) -> dict:
    """pca_data is already preprocessed per --pca_norm; sigma_z/mu_z were
    encoded from the VAEase-preprocessed data. gt_basis_orthonormal must
    be (k_gt, d) with orthonormal ROWS -- vaease.py builds
    subspace_basis_orthonormal as (d, k_gt) columns via torch.linalg.qr,
    so transpose at the call site. Pass None to skip all GT sections."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A_gt, k_gt = None, 0
    if gt_basis_orthonormal is not None:
        A_gt = torch.as_tensor(gt_basis_orthonormal, dtype=torch.float32)
        assert_orthonormal(A_gt, "gt_basis_orthonormal")
        k_gt = A_gt.shape[0]

    # ---- arm 1: PCA on the gradient matrix ----------------------------
    # Cut at the spectrum's own 95% cumulative-variance point, matching the
    # eff95 criterion reported for the VAEase arms. Cutting at k_gt instead
    # would hand this arm the dimension it is being scored against. n_top,
    # when set, caps it so every arm can be matched at a fixed size.
    V_pca_full, evr, eigvals = pca_on_gradients(pca_data, center=pca_centered)
    k_pca = variance_cut(evr, pca_variance_frac)
    capped = bool(n_top) and k_pca > n_top
    if capped:
        k_pca = n_top
    k_pca = max(1, min(k_pca, V_pca_full.shape[0]))
    V_pca = V_pca_full[:k_pca]
    assert_orthonormal(V_pca, "pca_raw")
    print(f"  [pca_raw] {k_pca} of {V_pca_full.shape[0]} components at "
          f"{pca_variance_frac:.0%} cumulative variance"
          + (f" (capped at n_top={n_top})" if capped else ""))

    plot_pca_spectrum(evr, k_pca, out_dir / "pca_raw_scree.png",
                      f"PCA of gradients (norm={pca_norm}, centered={pca_centered}, "
                      f"N={len(pca_data)})", frac=pca_variance_frac)
    (out_dir / "pca_raw_spectrum.json").write_text(json.dumps({
        "explained_variance_ratio": evr.cpu().numpy().tolist(),
        "eigenvalues": eigvals.cpu().numpy().tolist(),
        "cumulative": np.cumsum(evr.cpu().numpy()).tolist(),
        "norm": pca_norm, "centered": pca_centered,
        "variance_frac": float(pca_variance_frac),
        "k_pca_truncation": int(k_pca),
        "capped_by_n_top": bool(capped),
    }, indent=2))

    # ---- arms 2..n: VAEase, one per selector, NOT truncated ------------
    if activation_counts is None:
        raise ValueError("activation_counts is required (per-neuron firing counts)")
    z_det = (1.0 - sigma_z) * mu_z
    importance_all = z_det.var(axis=0) * (W_dec ** 2).sum(axis=0)

    arms = {"pca_raw": {"ortho": V_pca, "ordered": V_pca_full,
                        "rank": int(V_pca.shape[0]),
                        "k_selected": int(V_pca.shape[0]), "eff95": int(k_pca)}}

    for mode in selectors:
        idx = select_neurons(sigma_z, mode, group_active_indices,
                             activation_counts, importance_all,
                             n_top=n_top, freq_frac=freq_frac)
        sub = vaease_subspace(W_dec, idx, importance_all)
        if sub["rank"] == 0:
            print(f"  [{mode}] no active dims -- skipped")
            continue
        assert_orthonormal(sub["ortho"], f"vaease_{mode}")
        arms[f"vaease_{mode}"] = {
            "ortho": sub["ortho"], "ordered": sub["ortho"],
            "rank": sub["rank"], "k_selected": sub["k_selected"],
            "eff95": sub["effective_rank"].get("n_components_95pct"),
            "detail": sub,
        }
        print(f"  [{mode}] {sub['k_selected']} dims selected -> span rank {sub['rank']}"
              f" (eff95={sub['effective_rank'].get('n_components_95pct')})"
              f"  indices {sub['indices'][:12]}"
              f"{'...' if len(sub['indices']) > 12 else ''}")

    # ---- containment ---------------------------------------------------
    vs_gt, curves = {}, {}
    if A_gt is not None:
        for name, arm in arms.items():
            c = containment(A_gt, arm["ortho"])
            c.update(rank=arm["rank"], k_selected=arm["k_selected"], eff95=arm["eff95"])
            vs_gt[name] = c
            curves[name] = containment_curve(A_gt, arm["ordered"],
                                             k_max=min(32, arm["ordered"].shape[0]))
        plot_containment_curves(
            curves, k_gt, out_dir / "subspace_containment_curves.png",
            "Containment in ground-truth conditioning subspace vs number of directions",
        )

    vs_pca = {}
    for name, arm in arms.items():
        if name == "pca_raw":
            continue
        c = containment(V_pca, arm["ortho"])
        c.update(rank=arm["rank"], k_selected=arm["k_selected"], eff95=arm["eff95"])
        vs_pca[name] = c

    payload = {
        "n_gradients": int(len(pca_data)),
        "d": int(pca_data.shape[1]),
        "k_gt": int(k_gt),
        "k_pca": int(k_pca),
        "pca_variance_frac": float(pca_variance_frac),
        "pca_capped_by_n_top": bool(capped),
        "gt_names": list(gt_names),
        "pca_norm": pca_norm,
        "vae_norm": vae_norm,
        "pca_centered": bool(pca_centered),
        "selectors": list(selectors),
        "n_top": int(n_top),
        "vs_gt": vs_gt,
        "vs_pca": vs_pca,
        "curves": curves,
        "vaease_detail": {k: {kk: vv for kk, vv in v["detail"].items()
                              if kk not in ("raw", "ortho")}
                          for k, v in arms.items() if "detail" in v},
    }
    (out_dir / "subspace_comparison.json").write_text(json.dumps(payload, indent=2))
    write_summary_table(payload, out_dir / "subspace_comparison.txt")
    torch.save({name: arm["ortho"] for name, arm in arms.items()},
               out_dir / "subspace_directions.pt")

    print(f"\n  Saved -> {out_dir / 'subspace_comparison.txt'} (+ .json)")
    print(f"  Saved -> {out_dir / 'pca_raw_scree.png'}")
    if curves:
        print(f"  Saved -> {out_dir / 'subspace_containment_curves.png'}")
    print(f"  Saved -> {out_dir / 'subspace_directions.pt'}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load_checkpoint", required=True,
                   help="trained vaease_checkpoint.pt")
    p.add_argument("--grads", nargs="+", required=True,
                   help="gradient file(s) or a directory (same as vaease.py --grads)")
    p.add_argument("--glob_pattern", default="grads_t*.npy")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="diffusion checkpoint dir (unet_ema/, conditioner.pt). "
                        "Optional: without it the GT subspace is skipped and only "
                        "the PCA-vs-VAEase pairwise comparison is written.")
    p.add_argument("--cond_input_dim", type=int, default=9)
    p.add_argument("--base_prompt", type=float, nargs="+",
                   default=[0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    p.add_argument("--pca_norm", default="l2",
                   choices=["none", "l2", "standardize", "l2_then_standardize", "match_vae"],
                   help="preprocessing for the PCA arm only. Default l2: PCA on "
                        "unnormalized gradients is dominated by per-sample loss "
                        "magnitude rather than direction. match_vae uses the "
                        "checkpoint's own norm mode instead, removing the "
                        "preprocessing confound between the two arms.")
    p.add_argument("--pca_variance_frac", type=float, default=0.9,
                   help="cumulative-variance point at which the PCA arm is cut. "
                        "0.95 matches the eff95 criterion reported for the VAEase "
                        "arms, so both arms are cut the same way.")
    p.add_argument("--no_center", action="store_true",
                   help="uncentered PCA (second moment about the origin), keeping any "
                        "shared mean gradient direction inside the subspace")
    p.add_argument("--selectors", nargs="+",
                   default=["group_averaged", "frequency_topn", "importance_topn"],
                   choices=["group_averaged", "frequency_topn", "frequency_thresh",
                            "importance_topn", "union"])
    p.add_argument("--n_top", type=int, default=0,
                   help="cap for the *_topn selectors AND for the PCA arm. 0 (the "
                        "default) leaves the PCA arm at its variance-based count and "
                        "the ranked selectors at the group_averaged size. Set it to "
                        "match every arm at one fixed dimension.")
    p.add_argument("--freq_frac", type=float, default=0.05)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # vaease imports are local so this module stays importable on its own
    from vaease import (load_gradients, load_vaease_checkpoint, Normalizer,
                        encode_dataset_full, group_averaged_active_dims,
                        compute_neuron_activation_counts, build_alignment_tables)
    import torch.nn.functional as F

    data = load_gradients(args.grads, glob_pattern=args.glob_pattern)
    model = load_vaease_checkpoint(args.load_checkpoint, device=device)

    ckpt_meta = torch.load(args.load_checkpoint, map_location="cpu")
    vae_norm = ckpt_meta.get("norm", "none")
    mean = std = None
    npz = Path(args.load_checkpoint).parent / "normalizer.npz"
    if npz.exists():
        nz = Normalizer.load(npz)
        mean, std = nz.mean, nz.std
        print(f"Loaded checkpoint normalizer from {npz}")

    if not model.linear_decoder:
        raise SystemExit(
            "this analysis needs linear_decoder=True: with an MLP decoder there is "
            "no single global direction per latent dim, only an input-dependent "
            "Jacobian. Retrain without --mlp_decoder.")

    # VAEase side: the checkpoint's own preprocessing (default 'none' = raw)
    vae_data = apply_norm(data, vae_norm, mean, std)
    # PCA side: L2 by default, or matched
    pca_mode = vae_norm if args.pca_norm == "match_vae" else args.pca_norm
    pca_data = apply_norm(data, pca_mode, mean, std)
    print(f"PCA arm on norm={pca_mode!r}; VAEase arm on norm={vae_norm!r} "
          f"(N={len(data)}, d={data.shape[1]})")

    mu_z, sigma_z = encode_dataset_full(
        model, torch.from_numpy(vae_data.astype(np.float32)), device=device)
    group_all = group_averaged_active_dims(sigma_z, labels=None)["all"]
    neuron_counts = compute_neuron_activation_counts(sigma_z)
    W_dec = model.decoder.weight.detach().cpu().numpy()
    print(f"group_averaged active dims: {group_all['active_dims']} "
          f"(sigma^2 threshold {group_all['threshold']:.4f})")

    # ---- optional GT subspace ----
    A_gt_rows, subspace_names = None, ()
    basis_norm = basis_ortho = None
    if args.checkpoint:
        if len(args.base_prompt) != args.cond_input_dim:
            raise SystemExit(f"--base_prompt has {len(args.base_prompt)} values but "
                             f"--cond_input_dim={args.cond_input_dim}")
        from configs.config_64 import Config64 as Config
        from models.conditioner import ShapeConditioningEncoder
        from directions import build_valid_subspace_basis, resolve_feature_names

        cfg = Config(cond_input_dim=args.cond_input_dim)
        ck = Path(args.checkpoint)
        conditioner = ShapeConditioningEncoder(
            cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim).to(device)
        conditioner.load_state_dict(
            torch.load(ck / "conditioner.pt", map_location=device, weights_only=True))
        conditioner.eval()

        raw_basis, subspace_names = build_valid_subspace_basis(
            conditioner, args.base_prompt, resolve_feature_names(args.cond_input_dim),
            args.cond_input_dim, device, subtract_baseline=True)
        basis = torch.from_numpy(raw_basis.astype(np.float32))
        basis_norm = F.normalize(basis, dim=0)
        basis_ortho, _ = torch.linalg.qr(basis_norm)     # (d, k_gt) columns
        A_gt_rows = basis_ortho.T                        # -> (k_gt, d) rows
        print(f"GT subspace: {len(subspace_names)} dims {list(subspace_names)}")

    # Deliberately NOT defaulting to k_gt: routing the ground-truth dimension
    # in through the cap would put the PCA arm straight back where the 95% cut
    # was meant to move it away from. 0 means "do not cap"; the ranked
    # selectors then fall back to the group_averaged size.
    selector_n_top = args.n_top or max(group_all["active_dims"], 1)

    print("\n=== Raw-gradient PCA + subspace containment ===")
    run_subspace_comparison(
        pca_data=pca_data, sigma_z=sigma_z, mu_z=mu_z, W_dec=W_dec,
        out_dir=out_dir,
        gt_basis_orthonormal=A_gt_rows, gt_names=subspace_names,
        group_active_indices=group_all["active_dim_indices"],
        activation_counts=neuron_counts,
        selectors=tuple(args.selectors),
        n_top=selector_n_top if args.n_top else 0,
        freq_frac=args.freq_frac,
        pca_centered=not args.no_center, pca_norm=pca_mode, vae_norm=vae_norm,
        pca_variance_frac=args.pca_variance_frac,
    )

    # same two alignment tables vaease.py builds for its VAEase arms,
    # for the PCA arm, so all of them land side by side in out_dir
    if basis_norm is not None:
        V_pca_full, evr_full, _ = pca_on_gradients(pca_data, center=not args.no_center)
        k_tab = variance_cut(evr_full, args.pca_variance_frac)
        if args.n_top:
            k_tab = min(k_tab, args.n_top)
        build_alignment_tables(V_pca_full[:max(1, k_tab)],
                               basis_norm, basis_ortho, list(subspace_names),
                               "pca_raw", out_dir)

    print(f"\nDone. Outputs under {out_dir}/")


if __name__ == "__main__":
    main()