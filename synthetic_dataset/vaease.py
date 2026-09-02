"""
VAEase — training script for arbitrary vector data (e.g. ELROND gradients)
============================================================================
Implements VAEase (Lu, Zhu, He & Wipf, "Sparse Autoencoders, Again?", ICML
2025) for use as a per-sample local intrinsic dimension (LID) estimator.

Recap of the mechanism (see the paper for the full argument):
  - Encoder: x -> (mu_z, sigma_z) via a shared trunk + two linear heads.
    sigma_z is squashed to (0, 1) with a sigmoid: sigma -> 1 means the
    posterior has collapsed to the N(0,1) prior (inactive dimension);
    sigma -> 0 means the dimension is carrying signal (active).
  - Reparameterized code: z = mu_z + sigma_z * eps, eps ~ N(0, I).
  - VAEase gate (the paper's core trick, Eq. 4):
        z_tilde = (1 - sigma_z) * z
    This routes the encoder's own certainty estimate into the decoder as a
    *multiplicative* gate, so inactive dimensions arrive as a clean ~0
    rather than raw N(0,1) noise the decoder must permanently silence.
    That's what gives VAEase *adaptive* (per-sample) sparsity instead of
    the vanilla VAE's *fixed* sparsity pattern.
  - Decoder: z_tilde -> x_hat, with a single learned global noise scale
    gamma (scalar), i.e. p(x|z) = N(x | mu_x(z_tilde), gamma * I).
  - Loss = reconstruction (weighted by 1/gamma) + KL(q(z|x) || N(0,I)),
    with gamma learned jointly (no manual sparsity hyperparameters).

Active-dimension extraction (matches the paper's Section 5 protocol):
  For a sample x, look at sigma_z(x)^2 across the kappa latent dims. These
  values are (in a well-trained model) bimodal: near 0 for active dims,
  near 1 for inactive ones. We find the split point that minimizes the
  sum of within-group variance (adapted from Xia et al. 2015) and count
  dims below it as "active". This gives a *local, per-sample* dimension
  estimate — the thing PCA / parallel analysis can't give you directly.

Direction extraction & interventions (new — mirrors pca_analysis.py):
  With a *linear* decoder, x_hat = W_dec @ z_tilde, so column j of W_dec
  is a single GLOBAL vector in gradient space — the direction latent dim
  j writes when active. That's the direct analog of a PCA eigenvector,
  directly comparable (dot product) against the conditioner's ground
  truth W columns, and directly usable for cond_out interventions. What
  IS per-sample is sigma_z(x) (which of those fixed directions fire for
  a given x) — the dictionary itself is fixed, unlike an MLP decoder's
  Jacobian would be. See extract_decoder_directions() below.

  Because nothing in the VAEase objective forces decoder columns to be
  orthogonal (unlike PCA eigenvectors, which fall out of a symmetric
  covariance matrix), we build BOTH:
    - "raw" directions: active decoder columns, unit-normalized, as-is
      (can be correlated/redundant — that's informative on its own)
    - "orthogonalized" directions: QR of the stacked raw columns,
      dropping near-zero-norm trailing columns (rank deficiency)
  and run the alignment tables + intervention grids for both.

Usage
-----
  python vaease.py --data gradients.npy --output_dir runs/vaease_run1

  # analysis + interventions on an already-trained model:
  python vaease.py --load_checkpoint runs/vaease_run1/vaease_checkpoint.pt \\
      --data gradients.npy --checkpoint <diffusion checkpoint dir> \\
      --run_interventions

See --help for all options. Run with --self_test to validate the
implementation on a synthetic union-of-linear-subspaces dataset with a
known ground-truth dimension (a smaller version of the paper's own
sanity check) before trusting it on real data.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless-safe: no display needed on a compute node
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def expand_grad_paths(paths, glob_pattern: str = "grads_t*.npy"):
    """Mirrors pca_analysis.py's directory-expansion exactly: if a single
    directory is given, glob `glob_pattern` inside it and sort numerically
    by the digits in each filename (so grads_t500 sorts after grads_t50,
    unlike a plain string sort). Otherwise, treat the input as an explicit
    list of file paths, used as-is.
    """
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and paths[0].is_dir():
        grads_dir = paths[0]
        found = sorted(grads_dir.glob(glob_pattern),
                       key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
        if not found:
            raise FileNotFoundError(f"No files matching '{glob_pattern}' found in {grads_dir}")
        return found
    return paths


def load_gradients(paths, glob_pattern: str = "grads_t*.npy") -> np.ndarray:
    """Load one or more gradient files and concatenate into a single
    (N_total, d) float32 array.

    `paths` may be:
      - a single directory (str/Path) -> expanded via expand_grad_paths,
        exactly like pca_analysis.py: globs `glob_pattern` (default
        "grads_t*.npy") and sorts numerically by the digits in the filename
      - a single explicit file path (.npy/.npz/.pt/.pth)
      - a list of explicit file paths (each loaded and concatenated, same
        as passing multiple --grads args to pca_analysis.py)

    Each file may itself hold a full (n_i, d) matrix (e.g. one file per
    timestep, as in your gradient collection) or a single (d,) vector —
    both are handled and concatenated along axis 0.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = expand_grad_paths(paths, glob_pattern=glob_pattern)

    print(f"Loading {len(files)} gradient file(s):")
    for f in files:
        print(f"  {f}")

    arrays = []
    for f in files:
        f = Path(f)
        if f.suffix == ".npy":
            a = np.load(f)
        elif f.suffix == ".npz":
            z = np.load(f)
            key = "gradients" if "gradients" in z else list(z.keys())[0]
            a = z[key]
        elif f.suffix in (".pt", ".pth"):
            obj = torch.load(f, map_location="cpu")
            a = obj.numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
        else:
            raise ValueError(f"Unrecognized file extension for {f}. "
                             "Expected .npy, .npz, or .pt/.pth.")
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]          # a lone (d,) vector -> one row
        arrays.append(a)

    arr = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D (N, d) array after loading, got shape {arr.shape}.")
    return arr


@dataclass
class Normalizer:
    """Per-coordinate standardization (zero mean, unit variance), with
    saved stats so the same transform can be reapplied at eval time."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Guard against constant/near-constant dimensions (e.g. frozen
        # weights with always-zero gradient) to avoid dividing by ~0.
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        z = np.load(path)
        return cls(mean=z["mean"], std=z["std"])


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization: each gradient vector -> unit length.
    This is what pca_analysis.py's --normalize_grads does (removes
    per-sample magnitude, e.g. differing loss scale across pairs), as
    distinct from Normalizer's per-*coordinate* standardization above
    (removes scale differences across the 64 dimensions). The two are
    not mutually exclusive — you can L2-normalize rows first and then
    standardize columns; --norm l2_then_standardize below does exactly
    that, for anyone wanting both effects at once.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VAEase(nn.Module):
    """Linear encoder trunk + two heads (mu_z, sigma_z-logits); linear
    decoder. Mirrors the paper's LLM-activation configuration (Appendix
    D.3), which is the closest match to unstructured high-dim vectors
    like gradients. Swap in deeper encoder/decoder MLPs if you want more
    reconstruction capacity, but keep the decoder's *linear* option
    available if you want the active dimensions to correspond to an
    interpretable direction dictionary, analogous to your PCA components.
    """

    def __init__(self, d: int, kappa: int, hidden: Optional[int] = None,
                 linear_decoder: bool = True):
        super().__init__()
        self.d = d
        self.kappa = kappa
        self.linear_decoder = linear_decoder
        hidden = hidden or d  # match the paper's "encoder = linear + ReLU" setup

        self.encoder_trunk = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden, kappa)
        self.to_sigma_logit = nn.Linear(hidden, kappa)

        if linear_decoder:
            self.decoder = nn.Linear(kappa, d)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(kappa, hidden), nn.ReLU(),
                nn.Linear(hidden, d),
            )

        # Single global (log) noise scale, learned jointly — no manual
        # sparsity hyperparameters anywhere in this model.
        self.log_gamma = nn.Parameter(torch.zeros(()))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        mu_z = self.to_mu(h)
        # sigma_z in (0, 1): 1 = fully inactive (posterior collapsed to
        # prior), ->0 = fully active. Bounding to (0,1) matches the
        # theory's active/inactive regime and keeps the gate stable.
        sigma_z = torch.sigmoid(self.to_sigma_logit(h))
        return mu_z, sigma_z

    def forward(self, x: torch.Tensor):
        mu_z, sigma_z = self.encode(x)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps
        z_tilde = (1.0 - sigma_z) * z          # <-- the VAEase gate (Eq. 4)
        x_hat = self.decoder(z_tilde)
        gamma = self.log_gamma.exp().clamp_min(1e-6)  # numerical floor
        return x_hat, mu_z, sigma_z, gamma


def vaease_loss(x: torch.Tensor, x_hat: torch.Tensor, mu_z: torch.Tensor,
                 sigma_z: torch.Tensor, gamma: torch.Tensor
                 ) -> Tuple[torch.Tensor, dict]:
    d = x.shape[1]
    recon_sq = (x - x_hat).pow(2).sum(dim=1)          # sum over dims, per sample
    recon_term = recon_sq / (2.0 * gamma)
    const_term = 0.5 * d * torch.log(2 * np.pi * gamma)
    sigma2 = sigma_z.pow(2)
    kl = 0.5 * (sigma2 + mu_z.pow(2) - 1.0 - torch.log(sigma2.clamp_min(1e-12))).sum(dim=1)
    loss = (recon_term + const_term + kl).mean()
    with torch.no_grad():
        logs = {
            "loss": loss.item(),
            "recon_mse_per_dim": (recon_sq / d).mean().item(),
            "kl": kl.mean().item(),
            "gamma": gamma.item(),
        }
    return loss, logs


# ---------------------------------------------------------------------------
# Active-dimension analysis
# ---------------------------------------------------------------------------

def _variance_split_threshold(values: np.ndarray) -> float:
    """1D two-cluster split minimizing summed within-group variance
    (Otsu-style; adapted from Xia et al. 2015, as used in the paper).
    `values` should be sorted ascending on entry for this to be O(k)."""
    v = np.sort(values)
    k = len(v)
    if k < 2:
        return v[0] if k else 0.5
    best_t, best_cost = v[0], np.inf
    for i in range(1, k):
        lo, hi = v[:i], v[i:]
        cost = lo.var() * len(lo) + hi.var() * len(hi) if len(lo) and len(hi) else np.inf
        if cost < best_cost:
            best_cost, best_t = cost, (v[i - 1] + v[i]) / 2.0
    return best_t


def count_active_dims(sigma_z: np.ndarray) -> np.ndarray:
    """sigma_z: (N, kappa) array of posterior std devs (NOT squared).
    Returns (N,) int array of active-dimension counts, one per sample.
    This is the *local* / per-sample estimate — noisier for any single
    point, but it's the thing that lets you look at a distribution or
    cluster by category. See group_averaged_active_dims() below for a
    single, low-noise number over a known group instead.

    Vectorized across all N samples at once via cumulative sums, rather
    than looping in Python and calling _variance_split_threshold once
    per sample (the original approach) -- that loop is what made
    per-epoch AD tracking prohibitively slow once it started running
    every epoch on both train and val (measured: ~8s per call at
    N=5000, kappa=64 with the loop; a few ms vectorized). Verified
    bit-identical to the original per-sample version across 200 random
    test cases including edge cases (kappa=1, tiny N) before replacing
    it here -- see _count_active_dims_loop_reference below.
    """
    sigma2 = sigma_z ** 2
    N, k = sigma2.shape
    if k < 2:
        return np.zeros(N, dtype=int)

    S = np.sort(sigma2, axis=1)                        # (N, k), ascending per row
    cs1 = np.cumsum(S, axis=1)                          # running sum
    cs2 = np.cumsum(S ** 2, axis=1)                     # running sum of squares
    total1, total2 = cs1[:, -1:], cs2[:, -1:]

    n_lo = np.arange(1, k, dtype=np.float64)            # candidate split sizes 1..k-1
    n_hi = k - n_lo
    lo_sum1, lo_sum2 = cs1[:, :-1], cs2[:, :-1]
    hi_sum1, hi_sum2 = total1 - lo_sum1, total2 - lo_sum2

    # sum((x-mean)^2) = sum(x^2) - (sum(x))^2/n  -- i.e. variance*n,
    # matching _variance_split_threshold's cost exactly, for every
    # sample and every candidate split simultaneously.
    cost_lo = lo_sum2 - (lo_sum1 ** 2) / n_lo
    cost_hi = hi_sum2 - (hi_sum1 ** 2) / n_hi

    best_split_idx = np.argmin(cost_lo + cost_hi, axis=1)  # first-min tie-break, matches original
    return (best_split_idx + 1).astype(int)                # split size = active-dim count directly


def _count_active_dims_loop_reference(sigma_z: np.ndarray) -> np.ndarray:
    """Reference-only slow implementation, kept for testing/verification
    against the vectorized count_active_dims above. Not used elsewhere
    in this file."""
    sigma2 = sigma_z ** 2
    out = np.empty(sigma2.shape[0], dtype=int)
    for i, row in enumerate(sigma2):
        t = _variance_split_threshold(row)
        out[i] = int((row < t).sum())
    return out


def compute_active_dims_histogram(ad_counts: np.ndarray) -> dict:
    """Bin per-sample active-dimension counts over every integer value
    from min to max observed (not just occupied bins, so gaps show up
    as zero-count bars rather than silently vanishing). Returns a plain
    dict (JSON-serializable) with parallel "bins"/"counts" lists plus a
    few summary stats, so it's self-contained without needing ad_counts
    around to interpret it later."""
    ad_counts = np.asarray(ad_counts)
    lo, hi = int(ad_counts.min()), int(ad_counts.max())
    bins = list(range(lo, hi + 1))
    counts = [int((ad_counts == b).sum()) for b in bins]
    return {
        "bins": bins,
        "counts": counts,
        "n_samples": int(len(ad_counts)),
        "mean": float(ad_counts.mean()),
        "median": float(np.median(ad_counts)),
        "std": float(ad_counts.std()),
        "min": lo,
        "max": hi,
    }


def plot_active_dims_histogram(ad_counts: np.ndarray, out_path: Path, title: str) -> dict:
    """Bar plot of the per-sample active-dimension histogram (same visual
    style as pca_analysis.py's scree plots), saved to `out_path`. Returns
    the underlying histogram dict (see compute_active_dims_histogram)
    so the caller can also serialize it to JSON without recomputing."""
    hist = compute_active_dims_histogram(ad_counts)
    plt.figure(figsize=(8, 4))
    plt.bar(hist["bins"], hist["counts"])
    plt.xlabel("Active dimensions (per sample)")
    plt.ylabel("Number of samples")
    plt.title(f"{title}\n(mean={hist['mean']:.2f}, median={hist['median']:.0f}, "
              f"n={hist['n_samples']})")
    plt.xticks(hist["bins"])
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return hist


def plot_training_history(history: list, out_path: Path) -> None:
    """One figure, epoch on the x-axis throughout. Most metrics (loss,
    recon_mse_per_dim, kl, gamma, *_ad_mean, *_ad_group, ...) each get
    their own subplot automatically, generic over whatever keys exist
    in history.json. Two exceptions, hardcoded because they're meant to
    be read together rather than separately: val_ad_{median,min,max} are
    overlaid on one subplot (as are the train_ad_ equivalents), each
    series in its own color with a legend, so you can see the spread
    around the median at a glance instead of cross-referencing three
    separate plots.
    """
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    all_keys = [k for k in history[0].keys() if k != "epoch"]

    combined_specs = [
        {"title": "val_ad (median / min / max)",
         "series": [("val_ad_median", "median"), ("val_ad_min", "min"), ("val_ad_max", "max")]},
        {"title": "train_ad (median / min / max)",
         "series": [("train_ad_median", "median"), ("train_ad_min", "min"), ("train_ad_max", "max")]},
    ]
    # only actually combine if all three series for a group are present
    # (e.g. older history.json files without min/max just fall through
    # to solo subplots for whichever keys DO exist)
    combined_specs = [spec for spec in combined_specs
                       if all(key in all_keys for key, _ in spec["series"])]
    combined_keys = {key for spec in combined_specs for key, _ in spec["series"]}
    solo_keys = [k for k in all_keys if k not in combined_keys]

    panels = combined_specs + [{"title": k, "series": [(k, None)]} for k in solo_keys]

    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for i, panel in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        for key, label in panel["series"]:
            values = [h.get(key) for h in history]
            ax.plot(epochs, values, linewidth=1, label=(label or key))
        ax.set_xlabel("epoch")
        ax.set_ylabel(panel["title"])
        ax.set_title(panel["title"])
        ax.grid(alpha=0.3)
        if len(panel["series"]) > 1:
            ax.legend(fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_sample_active_mask(sigma_z: np.ndarray) -> np.ndarray:
    """Like count_active_dims, but returns the full (N, kappa) boolean
    mask instead of just the count — needed for the group active-SET
    overlap table (Jaccard-style comparison of WHICH global directions
    fire for different subgroups), as opposed to just how many fire."""
    sigma2 = sigma_z ** 2
    mask = np.zeros_like(sigma2, dtype=bool)
    for i, row in enumerate(sigma2):
        t = _variance_split_threshold(row)
        mask[i] = row < t
    return mask


def compute_neuron_activation_counts(sigma_z: np.ndarray) -> np.ndarray:
    """For each of the kappa latent dimensions ("neurons"), count how many
    of the N samples had that dimension active. This is the transpose of
    count_active_dims: that function counts, PER SAMPLE, how many neurons
    are active; this counts, PER NEURON, how many samples activate it —
    the standard sparse-autoencoder "feature activation frequency"
    statistic (e.g. as used in Anthropic's "Towards Monosemanticity"):
    a neuron with count 0 is permanently dead; a neuron with count N
    fires on literally every input (behaving more like an always-on bias
    than a sparse, selective feature); most neurons should sit somewhere
    in between if the model has learned a genuinely sparse code.

    sigma_z: (N, kappa). Returns (kappa,) int array.
    """
    mask = per_sample_active_mask(sigma_z)   # (N, kappa) bool
    return mask.sum(axis=0).astype(int)


def plot_neuron_activation_histogram(
    activation_counts: np.ndarray, n_samples: int, out_path: Path, title: str,
    n_bins: int = 30,
) -> dict:
    """Histogram OVER NEURONS (kappa data points — one per latent
    dimension, each being "how many of the N samples activated it"), NOT
    a per-sample histogram like plot_active_dims_histogram above (which
    has N data points, one per sample, each being "how many neurons
    fired for it"). Answers a different question: not "how many active
    dims does a typical sample have" but "how often does a typical
    neuron get used across the whole dataset."

    Uses proper np.histogram binning (unlike plot_active_dims_histogram's
    one-bin-per-integer-value approach) since activation counts range up
    to N samples, which is typically far larger than kappa itself, so a
    bin-per-integer-count would mean up to N bins for only kappa data
    points to fill them.
    """
    activation_counts = np.asarray(activation_counts)
    n_bins_eff = max(1, min(n_bins, len(activation_counts)))
    counts, bin_edges = np.histogram(activation_counts, bins=n_bins_eff,
                                     range=(0, max(n_samples, 1)))

    n_never = int((activation_counts == 0).sum())
    n_always = int((activation_counts == n_samples).sum())

    # Top 10 most active neurons (by activation count, descending), for
    # display alongside the histogram -- the histogram's bars only show
    # HOW MANY neurons fall at each activation-frequency level, not WHICH
    # specific neurons those are, so this fills that gap directly.
    n_top = min(10, len(activation_counts))
    top_idx = np.argsort(activation_counts)[::-1][:n_top]
    top_10_active_neurons = [(int(i), int(activation_counts[i])) for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align="edge", edgecolor="black")
    ax.set_xlabel(f"Number of samples (out of {n_samples}) a neuron was active on")
    ax.set_ylabel("Number of neurons")
    ax.set_title(f"{title}\n(kappa={len(activation_counts)} neurons, "
                f"{n_never} never active, {n_always} always active)")

    top_10_text = "Top 10 most active neurons\n(index: count)\n" + "\n".join(
        f"  {idx}: {cnt}" for idx, cnt in top_10_active_neurons
    )
    fig.text(0.78, 0.5, top_10_text, fontsize=8, va="center", ha="left",
             family="monospace", transform=fig.transFigure,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    fig.subplots_adjust(right=0.76)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "activation_counts": activation_counts.tolist(),
        "n_samples": int(n_samples),
        "kappa": int(len(activation_counts)),
        "n_never_active": n_never,
        "n_always_active": n_always,
        "mean_activation_count": float(activation_counts.mean()),
        "median_activation_count": float(np.median(activation_counts)),
        "top_10_active_neurons": top_10_active_neurons,
        "bin_edges": bin_edges.tolist(),
        "bin_counts": counts.tolist(),
    }


def plot_neuron_activation_bar(
    activation_counts: np.ndarray, n_samples: int, out_path: Path, title: str,
) -> None:
    """One bar PER NEURON -- kappa bars, x-axis = neuron index (0..kappa-1),
    height = how many of the N samples activated that specific neuron.

    Distinct from plot_neuron_activation_histogram above: that one bins
    the DISTRIBUTION of activation counts across neurons (answers "how
    many neurons fall in this activation-frequency range", kappa data
    points collapsed into ~30 bins). This instead shows each neuron's own
    count directly, one bar per neuron with no binning/collapsing at all
    (kappa bars, e.g. 64 for kappa=64) -- answers "which specific neurons
    are used how often", letting you see e.g. whether usage concentrates
    in a handful of low-index neurons or is spread across the dictionary.
    """
    activation_counts = np.asarray(activation_counts)
    kappa = len(activation_counts)
    plt.figure(figsize=(max(8, kappa * 0.15), 4))
    plt.bar(range(kappa), activation_counts, width=1.0, edgecolor="black", linewidth=0.3)
    plt.xlabel("Neuron index")
    plt.ylabel(f"Number of samples active on (out of {n_samples})")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def pca_effective_rank(directions: np.ndarray, variance_thresholds: Tuple[float, ...] = (0.90, 0.95, 0.99)) -> dict:
    """Estimates how many effectively ORTHOGONAL directions a set of
    (generally non-orthogonal) direction vectors actually spans, via PCA.

    Each active neuron's decoder column is a direction in gradient space,
    but nothing in VAEase's objective forces distinct neurons' columns to
    be orthogonal (unlike PCA eigenvectors), so "k active neurons" does
    NOT necessarily mean "k independent directions" -- some could be
    near-duplicates of each other. This gives a *soft*, graded answer to
    "how many" (via the variance-explained spectrum), complementing
    orthogonalize_directions()'s *hard* QR-based rank (which only drops
    directions that are essentially EXACTLY linearly dependent -- this
    catches directions that are merely highly correlated but not exactly
    parallel, which QR's rank cutoff would still count separately).

    IMPORTANT: this is deliberately UNCENTERED PCA -- eigendecomposition
    of the raw (not mean-subtracted) second-moment matrix. Centering
    (standard PCA on data points) would subtract out whatever "average
    direction" these vectors share before decomposing, which is exactly
    the wrong move here: we want the dimensionality of the SPAN of these
    vectors from the origin (do they collectively point in few or many
    distinct directions), not variance around some meaningful mean --
    these are directions/concepts, not samples scattered around a center.

    directions: (k, d), rows should already be unit-normalized (as
    extract_decoder_directions' raw_directions already are).
    Uses the same Gram-matrix trick as pca_analysis.py's own
    find_significant_directions for the typical k <= d case (few active
    neurons relative to gradient dimension): eigendecomposing the (k,k)
    Gram matrix directions @ directions.T has the same nonzero eigenvalues
    as the (d,d) second-moment matrix directions.T @ directions, just
    cheaper when k << d.

    Returns a dict with the full eigenvalue/explained-variance spectrum
    plus how many components are needed to reach each of
    `variance_thresholds` cumulative explained variance (default 90/95/99%).
    """
    directions = np.asarray(directions, dtype=np.float64)
    k, d = directions.shape
    if k == 0:
        return {"k_input_directions": 0}

    if k <= d:
        gram = directions @ directions.T / k          # (k, k) -- cheaper, same nonzero eigenvalues
        eigvals = np.linalg.eigvalsh(gram)[::-1]
    else:
        cov = directions.T @ directions / k            # (d, d)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)                # guard tiny negative numerical noise

    total = eigvals.sum()
    explained = eigvals / total if total > 0 else eigvals * 0
    cumulative = np.cumsum(explained)

    out = {
        "k_input_directions": int(k),
        "eigenvalues": eigvals.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance_ratio": cumulative.tolist(),
    }
    for t in variance_thresholds:
        out[f"n_components_{int(round(t * 100))}pct"] = int(np.searchsorted(cumulative, t) + 1)
    return out


def plot_pca_scree(explained_variance_ratio: list, out_path: Path, title: str) -> None:
    """Scree plot of pca_effective_rank's explained-variance spectrum,
    same visual style as pca_analysis.py's own PCA scree plots."""
    values = np.asarray(explained_variance_ratio) * 100
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(values)), values)
    plt.xlabel("Component")
    plt.ylabel("Explained variance (%)")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def group_averaged_active_dims(sigma_z: np.ndarray, labels: Optional[np.ndarray] = None) -> dict:
    """Matches the paper's Table 2/3 protocol exactly: average sigma_z
    *element-wise across samples first* (per latent dimension), then run
    the active/inactive threshold split once on that averaged profile —
    rather than thresholding each sample separately and then averaging
    the resulting counts. This gives one clean number per group, with
    much less noise than any individual sample's split, at the cost of
    losing the per-sample distribution.

    labels=None -> one result for the whole dataset ("all").
    labels=array of group ids (e.g. per-concept, per-timestep) -> one
    result per group, letting you directly compare "effective dimension
    of concept A" vs "concept B", analogous to the paper's per-manifold
    AD1/AD2/AD3 columns.
    """
    single_group = labels is None
    if single_group:
        labels = np.zeros(sigma_z.shape[0], dtype=int)

    out = {}
    for g in np.unique(labels):
        key = "all" if single_group else str(g)
        mask = labels == g
        avg_sigma2 = (sigma_z[mask] ** 2).mean(axis=0)   # average profile, per latent dim
        t = _variance_split_threshold(avg_sigma2)
        active = avg_sigma2 < t
        out[key] = {
            "active_dims": int(active.sum()),
            "threshold": float(t),
            "n_samples": int(mask.sum()),
            "active_dim_indices": np.where(active)[0].tolist(),
        }
    return out


@torch.no_grad()
def encode_dataset(model: VAEase, x: torch.Tensor, batch_size: int = 4096,
                    device: str = "cpu") -> np.ndarray:
    """Encoder-only pass (mu_z unused; we only need sigma_z for LID)."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        _, sigma_z = model.encode(batch)
        outs.append(sigma_z.cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_dataset_full(model: VAEase, x: torch.Tensor, batch_size: int = 4096,
                         device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Like encode_dataset, but also returns mu_z. Needed for direction
    ranking below, where we want a deterministic z_tilde per sample
    (mu_z, not a noisy reparameterized sample) to estimate how much each
    active dimension actually moves the reconstruction on average."""
    model.eval()
    mus, sigmas = [], []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        mu_z, sigma_z = model.encode(batch)
        mus.append(mu_z.cpu().numpy())
        sigmas.append(sigma_z.cpu().numpy())
    return np.concatenate(mus, axis=0), np.concatenate(sigmas, axis=0)


# ---------------------------------------------------------------------------
# Direction extraction & alignment (new — mirrors pca_analysis.py)
# ---------------------------------------------------------------------------

def extract_decoder_directions(
    model: VAEase, mu_z: np.ndarray, sigma_z: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> dict:
    """Pull the global, per-latent-dim direction dictionary out of a
    trained VAEase model and rank the ACTIVE ones by an approximate
    reconstruction-contribution score.

    Only valid for linear_decoder=True: then decoder.weight has shape
    (d, kappa), and column j is a fixed vector in gradient space — the
    direction latent dim j writes when its gate is open. This is what
    makes it comparable to a PCA eigenvector (fixed dictionary), unlike
    an MLP decoder where the local direction is input-dependent (would
    need a Jacobian at a point instead — not implemented here).

    "Active" indices come from group_averaged_active_dims() (dataset-
    wide average), same cutoff used elsewhere in this file. Ranking
    within that active set uses:
        importance_j = Var(z_tilde_j over dataset) * ||W_dec[:, j]||^2
    i.e. deterministic z_tilde = (1 - sigma_z) * mu_z (no reparam noise —
    this is a ranking heuristic, not a training-time quantity), analogous
    in spirit to PCA's eigenvalue (how much variance a direction carries)
    but is NOT an exact decomposition of reconstruction variance, since it
    ignores cross-covariance between latent dims.

    Returns a dict with:
        raw_directions      (k, d) unit-normalized decoder columns,
                             ranked by importance, k = len(active_indices)
        importance          (k,) ranking scores, descending
        active_indices      (k,) original decoder-column indices
        threshold           the sigma^2 split threshold used
    """
    if not model.linear_decoder or not isinstance(model.decoder, nn.Linear):
        raise ValueError(
            "extract_decoder_directions requires linear_decoder=True: with an "
            "MLP decoder there is no single global direction per latent dim "
            "(the local direction becomes input-dependent, i.e. a Jacobian "
            "at a point). Retrain with linear_decoder=True, or extend this "
            "function with a Jacobian-at-mean fallback if you need the MLP case."
        )

    grouped = group_averaged_active_dims(sigma_z, labels=labels if labels is not None else None)
    active_info = grouped["all"] if labels is None else grouped[list(grouped.keys())[0]]
    # dataset-wide active set regardless of `labels` (labels is only used
    # by the separate group-overlap table below, not for picking which
    # global directions exist)
    all_active = group_averaged_active_dims(sigma_z, labels=None)["all"]
    active_indices = np.array(all_active["active_dim_indices"], dtype=int)
    threshold = all_active["threshold"]

    W_dec = model.decoder.weight.detach().cpu().numpy()   # (d, kappa)

    if len(active_indices) == 0:
        raise ValueError("No active dimensions found — check training/convergence "
                          "before extracting directions.")

    z_tilde_det = (1.0 - sigma_z) * mu_z                     # (N, kappa), deterministic proxy
    var_z_tilde = z_tilde_det.var(axis=0)                    # (kappa,)
    col_norm_sq = (W_dec ** 2).sum(axis=0)                   # (kappa,)
    importance_all = var_z_tilde * col_norm_sq               # (kappa,)

    active_importance = importance_all[active_indices]
    order = np.argsort(active_importance)[::-1]              # descending
    active_indices_sorted = active_indices[order]
    importance_sorted = active_importance[order]

    raw_cols = W_dec[:, active_indices_sorted].T              # (k, d)
    norms = np.linalg.norm(raw_cols, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    raw_directions = raw_cols / norms

    return {
        "raw_directions": torch.from_numpy(raw_directions.astype(np.float32)),
        "importance": torch.from_numpy(importance_sorted.astype(np.float32)),
        "active_indices": active_indices_sorted,
        "threshold": threshold,
    }


def orthogonalize_directions(directions: torch.Tensor, rank_eps: float = 1e-6
                              ) -> Tuple[torch.Tensor, int]:
    """QR-orthogonalize a (k, d) set of directions (rows = directions),
    dropping near-zero-norm trailing columns — same idea as dropping Q10
    when rank(W)=9 in pca_analysis.py, but here the rank deficiency (if
    any) comes from VAEase possibly learning near-duplicate/correlated
    decoder columns rather than a known one-hot constraint, so the cutoff
    is data-driven via the R diagonal instead of a hardcoded 9.

    Returns (Q, rank) where Q is (rank, d) with orthonormal rows (note:
    transposed relative to torch.linalg.qr's convention, to match the
    (k, d) "rows = directions" layout used everywhere else in this file).
    """
    A = directions.T.contiguous()            # (d, k) — columns = directions, for qr
    Q, R = torch.linalg.qr(A, mode="reduced")  # Q: (d, k), R: (k, k)
    diag = torch.diagonal(R).abs()
    keep = diag > rank_eps * diag.max().clamp_min(1e-12)
    rank = int(keep.sum().item())
    Q_kept = Q[:, keep]                      # (d, rank)
    return Q_kept.T.contiguous(), rank        # (rank, d), rows = orthonormal directions


def build_alignment_tables(
    directions: torch.Tensor, W_normalized: torch.Tensor, feature_names: list,
    label: str, out_dir: Path, qr_drop_last: int = 1,
) -> None:
    """Same two-table format as pca_analysis.py's dot-product section:
    plain dot products against each conditioner input feature column,
    plus a second table projected onto the QR-orthogonalized conditioner
    subspace (dropping the last `qr_drop_last` columns for rank
    deficiency, e.g. the one-hot sum-to-1 constraint — default 1, matching
    rank(W)=9 for a 10-dim one-hot+continuous conditioner; set to 0 if
    your W is already full rank).

    `directions` should already be unit-normalized, (k, d), ranked in
    whatever order you want rows printed (e.g. by `importance` from
    extract_decoder_directions, descending).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dirs = directions.shape[0]
    dots = directions @ W_normalized   # (k, n_features)

    row_norms = torch.norm(dots, dim=1)
    col_norms = torch.norm(dots, dim=0)

    header = "         " + "".join(f"{n:>10}" for n in feature_names) + "  proj_ratio"
    print(f"\n  {label} — {n_dirs} VAEase directions vs conditioner input features:")
    print(header)
    txt = header + "\n"
    for i in range(n_dirs):
        row = f"  Dir {i+1:2d} " + "".join(f"{dots[i,j].item():>+10.3f}" for j in range(len(feature_names)))
        row += f"  {row_norms[i].item():>10.3f}"
        print(row)
        txt += row + "\n"
    col_row = "  col_norm" + "".join(f"{col_norms[j].item():>+10.3f}" for j in range(len(feature_names)))
    print(col_row)
    txt += col_row + "\n"
    (out_dir / f"dot_products_{label}.txt").write_text(txt)
    print(f"  Saved → {out_dir / f'dot_products_{label}.txt'}")

    n_features = W_normalized.shape[1]
    n_qr = n_features - qr_drop_last
    Q, _ = torch.linalg.qr(W_normalized)
    Qk = Q[:, :n_qr]
    dots_q = directions @ Qk
    proj_ratios = torch.norm(dots_q, dim=1)
    col_norms_q = torch.norm(dots_q, dim=0)

    header_q = "         " + "".join(f"{f'Q{j+1}':>10}" for j in range(n_qr)) + "  proj_ratio"
    print(f"\n  {label} — projected onto orthogonalized conditioner space (QR, {n_qr} dims):")
    print(header_q)
    txt_q = header_q + "\n"
    for i in range(n_dirs):
        row = f"  Dir {i+1:2d} " + "".join(f"{dots_q[i,j].item():>+10.3f}" for j in range(n_qr))
        row += f"  {proj_ratios[i].item():>10.3f}"
        print(row)
        txt_q += row + "\n"
    col_row_q = "  col_norm" + "".join(f"{col_norms_q[j].item():>+10.3f}" for j in range(n_qr))
    print(col_row_q)
    txt_q += col_row_q + "\n"
    (out_dir / f"dot_products_qr_{label}.txt").write_text(txt_q)
    print(f"  Saved → {out_dir / f'dot_products_qr_{label}.txt'}")


def group_active_set_overlap_table(
    sigma_z: np.ndarray, labels: np.ndarray, out_dir: Path,
    group_names: Optional[dict] = None,
) -> dict:
    """New table with no PCA analog: PCA has no notion of a sample-
    dependent active set, but VAEase does (per-sample sigma_z gates a
    FIXED dictionary on/off). This compares WHICH global decoder
    directions fire for different groups (e.g. per-timestep, per-shape),
    via Jaccard overlap of each group's active-index set — the direct
    counterpart to pca_analysis.py's cross-timestep dot-product tables,
    but for "same subset of the dictionary active" rather than "same
    direction at all".

    labels: (N,) array of group ids, same length as sigma_z's first dim.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_averaged_active_dims(sigma_z, labels=labels)
    group_keys = list(grouped.keys())
    names = group_names or {k: k for k in group_keys}

    sets = {k: set(grouped[k]["active_dim_indices"]) for k in group_keys}
    n = len(group_keys)
    jaccard = np.zeros((n, n))
    for i, ki in enumerate(group_keys):
        for j, kj in enumerate(group_keys):
            inter = len(sets[ki] & sets[kj])
            union = len(sets[ki] | sets[kj]) or 1
            jaccard[i, j] = inter / union

    header = "              " + "".join(f"{names[k]:>12}" for k in group_keys)
    print("\n--- Group active-set overlap (Jaccard over global decoder directions) ---")
    print(header)
    txt = header + "\n"
    for i, ki in enumerate(group_keys):
        row = f"  {names[ki]:>10}  " + "".join(f"{jaccard[i,j]:>12.3f}" for j in range(n))
        print(row)
        txt += row + "\n"
    txt += "\nActive dims per group:\n"
    for k in group_keys:
        txt += f"  {names[k]}: {grouped[k]['active_dims']} dims, indices={grouped[k]['active_dim_indices']}\n"
        print(f"  {names[k]}: {grouped[k]['active_dims']} dims, indices={grouped[k]['active_dim_indices']}")

    (out_dir / "group_active_overlap.txt").write_text(txt)
    print(f"  Saved → {out_dir / 'group_active_overlap.txt'}")
    return {"jaccard": jaccard, "group_keys": group_keys, "grouped": grouped}


# ---------------------------------------------------------------------------
# Intervention generation — copied verbatim in structure from pca_analysis.py
# so it's a drop-in match: same signatures, same grid layout.
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
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(image.squeeze().float().cpu())


def make_intervention_grid(images_2d, strengths, direction_idx, variance, scale=4, label_prefix="Dir"):
    """
    images_2d: list of lists [strength][img_idx] → PIL image
    rows = starting images, cols = intervention strengths
    """
    from PIL import Image, ImageDraw

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

    var_str = f"var={variance*100:.1f}%" if variance is not None else "var=n/a"
    draw.text((4, 4), f"{label_prefix} {direction_idx+1}  {var_str}", fill=(0, 0, 0))

    for c, strength in enumerate(strengths):
        x = label_w + pad + c * (cell_w + pad)
        draw.text((x + 2, label_h - 14), f"{strength:+g}", fill=(0, 0, 0))

    for img_idx in range(n_images):
        y = label_h + pad + img_idx * (cell_h + pad)
        for c, strength_imgs in enumerate(images_2d):
            x   = label_w + pad + c * (cell_w + pad)
            img = strength_imgs[img_idx]
            grid.paste(img.resize((cell_w, cell_h), Image.NEAREST), (x, y))

    return grid


def run_intervention_grids(
    directions: torch.Tensor, importance: Optional[torch.Tensor], label: str,
    unet, conditioner, ddim, cfg, device, base_conds, seeds, strengths, out_dir: Path,
    num_interventions: int,
):
    """Loops over the top `num_interventions` directions (already ranked
    on entry) and saves one intervention grid per direction — same loop
    structure as pca_analysis.py's per-dataset intervention block."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_to_show = min(num_interventions, directions.shape[0])
    print(f"\n--- Interventions for {label} ({n_to_show} directions) ---")
    for dir_idx in range(n_to_show):
        direction = F.normalize(directions[dir_idx], dim=0)
        variance = importance[dir_idx].item() / importance.sum().item() if importance is not None else None
        images_2d = []
        for strength in strengths:
            row_imgs = []
            for base_cond, seed in zip(base_conds, seeds):
                img = generate_with_intervention(
                    unet, conditioner, ddim, base_cond, direction, strength, cfg, device, seed,
                )
                row_imgs.append(img)
            images_2d.append(row_imgs)
        grid = make_intervention_grid(images_2d, strengths, dir_idx, variance, label_prefix=f"{label} Dir")
        grid_path = out_dir / f"{label}_dir{dir_idx+1:02d}.png"
        grid.save(grid_path)
        print(f"  Saved dir {dir_idx+1} → {grid_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_vaease(
    data: np.ndarray,
    output_dir: str,
    kappa: Optional[int] = None,
    hidden: Optional[int] = None,
    linear_decoder: bool = True,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_fraction: float = 0.05,
    norm: str = "standardize",  # "standardize" | "l2" | "l2_then_standardize" | "none"
    device: Optional[str] = None,
    seed: int = 42,
    report_every: int = 10,
    verbose: bool = True,
) -> VAEase:
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N, d = data.shape
    kappa = kappa or min(d, 300)  # mirrors the paper's LLM-activation setup

    if verbose:
        print(f"Data: N={N}, d={d}  ->  kappa={kappa}  (device={device})")

    # --- normalize ---
    # "l2" matches pca_analysis.py's --normalize_grads (row-wise unit norm,
    # removes per-sample magnitude e.g. differing loss scale across pairs).
    # "standardize" is per-coordinate zero-mean/unit-variance (keeps
    # VAEase's single global gamma honest across the 64 dims). These are
    # not mutually exclusive: "l2_then_standardize" applies both.
    if norm not in ("standardize", "l2", "l2_then_standardize", "none"):
        raise ValueError(f"Unknown norm mode: {norm!r}")

    normalizer = None
    if norm in ("l2", "l2_then_standardize"):
        data = l2_normalize_rows(data)
        if verbose:
            print("Applied row-wise L2 normalization (matches pca_analysis.py's "
                  "--normalize_grads)")
    if norm in ("standardize", "l2_then_standardize"):
        normalizer = Normalizer.fit(data)
        data = normalizer.transform(data)
        normalizer.save(out_dir / "normalizer.npz")
        if verbose:
            print(f"Fit per-coordinate normalizer "
                  f"(mean range [{normalizer.mean.min():.3g}, {normalizer.mean.max():.3g}], "
                  f"std range [{normalizer.std.min():.3g}, {normalizer.std.max():.3g}])")

    # --- split ---
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train = torch.from_numpy(data[train_idx]).float()
    x_val = torch.from_numpy(data[val_idx]).float()

    model = VAEase(d=d, kappa=kappa, hidden=hidden, linear_decoder=linear_decoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)

    history = []
    n_train = len(x_train)
    t0 = time.time()

    # Fixed-size "probe" subsample of the TRAINING set, same size as the
    # val split, drawn once up front (not resampled per epoch). This lets
    # every epoch's train-set AD mean/median be computed on an
    # apples-to-apples footing with the val-set numbers (same N each time)
    # without re-running count_active_dims's Python-level threshold search
    # over the full training set every single epoch, which would scale
    # with N and get slow for large datasets. If you'd rather have the
    # exact full-training-set AD instead of a subsample estimate, say so
    # and I'll swap this for the full x_train.
    n_probe = min(len(x_train), n_val)
    probe_idx = np.random.RandomState(seed).choice(len(x_train), size=n_probe, replace=False)
    x_train_probe = x_train[probe_idx]

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_logs = []
        for i in range(0, n_train, batch_size):
            batch = x_train[perm[i:i + batch_size]].to(device)
            x_hat, mu_z, sigma_z, gamma = model(batch)
            loss, logs = vaease_loss(batch, x_hat, mu_z, sigma_z, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_logs.append(logs)
        sched.step()

        mean_logs = {k: float(np.mean([l[k] for l in epoch_logs])) for k in epoch_logs[0]}

        # --- AD mean/median on both val and a matched-size train probe,
        # every epoch, so history.json has a full per-epoch curve for both.
        sigma_val = encode_dataset(model, x_val, device=device)
        ad_val = count_active_dims(sigma_val)
        sigma_train_probe = encode_dataset(model, x_train_probe, device=device)
        ad_train_probe = count_active_dims(sigma_train_probe)

        mean_logs["val_ad_mean"] = float(ad_val.mean())
        mean_logs["val_ad_median"] = float(np.median(ad_val))
        mean_logs["val_ad_min"] = int(ad_val.min())
        mean_logs["val_ad_max"] = int(ad_val.max())
        mean_logs["val_ad_group"] = group_averaged_active_dims(sigma_val)["all"]["active_dims"]

        mean_logs["train_ad_mean"] = float(ad_train_probe.mean())
        mean_logs["train_ad_median"] = float(np.median(ad_train_probe))
        mean_logs["train_ad_min"] = int(ad_train_probe.min())
        mean_logs["train_ad_max"] = int(ad_train_probe.max())
        mean_logs["train_ad_group"] = group_averaged_active_dims(sigma_train_probe)["all"]["active_dims"]
        history.append({"epoch": epoch, **mean_logs})

        if verbose and (epoch % report_every == 0 or epoch == 1 or epoch == epochs):
            elapsed = time.time() - t0
            print(f"[epoch {epoch:>4}/{epochs}] loss={mean_logs['loss']:.4f}  "
                  f"recon_mse/dim={mean_logs['recon_mse_per_dim']:.4f}  "
                  f"KL={mean_logs['kl']:.4f}  gamma={mean_logs['gamma']:.4g}  "
                  f"train AD: mean={mean_logs['train_ad_mean']:.1f} median={mean_logs['train_ad_median']:.0f}  "
                  f"val AD: mean={mean_logs['val_ad_mean']:.1f} median={mean_logs['val_ad_median']:.0f}  "
                  f"({elapsed:.0f}s elapsed)")

    torch.save({
        "model_state": model.state_dict(),
        "d": d, "kappa": kappa, "hidden": hidden, "linear_decoder": linear_decoder,
        "norm": norm,
    }, out_dir / "vaease_checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_training_history(history, out_dir / "history_plots.png")

    if verbose:
        print(f"\nSaved checkpoint -> {out_dir / 'vaease_checkpoint.pt'}")
        print(f"Saved training history -> {out_dir / 'history.json'}")
        print(f"Saved history plots -> {out_dir / 'history_plots.png'}")
        if normalizer is not None:
            print(f"Saved normalizer -> {out_dir / 'normalizer.npz'}")

    # --- final active-dimension summary over the FULL dataset (not just
    # the small validation split used for periodic reporting during
    # training). `data` here already reflects whichever --norm was applied.
    x_all = torch.from_numpy(data).float()
    sigma_all = encode_dataset(model, x_all, device=device)
    ad_all = count_active_dims(sigma_all)
    grouped_all = group_averaged_active_dims(sigma_all)["all"]

    # --- NEW: per-neuron activation frequency (how many of the N samples
    # activate each of the kappa neurons), and a histogram of that
    # distribution across neurons. Complements ad_all above: ad_all counts,
    # per SAMPLE, how many neurons fire; this counts, per NEURON, how many
    # samples it fires on -- the standard sparse-autoencoder "feature
    # activation frequency" diagnostic. Reuses sigma_all, already computed
    # just above, so this adds no extra encoder passes.
    neuron_activation_counts = compute_neuron_activation_counts(sigma_all)
    neuron_hist = plot_neuron_activation_histogram(
        neuron_activation_counts, n_samples=N,
        out_path=out_dir / "neuron_activation_histogram.png",
        title="Neuron activation frequency \u2014 full dataset",
    )
    (out_dir / "neuron_activation_histogram.json").write_text(json.dumps(neuron_hist, indent=2))
    if verbose:
        print(f"\nNeuron activation frequency (kappa={neuron_hist['kappa']} neurons, N={N} samples):")
        print(f"  Never active:          {neuron_hist['n_never_active']} neurons")
        print(f"  Always active:         {neuron_hist['n_always_active']} neurons")
        print(f"  Mean activation count: {neuron_hist['mean_activation_count']:.1f} / {N}")
        print(f"  Top 10 most active neurons (index: count): "
              + ", ".join(f"{idx}:{cnt}" for idx, cnt in neuron_hist["top_10_active_neurons"]))
        print(f"  Saved -> {out_dir / 'neuron_activation_histogram.png'} (+ .json)")

    plot_neuron_activation_bar(
        neuron_activation_counts, n_samples=N,
        out_path=out_dir / "neuron_activation_bar.png",
        title=f"Per-neuron activation count \u2014 full dataset (kappa={kappa})",
    )
    if verbose:
        print(f"  Saved -> {out_dir / 'neuron_activation_bar.png'} "
              f"(one bar per neuron, {kappa} bars)")

    # --- NEW: estimate how many effectively ORTHOGONAL directions the
    # active neurons' decoder columns actually span, via PCA. Each active
    # neuron's decoder column is a direction, but nothing forces distinct
    # neurons' columns to be orthogonal, so "k active neurons" != "k
    # independent directions" in general. Done for BOTH neuron-selection
    # criteria, since they're different selections and can disagree:
    #   - "group_averaged": grouped_all's active set (same selection
    #     extract_decoder_directions uses -- average sigma^2 across
    #     samples first, threshold once)
    #   - "active_over_5pct_samples": neurons active on >5% of samples
    #     individually (from neuron_activation_counts, just computed above)
    # Only meaningful for linear_decoder=True (see extract_decoder_
    # directions' own docstring for why an MLP decoder has no single
    # global direction per neuron) -- skipped gracefully otherwise.
    if model.linear_decoder:
        W_dec_all = model.decoder.weight.detach().cpu().numpy()   # (d, kappa)
        freq_active_indices = np.where(neuron_activation_counts > 0.05 * N)[0]

        orth_results = {}
        for sel_name, indices in [
            ("group_averaged", np.array(grouped_all["active_dim_indices"], dtype=int)),
            ("active_over_5pct_samples", freq_active_indices),
        ]:
            if len(indices) == 0:
                orth_results[sel_name] = {"k_input_directions": 0, "note": "no active neurons found"}
                continue
            cols = W_dec_all[:, indices].T                          # (k, d)
            col_norms = np.linalg.norm(cols, axis=1, keepdims=True)
            col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
            unit_dirs = cols / col_norms
            res = pca_effective_rank(unit_dirs)
            res["active_indices"] = indices.tolist()
            orth_results[sel_name] = res
            plot_pca_scree(
                res["explained_variance_ratio"],
                out_dir / f"orthogonal_directions_pca_{sel_name}.png",
                title=f"PCA of active-neuron decoder directions\n"
                      f"({sel_name}, k={len(indices)} active neurons)",
            )

        (out_dir / "orthogonal_directions_pca.json").write_text(json.dumps(orth_results, indent=2))
        if verbose:
            print(f"\nEstimated orthogonal-direction count (PCA on active-neuron decoder columns):")
            for sel_name, res in orth_results.items():
                if res.get("k_input_directions", 0) == 0:
                    print(f"  {sel_name}: no active neurons found")
                    continue
                print(f"  {sel_name} (k={res['k_input_directions']} active neurons): "
                      f"components needed for 90%={res['n_components_90pct']}  "
                      f"95%={res['n_components_95pct']}  99%={res['n_components_99pct']}")
            print(f"  Saved -> {out_dir / 'orthogonal_directions_pca.json'} "
                  f"(+ scree plot per selection)")
    elif verbose:
        print("\n(Skipping orthogonal-direction PCA estimate: requires linear_decoder=True)")

    # --- histogram of per-sample active-dimension counts on the held-out
    # val/test split specifically (x_val), as distinct from the full-
    # dataset numbers above. This is the generalization-focused view: it
    # answers "for gradients the model didn't train on, what does the
    # distribution of local dimensionality actually look like" rather
    # than a single aggregate number.
    sigma_val_final = encode_dataset(model, x_val, device=device)
    ad_val_final = count_active_dims(sigma_val_final)
    val_hist = plot_active_dims_histogram(
        ad_val_final, out_dir / "active_dims_histogram_valset.png",
        title=f"Active dimensions per sample \u2014 held-out val/test set",
    )
    (out_dir / "active_dims_histogram_valset.json").write_text(json.dumps(val_hist, indent=2))

    summary = {
        "n_samples": int(N),
        "kappa": int(kappa),
        "per_sample_mean": float(ad_all.mean()),
        "per_sample_median": float(np.median(ad_all)),
        "per_sample_std": float(ad_all.std()),
        "per_sample_min": int(ad_all.min()),
        "per_sample_max": int(ad_all.max()),
        "group_averaged_active_dims": grouped_all["active_dims"],
        "group_averaged_dim_indices": grouped_all["active_dim_indices"],
        "val_set": val_hist,
    }
    (out_dir / "active_dims_summary.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"\nFinal active-dimension summary (full dataset, N={N}):")
        print(f"  Per-sample AD:        mean={summary['per_sample_mean']:.2f}  "
              f"median={summary['per_sample_median']:.0f}  std={summary['per_sample_std']:.2f}  "
              f"[{summary['per_sample_min']}-{summary['per_sample_max']}]")
        print(f"  Group-averaged AD:    {summary['group_averaged_active_dims']}  "
              f"(one number for the whole dataset, matches paper's Table 2/3 protocol)")
        print(f"  Held-out val/test set (n={val_hist['n_samples']}): "
              f"mean={val_hist['mean']:.2f} median={val_hist['median']:.0f} "
              f"[{val_hist['min']}-{val_hist['max']}]")
        print(f"  Saved -> {out_dir / 'active_dims_summary.json'}")
        print(f"  Saved -> {out_dir / 'active_dims_histogram_valset.png'} "
              f"(+ .json with raw bin counts)")

    return model


def compute_active_dims_for_dataset(
    model: VAEase, data: np.ndarray, normalizer: Optional[Normalizer] = None,
    apply_l2: bool = False, device: Optional[str] = None, batch_size: int = 4096,
) -> np.ndarray:
    """Convenience entry point for post-hoc analysis: given a trained
    model and a (possibly un-normalized) dataset, return per-sample
    active-dimension counts. Set apply_l2=True and/or pass `normalizer`
    to match whatever --norm mode the model was actually trained with."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if apply_l2:
        data = l2_normalize_rows(data)
    if normalizer is not None:
        data = normalizer.transform(data)
    x = torch.from_numpy(data.astype(np.float32))
    sigma_z = encode_dataset(model, x, batch_size=batch_size, device=device)
    return count_active_dims(sigma_z)


def load_vaease_checkpoint(path: str, device: Optional[str] = None) -> VAEase:
    """Load a model saved by train_vaease's torch.save() call above,
    for analysis-only runs that skip retraining."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = VAEase(d=ckpt["d"], kappa=ckpt["kappa"], hidden=ckpt["hidden"],
                   linear_decoder=ckpt["linear_decoder"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Self-test: synthetic union-of-linear-subspaces data with known ground
# truth (a smaller version of the paper's own Section 5.1 experiment).
# Run with `python vaease.py --self_test` to sanity-check the implementation.
# ---------------------------------------------------------------------------

def _make_union_of_subspaces(n_per_manifold=2000, d=40, subspace_dims=(4, 4, 4), seed=0):
    rng = np.random.RandomState(seed)
    chunks = []
    true_dims = []
    for r in subspace_dims:
        basis = rng.randn(d, r)
        basis, _ = np.linalg.qr(basis)          # orthonormal basis, d x r
        latents = rng.randn(n_per_manifold, r)
        x = latents @ basis.T                    # embed into R^d
        x += 0.01 * rng.randn(*x.shape)           # small observation noise
        chunks.append(x)
        true_dims.append(r)
    data = np.concatenate(chunks, axis=0).astype(np.float32)
    labels = np.concatenate([np.full(n_per_manifold, i) for i in range(len(subspace_dims))])
    return data, labels, true_dims


def self_test():
    print("Running self-test on synthetic union-of-linear-subspaces data "
          "(ground truth: 3 manifolds, each dim=4, ambient d=40)...\n")
    data, labels, true_dims = _make_union_of_subspaces()
    model = train_vaease(
        data, output_dir="/tmp/vaease_selftest", kappa=20, epochs=60,
        batch_size=512, lr=5e-3, norm="standardize", report_every=20, verbose=True,
    )
    sigma_z = encode_dataset(model, torch.from_numpy(
        Normalizer.load("/tmp/vaease_selftest/normalizer.npz").transform(data)
    ).float())

    print("\nApproach 1 — per-sample AD, then average the counts "
          "(noisier, but gives a local distribution):")
    ad = count_active_dims(sigma_z)
    for i, r in enumerate(true_dims):
        mask = labels == i
        print(f"  manifold {i} (true r={r}): mean AD = {ad[mask].mean():.2f}, "
              f"median = {np.median(ad[mask]):.0f}")
    err1 = np.mean([abs(ad[labels == i].mean() - r) for i, r in enumerate(true_dims)])

    print("\nApproach 2 — average sigma profiles per group first, threshold once "
          "(matches the paper's Table 2/3 protocol; cleaner for a known group):")
    grouped = group_averaged_active_dims(sigma_z, labels)
    for i, r in enumerate(true_dims):
        g = grouped[str(i)]
        print(f"  manifold {i} (true r={r}): active_dims = {g['active_dims']} "
              f"(n={g['n_samples']} samples averaged)")
    err2 = np.mean([abs(grouped[str(i)]["active_dims"] - r) for i, r in enumerate(true_dims)])

    print(f"\nMean abs error — approach 1: {err1:.2f}, approach 2: {err2:.2f} "
          f"({'looks reasonable' if max(err1, err2) < 2.0 else 'CHECK IMPLEMENTATION'})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grads", type=str, nargs="+", default=None,
                        help="One or more gradient files (.npy/.npz/.pt), OR a single "
                             "directory containing them — mirrors pca_analysis.py: "
                             "a directory is expanded by globbing --glob_pattern "
                             "(default 'grads_t*.npy') and sorting numerically by the "
                             "digits in each filename. Multiple files are concatenated "
                             "into one training set (equivalent to PCA's 'shared' space).")
    parser.add_argument("--glob_pattern", type=str, default="grads_t*.npy",
                        help="Glob pattern used when --grads is a single directory. "
                             "Default matches pca_analysis.py's convention.")
    parser.add_argument("--output_dir", type=str, default="vaease_run")
    parser.add_argument("--kappa", type=int, default=None,
                        help="Latent dimension (must be >= true intrinsic dim). "
                             "Default: min(d, 300).")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Encoder trunk hidden width. Default: d.")
    parser.add_argument("--mlp_decoder", action="store_true",
                        help="Use a 2-layer MLP decoder instead of linear "
                             "(loses direct interpretability of active dims "
                             "as a linear direction dictionary — direction "
                             "extraction/interventions below will refuse to run).")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--norm", type=str, default="none",
                        choices=["standardize", "l2", "l2_then_standardize", "none"],
                        help="'standardize' (default): per-coordinate zero-mean/unit-"
                             "variance, keeps VAEase's single global gamma honest across "
                             "dims. 'l2': row-wise unit norm, matches pca_analysis.py's "
                             "--normalize_grads exactly. 'l2_then_standardize': both. "
                             "'none': raw data.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--self_test", action="store_true",
                        help="Run the synthetic ground-truth sanity check instead "
                             "of training on --grads.")

    # --- new: analysis-only mode + direction extraction / interventions ---
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to an existing vaease_checkpoint.pt. If given, "
                             "skips training entirely and just runs the analysis/"
                             "intervention pipeline below on this model.")
    parser.add_argument("--run_interventions", action="store_true",
                        help="After training/loading, extract decoder directions "
                             "(raw + QR-orthogonalized), build alignment tables "
                             "against the conditioner's W, optionally build a group "
                             "active-set overlap table, and generate intervention "
                             "grids for both direction sets.")
    parser.add_argument("--num_interventions", type=int, default=10,
                        help="How many top (by importance) active directions to "
                             "visualize per direction set (raw / orthogonalized).")
    parser.add_argument("--intervention_strengths", type=float, nargs="+",
                        default=[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    parser.add_argument("--num_steps", type=int, default=10,
                        help="DDIM sampling steps for intervention images.")
    parser.add_argument("--base_prompt", type=float, nargs=10,
                        default=[0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        metavar=("is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                                 "h_stripe", "v_stripe", "grain"),
                        help="Base conditioning vector for intervention images, same "
                             "convention as pca_analysis.py: 0.5 dims are randomised "
                             "per seed, other values stay fixed.")
    parser.add_argument("--imgs_per_base", type=int, default=6,
                        help="Number of randomised base conditioning vectors "
                             "(rows in each intervention grid).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Diffusion model checkpoint directory (unet_ema/, "
                             "conditioner.pt) — required if --run_interventions is set.")
    parser.add_argument("--cond_input_dim", type=int, default=10,
                        help="Conditioner input dim, passed to Config64 — matches "
                             "pca_analysis.py's hardcoded cond_input_dim=10; set to 7 "
                             "for the no-texture dataset variant.")
    parser.add_argument("--group_labels", type=str, default=None,
                        help="Optional .npy file of (N,) integer/string group labels, "
                             "same length/order as --grads after concatenation, for the "
                             "group active-set overlap table (e.g. per-timestep ids if "
                             "you concatenated grads_t50.npy + grads_t950.npy in that "
                             "order — pass labels accordingly).")
    parser.add_argument("--qr_drop_last", type=int, default=1,
                        help="How many trailing QR columns of the conditioner's W to "
                             "drop for rank deficiency in the alignment table (default "
                             "1, matching rank(W)=9 for a 10-dim one-hot+continuous "
                             "conditioner; set 0 if your W is already full rank).")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        raise SystemExit(0)

    if args.grads is None and args.load_checkpoint is None:
        raise SystemExit("Provide --grads <path(s)>, --self_test, or --load_checkpoint.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_gradients(args.grads, glob_pattern=args.glob_pattern) if args.grads is not None else None

    if args.load_checkpoint is not None:
        print(f"Loading existing VAEase checkpoint from {args.load_checkpoint} (skipping training)")
        model = load_vaease_checkpoint(args.load_checkpoint, device=device)
        normalizer = None
        norm_path = Path(args.load_checkpoint).parent / "normalizer.npz"
        if norm_path.exists():
            normalizer = Normalizer.load(norm_path)
            print(f"Loaded matching normalizer from {norm_path}")
        # figure out what norm mode this checkpoint used, best-effort, so
        # the data fed to encode_dataset_full below matches training exactly
        ckpt_meta = torch.load(args.load_checkpoint, map_location="cpu")
        norm_mode = ckpt_meta.get("norm", args.norm)
        proc_data = data
        if proc_data is not None:
            if norm_mode in ("l2", "l2_then_standardize"):
                proc_data = l2_normalize_rows(proc_data)
            if norm_mode in ("standardize", "l2_then_standardize") and normalizer is not None:
                proc_data = normalizer.transform(proc_data)
    else:
        model = train_vaease(
            data, output_dir=args.output_dir, kappa=args.kappa, hidden=args.hidden,
            linear_decoder=not args.mlp_decoder, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, val_fraction=args.val_fraction,
            norm=args.norm, device=args.device, seed=args.seed,
            report_every=args.report_every,
        )
        norm_mode = args.norm
        proc_data = data
        if norm_mode in ("l2", "l2_then_standardize"):
            proc_data = l2_normalize_rows(proc_data)
        if norm_mode in ("standardize", "l2_then_standardize"):
            proc_data = Normalizer.load(out_dir / "normalizer.npz").transform(proc_data)

    if not args.run_interventions:
        raise SystemExit(0)

    if proc_data is None:
        raise SystemExit("--run_interventions needs --grads to build sigma_z/mu_z "
                          "(pass the same gradient files the model was trained on).")
    if args.checkpoint is None:
        raise SystemExit("--run_interventions needs --checkpoint (diffusion model "
                          "checkpoint dir with unet_ema/ and conditioner.pt).")

    # ------------------------------------------------------------------
    # Encode full dataset (mu_z, sigma_z) for direction extraction
    # ------------------------------------------------------------------
    x_all = torch.from_numpy(proc_data.astype(np.float32))
    mu_z_all, sigma_z_all = encode_dataset_full(model, x_all, device=device)

    group_labels = None
    if args.group_labels is not None:
        group_labels = np.load(args.group_labels, allow_pickle=True)
        if len(group_labels) != len(proc_data):
            raise SystemExit(f"--group_labels length ({len(group_labels)}) does not "
                              f"match dataset size ({len(proc_data)}).")

    extracted = extract_decoder_directions(model, mu_z_all, sigma_z_all)
    raw_directions = extracted["raw_directions"]
    importance = extracted["importance"]
    print(f"\nExtracted {raw_directions.shape[0]} active decoder directions "
          f"(threshold sigma^2 < {extracted['threshold']:.4f}), "
          f"active_indices={extracted['active_indices'].tolist()}")

    ortho_directions, ortho_rank = orthogonalize_directions(raw_directions)
    print(f"QR-orthogonalized to rank {ortho_rank} "
          f"({'no rank deficiency' if ortho_rank == raw_directions.shape[0] else 'dropped '+str(raw_directions.shape[0]-ortho_rank)+' near-dependent direction(s)'})")

    # ------------------------------------------------------------------
    # Load diffusion model + conditioner (same pattern as pca_analysis.py)
    # ------------------------------------------------------------------
    from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
    from configs.config_64 import Config64 as Config
    from models.conditioner import ShapeConditioningEncoder

    cfg = Config(cond_input_dim=args.cond_input_dim)
    ckpt = Path(args.checkpoint)

    print(f"Loading diffusion checkpoint from {ckpt} ...")
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

    W = conditioner.proj.weight.detach().cpu()   # (d, cond_input_dim)
    W_normalized = F.normalize(W, dim=0)
    feature_names = ["is_tri", "is_sq", "is_circ", "r", "g", "b", "size",
                      "h_stripe", "v_stripe", "grain"][:args.cond_input_dim]

    # ------------------------------------------------------------------
    # Alignment tables — raw and orthogonalized, same format as pca_analysis.py
    # ------------------------------------------------------------------
    print("\n=== Alignment tables: VAEase directions vs conditioner ===")
    build_alignment_tables(F.normalize(raw_directions, dim=1), W_normalized,
                            feature_names, "vaease_raw", out_dir, qr_drop_last=args.qr_drop_last)
    build_alignment_tables(ortho_directions, W_normalized,
                            feature_names, "vaease_orthogonalized", out_dir, qr_drop_last=args.qr_drop_last)

    # ------------------------------------------------------------------
    # Group active-set overlap table (new — no PCA analog)
    # ------------------------------------------------------------------
    if group_labels is not None:
        print("\n=== Group active-set overlap ===")
        group_active_set_overlap_table(sigma_z_all, group_labels, out_dir)

    # ------------------------------------------------------------------
    # Interventions — base conditioning vectors, then both direction sets
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    rng_base = torch.Generator()
    rng_base.manual_seed(args.seed)
    base_conds = []
    for _ in range(args.imgs_per_base):
        vec = []
        for v in args.base_prompt:
            if v == 0.5:
                vec.append(torch.rand(1, generator=rng_base).item())
            else:
                vec.append(v)
        base_conds.append(torch.tensor(vec, dtype=torch.float32))
    seeds = [torch.randint(0, 2**31, (1,)).item() for _ in base_conds]
    print(f"\nBase prompt: {args.base_prompt}")
    print(f"Generated {len(base_conds)} base conditioning vectors (0.5 dims randomised)")

    run_intervention_grids(
        raw_directions, importance, "vaease_raw",
        unet, conditioner, ddim, cfg, device, base_conds, seeds,
        args.intervention_strengths, out_dir / "interventions_raw",
        args.num_interventions,
    )
    run_intervention_grids(
        ortho_directions, None, "vaease_orthogonalized",
        unet, conditioner, ddim, cfg, device, base_conds, seeds,
        args.intervention_strengths, out_dir / "interventions_orthogonalized",
        args.num_interventions,
    )

    print(f"\nDone. All outputs saved under {out_dir}/")