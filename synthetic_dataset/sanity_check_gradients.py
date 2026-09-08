"""
sanity_check_gradients.py
--------------------------
Sanity-checks whether cond_out gradients (as computed by collect_gradients.py)
actually correspond to real semantic differences -- circles only throughout.

Design: for --n_pairs independent pairs, render TWO ground-truth circles
(circle A, circle B), each with its OWN fully random, deterministic
(seeded) continuous features (deterministic_random_vec) -- e.g. for
cond_input_dim=7: A=[0,0,1, .32,.71,.05,.44], B=[0,0,1, .88,.02,.63,.19],
completely independent random draws, printed under each image in every
plot. The gradient's cond_out, however, is computed AS IF from the fixed
NEUTRAL prompt (every continuous feature at exactly 0.5) -- NOT from
circle A's own random values. This is a deliberate mismatch, not an
oversight: rendering a concrete, reproducible circle needs specific
numeric values (the neutral prompt alone can't produce two distinct
circles to compare), but the gradient stays anchored to the same neutral
point used throughout this script, matching build_circle_basis_matrix's
comparison basis. The actual gradient recipe -- imported directly from
collect_gradients.py (noise_image, predict_x0), not reimplemented --
ELROND Eq. 1-4:
  1. Noise circle A's image to timestep t
  2. ONE UNet call at exactly t, conditioned on the NEUTRAL prompt (not
     circle A's own random vector) -- grad tracked on cond_out --
     closed-form clean-image estimate:
     x0_hat = (z_t - sigma_t*eps_theta(z_t,t,c)) / alpha_t
  3. Loss = MSE(x0_hat, circle_B_image) -- against the RAW target, not a
     denoised version of it
  4. Backprop -> gradient w.r.t. cond_out
No multi-step DDIM trajectory anywhere -- one model call per gradient,
always evaluated at exactly the declared timestep (see
collect_gradients.py's module docstring for why this replaced an
earlier multi-step version). Both circles are ground-truth RENDERED
(render_image), not network-generated -- there's no reverse-diffusion
sampling process to visualize here at all.
Circle B's own conditioning vector is never fed to the model -- it only
supplies the pixel target for the loss, same as collect_gradients.py's
own average_anchor mode (one fixed anchor prompt, many different targets).

Comparisons are against the "circle vector subspace" basis, NOT raw
columns of conditioner.proj.weight: a bare weight column corresponds to
conditioner(e_i), i.e. an input with shape=[0,0,0] -- not a one-hot, not
a valid circle, not a valid anything, a vector no real image was ever
conditioned on. Instead, build_circle_basis_matrix defines each feature's
basis direction as conditioner(valid circle with that feature=1) -
conditioner(the same neutral circle) -- always a real, valid circle
conditioning vector on both ends, matching the same neutral anchor the
actual gradient's cond_out is computed with respect to.

Sign convention: every print/plot reports -gradient (alignment_scores(
-grad_a, W)), the direction that would DECREASE the loss -- not the raw
gradient itself, which points toward INCREASING the loss (away from
matching circle B). A gradient that's actually tracking real semantics
should show POSITIVE alignment between -gradient and the true
conditioner(circle B) - conditioner(neutral) direction. Since circle A
and B differ randomly across EVERY continuous feature (not one isolated
feature), there's no single "intended" feature to highlight anymore --
every bar chart shows all features on equal footing.

For every pair, prints the actual scalar products (cosine alignment) of
the gradient against the circle-subspace basis -- both at each of
--timesteps individually, and averaged over a dense --sweep_step sweep
for a statistic that isn't sensitive to which few timesteps happened to
be hand-picked. Each sweep point is now a single model call (ELROND
Eq. 2), so this sweep is dramatically cheaper than it used to be under
the old multi-step recipe -- denser sweeps (lower --sweep_step) are
correspondingly cheap to run too.

Usage:
    python sanity_check_gradients.py \\
        --checkpoint outputs/checkpoints/outputs_64_acc_no_text/checkpoint-epoch-0200 \\
        --out_dir outputs/sanity_checks --cond_input_dim 7 \\
        --timesteps 900 500 100 --n_samples 5 --n_pairs 4

Must be run from the same directory as collect_gradients.py (or with it
on the Python path), since it imports noise_image/predict_x0 from there
directly rather than reimplementing them -- this is deliberate, so the
sanity check can never silently drift out of sync with your actual
gradient collection recipe.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DConditionModel

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder
from collect_gradients import noise_image, predict_x0, FEATURE_NAMES_FULL, N_SHAPE_DIMS

# Direct, unambiguous import -- NOT a try/except fallback through
# dataset_64_update. That fallback used to make sense when both names
# were in circulation, but now creates a real risk: if an OLDER,
# pre-position dataset_64_update.py (still h_stripe/v_stripe) happens to
# exist anywhere on the path, `try: from dataset_64_update import ...`
# would succeed silently and this script would end up rendering with
# the WRONG (stale, stripes-based) SampleVector64/ImageGenerator64 --
# even though every other line here already correctly expects pos_x/
# pos_y. dataset_64.py is the confirmed, current, position-based
# generator, so that's what gets imported, unconditionally.
from dataset_64_position import SampleVector64, ImageGenerator64


# ---------------------------------------------------------------------------
# Rendering ground-truth images from a raw conditioning vector
# ---------------------------------------------------------------------------

NEUTRAL_FULL_TAIL = [0.0, 0.0, 1.0,   # shape one-hot (only used if cond_vec omits shape entirely)
                     0.5, 0.5, 0.5,   # r, g, b
                     0.5,             # size
                     0.5, 0.5]        # pos_x, pos_y -- default to 0.5 (CENTERED), not 0.0 --
                                       # 0.0 would render off-center, not a neutral default,
                                       # matching dataset_64.py's own GROUP_DEFAULTS
                                       # convention for position specifically


def render_image(cond_vec: List[float], seed: int = 0) -> Tuple[torch.Tensor, "PIL.Image.Image"]:
    """cond_vec: list of length cond_input_dim, standard prefix order.
    Trailing dims not provided are padded with neutral defaults (0.5 for
    r/g/b/size/pos_x/pos_y -- centered, matching every other neutral
    default here).
    Returns (image_tensor[3,64,64] in [-1,1], raw_PIL_image)."""
    full = list(cond_vec) + NEUTRAL_FULL_TAIL[len(cond_vec):]
    vec = SampleVector64.from_list(full)
    img = ImageGenerator64().generate(vec)
    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    normalized = arr * 2.0 - 1.0   # matches collect_gradients.py's own [0,1] -> [-1,1] convention exactly
    return normalized, img


def neutral_shape_prompt(shape_id: int, cond_input_dim: int) -> list:
    """The fixed neutral baseline prompt: shape one-hot fixed, EVERY
    continuous feature at exactly 0.5 -- e.g. for shape_id=2 (circle),
    cond_input_dim=7: [0,0,1, 0.5,0.5,0.5, 0.5]. This is the anchor
    circle: the gradient is computed with respect to THIS prompt's
    conditioning, and every extreme case is this same vector with
    exactly one feature flipped to 1.0."""
    one_hot = [0.0, 0.0, 0.0]
    one_hot[shape_id] = 1.0
    return one_hot[:min(3, cond_input_dim)] + [0.5] * (cond_input_dim - N_SHAPE_DIMS)


def deterministic_random_vec(shape_id: int, cond_input_dim: int, seed: int) -> list:
    """A fully random (every continuous feature independently sampled
    uniform in [0,1)), but fully DETERMINISTIC given `seed`, circle
    vector -- used to render two concrete, reproducible ground-truth
    circles to compare, as distinct from neutral_shape_prompt (the fixed
    anchor gradients are computed with respect to -- see main()'s NOTE
    on the deliberate mismatch this creates). Same shape one-hot
    convention as neutral_shape_prompt."""
    rng = random.Random(seed)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[shape_id] = 1.0
    return one_hot[:min(3, cond_input_dim)] + [rng.random() for _ in range(cond_input_dim - N_SHAPE_DIMS)]


def format_vector_caption(vec_list: list, feature_names: list) -> str:
    """Readable, feature-name-labeled caption for a raw conditioning
    vector's continuous part -- e.g. 'r=0.64  g=0.03  b=0.28  size=0.22'
    instead of an unlabeled list of numbers you'd have to count
    positions in to interpret. Skips the shape one-hot prefix (never
    varies in this script -- always circle -- so it's not worth the
    space). Wrapped onto multiple lines at 4 features per line so it
    stays readable rather than running off the edge of the plot for
    cond_input_dim values with more continuous features."""
    continuous_vals = vec_list[N_SHAPE_DIMS:]
    parts = [f"{name}={v:.2f}" for name, v in zip(feature_names, continuous_vals)]
    lines = ["   ".join(parts[i:i + 4]) for i in range(0, len(parts), 4)]
    return "\n".join(lines)


def continuous_feature_names(cond_input_dim: int) -> list:
    """The continuous (non-shape) feature names for this cond_input_dim --
    r,g,b,size,[pos_x,pos_y] as applicable. Shape is excluded:
    every comparison in this script holds shape=circle fixed (per request:
    circles only), so there's no "is_circ direction" to speak of when
    shape never actually varies."""
    return FEATURE_NAMES_FULL[N_SHAPE_DIMS:cond_input_dim]


def circle_probe_vector(feature_name: str, value: float, base_vec_list: list, cond_input_dim: int) -> list:
    """`base_vec_list` (the fixed neutral base circle -- see
    neutral_shape_prompt) with ONE named continuous feature overridden to
    `value` -- e.g. circle_probe_vector('size', 1.0, [0,0,1,.5,.5,.5,.5], 7)
    -> [0,0,1, .5,.5,.5, 1.0]. Everything else in base_vec_list (shape
    one-hot and every other continuous feature) is preserved exactly."""
    names = continuous_feature_names(cond_input_dim)
    vec = list(base_vec_list)
    vec[N_SHAPE_DIMS + names.index(feature_name)] = value
    return vec


def build_circle_basis_matrix(conditioner, feature_names: list, cond_input_dim: int, device,
                               base_vec_list: list) -> np.ndarray:
    """The 'circle vector subspace' basis, as a (64, n_features) matrix
    aligned with `feature_names`: column i is the direction in cond_out
    space from the fixed neutral base circle (base_vec_list, everything
    at 0.5) to that same base circle with feature_names[i] maxed to
    1.0 -- e.g. conditioner([0,0,1, 1,.5,.5,.5]) -
    conditioner([0,0,1, .5,.5,.5,.5]) for the 'r' column. Matches
    neutral_shape_prompt's baseline exactly, so this basis reflects the
    same anchor the actual gradients are computed with respect to.

    This replaces using raw columns of conditioner.proj.weight directly:
    a bare weight column corresponds to conditioner(e_i), i.e. an input
    with shape=[0,0,0] -- not a one-hot at all, not a valid circle, not a
    valid ANYTHING, a vector no real image was ever conditioned on.
    Every vector used to build THIS basis, by contrast, is a genuine,
    valid circle conditioning vector -- so alignment against it reflects
    a realistic, in-context movement (one real circle to another), not
    movement along a direction the model never saw a valid input for.
    """
    names = feature_names
    base_vec = torch.tensor(base_vec_list, dtype=torch.float32)
    with torch.no_grad():
        base_out = conditioner(base_vec.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        cols = []
        for name in names:
            probe_vec = torch.tensor(
                circle_probe_vector(name, 1.0, base_vec_list, cond_input_dim), dtype=torch.float32
            )
            probe_out = conditioner(probe_vec.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            cols.append(probe_out - base_out)
    return np.stack(cols, axis=1)   # (64, n_features)


# ---------------------------------------------------------------------------
# Gradient computation -- uses collect_gradients.py's own functions verbatim
# ---------------------------------------------------------------------------

def compute_pairwise_gradient(
    unet, conditioner, ddpm_scheduler,
    image_a: torch.Tensor, vec_a: torch.Tensor, image_b: torch.Tensor,
    device, timestep: int, n_samples: int = 1, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exactly collect_gradients.py's per-pair recipe (ELROND Eq. 1-4):
    noise image_a to `timestep`, ONE model call for the closed-form x0
    estimate conditioned on vec_a (gradient tracked on cond_out), loss
    against the RAW image_b, backprop. No multi-step DDIM chain -- no
    ddim_scheduler needed at all, since there's no trajectory to run
    (this is the direct consequence of switching collect_gradients.py's
    own recipe from a multi-step loop to ELROND's actual one-step
    formula; see collect_gradients.py's module docstring for the full
    story of why).
    Averages over n_samples independent noise draws (collect_gradients.py
    itself does NOT do this per-pair -- it relies on volume across many
    DIFFERENT pairs for noise reduction instead -- so n_samples=1 here
    reproduces exactly one of its pairs; n_samples>1 is purely this
    script's own choice to get a less noisy read on any ONE controlled
    pair, since we don't have thousands of pairs to average over here).
    Returns (mean_gradient, std_gradient), both (64,).
    """
    image_a = image_a.unsqueeze(0).to(device)
    image_b = image_b.unsqueeze(0).to(device)
    grads = []
    for i in range(n_samples):
        noisy_a = noise_image(image_a, timestep, ddpm_scheduler, device, seed=seed + i)
        cond_out = conditioner(vec_a.unsqueeze(0).to(device)).detach().requires_grad_(True)
        x0_pred = predict_x0(unet, cond_out, noisy_a, timestep, ddpm_scheduler)
        loss = F.mse_loss(x0_pred.float(), image_b.float())
        grad = torch.autograd.grad(loss, cond_out)[0]
        grads.append(grad.detach().cpu().numpy().reshape(-1))
    return np.mean(grads, axis=0), np.std(grads, axis=0)


def alignment_scores(direction: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine similarity between `direction` (64,) and every column of W
    (64, n_features)."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return d @ W_norm


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1D arrays, pure numpy
    (no scipy dependency). Returns nan if either array has zero variance
    (e.g. every pair happened to give the exact same value for this
    feature -- degenerate, no correlation is well-defined there)."""
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def trapezoidal_integrate(y: np.ndarray, axis: int = 0) -> np.ndarray:
    """Trapezoidal-rule integration of `y` along `axis`, implemented
    directly rather than via numpy's trapz/trapezoid (the function was
    renamed trapz -> trapezoid in numpy 2.0, so depending on either name
    breaks on some environment's numpy version).
    Sample spacing (dx) is deliberately left implicit/unit -- irrelevant
    here, since every caller immediately feeds the result through
    alignment_scores, which L2-normalizes it to a unit vector; any
    constant positive rescaling of dx has zero effect on that direction."""
    y = np.asarray(y)
    n = y.shape[axis]
    weights = np.ones(n)
    weights[0] = 0.5
    weights[-1] = 0.5
    y_moved = np.moveaxis(y, axis, 0)
    weighted = y_moved * weights.reshape([-1] + [1] * (y_moved.ndim - 1))
    return weighted.sum(axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_case_at_timestep(
    title: str, images: List["PIL.Image.Image"], image_labels: List[str],
    grad_align: np.ndarray, expected_align: np.ndarray, feature_names: list,
    highlight_features: Optional[List[str]], out_path: Path,
    image_vector_texts: Optional[List[str]] = None,
) -> None:
    """One figure: images on the left; on the right, TWO aligned bar
    charts sharing the same x-axis (ALL features, not just the one this
    pair varies -- so you can directly compare "is the gradient actually
    bigger in the intended direction than in the others"):
      1. -gradient's alignment (`grad_align` here is ALREADY the negated
         gradient -- see main(), where it's computed directly as
         alignment_scores(-grad_a, W) rather than negated for display)
      2. the TRUE reference: raw conditioning-vector difference
         (vec_b[feature] - vec_a[feature]), the same quantity the
         correlation analysis below calls "true_diffs" -- NOT a
         conditioner-embedding-space projection
    The feature(s) this pair was designed to vary are colored orange;
    everything else is blue, so the "intended" bar is easy to spot
    without losing the full comparison.
    `image_vector_texts`, if given (one string per image, e.g. the exact
    conditioning vector used to render that circle), is printed directly
    below that image -- for cases where the image's own numeric values
    are the point (e.g. two independently random ground-truth circles),
    not just its qualitative label."""
    grad_align = np.asarray(grad_align)
    expected_align = np.asarray(expected_align)
    colors = ["tab:orange" if (highlight_features and f in highlight_features) else "tab:blue"
              for f in feature_names]

    n_img = len(images)
    n_rows = max(n_img, 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.4 * n_rows), squeeze=False)

    for i in range(n_img):
        axes[i][0].imshow(images[i])
        axes[i][0].set_title(image_labels[i], fontsize=10)
        axes[i][0].axis("off")
        if image_vector_texts:
            axes[i][0].text(0.5, -0.12, image_vector_texts[i], transform=axes[i][0].transAxes,
                            ha="center", va="top", fontsize=11, family="monospace")
    for i in range(n_img, n_rows):
        axes[i][0].axis("off")

    ax_neg = axes[0][1]
    ax_neg.bar(feature_names, grad_align, color=colors)
    ax_neg.axhline(0, color="black", linewidth=0.5)
    ax_neg.set_ylim(-1, 1)
    ax_neg.set_ylabel("cosine alignment")
    ax_neg.set_title("-gradient  (direction that would DECREASE the loss)", fontsize=9)
    ax_neg.tick_params(axis="x", rotation=45)

    ax2 = axes[1][1]
    ax2.bar(feature_names, expected_align, color=colors)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("raw conditioning diff")
    ax2.set_title("true direction: raw conditioning diff (vec_b - vec_a)  [reference]", fontsize=9)
    ax2.tick_params(axis="x", rotation=45)

    for i in range(2, n_rows):
        axes[i][1].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alignment_vs_timestep(
    title: str, timesteps: list, curves: dict, expected_align: np.ndarray, feature_names: list,
    highlight_features: Optional[List[str]], images: List["PIL.Image.Image"], image_labels: List[str],
    out_path: Path, image_vector_texts: Optional[List[str]] = None,
) -> None:
    """Images on the left (this pair's two circles); on the right, ALL
    features' -gradient alignment plotted against timestep (`curves`
    already holds -gradient values -- see main()) -- checks
    collect_gradients.py's own stated hypothesis that high t captures
    coarse/shape structure and low t captures fine detail, and lets you
    compare the intended feature's curve against every other feature's,
    not just see it in isolation. The feature(s) this pair varies are
    drawn thicker and full-opacity, everything else thin and faded so
    the intended curve still stands out in what is otherwise a fairly
    dense plot.
    The constant true-direction reference is deliberately NOT drawn here
    (it doesn't vary with timestep, and is already shown clearly in the
    per-timestep bar chart and the sweep-average plot).
    `image_vector_texts`, if given, printed directly below each image --
    see plot_case_at_timestep's own docstring for why."""
    n_img = max(len(images), 1)

    fig = plt.figure(figsize=(11, 3.6 * n_img))
    gs = fig.add_gridspec(n_img, 2)

    for i, (img, label) in enumerate(zip(images, image_labels)):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(img)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis("off")
        if image_vector_texts:
            ax_img.text(0.5, -0.12, image_vector_texts[i], transform=ax_img.transAxes,
                       ha="center", va="top", fontsize=11, family="monospace")

    ax = fig.add_subplot(gs[:, 1])   # spans the full height, alongside the stacked images
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(feature_names):
        is_hl = bool(highlight_features and name in highlight_features)
        color = color_cycle[i % len(color_cycle)]
        vals = np.asarray(curves[name])
        ax.plot(timesteps, vals, marker="o", linestyle="-", color=color,
               linewidth=2.5 if is_hl else 1, alpha=1.0 if is_hl else 0.35,
               label=name)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("cosine alignment of -gradient with feature direction")
    ax.set_ylim(-1, 1)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_xaxis()   # low t (fine detail) on the right, matching diffusion convention
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sweep_average_bar(
    title: str, sweep_mean: dict, sweep_std: dict, expected_align: np.ndarray, feature_names: list,
    highlight_features: Optional[List[str]], images: List["PIL.Image.Image"], image_labels: List[str],
    out_path: Path, image_vector_texts: Optional[List[str]] = None,
) -> None:
    """Images on the left; on the right, ALL features, each with TWO
    grouped bars: -gradient (mean +/- std across the sweep -- `sweep_mean`
    already holds -gradient values, see main()) and the true reference --
    the raw conditioning-vector difference (vec_b[feature] -
    vec_a[feature]), the same quantity the correlation analysis calls
    "true_diffs" -- so you can directly compare the intended feature's
    bars against every other feature's, not just see it in isolation. The
    feature(s) this pair varies get a light orange background band
    behind their whole group, so the "intended" one is still easy to
    spot in a wider chart.
    A gradient bar that's both tall AND has a small error bar relative to
    its height is a real, timestep-independent signal; an error bar as
    big as the bar itself means don't trust that number.
    `image_vector_texts`, if given, printed directly below each image --
    see plot_case_at_timestep's own docstring for why."""
    means = np.array([sweep_mean[n] for n in feature_names])
    stds = np.array([sweep_std[n] for n in feature_names])
    refs = np.asarray(expected_align)

    n_img = max(len(images), 1)
    fig = plt.figure(figsize=(max(9, len(feature_names) * 1.1), 3.6 * n_img))
    gs = fig.add_gridspec(n_img, 2)

    for i, (img, label) in enumerate(zip(images, image_labels)):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(img)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis("off")
        if image_vector_texts:
            ax_img.text(0.5, -0.12, image_vector_texts[i], transform=ax_img.transAxes,
                       ha="center", va="top", fontsize=11, family="monospace")

    ax = fig.add_subplot(gs[:, 1])
    x = np.arange(len(feature_names))
    for i, name in enumerate(feature_names):
        if highlight_features and name in highlight_features:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.15, zorder=0)
    ax.bar(x - 0.18, means, width=0.36, yerr=stds, color="tab:purple", capsize=4, label="-gradient (mean +/- std)")
    ax.bar(x + 0.18, refs, width=0.36, color="tab:gray", label="true diff in conditioning (reference)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_xlim(-0.5, len(feature_names) - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("value  (-gradient: cosine alignment;  reference: raw diff)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc_mixed_dz0.1_pos/checkpoint-epoch-0200")
    parser.add_argument("--out_dir", type=str, default="outputs/sanity_checks/det_circles_corr_pos/")
    parser.add_argument("--cond_input_dim", type=int, default=10)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[900, 500, 100],
                        help="Timesteps to check, matching collect_gradients.py's own "
                             "--timesteps convention. A representative spread (high/mid/low) "
                             "lets you check its stated coarse-vs-fine hypothesis.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Independent noise draws averaged per (case, timestep) -- "
                             "this script's own noise-reduction choice, NOT part of "
                             "collect_gradients.py's actual recipe (which relies on pair "
                             "volume instead). Each draw is now just ONE model call "
                             "(ELROND Eq. 2), cheap compared to the old multi-step "
                             "recipe -- default 3 is a reasonable, generous starting point.")
    parser.add_argument("--sweep_step", type=int, default=5,
                        help="Dense timestep sweep for a robust averaged statistic, in "
                             "addition to the few --timesteps above: every multiple of "
                             "this value up to num_train_timesteps (default step 5, "
                             "num_train_timesteps=1000 -> 200 timesteps). A couple of "
                             "hand-picked timesteps can look meaningful by pure chance; "
                             "averaging alignment across a full dense sweep is a much more "
                             "reliable signal that it's real. Each sweep point is now a "
                             "single model call (ELROND Eq. 2), so this dense a sweep is "
                             "cheap; raise this (e.g. 50 -> 20 timesteps) to cut cost "
                             "further, or set 0 to disable the sweep entirely.")
    parser.add_argument("--sweep_n_samples", type=int, default=1,
                        help="Noise draws per sweep timestep. Default 1 since averaging "
                             "over ~100 different timesteps already reduces variance a "
                             "lot on its own -- unlike --n_samples above, which has to "
                             "do all its averaging at a single fixed timestep.")
    parser.add_argument("--n_pairs", type=int, default=5,
                        help="Number of independent random circle pairs to test. Each "
                             "pair is two ground-truth rendered circles with fully "
                             "random, deterministic (seeded) continuous features -- see "
                             "deterministic_random_vec -- compared against each other, "
                             "with the gradient's cond_out computed as if from the fixed "
                             "neutral prompt (0.5 everywhere), not either circle's own "
                             "actual random values (see the NOTE printed at startup).")
    parser.add_argument("--n_correlation_pairs", type=int, default=50,
                        help="Number of ADDITIONAL random pairs (separate from --n_pairs, "
                             "which is just for illustration) used for a proper statistical "
                             "check: across this many pairs, does the gradient's cosine "
                             "alignment with each basis direction actually correlate with "
                             "how much that feature's TRUE value differs between the two "
                             "circles? A real Pearson correlation coefficient per feature "
                             "per timestep, not just a few example bar charts. Set 0 to "
                             "skip this section entirely.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(cond_input_dim=args.cond_input_dim)
    ckpt = Path(args.checkpoint)
    unet_dir = ckpt / "unet_ema"
    cond_path = ckpt / "conditioner.pt"
    problems = []
    if not ckpt.is_dir():
        problems.append(f"--checkpoint directory does not exist: {ckpt.resolve()}")
    elif not unet_dir.is_dir():
        problems.append(f"no 'unet_ema' subdirectory found in {ckpt.resolve()} "
                        f"(looked for {unet_dir.resolve()})")
    elif not (unet_dir / "config.json").is_file():
        problems.append(f"{unet_dir.resolve()} exists but has no config.json in it "
                        f"(checkpoint save may not have finished, or this isn't a "
                        f"diffusers model directory)")
    if not cond_path.is_file():
        problems.append(f"no conditioner.pt found at {cond_path.resolve()}")
    if problems:
        # Fail fast with a clear, local message. Without this check,
        # diffusers' from_pretrained silently falls back to treating the
        # path as a HuggingFace Hub repo ID once it can't find a local
        # config.json, which then fails on an offline compute node with a
        # confusing "couldn't connect to huggingface.co" network error
        # that has nothing to do with the actual (local path) problem.
        raise SystemExit("--checkpoint doesn't look right:\n  " + "\n  ".join(problems) +
                         f"\n\nCurrent working directory: {Path.cwd()}"
                         f"\n(if --checkpoint is a relative path, it's resolved against this)")
    print(f"Loading checkpoint from {ckpt} ...")
    unet = UNet2DConditionModel.from_pretrained(unet_dir).to(device)
    unet.eval()
    conditioner = ShapeConditioningEncoder(
        cfg.cond_input_dim, cfg.cond_hidden_dim, cfg.cond_output_dim,
    ).to(device)
    conditioner.load_state_dict(
        torch.load(cond_path, map_location=device, weights_only=True)
    )
    conditioner.eval()

    feature_names = continuous_feature_names(args.cond_input_dim)
    if not feature_names:
        raise SystemExit(f"No continuous features available for cond_input_dim={args.cond_input_dim}")

    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )

    # One fixed neutral base circle -- used ONLY for cond_out (the actual
    # gradient anchor) and the basis/expected-direction comparison, NOT
    # for rendering either of the two circles being compared below.
    base_vec_list = neutral_shape_prompt(shape_id=2, cond_input_dim=args.cond_input_dim)  # 2 = circle
    vec_a = torch.tensor(base_vec_list, dtype=torch.float32)
    W = build_circle_basis_matrix(conditioner, feature_names, args.cond_input_dim, device, base_vec_list)
    print(f"Circle vector subspace basis ({len(feature_names)} dims): {feature_names}")
    print(f"\nBase prompt (neutral circle, used ONLY for cond_out) = {base_vec_list}")
    print("NOTE ON THIS VARIANT: each pair below is TWO ground-truth rendered circles "
          "with their OWN fully random, deterministic continuous values (printed under "
          "each image) -- NOT generated from the neutral prompt above. The gradient's "
          "cond_out is still computed AS IF from that neutral prompt, matching every "
          "other case in this script -- it is NOT the conditioning either circle was "
          "actually rendered with. This is a deliberate mismatch: rendering a concrete, "
          "reproducible circle needs specific numeric values, which the neutral prompt "
          "alone doesn't provide -- so the two circles get their own random values for "
          "rendering, while the gradient stays anchored to the same neutral point used "
          "everywhere else in this script.")
    print("NOTE: every print/plot below reports -gradient (i.e. alignment_scores(-grad_a, W)), "
          "the direction that would DECREASE the loss -- not the raw gradient itself, which "
          "points toward increasing the loss (away from matching image B). A gradient that's "
          "actually tracking real semantics should show POSITIVE alignment between -gradient "
          "and the true (B-A) direction.")

    summary = []
    for i in range(args.n_pairs):
        case = {"name": f"pair{i}", "label": f"pair {i}", "highlight": []}   # no single
        # "intended" feature anymore -- both circles vary randomly across ALL features.

        vec_a_random_list = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                                      seed=args.seed + 2 * i)
        vec_b_random_list = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                                      seed=args.seed + 2 * i + 1)
        img_a_t, img_a_pil = render_image(vec_a_random_list, seed=args.seed)
        img_b_t, img_b_pil = render_image(vec_b_random_list, seed=args.seed)
        img_a_pil.save(out_dir / f"{case['name']}_circleA.png")
        img_b_pil.save(out_dir / f"{case['name']}_circleB.png")
        vector_texts = [
            format_vector_caption(vec_a_random_list, feature_names),
            format_vector_caption(vec_b_random_list, feature_names),
        ]
        print(f"\n=== {case['label']}: circle A {vec_a_random_list} vs circle B {vec_b_random_list} ===")

        # The TRUE reference: the simple, raw difference between circle B's
        # and circle A's own actual conditioning-vector VALUES, feature by
        # feature (vec_b[feature] - vec_a[feature]) -- the exact same
        # quantity used as "true_diffs" in the correlation analysis below,
        # NOT a conditioner-embedding-space projection (that's what this
        # used to be: conditioner(B) - conditioner(A), then projected onto
        # W). This answers "how much did this feature's raw value actually
        # change between the two circles" directly, with no conditioner
        # network involved at all, and no dependence on the basis W --
        # matching what the correlation section actually correlates
        # against, not a differently-derived quantity that happens to look
        # similar.
        expected_align = np.array(vec_b_random_list[N_SHAPE_DIMS:]) - np.array(vec_a_random_list[N_SHAPE_DIMS:])

        curves = {name: [] for name in feature_names}
        for t in args.timesteps:
            grad_a, std_a = compute_pairwise_gradient(
                unet, conditioner, ddpm, img_a_t, vec_a, img_b_t,
                device, timestep=t, n_samples=args.n_samples, seed=args.seed,
            )
            grad_align = alignment_scores(-grad_a, W)
            noise_ratio = float(np.mean(std_a) / (np.mean(np.abs(grad_a)) + 1e-8))

            for name, val in zip(feature_names, grad_align):
                curves[name].append(float(val))

            print(f"  t={t:>4}  noise_ratio={noise_ratio:.2f}")
            print("    scalar products (-gradient . basis vector) per feature:")
            print("      " + "   ".join(f"{n}={v:+.3f}" for n, v in zip(feature_names, grad_align)))
            if case["highlight"]:
                signed = float(np.mean([v for n, v in zip(feature_names, grad_align) if n in case["highlight"]]))
                abs_changed = float(np.mean([abs(v) for n, v in zip(feature_names, grad_align) if n in case["highlight"]]))
                abs_other = float(np.mean([abs(v) for n, v in zip(feature_names, grad_align) if n not in case["highlight"]]))
                print(f"    signed alignment of -gradient on changed feature(s): {signed:+.3f}  "
                      f"(want POSITIVE -- see note above)")
                print(f"    |alignment| on changed feature(s): {abs_changed:.3f}   vs unrelated: {abs_other:.3f}  "
                      f"(want changed > unrelated)")

            plot_case_at_timestep(
                title=f"circle A  vs  circle B   t={t}  (n_samples={args.n_samples})",
                images=[img_a_pil, img_b_pil], image_labels=["circle A", "circle B"],
                grad_align=grad_align, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                out_path=out_dir / f"{case['name']}_t{t}.png",
                image_vector_texts=vector_texts,
            )

        plot_alignment_vs_timestep(
            title=f"{case['label']}: alignment vs timestep ({len(args.timesteps)} hand-picked points)",
            timesteps=args.timesteps, curves=curves, expected_align=expected_align,
            feature_names=feature_names, highlight_features=case["highlight"],
            images=[img_a_pil, img_b_pil], image_labels=["circle A", "circle B"],
            out_path=out_dir / f"{case['name']}_vs_timestep.png",
            image_vector_texts=vector_texts,
        )
        print(f"  Saved per-timestep plots and {case['name']}_vs_timestep.png -> {out_dir}/")

        # --- dense timestep sweep: a handful of hand-picked timesteps can
        # look meaningful (or meaningless) by pure chance. Averaging over a
        # full sweep (every --sweep_step up to num_train_timesteps -- 100
        # timesteps at the default step of 10) is a much more reliable
        # summary statistic than trusting any one or two points.
        sweep_result = None
        if args.sweep_step > 0:
            # valid timestep indices are 0..num_train_timesteps-1 (alphas_cumprod
            # has exactly num_train_timesteps entries) -- starting the sweep at 0
            # rather than at args.sweep_step also gives exactly 100 timesteps at
            # the default step of 10 for num_train_timesteps=1000, as requested.
            sweep_timesteps = list(range(0, cfg.num_train_timesteps, args.sweep_step))
            print(f"  Running dense sweep: {len(sweep_timesteps)} timesteps "
                  f"({sweep_timesteps[0]}..{sweep_timesteps[-1]}, step {args.sweep_step}), "
                  f"sweep_n_samples={args.sweep_n_samples} ...")
            sweep_curves = {name: [] for name in feature_names}
            for t in sweep_timesteps:
                grad_sweep, _ = compute_pairwise_gradient(
                    unet, conditioner, ddpm, img_a_t, vec_a, img_b_t,
                    device, timestep=t, n_samples=args.sweep_n_samples, seed=args.seed,
                )
                sweep_align = alignment_scores(-grad_sweep, W)
                for name, val in zip(feature_names, sweep_align):
                    sweep_curves[name].append(float(val))

            sweep_mean = {name: float(np.mean(vals)) for name, vals in sweep_curves.items()}
            sweep_std = {name: float(np.std(vals)) for name, vals in sweep_curves.items()}

            print(f"  [avg over {len(sweep_timesteps)} timesteps] scalar products (-gradient . basis vector):")
            print("    " + "   ".join(f"{n}={sweep_mean[n]:+.3f}(+/-{sweep_std[n]:.3f})" for n in feature_names))
            if case["highlight"]:
                signed_sweep = float(np.mean([sweep_mean[n] for n in case["highlight"]]))
                abs_sweep_changed = float(np.mean([abs(sweep_mean[n]) for n in case["highlight"]]))
                abs_sweep_other = float(np.mean([abs(sweep_mean[n]) for n in feature_names if n not in case["highlight"]]))
                print(f"  [avg over {len(sweep_timesteps)} timesteps] signed alignment of -gradient on "
                      f"changed feature(s): {signed_sweep:+.3f}  (want POSITIVE -- see note above)")
                print(f"  [avg over {len(sweep_timesteps)} timesteps] |alignment| changed feature(s): "
                      f"{abs_sweep_changed:.3f}   vs unrelated: {abs_sweep_other:.3f}  (want changed > unrelated)")

            plot_alignment_vs_timestep(
                title=f"{case['label']}: alignment vs timestep ({len(sweep_timesteps)}-timestep dense sweep)",
                timesteps=sweep_timesteps, curves=sweep_curves, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                images=[img_a_pil, img_b_pil], image_labels=["circle A", "circle B"],
                out_path=out_dir / f"{case['name']}_sweep_vs_timestep.png",
                image_vector_texts=vector_texts,
            )
            plot_sweep_average_bar(
                title=f"{case['label']}: alignment averaged over {len(sweep_timesteps)} timesteps",
                sweep_mean=sweep_mean, sweep_std=sweep_std, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                images=[img_a_pil, img_b_pil], image_labels=["circle A", "circle B"],
                out_path=out_dir / f"{case['name']}_sweep_average.png",
                image_vector_texts=vector_texts,
            )
            print(f"  Saved -> {case['name']}_sweep_vs_timestep.png, {case['name']}_sweep_average.png")

            sweep_result = {"sweep_timesteps": sweep_timesteps, "sweep_mean": sweep_mean, "sweep_std": sweep_std}

        summary.append({"case": case["name"], "label": case["label"], "highlight": case["highlight"],
                        "vec_a_random": vec_a_random_list, "vec_b_random": vec_b_random_list,
                        "timesteps": args.timesteps, "curves": curves,
                        "sweep": sweep_result,
                        "expected_align": expected_align.tolist(), "feature_names": feature_names})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved illustrative-pair summary -> {out_dir / 'summary.json'}")

    # ------------------------------------------------------------------
    # Correlation analysis: across MANY random pairs (not just the
    # --n_pairs illustrated above), does the gradient's cosine alignment
    # with each basis direction actually correlate with how much that
    # feature's TRUE value differs between the two circles? A real
    # statistic, not just a handful of example bar charts. Uses fresh
    # seeds well clear of the illustrative pairs' own, so nothing overlaps.
    # ------------------------------------------------------------------
    if args.n_correlation_pairs > 0:
        print(f"\n{'='*60}")
        print(f"Correlation analysis: {args.n_correlation_pairs} random pairs")
        print(f"{'='*60}")

        grad_dots = {t: {name: [] for name in feature_names} for t in args.timesteps}
        true_diffs = {name: [] for name in feature_names}

        corr_seed_offset = args.n_pairs * 2 + 10_000   # clear of the illustrative pairs' seeds
        for k in range(args.n_correlation_pairs):
            vec_a_k = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                               seed=args.seed + corr_seed_offset + 2 * k)
            vec_b_k = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                               seed=args.seed + corr_seed_offset + 2 * k + 1)
            img_a_k, _ = render_image(vec_a_k, seed=args.seed)
            img_b_k, _ = render_image(vec_b_k, seed=args.seed)

            for name_idx, name in enumerate(feature_names):
                true_diffs[name].append(vec_b_k[N_SHAPE_DIMS + name_idx] - vec_a_k[N_SHAPE_DIMS + name_idx])

            for t in args.timesteps:
                grad_k, _ = compute_pairwise_gradient(
                    unet, conditioner, ddpm, img_a_k, vec_a, img_b_k,
                    device, timestep=t, n_samples=1, seed=args.seed,
                )
                dots_k = alignment_scores(-grad_k, W)
                for name, val in zip(feature_names, dots_k):
                    grad_dots[t][name].append(float(val))

            if (k + 1) % max(1, args.n_correlation_pairs // 10) == 0:
                print(f"  {k + 1}/{args.n_correlation_pairs} pairs done")

        correlation_summary = {}
        for t in args.timesteps:
            print(f"\n  t={t}: correlation(-gradient . basis direction, true feature difference)")
            correlation_summary[t] = {}
            for name in feature_names:
                r = pearson_correlation(np.array(grad_dots[t][name]), np.array(true_diffs[name]))
                correlation_summary[t][name] = r
                print(f"    {name}: r={r:+.3f}")

            n_feat = len(feature_names)
            fig, axes = plt.subplots(1, n_feat, figsize=(3.2 * n_feat, 3.4), squeeze=False)
            for j, name in enumerate(feature_names):
                ax = axes[0][j]
                x = np.array(true_diffs[name])
                y = np.array(grad_dots[t][name])
                ax.scatter(x, y, s=14, alpha=0.6)
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
                ax.set_xlabel(f"true diff in {name}")
                ax.set_ylabel("-gradient . basis (cosine)")
                ax.set_title(f"{name}: r={correlation_summary[t][name]:+.3f}", fontsize=10)
            fig.suptitle(f"Gradient vs. ground truth across {args.n_correlation_pairs} random pairs, t={t}")
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            scatter_path = out_dir / f"correlation_scatter_t{t}.png"
            fig.savefig(scatter_path, dpi=150)
            plt.close(fig)
            print(f"  Saved -> {scatter_path}")

        # ------------------------------------------------------------
        # ALSO: one correlation per feature using the gradient's
        # alignment computed from the RAW gradient INTEGRATED across a
        # dense timestep sweep -- "all timesteps at once" -- rather than
        # averaging the already-cosine-normalized alignment scores from
        # each timestep separately. This is a real methodological
        # difference, not just a rewording: alignment_scores L2-
        # normalizes its input to a unit vector, so averaging PER-
        # TIMESTEP alignment scores treats every timestep's DIRECTION
        # equally regardless of that timestep's own gradient MAGNITUDE --
        # a timestep where the raw gradient is huge (strong, informative
        # signal) counts exactly as much as one where it's tiny (mostly
        # noise). Integrating the RAW gradient vectors first (via
        # trapezoidal_integrate), and only normalizing ONCE at the end,
        # lets high-magnitude timesteps naturally dominate the aggregate
        # direction, which is the more principled way to combine
        # evidence across timesteps of very different informativeness
        # (recall: gradient magnitude at high t can be ~100x that at low
        # t, from Eq. 2's sigma_t/alpha_t scaling).
        # Reuses the exact same --sweep_step-driven dense sweep already
        # used for the illustrative pairs' sweep-average plots above
        # (default step=5 -> 200 timesteps), so this is literally the
        # same 200 timesteps "used in plots", not a separate,
        # differently-sized sweep invented just for this section.
        # ------------------------------------------------------------
        if args.sweep_step > 0:
            dense_sweep_timesteps = list(range(0, cfg.num_train_timesteps, args.sweep_step))
            n_dense = len(dense_sweep_timesteps)
            print(f"\n  Computing gradient integrated across {n_dense} timesteps (same dense "
                  f"sweep as the sweep-average plots) for {args.n_correlation_pairs} pairs "
                  f"-- {args.n_correlation_pairs * n_dense} additional model calls total ...")

            integrated_grad_dots = {name: [] for name in feature_names}
            for k in range(args.n_correlation_pairs):
                vec_a_k = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                                   seed=args.seed + corr_seed_offset + 2 * k)
                vec_b_k = deterministic_random_vec(shape_id=2, cond_input_dim=args.cond_input_dim,
                                                   seed=args.seed + corr_seed_offset + 2 * k + 1)
                img_a_k, _ = render_image(vec_a_k, seed=args.seed)
                img_b_k, _ = render_image(vec_b_k, seed=args.seed)

                # Collect RAW (unnormalized) -gradient vectors across the
                # dense sweep -- NOT yet converted to cosine alignment.
                raw_neg_grads = []   # list of (64,) arrays, one per sweep timestep
                for t in dense_sweep_timesteps:
                    grad_k, _ = compute_pairwise_gradient(
                        unet, conditioner, ddpm, img_a_k, vec_a, img_b_k,
                        device, timestep=t, n_samples=1, seed=args.seed,
                    )
                    raw_neg_grads.append(-grad_k)   # sign convention matches -gradient everywhere else
                raw_neg_grads = np.stack(raw_neg_grads, axis=0)   # (n_dense, 64)

                integrated_direction = trapezoidal_integrate(raw_neg_grads, axis=0)   # (64,)
                integrated_align = alignment_scores(integrated_direction, W)          # (n_features,)
                for name, val in zip(feature_names, integrated_align):
                    integrated_grad_dots[name].append(float(val))

                if (k + 1) % max(1, args.n_correlation_pairs // 10) == 0:
                    print(f"    {k + 1}/{args.n_correlation_pairs} pairs done")

            print(f"\n  Correlation using gradient integrated across all {n_dense} timesteps:")
            integrated_correlation = {}
            for name in feature_names:
                r = pearson_correlation(np.array(integrated_grad_dots[name]), np.array(true_diffs[name]))
                integrated_correlation[name] = r
                print(f"    {name}: r={r:+.3f}")

            n_feat = len(feature_names)
            fig, axes = plt.subplots(1, n_feat, figsize=(3.2 * n_feat, 3.4), squeeze=False)
            for j, name in enumerate(feature_names):
                ax = axes[0][j]
                x = np.array(true_diffs[name])
                y = np.array(integrated_grad_dots[name])
                ax.scatter(x, y, s=14, alpha=0.6, color="tab:green")
                ax.axhline(0, color="gray", linewidth=0.5)
                ax.axvline(0, color="gray", linewidth=0.5)
                ax.set_xlabel(f"true diff in {name}")
                ax.set_ylabel(f"-gradient . basis (integrated over {n_dense} t)")
                ax.set_title(f"{name}: r={integrated_correlation[name]:+.3f}", fontsize=10)
            fig.suptitle(f"Gradient (integrated over {n_dense} timesteps) vs ground truth, "
                        f"{args.n_correlation_pairs} pairs")
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            integrated_scatter_path = out_dir / "correlation_scatter_integrated.png"
            fig.savefig(integrated_scatter_path, dpi=150)
            plt.close(fig)
            print(f"  Saved -> {integrated_scatter_path}")

            correlation_summary["integrated_over_dense_sweep"] = integrated_correlation

        (out_dir / "correlation_summary.json").write_text(json.dumps(correlation_summary, indent=2))
        print(f"\nSaved correlation summary -> {out_dir / 'correlation_summary.json'}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()