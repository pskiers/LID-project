"""
sanity_check_gradients.py
--------------------------
Sanity-checks whether cond_out gradients (as computed by collect_gradients.py)
actually correspond to real semantic differences -- circles only throughout.

Design: for every continuous feature (r, g, b, size, and h_stripe/v_stripe/
grain if cond_input_dim includes them), build a synthetic (ground-truth
rendered) "extreme" circle pushing that ONE feature to 0.0 or 1.0 relative
to the neutral (0.5 everywhere) base circle. For each such (base, extreme)
pair, a FRESH network-generated base image is sampled (full reverse-
diffusion from noise, conditioned on the general/neutral circle prompt --
e.g. for cond_input_dim=7: [0,0,1, 0.5,0.5,0.5, 0.5]) -- a different base
image per pair, not one image reused everywhere, so results aren't an
artifact of one specific noise realization. The gradient itself is
computed using collect_gradients.py's exact recipe -- imported directly
from there (noise_image, denoise_with_grad), not reimplemented:
  1. Noise the base image to timestep t
  2. Denoise with UNet conditioned on the base's OWN prompt -- grad
     tracked on cond_out, through the FULL multi-step DDIM chain
  3. Loss = MSE(denoised_base, extreme_image)
  4. Backprop -> gradient w.r.t. cond_out
The extreme's own conditioning vector is never fed to the model -- it
only supplies the pixel target for the loss, same as collect_gradients.py's
own average_anchor mode (one fixed anchor prompt, many different targets).

Comparisons are against the "circle vector subspace" basis, NOT raw
columns of conditioner.proj.weight: a bare weight column corresponds to
conditioner(e_i), i.e. an input with shape=[0,0,0] -- not a one-hot, not
a valid circle, not a valid anything, a vector no real image was ever
conditioned on. Instead, build_circle_basis_matrix defines each feature's
basis direction as conditioner(valid circle with that feature=1) -
conditioner(base circle) -- always a real, valid circle conditioning
vector on both ends, so alignment against it reflects a realistic,
in-context movement.

Sign convention: every print/plot reports -gradient (alignment_scores(
-grad_a, W)), the direction that would DECREASE the loss -- not the raw
gradient itself, which points toward INCREASING the loss (away from
matching the extreme image). A gradient that's actually tracking real
semantics should show POSITIVE alignment between -gradient and the true
conditioner(extreme) - conditioner(base) direction.

Every plot comparing a pair shows ALL features (not just the one that
pair varies), so you can directly compare the intended feature's
alignment against every other feature's -- with the feature(s) this pair
actually varies visually distinguished (orange bars / shaded band / bold
line) so the "intended" one is still easy to spot in the full comparison.

For every pair, prints the actual scalar products (cosine alignment) of
the gradient against the circle-subspace basis -- both at each of
--timesteps individually, and averaged over a dense --sweep_step sweep
(100 timesteps by default) for a statistic that isn't sensitive to which
few timesteps happened to be hand-picked.

Usage:
    python sanity_check_gradients.py \\
        --checkpoint outputs/checkpoints/outputs_64_acc_no_text/checkpoint-epoch-0200 \\
        --out_dir outputs/sanity_checks --cond_input_dim 7 \\
        --timesteps 900 500 100 --n_samples 5

Must be run from the same directory as collect_gradients.py (or with it
on the Python path), since it imports noise_image/denoise_with_grad from
there directly rather than reimplementing them -- this is deliberate, so
the sanity check can never silently drift out of sync with your actual
gradient collection recipe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel

from configs.config_64 import Config64 as Config
from models.conditioner import ShapeConditioningEncoder
from collect_gradients import noise_image, denoise_with_grad, FEATURE_NAMES_FULL, N_SHAPE_DIMS

try:
    from dataset_64_update import SampleVector64, ImageGenerator64
except ImportError:
    from dataset_64 import SampleVector64, ImageGenerator64


# ---------------------------------------------------------------------------
# Rendering ground-truth images from a raw conditioning vector
# ---------------------------------------------------------------------------

NEUTRAL_FULL_TAIL = [0.0, 0.0, 1.0,   # shape one-hot (only used if cond_vec omits shape entirely)
                     0.5, 0.5, 0.5,   # r, g, b
                     0.5,             # size
                     0.0, 0.0, 0.0]   # h_stripe, v_stripe, grain


def render_image(cond_vec: List[float], seed: int = 0) -> Tuple[torch.Tensor, "PIL.Image.Image"]:
    """cond_vec: list of length cond_input_dim, standard prefix order.
    Trailing dims not provided are padded with neutral defaults (0.5 for
    r/g/b/size, 0.0 i.e. no texture for h_stripe/v_stripe/grain).
    Returns (image_tensor[3,64,64] in [-1,1], raw_PIL_image)."""
    full = list(cond_vec) + NEUTRAL_FULL_TAIL[len(cond_vec):]
    vec = SampleVector64.from_list(full)
    img = ImageGenerator64().generate(vec, seed=seed)
    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    normalized = arr * 2.0 - 1.0   # matches collect_gradients.py's own [0,1] -> [-1,1] convention exactly
    return normalized, img


def zero_shape_prompt(shape_id: int, cond_input_dim: int) -> list:
    """The all-zero baseline prompt: shape one-hot fixed, EVERY continuous
    feature at exactly 0.0 -- e.g. for shape_id=2 (circle), cond_input_dim=7:
    [0,0,1, 0,0,0, 0]. This is the anchor circle: gradients are always
    computed with respect to THIS prompt's conditioning, and every extreme
    case is this same vector with exactly one feature flipped to 1.0 --
    as interpretable a contrast as this feature space allows: "nothing
    is set" vs. "exactly one thing is set to its maximum"."""
    one_hot = [0.0, 0.0, 0.0]
    one_hot[shape_id] = 1.0
    return one_hot[:min(3, cond_input_dim)] + [0.0] * (cond_input_dim - N_SHAPE_DIMS)


def continuous_feature_names(cond_input_dim: int) -> list:
    """The continuous (non-shape) feature names for this cond_input_dim --
    r,g,b,size,[h_stripe,v_stripe,grain] as applicable. Shape is excluded:
    every comparison in this script holds shape=circle fixed (per request:
    circles only), so there's no "is_circ direction" to speak of when
    shape never actually varies."""
    return FEATURE_NAMES_FULL[N_SHAPE_DIMS:cond_input_dim]


def circle_probe_vector(feature_name: str, value: float, cond_input_dim: int) -> list:
    """All-zero baseline circle prompt (shape=circle, every continuous
    feature at exactly 0.0) with ONE named continuous feature overridden
    to `value` -- e.g. circle_probe_vector('size', 1.0, 7) ->
    [0,0,1, 0,0,0, 1.0]."""
    names = continuous_feature_names(cond_input_dim)
    vec = [0.0, 0.0, 1.0] + [0.0] * len(names)
    vec[N_SHAPE_DIMS + names.index(feature_name)] = value
    return vec


def build_circle_basis_matrix(conditioner, feature_names: list, cond_input_dim: int, device) -> np.ndarray:
    """The 'circle vector subspace' basis, as a (64, n_features) matrix
    aligned with `feature_names`: column i is the direction in cond_out
    space from the ALL-ZERO base circle (shape=circle, everything else
    0.0) to a circle with feature_names[i] maxed to 1.0 -- e.g.
    conditioner([0,0,1, 1,0,0,0]) - conditioner([0,0,1, 0,0,0,0])
    for the 'r' column. Matches zero_shape_prompt's baseline exactly, so
    this basis reflects the same anchor the actual gradients are computed
    with respect to.

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
    base_vec = torch.tensor([0.0, 0.0, 1.0] + [0.0] * len(names), dtype=torch.float32)
    with torch.no_grad():
        base_out = conditioner(base_vec.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        cols = []
        for name in names:
            probe_vec = torch.tensor(circle_probe_vector(name, 1.0, cond_input_dim), dtype=torch.float32)
            probe_out = conditioner(probe_vec.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            cols.append(probe_out - base_out)
    return np.stack(cols, axis=1)   # (64, n_features)


def _predict_x0(unet, z_t: torch.Tensor, t: int, cond_out: torch.Tensor, scheduler) -> torch.Tensor:
    """The model's current estimate of the final clean image, given the
    current noisy state z_t at timestep t -- same math as
    visualize_x0.py's predict_x0: x0_pred = (z_t - sigma_t*eps) / alpha_t.
    Does NOT alter the denoising trajectory itself, purely a snapshot."""
    t_tensor = torch.tensor([t], device=z_t.device)
    noise_pred = unet(z_t, t_tensor, encoder_hidden_states=cond_out).sample
    alpha_prod = scheduler.alphas_cumprod[t]
    alpha_t = alpha_prod ** 0.5
    sigma_t = (1 - alpha_prod) ** 0.5
    return ((z_t - sigma_t * noise_pred) / alpha_t).clamp(-1, 1)


@torch.no_grad()
def generate_sample(unet, conditioner, ddim_scheduler, vec: torch.Tensor, device, seed: int = 0,
                     x0_snapshot_interval: int = 100,
                     ) -> Tuple[torch.Tensor, "PIL.Image.Image", dict]:
    """Full reverse-diffusion sample generation from pure noise,
    conditioned on `vec` -- an actual NETWORK-GENERATED image (what the
    model itself would draw for this prompt), as distinct from
    render_image()'s ground-truth synthetic render.

    Also snapshots the model's x0 prediction (visualize_x0.py's
    predict_x0 -- what it currently thinks the final clean image will
    look like) at every `x0_snapshot_interval` raw timesteps along this
    SAME generation trajectory -- no extra diffusion runs, just extra
    bookkeeping during the one pass already happening. For each target
    multiple of x0_snapshot_interval, uses the actual schedule entry
    closest to it from above (i.e. the first point in the trajectory
    that has reached or passed that noise level) -- for any --num_steps
    whose resulting schedule spacing (1000/num_steps) evenly divides
    x0_snapshot_interval (e.g. num_steps=10 -> spacing 100, or
    num_steps=20 -> spacing 50, both divide the default interval of 100
    evenly), every target lands on an exact schedule point; for other
    --num_steps values this is the nearest available approximation instead.

    Returns (image_tensor[3,64,64] in [-1,1], PIL image,
    {timestep: PIL image} of x0 snapshots, descending by timestep).
    """
    from torchvision.transforms.functional import to_pil_image
    torch.manual_seed(seed)
    cond_out = conditioner(vec.unsqueeze(0).to(device))
    image = torch.randn(1, 3, 64, 64, device=device)

    schedule = ddim_scheduler.timesteps.tolist()
    max_t = schedule[0]
    remaining_targets = [t for t in range(0, max_t + 1, x0_snapshot_interval) if t <= max_t]
    remaining_targets = sorted(set(remaining_targets), reverse=True)   # descending, e.g. [900,...,100,0]
    x0_snapshots = {}

    for t in schedule:
        while remaining_targets and t <= remaining_targets[0]:
            target = remaining_targets.pop(0)
            x0_pred = _predict_x0(unet, image, t, cond_out, ddim_scheduler)
            x0_snapshots[target] = to_pil_image(((x0_pred.squeeze(0) + 1) / 2).cpu())
        noise_pred = unet(image, t, encoder_hidden_states=cond_out).sample
        image = ddim_scheduler.step(noise_pred, t, image).prev_sample

    final_pil_for_fallback = to_pil_image(((image.clamp(-1, 1).squeeze(0) + 1) / 2).cpu())
    for target in remaining_targets:   # any targets below the schedule's last step (e.g. exact 0)
        x0_snapshots[target] = final_pil_for_fallback

    image = image.clamp(-1, 1)
    pil = to_pil_image(((image.squeeze(0) + 1) / 2).cpu())
    return image.squeeze(0), pil, x0_snapshots


def plot_x0_grid(x0_snapshots: dict, out_path: Path, title: str) -> None:
    """Single-row grid of x0 predictions across the captured timesteps,
    descending left (high noise) to right (clean) -- same idea as
    visualize_x0.py's grid, applied here to one pair's own
    network-generated base image."""
    timesteps_sorted = sorted(x0_snapshots.keys(), reverse=True)
    images = [x0_snapshots[t] for t in timesteps_sorted]
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.3), squeeze=False)
    for ax, img, t in zip(axes[0], images, timesteps_sorted):
        ax.imshow(img)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def extreme_cases(cond_input_dim: int) -> List[dict]:
    """ALL circles -- shape held fixed throughout, per request. For every
    continuous feature, ONE case: that feature flipped to 1.0 relative to
    the ALL-ZERO base circle (zero_shape_prompt) -- the most interpretable
    contrast this feature space allows, "nothing set" vs "exactly one
    thing set to maximum". (The old design also tested flipping to 0.0,
    but with an all-zero base that case is now identical to the base
    itself -- a trivial, degenerate pair -- so it's dropped.)
    Each of these IS one of the circle_basis probe vectors (see
    build_circle_basis_matrix) -- i.e. the "true difference direction"
    for each case is, by construction, (close to) a pure unit vector in
    the basis it's being compared against. That's deliberate: it makes
    the reference panel a clean check that the basis itself is sound,
    and isolates the genuinely open question -- does the GRADIENT show
    the same clean alignment, or does it not."""
    names = continuous_feature_names(cond_input_dim)
    label_map = {"r": "red", "g": "green", "b": "blue", "size": "size",
                "h_stripe": "h-stripe", "v_stripe": "v-stripe", "grain": "grain"}
    cases = []
    for name in names:
        cases.append({
            "name": f"circle_{name}_1",
            "vec": circle_probe_vector(name, 1.0, cond_input_dim),
            "label": f"circle ({label_map.get(name, name)}=1)",
            "highlight": [name],
        })
    return cases


# ---------------------------------------------------------------------------
# Gradient computation -- uses collect_gradients.py's own functions verbatim
# ---------------------------------------------------------------------------

def compute_pairwise_gradient(
    unet, conditioner, ddpm_scheduler, ddim_scheduler,
    image_a: torch.Tensor, vec_a: torch.Tensor, image_b: torch.Tensor,
    device, timestep: int, n_samples: int = 1, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exactly collect_gradients.py's per-pair recipe: noise image_a to
    `timestep`, denoise through the FULL ddim chain conditioned on vec_a
    (gradient tracked on cond_out), loss against image_b, backprop.
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
        denoised_a = denoise_with_grad(unet, cond_out, noisy_a, ddim_scheduler, resume_from_timestep=timestep)
        loss = F.mse_loss(denoised_a.float(), image_b.float())
        grad = torch.autograd.grad(loss, cond_out)[0]
        grads.append(grad.detach().cpu().numpy().reshape(-1))
    return np.mean(grads, axis=0), np.std(grads, axis=0)


def alignment_scores(direction: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Cosine similarity between `direction` (64,) and every column of W
    (64, n_features)."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return d @ W_norm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_case_at_timestep(
    title: str, images: List["PIL.Image.Image"], image_labels: List[str],
    grad_align: np.ndarray, expected_align: np.ndarray, feature_names: list,
    highlight_features: Optional[List[str]], out_path: Path,
) -> None:
    """One figure: images on the left; on the right, TWO aligned bar
    charts sharing the same x-axis (ALL features, not just the one this
    pair varies -- so you can directly compare "is the gradient actually
    bigger in the intended direction than in the others"):
      1. -gradient's alignment (`grad_align` here is ALREADY the negated
         gradient -- see main(), where it's computed directly as
         alignment_scores(-grad_a, W) rather than negated for display)
      2. the TRUE expected direction (conditioner(B) - conditioner(A))
    The feature(s) this pair was designed to vary are colored orange;
    everything else is blue, so the "intended" bar is easy to spot
    without losing the full comparison."""
    grad_align = np.asarray(grad_align)
    expected_align = np.asarray(expected_align)
    colors = ["tab:orange" if (highlight_features and f in highlight_features) else "tab:blue"
              for f in feature_names]

    n_img = len(images)
    n_rows = max(n_img, 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.0 * n_rows), squeeze=False)

    for i in range(n_img):
        axes[i][0].imshow(images[i])
        axes[i][0].set_title(image_labels[i], fontsize=10)
        axes[i][0].axis("off")
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
    ax2.set_ylabel("cosine alignment")
    ax2.set_title("true direction: conditioner(B) - conditioner(A)  [reference]", fontsize=9)
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
    out_path: Path,
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
    per-timestep bar chart and the sweep-average plot)."""
    n_img = max(len(images), 1)

    fig = plt.figure(figsize=(11, 3.2 * n_img))
    gs = fig.add_gridspec(n_img, 2)

    for i, (img, label) in enumerate(zip(images, image_labels)):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(img)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis("off")

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
    out_path: Path,
) -> None:
    """Images on the left; on the right, ALL features, each with TWO
    grouped bars: -gradient (mean +/- std across the sweep -- `sweep_mean`
    already holds -gradient values, see main()) and the true expected
    direction -- so you can directly compare the intended feature's bars
    against every other feature's, not just see it in isolation. The
    feature(s) this pair varies get a light orange background band
    behind their whole group, so the "intended" one is still easy to
    spot in a wider chart.
    A gradient bar that's both tall AND has a small error bar relative to
    its height is a real, timestep-independent signal; an error bar as
    big as the bar itself means don't trust that number."""
    means = np.array([sweep_mean[n] for n in feature_names])
    stds = np.array([sweep_std[n] for n in feature_names])
    refs = np.asarray(expected_align)

    n_img = max(len(images), 1)
    fig = plt.figure(figsize=(max(9, len(feature_names) * 1.1), 3.2 * n_img))
    gs = fig.add_gridspec(n_img, 2)

    for i, (img, label) in enumerate(zip(images, image_labels)):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(img)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis("off")

    ax = fig.add_subplot(gs[:, 1])
    x = np.arange(len(feature_names))
    for i, name in enumerate(feature_names):
        if highlight_features and name in highlight_features:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.15, zorder=0)
    ax.bar(x - 0.18, means, width=0.36, yerr=stds, color="tab:purple", capsize=4, label="-gradient (mean +/- std)")
    ax.bar(x + 0.18, refs, width=0.36, color="tab:gray", label="true direction (reference)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_xlim(-0.5, len(feature_names) - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("cosine alignment")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/sanity_checks/num_steps_100/")
    parser.add_argument("--cond_input_dim", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=100,
                        help="DDIM steps, matching collect_gradients.py's --num_steps. "
                             "Each one adds a chained UNet call to backprop through, so "
                             "this directly controls how expensive each gradient is.")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[900, 500, 100],
                        help="Timesteps to check, matching collect_gradients.py's own "
                             "--timesteps convention. A representative spread (high/mid/low) "
                             "lets you check its stated coarse-vs-fine hypothesis.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Independent noise draws averaged per (case, timestep) -- "
                             "this script's own noise-reduction choice, NOT part of "
                             "collect_gradients.py's actual recipe (which relies on pair "
                             "volume instead). Each draw re-runs the full multi-step DDIM "
                             "backprop, so keep this modest.")
    parser.add_argument("--sweep_step", type=int, default=50,
                        help="Dense timestep sweep for a robust averaged statistic, in "
                             "addition to the few --timesteps above: every multiple of "
                             "this value up to num_train_timesteps (default step 10, "
                             "num_train_timesteps=1000 -> exactly 100 timesteps). A "
                             "couple of hand-picked timesteps can look meaningful by pure "
                             "chance; averaging alignment across a full dense sweep is a "
                             "much more reliable signal that it's real. Set higher (e.g. "
                             "50 -> 20 timesteps) to cut cost if this is too slow, or 0 "
                             "to disable the sweep entirely.")
    parser.add_argument("--sweep_n_samples", type=int, default=1,
                        help="Noise draws per sweep timestep. Default 1 since averaging "
                             "over ~100 different timesteps already reduces variance a "
                             "lot on its own -- unlike --n_samples above, which has to "
                             "do all its averaging at a single fixed timestep.")
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
    W = build_circle_basis_matrix(conditioner, feature_names, args.cond_input_dim, device)
    print(f"Circle vector subspace basis ({len(feature_names)} dims): {feature_names}")

    ddpm = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_steps)

    cases = extreme_cases(args.cond_input_dim)
    if not cases:
        raise SystemExit(f"No extreme cases available for cond_input_dim={args.cond_input_dim}")

    base_vec_list = zero_shape_prompt(shape_id=2, cond_input_dim=args.cond_input_dim)  # 2 = circle
    vec_a = torch.tensor(base_vec_list, dtype=torch.float32)
    print(f"\nBase prompt (all-zero circle) = {base_vec_list}")
    print("NOTE: every print/plot below reports -gradient (i.e. alignment_scores(-grad_a, W)), "
          "the direction that would DECREASE the loss -- not the raw gradient itself, which "
          "points toward increasing the loss (away from matching image B). A gradient that's "
          "actually tracking real semantics should show POSITIVE alignment between -gradient "
          "and the true (B-A) direction.")

    summary = []
    for i, case in enumerate(cases):
        # A fresh network-generated base image PER PAIR (not one image reused
        # for every comparison) -- so results aren't just an artifact of
        # whatever one specific noise realization the first sample happened
        # to produce.
        img_a_t, img_a_pil, x0_snapshots = generate_sample(
            unet, conditioner, ddim, vec_a, device, seed=args.seed + i,
        )
        img_a_pil.save(out_dir / f"{case['name']}_base.png")
        plot_x0_grid(
            x0_snapshots, out_dir / f"{case['name']}_x0_predictions.png",
            title=f"x0 prediction during base circle generation -- {case['label']}",
        )

        img_b_t, img_b_pil = render_image(case["vec"], seed=args.seed)
        vec_b = torch.tensor(case["vec"], dtype=torch.float32)

        with torch.no_grad():
            expected_direction = (
                conditioner(vec_b.unsqueeze(0).to(device)) - conditioner(vec_a.unsqueeze(0).to(device))
            ).squeeze().cpu().numpy()
        expected_align = alignment_scores(expected_direction, W)

        print(f"\n=== pair: base (random circle)  vs  {case['label']}  (base image seed={args.seed + i}) ===")
        curves = {name: [] for name in feature_names}
        for t in args.timesteps:
            grad_a, std_a = compute_pairwise_gradient(
                unet, conditioner, ddpm, ddim, img_a_t, vec_a, img_b_t,
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
                title=f"base circle  vs  {case['label']}   t={t}  (n_samples={args.n_samples})",
                images=[img_a_pil, img_b_pil], image_labels=["base (network-generated)", case["label"]],
                grad_align=grad_align, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                out_path=out_dir / f"{case['name']}_t{t}.png",
            )

        plot_alignment_vs_timestep(
            title=f"base vs {case['label']}: alignment vs timestep ({len(args.timesteps)} hand-picked points)",
            timesteps=args.timesteps, curves=curves, expected_align=expected_align,
            feature_names=feature_names, highlight_features=case["highlight"],
            images=[img_a_pil, img_b_pil], image_labels=["base (network-generated)", case["label"]],
            out_path=out_dir / f"{case['name']}_vs_timestep.png",
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
                    unet, conditioner, ddpm, ddim, img_a_t, vec_a, img_b_t,
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
                title=f"base vs {case['label']}: alignment vs timestep ({len(sweep_timesteps)}-timestep dense sweep)",
                timesteps=sweep_timesteps, curves=sweep_curves, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                images=[img_a_pil, img_b_pil], image_labels=["base (network-generated)", case["label"]],
                out_path=out_dir / f"{case['name']}_sweep_vs_timestep.png",
            )
            plot_sweep_average_bar(
                title=f"base vs {case['label']}: alignment averaged over {len(sweep_timesteps)} timesteps",
                sweep_mean=sweep_mean, sweep_std=sweep_std, expected_align=expected_align,
                feature_names=feature_names, highlight_features=case["highlight"],
                images=[img_a_pil, img_b_pil], image_labels=["base (network-generated)", case["label"]],
                out_path=out_dir / f"{case['name']}_sweep_average.png",
            )
            print(f"  Saved -> {case['name']}_sweep_vs_timestep.png, {case['name']}_sweep_average.png")

            sweep_result = {"sweep_timesteps": sweep_timesteps, "sweep_mean": sweep_mean, "sweep_std": sweep_std}

        summary.append({"case": case["name"], "label": case["label"], "highlight": case["highlight"],
                        "base_image_seed": args.seed + i,
                        "timesteps": args.timesteps, "curves": curves,
                        "sweep": sweep_result,
                        "expected_align": expected_align.tolist(), "feature_names": feature_names})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Saved summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()