"""
Geometric Shape Dataset Generator — 64x64
==========================================
Generates synthetic 64x64 image datasets of geometric shapes, with a
*configurable set of underlying generative factors* ("features").

Full factor schema (10-dim, used when all feature groups are active):
  [0]  is_triangle  : float  1.0 if triangle, else 0.0  (one-hot)
  [1]  is_square    : float  1.0 if square,   else 0.0  (one-hot)
  [2]  is_circle    : float  1.0 if circle,   else 0.0  (one-hot)
  [3]  r            : float [0,1] → red channel
  [4]  g            : float [0,1] → green channel
  [5]  b            : float [0,1] → blue channel
  [6]  size         : float [0,1] → half-size maps to 10–30px (out of 32)
  [7]  h_stripe     : float [0,1] → horizontal stripe frequency (0=none)
  [8]  v_stripe     : float [0,1] → vertical stripe frequency (0=none)
  [9]  grain        : float [0,1] → grain/noise strength (0=none)

Feature groups (select any non-empty subset with --features)
--------------------------------------------------------------
  shape     -> is_triangle, is_square, is_circle   (3 dims)
  color     -> r, g, b                             (3 dims)
  size      -> size                                (1 dim)
  stripes   -> h_stripe, v_stripe                   (2 dims)
  grain     -> grain                                (1 dim)

Any group you *don't* include is:
  - dropped entirely from the stored vector (vectors.npy shrinks accordingly)
  - fixed to a neutral, non-perturbed constant during rendering, so the
    excluded factor contributes *no* variance to the images at all
    (not just to the label vector).

This lets you construct datasets with an exact, known intrinsic/generative
dimension — e.g. `--features shape,color,size` yields images that vary
along exactly 7 dimensions (3 one-hot + 3 color + 1 size), with zero
texture variation, which is useful as ground truth for validating
intrinsic-dimension / sparse-representation estimators.

You can still pin an *included* feature to a fixed value with --fix_*
(it stays in the vector as a constant column, and is still subject to
stochastic rendering perturbation around that fixed value, same as
before). You can also use --fix_* on an *excluded* group to override its
default neutral constant (e.g. force grain=0.2 as unmodeled nuisance
texture while still excluding it from the label vector).

Perturbation (stochastic rendering) — unchanged from before, and only
applied to *active* (included) continuous features:
  Original:  width = 1 - 2|v - 0.5|
             zero only exactly at v=0 and v=1

  New (flat-top, default): dead zones at [0, dead_zone] and [1-dead_zone, 1]
             zero width (deterministic) for v <= dead_zone or v >= 1-dead_zone
             linearly rises to max width at v=0.5
             default dead_zone = 0.1

  Set --dead_zone 0.0 to recover original behaviour exactly.

images_cached.npy (a single stacked uint8 array consumed by ShapeDataset64)
is now built automatically at the end of generation. Pass --no_cache to
skip this (e.g. for very large datasets where you'd rather stream images
lazily), and build it later with --cache against an existing output_dir.

Usage
-----
  python dataset_64.py --output_dir data_64 --n_samples 20000
  python dataset_64.py --output_dir data_64 --features shape,color,size     # no textures, 7-dim
  python dataset_64.py --output_dir data_64 --features color,size          # fixed shape, 4-dim
  python dataset_64.py --output_dir data_64 --features shape               # fixed color/size/texture, 3-dim
  python dataset_64.py --output_dir data_64 --dead_zone 0.0   # original perturbation behaviour
  python dataset_64.py --output_dir data_64 --no_perturb
  python dataset_64.py --output_dir data_64 --no_cache                     # skip auto-caching
  python dataset_64.py --output_dir data_64 --cache                        # (re)build cache only, no generation
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

# NOTE: torch is deliberately NOT imported at module level. It's only needed
# by ShapeDataset64 (the training-time loader), and importing it can be slow
# on network filesystems (many small files). Dataset generation (everything
# reached via the CLI __main__ block / DatasetGenerator64) is pure
# NumPy/PIL and never touches torch, so it shouldn't have to pay that cost.


SHAPE_NAMES = {0: "triangle", 1: "square", 2: "circle"}

# ---------------------------------------------------------------------------
# Full generative-factor schema (fixed, canonical order). Individual dataset
# instances may use only a subset of these, selected via feature groups.
# ---------------------------------------------------------------------------
FULL_COLUMNS = [
    "is_triangle", "is_square", "is_circle",
    "r", "g", "b", "size", "h_stripe", "v_stripe", "grain",
]
FULL_DIM = len(FULL_COLUMNS)

# Backwards-compatible aliases (represent the *full* 10-dim schema).
VECTOR_COLUMNS = FULL_COLUMNS
VECTOR_DIM = FULL_DIM

RESOLUTION = (64, 64)

# Feature groups: canonical order + which full-schema columns they own.
FEATURE_ORDER = ["shape", "color", "size", "stripes", "grain"]
FEATURE_GROUPS = {
    "shape":   (0, 3),   # is_triangle, is_square, is_circle
    "color":   (3, 6),   # r, g, b
    "size":    (6, 7),   # size
    "stripes": (7, 9),   # h_stripe, v_stripe
    "grain":   (9, 10),  # grain
}

# Neutral constants used when a feature group is excluded and no --fix_*
# override was supplied for its constituent attribute(s).
GROUP_DEFAULTS = {
    "shape":   {"shape_id": 2},              # default to circle
    "color":   {"r": 0.5, "g": 0.5, "b": 0.5},  # neutral mid-gray
    "size":    {"size": 0.5},                # mid size
    "stripes": {"h_stripe": 0.0, "v_stripe": 0.0},  # no stripes
    "grain":   {"grain": 0.0},               # no grain
}


def _resolve_features(features: Union[str, Sequence[str], None]) -> List[str]:
    """Parse and validate a feature spec into a canonical-order list of
    active group names (canonical order regardless of input order)."""
    if features is None:
        requested = list(FEATURE_ORDER)
    elif isinstance(features, str):
        requested = [f.strip() for f in features.split(",") if f.strip()]
    else:
        requested = list(features)

    unknown = sorted(set(requested) - set(FEATURE_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {unknown}. Valid groups: {FEATURE_ORDER}"
        )
    if not requested:
        raise ValueError("At least one feature group must be active.")

    # de-duplicate while imposing canonical order
    return [g for g in FEATURE_ORDER if g in requested]


def _active_columns(active_features: Sequence[str]) -> List[int]:
    cols: List[int] = []
    for g in active_features:
        start, end = FEATURE_GROUPS[g]
        cols.extend(range(start, end))
    return cols


def _active_vector_columns(active_features: Sequence[str]) -> List[str]:
    return [FULL_COLUMNS[i] for i in _active_columns(active_features)]


def _extract_active(full_vec_list: Sequence[float], active_features: Sequence[str]) -> List[float]:
    idx = _active_columns(active_features)
    return [full_vec_list[i] for i in idx]


@dataclass
class SampleVector64:
    shape_id: int
    r: float
    g: float
    b: float
    size: float
    h_stripe: float
    v_stripe: float
    grain: float

    @property
    def shape_name(self) -> str:
        return SHAPE_NAMES[self.shape_id]

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        r = int(self.r * 220)
        g = int(self.g * 220)
        b = int(self.b * 220)
        if r + g + b > 580:
            f = 580 / max(r + g + b, 1)
            r, g, b = int(r * f), int(g * f), int(b * f)
        if r + g + b < 30:
            r, g, b = 60, 60, 60
        return (r, g, b)

    @classmethod
    def random(cls, rng=None):
        rng = rng or random.Random()
        return cls(
            shape_id=rng.randint(0, 2),
            r=rng.random(), g=rng.random(), b=rng.random(),
            size=rng.random(), h_stripe=rng.random(),
            v_stripe=rng.random(), grain=rng.random(),
        )

    def to_list(self):
        """Full 10-dim factor vector (canonical order). Used internally for
        rendering; storage uses only the active-feature subset."""
        one_hot = [0.0, 0.0, 0.0]
        one_hot[self.shape_id] = 1.0
        return one_hot + [self.r, self.g, self.b, self.size,
                          self.h_stripe, self.v_stripe, self.grain]

    @classmethod
    def from_list(cls, vec):
        """Reconstructs from a *full* 10-dim factor vector."""
        return cls(
            shape_id=int(np.argmax(vec[:3])),
            r=vec[3], g=vec[4], b=vec[5], size=vec[6],
            h_stripe=vec[7], v_stripe=vec[8], grain=vec[9],
        )


def _perturb(v: float, rng: random.Random, dead_zone: float = 0.1) -> float:
    """
    Flat-top perturbation with dead zones near extremes.

    dead_zone=0.1 (default):
      v <= 0.1 or v >= 0.9  →  no perturbation (deterministic)
      v = 0.5               →  width = 0.8 (sample from [0.1, 0.9])
      v = 0.3               →  width = 0.4

    dead_zone=0.0:
      reduces to original  width = 1 - 2|v - 0.5|
      (only exactly v=0 and v=1 are deterministic)
    """
    if v <= dead_zone or v >= 1.0 - dead_zone:
        return v

    # width rises linearly from 0 at dead_zone to (1-2*dead_zone) at v=0.5
    active_half = 0.5 - dead_zone
    width = 2.0 * (active_half - abs(v - 0.5))

    if width <= 0:
        return v

    lo = max(0.0, v - width / 2)
    hi = min(1.0, v + width / 2)
    return rng.uniform(lo, hi)


class ImageGenerator64:
    def generate(self, vec: SampleVector64, seed: int = 0) -> Image.Image:
        W, H = RESOLUTION
        color = vec.color_rgb
        half = 10 + int(vec.size * 20)
        arr = np.ones((H, W, 3), dtype=np.uint8) * 255
        mask = self._make_mask(vec.shape_id, W // 2, H // 2, half, W, H)
        arr[mask] = color
        if vec.h_stripe > 0.05 or vec.v_stripe > 0.05:
            self._apply_stripes(arr, mask, color, vec.h_stripe, vec.v_stripe, half)
        if vec.grain > 0.05:
            self._apply_grain(arr, mask, vec.grain, seed)
        return Image.fromarray(arr)

    @staticmethod
    def _make_mask(shape_id, cx, cy, half, W, H):
        mask = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(mask)
        if shape_id == 0:
            ht = int(half * math.sqrt(3))
            pts = [(cx, cy - ht*2//3), (cx-half, cy+ht//3), (cx+half, cy+ht//3)]
            d.polygon(pts, fill=255)
        elif shape_id == 1:
            d.rectangle([cx-half, cy-half, cx+half, cy+half], fill=255)
        else:
            d.ellipse([cx-half, cy-half, cx+half, cy+half], fill=255)
        return np.array(mask) > 0

    @staticmethod
    def _apply_stripes(arr, mask, color, h_freq, v_freq, half):
        max_band = max(1, half // 3)
        band_h = max(1, int(max_band - h_freq*(max_band-1))) if h_freq > 0.05 else None
        band_w = max(1, int(max_band - v_freq*(max_band-1))) if v_freq > 0.05 else None
        base = np.array(color, dtype=np.float32)
        stripe = base * 0.4
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if mask[y, x]:
                    h_dark = band_h is not None and (y // band_h) % 2 == 1
                    v_dark = band_w is not None and (x // band_w) % 2 == 1
                    if h_dark or v_dark:
                        arr[y, x] = np.clip(stripe, 0, 255).astype(np.uint8)
                    else:
                        arr[y, x] = np.clip(base, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_grain(arr, mask, grain_strength, seed):
        rng = np.random.RandomState(seed)
        noise = rng.randint(-int(grain_strength*80), int(grain_strength*80)+1,
                            (arr.shape[0], arr.shape[1], 3))
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if mask[y, x]:
                    arr[y, x] = np.clip(
                        arr[y, x].astype(np.int32) + noise[y, x], 0, 255
                    ).astype(np.uint8)


def _render_worker(args: tuple) -> tuple:
    (i, full_vec_list, active_vec_list, active_features,
     img_dir, image_format, perturb, dead_zone, want_cache) = args

    vec = SampleVector64.from_list(full_vec_list)

    if perturb:
        rng_render = random.Random(i)
        perturbed = SampleVector64(
            shape_id=vec.shape_id,
            r        =_perturb(vec.r,        rng_render, dead_zone) if "color"   in active_features else vec.r,
            g        =_perturb(vec.g,        rng_render, dead_zone) if "color"   in active_features else vec.g,
            b        =_perturb(vec.b,        rng_render, dead_zone) if "color"   in active_features else vec.b,
            size     =_perturb(vec.size,     rng_render, dead_zone) if "size"    in active_features else vec.size,
            h_stripe =_perturb(vec.h_stripe, rng_render, dead_zone) if "stripes" in active_features else vec.h_stripe,
            v_stripe =_perturb(vec.v_stripe, rng_render, dead_zone) if "stripes" in active_features else vec.v_stripe,
            grain    =_perturb(vec.grain,    rng_render, dead_zone) if "grain"   in active_features else vec.grain,
        )
    else:
        perturbed = vec

    img = ImageGenerator64().generate(perturbed, seed=i)
    # Only materialize the in-memory array if we'll need it for the cache —
    # avoids the extra memory/IPC cost entirely when caching is disabled.
    img_arr = np.array(img, dtype=np.uint8) if want_cache else None
    ext = "png" if image_format.upper() == "PNG" else "jpg"
    fname = f"{i:06d}_{vec.shape_name}.{ext}"
    save_kwargs = {"quality": 95} if ext == "jpg" else {}
    img.save(Path(img_dir) / fname, format=image_format.upper(), **save_kwargs)
    return i, fname, active_vec_list, img_arr


@dataclass
class DatasetConfig64:
    output_dir:   str            = "data_64"
    n_samples:    int            = 20000
    resolution:   Tuple[int,int] = (64, 64)
    seed:         Optional[int]  = 42
    image_format: str            = "PNG"
    num_workers:  int            = 0
    verbose:      bool           = True
    perturb:      bool           = True
    dead_zone:    float          = 0.1
    features:     str            = ",".join(FEATURE_ORDER)
    cache:        bool           = True
    fix_shape:    Optional[int]  = None
    fix_r:        Optional[float]= None
    fix_g:        Optional[float]= None
    fix_b:        Optional[float]= None
    fix_size:     Optional[float]= None
    fix_h_stripe: Optional[float]= None
    fix_v_stripe: Optional[float]= None
    fix_grain:    Optional[float]= None

    def to_dict(self):
        d = asdict(self)
        d["resolution"] = list(d["resolution"])
        return d


class DatasetGenerator64:
    def __init__(self, output_dir="data_64", n_samples=20000, seed=42,
                 image_format="PNG", num_workers=0, verbose=True,
                 perturb=True, dead_zone=0.1, features=None, cache=True,
                 fix_shape=None, fix_r=None, fix_g=None, fix_b=None,
                 fix_size=None, fix_h_stripe=None, fix_v_stripe=None, fix_grain=None):
        active_features = _resolve_features(features)
        self.cfg = DatasetConfig64(
            output_dir=output_dir, n_samples=n_samples, seed=seed,
            image_format=image_format, num_workers=num_workers,
            verbose=verbose, perturb=perturb, dead_zone=dead_zone,
            features=",".join(active_features), cache=cache,
            fix_shape=fix_shape, fix_r=fix_r, fix_g=fix_g, fix_b=fix_b,
            fix_size=fix_size, fix_h_stripe=fix_h_stripe,
            fix_v_stripe=fix_v_stripe, fix_grain=fix_grain,
        )
        self.active_features = active_features
        self.active_vector_columns = _active_vector_columns(active_features)
        self.active_dim = len(self.active_vector_columns)
        self._rng = random.Random(seed)

    def _apply_group_defaults(self, v: SampleVector64) -> None:
        """For any excluded feature group, force its attribute(s) to a
        neutral constant — unless a --fix_* override already set it."""
        fix = self.cfg
        active = self.active_features

        if "shape" not in active and fix.fix_shape is None:
            v.shape_id = GROUP_DEFAULTS["shape"]["shape_id"]

        if "color" not in active:
            if fix.fix_r is None: v.r = GROUP_DEFAULTS["color"]["r"]
            if fix.fix_g is None: v.g = GROUP_DEFAULTS["color"]["g"]
            if fix.fix_b is None: v.b = GROUP_DEFAULTS["color"]["b"]

        if "size" not in active and fix.fix_size is None:
            v.size = GROUP_DEFAULTS["size"]["size"]

        if "stripes" not in active:
            if fix.fix_h_stripe is None: v.h_stripe = GROUP_DEFAULTS["stripes"]["h_stripe"]
            if fix.fix_v_stripe is None: v.v_stripe = GROUP_DEFAULTS["stripes"]["v_stripe"]

        if "grain" not in active and fix.fix_grain is None:
            v.grain = GROUP_DEFAULTS["grain"]["grain"]

    def generate(self):
        out_dir = Path(self.cfg.output_dir)
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        N = self.cfg.n_samples
        fix = self.cfg
        job_args = []
        for i in range(N):
            v = SampleVector64.random(self._rng)
            if fix.fix_shape    is not None: v.shape_id  = fix.fix_shape
            if fix.fix_r        is not None: v.r         = fix.fix_r
            if fix.fix_g        is not None: v.g         = fix.fix_g
            if fix.fix_b        is not None: v.b         = fix.fix_b
            if fix.fix_size     is not None: v.size      = fix.fix_size
            if fix.fix_h_stripe is not None: v.h_stripe  = fix.fix_h_stripe
            if fix.fix_v_stripe is not None: v.v_stripe  = fix.fix_v_stripe
            if fix.fix_grain    is not None: v.grain     = fix.fix_grain

            self._apply_group_defaults(v)

            full_vec_list = v.to_list()
            active_vec_list = _extract_active(full_vec_list, self.active_features)
            job_args.append((i, full_vec_list, active_vec_list, self.active_features,
                             str(img_dir), self.cfg.image_format, self.cfg.perturb,
                             self.cfg.dead_zone, self.cfg.cache))
        nw = self.cfg.num_workers
        if nw == -1:
            nw = cpu_count()
        results = (self._generate_parallel(job_args, nw)
                   if nw > 0 else self._generate_sequential(job_args))
        results.sort(key=lambda x: x[0])
        vec_arr   = np.array([r[2] for r in results], dtype=np.float32)
        filenames = [r[1] for r in results]
        mmap = np.memmap(out_dir / "vectors.npy", dtype=np.float32,
                         mode="w+", shape=(N, self.active_dim))
        mmap[:] = vec_arr
        mmap.flush()
        del mmap
        (out_dir / "filenames.txt").write_text("\n".join(filenames))

        disabled = [g for g in FEATURE_ORDER if g not in self.active_features]
        (out_dir / "vectors_meta.json").write_text(json.dumps({
            "shape": [N, self.active_dim], "dtype": "float32",
            "columns": self.active_vector_columns,
            "features": self.active_features,
            "disabled_features": disabled,
            "full_schema": FULL_COLUMNS,
            "perturbation": {
                "dead_zone": self.cfg.dead_zone,
                "description": (
                    f"No perturbation for v <= {self.cfg.dead_zone} or "
                    f"v >= {1-self.cfg.dead_zone}. "
                    "Linear interpolation to max width at v=0.5. "
                    "Set dead_zone=0.0 for original behaviour. "
                    "Disabled feature groups are never perturbed (held exactly constant)."
                ),
            },
        }, indent=2))
        (out_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        if self.cfg.verbose:
            nw_str = f"{nw} workers" if nw > 0 else "single-process"
            print(f"\n✓ {N} images → {out_dir}/images/ ({nw_str})")
            print(f"✓ vectors   → {out_dir}/vectors.npy {vec_arr.shape} float32")
            print(f"  Active features : {self.active_features}  (dim={self.active_dim})")
            print(f"  Columns         : {self.active_vector_columns}")
            if disabled:
                print(f"  Disabled        : {disabled} (fixed constant, no rendered variation)")

            if "is_triangle" in self.active_vector_columns:
                ci = {name: self.active_vector_columns.index(name)
                      for name in ("is_triangle", "is_square", "is_circle")}
                counts = {
                    "triangle": int((vec_arr[:, ci["is_triangle"]] == 1).sum()),
                    "square":   int((vec_arr[:, ci["is_square"]] == 1).sum()),
                    "circle":   int((vec_arr[:, ci["is_circle"]] == 1).sum()),
                }
                print(f"  Shape counts: {counts}")
            else:
                fixed_shape_id = fix.fix_shape if fix.fix_shape is not None else GROUP_DEFAULTS["shape"]["shape_id"]
                print(f"  Shape: fixed to '{SHAPE_NAMES[fixed_shape_id]}' (feature disabled)")

        if self.cfg.cache:
            image_arr = np.stack([r[3] for r in results])  # already in memory, no disk re-read
            np.save(out_dir / "images_cached.npy", image_arr)
            if self.cfg.verbose:
                print(f"✓ cache      → {out_dir / 'images_cached.npy'} {image_arr.shape} uint8 "
                      f"(built in-memory, no extra file reads)")
        elif self.cfg.verbose:
            print("  (skipped images_cached.npy — pass cache=True or run --cache separately)")
        return vec_arr

    def _generate_sequential(self, job_args):
        results = []
        N = len(job_args)
        for args in job_args:
            i, fname, vec_list, img_arr = _render_worker(args)
            results.append((i, fname, vec_list, img_arr))
            if self.cfg.verbose and (i % max(1, N // 20) == 0):
                print(f"  [{i+1:>{len(str(N))}}/{N}] {fname}")
        return results

    def _generate_parallel(self, job_args, num_workers):
        N = len(job_args)
        results = []
        width = len(str(N))
        with Pool(processes=num_workers) as pool:
            for i, fname, vec_list, img_arr in pool.imap_unordered(
                _render_worker, job_args, chunksize=max(1, N // (num_workers * 4))
            ):
                results.append((i, fname, vec_list, img_arr))
                if self.cfg.verbose and (len(results) % max(1, N // 20) == 0):
                    print(f"  [{len(results):>{width}}/{N}] {fname}")
        return results


class ShapeDataset64:
    def __init__(self, root, max_samples=None):
        import torch  # lazy: only this class needs torch, imported on first use
        self.root = Path(root)
        meta = json.loads((self.root / "vectors_meta.json").read_text())
        vec_dim = meta["shape"][1]
        self.vector_columns = meta["columns"]
        self.active_features = meta.get("features", FEATURE_ORDER)

        images_np = np.load(self.root / "images_cached.npy")
        if max_samples is not None:
            images_np = images_np[:max_samples]
        images_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
        del images_np
        self._images = (images_t / 255.0 - 0.5) / 0.5
        del images_t
        self._mmap = np.memmap(
            self.root / "vectors.npy", dtype=np.float32, mode="r"
        ).reshape(-1, vec_dim)
        if max_samples is not None:
            self._mmap = self._mmap[:max_samples]

    def __len__(self): return len(self._images)
    def __getitem__(self, i):
        import torch  # lazy: cheap after first load, avoids module-level cost
        return self._images[i], torch.from_numpy(self._mmap[i].copy())


def build_cache(root: str, verbose: bool = True):
    root = Path(root)
    filenames = (root / "filenames.txt").read_text().splitlines()
    if verbose:
        print(f"  Caching {len(filenames)} images → images_cached.npy ...")
    out = [np.array(Image.open(root / "images" / f).convert("RGB"), dtype=np.uint8)
           for f in filenames]
    arr = np.stack(out)
    np.save(root / "images_cached.npy", arr)
    if verbose:
        print(f"✓ cache      → {root / 'images_cached.npy'} {arr.shape} uint8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",   default="data_64",  type=str)
    parser.add_argument("--n_samples",    default=20000,      type=int)
    parser.add_argument("--seed",         default=42,         type=int)
    parser.add_argument("--format",       default="PNG",      choices=["PNG", "JPEG"])
    parser.add_argument("--num_workers",  default=0,          type=int)
    parser.add_argument("--no_perturb",   action="store_true")
    parser.add_argument("--no_cache",     action="store_true",
                        help="Skip building images_cached.npy after generation "
                             "(it is built automatically by default). Use "
                             "--cache later to build it separately.")
    parser.add_argument("--dead_zone",    default=0.1,        type=float,
                        help="Dead zone at each end of [0,1] with no perturbation "
                             "(default 0.1). Set to 0.0 for original behaviour.")
    parser.add_argument("--features",     default=",".join(FEATURE_ORDER), type=str,
                        help="Comma-separated list of feature groups to include: "
                             f"{FEATURE_ORDER}. Excluded groups are fixed to a "
                             "neutral constant and dropped from vectors.npy — e.g. "
                             "--features shape,color,size gives a 7-dim manifold "
                             "with no texture variation at all.")
    parser.add_argument("--fix_shape",    default=None, type=int, choices=[0,1,2])
    parser.add_argument("--fix_r",        default=None, type=float)
    parser.add_argument("--fix_g",        default=None, type=float)
    parser.add_argument("--fix_b",        default=None, type=float)
    parser.add_argument("--fix_size",     default=None, type=float)
    parser.add_argument("--fix_h_stripe", default=None, type=float)
    parser.add_argument("--fix_v_stripe", default=None, type=float)
    parser.add_argument("--fix_grain",    default=None, type=float)
    parser.add_argument("--cache",        action="store_true")
    args = parser.parse_args()
    if args.cache:
        build_cache(args.output_dir)
    else:
        DatasetGenerator64(
            output_dir=args.output_dir, n_samples=args.n_samples,
            seed=args.seed, image_format=args.format,
            num_workers=args.num_workers, verbose=True,
            perturb=not args.no_perturb, dead_zone=args.dead_zone,
            features=args.features, cache=not args.no_cache,
            fix_shape=args.fix_shape, fix_r=args.fix_r,
            fix_g=args.fix_g, fix_b=args.fix_b, fix_size=args.fix_size,
            fix_h_stripe=args.fix_h_stripe, fix_v_stripe=args.fix_v_stripe,
            fix_grain=args.fix_grain,
        ).generate()