"""
Geometric Shape Dataset Generator — 64x64
==========================================
Generates synthetic 64x64 image datasets of geometric shapes.

Each image is parameterized by a 10-dimensional sample vector:
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

Rendering notes:
  - White background
  - Stripes darken the shape color by 60%; crossings use same stripe color
  - Grain added on top of everything as pixel-level noise
  - h_stripe and v_stripe are fully independent and combinable

Output layout
-------------
  <output_dir>/
    images/           ← PNG files
    vectors.npy       ← float32 memmap, shape (N, 10)
    filenames.txt
    vectors_meta.json
    config.json

Usage
-----
  python dataset_64.py --output_dir data_64 --n_samples 20000
  python dataset_64.py --cache --output_dir data_64
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPE_NAMES    = {0: "triangle", 1: "square", 2: "circle"}
VECTOR_COLUMNS = [
    "is_triangle", "is_square", "is_circle",
    "r", "g", "b",
    "size",
    "h_stripe", "v_stripe", "grain",
]
VECTOR_DIM  = 10
RESOLUTION  = (64, 64)


# ---------------------------------------------------------------------------
# Sample vector
# ---------------------------------------------------------------------------

@dataclass
class SampleVector64:
    shape_id: int    # 0=triangle, 1=square, 2=circle
    r: float         # 0–1
    g: float         # 0–1
    b: float         # 0–1
    size: float      # 0–1
    h_stripe: float  # 0–1  horizontal stripe frequency
    v_stripe: float  # 0–1  vertical stripe frequency
    grain: float     # 0–1  grain strength

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
    def random(cls, rng: Optional[random.Random] = None) -> "SampleVector64":
        rng = rng or random.Random()
        return cls(
            shape_id=rng.randint(0, 2),
            r=rng.random(),
            g=rng.random(),
            b=rng.random(),
            size=rng.random(),
            h_stripe=rng.random(),
            v_stripe=rng.random(),
            grain=rng.random(),
        )

    def to_list(self) -> list:
        one_hot = [0.0, 0.0, 0.0]
        one_hot[self.shape_id] = 1.0
        return one_hot + [
            self.r, self.g, self.b, self.size,
            self.h_stripe, self.v_stripe, self.grain,
        ]

    @classmethod
    def from_list(cls, vec: list) -> "SampleVector64":
        return cls(
            shape_id=int(np.argmax(vec[:3])),
            r=vec[3], g=vec[4], b=vec[5],
            size=vec[6],
            h_stripe=vec[7],
            v_stripe=vec[8],
            grain=vec[9],
        )


# ---------------------------------------------------------------------------
# Image generator
# ---------------------------------------------------------------------------

class ImageGenerator64:
    """
    Generates a single 64x64 PIL Image from a SampleVector64.

    Pipeline:
      1. White background
      2. Fill shape with base color
      3. Apply horizontal and/or vertical stripes (darkening, combinable)
         - stripe pixels darken to 40% of base color
         - crossings use same stripe color (not double-darkened)
      4. Apply grain on top
    """

    def generate(self, vec: SampleVector64, seed: int = 0) -> Image.Image:
        W, H = RESOLUTION
        color = vec.color_rgb
        # size [0,1] → half in px: min=10, max=30
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
    def _make_mask(shape_id, cx, cy, half, W, H) -> np.ndarray:
        mask = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(mask)
        if shape_id == 0:
            ht = int(half * math.sqrt(3))
            pts = [
                (cx,        cy - ht * 2 // 3),
                (cx - half, cy + ht // 3),
                (cx + half, cy + ht // 3),
            ]
            d.polygon(pts, fill=255)
        elif shape_id == 1:
            d.rectangle([cx - half, cy - half, cx + half, cy + half], fill=255)
        else:
            d.ellipse([cx - half, cy - half, cx + half, cy + half], fill=255)
        return np.array(mask) > 0

    @staticmethod
    def _apply_stripes(arr, mask, color, h_freq, v_freq, half):
        max_band     = max(1, half // 3)
        band_h = max(1, int(max_band - h_freq * (max_band - 1))) if h_freq > 0.05 else None
        band_w = max(1, int(max_band - v_freq * (max_band - 1))) if v_freq > 0.05 else None
        base   = np.array(color, dtype=np.float32)
        stripe = base * 0.4   # stripe color = 40% of base
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
        rng   = np.random.RandomState(seed)
        noise = rng.randint(
            -int(grain_strength * 80),
            int(grain_strength * 80) + 1,
            (arr.shape[0], arr.shape[1], 3),
        )
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if mask[y, x]:
                    arr[y, x] = np.clip(
                        arr[y, x].astype(np.int32) + noise[y, x], 0, 255
                    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _render_worker(args: tuple) -> tuple:
    i, vec_list, img_dir, image_format, perturb = args
    vec = SampleVector64.from_list(vec_list)

    if perturb:
        rng_render = random.Random(i)
        perturbed = SampleVector64(
            shape_id=vec.shape_id,
            r        =_perturb(vec.r,        rng_render),
            g        =_perturb(vec.g,        rng_render),
            b        =_perturb(vec.b,        rng_render),
            size     =_perturb(vec.size,     rng_render),
            h_stripe =_perturb(vec.h_stripe, rng_render),
            v_stripe =_perturb(vec.v_stripe, rng_render),
            grain    =_perturb(vec.grain,    rng_render),
        )
    else:
        perturbed = vec

    img = ImageGenerator64().generate(perturbed, seed=i)
    ext   = "png" if image_format.upper() == "PNG" else "jpg"
    fname = f"{i:06d}_{vec.shape_name}.{ext}"
    save_kwargs = {"quality": 95} if ext == "jpg" else {}
    img.save(Path(img_dir) / fname, format=image_format.upper(), **save_kwargs)
    return i, fname, vec_list


def _perturb(v: float, rng: random.Random) -> float:
    """
    Perturb a value in [0,1] with width proportional to distance from ends.
    width = 1 - 2*|v - 0.5|
      v=0.5 → width=1.0 (sample from full [0,1])
      v=0   → width=0.0 (unchanged)
      v=1   → width=0.0 (unchanged)
    """
    width = 1.0 - 2.0 * abs(v - 0.5)
    if width <= 0:
        return v
    lo = max(0.0, v - width / 2)
    hi = min(1.0, v + width / 2)
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

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
    fix_shape:    Optional[int]  = None
    fix_r:        Optional[float]= None
    fix_g:        Optional[float]= None
    fix_b:        Optional[float]= None
    fix_size:     Optional[float]= None
    fix_h_stripe: Optional[float]= None
    fix_v_stripe: Optional[float]= None
    fix_grain:    Optional[float]= None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolution"] = list(d["resolution"])
        return d


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class DatasetGenerator64:
    """
    Generates a 64x64 geometric shape dataset.

    Parameters
    ----------
    output_dir   : str
    n_samples    : int
    seed         : int | None
    image_format : str      "PNG" or "JPEG"
    num_workers  : int      0=single-process, -1=all cores, N=N workers
    verbose      : bool
    """

    def __init__(
        self,
        output_dir:   str           = "data_64",
        n_samples:    int           = 20000,
        seed:         Optional[int] = 42,
        image_format: str           = "PNG",
        num_workers:  int           = 0,
        verbose:      bool          = True,
        perturb:      bool          = True,
        fix_shape:    Optional[int]   = None,
        fix_r:        Optional[float] = None,
        fix_g:        Optional[float] = None,
        fix_b:        Optional[float] = None,
        fix_size:     Optional[float] = None,
        fix_h_stripe: Optional[float] = None,
        fix_v_stripe: Optional[float] = None,
        fix_grain:    Optional[float] = None,
    ):
        self.cfg = DatasetConfig64(
            output_dir=output_dir,
            n_samples=n_samples,
            seed=seed,
            image_format=image_format,
            num_workers=num_workers,
            verbose=verbose,
            perturb=perturb,
            fix_shape   =fix_shape,
            fix_r       =fix_r,
            fix_g       =fix_g,
            fix_b       =fix_b,
            fix_size    =fix_size,
            fix_h_stripe=fix_h_stripe,
            fix_v_stripe=fix_v_stripe,
            fix_grain   =fix_grain,
        )
        self._rng = random.Random(seed)

    def generate(self) -> np.ndarray:
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
            job_args.append((i, v.to_list(), str(img_dir), self.cfg.image_format, self.cfg.perturb))

        nw = self.cfg.num_workers
        if nw == -1:
            nw = cpu_count()

        results = self._generate_parallel(job_args, nw) if nw > 0 else self._generate_sequential(job_args)
        results.sort(key=lambda x: x[0])

        vec_arr   = np.array([r[2] for r in results], dtype=np.float32)
        filenames = [r[1] for r in results]

        mmap = np.memmap(out_dir / "vectors.npy", dtype=np.float32, mode="w+", shape=(N, VECTOR_DIM))
        mmap[:] = vec_arr
        mmap.flush()
        del mmap

        (out_dir / "filenames.txt").write_text("\n".join(filenames))
        (out_dir / "vectors_meta.json").write_text(json.dumps({
            "shape":   [N, VECTOR_DIM],
            "dtype":   "float32",
            "columns": VECTOR_COLUMNS,
            "notes": {
                "is_triangle / is_square / is_circle": "one-hot shape encoding",
                "r / g / b":   "linear [0,1]; scaled to 0–220 during rendering",
                "size":        "linear [0,1]; maps half-size to 10–30px on 64px canvas",
                "h_stripe":    "horizontal stripe frequency [0,1]; 0=no stripes",
                "v_stripe":    "vertical stripe frequency [0,1]; 0=no stripes",
                "grain":       "pixel-level noise strength [0,1]; 0=no grain",
            },
        }, indent=2))
        (out_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))

        if self.cfg.verbose:
            counts = {
                "triangle": int((vec_arr[:, 0] == 1).sum()),
                "square":   int((vec_arr[:, 1] == 1).sum()),
                "circle":   int((vec_arr[:, 2] == 1).sum()),
            }
            workers_str = f"{nw} workers" if nw > 0 else "single-process"
            print(f"\n✓ {N} images  →  {out_dir}/images/  ({workers_str})")
            print(f"✓ vectors     →  {out_dir}/vectors.npy  {vec_arr.shape} float32")
            print(f"  Shape counts: {counts}")

        return vec_arr

    def _generate_sequential(self, job_args):
        results = []
        N = len(job_args)
        for args in job_args:
            i, fname, vec_list = _render_worker(args)
            results.append((i, fname, vec_list))
            if self.cfg.verbose and (i % max(1, N // 20) == 0):
                print(f"  [{i+1:>{len(str(N))}}/{N}] {fname}")
        return results

    def _generate_parallel(self, job_args, num_workers):
        N = len(job_args)
        results = []
        width = len(str(N))
        with Pool(processes=num_workers) as pool:
            for i, fname, vec_list in pool.imap_unordered(
                _render_worker, job_args, chunksize=max(1, N // (num_workers * 4))
            ):
                results.append((i, fname, vec_list))
                if self.cfg.verbose and (len(results) % max(1, N // 20) == 0):
                    print(f"  [{len(results):>{width}}/{N}] {fname}")
        return results


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ShapeDataset64:
    """
    Loads 64x64 shape images and conditioning vectors into RAM.

    Returns (per __getitem__)
    -------------------------
    image  : torch.Tensor (3, 64, 64) float32 normalized to [-1, 1]
    vector : torch.Tensor (10,)       float32
    """

    def __init__(self, root, max_samples=None):
        self.root = Path(root)
        images_np = np.load(self.root / "images_cached.npy")  # (N, 64, 64, 3) uint8
        if max_samples is not None:
            images_np = images_np[:max_samples]
        images_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
        del images_np
        self._images = (images_t / 255.0 - 0.5) / 0.5
        del images_t
        self._mmap = np.memmap(
            self.root / "vectors.npy", dtype=np.float32, mode="r"
        ).reshape(-1, VECTOR_DIM)
        if max_samples is not None:
            self._mmap = self._mmap[:max_samples]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, i):
        return self._images[i], torch.from_numpy(self._mmap[i].copy())


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(root: str):
    """Convert PNG images to a single numpy cache. Run once after generating."""
    root = Path(root)
    filenames = (root / "filenames.txt").read_text().splitlines()
    print(f"Building cache for {len(filenames)} images...")
    out = [np.array(Image.open(root / "images" / f).convert("RGB"), dtype=np.uint8)
           for f in filenames]
    arr = np.stack(out)
    np.save(root / "images_cached.npy", arr)
    print(f"Saved → {root / 'images_cached.npy'}  shape: {arr.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a 64x64 geometric shape dataset.")
    parser.add_argument("--output_dir",  default="data_64",  type=str)
    parser.add_argument("--n_samples",   default=20000,      type=int)
    parser.add_argument("--seed",        default=42,         type=int)
    parser.add_argument("--format",      default="PNG",      choices=["PNG", "JPEG"])
    parser.add_argument("--num_workers", default=0,          type=int,
                        help="0=single-process, -1=all cores")
    parser.add_argument("--no_perturb",  action="store_true",
                        help="Disable stochastic feature perturbation during rendering")
    parser.add_argument("--fix_shape",   default=None,       type=int, choices=[0, 1, 2],
                        help="Fix shape: 0=triangle, 1=square, 2=circle (default: random)")
    parser.add_argument("--fix_r",       default=None,       type=float, help="Fix r in [0,1]")
    parser.add_argument("--fix_g",       default=None,       type=float, help="Fix g in [0,1]")
    parser.add_argument("--fix_b",       default=None,       type=float, help="Fix b in [0,1]")
    parser.add_argument("--fix_size",    default=None,       type=float, help="Fix size in [0,1]")
    parser.add_argument("--fix_h_stripe",default=None,       type=float, help="Fix h_stripe in [0,1]")
    parser.add_argument("--fix_v_stripe",default=None,       type=float, help="Fix v_stripe in [0,1]")
    parser.add_argument("--fix_grain",   default=None,       type=float, help="Fix grain in [0,1]")
    parser.add_argument("--cache",       action="store_true",
                        help="Build images_cached.npy instead of generating")
    args = parser.parse_args()

    if args.cache:
        build_cache(args.output_dir)
    else:
        gen = DatasetGenerator64(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            image_format=args.format,
            num_workers=args.num_workers,
            verbose=True,
            perturb=not args.no_perturb,
            fix_shape   =args.fix_shape,
            fix_r       =args.fix_r,
            fix_g       =args.fix_g,
            fix_b       =args.fix_b,
            fix_size    =args.fix_size,
            fix_h_stripe=args.fix_h_stripe,
            fix_v_stripe=args.fix_v_stripe,
            fix_grain   =args.fix_grain,
        )
        gen.generate()
        print(f"\nNow build the cache:")
        print(f"  python dataset_64.py --cache --output_dir {args.output_dir}")