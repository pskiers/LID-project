"""
Geometric Shape Dataset Generator — 32x32
==========================================
Generates synthetic 32x32 image datasets of geometric shapes.

Each image is parameterized by a 10-dimensional sample vector:
  [0]  is_triangle  : float  1.0 if triangle, else 0.0  (one-hot)
  [1]  is_square    : float  1.0 if square,   else 0.0  (one-hot)
  [2]  is_circle    : float  1.0 if circle,   else 0.0  (one-hot)
  [3]  r            : float [0,1] → red channel
  [4]  g            : float [0,1] → green channel
  [5]  b            : float [0,1] → blue channel
  [6]  size         : float [0,1] → half-size maps to 5–13px (out of 16)
  [7]  v_stripe     : float [0,1] → vertical stripe frequency (0=none)
  [8]  grain        : float [0,1] → grain/noise strength (0=none)

32×32 tuning notes vs 64×64
----------------------------
  size range     : 10–30px → 5–13px  (keeps shapes inside canvas with margin)
  stripe band    : half//3 → fixed 2–4px range, floor of 2px so stripes
                   are always at least 2px wide and visible
  grain range    : ±80     → ±40     (±80 obliterates detail at 32px)
  stripe/grain   : pixel loops replaced with vectorised numpy ops

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
  python dataset_32.py --output_dir data_32 --n_samples 50000
  python dataset_32.py --cache --output_dir data_32
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
    "v_stripe", "grain",
]
VECTOR_DIM = 9
RESOLUTION = (32, 32)

# 32×32 rendering constants
HALF_MIN  =  5   # px — smallest shape half-radius
HALF_MAX  = 13   # px — largest shape half-radius (leaves ~3px margin on each side)
BAND_W    =  3   # px — fixed stripe band width (3px per stripe + 3px gap at 32px)
GRAIN_AMP = 40   # max ± grain offset per channel (was 80 at 64px)


# ---------------------------------------------------------------------------
# Sample vector
# ---------------------------------------------------------------------------

@dataclass
class SampleVector32:
    shape_id: int    # 0=triangle, 1=square, 2=circle
    r: float         # 0–1
    g: float         # 0–1
    b: float         # 0–1
    size: float      # 0–1
    v_stripe: float  # 0–1
    grain: float     # 0–1

    @property
    def shape_name(self) -> str:
        return SHAPE_NAMES[self.shape_id]

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        r = int(self.r * 255)
        g = int(self.g * 255)
        b = int(self.b * 255)
        # Floor: prevents near-black shapes
        MIN_CH = 80
        r, g, b = max(r, MIN_CH), max(g, MIN_CH), max(b, MIN_CH)
        # Ceiling: prevents near-white shapes on white background
        if r + g + b > 630:
            f = 630 / max(r + g + b, 1)
            r, g, b = int(r * f), int(g * f), int(b * f)
        return (r, g, b)

    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> "SampleVector32":
        rng = rng or random.Random()
        return cls(
            shape_id=rng.randint(0, 2),
            r=rng.random(), g=rng.random(), b=rng.random(),
            size=rng.random(),
            v_stripe=rng.random(),
            grain=rng.random(),
        )

    def to_list(self) -> list:
        one_hot = [0.0, 0.0, 0.0]
        one_hot[self.shape_id] = 1.0
        return one_hot + [
            self.r, self.g, self.b, self.size,
            self.v_stripe, self.grain,
        ]

    @classmethod
    def from_list(cls, vec: list) -> "SampleVector32":
        return cls(
            shape_id=int(np.argmax(vec[:3])),
            r=vec[3], g=vec[4], b=vec[5],
            size=vec[6],
            v_stripe=vec[7],
            grain=vec[8],
        )


# ---------------------------------------------------------------------------
# Image generator
# ---------------------------------------------------------------------------

class ImageGenerator32:
    """
    Generates a single 32×32 PIL Image from a SampleVector32.

    Pipeline
    --------
    1. White background
    2. Fill shape with base colour
    3. Apply vertical stripes (vectorised)
       stripe colour = 40% of base
    4. Apply grain on top (vectorised)
    """

    def generate(self, vec: SampleVector32, seed: int = 0) -> Image.Image:
        W, H = RESOLUTION
        color = vec.color_rgb
        half  = HALF_MIN + int(vec.size * (HALF_MAX - HALF_MIN))

        arr  = np.full((H, W, 3), 255, dtype=np.uint8)
        mask = self._make_mask(vec.shape_id, W // 2, H // 2, half, W, H)
        arr[mask] = color

        if vec.v_stripe > 0.05:
            self._apply_stripes(arr, mask, color, vec.v_stripe)

        if vec.grain > 0.05:
            self._apply_grain(arr, mask, vec.grain, seed)

        return Image.fromarray(arr)

    @staticmethod
    def _make_mask(shape_id, cx, cy, half, W, H) -> np.ndarray:
        img = Image.new("L", (W, H), 0)
        d   = ImageDraw.Draw(img)
        if shape_id == 0:
            ht  = int(half * math.sqrt(3))
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
        return np.array(img) > 0

    @staticmethod
    def _apply_stripes(arr, mask, color, v_stripe):
        """
        Fixed-width vertical stripes.

        v_stripe [0,1] controls darkness only:
          0.0 → stripe colour = base colour (invisible)
          0.5 → stripe colour = 50% of base (medium dark)
          1.0 → stripe colour = black (0, 0, 0)

        Band width is fixed at BAND_W px so stripes are always
        clearly legible regardless of v_stripe value.
        """
        W      = arr.shape[1]
        # darkness: 0 = same as base, 1 = black
        factor = 1.0 - v_stripe
        stripe = np.clip(np.array(color, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
        v_dark = (np.arange(W) // BAND_W) % 2 == 1   # (W,) alternating bands
        apply  = mask & v_dark[None, :]               # (H, W) inside shape only
        arr[apply] = stripe

    @staticmethod
    def _apply_grain(arr, mask, grain_strength, seed):
        """Vectorised grain: add per-pixel RGB noise inside the shape."""
        rng   = np.random.RandomState(seed)
        amp   = int(grain_strength * GRAIN_AMP)
        if amp == 0:
            return
        H, W  = arr.shape[:2]
        noise = rng.randint(-amp, amp + 1, (H, W, 3), dtype=np.int32)
        noisy = np.clip(arr.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        arr[mask] = noisy[mask]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _render_worker(args: tuple) -> tuple:
    i, vec_list, img_dir, image_format = args
    vec   = SampleVector32.from_list(vec_list)
    img   = ImageGenerator32().generate(vec, seed=i)
    ext   = "png" if image_format.upper() == "PNG" else "jpg"
    fname = f"{i:06d}_{vec.shape_name}.{ext}"
    save_kwargs = {"quality": 95} if ext == "jpg" else {}
    img.save(Path(img_dir) / fname, format=image_format.upper(), **save_kwargs)
    return i, fname, vec_list


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig32:
    output_dir:   str            = "data_32"
    n_samples:    int            = 20000
    resolution:   Tuple[int,int] = (32, 32)
    seed:         Optional[int]  = 42
    image_format: str            = "PNG"
    num_workers:  int            = 0
    verbose:      bool           = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolution"] = list(d["resolution"])
        return d


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

class DatasetGenerator32:
    """
    Generates a 32×32 geometric shape dataset.

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
        output_dir:   str           = "data_32",
        n_samples:    int           = 20000,
        seed:         Optional[int] = 42,
        image_format: str           = "PNG",
        num_workers:  int           = 0,
        verbose:      bool          = True,
    ):
        self.cfg  = DatasetConfig32(
            output_dir=output_dir, n_samples=n_samples,
            seed=seed, image_format=image_format,
            num_workers=num_workers, verbose=verbose,
        )
        self._rng = random.Random(seed)

    def generate(self) -> np.ndarray:
        out_dir = Path(self.cfg.output_dir)
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        N        = self.cfg.n_samples
        job_args = [
            (i, SampleVector32.random(self._rng).to_list(),
             str(img_dir), self.cfg.image_format)
            for i in range(N)
        ]

        nw = cpu_count() if self.cfg.num_workers == -1 else self.cfg.num_workers
        results = (
            self._generate_parallel(job_args, nw) if nw > 0
            else self._generate_sequential(job_args)
        )
        results.sort(key=lambda x: x[0])

        vec_arr   = np.array([r[2] for r in results], dtype=np.float32)
        filenames = [r[1] for r in results]

        mmap = np.memmap(out_dir / "vectors.npy", dtype=np.float32,
                         mode="w+", shape=(N, VECTOR_DIM))
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
                "r / g / b":   "linear [0,1]; floor 80/255, ceiling 630/765",
                "size":        f"linear [0,1]; half-size maps to {HALF_MIN}–{HALF_MAX}px",
                "v_stripe":    f"vertical stripe darkness [0,1]; 0=invisible, 1=black; width fixed at {BAND_W}px",
                "grain":       f"pixel noise ±{GRAIN_AMP} per channel",
            },
        }, indent=2))
        (out_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))

        if self.cfg.verbose:
            counts = {
                "triangle": int((vec_arr[:, 0] == 1).sum()),
                "square":   int((vec_arr[:, 1] == 1).sum()),
                "circle":   int((vec_arr[:, 2] == 1).sum()),
            }
            ws = f"{nw} workers" if nw > 0 else "single-process"
            print(f"\n✓ {N} images  →  {out_dir}/images/  ({ws})")
            print(f"✓ vectors     →  {out_dir}/vectors.npy  {vec_arr.shape} float32")
            print(f"  Shape counts: {counts}")

        return vec_arr

    def _generate_sequential(self, job_args):
        results, N = [], len(job_args)
        for args in job_args:
            i, fname, vec_list = _render_worker(args)
            results.append((i, fname, vec_list))
            if self.cfg.verbose and (i % max(1, N // 20) == 0):
                print(f"  [{i+1:>{len(str(N))}}/{N}] {fname}")
        return results

    def _generate_parallel(self, job_args, num_workers):
        N, results = len(job_args), []
        width = len(str(N))
        with Pool(processes=num_workers) as pool:
            for i, fname, vec_list in pool.imap_unordered(
                _render_worker, job_args,
                chunksize=max(1, N // (num_workers * 4))
            ):
                results.append((i, fname, vec_list))
                if self.cfg.verbose and (len(results) % max(1, N // 20) == 0):
                    print(f"  [{len(results):>{width}}/{N}] {fname}")
        return results


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ShapeDataset32:
    """
    Loads 32×32 shape images and conditioning vectors.

    Keeps images in RAM (32×32×3 uint8 = 3kb/image; 20k images = ~60MB).

    Returns per __getitem__
    -----------------------
    image  : torch.Tensor (3, 32, 32) float32 normalised to [-1, 1]
    vector : torch.Tensor (9,)        float32
    """

    def __init__(self, root, max_samples=None):
        self.root = Path(root)
        images_np = np.load(self.root / "images_cached.npy")  # (N, 32, 32, 3)
        if max_samples is not None:
            images_np = images_np[:max_samples]
        images_t     = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
        del images_np
        self._images = (images_t / 255.0 - 0.5) / 0.5
        del images_t
        self._mmap   = np.memmap(
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
    """Convert PNG images to a single numpy array. Run once after generating."""
    root      = Path(root)
    filenames = (root / "filenames.txt").read_text().splitlines()
    print(f"Building cache for {len(filenames)} images...")
    arr = np.stack([
        np.array(Image.open(root / "images" / f).convert("RGB"), dtype=np.uint8)
        for f in filenames
    ])
    np.save(root / "images_cached.npy", arr)
    mb = arr.nbytes / 1024 ** 2
    print(f"Saved → {root / 'images_cached.npy'}  shape: {arr.shape}  ({mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a 32×32 geometric shape dataset.")
    parser.add_argument("--output_dir",  default="data_32",  type=str)
    parser.add_argument("--n_samples",   default=20000,      type=int)
    parser.add_argument("--seed",        default=42,         type=int)
    parser.add_argument("--format",      default="PNG",      choices=["PNG", "JPEG"])
    parser.add_argument("--num_workers", default=0,          type=int,
                        help="0=single-process, -1=all cores")
    parser.add_argument("--cache",       action="store_true",
                        help="Build images_cached.npy instead of generating")
    args = parser.parse_args()

    if args.cache:
        build_cache(args.output_dir)
    else:
        gen = DatasetGenerator32(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            image_format=args.format,
            num_workers=args.num_workers,
            verbose=True,
        )
        gen.generate()
        print(f"\nNow build the cache:")
        print(f"  python dataset_32.py --cache --output_dir {args.output_dir}")