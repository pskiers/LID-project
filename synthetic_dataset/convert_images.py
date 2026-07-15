# cache_dataset.py
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

root = Path("data_128")  # change to your actual path
filenames = (root / "filenames.txt").read_text().splitlines()

out = []
for fname in tqdm(filenames):
    img = Image.open(root / "images" / fname).convert("RGB")
    out.append(np.array(img, dtype=np.uint8))

np.save(root / "images_cached.npy", np.stack(out))
print(f"Done — saved {len(out)} images, shape {np.stack(out).shape}")