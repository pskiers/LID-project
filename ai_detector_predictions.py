"""
ai_detector_predictions.py

Usage:
    ai_detector_predictions.py /path/to/mapping.json /path/to/images_dir /path/to/output_predictions.json

This script requires:
    pip install transformers torch pillow tqdm

By default it uses the Hugging Face model "Ateeqq/ai-vs-human-image-detector".
Change MODEL_NAME to try a different HF model (e.g. "prithivMLmods/deepfake-detector-model-v1").
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

# --- CONFIG ---
MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"  # default recommendation; change if you prefer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------


def load_model_and_processor(model_name: str):
    """
    Loads an image classification model + processor from Hugging Face.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return processor, model


def get_ai_label_id(config) -> Optional[int]:
    """
    Inspect model.config.id2label (if available) and return the label id
    that likely corresponds to 'ai-generated' or 'fake' class.
    If not found, return None (caller will fall back to class index 1).
    """
    id2label = getattr(config, "id2label", None)
    if not id2label:
        return None
    # Look for commonly used substrings indicating AI/fake class
    ai_keywords = ["fake", "ai", "synthetic", "generated", "genuine?no", "bot"]
    for idx, label in id2label.items():
        if any(k in str(label).lower() for k in ai_keywords):
            try:
                return int(idx)
            except Exception:
                pass
    # If not found, attempt heuristics: if two-class, check label text for 'real' and pick the other
    labels = [str(l).lower() for _, l in sorted(id2label.items(), key=lambda x: int(x[0]))]
    if len(labels) == 2:
        if "real" in labels[0] and "fake" not in labels[0]:
            return 1
        if "real" in labels[1] and "fake" not in labels[1]:
            return 0
    return None


def predict_image_probability(
    image_path: Path,
    processor,
    model,
    ai_label_id: Optional[int] = None,
    batch_size: int = 1
) -> float:
    """
    Returns probability (0..1) that the image is AI-generated (as detected by the model).
    If ai_label_id is None, we fallback to class index 1 (common binary mapping).
    """
    img = Image.open(image_path).convert("RGB")
    # Single-image processing
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, num_labels)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # numpy array (num_labels,)
    num_labels = probs.shape[0]

    if ai_label_id is None:
        # fallback heuristics
        if num_labels == 2:
            ai_id = 1
        else:
            # if >2 classes, choose the class with maximum probability that contains common keywords
            ai_id = int(probs.argmax())
    else:
        ai_id = int(ai_label_id)
        if ai_id < 0 or ai_id >= num_labels:
            # invalid id, fallback
            ai_id = int(probs.argmax())

    return float(probs[ai_id])


def batch_predict_and_save(
    mapping_json_path: str,
    images_folder: str,
    output_json_path: str,
    model_name: str = MODEL_NAME
) -> Dict[str, float]:
    """
    mapping_json_path: path to JSON with {"filename": "prompt", ...}
    images_folder: folder containing the image files named as in the JSON keys
    output_json_path: where to save {"prompt_or_prompt#n": probability}
    Returns the dictionary of predictions.
    """
    # load mapping
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        filename_to_prompt: Dict[str, str] = json.load(f)

    processor, model = load_model_and_processor(model_name)
    ai_label_id = get_ai_label_id(model.config)

    results = {}
    prompt_counts = {}  # used to disambiguate duplicate prompts (append __1, __2, ...)

    images_folder = Path(images_folder)
    for fname, prompt in tqdm(filename_to_prompt.items(), desc="images"):
        img_path = images_folder / fname
        if not img_path.exists():
            # try some fallback variants (common): filename without path or with different extensions
            # but for now we'll skip with a warning
            print(f"WARNING: file not found: {img_path} (skipping)")
            continue
        try:
            prob = predict_image_probability(img_path, processor, model, ai_label_id=ai_label_id)
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}; skipping.")
            continue

        # ensure unique keys if prompt repeated: append suffix __n
        key = prompt
        count = prompt_counts.get(prompt, 0)
        if count > 0:
            key = f"{prompt}__{count+1}"
        prompt_counts[prompt] = count + 1

        results[key] = prob

    # Save results JSON
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions for {len(results)} items to: {out_path}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python ai_detector_batch.py mapping.json /path/to/images_dir output_predictions.json")
        sys.exit(1)
    mapping_json = sys.argv[1]
    images_dir = sys.argv[2]
    out_json = sys.argv[3]
    batch_predict_and_save(mapping_json, images_dir, out_json, model_name=MODEL_NAME)
