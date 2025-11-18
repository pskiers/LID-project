import os
import json
import requests
from tqdm.auto import tqdm
from datasets import load_dataset
from PIL import Image

def download_laion_aesthetic(n_images=500, output_dir="laion_aesthetic", captions_file="captions.json"):
    # 1) Prepare output directories/files
    os.makedirs(output_dir, exist_ok=True)
    captions_path = os.path.join(output_dir, captions_file)
    captions_dict = {}

    # 2) Stream the dataset
    ds = load_dataset(
        "laion/laion2B-en-aesthetic",
        split="train",           # the dataset only has a 'train' split
        streaming=True
    )

    # 3) Shuffle & take N items
    #    (buffer_size controls randomness; adjust as needed)
    # stream = ds.shuffle(buffer_size=50_000)#.take(n_images)

    # 4) Download each image
    for idx, item in enumerate(tqdm(ds, desc="Downloading images")):
        if len(captions_dict) >= n_images:
            break
        url = item.get("URL")
        caption = item.get("TEXT")
        width = item.get("WIDTH")
        height = item.get("HEIGHT")
        if not url or not caption or not width or not height:
            continue
        if int(width) != 1024 or int(height) != 1024:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            # skip on any download error
            print(f"  [!] Skipping #{idx}: {e}")
            continue

        # Guess file extension
        content_type = resp.headers.get("content-type", "")
        ext = content_type.split("/")[-1].split(";")[0] or "jpg"
        # Some URLs may mis-report; force jpg if weird
        if ext.lower() not in ("jpg","jpeg","png","webp","gif"):
            ext = "jpg"

        # Build filename and save
        filename = f"{idx:04d}.{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        try:
            img = Image.open(filepath).convert("RGB")
            if img.width != 1024 or img.height != 1024:
                raise ValueError(f"Image {filename} has unexpected size: {img.size}")
        except Exception:
            os.remove(filepath)
            continue
        # Record caption
        captions_dict[filename] = caption


    # 5) Write captions to JSON
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Downloaded {len(captions_dict)} images.")
    print(f"Images + captions saved in `{output_dir}/`")


def download_elsa(models: list[str], n_images=500, output_dir="elsa", captions_file="captions.json"):
    models += ["real"]
    # 1) Prepare output directories/files
    os.makedirs(output_dir, exist_ok=True)
    for model in models:
        os.makedirs(os.path.join(output_dir, model), exist_ok=True)
    captions_paths = {model: os.path.join(output_dir, model, captions_file) for model in models}
    all_captions = {model: {} for model in models}

    # 2) Stream the dataset
    ds = load_dataset(
        "elsaEU/ELSA_D3",
        split="train",           # the dataset only has a 'train' split
        streaming=True
    )

    # 3) Shuffle & take N items
    #    (buffer_size controls randomness; adjust as needed)
    # stream = ds.shuffle(buffer_size=50_000)#.take(n_images)

    # 4) Download each image
    for idx, item in enumerate(tqdm(ds, desc="Downloading images")):
        if len(all_captions["real"]) >= n_images:
            break
        url = item.get("url")
        caption = item.get("positive_prompt")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            # skip on any download error
            print(f"  [!] Skipping #{idx}: {e}")
            continue

        # Guess file extension
        content_type = resp.headers.get("content-type", "")
        ext = content_type.split("/")[-1].split(";")[0] or "jpg"
        # Some URLs may mis-report; force jpg if weird
        if ext.lower() not in ("jpg","jpeg","png","webp","gif"):
            ext = "jpg"

        # Build filename and save
        filename = f"{idx:04d}.{ext}"
        filepath = os.path.join(output_dir, "real", filename)
        with open(filepath, "wb") as f:
            f.write(resp.content)
        try:
            img = Image.open(filepath).convert("RGB")
            if img.width > 1024 or img.height > 1024:
                raise ValueError(f"Image {filename} has unexpected size: {img.size}")
        except Exception:
            os.remove(filepath)
            continue
        # Record caption
        all_captions["real"][filename] = caption

        for i in range(4):
            if item.get(f"model_gen{i}") in models:
                img = item.get(f"image_gen{i}", None)
                if img is None:
                    continue
                if img.width > 1024 or img.height > 1024:
                    continue
                filepath = os.path.join(output_dir, item.get(f"model_gen{i}"), filename)
                img.save(filepath)
                all_captions[item.get(f"model_gen{i}")][filename] = caption

    # 5) Write captions to JSON
    for model in models: 
        with open(captions_paths[model], "w", encoding="utf-8") as f:
            json.dump(all_captions[model], f, ensure_ascii=False, indent=2)

    print(f"\nDone! Downloaded {len(all_captions['real'])} images.")
    print(f"Images + captions saved in `{output_dir}/`")



if __name__ == "__main__":
    # download_laion_aesthetic(
    #     n_images=500,
    #     output_dir="data/laion-2b-1024x1024",
    #     captions_file="captions.json"
    # )
    download_elsa(
        models=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-3-medium-diffusers"],
        n_images=500,
        output_dir="data/elsa",
        captions_file="captions.json"
    )