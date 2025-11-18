import os
import json
import argparse
import torch
from PIL import Image
from typing import Literal
from tqdm.auto import tqdm
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, AutoPipelineForText2Image


def load_prompts(prompts_path: str) -> dict[str, str]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_samples_json(samples: dict[str, str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def chunk_list(lst, size):
    """Yield successive chunks of size `size` from list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def main(
    prompts_json: str = "prompts.json",
    output_dir: str = "sdxl_samples",
    samples_json: str = "samples.json",
    model_id: Literal["sdxl", "sd3", "sd1.4", "sdxl_turbo"] = "sdxl",
    device: str = "cuda:0",
    n_samples_per_prompt: int = 3,
    batch_size: int = 16,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    base_seed: int = 42,
    lora_path: str | None = None,
):
    # 1) Load prompts
    prompts = load_prompts(prompts_json)
    os.makedirs(output_dir, exist_ok=True)

    # 2) Load SDXL pipeline once
    ModelCls, hf_path = {
        "sdxl": (StableDiffusionXLPipeline, "stabilityai/stable-diffusion-xl-base-1.0"),
        "sd3": (StableDiffusion3Pipeline, "stabilityai/stable-diffusion-3-medium-diffusers"),
        "sd1.4": (StableDiffusionPipeline, "CompVis/stable-diffusion-v1-4"),
        "sdxl_turbo": (AutoPipelineForText2Image, "stabilityai/sdxl-turbo"),
    }[model_id]
    pipe = ModelCls.from_pretrained(
        hf_path,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir="./model_cache",
    ).to(device)

    if lora_path is not None:
        print(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()
        pipe.to(device)

    # 3) Prepare flattened list of (prompt, filename_prefix)
    tasks = []  # list of tuples (prompt, filename_prefix)
    if isinstance(prompts, dict):
        for base_name, prompt in prompts.items():
            for i in range(n_samples_per_prompt):
                tasks.append((prompt, f"{base_name}_{i:02d}"))
    elif isinstance(prompts, list):
        for i, prompt in enumerate(prompts):
            for j in range(n_samples_per_prompt):
                tasks.append((prompt, f"sample_{i:04d}_{j:02d}"))
    else:
        raise ValueError("Prompts JSON must be a dict or a list.")

    samples_mapping: dict[str, str] = {}

    # 4) Process in batches
    for batch_idx, batch in enumerate(tqdm(list(chunk_list(tasks, batch_size)), desc="Batches")):
        prompts_batch = [p for p, _ in batch]
        prefixes_batch = [pref for _, pref in batch]

        # Create a generator per sample for reproducibility
        gens = [torch.Generator(device).manual_seed(base_seed + batch_idx * batch_size + idx)
                for idx in range(len(prompts_batch))]

        # 5) Generate batch
        outputs = pipe(
            prompt=prompts_batch,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gens,
        )

        # 6) Save images and record mapping
        for img, prefix in zip(outputs.images, prefixes_batch):
            filename = f"{prefix}.png"
            path = Path(output_dir) / filename
            img.save(path)
            samples_mapping[filename] = prompts_batch[prefixes_batch.index(prefix)]

    # 7) Save the samples.json
    save_samples_json(samples_mapping, os.path.join(output_dir, samples_json))
    print(f"Done! Generated {len(samples_mapping)} images in '{output_dir}/'")
    print(f"Sample mapping saved to '{output_dir}/{samples_json}'")


def convert_images_to_jpg(source_dir, target_dir, samples_json):
    # Make sure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Supported image formats
    valid_extensions = {'.png', '.jpeg', '.jpg', '.bmp', '.webp', '.tiff'}

    for filename in os.listdir(source_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_extensions:
            continue  # skip non-image files

        try:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, f"{name}.jpg")

            with Image.open(src_path) as img:
                # Convert image to RGB if not already
                rgb_img = img.convert("RGB")
                rgb_img.save(dst_path, "JPEG", quality=95)

            print(f"Converted: {filename} -> {dst_path}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

    with open(os.path.join(source_dir, samples_json), "r", encoding="utf-8") as f:
        samples_dict = json.load(f)
    samples_dict_jpg = {f"{os.path.splitext(k)[0]}.jpg": v for k, v in samples_dict.items()}
    with open(os.path.join(target_dir, samples_json), "w", encoding="utf-8") as f:
        json.dump(samples_dict_jpg, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompts_json", type=str, default="prompts.json", help="Path to prompts JSON file.")
    argparser.add_argument("--output_dir", type=str, default="sdxl_samples", help="Directory to save generated images.")
    argparser.add_argument("--samples_json", type=str, default="samples.json", help="Filename for samples mapping JSON.")
    argparser.add_argument("--model_id", type=str, default="sdxl", choices=["sdxl", "sd3", "sd1.4", "sdxl_turbo"], help="Model ID for the diffusion model.")
    argparser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on.")
    argparser.add_argument("--n_samples_per_prompt", type=int, default=3, help="Number of samples to generate per prompt.")
    argparser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    argparser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation.")
    argparser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    argparser.add_argument("--base_seed", type=int, default=42, help="Base seed for random number generation.")
    argparser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights (if any).")
    args = argparser.parse_args()

    main(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        base_seed=args.base_seed,
        lora_path=args.lora_path,
        samples_json=args.samples_json,
        prompts_json=args.prompts_json,
        output_dir=args.output_dir,
        n_samples_per_prompt=args.n_samples_per_prompt,
    )
    # convert_images_to_jpg(args.output_dir, args.output_dir + "_jpg", args.samples_json)
