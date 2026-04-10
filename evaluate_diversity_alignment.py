import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from dreamsim import dreamsim
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, max_images=None):
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
        )

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            return Image.new("RGB", (224, 224))


def collate_fn(batch):
    return batch


def get_all_embeddings(dataset, model_func, batch_size, device, desc="Extracting Features"):
    """
    Runs the model over the dataset and returns a single Tensor of all embeddings.
    Shape: (Num_Images, Embedding_Dim)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    all_embeds = []

    print(f"--- {desc} ---")
    with torch.no_grad():
        for batch_imgs in tqdm(dataloader):
            # model_func should take a list of PIL images and return a tensor (Batch, Dim)
            embeds = model_func(batch_imgs)
            all_embeds.append(embeds.cpu())  # Move to CPU to save VRAM during accumulation

    # Stack into one big tensor
    return torch.cat(all_embeds, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Efficient All-Pairs Diversity & Alignment")
    parser.add_argument("--folder", type=str, required=True, help="Folder with images")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for generation")
    parser.add_argument("--max_images", type=int, default=None, help="Limit total images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extraction")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Dataset
    dataset = ImageFolderDataset(args.folder, args.max_images)
    if len(dataset) < 2:
        print("Need at least 2 images to calculate diversity.")
        return
    print(f"Processing {len(dataset)} images...")

    # ==========================================
    # METRIC 2: CLIP Alignment
    # ==========================================
    print("\n[2/2] Calculating CLIP Alignment...")
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    def get_clip_image_embeds(images):
        inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
        outputs = clip_model.get_image_features(**inputs)
        return outputs / outputs.norm(p=2, dim=-1, keepdim=True)

    # Extract Image Embeddings
    img_embeds = get_all_embeddings(dataset, get_clip_image_embeds, args.batch_size, device, "CLIP Image Embeddings")
    img_embeds = img_embeds.to(device)

    # Get Text Embedding (Just one)
    text_inputs = clip_processor(text=[args.prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embed = clip_model.get_text_features(**text_inputs)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)

    # Calculate similarity (Image Embeds @ Text Embed.T)
    # shape: (N, 1)
    alignment_scores = torch.matmul(img_embeds, text_embed.t()).squeeze()
    valid_mask = alignment_scores > 0.0
    alignment_scores = alignment_scores[valid_mask]

    avg_clip_score = alignment_scores.mean().item()
    avg_clip_dist = 1.0 - avg_clip_score

    # ==========================================
    # METRIC 1: DreamSim Diversity (All Pairs)
    # ==========================================
    print("\n[1/2] Calculating DreamSim Diversity...")
    # Initialize DreamSim
    ds_model, ds_preprocess = dreamsim(pretrained=True, device=device)

    # Define helper to preprocess and embed a batch
    def get_dreamsim_embeds(images):
        # 1. Preprocess images
        # ds_preprocess(img) returns shape (1, 3, H, W)
        processed_imgs = [ds_preprocess(img) for img in images]

        # 2. Combine into a single batch
        # torch.cat flattens the list of (1, 3, H, W) into (Batch, 3, H, W)
        tensors = torch.cat(processed_imgs, dim=0).to(device)

        # 3. Get embeddings
        return ds_model.embed(tensors)

    # Extract all embeddings at once
    # Result shape: (N, D)
    ds_embeddings = get_all_embeddings(dataset, get_dreamsim_embeds, args.batch_size, device, "DreamSim Embeddings")
    ds_embeddings = ds_embeddings[valid_mask.to(ds_embeddings.device)]

    # Move embeddings back to GPU for fast matrix math
    ds_embeddings = ds_embeddings.to(device)

    # Normalize embeddings (DreamSim distance is usually cosine distance)
    ds_embeddings = F.normalize(ds_embeddings, p=2, dim=1)

    # Compute Similarity Matrix (N x N)
    # Matrix multiplication: (N, D) @ (D, N) -> (N, N)
    sim_matrix = torch.mm(ds_embeddings, ds_embeddings.t())

    # Convert Similarity to Distance (1 - Sim)
    dist_matrix = 1.0 - sim_matrix

    # We only want the average of the upper triangle (unique pairs, excluding diagonal)
    # Get indices for upper triangle
    # n = len(dataset)
    n = len(ds_embeddings)
    rows, cols = torch.triu_indices(n, n, offset=1)

    average_diversity = dist_matrix[rows, cols].mean().item()

    # Clean up DreamSim to free VRAM for CLIP
    del ds_model, ds_embeddings, sim_matrix, dist_matrix
    torch.cuda.empty_cache()

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS FOR: '{args.prompt}'")
    print(f"Images Processed: {n}")
    print(f"Pairs Calculated: {len(rows)} (All unique pairs)")
    print("-" * 50)
    print(f"DreamSim Diversity (Higher = More Diverse):  {average_diversity:.4f}")
    print(f"CLIP Score         (Higher = Better align):  {avg_clip_score:.4f}")
    print(f"CLIP Distance      (Lower  = Better align):  {avg_clip_dist:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
