import os
import random
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


def mean_pairwise_cosine_chunked(embeddings, chunk=1024):
    """Mean of 1 - cos over all unique pairs, without materializing the N x N
    matrix or torch.triu_indices.

    At N=10,000 the full matrix is 400 MB and triu_indices allocates two
    50M-element int64 tensors on top of it (~800 MB more), which is the
    usual OOM point. Row-blocking keeps peak memory at chunk x N.
    """
    n = embeddings.shape[0]
    total, count = 0.0, 0
    for i in range(0, n, chunk):
        block = embeddings[i:i + chunk] @ embeddings.t()        # (b, N)
        b = block.shape[0]
        for r in range(b):
            # only the strict upper triangle: columns after this row's index
            row = block[r, i + r + 1:]
            if row.numel():
                total += (1.0 - row).sum().item()
                count += row.numel()
    return total / max(count, 1), count


def sampled_proper_dreamsim(dataset, ds_model, ds_preprocess, device,
                            n_pairs=5000, seed=0, batch=32):
    """Mean DreamSim distance over randomly sampled pairs, using the model's
    own distance ds_model(a, b) rather than cosine on the embeddings.

    Sampling rather than all-pairs because ds_model is a forward pass per
    pair: 10,000 images is 50M pairs. 5,000 pairs gives a standard error
    around 0.003 on a mean of ~0.3, well below the differences being
    compared -- and the paper itself samples 2,000 pairs per concept for
    its own before/after table.
    """
    rng = random.Random(seed)
    n = len(dataset)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            pairs.append((a, b))

    dists = []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch), desc="DreamSim (proper, sampled)"):
            chunk = pairs[i:i + batch]
            ta = torch.cat([ds_preprocess(dataset[a]) for a, _ in chunk]).to(device)
            tb = torch.cat([ds_preprocess(dataset[b]) for _, b in chunk]).to(device)
            d = ds_model(ta, tb)
            dists.append(d.detach().float().cpu().reshape(-1))
    return torch.cat(dists)


def compare_distance_definitions(dataset, ds_model, ds_preprocess, device,
                                 n_pairs=200, seed=0):
    """Both distance definitions on the same sampled pairs.

    The question this answers: is `1 - cosine` on DreamSim embeddings a
    usable stand-in for ds_model(a, b)?

      - means close                  -> the fast path's numbers are fine as they are
      - means differ, Spearman ~1    -> rankings across concepts and lambda values
                                        hold, but absolute values are not comparable
                                        to the paper's; say so when reporting
      - Spearman low                 -> the fast path is measuring something else;
                                        use --dreamsim_mode proper
    """
    rng = random.Random(seed)
    n = len(dataset)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            pairs.append((a, b))

    proper, cosine = [], []
    with torch.no_grad():
        for a, b in tqdm(pairs, desc="Comparing distance definitions"):
            ia = ds_preprocess(dataset[a]).to(device)
            ib = ds_preprocess(dataset[b]).to(device)
            proper.append(float(ds_model(ia, ib).reshape(-1)[0]))
            ea = F.normalize(ds_model.embed(ia), p=2, dim=1)
            eb = F.normalize(ds_model.embed(ib), p=2, dim=1)
            cosine.append(float(1.0 - (ea @ eb.t()).reshape(-1)[0]))

    p, c = np.array(proper), np.array(cosine)
    pearson = float(np.corrcoef(p, c)[0, 1])
    # Spearman via the correlation of ranks -- no scipy dependency
    spearman = float(np.corrcoef(p.argsort().argsort(), c.argsort().argsort())[0, 1])

    print("\n" + "=" * 62)
    print(f"DISTANCE DEFINITION COMPARISON  ({len(pairs)} random pairs)")
    print("-" * 62)
    print(f"  ds_model(a, b)   mean={p.mean():.4f}  sd={p.std():.4f}  "
          f"[{p.min():.4f}, {p.max():.4f}]")
    print(f"  1 - cos(emb)     mean={c.mean():.4f}  sd={c.std():.4f}  "
          f"[{c.min():.4f}, {c.max():.4f}]")
    print(f"  mean ratio proper/cosine = {p.mean() / max(c.mean(), 1e-12):.3f}")
    print(f"  Pearson  r = {pearson:.4f}")
    print(f"  Spearman r = {spearman:.4f}")
    if abs(p.mean() - c.mean()) < 0.02:
        print("  -> means agree: the fast path is fine as it stands")
    elif spearman > 0.95:
        print("  -> means differ but ranking is preserved: your own comparisons")
        print("     hold, absolute values are not comparable to the paper's")
    else:
        print("  -> ranking differs too: use --dreamsim_mode proper")
    print("=" * 62)
    return {"pearson": pearson, "spearman": spearman,
            "proper_mean": float(p.mean()), "cosine_mean": float(c.mean())}


def main():
    parser = argparse.ArgumentParser(description="Efficient All-Pairs Diversity & Alignment")
    parser.add_argument("--folder", type=str, required=True, help="Folder with images")
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt used for generation. Only CLIP needs it; "
                             "otherwise it is just a label on the report.")
    parser.add_argument("--max_images", type=int, default=10000, help="Limit total images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extraction")
    parser.add_argument("--clip", action="store_true", default=False,
                        help="also compute CLIP alignment. Off by default: the "
                             "README points at Taited/clip-score for CLIP, so "
                             "this section is opt-in rather than something you "
                             "get by accident.")
    parser.add_argument(
        "--dreamsim_mode", default="cosine", choices=["cosine", "proper", "both"],
        help="cosine (default): 1 - cos on DreamSim embeddings, all pairs, fast. "
             "proper: ds_model(a, b) on sampled pairs, which is DreamSim's own "
             "distance. both: run each and report the difference.")
    parser.add_argument("--n_pairs", type=int, default=5000,
                        help="pairs sampled for --dreamsim_mode proper")
    parser.add_argument("--compare_pairs", type=int, default=200,
                        help="pairs used for the head-to-head in --dreamsim_mode both")
    parser.add_argument("--pair_seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=1024,
                        help="row block for the all-pairs cosine computation")

    args = parser.parse_args()
    if args.clip and not args.prompt:
        parser.error("--clip needs --prompt (the text side of the alignment score)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Dataset
    dataset = ImageFolderDataset(args.folder, args.max_images)
    if len(dataset) < 2:
        print("Need at least 2 images to calculate diversity.")
        return
    print(f"Processing {len(dataset)} images...")

    avg_clip_score = avg_clip_dist = None

    # ==========================================
    # METRIC 2: CLIP Alignment
    # ==========================================
    if args.clip:
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

        avg_clip_score = alignment_scores.mean().item()
        avg_clip_dist = 1.0 - avg_clip_score

        # CLIP is done with; free it before DreamSim loads rather than after,
        # so the two models are never resident at the same time
        del clip_model, img_embeds
        torch.cuda.empty_cache()

    # ==========================================
    # METRIC 1: DreamSim Diversity
    # ==========================================
    print("\n[1/2] Calculating DreamSim Diversity...")
    ds_model, ds_preprocess = dreamsim(pretrained=True, device=device)

    cosine_diversity = proper_diversity = None
    n_pairs_cosine = n_pairs_proper = 0
    comparison = None

    if args.dreamsim_mode in ("cosine", "both"):
        def get_dreamsim_embeds(images):
            processed_imgs = [ds_preprocess(img) for img in images]
            tensors = torch.cat(processed_imgs, dim=0).to(device)
            return ds_model.embed(tensors)

        ds_embeddings = get_all_embeddings(
            dataset, get_dreamsim_embeds, args.batch_size, device, "DreamSim Embeddings")
        ds_embeddings = F.normalize(ds_embeddings.to(device), p=2, dim=1)
        cosine_diversity, n_pairs_cosine = mean_pairwise_cosine_chunked(
            ds_embeddings, chunk=args.chunk)
        del ds_embeddings
        torch.cuda.empty_cache()

    if args.dreamsim_mode in ("proper", "both"):
        d = sampled_proper_dreamsim(dataset, ds_model, ds_preprocess, device,
                                    n_pairs=args.n_pairs, seed=args.pair_seed,
                                    batch=args.batch_size)
        proper_diversity = d.mean().item()
        n_pairs_proper = d.numel()
        # standard error, so it is obvious whether a gap between two runs is
        # larger than the sampling noise
        proper_se = d.std().item() / np.sqrt(max(n_pairs_proper, 1))

    if args.dreamsim_mode == "both":
        comparison = compare_distance_definitions(
            dataset, ds_model, ds_preprocess, device,
            n_pairs=args.compare_pairs, seed=args.pair_seed)

    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "=" * 62)
    print(f"FINAL RESULTS FOR: '{args.prompt}'")
    print(f"Folder: {args.folder}")
    print(f"Images Processed: {len(dataset)}")
    print("-" * 62)
    if cosine_diversity is not None:
        print(f"DreamSim, 1 - cos on embeddings (all {n_pairs_cosine:,} pairs): "
              f"{cosine_diversity:.4f}")
    if proper_diversity is not None:
        print(f"DreamSim, ds_model(a,b) ({n_pairs_proper:,} sampled pairs):      "
              f"{proper_diversity:.4f}  (se {proper_se:.4f})")
    if comparison is not None:
        print(f"  ratio proper/cosine on matched pairs: "
              f"{comparison['proper_mean'] / max(comparison['cosine_mean'], 1e-12):.3f}"
              f"   Spearman {comparison['spearman']:.4f}")
    if avg_clip_score is not None:
        print("-" * 62)
        print(f"CLIP Score    (Higher = Better align):  {avg_clip_score:.4f}")
        print(f"CLIP Distance (Lower  = Better align):  {avg_clip_dist:.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()