import os
from datasets import load_dataset
from tqdm import tqdm

# Flat directory
output_dir = "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/data/parrot_images"
os.makedirs(output_dir, exist_ok=True)

print("Starting flat download (streaming mode)...")

# Streaming avoids the 429 error by not fetching all metadata at once
ds = load_dataset("parrotzone/sdxl-1.0", split="train", streaming=True, token=os.environ["HF_TOKEN"])

for i, item in enumerate(tqdm(ds, total=8872)):
    try:
        artist_label = item.get('artist', item.get('label', f'style_{i}'))
        # Clean the name to ensure it's a valid filename
        clean_name = "".join([c for c in str(artist_label) if c.isalnum() or c in (' ', '_', '-')]).rstrip()
        clean_name = clean_name.replace(" ", "_")
        
        filename = f"{i:05d}_{clean_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save directly to the flat folder
        item['image'].save(filepath)
        
    except Exception as e:
        print(f"\nError saving image {i}: {e}")
        continue 

print(f"\nFinished! Total images in {output_dir}: {len(os.listdir(output_dir))}")