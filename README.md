# ELROND

## Setup environment
```
bash slurm_scripts/make_env_local.sh
```

## Run discovery
### Sample images
```
python sample_images_with_prompts.py \
    --prompts_json "monster_prompt.json" \
    --output_dir data/sdxl_dmd_monster \
    --model_id sdxl-dmd \
    --n_samples_per_prompt 10000 \
    --batch_size 3 \
    --base_seed 4 \
    --guidance_scale 0.0 \
    --num_inference_steps 4
```
### Collect gradients
```
cd diffusion_memorization
python token_subspace.py \
    name=sdxl-dmd-monster \
    model.model_id=sdxl-dmd \
    model.num_inference_steps=4 \
    model.guidance_scale=0 \
    data.prompt="A picture of a monster" \
    data.token="monster" \
    data.timesteps=[4] \
    data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_monster \
    remove_common=false \
    remove_significant_directions=false \
    randomize_pairs=true
```
### Train SAE (optional)
```
cd universal-diffsae
python -m src.scripts.train \
  --dataset_path ../diffusion_memorization/outputs/sdxl-dmd-monster/grads/t4/shards \
  --hookpoints doesnotmatter \
  --lr 0.00003 \
  --lr_warmup_steps 1000 \
  --wandb_project "sae_text_token_grads" \
  --expansion_factor 1 \
  --k 32 \
  --batch_topk false \
  --num_epochs 20 \
  --save_every 1000 \
  --run_name sdxl-dmd-monster-k32-exp-factor-1-notopk \
  --effective_batch_size 512 \
  --seed 31
```
### Sample with random directions
SAE:
```
python sample_with_directions.py \
  name=sdxl-dmd-monster-samples-with-directions \
  prompt="A picture of a monster" \
  token="monster" \
  min_intervention_strenght=50 \
  max_intervention_strenght=60 \
  batch_size=4 \
  directions_path="../universal-diffsae/sae-ckpts/sae_text_token_grads/sdxl-dmd-monster-k32-exp-factor-1-notopk_t4" \
  direction_type="sae"
```
PCA:
```
python sample_with_directions.py \
  name=sdxl-dmd-monster-samples-with-directions \
  prompt="A picture of a monster" \
  token="monster" \
  min_intervention_strenght=30 \
  max_intervention_strenght=40 \
  batch_size=4 \
  directions_path="outputs/sdxl-dmd-monster/grads/t4/shards" \
  direction_type="grad"
```