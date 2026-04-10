# ELROND

## Setup environment
Make a python venv and run:
```bash
bash slurm_scripts/make_env_local.sh
```
If you are using some sort of cluster then you there is a slurm script for making an environment at `slurm_scripts/make_env.sh` (probably won't work for you unless you are using helios, but it still may be easier to modify this one than write it from scratch).

## ELROND directions discovery
### 1. Sample images
Use `sample_images_with_prompts.py`. For more info run:
```bash
python sample_images_with_prompts.py -h
```
Example command for sdxl-dmd:
```bash
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
For more examples you can try searching `slurm_scripts/collect_lids.sh`.
### 2. Collect gradients
Once you have sampled images for a concept collect gradients with `diffusion_memorization/token_subspace.py`. This script is configured with `hydra` so to get your token gradient subspace either change `diffusion_memorization/configs/token_subspace.yaml`, make your own config, or use command line `hydra` utillites.

Some general directions (for more you'll just have to look it up in the code/config - sorry):
* **output path** - in folder `out_dir` a new folder `name` is created and there the gradients are stored
* **model version** - in general (so unless loading model is straight forward unlike sdxl-dmd) specify model class like in `model.model_class` and hugging face checkpoint path in `model.model_id`.
* **device** - this one is tricky. We need to do backprop through the model and the model is usually quite big so one gpu is usually not enough so we need to split it between multiple gpu. If you can fit on a single gpu then just use `model.device_map=null` and be happy. If it does not then you can try `model.device_map=auto` or `model.device_map=balanced`, which may but probably won't be implemented for your model, and may, but probably won't fix your CUDA OOM. If all fails you need to write a custom device map, where you basically explicitly tell `accelerate` exactly which module should go onto what gpu (super fun!!!). What is even more fun is that when you do that you may also get some "tensors on separate devices" errors, and if you do you just have to go and fix them in `diffusers` source code (which is why it is a submodule and why it is installed inplace). So yeah - GL HF doing that. Some general directions for writting those maps - just print out the main modules in your model, spread them more or less equally, and then if you still get CUDA OOM then juggle those modules between gpus and if it still does not help then take the module that is problematic/big and split it between gpus. When you get CUDA OOM you should be able to see in traceback (just be carefull, because you can get it during both forward and backward passes) on which module you got it, so just try to get a bit further - one module/layer/whatever at a time. For some reference - the device map you see in the `diffusion_memorization/configs/token_subspace.yaml` works on 4 RTX5000 (24 GB) with SDXL (so like 3.5B params)
```bash
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
For more examples you can try searching `slurm_scripts/collect_lids.sh`.
### Train SAE (optional)
If you want to decompose the gradiets with SAE then you train it like so:
```bash
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
For more examples you can try searching `slurm_scripts/collect_lids.sh`. Also for help use:
```bash
python -m src.scripts.train -h
```
### Sample with random directions
For some help, just look at the code or config (`diffusion_memorization/configs/sample_with_directions.yaml`)
#### SAE
For SAE you need to have one trained. Example command:
```bash
cd diffusion_memorization
python sample_with_directions.py \
  name=sdxl-dmd-monster-samples-with-directions \
  prompt="A picture of a monster" \
  token="monster" \
  min_intervention_strenght=50 \
  max_intervention_strenght=60 \
  batch_size=4 \
  directions_path="../universal-diffsae/sae-ckpts/sae_text_token_grads/sdxl-dmd-monster-k-32-exp-factor-1-t50-s31_t4" \
  direction_type="sae"
```
For more examples you can try searching `slurm_scripts/collect_lids.sh`.
#### PCA
You just need to have gradients sampled - PCA is done automatically in the script. Example command:
```bash
cd diffusion_memorization
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
For more examples you can try searching `slurm_scripts/collect_lids.sh`.

## Evaluating sampled images
### Mean DreamSim (diversity)
For mean DreamSim distance use `evaluate_diversity_alignment.py`. For help run:
```bash
python evaluate_diversity_alignment.py -h
```
NOTE: DO NOT USE IT FOR CLIP SCORE

### Clip score (text alignment)
We this repo: https://github.com/Taited/clip-score

### Qualitative (see how some directions look like)
For some example directions you can use `diffusion_memorization/compare_subspaces.py`. For help:
```bash
python compare_subspaces.py -h
```
This script can do a bit more than just show how some directions look like (hence the name), but a general example command that just visualises some directions looks like this:
```bash
# For SAE directions
python compare_subspaces.py \
    --dirs grad:outputs/sdxl-dmd-person/grads/t4/shards \
    --prompt="A picture of a person" \
    --token="person" \
    --sae_low=-6 \
    --sae_high=0 \
    --output_dir=outputs/intreventions/sdxl-dmd-person-pca-3-rand \
    --random_n=3 \
    --model="sdxl-dmd" \
    --num_samples=8 \
    --batch_size=4 \
    --intervention_strengths 0 1 5 10 30 50 70 100 150 200 \
    --num_interventions=8

# For PCA directions
python compare_subspaces.py \
    --dirs grad:outputs/sdxl-dmd-cat/grads/t4/shards \
    --prompt="A picture of a cat" \
    --token="cat" \
    --sae_low=-6 \
    --sae_high=0 \
    --output_dir=outputs/intreventions/sdxl-cat-random \
    --random_n=1 \
    --model="sdxl" \
    --num_samples=8 \
    --batch_size=4 \
    --intervention_strengths 0 30 50 70 100 \
    --num_interventions=2
```
For more examples you can try searching `slurm_scripts/collect_lids.sh`.

### Paper plots
You can find most in `diffusion_memorization/paper_plots_scripts`. Most of them were LLM-generated and heavily modified with no regard for nice code, so use at your own risk. Example:
```bash
python -m paper_plots_scripts.make_highlight
```

### Random experiments
You can find most in `diffusion_memorization/random_experiments`. Most of them were LLM-generated and heavily modified with no regard for nice code, so use at your own risk. Example:
```bash
python -m random_experiments.sample_sdxl_and_sd15
```