#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=plgdiffusion2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --output=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source new_env/bin/activate
cd diffusion_memorization
# cd diffusers/examples/unconditional_image_generation
# cd universal-diffsae
export SSL_CERT_FILE=/net/home/plgrid/plgpawel269/cacert.pem


# python calculate_images_lid.py --config-name img_lids2.yaml \
#     dataset_path=/net/scratch/hscra/plgrid/plgpawel269/LID-project/data/sdxl_dogs_lora_prior_jpg \
#     name=dog-lora-prior-lid-estimates-sdxl-unconditional-ai-prompts-0.05 \
#     conditional=False


# python token_subspace.py \
#     name=sdxl-turbo-pig-remove-mean \
#     model.model_id=stabilityai/sdxl-turbo \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0 \
#     data.prompt="A photograph of the inside of a subway train. There are frogs sitting on the seats. One of them is reading a newspaper. The window shows the river in the background." \
#     data.token="frogs" \
#     data.timesteps=[4,2,1] \
#     data.image_folder=/net/scratch/hscra/plgrid/plgpawel269/LID-project/data/sdxl_turbo_subspace_tokens/frog \
#     remove_common=true \
#     remove_significant_directions=false \
#     randomize_pairs=false

# python token_subspace.py \
#     name=sdxl-lora-prior-dog-astronaut \
#     model.model_id=stabilityai/stable-diffusion-xl-base-1.0 \
#     model.num_inference_steps=50 \
#     model.guidance_scale=7.5 \
#     data.prompt="A sks dog wearing aviator goggles piloting an airplane" \
#     data.token="dog" \
#     data.timesteps=[38,25] \
#     data.image_folder=/net/scratch/hscra/plgrid/plgpawel269/LID-project/data/tokens_lora_prior_preservation/sks_aviator_googles \
#     remove_common=true \
#     remove_significant_directions=false \
#     randomize_pairs=true \
#     max_pairs=4000

# python sample_images_with_prompts.py \
#     --output_dir=/net/scratch/hscra/plgrid/plgpawel269/LID-project/data/concept_complexity \
#     --prompts_json=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/captions3.json \
#     --n_samples_per_prompt=10000 #\
#     # --lora_path=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusers/examples/dreambooth/lora-trained-xl-prior_preservation/pytorch_lora_weights.safetensors
#     # --lora_path=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusers/examples/dreambooth/lora-trained-xl/pytorch_lora_weights.safetensors

# python compare_subspaces.py \
#     --dirA=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-monster-a-lot/grads/t50 \
#     --dirB=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/sdxl-monster-a-lot/grads/t50 \
#     --prompt="A picture of a monster" \
#     --token="monster" \
#     --output_dir=/net/scratch/hscra/plgrid/plgpawel269/LID-project/diffusion_memorization/outputs/interventions/monster_concept_8k_10k


# accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional_small_loop.py \
#   --dataset_name="uoft-cs/cifar100" \
#   --resolution=32 --center_crop --random_flip \
#   --output_dir="cifar100-size-small-1000-steps" \
#   --train_batch_size=256 \
#   --num_epochs=1000 \
#   --gradient_accumulation_steps=1 \
#   --use_ema \
#   --learning_rate=1e-4 \
#   --lr_warmup_steps=500 \
#   --mixed_precision="fp16" \
#   --logger="wandb" \
#   --save_model_epochs 100 \
#   --save_images_epochs 100 \
#   --ddpm_num_steps 1000 \
#   --ddpm_num_inference_steps 1000 \
#   --channels 128 128 \
#   --down_block_types AttnDownBlock2D DownBlock2D \
#   --up_block_types UpBlock2D AttnUpBlock2D \
#   --T 3 \
#   --n 6 \
#   --N_supervision 4

# accelerate launch --mixed_precision="fp16" --num_processes=1 sample.py \
# accelerate launch --mixed_precision="fp16" --multi_gpu sample.py \
#   --model_config_name_or_path cifar100-small-size-small-llm/checkpoint-49000/unet_ema \
#   --resolution=32 \
#   --output_dir="cifar100-small-size-small-llm-ddim250-samples" \
#   --ddpm_num_steps 1000 \
#   --ddpm_num_inference_steps 250 \
#   --num_samples 10000 \
#   --max_bs 5000 \
#   --use_ddim


# accelerate launch --mixed_precision="fp16" --multi_gpu sample_standard.py \
#   --model_config_name_or_path cifar100-standard/checkpoint-49000/unet_ema \
#   --resolution=32 \
#   --output_dir="cifar100-standard-samples" \
#   --ddpm_num_steps 1000 \
#   --ddpm_num_inference_steps 1000 \
#   --num_samples 10000 \
#   --max_bs 250

# python -m src.scripts.train \
#   --dataset_path /home/pskiers/LID-project/diffusion_memorization/outputs/sdxl-toy-a-lot/grads/t50 \
#   --hookpoints doesnotmatter \
#   --lr 0.00003 \
#   --lr_warmup_steps 1000 \
#   --wandb_project "sae_text_token_grads" \
#   --expansion_factor 8 \
#   --k 32 \
#   --batch_topk true \
#   --num_epochs 1000 \
#   --save_every 1000 \
#   --run_name toy-k32-exp-factor-8 \
#   --effective_batch_size 512

# cd ../diffusion_memorization/
# python compare_subspaces.py \
#     --dirs sae:../universal-diffsae/sae-ckpts/sae_text_token_grads/toy-new-32-exp-factor-8-t38_t38 \
#     --prompt="A picture of a toy" \
#     --token="toy" \
#     --sae_low=-6 \
#     --sae_high=0 \
#     --output_dir=outputs/intreventions/toy_concept_sae_k32_exp_factor8_38

# python -m src.scripts.train \
#   --dataset_path /home/pskiers/LID-project/diffusion_memorization/outputs/sdxl-toy-new/grads/t38/shards \
#   --hookpoints doesnotmatter \
#   --lr 0.00003 \
#   --lr_warmup_steps 1000 \
#   --wandb_project "sae_text_token_grads" \
#   --expansion_factor 8 \
#   --k 32 \
#   --batch_topk true \
#   --num_epochs 20 \
#   --save_every 1000 \
#   --run_name toy-new-32-exp-factor-8-t38 \
#   --effective_batch_size 512


# python token_subspace.py \
#     name=sdxl-turbo-monster-2 \
#     model.model_id=stabilityai/sdxl-turbo \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0 \
#     data.prompt="A picture of a monster" \
#     data.token="monster" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_monster \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true


# python token_subspace.py \
#     name=sdxl-dmd-car \
#     model.model_id=sdxl-dmd \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0.0 \
#     data.prompt="A picture of a car" \
#     data.token="car" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_car \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python token_subspace.py \
#     name=sdxl-dmd-monster \
#     model.model_id=sdxl-dmd \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0.0 \
#     data.prompt="A picture of a monster" \
#     data.token="monster" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_monster \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python token_subspace.py \
#     name=sdxl-dmd-dog \
#     model.model_id=sdxl-dmd \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0 \
#     data.prompt="A picture of a dog" \
#     data.token="dog" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_dog \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python token_subspace.py \
#     name=sdxl-dmd-cat \
#     model.model_id=sdxl-dmd \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0 \
#     data.prompt="A picture of a cat" \
#     data.token="cat" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_cat \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python token_subspace.py \
#     name=sdxl-dmd-person \
#     model.model_id=sdxl-dmd \
#     model.num_inference_steps=4 \
#     model.guidance_scale=0 \
#     data.prompt="A picture of a person" \
#     data.token="person" \
#     data.timesteps=[4] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_dmd_person \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python token_subspace.py \
#     name=sdxl-cat \
#     model.num_inference_steps=50 \
#     model.guidance_scale=7.5 \
#     data.prompt="A picture of a cat" \
#     data.token="cat" \
#     data.timesteps=[50] \
#     data.image_folder=/home/pskiers/LID-project/data/sdxl_cat \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true

# python -m src.scripts.train \
#   --dataset_path /home/pskiers/LID-project/diffusion_memorization/outputs/sdxl-dmd-person/grads/t4/shards \
#   --hookpoints doesnotmatter \
#   --lr 0.00003 \
#   --lr_warmup_steps 1000 \
#   --wandb_project "sae_text_token_grads" \
#   --expansion_factor 1 \
#   --k 32 \
#   --batch_topk false \
#   --num_epochs 20 \
#   --save_every 1000 \
#   --run_name sdxl-dmd-person-k32-exp-factor-1-notopk \
#   --effective_batch_size 512 \
#   --seed 31



# python compare_subspaces.py \
#     --dirs sae:../universal-diffsae/sae-ckpts/sae_text_token_grads/sdxl-dmd-person-k32-exp-factor-1-notopk_t4 \
#     --prompt="A picture of a person" \
#     --token="person" \
#     --sae_low=-6 \
#     --sae_high=0 \
#     --output_dir=outputs/intreventions/sdxl-dmd_person_sae_k32_exp_factor1_notopk_0-6-random-3 \
#     --random_n=3 \
#     --model="sdxl-dmd" \
#     --num_samples=8 \
#     --batch_size=4 \
#     --intervention_strengths 0 1 5 10 30 50 70 100 150 200 \
#     --num_interventions=8

# python compare_subspaces.py \
#     --dirs grad:outputs/sdxl-dmd-person/grads/t4/shards \
#     --prompt="A picture of a person" \
#     --token="person" \
#     --sae_low=-6 \
#     --sae_high=0 \
#     --output_dir=outputs/intreventions/sdxl-dmd-person-pca-3-rand \
#     --random_n=3 \
#     --model="sdxl-dmd" \
#     --num_samples=8 \
#     --batch_size=4 \
#     --intervention_strengths 0 1 5 10 30 50 70 100 150 200 \
#     --num_interventions=8

# python compare_subspaces.py \
#     --dirs grad:outputs/sdxl-dmd-cat/grads/t4/shards \
#     --prompt="A picture of a cat" \
#     --token="cat" \
#     --sae_low=-6 \
#     --sae_high=0 \
#     --output_dir=outputs/intreventions/sdxl-cat-random \
#     --random_n=1 \
#     --model="sdxl" \
#     --num_samples=8 \
#     --batch_size=4 \
#     --intervention_strengths 0 30 50 70 100 \
#     --num_interventions=2



# python sample_with_directions.py \
#   name=sdxl-dmd-person-samples-with-directions \
#   prompt="A picture of a person" \
#   token="person" \
#   min_intervention_strenght=50 \
#   max_intervention_strenght= \
#   batch_size=4 \
#   directions_path="../universal-diffsae/sae-ckpts/sae_text_token_grads/sdxl-dmd-person-k32-exp-factor-1-notopk_t4" \
#   direction_type="sae"

# python sample_with_directions.py \
#   name=sdxl-dmd-person-samples-with-directions \
#   prompt="A picture of a person" \
#   token="person" \
#   min_intervention_strenght=30 \
#   max_intervention_strenght=40 \
#   batch_size=4 \
#   directions_path="outputs/sdxl-dmd-person/grads/t4/shards" \
#   direction_type="grad"




# python token_subspace.py \
#     name=sdxl-an-image-of-a-monster \
#     model.num_inference_steps=50 \
#     model.guidance_scale=7.5 \
#     data.prompt="An image of a monster" \
#     data.token="monster" \
#     data.timesteps=[50] \
#     data.image_folder=/data/pskiers/concept_dimention/sdxl_monster \
#     remove_common=false \
#     remove_significant_directions=false \
#     randomize_pairs=true \
#     max_pairs=10000 \
#     shard_size=500


python token_subspace.py \
    name=sdxl-just-monster \
    model.num_inference_steps=50 \
    model.guidance_scale=7.5 \
    data.prompt="monster" \
    data.token="monster" \
    data.timesteps=[50] \
    data.image_folder=/data/pskiers/concept_dimention/sdxl_monster \
    remove_common=false \
    remove_significant_directions=false \
    randomize_pairs=true \
    max_pairs=10000 \
    shard_size=500
