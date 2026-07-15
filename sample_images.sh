#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source env/bin/activate


python sample_images_with_prompts.py \
    --prompts_json "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/art_prompt.json" \
    --output_dir "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/data/sdxl_art-40" \
    --model_id "sdxl" \
    --n_samples_per_prompt 10000 \
    --batch_size 16 \
    --base_seed 4 \
    --guidance_scale 7.0 \
    --num_inference_steps 40
    