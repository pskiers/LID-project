#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source env/bin/activate


python sample_images_with_prompts.py \
    --prompts_json "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/troll_prompt.json" \
    --output_dir "/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/data/sdxl-dmd-troll" \
    --model_id "sdxl-dmd" \
    --n_samples_per_prompt 5000 \
    --batch_size 16 \
    --base_seed 80000 \
    --guidance_scale 0 \
    --num_inference_steps 4
    