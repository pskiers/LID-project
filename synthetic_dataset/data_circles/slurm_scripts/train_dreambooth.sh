#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=2880
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --output=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source new_env/bin/activate
cd diffusers/examples/dreambooth/
export SSL_CERT_FILE=/net/home/plgrid/plgpawel269/cacert.pem


export MODEL_NAME="stabilityai/sdxl-turbo"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl-turbo"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --cache_dir /net/scratch/hscra/plgrid/plgpawel269/LID-project/model_cache