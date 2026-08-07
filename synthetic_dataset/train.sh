#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=plgideascv1cl-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source env/bin/activate
cd synthetic_dataset
# cd diffusers/examples/unconditional_image_generation
# cd universal-diffsae
export SSL_CERT_FILE=/net/home/plgrid/plgekaczmarczyk/cacert.pem


python collect_gradients.py --data_dir outputs/samples_for_grads/circle_no_text_retrained --out_dir outputs/gradients/t_mult_circle_retrained --cond_input_dim 7 --prompt 0 0 1 0.5 0.5 0.5 0.5 --checkpoint  /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/checkpoints/outputs_64_acc_no_text/checkpoint-epoch-0200/
#python train_pixel_acc.py
#python collect_gradients.py
#python dataset_64_update.py --output_dir data_64_no_text --features shape,color,size --n_samples 100000
#python vaease.py --grads /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/synthetic_dataset/outputs/gradients/t_mult_10k --output_dir runs/vaease_run1