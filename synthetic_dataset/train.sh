#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
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

#python collect_gradients.py --multiple_anchors
python train_pixel_acc.py
#python dataset_64_update.py --output_dir data_64_no_text --features shape,color,size --n_samples 100000