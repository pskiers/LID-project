#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source env/bin/activate
cd diffusion_memorization
# cd diffusers/examples/unconditional_image_generation
# cd universal-diffsae
export SSL_CERT_FILE=/net/home/plgrid/plgekaczmarczyk/cacert.pem


cd diffusion_memorization


#python sample_with_multi_directions.py \
#  name=sdxl-dmd-art-samples-with-directions-land-2-8-in \
#  directions_path="outputs/sdxl-art-pca/grads/t4/shards" \
#  direction_type="grad"
python calculate_fid.py


