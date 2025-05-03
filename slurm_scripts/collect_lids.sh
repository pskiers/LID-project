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
source env/bin/activate
cd diffusion_memorization
export SSL_CERT_FILE=/net/home/plgrid/plgpawel269/cacert.pem


python store_attributions.py
