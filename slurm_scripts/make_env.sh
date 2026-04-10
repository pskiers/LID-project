#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2880
#SBATCH --account=plgdynamic2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --output=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgpawel269/LID-project/slurm/stderr/error_%j.err

# IMPORTANT: load the modules for machine learning tasks and libraries
module load ML-bundle/24.06a

cd $SCRATCH/LID-project

# create and activate the virtual environment
python3.11 -m venv env/
source env/bin/activate

# install one of torch versions available at Helios wheel repo
pip install --no-cache torch==2.4.1+cu124.post2
pip install --no-cache torchvision==0.19.0+cu124

# install the rest of requirements, for example via requirements file
pip install --no-cache -r helios_requirements.txt
cd diffusers/examples/dreambooth
pip install --no-cache -r requirements.txt
pip install --no-cache -r requirements_sdxl.txt
pip install --no-cache -r requirements_flax.txt
