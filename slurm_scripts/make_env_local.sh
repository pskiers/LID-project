python3 -m venv env/
source env/bin/activate

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

pip install --no-cache -r helios_requirements.txt
cd diffusers/examples/dreambooth
pip install --no-cache -r requirements.txt
pip install --no-cache -r requirements_sdxl.txt
pip install --no-cache -r requirements_flax.txt
pip install simple_parsing==0.1.7
pip install einops==0.8.0
pip install natsort==8.4.0
