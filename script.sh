#!/bin/bash

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install gsplat
cd submodules/gsplat/examples
pip install -r requirements.txt
git clone https://github.com/rmbrualla/pycolmap.git
cd pycolmap
vim pyproject.toml
mv pycolmap/ pycolmap2/
python3 -m pip install -e .
cd ~/HunyuanWorld-Mirror 
mkdir houtput
cp -r /mnt/temp-data-volume/video ./houtput
python infer.py --input_path ./houtput/video --output_path ./houtputOpt --save_colmap --save_gs

