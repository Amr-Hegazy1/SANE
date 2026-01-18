#!/bin/bash
#SBATCH -w gpu-nvidia
#SBATCH --gres gpu:nvidia:1
#SBATCH -t 10:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

cd /storage/home/amr.hegazy/nobackup/SANE

source /storage/home/amr.hegazy/nobackup/SANE/.venv/bin/activate

# Your commands here
python experiments/llm_extension/pretrain_scaled_llm.py