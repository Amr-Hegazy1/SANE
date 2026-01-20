#!/bin/bash
#SBATCH -w gpu-intel-pvc
#SBATCH -t 7:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

cd /storage/home/amr.hegazy/nobackup/SANE

source /storage/home/amr.hegazy/nobackup/SANE/.venv/bin/activate



# Your commands here
python experiments/llm_extension/prepare_scaled_zoo.py