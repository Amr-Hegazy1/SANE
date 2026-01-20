#!/bin/bash
#SBATCH -w gpu-nvidia
#SBATCH --gres gpu:nvidia:1
#SBATCH -t 10:00:00
#SBATCH -o output_%j.txt
#SBATCH -e error_%j.txt

cd /storage/home/amr.hegazy/nobackup/SANE

source /storage/home/amr.hegazy/nobackup/SANE/.venv/bin/activate

# Your commands here
python experiments/llm_extension/analyze_llm_embeddings.py --save_json experiments/llm_extension/embedding_metrics.json

# python experiments/llm_extension/analyze_llm_embeddings.py --model_level --chunk_size 4096 --halo_size 128 --save_json experiments/llm_extension/embedding_metrics_model_level.json
