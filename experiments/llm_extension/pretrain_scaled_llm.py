import logging
import os
import sys
import json
from pathlib import Path
import torch
import ray
import ray.air
from ray import tune

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent)) 

from SANE.models.def_AE_trainable import AE_trainable
from llm_dataset import LLMLayerDataset 

logging.basicConfig(level=logging.INFO)

def main():
    # --- Load Data Info First ---
    dataset_path = Path("data/scaled_llm_zoo/scaled_llm_dataset.pt").absolute()
    meta_path = str(dataset_path).replace(".pt", "_info_test.json")
    
    if not os.path.exists(meta_path):
        print("Run prepare_scaled_zoo.py first!")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    TOKEN_DIM = meta['token_dim']
    # Approximate max sequence length (Intermediate size usually ~2.7x hidden)
    # Qwen 1.8B Inter is ~5500. Let's be safe.
    SEQ_LEN = 16384 
    NUM_LAYERS = 64 # Max layers in likely models (Llama-70B has 80, 7B has 32)

    print(f"Auto-Configuring SANE for Token Dim: {TOKEN_DIM}")

    # --- Resources ---
    gpus_per_trial = 1 if torch.cuda.is_available() else 0
    cpus_per_trial = 4
    experiment_name = "sane_llm_scaled_multiarch"
    
    # --- Config ---
    config = {}
    config["seed"] = 42
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["gpu_id"] = 0
    config["training::precision"] = "amp"
    
    config["dataset::dump"] = dataset_path
    config["trainset::batchsize"] = 4 # Reduce batch size as tokens are bigger (padded)
    
    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = False 
    
    # Auto-set dimensions
    config["ae:i_dim"] = TOKEN_DIM 
    config["ae:lat_dim"] = 1024 # Increased latent dim for multi-arch support
    config["ae:d_model"] = 1024
    config["ae:nhead"] = 8
    config["ae:num_layers"] = 6 # Deeper for more complex zoo
    config["ae:max_positions"] = [SEQ_LEN, NUM_LAYERS, SEQ_LEN]
    
    config["training::windowsize"] = 128 
    config["training::permutation_number"] = 0
    
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 1e-5
    config["optim::scheduler"] = "OneCycleLR"
    
    config["training::epochs_train"] = 15 # More epochs for variety
    config["training::output_epoch"] = 5
    config["training::test_epochs"] = 1
    
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.5 
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    config["monitor_memory"] = True
    
    output_dir = Path("experiments/llm_extension/results_scaled").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ray.init(num_cpus=cpus_per_trial, num_gpus=gpus_per_trial)
    
    experiment = tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={"training_iteration": config["training::epochs_train"]},
        config=config,
        local_dir=str(output_dir),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=2,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        )
    )
    
    tune.run_experiments(experiment, verbose=1)
    ray.shutdown()

if __name__ == "__main__":
    main()