import logging
import sys
from pathlib import Path
import torch
import ray
import ray.air # Required for CheckpointConfig
from ray import tune

# Ensure we can import the core SANE modules and our custom dataset
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent)) 

from llm_trainable import LLM_AE_trainable

logging.basicConfig(level=logging.INFO)

# --- Configuration for TinyLlama-1.1B ---
TOKEN_DIM = 6144 
SEQ_LEN = 5632
NUM_LAYERS = 22

PATH_ROOT = Path("./")

def main():
    # --- Resources ---
    gpus_per_trial = 1 if torch.cuda.is_available() else 0
    cpus_per_trial = 4
    
    experiment_name = "sane_llm_tinyllama_neuron_view"
    
    # --- SANE Configuration ---
    config = {}
    config["seed"] = 42
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["gpu_id"] = 0
    config["training::precision"] = "amp"
    
    config["dataset::dump"] = Path("data/llm_zoo/real_llm_dataset.pt").absolute()
    config["trainset::batchsize"] = 8 
    
    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = False 
    
    config["ae:i_dim"] = TOKEN_DIM 
    config["ae:lat_dim"] = 512 
    config["ae:d_model"] = 1024
    config["ae:nhead"] = 8
    config["ae:num_layers"] = 4 
    config["ae:max_positions"] = [SEQ_LEN + 100, NUM_LAYERS + 10, SEQ_LEN + 100]
    
    config["training::windowsize"] = 128 
    config["training::permutation_number"] = 0
    config["trainset::add_noise_view_1"] = 0.05
    config["trainset::add_noise_view_2"] = 0.05
    # LLM-specific loader settings
    config["llm::bucket_size"] = 1024
    
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 1e-5
    config["optim::scheduler"] = "OneCycleLR"
    
    config["training::epochs_train"] = 10
    config["training::output_epoch"] = 2
    config["training::test_epochs"] = 1
    
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.5 
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"

    config["monitor_memory"] = True
    
    # --- Ray Init ---
    # Use absolute path for safety
    output_dir = PATH_ROOT.joinpath("experiments/llm_extension/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ray.init(num_cpus=cpus_per_trial, num_gpus=gpus_per_trial)
    
    # FIXED: Use ray.air.CheckpointConfig
    experiment = tune.Experiment(
        name=experiment_name,
        run=LLM_AE_trainable,
        stop={"training_iteration": config["training::epochs_train"]},
        config=config,
        local_dir=output_dir,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        )
    )
    
    print(f"Starting Training: {experiment_name}")
    print(f"Checkpoints will be saved to: {output_dir}")
    
    tune.run_experiments(experiment, verbose=1)
    
    ray.shutdown()

if __name__ == "__main__":
    main()
