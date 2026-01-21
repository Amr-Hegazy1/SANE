import logging
logging.basicConfig(level=logging.INFO)

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import json
from pathlib import Path

import torch
import ray

from SANE.models.def_AE_trainable import AE_trainable

PATH_ROOT = Path("./")


def main():
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    # resources (tune)
    cpus_per_trial = 10
    gpus_per_trial = 1
    gpus = 1
    cpus = gpus * cpus_per_trial
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    experiment_name = "sane_llm_gemma3"

    config = {}
    config["seed"] = 32
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["device_no"] = 0
    config["training::precision"] = "amp" if torch.cuda.is_available() else "fp32"
    config["trainset::batchsize"] = 16

    # Transformer backbone inside SANE
    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = False  # torch.compile can be unstable on some setups

    # For reference-free canonicalization we can set permutation_number=0.
    config["training::permutation_number"] = 0
    config["testing::permutation_number"] = 0

    # windows
    config["training::windowsize"] = 256

    # dataset
    data_path = Path("data/dataset_llm_gemma3").absolute()
    config["dataset::dump"] = data_path / "dataset.pt"
    config["downstreamtask::dataset"] = None

    # infer max_positions + token dim from dataset_info
    info_path = data_path / "dataset_info_test.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        config["ae:i_dim"] = int(info["tokensize"])
        config["ae:max_positions"] = info["max_positions"]
    else:
        # fallbacks
        config["ae:i_dim"] = 4096
        config["ae:max_positions"] = [20000, 128, 256]

    # SANE AE size
    config["ae:lat_dim"] = 256
    config["ae:d_model"] = 1024
    config["ae:nhead"] = 8
    config["ae:num_layers"] = 8

    # optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 1e-9
    config["optim::scheduler"] = "OneCycleLR"

    # ssl training
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    
    # Skip GPU wait check to avoid blocking on active GPUs
    config["training::skip_gpu_wait"] = True

    config["training::epochs_train"] = 20
    config["training::output_epoch"] = 10
    config["training::test_epochs"] = 1

    # augmentations
    config["trainloader::workers"] = 4
    config["trainset::add_noise_view_1"] = 0.05
    config["trainset::add_noise_view_2"] = 0.05
    config["trainset::noise_multiplicative"] = True
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    config["callbacks"] = []
    config["resources"] = resources_per_trial

    output_dir = PATH_ROOT / "sane_pretraining"
    output_dir.mkdir(parents=True, exist_ok=True)

    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )
    print(f"started ray. dashboard: {context.dashboard_url}")

    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={"training_iteration": config["training::epochs_train"]},
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=output_dir,
        resources_per_trial=resources_per_trial,
    )

    ray.tune.run_experiments(
        experiments=experiment,
        resume=False,
        reuse_actors=False,
        verbose=2,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
