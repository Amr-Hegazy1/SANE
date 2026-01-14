import torch
import torch.nn as nn
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KernelDensity

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent))

from SANE.models.def_AE_module import AEModule
from llm_dataset import LLMLayerDataset

# Config (Must match training)
TOKEN_DIM = 6144 
SEQ_LEN = 5632

def get_layer_weights(dataset_train, dataset_test, layer_idx):
    """Extracts all samples corresponding to a specific layer ID from both datasets."""
    weights = []
    
    # Helper to search a dataset
    def search_ds(ds):
        found = []
        # Check if properties exist and have entries
        if 'layer_id' not in ds.properties:
            return found
            
        for i in range(len(ds)):
            # ds.properties['layer_id'] is a list of floats
            lid = ds.properties['layer_id'][i]
            if int(lid) == layer_idx:
                found.append(ds[i][0]) # Append weight tokens
        return found

    weights.extend(search_ds(dataset_train))
    weights.extend(search_ds(dataset_test))
    
    if not weights:
        # Fallback: check what layers ARE available
        available_layers = set()
        if 'layer_id' in dataset_train.properties:
            available_layers.update([int(x) for x in dataset_train.properties['layer_id']])
        if 'layer_id' in dataset_test.properties:
            available_layers.update([int(x) for x in dataset_test.properties['layer_id']])
        
        raise ValueError(f"No samples found for layer {layer_idx}. Available layers: {sorted(list(available_layers))}")
        
    return torch.stack(weights)

def compute_svd_spectrum(matrix):
    """Computes singular values of a matrix (flattened tokens to matrix)."""
    # Matrix shape: [Neurons, Inputs] -> [5632, 6144]
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Compute full SVD
    # We only need singular values (S), so compute_uv=False saves massive time
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        # Fallback for stability
        print("Warning: SVD convergence failed, using zero spectrum")
        return np.zeros(10)
        
    # Normalize
    if np.max(s) > 0:
        s = s / np.max(s)
    return s

def main():
    # 1. Setup Paths
    experiment_dir = Path("/storage/home/amr.hegazy/ray_results/sane_llm_tinyllama_neuron_view")
    # Find the latest trial folder
    trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("AE_trainable")]
    if not trial_dirs:
        print(f"No training results found in {experiment_dir}")
        return
        
    latest_trial = sorted(trial_dirs, key=lambda x: x.stat().st_mtime)[-1]
    
    checkpoint_path = latest_trial.joinpath("checkpoint_000005/state.pt")
    config_path = latest_trial.joinpath("params.json")
    
    print(f"Loading model from: {latest_trial.name}")
    
    # 2. Load Model
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['device'] = 'cpu'
    config['gpu_id'] = -1
    # FIX: Dummy steps for scheduler init
    config['training::steps_per_epoch'] = 100 
    
    model = AEModule(config)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        # Try finding any checkpoint
        chkpts = list(latest_trial.glob("checkpoint_*/state.pt"))
        if chkpts:
            checkpoint_path = chkpts[-1]
            print(f"Falling back to {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoints found.")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(state['model'])
    model.eval()
    
    # 3. Load Data (Train + Test to ensure coverage)
    print("Loading Datasets...")
    dataset_path = Path("data/llm_zoo/real_llm_dataset.pt")
    dataset_dict = torch.load(dataset_path, weights_only=False)
    trainset = dataset_dict['trainset']
    testset = dataset_dict['testset']
    
    # 4. Target a specific Layer
    # We try Layer 10, but handle if it's missing by auto-selecting available ones
    TARGET_LAYER = 10
    
    try:
        real_weights = get_layer_weights(trainset, testset, TARGET_LAYER)
    except ValueError as e:
        print(e)
        print("Switching target layer...")
        # Auto-select the layer with most samples in trainset
        l_ids = [int(x) for x in trainset.properties['layer_id']]
        from collections import Counter
        TARGET_LAYER = Counter(l_ids).most_common(1)[0][0]
        print(f"New Target Layer: {TARGET_LAYER}")
        real_weights = get_layer_weights(trainset, testset, TARGET_LAYER)

    print(f"Sampling variations of Layer {TARGET_LAYER} (Found {real_weights.shape[0]} anchors)...")
    
    # 5. Embed Anchors
    seq_len = real_weights.shape[1]
    pos = torch.zeros((1, seq_len, 3), dtype=torch.int)
    pos[:, :, 0] = torch.arange(seq_len)
    pos[:, :, 1] = TARGET_LAYER
    pos[:, :, 2] = torch.arange(seq_len)
    
    pos = pos.repeat(real_weights.shape[0], 1, 1)
    
    print("Encoding anchors...")
    with torch.no_grad():
        z_anchors = model.forward_encoder(real_weights, pos)
        
    # 6. Sample using KDE
    z_flat = z_anchors.reshape(-1, config['ae:lat_dim'])
    
    print("Fitting KDE on latent representations...")
    # Subsample for speed
    if z_flat.shape[0] > 5000:
        idx = torch.randperm(z_flat.shape[0])[:5000]
        z_train = z_flat[idx]
    else:
        z_train = z_flat
        
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(z_train.numpy())
    
    # Generate NEW Latents
    print("Sampling new latents...")
    z_new_flat = kde.sample(seq_len) 
    z_new = torch.tensor(z_new_flat, dtype=torch.float32).unsqueeze(0) 
    
    # 7. Decode
    print("Decoding to weights...")
    pos_new = pos[0:1] 
    with torch.no_grad():
        w_generated = model.forward_decoder(z_new, pos_new)
        
    # 8. Comparison Analysis (SVD Spectrum)
    print("Computing Spectral Analysis...")
    
    w_real = real_weights[0]
    
    svd_real = compute_svd_spectrum(w_real)
    svd_gen = compute_svd_spectrum(w_generated[0])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(svd_real, label=f'Real Layer {TARGET_LAYER}', linewidth=2, color='black', alpha=0.7)
    plt.plot(svd_gen, label='SANE Generated Layer', linestyle='--', linewidth=2, color='red')
    plt.yscale('log')
    plt.title(f"Spectral Density: Real vs Generated LLM Layer (TinyLlama)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("experiments/llm_extension")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir.joinpath("spectral_analysis.png")
    plt.savefig(plot_path)
    print(f"Analysis plot saved to {plot_path.absolute()}")
    
    # Save Generated Weights
    torch.save(w_generated, output_dir.joinpath("generated_layer.pt"))

if __name__ == "__main__":
    main()