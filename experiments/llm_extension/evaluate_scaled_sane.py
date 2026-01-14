import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent))

from SANE.models.def_AE_module import AEModule
from llm_dataset import LLMLayerDataset

def get_holdout_sample(dataset, model_name_substr="gemma", layer_idx=10):
    """Finds a specific layer from a specific holdout architecture."""
    # dataset.data is a list of dicts
    candidates = []
    for i, sample in enumerate(dataset.data):
        info = sample.get('info', '')
        # Check model name match
        if model_name_substr.lower() in info.lower():
            # Check layer match (dataset stores 'props' as list, layer_id is index 0)
            lid = int(sample['props'][0])
            if lid == layer_idx:
                candidates.append(i)
    
    if not candidates:
        print(f"Warning: No samples found for {model_name_substr} Layer {layer_idx}")
        # Return random sample to avoid crash
        return dataset[0]
        
    # Return first match
    return dataset[candidates[0]]

def compute_svd_spectrum(matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    # Remove padding for fair comparison if possible
    # But since SANE output is padded, we analyze the full padded matrix
    # to see if SANE correctly predicts "zeros" in the padded region.
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except:
        return np.zeros(10)
    if np.max(s) > 0:
        s = s / np.max(s)
    return s

def main():
    # 1. Load Model
    experiment_dir = Path("/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch")
    # Get latest trial
    trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("AE_trainable")]
    latest_trial = sorted(trial_dirs, key=lambda x: x.stat().st_mtime)[-1]
    
    # Checkpoints are saved at intervals (5, 10, 15)
    # Find latest checkpoint folder
    chkpts = sorted(list(latest_trial.glob("checkpoint_*")))
    checkpoint_path = chkpts[-1].joinpath("state.pt")
    config_path = latest_trial.joinpath("params.json")
    
    print(f"Loading Scaled SANE from: {latest_trial.name}")
    print(f"Checkpoint: {checkpoint_path.name}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['device'] = 'cpu'
    config['gpu_id'] = -1
    config['training::steps_per_epoch'] = 100 
    
    model = AEModule(config)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(state['model'])
    model.eval()
    
    # 2. Load Holdout Dataset
    print("Loading Scaled Zoo...")
    dataset_path = Path("data/scaled_llm_zoo/scaled_llm_dataset.pt")
    dataset_dict = torch.load(dataset_path, weights_only=False)
    # The 'testset' contains the HOLDOUT models (Gemma/Qwen 1.8B)
    testset = dataset_dict['testset']
    
    # 3. Select a Gemma Layer
    target_arch = "gemma"
    target_layer = 12
    print(f"Attempting Zero-Shot Reconstruction of: {target_arch} (Layer {target_layer})")
    
    # Get raw sample (before transform) to see structure
    # Note: testset is LLMLayerDataset. __getitem__ handles padding.
    # We use __getitem__ to get the padded tensor SANE expects.
    
    # Find index first
    sample_idx = -1
    for i, d in enumerate(testset.data):
        if target_arch in d['info'].lower() and int(d['props'][0]) == target_layer:
            sample_idx = i
            print(f"Found sample: {d['info']}")
            break
            
    if sample_idx == -1:
        print("Could not find exact match, picking first sample from testset.")
        sample_idx = 0
        
    # Get padded input
    w_in, mask, pos, props = testset[sample_idx]
    
    # Add Batch Dim
    w_in = w_in.unsqueeze(0)
    pos = pos.unsqueeze(0)
    
    # 4. Reconstruct
    print("Reconstructing...")
    with torch.no_grad():
        z = model.forward_encoder(w_in, pos)
        w_out = model.forward_decoder(z, pos)
        
    # 5. Analysis
    # We compare the SVD of the Real (Padded) vs Reconstructed (Padded)
    # Ideally, SANE should reconstruct the weights AND the zero-padding.
    
    svd_real = compute_svd_spectrum(w_in[0])
    svd_recon = compute_svd_spectrum(w_out[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(svd_real, label=f'Real {target_arch} Layer {target_layer}', linewidth=2, color='black', alpha=0.7)
    plt.plot(svd_recon, label='SANE Zero-Shot Reconstruction', linestyle='--', linewidth=2, color='red')
    plt.yscale('log')
    plt.title(f"Zero-Shot Generalization: {target_arch.capitalize()} (Unseen Architecture)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Singular Value (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path("experiments/llm_extension/zero_shot_analysis.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path.absolute()}")
    
    # 6. Sparsity/Padding Check
    # Check if SANE correctly outputs zeros where it should (the padding region)
    # TinyLlama Hidden=2048. Gemma Hidden=2048.
    # If we padded to Qwen (Hidden=4096+), there is a lot of padding.
    # Let's check the magnitude of the last 100 columns.
    last_cols_real = w_in[0, :, -100:].abs().mean()
    last_cols_recon = w_out[0, :, -100:].abs().mean()
    print(f"\nPadding Region Mean Magnitude:")
    print(f"  Real:  {last_cols_real:.6f}")
    print(f"  Recon: {last_cols_recon:.6f}")
    if last_cols_recon < 0.01:
        print(">> SUCCESS: SANE correctly predicts zero-padding for smaller architectures!")
    else:
        print(">> Note: SANE generated noise in the padding region (common in zero-shot).")

if __name__ == "__main__":
    main()