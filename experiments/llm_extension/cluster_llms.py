import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent))

from SANE.models.def_AE_module import AEModule
from llm_dataset import LLMLayerDataset

def get_masked_mean_embedding(z, mask):
    """
    Computes mean embedding over the sequence dimension, ignoring padding.
    z: [Batch, Seq_Len, Latent_Dim]
    mask: [Batch, Seq_Len, Token_Dim] -> we slice to [Batch, Seq_Len, 1]
    """
    # Mask is boolean, convert to float
    mask_float = mask[:, :, 0:1].float() # Take first feature dim of mask
    
    # Sum of valid tokens
    # z * mask zeros out padding
    sum_z = (z * mask_float).sum(dim=1) # [Batch, Latent]
    
    # Count of valid tokens
    count_z = mask_float.sum(dim=1) # [Batch, 1]
    
    # Avoid div by zero (though unlikely in valid data)
    count_z = torch.clamp(count_z, min=1.0)
    
    return sum_z / count_z

def parse_model_info(info_str):
    """
    Parses 'TinyLlama/TinyLlama-1.1B-Chat-v1.0_L10' into components.
    """
    # Remove Layer suffix
    parts = info_str.split('_L')
    model_name = parts[0]
    
    family = "Unknown"
    if "llama" in model_name.lower():
        family = "Llama"
    elif "qwen" in model_name.lower():
        family = "Qwen"
    elif "gemma" in model_name.lower():
        family = "Gemma"
    elif "mistral" in model_name.lower():
        family = "Mistral"
        
    return model_name, family

def main():
    # 1. Load SANE Model
    experiment_dir = Path("/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch")
    trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("AE_trainable")]
    latest_trial = sorted(trial_dirs, key=lambda x: x.stat().st_mtime)[-1]
    
    chkpts = sorted(list(latest_trial.glob("checkpoint_*")))
    checkpoint_path = chkpts[-1].joinpath("state.pt")
    config_path = latest_trial.joinpath("params.json")
    
    print(f"Loading SANE from: {latest_trial.name}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['device'] = 'cuda'
    config['gpu_id'] = 0
    config['training::steps_per_epoch'] = 100 
    
    model = AEModule(config)
    state = torch.load(checkpoint_path, map_location='cuda')
    model.model.load_state_dict(state['model'])
    model.eval()
    
    # 2. Load Dataset (Train + Test/Holdout)
    print("Loading Dataset...")
    dataset_path = Path("data/scaled_llm_zoo/scaled_llm_dataset.pt")
    dataset_dict = torch.load(dataset_path, weights_only=False)
    
    # Combine both datasets for global clustering
    combined_data = dataset_dict['trainset'].data + dataset_dict['testset'].data
    # We need to wrap this in the Dataset class to handle padding correctly
    full_dataset = LLMLayerDataset(combined_data, property_keys=None)
    
    print(f"Extracting embeddings for {len(full_dataset)} layers...")
    
    embeddings = []
    metadata = []
    
    # Batch processing for speed
    BATCH_SIZE = 8
    dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (w, mask, pos, props) in enumerate(dataloader):
            # Forward Encoder
            w = w.to('cuda')
            mask = mask.to('cuda')
            pos = pos.to('cuda')
            
            z = model.forward_encoder(w, pos)
            
            # Pool: [Batch, Seq, Latent] -> [Batch, Latent]
            # This condenses the whole layer into one vector
            z_pooled = get_masked_mean_embedding(z, mask)
            
            embeddings.append(z_pooled.cpu().numpy())
            
            # Metadata extraction relies on index mapping
            # Since shuffle=False, we can map batch indices back to dataset.data
            start_idx = batch_idx * BATCH_SIZE
            for i in range(w.shape[0]):
                global_idx = start_idx + i
                info_str = full_dataset.data[global_idx]['info']
                model_name, family = parse_model_info(info_str)
                layer_id = int(props[i][0].item())
                
                metadata.append({
                    "Model": model_name,
                    "Family": family,
                    "Layer": layer_id,
                    "Label": f"{family} L{layer_id}"
                })
                
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * BATCH_SIZE} layers...")

    X = np.concatenate(embeddings, axis=0)
    df_meta = pd.DataFrame(metadata)
    
    print(f"Embedding Matrix Shape: {X.shape}")
    
    # 3. Dimensionality Reduction
    print("Running PCA...")
    pca = PCA(n_components=50) # Reduce noise before t-SNE
    X_pca = pca.fit_transform(X)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)
    
    df_meta['x'] = X_tsne[:, 0]
    df_meta['y'] = X_tsne[:, 1]
    
    # 4. Plotting
    sns.set_context("talk")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: By Architecture Family
    sns.scatterplot(
        data=df_meta, x='x', y='y', 
        hue='Family', style='Family', 
        palette='bright', s=100, alpha=0.8, ax=axes[0]
    )
    axes[0].set_title("SANE Space: Clustered by Architecture")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: By Layer Depth
    # Create a continuous color map
    points = axes[1].scatter(
        df_meta['x'], df_meta['y'], 
        c=df_meta['Layer'], cmap='viridis', 
        s=100, alpha=0.8
    )
    axes[1].set_title("SANE Space: Layer Trajectories")
    fig.colorbar(points, ax=axes[1], label='Layer Depth')
    
    plt.tight_layout()
    output_path = Path("experiments/llm_extension/llm_clusters.png")
    plt.savefig(output_path)
    print(f"Cluster plot saved to {output_path.absolute()}")

if __name__ == "__main__":
    main()