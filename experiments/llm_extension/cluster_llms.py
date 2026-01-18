import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import pandas as pd
from pathlib import Path
from torch.utils.data import ConcatDataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent))

from SANE.models.def_AE_module import AEModule
from llm_dataset import LLMLayerDataset

def get_masked_mean_embedding(z, mask_seq):
    """Mean-pool over sequence dimension, ignoring padding.

    z: [B, T, D]
    mask_seq: [B, T] bool (True for valid tokens)
    """
    if mask_seq.ndim != 2:
        raise ValueError(f"mask_seq must be [B, T], got {tuple(mask_seq.shape)}")
    mask_float = mask_seq.to(dtype=z.dtype).unsqueeze(-1)  # [B, T, 1]
    sum_z = (z * mask_float).sum(dim=1)
    count_z = mask_float.sum(dim=1).clamp(min=1.0)
    return sum_z / count_z

def parse_model_info(info_str):
    """
    Parses 'TinyLlama/TinyLlama-1.1B-Chat-v1.0_L10' into components.
    """
    # Remove Layer suffix
    parts = info_str.split('_L')
    model_name = parts[0]

    mn = model_name.lower()
    # Keep these aligned with families appearing in prepare_scaled_zoo.py
    if "tinyllama" in mn:
        family = "TinyLlama"
    elif "smollm" in mn:
        family = "SmolLM"
    elif "stablelm" in mn:
        family = "StableLM"
    elif "microsoft/phi" in mn or "/phi-" in mn or "phi-" in mn:
        family = "Phi"
    elif "qwen" in mn:
        family = "Qwen"
    elif "gemma" in mn:
        family = "Gemma"
    elif "mistral" in mn:
        family = "Mistral"
    elif "open_llama" in mn or "openlm-research/open_llama" in mn:
        family = "OpenLLaMA"
    elif "meta-llama/llama" in mn or "llama-" in mn or ("llama" in mn and "tinyllama" not in mn and "open_llama" not in mn):
        family = "Llama"
    elif mn.startswith("01-ai/yi") or "/yi-" in mn:
        family = "Yi"
    elif "deepseek" in mn:
        family = "DeepSeek"
    elif "falcon" in mn:
        family = "Falcon"
    elif "mpt" in mn:
        family = "MPT"
    elif "olmo" in mn:
        family = "OLMo"
    elif "pythia" in mn:
        family = "Pythia"
    elif "bloom" in mn:
        family = "BLOOM"
    else:
        family = "Other"

    return model_name, family


def _select_trial_dir(experiment_dir: Path, expected_i_dim: int) -> Path:
    """Pick the newest trial whose params.json matches expected_i_dim.

    Falls back to newest trial overall if no exact match is found.
    """
    trial_dirs = [
        d
        for d in experiment_dir.iterdir()
        if d.is_dir()
        and (
            d.name.startswith("LLM_AE_trainable")
            or d.name.startswith("AE_trainable")
            or d.name.startswith("Trainable")
        )
    ]
    if not trial_dirs:
        # last resort: any directory
        trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]

    def _read_i_dim(d: Path):
        cfg_path = d / "params.json"
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            return int(cfg.get("ae:i_dim", 0))
        except Exception:
            return None

    matching = [d for d in trial_dirs if _read_i_dim(d) == int(expected_i_dim)]
    if matching:
        return sorted(matching, key=lambda x: x.stat().st_mtime)[-1]
    return sorted(trial_dirs, key=lambda x: x.stat().st_mtime)[-1]


def _match_token_dim(w: torch.Tensor, expected_dim: int):
    """Pad/truncate token dim to match the model input dim."""
    curr = int(w.shape[-1])
    expected_dim = int(expected_dim)
    if curr == expected_dim:
        return w

    if curr > expected_dim:
        return w[..., :expected_dim]

    pad_dim = expected_dim - curr
    w_pad = torch.zeros((*w.shape[:-1], pad_dim), device=w.device, dtype=w.dtype)
    return torch.cat([w, w_pad], dim=-1)


def _slice_window(w: torch.Tensor, pos: torch.Tensor, mask_seq: torch.Tensor, windowsize: int):
    """Slice a fixed-length window from a (padded) sequence.

    Assumes padding is at the end (mask_seq is True for valid prefix).
    """
    windowsize = int(windowsize)
    if windowsize <= 0:
        return w, pos, mask_seq
    if w.shape[1] <= windowsize:
        return w, pos, mask_seq

    B = w.shape[0]
    out_w, out_pos, out_mask = [], [], []
    for i in range(B):
        valid_len = int(mask_seq[i].sum().item())
        if valid_len <= 0:
            # Extremely defensive: ensure at least one valid token to avoid NaNs
            valid_len = 1
            mask_seq[i, 0] = True
        if valid_len <= windowsize:
            start = 0
        else:
            start = max(0, (valid_len - windowsize) // 2)
        end = start + windowsize
        out_w.append(w[i, start:end])
        out_pos.append(pos[i, start:end])
        out_mask.append(mask_seq[i, start:end])

    return torch.stack(out_w, dim=0), torch.stack(out_pos, dim=0), torch.stack(out_mask, dim=0)

def main():
    # 1. Load Dataset + determine expected token dim
    dataset_path = Path("data/scaled_llm_zoo/scaled_llm_dataset.pt")
    meta_path = Path(str(dataset_path).replace(".pt", "_info_test.json"))
    expected_token_dim = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        expected_token_dim = int(meta.get("token_dim", 0))

    # 2. Select a matching SANE checkpoint (by ae:i_dim)
    experiment_dir = Path("/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch")
    latest_trial = _select_trial_dir(experiment_dir, expected_token_dim or 0)
    
    chkpts = sorted(list(latest_trial.glob("checkpoint_*")))
    checkpoint_path = chkpts[-1].joinpath("state.pt")
    config_path = latest_trial.joinpath("params.json")
    
    print(f"Loading SANE from: {latest_trial.name}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_i_dim = int(config.get("ae:i_dim", 0))
    windowsize = int(config.get("training::windowsize", 128))
    if expected_token_dim and model_i_dim and expected_token_dim != model_i_dim:
        print(f"WARNING: Dataset token_dim={expected_token_dim} but checkpoint ae:i_dim={model_i_dim}. Will pad/truncate inputs to match.")
    
    config['device'] = 'cuda'
    config['gpu_id'] = 0
    config['training::steps_per_epoch'] = 100 
    
    model = AEModule(config)
    state = torch.load(checkpoint_path, map_location='cuda')
    model.model.load_state_dict(state['model'])
    model.eval()
    
    # 3. Load Dataset (Train + Test/Holdout)
    print("Loading Dataset...")
    dataset_dict = torch.load(dataset_path, weights_only=False)

    trainset = dataset_dict['trainset']
    testset = dataset_dict['testset']

    # Combine both datasets for global clustering
    # New scaled zoo uses a lazy, batch-backed dataset (LLMBatchDataset) that has `refs`.
    if hasattr(trainset, 'refs') and hasattr(testset, 'refs'):
        full_dataset = ConcatDataset([trainset, testset])
        info_list = [r.get('info', '') for r in trainset.refs] + [r.get('info', '') for r in testset.refs]
    else:
        combined_data = trainset.data + testset.data
        full_dataset = LLMLayerDataset(combined_data, property_keys=None)
        info_list = [d.get('info', '') for d in combined_data]
    
    print(f"Extracting embeddings for {len(full_dataset)} layers...")
    
    embeddings = []
    metadata = []
    
    # Batch processing for speed
    BATCH_SIZE = 4
    dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (w, mask, pos, props) in enumerate(dataloader):
            # Forward Encoder
            w = w.to('cuda', non_blocking=True)
            mask = mask.to('cuda', non_blocking=True)
            pos = pos.to('cuda', non_blocking=True)

            # Dataset mask is [B, T, token_dim]. Convert to a 1D token validity mask.
            # (Padding is per-token, not per-feature for attention.)
            mask_seq = mask[..., 0].to(torch.bool)  # [B, T]

            # Ensure dtype matches the tokenizer / model weights
            try:
                model_dtype = next(model.model.parameters()).dtype
            except Exception:
                model_dtype = torch.float32
            w = w.to(dtype=model_dtype)

            # Ensure token_dim matches what the checkpoint expects
            if model_i_dim:
                w = _match_token_dim(w, model_i_dim)

            # Slice a small window to keep attention tractable
            w, pos, mask_seq = _slice_window(w, pos, mask_seq, windowsize)

            # Build SDPA attention mask: True means "masked out".
            # Shape [B, 1, 1, T] broadcasts to [B, n_head, T, T].
            attn_mask = (~mask_seq)[:, None, None, :]

            # Pass mask so the encoder can ignore padding
            z = model.forward_encoder(w, pos, attn_mask)
            
            # Pool: [Batch, Seq, Latent] -> [Batch, Latent]
            # This condenses the whole layer into one vector
            z_pooled = get_masked_mean_embedding(z, mask_seq)
            
            embeddings.append(z_pooled.cpu().numpy())
            
            # Metadata extraction relies on index mapping
            # Since shuffle=False, we can map batch indices back to dataset.data
            start_idx = batch_idx * BATCH_SIZE
            for i in range(w.shape[0]):
                global_idx = start_idx + i
                info_str = info_list[global_idx] if global_idx < len(info_list) else ""
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