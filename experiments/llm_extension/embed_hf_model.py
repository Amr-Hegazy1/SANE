import torch
import torch.nn.functional as F
import json
import sys
import argparse
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# Add paths to SANE modules
sys.path.append(str(Path(__file__).parent.parent.parent)) 
sys.path.append(str(Path(__file__).parent))

from SANE.models.def_AE_module import AEModule

# Standard keys for extraction
KEY_MAPPINGS = {
    "LlamaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "MistralForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Qwen2ForCausalLM": ("gate_proj", "up_proj", "down_proj"), 
    "GemmaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Gemma2ForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "OPTForCausalLM": ("fc1", "fc1", "fc2"), # OPT is weird, just a placeholder
}

def load_sane_model(results_dir):
    """Loads the best SANE checkpoint from the results directory."""
    results_path = Path(results_dir)
    # Find latest trial
    trial_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("AE_trainable")]
    if not trial_dirs:
        raise FileNotFoundError(f"No SANE training results found in {results_dir}")
    latest_trial = sorted(trial_dirs, key=lambda x: x.stat().st_mtime)[-1]
    
    # Find metadata to get token dimension
    # (We assume metadata is in data/scaled_llm_zoo based on previous steps)
    # If not found, we try to infer from config
    chkpts = sorted(list(latest_trial.glob("checkpoint_*")))
    checkpoint_path = chkpts[-1].joinpath("state.pt")
    config_path = latest_trial.joinpath("params.json")
    
    print(f"Loading SANE from: {latest_trial.name}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Force CPU for inference to avoid OOM with big models loaded
    config['device'] = 'cuda'
    config['gpu_id'] = 0
    config['training::steps_per_epoch'] = 100 
    
    model = AEModule(config)
    state = torch.load(checkpoint_path, map_location='cuda')
    model.model.load_state_dict(state['model'])
    model.eval()
    
    # Return model and expected token dim
    return model, config['ae:i_dim']

def extract_and_pad_layer(state_dict, layer_idx, arch_type, target_token_dim):
    """
    Extracts MLP weights, tokenizes them, and pads to the SANE input dimension.
    """
    keys = KEY_MAPPINGS.get(arch_type, ("gate_proj", "up_proj", "down_proj"))
    prefix = f"model.layers.{layer_idx}.mlp"
    
    try:
        # 1. Extract
        # Try specific keys first
        try:
            w_gate = state_dict[f"{prefix}.{keys[0]}.weight"]
            w_up   = state_dict[f"{prefix}.{keys[1]}.weight"]
            w_down = state_dict[f"{prefix}.{keys[2]}.weight"]
        except KeyError:
            # Fallback for Qwen/Phi sometimes mapping differently
            # Simple heuristic: filter keys containing layer_idx and 'mlp'
            print(f"  Standard keys failed for {prefix}, skipping...")
            return None

        # 2. Tokenize (Neuron View)
        # Down needs transpose: [Hidden, Inter] -> [Inter, Hidden]
        w_down_t = w_down.t()
        
        # Current Token Dim = Hidden + Hidden + Hidden
        curr_dim = w_gate.shape[1] + w_up.shape[1] + w_down_t.shape[1]
        
        # 3. Pad features (Hidden Dim)
        # SANE expects fixed feature size (target_token_dim)
        # We assume the 3 matrices contribute equally to the mismatch, 
        # but simplistically we can just pad the result or pad components.
        # Padding components is safer if we want to reverse it later.
        
        # Calculate target hidden size from target_token_dim (assuming 3 parts)
        target_h = target_token_dim // 3
        curr_h = w_gate.shape[1]
        
        if curr_h < target_h:
            pad = target_h - curr_h
            w_gate = F.pad(w_gate, (0, pad))
            w_up = F.pad(w_up, (0, pad))
            w_down_t = F.pad(w_down_t, (0, pad))
        elif curr_h > target_h:
            # Truncate? Or fail? SANE can't handle larger than it trained on.
            print(f"  WARNING: Model hidden dim {curr_h} > SANE dim {target_h}. Truncating (Data Loss!).")
            w_gate = w_gate[:, :target_h]
            w_up = w_up[:, :target_h]
            w_down_t = w_down_t[:, :target_h]

        # Concatenate: [Inter, Target_Token_Dim]
        tokens = torch.cat([w_gate, w_up, w_down_t], dim=1)
        
        return tokens
        
    except Exception as e:
        print(f"  Error extracting layer {layer_idx}: {e}")
        return None

def get_layer_embedding(sane_model, tokens, layer_idx):
    """Passes tokens through SANE and pools the result."""
    # Create inputs
    # tokens: [Seq, Dim]
    # Pos: [1, Seq, 3] -> [0, Layer_ID, 0..N]
    seq_len = tokens.shape[0]
    pos = torch.zeros((1, seq_len, 3), dtype=torch.int)
    pos[:, :, 0] = torch.arange(seq_len)
    pos[:, :, 1] = layer_idx
    pos[:, :, 2] = torch.arange(seq_len)
    
    input_tensor = tokens.unsqueeze(0) # [1, Seq, Dim]
    
    with torch.no_grad():
        # Encode -> [1, Seq, Latent]
        input_tensor = input_tensor.to('cuda')
        pos = pos.to('cuda')
        
        z = sane_model.forward_encoder(input_tensor, pos)
        
        # Mean Pool (No mask needed here as we aren't batching variable lengths)
        z_pooled = z.mean(dim=1).squeeze(0) # [Latent]
        
    return z_pooled

def main():
    parser = argparse.ArgumentParser(description="Embed a Hugging Face model using SANE")
    parser.add_argument("--model_id", type=str, required=True, help="HF Model ID (e.g. google/gemma-2b)")
    parser.add_argument("--sane_dir", type=str, default="/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch", help="Path to SANE results")
    parser.add_argument("--output", type=str, default="embeddings.pt", help="Output file path")
    args = parser.parse_args()

    # 1. Load SANE
    print("Loading SANE...")
    sane_model, target_dim = load_sane_model(args.sane_dir)
    print(f"SANE expects Token Dimension: {target_dim}")

    # 2. Load HF Model
    print(f"Downloading/Loading {args.model_id}...")
    try:
        config = AutoConfig.from_pretrained(args.model_id)
        arch = config.architectures[0]
        # Load to CPU to allow processing big models without VRAM
        hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, device_map="cpu")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    embeddings = []
    
    # 3. Process Layers
    num_layers = config.num_hidden_layers
    print(f"Processing {num_layers} layers...")
    
    state_dict = hf_model.state_dict()
    
    for l_id in tqdm(range(num_layers)):
        # Extract & Pad
        tokens = extract_and_pad_layer(state_dict, l_id, arch, target_dim)
        
        if tokens is not None:
            # Embed
            z = get_layer_embedding(sane_model, tokens, l_id)
            embeddings.append(z)
        else:
            print(f"Warning: Layer {l_id} could not be processed.")

    if not embeddings:
        print("No embeddings generated.")
        return

    # Stack
    embeddings = torch.stack(embeddings) # [Num_Layers, Latent_Dim]
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # 4. Save
    result = {
        "model_id": args.model_id,
        "embeddings": embeddings,
        "config": config.to_dict()
    }
    torch.save(result, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()