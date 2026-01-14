import torch
import os
import json
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from llm_dataset import LLMLayerDataset

# --- Configuration ---
# Models to Train on (The "Known" World)
TRAIN_ZOO = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Qwen/Qwen1.5-0.5B", # Small Qwen
    # Uncomment if you have RAM (requires ~15GB system RAM per model load)
    # "openlm-research/open_llama_3b_v2", 
]

# Models to Test on (The "Unseen" World - Different architectures)
HOLDOUT_ZOO = [
    "google/gemma-2b", # Different architecture details
    "Qwen/Qwen1.5-1.8B", # Same family, different scale
]

# Mapping different key names to a standard (Gate, Up, Down) tuple
# Llama/Mistral: gate_proj, up_proj, down_proj
# Qwen: w1 (up?), w2 (gate?), c_proj (down) -> Actually Qwen is: w2(gate), w1(up), c_proj(down) usually
# Let's map dynamically based on shapes if needed, or hardcode known mappings.
KEY_MAPPINGS = {
    "LlamaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "MistralForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Qwen2ForCausalLM": ("gate_proj", "up_proj", "down_proj"), # Qwen1.5 uses standard names in HF
    "GemmaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Gemma2ForCausalLM": ("gate_proj", "up_proj", "down_proj"),
}

def get_max_hidden_size(model_list):
    """Scans config of all models to find the global maximum hidden dimension."""
    max_h = 0
    print("Scanning models for dimensions...")
    for m in model_list:
        try:
            cfg = AutoConfig.from_pretrained(m)
            # Handle standard naming variations
            h = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 0))
            if h > max_h:
                max_h = h
            print(f"  {m}: {h}")
        except Exception as e:
            print(f"  Could not load config for {m}: {e}")
    return max_h

def extract_mlp_weights(state_dict, layer_idx, arch_type):
    """
    Extracts weights using architecture-specific logic.
    Returns (Gate, Up, Down) tensors or None.
    """
    keys = KEY_MAPPINGS.get(arch_type, ("gate_proj", "up_proj", "down_proj"))
    prefix = f"model.layers.{layer_idx}.mlp"
    
    # Gemma/Qwen/Llama all map to 'model.layers' in recent HF versions
    # If keys fail, try finding them via string matching
    try:
        w_gate = state_dict[f"{prefix}.{keys[0]}.weight"]
        w_up   = state_dict[f"{prefix}.{keys[1]}.weight"]
        w_down = state_dict[f"{prefix}.{keys[2]}.weight"]
        return w_gate, w_up, w_down
    except KeyError:
        return None, None, None

def process_model_zoo(model_list, max_hidden_dim, zoo_name):
    samples = []
    
    for model_name in model_list:
        print(f"\nProcessing {model_name}...")
        try:
            config = AutoConfig.from_pretrained(model_name)
            arch = config.architectures[0]
            curr_hidden = config.hidden_size
            num_layers = config.num_hidden_layers
            
            print(f"  Arch: {arch} | Hidden: {curr_hidden} | Layers: {num_layers}")
            
            # Load weights to CPU
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
            state_dict = model.state_dict()
            
            for l_id in tqdm(range(num_layers), desc=f"  Extracting"):
                w_gate, w_up, w_down = extract_mlp_weights(state_dict, l_id, arch)
                
                if w_gate is not None:
                    # 1. Structure-Aware Tokenization (Concat Neurons)
                    # Down needs transpose to align: [Hidden, Inter] -> [Inter, Hidden]
                    w_down_t = w_down.t()
                    
                    # 2. PAD to Global Max Dimension
                    # We pad the Hidden dimension (feature dim), NOT the Intermediate (sequence dim)
                    # Current shape: [Inter, Current_Hidden]
                    if curr_hidden < max_hidden_dim:
                        pad_size = max_hidden_dim - curr_hidden
                        # Pad columns (features)
                        w_gate = torch.nn.functional.pad(w_gate, (0, pad_size))
                        w_up = torch.nn.functional.pad(w_up, (0, pad_size))
                        w_down_t = torch.nn.functional.pad(w_down_t, (0, pad_size))
                        
                    # 3. Create Neuron Tokens
                    # Shape: [Inter, Max_Hidden * 3]
                    tokens = torch.cat([w_gate, w_up, w_down_t], dim=1)
                    
                    # 4. Create Metadata
                    seq_len = tokens.shape[0]
                    
                    # Pos: [Global_Seq, Layer_ID, Neuron_ID]
                    pos = torch.zeros((seq_len, 3), dtype=torch.int)
                    pos[:, 0] = torch.arange(seq_len)
                    pos[:, 1] = l_id
                    pos[:, 2] = torch.arange(seq_len)
                    
                    # Props: [Layer_ID, Hidden_Size (normalized)]
                    # Storing original hidden size helps us analyze if SANE learns scale
                    props = [float(l_id), float(curr_hidden)]
                    
                    samples.append({
                        'w': tokens.detach().cpu(),
                        'p': pos,
                        'props': props,
                        'info': f"{model_name}_L{l_id}"
                    })
            
            # Cleanup to save RAM
            del model
            del state_dict
            gc.collect()
            
        except Exception as e:
            print(f"  Failed: {e}")
            
    return samples

def main():
    output_dir = Path("data/scaled_llm_zoo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Global Dimension
    all_models = TRAIN_ZOO + HOLDOUT_ZOO
    max_hidden = get_max_hidden_size(all_models)
    token_dim = max_hidden * 3
    print(f"\nGlobal Max Hidden Size: {max_hidden}")
    print(f"Unified SANE Token Dimension: {token_dim}")
    
    # 2. Process Train Zoo
    print("\n=== Building TRAIN Zoo ===")
    train_samples = process_model_zoo(TRAIN_ZOO, max_hidden, "train")
    
    # 3. Process Holdout Zoo
    print("\n=== Building HOLDOUT Zoo ===")
    holdout_samples = process_model_zoo(HOLDOUT_ZOO, max_hidden, "holdout")
    
    # 4. Save
    prop_keys = {
        "result_keys": ["layer_id", "orig_hidden_size"],
        "config_keys": []
    }
    
    # We use all train_samples for training (can split val internally later if needed)
    # And holdout for OOD testing
    trainset = LLMLayerDataset(train_samples, prop_keys)
    holdoutset = LLMLayerDataset(holdout_samples, prop_keys)
    
    dataset_dict = {
        "trainset": trainset,
        "testset": holdoutset # We put holdout in 'testset' slot for SANE eval
    }
    
    save_path = output_dir.joinpath("scaled_llm_dataset.pt")
    torch.save(dataset_dict, save_path)
    
    # Save Metadata including dimensions
    meta = {
        "properties": prop_keys["result_keys"],
        "max_hidden": max_hidden,
        "token_dim": token_dim,
        "train_models": TRAIN_ZOO,
        "holdout_models": HOLDOUT_ZOO
    }
    with open(str(save_path).replace(".pt", "_info_test.json"), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nDone. Dataset saved to {save_path}")

if __name__ == "__main__":
    main()