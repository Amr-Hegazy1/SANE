import torch
import os
import json
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from llm_dataset import LLMLayerDataset
from safetensors import safe_open
from safetensors.torch import load_file

# --- Configuration ---
# Models to Train on (The "Known" World)
TRAIN_ZOO = [
    # Small Models (< 2B)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "Qwen/Qwen1.5-0.5B",
    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-360M",
    "HuggingFaceTB/SmolLM-1.7B",
    "stabilityai/stablelm-2-1_6b",
    "microsoft/phi-1_5",
    
    # Medium Models (2B-10B)
    "HuggingFaceTB/SmolLM3-3B",
    "openlm-research/open_llama_3b_v2",
    "stabilityai/stablelm-base-alpha-7b",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-2-7b-hf",
    "01-ai/Yi-6B",
    "01-ai/Yi-9B",
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen2-7B",
    
    # Large Models (10B+)
    "meta-llama/Llama-2-13b-hf",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen1.5-14B",
    "01-ai/Yi-34B",
]

# Models to Test on (The "Unseen" World - Different architectures & scales)
HOLDOUT_ZOO = [
    # Gemma Family (Different architecture)
    "google/gemma-2b",
    "google/gemma-7b",
    "google/gemma-2-9b",
    "google/gemma-2-27b",
    
    # Qwen Variants (Same family, different scales for interpolation/extrapolation)
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2.5-7B",
    
    # Phi Family (Different architecture)
    "microsoft/phi-2",
    "microsoft/phi-3-mini-4k-instruct",
    
    # DeepSeek (Different architecture)
    "deepseek-ai/deepseek-llm-7b-base",
    
    # Falcon (Different architecture)
    "tiiuae/falcon-7b",
    
    # MPT (Different architecture)
    "mosaicml/mpt-7b",
    
    # OLMo (Different architecture)
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-hf",
    
    # Pythia (Different training approach)
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    
    # BLOOM (Multilingual, different architecture)
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
]

# Mapping different key names to a standard (Gate, Up, Down) tuple
KEY_MAPPINGS = {
    "LlamaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "MistralForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Qwen2ForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "GemmaForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "Gemma2ForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "YiForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "SmolLMForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "PhiForCausalLM": ("fc1", "fc2", "fc2"),  # Phi uses different naming
    "StableLmForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "FalconForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),
    "MPTForCausalLM": ("up_proj", "up_proj", "down_proj"),
    "OLMoForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "GPTNeoXForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),  # Pythia
    "BloomForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),
    "DeepseekForCausalLM": ("gate_proj", "up_proj", "down_proj"),
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

def load_state_dict_efficiently(model_name):
    """
    Try to load model weights using safetensors first (more memory efficient),
    fallback to standard loading if not available.
    """
    from huggingface_hub import snapshot_download
    
    try:
        # Download model files
        model_path = snapshot_download(repo_id=model_name, allow_patterns=["*.safetensors", "*.bin"])
        
        # Try safetensors first
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        if safetensors_files:
            print(f"  Loading via safetensors (memory efficient)")
            state_dict = {}
            for sf in safetensors_files:
                with safe_open(sf, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            return state_dict
    except:
        pass
    
    # Fallback to standard loading
    print(f"  Loading via standard method")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # Use fp16 to save memory
        device_map="cpu",
        low_cpu_mem_usage=True  # Enable low memory mode
    )
    state_dict = model.state_dict()
    del model
    gc.collect()
    return state_dict

def extract_mlp_weights(state_dict, layer_idx, arch_type):
    """
    Extracts weights using architecture-specific logic.
    Returns (Gate, Up, Down) tensors or None.
    Converts to float32 only when needed.
    """
    keys = KEY_MAPPINGS.get(arch_type, ("gate_proj", "up_proj", "down_proj"))
    prefix = f"model.layers.{layer_idx}.mlp"
    
    try:
        w_gate = state_dict[f"{prefix}.{keys[0]}.weight"].float()
        w_up   = state_dict[f"{prefix}.{keys[1]}.weight"].float()
        w_down = state_dict[f"{prefix}.{keys[2]}.weight"].float()
        return w_gate, w_up, w_down
    except KeyError:
        return None, None, None

def process_layer(w_gate, w_up, w_down, curr_hidden, max_hidden_dim, layer_id):
    """
    Process a single layer's weights into a sample.
    Isolated function to enable better garbage collection.
    """
    # Down needs transpose to align: [Hidden, Inter] -> [Inter, Hidden]
    w_down_t = w_down.t()
    
    # PAD to Global Max Dimension if needed
    if curr_hidden < max_hidden_dim:
        pad_size = max_hidden_dim - curr_hidden
        w_gate = torch.nn.functional.pad(w_gate, (0, pad_size))
        w_up = torch.nn.functional.pad(w_up, (0, pad_size))
        w_down_t = torch.nn.functional.pad(w_down_t, (0, pad_size))
    
    # Create Neuron Tokens - Shape: [Inter, Max_Hidden * 3]
    tokens = torch.cat([w_gate, w_up, w_down_t], dim=1)
    
    # Create Metadata
    seq_len = tokens.shape[0]
    
    # Pos: [Global_Seq, Layer_ID, Neuron_ID]
    pos = torch.zeros((seq_len, 3), dtype=torch.int)
    pos[:, 0] = torch.arange(seq_len)
    pos[:, 1] = layer_id
    pos[:, 2] = torch.arange(seq_len)
    
    # Props: [Layer_ID, Hidden_Size (normalized)]
    props = [float(layer_id), float(curr_hidden)]
    
    # Move to CPU and detach immediately
    sample = {
        'w': tokens.detach().cpu(),
        'p': pos,
        'props': props
    }
    
    # Cleanup intermediate tensors
    del w_gate, w_up, w_down, w_down_t, tokens, pos
    
    return sample

def process_model_zoo(model_list, max_hidden_dim, zoo_name, output_dir, batch_save_size=50):
    """
    Process models and save samples in batches to avoid memory buildup.
    """
    samples = []
    batch_counter = 0
    
    for model_name in model_list:
        print(f"\nProcessing {model_name}...")
        try:
            config = AutoConfig.from_pretrained(model_name)
            arch = config.architectures[0]
            curr_hidden = config.hidden_size
            num_layers = config.num_hidden_layers
            
            print(f"  Arch: {arch} | Hidden: {curr_hidden} | Layers: {num_layers}")
            
            # Load weights efficiently
            state_dict = load_state_dict_efficiently(model_name)
            
            for l_id in tqdm(range(num_layers), desc=f"  Extracting"):
                w_gate, w_up, w_down = extract_mlp_weights(state_dict, l_id, arch)
                
                if w_gate is not None:
                    sample = process_layer(w_gate, w_up, w_down, curr_hidden, max_hidden_dim, l_id)
                    sample['info'] = f"{model_name}_L{l_id}"
                    samples.append(sample)
                    
                    # Clear individual weights from memory
                    del w_gate, w_up, w_down
                    
                    # Save batch if reached threshold
                    if len(samples) >= batch_save_size:
                        batch_file = output_dir / f"{zoo_name}_batch_{batch_counter}.pt"
                        torch.save(samples, batch_file)
                        print(f"    Saved batch {batch_counter} ({len(samples)} samples)")
                        samples = []
                        batch_counter += 1
                        gc.collect()
            
            # Cleanup model state dict
            del state_dict
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save remaining samples
    if samples:
        batch_file = output_dir / f"{zoo_name}_batch_{batch_counter}.pt"
        torch.save(samples, batch_file)
        print(f"    Saved final batch {batch_counter} ({len(samples)} samples)")
    
    return batch_counter + 1

def merge_batches(output_dir, zoo_name, num_batches, prop_keys):
    """Merge batch files into final dataset."""
    print(f"\nMerging {num_batches} batches for {zoo_name}...")
    all_samples = []
    
    for i in range(num_batches):
        batch_file = output_dir / f"{zoo_name}_batch_{i}.pt"
        if batch_file.exists():
            samples = torch.load(batch_file)
            all_samples.extend(samples)
            batch_file.unlink()  # Delete batch file after loading
            
            if (i + 1) % 10 == 0:
                gc.collect()
    
    return LLMLayerDataset(all_samples, prop_keys)

def main():
    output_dir = Path("data/scaled_llm_zoo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Global Dimension
    all_models = TRAIN_ZOO + HOLDOUT_ZOO
    max_hidden = get_max_hidden_size(all_models)
    token_dim = max_hidden * 3
    print(f"\nGlobal Max Hidden Size: {max_hidden}")
    print(f"Unified SANE Token Dimension: {token_dim}")
    
    prop_keys = {
        "result_keys": ["layer_id", "orig_hidden_size"],
        "config_keys": []
    }
    
    # 2. Process Train Zoo in batches
    print("\n=== Building TRAIN Zoo ===")
    train_batches = process_model_zoo(TRAIN_ZOO, max_hidden, "train", output_dir)
    
    # 3. Process Holdout Zoo in batches
    print("\n=== Building HOLDOUT Zoo ===")
    holdout_batches = process_model_zoo(HOLDOUT_ZOO, max_hidden, "holdout", output_dir)
    
    # 4. Merge batches and create final dataset
    print("\n=== Creating Final Dataset ===")
    trainset = merge_batches(output_dir, "train", train_batches, prop_keys)
    holdoutset = merge_batches(output_dir, "holdout", holdout_batches, prop_keys)
    
    dataset_dict = {
        "trainset": trainset,
        "testset": holdoutset
    }
    
    save_path = output_dir / "scaled_llm_dataset.pt"
    torch.save(dataset_dict, save_path)
    
    # Save Metadata
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
    print(f"Train samples: {len(trainset)}, Holdout samples: {len(holdoutset)}")

if __name__ == "__main__":
    main()