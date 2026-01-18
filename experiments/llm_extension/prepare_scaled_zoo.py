import torch
import os
import json
import gc
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# Ensure local imports work regardless of current working directory
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_dataset import LLMLayerDataset, LLMBatchDataset
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
    # Phi MLP is 2-layer (fc1: [inter, hidden], fc2: [hidden, inter]).
    # To fit SANE's (Gate, Up, Down) tokenization, we duplicate fc1.
    "PhiForCausalLM": ("fc1", "fc1", "fc2"),
    "StableLmForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "FalconForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),
    "MPTForCausalLM": ("up_proj", "up_proj", "down_proj"),
    "OLMoForCausalLM": ("gate_proj", "up_proj", "down_proj"),
    "GPTNeoXForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),  # Pythia
    "BloomForCausalLM": ("dense_h_to_4h", "dense_h_to_4h", "dense_4h_to_h"),
    "DeepseekForCausalLM": ("gate_proj", "up_proj", "down_proj"),
}


def _cfg_get(cfg, *names, default=0):
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


def get_global_max_dims(model_list):
    """Scan configs to find global max hidden and intermediate sizes."""
    max_h = 0
    max_inter = 0
    print("Scanning models for dimensions...")
    for m in model_list:
        try:
            cfg = AutoConfig.from_pretrained(m)
            h = _cfg_get(cfg, "hidden_size", "d_model", default=0)
            inter = _cfg_get(cfg, "intermediate_size", "ffn_dim", "n_inner", default=0)
            if inter == 0 and h:
                inter = 4 * int(h)
            max_h = max(max_h, int(h))
            max_inter = max(max_inter, int(inter))
            print(f"  {m}: hidden={h}, inter={inter}")
        except Exception as e:
            print(f"  Could not load config for {m}: {e}")
    return max_h, max_inter


class _SafeTensorsAccessor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.files = sorted(self.model_path.glob("*.safetensors"))
        self.key_to_file = {}
        for sf in self.files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.key_to_file[key] = sf

    def get(self, key: str):
        sf = self.key_to_file.get(key)
        if sf is None:
            raise KeyError(key)
        with safe_open(sf, framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def get_weight_accessor(model_name, allow_bin_fallback=False):
    """Return (get_tensor_fn, cleanup_fn, mode_str).

    Safetensors path streams single tensors (no full state_dict).
    Bin fallback loads the model but avoids building a full state_dict.
    """
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=model_name, allow_patterns=["*.safetensors", "*.bin", "*.json"])
    st_files = list(Path(model_path).glob("*.safetensors"))
    if st_files:
        accessor = _SafeTensorsAccessor(model_path)
        return accessor.get, (lambda: None), "safetensors-stream"

    if not allow_bin_fallback:
        raise RuntimeError("No .safetensors found; rerun with allow_bin_fallback=True to load .bin weights")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    def _get(key: str):
        if not key.endswith(".weight"):
            raise KeyError(key)
        parts = key.split(".")
        # model.layers.<layer>.mlp.<name>.weight
        if len(parts) >= 6 and parts[0] == "model" and parts[1] == "layers" and parts[3] == "mlp":
            layer_idx = int(parts[2])
            proj_name = parts[4]
            mlp = model.model.layers[layer_idx].mlp
            if not hasattr(mlp, proj_name):
                raise KeyError(key)
            return getattr(mlp, proj_name).weight
        raise KeyError(key)

    def _cleanup():
        nonlocal model
        del model
        gc.collect()

    return _get, _cleanup, "bin-model"

def extract_mlp_weights(get_tensor, layer_idx, arch_type):
    """
    Extracts weights using architecture-specific logic.
    Returns (Gate, Up, Down) tensors or None.
    Converts to float32 only when needed.
    """
    prefix = f"model.layers.{layer_idx}.mlp"

    # Phi-3 uses fused gate+up projection
    if arch_type == "Phi3ForCausalLM":
        try:
            w_gate_up = get_tensor(f"{prefix}.gate_up_proj.weight")
            w_down = get_tensor(f"{prefix}.down_proj.weight")
            if w_gate_up.shape[0] % 2 != 0:
                return None, None, None
            half = w_gate_up.shape[0] // 2
            w_gate = w_gate_up[:half, :]
            w_up = w_gate_up[half:, :]
            return w_gate, w_up, w_down
        except KeyError:
            return None, None, None

    keys = KEY_MAPPINGS.get(arch_type, ("gate_proj", "up_proj", "down_proj"))
    try:
        w_gate = get_tensor(f"{prefix}.{keys[0]}.weight")
        w_up = get_tensor(f"{prefix}.{keys[1]}.weight")
        w_down = get_tensor(f"{prefix}.{keys[2]}.weight")
        return w_gate, w_up, w_down
    except KeyError:
        return None, None, None

def process_layer(
    w_gate,
    w_up,
    w_down,
    curr_hidden,
    max_hidden_dim,
    layer_id,
    out_dtype=torch.float16,
    standardize=True,
    align_neurons=True,
):
    """
    Process a single layer's weights into a sample.
    Isolated function to enable better garbage collection.
    """
    # Standardize per-layer (paper preprocessing)
    if standardize:
        # Use float32 for stable stats
        w_gate_f = w_gate.float()
        w_up_f = w_up.float()
        w_down_f = w_down.float()
        w_down_t_f = w_down_f.t()
        all_vals = torch.cat([w_gate_f.flatten(), w_up_f.flatten(), w_down_t_f.flatten()])
        mean = all_vals.mean()
        std = all_vals.std().clamp_min(1e-6)
        w_gate = (w_gate_f - mean) / std
        w_up = (w_up_f - mean) / std
        w_down = (w_down_f - mean) / std
    # cast to target dtype
    w_gate = w_gate.to(dtype=out_dtype)
    w_up = w_up.to(dtype=out_dtype)
    w_down = w_down.to(dtype=out_dtype)

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

    # Align neuron ordering (simple canonicalization via L2 norm sort)
    if align_neurons:
        norms = torch.norm(tokens.float(), p=2, dim=1)
        order = torch.argsort(norms, descending=True)
        tokens = tokens.index_select(0, order)
    
    # Create Metadata
    seq_len = tokens.shape[0]
    
    # Pos: [Token_Index, Layer_ID, Neuron_ID]
    pos = torch.zeros((seq_len, 3), dtype=torch.int)
    pos[:, 0] = torch.arange(seq_len)
    pos[:, 1] = layer_id
    pos[:, 2] = torch.arange(seq_len)
    
    # Props: [Layer_ID, Hidden_Size (normalized)]
    props = [float(layer_id), float(curr_hidden)]
    
    # Move to CPU and detach immediately
    sample = {
        'w': tokens.detach().cpu(),
        'p': pos.detach().cpu(),
        'props': props
    }
    
    # Cleanup intermediate tensors
    del w_gate, w_up, w_down, w_down_t, tokens, pos
    
    return sample

def process_model_zoo(
    model_list,
    max_hidden_dim,
    zoo_name,
    output_dir,
    batch_save_size=1,
    allow_bin_fallback=True,
    out_dtype=torch.float16,
):
    """
    Process models and save samples in batches to avoid memory buildup.
    """
    samples = []
    batch_counter = 0
    refs = []
    
    for model_name in model_list:
        print(f"\nProcessing {model_name}...")
        try:
            config = AutoConfig.from_pretrained(model_name)
            arch = config.architectures[0]
            curr_hidden = _cfg_get(config, "hidden_size", "d_model", default=0)
            num_layers = _cfg_get(config, "num_hidden_layers", "n_layer", default=0)
            
            print(f"  Arch: {arch} | Hidden: {curr_hidden} | Layers: {num_layers}")
            
            get_tensor, cleanup, mode = get_weight_accessor(model_name, allow_bin_fallback=allow_bin_fallback)
            print(f"  Weight mode: {mode}")
            
            for l_id in tqdm(range(num_layers), desc=f"  Extracting"):
                w_gate, w_up, w_down = extract_mlp_weights(get_tensor, l_id, arch)
                
                if w_gate is not None:
                    sample = process_layer(w_gate, w_up, w_down, curr_hidden, max_hidden_dim, l_id, out_dtype=out_dtype)
                    sample['info'] = f"{model_name}_L{l_id}"
                    samples.append(sample)
                    
                    # Clear individual weights from memory
                    del w_gate, w_up, w_down
                    
                    # Save batch if reached threshold
                    if len(samples) >= batch_save_size:
                        batch_file = output_dir / f"{zoo_name}_batch_{batch_counter}.pt"
                        torch.save(samples, batch_file)
                        print(f"    Saved batch {batch_counter} ({len(samples)} samples)")

                        for j, s in enumerate(samples):
                            refs.append({
                                "file": str(batch_file.resolve()),
                                "idx": j,
                                "props": s.get("props", []),
                                "info": s.get("info", ""),
                            })

                        samples = []
                        batch_counter += 1
                        gc.collect()
            
            cleanup()
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

        for j, s in enumerate(samples):
            refs.append({
                "file": str(batch_file.resolve()),
                "idx": j,
                "props": s.get("props", []),
                "info": s.get("info", ""),
            })
    
    return refs

def write_index(output_dir, zoo_name, refs):
    index_path = output_dir / f"{zoo_name}_index.json"
    with open(index_path, "w") as f:
        json.dump(refs, f)
    return index_path

def main():
    output_dir = Path("data/scaled_llm_zoo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Global Dimensions
    all_models = TRAIN_ZOO + HOLDOUT_ZOO
    max_hidden, max_inter = get_global_max_dims(all_models)
    token_dim = max_hidden * 3
    print(f"\nGlobal Max Hidden Size: {max_hidden}")
    print(f"Global Max Intermediate Size: {max_inter}")
    print(f"Unified SANE Token Dimension: {token_dim}")
    
    prop_keys = {
        "result_keys": ["layer_id", "orig_hidden_size"],
        "config_keys": []
    }
    
    # 2. Process Train Zoo in batches
    print("\n=== Building TRAIN Zoo ===")
    train_refs = process_model_zoo(TRAIN_ZOO, max_hidden, "train", output_dir, batch_save_size=1)
    train_index = write_index(output_dir, "train", train_refs)
    print(f"Wrote train index: {train_index}")
    
    # 3. Process Holdout Zoo in batches
    print("\n=== Building HOLDOUT Zoo ===")
    holdout_refs = process_model_zoo(HOLDOUT_ZOO, max_hidden, "holdout", output_dir, batch_save_size=1)
    holdout_index = write_index(output_dir, "holdout", holdout_refs)
    print(f"Wrote holdout index: {holdout_index}")

    # 4. Create Final Dataset (Lazy; does NOT merge batches into one huge list)
    print("\n=== Creating Final Dataset (Lazy) ===")
    trainset = LLMBatchDataset(train_refs, token_dim=token_dim, max_seq_len=max_inter, property_keys=prop_keys)
    holdoutset = LLMBatchDataset(holdout_refs, token_dim=token_dim, max_seq_len=max_inter, property_keys=prop_keys)
    
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
        "max_intermediate": max_inter,
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
