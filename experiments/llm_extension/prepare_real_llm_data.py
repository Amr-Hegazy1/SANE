import torch
import os
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from llm_dataset import LLMLayerDataset

MODEL_ZOO = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
]

def tokenize_llama_mlp_layer(model_state_dict, layer_idx, hidden_size, intermediate_size, standardize=True, align_neurons=True):
    prefix = f"model.layers.{layer_idx}.mlp"
    try:
        w_gate = model_state_dict[f"{prefix}.gate_proj.weight"]
        w_up   = model_state_dict[f"{prefix}.up_proj.weight"]
        w_down = model_state_dict[f"{prefix}.down_proj.weight"]
    except KeyError:
        return None

    # Standardize per-layer
    if standardize:
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

    w_down_t = w_down.t()
    tokens = torch.cat([w_gate, w_up, w_down_t], dim=1)

    # Align neuron ordering (simple L2 norm sort)
    if align_neurons:
        norms = torch.norm(tokens.float(), p=2, dim=1)
        order = torch.argsort(norms, descending=True)
        tokens = tokens.index_select(0, order)

    return tokens.detach().cpu()

def main():
    output_dir = Path("data/llm_zoo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    print(f"Preparing Real LLM Data from: {MODEL_ZOO}")

    for model_name in MODEL_ZOO:
        try:
            config = AutoConfig.from_pretrained(model_name)
            hidden_size = config.hidden_size
            inter_size = config.intermediate_size
            num_layers = config.num_hidden_layers
            
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
            state_dict = model.state_dict()
            
            for l_id in tqdm(range(num_layers), desc=f"Processing Layers of {model_name}"):
                w_tokens = tokenize_llama_mlp_layer(state_dict, l_id, hidden_size, inter_size)
                
                if w_tokens is not None:
                    seq_len = w_tokens.shape[0]
                    
                    pos = torch.zeros((seq_len, 3), dtype=torch.int)
                    pos[:, 0] = torch.arange(seq_len)
                    pos[:, 1] = l_id 
                    pos[:, 2] = torch.arange(seq_len)

                    # Properties list corresponding to keys below
                    props = [float(l_id)]

                    all_samples.append({
                        'w': w_tokens,
                        'p': pos,
                        'props': props,
                        'info': f"{model_name}_L{l_id}"
                    })
            
            del model
            del state_dict
            
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")

    import random
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    # 1. Save Dataset
    prop_keys = {
        "result_keys": ["layer_id"],
        "config_keys": []
    }
    
    trainset = LLMLayerDataset(train_data, prop_keys)
    testset = LLMLayerDataset(test_data, prop_keys)
    
    dataset_dict = {
        "trainset": trainset,
        "testset": testset
    }
    
    save_path = output_dir.joinpath("real_llm_dataset.pt")
    torch.save(dataset_dict, save_path)
    print(f"Dataset saved to {save_path.absolute()}")

    # 2. Save Metadata JSON (Fix for KeyError)
    # AE_trainable looks for [dataset_name]_info_test.json to find property keys
    json_path = str(save_path).replace(".pt", "_info_test.json")
    metadata = {
        "properties": prop_keys["result_keys"]
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {json_path}")

if __name__ == "__main__":
    main()
