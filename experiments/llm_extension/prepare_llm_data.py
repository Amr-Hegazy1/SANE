import torch
import torch.nn as nn
import os
from pathlib import Path
from llm_dataset import LLMLayerDataset

# Configuration for our synthetic "TinyLlama"
CONFIG = {
    "num_models": 20,
    "hidden_size": 256,
    "intermediate_size": 1024,
    "num_layers": 4,
    "vocab_size": 1000
}

def create_synthetic_mlp(hidden, inter):
    """Creates random MLP weights representing Llama structure (Gate, Up, Down)"""
    # Llama MLP has 3 matrices
    # Gate: [inter, hidden]
    # Up:   [inter, hidden]
    # Down: [hidden, inter]
    gate_proj = torch.randn(inter, hidden) * 0.02
    up_proj = torch.randn(inter, hidden) * 0.02
    down_proj = torch.randn(hidden, inter) * 0.02
    return gate_proj, up_proj, down_proj

def tokenize_mlp_layer(gate, up, down, model_id, layer_id):
    """
    Tokenizes MLP weights by grouping them into Neurons.
    Token[i] = Concat(Gate[i,:], Up[i,:], Down[:,i])
    """
    inter_size = gate.shape[0]
    tokens = []
    positions = []
    
    # Iterate over neurons (intermediate dimension)
    for i in range(inter_size):
        # Flatten and concatenate inputs/outputs related to this specific neuron
        # Token dim = hidden + hidden + hidden = 3 * hidden
        token = torch.cat([gate[i, :], up[i, :], down[:, i]])
        tokens.append(token)
        
        # Position encoding: [Model_ID, Layer_ID, Neuron_ID]
        # We assume Model_ID maps to global sequence position in SANE
        positions.append(torch.tensor([model_id, layer_id, i], dtype=torch.int))
        
    return torch.stack(tokens), torch.stack(positions)

def main():
    print(f"Generating {CONFIG['num_models']} synthetic models...")
    
    all_samples = []
    
    for m_id in range(CONFIG['num_models']):
        # Simulate properties (e.g., accuracy)
        model_acc = 0.5 + torch.rand(1).item() * 0.4
        
        for l_id in range(CONFIG['num_layers']):
            # 1. Generate Weights
            gate, up, down = create_synthetic_mlp(CONFIG['hidden_size'], CONFIG['intermediate_size'])
            
            # 2. Tokenize (Structure-Aware)
            w_tokens, pos_tokens = tokenize_mlp_layer(gate, up, down, m_id, l_id)
            
            # 3. Store
            # Properties: [Accuracy, Layer_ID] (Adding Layer_ID to props to predict depth later)
            props = [model_acc, float(l_id)] 
            
            all_samples.append({
                'w': w_tokens, 
                'p': pos_tokens, 
                'props': props
            })
            
    # Split into Train/Test
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    print(f"Total Samples (Layers): {len(all_samples)}")
    print(f"Token Dimension: {train_data[0]['w'].shape[1]} (3 * {CONFIG['hidden_size']})")
    print(f"Sequence Length: {train_data[0]['w'].shape[0]}")
    
    # Create Datasets
    # We define property keys to allow SANE to linear probe them
    prop_keys = {
        "result_keys": ["test_acc", "layer_id"],
        "config_keys": []
    }
    
    trainset = LLMLayerDataset(train_data, prop_keys)
    testset = LLMLayerDataset(test_data, prop_keys)
    
    dataset_dict = {
        "trainset": trainset,
        "testset": testset
    }
    
    # Save
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir.joinpath("llm_dataset.pt")
    torch.save(dataset_dict, save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    main()