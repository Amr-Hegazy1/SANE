import torch
from transformers import AutoModel
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))

from SANE.llm_adaptation.universal_tokenizer import UniversalWeightTokenizer


def test_model(model_id):
    print(f"Loading model: {model_id}...")
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return False

    print("Model loaded. Extracting state_dict...")
    state_dict = model.state_dict()

    print(f"State dict keys sample: {list(state_dict.keys())[:5]}")

    tokenizer = UniversalWeightTokenizer(chunk_size=64)

    print("Tokenizing...")
    try:
        chunks, pos = tokenizer.tokenize(state_dict)
    except Exception as e:
        print(f"Tokenization failed: {e}")
        for k in list(state_dict.keys())[:20]:
            match = tokenizer.layer_pattern.search(k)
            print(f"Key: {k} -> Match: {match.group(0) if match else 'None'}")
        return False

    print("\n=== Tokenization Results ===")
    print(f"Model: {model_id}")
    print(f"Total Parameters (approx): {chunks.numel()}")
    print(f"Total Chunks: {chunks.shape[0]}")
    print(f"Chunk Shape: {chunks.shape}")

    layers_found = set()
    for p in pos:
        layers_found.add(float(p[0]))

    print(f"Unique Layer Progress Values: {len(layers_found)}")
    print(f"First 5 Positions (Embeddings?):\n{pos[:5]}")
    print(f"Last 5 Positions (Head?):\n{pos[-5:]}")

    return True


if __name__ == "__main__":
    target_model = "google/gemma-3-270m-it"

    fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    if not test_model(target_model):
        print(
            f"\n\nCould not load {target_model} (likely auth/gated). Falling back to {fallback_model} to verify logic on Llama architecture."
        )
        test_model(fallback_model)
