import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import argparse
import numpy as np
from typing import List, Dict
import json

try:
    from src.SANE.llm_adaptation.universal_tokenizer import UniversalWeightTokenizer
    from src.SANE.llm_adaptation.dataset_universal import UniversalWeightDataset
    from src.SANE.llm_adaptation.universal_sane import UniversalSANE
except ImportError:
    from universal_tokenizer import UniversalWeightTokenizer
    from dataset_universal import UniversalWeightDataset
    from universal_sane import UniversalSANE


def get_model_embedding(
    model: torch.nn.Module,
    chunks: torch.Tensor,
    positions: torch.Tensor,
    window_size: int = 1024,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()

    num_tokens = chunks.shape[0]

    remainder = num_tokens % window_size
    if remainder > 0:
        pad_len = window_size - remainder
        chunks = torch.cat(
            [chunks, torch.zeros(pad_len, chunks.shape[1]).to(chunks.dtype)], dim=0
        )
        positions = torch.cat(
            [positions, torch.zeros(pad_len, 2).to(positions.dtype)], dim=0
        )

    num_windows = chunks.shape[0] // window_size

    latents_sum = 0
    total_tokens = 0

    for i in range(0, num_windows, batch_size):
        end_idx = min(i + batch_size, num_windows)
        batch_windows_chunks = []
        batch_windows_pos = []

        for w in range(i, end_idx):
            start = w * window_size
            end = start + window_size
            batch_windows_chunks.append(chunks[start:end])
            batch_windows_pos.append(positions[start:end])

        b_chunks = torch.stack(batch_windows_chunks).to(device)
        b_pos = torch.stack(batch_windows_pos).to(device)

        b_chunks = b_chunks.to(model.input_proj.weight.dtype)
        b_pos = b_pos.to(model.input_proj.weight.dtype)

        with torch.no_grad():
            _, latents = model(b_chunks, b_pos)

        latents_sum += latents.sum(dim=(0, 1)).cpu().numpy()
        total_tokens += latents.shape[0] * latents.shape[1]

    global_mean_embedding = latents_sum / total_tokens
    return global_mean_embedding


def evaluate(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    print(f"Loading SANE from {args.checkpoint_path}...")
    sane_model = UniversalSANE(
        input_dim=args.chunk_size,
        d_model=args.d_model,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
    )

    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    sane_model.load_state_dict(state_dict)
    sane_model = accelerator.prepare(sane_model)
    sane_model.eval()

    if args.models_json:
        with open(args.models_json, "r") as f:
            model_list = json.load(f)
    elif args.models:
        model_list = args.models.split(",")
    else:
        model_list = ["google/gemma-3-270m-it"]

    dataset = UniversalWeightDataset(
        model_paths=model_list,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        force_reprocess=False,
    )

    results = {}

    print("Computing embeddings...")
    for i in tqdm(range(len(dataset))):
        model_name = dataset.model_paths[i]
        chunks, pos = dataset[i]

        embedding = get_model_embedding(
            sane_model,
            chunks,
            pos,
            window_size=args.window_size,
            batch_size=args.batch_size,
            device=accelerator.device,
        )

        results[model_name] = embedding.tolist()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "model_embeddings.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Embeddings saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained SANE checkpoint (.pt)",
    )
    parser.add_argument(
        "--models", type=str, help="Comma separated list of HF model IDs"
    )
    parser.add_argument(
        "--models_json", type=str, help="Path to JSON file containing list of model IDs"
    )
    parser.add_argument("--output_dir", type=str, default="./output_eval")
    parser.add_argument("--cache_dir", type=str, default="./data/universal_cache")

    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    args = parser.parse_args()
    evaluate(args)
