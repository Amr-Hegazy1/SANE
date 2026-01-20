import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Union
import logging
import os
import glob
from tqdm import tqdm
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

try:
    from .universal_tokenizer import UniversalWeightTokenizer
except ImportError:
    from universal_tokenizer import UniversalWeightTokenizer


class UniversalWeightDataset(Dataset):
    def __init__(
        self,
        model_paths: List[str],
        cache_dir: str = "./data/universal_cache",
        chunk_size: int = 64,
        force_reprocess: bool = False,
    ):
        self.model_paths = model_paths
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.tokenizer = UniversalWeightTokenizer(chunk_size=chunk_size)

        self.data_files = []
        self._prepare_dataset(force_reprocess)

    def _prepare_dataset(self, force_reprocess: bool):
        print(f"Preparing dataset with {len(self.model_paths)} models...")

        for model_path in tqdm(self.model_paths):
            safe_name = model_path.replace("/", "_").replace("\\", "_")
            cache_path = self.cache_dir / f"{safe_name}_chunks.pt"

            if cache_path.exists() and not force_reprocess:
                self.data_files.append(cache_path)
                continue

            try:
                print(f"Processing {model_path}...")

                local_dir = model_path
                if not os.path.exists(model_path):
                    local_dir = snapshot_download(
                        model_path,
                        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model"],
                        ignore_patterns=["*.msgpack", "*.h5"],
                    )

                try:
                    config = AutoConfig.from_pretrained(
                        local_dir, trust_remote_code=True
                    )
                    num_layers = getattr(config, "num_hidden_layers", None)
                    if num_layers is None:
                        num_layers = getattr(config, "n_layer", None)
                    if num_layers is None:
                        num_layers = getattr(config, "num_layers", None)

                    if num_layers is None:
                        print(
                            f"Warning: Could not determine num_layers for {model_path}. Using default estimation."
                        )
                    else:
                        num_layers += 1
                except Exception as e:
                    print(
                        f"Config load failed: {e}. Proceeding without global layer count."
                    )
                    num_layers = None

                files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
                if not files:
                    files = sorted(glob.glob(os.path.join(local_dir, "*.bin")))

                if not files:
                    print(f"No weights found for {model_path}")
                    continue

                all_chunks = []
                all_pos = []

                for file in tqdm(files, desc=f"Shards {model_path}", leave=False):
                    if file.endswith(".safetensors"):
                        state_dict = load_safetensors(file)
                    else:
                        state_dict = torch.load(file, map_location="cpu")

                    chunks, pos = self.tokenizer.tokenize(
                        state_dict, total_layers_override=num_layers
                    )

                    if chunks.numel() > 0:
                        all_chunks.append(chunks)
                        all_pos.append(pos)

                    del state_dict
                    del chunks
                    del pos
                    torch.cuda.empty_cache()

                if all_chunks:
                    full_chunks = torch.cat(all_chunks, dim=0)
                    full_pos = torch.cat(all_pos, dim=0)

                    torch.save(
                        {"chunks": full_chunks, "positions": full_pos}, cache_path
                    )
                    self.data_files.append(cache_path)

                    del full_chunks
                    del full_pos

            except Exception as e:
                print(f"Failed to process {model_path}: {e}")
                import traceback

                traceback.print_exc()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data["chunks"], data["positions"]


if __name__ == "__main__":
    models = ["google/gemma-3-270m-it"]

    dataset = UniversalWeightDataset(models, force_reprocess=True)

    print(f"Dataset size: {len(dataset)}")
    chunks, pos = dataset[0]
    print(f"Item 0 Chunks: {chunks.shape}")
    print(f"Item 0 Pos: {pos.shape}")
