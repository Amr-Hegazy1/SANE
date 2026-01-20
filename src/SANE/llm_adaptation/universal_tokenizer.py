import torch
import re
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging


class UniversalWeightTokenizer:
    def __init__(self, chunk_size: int = 64):
        self.chunk_size = chunk_size
        self.layer_keywords = {"layers", "blocks", "h", "transformer"}

    def _get_layer_info(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[int, Dict[str, int]]:
        max_layer = 0
        key_to_layer = {}

        for key in state_dict.keys():
            parts = key.split(".")
            layer_idx = -1

            for i, part in enumerate(parts[:-1]):
                if part in self.layer_keywords:
                    if parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break

            key_to_layer[key] = layer_idx
            if layer_idx > max_layer:
                max_layer = layer_idx

        return max_layer + 1, key_to_layer

    def _topological_sort(self, keys: List[str]) -> List[str]:
        def sort_key(k):
            if "embed" in k or "wte" in k:
                return (0, 0, k)

            parts = k.split(".")
            layer_idx = -1
            for i, part in enumerate(parts[:-1]):
                if part in self.layer_keywords and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            if layer_idx != -1:
                return (1, layer_idx, k)

            if "lm_head" in k or "final" in k or "output" in k or "norm" in k:
                return (2, 0, k)

            return (1.5, 0, k)

        return sorted(keys, key=sort_key)

    def tokenize(
        self,
        state_dict: Dict[str, torch.Tensor],
        total_layers_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks_list = []
        pos_list = []

        if total_layers_override is not None:
            num_layers = total_layers_override
            _, key_to_layer = self._get_layer_info(state_dict)
        else:
            num_layers, key_to_layer = self._get_layer_info(state_dict)

        sorted_keys = self._topological_sort(list(state_dict.keys()))

        for key in sorted_keys:
            tensor = state_dict[key]

            if not tensor.is_floating_point():
                continue

            flat_w = tensor.view(-1)
            num_elements = flat_w.numel()

            remainder = num_elements % self.chunk_size
            if remainder > 0:
                padding = self.chunk_size - remainder
                flat_w = torch.cat(
                    [
                        flat_w,
                        torch.zeros(padding, device=flat_w.device, dtype=flat_w.dtype),
                    ]
                )

            num_chunks = flat_w.numel() // self.chunk_size
            layer_chunks = flat_w.view(num_chunks, self.chunk_size)

            chunks_list.append(layer_chunks)

            layer_idx = key_to_layer.get(key, -1)

            if layer_idx == -1:
                if "embed" in key:
                    layer_progress = 0.0
                elif "lm_head" in key or "final" in key:
                    layer_progress = 1.0
                else:
                    layer_progress = 0.5
            else:
                layer_progress = layer_idx / max(1, num_layers - 1)

            intra_progress = torch.linspace(0, 1, num_chunks)

            layer_prog_tensor = torch.full((num_chunks,), layer_progress)

            key_pos = torch.stack([layer_prog_tensor, intra_progress], dim=1)
            pos_list.append(key_pos)

        if not chunks_list:
            return torch.empty(0, self.chunk_size), torch.empty(0, 2)

        return torch.cat(chunks_list, dim=0), torch.cat(pos_list, dim=0)


if __name__ == "__main__":
    tokenizer = UniversalWeightTokenizer(chunk_size=4)

    mock_state = {
        "model.embed_tokens.weight": torch.randn(10, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 8),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8),
        "lm_head.weight": torch.randn(10, 8),
    }

    chunks, pos = tokenizer.tokenize(mock_state)

    print(f"Total Chunks: {chunks.shape[0]}")
    print(f"Chunk Shape: {chunks.shape}")
    print(f"Pos Shape: {pos.shape}")
    print(f"First 5 Pos (Embeddings):\n{pos[:5]}")
    print(f"Middle Pos (Layer 0):\n{pos[25:30]}")
    print(f"Last 5 Pos (Head):\n{pos[-5:]}")
