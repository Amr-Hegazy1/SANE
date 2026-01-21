"""Datasets for training SANE on HF LLM token files.

This dataset is intentionally simple and *append-only* relative to SANE.
It expects pre-tokenized samples saved as torch files (.pt), each containing:

- 'w': tokens (float)        [N, tokensize]
- 'm': mask (bool)           [N, tokensize]
- 'p': positions (long)      [N, 3]
- 'props': optional tensor   [P] or None
- 'meta': optional dict

...Then SANE's existing augmentation pipeline (WindowCutter, Noise, etc.) can be
attached via `dataset.transforms` in the trainable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class HFLlmTokenFileDataset(Dataset):
    """Dataset over a directory of tokenized LLM checkpoints.

    Directory layout (recommended):
      root/
        train/
          sample_000.pt
          sample_001.pt
        test/
          ...

    Each file is a dict (see module docstring).

    `samples_per_file` allows oversampling each token file by drawing multiple
    random windows across epochs.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms=None,
        samples_per_file: int = 1,
        cache_files: int = 2,
    ):
        self.root = Path(root).absolute()
        self.split = split
        self.transforms = transforms
        self.samples_per_file = int(samples_per_file)
        self.cache_files = int(cache_files)

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Split directory not found: {split_dir}")

        self.files = sorted(
            [split_dir / f for f in os.listdir(split_dir) if f.endswith(".pt")]
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files found under {split_dir}")

        # very small LRU cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []

        # property keys may be absent (ssl-only)
        self.property_keys: Optional[Dict[str, Any]] = None
        self.properties: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.files) * self.samples_per_file

    def _load(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        
        item = torch.load(path, map_location="cpu")

        # maintain cache
        self._cache[key] = item
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_files:
            k0 = self._cache_order.pop(0)
            self._cache.pop(k0, None)
        return item

    def __getitem__(self, idx: int):
        file_idx = idx // self.samples_per_file
        path = self.files[file_idx]
        item = self._load(path)

        ddx = item["w"]
        mdx = item["m"]
        pos = item["p"]
        props = item.get("props", None)
        if props is None:
            props = torch.tensor([])

        if self.transforms:
            ddx, mdx, pos = self.transforms(ddx, mdx, pos)

        return ddx, mdx, pos, props
