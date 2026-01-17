import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


def _infer_seq_lens_from_data(dataset) -> List[int]:
    """
    Infer sequence lengths from the underlying raw data if available.
    Falls back to computing from mask if needed (slower).
    """
    if hasattr(dataset, "data"):
        data = getattr(dataset, "data")
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            if "w" in data[0]:
                return [int(d["w"].shape[0]) for d in data]
    # Fallback: compute from mask in __getitem__
    lengths = []
    for i in range(len(dataset)):
        ddx, mask, pos, props = dataset[i]
        # mask is [seq, dim], with full rows True for real tokens
        seq_len = int(mask[:, 0].sum().item()) if mask.numel() > 0 else int(ddx.shape[0])
        lengths.append(seq_len)
    return lengths


class LLMWindowedDataset(Dataset):
    """
    Wraps an LLM dataset and returns mask-aware random windows.
    Ensures windows are sampled only from real (non-padded) tokens.
    """

    def __init__(self, base_dataset: Dataset, windowsize: int):
        self.base_dataset = base_dataset
        self.windowsize = int(windowsize)
        self.seq_lens = _infer_seq_lens_from_data(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        ddx, mask, pos, props = self.base_dataset[index]

        seq_len = int(self.seq_lens[index])
        seq_len = min(seq_len, ddx.shape[0])

        # Ensure we have at least windowsize tokens to avoid downstream errors
        if ddx.shape[0] < self.windowsize:
            pad_len = self.windowsize - ddx.shape[0]
            ddx = F.pad(ddx, (0, 0, 0, pad_len), value=0)
            mask = F.pad(mask, (0, 0, 0, pad_len), value=0)
            pos = F.pad(pos, (0, 0, 0, pad_len), value=0)

        if seq_len <= self.windowsize:
            start = 0
        else:
            start = random.randint(0, seq_len - self.windowsize)
        end = start + self.windowsize

        ddx = ddx[start:end]
        mask = mask[start:end]
        pos = pos[start:end]
        return ddx, mask, pos, props


class BucketedBatchSampler(Sampler[List[int]]):
    """
    Groups indices into buckets by sequence length to reduce padding variance.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        bucket_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

        buckets = {}
        for idx, l in enumerate(self.lengths):
            bucket_id = int(l // self.bucket_size)
            buckets.setdefault(bucket_id, []).append(idx)
        self.buckets = buckets

        # Precompute length for __len__
        total = 0
        for _, idxs in self.buckets.items():
            n = len(idxs) // self.batch_size
            if not self.drop_last and len(idxs) % self.batch_size != 0:
                n += 1
            total += n
        self._len = total

    def __iter__(self):
        bucket_ids = list(self.buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_ids)

        for b_id in bucket_ids:
            idxs = list(self.buckets[b_id])
            if self.shuffle:
                random.shuffle(idxs)
            # yield batches from this bucket
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        return self._len


def build_llm_dataloaders(
    dataset_dict,
    batch_size: int,
    windowsize: int,
    bucket_size: Optional[int] = None,
    num_workers: int = 2,
    drop_last: bool = True,
):
    """
    Build DataLoaders with bucketed sampling and mask-aware windowing.
    """
    if bucket_size is None:
        bucket_size = max(int(windowsize) * 8, int(windowsize))

    trainset = dataset_dict["trainset"]
    testset = dataset_dict["testset"]
    valset = dataset_dict.get("valset", None)

    trainset_w = LLMWindowedDataset(trainset, windowsize=windowsize)
    testset_w = LLMWindowedDataset(testset, windowsize=windowsize)
    valset_w = LLMWindowedDataset(valset, windowsize=windowsize) if valset else None

    train_lengths = _infer_seq_lens_from_data(trainset)
    test_lengths = _infer_seq_lens_from_data(testset)
    val_lengths = _infer_seq_lens_from_data(valset) if valset else None

    train_sampler = BucketedBatchSampler(
        lengths=train_lengths,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle=True,
        drop_last=drop_last,
    )
    test_sampler = BucketedBatchSampler(
        lengths=test_lengths,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle=False,
        drop_last=drop_last,
    )
    val_sampler = (
        BucketedBatchSampler(
            lengths=val_lengths,
            batch_size=batch_size,
            bucket_size=bucket_size,
            shuffle=False,
            drop_last=drop_last,
        )
        if val_lengths is not None
        else None
    )

    trainloader = DataLoader(
        trainset_w,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=4,
    )
    testloader = DataLoader(
        testset_w,
        batch_sampler=test_sampler,
        num_workers=num_workers,
        prefetch_factor=4,
    )
    if valset_w is not None:
        valloader = DataLoader(
            valset_w,
            batch_sampler=val_sampler,
            num_workers=num_workers,
            prefetch_factor=4,
        )
    else:
        valloader = None

    return trainset_w, testset_w, valset_w, trainloader, testloader, valloader
