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

    def __init__(self, base_dataset: Dataset, windowsize: int, enable_properties: bool = False):
        self.base_dataset = base_dataset
        self.windowsize = int(windowsize)
        # Downstream tasks in core SANE expect `.properties` and can call `__get_weights__`,
        # which materializes the entire dataset into a single tensor. For large LLM zoos
        # this is prohibitively expensive, so keep it opt-in.
        self.base_properties = getattr(base_dataset, "properties", None)
        self.properties = self.base_properties if enable_properties else None
        # Avoid expensive full-dataset length inference at startup.
        # We'll infer the valid (unpadded) seq length from the mask on each __getitem__.
        self.seq_lens = None

    def __get_weights__(self):
        """Delegates to the underlying dataset when downstream tasks are enabled."""
        if self.properties is None:
            raise AttributeError("__get_weights__ is disabled because downstream properties are disabled")
        if hasattr(self.base_dataset, "__get_weights__"):
            return self.base_dataset.__get_weights__()
        raise AttributeError("Base dataset does not implement __get_weights__")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        ddx, mask, pos, props = self.base_dataset[index]

        # mask is [seq, dim] (bool) with full rows True for real tokens
        seq_len = int(mask[:, 0].sum().item()) if mask.numel() > 0 else int(ddx.shape[0])
        seq_len = min(seq_len, int(ddx.shape[0]))

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
    enable_downstream: bool = False,
):
    """
    Build DataLoaders with bucketed sampling and mask-aware windowing.
    """
    if bucket_size is None:
        bucket_size = max(int(windowsize) * 8, int(windowsize))

    trainset = dataset_dict["trainset"]
    testset = dataset_dict["testset"]
    valset = dataset_dict.get("valset", None)

    trainset_w = LLMWindowedDataset(trainset, windowsize=windowsize, enable_properties=enable_downstream)
    testset_w = LLMWindowedDataset(testset, windowsize=windowsize, enable_properties=enable_downstream)
    valset_w = LLMWindowedDataset(valset, windowsize=windowsize, enable_properties=enable_downstream) if valset else None

    # After windowing, all samples have the same effective sequence length (windowsize).
    # So we can avoid scanning/loading the whole dataset just to infer lengths.
    train_lengths = [int(windowsize)] * len(trainset)
    test_lengths = [int(windowsize)] * len(testset)
    val_lengths = [int(windowsize)] * len(valset) if valset else None

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

    dl_kwargs = {
        "num_workers": num_workers,
    }
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = 4

    trainloader = DataLoader(
        trainset_w,
        batch_sampler=train_sampler,
        **dl_kwargs,
    )
    testloader = DataLoader(
        testset_w,
        batch_sampler=test_sampler,
        **dl_kwargs,
    )
    if valset_w is not None:
        valloader = DataLoader(
            valset_w,
            batch_sampler=val_sampler,
            **dl_kwargs,
        )
    else:
        valloader = None

    return trainset_w, testset_w, valset_w, trainloader, testloader, valloader
