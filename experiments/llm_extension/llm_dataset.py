import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
from pathlib import Path


class LLMBatchDataset(Dataset):
    """Lazy dataset backed by on-disk batch files.

    This avoids loading the entire zoo into RAM. Each item references a
    (batch_file, sample_idx) pair. Only the required batch file is loaded
    (and cached) on demand.

    refs: list of dicts with keys:
      - 'file': str path to a .pt batch file
      - 'idx': int index within that batch list
      - 'props': list[float]
      - optional: 'info'
    """

    def __init__(self, refs, token_dim, max_seq_len, property_keys=None):
        self.refs = refs
        self.token_dim = int(token_dim)
        self.max_seq_len = int(max_seq_len)
        self.transforms = None

        self._cache_file = None
        self._cache_data = None

        self.properties = {}
        if property_keys and 'result_keys' in property_keys:
            keys = property_keys['result_keys']
            for k in keys:
                self.properties[k] = []
            for r in self.refs:
                vals = r.get('props', [])
                for i, k in enumerate(keys):
                    self.properties[k].append(vals[i] if i < len(vals) else None)

    def __len__(self):
        return len(self.refs)

    def _load_batch(self, file_path: str):
        # Ray typically chdirs into a trial dir, so relative paths saved in refs
        # may not resolve. We try a couple of stable bases.
        p = Path(file_path)
        if not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parents[2]
            candidate = repo_root / p
            if candidate.exists():
                p = candidate

        resolved = str(p.resolve()) if p.exists() else str(p)

        if self._cache_file == resolved and self._cache_data is not None:
            return
        if not Path(resolved).exists():
            raise FileNotFoundError(f"Batch file not found: {file_path} (resolved: {resolved})")

        self._cache_data = torch.load(resolved, map_location='cpu', weights_only=False)
        self._cache_file = resolved

    def __getitem__(self, index):
        ref = self.refs[index]
        batch_file = ref['file']
        sample_idx = int(ref['idx'])

        self._load_batch(batch_file)
        sample = self._cache_data[sample_idx]

        ddx = sample['w']
        pos = sample['p']
        props = torch.tensor(ref.get('props', sample.get('props', [])))

        curr_seq = ddx.shape[0]
        if curr_seq < self.max_seq_len:
            pad_len = self.max_seq_len - curr_seq
            ddx = F.pad(ddx, (0, 0, 0, pad_len), value=0)
            pos = F.pad(pos, (0, 0, 0, pad_len), value=0)
            mask = torch.zeros((self.max_seq_len, self.token_dim), dtype=torch.bool)
            mask[:curr_seq, :] = True
        else:
            if curr_seq > self.max_seq_len:
                ddx = ddx[: self.max_seq_len]
                pos = pos[: self.max_seq_len]
            mask = torch.ones((self.max_seq_len, self.token_dim), dtype=torch.bool)

        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)

        return ddx, mask, pos, props

class LLMLayerDataset(Dataset):
    """
    Custom Dataset for LLM Layers.
    Treats each Layer of an LLM as a sample.
    Each sample is a sequence of Neurons (tokens).
    Handles variable sequence lengths via padding.
    """
    def __init__(self, data_list, property_keys=None):
        """
        data_list: list of dicts {'w': tokens, 'p': pos, 'props': [val1, val2...]}
        property_keys: dict {'result_keys': ['name1', 'name2'...]} mapping prop indices to names
        """
        self.data = data_list
        self.transforms = None
        
        # 1. Determine Max Sequence Length for Padding
        if len(self.data) > 0:
            self.max_seq_len = max([d['w'].shape[0] for d in self.data])
            self.token_dim = self.data[0]['w'].shape[1]
            print(f"LLMLayerDataset: Max sequence length found: {self.max_seq_len}")
        else:
            self.max_seq_len = 0
            self.token_dim = 0

        # Required for SANE Downstream Task Learner
        # We must pad these positions too if we want to stack them
        self.pos = []
        for d in self.data:
            p = d['p']
            # Pad Position: [Seq, 3] -> [Max_Seq, 3]
            if p.shape[0] < self.max_seq_len:
                pad_len = self.max_seq_len - p.shape[0]
                # Pad with 0s (ignored by mask anyway)
                p = F.pad(p, (0, 0, 0, pad_len), value=0)
            self.pos.append(p)
            
        self.positions = self.pos[0] if len(self.pos) > 0 else None
        self.mask = None

        # Convert row-based properties to column-based dictionary
        self.properties = {}
        if property_keys and 'result_keys' in property_keys:
            keys = property_keys['result_keys']
            for k in keys:
                self.properties[k] = []
            
            for item in data_list:
                vals = item['props']
                for i, k in enumerate(keys):
                    self.properties[k].append(vals[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        
        ddx = sample['w'] # [Sequence, TokenDim]
        pos = sample['p'] # [Sequence, 3]
        props = torch.tensor(sample['props']) 
        
        # 2. Dynamic Padding
        curr_seq = ddx.shape[0]
        if curr_seq < self.max_seq_len:
            pad_len = self.max_seq_len - curr_seq
            
            # Pad Weights: [Seq, Dim] -> [Max_Seq, Dim]
            ddx = F.pad(ddx, (0, 0, 0, pad_len), value=0)
            
            # Pad Positions: [Seq, 3] -> [Max_Seq, 3]
            pos = F.pad(pos, (0, 0, 0, pad_len), value=0)
            
            # Create Mask: 1 for Real, 0 for Pad
            # Shape must match ddx: [Max_Seq, Dim]
            mask = torch.zeros((self.max_seq_len, self.token_dim), dtype=torch.bool)
            mask[:curr_seq, :] = True
        else:
            # No padding needed
            mask = torch.ones_like(ddx, dtype=torch.bool)
        
        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)
            
        return ddx, mask, pos, props

    def __get_weights__(self):
        # This function constructs the full tensor for downstream tasks.
        # Since we handle padding in __init__/_getitem__, we need to replicate it here
        # or iterate. For SANE, iterating __getitem__ is safer to ensure padding consistency.
        
        weights = []
        masks = []
        for i in range(len(self)):
            d, m, _, _ = self.__getitem__(i)
            weights.append(d)
            masks.append(m)
            
        return torch.stack(weights), torch.stack(masks)