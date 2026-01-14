import torch
from torch.utils.data import Dataset
import logging

class LLMLayerDataset(Dataset):
    """
    Custom Dataset for LLM Layers.
    Treats each Layer of an LLM as a sample.
    Each sample is a sequence of Neurons (tokens).
    """
    def __init__(self, data_list, property_keys=None):
        """
        data_list: list of dicts {'w': tokens, 'p': pos, 'props': [val1, val2...]}
        property_keys: dict {'result_keys': ['name1', 'name2'...]} mapping prop indices to names
        """
        self.data = data_list
        self.transforms = None
        
        # Required for SANE Downstream Task Learner
        self.pos = [d['p'] for d in self.data]
        self.positions = self.data[0]['p'] if len(self.data) > 0 else None
        self.mask = None

        # Fix: Convert row-based properties to column-based dictionary
        self.properties = {}
        if property_keys and 'result_keys' in property_keys:
            keys = property_keys['result_keys']
            # Initialize columns
            for k in keys:
                self.properties[k] = []
            
            # Populate columns
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
        # props is just the list of values for this sample
        props = torch.tensor(sample['props']) 
        
        mask = torch.ones_like(ddx, dtype=torch.bool)
        
        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)
            
        return ddx, mask, pos, props

    def __get_weights__(self):
        weights = torch.stack([d['w'] for d in self.data])
        masks = torch.ones_like(weights, dtype=torch.bool)
        return weights, masks