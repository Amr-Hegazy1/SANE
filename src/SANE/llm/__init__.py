"""LLM utilities for SANE.

This subpackage is intentionally additive: it does not modify SANE's existing CNN/NN
pipelines, but provides:

- Reference-free canonicalization (head/neuron ordering) for Transformer LLMs
- Patch-based weight tokenization suited for large matrices
- Dataset utilities for training SANE on HF models

"""

from .gemma3 import (
    load_hf_gemma3,
    detect_gemma3_prefix,
    canonicalize_gemma3_reference_free,
    tokenize_gemma3_patches,
)

from .dataset_hf_llm import HFLlmTokenFileDataset
