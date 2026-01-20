# Universal SANE: Model-Agnostic LLM Weight Embeddings

This directory contains the implementation of **Universal SANE**, an extension of SANE designed to learn representations of **Large Language Models (LLMs)** across different architectures (Llama, Mistral, Qwen, etc.) and sizes.

## 🧠 Core Concept: The Universal Weight Encoder

Unlike the original SANE which was tied to specific architectures (e.g., ResNet-18) via fixed tokenization and integer positional IDs, Universal SANE treats neural network weights as a continuous stream of "functional matter."

### Key Innovations

1.  **Dimension-Agnostic Chunking**:
    Instead of tokenizing by "Neuron" (which varies in size from 4096 to 12000+), we chop all weights into fixed-size **Chunks** (default: 64 floats). A larger model simply produces a longer stream of chunks.

2.  **Relative Topological Positioning**:
    Instead of absolute Layer IDs (which break when comparing 12-layer vs 80-layer models), we use **Continuous Progress Coordinates** (0.0 to 1.0).
    *   Embeddings $\approx 0.0$
    *   Middle Layers $\approx 0.5$
    *   Head/Output $\approx 1.0$
    
    This allows SANE to generalize: "This chunk belongs to the late-stage reasoning layers" regardless of the model's total depth.

3.  **Universal Tokenizer**:
    Robustly parses standard LLM naming conventions (`layers.0`, `h.5`, `blocks.12`) to automatically assign these topological coordinates.

## 🚀 Quick Start

### 1. Training (Pre-training SANE)

Train a Universal SANE model on a "Zoo" of LLMs. You define the zoo simply by listing HuggingFace Model IDs. The system handles downloading, caching, and streaming.

```bash
accelerate launch src/SANE/llm_adaptation/train_universal_sane.py \
    --epochs 10 \
    --batch_size 4 \
    --window_size 1024 \
    --save_every 1 \
    --mixed_precision fp16
```

*Note: By default, the script uses a demo list (`gemma-3-270m-it`). Edit `train_universal_sane.py` to add `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.1`, etc.*

### 2. Evaluation (Generating Embeddings)

Once trained, use the checkpoint to generate global embeddings for any LLM.

```bash
python src/SANE/llm_adaptation/eval_universal_sane.py \
    --checkpoint_path output_sane/checkpoint_epoch_2.pt \
    --models "google/gemma-3-270m-it,TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --output_dir output_eval
```

This produces `model_embeddings.json` mapping Model Name $\to$ 512-dim Vector.

### 3. Visualization (Clustering)

Visualize the "Weight Space" of your LLMs.

```bash
python src/SANE/llm_adaptation/cluster_llms.py \
    --input output_eval/model_embeddings.json \
    --output output_eval/embedding_plot.png
```

## 📂 File Structure

*   `universal_tokenizer.py`: Logic for chunking weights and calculating relative positions.
*   `dataset_universal.py`: PyTorch Dataset that caches tokenized models to disk.
*   `universal_sane.py`: The Transformer Autoencoder with Continuous Sinusoidal Embeddings.
*   `train_universal_sane.py`: Multi-GPU training loop using 🤗 Accelerate.
*   `eval_universal_sane.py`: Inference script for downstream tasks.
*   `cluster_llms.py`: PCA visualization of learned embeddings.

## 🔧 Extending

To support custom model architectures (e.g., Mamba, RWKV) with weird naming conventions, simply update the `layer_keywords` set in `universal_tokenizer.py`:

```python
self.layer_keywords = {'layers', 'blocks', 'h', 'transformer', 'mixer', 'rwkv'}
```
