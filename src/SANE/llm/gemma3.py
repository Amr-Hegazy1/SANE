"""Gemma 3 (text) helpers for SANE.

Primary target (initial): `google/gemma-3-270m-it`.

Key features
- Reference-free canonicalization of *symmetry* dimensions:
  - Attention head order (q/k/v/o projections)
  - MLP intermediate neuron order (SwiGLU gate/up + down projection)
- Patch-based tokenization for large matrices, producing a fixed token width.

Notes
- Gemma 3 attention can be MQA/GQA. Canonicalization supports both by permuting
  KV heads (if >1) and corresponding Q groups, plus optionally Q-head order within
  each KV group.
- This module does not require editing any existing SANE files.

"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# -------------------------------
# Hugging Face loading
# -------------------------------

def load_hf_gemma3(
    model_id_or_path: str,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load a Gemma 3 text-only causal LM from Hugging Face.

    Returns:
        (state_dict, config_dict)

    This function uses `transformers` if installed.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    except Exception as e:
        raise ImportError(
            "Loading HF Gemma 3 requires `transformers`. Install e.g. `pip install transformers`. "
            "Gemma 3 may require a recent/dev transformers version."
        ) from e

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype={dtype}. Use float32/float16/bfloat16.")

    config = AutoConfig.from_pretrained(model_id_or_path, token=token, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        token=token,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": device} if device != "cpu" else None,
    )
    state_dict = model.state_dict()

    # keep config minimal and serializable
    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return state_dict, cfg


# -------------------------------
# Key resolution
# -------------------------------

def detect_gemma3_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    """Detect the prefix that precedes `layers.{i}.*` in a Gemma 3 text model state_dict.

    HF keys often look like:
      - `model.layers.0.self_attn.q_proj.weight`
      - `language_model.model.layers.0.self_attn.q_proj.weight`
      - `base_model.model.language_model.model.layers.0...` (when wrapped)

    We return the prefix up to (and including) the trailing dot.
    """
    target_suffix = "layers.0.self_attn.q_proj.weight"
    for k in state_dict.keys():
        if k.endswith(target_suffix):
            return k[: -len(target_suffix)]

    # fallback: regex search for any layer index
    pat = re.compile(r"^(.*)layers\\.\\d+\\.self_attn\\.q_proj\\.weight$")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            return m.group(1)

    raise KeyError(
        "Could not detect Gemma3 layer prefix. Expected a key ending with "
        f"`{target_suffix}` (or similar)."
    )


@dataclass
class Gemma3TextHyper:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool

    @staticmethod
    def from_config(cfg: Dict) -> "Gemma3TextHyper":
        # Gemma 3 text config nested vs flat differs across wrappers.
        # Common HF: cfg already corresponds to Gemma3TextConfig.
        def _get(*keys, default=None):
            for kk in keys:
                if kk in cfg:
                    return cfg[kk]
            return default

        hidden_size = int(_get("hidden_size"))
        intermediate_size = int(_get("intermediate_size"))
        num_hidden_layers = int(_get("num_hidden_layers"))
        num_attention_heads = int(_get("num_attention_heads"))
        num_key_value_heads = int(_get("num_key_value_heads", default=num_attention_heads))
        head_dim = int(_get("head_dim", default=hidden_size // num_attention_heads))
        attention_bias = bool(_get("attention_bias", default=False))

        return Gemma3TextHyper(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
        )


# -------------------------------
# Reference-free canonicalization
# -------------------------------

def _svd_signature(mat: torch.Tensor, topk: int = 8) -> torch.Tensor:
    """Compute a deterministic per-matrix signature.

    Returns a 1D tensor of length (topk + 2):
      [topk singular values (desc), frob_norm, mean_abs]
    """
    m = mat.detach().float()
    # Use SVD vals; robust, deterministic on CPU.
    s = torch.linalg.svdvals(m)
    k = min(topk, s.numel())
    top = s[:k]
    if k < topk:
        top = torch.nn.functional.pad(top, (0, topk - k))
    frob = torch.linalg.norm(m)
    mean_abs = m.abs().mean()
    return torch.cat([top, frob.view(1), mean_abs.view(1)], dim=0)


def _lexsort(signatures: torch.Tensor, decimals: int = 6) -> torch.LongTensor:
    """Lexicographic sort over signature rows (deterministic).

    We round to improve stability under tiny float noise.
    """
    import numpy as np

    sig = signatures.detach().cpu().numpy()
    sig = np.round(sig, decimals=decimals)
    # np.lexsort sorts by last key first
    keys = [sig[:, j] for j in range(sig.shape[1] - 1, -1, -1)]
    order = np.lexsort(keys)
    return torch.as_tensor(order, dtype=torch.long)


def _permute_rows_by_blocks(weight: torch.Tensor, perm: torch.LongTensor, block: int) -> torch.Tensor:
    """Permute contiguous row blocks of size `block` according to `perm`.

    weight: [n*block, in]
    perm: [n]
    """
    out, inn = weight.shape
    assert out % block == 0, "out_features must be divisible by block"
    n = out // block
    w = weight.view(n, block, inn)
    w = w.index_select(0, perm.to(w.device))
    return w.reshape(out, inn)


def _permute_cols_by_blocks(weight: torch.Tensor, perm: torch.LongTensor, block: int) -> torch.Tensor:
    """Permute contiguous column blocks of size `block` according to `perm`.

    weight: [out, n*block]
    perm: [n]
    """
    out, inn = weight.shape
    assert inn % block == 0, "in_features must be divisible by block"
    n = inn // block
    w = weight.view(out, n, block)
    w = w.index_select(1, perm.to(w.device))
    return w.reshape(out, inn)


def canonicalize_gemma3_reference_free(
    state_dict: Dict[str, torch.Tensor],
    cfg: Dict,
    *,
    prefix: Optional[str] = None,
    signature_topk: int = 8,
) -> Dict[str, torch.Tensor]:
    """Return a *new* state_dict with reference-free canonicalization applied.

    Canonicalizes within each decoder layer:
      - attention head order (q_proj and corresponding o_proj; plus kv if kv_heads>1)
      - MLP intermediate neuron order (gate/up rows and down columns)

    The model function is preserved (only symmetry dimensions are permuted).
    """
    hyper = Gemma3TextHyper.from_config(cfg)
    if prefix is None:
        prefix = detect_gemma3_prefix(state_dict)

    out: Dict[str, torch.Tensor] = dict(state_dict)  # shallow copy

    # helper to get/set with prefix
    def k_layer(i: int, suffix: str) -> str:
        return f"{prefix}layers.{i}.{suffix}"

    for layer_idx in range(hyper.num_hidden_layers):
        # --- Attention ---
        qk = k_layer(layer_idx, "self_attn.q_proj.weight")
        kk = k_layer(layer_idx, "self_attn.k_proj.weight")
        vk = k_layer(layer_idx, "self_attn.v_proj.weight")
        ok = k_layer(layer_idx, "self_attn.o_proj.weight")

        Wq = out[qk]
        Wk = out[kk]
        Wv = out[vk]
        Wo = out[ok]

        n_q = hyper.num_attention_heads
        n_kv = hyper.num_key_value_heads
        hd = hyper.head_dim

        # sanity
        assert Wq.shape[0] == n_q * hd, f"Unexpected q_proj out_features: {Wq.shape}"
        assert Wo.shape[1] == n_q * hd, f"Unexpected o_proj in_features: {Wo.shape}"
        assert Wk.shape[0] == n_kv * hd, f"Unexpected k_proj out_features: {Wk.shape}"
        assert Wv.shape[0] == n_kv * hd, f"Unexpected v_proj out_features: {Wv.shape}"

        # Step 1: permute KV heads (if >1), plus corresponding Q groups.
        group_size = n_q // max(n_kv, 1)
        if n_kv > 1:
            sig_kv = []
            for h in range(n_kv):
                k_block = Wk[h * hd : (h + 1) * hd, :]
                v_block = Wv[h * hd : (h + 1) * hd, :]
                sig = torch.cat([
                    _svd_signature(k_block, topk=signature_topk),
                    _svd_signature(v_block, topk=signature_topk),
                ])
                sig_kv.append(sig)
            sig_kv_t = torch.stack(sig_kv, dim=0)
            perm_kv = _lexsort(sig_kv_t)

            # apply to K/V
            Wk = _permute_rows_by_blocks(Wk, perm_kv, hd)
            Wv = _permute_rows_by_blocks(Wv, perm_kv, hd)

            # apply to Q by KV groups
            # reshape Q as [kv, group, hd, in]
            q_out, q_in = Wq.shape
            Wq_g = Wq.view(n_kv, group_size, hd, q_in)
            Wq_g = Wq_g.index_select(0, perm_kv.to(Wq_g.device))
            Wq = Wq_g.reshape(q_out, q_in)

            # apply to O columns by KV groups
            o_out, o_in = Wo.shape
            Wo_g = Wo.view(o_out, n_kv, group_size, hd)
            Wo_g = Wo_g.index_select(1, perm_kv.to(Wo_g.device))
            Wo = Wo_g.reshape(o_out, o_in)

        # Step 2: permute Q heads *within each KV group*.
        # For MQA (n_kv==1), this is simply a full head permutation.
        q_out, q_in = Wq.shape
        Wq_g2 = Wq.view(n_kv, group_size, hd, q_in)
        o_out, o_in = Wo.shape
        Wo_g2 = Wo.view(o_out, n_kv, group_size, hd)

        for g in range(n_kv):
            sig_q = []
            for h_in_g in range(group_size):
                q_block = Wq_g2[g, h_in_g, :, :]
                sig_q.append(_svd_signature(q_block, topk=signature_topk))
            sig_q_t = torch.stack(sig_q, dim=0)
            perm_q = _lexsort(sig_q_t)

            Wq_g2[g] = Wq_g2[g].index_select(0, perm_q.to(Wq_g2.device))
            Wo_g2[:, g] = Wo_g2[:, g].index_select(1, perm_q.to(Wo_g2.device))

        Wq = Wq_g2.reshape(q_out, q_in)
        Wo = Wo_g2.reshape(o_out, o_in)

        out[qk] = Wq
        out[kk] = Wk
        out[vk] = Wv
        out[ok] = Wo

        # --- MLP (SwiGLU) ---
        gk = k_layer(layer_idx, "mlp.gate_proj.weight")
        uk = k_layer(layer_idx, "mlp.up_proj.weight")
        dk = k_layer(layer_idx, "mlp.down_proj.weight")

        Wg = out[gk]
        Wu = out[uk]
        Wd = out[dk]

        inter = hyper.intermediate_size
        assert Wg.shape[0] == inter and Wu.shape[0] == inter and Wd.shape[1] == inter, (
            f"Unexpected MLP shapes: gate {Wg.shape}, up {Wu.shape}, down {Wd.shape}"
        )

        # neuron signature (row norms + down col norm)
        sig_n = torch.stack(
            [
                Wg.detach().float().norm(dim=1),
                Wu.detach().float().norm(dim=1),
                Wd.detach().float().norm(dim=0),
            ],
            dim=1,
        )
        perm_n = _lexsort(sig_n)

        out[gk] = Wg.index_select(0, perm_n.to(Wg.device))
        out[uk] = Wu.index_select(0, perm_n.to(Wu.device))
        out[dk] = Wd.index_select(1, perm_n.to(Wd.device))

    return out


# -------------------------------
# Patch tokenization
# -------------------------------

@torch.no_grad()
def _tokenize_vector(vec: torch.Tensor, tokensize: int) -> Tuple[torch.Tensor, torch.Tensor]:
    v = vec.detach().flatten()
    tok = torch.zeros(tokensize, dtype=v.dtype)
    m = torch.zeros(tokensize, dtype=torch.bool)
    n = min(tokensize, v.numel())
    tok[:n] = v[:n]
    m[:n] = True
    return tok, m


@torch.no_grad()
def _tokenize_matrix_patches(
    mat: torch.Tensor,
    *,
    patch_rows: int,
    patch_cols: int,
    tokensize: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a 2D matrix into fixed-size patch tokens.

    Returns:
      tokens: [n_tokens, tokensize]
      mask:   [n_tokens, tokensize] (True where real weights)
      row_ids: [n_tokens] patch row-block index
      col_ids: [n_tokens] patch col-block index
    """
    assert mat.dim() == 2
    out_dim, in_dim = mat.shape

    n_rb = math.ceil(out_dim / patch_rows)
    n_cb = math.ceil(in_dim / patch_cols)

    tokens = []
    masks = []
    row_ids = []
    col_ids = []

    for rb in range(n_rb):
        r0 = rb * patch_rows
        r1 = min((rb + 1) * patch_rows, out_dim)
        for cb in range(n_cb):
            c0 = cb * patch_cols
            c1 = min((cb + 1) * patch_cols, in_dim)

            patch = mat[r0:r1, c0:c1].detach()
            # pad patch to (patch_rows, patch_cols)
            if patch.shape[0] != patch_rows or patch.shape[1] != patch_cols:
                pad_r = patch_rows - patch.shape[0]
                pad_c = patch_cols - patch.shape[1]
                patch = torch.nn.functional.pad(patch, (0, pad_c, 0, pad_r))

            flat = patch.flatten()
            assert flat.numel() == tokensize, f"patch flatten {flat.numel()} != tokensize {tokensize}"

            mask = torch.zeros(tokensize, dtype=torch.bool)
            # real region is (r1-r0) x (c1-c0)
            real_r = r1 - r0
            real_c = c1 - c0
            # mark real elements in row-major order
            # indices: i*patch_cols + j
            for rr in range(real_r):
                start = rr * patch_cols
                mask[start : start + real_c] = True

            tokens.append(flat)
            masks.append(mask)
            row_ids.append(rb)
            col_ids.append(cb)

    return (
        torch.stack(tokens, dim=0),
        torch.stack(masks, dim=0),
        torch.tensor(row_ids, dtype=torch.long),
        torch.tensor(col_ids, dtype=torch.long),
    )


@dataclass
class Gemma3TokenizationConfig:
    tokensize: int = 4096
    patch_rows: int = 64
    patch_cols: int = 64
    include_embeddings: bool = False
    include_lm_head: bool = False


@torch.no_grad()
def tokenize_gemma3_patches(
    state_dict: Dict[str, torch.Tensor],
    cfg: Dict,
    *,
    prefix: Optional[str] = None,
    tcfg: Gemma3TokenizationConfig = Gemma3TokenizationConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Tokenize Gemma 3 text model weights into (tokens, mask, pos).

    Output shapes:
      tokens: [N, tokensize]
      mask:   [N, tokensize]
      pos:    [N, 3]  (block_token_index, block_id, group_id)

    - block_id enumerates: embed_tokens (0, optional), layer 0..L-1 (offset), final_norm, lm_head (optional)
    - block_token_index is *within-block* token ordinal, including module-type offsets
    - group_id is module-specific grouping (e.g., head id for attention patches, neuron-block id for MLP, else 0)

    Returns also a `meta` dict describing boundaries.
    """
    hyper = Gemma3TextHyper.from_config(cfg)
    if prefix is None:
        prefix = detect_gemma3_prefix(state_dict)

    tokens_all: List[torch.Tensor] = []
    masks_all: List[torch.Tensor] = []
    pos_all: List[torch.Tensor] = []

    meta = {
        "blocks": [],  # list of dicts per block
        "tokensize": tcfg.tokensize,
        "patch_rows": tcfg.patch_rows,
        "patch_cols": tcfg.patch_cols,
    }

    # determine special keys outside prefix
    # try a few common candidates
    def _first_existing(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in state_dict:
                return c
        return None

    embed_key = _first_existing([
        f"{prefix}embed_tokens.weight",
        f"model.embed_tokens.weight",
        f"language_model.model.embed_tokens.weight",
        "embed_tokens.weight",
    ])

    norm_key = _first_existing([
        f"{prefix}norm.weight",
        f"model.norm.weight",
        f"language_model.model.norm.weight",
        "norm.weight",
    ])

    lm_head_key = _first_existing([
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ])

    block_id = 0

    def add_block(block_name: str, entries: List[Tuple[str, torch.Tensor, str]]):
        """Add a block to outputs.

        entries: list of (param_name, tensor, module_type)
          module_type used to define module offsets and group id rules.
        """
        nonlocal block_id
        b_start = sum(t.shape[0] for t in tokens_all) if tokens_all else 0

        # first pass to tokenize each entry and compute module offsets
        module_offsets: Dict[str, int] = {}
        module_token_counts: Dict[str, int] = {}

        tokenized_entries = []
        running = 0

        for pname, tensor, mtype in entries:
            if tensor.dim() == 1:
                tok, m = _tokenize_vector(tensor, tcfg.tokensize)
                tok = tok.unsqueeze(0)
                m = m.unsqueeze(0)
                row_ids = torch.zeros(1, dtype=torch.long)
                col_ids = torch.zeros(1, dtype=torch.long)
            elif tensor.dim() == 2:
                tok, m, row_ids, col_ids = _tokenize_matrix_patches(
                    tensor,
                    patch_rows=tcfg.patch_rows,
                    patch_cols=tcfg.patch_cols,
                    tokensize=tcfg.tokensize,
                )
            else:
                raise NotImplementedError(f"Only 1D/2D tensors supported, got {pname} with shape {tensor.shape}")

            module_offsets[pname] = running
            module_token_counts[pname] = tok.shape[0]
            running += tok.shape[0]

            tokenized_entries.append((pname, tensor, mtype, tok, m, row_ids, col_ids))

        # second pass to build pos
        for pname, tensor, mtype, tok, m, row_ids, col_ids in tokenized_entries:
            n_tok = tok.shape[0]
            # within-block token indices
            pos0 = torch.arange(n_tok, dtype=torch.long) + module_offsets[pname]

            # group id: attention heads or mlp neuron blocks
            pos2 = torch.zeros(n_tok, dtype=torch.long)
            if mtype == "attn_q":
                # row block index -> row start -> head id
                row_start = row_ids * tcfg.patch_rows
                pos2 = (row_start // hyper.head_dim).clamp(min=0)
            elif mtype in {"attn_k", "attn_v"}:
                row_start = row_ids * tcfg.patch_rows
                pos2 = (row_start // hyper.head_dim).clamp(min=0)
            elif mtype == "attn_o":
                col_start = col_ids * tcfg.patch_cols
                pos2 = (col_start // hyper.head_dim).clamp(min=0)
            elif mtype in {"mlp_gate", "mlp_up"}:
                # neuron block id
                pos2 = row_ids
            elif mtype == "mlp_down":
                pos2 = col_ids

            pos1 = torch.full((n_tok,), block_id, dtype=torch.long)
            pos = torch.stack([pos0, pos1, pos2], dim=1)

            tokens_all.append(tok)
            masks_all.append(m)
            pos_all.append(pos)

        b_end = sum(t.shape[0] for t in tokens_all)
        meta["blocks"].append(
            {
                "block_id": block_id,
                "name": block_name,
                "start": b_start,
                "end": b_end,
                "module_offsets": module_offsets,
                "module_token_counts": module_token_counts,
            }
        )
        block_id += 1

    # --- optional embedding block ---
    if tcfg.include_embeddings and embed_key is not None:
        add_block("embed_tokens", [(embed_key, state_dict[embed_key], "embed")])

    # --- per-layer blocks ---
    for i in range(hyper.num_hidden_layers):
        def k_layer(suffix: str) -> str:
            return f"{prefix}layers.{i}.{suffix}"

        entries = [
            (k_layer("input_layernorm.weight"), state_dict[k_layer("input_layernorm.weight")], "norm"),
            (k_layer("self_attn.q_proj.weight"), state_dict[k_layer("self_attn.q_proj.weight")], "attn_q"),
            (k_layer("self_attn.k_proj.weight"), state_dict[k_layer("self_attn.k_proj.weight")], "attn_k"),
            (k_layer("self_attn.v_proj.weight"), state_dict[k_layer("self_attn.v_proj.weight")], "attn_v"),
            (k_layer("self_attn.o_proj.weight"), state_dict[k_layer("self_attn.o_proj.weight")], "attn_o"),
            (k_layer("self_attn.q_norm.weight"), state_dict.get(k_layer("self_attn.q_norm.weight"), torch.zeros(hyper.head_dim)), "norm"),
            (k_layer("self_attn.k_norm.weight"), state_dict.get(k_layer("self_attn.k_norm.weight"), torch.zeros(hyper.head_dim)), "norm"),
            (k_layer("post_attention_layernorm.weight"), state_dict[k_layer("post_attention_layernorm.weight")], "norm"),
            (k_layer("pre_feedforward_layernorm.weight"), state_dict[k_layer("pre_feedforward_layernorm.weight")], "norm"),
            (k_layer("mlp.gate_proj.weight"), state_dict[k_layer("mlp.gate_proj.weight")], "mlp_gate"),
            (k_layer("mlp.up_proj.weight"), state_dict[k_layer("mlp.up_proj.weight")], "mlp_up"),
            (k_layer("mlp.down_proj.weight"), state_dict[k_layer("mlp.down_proj.weight")], "mlp_down"),
            (k_layer("post_feedforward_layernorm.weight"), state_dict[k_layer("post_feedforward_layernorm.weight")], "norm"),
        ]
        add_block(f"layer_{i}", entries)

    # --- final norm ---
    if norm_key is not None:
        add_block("final_norm", [(norm_key, state_dict[norm_key], "norm")])

    # --- optional lm head ---
    if tcfg.include_lm_head and lm_head_key is not None:
        add_block("lm_head", [(lm_head_key, state_dict[lm_head_key], "lm_head")])

    tokens = torch.cat(tokens_all, dim=0) if tokens_all else torch.empty((0, tcfg.tokensize))
    masks = torch.cat(masks_all, dim=0) if masks_all else torch.empty((0, tcfg.tokensize), dtype=torch.bool)
    pos = torch.cat(pos_all, dim=0) if pos_all else torch.empty((0, 3), dtype=torch.long)

    # dataset info
    meta["max_positions"] = (pos.max(dim=0).values + 1).tolist() if pos.numel() else [0, 0, 0]
    meta["num_tokens"] = int(tokens.shape[0])

    return tokens, masks, pos, meta
