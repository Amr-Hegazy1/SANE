import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add paths
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "src"))
sys.path.append(str(Path(__file__).parent))



def get_masked_mean_embedding(z, mask):
    """
    Computes mean embedding over sequence dimension, ignoring padding.
    z: [B, T, D], mask: [B, T, token_dim]
    """
    mask_float = mask[:, :, 0:1].float()
    sum_z = (z * mask_float).sum(dim=1)
    count_z = mask_float.sum(dim=1)
    count_z = torch.clamp(count_z, min=1.0)
    return sum_z / count_z


def parse_model_info(info_str):
    """
    Parses 'TinyLlama/TinyLlama-1.1B-Chat-v1.0_L10' into components.
    """
    # Remove Layer suffix
    parts = info_str.split('_L')
    model_name = parts[0]

    mn = model_name.lower()
    # Keep these aligned with families appearing in prepare_scaled_zoo.py
    if "tinyllama" in mn:
        family = "TinyLlama"
    elif "smollm" in mn:
        family = "SmolLM"
    elif "stablelm" in mn:
        family = "StableLM"
    elif "microsoft/phi" in mn or "/phi-" in mn or "phi-" in mn:
        family = "Phi"
    elif "qwen" in mn:
        family = "Qwen"
    elif "gemma" in mn:
        family = "Gemma"
    elif "mistral" in mn:
        family = "Mistral"
    elif "open_llama" in mn or "openlm-research/open_llama" in mn:
        family = "OpenLLaMA"
    elif "meta-llama/llama" in mn or "llama-" in mn or ("llama" in mn and "tinyllama" not in mn and "open_llama" not in mn):
        family = "Llama"
    elif mn.startswith("01-ai/yi") or "/yi-" in mn:
        family = "Yi"
    elif "deepseek" in mn:
        family = "DeepSeek"
    elif "falcon" in mn:
        family = "Falcon"
    elif "mpt" in mn:
        family = "MPT"
    elif "olmo" in mn:
        family = "OLMo"
    elif "pythia" in mn:
        family = "Pythia"
    elif "bloom" in mn:
        family = "BLOOM"
    else:
        family = "Other"

    return model_name, family


def load_latest_trial(experiment_dir: Path):
    trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("AE_trainable")]
    if not trial_dirs:
        raise FileNotFoundError(f"No training results found in {experiment_dir}")
    latest_trial = Path("/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch/LLM_AE_trainable_12b15_00000_0_2026-01-19_10-06-00/")
    chkpts = sorted(list(latest_trial.glob("checkpoint_*")))
    if not chkpts:
        raise FileNotFoundError(f"No checkpoints found in {latest_trial}")
    checkpoint_path = chkpts[-1].joinpath("state.pt")
    config_path = latest_trial.joinpath("params.json")
    return latest_trial, checkpoint_path, config_path


def _get_info_list(dataset):
    if hasattr(dataset, "data"):
        data = getattr(dataset, "data")
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return [d.get("info", "") for d in data]
    if hasattr(dataset, "refs"):
        refs = getattr(dataset, "refs")
        if isinstance(refs, list):
            return [r.get("info", "") for r in refs]
    return [""] * len(dataset)


def compute_embeddings(
    model,
    dataset,
    info_list=None,
    device="cuda",
    batch_size=8,
    limit=None,
    allow_mismatch=False,
):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    metadata = []

    with torch.no_grad():
        model_dtype = next(model.model.parameters()).dtype
        expected_dim = model.model.tokenizer.in_features
        warned_dim = False
        seen = 0
        for batch_idx, (w, mask, pos, props) in enumerate(loader):
            if limit is not None and seen >= limit:
                break
            # Align token dimension with model input dim
            if w.shape[-1] != expected_dim:
                if not allow_mismatch:
                    raise RuntimeError(
                        f"Token dim {w.shape[-1]} does not match model i_dim {expected_dim}. "
                        "Rebuild dataset or pass --allow_mismatch to pad/truncate."
                    )
                if not warned_dim:
                    print(
                        f"Warning: token_dim {w.shape[-1]} != model i_dim {expected_dim}. "
                        "Padding/truncating inputs for analysis."
                    )
                    warned_dim = True
                if w.shape[-1] > expected_dim:
                    w = w[..., :expected_dim]
                    mask = mask[..., :expected_dim]
                else:
                    pad = expected_dim - w.shape[-1]
                    w = torch.nn.functional.pad(w, (0, pad), value=0)
                    mask = torch.nn.functional.pad(mask, (0, pad), value=0)
            w = w.to(device=device, dtype=model_dtype)
            mask = mask.to(device)
            pos = pos.to(device)

            z = model.forward_encoder(w, pos)
            z_pooled = get_masked_mean_embedding(z, mask)
            embeddings.append(z_pooled.cpu().numpy())

            start_idx = batch_idx * batch_size
            for i in range(w.shape[0]):
                global_idx = start_idx + i
                if global_idx >= len(dataset):
                    break
                if info_list is None:
                    info_list = _get_info_list(dataset)
                info_str = info_list[global_idx] if global_idx < len(info_list) else ""
                model_name, family = parse_model_info(info_str)
                layer_id = int(props[i][0].item()) if props.ndim > 1 else int(props[i].item())
                hidden_size = None
                if props.shape[-1] > 1:
                    hidden_size = float(props[i][1].item())
                metadata.append(
                    {
                        "model": model_name,
                        "family": family,
                        "layer_id": layer_id,
                        "hidden_size": hidden_size,
                    }
                )

            seen += w.shape[0]

    X = np.concatenate(embeddings, axis=0)
    return X, metadata


def knn_architecture_accuracy(X, y, k=5, folds=5, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return float(np.mean(accs)), float(np.std(accs))


def linear_probe_r2(X, y, folds=5, seed=42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in kf.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y[train_idx])
        scores.append(reg.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores)), float(np.std(scores))


def _zscore(x):
    x = x.astype(np.float32)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0 or not np.isfinite(std):
        return x - mean
    return (x - mean) / std


def residualize_embeddings(X, covariates):
    """
    Regress out covariates from embeddings.
    covariates: np.ndarray [n_samples, n_covariates]
    """
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(covariates, X)
    X_hat = reg.predict(covariates)
    return X - X_hat


def encode_sequence_with_halo(model, tokens, positions, device, expected_dim, halo_size=0, chunk_size=4096):
    """
    Encode a long token sequence with optional haloing. Returns per-token embeddings.
    """
    model_dtype = next(model.model.parameters()).dtype
    n = tokens.shape[0]
    if tokens.shape[1] != expected_dim:
        raise RuntimeError(f"Token dim {tokens.shape[1]} != model i_dim {expected_dim}")

    embeddings = []
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            ctx_start = max(0, start - halo_size)
            ctx_end = min(n, end + halo_size)

            t_chunk = tokens[ctx_start:ctx_end]
            p_chunk = positions[ctx_start:ctx_end]

            t_chunk = t_chunk.to(device=device, dtype=model_dtype).unsqueeze(0)
            p_chunk = p_chunk.to(device=device).unsqueeze(0)

            z = model.forward_encoder(t_chunk, p_chunk).squeeze(0)

            # Strip halo region
            inner_start = start - ctx_start
            inner_end = inner_start + (end - start)
            embeddings.append(z[inner_start:inner_end].cpu())

    return torch.cat(embeddings, dim=0)


def compute_model_level_embeddings(
    model,
    dataset,
    info_list,
    device="cuda",
    expected_dim=0,
    halo_size=0,
    chunk_size=4096,
    allow_mismatch=False,
    max_positions=None,
):
    """
    Concatenate layer tokens into a model-level sequence and encode with haloing.
    Returns model-level embeddings and metadata per model.
    """
    # group indices by model name
    groups = {}
    for idx, info in enumerate(info_list):
        model_name, family = parse_model_info(info)
        groups.setdefault(model_name, {"family": family, "indices": []})
        groups[model_name]["indices"].append(idx)

    model_embeddings = []
    model_meta = []

    for model_name, g in groups.items():
        # collect layers, ordered by layer_id
        layer_items = []
        for idx in g["indices"]:
            w, mask, pos, props = dataset[idx]
            layer_id = int(props[0].item()) if props.ndim > 0 else 0
            # strip padding
            if mask.ndim == 2:
                valid = mask[:, 0].bool()
                w = w[valid]
            if w.shape[0] == 0:
                continue
            layer_items.append((layer_id, w))

        if not layer_items:
            continue

        layer_items.sort(key=lambda x: x[0])
        tokens = torch.cat([li[1] for li in layer_items], dim=0)

        if tokens.shape[1] != expected_dim:
            if not allow_mismatch:
                raise RuntimeError(
                    f"Token dim {tokens.shape[1]} does not match model i_dim {expected_dim}. "
                    "Rebuild dataset or pass --allow_mismatch to pad/truncate."
                )
            if tokens.shape[1] > expected_dim:
                tokens = tokens[:, :expected_dim]
            else:
                pad = expected_dim - tokens.shape[1]
                tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)

        # build positions [n,l,k]; n is global token index, k is index within layer
        positions = torch.zeros((tokens.shape[0], 3), dtype=torch.int)
        offset = 0
        for layer_id, w in layer_items:
            n_layer = w.shape[0]
            positions[offset : offset + n_layer, 0] = torch.arange(offset, offset + n_layer)
            positions[offset : offset + n_layer, 1] = layer_id
            positions[offset : offset + n_layer, 2] = torch.arange(0, n_layer)
            offset += n_layer

        # Clamp positions to embedding table sizes if provided
        if max_positions is not None and len(max_positions) == 3:
            for dim in range(3):
                if max_positions[dim] is not None:
                    max_val = int(max_positions[dim]) - 1
                    if max_val >= 0:
                        positions[:, dim] = positions[:, dim].clamp_max(max_val)

        # Encode with haloing if needed
        z = encode_sequence_with_halo(
            model=model,
            tokens=tokens,
            positions=positions,
            device=device,
            expected_dim=expected_dim,
            halo_size=halo_size,
            chunk_size=chunk_size,
        )
        z_mean = z.mean(dim=0).numpy()
        model_embeddings.append(z_mean)
        model_meta.append({"model": model_name, "family": g["family"]})

    X = np.stack(model_embeddings, axis=0)
    return X, model_meta


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM SANE embeddings (no t-SNE)")
    parser.add_argument("--experiment_dir", type=str, default="/storage/home/amr.hegazy/ray_results/sane_llm_scaled_multiarch")
    parser.add_argument("--dataset_path", type=str, default="data/scaled_llm_zoo/scaled_llm_dataset.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--holdout_only", action="store_true")
    parser.add_argument("--save_json", type=str, default="experiments/llm_extension/embedding_metrics.json")
    parser.add_argument("--allow_mismatch", action="store_true", help="Pad/truncate tokens if token_dim != model i_dim")
    parser.add_argument("--model_level", action="store_true", help="Compute model-level embeddings by concatenating layers")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for model-level encoding")
    parser.add_argument("--halo_size", type=int, default=0, help="Halo size for model-level encoding")
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        raise SystemExit(
            "scikit-learn is required for this analysis. Install it via `pip install scikit-learn`."
        )

    try:
        from SANE.models.def_AE_module import AEModule
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency for SANE imports: {e}. "
            "Ensure the SANE package and its deps (e.g., einops) are installed or PYTHONPATH includes src/."
        )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    experiment_dir = Path(args.experiment_dir)
    dataset_path = Path(args.dataset_path)

    latest_trial, checkpoint_path, config_path = load_latest_trial(experiment_dir)
    print(f"Loading SANE from: {latest_trial.name}")

    with open(config_path, "r") as f:
        config = json.load(f)
    config["device"] = args.device
    config["gpu_id"] = 0 if args.device == "cuda" else -1
    config["training::steps_per_epoch"] = 100

    model = AEModule(config)
    state = torch.load(checkpoint_path, map_location=args.device)
    model.model.load_state_dict(state["model"])
    model.eval()

    print(f"Loading dataset: {dataset_path}")
    dataset_dict = torch.load(dataset_path, weights_only=False)
    if args.holdout_only:
        datasets = [dataset_dict["testset"]]
    else:
        datasets = [dataset_dict["trainset"], dataset_dict["testset"]]

    expected_dim = model.model.tokenizer.in_features
    max_positions = config.get("ae:max_positions", None)

    if args.model_level:
        all_X = []
        all_meta = []
        for ds in datasets:
            info_list = _get_info_list(ds)
            X, meta = compute_model_level_embeddings(
                model=model,
                dataset=ds,
                info_list=info_list,
                device=args.device,
                expected_dim=expected_dim,
                halo_size=args.halo_size,
                chunk_size=args.chunk_size,
                allow_mismatch=args.allow_mismatch,
                max_positions=max_positions,
            )
            all_X.append(X)
            all_meta.extend(meta)
        X = np.concatenate(all_X, axis=0)
        meta = all_meta
    else:
        all_X = []
        all_meta = []
        for ds in datasets:
            info_list = _get_info_list(ds)
            X, meta = compute_embeddings(
                model=model,
                dataset=ds,
                info_list=info_list,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
                allow_mismatch=args.allow_mismatch,
            )
            all_X.append(X)
            all_meta.extend(meta)
        X = np.concatenate(all_X, axis=0)
        meta = all_meta

    families = np.array([m["family"] for m in meta])
    layer_ids = np.array([m.get("layer_id", np.nan) for m in meta], dtype=np.float32)
    hidden_sizes = np.array(
        [m.get("hidden_size", np.nan) for m in meta],
        dtype=np.float32,
    )

    results = {}

    # kNN accuracy by architecture family
    acc_mean, acc_std = knn_architecture_accuracy(X, families, k=5, folds=5)
    results["knn_architecture_acc_mean"] = acc_mean
    results["knn_architecture_acc_std"] = acc_std

    # Silhouette score (sample if too large)
    if X.shape[0] > 5000:
        idx = np.random.choice(X.shape[0], 5000, replace=False)
        sil = silhouette_score(X[idx], families[idx])
    else:
        sil = silhouette_score(X, families)
    results["silhouette_architecture"] = float(sil)

    # Linear probe for layer depth (only if layer_ids present)
    if np.isfinite(layer_ids).any():
        valid = np.isfinite(layer_ids)
        r2_mean, r2_std = linear_probe_r2(X[valid], layer_ids[valid], folds=5)
        results["layer_depth_r2_mean"] = r2_mean
        results["layer_depth_r2_std"] = r2_std

    # Linear probe for hidden size (if available)
    if np.isfinite(hidden_sizes).any():
        valid = np.isfinite(hidden_sizes)
        r2_h_mean, r2_h_std = linear_probe_r2(X[valid], hidden_sizes[valid], folds=5)
        results["hidden_size_r2_mean"] = r2_h_mean
        results["hidden_size_r2_std"] = r2_h_std

    # Size-regressed metrics (hidden size only)
    if np.isfinite(hidden_sizes).any():
        valid = np.isfinite(hidden_sizes)
        cov = _zscore(hidden_sizes[valid]).reshape(-1, 1)
        X_res = residualize_embeddings(X[valid], cov)
        acc_mean, acc_std = knn_architecture_accuracy(X_res, families[valid], k=5, folds=5)
        results["knn_architecture_acc_mean_regress_hidden"] = acc_mean
        results["knn_architecture_acc_std_regress_hidden"] = acc_std
        if X_res.shape[0] > 5000:
            idx = np.random.choice(X_res.shape[0], 5000, replace=False)
            sil = silhouette_score(X_res[idx], families[valid][idx])
        else:
            sil = silhouette_score(X_res, families[valid])
        results["silhouette_architecture_regress_hidden"] = float(sil)
        results["n_samples_regress_hidden"] = int(X_res.shape[0])

    # Depth-regressed metrics (layer depth only)
    if np.isfinite(layer_ids).any():
        valid = np.isfinite(layer_ids)
        cov = _zscore(layer_ids[valid]).reshape(-1, 1)
        X_res = residualize_embeddings(X[valid], cov)
        acc_mean, acc_std = knn_architecture_accuracy(X_res, families[valid], k=5, folds=5)
        results["knn_architecture_acc_mean_regress_depth"] = acc_mean
        results["knn_architecture_acc_std_regress_depth"] = acc_std
        if X_res.shape[0] > 5000:
            idx = np.random.choice(X_res.shape[0], 5000, replace=False)
            sil = silhouette_score(X_res[idx], families[valid][idx])
        else:
            sil = silhouette_score(X_res, families[valid])
        results["silhouette_architecture_regress_depth"] = float(sil)
        results["n_samples_regress_depth"] = int(X_res.shape[0])

    # Regress out both size + depth when available
    if np.isfinite(hidden_sizes).any() and np.isfinite(layer_ids).any():
        valid = np.isfinite(hidden_sizes) & np.isfinite(layer_ids)
        if np.any(valid):
            cov = np.stack([_zscore(hidden_sizes[valid]), _zscore(layer_ids[valid])], axis=1)
            X_res = residualize_embeddings(X[valid], cov)
            acc_mean, acc_std = knn_architecture_accuracy(X_res, families[valid], k=5, folds=5)
            results["knn_architecture_acc_mean_regress_hidden_depth"] = acc_mean
            results["knn_architecture_acc_std_regress_hidden_depth"] = acc_std
            if X_res.shape[0] > 5000:
                idx = np.random.choice(X_res.shape[0], 5000, replace=False)
                sil = silhouette_score(X_res[idx], families[valid][idx])
            else:
                sil = silhouette_score(X_res, families[valid])
            results["silhouette_architecture_regress_hidden_depth"] = float(sil)
            results["n_samples_regress_hidden_depth"] = int(X_res.shape[0])

    # Basic counts
    fam_counts = {f: int(np.sum(families == f)) for f in np.unique(families)}
    results["family_counts"] = fam_counts
    results["n_samples"] = int(X.shape[0])

    print("\n=== Embedding Metrics ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
