import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import argparse
from pathlib import Path
import json
import torch

from SANE.llm import (
    load_hf_gemma3,
    detect_gemma3_prefix,
    canonicalize_gemma3_reference_free,
    tokenize_gemma3_patches,
)
from SANE.llm.gemma3 import Gemma3TokenizationConfig
from SANE.llm.dataset_hf_llm import HFLlmTokenFileDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/gemma-3-270m-it")
    ap.add_argument("--out", type=str, default="../../data/dataset_llm_gemma3")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float16","bfloat16"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no_canonicalize", action="store_true")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--include_lm_head", action="store_true")
    ap.add_argument("--tokensize", type=int, default=4096)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--train_samples_per_file", type=int, default=256)
    ap.add_argument("--test_samples_per_file", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} (device={args.device}, dtype={args.dtype})")
    sd, cfg = load_hf_gemma3(args.model, device=args.device, dtype=args.dtype, token=args.hf_token)
    prefix = detect_gemma3_prefix(sd)
    print(f"Detected Gemma3 prefix: {prefix!r}")

    if not args.no_canonicalize:
        print("Applying reference-free canonicalization (heads + MLP neurons)")
        sd = canonicalize_gemma3_reference_free(sd, cfg, prefix=prefix)

    tcfg = Gemma3TokenizationConfig(
        tokensize=args.tokensize,
        patch_rows=args.patch,
        patch_cols=args.patch,
        include_embeddings=args.include_embeddings,
        include_lm_head=args.include_lm_head,
    )

    print("Tokenizing weights into patch tokens")
    w, m, p, meta = tokenize_gemma3_patches(sd, cfg, prefix=prefix, tcfg=tcfg)

    # save sample files
    sample = {
        "w": w.cpu(),
        "m": m.cpu(),
        "p": p.cpu(),
        "props": torch.tensor([]),
        "meta": meta,
        "model_id": args.model,
    }
    torch.save(sample, out_dir / "train" / "sample_000.pt")
    torch.save(sample, out_dir / "test" / "sample_000.pt")

    # dataset object
    trainset = HFLlmTokenFileDataset(out_dir, split="train", samples_per_file=args.train_samples_per_file)
    testset = HFLlmTokenFileDataset(out_dir, split="test", samples_per_file=args.test_samples_per_file)

    dataset = {"trainset": trainset, "testset": testset}
    torch.save(dataset, out_dir / "dataset.pt")

    # dataset info (handy for config)
    max_positions = (p.max(dim=0).values + 1).tolist()
    info = {
        "model": args.model,
        "tokensize": int(w.shape[1]),
        "n_tokens": int(w.shape[0]),
        "max_positions": max_positions,
        "tokenization": {
            "patch": args.patch,
            "include_embeddings": bool(args.include_embeddings),
            "include_lm_head": bool(args.include_lm_head),
            "canonicalized": (not args.no_canonicalize),
        },
    }
    (out_dir / "dataset_info_test.json").write_text(json.dumps(info, indent=2))
    (out_dir / "dataset_info_train.json").write_text(json.dumps(info, indent=2))

    print("Done.")
    print("Suggested AE config:")
    print(f"  ae:i_dim = {w.shape[1]}")
    print(f"  ae:max_positions = {max_positions}")


if __name__ == "__main__":
    main()
