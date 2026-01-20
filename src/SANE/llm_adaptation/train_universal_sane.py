import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import argparse
from typing import List, Tuple

try:
    from src.SANE.llm_adaptation.universal_tokenizer import UniversalWeightTokenizer
    from src.SANE.llm_adaptation.dataset_universal import UniversalWeightDataset
    from src.SANE.llm_adaptation.universal_sane import UniversalSANE
except ImportError:
    from universal_tokenizer import UniversalWeightTokenizer
    from dataset_universal import UniversalWeightDataset
    from universal_sane import UniversalSANE


import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        sim_matrix = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -float("inf"))

        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ],
            dim=0,
        )

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def window_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], window_size: int = 1024
):
    batch_chunks_1 = []
    batch_pos_1 = []
    batch_chunks_2 = []
    batch_pos_2 = []

    for chunks, pos in batch:
        seq_len = chunks.shape[0]

        def get_window():
            if seq_len < window_size:
                pad_len = window_size - seq_len
                c = torch.cat([chunks, torch.zeros(pad_len, chunks.shape[1])], dim=0)
                p = torch.cat([pos, torch.zeros(pad_len, 2)], dim=0)
                return c, p
            else:
                start = torch.randint(0, seq_len - window_size + 1, (1,)).item()
                return chunks[start : start + window_size], pos[
                    start : start + window_size
                ]

        c1, p1 = get_window()
        c2, p2 = get_window()

        batch_chunks_1.append(c1)
        batch_pos_1.append(p1)
        batch_chunks_2.append(c2)
        batch_pos_2.append(p2)

    # Stack view 1 and view 2
    # Result: [2*B, window_size, dim]
    # First B are view 1, next B are view 2 (corresponding pairs)
    all_chunks = torch.cat(
        [torch.stack(batch_chunks_1), torch.stack(batch_chunks_2)], dim=0
    )
    all_pos = torch.cat([torch.stack(batch_pos_1), torch.stack(batch_pos_2)], dim=0)

    return all_chunks, all_pos


def train(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Comprehensive Model Zoo (Diverse Architectures & Sizes)
    model_list = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-v0.1",
        "Qwen/Qwen1.5-0.5B",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-7B",
        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "google/gemma-2b",
        "google/gemma-7b",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-3-1b-pt",
        "google/gemma-3-4b-pt",
        "microsoft/phi-1_5",
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/gpt-neo-125M",
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "allenai/OLMo-1B-hf",
        "state-spaces/mamba-130m-hf",
        "state-spaces/mamba-370m-hf",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-3b",
        "stabilityai/stablelm-2-zephyr-1_6b",
        "stabilityai/stablelm-3b-4e1t",
        "cerebras/Cerebras-GPT-111M",
        "cerebras/Cerebras-GPT-256M",
        "cerebras/btlm-3b-8k-base",
    ]

    model_list = sorted(list(set(model_list)))

    if accelerator.is_main_process:
        print(f"Zoo populated with {len(model_list)} models.")

    dataset = UniversalWeightDataset(
        model_paths=model_list,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        force_reprocess=False,
    )

    collate_fn = lambda b: window_collate_fn(b, window_size=args.window_size)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = UniversalSANE(
        input_dim=args.chunk_size,
        d_model=args.d_model,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    criterion = torch.nn.MSELoss()
    criterion_contra = NTXentLoss(temperature=0.1)

    print(f"Starting training on {accelerator.device}...")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)

        for batch_chunks, batch_pos in progress_bar:
            with accelerator.accumulate(model):
                recon, z = model(batch_chunks, batch_pos, return_projected=True)

                target = batch_chunks.to(recon.dtype)
                loss_recon = criterion(recon, target)

                z1, z2 = torch.chunk(z, 2, dim=0)
                loss_contra = criterion_contra(z1, z2)

                loss = loss_recon + (args.lambda_contrastive * loss_contra)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_description(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} (Rec: {loss_recon.item():.4f}, Con: {loss_contra.item():.4f})"
                )

                global_step += 1

        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

        if epoch % args.save_every == 0:
            out_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), out_path)
            accelerator.print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="./data/universal_cache")
    parser.add_argument("--output_dir", type=str, default="./output_sane")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--lambda_contrastive",
        type=float,
        default=0.1,
        help="Weight for contrastive loss",
    )

    args = parser.parse_args()
    train(args)
