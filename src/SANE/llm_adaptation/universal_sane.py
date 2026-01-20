import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class ContinuousSinePE(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        scaled_pos = positions.unsqueeze(-1) * 1000.0

        args = scaled_pos * self.freqs.unsqueeze(0).unsqueeze(0)

        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            pe = torch.cat([pe, torch.zeros_like(pe[:, :, :1])], dim=-1)

        return pe


class UniversalSANE(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pe_layer = ContinuousSinePE(d_model)
        self.pe_intra = ContinuousSinePE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.output_proj = nn.Linear(d_model, input_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128)
        )

    def forward(
        self,
        chunks: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_projected: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = chunks.to(self.input_proj.weight.dtype)
        positions = positions.to(self.input_proj.weight.dtype)

        x = self.input_proj(chunks)

        pe_l = self.pe_layer(positions[:, :, 0])
        pe_i = self.pe_intra(positions[:, :, 1])
        x = x + pe_l + pe_i

        latent = self.encoder(x, src_key_padding_mask=padding_mask)

        decoded = self.decoder(
            latent + pe_l + pe_i,
            latent,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )

        recon = self.output_proj(decoded)

        if return_projected:
            latent_mean = latent.mean(dim=1)
            projected = self.projection_head(latent_mean)
            return recon, projected

        return recon, latent


if __name__ == "__main__":
    model = UniversalSANE(
        input_dim=64, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2
    )

    batch_size = 2
    seq_len = 100

    dummy_chunks = torch.randn(batch_size, seq_len, 64)
    dummy_pos = torch.rand(batch_size, seq_len, 2)

    recon, latent = model(dummy_chunks, dummy_pos)

    print(f"Model Config: UniversalSANE")
    print(f"Input: {dummy_chunks.shape}")
    print(f"Recon: {recon.shape}")
    print(f"Latent: {latent.shape}")
