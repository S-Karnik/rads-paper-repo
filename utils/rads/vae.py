from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "CLIPVAE",
]


class CLIPVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        seq_len: int = 77,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len, self.embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.bottleneck_norm = nn.LayerNorm(seq_len * embed_dim)

        self.fc_mu = nn.Linear(seq_len * embed_dim, latent_dim)
        self.fc_var = nn.Linear(seq_len * embed_dim, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, seq_len * embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x += self.pos_embed
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        x = self.bottleneck_norm(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-6, max=6)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        x = self.fc_decoder_input(z).view(batch_size, self.seq_len, self.embed_dim)
        x += self.pos_embed
        out = self.transformer_decoder(x)
        return out

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
