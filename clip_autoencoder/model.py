import torch
import torch.nn as nn

# --- VAE ---
class CLIPVAE(nn.Module):
    def __init__(self, latent_dim=64, seq_len=77, embed_dim=768, n_layers=4, n_heads=8, dropout=0.1, decoder_final_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=2048, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Flattened bottleneck norm
        self.bottleneck_norm = nn.LayerNorm(seq_len * embed_dim)
        
        self.fc_mu = nn.Linear(seq_len * embed_dim, latent_dim)
        self.fc_var = nn.Linear(seq_len * embed_dim, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, seq_len * embed_dim)
        
        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=2048, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        # Note: We pass `norm=None` here and apply it manually to ensure clarity, 
        # though PyTorch's TransformerEncoder can take a norm arg.
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        # --- FIX: Add Final Layer Norm ---
        # Essential because Pre-Norm transformers have unbounded variance growth
        self.decoder_final_norm = nn.LayerNorm(embed_dim) if decoder_final_norm else None

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-6, max=6)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        batch_size = x.shape[0]
        x = x + self.pos_embed
        x_encoded = self.transformer_encoder(x)
        
        x_flat = x_encoded.reshape(batch_size, -1)
        x_flat = self.bottleneck_norm(x_flat)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_var(x_flat)
        return mu, logvar

    def decode(self, z):
        batch_size = z.shape[0]
        x_decode = self.fc_decoder_input(z).view(batch_size, self.seq_len, self.embed_dim)
        x_decode = x_decode + self.pos_embed
        
        out = self.transformer_decoder(x_decode)
        
        # --- FIX: Apply Final Norm ---
        out = self.decoder_final_norm(out) if self.decoder_final_norm is not None else out
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
