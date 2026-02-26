"""
Variational Autoencoder (VAE) — Generative AI component for Variant 6.

Compresses 228-dim Chef's Hat observations into a 32-dim latent space.
The encoder is used as a frozen feature extractor by the DQN agent.
The decoder's reconstruction error is used as a curiosity / intrinsic reward.

Observation layout (from official ChefsHatGYM source):
    obs[0:11]    board state (pizza area)
    obs[11:28]   current player hand
    obs[28:228]  valid action mask (200 values, 1 = valid, 0 = invalid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

OBS_DIM    = 228
LATENT_DIM = 32


class VAE(nn.Module):

    def __init__(self, input_dim=OBS_DIM, hidden_dim=128, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_fc1   = nn.Linear(input_dim,  hidden_dim)
        self.enc_fc2   = nn.Linear(hidden_dim, 64)
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim,  64)
        self.dec_fc2 = nn.Linear(64,          hidden_dim)
        self.dec_fc3 = nn.Linear(hidden_dim,  input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent(self, x):
        """Deterministic encode — used at inference time."""
        mu, _ = self.encode(x)
        return mu

    def reconstruction_error(self, x):
        """MSE between x and its reconstruction — used as curiosity signal."""
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return F.mse_loss(recon, x, reduction="none").mean(dim=-1)

    @staticmethod
    def loss(x_recon, x, mu, logvar, beta=1.0):
        recon = F.mse_loss(x_recon, x, reduction="sum")
        kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[VAE] Saved → {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[VAE] Loaded ← {path}")
