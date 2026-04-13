from typing import Any

import torch # or tensorflow/keras depending on your model
import joblib

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
from typing import Any


import torch
import torch.nn as nn
from typing import Any


class VAE(nn.Module):
    """
    Updated Beefed VAE Architecture.
    Matches confirmed dimensions: 24 Input Features -> 12 Latent Dimensions.
    """

    def __init__(self, input_dim=24, latent_dim=12):
        super(VAE, self).__init__()

        # Encoder: 24 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Latent space: 32 -> 12 (mu and logvar)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder: 12 -> 32 -> 64 -> 128 -> 24
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            # Index 5: 64 -> 128
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # Index 8: 128 -> 24 (Matches Output Features)
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class NeuroPillar:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_dim = 24

        # 🚀 INITIALIZE WITH BEEFED DIMENSIONS: latent_dim=12
        self.model = VAE(input_dim=self.model_input_dim, latent_dim=12).to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✅ NeuroPillar: Beefed VAE weights loaded (Architecture: 128-64-32-12)")
        except RuntimeError as e:
            print(f"❌ Still a Mismatch: {e}")

    def predict(self, input_data: Any) -> float:
        """Calculates Reconstruction Loss (MSE) for the 24-feature vector."""
        # Convert to tensor and move to device
        x = torch.as_tensor(input_data, dtype=torch.float32).to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure strict 24-feature shape
        if x.shape[1] != self.model_input_dim:
            x = x[:, :self.model_input_dim]

        with torch.no_grad():
            reconstructed, _, _ = self.model(x)
            # MSE reconstruction loss - higher loss = more 'anomalous' (potential fraud)
            loss = torch.mean((x - reconstructed) ** 2)

        return float(loss.item())