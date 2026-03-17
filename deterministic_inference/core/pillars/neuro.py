import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12): # 🚀 Removed hardcoded 24
        super(VAE, self).__init__()

        # ENCODER: Beefed to handle 27+ features
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

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # DECODER: Dynamically projects back to input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim) # 🚀 Now maps to 27
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class NeuroPillar:
    def __init__(self, model_path, input_dim=27): # 🚀 Pass in the detected 27
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_dim = input_dim

        # INITIALIZE
        self.model = VAE(input_dim=self.model_input_dim, latent_dim=12).to(self.device)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ NeuroPillar: Loaded champion weights for {input_dim} features.")
            except Exception as e:
                print(f"⚠️ Weights incompatible with {input_dim} features. Training recommended.")
        else:
            print("🛠️ Training Mode: No weights found. Ready for initial fit.")

    def predict(self, input_data: Any) -> float:
        """Calculates Reconstruction Loss (MSE) for the full feature vector."""
        x = torch.as_tensor(input_data, dtype=torch.float32).to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 🚀 NO MORE TRUNCATING: We use all features
        with torch.no_grad():
            reconstructed, _, _ = self.model(x)
            loss = torch.mean((x - reconstructed) ** 2)

        return float(loss.item())