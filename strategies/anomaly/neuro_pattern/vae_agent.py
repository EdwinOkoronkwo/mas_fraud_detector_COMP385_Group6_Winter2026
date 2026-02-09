import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset
from autogen_agentchat.agents import AssistantAgent
from utils.logger import setup_logger

logger = setup_logger("VAE_Tool")


# --- PyTorch VAE Architecture ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        # Encoder: Compresses input into distribution parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(12, latent_dim)
        self.fc_logvar = nn.Linear(12, latent_dim)

        # Decoder: Reconstructs from latent sample
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim)
        )

    def reparameterize(self, mu, logvar):
        """The Reparameterization Trick: samples from N(mu, var) while allowing backprop."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """ELBO Loss: Reconstruction Error + KL Divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# --- Main Tool Function ---
def train_vae_sim(db_path: str) -> str:
    try:
        # Check for hardware acceleration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Data Acquisition
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X_raw = df.drop(columns=['is_fraud'], errors='ignore').select_dtypes(include=[np.number]).values

        X_tensor = torch.FloatTensor(X_raw).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 2. Model Initialization
        input_dim = X_raw.shape[1]
        model = VAE(input_dim=input_dim, latent_dim=5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 3. Training Loop
        logger.info(f"Starting PyTorch VAE Training on {device}...")
        model.train()
        epochs = 100
        for epoch in range(epochs):
            for batch in dataloader:
                data = batch[0]
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                optimizer.step()

        # 4. Inference & Anomaly Calculation
        model.eval()
        with torch.no_grad():
            reconstructed, _, _ = model(X_tensor)
            # Calculate MSE on CPU for numpy processing
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        threshold = np.percentile(mse, 97)
        anomaly_count = int(np.sum(mse > threshold))

        # 5. Persistence & Visualization
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_vae.pth")
        torch.save(model.state_dict(), model_path)

        os.makedirs("reports", exist_ok=True)
        plot_path = "reports/vae_reconstruction.png"

        plt.figure(figsize=(10, 6))
        plt.hist(mse, bins=50, color='darkorchid', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        plt.title(f"PyTorch VAE: Anomaly Distribution\nDetected {anomaly_count} Potential Fraud Cases")
        plt.xlabel("Reconstruction Error (MSE)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "PyTorch_Variational_Autoencoder",
            "saved_model_path": model_path,
            "anomaly_count": anomaly_count,
            "metrics": {
                "avg_mse": round(float(np.mean(mse)), 6),
                "threshold": round(float(threshold), 6)
            },
            "plot_url": plot_path,
            "status": "SUCCESS"
        })

    except Exception as e:
        logger.error(f"VAE Error: {str(e)}")
        return json.dumps({"status": "ERROR", "message": str(e)})


class VAEAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="VAE_Agent",
            model_client=model_client,
            tools=[train_vae_sim],
            system_message="""You are the Variational Inference Expert.
            You use probabilistic reconstruction (VAEs) to identify fraud.
            You specialize in spotting transactions that deviate from the normal latent distribution.
            You MUST return a JSON containing 'model', 'anomaly_count', and 'plot_url'."""
        )