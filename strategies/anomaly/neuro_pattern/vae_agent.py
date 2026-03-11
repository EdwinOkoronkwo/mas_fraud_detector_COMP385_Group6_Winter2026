import json
import os

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset
from autogen_agentchat.agents import AssistantAgent

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("VAE_Tool")

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# --- 1. PYTORCH VAE ARCHITECTURE ---

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset


# --- 1. THE STABLE VAE ARCHITECTURE ---

class VAE(nn.Module):
    def __init__(self, input_dim=24, latent_dim=8):
        super(VAE, self).__init__()

        # 🚀 WIDER & DEEPER ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),      # Increased from 64
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),              # LeakyReLU helps prevent dead neurons
            nn.Linear(128, 64),             # Increased from 32
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # 🚀 WIDER & DEEPER DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim)        # Back to 24 features
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


def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Weighted total loss
    return recon_loss + (beta * kl_loss)


# --- 2. THE AGENT TOOL ---

def train_vae_sim(data_path: str = "data/temp_split.joblib", latent_dim: int = 8, threshold_p: float = 98.5) -> str:
    """Beefed-up VAE Training: CPU-optimized with Early Stopping and expanded layers."""
    try:
        # Laptop safety: Force CPU
        device = torch.device("cpu")

        if not os.path.exists(data_path):
            return json.dumps({"status": "ERROR", "message": f"Bundle not found at {data_path}"})

        data_bundle = joblib.load(data_path)
        X_train_raw, y_train = data_bundle['train']
        X_val_raw, y_val = data_bundle['val']

        X_train = X_train_raw.values if hasattr(X_train_raw, 'values') else X_train_raw
        X_val = X_val_raw.values if hasattr(X_val_raw, 'values') else X_val_raw

        X_tensor_train = torch.FloatTensor(X_train.astype(np.float32)).to(device)
        X_tensor_val = torch.FloatTensor(X_val.astype(np.float32)).to(device)

        # 🚀 CPU FIX: Batch size 64 is much better for laptop thermal/cache performance
        dataloader = DataLoader(TensorDataset(X_tensor_train), batch_size=64, shuffle=True)

        input_dim = X_train.shape[1]
        model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # --- EARLY STOPPING LOGIC ---
        best_val_loss = float('inf')
        patience = 5
        trigger_times = 0
        best_model_state = None

        # --- TRAINING LOOP ---
        model.train()
        for epoch in range(50):
            epoch_loss = 0
            for batch in dataloader:
                data = batch[0]
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss = vae_loss_function(recon, data, mu, logvar)

                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            # --- VALIDATION CHECK (Per Epoch) ---
            model.eval()
            with torch.no_grad():
                val_recon, v_mu, v_logvar = model(X_tensor_val)
                val_loss = vae_loss_function(val_recon, X_tensor_val, v_mu, v_logvar).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                best_model_state = model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"🛑 Early Stopping at epoch {epoch}")
                    break

        # Load the best performing version before saving
        if best_model_state:
            model.load_state_dict(best_model_state)

        # --- EVALUATION ON VALIDATION SET ---
        model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = model(X_tensor_val)
            mse = torch.mean((X_tensor_val - reconstructed) ** 2, dim=1).cpu().numpy()

        threshold = np.percentile(mse, threshold_p)
        y_pred = (mse > threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)

        # Save artifact
        model_save_path = "models/champion_vae.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_save_path)

        return json.dumps({
            "model": "variational_ae",
            "status": "SUCCESS",
            "metrics": {
                "f1": float(round(f1, 4)),
                "recall": float(round(recall, 4)),
                "precision": float(round(precision, 4)),
                "avg_mse": float(round(np.mean(mse), 6)),
                "threshold": float(round(threshold, 6))
            },
            "path": model_save_path
        })

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class VAEAgent:
    def __init__(self, model_client, settings):
        self.settings = settings

        # 🚀 FIX: Force alignment with the DataSpecialist output
        bundle_path = "data/temp_split.joblib"

        self.agent = AssistantAgent(
            name="VAE_Agent",
            model_client=model_client,
            tools=[train_vae_sim],
            system_message=f"""
                    You are the VAE Anomaly Specialist. 
                    Your goal is to maximize Recall while keeping Precision > 0.05.

                    HYPERPARAMETER GUIDANCE:
                    - 'latent_dim': Try 8 or 10. A bottleneck that is too tight (5) loses all signal.
                    - 'threshold_p': Use 99.4 to 99.8. This aligns with the 0.6% fraud prevalence. 
                      Higher percentile = fewer False Positives.
                    - 'beta': If reconstruction error (MSE) is too high, lower beta to 0.01.

                    STRATEGY:
                    1. Tight Filter: (latent=8, threshold_p=99.6).
                    2. High Sensitivity: (latent=12, threshold_p=99.0).

                    Retrain and compare. We need the VAE to detect the 'Neuro-Anomalies' 
                    the XGBoost (Gold) model might miss.
                    """
        )



# --- PyTorch VAE Architecture ---
# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim=5):
#         super(VAE, self).__init__()
#         # Encoder: Compresses input into distribution parameters
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 20),
#             nn.ReLU(),
#             nn.Linear(20, 12),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(12, latent_dim)
#         self.fc_logvar = nn.Linear(12, latent_dim)
#
#         # Decoder: Reconstructs from latent sample
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 12),
#             nn.ReLU(),
#             nn.Linear(12, 20),
#             nn.ReLU(),
#             nn.Linear(20, input_dim)
#         )
#
#     def reparameterize(self, mu, logvar):
#         """The Reparameterization Trick: samples from N(mu, var) while allowing backprop."""
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         h = self.encoder(x)
#         mu, logvar = self.fc_mu(h), self.fc_logvar(h)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar
#
#
# def vae_loss_function(recon_x, x, mu, logvar):
#     """ELBO Loss: Reconstruction Error + KL Divergence."""
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl_loss
#
#
# # --- Main Tool Function ---
# def train_vae_sim(db_path: str) -> str:
#     try:
#         # Check for hardware acceleration
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 1. Data Acquisition
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X_raw = df.drop(columns=['is_fraud'], errors='ignore').select_dtypes(include=[np.number]).values
#
#         X_tensor = torch.FloatTensor(X_raw).to(device)
#         dataset = TensorDataset(X_tensor)
#         dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#
#         # 2. Model Initialization
#         input_dim = X_raw.shape[1]
#         model = VAE(input_dim=input_dim, latent_dim=5).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#         # 3. Training Loop
#         logger.info(f"Starting PyTorch VAE Training on {device}...")
#         model.train()
#         epochs = 100
#         for epoch in range(epochs):
#             for batch in dataloader:
#                 data = batch[0]
#                 optimizer.zero_grad()
#                 recon_batch, mu, logvar = model(data)
#                 loss = vae_loss_function(recon_batch, data, mu, logvar)
#                 loss.backward()
#                 optimizer.step()
#
#         # 4. Inference & Anomaly Calculation
#         model.eval()
#         with torch.no_grad():
#             reconstructed, _, _ = model(X_tensor)
#             # Calculate MSE on CPU for numpy processing
#             mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
#
#         threshold = np.percentile(mse, 97)
#         anomaly_count = int(np.sum(mse > threshold))
#
#         # 5. Persistence & Visualization
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_vae.pth")
#         torch.save(model.state_dict(), model_path)
#
#         os.makedirs("reports", exist_ok=True)
#         plot_path = "reports/vae_reconstruction.png"
#
#         plt.figure(figsize=(10, 6))
#         plt.hist(mse, bins=50, color='darkorchid', alpha=0.7)
#         plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
#         plt.title(f"PyTorch VAE: Anomaly Distribution\nDetected {anomaly_count} Potential Fraud Cases")
#         plt.xlabel("Reconstruction Error (MSE)")
#         plt.ylabel("Frequency")
#         plt.legend()
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "PyTorch_Variational_Autoencoder",
#             "saved_model_path": model_path,
#             "anomaly_count": anomaly_count,
#             "metrics": {
#                 "avg_mse": round(float(np.mean(mse)), 6),
#                 "threshold": round(float(threshold), 6)
#             },
#             "plot_url": plot_path,
#             "status": "SUCCESS"
#         })
#
#     except Exception as e:
#         logger.error(f"VAE Error: {str(e)}")
#         return json.dumps({"status": "ERROR", "message": str(e)})
#
#
# class VAEAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="VAE_Agent",
#             model_client=model_client,
#             tools=[train_vae_sim],
#             system_message="""You are the Variational Inference Expert.
#             You use probabilistic reconstruction (VAEs) to identify fraud.
#             You specialize in spotting transactions that deviate from the normal latent distribution.
#             You MUST return a JSON containing 'model', 'anomaly_count', and 'plot_url'."""
#       )