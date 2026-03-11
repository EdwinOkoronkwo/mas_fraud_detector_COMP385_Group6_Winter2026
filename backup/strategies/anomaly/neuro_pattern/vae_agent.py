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

from mas_fraud_detector.config.settings import TEMP_SPLIT_PATH
from mas_fraud_detector.utils.logger import setup_logger

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

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 12),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(12, latent_dim)
        self.fc_logvar = nn.Linear(12, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim)
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

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def validate_neuro_champion(mse_scores, labels, threshold_percentile=99.0):
    """
    Breaks the '1,297' pattern by calculating actual performance
    against the ground truth labels.
    """
    threshold = np.percentile(mse_scores, threshold_percentile)
    # Predict 1 for Anomaly, 0 for Normal
    y_pred = (mse_scores > threshold).astype(int)
    y_true = labels.values

    # Calculate Metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

# --- 2. THE FLEXIBLE VAE TOOL ---

def train_vae_sim(db_path: str, latent_dim: int = 5, threshold_p: int = 97) -> str:
    """Agent-accessible tool for VAE training with Ground-Truth Validation."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)

        # Keep labels for validation
        y_true = df['is_fraud'].values
        X_raw = df.drop(columns=['is_fraud'], errors='ignore').select_dtypes(include=[np.number]).values

        X_tensor = torch.FloatTensor(X_raw).to(device)
        dataloader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)

        input_dim = X_raw.shape[1]
        model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Training
        model.train()
        for epoch in range(50):
            for batch in dataloader:
                data = batch[0]
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                loss = vae_loss_function(recon, data, mu, logvar)
                loss.backward()
                optimizer.step()

        # Scoring
        model.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            final_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_tensor.shape[0]

        # Consolidated Threshold & Pattern Breaking
        threshold = np.percentile(mse, threshold_p)
        y_pred = (mse > threshold).astype(int)
        anomaly_count = int(np.sum(y_pred))

        # Precision/Recall Logic
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0.0

        # Save and Plot (as per your original logic)
        model_save_path = "models/champion_vae.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_save_path)

        return json.dumps({
            "model": "variational_ae",
            "status": "SUCCESS",
            "anomaly_count": anomaly_count,
            "metrics": {
                "avg_mse": round(float(np.mean(mse)), 6),
                "kl_divergence": round(float(final_kl), 6),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "true_positives": int(tp)
            },
            "saved_model_path": model_save_path
        })
    except Exception as e:
        return json.dumps({"model": "variational_ae", "status": "ERROR", "error_message": str(e)})

# def train_vae_sim(db_path: str, latent_dim: int = 5, threshold_p: int = 97) -> str:
#     """Agent-accessible tool for VAE training with standardized JSON output."""
#     try:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X_raw = df.drop(columns=['is_fraud'], errors='ignore').select_dtypes(include=[np.number]).values
#
#         X_tensor = torch.FloatTensor(X_raw).to(device)
#         dataloader = DataLoader(TensorDataset(X_tensor), batch_size=64, shuffle=True)
#
#         input_dim = X_raw.shape[1]
#         model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#         # Training Loop
#         model.train()
#         for epoch in range(50):
#             for batch in dataloader:
#                 data = batch[0]
#                 optimizer.zero_grad()
#                 recon, mu, logvar = model(data)
#                 loss = vae_loss_function(recon, data, mu, logvar)
#                 loss.backward()
#                 optimizer.step()
#
#         # Inference & Scoring
#         model.eval()
#         with torch.no_grad():
#             reconstructed, mu, logvar = model(X_tensor)
#             # Individual MSE for anomaly scoring
#             mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
#             # Final KL Divergence for the report
#             final_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_tensor.shape[0]
#
#         # 2. Get Ground Truth from the same DataFrame
#         y_true = df['is_fraud'].values
#
#         # 3. BREAK THE PATTERN: Calculate actual Fraud detection
#         threshold = np.percentile(mse, threshold_percentile)
#         y_pred = (mse > threshold).astype(int)
#
#         # 4. Metrics calculation
#         tp = np.sum((y_pred == 1) & (y_true == 1))
#         fp = np.sum((y_pred == 1) & (y_true == 0))
#         precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
#         recall = float(tp / np.sum(y_true == 1))
#
#         threshold = np.percentile(mse, threshold_p)
#         anomaly_count = int(np.sum(mse > threshold))
#
#         # Save model for the Registry
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_save_path = os.path.join(model_dir, "champion_vae.pth")
#         torch.save(model.state_dict(), model_save_path)
#
#         # Plotting
#         plot_path = "reports/vae_reconstruction.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.figure(figsize=(10, 6))
#         plt.hist(mse, bins=50, color='darkorchid', alpha=0.7)
#         plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
#         plt.title(f"VAE Anomaly Distribution (Count: {anomaly_count})")
#         plt.legend()
#         plt.savefig(plot_path);
#         plt.close()
#
#         # Return structured JSON for the Selector/Critic
#         return json.dumps({
#             "model": "variational_ae",
#             "status": "SUCCESS",
#             "anomaly_count": anomaly_count,
#             "metrics": {
#                 "avg_mse": round(float(np.mean(mse)), 6),
#                 "kl_divergence": round(float(final_kl), 6),
#                 "precision": round(precision, 4),
#                 "recall": round(recall, 4),
#                 "true_positives": int(tp)  # This is the "S
#             },
#             "plot_url": plot_path,
#             "saved_model_path": model_save_path
#         })
#
#     except Exception as e:
#         # Standardized error return
#         return json.dumps({
#             "model": "variational_ae",
#             "status": "ERROR",
#             "error_message": str(e)
#         })

# --- 3. THE VAE AGENT ---

import functools


class VAEAgent:
    def __init__(self, model_client):
        # Signature is now just (self, model_client)
        self.agent = AssistantAgent(
            name="VAE_Agent",
            model_client=model_client,
            tools=[train_vae_sim], # Tool signature is now clean: (latent_dim, threshold_p)
            system_message="""You are the Variational Inference Expert.
            Propose 'latent_dim' and 'threshold_p' to identify fraud.
            You MUST return a JSON containing 'model', 'anomaly_count', and 'plot_url'."""
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