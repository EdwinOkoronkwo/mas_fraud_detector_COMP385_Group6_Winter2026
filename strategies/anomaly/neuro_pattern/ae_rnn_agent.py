from autogen_agentchat.agents import AssistantAgent

import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, recall_score, precision_score
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.neural_network import MLPRegressor

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# Global Database Path
TEMP_SPLIT_PATH = "mas_fraud_detector/data/database.sqlite"


def train_rnn_ae_flexible(data_path: str = "data/temp_split.joblib",
                          hidden_dim: int = 64,
                          latent_dim: int = 16,
                          threshold_percentile: float = 99.4) -> str:
    """Consolidated Neuro-Anomaly Tool: Uses LSTM-Autoencoder logic."""
    try:
        # 🚀 Fix 1: Device Agnostic (Use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🛡️ Force correct pathing
        if not os.path.exists(data_path):
            data_path = "data/temp_split.joblib"

        data_bundle = joblib.load(data_path)
        X_train_raw, _ = data_bundle['train']
        X_val_raw, y_val = data_bundle['val']

        X_train = X_train_raw.values if hasattr(X_train_raw, 'values') else X_train_raw
        X_val = X_val_raw.values if hasattr(X_val_raw, 'values') else X_val_raw

        # Move tensors to device
        X_train_t = torch.tensor(X_train.astype(np.float32)).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val.astype(np.float32)).unsqueeze(1).to(device)

        train_loader = DataLoader(TensorDataset(X_train_t), batch_size=1024, shuffle=True)
        input_dim = X_train.shape[1]

        class Residual_LSTM_AE(nn.Module):
            def __init__(self, in_dim, h_dim, l_dim):
                super().__init__()
                self.enc = nn.LSTM(in_dim, h_dim, batch_first=True)
                self.latent_proj = nn.Linear(h_dim, l_dim)
                self.dec_proj = nn.Linear(l_dim, h_dim)
                self.dec = nn.LSTM(h_dim, in_dim, batch_first=True)

            def forward(self, x):
                _, (h, _) = self.enc(x)
                lat = self.latent_proj(h.permute(1, 0, 2))
                out = self.dec_proj(lat)
                out, _ = self.dec(out)
                return out

        model = Residual_LSTM_AE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(30):
            for batch in train_loader:
                data = batch[0]
                optimizer.zero_grad()
                loss = criterion(model(data), data)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            recons = model(X_val_t)
            # Calculate MSE on device, then move to CPU for numpy
            mse = torch.mean((X_val_t - recons) ** 2, dim=(1, 2)).cpu().numpy()

        threshold = np.percentile(mse, threshold_percentile)
        y_pred = (mse > threshold).astype(int)

        # 🚀 Fix 2: Consistent Champion Naming
        save_path = "models/champion_rnn_ae.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), save_path)

        return json.dumps({
            "model": "rnn_autoencoder",
            "status": "SUCCESS",
            "metrics": {
                "f1": round(float(f1_score(y_val, y_pred)), 4),
                "recall": round(float(recall_score(y_val, y_pred)), 4),
                "precision": round(float(precision_score(y_val, y_pred)), 4),
                "threshold_value": round(float(threshold), 6)
            },
            "path": save_path
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

# --- 2. THE STANDARD AE TOOL (FIXED PATH) ---


# --- 3. THE AGENTS ---

class RNNAgent:
    def __init__(self, model_client, settings):
        self.settings = settings

        # 🚀 FIX: Ensure the agent knows to look for the Specialist Bundle
        # We define the bundle path based on your standard project structure
        bundle_path = "data/temp_split.joblib"

        self.agent = AssistantAgent(
            name="RNN_Agent",
            model_client=model_client,
            tools=[train_rnn_ae_flexible],
            system_message=f"""
                    You are the Temporal Pattern Specialist. 
                    XGBoost (Gold) is currently at F1=0.49. To beat it, you must be precise.

                    ### STRATEGY:
                    - THRESHOLD: Do not use 99.0. Use 99.4 or 99.6. 
                      (This matches the 0.6% actual fraud prevalence).
                    - DIMENSIONS: hidden_dim=64, latent_dim=16. 
                    - If F1 is low, increase the threshold_percentile to reduce False Positives.
                    """
        )
# def train_rnn_ae(db_path: str) -> str:
#     """
#     Residual LSTM-based Autoencoder with Early Stopping, Huber Loss,
#     and Dimension-Corrected residual connections.
#     """
#     try:
#         import torch
#         import torch.nn as nn
#         from torch.utils.data import DataLoader, TensorDataset, random_split
#         import os
#         import pandas as pd
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from sqlalchemy import create_engine
#         import json
#         import logging
#
#         logger = logging.getLogger(__name__)
#
#         # --- DATA PREPARATION ---
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float32)
#
#         # Reshape for LSTM: [Batch, TimeSteps(1), Features]
#         X_tensor = torch.tensor(X).unsqueeze(1)
#         input_dim = X.shape[1]
#
#         # Split for Early Stopping (80% Train, 20% Validation)
#         train_size = int(0.8 * len(X_tensor))
#         val_size = len(X_tensor) - train_size
#         train_ds, val_ds = random_split(TensorDataset(X_tensor), [train_size, val_size])
#
#         train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=1024)
#
#         # --- MODEL DEFINITION ---
#         class Residual_LSTM_AE(nn.Module):
#             def __init__(self, in_dim, hidden_dim=32, latent_dim=16):
#                 super().__init__()
#                 # Encoder: Compresses signal
#                 self.enc1 = nn.LSTM(in_dim, hidden_dim, batch_first=True)
#                 self.enc2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
#
#                 # Decoder: Reconstructs signal
#                 self.dec1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
#                 self.dec2 = nn.LSTM(hidden_dim, in_dim, batch_first=True)
#
#                 self.dropout = nn.Dropout(0.1)
#
#             def forward(self, x):
#                 # Encoding
#                 out1, _ = self.enc1(x)
#                 out1 = self.dropout(out1)
#                 _, (h2, _) = self.enc2(out1)
#
#                 # Bottleneck permutation [1, B, L] -> [B, 1, L]
#                 latent = h2.permute(1, 0, 2)
#
#                 # Decoding
#                 out3, _ = self.dec1(latent)
#                 out3 = self.dropout(out3)
#                 out4, _ = self.dec2(out3)
#
#                 # Residual Connection: Focuses learning on the reconstruction error
#                 return out4 + x
#
#         # --- TRAINING SETUP ---
#         model = Residual_LSTM_AE(input_dim)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
#         criterion = nn.SmoothL1Loss()  # Huber Loss: Robust to extreme transaction amounts
#
#         # Early Stopping State
#         patience = 5
#         best_val_loss = float('inf')
#         early_stop_counter = 0
#         best_model_state = None
#
#         # --- TRAINING LOOP ---
#         epochs = 50
#         for epoch in range(epochs):
#             model.train()
#             total_train_loss = 0
#             for batch in train_loader:
#                 inputs = batch[0]
#                 optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = criterion(output, inputs)
#                 loss.backward()
#                 optimizer.step()
#                 total_train_loss += loss.item()
#
#             # Validation Phase
#             model.eval()
#             total_val_loss = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     inputs = batch[0]
#                     val_out = model(inputs)
#                     total_val_loss += criterion(val_out, inputs).item()
#
#             avg_val_loss = total_val_loss / len(val_loader)
#
#             # Early Stopping Check
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 best_model_state = model.state_dict()
#                 early_stop_counter = 0
#             else:
#                 early_stop_counter += 1
#
#             if early_stop_counter >= patience:
#                 logger.info(f"Early stopping at epoch {epoch}")
#                 break
#
#         # Restore Best Weights
#         if best_model_state:
#             model.load_state_dict(best_model_state)
#
#         # --- PERSISTENCE ---
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_rnn_ae.pth")
#         torch.save(model.state_dict(), model_path)
#
#         # --- ANOMALY SCORING ---
#         model.eval()
#         with torch.no_grad():
#             full_output = model(X_tensor)
#             # Use MSE for final scoring to emphasize large deviations
#             mse = torch.mean((X_tensor - full_output) ** 2, dim=(1, 2)).numpy()
#
#         threshold = np.percentile(mse, 99)
#         anomaly_count = int(np.sum(mse > threshold))
#
#         # --- VISUALIZATION ---
#         plot_path = "reports/rnn_ae_dist.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.figure(figsize=(8, 5))
#         plt.hist(mse, bins=50, color='gold', edgecolor='black', alpha=0.7)
#         plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
#         plt.title(f"Residual RNN-AE Reconstruction Error\nAnomalies Detected: {anomaly_count}")
#         plt.xlabel("Mean Squared Error")
#         plt.ylabel("Frequency")
#         plt.legend()
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "Residual_RNN_Autoencoder",
#             "saved_model_path": model_path,
#             "anomaly_count": anomaly_count,
#             "metrics": {
#                 "avg_mse": round(float(np.mean(mse)), 6),
#                 "best_val_loss": round(float(best_val_loss), 6)
#             },
#             "plot_url": plot_path,
#             "status": "SUCCESS"
#         })
#
#     except Exception as e:
#         return json.dumps({"status": "ERROR", "message": str(e)})
#
#
# class RNNAgent:
#     def __init__(self, model_client):
#         self.agent = AssistantAgent(
#             name="RNN_Agent",
#             model_client=model_client,
#             tools=[train_rnn_ae],
#             system_message="""You are the RNN Sequence Expert.
#             You use LSTM Autoencoders to capture non-linear relationships between features.
#             You MUST return a JSON containing 'model': 'RNN_Autoencoder'."""
#         )
# # Update neuro_selector order:
# AE_Agent -> VAE_Agent -> RNN_Agent -> Neuro_Critic