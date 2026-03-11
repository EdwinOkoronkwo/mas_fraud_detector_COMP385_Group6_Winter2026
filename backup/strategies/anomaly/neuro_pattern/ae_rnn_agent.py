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
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.neural_network import MLPRegressor

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# Global Database Path
TEMP_SPLIT_PATH = "mas_fraud_detector/data/database.sqlite"


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
def train_rnn_ae_flexible(hidden_dim: int, latent_dim: int, threshold_percentile: float = 99.0) -> str:
    """
    Residual LSTM-based Autoencoder with Agent-defined dimensions.
    """
    # Uses the global TEMP_SPLIT_PATH defined at the top of your file
    try:
        # Data Prep
        engine = create_engine(f"sqlite:///{os.path.abspath(TEMP_SPLIT_PATH)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float32)

        X_tensor = torch.tensor(X).unsqueeze(1)  # [Batch, TimeSteps(1), Features]
        input_dim = X.shape[1]

        # Split for Early Stopping
        train_size = int(0.8 * len(X_tensor))
        val_size = len(X_tensor) - train_size
        train_ds, val_ds = random_split(TensorDataset(X_tensor), [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024)

        # Model Architecture
        class Residual_LSTM_AE(nn.Module):
            def __init__(self, in_dim, h_dim, l_dim):
                super().__init__()
                self.enc1 = nn.LSTM(in_dim, h_dim, batch_first=True)
                self.enc2 = nn.LSTM(h_dim, l_dim, batch_first=True)
                self.dec1 = nn.LSTM(l_dim, h_dim, batch_first=True)
                self.dec2 = nn.LSTM(h_dim, in_dim, batch_first=True)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                out1, _ = self.enc1(x);
                out1 = self.dropout(out1)
                _, (h2, _) = self.enc2(out1)
                latent = h2.permute(1, 0, 2)
                out3, _ = self.dec1(latent);
                out3 = self.dropout(out3)
                out4, _ = self.dec2(out3)
                return out4

        model = Residual_LSTM_AE(input_dim, hidden_dim, latent_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()

        # Training with Early Stopping
        best_val_loss = float('inf')
        patience, counter = 5, 0
        for epoch in range(30):  # Epochs reduced for agent speed
            model.train()
            for batch in train_loader:
                optimizer.zero_grad();
                loss = criterion(model(batch[0]), batch[0]);
                loss.backward();
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader: val_loss += criterion(model(batch[0]), batch[0]).item()
            avg_val = val_loss / len(val_loader)
            if avg_val < best_val_loss:
                best_val_loss = avg_val; counter = 0
            else:
                counter += 1
            if counter >= patience: break

        # Scoring
        model.eval()
        with torch.no_grad():
            full_out = model(X_tensor)
            mse = torch.mean((X_tensor - full_out) ** 2, dim=(1, 2)).numpy()

        # 2. Get Ground Truth from the same DataFrame
        y_true = df['is_fraud'].values

        # 3. BREAK THE PATTERN: Calculate actual Fraud detection
        threshold = np.percentile(mse, threshold_percentile)
        y_pred = (mse > threshold).astype(int)

        # 4. Metrics calculation
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / np.sum(y_true == 1))

        threshold = np.percentile(mse, threshold_percentile)
        anomaly_count = int(np.sum(mse > threshold))

        # Save & Plot
        plot_path = "reports/rnn_ae_dist.png"
        plt.figure(figsize=(8, 5))
        plt.hist(mse, bins=50, color='gold', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--')
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "rnn_autoencoder",  # Matches your selector key
            "status": "SUCCESS",
            "anomaly_count": int(anomaly_count),
            "metrics": {
                "avg_mse": round(float(np.mean(mse)), 6),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "true_positives": int(tp)  # This is the "Smoking Gun"
            }
        })
    except Exception as e:
        return json.dumps({
            "model": "rnn_autoencoder",
            "status": "ERROR",
            "error_message": str(e)
        })

# --- 2. THE STANDARD AE TOOL ---

def train_ae_flexible(hidden_layers: list, threshold_percentile: float = 95.0) -> str:
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(TEMP_SPLIT_PATH)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values

        from sklearn.neural_network import MLPRegressor
        full_arch = tuple(hidden_layers + hidden_layers[:-1][::-1])
        model = MLPRegressor(hidden_layer_sizes=full_arch, activation='tanh', random_state=42)
        model.fit(X, X)

        mse = np.mean(np.power(X - model.predict(X), 2), axis=1)
        threshold = np.percentile(mse, threshold_percentile)
        anomaly_count = int(np.sum(mse > threshold))

        # Persist model
        import joblib
        joblib.dump(model, "models/champion_ae.joblib")

        return json.dumps({
            "model": "autoencoder_ae",
            "status": "SUCCESS",
            "anomaly_count": int(anomaly_count),  # Ensure it's a standard int
            "metrics": {
                "avg_mse": round(float(np.mean(mse)), 6)
            },
            "metadata": {
                "threshold_used": "95th_percentile",
                "saved_at": "models/champion_ae.joblib"
            }
        })
    except Exception as e:
        return json.dumps({"model": "autoencoder_ae", "status": "ERROR", "error_message": str(e)})
# --- 3. THE AGENTS ---

class RNNAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="RNN_Agent",
            model_client=model_client,
            tools=[train_rnn_ae_flexible],  # Ensure the tool is passed here
            system_message="""You are a Technical Execution Agent.

            CRITICAL: You have access to the tool 'train_rnn_ae_flexible'. 
            Do NOT say you cannot access files. 
            Do NOT provide code snippets.

            TASK: 
            Call 'train_rnn_ae_flexible' using the db_path provided in the user request.
            Use hidden_dim=64, latent_dim=16, and threshold_percentile=99.

            Once the tool returns, output the anomaly count as a JSON-compatible string:
            {"model": "rnn_autoencoder", "status": "SUCCESS", "anomaly_count": X}"""
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