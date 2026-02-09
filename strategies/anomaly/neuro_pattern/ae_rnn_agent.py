from autogen_agentchat.agents import AssistantAgent


def train_rnn_ae(db_path: str) -> str:
    """
    Residual LSTM-based Autoencoder with Early Stopping, Huber Loss,
    and Dimension-Corrected residual connections.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sqlalchemy import create_engine
        import json
        import logging

        logger = logging.getLogger(__name__)

        # --- DATA PREPARATION ---
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float32)

        # Reshape for LSTM: [Batch, TimeSteps(1), Features]
        X_tensor = torch.tensor(X).unsqueeze(1)
        input_dim = X.shape[1]

        # Split for Early Stopping (80% Train, 20% Validation)
        train_size = int(0.8 * len(X_tensor))
        val_size = len(X_tensor) - train_size
        train_ds, val_ds = random_split(TensorDataset(X_tensor), [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024)

        # --- MODEL DEFINITION ---
        class Residual_LSTM_AE(nn.Module):
            def __init__(self, in_dim, hidden_dim=32, latent_dim=16):
                super().__init__()
                # Encoder: Compresses signal
                self.enc1 = nn.LSTM(in_dim, hidden_dim, batch_first=True)
                self.enc2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)

                # Decoder: Reconstructs signal
                self.dec1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
                self.dec2 = nn.LSTM(hidden_dim, in_dim, batch_first=True)

                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                # Encoding
                out1, _ = self.enc1(x)
                out1 = self.dropout(out1)
                _, (h2, _) = self.enc2(out1)

                # Bottleneck permutation [1, B, L] -> [B, 1, L]
                latent = h2.permute(1, 0, 2)

                # Decoding
                out3, _ = self.dec1(latent)
                out3 = self.dropout(out3)
                out4, _ = self.dec2(out3)

                # Residual Connection: Focuses learning on the reconstruction error
                return out4 + x

        # --- TRAINING SETUP ---
        model = Residual_LSTM_AE(input_dim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()  # Huber Loss: Robust to extreme transaction amounts

        # Early Stopping State
        patience = 5
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        # --- TRAINING LOOP ---
        epochs = 50
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch in train_loader:
                inputs = batch[0]
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, inputs)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validation Phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    val_out = model(inputs)
                    total_val_loss += criterion(val_out, inputs).item()

            avg_val_loss = total_val_loss / len(val_loader)

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore Best Weights
        if best_model_state:
            model.load_state_dict(best_model_state)

        # --- PERSISTENCE ---
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_rnn_ae.pth")
        torch.save(model.state_dict(), model_path)

        # --- ANOMALY SCORING ---
        model.eval()
        with torch.no_grad():
            full_output = model(X_tensor)
            # Use MSE for final scoring to emphasize large deviations
            mse = torch.mean((X_tensor - full_output) ** 2, dim=(1, 2)).numpy()

        threshold = np.percentile(mse, 99)
        anomaly_count = int(np.sum(mse > threshold))

        # --- VISUALIZATION ---
        plot_path = "reports/rnn_ae_dist.png"
        os.makedirs("reports", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.hist(mse, bins=50, color='gold', edgecolor='black', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
        plt.title(f"Residual RNN-AE Reconstruction Error\nAnomalies Detected: {anomaly_count}")
        plt.xlabel("Mean Squared Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "Residual_RNN_Autoencoder",
            "saved_model_path": model_path,
            "anomaly_count": anomaly_count,
            "metrics": {
                "avg_mse": round(float(np.mean(mse)), 6),
                "best_val_loss": round(float(best_val_loss), 6)
            },
            "plot_url": plot_path,
            "status": "SUCCESS"
        })

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class RNNAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="RNN_Agent",
            model_client=model_client,
            tools=[train_rnn_ae],
            system_message="""You are the RNN Sequence Expert. 
            You use LSTM Autoencoders to capture non-linear relationships between features.
            You MUST return a JSON containing 'model': 'RNN_Autoencoder'."""
        )
# Update neuro_selector order:
# AE_Agent -> VAE_Agent -> RNN_Agent -> Neuro_Critic