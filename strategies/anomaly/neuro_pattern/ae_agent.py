import json
import os

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogen_agentchat.agents import AssistantAgent
from sqlalchemy import create_engine
from sklearn.neural_network import MLPRegressor

from utils.logger import setup_logger

logger = setup_logger("AE_Tool")
def train_autoencoder(db_path: str) -> str:
    """Standard Autoencoder tool for finding reconstruction anomalies."""
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values

        # Deep Bottleneck: Input -> 16 -> 8 -> 4 (Latent) -> 8 -> 16 -> Output
        model = MLPRegressor(hidden_layer_sizes=(16, 8, 4, 8, 16),
                             activation='tanh', solver='adam', max_iter=250, random_state=42)
        model.fit(X, X)

        # SAVE THE MODEL
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_ae.joblib")
        joblib.dump(model, model_path)
        logger.info(f"--- AE: Model persisted to {model_path} ---")

        # Calculate Reconstruction Error
        reconstruction = model.predict(X)
        mse = np.mean(np.power(X - reconstruction, 2), axis=1)
        threshold = np.percentile(mse, 95)  # Label top 5% as outliers
        anomaly_count = int(np.sum(mse > threshold))

        # Visualization
        plt.figure(figsize=(8, 5))
        plt.hist(mse, bins=50, color='teal', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='--', label=f'95th Percentile: {threshold:.4f}')
        plt.title(f"Standard Autoencoder: {anomaly_count} Anomalies")
        plt.xlabel("Reconstruction MSE")
        plot_path = "reports/ae_reconstruction.png"
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "Autoencoder",
            "saved_model_path": model_path,
            "anomaly_count": anomaly_count,
            "metrics": {"avg_mse": round(float(np.mean(mse)), 6)},
            "plot_url": plot_path,
            "status": "SUCCESS"
        })
    except Exception as e:
        return f"AE_TOOL_ERROR: {str(e)}"


class AEAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="AE_Agent",
            model_client=model_client,
            tools=[train_autoencoder],
            system_message="""You are the Neural Reconstruction Expert. 
            Your goal is to find fraud by training a Standard Autoencoder. 
            Points that the model CANNOT reconstruct (high MSE) are your anomalies.
            You MUST return a JSON containing 'model', 'anomaly_count', and 'plot_url'."""
        )