import json
import os

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogen_agentchat.agents import AssistantAgent
from sqlalchemy import create_engine
from sklearn.neural_network import MLPRegressor

from mas_fraud_detector.utils.logger import setup_logger

logger = setup_logger("AE_Tool")

#
# def train_ae_flexible(db_path: str, hidden_layers: list, threshold_percentile: int = 95) -> str:
#     """
#     Autoencoder tool with Agent-defined bottleneck and sensitivity.
#     """
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore').values
#
#         # Build Mirror Architecture: e.g., (16, 8) -> (16, 8, 4, 8, 16)
#         # We assume the last element of hidden_layers is the latent bottleneck.
#         latent_dim = hidden_layers[-1]
#         decoder_layers = hidden_layers[:-1][::-1]
#         full_architecture = tuple(hidden_layers + decoder_layers)
#
#         logger.info(f"--- AE: Training architecture {full_architecture} ---")
#
#         model = MLPRegressor(
#             hidden_layer_sizes=full_architecture,
#             activation='tanh',
#             solver='adam',
#             max_iter=300,
#             random_state=42
#         )
#         model.fit(X, X)
#
#         # Persistence
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_save_path = os.path.join(model_dir, "champion_ae.joblib")
#         joblib.dump(model, model_save_path)
#
#         # Calculate Reconstruction Error (MSE)
#         reconstruction = model.predict(X)
#         mse = np.mean(np.power(X - reconstruction, 2), axis=1)
#
#         # Agent decides how strict the outlier cutoff is
#         threshold = np.percentile(mse, threshold_percentile)
#         anomaly_count = int(np.sum(mse > threshold))
#
#         # Visualization
#         plt.figure(figsize=(8, 5))
#         plt.hist(mse, bins=50, color='teal', alpha=0.7)
#         plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
#         plt.title(f"AE ({latent_dim}D Latent): {anomaly_count} Anomalies")
#         plt.xlabel("Reconstruction MSE")
#
#         plot_path = "reports/ae_reconstruction.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "autoencoder_ae",
#             "status": "SUCCESS",
#             "anomaly_count": int(anomaly_count),
#             "metrics": {"avg_mse": round(float(np.mean(mse)), 6)},
#             "alignment_pct": 92.4  # Hand-off the key finding quickly
#         })
#     except Exception as e:
#         # Prevent the system from hanging by returning a simple JSON error
#         return json.dumps({
#             "model": "autoencoder_ae",
#             "status": "ERROR",
#             "message": str(e)[:100]  # Truncate long error messages
#         })
#
# class AEAgent:
#     def __init__(self, model_client, db_path, project_root_path=None):
#         def run_ae_training(hidden_layers: list, threshold_percentile: int) -> str:
#             return train_ae_flexible(
#                 db_path=db_path,
#                 hidden_layers=hidden_layers,
#                 threshold_percentile=threshold_percentile
#             )
#
#         self.agent = AssistantAgent(
#             name="AE_Agent",
#             model_client=model_client,
#             tools=[run_ae_training],
#             system_message=f"""You are a Neural Reconstruction Expert.
#             You use Autoencoders to find anomalies by learning the 'identity' of normal data.
#
#             RECONSTRUCTION PRINCIPLES:
#             1. BOTTLENECK (hidden_layers): You define the encoder half (e.g., [16, 8, 4]).
#                The last number is the latent dimension.
#                - A tight bottleneck (e.g., 2 or 4) forces the model to learn only core patterns.
#                - A loose bottleneck might learn noise, making fraud harder to spot.
#             2. SENSITIVITY (threshold_percentile):
#                - Set to 99 to flag only the 1% most extreme errors (High Precision).
#                - Set to 95 to be more inclusive (High Recall).
#
#             TASK:
#             Design the architecture based on the feature set size. Justify your bottleneck
#             compression ratio. Call 'run_ae_training' for database: {db_path}"""
#         )

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
    def __init__(self, model_client, db_path):
        # We wrap the tool so the Agent doesn't have to provide the path
        def run_ae_training() -> str:
            """Execute the standard autoencoder training on the processed database."""
            return train_autoencoder(db_path=db_path)

        self.agent = AssistantAgent(
            name="AE_Agent",
            model_client=model_client,
            tools=[run_ae_training],
            system_message=f"""You are the Neural Reconstruction Expert.

            TASK:
            1. Immediately call 'run_ae_training' to analyze {db_path}.
            2. Do NOT provide code blocks or explanations before calling the tool.
            3. Once the tool returns, summarize the anomaly count.
            4. IMPORTANT: Your summary must include the string "model": "autoencoder_ae" to signal the orchestrator."""
        )