# mas_fraud_detector/strategies/unsupervised/som_agent.py
# mas_fraud_detector/strategies/unsupervised/som_agent.py
import json
import os

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from minisom import MiniSom # Ensure this is installed: pip install minisom
from autogen_agentchat.agents import AssistantAgent

from mas_fraud_detector.schema.som_verification import SOMVerification
from mas_fraud_detector.utils.logger import setup_logger

logger = setup_logger("SOM_Tool")


def train_som_flexible(db_path: str, grid_size: int, threshold_percentile: int = 95) -> str:
    """
    SOM training with Agent-defined grid topology and anomaly sensitivity.
    """
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float64)

        logger.info(
            f"--- SOM: Training {grid_size}x{grid_size} grid (Sensitivity: {threshold_percentile}th percentile) ---")

        # 1. Training
        som = MiniSom(grid_size, grid_size, X.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X)
        som.train_random(X, 1000)

        # 2. Persistence
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_som.joblib")
        joblib.dump(som, model_path)

        # 3. Anomaly Detection via Quantization Error
        # Points that don't fit well into the map topology are anomalies
        q_errors = np.array([som.quantization_error([x]) for x in X])
        threshold = np.percentile(q_errors, threshold_percentile)
        anomaly_count = int(np.sum(q_errors > threshold))
        avg_q_error = np.mean(q_errors)

        # 4. Visualization (U-Matrix)
        plt.figure(figsize=(grid_size / 2, grid_size / 2))
        plt.pcolor(som.distance_map().T, cmap='bone_r')  # Darker areas = further apart
        plt.colorbar()
        plt.title(f"SOM U-Matrix ({grid_size}x{grid_size}) - Anomalies: {anomaly_count}")

        plot_path = "reports/som_umatrix.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "Self-Organizing Map",
            "status": "SUCCESS",
            "parameters_used": {
                "grid_size": f"{grid_size}x{grid_size}",
                "threshold_percentile": threshold_percentile
            },
            "anomaly_count": anomaly_count,
            "metrics": {"avg_quantization_error": round(avg_q_error, 4)},
            "plot_url": plot_path,
            "saved_model_path": model_path
        })
    except Exception as e:
        return f"SOM FLEXIBLE ERROR: {str(e)}"


class SOMAgent:
    def __init__(self, model_client, db_path):
        def run_som_training(grid_size: int, threshold_percentile: int) -> str:
            return train_som_flexible(
                db_path=db_path,
                grid_size=grid_size,
                threshold_percentile=threshold_percentile
            )

        self.agent = AssistantAgent(
            name="SOM_Agent",
            model_client=model_client,
            tools=[run_som_training],
            # We removed response_format to stop the warning
            system_message=f"""[INST] You are a Topological Mapping Specialist.

            DB: {db_path}

            STRICT OUTPUT RULE:
            Your final response must be a single JSON object. Do not include any text before, after, or around the JSON.

            JSON SCHEMA:
            {{
                "grid_size": int,
                "threshold_percentile": int,
                "quantization_error": float,
                "anomaly_count": int,
                "topology_reasoning": "string"
            }}

            ACTION:
            1. Call 'run_som_training'.
            2. Output the result using the JSON SCHEMA above. [/INST]"""
        )
#
# def train_som(db_path: str) -> str:
#     """SOM with U-Matrix visualization and error-based anomaly counting."""
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float64)
#
#         # 1. Mini-Grid Search for best grid size
#         best_q_error = float('inf')
#         best_size = 10
#         for size in [7, 10]:
#             som = MiniSom(size, size, X.shape[1], sigma=1.0, learning_rate=0.5)
#             som.random_weights_init(X)
#             som.train_random(X, 500)
#             err = som.quantization_error(X)
#             if err < best_q_error:
#                 best_q_error = err
#                 best_size = size
#
#         # 2. Final Training
#         final_som = MiniSom(best_size, best_size, X.shape[1])
#         final_som.random_weights_init(X)
#         final_som.train_random(X, 1000)
#
#         # SAVE MODEL (MiniSom objects use pickle/joblib)
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_som.joblib")
#         joblib.dump(final_som, model_path)
#         logger.info(f"--- SOM: Model persisted to {model_path} ---")
#
#         # 3. ANOMALY COUNT: Use Quantization Error threshold (95th percentile)
#         # Any point far from its matching neuron is an anomaly
#         q_errors = np.array([final_som.quantization_error([x]) for x in X])
#         threshold = np.percentile(q_errors, 95)
#         anomaly_count = int(np.sum(q_errors > threshold))
#
#         # 4. U-Matrix Plot
#         plt.figure(figsize=(best_size / 2, best_size / 2))
#         plt.pcolor(final_som.distance_map().T, cmap='bone_r')
#         plt.colorbar()
#         plt.title(f"SOM U-Matrix (Detected Anomalies: {anomaly_count})")
#         plot_path = f"reports/som_umatrix.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "SOM",
#             "best_grid": f"{best_size}x{best_size}",
#             "anomaly_count": anomaly_count, # CRITICAL: For Critic cross-check
#             "metrics": {"quantization_error": round(best_q_error, 4)},
#             "plot_url": plot_path,
#             "status": "SUCCESS"
#         })
#     except Exception as e:
#         return f"SOM ERROR: {str(e)}"
#
# class SOMAgent:
#     def __init__(self, model_client, db_path):
#         self.agent = AssistantAgent(
#             name="SOM_Agent",
#             model_client=model_client,
#             tools=[train_som],
#             system_message=f"""[INST] You are a SOM specialist.
#             1. Use database: {db_path}
#             2. Call 'train_som' immediately.
#             3. Repeat the JSON result and stop. [/INST]"""
#         )