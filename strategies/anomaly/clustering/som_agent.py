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

from utils.logger import setup_logger

logger = setup_logger("SOM_Tool")


def train_som(db_path: str) -> str:
    """SOM with U-Matrix visualization and error-based anomaly counting."""
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore').values.astype(np.float64)

        # 1. Mini-Grid Search for best grid size
        best_q_error = float('inf')
        best_size = 10
        for size in [7, 10]:
            som = MiniSom(size, size, X.shape[1], sigma=1.0, learning_rate=0.5)
            som.random_weights_init(X)
            som.train_random(X, 500)
            err = som.quantization_error(X)
            if err < best_q_error:
                best_q_error = err
                best_size = size

        # 2. Final Training
        final_som = MiniSom(best_size, best_size, X.shape[1])
        final_som.random_weights_init(X)
        final_som.train_random(X, 1000)

        # SAVE MODEL (MiniSom objects use pickle/joblib)
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_som.joblib")
        joblib.dump(final_som, model_path)
        logger.info(f"--- SOM: Model persisted to {model_path} ---")

        # 3. ANOMALY COUNT: Use Quantization Error threshold (95th percentile)
        # Any point far from its matching neuron is an anomaly
        q_errors = np.array([final_som.quantization_error([x]) for x in X])
        threshold = np.percentile(q_errors, 95)
        anomaly_count = int(np.sum(q_errors > threshold))

        # 4. U-Matrix Plot
        plt.figure(figsize=(best_size / 2, best_size / 2))
        plt.pcolor(final_som.distance_map().T, cmap='bone_r')
        plt.colorbar()
        plt.title(f"SOM U-Matrix (Detected Anomalies: {anomaly_count})")
        plot_path = f"reports/som_umatrix.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "SOM",
            "best_grid": f"{best_size}x{best_size}",
            "anomaly_count": anomaly_count, # CRITICAL: For Critic cross-check
            "metrics": {"quantization_error": round(best_q_error, 4)},
            "plot_url": plot_path,
            "status": "SUCCESS"
        })
    except Exception as e:
        return f"SOM ERROR: {str(e)}"

class SOMAgent:
    def __init__(self, model_client, db_path):
        self.agent = AssistantAgent(
            name="SOM_Agent",
            model_client=model_client,
            tools=[train_som],
            system_message=f"""[INST] You are a SOM specialist.
            1. Use database: {db_path}
            2. Call 'train_som' immediately.
            3. Repeat the JSON result and stop. [/INST]"""
        )