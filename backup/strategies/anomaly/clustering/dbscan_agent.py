# mas_fraud_detector/strategies/unsupervised/dbscan_agent.py
# mas_fraud_detector/strategies/unsupervised/dbscan_agent.py
import json
import os

import joblib
import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from mas_fraud_detector.utils.logger import setup_logger

logger = setup_logger("KMeans_Tool")


def train_dbscan_flexible(db_path: str, eps: float = 0.5, min_samples: int = 5) -> str:
    """
    DBSCAN training tool where hyperparameters are controlled by the Agent.
    """
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore')

        # Execute with Agent's parameters
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        anomaly_count = list(labels).count(-1)
        noise_ratio = anomaly_count / len(X)

        # Save model
        model_path = "mas_fraud_detector/models/champion_dbscan.joblib"
        joblib.dump(db, model_path)

        # Visualization (PCA)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=5, alpha=0.5)
        plt.title(f"DBSCAN: {anomaly_count} Anomalies (eps={eps}, min_samples={min_samples})")
        plot_path = "reports/dbscan_anomalies.png"
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "DBSCAN",
            "params": {"eps": eps, "min_samples": min_samples},
            "metrics": {"noise_ratio": round(noise_ratio, 4)},
            "anomaly_count": anomaly_count,
            "status": "SUCCESS"
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class DBSCANAgent:
    def __init__(self, model_client, data_path):
        self.agent = AssistantAgent(
            name="DBSCAN_Agent",
            model_client=model_client,
            tools=[train_dbscan_flexible],
            system_message=f"""
            [INST] You are the DBSCAN Hyperparameter Specialist.
            DATABASE_PATH: {data_path}

            GOAL: Find an 'eps' and 'min_samples' configuration where the noise_ratio is between 0.005 (0.5%) and 0.02 (2%).

            STRATEGY:
            1. Start with eps=0.5, min_samples=5.
            2. If noise_ratio > 0.02, your epsilon is too small (clusters are too tight). INCREASE eps.
            3. If noise_ratio < 0.005, your epsilon is too large (everything is merging). DECREASE eps.
            4. Once the goal is met, provide the final JSON and STOP. [/INST]
            """
        )
# def train_dbscan(db_path: str) -> str:
#     """DBSCAN with EPS tuning and anomaly scatter visualization."""
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore')
#
#         # 1. Tuning EPS
#         # We sample for speed; tuning on 5k rows is enough to find a good EPS
#         X_sample = X.sample(n=min(5000, len(X)), random_state=42)
#         best_eps = 0.5
#         for e in [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
#             db = DBSCAN(eps=e, min_samples=5).fit(X_sample)
#             noise_ratio = list(db.labels_).count(-1) / len(X_sample)
#             if 0.01 < noise_ratio < 0.10:  # Aim for 1-10% noise
#                 best_eps = e
#                 break
#
#         # 2. Final Execution
#         final_db = DBSCAN(eps=best_eps, min_samples=5).fit(X)
#         labels = final_db.labels_
#         anomaly_count = list(labels).count(-1)
#
#         # SAVE MODEL
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_dbscan.joblib")
#         joblib.dump(final_db, model_path)
#         logger.info(f"--- DBSCAN: Model persisted to {model_path} ---")
#
#         # 3. Visualization via PCA
#         pca = PCA(n_components=2)
#         # Sample for the plot to keep it clean (3000 points)
#         X_pca = pca.fit_transform(X)
#         X_plot = X_pca[:3000]
#         labels_plot = labels[:3000]
#
#         plt.figure(figsize=(10, 6))
#         # Plot normal clusters in a muted color
#         plt.scatter(X_plot[labels_plot != -1, 0], X_plot[labels_plot != -1, 1],
#                     c=labels_plot[labels_plot != -1], cmap='viridis', s=5, alpha=0.5, label='Clusters')
#         # Plot anomalies in bright red
#         plt.scatter(X_plot[labels_plot == -1, 0], X_plot[labels_plot == -1, 1],
#                     c='red', s=15, marker='x', label='Anomalies (Noise)')
#
#         plt.title(f"DBSCAN: {anomaly_count} Anomalies Detected (eps={best_eps})")
#         plt.legend()
#         plot_path = "reports/dbscan_anomalies.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "DBSCAN",
#             "best_eps": best_eps,
#             "saved_model_path": model_path,
#             "metrics": {"noise_ratio": round(anomaly_count / len(X), 4)},
#             "anomaly_count": anomaly_count,
#             "plot_url": plot_path,
#             "status": "SUCCESS"
#         })
#     except Exception as e:
#         return f"DBSCAN ERROR: {str(e)}"
#
#
#
# class DBSCANAgent:
#     def __init__(self, model_client, data_path):
#         self.agent = AssistantAgent(
#             name="DBSCAN_Agent",
#             model_client=model_client,
#             tools=[train_dbscan],
#             # Injecting the data_path ensures Mistral doesn't "hallucinate" a filename
#             system_message=f"""[INST] You are a specialized DBSCAN execution unit.
#             1. Data Source: {data_path}
#             2. ACTION: Call 'train_dbscan' immediately using this path.
#             3. Use default parameters (eps=0.5, min_samples=5) unless history suggests otherwise.
#             4. Speak ONLY in English. Do not explain your actions. [/INST]"""
#         )