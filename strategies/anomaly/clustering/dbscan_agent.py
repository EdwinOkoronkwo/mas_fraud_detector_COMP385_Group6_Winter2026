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

from utils.logger import setup_logger

logger = setup_logger("KMeans_Tool")

from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.metrics import silhouette_score, davies_bouldin_score


import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from autogen_agentchat.agents import AssistantAgent


def train_dbscan(db_path: str, eps: float = 0.5, min_samples: int = 5) -> str:
    import os, json, joblib, numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN

    try:
        # 1. LOAD FROM STANDARDIZED BUNDLE (The Specialist's Output)
        bundle_path = "data/temp_split.joblib"
        if not os.path.exists(bundle_path):
            return json.dumps({"status": "ERROR", "message": "Standardized data bundle missing."})

        data_bundle = joblib.load(bundle_path)
        # We use the 'train' bundle features
        X_raw, _ = data_bundle['train']

        # 🚀 FIX 1: Handle NumPy (from SMOTE) vs Pandas
        # No need for manual drops; the Specialist already removed 'is_fraud', 'cc_num', etc.
        if hasattr(X_raw, 'values'):
            X = X_raw.values.astype(np.float32)
        else:
            X = X_raw.astype(np.float32)

        # 🚀 FIX 2: Sampling for O(N^2) Complexity
        # DBSCAN will freeze your system if you run it on 129k+ rows. 10k is a safe limit.
        if len(X) > 10000:
            indices = np.random.choice(len(X), 10000, replace=False)
            X_sampled = X[indices]
        else:
            X_sampled = X

        # 2. EXECUTION
        # n_jobs=-1 is vital for speed here.
        dbscan_model = DBSCAN(eps=eps, min_samples=int(min_samples), n_jobs=-1).fit(X_sampled)

        # 3. METRICS
        labels = dbscan_model.labels_
        # -1 in DBSCAN represents 'Noise' (The Anomaly)
        anomaly_count = np.count_nonzero(labels == -1)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

        # PERSISTENCE
        model_path = "models/champion_dbscan.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(dbscan_model, model_path)

        return json.dumps({
            "model": "DBSCAN",
            "status": "SUCCESS",
            "metrics": {
                "input_dim": int(X.shape[1]), # Should be 24
                "anomaly_count": int(anomaly_count),
                "clusters_found": int(cluster_count)
            },
            "params": {"eps": eps, "min_samples": int(min_samples)},
            "path": model_path
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

class DBSCANAgent:
    def __init__(self, model_client, settings):
        self.settings = settings
        self.agent = AssistantAgent(
            name="DBSCAN_Agent",
            model_client=model_client,
            tools=[train_dbscan],
            system_message=f"""
            You are the Density-Based Spatial Clustering Expert.
            Data: {self.settings.DB_PATH}
    
            TASK:
            1. Suggest an 'eps' value for DBSCAN. 
            2. If the previous agent reported -1 clusters, INCREASE your 'eps' (e.g., try 2.5 or 5.0).
            3. Call 'train_dbscan' with your suggestion.
            4. If the tool returns a 'RETRY' status, adjust your 'eps' and try again immediately.
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