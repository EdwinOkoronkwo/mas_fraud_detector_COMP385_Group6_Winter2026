# mas_fraud_detector/strategies/unsupervised/kmeans_agent.py
# mas_fraud_detector/strategies/unsupervised/kmeans_agent.py
import json
import os

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from autogen_agentchat.agents import AssistantAgent
from sklearn.metrics import silhouette_score

from utils.logger import setup_logger

logger = setup_logger("KMeans_Tool")


def train_kmeans(db_path: str) -> str:
    """Tool for K-Means with Persistence and PCA Visualization."""
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore')

        # 1. Mini-Grid Search for K (on 10k sample)
        X_sample = X.sample(n=min(10000, len(X)), random_state=42)
        best_score, best_k = -1, 2
        for k in [2, 3, 5, 7, 9]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_tmp = km.fit_predict(X_sample)
            score = silhouette_score(X_sample, labels_tmp)
            if score > best_score:
                best_score, best_k = score, k

        # 2. FINAL EXECUTION & PERSISTENCE
        final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_km.fit(X)  # Fit on full scaled data

        # --- SAVE THE MODEL ---
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_kmeans.joblib")
        joblib.dump(final_km, model_path)
        logger.info(f"--- K-MEANS: Model persisted to {model_path} ---")

        # 3. Visualization (PCA to 2D)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.sample(n=min(2000, len(X))))
        plt.figure(figsize=(10, 6))
        # Use labels from the actual data points sampled for PCA
        full_labels = final_km.labels_
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=full_labels[:len(X_pca)], cmap='viridis', s=5)
        plt.title(f"K-Means Clustering (k={best_k})")
        plot_path = "reports/kmeans_clusters.png"
        plt.savefig(plot_path)
        plt.close()

        # Smaller cluster calculation
        counts = np.bincount(full_labels)
        anomaly_count = int(np.min(counts))

        return json.dumps({
            "model": "K-Means",
            "best_k": best_k,
            "saved_model_path": model_path,
            "metrics": {"silhouette": round(best_score, 4)},
            "anomaly_count": anomaly_count,
            "plot_url": plot_path,
            "status": "SUCCESS"
        })
    except Exception as e:
        return f"KMEANS ERROR: {str(e)}"

class KMeansAgent:
    def __init__(self, model_client, data_path):
        # We pass the db_path directly into the instructions
        self.agent = AssistantAgent(
            name="KMeans_Agent",
            model_client=model_client,
            tools=[train_kmeans],
            system_message=f"""[INST] You are a specialized KMeans execution unit.
            1. Use the database at path: {data_path}
            2. ACTION: Call 'train_kmeans' immediately using this path.
            3. Do not provide any conversational preamble or explanations.
            4. Speak ONLY in English. [/INST]"""
        )