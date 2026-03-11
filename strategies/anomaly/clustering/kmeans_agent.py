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


# def train_kmeans(db_path: str) -> str:
#     """Tool for K-Means with Persistence and PCA Visualization."""
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore')
#
#         # 1. Mini-Grid Search for K (on 10k sample)
#         X_sample = X.sample(n=min(10000, len(X)), random_state=42)
#         best_score, best_k = -1, 2
#         for k in [2, 3, 5, 7, 9]:
#             km = KMeans(n_clusters=k, random_state=42, n_init=10)
#             labels_tmp = km.fit_predict(X_sample)
#             score = silhouette_score(X_sample, labels_tmp)
#             if score > best_score:
#                 best_score, best_k = score, k
#
#         # 2. FINAL EXECUTION & PERSISTENCE
#         final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
#         final_km.fit(X)  # Fit on full scaled data
#
#         # --- SAVE THE MODEL ---
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_kmeans.joblib")
#         joblib.dump(final_km, model_path)
#         logger.info(f"--- K-MEANS: Model persisted to {model_path} ---")
#
#         # 3. Visualization (PCA to 2D)
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X.sample(n=min(2000, len(X))))
#         plt.figure(figsize=(10, 6))
#         # Use labels from the actual data points sampled for PCA
#         full_labels = final_km.labels_
#         plt.scatter(X_pca[:, 0], X_pca[:, 1], c=full_labels[:len(X_pca)], cmap='viridis', s=5)
#         plt.title(f"K-Means Clustering (k={best_k})")
#         plot_path = "reports/kmeans_clusters.png"
#         plt.savefig(plot_path)
#         plt.close()
#
#         # Smaller cluster calculation
#         counts = np.bincount(full_labels)
#         anomaly_count = int(np.min(counts))
#
#         return json.dumps({
#             "model": "K-Means",
#             "best_k": best_k,
#             "saved_model_path": model_path,
#             "metrics": {"silhouette": round(best_score, 4)},
#             "anomaly_count": anomaly_count,
#             "plot_url": plot_path,
#             "status": "SUCCESS"
#         })
#     except Exception as e:
#         return f"KMEANS ERROR: {str(e)}"

from sklearn.cluster import KMeans


def train_kmeans_flexible(n_clusters: int = 8, data_path: str = "data/temp_split.joblib") -> str:
    """
    Surgically aligned K-Means tool.
    Pulls 24 features from DataSpecialist and calculates outlier distances.
    """
    try:
        # 1. LOAD FROM BUNDLE (Not SQL)
        if not os.path.exists(data_path):
            return json.dumps({"status": "ERROR", "message": f"Bundle missing at {data_path}"})

        data_bundle = joblib.load(data_path)
        # We use the 'train' features to build the 'normal' clusters
        X_raw, _ = data_bundle['train']

        # 🚀 FIX: Handle NumPy vs Pandas to get the 24 clean features
        # No need for manual drops anymore because the Specialist did it.
        X = X_raw.values if hasattr(X_raw, 'values') else X_raw
        X = X.astype(np.float32)

        # 2. EXECUTION
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
        kmeans.fit(X)

        # 3. ANOMALY DETECTION (Distance-based)
        # transform() gives distance to every centroid; we take the distance to the CLOSEST one.
        distances = kmeans.transform(X)
        min_distances = distances.min(axis=1)

        # 99th percentile threshold: Points furthest from any cluster are 'anomalies'
        threshold = np.percentile(min_distances, 99)
        y_pred = (min_distances > threshold).astype(int)
        anomaly_count = np.sum(y_pred)

        # 4. METRICS
        # Silhouette score is O(N^2), so we sample 5000 rows for speed
        sample_size = min(len(X), 5000)
        sil = silhouette_score(X, kmeans.labels_, sample_size=sample_size)

        # 5. PERSISTENCE
        model_path = "models/champion_kmeans.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(kmeans, model_path)

        return json.dumps({
            "model": "K-Means",
            "status": "SUCCESS",
            "params": {"n_clusters": int(n_clusters)},
            "metrics": {
                "inertia": round(float(kmeans.inertia_), 2),
                "silhouette": round(float(sil), 4),
                "anomaly_count": int(anomaly_count),
                "feature_count": X.shape[1]  # Should be 24
            },
            "path": model_path
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class KMeansAgent:
    def __init__(self, model_client, settings):
        self.settings = settings
        self.agent = AssistantAgent(
            name="KMeans_Agent",
            model_client=model_client,
            tools=[train_kmeans_flexible],
            system_message=f"""
            You are the Centroid Specialist.
            DATABASE_PATH: {self.settings.DB_PATH}

            TASK: Identify structural outliers via K-Means distance.
            Try n_clusters=8 and verify the Silhouette score.
            """
        )