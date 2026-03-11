# mas_fraud_detector/strategies/unsupervised/iso_forest_agent.py
# mas_fraud_detector/strategies/unsupervised/iso_forest_agent.py
import json
import os

import joblib
from mas_fraud_detector.utils.logger import setup_logger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from autogen_agentchat.agents import AssistantAgent
logger = setup_logger("IsoForest_Tool")

#
# def train_iso_forest_flexible(db_path: str, contamination: float, n_estimators: int = 100) -> str:
#     """
#     Isolation Forest execution with Agent-defined contamination levels.
#     """
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#         X = df.drop(columns=['is_fraud'], errors='ignore')
#
#         logger.info(f"--- ISO_FOREST: Training with contamination={contamination}, estimators={n_estimators} ---")
#
#         # 1. Execution
#         iso = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
#         preds = iso.fit_predict(X)
#
#         # Calculate anomaly stats
#         anomaly_count = int(np.sum(preds == -1))
#         avg_score = np.mean(iso.decision_function(X[preds == -1]))
#
#         # 2. SAVE THE MODEL
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, "champion_isoforest.joblib")
#         joblib.dump(iso, model_path)
#
#         # 3. Visualization via PCA (Reduced set for plotting speed)
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X)
#
#         # Create a mesh grid for the contour plot
#         xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 50),
#                              np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 50))
#
#         # Fit a small viz model on PCA space to show boundaries
#         iso_viz = IsolationForest(contamination=contamination).fit(X_pca)
#         Z = iso_viz.decision_function(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
#
#         plt.figure(figsize=(8, 6))
#         plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#         plt.scatter(X_pca[:2000, 0], X_pca[:2000, 1], c='white', s=1, alpha=0.5)
#         plt.title(f"Isolation Forest Boundaries (Detected Anomalies: {anomaly_count})")
#
#         plot_path = "reports/isoforest_contours.png"
#         os.makedirs("reports", exist_ok=True)
#         plt.savefig(plot_path)
#         plt.close()
#
#         return json.dumps({
#             "model": "isolation_forest",  # <--- CRITICAL: Selector trigger
#             "status": "SUCCESS",
#             "anomaly_count": int(anomaly_count),
#             "metrics": {
#                 "contamination": contamination,
#                 "n_estimators": n_estimators
#             },
#             "saved_model_path": "mas_fraud_detector/models/champion_iso.joblib"
#         })
#     except Exception as e:
#         return json.dumps({"model": "isolation_forest", "status": "ERROR", "message": str(e)})
#
#
# class IsolationForestAgent:
#     def __init__(self, model_client, db_path, project_root_path=None):
#         def run_iso_training(contamination: float, n_estimators: int) -> str:
#             return train_iso_forest_flexible(
#                 db_path=db_path,
#                 contamination=contamination,
#                 n_estimators=n_estimators
#             )
#
#         self.agent = AssistantAgent(
#             name="IsoForest_Agent",
#             model_client=model_client,
#             tools=[run_iso_training],
#             system_message=f"""You are an Outlier Detection Specialist.
#             Isolation Forest isolates observations by randomly selecting a feature and a split value.
#             Anomalies are easier to isolate and thus have shorter path lengths in the trees.
#
#             STRATEGIC PARAMETERS:
#             1. CONTAMINATION: This is your 'expected fraud rate'.
#                - If the EDA showed a 175:1 ratio, a contamination of ~0.005 to 0.01 is logical.
#                - Increasing this will flag more transactions as suspicious but increase False Positives.
#             2. N_ESTIMATORS: The number of isolation trees. 100 is standard, 200 is more robust.
#
#             TASK:
#             Justify your contamination level based on the known fraud distribution in the dataset.
#             Call 'run_iso_training' with your expert parameters. Data: {db_path}"""
#         )


def train_iso_forest(db_path: str) -> str:
    """Isolation Forest with Decision Boundary visualization and anomaly counting."""
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        X = df.drop(columns=['is_fraud'], errors='ignore')

        # 1. Tuning Contamination
        best_score = -1
        best_contam = 0.1
        for c in [0.01, 0.05, 0.1]:
            iso = IsolationForest(contamination=c, random_state=42)
            preds = iso.fit_predict(X)
            avg_score = np.mean(iso.decision_function(X[preds == -1]))
            if avg_score > best_score:
                best_score = avg_score
                best_contam = c

        # 2. ANOMALY COUNT: Generate binary labels for the Critic
        final_iso = IsolationForest(contamination=best_contam, random_state=42)
        final_preds = final_iso.fit_predict(X)
        # Convert scikit-learn labels: -1 (anomaly) -> 1, 1 (normal) -> 0
        anomaly_count = int(np.sum(final_preds == -1))

        # --- SAVE THE MODEL FOR INFERENCE ---
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "champion_isoforest.joblib")
        joblib.dump(final_iso, model_path)
        logger.info(f"--- ISOFOREST: Model persisted to {model_path} ---")

        # 3. Visualization via PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        iso_2d = IsolationForest(contamination=best_contam).fit(X_pca)

        xx, yy = np.meshgrid(np.linspace(X_pca[:,0].min(), X_pca[:,0].max(), 50),
                             np.linspace(X_pca[:,1].min(), X_pca[:,1].max(), 50))
        Z = iso_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='white', s=1, alpha=0.5)
        plt.title(f"Isolation Forest Boundaries (Anomalies: {anomaly_count})")
        plot_path = "reports/isoforest_contours.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        plt.close()

        return json.dumps({
            "model": "Isolation Forest",
            "best_contamination": best_contam,
            "anomaly_count": anomaly_count, # CRITICAL: For Critic cross-check
            "saved_model_path": model_path,  # Report the path to the Aggregator
            "metrics": {"avg_anomaly_score": round(best_score, 4)},
            "plot_url": plot_path,
            "status": "SUCCESS"
        })
    except Exception as e:
        return f"ISOFOREST ERROR: {str(e)}"


from functools import partial


class IsolationForestAgent:
    def __init__(self, model_client, db_path):
        # We use partial to lock the db_path into the function immediately
        # This prevents the 'NoneType' error by pre-binding the argument
        run_iso_tool = partial(train_iso_forest, db_path=db_path)

        # We must give the partial function a name and docstring for the LLM to see it
        run_iso_tool.__name__ = "run_iso_training"
        run_iso_tool.__doc__ = "Executes Isolation Forest training on the database."

        self.agent = AssistantAgent(
            name="IsoForest_Agent",
            model_client=model_client,
            tools=[run_iso_tool],
            system_message=f"""You are an Isolation Forest specialist.
            1. Target Database: {db_path}
            2. ACTION: Call 'run_iso_training' immediately.
            3. Do not ask for parameters; they are pre-configured.
            4. Report the anomaly_count and saved_model_path when finished."""
        )

# class IsolationForestAgent:
#     def __init__(self, model_client, db_path):
#         self.agent = AssistantAgent(
#             name="IsoForest_Agent",
#             model_client=model_client,
#             tools=[train_iso_forest],
#             system_message=f"""[INST] You are an Isolation Forest specialist.
#             1. Use database: {db_path}
#             2. Call 'train_iso_forest'.
#             3. Repeat JSON and stop. [/INST]"""
#         )