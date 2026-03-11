import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from autogen_agentchat.agents import AssistantAgent
from typing import Dict, Any
import joblib
from mas_fraud_detector.config.settings import TEMP_SPLIT_PATH
from mas_fraud_detector.tools.training.supervised_common_tools import save_confusion_matrix
from mas_fraud_detector.utils.logger import setup_logger

logger = setup_logger("RF_Tool")

def train_rf_flexible(data_pickle_path: str, n_estimators: int, max_depth: int = None, min_samples_split: int = 2) -> str:
    """
    Trains a Random Forest with specific hyperparameters proposed by the Agent.
    """
    try:
        if not os.path.exists(data_pickle_path):
            return f"ERROR: Temp data not found at {data_pickle_path}"

        X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)

        logger.info(f"--- RF: Training Forest with {n_estimators} trees, Max Depth: {max_depth} ---")

        # Initialize with Agent-provided params
        # Note: we keep class_weight='balanced' to help with the 175:1 imbalance
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        rf.fit(X_train, y_train)

        # Persistence
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "champion_rf.joblib")
        joblib.dump(rf, model_save_path)

        # Evaluation
        preds = rf.predict(X_test)
        probs = rf.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, preds)
        plot_path = save_confusion_matrix(cm, "Random Forest")

        metrics = {
            "model": "random forest",
            "status": "SUCCESS",
            "parameters_used": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            },
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
            }
        }
        return json.dumps(metrics, indent=2)

    except Exception as e:
        return f"RF FLEXIBLE ERROR: {str(e)}"

class RFAgent:
    def __init__(self, model_client, project_root_path=None):
        def run_rf_training(n_estimators: int, max_depth: int, min_samples_split: int) -> str:
            return train_rf_flexible(
                data_pickle_path=TEMP_SPLIT_PATH,
                n_estimators=n_estimators,
                # Agent can use 0 for unlimited, or choose a specific integer
                max_depth=max_depth if max_depth > 0 else None,
                min_samples_split=min_samples_split
            )

        self.agent = AssistantAgent(
            name="RF_Agent",
            model_client=model_client,
            tools=[run_rf_training],
            system_message="""You are an Ensemble Learning Strategist specializing in Random Forests. 
            Your goal is to design a 'Forest' that generalizes well on highly imbalanced data.

            DESIGN PRINCIPLES:
            1. FOREST SIZE (n_estimators): Scale the number of trees based on the feature complexity. 
               More trees (e.g., 250-500) reduce variance but increase computation time.
            2. TREE DEPTH (max_depth): 
               - For imbalanced data, deep trees risk 'memorizing' the few fraud cases (overfitting).
               - Shallower trees act as stronger regularizers. 
               - Set max_depth=0 if you want the trees to grow until all leaves are pure.
            3. SPLIT CONSERVATISM (min_samples_split): Higher values (e.g., 10+) prevent the model 
               from creating nodes based on a tiny, noisy subset of data.

            TASK:
            Justify your forest architecture based on the noise and imbalance reported by the EDA specialist. 
            Then, call 'run_rf_training' with your chosen parameters."""
        )

# class RFAgent:
#     def __init__(self, model_client, project_root_path=None):
#         # The wrapper now accepts the parameters the Agent decides on
#         def run_rf_training(n_estimators: int, max_depth: int, min_samples_split: int) -> str:
#             return train_rf_flexible(
#                 data_pickle_path=TEMP_SPLIT_PATH,
#                 n_estimators=n_estimators,
#                 max_depth=max_depth if max_depth > 0 else None,  # Allow agent to send 0 for None
#                 min_samples_split=min_samples_split
#             )
#
#         self.agent = AssistantAgent(
#             name="RF_Agent",
#             model_client=model_client,
#             tools=[run_rf_training],
#             system_message="""You are the Ensemble Learning Specialist (Random Forest).
#             You provide expert-guided hyperparameters instead of using brute-force search.
#
#             GUIDELINES:
#             1. 'n_estimators': Use 100 for fast iteration, 200-300 for higher stability.
#             2. 'max_depth': Use 10-20 to prevent overfitting on noisy fraud data. Use 0 for 'None' (unlimited).
#             3. 'min_samples_split': Increase this (e.g., 5 or 10) to make the model more conservative.
#
#             TASK:
#             Analyze the EDA and Data Foundation reports. If the class imbalance is severe,
#             choose a conservative 'max_depth' to avoid memorizing noise.
#             Call 'run_rf_training' with your expert selection."""
#         )
#
# def train_rf_with_grid_search(data_pickle_path: str) -> str:
#     """
#     Performs Grid Search CV on Random Forest.
#     Aims for high stability and low variance in fraud detection.
#     """
#     try:
#         if not os.path.exists(data_pickle_path):
#             return f"ERROR: Temp data not found at {data_pickle_path}"
#
#         logger.info("--- RF: Loading pre-split data... ---")
#         X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)
#
#         logger.info(f"--- ANN: Starting Grid Search (Data size: {len(X_train)} rows) ---")
#
#         # 1. Define the Parameter Grid
#         # n_estimators: Number of trees in the forest
#         # max_features: Number of features to consider at every split
#         # max_depth: How deep each tree can go
#         param_grid = {
#             'n_estimators': [100, 200],
#             'max_depth': [10, 20, None],
#             'max_features': ['sqrt', 'log2'],
#             'min_samples_split': [2, 5]
#         }
#
#         # 2. Initialize the Random Forest Classifier
#         # bootstrap=True is default and essential for the "Forest" effect
#         rf = RandomForestClassifier(
#             random_state=42,
#             n_jobs=-1,  # Parallelize across all CPU cores
#             class_weight='balanced_subsample'  # Extra protection for class weight
#         )
#
#         # 3. Setup Grid Search
#         # We use F1-score here to ensure a balance between catching fraud
#         # and not flagging every single transaction.
#         grid_search = GridSearchCV(
#             estimator=rf,
#             param_grid=param_grid,
#             scoring='f1',
#             cv=3,
#             n_jobs=-1,
#             verbose=1
#
#         )
#
#         # 4. Train
#         grid_search.fit(X_train, y_train)
#         logger.info("--- ANN: Training in progress (this may take 1-2 minutes)... ---")
#         best_model = grid_search.best_estimator_
#
#         # --- ADD THIS: PERSISTENCE STEP ---
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_save_path = os.path.join(model_dir, "champion_rf.joblib")
#
#         joblib.dump(best_model, model_save_path)
#         logger.info(f"--- RF: Model persisted to {model_save_path} ---")
#
#         # 5. Evaluate
#         preds = best_model.predict(X_test)
#         probs = best_model.predict_proba(X_test)[:, 1]
#         cm = confusion_matrix(y_test, preds)
#         plot_path = save_confusion_matrix(cm, "Random Forest")
#
#         metrics = {
#             "model": "random forest",  # MATCHES SELECTOR
#             "saved_path": model_save_path,  # Add this to the JSON so the Aggregator knows
#             "best_params": grid_search.best_params_,
#             "plot_url": plot_path,
#             "recall": round(recall_score(y_test, preds), 4),
#             "f1_score": round(f1_score(y_test, preds), 4),
#             "roc_auc": round(roc_auc_score(y_test, probs), 4),
#             "confusion_matrix": {
#                 "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
#                 "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
#             }
#         }
#         logger.info(f"--- RF: Best Params Found: {grid_search.best_params_} ---")
#         return json.dumps(metrics, indent=2)
#
#     except Exception as e:
#         return f"RANDOM FOREST GRID SEARCH ERROR: {str(e)}"
#
#
# class RFAgent:
#     def __init__(self, model_client, project_root_path):
#
#         def run_rf_training() -> str:
#             return train_rf_with_grid_search(TEMP_SPLIT_PATH)
#
#         self.agent = AssistantAgent(
#             name="RF_Agent",
#             model_client=model_client,
#             tools=[run_rf_training],
#             system_message="""You are a code execution unit.
#                 Your ONLY action is to call 'run_rf_training'.
#                 Do not explain. Do not predict results.
#                 If you do not call the tool, you have failed the mission."""
#         )