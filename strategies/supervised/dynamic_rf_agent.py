from datetime import datetime

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
from config.settings import settings
from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("Dynamic RF_Tool")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score


def train_dynamic_rf_tool(data_path: str = "data/temp_split.joblib",
                          n_estimators: int = 100,
                          max_depth: int = None,
                          min_samples_leaf: int = 1,
                          is_champion: bool = False) -> str:
    try:
        # 1. Load Data
        data = joblib.load(data_path)
        X_train, y_train = data['train']
        X_val, y_val = data['val']

        # 🚀 FIX: Ensure we use raw values to avoid 'Feature Name' errors later
        # This handles both Pandas and NumPy inputs gracefully
        X_tr_raw = X_train.values if hasattr(X_train, 'values') else X_train
        y_tr_raw = y_train.values if hasattr(y_train, 'values') else y_train
        X_v_raw = X_val.values if hasattr(X_val, 'values') else X_val
        y_v_raw = y_val.values if hasattr(y_val, 'values') else y_val

        # 2. Initialize and Train
        # Note: We keep class_weight but SMOTE has already done the heavy lifting
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr_raw, y_tr_raw)

        # 3. Evaluation at 0.5 Threshold (Standard for RF)
        # If you want to match XGBoost's 0.3 threshold, use predict_proba here
        preds = model.predict(X_v_raw)

        # Determine Path
        save_path = "models/champion_rf.pkl" if is_champion else "models/challenger_rf.pkl"

        metrics = {
            "status": "SUCCESS",
            "agent": "Dynamic_RF_Agent",
            "type": "random_forest",
            "feature_count": X_tr_raw.shape[1],
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf
            },
            # Explicitly cast to float for JSON safety
            "f1": round(float(f1_score(y_v_raw, preds)), 4),
            "recall": round(float(recall_score(y_v_raw, preds)), 4),
            "precision": round(float(precision_score(y_v_raw, preds)), 4),
            "model_path": save_path
        }

        # 4. Physical Save
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, save_path)

        return json.dumps(metrics)

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class DynamicRFAgent:
    def __init__(self, model_client, data_path):
        self.data_path = data_path

        # The wrapper ensures the path and save flag are handled automatically
        def run_rf_tournament(n_estimators: int, max_depth: int = None, min_samples_leaf: int = 1) -> str:
            """
            Trains the Random Forest and PHYSICALLY SAVES it as the champion.
            """
            return train_dynamic_rf_tool(
                data_path=self.data_path,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                is_champion=True  # This is the trigger for the joblib.dump logic
            )

        self.agent = AssistantAgent(
            name="Dynamic_RF_Challenger",
            model_client=model_client,
            tools=[run_rf_tournament],
            system_message=f"""
            You are a Random Forest Optimizer. 

            CRITICAL PROTOCOL:
            1. You MUST call 'run_rf_tournament' to execute the training. The model is NOT saved unless you call this tool.
            2. After calling the tool, review the JSON output. 
            3. If the results are satisfactory, output the final metrics and terminate.

            MISSION: Optimize F1 score using 'balanced_subsample' for fraud detection.
            DATA_PATH: {self.data_path}
            """
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