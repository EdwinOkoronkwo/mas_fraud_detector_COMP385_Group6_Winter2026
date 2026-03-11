import pandas as pd
import numpy as np
import os
import json
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from autogen_agentchat.agents import AssistantAgent
from typing import Dict, Any
import joblib
from mas_fraud_detector.config.settings import TEMP_SPLIT_PATH
from mas_fraud_detector.tools.training.supervised_common_tools import save_confusion_matrix
from mas_fraud_detector.utils.logger import setup_logger

logger = setup_logger("XGB_Tool")


def train_xgb_flexible(data_pickle_path: str, n_estimators: int, max_depth: int, learning_rate: float,
                       threshold: float = 0.5) -> str:
    """
    Trains XGBoost with specific hyperparameters and a custom decision threshold proposed by the Agent.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        try:
            if not os.path.exists(data_pickle_path):
                return f"ERROR: Temp data not found at {data_pickle_path}"

            X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)

            logger.info(f"--- XGB: Training with {n_estimators} trees, LR: {learning_rate}, Threshold: {threshold} ---")

            # Initialize with Agent-provided params
            # tree_method='hist' is kept for speed efficiency
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                tree_method='hist',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            # Persistence
            model_dir = "mas_fraud_detector/models"
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, "champion_xgb.joblib")
            joblib.dump(model, model_save_path)

            # Evaluation with custom thresholding
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= threshold).astype(int)

            cm = confusion_matrix(y_test, preds)
            plot_path = save_confusion_matrix(cm, "Extreme Gradient Boosting")

            metrics = {
                "model": "extreme gradient boosting",
                "status": "SUCCESS",
                "parameters_used": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "threshold": threshold
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
            return f"XGBOOST FLEXIBLE ERROR: {str(e)}"


class XGBAgent:
    def __init__(self, model_client, project_root_path=None):
        def run_xgb_training(n_estimators: int, max_depth: int, learning_rate: float, threshold: float) -> str:
            return train_xgb_flexible(
                data_pickle_path=TEMP_SPLIT_PATH,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                threshold=threshold
            )

        self.agent = AssistantAgent(
            name="XGB_Agent",
            model_client=model_client,
            tools=[run_xgb_training],
            system_message="""You are a Gradient Boosting Architect (XGBoost Specialist). 
            Your goal is to build a high-performance ensemble that targets rare fraud events.

            BOOSTING STRATEGY:
            1. LEARNING PACE (learning_rate & n_estimators): 
               - A smaller learning_rate (e.g., 0.01) makes the model more robust to noise but 
                 requires more estimators (trees) to converge. 
               - A higher rate (e.g., 0.1) converges faster but risks overfitting minority outliers.
            2. COMPLEXITY (max_depth): Control the interaction depth. In fraud, shallow trees (3-6) 
               often generalize better than deep, complex trees that might 'memorize' noise.
            3. DETECTION SENSITIVITY (threshold): This is your tactical lever. 
               - Lowering the threshold (e.g., below 0.5) prioritizes RECALL (catching more fraud). 
               - Raising it prioritizes PRECISION (reducing false alarms).

            TASK:
            Propose a configuration that maximizes fraud capture while maintaining operational 
            feasibility. Justify your threshold choice based on the cost of missing a fraud 
            case versus the cost of a false alarm. Call 'run_xgb_training' to proceed."""
        )

# class XGBAgent:
#     def __init__(self, model_client, project_root_path=None):
#         # Wrapper passing Agent's expert decisions to the tool
#         def run_xgb_training(n_estimators: int, max_depth: int, learning_rate: float, threshold: float) -> str:
#             return train_xgb_flexible(
#                 data_pickle_path=TEMP_SPLIT_PATH,
#                 n_estimators=n_estimators,
#                 max_depth=max_depth,
#                 learning_rate=learning_rate,
#                 threshold=threshold
#             )
#
#         self.agent = AssistantAgent(
#             name="XGB_Agent",
#             model_client=model_client,
#             tools=[run_xgb_training],
#             system_message="""You are the Gradient Boosting Specialist (XGBoost).
#             You use your expertise to optimize boosting rounds and decision thresholds.
#
#             GUIDELINES:
#             1. 'learning_rate': Use 0.01 for a conservative "slow learn" or 0.1 for faster convergence.
#             2. 'max_depth': Keep it low (3-6) to avoid overfitting the noise in fraud data.
#             3. 'threshold': In fraud, the cost of a False Negative is high. If you want to increase
#                RECALL, propose a lower threshold (e.g., 0.2 or 0.3) instead of the default 0.5.
#
#             TASK:
#             Review the Data Foundation details. Propose a configuration that balances
#             high recall without completely destroying precision.
#             Call 'run_xgb_training' with your selection."""
#         )

# def train_xgb_with_grid_search(data_pickle_path: str) -> str:
#     # Silence XGBoost internal warnings
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
#         try:
#             if not os.path.exists(data_pickle_path):
#                 return f"ERROR: Temp data not found at {data_pickle_path}"
#
#             X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)
#
#             # SPEED FIX: Consistent with LR to ensure the tournament finishes
#             if len(X_train) > 20000:
#                 X_train = X_train.sample(20000, random_state=42)
#                 y_train = y_train.loc[X_train.index]
#
#             param_grid = {
#                 'n_estimators': [100, 200],
#                 'max_depth': [3, 5, 7],
#                 'learning_rate': [0.01, 0.1],
#                 'subsample': [0.8]
#             }
#
#             # REMOVED: use_label_encoder (deprecated)
#             # ADDED: tree_method='hist' for a massive speed boost
#             xgb = XGBClassifier(
#                 eval_metric='logloss',
#                 tree_method='hist',
#                 random_state=42,
#                 n_jobs=-1
#             )
#
#             grid_search = GridSearchCV(
#                 estimator=xgb,
#                 param_grid=param_grid,
#                 scoring='recall',
#                 cv=2, # Reduced from 3 for speed
#                 n_jobs=-1
#             )
#
#             grid_search.fit(X_train, y_train)
#             best_model = grid_search.best_estimator_
#
#             # --- ADD THIS: PERSISTENCE STEP ---
#             model_dir = "mas_fraud_detector/models"
#             os.makedirs(model_dir, exist_ok=True)
#             model_save_path = os.path.join(model_dir, "champion_xgb.joblib")
#
#             joblib.dump(best_model, model_save_path)
#             logger.info(f"--- XGB: Model persisted to {model_save_path} ---")
#
#             # 1. Get probabilities instead of hard 0/1 predictions
#             probs = best_model.predict_proba(X_test)[:, 1]
#
#             # 2. Find the best threshold for recall (e.g., trying 0.1, 0.2, 0.3)
#             # Or just force a lower threshold to see if it beats LR
#             best_threshold = 0.2
#             preds = (probs >= best_threshold).astype(int)
#
#             # 3. Recalculate Confusion Matrix with the new threshold
#             cm = confusion_matrix(y_test, preds)
#             plot_path = save_confusion_matrix(cm, "Extreme Gradient Boosting")
#
#             metrics = {
#                 "model": "extreme gradient boosting",
#                 "saved_path": model_save_path,  # Add this to the JSON so the Aggregator knows
#                 "best_params": grid_search.best_params_,
#                 "plot_url": plot_path,
#                 "recall": round(recall_score(y_test, preds), 4),
#                 "f1_score": round(f1_score(y_test, preds), 4),
#                 "roc_auc": round(roc_auc_score(y_test, probs), 4),
#                 "confusion_matrix": {
#                     "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
#                     "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
#                 }
#             }
#             return json.dumps(metrics, indent=2)
#
#         except Exception as e:
#             return f"XGBOOST GRID SEARCH ERROR: {str(e)}"
#
#
# class XGBAgent:
#     def __init__(self, model_client, project_root_path):
#
#         def run_xgb_training() -> str:
#             return train_xgb_with_grid_search(TEMP_SPLIT_PATH)
#
#         self.agent = AssistantAgent(
#             name="XGB_Agent",
#             model_client=model_client,
#             tools=[run_xgb_training],
#             system_message="""You are a code execution unit.
#                 Your ONLY action is to call 'run_xgb_training'.
#                 Do not explain. Do not predict results.
#                 If you do not call the tool, you have failed the mission."""
#         )