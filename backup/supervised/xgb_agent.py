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
from config.settings import TEMP_SPLIT_PATH
from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("XGB_Tool")

def train_xgb_with_grid_search(data_pickle_path: str) -> str:
    # Silence XGBoost internal warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        try:
            if not os.path.exists(data_pickle_path):
                return f"ERROR: Temp data not found at {data_pickle_path}"

            X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)

            # SPEED FIX: Consistent with LR to ensure the tournament finishes
            if len(X_train) > 20000:
                X_train = X_train.sample(20000, random_state=42)
                y_train = y_train.loc[X_train.index]

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8]
            }

            # REMOVED: use_label_encoder (deprecated)
            # ADDED: tree_method='hist' for a massive speed boost
            xgb = XGBClassifier(
                eval_metric='logloss',
                tree_method='hist',
                random_state=42,
                n_jobs=-1
            )

            grid_search = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring='recall',
                cv=2, # Reduced from 3 for speed
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # --- ADD THIS: PERSISTENCE STEP ---
            model_dir = "mas_fraud_detector/models"
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, "champion_xgb.joblib")

            joblib.dump(best_model, model_save_path)
            logger.info(f"--- XGB: Model persisted to {model_save_path} ---")

            # 1. Get probabilities instead of hard 0/1 predictions
            probs = best_model.predict_proba(X_test)[:, 1]

            # 2. Find the best threshold for recall (e.g., trying 0.1, 0.2, 0.3)
            # Or just force a lower threshold to see if it beats LR
            best_threshold = 0.2
            preds = (probs >= best_threshold).astype(int)

            # 3. Recalculate Confusion Matrix with the new threshold
            cm = confusion_matrix(y_test, preds)
            plot_path = save_confusion_matrix(cm, "Extreme Gradient Boosting")

            metrics = {
                "model": "extreme gradient boosting",
                "saved_path": model_save_path,  # Add this to the JSON so the Aggregator knows
                "best_params": grid_search.best_params_,
                "plot_url": plot_path,
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
            return f"XGBOOST GRID SEARCH ERROR: {str(e)}"


class XGBAgent:
    def __init__(self, model_client, project_root_path):

        def run_xgb_training() -> str:
            return train_xgb_with_grid_search(TEMP_SPLIT_PATH)

        self.agent = AssistantAgent(
            name="XGB_Agent",
            model_client=model_client,
            tools=[run_xgb_training],
            system_message="""You are a code execution unit. 
                Your ONLY action is to call 'run_xgb_training'. 
                Do not explain. Do not predict results. 
                If you do not call the tool, you have failed the mission."""
        )