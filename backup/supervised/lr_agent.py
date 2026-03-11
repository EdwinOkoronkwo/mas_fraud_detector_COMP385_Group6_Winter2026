import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from autogen_agentchat.agents import AssistantAgent
from typing import Dict, Any
from config.settings import TEMP_SPLIT_PATH

from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger
import warnings
logger = setup_logger("LR_Tool")
import joblib

from sklearn.preprocessing import StandardScaler

def train_lr_with_grid_search(data_pickle_path: str) -> str:
    # Use context manager to catch and silence Sklearn FutureWarnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            if not os.path.exists(data_pickle_path):
                return f"ERROR: Temp data not found at {data_pickle_path}"

            X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)

            # SPEED FIX: Subsample to 20k to prevent ReadTimeout
            if len(X_train) > 20000:
                X_train = X_train.sample(20000, random_state=42)
                y_train = y_train.loc[X_train.index]

            # Scaling is mandatory for Logistic Regression stability
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            param_grid = {
                'C': [0.1, 1.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }

            lr = LogisticRegression(max_iter=1000, random_state=42)
            grid_search = GridSearchCV(
                estimator=lr,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=2,
                n_jobs=-1
            )

            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_

            # --- ADD THIS: PERSISTENCE STEP ---
            model_dir = "mas_fraud_detector/models"
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, "champion_lr.joblib")

            joblib.dump(best_model, model_save_path)
            logger.info(f"--- LR: Model persisted to {model_save_path} ---")

            preds = best_model.predict(X_test_scaled)
            probs = best_model.predict_proba(X_test_scaled)[:, 1]
            cm = confusion_matrix(y_test, preds)
            plot_path = save_confusion_matrix(cm, "Linear Regression")

            metrics = {
                "model": "logistic regression",
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
            return f"LOGISTIC REGRESSION ERROR: {str(e)}"



class LRAgent:
    def __init__(self, model_client, project_root_path):

        def run_lr_training() -> str:
            return train_lr_with_grid_search(TEMP_SPLIT_PATH)

        self.agent = AssistantAgent(
            name="LR_Agent",
            model_client=model_client,
            tools=[run_lr_training],
            system_message="""You are a code execution unit. 
                Your ONLY action is to call 'run_lr_training'. 
                Do not explain. Do not predict results. 
                If you do not call the tool, you have failed the mission."""
        )