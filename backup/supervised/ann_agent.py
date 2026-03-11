import pandas as pd
import numpy as np
import os
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from autogen_agentchat.agents import AssistantAgent
from config.settings import TEMP_SPLIT_PATH
from typing import Dict, Any
import joblib

from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger
logger = setup_logger("ANN_Tool")

def train_ann_with_grid_search(data_pickle_path: str) -> str:
    """
    Performs Grid Search CV on an ANN to find optimum hyperparameters.
    """
    try:
        if not os.path.exists(data_pickle_path):
            return f"ERROR: Temp data not found at {data_pickle_path}"
        logger.info("--- ANN: Loading pre-split data... ---")
        X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)

        logger.info(f"--- ANN: Starting Grid Search (Data size: {len(X_train)} rows) ---")
        # 1. Define the Parameter Grid
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (100,), (64, 32, 16)],
            'alpha': [0.0001, 0.001],  # L2 penalty (Regularization)
            'learning_rate_init': [0.001, 0.01]
        }

        # 2. Initialize the Base Model
        mlp = MLPClassifier(max_iter=300, random_state=42, early_stopping=True)

        # 3. Setup Grid Search (3-Fold CV to save time/CPU)
        grid_search = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            scoring='f1',  # We optimize for F1 to balance Precision and Recall
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        # 4. Train
        logger.info("--- ANN: Training in progress (this may take 1-2 minutes)... ---")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # --- ADD THIS: PERSISTENCE STEP ---
        model_dir = "mas_fraud_detector/models"
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "champion_ann.joblib")

        joblib.dump(best_model, model_save_path)
        logger.info(f"--- ANN: Model persisted to {model_save_path} ---")

        # 5. Evaluate Best Model
        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, preds)
        plot_path = save_confusion_matrix(cm, "Artificial Neural Network")

        metrics = {
            "model": "artificial neural network",  # MATCHES SELECTOR
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

        logger.info(f"--- ANN: Best Params Found: {grid_search.best_params_} ---")
        return json.dumps(metrics, indent=2)

    except Exception as e:
        return f"ANN GRID SEARCH ERROR: {str(e)}"


class ANNAgent:
    def __init__(self, model_client, project_root_path):
        # The wrapper CLOSES over the absolute path from settings
        def run_ann_training() -> str:
            # We explicitly call your tool with the constant path
            return train_ann_with_grid_search(TEMP_SPLIT_PATH)

        self.agent = AssistantAgent(
            name="ANN_Agent",
            model_client=model_client,
            tools=[run_ann_training],
            system_message="""You are a code execution unit. 
                Your ONLY action is to call 'run_ann_training'. 
                Do not explain. Do not predict results. 
                If you do not call the tool, you have failed the mission."""
        )