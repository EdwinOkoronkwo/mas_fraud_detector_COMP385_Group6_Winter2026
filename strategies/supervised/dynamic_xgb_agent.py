import json
import os

import joblib
from autogen_agentchat.agents import AssistantAgent
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

import os
import json
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score


# --- TOOL DEFINITION ---
import joblib
import json
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score


import numpy as np
# ... (rest of your imports)

def train_dynamic_xgb_tool(n_estimators: int, max_depth: int, learning_rate: float,
                           scale_pos_weight: float = 1.0, # 👈 Added this parameter
                           data_path: str = "data/temp_split.joblib",
                           save_path: str = "models/gold_xgb.pkl") -> str:
    try:
        data = joblib.load(data_path)
        X_train, y_train = data['train']
        X_val, y_val = data['val']

        # 🚀 FIX: Convert to numpy only IF it's a DataFrame, otherwise use as-is
        # This handles both pandas.DataFrame and numpy.ndarray safely
        X_train_raw = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_raw = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_raw = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_raw = y_val.values if hasattr(y_val, 'values') else y_val

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=20,  # 👈 Use the agent's suggestion
            random_state=42,
            tree_method='hist'
        )

        # Train on raw values
        model.fit(X_train_raw, y_train_raw)

        # Evaluate at the Pipeline's 0.3 Threshold
        # (This is great—0.3 is better for fraud recall than the 0.5 default)
        y_probs = model.predict_proba(X_val_raw)[:, 1]
        preds = (y_probs >= 0.3).astype(int)

        metrics = {
            "f1": round(float(f1_score(y_val_raw, preds)), 4),
            "recall": round(float(recall_score(y_val_raw, preds)), 4),
            "precision": round(float(precision_score(y_val_raw, preds)), 4),
            "feature_count": X_train_raw.shape[1],
            "status": "SUCCESS"
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        return json.dumps(metrics)

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})
# --- AGENT CLASS ---


class DynamicXGBAgent:
    def __init__(self, model_client, temp_data_path):
        self.temp_data_path = temp_data_path

        # Updated tool wrapper to include scale_pos_weight
        def run_dynamic_xgb(n_estimators: int, max_depth: int,
                            learning_rate: float, scale_pos_weight: float) -> str:
            return train_dynamic_xgb_tool(
                data_path=self.temp_data_path,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,  # 👈 Passed through
                save_path="models/gold_xgb.pkl"
            )

        self.agent = AssistantAgent(
            name="Dynamic_XGB_Agent",
            model_client=model_client,
            tools=[run_dynamic_xgb],
            system_message=f"""
                    You are a Senior ML Engineer. Current ANN Champion is at F1=0.63. 
                    XGBoost MUST beat this to be certified as GOLD.

                    CRITICAL PARAMETER: 'scale_pos_weight'
                    - Your previous attempts at scale=100+ resulted in 90%+ Recall but failed 
                      on Precision (F1 crashed).
                    - NEW RANGE: Try values between 10.0 and 50.0. 
                    - Logic: Since we evaluate at a 0.3 threshold, a lower scale (e.g., 15-30) 
                      usually yields the highest F1 by protecting Precision.

                    HYPERPARAMETER GUIDANCE:
                    - 'max_depth': 4-7. 
                    - 'n_estimators': 400-800.
                    - 'learning_rate': 0.01-0.05.

                    STRATEGY:
                    1. Attempt 1 (The Balanced Guard): (scale=20, n=500, d=4, lr=0.05).
                    2. Attempt 2 (The Interaction Search): (scale=35, n=600, d=6, lr=0.02).
                    3. Attempt 3: Pivot based on which attempt got closer to F1=0.77.
                    """
        )