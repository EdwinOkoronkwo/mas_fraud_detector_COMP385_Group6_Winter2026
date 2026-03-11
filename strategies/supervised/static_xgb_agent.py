import json

import joblib
from autogen_agentchat.agents import AssistantAgent
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from utils.logger import setup_logger

logger = setup_logger("Static XGB_Tool")


def train_static_xgb_tool(data_path: str = "data/temp_split.joblib") -> str:
    """Trains a 'Vanilla' XGBoost as a true baseline (no tuning, no GridSearch)."""
    import os
    import joblib
    import json
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score, recall_score, precision_score

    try:
        if not os.path.exists(data_path):
            return f"ERROR: Data split not found at {data_path}"

        # 1. Load Data (Dictionary Format)
        data = joblib.load(data_path)
        X_train, y_train = data['train']
        X_val, y_val = data['val']

        # 2. Initialize Vanilla Model
        # We use standard defaults to create a "floor" for the tournament
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        )

        # 3. Train
        model.fit(X_train, y_train)

        # 4. Evaluate
        preds = model.predict(X_val)

        metrics = {
            "agent": "Static_XGB_Baseline",
            "params": "Vanilla Defaults (n=100, d=3, lr=0.1)",
            "f1": round(f1_score(y_val, preds), 4),
            "recall": round(recall_score(y_val, preds), 4),
            "precision": round(precision_score(y_val, preds), 4),
            "status": "BASELINE_ESTABLISHED"
        }

        return json.dumps(metrics)

    except Exception as e:
        return f"STATIC_XGB_ERROR: {str(e)}"

from autogen_agentchat.agents import AssistantAgent

class StaticXGBAgent:
    def __init__(self, model_client, temp_data_path):
        self.temp_data_path = temp_data_path

        # Tool with NO arguments for the LLM, but path-aware logic
        def run_static_xgb() -> str:
            """
            Executes a standard XGBoost run with best-practice default parameters:
            n_estimators=100, max_depth=3, learning_rate=0.1.
            """
            # We pass the path directly to the tool here
            return train_static_xgb_tool(data_path=self.temp_data_path)

        self.agent = AssistantAgent(
            name="Static_XGB_Agent",
            model_client=model_client,
            tools=[run_static_xgb],
            system_message=f"""You are the Reliability Specialist.
            Your data is located at: {self.temp_data_path}
            Your ONLY task is to execute the 'run_static_xgb' tool to establish 
             the standard performance benchmark for the supervised team."""
        )