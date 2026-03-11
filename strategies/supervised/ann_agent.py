# import pandas as pd
# import numpy as np
# import os
# import json
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
# from autogen_agentchat.agents import AssistantAgent
# from mas_fraud_detector.config.settings import TEMP_SPLIT_PATH
# from typing import Dict, Any
# import joblib
#
# from mas_fraud_detector.tools.training.supervised_common_tools import save_confusion_matrix
# from mas_fraud_detector.utils.logger import setup_logger
# logger = setup_logger("ANN_Tool")

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("ANN_Tool")


import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from tools.training.supervised_common_tools import save_confusion_matrix
from utils.logger import setup_logger

logger = setup_logger("ANN_Tool")

import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score



from autogen_agentchat.agents import AssistantAgent
from config.settings import settings

import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix

import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score

import os
import joblib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score


def train_ann_flexible(hidden_layers: list, alpha: float, learning_rate: float, **kwargs) -> str:
    """
    Trains an MLPClassifier (ANN) and saves it.
    Optimized to pull from the DataSpecialist's 24-feature bundle.
    """
    import os
    import joblib
    import json
    import numpy as np
    import pandas as pd
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, recall_score

    try:
        # 🟢 RESOLUTION LOGIC: Correctly identifies the bundle path
        data_file = kwargs.get('data_path', "data/temp_split.joblib")

        if not os.path.exists(data_file):
            return json.dumps({"status": "ERROR", "message": f"Data not found at {data_file}"})

        # 1. Load Dictionary (Matches DataSpecialist output)
        data = joblib.load(data_file)
        X_train, y_train = data['train']
        X_val, y_val = data['val']

        # 2. NUMERIC SHIELD: Ensures NumPy arrays from SMOTE are handled safely
        # We convert to DataFrame ONLY to use select_dtypes, ensuring we hit exactly 24 features.
        X_train_df = pd.DataFrame(X_train).select_dtypes(include=[np.number]).fillna(0)
        X_val_df = pd.DataFrame(X_val).select_dtypes(include=[np.number]).fillna(0)

        # 3. Model Training (Scikit-Learn MLP)
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            alpha=alpha,
            learning_rate_init=learning_rate,
            max_iter=500,
            random_state=42,
            early_stopping=True
        )

        mlp.fit(X_train_df, y_train)

        # 4. Evaluation (Standard 0.5 Threshold for MLP)
        preds = mlp.predict(X_val_df)

        # 5. Save model for the Aggregator
        save_path = "models/champion_ann.joblib"
        os.makedirs("models", exist_ok=True)
        joblib.dump(mlp, save_path)

        return json.dumps({
            "agent": "ANN_Agent",
            "status": "SUCCESS",
            "metrics": {
                "f1": round(float(f1_score(y_val, preds)), 4),
                "recall": round(float(recall_score(y_val, preds)), 4),
                "feature_count": X_train_df.shape[1]
            },
            "path": save_path
        })

    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})
class ANNAgent:
    def __init__(self, model_client, project_root_path=None):
        """
        ANN Agent that autonomously designs its own neural architecture.
        """

        def run_ann_training(hidden_layers: list, alpha: float, learning_rate: float) -> str:
            return train_ann_flexible(
                data_pickle_path=settings.TEMP_SPLIT_PATH,
                hidden_layers=hidden_layers,
                alpha=alpha,
                learning_rate=learning_rate
            )

        self.agent = AssistantAgent(
            name="ANN_Agent",
            model_client=model_client,
            tools=[run_ann_training],
            system_message="""You are a Deep Learning Architect specializing in Fraud Detection.

            CHALLENGE: A Static XGBoost model has already achieved an F1-Score of 0.73. 
            Your mission is to design a Neural Network architecture that surpasses this.

            DESIGN STRATEGY:
            1. ARCHITECTURE (Bottlenecking): Use a 'Wide-to-Deep' approach. 
               - Layer 1: Start wide (e.g., 64 or 128 units) to capture all 24 feature interactions.
               - Layer 2-3: Taper down (e.g., 32, 16) to create a bottleneck that forces the model to learn abstract fraud patterns.
            2. CONVERGENCE: Use a small learning_rate (e.g., 0.001) to ensure the model doesn't overstep the global minimum in the loss landscape.
            3. REGULARIZATION: Set 'alpha' (L2 penalty) between 0.001 and 0.01 to prevent overfitting on the minority class.

            TASK:
            - Briefly explain why your specific layer choices will help capture non-linearities that XGBoost might miss.
            - Call 'run_ann_training' with your custom parameters."""
        )

# class ANNAgent:
#     def __init__(self, model_client, project_root_path=None):
#         """
#         ANN Agent that proposes its own hyperparameters to the training tool.
#         """
#
#         # 1. The wrapper MUST take the arguments the LLM will provide
#         def run_ann_training(hidden_layers: list, alpha: float, learning_rate: float) -> str:
#             # These arguments are now dynamically filled by the LLM's reasoning
#             return train_ann_flexible(
#                 data_pickle_path=TEMP_SPLIT_PATH,
#                 hidden_layers=hidden_layers,
#                 alpha=alpha,
#                 learning_rate=learning_rate
#             )
#
#         self.agent = AssistantAgent(
#             name="ANN_Agent",
#             model_client=model_client,
#             tools=[run_ann_training],  # The agent now "sees" the parameters in the tool signature
#             system_message="""You are a Deep Learning Expert specializing in Fraud Detection.
#
#             Instead of using brute-force grid search, you evaluate the task and propose
#             optimal hyperparameters based on the following logic:
#
#             - ARCHITECTURE: Use hidden_layers=[64, 32, 16] for complex fraud patterns.
#               Use [32, 16] for simpler, smaller datasets.
#             - REGULARIZATION: If the EDA report shows high noise, increase 'alpha' (e.g., 0.01) to prevent overfitting.
#             - STABILITY: Use a learning_rate of 0.001 for steady convergence.
#
#             You MUST call 'run_ann_training' with your chosen parameters to train the model."""
#         )

## AutoML (Automated ML)

#
# def train_ann_with_grid_search(data_pickle_path: str) -> str:
#     """
#     Performs Grid Search CV on an ANN to find optimum hyperparameters.
#     """
#     try:
#         if not os.path.exists(data_pickle_path):
#             return f"ERROR: Temp data not found at {data_pickle_path}"
#         logger.info("--- ANN: Loading pre-split data... ---")
#         X_train, X_test, y_train, y_test = joblib.load(data_pickle_path)
#
#         logger.info(f"--- ANN: Starting Grid Search (Data size: {len(X_train)} rows) ---")
#         # 1. Define the Parameter Grid
#         param_grid = {
#             'hidden_layer_sizes': [(64, 32), (100,), (64, 32, 16)],
#             'alpha': [0.0001, 0.001],  # L2 penalty (Regularization)
#             'learning_rate_init': [0.001, 0.01]
#         }
#
#         # 2. Initialize the Base Model
#         mlp = MLPClassifier(max_iter=300, random_state=42, early_stopping=True)
#
#         # 3. Setup Grid Search (3-Fold CV to save time/CPU)
#         grid_search = GridSearchCV(
#             estimator=mlp,
#             param_grid=param_grid,
#             scoring='f1',  # We optimize for F1 to balance Precision and Recall
#             cv=3,
#             n_jobs=-1,
#             verbose=1
#         )
#
#         # 4. Train
#         logger.info("--- ANN: Training in progress (this may take 1-2 minutes)... ---")
#         grid_search.fit(X_train, y_train)
#         best_model = grid_search.best_estimator_
#
#         # --- ADD THIS: PERSISTENCE STEP ---
#         model_dir = "mas_fraud_detector/models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_save_path = os.path.join(model_dir, "champion_ann.joblib")
#
#         joblib.dump(best_model, model_save_path)
#         logger.info(f"--- ANN: Model persisted to {model_save_path} ---")
#
#         # 5. Evaluate Best Model
#         preds = best_model.predict(X_test)
#         probs = best_model.predict_proba(X_test)[:, 1]
#         cm = confusion_matrix(y_test, preds)
#         plot_path = save_confusion_matrix(cm, "Artificial Neural Network")
#
#         metrics = {
#             "model": "artificial neural network",  # MATCHES SELECTOR
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
#
#         logger.info(f"--- ANN: Best Params Found: {grid_search.best_params_} ---")
#         return json.dumps(metrics, indent=2)
#
#     except Exception as e:
#         return f"ANN GRID SEARCH ERROR: {str(e)}"
#
#
# class ANNAgent:
#     def __init__(self, model_client, project_root_path):
#         # The wrapper CLOSES over the absolute path from settings
#         def run_ann_training() -> str:
#             # We explicitly call your tool with the constant path
#             return train_ann_with_grid_search(TEMP_SPLIT_PATH)
#
#         self.agent = AssistantAgent(
#             name="ANN_Agent",
#             model_client=model_client,
#             tools=[run_ann_training],
#             system_message="""You are a code execution unit.
#                 Your ONLY action is to call 'run_ann_training'.
#                 Do not explain. Do not predict results.
#                 If you do not call the tool, you have failed the mission."""
#         )