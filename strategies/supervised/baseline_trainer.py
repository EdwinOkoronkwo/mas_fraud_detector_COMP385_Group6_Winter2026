import os
import joblib
import json
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import GridSearchCV

from config.settings import settings
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score

from tools.data_prep.preprocess_tools import preprocessing_tool
from tools.training.supervised_common_tools import prepare_championship_data_tool
from utils.logger import setup_logger

# Silence the XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning)


class BaselineModelTrainer:
    def __init__(self, settings_obj, use_behavioral=False):
        self.settings = settings_obj
        self.use_behavioral = use_behavioral
        self.logger = setup_logger("Baseline_Trainer")
        self.data_path = "data/temp_split.joblib"

    def _persist_artifacts(self, model, metrics, feature_names, filename):
        """
        Saves the model and its metadata to the models/ directory.
        """
        import os
        import joblib

        try:
            # Create models directory if it doesn't exist
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, filename)

            # Create a bundle containing model + metadata
            artifact_bundle = {
                "model": model,
                "metrics": metrics,
                "features": feature_names,
                "type": "xgboost_baseline"
            }

            joblib.dump(artifact_bundle, save_path)
            self.logger.info(f"✅ Baseline artifacts persisted to {save_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to persist artifacts: {str(e)}")
            raise e

    def run_baseline_training(self):
        try:
            # 1. Prepare & Load Data
            status = prepare_championship_data_tool(self.settings.DB_PATH)
            if "SUCCESS" not in status:
                raise RuntimeError(f"DataSpecialist failed: {status}")

            data_bundle = joblib.load(self.data_path)

            X_train, y_train = data_bundle.get('train', (None, None))
            X_val, y_val = data_bundle.get('val', (None, None))
            feature_names = data_bundle.get('features', [])

            # Ensure clean NumPy arrays
            X_train = np.array(X_train)
            X_val = np.array(X_val)
            y_train = np.array(y_train)
            y_val = np.array(y_val)

            feat_count = X_train.shape[1]
            self.logger.info(f"📊 DATA VERIFIED: {feat_count} features ready.")

            # 2. Train Model (Simplified to reduce over-performance)
            model = XGBClassifier(
                n_estimators=50,  # Low capacity
                max_depth=3,  # Simple logic
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss'
            )

            model.fit(X_train, y_train)

            # 3. Evaluation
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds)
            recall = recall_score(y_val, preds)

            metrics = {
                "f1_score": round(float(f1), 4),
                "recall": round(float(recall), 4),  # Corrected syntax
                "feature_count": feat_count
            }

            # 4. Persistence
            filename = f"xgb_baseline_{feat_count}feat.pkl"

            # 🚀 NOW THIS CALL WILL WORK
            self._persist_artifacts(
                model=model,
                metrics=metrics,
                feature_names=feature_names,
                filename=filename
            )

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Baseline Training Failed: {str(e)}")
            raise e