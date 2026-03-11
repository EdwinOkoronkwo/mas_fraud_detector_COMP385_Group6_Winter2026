import pandas as pd
import torch
import numpy as np
import joblib
import os

import os

from inference.pillars.supervised_pillar import SupervisedPillar
import os
import joblib
from inference.pillars.supervised_pillar import SupervisedPillar
from utils.logger import setup_logger

logger = setup_logger("Inference_Engine")


class InferenceEngine:
    def __init__(self, infra_manager):
        self.infra = infra_manager
        self.logger = logger

        # Load Registry Assets
        registry = self.infra.get_asset('registry')
        model_filename = registry['model_info']['file']
        model_path = os.path.join(self.infra.settings.MODELS_DIR, model_filename)
        golden_features = registry['inference_params']['expected_features']

        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model file missing: {model_path}")
            raise FileNotFoundError(f"Model not found at: {model_path}")

        self.logger.info(f"🚀 Initializing {registry['model_info']['name']} with {len(golden_features)} Golden Features")

        # Initialize the pillar
        self._champion = SupervisedPillar(model_path, golden_features)

        # --- NEW: Result Cache for Accuracy Tracking ---
        self.last_run_result = None

    @property
    def champion(self):
        return self._champion

    def execute_supervised_inference(self, scaled_data: dict) -> dict:
        """
        Takes the scaled dictionary from the Preprocessor and runs it through the XGBoost model.
        """
        try:
            # 1. Run the model
            score = self.champion.predict(scaled_data)

            # 2. Prepare the result
            result = {
                "probability": round(float(score), 4),
                "prediction": 1 if score > 0.5 else 0,  # Map to binary
                "risk_level": "HIGH" if score > 0.5 else "LOW",
                "status": "SUCCESS",
                "model_version": "gold_xgb_v2"
            }

            # 3. CRITICAL: Save to the INSTANCE so the Orchestrator can see it
            self.last_run_result = result
            self.logger.info(f"✅ Cache Updated: Pred={result['prediction']} (Prob={result['probability']})")

            return result
        except Exception as e:
            self.logger.error(f"Inference Crash: {str(e)}")
            return {"status": "ERROR", "message": str(e)}