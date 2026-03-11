import os
import json
import joblib
from utils.logger import setup_logger


import os
import joblib
from utils.logger import setup_logger

import os
import json
import joblib
import pandas as pd
from utils.logger import setup_logger


class InfrastructureManager:
    """The Handshake Handler: Verifies the Supervised Champion & Assets."""

    def __init__(self, settings_obj):
        self.settings = settings_obj
        self.logger = setup_logger("Infrastructure_Manager")
        self.assets = {}
        self.manifest = None

    def verify_and_load(self):
        self.logger.info("🤝 STARTING INFRASTRUCTURE HANDSHAKE...")

        # 1. Define Paths
        registry_path = os.path.join(self.settings.REPORT_DIR, "champion_registry.json")
        scaler_path = os.path.join(self.settings.MODELS_DIR, "scaler.joblib")

        # 2. Load Registry
        if not os.path.exists(registry_path):
            self.logger.error(f"❌ Registry NOT FOUND at {registry_path}")
            return False

        with open(registry_path, 'r') as f:
            self.manifest = json.load(f)
            self.assets['registry'] = self.manifest

        # 3. Dynamically locate the Supervised Champion using your Manifest keys
        # The JSON you sent uses ['model_info']['file']
        champion_filename = self.manifest['model_info']['file']
        champion_path = os.path.join(self.settings.MODELS_DIR, champion_filename)

        if os.path.exists(champion_path):
            self.logger.info(f"✅ Supervised Champion FOUND: {champion_path}")
            self.assets['supervised_model'] = joblib.load(champion_path)
        else:
            self.logger.error(f"❌ Supervised Champion MISSING at {champion_path}")
            return False

        # 4. Extract Features directly from the Manifest
        # This replaces the need for a separate feature_names.joblib
        if 'inference_params' in self.manifest:
            self.assets['feature_names'] = self.manifest['inference_params']['expected_features']
            self.logger.info(f"📋 FEATURE SCHEMA LOADED: {len(self.assets['feature_names'])} features.")
        else:
            self.logger.error("❌ Feature Names MISSING in Registry JSON.")
            return False

        # 5. Check Scaler
        if os.path.exists(scaler_path):
            self.assets['scaler'] = joblib.load(scaler_path)
            self.logger.info("✅ Scaler LOADED.")
        else:
            self.logger.warning("⚠️ Scaler MISSING. Ensure Preprocessor can handle raw data.")

        self.logger.info("🤝 HANDSHAKE COMPLETE: Supervised Pillar Ready.")
        return True

    def get_asset(self, key):
        """Safe retrieval of loaded assets (model, scaler, registry)."""
        return self.assets.get(key)