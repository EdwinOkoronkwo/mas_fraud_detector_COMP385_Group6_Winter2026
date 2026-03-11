import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Any

import os
import json
import joblib
import numpy as np

import os
import json
import joblib
import numpy as np


class InfrastructureManager:
    def __init__(self):
        self.root = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
        self.registry_path = os.path.join(self.root, "reports", "champion_registry.json")
        self.model_dir = os.path.join(self.root, "models")
        self.db_path = os.path.join(self.root, "data", "database.sqlite")

        # MAS Architecture Blueprint
        self.input_dim = 24

        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def extract_model_input(self, processed_vector):
        """Standardizes input to 24-feature Float32 NumPy array."""
        if hasattr(processed_vector, "toarray"):
            processed_vector = processed_vector.toarray()

        if hasattr(processed_vector, "values"):
            processed_vector = processed_vector.values

        # Strict positional slicing to 24 features
        return processed_vector[:, :self.input_dim].astype(np.float32)

    def get_features(self):
        return self.registry.get('features_used', [])[:self.input_dim]

    # --- CORE MODELS ONLY ---

    def get_preprocessor_path(self):
        return os.path.join(self.model_dir, "preprocessor_base.joblib")

    def get_gold_model_path(self):
        """The Dynamic XGB Champion (F1: 0.62)."""
        return os.path.join(self.model_dir, "gold_champion.pkl")

    def get_baseline_model_path(self):
        """The 24-feat static baseline."""
        return os.path.join(self.model_dir, "xgb_baseline_24feat.pkl")

    def get_neuro_model_path(self):
        """The Champion VAE (Latent Dim: 8)."""
        return os.path.join(self.model_dir, "champion_vae.pth")

    def get_cluster_model_path(self):
        """🚀 ADDED: Points to the DBSCAN/K-Means Champion."""
        return os.path.join(self.model_dir, "champion_dbscan.joblib")

    def get_rnn_model_path(self):
        return os.path.join(self.model_dir, "champion_rnn_ae.pth")