import os
import json
import joblib
import numpy as np
from typing import Any

class InfrastructureManager:
    def __init__(self):
        # 📂 Centralized Path Management
        self.root = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
        self.model_dir = os.path.join(self.root, "models")
        self.promoted_dir = os.path.join(self.model_dir, "promoted") 
        self.registry_path = os.path.join(self.root, "reports", "champion_registry.json")
        
        # 🚀 DATABASE PATH (Restored for DataHandler)
        self.db_path = os.path.join(self.root, "data", "database.sqlite")
        
        # 🚀 CONSTANT: The final 27-feature architecture
        self.input_dim = 27 

        # Load registry if exists
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def extract_model_input(self, processed_vector: Any) -> np.ndarray:
        """Standardizes input to 27-feature Float32 NumPy array."""
        if hasattr(processed_vector, "toarray"):
            processed_vector = processed_vector.toarray()

        if not isinstance(processed_vector, np.ndarray):
            processed_vector = np.array(processed_vector)

        if processed_vector.ndim == 1:
            processed_vector = processed_vector.reshape(1, -1)

        final_vector = np.ascontiguousarray(
            processed_vector[:, :self.input_dim], 
            dtype=np.float32
        )
        return final_vector

    # --- PROMOTED MODEL PATHS ---
    
    def get_preprocessor_path(self):
        return os.path.join(self.promoted_dir, "preprocessor_base.joblib")

    def get_gold_model_path(self):
        return os.path.join(self.promoted_dir, "gold_champion.pkl")

    def get_baseline_model_path(self):
        return os.path.join(self.promoted_dir, "xgb_baseline_27feat.pkl")

    def get_neuro_model_path(self):
        return os.path.join(self.promoted_dir, "champion_vae.pth")

    def get_cluster_model_path(self):
        return os.path.join(self.promoted_dir, "champion_dbscan.joblib")

    def get_features(self):
        """Returns feature names, capped at 27."""
        return self.registry.get('features_used', [])[:self.input_dim]