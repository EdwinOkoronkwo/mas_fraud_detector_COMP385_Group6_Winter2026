import json

import joblib
import torch
import os
import logging

logger = logging.getLogger(__name__)


class RegistryLoader:
    def __init__(self, registry_path: str, baseline_path: str):
        self.registry_path = registry_path
        self.baseline_path = baseline_path
        self.registry_data = self._read_json(registry_path)
        self.loaded_assets = {}

    def _read_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def load_all_assets(self, neuro_class):
        logger.info("📦 Loading all model assets into memory...")

        # 1. Load the Gold Tier (Supervised)
        self.loaded_assets['supervised'] = joblib.load(self.registry_data['supervised']['path'])

        # 2. Load the Neuro Tier (VAE)
        feat_count = len(self.registry_data['supervised']['features'])
        model = neuro_class(input_dim=feat_count)
        model.load_state_dict(torch.load(self.registry_data['neuro']['path']))
        model.eval()
        self.loaded_assets['neuro'] = model

        # 3. Load the Static Baseline (For Comparison)
        if os.path.exists(self.baseline_path):
            self.loaded_assets['baseline'] = joblib.load(self.baseline_path)
            logger.info("✅ Static Baseline loaded successfully.")

        return self.loaded_assets