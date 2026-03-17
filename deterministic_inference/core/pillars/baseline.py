from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any

class BaselinePillar:
    def __init__(self, model_path, feature_list):
        loaded = joblib.load(model_path)
        self.model = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded
        
        # 🚀 STEP 1: Get the names the model was ACTUALLY trained with
        if hasattr(self.model, "get_booster"):
            self.internal_names = self.model.get_booster().feature_names
        else:
            self.internal_names = self.model.feature_names
            
        self.expected_dim = len(self.internal_names) if self.internal_names else 27

    def predict(self, input_data: Any) -> float:
        # 1. Standardize to NumPy array
        raw_data = input_data.values if hasattr(input_data, "values") else np.array(input_data)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        # 2. Slice/Pad to match the model's training dimension
        # If model expects 27 but we have more/less, we must align here
        if raw_data.shape[1] > self.expected_dim:
            raw_data = raw_data[:, :self.expected_dim]
        
        # 🚀 STEP 2: Create DMatrix using the model's OWN feature names
        # This guarantees name-alignment and prevents the 'Frozen' score bug
        import xgboost as xgb
        dmat = xgb.DMatrix(
            raw_data.astype(np.float32), 
            feature_names=self.internal_names
        )

        # 3. Predict using the booster directly
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        raw_score = booster.predict(dmat)[0]

        return float(raw_score)