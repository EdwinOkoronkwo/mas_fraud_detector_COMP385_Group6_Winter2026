from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any


class SupervisedPillar:
    def __init__(self, model_path, feature_list):
        # 1. LOAD MODEL
        loaded = joblib.load(model_path)
        self.model = loaded.get('model', loaded) if isinstance(loaded, dict) else loaded

        # 🚀 THE ALIGNMENT FIX: 
        # Ask the model exactly what names it was trained with.
        if hasattr(self.model, "get_booster"):
            self.internal_names = self.model.get_booster().feature_names
        elif hasattr(self.model, "feature_names"):
            self.internal_names = self.model.feature_names
        else:
            # Fallback to provided list if booster metadata is missing
            self.internal_names = feature_list

        self.expected_dim = len(self.internal_names) if self.internal_names else 27

    def predict(self, input_data: Any) -> float:
        """Robust inference for the Supervised (Gold) model."""
        # 1. UNIFY INPUT TO NUMPY
        if hasattr(input_data, "values"):
            raw_data = input_data.values
        else:
            raw_data = np.array(input_data)

        # 2. ENSURE 2D & FLOAT32
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        
        # 3. STRICT DIMENSION ALIGNMENT
        if raw_data.shape[1] > self.expected_dim:
            raw_data = raw_data[:, :self.expected_dim]
        elif raw_data.shape[1] < self.expected_dim:
             raise ValueError(f"Gold Mismatch: Expected {self.expected_dim}, got {raw_data.shape[1]}")

        # 🚀 STEP 4: CREATE ALIGNED DMATRIX
        # Passing internal_names here stops the 'Frozen' score and the Name Error.
        import xgboost as xgb
        dmat = xgb.DMatrix(
            raw_data.astype(np.float32), 
            feature_names=self.internal_names
        )

        # 5. RUN INFERENCE
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        raw_score = booster.predict(dmat)[0]

        return float(raw_score)