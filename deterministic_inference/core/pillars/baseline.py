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
        # 1. LOAD MODEL (Handles the 24-feat .pkl we just generated)
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            self.model = loaded.get('model', loaded)
        else:
            self.model = loaded

        self.feature_list = feature_list  # The 24 ground-truth features
        self.is_sklearn = hasattr(self.model, 'predict_proba')

    def predict(self, input_data: Any) -> float:
        """
        Robust 24-feature inference for the Baseline specialist.
        Bypasses KeyErrors by using positional slicing.
        """
        # 1. UNIFY INPUT TO NUMPY (Positional Safety)
        if isinstance(input_data, pd.DataFrame):
            # Try to use columns, fallback to raw values if headers are missing
            try:
                raw_data = input_data[self.feature_list].values
            except KeyError:
                raw_data = input_data.values[:, :24]

        elif isinstance(input_data, np.ndarray):
            raw_data = input_data
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
        else:
            # Fallback for list/dict
            raw_data = np.array(input_data)
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)

        # 2. STRICT 24-FEATURE SLICING
        # If the preprocessor sent 27 features but we only want 24:
        if raw_data.shape[1] > 24:
            raw_data = raw_data[:, :24]
        elif raw_data.shape[1] < 24:
            raise ValueError(f"Baseline Mismatch: Expected 24 features, got {raw_data.shape[1]}")

        # 3. RUN INFERENCE
        # Ensure float32 for XGBoost compatibility
        raw_data = raw_data.astype(np.float32)

        if self.is_sklearn:
            # For Scikit-Learn based models
            return float(self.model.predict_proba(raw_data)[0][1])
        else:
            # For the Baseline XGBoost (using DMatrix)
            # We omit feature_names here because raw_data is now a position-matched array
            dmatrix = xgb.DMatrix(raw_data)
            return float(self.model.predict(dmatrix)[0])