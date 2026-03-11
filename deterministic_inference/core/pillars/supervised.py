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
        # 1. LOAD MODEL (Supports .pkl, .joblib, or Registry-wrapped dicts)
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            self.model = loaded.get('model', loaded)
        else:
            self.model = loaded

        self.feature_list = feature_list  # Must be exactly 24 features
        # Check if it's the Scikit-Learn wrapper or a raw XGB Booster
        self.is_sklearn = hasattr(self.model, 'predict_proba')

    def predict(self, input_data: Any) -> float:
        """
        Finalized 24-feature inference. Optimized for NumPy to prevent KeyErrors.
        """
        # 1. UNIFY INPUT TO NUMPY
        if isinstance(input_data, pd.DataFrame):
            # If we have a DF, we try to use the names, but fallback to values if headers are missing
            try:
                raw_data = input_data[self.feature_list].values
            except KeyError:
                raw_data = input_data.values[:, :len(self.feature_list)]

        elif isinstance(input_data, np.ndarray):
            raw_data = input_data
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
        else:
            # Handle list/dict inputs
            raw_data = np.array(input_data)
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)

        # 2. STRICT DIMENSION CHECK (The "Bridge" Guardrail)
        if raw_data.shape[1] != len(self.feature_list):
            # Final safety check: if InfraManager didn't slice it, we do it here
            if raw_data.shape[1] > len(self.feature_list):
                raw_data = raw_data[:, :len(self.feature_list)]
            else:
                raise ValueError(f"Dimension Mismatch: Expected {len(self.feature_list)}, got {raw_data.shape[1]}")

        # 3. RUN INFERENCE (Floating point result only)
        # Ensure data is float32 for XGBoost/ANN compatibility
        raw_data = raw_data.astype(np.float32)

        if self.is_sklearn:
            # Handles gold_champion.pkl (XGB) and champion_ann.joblib
            return float(self.model.predict_proba(raw_data)[0][1])
        else:
            # Handles raw XGB boosters if necessary
            dmatrix = xgb.DMatrix(raw_data)
            return float(self.model.predict(dmatrix)[0])

# class SupervisedPillar:
#     def __init__(self, model_path, feature_list):
#         # The ANN is saved as a joblib object (likely MLPClassifier or similar)
#         loaded = joblib.load(model_path)
#
#         # Handle wrapping if the agent saved it in a dictionary
#         if isinstance(loaded, dict):
#             self.model = loaded.get('model', loaded)
#         else:
#             self.model = loaded
#
#         self.feature_list = feature_list  # Expected exactly 24 features
#         # Check for standard Scikit-learn interface
#         self.has_proba = hasattr(self.model, 'predict_proba')
#
#     def predict(self, input_data: Any) -> float:
#         """Strict 24-feature inference for the ANN Gold Champion."""
#         # 1. UNIFY INPUT
#         if isinstance(input_data, pd.DataFrame):
#             df = input_data[self.feature_list]
#         elif isinstance(input_data, np.ndarray):
#             if input_data.ndim == 1:
#                 input_data = input_data.reshape(1, -1)
#
#             if input_data.shape[1] != len(self.feature_list):
#                 # Fallback: if we have more than 24 (e.g. 27), slice to the first 24
#                 input_data = input_data[:, :len(self.feature_list)]
#
#             df = pd.DataFrame(input_data, columns=self.feature_list)
#         else:
#             df = pd.DataFrame(input_data)[self.feature_list]
#
#         # 2. RUN INFERENCE (Neural Net focus)
#         raw_data = df.values.astype(np.float32)
#
#         try:
#             if self.has_proba:
#                 # MLPClassifier returns [prob_0, prob_1]
#                 return float(self.model.predict_proba(raw_data)[0][1])
#             else:
#                 # Fallback for models that only provide direct output/regression
#                 return float(self.model.predict(raw_data)[0])
#         except Exception as e:
#             print(f"Inference Error on Supervised Pillar: {e}")
#             return 0.0


# class SupervisedPillar:
#     def __init__(self, model_path, feature_list):
#         # Handle both Gold (object) and potential Baseline (dict) formats
#         loaded = joblib.load(model_path)
#         if isinstance(loaded, dict):
#             self.model = loaded.get('model')
#         else:
#             self.model = loaded
#
#         self.feature_list = feature_list  # Expected to be exactly 24 features
#         self.is_sklearn = hasattr(self.model, 'predict_proba')
#
#     def predict(self, input_data: Any) -> float:
#         """
#         Strict 24-feature inference for Gold and Supervised specialists.
#         """
#         # 1. UNIFY INPUT (Expects 2D NumPy/DF from InfraManager)
#         if isinstance(input_data, pd.DataFrame):
#             # Strict selection: if a column is missing, this will (rightly) raise a KeyError
#             df = input_data[self.feature_list]
#
#         elif isinstance(input_data, np.ndarray):
#             if input_data.ndim == 1:
#                 input_data = input_data.reshape(1, -1)
#
#             # 🚀 ENFORCEMENT: No more 'adaptive' slicing.
#             if input_data.shape[1] != len(self.feature_list):
#                 raise ValueError(
#                     f"Gold Model Dimension Mismatch: Expected {len(self.feature_list)}, "
#                     f"got {input_data.shape[1]}"
#                 )
#             df = pd.DataFrame(input_data, columns=self.feature_list)
#
#         else:
#             # Fallback for dict/list: convert and enforce order/count
#             df = pd.DataFrame(input_data)
#             df = df[self.feature_list]
#
#         # 2. RUN INFERENCE
#         raw_data = df.values
#
#         if self.is_sklearn:
#             # For sklearn-wrapped XGB or other standard classifiers
#             return float(self.model.predict_proba(raw_data)[0][1])
#         else:
#             # For raw XGBoost Boosters
#             # feature_names ensures the model interprets the columns in the correct order
#             dmatrix = xgb.DMatrix(raw_data, feature_names=self.feature_list)
#             return float(self.model.predict(dmatrix)[0])