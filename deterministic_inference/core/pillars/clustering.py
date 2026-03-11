from typing import Any

import joblib
import numpy as np

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


class ClusteringPillar:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        # Extract reference points for distance calculation
        if hasattr(self.model, 'components_'):
            # DBSCAN stores the core samples it used to define clusters
            self.core_samples = self.model.components_
        else:
            # Fallback for K-Means or other centroid-based models
            self.core_samples = getattr(self.model, 'cluster_centers_', None)

        if self.core_samples is None:
            raise AttributeError("Clustering Model must be fitted (DBSCAN/K-Means).")

        # Lock the expected dimension (Verified as 24 in audit)
        self.expected_dim = self.core_samples.shape[1]

    def predict_raw(self, input_data: Any) -> float:
        """
        Calculates the distance to the nearest cluster/core sample.
        Enforces strict 24-feature input.
        """
        # 1. STANDARDIZE INPUT (Expects 2D NumPy from InfraManager)
        features = np.array(input_data)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 🚀 ENFORCEMENT: No more slicing.
        if features.shape[1] != self.expected_dim:
            raise ValueError(f"Clustering Dimension Mismatch: Expected {self.expected_dim}, got {features.shape[1]}")

        # 2. CALCULATE MINIMUM DISTANCE
        # We find how far this transaction is from the nearest "normal" cluster core.
        # High distance = High Anomaly Score.
        _, distances = pairwise_distances_argmin_min(features, self.core_samples)

        return float(distances[0])