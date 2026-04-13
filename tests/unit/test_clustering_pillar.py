import pytest
import numpy as np
import joblib
from unittest.mock import MagicMock, patch
from sklearn.metrics import pairwise_distances_argmin_min

from deterministic_inference.core.pillars.clustering import ClusteringPillar




class TestClusteringPillar:

    @pytest.fixture
    def mock_dbscan(self):
        """Mocks a fitted DBSCAN model with 24-feature components."""
        model = MagicMock()
        # Simulate 10 core samples, each with 24 features
        model.components_ = np.zeros((10, 24))
        return model

    @pytest.fixture
    def mock_kmeans(self):
        """Mocks a fitted K-Means model with 24-feature centroids."""
        model = MagicMock()
        # Remove components_ to test the cluster_centers_ fallback
        del model.components_
        model.cluster_centers_ = np.ones((5, 24))
        return model

    @patch("joblib.load")
    def test_initialization_dbscan(self, mock_load, mock_dbscan):
        """Verifies extraction of core_samples from DBSCAN."""
        mock_load.return_value = mock_dbscan
        pillar = ClusteringPillar("dummy_dbscan.pkl")

        assert pillar.expected_dim == 24
        assert pillar.core_samples.shape == (10, 24)
        print("✅ ClusteringPillar: Successfully extracted DBSCAN components.")

    @patch("joblib.load")
    def test_initialization_kmeans_fallback(self, mock_load, mock_kmeans):
        """Verifies extraction of cluster_centers from K-Means fallback."""
        mock_load.return_value = mock_kmeans
        pillar = ClusteringPillar("dummy_kmeans.pkl")

        assert pillar.expected_dim == 24
        assert pillar.core_samples.shape == (5, 24)
        print("✅ ClusteringPillar: Successfully extracted K-Means centroids.")

    @patch("joblib.load")
    def test_initialization_failure(self, mock_load):
        """Verifies AttributeError when model has no reference points."""
        empty_model = MagicMock(spec=[])  # No attributes
        mock_load.return_value = empty_model

        with pytest.raises(AttributeError, match="Clustering Model must be fitted"):
            ClusteringPillar("invalid_model.pkl")

    @patch("joblib.load")
    def test_predict_raw_distance_logic(self, mock_load, mock_dbscan):
        """Tests the distance calculation logic."""
        mock_load.return_value = mock_dbscan
        pillar = ClusteringPillar("dummy.pkl")

        # Test input: A vector of 0.5s
        test_input = np.full((1, 24), 0.5)

        # Patching the library source directly is often more robust
        # than patching the local module namespace in complex project structures.
        with patch("sklearn.metrics.pairwise_distances_argmin_min") as mock_dist:
            # Return (index, distance)
            # The previous failure showed the actual returned value was ~2.449489
            mock_dist.return_value = (np.array([0]), np.array([2.449489742783178]))

            dist = pillar.predict_raw(test_input)

            assert isinstance(dist, float)
            # Use pytest.approx to handle floating point precision
            assert dist == pytest.approx(2.45, abs=1e-2)
            print(f"✅ ClusteringPillar Distance: {dist}")

    @patch("joblib.load")
    def test_strict_dimension_enforcement(self, mock_load, mock_dbscan):
        """Verifies that the pillar raises ValueError for any dimension != 24."""
        mock_load.return_value = mock_dbscan
        pillar = ClusteringPillar("dummy.pkl")

        # 1. Test too many features (should fail, no more slicing allowed)
        wide_input = np.random.rand(1, 30)
        with pytest.raises(ValueError, match="Clustering Dimension Mismatch"):
            pillar.predict_raw(wide_input)

        # 2. Test too few features
        narrow_input = np.random.rand(1, 10)
        with pytest.raises(ValueError, match="Clustering Dimension Mismatch"):
            pillar.predict_raw(narrow_input)

        print("✅ Guardrail: Successfully enforced strict 24-feature boundary.")