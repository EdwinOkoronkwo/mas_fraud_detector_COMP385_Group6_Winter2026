import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import MagicMock, patch

from deterministic_inference.core.pillars.supervised import SupervisedPillar

import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import MagicMock, patch



class TestSupervisedPillar:

    @pytest.fixture
    def feature_list(self):
        """The strict 24-feature list required by the MAS."""
        return [f"f{i}" for i in range(24)]

    @pytest.fixture
    def mock_model(self):
        """A mock Scikit-Learn style model."""
        model = MagicMock()
        # Returns [prob_class_0, prob_class_1]
        model.predict_proba.return_value = np.array([[0.1, 0.9]])
        return model

    @patch("joblib.load")
    def test_pillar_initialization_from_dict(self, mock_load, feature_list, mock_model):
        """
        Tests if the pillar correctly extracts the model from a Registry-wrapped dict
        by mocking joblib.load to avoid PicklingErrors.
        """
        # Configure the mock to return a dictionary
        mock_load.return_value = {'model': mock_model, 'metadata': {'version': 1.0}}

        pillar = SupervisedPillar("dummy_path.pkl", feature_list)

        assert pillar.is_sklearn is True
        assert pillar.model == mock_model
        assert pillar.feature_list == feature_list

    @patch("joblib.load")
    def test_input_unification_to_float32(self, mock_load, feature_list, mock_model):
        """
        CRITICAL TEST: Ensures input is cast to float32 and dimensions are enforced.
        """
        mock_load.return_value = mock_model
        pillar = SupervisedPillar("dummy_path.pkl", feature_list)

        # 1. Provide a DataFrame with 26 features (2 extra)
        # Pillar should slice this down to the first 24
        df_input = pd.DataFrame(np.random.rand(1, 26), columns=[f"f{i}" for i in range(26)])

        # 2. Run prediction
        score = pillar.predict(df_input)

        # 3. Verify math compatibility
        args, _ = mock_model.predict_proba.call_args
        passed_data = args[0]

        assert passed_data.dtype == np.float32  # Required for TensorFlow/XGB consistency
        assert passed_data.shape[1] == 24
        assert score == 0.9

    @patch("joblib.load")
    def test_dimension_mismatch_error(self, mock_load, feature_list, mock_model):
        """
        Verifies that the pillar raises a ValueError if data is smaller than 24 features.
        """
        mock_load.return_value = mock_model
        pillar = SupervisedPillar("dummy_path.pkl", feature_list)

        # Input only has 10 features (invalid)
        small_input = np.random.rand(1, 10)

        with pytest.raises(ValueError, match="Dimension Mismatch"):
            pillar.predict(small_input)

    @patch("joblib.load")
    @patch("xgboost.DMatrix")
    def test_raw_xgboost_booster_fallback(self, mock_dmatrix, mock_load, feature_list):
        """
        Verifies logic for raw XGB boosters that don't have predict_proba.
        """
        mock_booster = MagicMock(spec=['predict'])  # Only has 'predict', no 'predict_proba'
        mock_booster.predict.return_value = np.array([0.75])
        mock_load.return_value = mock_booster

        pillar = SupervisedPillar("dummy_path.pkl", feature_list)

        assert pillar.is_sklearn is False

        score = pillar.predict(np.random.rand(1, 24))

        assert score == 0.75
        mock_dmatrix.assert_called_once()


