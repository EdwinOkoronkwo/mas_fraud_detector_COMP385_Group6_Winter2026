import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import joblib

from agentic_inference.core.batch_processor import BatchProcessor
from deterministic_inference.core.scoring_engine import ScoringEngine
from deterministic_inference.core.weight_adapter import WeightAdapter


# Importing the components described in the prompt
# Note: In a real environment, ensure these are importable from your project structure


import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import joblib



class TestScoringAndIntegration:

    @pytest.fixture
    def scoring_engine(self):
        adapter = WeightAdapter(momentum=0.5)
        return ScoringEngine(adapter=adapter)

    def test_normalization_logic(self, scoring_engine):
        """Verifies that sigmoid normalization maps raw values to (0, 1)."""
        # Test Neuro Normalization (MSE)
        # Center is 0.10, Slope is 0.02
        low_mse = 0.02
        high_mse = 0.20

        score_low = scoring_engine._normalize_neuro(low_mse, 0.10, 0.02)
        score_high = scoring_engine._normalize_neuro(high_mse, 0.10, 0.02)

        assert 0 < score_low < 0.5  # Below center
        assert 0.5 < score_high < 1.0  # Above center
        assert score_high > score_low

    def test_consensus_boost_logic(self, scoring_engine):
        """Tests the CONSENSUS_BOOST mode when Gold and Neuro agree."""
        # gold > 0.5 and n_p > 0.5
        result = scoring_engine.compute_mas_score(
            gold_prob=0.8,
            neuro_mse=0.15,  # Will result in n_p > 0.5
            cluster_dist=1.0
        )

        assert result['mode'] == "CONSENSUS_BOOST"
        assert result['final_score'] == 0.99

    def test_weight_adaptation_cycle(self):
        """
        Verifies that weights shift dynamically.
        FIX: We now check that the relative trust in a failing agent drops,
        avoiding the 0.7 baseline trap.
        """
        adapter = WeightAdapter(momentum=0.8)  # High momentum to see shifts quickly

        # Capture initial state
        initial_weights = adapter.get_weights().copy()
        initial_cluster_weight = initial_weights['cluster']

        # Simulate 'cluster' agent failing repeatedly:
        # It predicts fraud (c_p=0.9) but the actual label is clean (actual=0) -> False Positive
        # While 'gold' and 'neuro' stay neutral/accurate.
        for _ in range(10):
            adapter.update_performance(actual=0, gold_p=0.01, n_p=0.01, c_p=0.9)

        new_weights = adapter.get_weights()

        # The 'cluster' agent should have been penalized for the high False Positive rate
        assert new_weights['cluster'] < initial_cluster_weight

        # The 'gold' agent's relative importance should grow compared to the failing 'cluster' agent
        assert new_weights['gold'] > new_weights['cluster']

        print(
            f"✅ Weight Shift Validated: Cluster weight dropped from {initial_cluster_weight:.2f} to {new_weights['cluster']:.2f}")

    @pytest.mark.asyncio
    async def test_batch_processor_extraction(self):
        """
        Tests the BatchProcessor's ability to extract metrics
        from a raw dictionary using the pipeline pillars.
        """
        # Mocking the pipeline and its pillars
        mock_pipeline = MagicMock()
        mock_pipeline.base_pillar.predict.return_value = 0.1
        mock_pipeline.gold_pillar.predict.return_value = 0.7
        mock_pipeline.neuro_pillar.predict.return_value = 0.12
        mock_pipeline.cluster_pillar.predict_raw.return_value = 2.0

        # Mock preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.zeros((1, 24))

        with patch("joblib.load", return_value=mock_preprocessor):
            processor = BatchProcessor(mock_pipeline)

            sample_row = {
                "cc_num": "1234567890123456",
                "amt": 100.0,
                "merchant": "Test Shop",
                "actual_label": 1
            }

            metrics = processor._extract_metrics(sample_row)

            assert str(metrics['CC']).endswith("3456")
            assert metrics['GOLD'] == 0.7
            assert "mode" in metrics
            print("✅ BatchProcessor: Correctly extracted CC suffix and pillar metrics.")

    def test_safe_veto_logic(self, scoring_engine):
        """Tests GOLD_SAFE_VETO when top agents agree transaction is clean."""
        # gold < 0.15 and n_p < 0.15
        result = scoring_engine.compute_mas_score(
            gold_prob=0.05,
            neuro_mse=0.01,  # Very low MSE
            cluster_dist=0.5
        )

        assert result['mode'] == "GOLD_SAFE_VETO"
        assert result['final_score'] < 0.05  # Suppressed score