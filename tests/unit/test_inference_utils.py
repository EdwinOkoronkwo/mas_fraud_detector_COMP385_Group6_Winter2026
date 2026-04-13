import pytest
import pandas as pd
import numpy as np
import torch
import os
import json
import sqlite3
from unittest.mock import MagicMock, patch

from deterministic_inference.utils.data_handler import DataHandler
from deterministic_inference.utils.infrastructure import InfrastructureManager


class TestDataHandler:
    """Tests for the DataHandler SQL discovery and cleaning logic."""

    @pytest.fixture
    def mock_db_setup(self):
        """Sets up a mock connection and cursor for SQLite testing."""
        mock_conn = MagicMock()
        mock_cursor = mock_conn.cursor.return_value
        # Simulate finding the 'transactions' table
        mock_cursor.fetchall.return_value = [("transactions",)]
        return mock_conn

    @patch("sqlite3.connect")
    @patch("pandas.read_sql_query")
    def test_fetch_balanced_samples_logic(self, mock_read_sql, mock_connect, mock_db_setup):
        """Verifies table discovery and UNION ALL query construction."""
        mock_connect.return_value = mock_db_setup

        # Mock DataFrame return representing balanced classes
        mock_df = pd.DataFrame({
            'cc_num': ['1', '2'],
            'amt': [100, 200],
            'actual_label': [1, 0]
        })
        mock_read_sql.return_value = mock_df

        handler = DataHandler("dummy.db")
        df = handler.fetch_balanced_samples(2)

        assert len(df) == 2
        # Verify the UNION ALL query was parameterized correctly for balanced limits
        args, kwargs = mock_read_sql.call_args
        assert "UNION ALL" in args[0]
        assert "is_fraud = 1" in args[0]
        assert "is_fraud = 0" in args[0]
        assert kwargs['params'] == [1, 1]

    @patch("sqlite3.connect")
    @patch("pandas.read_sql_query")
    def test_get_transaction_by_cc_cleaning(self, mock_read_sql, mock_connect, mock_db_setup):
        """Verifies regex cleaning of merchant names and label type casting."""
        mock_connect.return_value = mock_db_setup

        # Dirty data from DB: prefix "fraud_" and underscores
        raw_df = pd.DataFrame([{
            'cc_num': '1234',
            'merchant': 'fraud_kristin_and_sons',
            'actual_label': 1.0  # Float from SQL
        }])
        mock_read_sql.return_value = raw_df

        handler = DataHandler("dummy.db")
        clean_df = handler.get_transaction_by_cc("1234")

        # Assertions on cleaning logic
        # 1. 'fraud_' removed 2. underscores to spaces 3. Title Case
        assert clean_df.iloc[0]['merchant'] == "Kristin And Sons"
        # 4. Integer casting for model compatibility
        assert clean_df.iloc[0]['actual_label'] == 1
        assert isinstance(clean_df.iloc[0]['actual_label'], (int, np.integer))


class TestInfrastructureManager:
    """Tests for the MAS Architecture blueprint and model input standardization."""

    def test_extract_model_input_torch_compatibility(self):
        """
        Verifies conversion of inputs to 24-feat NumPy.
        Uses PyTorch to verify the output meets neural pillar standards.
        """
        manager = InfrastructureManager()

        # Create an over-sized input (30 columns)
        data = np.random.rand(10, 30).astype(np.float64)
        df_input = pd.DataFrame(data)

        # Standardize via manager
        processed = manager.extract_model_input(df_input)

        # 1. Verify Slicing and Type
        assert processed.shape == (10, 24)
        assert processed.dtype == np.float32

        # 2. PyTorch Integration: Ensure result is a valid Torch tensor for Neuro Pillar
        torch_input = torch.from_numpy(processed)
        # Verify precision casting using Torch
        torch_input_cast = torch_input.to(torch.float32)

        assert torch_input_cast.shape == (10, 24)
        assert torch_input_cast.dtype == torch.float32

    @patch("os.path.exists")
    def test_path_resolution_integrity(self, mock_exists):
        """Verifies that the manager resolves paths to all core MAS pillars."""
        mock_exists.return_value = True
        manager = InfrastructureManager()

        # Ensure critical paths point to the 'models' directory defined in init
        assert "models" in manager.get_gold_model_path()
        assert "gold_champion.pkl" in manager.get_gold_model_path()

        assert "champion_vae.pth" in manager.get_neuro_model_path()
        assert "champion_dbscan.joblib" in manager.get_cluster_model_path()
        assert "champion_rnn_ae.pth" in manager.get_rnn_model_path()

    def test_registry_feature_loading(self):
        """Verifies feature name extraction from the champion registry."""
        manager = InfrastructureManager()
        # Mocking the registry content
        manager.registry = {
            'features_used': [f'feat_{i}' for i in range(50)]
        }

        features = manager.get_features()

        # Should be strictly capped at 24 features (input_dim)
        assert len(features) == 24
        assert features[0] == 'feat_0'
        assert features[-1] == 'feat_23'