import pytest
import pandas as pd
import numpy as np
import os

from tools.data_prep.preprocess_tools import drop_irrelevant_features, handle_categorical_encoding


class TestDataTools:

    @pytest.fixture
    def sample_raw_df(self):
        """Must have TWO rows to prevent drop_first=True from deleting the columns."""
        data = [
            {
                "category": "misc_net", "gender": "F", "amt": 4.97,
                "cc_num": 123, "first": "A", "last": "B", "street": "C"
            },
            {
                "category": "grocery_pos", "gender": "M", "amt": 50.0,
                "cc_num": 456, "first": "D", "last": "E", "street": "F"
            }
        ]
        return pd.DataFrame(data)

    def test_feature_dropping_logic(self, sample_raw_df):
        processed_df = drop_irrelevant_features(sample_raw_df)
        forbidden = ['cc_num', 'first', 'last', 'street']
        for col in forbidden:
            assert col not in processed_df.columns
        assert "amt" in processed_df.columns

    def test_categorical_encoding_to_int(self, sample_raw_df):
        # DEBUG: Ensure the fixture is actually providing 'category' and 'gender'
        assert 'category' in sample_raw_df.columns
        assert 'gender' in sample_raw_df.columns

        encoded_df = handle_categorical_encoding(sample_raw_df)

        # Look for the new columns created by pd.get_dummies
        new_cols = [c for c in encoded_df.columns if 'category_' in c or 'gender_' in c]

        assert len(new_cols) > 0, f"Encoding failed. Output columns: {encoded_df.columns.tolist()}"

        for col in new_cols:
            assert np.issubdtype(encoded_df[col].dtype, np.number)

    def test_preprocessing_tool_persistence(self):
        # Check for 'models' directory existence
        assert os.path.exists("models"), "Models directory is missing"