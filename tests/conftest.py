import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

import sys
import os

# This adds the project root to the Python path automatically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.fixture
def mock_transaction_data():
    """Provides a fake preprocessed vector for anomaly agents."""
    # Simulating a 22-feature vector typical of fraud detection models
    return np.random.rand(1, 22).astype(np.float32)

@pytest.fixture
def mock_llm_client():
    """Fakes the LLM response so we don't call Mistral/OpenAI during tests."""
    mock = MagicMock()
    mock.chat.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Mocked Agent Reasoning"))]
    )
    return mock

@pytest.fixture
def mock_db_session():
    """Fakes a database session to prevent writing to database.sqlite."""
    return MagicMock()