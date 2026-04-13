
import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch

from api.controllers.inference_controller import InferenceController


class TestInferenceController:

    @pytest.fixture
    def mock_pipeline(self):
        """Mocks the InferencePipeline and its sub-components."""
        pipeline = MagicMock()
        pipeline.run_batch = AsyncMock()
        pipeline.handler = MagicMock()
        return pipeline

    @pytest.fixture
    def sample_results_df(self):
        """Provides a sample results DataFrame matching the internal schema."""
        return pd.DataFrame([{
            'CC': '1234',
            'ACT': 0,
            'GOLD': 0.1,
            'N_RAW': 0.05,
            'N_CAL': 0.02,
            'C_RAW': 0.8,
            'C_CAL': 0.9,
            'MATH': 0.45,
            'mode': 'RAG',
            'explanation': 'Test audit successful.'
        }])

    @pytest.mark.asyncio
    @patch('api.controllers.inference_controller.SessionLocal')
    @patch('api.controllers.inference_controller.InferenceResult')
    async def test_run_ui_batch_sync_logic(self, mock_result_model, mock_session_local, mock_pipeline,
                                           sample_results_df):
        """
        Verifies that batch runs correctly trigger database synchronization.
        Updated to target the specific module for patching to avoid attribute errors.
        """
        mock_pipeline.run_batch.return_value = sample_results_df

        # Mock DB Session
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        controller = InferenceController(mock_pipeline)
        callback = MagicMock()

        # Execute
        results = await controller.run_ui_batch(n_samples=1, callback=callback)

        # Assertions
        assert len(results) == 1
        assert mock_db.add.called
        assert mock_db.commit.called
        callback.assert_any_call("SYSTEM", "✅ Database Sync Complete.")
        print("✅ InferenceController: Batch run and DB sync verified.")

    @pytest.mark.asyncio
    async def test_run_single_inference_handling(self, mock_pipeline, sample_results_df):
        """Verifies the diagnostic 'Smart Handling' logic for different result types."""
        controller = InferenceController(mock_pipeline)
        tx_data = {"CC": "1234", "Amount": 100}

        # Scenario 1: Pipeline returns a DataFrame
        mock_pipeline.run_batch.return_value = sample_results_df
        res_df = await controller.run_single_inference(tx_data, params={})
        assert isinstance(res_df, dict)
        assert res_df['CC'] == '1234'

        # Scenario 2: Pipeline returns a raw dict (Fallback)
        mock_pipeline.run_batch.return_value = {"status": "success"}
        res_dict = await controller.run_single_inference(tx_data, params={})
        assert res_dict["status"] == "success"

        print("✅ InferenceController: Single inference result mapping verified.")

    def test_get_transaction_by_cc_bridge(self, mock_pipeline):
        """Verifies the bridge between the controller and the DataHandler."""
        controller = InferenceController(mock_pipeline)

        # Mock successful match
        mock_pipeline.handler.get_transaction_by_cc.return_value = pd.DataFrame([{"CC": "9999", "Vendor": "TestStore"}])

        result = controller.get_transaction_by_cc("9999")
        assert result["Vendor"] == "TestStore"

        # Mock no match
        mock_pipeline.handler.get_transaction_by_cc.return_value = pd.DataFrame()
        assert controller.get_transaction_by_cc("0000") is None

        print("✅ InferenceController: Transaction bridge logic verified.")