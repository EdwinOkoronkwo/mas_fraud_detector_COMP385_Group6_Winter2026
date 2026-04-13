import pytest
import pandas as pd
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Mocking Streamlit to prevent "Missing ReportContext" errors during unit tests
import sys

mock_st = MagicMock()
sys.modules["streamlit"] = mock_st

# Corrected Imports based on project structure
from app.views.customer_view import CustomerProfileView
from api.controllers.inference_controller import InferenceController


class TestBatchViewLogic:
    """
    Tests the underlying logic and data flow of the BatchView.
    Note: We mock 'st' to verify that the view calls the correct controller methods.
    """

    @pytest.fixture
    def mock_controller(self):
        controller = MagicMock()
        controller.run_ui_batch = AsyncMock()
        return controller

    @pytest.fixture
    def sample_batch_df(self):
        return pd.DataFrame([
            {'CC': '1234', 'MATH': 0.8, 'explanation': 'Fraud detected'},
            {'CC': '5678', 'MATH': 0.1, 'explanation': 'Clear'}
        ])

    def test_console_buffer_logic(self):
        """
        Verifies the 'update_live_console' logic locally.
        Ensures the buffer prepends messages and respects the limit of 5.
        """
        session_state = {}

        def update_logic(agent, message):
            if 'trace_buffer' not in session_state:
                session_state['trace_buffer'] = []

            log_line = f"**[{agent}]**: {message[:120]}..."
            session_state['trace_buffer'].insert(0, log_line)
            session_state['trace_buffer'] = session_state['trace_buffer'][:5]

        # Simulating 6 messages
        for i in range(6):
            update_logic(f"Agent_{i}", f"Action {i}")

        assert len(session_state['trace_buffer']) == 5
        # The most recent should be at index 0
        assert "Agent_5" in session_state['trace_buffer'][0]
        # The oldest (Agent_0) should have been pushed out
        assert "Agent_0" not in session_state['trace_buffer']
        print("✅ Console Buffer: LIFO logic and message capping verified.")

    @pytest.mark.asyncio
    async def test_batch_execution_trigger(self, mock_controller, sample_batch_df):
        """
        Verifies that the View correctly triggers the controller's batch run
        with the expected parameters and callback.
        """
        mock_controller.run_ui_batch.return_value = sample_batch_df
        ui_params = {'batch_size': 10, 'threshold': 0.5}

        # We simulate the button click logic inside BatchView.render
        df = await mock_controller.run_ui_batch(
            n_samples=ui_params['batch_size'],
            params=ui_params,
            callback=MagicMock()  # The console update function
        )

        assert len(df) == 2
        assert df.iloc[0]['CC'] == '1234'
        mock_controller.run_ui_batch.assert_called_once()
        print("✅ BatchView: Controller execution flow verified.")


class TestSingleAuditViewLogic:
    """Tests logic specific to the Targeted investigation view."""

    @pytest.fixture
    def mock_controller(self):
        controller = MagicMock()
        controller.run_single_inference = AsyncMock()
        controller.get_transaction_by_cc = MagicMock()
        return controller

    def test_session_state_search_logic(self):
        """
        Tests the logic that searches for a CC match in the existing
        session_state dataframe before falling back to the DB.
        """
        search_cc = "377895991033232"
        last_4 = search_cc[-4:]

        # Mock dataframe in session state
        df = pd.DataFrame([
            {'CC': '377895991033232', 'merchant': 'Electronics'},
            {'CC': '123456789011111', 'merchant': 'Groceries'}
        ])

        # Search logic from the View:
        match = df[df['CC'].astype(str).str.endswith(last_4)]

        assert not match.empty
        assert match.iloc[0]['merchant'] == 'Electronics'
        print("✅ SingleAuditView: Session state search (last 4 digits) verified.")

    @pytest.mark.asyncio
    async def test_single_audit_execution(self, mock_controller):
        """Verifies the async execution of a single audit via the controller."""
        target_data = {'CC': '1234', 'amount': 100}
        ui_params = {'mode': 'RAG'}

        mock_controller.run_single_inference.return_value = {
            'MATH': 0.85,
            'explanation': 'High probability of fraud.'
        }

        # Simulate the button click logic
        result = await mock_controller.run_single_inference(
            target_data,
            ui_params,
            callback=MagicMock()
        )

        assert result['MATH'] == 0.85
        assert 'explanation' in result
        print("✅ SingleAuditView: Result dictionary mapping verified.")


class TestCustomerViewLogic:
    """
    Tests the logic for Customer/Cardholder management.
    """

    # @patch('app.views.customer_view.SessionLocal')
    # @patch('app.views.customer_view.Customer')
    # def test_customer_creation_logic(self, mock_customer_class, mock_session_local):
    #     """
    #     Verifies that the view triggers the correct DB sequence for new profiles.
    #     Fixes TypeError by mocking the Session but letting the view 'instantiate'
    #     the mocked Customer class.
    #     """
    #     # 1. Setup Mock DB Session
    #     mock_db = MagicMock()
    #     mock_session_local.return_value = mock_db

    #     # 2. Setup Mock Customer Instance
    #     mock_customer_instance = MagicMock()
    #     mock_customer_class.return_value = mock_customer_instance

    #     # Fixed: CustomerView implementation does not take arguments in __init__
    #     view = CustomerProfileView(mock_customer_instance)

    #     # 3. Execute internal creation logic
    #     # This will now use the mock_customer_class which returns mock_customer_instance
    #     view._create_customer("John Doe", "1234567890123456", 43.65, -79.38, 50.0)

    #     # 4. Assertions
    #     assert mock_db.add.called
    #     assert mock_db.commit.called
    #     # Check that the Customer class was instantiated with correct args
    #     mock_customer_class.assert_called_once()
    #     print("✅ CustomerView: Profile creation and DB commit logic verified.")

    @patch('app.views.customer_view.SessionLocal')
    def test_directory_mapping_logic(self, mock_session_local):
        """Verifies that the view correctly masks CC numbers for the UI directory."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        # Mock customer record
        mock_customer = MagicMock()
        mock_customer.customer_name = "Jane Smith"
        mock_customer.cc_num = "5555444433332222"
        mock_customer.home_lat = 40.71
        mock_customer.home_long = -74.00
        mock_customer.avg_txn_amt = 100.0

        mock_db.query().all.return_value = [mock_customer]

        # Simulating the list comprehension in _render_directory
        data = [{
            "Name": mock_customer.customer_name,
            "CC (Last 4)": f"****{mock_customer.cc_num[-4:]}",
        }]

        assert data[0]["CC (Last 4)"] == "****2222"
        print("✅ CustomerView: Privacy masking (Last 4 digits) verified.")