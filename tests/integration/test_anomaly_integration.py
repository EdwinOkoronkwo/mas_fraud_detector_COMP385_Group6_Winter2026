from unittest.mock import MagicMock

import pytest
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage

from strategies.anomaly.anomaly_team import create_anomaly_team

"""
MODULE: Anomaly Team Integration Testing
PURPOSE: 
Verifies the Unsupervised Learning Pipeline: RNN -> VAE -> KMeans -> DBSCAN.
This selector uses 'History Detection' to ensure that even if an agent 
speaks twice, the sequence only moves forward once specific model 
results are logged in the chat history.
"""

import pytest
from unittest.mock import MagicMock
from autogen_agentchat.messages import TextMessage


"""
MODULE: Anomaly Team Integration (Extraction Method)
PURPOSE: 
Because the 'anomaly_selector' is a closure inside 'create_anomaly_team', 
we cannot import it directly. This test initializes a mock team to 
extract the internal logic and verify the unsupervised state machine.
"""

import pytest
from unittest.mock import MagicMock
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent
from autogen_core import FunctionCall
from autogen_agentchat.agents import AssistantAgent



class TestAnomalyIntegration:

    @pytest.fixture
    def extracted_selector(self):
        """
        Fixture to extract the nested selector function.
        """
        mock_client = MagicMock()
        d1 = AssistantAgent(name="G1", model_client=mock_client)
        d2 = AssistantAgent(name="G2", model_client=mock_client)
        team = create_anomaly_team(agents_list=[d1, d2], model_client=mock_client)
        return team._selector_func

    def test_priority_one_tool_retention(self, extracted_selector):
        """
        FIX: Use a MagicMock to bypass Pydantic's strict 'no-extra-fields' rule.
        This allows us to provide exactly what the selector is looking for.
        """
        # Create a mock message that simulates the source and the tool_calls attribute
        mock_msg = MagicMock()
        mock_msg.source = "RNN_Agent"
        mock_msg.tool_calls = [{"name": "train_rnn", "args": "{}"}]

        # Ensure hasattr(mock_msg, 'tool_calls') returns True
        assert hasattr(mock_msg, 'tool_calls')

        # The selector should see the tool_calls and stay on RNN_Agent
        assert extracted_selector([mock_msg]) == "RNN_Agent"

    def test_unsupervised_relay_sequence(self, extracted_selector):
        """Verify: History-based state machine moves from RNN to VAE."""
        msgs = [TextMessage(content="Start", source="Planner")]
        assert extracted_selector(msgs) == "RNN_Agent"

        msgs.append(TextMessage(content="rnn_autoencoder complete", source="RNN_Agent"))
        assert extracted_selector(msgs) == "VAE_Agent"

    def test_dbscan_to_critic_transition(self, extracted_selector):
        """Verify: Moves to Critic only after the full model history is present."""
        history = "rnn_autoencoder, variational_ae, k-means, dbscan"
        msgs = [TextMessage(content=history, source="DBSCAN_Agent")]
        assert extracted_selector(msgs) == "Anomaly_Critic"

    def test_anomaly_termination(self, extracted_selector):
        """Verify: Returns None to terminate once 'audit table' is in history."""
        # History must contain all models + the specific termination keyword
        history = "rnn_autoencoder variational_ae k-means dbscan"
        msgs = [
            TextMessage(content=history, source="DBSCAN_Agent"),
            TextMessage(content="Here is the final audit table.", source="Anomaly_Critic")
        ]
        assert extracted_selector(msgs) is None