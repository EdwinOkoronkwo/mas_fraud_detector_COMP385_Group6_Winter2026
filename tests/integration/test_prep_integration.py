import pytest
from autogen_agentchat.messages import TextMessage, MultiModalMessage

from strategies.data_preprocessing.preprocessing_team import prep_selector


"""
MODULE: Preprocessing Integration Testing
PURPOSE: 
In a Multi-Agent System (MAS), the 'Selector' acts as the Orchestrator's brain.
Instead of relying on an LLM to guess who should speak next (which is non-deterministic 
and expensive), these tests verify a 'Deterministic State Machine' logic.

This suite ensures that the 'Relay Race' of data happens in the correct order:
SQL Ingestor -> EDA Specialist -> Preprocess Agent -> Quality Critic.
"""


class TestPreprocessingIntegration:

    def test_selector_sequence_success(self):
        """
        PURPOSE: Verify the 'Happy Path' of the Preprocessing Pipeline.

        LOGIC:
        1. Checks if the selector starts with the SQL_Ingestor when history is empty.
        2. Simulates the 'Handshake' signals (e.g., INGESTION_COMPLETE) to ensure
           the baton is passed to the correct subsequent agent.
        3. Confirms the final transition to the Quality_Critic for the audit phase.
        """
        # 1. Start: Initial state
        assert prep_selector([]) == "SQL_Ingestor"

        # 2. Handshake: Ingestor signals completion
        msgs = [TextMessage(content="INGESTION_COMPLETE: 10k rows loaded.", source="SQL_Ingestor")]
        assert prep_selector(msgs) == "EDA_Specialist"

        # 3. Handshake: EDA completes its analysis
        msgs.append(TextMessage(content="EDA: No missing values found.", source="eda_specialist"))
        assert prep_selector(msgs) == "Preprocess_Agent"

        # 4. Handshake: Preprocessor creates the .joblib artifact
        msgs.append(TextMessage(content="PREPROCESS_SUCCESS: Data ready for XGB.", source="Preprocess_Agent"))
        assert prep_selector(msgs) == "Quality_Critic"

    def test_selector_stays_on_failure(self):
        """
        PURPOSE: Verify Pipeline Resilience and Error Handling.

        LOGIC:
        In an autonomous system, we must prevent 'failing forward.'
        If an agent reports an error or fails to provide the mandatory
        'Success Signal', the selector must NOT move to the next stage.
        This test ensures the pipeline halts or retries the current step
        until the requirements are met.
        """
        # Simulating a failure message from SQL Ingestor
        msgs = [TextMessage(content="CRITICAL ERROR: Connection to Snowflake failed.", source="SQL_Ingestor")]

        # The selector should recognize the missing 'INGESTION_COMPLETE' signal
        # and stay on the current agent to allow for retry logic.
        assert prep_selector(msgs) == "SQL_Ingestor"

    def test_selector_handles_mixed_case_and_lists(self):
        """
        PURPOSE: Verify Data Robustness and Multi-Modal Compatibility.

        LOGIC:
        AutoGen v0.4+ agents can send 'MultiModal' content (lists of strings/images).
        Standard string parsing would crash if it encountered a list instead of a string.

        This test verifies that:
        1. The selector correctly iterates through list-based content.
        2. The selector is case-insensitive (handles 'ingestion_complete' vs 'INGESTION_COMPLETE').
        3. The system is protected against Pydantic validation crashes during message passing.
        """
        # Simulating a complex MultiModal message containing the success signal in a list
        msgs = [
            MultiModalMessage(
                content=["ingestion_complete", "Metadata: csv_source"],
                source="SQL_Ingestor"
            )
        ]

        # Verification that the code successfully 'hunts' for the signal inside the list
        assert prep_selector(msgs) == "EDA_Specialist"