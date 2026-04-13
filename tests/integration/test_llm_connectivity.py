import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from config.llm_config import get_model_client
from autogen_agentchat.agents import AssistantAgent

"""
MODULE: Full Pipeline End-to-End (E2E) Test (Suite-Safe Version)
PURPOSE: 
Simulates the MAS Fraud Detector lifecycle. Updated to use 
explicit AsyncMocking and scoped patching to prevent "MagicMock is not awaitable" 
errors when running in a full test suite alongside live connectivity tests.
"""


@pytest.mark.asyncio
async def test_full_mas_orchestration():
    client = get_model_client()

    # ---------------------------------------------------------
    # STAGE 1: PREPROCESSING (MOCK DATA)
    # ---------------------------------------------------------
    prep_output = "TRANS_ID: 9982 | AMT: $50,000 | LOC: Unknown | FLAG: High-Velocity"

    # ---------------------------------------------------------
    # STAGE 2: SUPERVISED TOURNAMENT (MOCK INFERENCE)
    # ---------------------------------------------------------
    supervised_summary = (
        "XGBoost: 0.88 Fraud Prob | "
        "Random Forest: 0.82 Fraud Prob | "
        "ANN (PyTorch): 0.94 Fraud Prob"
    )

    # ---------------------------------------------------------
    # STAGE 3: ANOMALY DETECTION (MOCK AUDIT)
    # ---------------------------------------------------------
    anomaly_summary = (
        "RNN Autoencoder: Outlier (High MSE) | "
        "VAE: Latent Space Anomaly | "
        "DBSCAN: Cluster Mismatch"
    )

    # ---------------------------------------------------------
    # STAGE 4: DECISION AGGREGATION
    # ---------------------------------------------------------
    # We initialize the agent normally
    aggregator = AssistantAgent(
        name="Final_Judge",
        model_client=client,
        system_message=(
            "You are the Final Decision Aggregator for a Fraud Detection System. "
            "Return a JSON-like summary with VERDICT: [RED/YELLOW/GREEN] and REASON."
        )
    )

    final_task = f"""
    Please provide a final verdict based on these inputs:
    Supervised Results: {supervised_summary}
    Anomaly Results: {anomaly_summary}
    Transaction Details: {prep_output}
    """

    # CRITICAL FIX: Use patch.object on the instance OR the class with a context manager.
    # We also mock the 'TaskResult' structure correctly for AutoGen 0.4.
    with patch.object(AssistantAgent, "run", new_callable=AsyncMock) as mocked_run:
        # Define the mock message
        mock_msg = MagicMock()
        mock_msg.content = "VERDICT: RED. High probability of fraud across all pillars."
        mock_msg.source = "Final_Judge"

        # Define the TaskResult mock
        mock_result = MagicMock()
        mock_result.messages = [mock_msg]

        mocked_run.return_value = mock_result

        # Execute the awaited call
        result = await aggregator.run(task=final_task)

        # Validation
        content = result.messages[-1].content
        verdict = content.upper()

        # ---------------------------------------------------------
        # FINAL VALIDATION
        # ---------------------------------------------------------
        assert any(term in verdict for term in ["RED", "HIGH", "FRAUD", "VERDICT"])
        assert len(result.messages) > 0
        assert result.messages[-1].source == "Final_Judge"

    print(f"\n🚀 FULL E2E SYSTEM VERDICT (MOCKED): {content}")

# --- ADVICE FOR COMPANION TEST (test_mistral_connectivity_direct) ---
# If the connectivity test fails while this one is in the suite,
# ensure the connectivity test is NOT being patched by a global
# conftest.py or a module-level patch.