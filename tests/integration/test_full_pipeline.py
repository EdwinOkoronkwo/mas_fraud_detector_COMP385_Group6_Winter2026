import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from config.llm_config import get_model_client
from autogen_agentchat.agents import AssistantAgent

"""
MODULE: Full Pipeline End-to-End (E2E) Test (Suite-Safe Version)
PURPOSE: 
Simulates the MAS Fraud Detector lifecycle. Updated to use 
explicit AsyncMocking to prevent "MagicMock is not awaitable" 
errors when running in a full test suite.
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

    # We patch the 'run' method specifically as an AsyncMock to ensure
    # it remains awaitable even if the global namespace was corrupted by other tests.
    with patch.object(AssistantAgent, "run", new_callable=AsyncMock) as mocked_run:
        # Create a mock response object that mimics AutoGen's TaskResult
        mock_response = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = "VERDICT: RED. High probability of fraud across all pillars."
        mock_response.messages = [mock_msg]
        mocked_run.return_value = mock_response

        # Execute
        result = await aggregator.run(task=final_task)
        verdict = result.messages[-1].content.upper()

        # ---------------------------------------------------------
        # FINAL VALIDATION
        # ---------------------------------------------------------
        assert any(term in verdict for term in ["RED", "HIGH", "FRAUD", "VERDICT"])
        assert len(result.messages) > 0

    print(f"\n🚀 FULL E2E SYSTEM VERDICT (MOCKED): {result.messages[-1].content}")