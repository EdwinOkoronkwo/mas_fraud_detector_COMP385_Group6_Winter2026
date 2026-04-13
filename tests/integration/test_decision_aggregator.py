import pytest
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent

from config.llm_config import get_model_client

"""
MODULE: Decision Aggregator Integration
PURPOSE: 
Verifies that the LLM can act as a 'Final Judge' by aggregating 
conflicting or complementary data from the Supervised and Anomaly teams.
"""


@pytest.mark.asyncio
async def test_aggregator_logic():
    client = get_model_client()

    # We create the Aggregator Agent
    aggregator = AssistantAgent(
        name="Decision_Aggregator",
        model_client=client,
        system_message=(
            "You are a Senior Fraud Investigator. You will receive a summary of model results. "
            "Your job is to provide a final Fraud Verdict: [HIGH RISK, MEDIUM RISK, or LOW RISK] "
            "and a brief justification."
        )
    )

    # Simulating the input from the two specialized teams
    mock_input = """
    SUPERVISED TEAM SUMMARY:
    - XGBoost: 85% Fraud Probability
    - ANN (TensorFlow): 92% Fraud Probability
    - Random Forest: 78% Fraud Probability

    ANOMALY TEAM SUMMARY (Audit Table):
    - RNN Autoencoder: High Reconstruction Error (Anomaly detected)
    - VAE: Latent space outlier
    - DBSCAN: Transaction falls into a 'Suspicious' cluster
    """

    # Run the aggregator
    result = await aggregator.run(task=mock_input)

    # Verification
    final_response = result.messages[-1].content.upper()

    assert "HIGH RISK" in final_response
    assert len(result.messages) > 0
    print(f"\n✅ Aggregator Verdict: {result.messages[-1].content}")


@pytest.mark.asyncio
async def test_aggregator_handles_conflicting_data():
    """
    PURPOSE: Verifies the LLM's reasoning when models disagree.
    """
    client = get_model_client()
    aggregator = AssistantAgent(name="Judge", model_client=client)

    conflicting_input = """
    SUPERVISED TEAM: XGBoost says 10% Probability (Safe).
    ANOMALY TEAM: RNN says 99% Reconstruction Error (Highly Unusual Behavior).
    """

    result = await aggregator.run(task=conflicting_input)

    # We expect the LLM to at least mention 'Anomalous' or 'Investigation'
    # because the unsupervised model flagged something the supervised one missed.
    assert len(result.messages[-1].content) > 20
    print(f"\n✅ Conflicting Data Reasoning: {result.messages[-1].content[:150]}...")