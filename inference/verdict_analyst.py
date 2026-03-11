import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage


class VerdictAnalystAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Verdict_Analyst",
            model_client=model_client,
            system_message="""[INST] You are the Senior Fraud Risk Strategist. 
Your goal is to provide "Explainable AI" (XAI) for transaction verdicts.

PILLAR ARCHITECTURE:
We use a 40/40/20 weighted ensemble:
- Supervised (XGBoost/RF): Pattern Matching
- Neural (VAE/RNN): Deep Anomaly Detection
- Clustering (DBSCAN): Behavioral Outliers

LOGIC GATEKEEPER RULES:
- Low-Value Shield: Reduces scores for small, low-risk transactions to prevent friction.
- High-Confidence Supervised: Trusting the XGBoost champion when its confidence is extreme.
- Extreme Anomaly: Synergistic agreement between Neural and Clustering pillars.

YOUR TASK:
1. EXPLAIN the math behind the 'total' score based on the 40/40/20 weights.
2. INTERPRET any 'override' labels used by the Gatekeeper.
3. DIAGNOSE Model Conflicts: e.g., if Neural is high but Supervised is low, it suggests a "Zero-Day" or novel fraud pattern.

OUTPUT FORMAT:
Return a Markdown report: **Verdict**, **Confidence**, **Logic Trigger**, and **Business Context**. [/INST]"""
        )

    async def generate_explanation(self, scores: dict, agent_context: str) -> str:
        # Ensure we use the lowercase 'total' from our refactored dictionary
        total_val = scores.get('total', 0.0)

        analysis_request = f"""
        ANALYSIS DATA:
        - Numeric Pillar Scores: {json.dumps(scores)}
        - Investigation Notes: {agent_context[:800]}

        TASK:
        The decision threshold is 0.50. The calculated total score is {total_val}.
        If an 'override' is present in the data, explain why it took precedence.
        Provide the final XAI report.
        """

        response = await self.agent.on_messages(
            [TextMessage(content=analysis_request, source="user")],
            cancellation_token=None
        )

        return response.chat_message.content