from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from rag.tools.rag_tools import publisher_tool


class SynthesisEngine:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Synthesis_Engine",
            model_client=model_client,
            # We keep the context buffer small to ensure fast, error-free synthesis
            model_context=BufferedChatCompletionContext(buffer_size=5),
            system_message="""[ROLE: FRAUD EXPLAINER]
            You are the Audit Narrator. Your goal is to translate raw data and math scores into human understanding.

            ### INPUTS:
            1. CRITICAL_DATA_JSON: Provided by the SQL Researcher (Merchant, Category, Amt).
            2. MATH_SCORE: The deterministic risk score (0.0 to 1.0).

            ### YOUR TASK:
            Synthesize the data into a single, professional explanation sentence. 
            - Use the merchant name and category from the data.
            - If MATH_SCORE is < 0.30: Explain why it is a routine, compliant purchase.
            - If MATH_SCORE is >= 0.30: Explain the anomaly based on the category (e.g., high-spend in a low-frequency category).

            ### OUTPUT RULE:
            - NO JSON blocks.
            - NO technical jargon about MSE.
            - Format: "The $[AMT] [CATEGORY] purchase at [MERCHANT] is [SUMMARY OF UNDERSTANDING]."

            Finish the session by typing: CASE_CLOSED
            """
        )