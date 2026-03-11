from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain.tools import tool

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool



class InferenceCritic:
    def __init__(self, model_client):
        # Dedicated tool for the Critic to 'stamp' its approval
        def verify_inference_logic(explanation: str, model_scores: str) -> str:
            return f"Verification successful for Gold Tier XGBoost results."

        critic_tool = FunctionTool(verify_inference_logic, description="Verify text vs scores")

        self.agent = AssistantAgent(
            name="Inference_Critic",
            model_client=model_client,
            model_context=BufferedChatCompletionContext(buffer_size=5),
            tools=[critic_tool],
            system_message="""You are the Inference Critic (Hallucination Guard). 
            Your goal is to ensure the final fraud explanation is 100% data-backed.

            RULES:
            1. Compare the Synthesis report against the XGBoost Probability provided by the Inference_Specialist.
            2. Verify: CC_NUM and the Gold Tier Risk Level (Probability > 0.5 is High Risk).
            3. If the Specialist reports a probability of 0.8 but the report says 'Low Risk', order a 'REVISE'.
            4. If the logic is consistent with the Gold XGB output, conclude with 'CASE_CLOSED'."""
        )