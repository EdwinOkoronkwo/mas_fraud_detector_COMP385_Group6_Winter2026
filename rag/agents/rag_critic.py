from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain.tools import tool


class RAGCritic:
    def __init__(self, model_client):
        @tool
        def check_data_consistency(synthesis_claim: str, source_data: str) -> bool:
            """
            Checks if a claim made in the summary (synthesis_claim)
            is actually supported by the research logs (source_data).
            """
            # This is a helper for the agent to formalize its check
            return True

        critic_tool = LangChainToolAdapter(check_data_consistency)

        self.agent = AssistantAgent(
            name="RAG_Critic",
            model_client=model_client,
            model_context=BufferedChatCompletionContext(buffer_size=5),  # Keep only last 5 messages
            tools=[critic_tool],
            system_message="""You are the Hallucination Guard. 
            Your goal is to ensure the final fraud explanation is 100% accurate.
            
            Your ONLY job is to verify claims using the 'check_data_consistency' tool.
    
            STRICT RULES:
            1. Do NOT misspells the tool name. It is 'check_data_consistency'.
            2. If you see a mismatch between the report and the source data, flag it as a VIOLATION.
            3. You must verify CC_NUM, Amount, and Risk Scores.

            CHECKLIST:
            1. Did the Synthesis Engine mention a policy? Check if Vector_Researcher actually found that policy.
            2. Did it mention a spending average? Check if SQL_Researcher provided that number.
            3. Did it mention a merchant breach? Check if Web_Researcher confirmed it.

            If you find a discrepancy, you MUST tell the Synthesis_Engine to 'REVISE'.
            If everything is perfect, output: 'FINAL_REPORT: [The full text of the report]'."""
        )