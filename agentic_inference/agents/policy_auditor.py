from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage


class PolicyAuditor:
    def __init__(self, model_client, vector_service):
        self.client = model_client
        self.retriever = vector_service.get_retriever()

        # We use a simple system message focused on narrative
        self.system_message = """
        You are a Financial Compliance Assistant. 
        Your task is to provide a brief, professional explanation of a transaction 
        based on provided policy documents and a mathematical risk score.

        RULES:
        1. Do not assign a new score.
        2. Do not use 'CRITICAL' unless the MATH_SCORE is high (> 0.50).
        3. If the score is low, explain why the transaction is compliant.
        4. Keep explanations to 2 sentences max.
        """

    async def get_explanation(self, tx_data, math_score):
        # 1. Retrieve the 'Why' from the Vector DB
        query = f"Policy rules for transaction: {tx_data}"
        docs = self.retriever.invoke(query)
        policy_context = "\n".join([d.page_content for d in docs])

        # 2. Build the Explanation Prompt
        prompt = (
            f"{self.system_message}\n\n"
            f"MATH_SCORE: {math_score}\n"
            f"POLICIES: {policy_context}\n"
            f"DATA: {tx_data}\n\n"
            "SUMMARY AND EXPLANATION:"
        )

        # Use a direct completion call to avoid AutoGen sequence errors
        response = await self.client.create_completion(prompt)
        return response.content