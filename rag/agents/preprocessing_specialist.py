from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from rag.tools.rag_tools import scale_transaction_data


class PreprocessingAgent:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Data_Preprocessor",
            model_client=model_client,
            tools=[scale_transaction_data],
            system_message="""You are the Data Integrity specialist. 

            TASK:
            1. Receive raw transaction details (amount, hour, distance).
            2. Call 'scale_data' to transform these into their mathematical scaled equivalents.
            3. Explicitly state the scaled values (s_amount, s_hour, s_dist) so the Inference_Specialist can use them.

            Do not interpret the risk. Just provide the clean data."""
        )