from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from inference.tools.preprocessing_tools import scale_transaction_data


class PreprocessingAgent:
    def __init__(self, model_client, infra_manager):
        self.model_client = model_client
        self.infra_manager = infra_manager

        # We wrap the tool so it automatically receives the infra_manager
        def scaling_tool(features_input: dict) -> dict:
            return scale_transaction_data(features_input, self.infra_manager)

        self.agent = AssistantAgent(
            name="Data_Preprocessor",
            model_client=model_client,
            tools=[scaling_tool], # Use the wrapper here
            system_message="""
            You are the Data Integrity Specialist.
            MISSION: Scale raw JSON data into the standardized numerical format verified by the Infrastructure Manager.

            REQUIRED ACTION:
            1. Call 'scaling_tool' with the raw JSON transaction provided by the Lead Investigator.
            2. When the tool returns the dictionary, output the key-value pairs clearly.
            3. Explicitly state: "Data is scaled. Inference_Specialist, please proceed with the model analysis." 
            """
        )