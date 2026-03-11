from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool


from autogen_core.tools import FunctionTool

from inference.tools.inference_engine import InferenceEngine
from rag.tools.rag_tools import  execute_champion_ensemble


from langchain_core.tools import tool as langchain_tool
from autogen_ext.tools.langchain import LangChainToolAdapter

from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent


class InferenceAgent:
    def __init__(self, model_client, engine):
        self.engine = engine

        # 1. Update description to emphasize it's the ONLY way to get results
        inference_tool = FunctionTool(
            self.engine.execute_supervised_inference,
            name="execute_supervised_inference",
            description="MANDATORY: The only tool to calculate fraud probability. Must be called as soon as scaled data is available."
        )

        self.agent = AssistantAgent(
            name="Inference_Specialist",
            model_client=model_client,
            tools=[inference_tool],
            system_message="""You are the Gold Tier Inference Specialist. 

            TRIGGER: As soon as you see a dictionary from the Data_Preprocessor, you MUST call 'execute_supervised_inference'.

            STRICT RULES:
            - DO NOT wait for permission from the Lead_Investigator.
            - DO NOT summarize the data. 
            - DO NOT guess the risk.
            - YOUR ONLY VALID OUTPUT is calling the 'execute_supervised_inference' tool.

            If you provide a text response without calling the tool, the pipeline fails."""
        )