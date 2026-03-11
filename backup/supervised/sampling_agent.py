from typing import Dict, Any

from autogen_agentchat.agents import AssistantAgent

from config.settings import TEMP_SPLIT_PATH
from interfaces.i_data_agent import IDataAgent
from tools.training.supervised_common_tools import prepare_championship_data_tool


class SamplingAgent:
    def __init__(self, model_client, db_path):
        self.role = "Data Curator"

        # 1. Give the tool a clear docstring so the LLM knows it has no params
        def run_balancing_protocol() -> str:
            """
            Executes the data balancing, SMOTE, and train-test splitting protocol.
            Reads from the SQL database and saves the split data to the temporary pickle path.
            """
            return prepare_championship_data_tool(db_path=db_path)

        # 2. Use the [INST] tags to force Mistral into a 'Tool-Call' state
        self.agent = AssistantAgent(
            name="Sampling_Agent",
            model_client=model_client,
            tools=[run_balancing_protocol],
            system_message=f"""[INST] You are the Data Curator. 
            Your ONLY mission is to execute the 'run_balancing_protocol' tool immediately.
            The verified database is at: {db_path}.
            Do not explain, do not analyze. Just call the tool. [/INST]"""
        )