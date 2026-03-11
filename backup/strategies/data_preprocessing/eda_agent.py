from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from mas_fraud_detector.interfaces.i_data_agent import IDataAgent
from mas_fraud_detector.tools.data_prep.eda_tools import generate_eda_plots_tool

# class EDAAgent(IDataAgent):
#     def __init__(self, config: Dict[str, Any], model_client: Any):
#         super().__init__(name="EDA_Specialist", role="Data Analyst", config=config)
#
#         # We wrap the tool so we can inject the DB_PATH from the config
#         def run_analysis() -> str:
#             return generate_eda_plots_tool(db_path=self.config.get("DB_PATH"))
#
#         self.agent = AssistantAgent(
#             name=self.name,
#             model_client=model_client,
#             tools=[run_analysis],
#             system_message=f"""You are the {self.role}.
#             Your goal is to identify patterns in the SQL data.
#             You are the EDA Specialist. Call 'run_eda_tool' immediately.
#             Your output must ONLY be the tool execution command. Once you receive the tool result,
#             summarize the fraud correlations briefly.
#             1. Run 'run_analysis'.
#             2. Report the fraud imbalance and top correlations.
#             3. Comment on the 'amt' statistics (outliers)."""
#         )

class EDAAgent(IDataAgent):
    def __init__(self, config: Dict[str, Any], model_client: Any):
        super().__init__(name="EDA_Specialist", role="Data Analyst", config=config)

        def run_analysis() -> str:
            return generate_eda_plots_tool(db_path=self.config.get("DB_PATH"))

        self.agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            tools=[run_analysis],
            system_message=f"""You are the {self.role}. 
            Your goal is to provide the 'Mathematical Grounding' for the Training Agents.

            1. Run 'run_analysis'.
            2. Report the EXACT fraud imbalance ratio (e.g., 1:175). 
               The Isolation Forest and VAE Agents will use this to set their anomaly thresholds.
            3. Highlight the variance in 'amt'. 
               The RNN_Agent needs this to decide if it should use Huber Loss or MSE.

            Summarize findings into a 'Data Profile' for the next agents."""
        )
    async def run(self, state: Any = None) -> Dict[str, Any]:
        """
        Implementation of the abstract method from IDataAgent.
        In a group chat, the team orchestrator handles the 'run', 
        but the class still needs this defined to be instantiable.
        """
        return {"status": "active", "agent": self.name}
