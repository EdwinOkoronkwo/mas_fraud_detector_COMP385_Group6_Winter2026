from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from interfaces.i_data_agent import IDataAgent
from interfaces.i_eda_agent import IEDAAgent
from tools.data_prep.eda_tools import generate_eda_plots_tool
from tools.data_prep.visualization_tools import execute_comprehensive_eda


class EDAAgent(IEDAAgent):
    def __init__(self, config: Dict[str, Any], model_client: Any):
        super().__init__(name="EDA_Specialist", role="Forensic Data Analyst", config=config)

        def run_visual_analysis() -> str:
            return execute_comprehensive_eda(
                db_path=self.config.get("DB_PATH"),
                output_dir=self.config.get("PLOT_DIR", "reports/plots")
            )

        self.agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            tools=[run_visual_analysis],
            system_message=f"""You are the {self.role}. 

                        OPERATING PROTOCOL:
                        1. Use 'run_visual_analysis' to inspect the post-processed data.
                        2. VERIFY that categorical strings (category, gender) have been successfully 
                           transformed into binary 'One-Hot' vectors.
                        3. AUDIT the numerical ranges: With Z-score scaling applied, most values 
                           for 'amt', 'lat', and 'long' should now be centered near 0.
                        4. IDENTIFY any 'Feature Explosion': Note how many new columns were created 
                           from the 'category' feature.
                        5. LOG the final row/column count for the Quality Critic's Phase 1 sign-off.

                        Goal: Ensure the dataset is mathematically compatible for supervised learning."""
        )
# class EDAAgent(IEDAAgent):
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
    async def run(self, state: Any = None) -> Dict[str, Any]:
        """
        Implementation of the abstract method from IDataAgent.
        In a group chat, the team orchestrator handles the 'run', 
        but the class still needs this defined to be instantiable.
        """
        return {"status": "active", "agent": self.name}
