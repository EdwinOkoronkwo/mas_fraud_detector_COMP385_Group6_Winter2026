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
            import os  # Ensure OS is inside the scope
            import sqlite3

            # 1. Ensure the directory exists
            plot_dir = self.config.get("PLOT_DIR", "reports/plots")
            os.makedirs(plot_dir, exist_ok=True)

            # 2. Call the execution logic
            # Ensure the underlying tool handles the 'suffix' internally
            # or pass it here if execute_comprehensive_eda requires it.
            return execute_comprehensive_eda(
                db_path=self.config.get("DB_PATH"),
                output_dir=plot_dir,
                suffix="Initial"  # 🟢 ADDED: This fixes the 'missing 1 required positional argument' error
            )

        self.agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            tools=[run_visual_analysis],
            system_message=f"""You are the {self.role}. 

            CRITICAL: When calling 'run_visual_analysis', the system will now 
            automatically provide the 'Initial' suffix. 

            Your goal is to successfully generate the plots. If the tool returns 
            a path, verify the files exist and signal success."""
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
