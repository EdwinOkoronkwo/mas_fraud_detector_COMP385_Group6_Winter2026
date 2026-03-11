from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from interfaces.i_data_agent import IDataAgent
from interfaces.i_ingestor_agent import IIngestorAgent
from tools.db_tools import migrate_csv_to_sql_tool, purge_existing_database


class SQLIngestorAgent(IIngestorAgent):
    def __init__(self, config: Dict[str, Any], model_client: Any):
        super().__init__(name="SQL_Ingestor", role="Data Migration Specialist", config=config)

        # Helper to wrap the migration tool with config context
        def run_migration() -> str:
            return migrate_csv_to_sql_tool(
                kaggle_path=self.config.get("KAGGLE_PATH"),
                db_path=self.config.get("DB_PATH"),
                sample_frac=0.1
            )

        self.agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            tools=[purge_existing_database, run_migration],
            system_message=f"""You are the {self.role}. 

        STRICT WORKFLOW:
        You MUST execute two tools in a single session.
        1. Call 'purge_existing_database' FIRST.
        2. IMMEDIATELY after the purge success, call 'run_migration'. 
           Do NOT wait for further instructions.

        REQUIRED OUTPUT:
        After 'run_migration' finishes, you must state: "INGESTION_COMPLETE".
        If you do not say "INGESTION_COMPLETE", the pipeline will hang."""
        )

    # --- THIS IS THE MISSING PIECE ---
    async def run(self, state: Any = None) -> Dict[str, Any]:
        """
        Implementation of the abstract method from IDataAgent.
        In the new architecture, this bridges the Orchestrator logic 
        to the AutoGen Agent logic.
        """
        self.log_info(f"Agent {self.name} is now active.")
        
        # For now, we return a status. In a full run, 
        # the Team/GroupChat handles the actual execution.
        return {"status": "success", "agent_name": self.name}


# class SQLIngestorAgent(IIngestorAgent):
#     def __init__(self, config: Dict[str, Any], model_client: Any):
#         super().__init__(name="SQL_Ingestor", role="Data Migration Specialist", config=config)
#
#         # Define the tool for the internal agent
#         def run_migration() -> str:
#             return migrate_csv_to_sql_tool(
#                 kaggle_path=self.config.get("KAGGLE_PATH"),
#                 db_path=self.config.get("DB_PATH"),
#                 sample_frac=0.1
#             )
#
#         self.agent = AssistantAgent(
#             name=self.name,
#             model_client=model_client,
#             tools=[run_migration],
#             system_message=f"""You are the {self.role}.
#                 Your ONLY task is to call 'migrate_csv_to_sql_tool'. Do not provide any introduction.
#                 Do not explain your actions. If the migration succeeds, report 'MIGRATION COMPLETE' and nothing else.
#                 Your task is to execute the 'run_migration' tool to move data to SQL.
#                 IMPORTANT: Once the tool finishes, you MUST state the exact row counts
#                 provided in the tool's output so the Quality Critic can verify them."""
#         )
