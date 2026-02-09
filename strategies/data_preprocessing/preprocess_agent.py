from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from interfaces.i_data_agent import IDataAgent
from tools.data_prep.preprocess_tools import apply_scaling_and_cleaning_tool


class PreprocessAgent(IDataAgent):
    def __init__(self, config: Dict[str, Any], model_client: Any):
        super().__init__(name="Preprocess_Agent", role="Feature Engineer", config=config)

        # Wrapped tool specifically for cleaning and scaling
        def run_cleaning_and_scaling() -> str:
            db_file = self.config.get("DB_PATH")
            return apply_scaling_and_cleaning_tool(db_path=db_file)

        self.agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            tools=[run_cleaning_and_scaling],
            system_message=f"""You are the {self.role}. 
            Your ONLY job is to execute 'run_cleaning_and_scaling'. 
        
        CRITICAL PIPELINE CHANGE: 
        1. You will use 'StandardScaler' to normalize features.
        2. You will NOT apply SMOTE here; balancing is now deferred to the Training Phase to prevent leakage.
        3. Once complete, confirm that the table 'cleaned_scaled_data' is ready for the Tournament.
        
        Do not talk. Execute the tool and report the SUCCESS/ERROR status."""
        )

    async def run(self, state: Any = None) -> Dict[str, Any]:
        return {"status": "success", "agent_name": self.name}