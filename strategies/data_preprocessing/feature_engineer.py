import os

import pandas as pd
from typing import Dict, List

import joblib
import pandas as pd
from typing import Dict, List

from tools.data_prep.feature_eng_tools import FraudToolbox
from utils.logger import setup_logger

tool_logger = setup_logger("Feature_Engineer")

import os
import joblib
import pandas as pd
from typing import Dict, Any, List
from autogen_agentchat.agents import AssistantAgent
from utils.logger import setup_logger


class FeatureEngineerAgent:
    def __init__(self, settings: Any, model_client: Any, preprocessor_path: str):
        # 1. Basic Setup
        self.settings = settings
        self.model_client = model_client
        self.preprocessor_path = preprocessor_path
        self.logger = setup_logger("Feature_Engineer")

        # 2. State for Inference (Phase 3)
        self.preprocessor = None
        self.expected_columns = []

        # 3. Soft-load Preprocessor (Safe for Phase 1)
        if os.path.exists(self.preprocessor_path):
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                self.logger.info("✅ Preprocessor loaded for Inference.")
            except Exception as e:
                self.logger.warning(f"⚠️ Preprocessor load failed: {e}")
        else:
            self.logger.info("🛠️ Training Mode: Preprocessor not found (expected in Phase 1).")

        # 4. Define the Tool for Phase 1 (The "Action")
        def run_feature_induction() -> str:
            """Induces behavioral features (ratios, temporal, velocity) into SQL."""
            db_path = getattr(self.settings, "DB_PATH", None)
            if not db_path:
                return "ERROR: No DB_PATH found in settings."

            # This calls the implementation logic using FraudToolbox
            return self._execute_induction(db_path)

        # 5. THE FIX: The AssistantAgent attribute the Orchestrator needs
        self.agent = AssistantAgent(
            name="Feature_Engineer",
            model_client=self.model_client,
            tools=[run_feature_induction],
            system_message="""You are the Feature Engineer.

            ### OPERATING PROTOCOL:
            1. THINK: Start with a 'THOUGHT:' block.
            2. REASON: Explain why creating behavioral features (Amount-to-Category ratios, Velocity) helps detect fraud.
            3. ACT: Call 'run_feature_induction' to update the SQL database.

            TASK: Add engineered features to the training table and signal completion."""
        )

    def _execute_induction(self, db_path: str) -> str:
        """Helper logic to perform the SQL updates using FraudToolbox."""
        from sqlalchemy import create_engine
        # Assuming FraudToolbox is imported or defined in scope
        try:
            engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
            df = pd.read_sql("SELECT * FROM train_transactions", engine)

            # Compute necessary stats
            cat_means = df.groupby('category')['amt'].mean().to_dict()

            # Apply Toolbox Methods
            df = FraudToolbox.compute_amt_to_cat_ratio(df, cat_means)
            df = FraudToolbox.extract_temporal_risk(df)
            df = FraudToolbox.calculate_velocity(df)

            # Persist back to SQL
            df.to_sql("train_transactions", engine, if_exists='replace', index=False)
            return "SUCCESS: Behavioral features induced. ENGINEERING_COMPLETE."
        except Exception as e:
            return f"ERROR: {str(e)}"

    def transform(self, raw_row: Dict) -> Dict:
        """Inference method (used in Phase 3, not by Orchestrator)."""
        if not self.preprocessor:
            raise RuntimeError("Preprocessor not loaded. Cannot transform.")

        df = pd.DataFrame([raw_row])
        try:
            processed_array = self.preprocessor.transform(df)
            return dict(zip(self.expected_columns, processed_array[0]))
        except Exception as e:
            self.logger.error(f"❌ Transformation failed: {e}")
            return {col: 0.0 for col in self.expected_columns}

    async def run(self, state: Any = None) -> Dict[str, Any]:
        """Standard run method to match the pipeline interface."""
        return {"status": "success", "agent_name": "Feature_Engineer"}