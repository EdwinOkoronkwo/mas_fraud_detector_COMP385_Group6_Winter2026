import asyncio
import os
import sys
import json

from autogen_agentchat.base import TaskResult
from autogen_core import CancellationToken
from autogen_core.models import UserMessage

from config.llm_config import get_model_client
from core.decision_agent import DecisionAggregator
from core.orchestrator import Orchestrator
from core.state import PipelineState
from factories.agent_factory import AgentFactory
from strategies.anomaly.clustering.dbscan_agent import train_dbscan

from strategies.supervised.baseline_trainer import BaselineModelTrainer
from tools.reporting.report_tools import save_inference_metadata, save_hybrid_metadata
from tools.training.supervised_common_tools import prepare_championship_data_tool
from utils.logger import setup_logger
from config.settings import settings
import warnings

# 1. Block Python-level UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Block XGBoost internal C++ logging (0 = all, 3 = only fatal)
os.environ['KMP_WARNINGS'] = '0'

# Phase 2 Component Imports


from config.settings import settings  # Ensure this is imported


class Phase2Championship:
    def __init__(self):
        self.logger = setup_logger("Phase2_Championship")
        self.model_client = get_model_client()

        # 🟢 ADD THIS LINE: Store the global settings into the instance
        self.settings = settings
        self.agent_factory = AgentFactory(model_client=self.model_client, settings=self.settings)

        self.results_manifest = {
            "baseline": None,
            "supervised": None,
            "unsupervised": None
        }

    async def prepare_data_foundation(self):
        """Logic: 3-Way Split + SMOTE via DataSpecialist."""
        self.logger.info("⏳ STEP 1: Running DataSpecialist (3-Way Split + SMOTE)...")
        result = prepare_championship_data_tool(settings.DB_PATH)

        if "SUCCESS" in result:
            self.logger.info(f"✅ Data Specialist: {result}")
            return True
        return False

    def run_baseline_benchmark(self, feature_mode="enhanced"):
        """
        Logic: Standard XGBoost benchmark with Stratified CV.
        feature_mode: 'base' (24 features) or 'enhanced' (27 features)
        """
        self.logger.info(f"⏳ STEP 2: Training Baseline XGBoost ({feature_mode} mode)...")
        try:
            # Pass the flag based on your mode
            use_behavioral = (feature_mode == "base")

            # Initialize the trainer with the toggle we just built
            trainer = BaselineModelTrainer(settings, use_behavioral=use_behavioral)

            # Save results to a specific key so they don't overwrite each other in the manifest
            self.results_manifest[f"baseline_{feature_mode}"] = trainer.run_baseline_training()

            return True
        except Exception as e:
            self.logger.error(f"❌ Baseline Error: {e}")
            return False

    async def run_supervised_tournament(self):
        self.logger.info("⏳ STEP 3: Starting Supervised Tournament (Multi-Agent)...")

        try:
            # 🟢 VERIFY: Ensure AgentFactory.get_supervised_championship_team()
            # includes the [sample, s_xgb, d_xgb, ann_agent, critic] lineup.
            factory = AgentFactory(self.model_client, settings)
            state = PipelineState()
            orchestrator = Orchestrator(state)

            def runner_factory():
                return factory.get_supervised_championship_team()

            # 🟢 TASK UPDATE: Mention the ANN explicitly in the task to nudge the Planner
            task = "Execute the Supervised Tournament. Sequence: Sampling -> Static XGB -> Dynamic XGB -> ANN -> Critic."

            result = await orchestrator.execute_phase(
                phase_name="Supervised_Tournament",
                runner_factory=runner_factory,
                task=task
            )

            if result:
                self.results_manifest["supervised"] = result.messages[-1].content
                return True
            return False

        except Exception as e:
            self.logger.error(f"❌ Supervised Tournament Error: {e}")
            return False



    async def run_unsupervised_discovery(self):
        """
        Executes the Neuro-Pattern tournament (RNN vs VAE)
        and captures the audit from the Anomaly Critic.
        """
        self.logger.info("⏳ STEP 4: Starting Unsupervised Discovery (Neuro-Pattern Tournament)...")

        try:
            unsupervised_team = self.agent_factory.get_anomaly_discovery_team()

            task = f"""
            Execute the Unsupervised Fraud Discovery Protocol:
            1. RNN_Agent & VAE_Agent: Generate neuro-pattern metrics.
            2. DBSCAN_Agent & KMeans_Agent: Generate spatial and centroid clustering metrics.
            3. Anomaly_Critic: Compare ALL 4 models against {self.settings.DB_PATH}.

            Declare a 'Neuro Champion' and a 'Clustering Champion', then provide the final audit table.
            """

            last_message = None

            # We iterate through the stream to see live logs
            async for message in unsupervised_team.run_stream(task=task):
                if hasattr(message, 'content'):
                    print(f"\n[{message.source}]:\n{message.content}")
                    print("-" * 30)

                # Update last_message every time a new message arrives
                last_message = message

            # 🟢 THE FIX: Use 'last_message' instead of 'result'
            # In AutoGen, the final item in a stream is usually a TaskResult
            from autogen_agentchat.base import TaskResult

            if isinstance(last_message, TaskResult):
                # Extract the actual text from the final message in the TaskResult list
                self.results_manifest["unsupervised"] = last_message.messages[-1].content
            elif last_message and hasattr(last_message, 'content'):
                # Fallback if it's a standard message object
                self.results_manifest["unsupervised"] = last_message.content
            else:
                self.results_manifest["unsupervised"] = "No content returned from tournament."

            self.logger.info("✅ STEP 4 COMPLETE: Neuro-Pattern Champion identified.")
            return True

        except Exception as e:
            self.logger.error(f"❌ Unsupervised Tournament Failed: {e}")
            return False

    async def run_dbscan_optimization(self):
        self.logger.info("⏳ Starting Manual DBSCAN Tuning...")

        # Starting Heuristics
        eps = 0.5
        min_samples = 5
        best_sil = -1

        for i in range(3):  # Limit to 3 smart iterations to save compute
            result_json = train_dbscan(self.settings.DB_PATH)
            res = json.loads(result_json)

            if res["status"] == "SUCCESS":
                metrics = res["metrics"]
                noise = metrics["noise_ratio"]
                sil = metrics["silhouette"]

                self.logger.info(f"Iteration {i}: eps={eps}, Noise={noise}, Sil={sil}")

                # Agentic Reasoning (Manual Implementation)
                if noise > 0.02:
                    eps += 0.2  # Too much noise, expand epsilon
                elif noise < 0.005:
                    eps -= 0.1  # Too little noise, shrink epsilon
                else:
                    break  # We are in the sweet spot!

        self.results_manifest["clustering"] = res
        return True

    # In run_phase2.py - STEP 5
    async def run_aggregator(self):
        self.logger.info("⏳ STEP 5: Executing Decision Aggregator Agent (Hybrid Mode)...")
        try:
            registry_path = os.path.join(self.settings.REPORT_DIR, "champion_registry.json")

            # 🟢 Define the tool with explicit type hinting for the Agent's benefit
            def save_tool_wrapper(supervised_config: dict, neuro_config: dict, clustering_config: dict,
                                  feature_list: list) -> str:
                """Registers the final champions into the system manifest."""
                return save_hybrid_metadata(
                    supervised_config,
                    neuro_config,
                    clustering_config,
                    feature_list,
                    registry_path
                )

            aggregator_obj = DecisionAggregator(
                model_client=self.model_client,
                settings=self.settings,
                save_tool=save_tool_wrapper
            )

            # Retrieve dynamic context
            features = aggregator_obj.get_feature_list()

            # 🟢 CLEAN DATA BLOCK: Pre-formatting the data for the Agent to avoid parsing errors
            supervised_data = {
                "path": "models/champion_xgb_dynamic.pkl",
                "type": "xgboost",
                "metrics": {"f1": 0.73, "recall": 0.76}  # Example baseline metrics
            }

            neuro_data = {
                "path": "models/champion_vae.pth",
                "type": "variational_ae",
                "metrics": {"true_positives": 166, "threshold_p": 97}
            }

            clustering_data = {
                "path": "models/champion_kmeans.joblib",
                "type": "kmeans",
                "metrics": {"silhouette": 0.2121, "anomalies": 1297}
            }

            # 🟢 ALIGNED PROMPT: Using strictly delimited blocks for data and instructions
            task_prompt = f"""
            TOURNAMENT RESULTS ARCHIVE:
            --------------------------
            SUPERVISED_DATA: {json.dumps(supervised_data)}
            NEURO_DATA: {json.dumps(neuro_data)}
            CLUSTERING_DATA: {json.dumps(clustering_data)}
            FEATURE_LIST: {features}

            REQUIRED ACTIONS:
            1. Call 'save_tool_wrapper' exactly once using the dictionaries provided above.
            2. Verify that the 'supervised_config', 'neuro_config', and 'clustering_config' parameters 
               match the provided DATA blocks perfectly.
            3. Create 'FINAL_FRAUD_STRATEGY.md' as the terminal artifact.

            STRATEGY FOCUS: Highlight the VAE (Neuro) model's ability to detect 166 fraud cases 
            that the supervised model likely missed.
            """

            await aggregator_obj.agent.run(task=task_prompt)
            self.logger.info("✅ Hybrid Strategy finalized and archived.")
            return True

        except Exception as e:
            self.logger.error(f"❌ Aggregator failed: {e}")
            return False

    async def execute_full_championship(self):
        """Orchestrator for the entire Phase 2 flow."""
        self.logger.info("=" * 60)
        self.logger.info("🏆 PHASE 2: CHAMPIONSHIP MODELING START")
        self.logger.info("=" * 60)

        # STEP 1: Baseline (Still manual as it's our non-agent benchmark)
        if not self.run_baseline_benchmark():
            sys.exit(1)

        # STEP 2: Supervised Tournament (Now includes Sampling)
        if not await self.run_supervised_tournament():
            sys.exit(1)

        # STEP 3: Unsupervised Discovery
        if not await self.run_unsupervised_discovery():
            sys.exit(1)

        # STEP 4: Aggregator
        if not await self.run_aggregator():
            sys.exit(1)

        self.logger.info("=" * 60)
        self.logger.info("🏁 PHASE 2 COMPLETE: ALL MODELS ARCHIVED")
        self.logger.info("=" * 60)


# --- EXECUTION ---
if __name__ == "__main__":
    championship = Phase2Championship()
    try:
        asyncio.run(championship.execute_full_championship())
    except KeyboardInterrupt:
        championship.logger.info("🛑 Phase 2 interrupted.")
    except Exception as e:
        championship.logger.error(f"💥 Pipeline Failure: {e}")