import logging

from autogen_agentchat.base import TaskResult

from config.settings import settings
from core.state import PipelineState
from core.orchestrator import Orchestrator
from factories.agent_factory import AgentFactory
from tools.data_prep.preprocess_tools import scrub_junk_columns

logger = logging.getLogger("FraudPipeline")


class FraudPipeline:
    def __init__(self, settings, model_client):
        self.settings = settings
        self.model_client = model_client

        # FIX: Pass 'settings' (the object), not 'settings.AGENT_CONFIG' (the dict)
        self.factory = AgentFactory(model_client, settings)

        self.orchestrator = Orchestrator(PipelineState())

    # --- 1. DEFINE THE TOOL INSIDE THE CLASS ---
    def save_inference_metadata(self, supervised_path: str, neural_path: str, clustering_path: str) -> str:
        """Saves the official model selection for the inference pipeline."""
        import json
        import os

        # We can use class-level settings here!
        registry_path = os.path.join(self.settings.PROJECT_ROOT, "models/champion_registry.json")

        metadata = {
            "champions": {
                "supervised": supervised_path,
                "neural": neural_path,
                "clustering": clustering_path
            },
            "ensemble_weights": {"supervised": 0.4, "neural": 0.4, "clustering": 0.2}
        }

        with open(registry_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return f"SUCCESS: Registry saved to {registry_path}"

    async def run_data_foundation(self):
        # 1. Execute the Agentic Phase
        result = await self.orchestrator.execute_phase(
            phase_name="Data_Foundation",
            runner_factory=self.factory.get_preprocessing_team,
            task="Migrate CSV to SQL and scale features."
        )

        # 2. Hard-Gate Scrubbing
        # Calling your saved function to enforce the 27-column contract
        # scrub_junk_columns(settings.DB_PATH)

        return result

    async def run_supervised_tournament(self):
        return await self.orchestrator.execute_phase(
            phase_name="Supervised_Championship",
            runner_factory=self.factory.get_supervised_championship_team,
            task=f"Start balancing protocol using {self.settings.DB_PATH}."
        )

    async def run_anomaly_discovery(self):
        instruction = f"Run clustering and neural patterns on {self.settings.DB_PATH}. Calculate alignment %."
        return await self.orchestrator.execute_phase(
            phase_name="Anomaly_Discovery",
            runner_factory=self.factory.get_anomaly_discovery_team,
            task=instruction
        )

    async def run_aggregation(self, results):
        agg_agent = self.factory.get_decision_aggregator_agent(
            save_method=self.save_inference_metadata
        )

        # 🔍 FIX: Search the message history for the actual metrics
        # Instead of just the last message, we look for the message from the 'Critic' or 'Summary' agent
        supervised_summary = next((m.content for m in reversed(results['sup'].messages)
                                   if "Recall" in str(m.content)), "No supervised data found")

        anomaly_summary = next((m.content for m in reversed(results['anom'].messages)
                                if "Champion" in str(m.content)), "No anomaly data found")

        agg_task = f"""
        Tournament results are finalized. USE THE DATA BELOW FOR THE REPORT:
        "Note: The Supervised model is at 'models/champion_xgboost.joblib', the Neuro model is at 'models/champion_vae.pth', and the Clustering model is at 'models/champion_dbscan.joblib'.

        ### SUPERVISED TOURNAMENT DATA:
        {supervised_summary}

        ### ANOMALY DISCOVERY DATA:
        {anomaly_summary}
        """

        # This ensures the LLM 'reads' the actual metrics before writing the MD file
        return await agg_agent.run(task=agg_task)




# class FraudPipeline:
#     def __init__(self, settings, model_client):
#         self.settings = settings
#         self.model_client = model_client
#         self.factory = AgentFactory(model_client, settings.AGENT_CONFIG)
#
#         self.orchestrator = Orchestrator(PipelineState())
#
#     # --- 1. DEFINE THE TOOL INSIDE THE CLASS ---
#     def save_inference_metadata(self, supervised_path: str, neural_path: str, clustering_path: str) -> str:
#         """Saves the official model selection for the inference pipeline."""
#         import json
#         import os
#
#         # We can use class-level settings here!
#         registry_path = os.path.join(self.settings.PROJECT_ROOT, "models/champion_registry.json")
#
#         metadata = {
#             "champions": {
#                 "supervised": supervised_path,
#                 "neural": neural_path,
#                 "clustering": clustering_path
#             },
#             "ensemble_weights": {"supervised": 0.4, "neural": 0.4, "clustering": 0.2}
#         }
#
#         with open(registry_path, "w") as f:
#             json.dump(metadata, f, indent=4)
#
#         return f"SUCCESS: Registry saved to {registry_path}"
#
#     async def run_data_foundation(self):
#         return await self.orchestrator.execute_phase(
#             phase_name="Data_Foundation",
#             runner_factory=self.factory.get_preprocessing_team,
#             task="Migrate CSV to SQL and scale features."
#         )
#     # Inside FraudPipeline
#     # async def run_data_foundation(self):
#     #     # This phase name should be unique or the runtime must be reset
#     #     return await self.orchestrator.execute_phase(
#     #         phase_name=f"Data_Foundation_{int(time.time())}",  # Add a timestamp to force uniqueness
#     #         runner_factory=self.factory.get_preprocessing_team,
#     #         task="Migrate CSV to SQL and scale features."
#     #     )
#
#     # async def run_supervised_tournament(self):
#     #     # This "task" string acts as a NEW USER MESSAGE.
#     #     # It resets the role order and satisfies the API.
#     #     instruction = f"Phase 1 is complete and verified. Please start the Supervised Tournament using data from {self.settings.DB_PATH}."
#     #
#     #     return await self.orchestrator.execute_phase(
#     #         phase_name="Supervised_Championship",
#     #         runner_factory=self.factory.get_supervised_championship_team,
#     #         task=instruction  # <--- This is the "User" role the API is looking for
#     #     )
#
#     async def run_supervised_tournament(self):
#         return await self.orchestrator.execute_phase(
#             phase_name="Supervised_Championship",
#             runner_factory=self.factory.get_supervised_championship_team,
#             task=f"Start balancing protocol using {self.settings.DB_PATH}."
#         )
#
#     async def run_anomaly_discovery(self):
#         instruction = f"Run clustering and neural patterns on {self.settings.DB_PATH}. Calculate alignment %."
#         return await self.orchestrator.execute_phase(
#             phase_name="Anomaly_Discovery",
#             runner_factory=self.factory.get_anomaly_discovery_team,
#             task=instruction
#         )
#
#     async def run_aggregation(self, results):
#         # No prompt here. No tool setup here.
#         # Just grab the specialized agent from the factory.
#         agg_agent = self.factory.get_decision_aggregator_agent(
#             save_method=self.save_inference_metadata
#         )
#
#         agg_task = f"""
#         The tournament is over.
#         Supervised Winner: {results['sup'].messages[-1].content}
#         Anomaly Winner: {results['anom'].messages[-1].content}
#         """
#
#         # Direct run ensures the tool-loop (iterations) completes
#         return await agg_agent.run(task=agg_task)


