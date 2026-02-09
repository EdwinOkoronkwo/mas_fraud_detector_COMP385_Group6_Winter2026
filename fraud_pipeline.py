import os
import json
import logging
from datetime import datetime

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from tools.reporting.report_tools import write_markdown_report

from config import settings
from config.llm_config import get_model_client
from config.settings import AGENT_CONFIG, TEMP_SPLIT_PATH
from core.decision_agent import DecisionAggregator
from core.state import PipelineState
from core.orchestrator import Orchestrator
from factories.agent_factory import AgentFactory
from autogen_core.models import UserMessage

from tools.reporting.report_tools import write_markdown_report
from tools.training.deployment_tools import persist_champion_model
from utils.logger import setup_logger


logger = logging.getLogger("FraudPipeline")


class FraudPipeline:
    def __init__(self, settings, model_client):
        self.settings = settings
        self.model_client = model_client
        self.factory = AgentFactory(model_client, settings.AGENT_CONFIG)

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
        return await self.orchestrator.execute_phase(
            phase_name="Data_Foundation",
            runner_factory=self.factory.get_preprocessing_team,
            task="Migrate CSV to SQL and scale features."
        )

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
        # No prompt here. No tool setup here.
        # Just grab the specialized agent from the factory.
        agg_agent = self.factory.get_decision_aggregator_agent(
            save_method=self.save_inference_metadata
        )

        agg_task = f"""
        The tournament is over. 
        Supervised Winner: {results['sup'].messages[-1].content}
        Anomaly Winner: {results['anom'].messages[-1].content}
        """

        # Direct run ensures the tool-loop (iterations) completes
        return await agg_agent.run(task=agg_task)


