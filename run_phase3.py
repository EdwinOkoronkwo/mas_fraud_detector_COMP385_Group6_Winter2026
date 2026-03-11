import os
import re
import asyncio
import pandas as pd
import joblib
from dotenv import load_dotenv

from agentic_inference.core.batch_processor import BatchProcessor
from agentic_inference.core.performance_evaluator import PerformanceEvaluator
from agentic_inference.services.vector_service import VectorService
from config.llm_config import get_model_client
from config.settings import settings
from core.orchestrator import Orchestrator
from core.state import PipelineState
from deterministic_inference.core.pillars.baseline import BaselinePillar
from deterministic_inference.core.pillars.clustering import ClusteringPillar
from deterministic_inference.core.pillars.neuro import NeuroPillar
from deterministic_inference.core.pillars.supervised import SupervisedPillar
from deterministic_inference.core.scoring_engine import ScoringEngine
from deterministic_inference.utils.data_handler import DataHandler
from deterministic_inference.utils.infrastructure import InfrastructureManager
from factories.agent_factory import AgentFactory


class InferencePipeline:
    def __init__(self, factory: AgentFactory, vector_service: VectorService, baseline_version="enhanced"):
        # 1. Infrastructure & Scoring
        self.infra = InfrastructureManager()
        self.scorer = ScoringEngine()
        self.handler = DataHandler(self.infra.db_path)

        # 2. State & Orchestration
        self.state = PipelineState()
        self.orchestrator = Orchestrator(self.state)
        self.factory = factory
        self.vector_service = vector_service

        # 3. Tracking
        self.last_weight_history = []

        # 4. Pillar Initialization
        self.mas_features = self.infra.get_features()
        self.gold_pillar = SupervisedPillar(self.infra.get_gold_model_path(), self.mas_features)
        self.neuro_pillar = NeuroPillar(self.infra.get_neuro_model_path())
        self.cluster_pillar = ClusteringPillar(self.infra.get_cluster_model_path())
        self.base_pillar = BaselinePillar(self.infra.get_baseline_model_path(), self.mas_features)

    def _parse_agent_explanation(self, text: str) -> str:
        # Added "BANKING OPERATIONAL POLICY.*?\d\)" to the removal pattern
        clean_text = re.sub(
            r"BANKING OPERATIONAL POLICY.*?RULE-[\w-]+:\s*|CASE_CLOSED|```json|```|\{.*\}",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        # Remove any lingering double spaces or leading dashes
        clean_text = re.sub(r"^\W+", "", clean_text)
        return clean_text if len(clean_text) >= 5 else "Neural Math complete."

    async def run_batch(self, n_samples: int = 100):
        """Delegates the heavy lifting to the BatchProcessor."""
        processor = BatchProcessor(self)
        final_df = await processor.execute(n_samples)

        # Sync weight history back to pipeline so main() can find it for evaluator
        self.last_weight_history = processor.weight_history
        return final_df


async def main():
    load_dotenv()
    infra = InfrastructureManager()

    # 1. Initialize Foundation
    m_client = get_model_client()
    my_v_service = VectorService()
    my_factory = AgentFactory(model_client=m_client, settings=settings)

    # 2. Fetch Data (SET TO 100 FOR FULL PLOT)
    handler = DataHandler(infra.db_path)
    df_raw = handler.fetch_balanced_samples(n_samples=100)
    print(f"⚖️ Dataset Loaded: {len(df_raw)} total rows")

    # 3. Initialize Pipeline
    pipeline = InferencePipeline(factory=my_factory, vector_service=my_v_service)

    # 4. Run Batch (Logs will appear here)
    print("🚀 Processing transactions and generating agent audits...")
    final_results = await pipeline.run_batch(n_samples=100)

    # 5. Reporting
    evaluator = PerformanceEvaluator()

    # A. The Audit Table
    print("\n" + "═" * 110)
    print(f"║ {'DETAILED FRAUD AUDIT TRAIL':^106} ║")
    print("═" * 110)
    print(final_results.to_string(index=False))

    # B. The Performance Metrics (Baseline vs MAS)
    print(evaluator.generate_report(final_results))

    # C. THE WEIGHT ADJUSTMENT TABLE (The part you were missing)
    # This shows the final Trust % for each agent
    print(evaluator.generate_weights_report(pipeline.scorer.adapter))

    # 6. Visualization
    evaluator.plot_weight_evolution(pipeline.last_weight_history, final_results)
    evaluator.plot_results(final_results)
    evaluator.plot_executive_results(final_results)

    # PRINT THE FINAL WEIGHT EVOLUTION TABLE
    print("\n" + "═" * 80)
    print(f"║ {'FINAL SYSTEM TRUST CALIBRATION':^76} ║")
    print("═" * 80)

    weights = pipeline.scorer.adapter.get_weights()
    perf = pipeline.scorer.adapter.agent_performance

    print(f"{'AGENT':<15} | {'TRUST WEIGHT':<15} | {'TP':<6} | {'FP':<6} | {'FN':<6}")
    print("-" * 80)
    for agent, w in weights.items():
        stats = perf[agent]
        print(
            f"{agent.upper():<15} | {w:>12.2%} | {int(stats['tp']):<6} | {int(stats['fp']):<6} | {int(stats['fn']):<6}")
    print("═" * 80)

    print(f"📈 Final Check: History contains {len(pipeline.last_weight_history)} data points.")
    print(f"✅ PIPELINE COMPLETE: Check 'reports/plots/' for visualizations.")

if __name__ == "__main__":
    asyncio.run(main())
