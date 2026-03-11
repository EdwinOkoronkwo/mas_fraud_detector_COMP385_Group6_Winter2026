
import pandas as pd
import numpy as np
import asyncio
import random
import joblib

class BatchProcessor:
    def __init__(self, pipeline):
        """
        Encapsulates the orchestration of a Phase 3 Fraud Audit batch.
        :param pipeline: The InferencePipeline instance containing models and infra.
        """
        self.pipeline = pipeline
        self.results = []
        self.weight_history = []

        # Load preprocessor once for the entire batch
        self.preprocessor = joblib.load(self.pipeline.infra.get_preprocessor_path())
        self.threshold = 0.30

    async def execute(self, n_samples: int):
        """The main entry point for the batch command."""
        df_raw = self.pipeline.handler.fetch_balanced_samples(n_samples)

        print(f"🚀 BatchProcessor: Starting execution for {n_samples} samples...")

        for _, row in df_raw.iterrows():
            raw_dict = row.to_dict()

            # 1. Specialist Inference (The Math)
            metrics = self._extract_metrics(raw_dict)

            # 2. Pillar Weight Calibration (The Learning)
            self._update_weights(metrics)

            # 3. RAG Audit (The Contextual Reasoning with Backoff)
            explanation = await self._conduct_rag_audit(metrics, raw_dict)

            # 4. Packaging
            self.results.append(self._package_row(metrics, explanation))

            # Standard pacing
            await asyncio.sleep(1.0)

        return pd.DataFrame(self.results)

    def _extract_metrics(self, raw_dict: dict) -> dict:
        """Handles the 'Sealed 24' math and pillar normalization."""
        raw_df = pd.DataFrame([raw_dict])
        if 'Unnamed: 0' not in raw_df.columns: raw_df['Unnamed: 0'] = 0

        # Transform using the class-level preprocessor
        vector = self.pipeline.infra.extract_model_input(self.preprocessor.transform(raw_df))

        # Pillar Predictions
        b_p = float(self.pipeline.base_pillar.predict(vector))
        g_p = float(self.pipeline.gold_pillar.predict(vector))
        n_raw = float(self.pipeline.neuro_pillar.predict(vector))
        c_raw = float(self.pipeline.cluster_pillar.predict_raw(vector))

        # Use the Scorer for normalization (including the fixed cluster sigmoid)
        mas_output = self.pipeline.scorer.compute_mas_score(g_p, n_raw, c_raw)

        return {
            "cc": str(raw_dict.get('cc_num'))[-4:],
            "actual": int(raw_dict.get('actual_label', 0)),
            "base_p": b_p,
            "gold_p": g_p,
            "n_norm": mas_output['n_p'],
            "c_norm": mas_output['c_p'],
            "n_raw": n_raw,
            "c_raw": c_raw,
            "math_score": mas_output['final_score']
        }

    def _update_weights(self, metrics: dict):
        """Passes performance to the Adapter and tracks weight evolution."""
        self.pipeline.scorer.adapter.update_performance(
            metrics['actual'],
            metrics['gold_p'],
            metrics['n_norm'],
            metrics['c_norm']
        )
        self.weight_history.append(self.pipeline.scorer.adapter.get_weights().copy())

    async def _conduct_rag_audit(self, metrics: dict, raw_data: dict) -> str:
        """Handles Agent communication with exponential backoff for 429 errors."""
        task_prompt = (
            f"Audit CC {metrics['cc']}. MAS Score: {round(metrics['math_score'], 3)}. "
            f"Tx: {raw_data.get('amt')} at {raw_data.get('category')}."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                audit_result = await asyncio.wait_for(
                    self.pipeline.orchestrator.execute_phase(
                        phase_name=f"Audit_{metrics['cc']}",
                        runner_factory=lambda: self.pipeline.factory.get_rag_audit_team(
                            self.pipeline.vector_service
                        ).get_team(),
                        task=task_prompt
                    ), timeout=90.0
                )
                return self.pipeline._parse_agent_explanation(audit_result.messages[-1].content)

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 12 # 12s, 24s...
                    print(f"⚠️ 429 Rate Limit hit for {metrics['cc']}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                return f"Neural Math complete (Audit unavailable: {str(e)[:15]})"

    def _package_row(self, m: dict, explanation: str) -> dict:
        """Standardizes the output keys for the PerformanceEvaluator."""
        return {
            "CC": m['cc'],
            "ACT": m['actual'],
            "BASE": round(m['base_p'], 3),
            "GOLD": round(m['gold_p'], 3),
            "NEU_RAW": round(m['n_raw'], 3),
            "NEU_NORM": round(m['n_norm'], 3),
            "CLU_RAW": round(m['c_raw'], 3),
            "CLU_NORM": round(m['c_norm'], 3),
            "MATH": round(m['math_score'], 3),
            "HIT": "✅" if (m['math_score'] >= self.threshold) == bool(m['actual']) else "❌",
            "EXPLANATION": explanation.replace("\n", " ")[:60] + ".."
        }