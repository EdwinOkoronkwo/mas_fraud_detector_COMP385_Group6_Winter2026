# import sqlite3
#
# import joblib
# import numpy as np
# from dotenv import load_dotenv
#
# from agentic_inference.core.performance_evaluator import PerformanceEvaluator
# from agentic_inference.services.vector_service import VectorService
# from config.llm_config import get_model_client
# from config.settings import settings
# from core.orchestrator import Orchestrator
# from core.state import PipelineState
# from deterministic_inference.core.pillars.baseline import BaselinePillar
# from deterministic_inference.core.pillars.clustering import ClusteringPillar
# from deterministic_inference.core.pillars.neuro import NeuroPillar
# from deterministic_inference.core.pillars.supervised import SupervisedPillar
# from deterministic_inference.core.scoring_engine import ScoringEngine
# from deterministic_inference.utils.data_handler import DataHandler
# from strategies.data_preprocessing.feature_engineer import FeatureEngineerAgent
# from deterministic_inference.utils.infrastructure import InfrastructureManager
#
# import pandas as pd
# import asyncio
# import re
#
# from factories.agent_factory import AgentFactory
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_colwidth', 80)
#
#
# class InferencePipeline:
#     def __init__(self, factory: AgentFactory, vector_service: VectorService, baseline_version="enhanced"):
#         # Infrastructure
#         self.infra = InfrastructureManager()
#         self.scorer = ScoringEngine()
#
#         # --- FEATURE LIST SEPARATION ---
#         # MAS (Gold, Neuro, Cluster) ALWAYS needs the full 27 features
#         self.mas_feature_list = self.infra.get_features(mode="gold")
#
#         # Baseline uses the specified version (24 or 27)
#         self.base_feature_list = self.infra.get_features(mode=baseline_version)
#
#         # Core Orchestration
#         self.state = PipelineState()
#         self.orchestrator = Orchestrator(self.state)
#         self.factory = factory
#         self.vector_service = vector_service
#         self.handler = DataHandler(self.infra.db_path)
#
#         # --- PILLAR INITIALIZATION ---
#         # Use mas_feature_list (27) for the Champion components
#         self.gold_pillar = SupervisedPillar(self.infra.get_gold_model_path(), self.mas_feature_list)
#         self.neuro_pillar = NeuroPillar(self.infra.get_neuro_model_path())
#         self.cluster_pillar = ClusteringPillar(self.infra.get_cluster_model_path())
#
#         # Use base_feature_list (24 or 27) for the Baseline component
#         self.base_pillar = BaselinePillar(
#             self.infra.get_baseline_model_path(version=baseline_version),
#             self.base_feature_list
#         )
#
#     def _parse_agent_explanation(self, text: str) -> str:
#         clean_text = re.sub(r"CASE_CLOSED|```json|```|\{.*\}", "", text, flags=re.DOTALL).strip()
#         return clean_text if len(clean_text) >= 5 else "Analysis complete."
#
#     async def run_batch(self, n_samples: int = 10):
#         df_raw = self.handler.fetch_balanced_samples(n_samples)
#         results = []
#         custom_threshold = 0.30
#
#         # Load the preprocessor (Enhanced/27 features)
#         preprocessor = joblib.load(self.infra.get_preprocessor_path())
#         cat_means = self.infra.registry.get('metadata', {}).get('cat_means', {})
#
#         print(f"🕵️ Starting XAI Audit for {n_samples} samples...")
#
#         for _, row in df_raw.iterrows():
#             raw_dict = row.to_dict()
#             cc_tail = str(raw_dict.get('cc_num'))[-4:]
#
#             # --- BEHAVIORAL INDUCTION ---
#             raw_df = pd.DataFrame([raw_dict])
#             category = raw_dict.get('category')
#             avg = cat_means.get(category, raw_dict.get('amt'))
#             raw_df['amt_to_cat_avg'] = raw_dict.get('amt') / avg if avg != 0 else 1.0
#             dt_object = pd.to_datetime(raw_dict.get('unix_time'), unit='s')
#             raw_df['high_risk_time'] = 1 if dt_object.hour in [23, 0, 1, 2, 3, 4] else 0
#             raw_df['txn_velocity'] = 1.0
#
#             # 1. Transform to 27 features
#             # 1. Transform produces 27 features (Gold standard)
#             processed_vector = preprocessor.transform(raw_df)
#
#             # 2. Reshape to 2D (Standardizing shape)
#             if hasattr(processed_vector, 'shape') and len(processed_vector.shape) == 3:
#                 processed_vector = processed_vector.reshape(processed_vector.shape[0], -1)
#
#             # 3. THE CRITICAL SLICE (Prevents the ValueError)
#             # If we are using the 24-feature baseline, give it only the first 24 columns
#             if len(self.base_feature_list) == 24:
#                 base_input = processed_vector[:, :24]
#                 print(f"DEBUG: Baseline Input Shape: {base_input.shape}")
#             else:
#                 base_input = processed_vector
#                 print(f"DEBUG: Baseline Input Shape: {base_input.shape}")
#
#             # 4. Predict (Now the dimensions match exactly!)
#             b_p = self.base_pillar.predict(base_input)
#
#             # --- 3. DUAL-PATH PREDICTION (The Fix) ---
#
#             # Path A: MAS (Full 27 features)
#             g_p = self.gold_pillar.predict(processed_vector)
#             n_mse = self.neuro_pillar.predict(processed_vector)
#             c_dist = self.cluster_pillar.predict_raw(processed_vector)
#
#             # Path B: Baseline (Slice to 24 if needed)
#             if len(self.base_feature_list) == 24:
#                 # Slice indices 0-23
#                 base_input = processed_vector[:, :24]
#             else:
#                 base_input = processed_vector
#
#             b_p = self.base_pillar.predict(base_input)
#
#             # --- SCORING & AGENTS ---
#             mas_scores = self.scorer.compute_mas_score(g_p, n_mse, c_dist)
#             math_score = mas_scores['final_score']
#
#             tx_data = f"AMT: {raw_dict.get('amt')}, CC_TAIL: {cc_tail}"
#             task_prompt = f"Audit CC {cc_tail}. Math Score: {round(math_score, 3)}. Data: {tx_data}."
#
#             try:
#                 audit_result = await asyncio.wait_for(
#                     self.orchestrator.execute_phase(
#                         phase_name=f"Audit_{cc_tail}",
#                         runner_factory=lambda: self.factory.get_rag_audit_team(self.vector_service).get_team(),
#                         task=task_prompt
#                     ),
#                     timeout=120.0
#                 )
#                 explanation = self._parse_agent_explanation(audit_result.messages[-1].content)
#             except Exception:
#                 explanation = "Neural Math analysis complete (Policy timeout)."
#
#             # Formatting for display
#             display_explanation = (explanation.replace("\n", " ")[:60] + "..")
#
#             results.append({
#                 "CC": cc_tail,
#                 "ACT": raw_dict['actual_label'],
#                 "BASE": round(b_p, 3),
#                 "GOLD": round(g_p, 3),
#                 "MATH": round(math_score, 3),
#                 "HIT": "✅" if (math_score >= custom_threshold) == raw_dict['actual_label'] else "❌",
#                 "EXPLANATION": display_explanation
#             })
#             await asyncio.sleep(0.5)
#
#         return pd.DataFrame(results)
#
#
# async def main():
#     # 1. Load environment variables
#     load_dotenv()
#
#     # --- A. Database Sanitization ---
#     infra = InfrastructureManager()
#     db_path = infra.db_path
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
#     cursor.execute("UPDATE test_transactions SET merchant = REPLACE(merchant, 'fraud_', '')")
#     conn.commit()
#     conn.close()
#     print("🧹 Database Sanitized: 'fraud_' merchant names removed.")
#
#     # 2. Initialize Model & Services
#     m_client = get_model_client()
#     my_v_service = VectorService()
#     my_factory = AgentFactory(model_client=m_client, settings=settings)
#
#     # 3. Initialize & Run Pipeline
#     pipeline = InferencePipeline(factory=my_factory, vector_service=my_v_service)
#     print("🚀 Starting Hybrid Deterministic-Agentic Pipeline...")
#     n_samples = 100
#     final_results = await pipeline.run_batch(n_samples=n_samples)
#
#     # 4. Reporting & Export
#     print("\n" + "=" * 85)
#     print(f"{'FINAL MAS FRAUD DETECTION SUMMARY':^85}")
#     print("=" * 85)
#     print(final_results.to_string(index=False))
#
#     evaluator = PerformanceEvaluator()
#
#     # --- ADDED THESE LINES ---
#     print(evaluator.generate_report(final_results))  # Terminal Table
#     evaluator.plot_results(final_results)  # Generates PNGs in /reports/plots
#     evaluator.save_to_csv(final_results)  # Saves the CSV
#     # -------------------------
#
#     print("=" * 85)
#
# if __name__ == "__main__":
#     asyncio.run(main())