
import pandas as pd
import numpy as np
import asyncio
import random
import joblib

import asyncio

import pandas as pd
import joblib
import asyncio

import pandas as pd
import joblib
import asyncio
import json

import pandas as pd
import joblib
import asyncio

import pandas as pd
import joblib
import asyncio

import asyncio
import pandas as pd
import joblib


class BatchProcessor:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = []
        self.weight_history = []
        self.preprocessor = joblib.load(self.pipeline.infra.get_preprocessor_path())
        self.threshold = 0.30

    async def execute(self, n_samples: int, params: dict = None, callback=None):
        """Standard Batch Processing Loop"""
        df_raw = self.pipeline.handler.fetch_balanced_samples(n_samples)
        self.results = []

        for i, row in df_raw.iterrows():
            # 🚀 Rate Limit Protection
            if i > 0:
                await asyncio.sleep(2.0)

            raw_dict = row.to_dict()
            metrics = self._extract_metrics(raw_dict, params)
            cc_val = str(raw_dict['cc_num'])

            # 🚀 Audit with Rehydration (Fetches identity from DB)
            audit_data = await self._conduct_rag_audit(
                metrics=metrics,
                raw_data=raw_dict,
                first="Lookup...",
                last="Pending",
                cc_display=cc_val[-4:],
                callback=callback
            )

            packaged = self._package_row(metrics, audit_data, raw_data=raw_dict)
            self.results.append(packaged)

        return pd.DataFrame(self.results)

    async def execute_manual(self, df_manual: pd.DataFrame, params: dict = None, callback=None):
        self.results = []

        for _, row in df_manual.iterrows():
            raw_dict = row.to_dict()
            cc_full = str(raw_dict.get('cc_num', '0000'))

            # 🚀 FIX: Manual mode must also provide the 3 new arguments
            first_name = str(raw_dict.get('first', 'Unknown'))
            last_name = str(raw_dict.get('last', 'Unknown'))
            cc_display = f"****{cc_full[-4:]}"

            if callback:
                callback("SYSTEM", "Executing Manual Compliance Investigation...")

            metrics = self._extract_metrics(raw_dict, params)
            self._update_weights(metrics)

            # 🚀 ALIGNED CALL (6 Args)
            audit_data = await self._conduct_rag_audit(
                metrics, raw_dict, first_name, last_name, cc_display, callback=callback
            )

            self.results.append(self._package_row(metrics, audit_data, raw_dict))

        return pd.DataFrame(self.results)

    # async def execute_manual(self, df_manual: pd.DataFrame, params: dict = None, callback=None):
    #     """Restored: Handles the 'Single Transaction Audit' view"""
    #     self.results = []
    #     for _, row in df_manual.iterrows():
    #         raw_dict = row.to_dict()
    #         cc_val = str(raw_dict['cc_num'])
    #
    #         if callback:
    #             callback("SYSTEM", f"Initiating Deep Audit for CC: ...{cc_val[-4:]}")
    #
    #         metrics = self._extract_metrics(raw_dict, params)
    #
    #         # Sync weights for manual audits to keep the model adaptive
    #         if hasattr(self, '_update_weights'):
    #             self._update_weights(metrics)
    #
    #         # Re-use the same audit logic (which fetches names from DB)
    #         audit_data = await self._conduct_rag_audit(
    #             metrics=metrics,
    #             raw_data=raw_dict,
    #             first="Manual",
    #             last="Audit",
    #             cc_display=cc_val[-4:],
    #             callback=callback
    #         )
    #
    #         self.results.append(self._package_row(metrics, audit_data, raw_dict))
    #
    #     return pd.DataFrame(self.results)

    def _extract_metrics(self, raw_dict: dict, params: dict = None) -> dict:
        raw_df = pd.DataFrame([raw_dict])
        if 'Unnamed: 0' not in raw_df.columns:
            raw_df['Unnamed: 0'] = 0

        vector = self.pipeline.infra.extract_model_input(self.preprocessor.transform(raw_df))

        # Model Pillar Calculations
        b_p = float(self.pipeline.base_pillar.predict(vector))
        g_p = float(self.pipeline.gold_pillar.predict(vector))
        n_raw = float(self.pipeline.neuro_pillar.predict(vector))
        c_raw = float(self.pipeline.cluster_pillar.predict_raw(vector))

        mas_output = self.pipeline.scorer.compute_mas_score(
            gold_prob=g_p, neuro_mse=n_raw, cluster_dist=c_raw, params=params
        )

        return {
            "CC": str(raw_dict['cc_num']),
            "ACT": int(raw_dict['actual_label']),  # Strictly mapping to your DB column
            "BASE": b_p,
            "GOLD": g_p,
            "MATH": mas_output['final_score'],
            "N_RAW": n_raw,
            "C_RAW": c_raw,
            "N_CAL": mas_output['n_p'],
            "C_CAL": mas_output['c_p'],
            "mode": mas_output['mode']
        }

    async def _conduct_rag_audit(self, metrics: dict, raw_data: dict, first: str, last: str, cc_display: str,
                                 callback=None) -> dict:
        """The 'Brain' of the system: Grounding + RAG reasoning"""
        cc_full = str(raw_data['cc_num'])

        # 1. FETCH ACTUAL IDENTITY (The Fix for 'Unknown User')
        db_record = self.pipeline.handler.get_transaction_by_cc(cc_full)
        if isinstance(db_record, pd.DataFrame) and not db_record.empty:
            actual_record = db_record.iloc[0].to_dict()
        else:
            actual_record = raw_data

        # Use variables from DB record
        real_f = actual_record.get('first', 'Unknown')
        real_l = actual_record.get('last', 'User')
        merchant = actual_record.get('merchant', 'Unknown Merchant')
        category = actual_record.get('category', 'General')

        if callback:
            callback("SYSTEM", f"Direct Grounding: Analysis for {real_f} {real_l}...")

        task_prompt = (
            f"INVESTIGATION REPORT: {real_f} {real_l}\n"
            f"IDENTIFIER: {cc_full} | CATEGORY: {category} | MERCHANT: {merchant}\n"
            f"RISK OVERVIEW: The MAS System has flagged this as HIGH RISK (Score: {metrics['MATH']:.3f}).\n"
            f"PILLAR BREAKDOWN:\n"
            f"- Supervised Risk (GOLD): {metrics['GOLD']:.3f}\n"
            f"- Neural Anomaly (MSE): {metrics['N_RAW']:.3f}\n"
            f"- Baseline Probability: {metrics['BASE']:.3f}\n\n"
            f"TASK: 1. Evaluate the '{category}' policy.\n"
            f"2. CRITICAL: Reconcile why the Baseline/GOLD scores are high even if MSE is low."
        )

        try:
            # 2. Agent Execution
            audit_result = await asyncio.wait_for(
                self.pipeline.orchestrator.execute_phase(
                    phase_name=f"Audit_{cc_display}",
                    runner_factory=lambda: self.pipeline.factory.get_rag_audit_team(
                        self.pipeline.vector_service
                    ).get_team(),
                    task=task_prompt
                ), timeout=90.0
            )

            # Trace and Narrative extraction
            trace_logs = [{"agent": getattr(m, 'source', 'AGENT'), "content": str(m.content)} for m in
                          audit_result.messages]
            narrative = self.pipeline._parse_agent_explanation(audit_result.messages[-1].content)

            return {
                "narrative": narrative,
                "trace": trace_logs,
                "found_first": real_f,
                "found_last": real_l,
                "found_merchant": merchant
            }

        except Exception as e:
            if callback:
                callback("SYSTEM", f"❌ RAG Failure: {str(e)[:50]}")
            return {"narrative": f"Audit Error: {str(e)}", "trace": [], "found_first": real_f, "found_last": real_l,
                    "found_merchant": merchant}

    def _package_row(self, m: dict, audit_data: dict, raw_data: dict) -> dict:
        """Consolidates findings for the UI DataFrame"""
        return {
            "CC": m['CC'],
            "first": audit_data['found_first'],
            "last": audit_data['found_last'],
            "merchant": audit_data['found_merchant'],
            "ACT": m['ACT'],
            "BASE": round(m['BASE'], 3),
            "GOLD": round(m['GOLD'], 3),
            "N_RAW": round(m['N_RAW'], 3),
            "MATH": round(m['MATH'], 3),
            "mode": m['mode'],
            "hit": "✅" if (m['MATH'] >= 0.30) == bool(m['ACT']) else "❌",
            "explanation": audit_data['narrative'],
            "trace": audit_data['trace']
        }

    def _update_weights(self, metrics: dict):
        """Self-Calibration Logic"""
        self.pipeline.scorer.adapter.update_performance(
            metrics['ACT'], metrics['GOLD'], metrics['N_CAL'], metrics['C_CAL']
        )
        self.weight_history.append(self.pipeline.scorer.adapter.get_weights().copy())



# class BatchProcessor:
#     def __init__(self, pipeline):
#         self.pipeline = pipeline
#         self.results = []
#         self.weight_history = []
#         # Load the preprocessor once for efficiency
#         self.preprocessor = joblib.load(self.pipeline.infra.get_preprocessor_path())
#         self.threshold = 0.30
#
#     async def execute(self, n_samples: int, params: dict = None, callback=None):
#         """Standard automated batch execution."""
#         df_raw = self.pipeline.handler.fetch_balanced_samples(n_samples)
#         self.results = []
#
#         for _, row in df_raw.iterrows():
#             raw_dict = row.to_dict()
#             cc_id = str(raw_dict.get('cc_num'))[-4:]
#
#             if callback:
#                 callback("SYSTEM", f"Started Deep-Dive Audit for CC ...{cc_id}")
#
#             # 1. Specialist Math Inference
#             metrics = self._extract_metrics(raw_dict, params)
#
#             if callback:
#                 callback("SQL_RESEARCHER", f"Analyzing merchant: {raw_dict.get('merchant')}")
#
#             # 2. Update Adaptive Weights
#             self._update_weights(metrics)
#
#             # 3. Agentic RAG Audit - 🚀 FIXED: Passed callback to enable live streaming
#             audit_data = await self._conduct_rag_audit(metrics, raw_dict, callback=callback)
#
#             if callback:
#                 callback("VECTOR_RESEARCHER", f"Cross-referenced Policy v2026.1 for Risk: {metrics['MATH']}")
#
#             # 4. Final Packaging
#             self.results.append(self._package_row(metrics, audit_data))
#
#         return pd.DataFrame(self.results)
#
#     async def execute_manual(self, df_manual: pd.DataFrame, params: dict = None, callback=None):
#         """Targeted execution for Single View audits."""
#         self.results = []
#
#         for _, row in df_manual.iterrows():
#             raw_dict = row.to_dict()
#
#             if callback:
#                 callback("SYSTEM", "Executing Manual Compliance Investigation...")
#
#             # 1. Specialist Math Inference
#             metrics = self._extract_metrics(raw_dict, params)
#
#             # 2. Update Adaptive Weights
#             self._update_weights(metrics)
#
#             if callback:
#                 callback("SQL_RESEARCHER", "Verifying specific transaction ledger entry...")
#
#             # 3. Agentic RAG Audit - 🚀 FIXED: Passed callback to enable live streaming
#             audit_data = await self._conduct_rag_audit(metrics, raw_dict, callback=callback)
#
#             if callback:
#                 callback("VECTOR_RESEARCHER", "Audit Complete. Generating narrative...")
#
#             # 4. Final Packaging
#             self.results.append(self._package_row(metrics, audit_data))
#
#         return pd.DataFrame(self.results)
#
#     def _extract_metrics(self, raw_dict: dict, params: dict = None) -> dict:
#         """Extracts and computes all 6 mathematical pillars + MAS Consensus."""
#         raw_df = pd.DataFrame([raw_dict])
#         if 'Unnamed: 0' not in raw_df.columns:
#             raw_df['Unnamed: 0'] = 0
#
#         vector = self.pipeline.infra.extract_model_input(self.preprocessor.transform(raw_df))
#
#         # Raw Model Predictions
#         b_p = float(self.pipeline.base_pillar.predict(vector))
#         g_p = float(self.pipeline.gold_pillar.predict(vector))
#         n_raw = float(self.pipeline.neuro_pillar.predict(vector))
#         c_raw = float(self.pipeline.cluster_pillar.predict_raw(vector))
#
#         # Consensus Scorer (Returns calibrated values and final score)
#         mas_output = self.pipeline.scorer.compute_mas_score(
#             gold_prob=g_p, neuro_mse=n_raw, cluster_dist=c_raw, params=params
#         )
#
#         return {
#             "CC": str(raw_dict.get('cc_num', '0000'))[-4:],
#             "ACT": int(raw_dict.get('actual_label', 0)),
#             "BASE": b_p,
#             "GOLD": g_p,
#             "MATH": mas_output['final_score'],
#             "N_RAW": n_raw,
#             "C_RAW": c_raw,
#             "N_CAL": mas_output['n_p'],
#             "C_CAL": mas_output['c_p'],
#             "mode": mas_output['mode']
#         }
#
#     async def _conduct_rag_audit(self, metrics: dict, raw_data: dict, callback=None) -> dict:
#         """Executes the MAS Agent phase and extracts a clean narrative."""
#         cc_id = metrics['CC']
#
#         if callback:
#             callback("ORCHESTRATOR", f"Initializing Phase: Audit_{cc_id}")
#
#         # Construct the task with user profile context
#         task_prompt = (
#             f"USER_PROFILE: {raw_data.get('name', 'Customer')} | Home: {raw_data.get('city', 'Unknown')}\n"
#             f"TRANSACTION: ${raw_data.get('amt')} at {raw_data.get('merchant', 'Unknown Merchant')}\n"
#             f"ANALYSIS: Neuro MSE is {metrics['N_RAW']}, Distance is {raw_data.get('distance', 0)} miles.\n\n"
#             f"TASK: Look at the rules in the chat history. Translate the matching condition into a single "
#             f"professional sentence for the audit report. \n"
#             f"EXAMPLE: 'For Alice, this $45 charge is flagged as High Risk because it matches the high-risk "
#             f"small-ticket pattern defined in RULE-PROBE-4837.'\n"
#             f"STRICT: Use your OUTPUT RULE format. Do not list Condition B or C."
#         )
#
#         try:
#             # 1. Run the Multi-Agent Group Chat
#             audit_result = await asyncio.wait_for(
#                 self.pipeline.orchestrator.execute_phase(
#                     phase_name=f"Audit_{cc_id}",
#                     runner_factory=lambda: self.pipeline.factory.get_rag_audit_team(
#                         self.pipeline.vector_service
#                     ).get_team(),
#                     task=task_prompt
#                 ), timeout=90.0
#             )
#
#             # 2. Extract Trace & Handle TextMessage Objects
#             trace_logs = []
#             for msg in audit_result.messages:
#                 name = getattr(msg, 'name', 'ORCHESTRATOR')
#                 content = msg.content if hasattr(msg, 'content') else str(msg)
#                 trace_logs.append({"agent": name, "content": content})
#
#             # 3. Extract the last message (The Synthesis Engine's output)
#             last_msg = audit_result.messages[-1]
#             final_raw_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
#
#             # Clean and format the final narrative
#             narrative = self.pipeline._parse_agent_explanation(final_raw_text)
#
#             if callback:
#                 callback("SYSTEM", f"✅ Audit {cc_id} Complete.")
#
#             return {"narrative": narrative, "trace": trace_logs}
#
#         except Exception as e:
#             if callback:
#                 callback("SYSTEM", f"❌ Audit {cc_id} Failed: {str(e)[:50]}")
#             return {"narrative": f"Audit unavailable: {str(e)}", "trace": []}
#
#     def _update_weights(self, metrics: dict):
#         """Updates the adaptive weighting system based on current performance."""
#         self.pipeline.scorer.adapter.update_performance(
#             metrics['ACT'], metrics['GOLD'], metrics['N_CAL'], metrics['C_CAL']
#         )
#         self.weight_history.append(self.pipeline.scorer.adapter.get_weights().copy())
#
#     def _package_row(self, m: dict, audit_data: dict) -> dict:
#         """Consolidates all metrics and RAG output into a single standardized dictionary."""
#         return {
#             "CC": m['CC'],
#             "ACT": m['ACT'],
#             "BASE": round(m['BASE'], 3),
#             "GOLD": round(m['GOLD'], 3),
#             "N_RAW": round(m['N_RAW'], 3),
#             "N_CAL": round(m['N_CAL'], 3),
#             "C_RAW": round(m['C_RAW'], 3),
#             "C_CAL": round(m['C_CAL'], 3),
#             "MATH": round(m['MATH'], 3),
#             "mode": m['mode'],
#             "hit": "✅" if (m['MATH'] >= self.threshold) == bool(m['ACT']) else "❌",
#             # Ensure keys match the UI requirements
#             "explanation": audit_data.get('narrative', "No narrative generated."),
#             "trace": audit_data.get('trace', [])
#         }
# class BatchProcessor:
#     def __init__(self, pipeline):
#         self.pipeline = pipeline
#         self.results = []
#         self.weight_history = []
#         self.preprocessor = joblib.load(self.pipeline.infra.get_preprocessor_path())
#         self.threshold = 0.30
#
#     async def execute(self, n_samples: int, params: dict = None):
#         df_raw = self.pipeline.handler.fetch_balanced_samples(n_samples)
#
#         for _, row in df_raw.iterrows():
#             raw_dict = row.to_dict()
#
#             # 1. Specialist Inference (The Math)
#             metrics = self._extract_metrics(raw_dict, params)
#
#             # 2. Pillar Weight Calibration (The Learning)
#             self._update_weights(metrics)
#
#             # ---------------------------------------------------------
#             # 🎯 3. RAG Audit (The Contextual Reasoning)
#             # THIS IS WHERE YOUR METHOD IS CALLED
#             # ---------------------------------------------------------
#             explanation = await self._conduct_rag_audit(metrics, raw_dict)
#
#             # 4. Packaging the results for the UI
#             self.results.append(self._package_row(metrics, explanation))
#
#         return pd.DataFrame(self.results)
#
#     def _extract_metrics(self, raw_dict: dict, params: dict = None) -> dict:
#         """Handles the 'Sealed 24' math and dynamic pillar normalization."""
#         raw_df = pd.DataFrame([raw_dict])
#         if 'Unnamed: 0' not in raw_df.columns: raw_df['Unnamed: 0'] = 0
#
#         vector = self.pipeline.infra.extract_model_input(self.preprocessor.transform(raw_df))
#
#         # Pillar Predictions
#         b_p = float(self.pipeline.base_pillar.predict(vector))
#         g_p = float(self.pipeline.gold_pillar.predict(vector))
#         n_raw = float(self.pipeline.neuro_pillar.predict(vector))
#         c_raw = float(self.pipeline.cluster_pillar.predict_raw(vector))
#
#         # 🎯 DYNAMIC CALL: Passing UI params to the Scorer
#         mas_output = self.pipeline.scorer.compute_mas_score(
#             gold_prob=g_p,
#             neuro_mse=n_raw,
#             cluster_dist=c_raw,
#             params=params
#         )
#
#         # 🚀 SYNCED RETURN KEYS: Matches PerformanceEvaluator requirements
#         return {
#             "CC": str(raw_dict.get('cc_num'))[-4:],
#             "ACT": int(raw_dict.get('actual_label', 0)),  # Renamed from 'actual'
#             "BASE": b_p,  # Renamed from 'base_p'
#             "GOLD": g_p,  # Renamed from 'gold_p'
#             "MATH": mas_output['final_score'],  # Renamed from 'math_score'
#             "N_RAW": n_raw,
#             "C_RAW": c_raw,
#             "N_CAL": mas_output['n_p'],  # For the Trinity Audit table
#             "C_CAL": mas_output['c_p'],  # For the Trinity Audit table
#             "mode": mas_output['mode']
#         }
#
#
#     def _update_weights(self, metrics: dict):
#         """Passes performance to the Adapter and tracks weight evolution."""
#         self.pipeline.scorer.adapter.update_performance(
#             metrics['ACT'],  # 🚀 Updated from 'actual'
#             metrics['GOLD'],  # 🚀 Updated from 'gold_p'
#             metrics['N_CAL'],  # 🚀 Updated from 'n_norm'
#             metrics['C_CAL']  # 🚀 Updated from 'c_norm'
#         )
#         self.weight_history.append(self.pipeline.scorer.adapter.get_weights().copy())
#
#     async def _conduct_rag_audit(self, metrics: dict, raw_data: dict) -> str:
#         """Handles Agent communication with exponential backoff."""
#         task_prompt = (
#             f"Audit CC {metrics['CC']}. "  # 🚀 Updated from 'cc'
#             f"MAS Score: {round(metrics['MATH'], 3)}. "  # 🚀 Updated from 'math_score'
#             f"Tx: {raw_data.get('amt')} at {raw_data.get('category')}."
#         )
#
#
#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 audit_result = await asyncio.wait_for(
#                     self.pipeline.orchestrator.execute_phase(
#                         phase_name=f"Audit_{metrics['cc']}",
#                         runner_factory=lambda: self.pipeline.factory.get_rag_audit_team(
#                             self.pipeline.vector_service
#                         ).get_team(),
#                         task=task_prompt
#                     ), timeout=90.0
#                 )
#                 return self.pipeline._parse_agent_explanation(audit_result.messages[-1].content)
#
#             except Exception as e:
#                 if "429" in str(e) and attempt < max_retries - 1:
#                     wait_time = (attempt + 1) * 12 # 12s, 24s...
#                     print(f"⚠️ 429 Rate Limit hit for {metrics['cc']}. Retrying in {wait_time}s...")
#                     await asyncio.sleep(wait_time)
#                     continue
#                 return f"Neural Math complete (Audit unavailable: {str(e)[:15]})"
#
#     def _package_row(self, m: dict, explanation: str) -> dict:
#         """Standardizes the output keys for the PerformanceEvaluator and SQL."""
#         return {
#             "CC": m['CC'],  # 🚀 Updated from 'cc_num'
#             "ACT": m['ACT'],  # 🚀 Updated from 'actual_label'
#             "BASE": round(m['BASE'], 3),
#             "GOLD": round(m['GOLD'], 3),
#             "MATH": round(m['MATH'], 3),
#             "N_RAW": round(m['N_RAW'], 3),
#             "N_CAL": round(m['N_CAL'], 3),
#             "C_RAW": round(m['C_RAW'], 3),
#             "C_CAL": round(m['C_CAL'], 3),
#             "mode": m['mode'],
#             "hit": "✅" if (m['MATH'] >= self.threshold) == bool(m['ACT']) else "❌",
#             "explanation": explanation
#         }