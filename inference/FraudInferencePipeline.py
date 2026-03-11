import os
from datetime import datetime
from config.llm_config import get_model_client
from inference.LogicGate import LogicGatekeeper
from inference.audit_cli import AuditCLI
from inference.performance_tracker import PerformanceTracker
from inference.report_generator import ReportGenerator
from inference.agent_orchestrator import AgentOrchestrator
from inference.data_handler import DataHandler
from inference.task_architect import TaskArchitect
from inference.transaction_processor import TransactionProcessor
from inference.verdict_analyst import VerdictAnalystAgent
from inference.xai_logger import XAILogger
from rag.tools.rag_tools import REGISTRY, scale_transaction_data, predict_baseline_shadow, extract_detailed_scores


class FraudInferencePipeline:
    def __init__(self, investigator_team, db_path: str):
        # 1. Base Engines
        self.data_handler = DataHandler(db_path)
        self.gatekeeper = LogicGatekeeper(registry_weights=REGISTRY.get("ensemble_weights"))
        self.reporter = ReportGenerator()

        # 2. Composite: Strategy/Execution Logic
        self.processor = TransactionProcessor(
            architect=TaskArchitect(),
            orchestrator=AgentOrchestrator(investigator_team),
            gatekeeper=self.gatekeeper,
            analyst=VerdictAnalystAgent(get_model_client()),
            threshold=self.reporter.THRESHOLD
        )

        # 3. Composite: Utilities
        self.xai_logger = XAILogger()
        self.perf_tracker = PerformanceTracker(threshold=self.reporter.THRESHOLD)
        self.cli = AuditCLI()

    async def run_batch_inference(self, n_samples=10):
        print(f"📊 STARTING STRICT BATCH INFERENCE: {n_samples} Samples")
        df = self.data_handler.fetch_balanced_samples(n_samples)
        results = []

        for _, row in df.iterrows():
            cc_last4 = str(row['cc_num'])[-4:]

            # 1. Prepare Data
            scaled_row = scale_transaction_data(row.to_dict())

            try:
                # 2. Execute Agent Pipeline
                tx_res = await self.processor.execute(row, scaled_row)

                # 3. Secure Mathematical Scores
                f_scores = tx_res.get('scores')

                # If scores are missing or have a None/0.0 total,
                # and logic hasn't already crashed, we force the verification here.
                if not f_scores or f_scores.get('total') is None:
                    raise ValueError("❌ PIPELINE BREAK: Processor returned empty scores dictionary.")

                # 4. Success Path
                baseline_score = predict_baseline_shadow(scaled_row)
                actual_fraud = row['actual_label'] == 1

                # Save logs and update metrics
                path = self.xai_logger.save_report(cc_last4, tx_res['xai_report'], f_scores)
                self.perf_tracker.update(f_scores, actual_fraud)

                # 1. Separate the numeric scores from the metadata strings
                numeric_scores = {k: f"{v:.4f}" for k, v in f_scores.items() if isinstance(v, (int, float))}

                threshold = self.reporter.THRESHOLD  # Assuming reporter is instance of ReportGenerator

                results.append({
                    "CC (Last 4)": f"...{cc_last4}",
                    "Amount": f"${row['amt']:.2f}",
                    "Actual": "FRAUD" if actual_fraud else "NORMAL",
                    "baseline": f"{baseline_score:.4f}",
                    **numeric_scores,
                    "Verdict": "FRAUD" if f_scores['total'] > threshold else "NORMAL",
                    "Notes": self.gatekeeper.generate_ai_insight(f_scores),
                    "Override": f_scores.get('override', 'None')
                })

            except Exception as e:
                # 🚨 THE DIAGNOSTIC CAPTURE
                print("\n" + "!" * 60)
                print(f"CRITICAL ERROR ENCOUNTERED ON CC ...{cc_last4}")
                print(f"ERROR TYPE: {type(e).__name__}")
                print(f"ERROR MESSAGE: {e}")

                # Attempt to show the last 500 characters of the AI's response if available
                if 'tx_res' in locals() and 'xai_report' in tx_res:
                    print("\n--- AGENT TEXT PRE-CRASH (FOOTER AREA) ---")
                    print(tx_res['xai_report'][-500:])
                    print("-" * 40)

                print("!" * 60 + "\n")
                # Re-raise so the program stops as you requested
                raise e

        # Recalibrate and Print
        accuracies = self.perf_tracker.get_final_accuracies()
        updated_weights = self.gatekeeper.update_performance_metrics(accuracies)
        self.reporter.print_detailed_reports(results, current_weights=updated_weights)
    # async def run_batch_inference(self, n_samples=10):
    #     df = self.data_handler.fetch_balanced_samples(n_samples)
    #     results = []
    #
    #     for _, row in df.iterrows():
    #         # 1. Execute Multi-Agent Investigation
    #         tx_res = await self.processor.execute(row)
    #
    #         # 2. SUCCESS: Capture the scores directly from the processor's memory
    #         # This replaces the extract_detailed_scores/regex logic
    #         f_scores = tx_res['scores']
    #
    #         actual_fraud = row['actual_label'] == 1
    #         cc_last4 = str(row['cc_num'])[-4:]
    #
    #         # 3. Save logs (Fixing the TypeError from before)
    #         path = self.xai_logger.save_report(cc_last4, tx_res['xai_report'], f_scores)
    #
    #         # 4. Update accuracy tracker with REAL numbers
    #         self.perf_tracker.update(f_scores, actual_fraud)
    #
    #         # 5. Get the Baseline (Which you confirmed works!)
    #         scaled_row = scale_transaction_data(row.to_dict())
    #         baseline_score = predict_baseline_shadow(scaled_row)
    #
    #         # 6. Append to results
    #         results.append({
    #             "CC (Last 4)": f"...{cc_last4}",
    #             "Amount": f"${row['amt']:.2f}",
    #             "Actual": "FRAUD" if actual_fraud else "NORMAL",
    #             "baseline": baseline_score,
    #             **f_scores,  # This will now unpack REAL scores into the table
    #             "Notes": self.gatekeeper.generate_ai_insight(f_scores, self.gatekeeper.weights),
    #             "xai_path": path,
    #             "xai_summary": self.xai_logger.get_summary(tx_res['xai_report'])
    #         })
    #
    #     # 7. Recalibrate weights and print
    #     accuracies = self.perf_tracker.get_final_accuracies()
    #     updated_weights = self.gatekeeper.update_performance_metrics(accuracies)
    #     self.reporter.print_detailed_reports(results, current_weights=updated_weights)
