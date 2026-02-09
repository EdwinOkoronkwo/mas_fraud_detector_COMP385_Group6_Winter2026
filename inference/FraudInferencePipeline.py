from inference.LogicGate import LogicGatekeeper
from inference.report_generator import ReportGenerator
from inference.agent_orchestrator import AgentOrchestrator
from inference.data_handler import DataHandler
from inference.task_architect import TaskArchitect


class FraudInferencePipeline:
    """The Master Conductor that composes the specialized inference classes.

    This class implements the 'Facade' pattern, providing a single entry point
    to run an end-to-end fraud investigation pipeline from data to report.
    """

    def __init__(self, investigator_team, db_path: str):
        """Assembles the pipeline by injecting necessary dependencies.

        Args:
            investigator_team: The initialized RAGTeam MAS instance.
            db_path: Path to the SQLite transaction database.
        """
        self.data_handler = DataHandler(db_path)
        self.architect = TaskArchitect()
        self.orchestrator = AgentOrchestrator(investigator_team)
        self.gatekeeper = LogicGatekeeper()
        self.reporter = ReportGenerator()

    async def run_batch_inference(self, n_samples=10):
        df = self.data_handler.fetch_balanced_samples(n_samples)
        results = []

        # Initialize performance tracking
        performance = {'lr': {'corr': 0, 'total': 0},
                       'rnn': {'corr': 0, 'total': 0},
                       'db': {'corr': 0, 'total': 0},
                       'ai': {'corr': 0, 'total': 0}}

        for _, row in df.iterrows():
            task = self.architect.create_investigation_task(row)
            raw_output = await self.orchestrator.execute_with_resilience(task)

            # 1. Define actuals immediately to avoid NameError
            actual_is_fraud = row['actual_label'] == 1
            actual_str = "FRAUD" if actual_is_fraud else "NORMAL"  # <--- DEFINED HERE

            # 2. Get Initial Verdict
            final_scores = self.gatekeeper.apply_final_verdict(raw_output, row['amt'])

            # 3. Update Model Performance
            for m in ['lr', 'rnn', 'db']:
                m_score = final_scores.get(m, 0.0)
                is_correct = (m_score > 0.5) == actual_is_fraud
                if is_correct: performance[m]['corr'] += 1
                performance[m]['total'] += 1

            # 4. Update AI Performance
            ai_is_correct = (final_scores.get('Total', 0.0) > 0.5) == actual_is_fraud
            if ai_is_correct: performance['ai']['corr'] += 1
            performance['ai']['total'] += 1

            # 5. Feed stats back and RE-RUN verdict to ensure AI Lead on report
            new_acc_snapshot = {m: (p['corr'] / p['total']) for m, p in performance.items()}
            self.gatekeeper.update_performance_metrics(new_acc_snapshot)

            # Re-fetch with updated weights
            final_scores = self.gatekeeper.apply_final_verdict(raw_output, row['amt'])
            predicted = "FRAUD" if final_scores.get('Total', 0.0) > 0.5 else "NORMAL"

            results.append({
                "CC (Last 4)": f"...{str(row['cc_num'])[-4:]}",
                "Amount": f"${row['amt']:.2f}",
                "Actual": actual_str,  # <--- NOW ACCESSIBLE
                "LR": final_scores.get('lr', 0.0),
                "RNN": final_scores.get('rnn', 0.0),
                "DB": final_scores.get('db', 0.0),
                "AI Score": final_scores.get('Total', 0.0),
                "Override": final_scores.get("Override", "None"),
                "Match": "✅" if actual_str == predicted else "❌"
            })

        self.reporter.print_detailed_reports(results)

    # async def run_batch_inference(self, n_samples=10):
    #     df = self.data_handler.fetch_balanced_samples(n_samples)
    #     results = []
    #
    #     for _, row in df.iterrows():
    #         task = self.architect.create_investigation_task(row)
    #         raw_output = await self.orchestrator.execute_with_resilience(task)
    #         final_scores = self.gatekeeper.apply_final_verdict(raw_output, row['amt'])
    #         actual = "FRAUD" if row['actual_label'] == 1 else "NORMAL"
    #         predicted = "FRAUD" if final_scores.get('Total', 0.0) > 0.5 else "NORMAL"
    #
    #         results.append({
    #             "CC (Last 4)": f"...{str(row['cc_num'])[-4:]}",
    #             "Amount": f"${row['amt']:.2f}",
    #             "Actual": actual,
    #             "LR": f"{final_scores.get('lr', 0.0):.4f}",
    #             "RNN": f"{final_scores.get('rnn', 0.0):.4f}",
    #             "DB": f"{final_scores.get('db', 0.0):.4f}",
    #             "AI Score": f"{final_scores.get('Total', 0.0):.4f}",
    #             "Override": final_scores.get("Override", "None"),  # 🟢 CAPTURE THIS!
    #             "Match": "✅" if actual == predicted else "❌"
    #         })
    #     self.reporter.print_detailed_reports(results)