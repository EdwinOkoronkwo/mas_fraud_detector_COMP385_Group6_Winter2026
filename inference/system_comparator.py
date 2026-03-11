import pandas as pd


class SystemComparator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    async def run_ab_test(self, n_samples=10):
        print("\n" + "═" * 60)
        print("🧪 STARTING A/B TEST: SINGLE-MODEL vs. MULTI-AGENT SYSTEM")
        print("═" * 60)

        rows = self.pipeline.data_handler.get_inference_data(n_samples)
        comparison_results = []
        all_logs = []  # To collect the missing XAI logs

        for _, row in rows.iterrows():
            mas_result = await self.pipeline.process_single_transaction(row)

            # --- LEGACY LOGIC ---
            raw_scores = mas_result.get('raw_scores', {'supervised': 0.0})
            legacy_score = raw_scores.get('supervised', 0.0)
            legacy_decision = "DECLINE" if legacy_score > 0.3 else "APPROVE"

            actual_label = "FRAUD" if row['is_fraud'] == 1 else "NORMAL"
            mas_correct = mas_result['final_decision'] == actual_label
            legacy_correct = legacy_decision == actual_label

            # --- LOG CAPTURE ---
            # Extract the explanation from the agent metadata
            explanation = mas_result.get('explanation', "No log provided by agent.")
            all_logs.append(f"Card ...{str(row['cc_num'])[-4:]}: {explanation}")

            comparison_results.append({
                "CC_Last4": str(row['cc_num'])[-4:],
                "Actual": actual_label,
                "Legacy_Decision": legacy_decision,
                "MAS_Decision": mas_result['final_decision'],
                "MAS_Benefit": "✅" if (mas_correct and not legacy_correct) else (
                    "❌" if (not mas_correct and legacy_correct) else "—")
            })

        self._print_comparison_report(comparison_results, all_logs)

    def _print_comparison_report(self, results, logs):
        df = pd.DataFrame(results)

        # 1. Show the Battle Table
        print("\n📊 DECISION COMPARISON")
        print(df.to_string(index=False))

        # 2. Show the missing XAI Logs
        print("\n" + "═" * 60)
        print("🧠 DETAILED AI EXPLAINABILITY (XAI) LOGS")
        print("═" * 60)
        for log in logs:
            print(log)

        # 3. Final Scoreboard
        legacy_acc = (df['Legacy_Decision'] == df['Actual']).mean() * 100
        mas_acc = (df['MAS_Decision'] == df['Actual']).mean() * 100
        print(f"\n🏆 LIFT ANALYSIS: MAS is {mas_acc - legacy_acc:+.1f}% more accurate than Single Model.")