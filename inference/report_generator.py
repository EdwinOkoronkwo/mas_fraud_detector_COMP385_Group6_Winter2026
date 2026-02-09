
import io
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter


class ReportGenerator:
    """
    Handles deep-dive reporting, accuracy metrics, and financial ROI auditing.
    Enforces the 'Accuracy = Savings' rule.
    """

    def _calculate_accuracy(self, results_list: list, score_key: str, threshold: float = 0.5) -> float:
        if not results_list: return 0.0
        matches = 0
        for res in results_list:
            # Handle both string and float scores
            score = float(res.get(score_key, 0.0))
            pred = "FRAUD" if score > threshold else "NORMAL"
            if pred == res["Actual"]:
                matches += 1
        return (matches / len(results_list)) * 100

    def print_detailed_reports(self, results_list: list) -> str:
        report_buffer = io.StringIO()
        models = {
            "LR (Logistic Regression)": "LR",
            "RNN (Neural Sequence)": "RNN",
            "DB (DBSCAN Clustering)": "DB",
            "AI (Final Ensemble)": "AI Score"
        }

        # 1. Performance Tables
        for title, key in models.items():
            report_buffer.write(f"\n📊 {title} REPORT\n")
            table_data = []
            for res in results_list:
                score = float(res.get(key, 0.0))
                pred = "FRAUD" if score > 0.5 else "NORMAL"
                table_data.append({
                    "CC (Last 4)": res["CC (Last 4)"],
                    "Amount": res["Amount"],
                    "Actual": res["Actual"],
                    "Score": f"{score:.4f}",
                    "Result": "✅" if pred == res["Actual"] else "❌"
                })
            report_buffer.write(tabulate(table_data, headers="keys", tablefmt="psql") + "\n")

        # 2. Final Accuracy Scoreboard
        report_buffer.write("\n🏆 FINAL ACCURACY SCOREBOARD\n" + "═" * 45 + "\n")
        accuracy_data = []
        chart_labels, chart_values = [], []

        for t, k in models.items():
            acc = self._calculate_accuracy(results_list, k)
            accuracy_data.append({"Model Component": t, "Accuracy": f"{acc:.2f}%"})
            chart_labels.append(t.split(" (")[0])
            chart_values.append(acc)

        report_buffer.write(tabulate(accuracy_data, headers="keys", tablefmt="fancy_grid") + "\n")

        # 3. Visualizations & ROI
        try:
            self._generate_performance_charts(chart_labels, chart_values, results_list)
            heatmap_status = self._generate_risk_heatmap(results_list)
            report_buffer.write(f"\n📈 [Charts Generated: performance_report.png, risk_heatmap.png]\n")
            report_buffer.write(f"🗺️ {heatmap_status}\n")
        except Exception as e:
            report_buffer.write(f"\n⚠️ Visualization Failed: {str(e)}\n")

        report_buffer.write(str(self.print_financial_impact(results_list)) + "\n")

        final_report = report_buffer.getvalue()
        print(final_report)
        return final_report

    def _generate_performance_charts(self, labels, accuracy_values, results_list):
        plt.switch_backend('Agg')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['#aec6cf', '#ffb347', '#b39eb5', '#77dd77']

        # --- Chart 1: Accuracy ---
        ax1.bar(labels, accuracy_values, color=colors)
        ax1.set_title('Detection Accuracy (%)', fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')

        # --- Chart 2: Net Financial Value ($) ---
        # APPLYING THE UTILITY WEIGHT: (Acc - 0.5) / 0.5
        roi_values = []
        keys = ["LR", "RNN", "DB", "AI Score"]

        for key in keys:
            acc = self._calculate_accuracy(results_list, key) / 100.0
            utility = max(0, (acc - 0.5) / 0.5)

            prevented = 0.0
            friction = 0.0
            for res in results_list:
                amt = float(str(res.get("Amount", "0")).replace('$', '').replace(',', ''))
                pred = "FRAUD" if float(res[key]) > 0.5 else "NORMAL"
                if pred == "FRAUD" and res["Actual"] == "FRAUD":
                    prevented += amt
                elif pred == "FRAUD" and res["Actual"] == "NORMAL":
                    friction += amt

            roi_values.append((prevented - friction) * utility)

        ax2.bar(labels, roi_values, color=colors)
        ax2.set_title('Net System Value (Utility Weighted $)', fontweight='bold')

        plt.tight_layout()
        plt.savefig("performance_report.png")
        plt.close()

    def _generate_risk_heatmap(self, results_list):
        plt.switch_backend('Agg')
        amounts = [float(str(res.get("Amount", "0")).replace('$', '').replace(',', '')) for res in results_list]
        scores = [float(res.get("AI Score", 0)) for res in results_list]

        plt.figure(figsize=(10, 6))
        plt.axhspan(0, 0.25, color='green', alpha=0.05, label='Safe Zone')
        plt.axhspan(0.25, 0.75, color='yellow', alpha=0.05, label='Ambiguity')
        plt.axhspan(0.75, 1.0, color='red', alpha=0.05, label='Danger')

        plt.scatter(amounts, scores, c=amounts, cmap='viridis', edgecolors='k', alpha=0.7)
        plt.axhline(y=0.5, color='black', linestyle='--')
        plt.title('AI Risk Topology')
        plt.savefig("risk_heatmap.png")
        plt.close()
        return "Heatmap generated successfully."

    def print_financial_impact(self, results_list):
        # ROI based on AI Score
        prevented, friction = 0.0, 0.0
        for res in results_list:
            amt = float(str(res.get("Amount", "0")).replace('$', '').replace(',', ''))
            pred = "FRAUD" if float(res["AI Score"]) > 0.5 else "NORMAL"
            if pred == "FRAUD" and res["Actual"] == "FRAUD":
                prevented += amt
            elif pred == "FRAUD" and res["Actual"] == "NORMAL":
                friction += amt

        lines = ["\n💰 FINANCIAL IMPACT ANALYSIS", "═" * 45]
        data = [["Net AI Value", f"${(prevented - friction):,.2f}"]]
        lines.append(tabulate(data, tablefmt="fancy_grid"))
        return "\n".join(lines)

