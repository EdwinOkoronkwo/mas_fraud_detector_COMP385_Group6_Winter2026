
import io
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter

from rag.tools.rag_tools import CHAMPIONS


class ReportGenerator:
    THRESHOLD = 0.25

    def _calculate_accuracy(self, results_list: list, score_key: str, threshold: float = THRESHOLD) -> float:
        if not results_list: return 0.0

        # Define internal thresholds for raw values to prevent 0% accuracy during observation
        # Once you have your data, update these numbers.
        INTERNAL_MSE_THRESH = 1.0  # Placeholder for Neural MSE
        INTERNAL_DIST_THRESH = 3.0  # Placeholder for Clustering Dist

        matches = 0
        for res in results_list:
            score = float(res.get(score_key, 0.0))

            # Apply specific threshold based on the pillar type
            current_thresh = threshold
            if score_key == "neural": current_thresh = INTERNAL_MSE_THRESH
            if score_key == "clustering": current_thresh = INTERNAL_DIST_THRESH

            pred = "FRAUD" if score >= current_thresh else "NORMAL"
            if pred == res["Actual"]:
                matches += 1
        return (matches / len(results_list)) * 100

    def print_detailed_reports(self, results_list: list, current_weights=None) -> str:
        report_buffer = io.StringIO()
        models = {
            "Manual Baseline (XGB)": "baseline",
            "Supervised Pillar (XGB/RF)": "supervised",
            "Neural Pillar (VAE/RNN)": "neural",
            "Clustering Pillar (DBSCAN)": "clustering",
            "AI (Final Ensemble)": "total"
        }

        for title, key in models.items():
            report_buffer.write(f"\n📊 {title} REPORT\n")
            table_data = []
            for res in results_list:
                score = float(res.get(key, 0.0))
                is_final_ensemble = (key == "total")
                is_supervised = (key in ["baseline", "supervised"])

                # --- NEW RAW OBSERVABILITY LOGIC ---
                if is_final_ensemble:
                    hitl_floor = self.THRESHOLD - 0.05
                    if score >= self.THRESHOLD:
                        pred, status_icon = "FRAUD", ("✅" if res["Actual"] == "FRAUD" else "❌")
                    elif score >= hitl_floor:
                        pred, status_icon = "HITL_PENDING", "⏳ HITL"
                    else:
                        pred, status_icon = "NORMAL", ("✅" if res["Actual"] == "NORMAL" else "❌")

                elif is_supervised:
                    pred = "FRAUD" if score >= self.THRESHOLD else "NORMAL"
                    status_icon = "✅" if pred == res["Actual"] else "❌"

                else:
                    # FOR NEURAL AND CLUSTERING: Show raw values without forced binary labels
                    # We compare to res["Actual"] just for the icon, but label as RAW
                    # Note: We assume score > 1.0 is a typical fraud spike for MSE/Dist
                    is_anomalous = score > 1.0
                    pred = "RAW_VAL"
                    # Icon shows if the raw spike aligns with actual fraud
                    if res["Actual"] == "FRAUD":
                        status_icon = "🔥 SPIKE" if is_anomalous else "☁️ LOW_ERR"
                    else:
                        status_icon = "✅ NORMAL" if not is_anomalous else "❌ NOISE"

                table_data.append({
                    "CC (Last 4)": res["CC (Last 4)"],
                    "Amount": res["Amount"],
                    "Actual": res["Actual"],
                    "Score": f"{score:.4f}",  # Preserves precision for small MSEs
                    "Verdict": pred,
                    "Result": status_icon,
                    "AI Reasoning": res.get("Notes", "N/A")
                })

            report_buffer.write(tabulate(table_data, headers="keys", tablefmt="psql") + "\n")

        # 2. Final Accuracy Scoreboard
        report_buffer.write("\n🏆 FINAL ACCURACY SCOREBOARD\n" + "═" * 45 + "\n")
        accuracy_data = []
        chart_labels = []
        chart_values = []

        # 2. Update the Pillars list for the Scoreboard
        pillars = [
            ("Manual Baseline", "baseline"),  # ADDED
            ("Supervised", "supervised"),
            ("Neural", "neural"),
            ("Clustering", "clustering")
        ]

        for pillar_name, key in pillars:
            acc = self._calculate_accuracy(results_list, key, threshold=self.THRESHOLD)

            # DYNAMIC NAME EXTRACTION
            path = CHAMPIONS.get(key, "N/A")
            actual_model = path.split('_')[-1].split('.')[0].upper() if "_" in path else "UNKNOWN"

            accuracy_data.append({
                "Pillar": pillar_name,
                "Champion Model": actual_model,
                "Accuracy": f"{acc:.2f}%"
            })

            chart_labels.append(f"{pillar_name}\n({actual_model})")
            chart_values.append(acc)

        # Add the Final AI Ensemble score
        ai_acc = self._calculate_accuracy(results_list, "total", threshold=self.THRESHOLD)
        accuracy_data.append({
            "Pillar": "FINAL ENSEMBLE",
            "Champion Model": "LogicGatekeeper",
            "Accuracy": f"{ai_acc:.2f}%"
        })
        chart_labels.append("AI ENSEMBLE")
        chart_values.append(ai_acc)

        report_buffer.write(tabulate(accuracy_data, headers="keys", tablefmt="fancy_grid") + "\n")

        # 2.5 ADAPTIVE WEIGHT RECALIBRATION
        if current_weights:
            report_buffer.write("\n🔄 ADAPTIVE WEIGHT RECALIBRATION\n")
            sorted_weights = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)
            weight_table = []
            for pillar, val in sorted_weights:
                trend = "▲" if val > 0.35 else "▼" if val < 0.25 else "—"
                weight_table.append({
                    "Pillar": pillar.capitalize(),
                    "Contribution": f"{val * 100:.1f}%",
                    "Status": "Leader" if val == max(current_weights.values()) else trend
                })
            report_buffer.write(tabulate(weight_table, headers="keys", tablefmt="simple") + "\n")
            top_p = max(current_weights, key=current_weights.get).upper()
            report_buffer.write(f"💡 STRATEGY: System is currently prioritizing {top_p} patterns.\n")
            report_buffer.write("═" * 45 + "\n")

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

        # Color palette: Grey for Baseline, Professional colors for AI
        colors = ['#808080', '#aec6cf', '#ffb347', '#b39eb5', '#77dd77']

        # --- Chart 1: Accuracy ---
        ax1.bar(labels, accuracy_values, color=colors)
        ax1.set_title('Detection Accuracy (%)', fontweight='bold')

        # Find the baseline value to draw a "Comparison Line"
        baseline_acc = accuracy_values[0] if accuracy_values else 50
        ax1.axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.6, label=f'Baseline ({baseline_acc:.1f}%)')
        ax1.legend()

        # --- Chart 2: Net Financial Value ($) ---
        roi_values = []
        # Added 'baseline' to the ROI calculation loop
        keys = ["baseline", "supervised", "neural", "clustering", "total"]

        for key in keys:
            acc = self._calculate_accuracy(results_list, key) / 100.0
            # Utility Weight: (Accuracy - Baseline) / (Max - Baseline)
            utility = max(0, (acc - 0.5) / 0.5)

            prevented = 0.0
            friction = 0.0

            for res in results_list:
                # Robust currency cleaning
                raw_amt = res.get("Amount", "0")
                amt = float(str(raw_amt).replace('$', '').replace(',', ''))

                # Prediction logic based on the 0.5 decision boundary
                score = float(res.get(key, 0.0))
                pred = "FRAUD" if score > 0.5 else "NORMAL"

                if pred == "FRAUD" and res["Actual"] == "FRAUD":
                    prevented += amt
                elif pred == "FRAUD" and res["Actual"] == "NORMAL":
                    friction += amt

            # ROI = (Money Saved - Money Wasted on False Positives) * Model Confidence
            roi_values.append((prevented - friction) * utility)

        ax2.bar(labels, roi_values, color=colors)
        ax2.set_title('Net System Value (Utility Weighted $)', fontweight='bold')
        ax2.axhline(y=0, color='black', linewidth=0.8)  # Baseline for profit/loss

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
        # Calculate for Ensemble
        e_prevented, e_friction = self._calculate_roi(results_list, "total")

        # Calculate for Baseline
        # Note: Ensure "baseline" exists in your results_list from the Pipeline!
        b_prevented, b_friction = self._calculate_roi(results_list, "baseline")

        net_ensemble = e_prevented - e_friction
        net_baseline = b_prevented - b_friction
        ai_lift = net_ensemble - net_baseline

        lines = [
            "\n💰 FINANCIAL IMPACT ANALYSIS",
            "═" * 45,
            f"Total Fraud Prevented (AI): ${e_prevented:,.2f}",
            f"Total Friction Cost (AI):   ${e_friction:,.2f}",
            f"NET AI VALUE:               ${net_ensemble:,.2f}",
            "─" * 45,
            f"NET BASELINE VALUE:         ${net_baseline:,.2f}",
            f"🚀 STRATEGIC AI LIFT:        ${ai_lift:,.2f}",
            "═" * 45
        ]
        return "\n".join(lines)

    def _calculate_roi(self, results_list: list, key: str) -> tuple:
        """
        Helper to calculate financial impact for a specific pillar or the ensemble.
        Returns: (total_fraud_prevented, total_friction_cost)
        """
        prevented = 0.0
        friction = 0.0

        for res in results_list:
            # Clean the amount string (removes $ and ,)
            raw_amt = res.get("Amount", "0")
            amt = float(str(raw_amt).replace('$', '').replace(',', ''))

            # Use the class threshold to determine if the model 'flagged' it
            score = float(res.get(key, 0.0))
            is_flagged = score >= self.THRESHOLD

            # Logic:
            # 1. Flagged + Actually Fraud = Prevention (Win)
            # 2. Flagged + Actually Normal = Friction (Loss)
            if is_flagged:
                if res["Actual"] == "FRAUD":
                    prevented += amt
                else:
                    friction += amt

        return prevented, friction

    def get_model_display_name(pillar_key):
        """
        Extracts 'XGB' or 'VAE' from 'models/champion_xgb.joblib'
        """
        path = CHAMPIONS.get(pillar_key, "Unknown")
        # Splits path by '/' and '_' to find the actual name bit
        try:
            filename = path.split('/')[-1]  # champion_xgb.joblib
            name_part = filename.split('_')[-1]  # xgb.joblib
            return name_part.split('.')[0].upper()  # XGB
        except:
            return pillar_key.upper()

    # Then, in your ReportGenerator:
    models = {
        f"{get_model_display_name('supervised')} (Supervised)": "supervised",
        f"{get_model_display_name('neural')} (Neural)": "neural",
        f"{get_model_display_name('clustering')} (Clustering)": "clustering",
        "AI (Final Ensemble)": "total"
    }

