import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick # For percentage formatting

class PerformanceEvaluator:
    @staticmethod
    def generate_report(df: pd.DataFrame):
        if df.empty: return "No data available."
        threshold = 0.30

        def calculate_metrics(prefix):
            tp = ((df[prefix] >= threshold) & (df['ACT'] == 1)).sum()
            tn = ((df[prefix] < threshold) & (df['ACT'] == 0)).sum()
            fp = ((df[prefix] >= threshold) & (df['ACT'] == 0)).sum()
            fn = ((df[prefix] < threshold) & (df['ACT'] == 1)).sum()
            acc = (tp + tn) / len(df)
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            return acc, pre, rec, f1, fp, fn

        m_acc, m_pre, m_rec, m_f1, m_fp, m_fn = calculate_metrics('MATH')
        b_acc, b_pre, b_rec, b_f1, b_fp, b_fn = calculate_metrics('BASE')

        report = [
            "\n" + "═" * 60,
            f"{'COMPREHENSIVE PERFORMANCE COMPARISON':^60}",
            "═" * 60,
            f"{'METRIC':<25} | {'BASELINE':<15} | {'MAS (NEURAL)':<15}",
            "-" * 60,
            f"{'Accuracy':<25} | {b_acc:>14.2%} | {m_acc:>14.2%}",
            f"{'Recall (Capture)':<25} | {b_rec:>14.2%} | {m_rec:>14.2%}",
            f"{'F1-Score':<25} | {b_f1:>14.2%} | {m_f1:>14.2%}",
            "-" * 60,
            f"{'False Negatives':<25} | {b_fn:>14} | {m_fn:>14}",
            "═" * 60
        ]
        return "\n".join(report)

    @staticmethod
    def save_to_csv(results_df: pd.DataFrame, filename="final_fraud_export.csv"):
        results_df.to_csv(filename, index=False)
        print(f"📊 Full audit data exported to {filename}")

    @staticmethod
    def print_calibration_audit(df: pd.DataFrame):
        print("\n" + "═" * 100)
        print(f"{'TRINITY PILLAR CALIBRATION AUDIT (RAW VS CALIB)':^100}")
        print("═" * 100)
        headers = f"{'CC':<6} | {'ACT':<4} | {'Gold':<6} | {'N_Raw':<7} | {'N_Cal':<7} | {'C_Raw':<7} | {'C_Cal':<7} | {'MAS'}"
        print(headers)
        print("-" * 100)
        for _, r in df.head(10).iterrows():
            print(
                f"{r['CC']:<6} | {r['ACT']:<4} | {r['GOLD_P']:<6.3f} | {r['N_RAW']:<7.3f} | {r['N_CAL']:<7.3f} | {r['C_RAW']:<7.3f} | {r['C_CAL']:<7.3f} | {r['MATH']:<6.3f}")
        print("═" * 100)

    @staticmethod
    @staticmethod
    def plot_results(df: pd.DataFrame, save_folder="reports/plots"):
        """
        Surgically updated to match the 'Sealed 24' result keys.
        Removed: C_CAL, N_CAL, GOLD_P
        Added: GOLD, BASE, MATH
        """
        os.makedirs(save_folder, exist_ok=True)
        sns.set_theme(style="whitegrid")
        threshold = 0.30

        # 🚀 THE CRITICAL MAPPING FIX
        # These keys must exist in your run_batch results list
        pillars = {
            'Baseline': 'BASE',
            'Supervised (Gold)': 'GOLD',
            'Final MAS': 'MATH'
        }

        pillar_colors = sns.color_palette("husl", len(pillars))
        metrics_data = []

        for label, col in pillars.items():
            if col not in df.columns:
                print(f"⚠️ Warning: Column {col} missing from results. Skipping plot.")
                continue

            tp = ((df[col] >= threshold) & (df['ACT'] == 1)).sum()
            tn = ((df[col] < threshold) & (df['ACT'] == 0)).sum()
            fp = ((df[col] >= threshold) & (df['ACT'] == 0)).sum()
            fn = ((df[col] < threshold) & (df['ACT'] == 1)).sum()

            acc = (tp + tn) / len(df) if len(df) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0

            metrics_data.append({
                'Pillar': label, 'Accuracy': acc, 'Recall': rec,
                'Precision': pre, 'F1-Score': f1, 'False Negatives': fn
            })

        if not metrics_data:
            print("❌ Error: No valid columns found to plot.")
            return

        m_df = pd.DataFrame(metrics_data)
        target_metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'False Negatives']

        for metric in target_metrics:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Pillar', y=metric, data=m_df, palette=pillar_colors, hue='Pillar', legend=False)
            plt.title(f'Pillar Comparison: {metric}', fontsize=15, fontweight='bold')

            if metric != 'False Negatives':
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                plt.ylim(0, 1.1)
            else:
                plt.ylabel('Count of Missed Frauds')

            for p in ax.patches:
                val = f'{p.get_height():.1%}' if metric != 'False Negatives' else f'{int(p.get_height())}'
                ax.annotate(val, (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.savefig(os.path.join(save_folder, f"metric_{metric.lower().replace(' ', '_')}.png"))
            plt.close()

    @staticmethod
    def plot_executive_results(df: pd.DataFrame, save_folder="reports/executive_plots"):
        """Generates high-level comparison plots comparing strictly Baseline vs. Final MAS."""
        os.makedirs(save_folder, exist_ok=True)
        sns.set_theme(style="whitegrid")
        threshold = 0.30

        # Simplified Mapping: Just the competitors
        competitors = {
            'Baseline (Static)': 'BASE',
            'Final MAS (Collective)': 'MATH'
        }

        # Distinct Colors: Red for Baseline, Green for MAS
        comp_colors = ["#e74c3c", "#2ecc71"]

        metrics_data = []
        for label, col in competitors.items():
            # Calculation logic remains identical for consistency
            tp = ((df[col] >= threshold) & (df['ACT'] == 1)).sum()
            tn = ((df[col] < threshold) & (df['ACT'] == 0)).sum()
            fp = ((df[col] >= threshold) & (df['ACT'] == 0)).sum()
            fn = ((df[col] < threshold) & (df['ACT'] == 1)).sum()

            acc = (tp + tn) / len(df) if len(df) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            pre = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0

            metrics_data.append({
                'System': label,
                'Accuracy': acc,
                'Recall': rec,
                'Precision': pre,
                'F1-Score': f1,
                'False Negatives': fn
            })

        m_df = pd.DataFrame(metrics_data)
        target_metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'False Negatives']

        for metric in target_metrics:
            plt.figure(figsize=(8, 6))

            # Plot only the two systems
            ax = sns.barplot(x='System', y=metric, data=m_df, palette=comp_colors, hue='System', legend=False)

            plt.title(f'Executive Summary: {metric}', fontsize=14, fontweight='bold')

            # Format Y-Axis as Percentage
            if metric != 'False Negatives':
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                plt.ylim(0, 1.1)
                plt.ylabel('Percentage')
            else:
                plt.ylabel('Total Missed Fraud (Count)')

            # Annotate bars with bold values
            for p in ax.patches:
                height = p.get_height()
                val = f'{height:.1%}' if metric != 'False Negatives' else f'{int(height)}'
                ax.annotate(val, (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

            # Clean filename for presentation
            file_name = f"executive_{metric.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(save_folder, file_name), bbox_inches='tight')
            plt.close()

        print(f"✅ Executive plots generated in: {save_folder}")

    @staticmethod
    def plot_weight_evolution(weight_history, df_results, save_folder="reports/plots"):
        """Generates trust curves with integer X-axis and percentage Y-axis."""
        if not weight_history or df_results.empty:
            return

        os.makedirs(save_folder, exist_ok=True)
        sns.set_theme(style="whitegrid")

        history_df = pd.DataFrame(weight_history)
        # Shift index by 1 so it starts at sample #1 instead of #0
        history_df.index = history_df.index + 1

        plt.figure(figsize=(12, 6))

        # Plot curves
        plt.plot(history_df.index, history_df['gold'], label='Supervised (Gold)', color='#3498db', linewidth=2)
        plt.plot(history_df.index, history_df['neuro'], label='Neuro (Anomaly)', color='#e67e22', linewidth=2)
        plt.plot(history_df.index, history_df['cluster'], label='Clustering', color='#9b59b6', linewidth=2)

        # Formatting X-Axis as actual integers
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Formatting Y-Axis as Percentages
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Add red markers for misses
        threshold = 0.30
        misses = df_results[(df_results['ACT'] == 1) & (df_results['MATH'] < threshold)]
        for idx in misses.index:
            plt.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.3)

        plt.title('Dynamic Agent Trust Evolution', fontsize=15, fontweight='bold')
        plt.xlabel('Number of Samples Processed')
        plt.ylabel('Influence Weight (%)')
        plt.legend(loc='best')
        plt.ylim(0, 1.05)

        plt.savefig(os.path.join(save_folder, "agent_trust_curve.png"), bbox_inches='tight')
        plt.close()

    @staticmethod
    def generate_weights_report(adapter):
        """Your terminal table for final trust levels."""
        w = adapter.get_weights()
        report = [
            "\n" + "═" * 40,
            f"{'FINAL AGENT TRUST (WEIGHTS)':^40}",
            "═" * 40,
            f"{'AGENT':<20} | {'INFLUENCE':<15}",
            "-" * 40,
            f"{'Supervised (Gold)':<20} | {w['gold']:>14.2%}",
            f"{'Neuro (Anomaly)':<20} | {w['neuro']:>14.2%}",
            f"{'Clustering':<20} | {w['cluster']:>14.2%}",
            "═" * 40
        ]
        return "\n".join(report)

    @staticmethod
    def print_calibration_audit(df: pd.DataFrame):
        """
        Cleaned audit print: No more 'Raw vs Calib' confusion.
        Shows the core specialists vs the final collective score.
        """
        print("\n" + "═" * 80)
        print(f"{'MAS SYSTEM INTEGRITY AUDIT: FINAL SCORES':^80}")
        print("═" * 80)
        headers = f"{'CC':<6} | {'ACT':<4} | {'Base':<6} | {'Gold':<6} | {'MAS Final'}"
        print(headers)
        print("-" * 80)

        # Display first 10 rows
        for _, r in df.head(10).iterrows():
            print(f"{r['CC']:<6} | {r['ACT']:<4} | {r['BASE']:<6.3f} | {r['GOLD']:<6.3f} | {r['MATH']:<9.3f}")
        print("═" * 80)