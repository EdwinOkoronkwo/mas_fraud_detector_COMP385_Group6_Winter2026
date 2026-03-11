import pandas as pd

import pandas as pd

import pandas as pd


class ReportCard:
    @staticmethod
    def generate_summary(df: pd.DataFrame):
        """Calculates performance metrics across Baseline, Trinity, and RAG-Hybrid."""
        total = len(df)
        fraud_df = df[df['ACT'] == 1]

        # 1. Calculate Hits for all three tracks
        # Note: We ensure the DataFrame has these 'HIT' columns from our pipeline loop
        df['TRINITY_HIT'] = (df['MATH'] >= 0.3) == df['ACT']
        df['HYBRID_HIT'] = (df['FINAL'] >= 0.3) == df['ACT']
        # BASE_HIT should already be in your DF from the base_pillar prediction

        # 2. Accuracy Calculations
        base_acc = df['BASE_HIT'].map({'✅': True, '❌': False}).mean() if isinstance(df['BASE_HIT'].iloc[0], str) else \
        df['BASE_HIT'].mean()
        trin_acc = df['TRINITY_HIT'].mean()
        hyb_acc = df['HYBRID_HIT'].mean()

        # 3. Recall Calculations (Fraud Detection Rate)
        fraud_df = df[df['ACT'] == 1]
        base_rec = (fraud_df['BASE_HIT'] == "✅").mean() if not fraud_df.empty else 0
        trin_rec = (fraud_df['TRINITY_HIT']).mean() if not fraud_df.empty else 0
        hyb_rec = (fraud_df['HYBRID_HIT']).mean() if not fraud_df.empty else 0

        print("\n" + "=" * 85)
        print(f"{'ULTIMATE HYBRID PERFORMANCE MATRIX (2026.1)':^85}")
        print("=" * 85)
        print(f"{'Metric':<25} | {'Baseline':<12} | {'MAS (Trinity)':<15} | {'RAG Hybrid':<12}")
        print("-" * 85)
        print(f"{'Overall Accuracy':<25} | {base_acc:>11.1%} | {trin_acc:>14.1%} | {hyb_acc:>12.1%}")
        print(f"{'Fraud Recall (Sens.)':<25} | {base_rec:>11.1%} | {trin_rec:>14.1%} | {hyb_rec:>12.1%}")
        print("-" * 85)

        # 4. Strategic Insight
        lift = (hyb_rec - base_rec) * 100
        print(f"STRATEGIC LIFT: Agentic RAG improved Fraud Capture by {lift:+.1f}% over Baseline.")
        print("=" * 85)

        return df