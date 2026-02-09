import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensures plots save to files without trying to open a window
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
from pathlib import Path

# --- PROJECT ANCHORING ---
# This ensures we always land in /mas_fraud_detector/processed_data
current_file = Path(__file__).resolve()
try:
    PROJECT_ROOT = [p for p in current_file.parents if p.name == 'mas_fraud_detector'][0]
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except IndexError:
    # Fallback if the folder structure is different than expected
    OUTPUT_DIR = "processed_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_eda_plots_tool(db_path: str):
    """
    Generates Correlation Heatmap and Initial Class Distribution.
    Saves artifacts to the centralized 'processed_data' folder.
    """
    abs_db_path = os.path.abspath(db_path)
    if not os.path.exists(abs_db_path):
        return f"EDA ERROR: Database file not found at {abs_db_path}"

    # Use a check to ensure we don't use a broken URI on Windows
    engine = create_engine(f"sqlite:///{abs_db_path}")

    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM train_transactions", conn)

        if df.empty:
            return "EDA ERROR: Table 'train_transactions' is empty."

        # 1. Feature Correlation
        # We drop non-predictive numeric IDs to keep the heatmap clean
        numeric_df = df.select_dtypes(include=[np.number])
        cols_to_drop = [c for c in ['txn_id', 'Unnamed: 0', 'index'] if c in numeric_df.columns]
        if cols_to_drop:
            numeric_df = numeric_df.drop(columns=cols_to_drop)

        corr_matrix = numeric_df.corr()
        top_corr = corr_matrix['is_fraud'].sort_values(ascending=False).head(4).to_dict()

        # 2. Visuals - Standardizing the save path
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Matrix (10% Sample)")

        corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
        plt.savefig(corr_path)
        plt.close()

        # 3. Stats Summary
        amt_stats = {}
        if 'amt' in df.columns:
            amt_stats = {
                "mean": float(df['amt'].mean()),
                "max": float(df['amt'].max()),
                "std": float(df['amt'].std())
            }

        counts = df['is_fraud'].value_counts().to_dict()

        return (
            f"EDA SUCCESS. "
            f"Artifact saved to: {corr_path}. "
            f"Class Distribution: {counts}. "
            f"Key Stats (Amount): {amt_stats}. "
            f"Top Fraud Correlations: {top_corr}."
        )

    except Exception as e:
        return f"EDA ERROR: {str(e)}"