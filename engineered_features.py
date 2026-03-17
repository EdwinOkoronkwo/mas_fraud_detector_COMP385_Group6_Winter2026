import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os


def audit_group6_features(db_path):
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)

    try:
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
        print(f"✅ Connection Successful: {db_path}")
    except Exception as e:
        print(f"❌ Error accessing table: {e}")
        return

    # 1. Separate Columns
    all_columns = df.columns.tolist()
    is_fraud_col = 'is_fraud' if 'is_fraud' in df.columns else None

    # Identify prefixes from Sklearn ColumnTransformer
    num_features = [c for c in all_columns if c.startswith('num__')]
    cat_features = [c for c in all_columns if c.startswith('cat__')]
    behavioral = [c for c in all_columns if any(x in c for x in ['ratio', 'velocity', 'risk'])]

    print("\n📝 --- ARCHITECTURE SUMMARY ---")
    print(f"Total Input Dimension: {len(all_columns) - (1 if is_fraud_col else 0)}")
    print(f"Numerical Features:    {len(num_features)}")
    print(f"Categorical (OHE):     {len(cat_features)}")
    print(f"Behavioral (Induced):  {len(behavioral)}")

    # 2. THE FULL MANIFEST
    print("\n📋 --- COMPLETE FEATURE LIST ---")
    for i, col in enumerate(all_columns):
        if col == is_fraud_col: continue
        marker = "⭐ [BEHAVIORAL]" if col in behavioral else "  "
        print(f"{i + 1:02}. {marker} {col}")

    # 3. NumPy Statistical Sample
    if all_columns:
        # Check a slice of 5 features for scaling health
        sample_cols = all_columns[:5]
        data_matrix = df[sample_cols].to_numpy()
        means = np.mean(data_matrix, axis=0)
        stds = np.std(data_matrix, axis=0)

        print("\n⚖️ --- SCALING SAMPLE (NUMPY) ---")
        for i, col in enumerate(sample_cols):
            print(f"{col:25} | Mean: {means[i]:>7.4f} | Std: {stds[i]:>7.4f}")

    if len(behavioral) == 0:
        print("\n⚠️  ALERT: No behavioral features detected in final table.")
        print("Check if preprocessing_tool is dropping columns not in its hardcoded num_cols list.")

    print("\n🚀 AUDIT COMPLETE.")


if __name__ == "__main__":
    db_path = r'C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\data\database.sqlite'
    audit_group6_features(db_path)