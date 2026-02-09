import os

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine


def apply_scaling_and_cleaning_tool(db_path: str, target_col: str = 'is_fraud') -> str:
    try:
        # --- 1. DYNAMIC PATH ANCHORING ---
        current_file = Path(__file__).resolve()
        project_root = next(p for p in current_file.parents if p.name == 'mas_fraud_detector')
        output_dir = os.path.join(project_root, "processed_data")
        os.makedirs(output_dir, exist_ok=True)

        # --- 2. DATABASE CONNECTION ---
        abs_db_path = os.path.abspath(db_path)
        engine = create_engine(f"sqlite:///{abs_db_path}")

        # --- 3. DATA LOADING ---
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM train_transactions", conn)

        if df.empty:
            return "PREPROCESS ERROR: 'train_transactions' table is empty."

        # Separate target and features
        y = df[target_col]
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')

        # Drop junk columns if they exist
        junk_cols = ['txn_id', 'Unnamed: 0', 'index']
        X = X.drop(columns=[c for c in junk_cols if c in X.columns])

        # --- 4. SCALE ---
        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X)

        # --- ADD THIS: SAVE THE SCALER FOR BRANDON ---
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        # ---------------------------------------------

        # --- 5. RECONSTRUCT & SAVE ---
        cleaned_df = pd.DataFrame(X_scaled_array, columns=X.columns)
        cleaned_df[target_col] = y.values
        cleaned_df.to_sql("cleaned_scaled_data", engine, if_exists='replace', index=False)

        return f"PREPROCESS SUCCESS: Data scaled and scaler saved to {scaler_path}"

    except Exception as e:
        return f"PREPROCESS ERROR: {str(e)}"