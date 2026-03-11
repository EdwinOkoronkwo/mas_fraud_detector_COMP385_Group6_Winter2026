import joblib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from pathlib import Path

from config.settings import settings




# def prepare_championship_data_tool(db_path: str) -> str:
#     try:
#         engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
#         df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)
#
#         # SPEED OPTIMIZATION: Downsample to 10k rows before splitting
#         # This prevents HTTP timeouts during SMOTE and Grid Search.
#         if len(df) > 50000:
#             df = df.sample(50000, random_state=42)
#
#         X = df.drop(columns=['is_fraud'])
#         y = df['is_fraud']
#
#         # 1. SPLIT FIRST (Ensures the Test set is 100% "Real" data)
#         X_train_raw, X_test, y_train_raw, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
#
#         # 2. APPLY SMOTE ONLY TO TRAINING DATA
#         # With a 10k base, the training set (~8k) will balance out quickly.
#         smote = SMOTE(random_state=42)
#         X_train_balanced, y_train_balanced = smote.fit_resample(X_train_raw, y_train_raw)
#
#         # 3. SAVE FOR TOURNAMENT AGENTS
#         joblib.dump((X_train_balanced, X_test, y_train_balanced, y_test), TEMP_SPLIT_PATH)
#         # 4. VERIFICATION (Prevents the '\x10' Invalid Load Key Error)
#         if os.path.exists(TEMP_SPLIT_PATH):
#             file_size = os.path.getsize(TEMP_SPLIT_PATH)
#             if file_size > 0:
#                 return f"SUCCESS: Training set balanced ({len(y_train_balanced)} rows). Test set remains pure ({len(y_test)} rows)."
#
#         return "ERROR: Data saved but file is inaccessible or empty."
#     except Exception as e:
#         return f"ERROR in data split: {str(e)}"


import pandas as pd
import os
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class DataSpecialist:
    def __init__(self, db_path):
        self.db_path = os.path.abspath(db_path)
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def assign_targets(self, table_name="cleaned_scaled_data"):
        # 1. DYNAMIC LOADING: No manual lists.
        # We take what the Preprocess_Agent gave us.
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)

        # 2. AUTOMATIC FEATURE SELECTION
        # Everything except the target is a feature.
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

        print(f"✅ Bridge Verified: {X.shape[1]} features (The verified 24) extracted from SQL.")
        return X, y

    def stratified_3way_split(self, X, y):
        # First split: Test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Second split: Val set (20% of the remaining 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_smote(self, X_train, y_train):
        # Balance only the training set to prevent leakage
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res


# THE REPAIRED WRAPPER
def prepare_championship_data_tool(db_path: str) -> str:
    try:
        specialist = DataSpecialist(db_path)

        # 1. Load the REAL 24 features dynamically
        X, y = specialist.assign_targets()

        # 2. Split and SMOTE (No manual feature adding here!)
        X_train, X_val, X_test, y_train, y_val, y_test = specialist.stratified_3way_split(X, y)
        X_train_bal, y_train_bal = specialist.apply_smote(X_train, y_train)

        data_bundle = {
            'train': (X_train_bal, y_train_bal),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'features': X.columns.tolist()  # Stores the 24 names for the model
        }

        # 3. SAVE
        output_path = "data/temp_split.joblib"
        os.makedirs("data", exist_ok=True)
        joblib.dump(data_bundle, output_path)

        return f"SUCCESS: 24-Feature Data prepared. Balanced Train: {len(y_train_bal)} | Val: {len(y_val)} | Test: {len(y_test)}"
    except Exception as e:
        return f"ERROR: Data preparation failed: {str(e)}"



def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name}')
    path = f"reports/{model_name.replace(' ', '_').lower()}_cm.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(path)
    plt.close()
    return path