import joblib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from pathlib import Path

from config.settings import TEMP_SPLIT_PATH


def prepare_championship_data_tool(db_path: str) -> str:
    try:
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")
        df = pd.read_sql("SELECT * FROM cleaned_scaled_data", engine)

        # SPEED OPTIMIZATION: Downsample to 10k rows before splitting
        # This prevents HTTP timeouts during SMOTE and Grid Search.
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)

        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']

        # 1. SPLIT FIRST (Ensures the Test set is 100% "Real" data)
        X_train_raw, X_test, y_train_raw, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 2. APPLY SMOTE ONLY TO TRAINING DATA
        # With a 10k base, the training set (~8k) will balance out quickly.
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_raw, y_train_raw)

        # 3. SAVE FOR TOURNAMENT AGENTS
        joblib.dump((X_train_balanced, X_test, y_train_balanced, y_test), TEMP_SPLIT_PATH)
        # 4. VERIFICATION (Prevents the '\x10' Invalid Load Key Error)
        if os.path.exists(TEMP_SPLIT_PATH):
            file_size = os.path.getsize(TEMP_SPLIT_PATH)
            if file_size > 0:
                return f"SUCCESS: Training set balanced ({len(y_train_balanced)} rows). Test set remains pure ({len(y_test)} rows)."

        return "ERROR: Data saved but file is inaccessible or empty."
    except Exception as e:
        return f"ERROR in data split: {str(e)}"




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