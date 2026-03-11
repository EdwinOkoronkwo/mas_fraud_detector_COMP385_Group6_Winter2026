import joblib
import pandas as pd
import os
from xgboost import XGBClassifier


def train_base_24_model():
    # 1. PATH CONFIGURATION
    root = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"

    # Corrected Filename from your folder
    DATA_FILENAME = "temp_split.joblib"

    data_path = os.path.join(root, "data", DATA_FILENAME)
    save_dir = os.path.join(root, "models", "baselines", "base")

    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    # 2. LOAD DATA
    print(f"📂 Loading data from {DATA_FILENAME}...")
    data = joblib.load(data_path)

    # Extracting from the joblib dictionary structure
    X_train, y_train = data['train']

    # 3. SELECTING THE BASE 24 FEATURES
    # We explicitly drop the 3 behavioral columns to create the simple version
    engineered_cols = ["amt_to_cat_avg", "high_risk_time", "txn_velocity"]
    X_train_base = X_train.drop(columns=[c for c in engineered_cols if c in X_train.columns])

    print(f"🚀 Training Base Model on {X_train_base.shape[1]} features...")
    print(f"Target Feature Count: 24 | Actual: {X_train_base.shape[1]}")

    # 4. TRAINING
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=12,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_base, y_train)

    # 5. SAVE
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "xgb_baseline_24.pkl")
    joblib.dump(model, save_path)

    print(f"✅ SUCCESS: 24-feature baseline saved to: {save_path}")


import joblib
import pandas as pd
import os
from xgboost import XGBClassifier


def train_enhanced_baseline():
    root = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
    data_path = os.path.join(root, "data", "temp_split.joblib")
    # This matches the folder your pipeline is looking for
    save_dir = os.path.join(root, "models", "baselines", "enhanced")

    # 1. LOAD
    data = joblib.load(data_path)
    X_train, y_train = data['train']

    # 2. FEATURE CHECK
    print(f"🚀 Training ENHANCED Model on {X_train.shape[1]} features...")

    # 3. TRAIN
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=12,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # 4. SAVE
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "xgb_baseline_enhanced.pkl")
    joblib.dump(model, save_path)

    print(f"✅ SUCCESS: Enhanced baseline saved to: {save_path}")



if __name__ == "__main__":
    train_base_24_model()
    train_enhanced_baseline()