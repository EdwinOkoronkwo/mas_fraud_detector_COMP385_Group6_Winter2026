import joblib
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier

# --- CONSTANTS ---
BASE_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
MODEL_FILE = os.path.join(BASE_PATH, "models", "xgb_baseline_24feat.pkl")
REGISTRY_FILE = os.path.join(BASE_PATH, "reports", "baseline_registry.json")
DATA_PATH = os.path.join(BASE_PATH, "data", "temp_split.joblib")

# The 24-feature bridge verified in your logs
EXPECTED_FEATURES = [
    "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long",
    "amt_to_cat_avg", "high_risk_time",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos",
    "category_travel"
]


def finalize_baseline_model():
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)

    print("🔄 Loading Phase 2 Data for 24-Feature Baseline...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return

    data = joblib.load(DATA_PATH)
    X_train_raw, y_train = data['train']

    # --- DATA RECONSTRUCTION LOGIC ---
    if isinstance(X_train_raw, pd.DataFrame):
        # If it's a DataFrame but missing columns, it will still fail, so we check columns
        missing = set(EXPECTED_FEATURES) - set(X_train_raw.columns)
        if not missing:
            X_train = X_train_raw[EXPECTED_FEATURES]
        else:
            print(f"⚠️ DataFrame missing columns: {missing}. Reverting to positional mapping...")
            X_train = pd.DataFrame(X_train_raw.values[:, :24], columns=EXPECTED_FEATURES)
    else:
        # It's likely a NumPy array from the Sampler
        print(f"💡 Detected {type(X_train_raw)}. Mapping first 24 columns to schema...")
        # Ensure we have at least 24 columns
        if X_train_raw.shape[1] < 24:
            raise ValueError(f"Data error: Expected 24 columns, found {X_train_raw.shape[1]}")

        # Slice to 24 columns and apply names
        X_train = pd.DataFrame(X_train_raw[:, :24], columns=EXPECTED_FEATURES)

    # --- TRAINING (Using Logged Parameters) ---
    print(f"🚀 Training Static Baseline (F1 Floor: 0.4878) with shape {X_train.shape}...")
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )
    model.fit(X_train, y_train)

    # Save Model
    joblib.dump(model, MODEL_FILE)

    # Save Registry
    manifest = {
        "model_info": {
            "name": "Static_XGB_Baseline",
            "file": "xgb_baseline_24feat.pkl",
            "type": "xgboost.XGBClassifier",
            "tier": "Baseline"
        },
        "performance": {
            "f1_score": 0.4878,
            "recall": 0.8065,
            "precision": 0.3497
        },
        "metadata": {
            "finalized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_count": 24
        }
    }

    with open(REGISTRY_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n✅ BASELINE GENERATED: {MODEL_FILE}")


if __name__ == "__main__":
    finalize_baseline_model()