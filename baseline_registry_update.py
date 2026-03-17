import joblib
import os
import pandas as pd
from xgboost import XGBClassifier

# --- CONSTANTS ---
BASE_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
MODELS_DIR = os.path.join(BASE_PATH, "models")
DATA_PATH = os.path.join(BASE_PATH, "data", "temp_split.joblib")

# The specific target for comparison
BASELINE_OUT = os.path.join(MODELS_DIR, "xgb_baseline_27feat.pkl")

# 🚀 27-Feature Schema
EXPECTED_FEATURES = [
    "num__amt", "num__zip", "num__lat", "num__long", "num__city_pop", "num__unix_time", 
    "num__merch_lat", "num__merch_long", "num__amt_to_cat_avg", "num__high_risk_time", 
    "num__txn_velocity", "cat__category_entertainment", "cat__category_food_dining", 
    "cat__category_gas_transport", "cat__category_grocery_net", "cat__category_grocery_pos", 
    "cat__category_health_fitness", "cat__category_home", "cat__category_kids_pets", 
    "cat__category_misc_net", "cat__category_misc_pos", "cat__category_personal_care", 
    "cat__category_shopping_net", "cat__category_shopping_pos", "cat__category_travel", 
    "cat__gender_F", "cat__gender_M"
]

def create_baseline():
    print(f"📊 Initializing Baseline with 50 trees...")
    
    try:
        # Load data
        data_path = os.path.join(BASE_PATH, "data", "temp_split.joblib")
        data = joblib.load(data_path)
        X_train_raw, y_train = data['train']

        # 🛠️ THE FIX: Handle Sparse Matrices or DataFrames safely
        if hasattr(X_train_raw, "toarray"):
            # If it's a sparse matrix (SciPy)
            X_train_dense = X_train_raw.toarray()
        elif hasattr(X_train_raw, "values"):
            # If it's already a DataFrame
            X_train_dense = X_train_raw.values
        else:
            # Standard NumPy array
            X_train_dense = X_train_raw

        # Now slice exactly 27 features safely
        X_train_sliced = X_train_dense[:, :27]
        X_train = pd.DataFrame(X_train_sliced, columns=EXPECTED_FEATURES)

        baseline_model = XGBClassifier(
            n_estimators=50, # 🎯 Strictly 50 trees for baseline
            max_depth=3,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42
        )
        
        print("🌲 Fitting baseline model...")
        baseline_model.fit(X_train, y_train)
        
        # Save ONLY the baseline
        joblib.dump(baseline_model, BASELINE_OUT)
        print(f"✅ Baseline Saved: {BASELINE_OUT}")

    except Exception as e:
        print(f"❌ Baseline training failed: {e}")


if __name__ == "__main__":
    create_baseline()