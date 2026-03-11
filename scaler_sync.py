import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 1. Define Paths (Aligning with your logs)
BASE_DIR = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
DATA_PATH = os.path.join(BASE_DIR, "data", "temp_split.joblib")
SCALER_OUT = os.path.join(BASE_DIR, "models", "scaler.joblib")

# The exact 24 features from your Gold Manifest
EXPECTED_FEATURES = [
    "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos",
    "category_travel", "gender_F", "gender_M"
]


def sync_scaler_to_gold_model():
    print("🔄 Loading Phase 2 Data...")
    data = joblib.load(DATA_PATH)
    X_train, _ = data['train']

    # 2. Filter X_train to match the 24 Gold features exactly
    # This ensures the scaler knows about category_entertainment, etc.
    X_train_gold = X_train[EXPECTED_FEATURES]

    print(f"📏 Fitting Scaler on {len(EXPECTED_FEATURES)} features...")
    scaler = StandardScaler()
    scaler.fit(X_train_gold)

    # 3. Save the new "Gold Scaler"
    joblib.dump(scaler, SCALER_OUT)
    print(f"✅ SUCCESS: New scaler saved to {SCALER_OUT}")
    print(f"🚀 Your PreprocessingAgent can now scale all 24 columns without errors.")


if __name__ == "__main__":
    sync_scaler_to_gold_model()