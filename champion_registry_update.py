import joblib
import json
import os
import shutil
from datetime import datetime

# --- CONSTANTS ---
BASE_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
MODELS_DIR = os.path.join(BASE_PATH, "models")
REPORTS_DIR = os.path.join(BASE_PATH, "reports")

# 🎯 SOURCE: Defined in your train_dynamic_xgb_tool
MODEL_SOURCE = os.path.join(MODELS_DIR, "gold_xgb.pkl")
# 🎯 DESTINATION: The standardized name for InfrastructureManager
MODEL_DEST = os.path.join(MODELS_DIR, "gold_champion.pkl")
REGISTRY_FILE = os.path.join(REPORTS_DIR, "champion_registry.json")

# 🚀 27-FEATURE SCHEMA
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

def promote_to_gold():
    print(f"🥇 Promoting {os.path.basename(MODEL_SOURCE)} to Gold Champion...")

    if not os.path.exists(MODEL_SOURCE):
        print(f"❌ Critical Error: Source file not found at {MODEL_SOURCE}")
        return

    # 1. Physical Promotion (Renaming for the MAS architecture)
    shutil.copy2(MODEL_SOURCE, MODEL_DEST)
    
    # 2. Registry Update with Peak Tournament Metrics
    manifest = {
        "status": "CERTIFIED_GOLD",
        "certified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": "Dynamic_XGB_Agent",
        "metrics": {
            "f1_score": 0.723,    # Peak performance from your tournament audit
            "recall": 0.721,
            "precision": 0.725
        },
        "feature_engineering": {
            "count": 27,
            "list": EXPECTED_FEATURES
        }
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ SUCCESS: {MODEL_DEST} is now the active MAS Gold Champion.")

if __name__ == "__main__":
    promote_to_gold()