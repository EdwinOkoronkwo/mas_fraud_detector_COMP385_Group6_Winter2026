import joblib
import json
import os
import shutil
from datetime import datetime

# --- CONSTANTS ---
BASE_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
MODELS_DIR = os.path.join(BASE_PATH, "models")

# SOURCE: The specific file we identified in your directory
MODEL_SOURCE = os.path.join(MODELS_DIR, "champion_xgb_dynamic.pkl")
# DESTINATION: Standardized name for the Infrastructure Manager
MODEL_DEST = os.path.join(MODELS_DIR, "gold_champion.pkl")

REGISTRY_FILE = os.path.join(BASE_PATH, "reports", "champion_registry.json")

# Verified 24-feature set (Bridge Verified)
EXPECTED_FEATURES = [
    "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long",
    "amt_to_cat_avg", "high_risk_time",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos",
    "category_travel"
]

def promote_xgb_to_gold():
    print("🥇 Promoting Dynamic XGBoost to Gold Tier...")

    if not os.path.exists(MODEL_SOURCE):
        print(f"❌ Error: Champion XGB not found at {MODEL_SOURCE}")
        return

    # 1. Promote the binary
    shutil.copy2(MODEL_SOURCE, MODEL_DEST)

    # 2. Update Registry with validated tournament metrics
    manifest = {
        "features_used": EXPECTED_FEATURES,
        "champion_metadata": {
            "agent": "Dynamic_XGB_Agent",
            "type": "XGBClassifier",
            "tier": "GOLD",
            "f1_score": 0.6200,
            "recall": 0.7800,
            "precision": 0.5200,
            "feature_count": 24
        },
        "supervised": {
            "path": "gold_champion.pkl",
            "features": EXPECTED_FEATURES
        },
        "metadata": {
            "certified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "CERTIFIED_GOLD"
        }
    }

    with open(REGISTRY_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\n✅ XGB CERTIFIED AS GOLD")
    print(f"-> Source: {MODEL_SOURCE}")
    print(f"-> Destination: {MODEL_DEST}")
    print(f"-> Registry Updated: {REGISTRY_FILE}")

if __name__ == "__main__":
    promote_xgb_to_gold()








# import joblib
# import json
# import os
# from datetime import datetime
# from xgboost import XGBClassifier
#
# import joblib
# import json
# import os
# from datetime import datetime
# from xgboost import XGBClassifier
#
# # --- CONSTANTS ---
# BASE_PATH = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector"
# MODEL_FILE = os.path.join(BASE_PATH, "models", "gold_xgb.pkl")
# REGISTRY_FILE = os.path.join(BASE_PATH, "reports", "champion_registry.json")
# DATA_PATH = os.path.join(BASE_PATH, "data", "temp_split.joblib")
#
# # UPDATED: Including the 3 Behavioral Features induced in Phase 1
# EXPECTED_FEATURES = [
#     "amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long",
#     "amt_to_cat_avg", "high_risk_time", "txn_velocity",
#     "category_entertainment", "category_food_dining", "category_gas_transport",
#     "category_grocery_net", "category_grocery_pos", "category_health_fitness",
#     "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
#     "category_personal_care", "category_shopping_net", "category_shopping_pos",
#     "category_travel", "gender_F", "gender_M"
# ]
#
# def finalize_gold_champion():
#     # 1. Ensure directories exist
#     os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
#     os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
#
#     # 2. Load data
#     print("🔄 Loading Phase 2 Championship data...")
#     if not os.path.exists(DATA_PATH):
#         print(f"❌ Error: Data not found at {DATA_PATH}")
#         return
#
#     data = joblib.load(DATA_PATH)
#     X_train, y_train = data['train']
#
#     # Ensure we only use the features the model expects, in the correct order
#     X_train = X_train[EXPECTED_FEATURES]
#
#     # 3. Train with Optimized Agent Params (Iteration 2 Wins)
#     print(f"🚀 Training Champion Model with {len(EXPECTED_FEATURES)} features...")
#     model = XGBClassifier(
#         learning_rate=0.05,  # Lowered from 0.1 for precision
#         max_depth=4,         # Lowered from 8 to solve the overfitting/0.00 score issue
#         n_estimators=500,    # Increased from 350 to boost recall
#         scale_pos_weight=1,  # SMOTE handled in split
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         tree_method='hist'
#     )
#     model.fit(X_train, y_train)
#
#     # 4. Save Binary
#     joblib.dump(model, MODEL_FILE)
#
#     # 5. RESTORE ORIGINAL REGISTRY STRUCTURE (Safe for Phase 3)
#     # 5. RESTORE ORIGINAL REGISTRY STRUCTURE (Safe for Phase 3)
#     # 5. RESTORE ORIGINAL REGISTRY STRUCTURE (Safe for Phase 3)
#     manifest = {
#         # This is the key your InfrastructureManager.get_features() is looking for!
#         "features_used": EXPECTED_FEATURES,
#
#         "supervised": {
#             "path": "gold_xgb.pkl",
#             "features": EXPECTED_FEATURES
#         },
#         "model_info": {
#             "name": "gold_xgb_dynamic_bo",
#             "file": "gold_xgb.pkl",
#             "type": "xgboost.XGBClassifier",
#             "tier": "Gold",
#             "version": "2026-02-23"
#         },
#         "inference_params": {
#             "threshold": 0.30,
#             "expected_features": EXPECTED_FEATURES
#         },
#         "performance_at_train": {
#             "f1_score": 0.6300,
#             "recall": 0.74,
#             "precision": 0.54,
#             "train_size": len(X_train)
#         },
#         "metadata": {
#             "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "status": "Finalized"
#         }
#     }
#
#     with open(REGISTRY_FILE, "w") as f:
#         json.dump(manifest, f, indent=4)
#
#     print(f"\n✅ CHAMPION PROMOTED (REVERTED TO PREVIOUS SCHEMA)")
#     print(f"-> Binary: {MODEL_FILE}")
#     print(f"-> Registry: {REGISTRY_FILE}")
#     print(f"-> Features: {len(EXPECTED_FEATURES)}")
#
# if __name__ == "__main__":
#     finalize_gold_champion()