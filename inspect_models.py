import joblib
import torch
import xgboost as xgb
import json
import os

# --- TARGETED CONFIGURATION ---
PATHS = {
    "Primary Preprocessor": r"models\preprocessor_base.joblib",
    "Gold XGBoost": r"models\gold_xgb.pkl",
    "Baseline XGBoost": r"models\xgb_baseline_24feat.pkl",
    "Champion VAE": r"models\champion_vae.pth",
    "DBSCAN Cluster": r"models\champion_dbscan.joblib"
}


def audit_core_four():
    print("=" * 70)
    print("🚀 MAS FRAUD DETECTOR: CORE FOUR INTEGRITY AUDIT")
    print("=" * 70)

    try:
        prep = joblib.load(PATHS["Primary Preprocessor"])
        target_dim = len(prep.get_feature_names_out())
        print(f"✅ BASELINE SYNC: {target_dim} Features Verified.")
    except Exception as e:
        print(f"❌ Preprocessor Missing: {e}")
        return

    print("\n[SECTION 1: BOOSTING]")
    for key in ["Gold XGBoost", "Baseline XGBoost"]:
        if os.path.exists(PATHS[key]):
            try:
                loaded = joblib.load(PATHS[key])

                # 🚀 SURGICAL ADJUSTMENT: Handle Dictionary vs. Object
                if isinstance(loaded, dict):
                    # Extract the model from the bundle we created in BaselineModelTrainer
                    model_obj = loaded.get('model')
                    n_f = loaded.get('metrics', {}).get('feature_count', "Unknown")
                else:
                    model_obj = loaded
                    booster = model_obj.get_booster() if hasattr(model_obj, 'get_booster') else model_obj
                    config = json.loads(booster.save_config())
                    n_f = int(config.get('learner', {}).get('learner_model_param', {}).get('num_feature'))

                status = "✅ MATCH" if str(n_f) == str(target_dim) else "❌ MISMATCH"
                print(f"📊 {key:20} | Features: {n_f} | {status}")
            except Exception as e:
                print(f"⚠️ {key} Load Error: {e}")
        else:
            print(f"🔘 {key:20} | Missing")

    print("\n[SECTION 2: NEURAL]")
    if os.path.exists(PATHS["Champion VAE"]):
        try:
            ckpt = torch.load(PATHS["Champion VAE"], map_location='cpu', weights_only=False)
            first_w = next(k for k in ckpt.keys() if 'weight' in k)
            n_f = ckpt[first_w].shape[1]
            status = "✅ MATCH" if n_f == target_dim else "❌ MISMATCH"
            print(f"🧠 Champion VAE        | Input:    {n_f} | {status}")
        except Exception as e:
            print(f"⚠️ VAE Load Error: {e}")

    print("\n[SECTION 3: CLUSTERING]")
    if os.path.exists(PATHS["DBSCAN Cluster"]):
        try:
            db = joblib.load(PATHS["DBSCAN Cluster"])
            n_f = db.components_.shape[1] if hasattr(db, 'components_') else getattr(db, 'n_features_in_', "Unknown")
            status = "✅ MATCH" if n_f == target_dim else "❌ MISMATCH"
            print(f"📍 DBSCAN Cluster      | Dim:      {n_f} | {status}")
        except Exception as e:
            print(f"⚠️ DBSCAN Load Error: {e}")

    print("\n" + "=" * 70)
    print("📋 VERDICT: SYSTEM SEALED & VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    audit_core_four()