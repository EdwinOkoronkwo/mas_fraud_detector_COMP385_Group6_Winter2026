import os
import shutil
from deterministic_inference.utils.infrastructure import InfrastructureManager

def promote_models():
    # Initialize Manager to get correct paths
    manager = InfrastructureManager()
    
    # Create the 'promoted' directory if it doesn't exist
    os.makedirs(manager.promoted_dir, exist_ok=True)

    # 🚀 MAP: { "Current Filename in /models" : "New Filename in /promoted" }
    promotion_map = {
        "gold_xgb.pkl": "gold_champion.pkl",
        "gold_champion.pkl": "gold_champion.pkl", # Fallback if already renamed
        "xgb_baseline_27feat.pkl": "xgb_baseline_27feat.pkl",
        "champion_vae.pth": "champion_vae.pth",
        "champion_dbscan.joblib": "champion_dbscan.joblib",
        "preprocessor_base.joblib": "preprocessor_base.joblib"
    }

    print(f"📂 Starting promotion to: {manager.promoted_dir}\n" + "="*50)

    for src_name, dest_name in promotion_map.items():
        src_path = os.path.join(manager.model_dir, src_name)
        dest_path = os.path.join(manager.promoted_dir, dest_name)
        
        if os.path.exists(src_path):
            # copy2 preserves metadata
            shutil.copy2(src_path, dest_path)
            print(f"✅ Promoted: {src_name} -> {dest_name}")
        else:
            # Skip with warning if the specific source version isn't found
            print(f"⚠️  Skipped: {src_name} (File not found in /models)")

    print("="*50 + "\n✅ Housekeeping: Promotion Complete.")

if __name__ == "__main__":
    promote_models()