
import os
import joblib

import pandas as pd

import joblib
import os

# Use a single, consistent path variable
MODEL_DIR = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\models"
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)

    # Check for feature names (Available if fitted on a DataFrame)
    if hasattr(scaler, 'feature_names_in_'):
        print(f"✅ Scaler expects {scaler.n_features_in_} features.")
        print(f"📋 FEATURES IN ORDER:\n{scaler.feature_names_in_}")

        # This is your "Smoking Gun" check:
        if scaler.n_features_in_ == 21:
            print("\n🚀 SUCCESS: Scaler is aligned with the full 21-feature schema.")
        else:
            print(f"\n⚠️ WARNING: Scaler is still set to {scaler.n_features_in_} features. VAE will crash.")
    else:
        print("❌ Scaler exists but doesn't have feature names stored.")
        print(f"Scaler expects {scaler.n_features_in_} features.")
else:
    print(f"❌ Scaler not found at {scaler_path}")