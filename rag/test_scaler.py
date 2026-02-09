
import os
import joblib

import pandas as pd

# Load the scaler you just saved in the training script
scaler = joblib.load(r"C:\Youtube\autogen\mas_fraud_detector\models\scaler.joblib")

# If it's a standard scikit-learn scaler, it stores feature names!
if hasattr(scaler, 'feature_names_in_'):
    print(f"✅ YOUR 9 FEATURES IN ORDER: {scaler.feature_names_in_}")
else:
    print("❌ Scaler doesn't have names. We must check the SQL table columns directly.")

MODEL_DIR = r"C:\Youtube\autogen\mas_fraud_detector\models"
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
print(f"Scaler expects {scaler.n_features_in_} features.")