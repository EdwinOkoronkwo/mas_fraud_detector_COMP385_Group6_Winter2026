# In your Phase1_Runner or verification script:
import joblib
import os
from datetime import datetime
import pandas as pd


def inspect_preprocessor(file_path: str = "models/preprocessor_base.joblib"):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        return

    # 1. Metadata: When was it created?
    stats = os.stat(file_path)
    creation_time = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
    file_size_kb = stats.st_size / 1024

    print(f"--- 📅 ARTIFACT METADATA ---")
    print(f"File Path: {file_path}")
    print(f"Created/Modified: {creation_time}")
    print(f"File Size: {file_size_kb:.2f} KB")
    print("-" * 30)

    # 2. Load and Unpack
    preprocessor = joblib.load(file_path)

    # 3. Inspect the Scaler (The 'num' branch)
    # We access the named transformer we defined in our tool
    scaler = preprocessor.named_transformers_['num']
    num_features = preprocessor.transformers_[0][2]  # The list of numeric columns

    print(f"--- 🔢 SCALER CONTENT (8 Features) ---")
    for name, mean, var in zip(num_features, scaler.mean_, scaler.var_):
        print(f"Feature: {name:12} | Mean: {mean:10.4f} | StdDev: {var ** 0.5:10.4f}")
    print("-" * 30)

    # 4. Inspect the One-Hot Encoder (The 'cat' branch)
    ohe = preprocessor.named_transformers_['cat']
    feature_names = ohe.get_feature_names_out()

    print(f"--- 🏷️ ONE-HOT CONTENT ({len(feature_names)} Features) ---")
    print(f"Encoded Categories: {list(feature_names)}")
    print("-" * 30)

    # 5. Final Verification
    total_features = len(num_features) + len(feature_names)
    print(f"✅ TOTAL MODEL INPUT DIMENSION: {total_features}")


# Execute the inspection
inspect_preprocessor()