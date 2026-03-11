import joblib
import json
import os

MODEL_PATH = r'C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\models\preprocessor.joblib'
MAP_PATH = r'C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\models\feature_map.json'

scaler = joblib.load(MODEL_PATH)

# Extract the exact names in the exact order
feature_names = list(scaler.get_feature_names_out())

# Clean up prefixes (Scikit-learn often adds 'cat__' or 'num__')
clean_names = [name.split('__')[-1] for name in feature_names]

with open(MAP_PATH, 'w') as f:
    json.dump(clean_names, f, indent=4)

import json
path = r"C:\CentennialCollege\AI_Capstone_Project\GroupProject\mas_fraud_detector\reports\champion_registry.json"
with open(path, 'r') as f:
    data = json.load(f)
    print(data.keys())

print(f"🎯 Feature map saved with {len(clean_names)} columns.")