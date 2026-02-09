

import os

# This is: .../mas_fraud_detector/config/settings.py
_config_dir = os.path.dirname(os.path.abspath(__file__))

# This is: .../mas_fraud_detector/
BASE_DIR = os.path.dirname(_config_dir)

# Define absolute paths
KAGGLE_PATH = os.path.join(BASE_DIR, "data", "kaggle")
DB_PATH = os.path.join(BASE_DIR, "data", "database.sqlite")
PROCESSED_TABLE_NAME = "cleaned_scaled_data"

AGENT_CONFIG = {
    "KAGGLE_PATH": KAGGLE_PATH,
    "DB_PATH": DB_PATH
}

# Add this to handle the tournament hand-off
TEMP_SPLIT_PATH = os.path.join(BASE_DIR, "data", "temp_split.pkl")

# Add this to your existing settings.py
# MODELS_DIR = os.path.join(BASE_DIR, "models", "champions")
# Ensure the data directory exists so os.path.exists() doesn't fail later
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Create a standard reports folder in your project root
REPORT_DIR = os.path.join(os.getcwd(), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

AGENT_CONFIG.update({
    "TEMP_SPLIT_PATH": TEMP_SPLIT_PATH
})

import os

# Root: .../mas_fraud_detector/
_config_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_config_dir)

# Absolute Paths
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "data", "database.sqlite"))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "models"))
POLICY_FILE = os.path.normpath(os.path.join(BASE_DIR, "data", "policies", "fraud_handbook.txt"))
REPORT_DIR = os.path.normpath(os.path.join(BASE_DIR, "reports"))

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)