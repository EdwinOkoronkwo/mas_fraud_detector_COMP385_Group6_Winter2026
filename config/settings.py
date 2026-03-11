import os


class Settings:
    def __init__(self):
        # Base setup
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.PROJECT_ROOT = self.BASE_DIR
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")

        # Paths
        self.DB_PATH = os.path.normpath(os.path.join(self.DATA_DIR, "database.sqlite"))
        self.TEMP_SPLIT_PATH = os.path.normpath(os.path.join(self.DATA_DIR, "temp_split.joblib"))
        self.MODELS_DIR = os.path.normpath(os.path.join(self.BASE_DIR, "models"))
        self.REPORT_DIR = os.path.normpath(os.path.join(self.BASE_DIR, "reports"))

        # Tables
        self.PROCESSED_TABLE_NAME = "cleaned_scaled_data"

        # Ensure directories
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        self.AGENT_CONFIG = {
            "model_client": "mistral-medium-latest",
            "temperature": 0.1,
            "max_tokens": 2048
        }

    def get(self, key, default=None):
        """Standard getter to satisfy the Orchestrator's .get calls."""
        return getattr(self, key, default)


# The single source of truth instance
settings = Settings()

import os

# --- 1. BASE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "models"))
REPORT_DIR = os.path.normpath(os.path.join(BASE_DIR, "reports"))

# --- 2. FILE PATHS ---
DB_PATH = os.path.normpath(os.path.join(DATA_DIR, "database.sqlite"))
TEMP_SPLIT_PATH = os.path.normpath(os.path.join(DATA_DIR, "temp_split.joblib"))

# --- 3. DATABASE TABLES ---
PROCESSED_TABLE_NAME = "cleaned_scaled_data"

# --- 4. AGENT CONFIGURATION ---
AGENT_CONFIG = {
    "model_client": "mistral-medium-latest",
    "temperature": 0.1,
    "max_tokens": 2048
}

# --- 5. INITIALIZE DIRECTORIES ---
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

