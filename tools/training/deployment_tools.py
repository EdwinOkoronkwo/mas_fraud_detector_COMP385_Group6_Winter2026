import joblib
import os
import json
from datetime import datetime
from config.settings import MODELS_DIR


def persist_champion_model(model_name: str, metrics_json: str) -> str:
    """
    Saves the winning model metrics to the models/champions directory.
    """
    try:
        # 1. No need to define paths here; we use the one from settings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace(' ', '_').lower()

        meta_filename = f"champion_{safe_name}_{timestamp}_metrics.json"
        save_path = os.path.join(MODELS_DIR, meta_filename)

        # 2. Write the metrics
        with open(save_path, 'w') as f:
            f.write(metrics_json)

        return f"SUCCESS: Champion ({model_name}) metrics persisted to {MODELS_DIR}"
    except Exception as e:
        return f"ERROR during persistence: {str(e)}"