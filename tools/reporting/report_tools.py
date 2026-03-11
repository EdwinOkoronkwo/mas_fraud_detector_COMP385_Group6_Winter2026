import json
import os
from datetime import datetime


def write_markdown_report(report_content: str, filename: str = "FINAL_FRAUD_STRATEGY.md") -> str:
    """
    Writes a formatted markdown report to the project's reports directory.
    Use this to persist the final ensemble strategy and mathematical findings.
    """
    try:
        # Get the current working directory where main.py is running
        base_dir = os.getcwd()
        reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        file_path = os.path.join(reports_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return f"SUCCESS: Report saved to {file_path}"
    except Exception as e:
        return f"ERROR: Failed to save report: {str(e)}"




def extract_supervised_metadata(self):
    """Extracts feature names and model config for the handoff."""
    # 1. Grab feature names from the data bundle
    if os.path.exists(self.settings.TEMP_SPLIT_PATH):
        data = joblib.load(self.settings.TEMP_SPLIT_PATH)
        # Assuming the bundle is a dict with 'X_train'
        feature_names = data['X_train'].columns.tolist()
    else:
        feature_names = []

    # 2. This will be passed to the 'save_tool'
    return feature_names


def save_inference_metadata(supervised_config: dict, feature_list: list, registry_path: str):
    """
    Saves the final champion registry to JSON.
    """
    # Defensive extraction to prevent 'KeyError'
    path = supervised_config.get("path", "models/dynamic_xgb_champion.joblib")
    m_type = supervised_config.get("type", "xgboost")
    params = supervised_config.get("params", {})

    registry = {
        "supervised": {
            "path": path,
            "type": m_type,
            "features": feature_list,
            "config": {
                "model_class": "XGBClassifier" if m_type == "xgboost" else "SklearnModel",
                "params": params
            }
        },
        "weights": {"supervised": 1.0, "neuro": 0.0, "clustering": 0.0},
        "timestamp": "2026-02-18"
    }

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)
    return f"SUCCESS: Registry saved to {registry_path}"


import json
import os
from datetime import datetime

import joblib
import os
from datetime import datetime
import json


def save_hybrid_metadata(supervised_config, neuro_config, clustering_config, feature_list, registry_path):
    """The 'save_tool' used by the Aggregator to lock in the winners."""
    try:
        registry_data = {
            "metadata": {"timestamp": "2026-02-21", "status": "Finalized"},
            "supervised": supervised_config, # Best of XGB or ANN
            "unsupervised_neuro": neuro_config, # VAE
            "unsupervised_clustering": clustering_config, # KMeans
            "features_used": feature_list
        }
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=4)
        return f"SUCCESS: Registry saved to {registry_path}"
    except Exception as e:
        return f"REGISTRY_ERROR: {str(e)}"

