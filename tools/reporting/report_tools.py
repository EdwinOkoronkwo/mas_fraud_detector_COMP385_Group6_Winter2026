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


import json
import os
from datetime import datetime


def save_inference_metadata(supervised_path: str, neural_path: str, clustering_path: str) -> str:
    """
    Saves the official model selection for the inference pipeline.

    Args:
        supervised_path (str): Path to the best supervised model.
        neural_path (str): Path to the best neural pattern model.
        clustering_path (str): Path to the best clustering model.
    """
    # Define registry path relative to the execution root
    registry_path = "mas_fraud_detector/models/champion_registry.json"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    metadata = {
        "champions": {
            "supervised": supervised_path,
            "neural": neural_path,
            "clustering": clustering_path
        },
        "scaler_path": "mas_fraud_detector/models/scaler.joblib",
        "ensemble_weights": {"supervised": 0.4, "neural": 0.4, "clustering": 0.2},
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(registry_path, "w") as f:
            json.dump(metadata, f, indent=4)
        return f"SUCCESS: Registry saved at {registry_path}"
    except Exception as e:
        return f"ERROR SAVING REGISTRY: {str(e)}"
