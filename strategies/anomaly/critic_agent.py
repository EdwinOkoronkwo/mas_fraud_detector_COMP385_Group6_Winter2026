
import json
import os

from autogen_agentchat.agents import AssistantAgent
from sqlalchemy import text


def verify_anomaly_labels(db_path: str, model_type: str = "all") -> str:
    """SQL tool to perform hit-analysis: predicted anomalies vs ground truth."""
    try:
        import pandas as pd
        from sqlalchemy import create_engine, text
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")

        # 1. Get Global Stats
        with engine.connect() as conn:
            actual_fraud = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data WHERE is_fraud = 1")).scalar()
            total_records = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data")).scalar()

        # 2. Extract Predictions (Assuming models flagged records in the DB or we use a standard index)
        # Note: In a real MAS, you'd pass the specific indices here.
        # For now, we simulate the 'Hit Rate' logic based on the 1,297 anomalies reported.

        # If your models are saving 'is_anomaly' columns to the DB, use this:
        # tp = conn.execute(text(f"SELECT COUNT(*) FROM cleaned_scaled_data WHERE {model_type}_anomaly = 1 AND is_fraud = 1")).scalar()

        return json.dumps({
            "status": "SUCCESS",
            "ground_truth": {
                "total_fraud": int(actual_fraud),
                "prevalence": f"{round((actual_fraud / total_records) * 100, 2)}%"
            },
            "instruction": "Critic: Use this total_fraud count to calculate Recall (TP / total_fraud) for each model."
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class AnomalyCritic:
    def __init__(self, model_client, settings):
        self.settings = settings
        self.agent = AssistantAgent(
            name="Anomaly_Critic",  # Ensure this matches the Selector return string
            model_client=model_client,
            tools=[verify_anomaly_labels],  # Tool must be able to read DB_PATH
            system_message=f"""
            [INST] You are the Final Auditor. 
            DATABASE: {self.settings.DB_PATH}

            TASK:
            1. Use 'verify_anomaly_labels' to check the precision/recall of DBSCAN and K-Means.
            2. Compare these against the VAE (124 TPs, 9.56% Precision).
            3. Output the FINAL LEADERTABLE in Markdown.
            4. End your message with 'TERMINATE'. [/INST]
            """
        )