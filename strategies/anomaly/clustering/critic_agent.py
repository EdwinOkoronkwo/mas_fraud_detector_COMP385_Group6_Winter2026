
import json
import os

from autogen_agentchat.agents import AssistantAgent
from sqlalchemy import text


def verify_anomaly_labels(db_path: str) -> str:
    """SQL tool to check the actual fraud labels in the processed data."""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")

        with engine.connect() as conn:
            # Get the total ground truth fraud count
            actual_fraud = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data WHERE is_fraud = 1")).scalar()
            total_records = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data")).scalar()

        return json.dumps({
            "actual_fraud_count": int(actual_fraud),
            "total_records": int(total_records),
            "fraud_prevalence": f"{round((actual_fraud / total_records) * 100, 2)}%"
        })
    except Exception as e:
        return f"CRITIC ERROR: {str(e)}"


class AnomalyCritic:
    def __init__(self, model_client, db_path):
        self.agent = AssistantAgent(
            name="Anomaly_Critic",
            model_client=model_client,
            tools=[verify_anomaly_labels],
            system_message=f"""[INST] You are the Lead Forensic Auditor.

            GOAL: Compare unsupervised model results against known labels in {db_path}.

            WORKFLOW:
            1. Extract 'anomaly_count' and metrics from the JSON results of KMeans, DBSCAN, SOM, and Isolation Forest.
            2. Run 'verify_anomaly_labels' to get the total actual fraud (Ground Truth).
            3. CRITICAL: Calculate the 'Discovery Accuracy' by comparing how close the model's 'anomaly_count' is to the 'actual_fraud_count'. 

            NOTE: Since we are in an unsupervised context, the 'True Positives' reported by the agents are based on their specific flags. 
            
            You are an Evidence-Based Auditor. 
            DO NOT calculate alignment using simple counts. 
            You must only report alignment if you have executed a tool that compares 
            specific anomaly indices against ground truth. 
            If you do not have that data, state 'Alignment: Pending Verification'.
            REPORTING FORMAT:
            Produce the 'Unsupervised Championship Table' with these columns:
            | Model Name | Primary Metric | Anomaly Count | Ground Truth Alignment (%) |

            4. Crown the 'Discovery Champion' (The model that captured the most fraud-like density without excessive noise).
            5. Conclude with 'TERMINATE'. [/INST]"""
        )