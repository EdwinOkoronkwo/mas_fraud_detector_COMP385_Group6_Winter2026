# mas_fraud_detector/strategies/unsupervised/critic_agent.py
import json
import os

from autogen_agentchat.agents import AssistantAgent
from sqlalchemy import text


def verify_anomaly_labels(db_path: str) -> str:
    """SQL tool to check the actual fraud labels in the processed data."""
    try:
        from sqlalchemy import create_engine, text
        import json
        import os

        engine = create_engine(f"sqlite:///{os.path.abspath(db_path)}")

        with engine.connect() as conn:
            actual_fraud = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data WHERE is_fraud = 1")).scalar()
            total_records = conn.execute(text("SELECT COUNT(*) FROM cleaned_scaled_data")).scalar()

        # FIXED: Returning standard JSON for consistent parsing
        return json.dumps({
            "status": "SUCCESS",
            "actual_fraud_count": int(actual_fraud),
            "total_records": int(total_records),
            "fraud_prevalence_pct": round((actual_fraud / total_records) * 100, 2)
        })
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})


class AnomalyCritic:
    def __init__(self, model_client, db_path):
        self.agent = AssistantAgent(
            name="Anomaly_Critic",
            model_client=model_client,
            tools=[verify_anomaly_labels],
            system_message=f"""[INST] You are the Lead Forensic Auditor.

            GOAL: Crown winners for both the Clustering Group and the Neuro-Pattern Group.

            WORKFLOW:
            1. CLUSTERING AUDIT: Review KMeans, DBSCAN, SOM, and Isolation Forest results.
            2. NEURO-PATTERN AUDIT: Review AE, VAE, and RNN-AE results. 
               - Look for 'Reconstruction MSE' and 'Anomaly Counts'.
            3. GROUND TRUTH: Run 'verify_anomaly_labels' to see actual fraud prevalence in {db_path}.

            EVALUATION:
            - Models with counts much higher than ground truth = High Noise (Low Weight).
            - Models with counts closer to ground truth = High Precision (Discovery Champion).
            
            CRITICAL GATEKEEPER RULE: 
            Do NOT produce a final 'Group Championship Table' if any model in 
            the Neuro-Pattern group is marked as 'Pending'. 
            If data is missing, output: 'DATA_GAP: Awaiting Neural metrics from VAE/RNN.' 
            and do NOT conclude with TERMINATE.

            REPORTING:
            Produce the 'Group Championship Table' for BOTH groups.
            Crown two champions: 'Clustering Champion' and 'Neuro-Pattern Champion'.
            Conclude with 'TERMINATE'. [/INST]"""
        )