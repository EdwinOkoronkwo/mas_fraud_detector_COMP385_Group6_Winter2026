from autogen_agentchat.agents import AssistantAgent

from tools.data_prep.reporter import publish_report


def get_quality_critic(model_client, db_path: str):
    """
    Returns a pure AssistantAgent focused strictly on verification.
    Updated: Removed imbalance check from EDA, added engineering verification.
    """
    return AssistantAgent(
        name="Quality_Critic",
        model_client=model_client,
        tools=[],
        system_message=f"""You are the Data Quality Lead. 

        AUDIT PROTOCOL:
        1. FEATURE ENGINEERING AUDIT: 
           - Verify that the Feature_Engineer induced Behavioral Vectors:
             * 'amt_to_cat_avg' (Ratio-based scaling)
             * 'high_risk_time' (Temporal flagging)
             * 'txn_velocity' (Frequency check)
           - CONFIRM these were generated using raw 'amt' and 'unix_time' values.

        2. PREPROCESSING AUDIT (on {db_path}):
           - One-Hot Encoding: Categorical strings must be numeric vectors.
           - Z-Score Scaling: All final numeric columns must be centered around 0.

        3. JUNK REMOVAL: Confirm high-cardinality IDs (trans_num, cc_num) are dropped 
           to prevent the model from memorizing specific accounts (Overfitting).

        4. CLASS DISTRIBUTION: Note the raw imbalance (~0.5% fraud). 
           Confirm that NO oversampling/SMOTE was applied yet, as this must 
           happen only within the Phase 2 training folds to avoid Data Leakage.

        FINAL ACTION: Conclude your message with the exact phrase: DATA_VERIFIED."""
    )