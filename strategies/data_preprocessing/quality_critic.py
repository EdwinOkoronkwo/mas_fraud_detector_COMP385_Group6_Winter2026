from autogen_agentchat.agents import AssistantAgent


def get_quality_critic(model_client, db_path: str):
    """
    Creates the Critic to verify the CLEANED and SCALED dataset (Pre-SMOTE).
    """
    return AssistantAgent(
        name="Quality_Critic",
        model_client=model_client,
        system_message=f"""You are the Data Quality Lead. 

        GOAL: Verify the migration and cleaning of data to: {db_path}

        CHECKS:
        1. Verify that the table 'cleaned_scaled_data' exists in the database.
        2. Verify that the 'is_fraud' column exists.
        3. NOTE: Row counts will be IMBALANCED (this is expected). Do NOT look for equal counts here.
        4. Check that features are scaled (StandardScaler applied).

        REPORT FORMAT (JSON):
        {{
            "status": "SUCCESS" or "FAILED",
            "row_counts": {{"cleaned_scaled_data": total_count}},
            "imbalance_ratio": "Report the fraud vs non-fraud count",
            "db_path": "{db_path}",
            "verification_msg": "Explain if the data is scaled and ready for the tournament."
        }}

        If the database is ready and the path matches, end with: DATA_VERIFIED."""
    )