from autogen_agentchat.agents import AssistantAgent

from tools.data_prep.reporter import publish_report



def get_quality_critic(model_client, db_path: str):
    """
    Returns a pure AssistantAgent focused strictly on verification.
    No tools = No role-alternation errors.
    """
    return AssistantAgent(
        name="Quality_Critic",
        model_client=model_client,
        tools=[],  # Cleaned: No tools attached
        system_message=f"""You are the Data Quality Lead. 

        AUDIT PROTOCOL:
        1. VERIFY that the EDAAgent identified the 175:1 class imbalance. 
        2. CONFIRM that the Preprocessing tool successfully executed on {db_path}:
           - One-Hot Encoding: Categorical strings are now numeric vectors.
           - Z-Score Scaling: 'amt' and 'city_pop' are centered around 0.
        3. CHECK for 'Junk' removal: Ensure IDs and names (trans_num, cc_num) are dropped.

        REASONING: Briefly state why keeping the data imbalanced at this stage 
        prevents data leakage during Phase 2 training.

        FINAL ACTION: Once you have reviewed the logs and confirmed the criteria, 
        conclude your message with the exact phrase: DATA_VERIFIED."""
    )