import autogen_agentchat.teams
from autogen_agentchat.conditions import TextMentionTermination


from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from strategies.data_preprocessing.quality_critic import get_quality_critic


def prep_selector(messages):
    processed_contents = []
    for m in messages:
        if isinstance(m.content, list):
            processed_contents.append(str(m.content))
        else:
            processed_contents.append(m.content or "")

    history_text = " ".join(processed_contents).upper()
    history_sources = [m.source.lower() for m in messages]

    # 1. SQL_Ingestor -> EDA_Specialist
    if "INGESTION_COMPLETE" not in history_text:
        return "SQL_Ingestor"

    # 2. EDA_Specialist -> Feature_Engineer
    if "eda_specialist" not in history_sources:
        return "EDA_Specialist"

    # 3. Feature_Engineer -> Preprocess_Agent
    # The Feature_Engineer should end its turn with "ENGINEERING_COMPLETE"
    # if "ENGINEERING_COMPLETE" not in history_text:
    #     return "Feature_Engineer"

    # 4. Preprocess_Agent -> Quality_Critic
    if "PREPROCESS_SUCCESS" not in history_text:
        return "Preprocess_Agent"

    # 5. Final Audit
    return "Quality_Critic"
def create_preprocessing_team(sql_ingestor, eda_specialist, feature_engineer, preprocess_agent, model_client):
    actual_db_path = sql_ingestor.config.get("DB_PATH")

    # The critic remains the final gatekeeper
    critic = get_quality_critic(model_client, actual_db_path)

    return SelectorGroupChat(
        participants=[
            sql_ingestor.agent,
            eda_specialist.agent,
            # feature_engineer.agent, # Added to the roster
            preprocess_agent.agent,
            critic
        ],
        model_client=model_client,
        selector_func=prep_selector,
        termination_condition=TextMentionTermination("DATA_VERIFIED")
    )