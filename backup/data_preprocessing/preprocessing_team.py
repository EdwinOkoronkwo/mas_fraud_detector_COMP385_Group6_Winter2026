import autogen_agentchat.teams
from autogen_agentchat.conditions import TextMentionTermination


from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from strategies.data_preprocessing.quality_critic import get_quality_critic


def prep_selector(messages):
    """
    Fixed Selector: Handles both text and tool call lists safely.
    """
    # Convert all content to strings (handles lists of tool calls)
    processed_contents = []
    for m in messages:
        if isinstance(m.content, list):
            # Flatten tool call/result lists into a string
            processed_contents.append(str(m.content))
        else:
            processed_contents.append(m.content or "")

    history_text = " ".join(processed_contents).upper()
    history_sources = [m.source.lower() for m in messages]

    # 1. Stay on Ingestor until it explicitly signals completion
    if "INGESTION_COMPLETE" not in history_text:
        return "SQL_Ingestor"

    # 2. Move to EDA
    if "eda_specialist" not in history_sources:
        return "EDA_Specialist"

    # 3. Move to Preprocessing
    if "preprocess_agent" not in history_sources:
        return "Preprocess_Agent"

    # 4. Final Handoff
    return "Quality_Critic"


def create_preprocessing_team(sql_ingestor, eda_specialist, preprocess_agent, model_client):
    actual_db_path = sql_ingestor.config.get("DB_PATH")

    # Ensure critic is the tool-less version we discussed
    critic = get_quality_critic(model_client, actual_db_path)

    return SelectorGroupChat(
        participants=[
            sql_ingestor.agent,
            eda_specialist.agent,
            preprocess_agent.agent,
            critic  # This is already an agent object from get_quality_critic
        ],
        model_client=model_client,
        selector_func=prep_selector,  # Using the new state-based selector
        termination_condition=TextMentionTermination("DATA_VERIFIED")
    )