import autogen_agentchat.teams
from autogen_agentchat.conditions import TextMentionTermination
from strategies.data_preprocessing.quality_critic import get_quality_critic

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination


def prep_selector(messages):
    """
    Ensures a strict sequence for Ollama:
    Ingest -> EDA -> Preprocess -> Critic
    """
    history = [m.source.lower() for m in messages]

    if "sql_ingestor" not in history:
        return "SQL_Ingestor"
    if "eda_specialist" not in history:
        return "EDA_Specialist"
    if "preprocess_agent" not in history:
        return "Preprocess_Agent"
    # Once the main 3 are done, hand off to the Quality Critic
    return "Quality_Critic"


def create_preprocessing_team(sql_ingestor, eda_specialist, preprocess_agent, model_client):
    """
    Assembles the data prep team using a fixed-sequence selector to prevent Ollama errors.
    """
    actual_db_path = sql_ingestor.config.get("DB_PATH")

    # Initialize the Critic with a unique name for the selector
    critic = get_quality_critic(model_client, actual_db_path)

    termination = TextMentionTermination("DATA_VERIFIED")

    # Change to SelectorGroupChat to use your custom selector_func
    team = SelectorGroupChat(
        participants=[
            sql_ingestor.agent,
            eda_specialist.agent,
            preprocess_agent.agent,
            critic
        ],
        model_client=model_client,  # Required for SelectorGroupChat
        selector_func=prep_selector,
        termination_condition=termination
    )

    return team