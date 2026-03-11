# mas_fraud_detector/strategies/unsupervised/unsupervised_team.py

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination


def clustering_selector(messages):
    # Standard Tool Handshake
    last_msg = messages[-1]
    last_speaker = getattr(last_msg, 'source', "")

    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return last_speaker

    # FIXED RELAY ORDER
    # This acts like Round Robin but stops at the Critic.
    relay = {
        "User": "KMeans_Agent",
        "KMeans_Agent": "DBSCAN_Agent",
        "DBSCAN_Agent": "SOM_Agent",
        "SOM_Agent": "IsoForest_Agent",
        "IsoForest_Agent": "Anomaly_Critic",
        "Anomaly_Critic": None  # TERMINATE
    }

    # If the last speaker is in our relay, pick the next one.
    # Otherwise, start with KMeans.
    next_agent = relay.get(last_speaker, "KMeans_Agent")

    print(f"--- [RELAY SELECTOR]: {last_speaker} -> {next_agent} ---")
    return next_agent


# FIND THIS FUNCTION AND REMOVE THE SETTER LOOP
def create_clustering_team(agents_list, model_client):
    all_participants = agents_list

    return SelectorGroupChat(
        participants=all_participants,
        model_client=model_client,
        selector_func=clustering_selector,
        termination_condition=TextMentionTermination("TERMINATE")
    )