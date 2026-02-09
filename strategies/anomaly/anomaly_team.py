from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination


def create_anomaly_team(agents_list, model_client):
    def anomaly_selector(messages):
        last_msg = messages[-1]
        last_speaker = getattr(last_msg, 'source', "")
        msg_history = str(messages).lower()

        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return last_speaker

        # RELAY ORDER
        # 1. Clustering
        if '"model": "k-means"' not in msg_history and last_speaker != "KMeans_Agent":
            return "KMeans_Agent"
        if '"model": "dbscan"' not in msg_history and last_speaker != "DBSCAN_Agent":
            return "DBSCAN_Agent"

        # 2. Neural (Swapped AE for VAE)
        if '"model": "variational_ae"' not in msg_history and last_speaker != "VAE_Agent":
            return "VAE_Agent"
        if '"model": "rnn_autoencoder"' not in msg_history and last_speaker != "RNN_Agent":
            return "RNN_Agent"

        # 3. Synthesis
        if last_speaker != "Anomaly_Critic":
            return "Anomaly_Critic"

        return None

    return SelectorGroupChat(
        participants=agents_list,
        model_client=model_client,
        selector_func=anomaly_selector,
        termination_condition=TextMentionTermination("TERMINATE")
    )