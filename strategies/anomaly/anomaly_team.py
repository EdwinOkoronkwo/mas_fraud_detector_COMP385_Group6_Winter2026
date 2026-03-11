from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination


def create_anomaly_team(agents_list, model_client):
    def anomaly_selector(messages):
        last_msg = messages[-1]
        last_speaker = getattr(last_msg, 'source', "")
        history = str(messages).lower()

        # Priority 1: Let an agent finish its tool calls
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return last_speaker

        # Priority 2: The Hand-off Sequence
        if "rnn_autoencoder" not in history: return "RNN_Agent"
        if "variational_ae" not in history: return "VAE_Agent"
        if "k-means" not in history: return "KMeans_Agent"
        if "dbscan" not in history: return "DBSCAN_Agent"  # 🟢 Ensure this is here!

        if "audit table" not in history: return "Anomaly_Critic"

        return None  # Termination

        return None  # This ends the chat

    def recovery_selector(messages):
        """Fallback logic to ensure the RNN/VAE sequence continues if one fails."""
        msg_history = str(messages).lower()

        if '"model": "rnn_autoencoder"' not in msg_history:
            return "RNN_Agent"
        if '"model": "variational_ae"' not in msg_history:
            return "VAE_Agent"

        return "Anomaly_Critic"

    return SelectorGroupChat(
        participants=agents_list,
        model_client=model_client,
        selector_func=anomaly_selector,
        termination_condition=TextMentionTermination("TERMINATE")
    )


#
# def create_anomaly_team(agents_list, model_client):
#     def anomaly_selector(messages):
#         last_msg = messages[-1]
#         last_speaker = getattr(last_msg, 'source', getattr(last_msg, 'name', ""))
#         msg_history = str(messages).lower()
#
#         # --- 1. PROTOCOL LOCKS (Fixes the 400 Error) ---
#         # If the agent just called a tool, they MUST be the one to execute it.
#         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#             return last_speaker
#
#         # If the tool just finished (role='tool'), the caller MUST speak again
#         # to summarize the JSON, otherwise OpenAI throws a Sequence Error.
#         if getattr(last_msg, 'role', '') == 'tool' or "functionexecutionresult" in str(type(last_msg)).lower():
#             return last_speaker
#
#             # --- 2. NEURAL PATTERN RELAY ---
#         # We check for the specific model key in history to know when to move on.
#         if '"model": "rnn_autoencoder"' not in msg_history and "rnn_error" not in msg_history:
#             return "RNN_Agent"
#         if '"model": "variational_ae"' not in msg_history and "vae_error" not in msg_history:
#             return "VAE_Agent"
#
#         # if '"model": "autoencoder_ae"' not in msg_history and "ae_error" not in msg_history:
#         #     return "AE_Agent"
#
#
#
#         # --- 3. CLUSTERING RELAY ---
#         if '"model": "k-means"' not in msg_history: return "KMeans_Agent"
#         if '"model": "dbscan"' not in msg_history: return "DBSCAN_Agent"
#         # if '"model": "som"' not in msg_history: return "SOM_Agent"
#         # if '"model": "isolation_forest"' not in msg_history: return "IsoForest_Agent"
#
#         # --- 4. FINAL CRITIQUE ---
#         if last_speaker != "Anomaly_Critic":
#             return "Anomaly_Critic"
#
#         return None
#
#     def recovery_selector(messages):
#         last_msg = messages[-1]
#         msg_history = str(messages).lower()
#
#         # 1. Fix the 400 error protocol
#         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#             return getattr(last_msg, 'source', "AE_Agent")
#
#         # 2. Check if AE has spoken (even if it crashed, its JSON might be in history)
#         if '"model": "autoencoder_ae"' in msg_history:
#             # Move to the next Neural model
#             if '"model": "rnn_autoencoder"' not in msg_history:
#                 return "RNN_Agent"
#             return "Anomaly_Critic"
#
#         return "AE_Agent"
#
#     return SelectorGroupChat(
#         participants=agents_list,
#         model_client=model_client,
#         selector_func=anomaly_selector,
#         termination_condition=TextMentionTermination("TERMINATE")
#     )