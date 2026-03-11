from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat


def create_neuro_team(agents_list, model_client):
    def neuro_selector(messages):
        last_msg = messages[-1]
        last_speaker = getattr(last_msg, 'source', "")
        msg_history = str(messages).lower()

        # 1. TOOL HANDSHAKE
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return last_speaker

        # 2. SEQUENTIAL PROGRESSION
        has_ae = '"model": "autoencoder"' in msg_history
        has_vae = '"model": "variational_ae"' in msg_history
        has_rnn = '"model": "rnn_autoencoder"' in msg_history

        if not has_ae and last_speaker != "AE_Agent":
            return "AE_Agent"
        if has_ae and not has_vae and last_speaker != "VAE_Agent":
            return "VAE_Agent"
        if has_vae and not has_rnn and last_speaker != "RNN_Agent":
            return "RNN_Agent"

        # 3. THE TERMINATION GATE
        # If all models are done and the Critic hasn't spoken yet, pick Critic.
        if has_ae and has_vae and has_rnn and last_speaker != "Neuro_Critic":
            return "Neuro_Critic"

        # 4. EMERGENCY BRAKE: If the Critic just spoke, STOP.
        # Returning None prevents the Selector from picking an Assistant twice.
        return None

    termination = TextMentionTermination("TERMINATE")

    return SelectorGroupChat(
        participants=agents_list,
        model_client=model_client,
        selector_func=neuro_selector,
        termination_condition=termination  # ADD THIS
    )