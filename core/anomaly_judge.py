from autogen_agentchat.agents import AssistantAgent


class AnomalyJudge:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Anomaly_Judge",
            model_client=model_client,
            system_message="""You are the Anomaly Selection Judge.
            1. Review results from RNN-AE and Standard Autoencoder.
            2. Selection Metric: Prioritize the model with the highest 'Ground Truth Alignment %'.
            3. CRITICAL: Only one model can be the 'Anomaly Champion'. 
            4. Output only the name of the winning model and its final alignment score."""
        )