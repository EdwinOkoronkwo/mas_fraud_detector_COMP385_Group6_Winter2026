from autogen_agentchat.agents import AssistantAgent

from strategies.anomaly.clustering.critic_agent import verify_anomaly_labels


class NeuroCritic:
    def __init__(self, model_client, db_path):
        self.agent = AssistantAgent(
            name="Neuro_Critic",
            model_client=model_client,
            tools=[verify_anomaly_labels],
            system_message=f"""[INST] You are the Neural Audit Lead.

            GOAL: Evaluate deep-learning reconstruction models against known labels in {db_path}.

            WORKFLOW:
           Compare the results from:
            1. REVIEW: Look for the JSON outputs from AE_Agent, VAE_Agent, and RNN_Agent.
            2. VERIFY: Call 'verify_anomaly_labels' using the indices provided by the champions.
            3. NO HALLUCINATION: If a model failed or its output is missing, report 'N/A'. 
               DO NOT calculate alignment percentages unless you have run 'verify_anomaly_labels'.
            4. PERSIST: Save the best performing model using 'persist_champion_model'.
            5. TERMINATE: End the session once the neural champion is verified.

            Run 'verify_anomaly_labels' to compare anomaly counts against the 733 actual frauds.
            Crown the champion and call 'persist_champion_model' with the result.
            Final word MUST be TERMINATE.

            REPORTING FORMAT:
            Produce the 'Neural Reconstruction Championship':
            | Model Name | Reconstruction MSE | Anomaly Count | Ground Truth Alignment |
            |------------|-------------------|---------------|-----------------------|

            4. Crown the 'Neuro Champion' (The model that isolated fraud most precisely via reconstruction error).
            5. Final word MUST be TERMINATE. [/INST]"""
        )