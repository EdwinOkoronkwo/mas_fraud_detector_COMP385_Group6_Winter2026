from autogen_agentchat.agents import AssistantAgent


class LeadInvestigator:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Lead_Investigator",
            model_client=model_client,
            system_message="""You are the Lead Fraud Investigator. 
            MISSION: Orchestrate the transition from Raw Data to Gold Tier Inference.

            WORKFLOW:
            1. **Coordinate**: Pass raw transaction data to the 'Data_Preprocessor'.
            2. **Monitor**: Ensure the 'Data_Preprocessor' produces the 24-feature aligned dictionary.
            3. **Handover**: Once data is scaled, instruct the 'Inference_Specialist' to execute 'gold_xgb' inference.
            4. **Finalize**: Only after the Inference_Critic has verified the results and the Synthesis report is complete, conclude with 'CASE_CLOSED'.

            STRICT RULE:
            We are using the Gold Tier XGBoost (24 features). Ignore any mention of 12 or 22 features.
            """
        )