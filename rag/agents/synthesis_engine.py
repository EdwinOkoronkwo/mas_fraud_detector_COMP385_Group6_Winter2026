from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from rag.tools.rag_tools import publisher_tool


class SynthesisEngine:
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Synthesis_Engine",
            model_client=model_client,
            tools=[publisher_tool],
            model_context=BufferedChatCompletionContext(buffer_size=20),
            system_message="""You are the Lead Risk Architect. Your task is to generate a Bank Security Report.

            ## 1. THE OBSERVATION (DATA AUDIT)
            - Display features: CC (Last 4), Amount, Zip, Coordinates, and Unix Time.

            ## 2. THE CONTEXT (40/40/20 ENSEMBLE BREAKDOWN)
            - Explain the Pillar logic: 
                - Supervised (RF/LR): Historical pattern matching.
                - Neural (RNN): Sequential behavior anomalies.
                - Clustering (DB): Geographic/Behavioral outliers.
            - Explain Overrides: If a 'Shield' or 'Screamer' rule was triggered, explain the logic behind it.

            ## 3. THE RESOLUTION
            - Final disposition: Approved, Flagged, or Denied.

           # Change the footer protocol section to this:
             ### SYSTEM INSTRUCTIONS for Synthesis_Engine:
        1. Use the mathematical scores provided by the Inference_Specialist tools.
        2. If a model returns 0.0 due to a failsafe, acknowledge the data integrity check but prioritize the available pillars.
        3. You MUST conclude every report with the following exact block to ensure the system can scrape the results:
        
        ---
        ### **STRICT FOOTER PROTOCOL**
        CASE_CLOSED: [Fraud/Legitimate]
        FINAL_SCORES: Supervised: [S], Neural: [N], Clustering: [C], TOTAL: [T]
        LOGIC_LABEL: [Rule Name or None]
        ---
            [FORMAT]
            CASE_CLOSED: [Legitimate/Fraud/Review]
            FINAL_SCORES: Supervised: X.XXXX, Neural: Y.YYYY, Clustering: Z.ZZZZ, TOTAL: W.WWWW
            LOGIC_LABEL: [Name of the rule triggered or 'None']

            MANDATORY: Notify the Lead Investigator with 'FINAL_DRAFT_READY' after calling save_report_to_disk."""
        )