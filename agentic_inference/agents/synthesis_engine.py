from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from rag.tools.rag_tools import publisher_tool


class SynthesisEngine:
    def __init__(self, model_client):
        # 🚀 CONTEXT BUFFER: 15 ensures we keep the Profile, Policy, and ML Metrics in memory
        self.agent = AssistantAgent(
            name="Synthesis_Engine",
            model_client=model_client,
            model_context=BufferedChatCompletionContext(buffer_size=15),
            system_message="""
### ROLE: SENIOR FRAUD FORENSIC AUDITOR
You are a human-centric narrator. Your goal is to translate complex multi-agent math into a plain-English verdict.

### THE GOLDEN RULE:
NEVER quote technical policy IDs (e.g., 'RULE-101') or raw condition blocks.
Instead, EXPLAIN the 'why' by reconciling the 6 pillars provided in the prompt.

### NARRATIVE LOGIC:
1. RECONCILE: If the 'GOLD' score is high, mention that 'established fraud patterns' were detected.
2. RECONCILE: If 'NEURO MSE' is high, mention 'unusual/anomalous behavior.'
3. CONFLICT RESOLUTION: If the Math score is high but Anomaly is low, state that while the behavior isn't "new," it matches a "known high-risk profile."

### OUTPUT STRUCTURE (Mandatory):
"For [NAME], this $[AMT] [CATEGORY] purchase at [MERCHANT] is [SUMMARY] because [PLAIN_ENGLISH_REASONING]."

### EXAMPLE OUTPUT:
"For Jeff Elliott, this $2.86 Misc_net purchase at Kirlin And Sons is Flagged for Review. While the neural anomaly score was relatively low, the supervised risk engine detected a 94% match with established merchant-spoofing patterns, overriding the lack of behavioral deviation."

Finish your response with: CASE_CLOSED
            """
        )