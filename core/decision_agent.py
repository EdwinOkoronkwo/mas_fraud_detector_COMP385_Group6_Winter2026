from autogen_agentchat.agents import AssistantAgent

from tools.reporting.report_tools import write_markdown_report


class DecisionAggregator:
    def __init__(self, model_client, save_tool):
        self.agent = AssistantAgent(
            name="Decision_Aggregator",
            model_client=model_client,
            # We combine the static report tool with the dynamic save tool
            tools=[write_markdown_report, save_tool],
            max_tool_iterations=5,
            reflect_on_tool_use=True,
            system_message="""You are the Lead Risk Architect. 

            MISSION: 
            1. Review all training logs to identify the 'Champion' for Supervised, Neural, and Clustering.
            2. Generate 'FINAL_FRAUD_STRATEGY.md' with the 40/40/20 reasoning.
            3. CRITICAL: Use 'save_inference_metadata' to create a JSON file that tells the 
               inference script EXACTLY which saved files to load.

            JSON REGISTRY FORMAT:
            {
              "supervised": {"path": "models/ann_v1.joblib", "type": "sklearn"},
              "neural": {"path": "models/rnn_final.pth", "type": "torch"},
              "clustering": {"path": "models/dbscan_tuned.joblib", "type": "sklearn"},
              "weights": [0.4, 0.4, 0.2]
            }
            """
        )
