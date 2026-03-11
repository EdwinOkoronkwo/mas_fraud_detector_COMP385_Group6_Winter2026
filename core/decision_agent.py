import os

from autogen_agentchat.agents import AssistantAgent

from tools.reporting.report_tools import write_markdown_report

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_core import CancellationToken

import os
import joblib
import json



class DecisionAggregator:
    def __init__(self, model_client, settings, save_tool, tournament_results=None):
        self.settings = settings

        # 1. Capture the actual metrics from the tournament tools
        results_context = tournament_results if tournament_results else "No results found yet."

        # 2. Get the 24 features directly from the source of truth
        feature_list = self.get_feature_list()

        self.agent = AssistantAgent(
            name="Decision_Aggregator",
            model_client=model_client,
            tools=[write_markdown_report, save_tool],
            system_message=f"""You are the Lead Risk Architect.

            ### DATA SOURCE: DataSpecialist Bundle (.joblib)
            ### FEATURES USED: {feature_list}

            ### REAL-TIME TOURNAMENT DATA:
            {results_context}

            ### YOUR MISSION:
            1. ANALYZE: Use the TP, Recall, and Precision from the data above.
            2. COMPARE: Evaluate the Supervised (XGB/RF), Neuro (MLP/VAE/RNN), and Clustering (DBSCAN/K-Means) results.
            3. CHAMPION: Pick the model with the best Recall-to-Precision balance for fraud detection.
            4. ARCHIVE: Call 'save_tool' to move the winning artifact to 'models/production_champion'.
            5. REPORT: Write 'FINAL_FRAUD_STRATEGY.md'. Include a table comparing the models and list the {len(feature_list)} features used.
            """
        )

    def get_feature_list(self):
        """
        STRICT SYNC: Pulls the feature list only from the DataSpecialist bundle.
        """
        try:
            # This path is defined by your DataSpecialist output
            bundle_path = "data/temp_split.joblib"

            if not os.path.exists(bundle_path):
                raise FileNotFoundError(f"Critical Error: {bundle_path} not found. DataSpecialist must run first.")

            data_bundle = joblib.load(bundle_path)

            # Extract the features list stored by the Specialist
            features = data_bundle.get('features', [])

            if not features:
                # If for some reason the list is missing, we extract from the training DataFrame columns
                X_train, _ = data_bundle.get('train')
                if hasattr(X_train, 'columns'):
                    features = X_train.columns.tolist()
                else:
                    # If it's a NumPy array, we report the count
                    return [f"Feature_{i}" for i in range(X_train.shape[1])]

            return features

        except Exception as e:
            return f"Error retrieving feature list: {str(e)}"