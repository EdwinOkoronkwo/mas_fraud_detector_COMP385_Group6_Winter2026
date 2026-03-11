import pandas as pd

class TaskArchitect:
    def create_investigation_task(self, row: pd.Series, scaled_features: dict) -> str:
        cc_str = str(row['cc_num']).split('.')[0][-4:] # Just last 4 for privacy/clarity
        amt = row['amt']

        # Inject the 22-feature vector into the prompt for the Specialist
        feature_context = "\n".join([f"- {k}: {v:.4f}" for k, v in scaled_features.items()])

        return (
            f"--- PHASE 4: SUPERVISED PILLAR VALIDATION ---\n"
            f"Target: CC ending in {cc_str} | Transaction: ${amt:.2f}\n\n"
            f"### SCALED FEATURE VECTOR:\n"
            f"{feature_context}\n\n"
            "OPERATIONAL REQUIREMENT:\n"
            "1. Pass the above vector to the 'execute_supervised_validation' tool.\n"
            "2. Report the raw probability and the weighted impact for the CHAMPION model.\n"
            "3. Do NOT estimate Neural or Clustering scores yet."
        )

# class TaskArchitect:
#     def create_investigation_task(self, row: pd.Series, scaled_features: dict) -> str:
#         cc_str = str(row['cc_num']).split('.')[0]
#         amt = row['amt']
#
#         sensitivity_note = ""
#         if amt > 1000:
#             sensitivity_note = "CRITICAL: High-value transaction. Prioritize RNN anomalies."
#         elif 300 < amt <= 1000:
#             sensitivity_note = "NOTICE: Use high-sensitivity threshold for borderline patterns."
#
#         # NEW: Inject the 22-feature vector into the prompt
#         feature_context = "\n".join([f"- {k}: {v:.4f}" for k, v in scaled_features.items()])
#
#         return (
#             f"Investigate CC {cc_str} for a transaction of ${amt:.2f}.\n"
#             f"{sensitivity_note}\n\n"
#             f"### SCALED FEATURE VECTOR (Input for Model Tools):\n"
#             f"{feature_context}\n\n"
#             "Provide a breakdown for RF, RNN, and DBSCAN, and the final 'TOTAL RISK SCORE'."
#         )