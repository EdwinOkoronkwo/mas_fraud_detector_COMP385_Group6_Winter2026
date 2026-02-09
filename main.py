# import asyncio
# import os
#
# from mas_fraud_detector.config import settings
# from mas_fraud_detector.config.llm_config import get_model_client
# from mas_fraud_detector.config.settings import AGENT_CONFIG, TEMP_SPLIT_PATH
# from mas_fraud_detector.core.decision_agent import DecisionAggregator
# from mas_fraud_detector.core.state import PipelineState
# from mas_fraud_detector.core.orchestrator import Orchestrator
# from mas_fraud_detector.factories.agent_factory import AgentFactory
# from autogen_core.models import UserMessage
#
# from mas_fraud_detector.tools.training.deployment_tools import persist_champion_model
# from mas_fraud_detector.utils.logger import setup_logger
#
# logger = setup_logger("Main")
#
#
# def print_agent_chat(phase_name, result):
#     """Helper to display the multi-agent conversation."""
#     print(f"\n{'=' * 20} {phase_name.upper()} CHAT LOGS {'=' * 20}")
#     for message in result.messages:
#         author = message.source
#         content = message.content
#         print(f"\n[{author.upper()}]:\n{content}")
#         print("-" * 30)
#
#
# async def main():
#     # 0. Infrastructure Guard
#     # Ensure all required directories exist before agents start working
#     for folder in ["mas_fraud_detector/models", "reports", "data/processed"]:
#         os.makedirs(folder, exist_ok=True)
#     # 1. Setup
#     model_client = get_model_client()
#     state = PipelineState()
#     orchestrator = Orchestrator(state)
#     factory = AgentFactory(model_client=model_client, config=AGENT_CONFIG)
#
#     try:
#         # --- PHASE 1: DATA FOUNDATION ---
#         logger.info("Starting Agentic Preprocessing Team...")
#         result_prep = await orchestrator.execute_phase(
#             phase_name="Data_Foundation",
#             runner_factory=factory.get_preprocessing_team,
#             task="""1. Migrate CSV data to SQL.
#                     2. Perform EDA and scale features."""
#         )
#         data_prep_final_text = result_prep.messages[-2].content if len(result_prep.messages) > 1 else "Data Verified"
#
#         # --- PHASE 2: SUPERVISED TOURNAMENT ---
#         logger.info("Starting Supervised Championship...")
#         task_s = f"Sampling_Agent, start the balancing protocol using {settings.DB_PATH} now."
#         result_train = await orchestrator.execute_phase(
#             phase_name="Supervised_Championship",
#             runner_factory=factory.get_supervised_championship_team,
#             task=task_s
#         )
#         # Capture summary for the final report
#         supervised_final_text = result_train.messages[-2].content if len(result_train.messages) > 1 else "N/A"
#
#         # --- PHASE 3: ANOMALY DISCOVERY ---
#         logger.info("Starting Unified Anomaly Discovery Team...")
#
#         # Define the string first
#         instruction_3 = f"""
#                 1. Run the clustering and neural reconstruction relay on {settings.DB_PATH}.
#                 2. Use the 'verify_anomaly_labels' tool to get the ground truth count (733).
#                 3. FOR EACH MODEL: Calculate the 'Ground Truth Alignment %'.
#                    - This is (Count of actual frauds detected by model / 733) * 100.
#                 4. Populate the final table with these percentages instead of 'Pending'.
#                 """
#
#         result_anomaly = await orchestrator.execute_phase(
#             phase_name="Anomaly_Discovery",
#             runner_factory=factory.get_anomaly_discovery_team,
#             task=instruction_3  # Changed from task_anomaly= to task=
#         )
#         print_agent_chat("Anomaly Discovery", result_anomaly)
#         # This summary now contains BOTH Clustering and Neural results for the Critic
#         anomaly_final_text = result_anomaly.messages[-2].content if len(result_anomaly.messages) > 1 else "N/A"
#
#         # --- PHASE 4: DECISION AGGREGATION ---
#         logger.info("--- Starting Decision Aggregation Phase ---")
#
#         # Capture all three phase summaries
#         data_prep_summary = str(result_prep.messages[-2].content) if result_prep else "Data was cleaned and scaled."
#         supervised_summary = str(result_train.messages[-2].content) if len(result_train.messages) > 1 else "N/A"
#         anomaly_summary = str(result_anomaly.messages[-2].content) if len(result_anomaly.messages) > 1 else "N/A"
#
#         agg_task = f"""
#         ### MISSION: Generate the 'FINAL_FRAUD_STRATEGY.md'
#         You are a Senior Risk Architect who specializes in ensemble detection logic and professional Markdown reporting
#         . Use the 'write_markdown_report' tool.
#
#         ### INPUT DATA:
#         1. **Data Prep:** {data_prep_summary}
#         2. **Supervised Phase:** {supervised_summary}
#         3. **Anomaly Phase:** {anomaly_summary}
#         4. **Data Prep Status (Quality Critic):** {data_prep_final_text}
#         5. **Supervised Results:** {supervised_final_text}
#         6. **Anomaly Discovery Results:** {anomaly_final_text}
#
#         ### REPORT REQUIREMENTS:
#         - **Data Foundation:** Discuss the preprocessing (Scaling, SMOTE, SQL migration) and why it was critical for the ANN's stability.
#         - **Aggregation Logic:** Explicitly justify the 40/40/20 weights.
#         - **Reasoning:** Explain that Supervised catches 'Knowns', Neural/RNN catches 'Sequences' (56% alignment), and Clustering catches 'Outliers'.
#         - **Sample Calculation:** Show how a transaction flagged by both ANN and RNN hits a Risk Score of 0.8.
#         - **MANDATORY:** Call the 'write_markdown_report' tool to save the file.
#         ### ENHANCED REPORT REQUIREMENTS:
#             - **Data Foundation Reasoning:** Use the 'row_counts' and 'imbalance_ratio' from the Quality Critic to explain WHY the supervised models needed SMOTE and why the anomaly models were run on the full 129k record set.
#             - **Aggregation Calculation:** Show a sample calculation.
#               - If a transaction is flagged by **ANN** (0.4) and **RNN** (0.4) = **0.8 Risk (High)**.
#               - If only flagged by **DBSCAN** (0.2) = **0.2 Risk (Low)**.
#             - **Strategic Justification:** Explain that the 40% RNN weight is chosen because it aligned with 56% of the 733 verified frauds—bridging the gap between pure outliers and known patterns.
#             - **MANDATORY:** You MUST call the 'write_markdown_report' tool
#         """
#
#         await orchestrator.execute_phase(
#             phase_name="Decision_Aggregation",
#             runner_factory=factory.get_decision_aggregator_agent,
#             task=agg_task
#         )
#         logger.info("--- [ALL PHASES COMPLETE] ---")
#
#     except Exception as e:
#         logger.critical(f"Pipeline Execution Failed: {e}")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())