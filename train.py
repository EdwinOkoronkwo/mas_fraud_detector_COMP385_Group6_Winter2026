import asyncio
import json
import os
import shutil

from config import settings
from config.llm_config import get_model_client
from fraud_pipeline import FraudPipeline
import asyncio
from config import settings
from config.llm_config import get_model_client
from fraud_pipeline import FraudPipeline
from strategies.data_preprocessing.agent_reporter import AgentReporter
from strategies.data_preprocessing.data_auditor import DataAuditor
# from strategies.supervised.baseline_trainer import ModelTrainer
from utils.logger import setup_logger

# Initialize Logger
logger = setup_logger("Phase1 and Phase 2 Runner")

import os
import shutil
import asyncio
import logging
from config import settings
# Ensure these imports match your project structure

from strategies.data_preprocessing.data_auditor import DataAuditor

# Setup a local logger for the runner
logger = logging.getLogger("Phase1_Runner")


async def run_training_system():
    # --- 1. NUCLEAR RESET (Physical Cleanse) ---
    folders_to_clean = ['./.cache', './__pycache__']

    # CHANGE: Match the .joblib extension used in ModelTrainer
    shared_data_path = os.path.join(settings.PROJECT_ROOT, 'data/temp_split.joblib')

    files_to_clean = [
        settings.DB_PATH,
        shared_data_path
    ]

    print("🛠️  PREPARING CLEAN SLATE...")
    for folder in folders_to_clean:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"🧹 Cleaned folder: {folder}")

    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted file: {file_path}")
            except PermissionError:
                print(f"⚠️ Could not delete {file_path}. Close other Python processes.")

    # --- 2. INITIALIZE ---
    model_client = get_model_client()
    pipeline = FraudPipeline(settings, model_client)
    auditor = DataAuditor(settings, logger)

    # ADD: Initialize the Trainer here
    # trainer = ModelTrainer(settings)

    try:
        # --- PHASE 1: DATA FOUNDATION ---
        print("\n🚀 RUNNING PHASE 1: DATA PREPARATION...")
        prep_result = await pipeline.run_data_foundation()

        # Check for Phase 1 consensus
        if not any("DATA_VERIFIED" in m.content for m in prep_result.messages[-2:]):
            print("⚠️ Phase 1 finished without explicit verification.")

        # --- 3. THE MANDATORY AUDIT GATE ---
        if auditor.verify_sql_handoff():
            print("✅ SQL Table Verified. Phase 1 Handoff Successful.")
        else:
            print("❌ SQL Verification Failed. Table is not model-ready. Aborting.")
            return

        # --- NEW: BASELINE & DATA SHARING PHASE ---
        print("\n🔨 ESTABLISHING NON-AI BASELINE...")
        # 1. Train the model and get paths
        # baseline_path, shared_data_path = trainer.train_manual_xgb()
        #
        # print(f"✅ Baseline established and exported independently.")

        # --- PHASE 2: SUPERVISED CHAMPIONSHIP ---
        print("\n🏆 STARTING PHASE 2: SUPERVISED TOURNAMENT...")
        sup_result = await pipeline.run_supervised_tournament()
        winner_msg = sup_result.messages[-1].content if sup_result.messages else "No result"
        print(f"🥇 TOURNAMENT WINNER: {winner_msg[:100]}...")

        # --- PHASE 3: ANOMALY DISCOVERY (Unsupervised) ---
        print("\n🔍 STARTING PHASE 3: ANOMALY DISCOVERY...")
        # This uncovers hidden fraud patterns via VAE/Isolation Forest
        anom_result = await pipeline.run_anomaly_discovery()
        anom_msg = anom_result.messages[-1].content if anom_result.messages else "No discovery logs found"
        print("🕵️ Anomaly Analysis Complete.")

        # --- PHASE 4: FINAL AGGREGATION & REPORTING ---
        print("\n📊 STARTING PHASE 4: FINAL AGGREGATION...")
        # Passing results to the 'CEO' agent for the final executive report
        await pipeline.run_aggregation({
            "prep": prep_result,
            "sup": sup_result,
            "anom": anom_result
        })

        print("\n🏁 PIPELINE EXECUTION FINISHED: All Models & Reports Saved.")

    except Exception as e:
        print(f"❌ Pipeline Critical Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(run_training_system())
    except KeyboardInterrupt:
        print("\n🛑 Execution stopped by user.")


# async def run_training_system():
#     # --- 1. NUCLEAR RESET ---
#     folders_to_clean = ['./.cache', './__pycache__']
#     files_to_clean = [settings.DB_PATH, os.path.join(settings.PROJECT_ROOT, 'data/temp_split.pkl')]
#
#     print("🛠️  PREPARING CLEAN SLATE...")
#     for folder in folders_to_clean:
#         if os.path.exists(folder):
#             shutil.rmtree(folder)
#             print(f"🧹 Cleaned folder: {folder}")
#
#     for file_path in files_to_clean:
#         if os.path.exists(file_path):
#             try:
#                 os.remove(file_path)
#                 print(f"🗑️ Deleted file: {file_path}")
#             except PermissionError:
#                 print(f"⚠️ Could not delete {file_path}. Close other Python processes.")
#
#     # --- 2. INITIALIZE ---
#     model_client = get_model_client()
#     pipeline = FraudPipeline(settings, model_client)
#     auditor = DataAuditor(settings, logger)
#
#     try:
#         # --- PHASE 1: DATA FOUNDATION ---
#         print("\n🚀 RUNNING PHASE 1 FROM SCRATCH...")
#         prep_result = await pipeline.run_data_foundation()
#
#         # USE PREP_RESULT: Validate agent consensus before auditing files
#         last_msg = prep_result.messages[-1].content if prep_result.messages else ""
#         if "DATA_VERIFIED" not in last_msg:
#             print(f"⚠️ Phase 1 finished without explicit verification. Last message: {last_msg[:100]}...")
#             # We continue anyway to let the Auditor have the final say
#
#         # --- 3. THE MANDATORY AUDIT GATE ---
#         if auditor.verify_sql_handoff():
#             print("✅ SQL Table Verified. Phase 1 Handoff Successful.")
#         else:
#             print("❌ SQL Verification Failed. Table is not model-ready. Aborting.")
#             return
#
#         # --- PHASE 2: SUPERVISED CHAMPIONSHIP ---
#         print("\n🏆 STARTING PHASE 2: SUPERVISED TOURNAMENT...")
#         sup_result = await pipeline.run_supervised_tournament()
#
#         # USE SUP_RESULT: Log the winner of the tournament
#         # This ensures you know which model became the 'Champion'
#         winner_msg = sup_result.messages[-1].content if sup_result.messages else "No result"
#         print(f"\n🥇 TOURNAMENT WINNER: {winner_msg}")
#
#         print("\n🏁 PIPELINE EXECUTION FINISHED.")
#
#     except Exception as e:
#         print(f"❌ Pipeline Critical Failure: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     try:
#         asyncio.run(run_training_system())
#     except KeyboardInterrupt:
#         print("\n🛑 Execution stopped by user.")

#
# async def run_training_system():
#     model_client = get_model_client()
#     pipeline = FraudPipeline(settings, model_client)
#
#     try:
#         # Phase 1: Data Preparation
#         prep_result = await pipeline.run_data_foundation() # Double-check this name!
#
#         # Phase 2: Supervised Championship
#         sup_result = await pipeline.run_supervised_tournament()
#
#         # Phase 3: Anomaly Discovery (Clustering & Neural)
#         anom_result = await pipeline.run_anomaly_discovery()
#
#         # Phase 4: Final Aggregation (Passing the dictionary for the agent)
#         await pipeline.run_aggregation({
#             "prep": prep_result,
#             "sup": sup_result,
#             "anom": anom_result
#         })
#
#         print("\n🏆 TRAINING COMPLETE: Models saved and Registry updated.")
#
#     except Exception as e:
#         print(f"❌ Pipeline Failed: {e}")
#         # Consider logging the full traceback here for easier debugging
#         import traceback
#         traceback.print_exc()
#
# if __name__ == "__main__":
#     asyncio.run(run_training_system())