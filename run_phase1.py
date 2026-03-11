import asyncio
from config.settings import settings
from config.llm_config import get_model_client
from fraud_pipeline import FraudPipeline
from strategies.data_preprocessing.agent_reporter import AgentReporter
from strategies.data_preprocessing.data_auditor import DataAuditor
from utils.logger import setup_logger

# Initialize Logger
logger = setup_logger("Phase1_Runner")


async def run_phase1():
    """
    Orchestrates the transition from Raw Kaggle CSV to
    Balanced, Scaled, and Serialized Training Data.
    """
    # 1. Setup Environment & Models
    logger.info("🚀 Initializing Phase 1 Environment...")
    model_client = get_model_client()

    # The Pipeline contains the SQL_Ingestor, Preprocessing_Agent, and Sampling_Agent
    pipeline = FraudPipeline(settings, model_client)

    # Audit and Reporting Tools
    auditor = DataAuditor(settings, logger)
    reporter = AgentReporter(settings, logger)

    # 2. Execute Multi-Agent Pipeline
    # This runs the full sequence: Purge -> Ingest -> Preprocess -> SMOTE/Split
    logger.info("⏳ Executing Phase 1 Pipeline (Ingestion & Preprocessing)...")
    result = await pipeline.run_data_foundation()

    # Get the last message content
    last_message = result.messages[-1].content

    # BROADENED SUCCESS CRITERIA:
    # If the agents finished the report or verified data, we proceed to Audit
    success_keywords = ["DATA_VERIFIED", "SUCCESS", "REPORT published"]

    if any(keyword in last_message for keyword in success_keywords):
        logger.info("✅ Pipeline tasks finished. Starting Final Audit & Handoff Inspection...")

        reporter.process_results(result)

        # This triggers the log you wanted: Path + Data Head
        auditor.verify_physical_artifacts()
        auditor.run_database_inspection()
        handoff_ok = auditor.verify_sql_handoff()

        if handoff_ok:
            logger.info("=" * 50)
            logger.info("🏁 PHASE 1 COMPLETE: SQL Foundation is locked.")
            logger.info(f"🔗 Phase 2 can now run: pd.read_sql('SELECT * FROM cleaned_scaled_data', engine)")
            logger.info("=" * 50)
        else:
            logger.error("❌ Phase 1 failed Quality Gate checks.")
    else:
        logger.error(f"❌ Pipeline failed. Last message: {last_message}")


if __name__ == "__main__":
    try:
        asyncio.run(run_phase1())
    except KeyboardInterrupt:
        logger.info("🛑 Phase 1 interrupted by user.")
    except Exception as e:
        logger.error(f"💥 Critical Error in Runner: {e}")