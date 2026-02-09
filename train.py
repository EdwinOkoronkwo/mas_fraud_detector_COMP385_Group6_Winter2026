import asyncio

from config import settings
from config.llm_config import get_model_client
from fraud_pipeline import FraudPipeline


async def run_training_system():
    model_client = get_model_client()
    pipeline = FraudPipeline(settings, model_client)

    try:
        # Phase 1: Data Preparation
        prep_result = await pipeline.run_data_foundation() # Double-check this name!

        # Phase 2: Supervised Championship
        sup_result = await pipeline.run_supervised_tournament()

        # Phase 3: Anomaly Discovery (Clustering & Neural)
        anom_result = await pipeline.run_anomaly_discovery()

        # Phase 4: Final Aggregation (Passing the dictionary for the agent)
        await pipeline.run_aggregation({
            "prep": prep_result,
            "sup": sup_result,
            "anom": anom_result
        })

        print("\n🏆 TRAINING COMPLETE: Models saved and Registry updated.")

    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        # Consider logging the full traceback here for easier debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_training_system())