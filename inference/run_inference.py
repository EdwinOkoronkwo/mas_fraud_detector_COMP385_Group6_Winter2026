# run_inference.py
import asyncio

from config.llm_config import get_model_client
from config.settings import DB_PATH
from inference.FraudInferencePipeline import FraudInferencePipeline
from rag.rag_team import RAGTeam
from rag.vector_service import VectorService




async def main():
    # 1. BOOTSTRAP: Initialize core services
    model_client = get_model_client()
    vector_service_inst = VectorService(persist_directory="./chroma_db")

    # 2. ASSEMBLE: Initialize the MAS team
    investigator_team = RAGTeam(model_client, DB_PATH, vector_service_inst)

    # 3. INJECT: Pass the assembled team into the Pipeline
    pipeline = FraudInferencePipeline(investigator_team, DB_PATH)

    # 4. EXECUTE
    await pipeline.run_batch_inference(n_samples=10)


if __name__ == "__main__":
    asyncio.run(main())