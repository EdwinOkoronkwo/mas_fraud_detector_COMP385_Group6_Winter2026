import asyncio
import os

from mas_fraud_detector.config.llm_config import get_model_client
from mas_fraud_detector.rag.rag_team import RAGTeam
from mas_fraud_detector.rag.vector_service import VectorService


# Set your API keys if not already in environment
# os.environ["TAVILY_API_KEY"] = "your_tavily_key"
# os.environ["OPENAI_API_KEY"] = "your_embedding_key"

async def run_fraud_investigation():
    # 1. Setup Client
    model_client = get_model_client()

    # 2. Setup Vector Store
    # Persistence ensures we don't re-ingest every time
    vector_service = VectorService(persist_directory="./chroma_db")
    policy_path = r"C:\Youtube\autogen\mas_fraud_detector\data\policies\fraud_handbook.txt"

    if not os.path.exists("./chroma_db"):
        print("Initial ingestion of policies...")
        vector_service.load_local_policies(policy_path)

    # 3. Initialize the Team Orchestrator
    db_path = r"C:\Youtube\autogen\mas_fraud_detector\data\database.sqlite"
    rag_team_manager = RAGTeam(model_client, db_path, vector_service)
    investigation_group = rag_team_manager.get_team()

    # 4. Define the Scenario
    # This mimics a message coming from your Anomaly Detection model
    test_task = (
        "New Fraud Alert: Transaction of $1,200 for customer Brandon Pittman. "
        "The purchase was made in Kyiv, Ukraine, which seems inconsistent with his home address. "
        "Investigate his transaction history for anomalies and cross-reference with our fraud policies."
    )

    print("\n" + "=" * 50)
    print("STARTING MULTI-AGENT RAG INVESTIGATION")
    print("=" * 50 + "\n")

    # 5. Run and Stream the Conversation
    async for message in investigation_group.run_stream(task=test_task):
        # Format the output for readability
        source = getattr(message, 'source', 'System')
        content = getattr(message, 'content', str(message))

        print(f"--- {source.upper()} ---")
        print(f"{content}\n")


if __name__ == "__main__":
    asyncio.run(run_fraud_investigation())