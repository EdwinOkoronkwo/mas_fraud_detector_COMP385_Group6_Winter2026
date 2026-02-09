import asyncio

from mas_fraud_detector.config.llm_config import get_model_client
from mas_fraud_detector.rag.rag_team import RAGTeam
from mas_fraud_detector.rag.vector_service import VectorService


async def main():
    # 1. Initialize our LLM Client (Mistral via OpenAI wrapper)
    model_client = get_model_client()

    # 2. Setup Vector Store & Ingest Policies
    # This only needs to be run once, or when policies change
    vector_service = VectorService(persist_directory="./chroma_db")
    # Point it to your data folder
    policy_path = r"C:\Youtube\autogen\mas_fraud_detector\data\policies\fraud_handbook.txt"
    vector_service.load_local_policies(policy_path)

    # 3. Initialize the RAG Team
    # We pass the db_path and the initialized vector_service
    rag_orchestrator = RAGTeam(
        model_client=model_client,
        db_path="database.sqlite",
        vector_service=vector_service
    )

    investigation_team = rag_orchestrator.get_team()

    # 4. Define the Fraud Alert to investigate
    # In a real app, this would come from your Anomaly Detection model
    high_risk_alert = (
        "CRITICAL ALERT (Risk Score: 0.85): User 1024 (John Doe) attempted a "
        "$4,500 withdrawal at 2:15 AM from a merchant in 'Lagos, Nigeria'. "
        "User's home location is 'New York, USA'."
    )

    print("\n--- Starting RAG Investigation ---\n")

    # 5. Run the Team Stream
    # This will show the conversation between researchers and the synthesis engine
    async for message in investigation_team.run_stream(task=high_risk_alert):
        if hasattr(message, 'source') and hasattr(message, 'content'):
            print(f"[{message.source.upper()}]: {message.content}\n")


if __name__ == "__main__":
    asyncio.run(main())