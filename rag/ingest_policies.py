import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_policies(vector_service, file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: Policy file not found at {file_path}")
        return

    # 1. Load the text
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # 2. Split into chunks (so the agent can find specific policies)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 3. Add to your existing vector service
    # Note: Assuming your vector_service has an 'add_documents' or 'add_texts' method
    vector_service.vector_store.add_documents(docs)
    print(f"✅ Successfully indexed {len(docs)} policy chunks from {file_path}")

# USE CASE:
# ingest_policies(vector_service, r"C:\Youtube\autogen\mas_fraud_detector\data\policies\fraud_handbook.txt")