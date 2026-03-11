import os
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorService:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory

        # Ensure directory exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="fraud_policies"
        )

    def load_local_policies(self, file_path):
        """Loads and indexes the policy handbook if it exists."""
        if not os.path.exists(file_path):
            print(f"⚠️ Policy file {file_path} not found.")
            return

        print(f"📖 Loading policies from: {file_path}")
        # Ensure we use UTF-8 to handle those policy emojis/symbols
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # UPDATED: Better splitter to avoid missing rules
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["POLICY_CODE:", "\n\n", "\n", " "]
        )
        docs = text_splitter.split_documents(documents)

        self.vector_store.add_documents(docs)
        print(f"✅ Successfully ingested {len(docs)} policy chunks into ChromaDB.")

    def get_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)
