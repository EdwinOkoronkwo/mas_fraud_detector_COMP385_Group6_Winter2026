from agentic_inference.agents.rag_team import RagAuditTeam
from agentic_inference.services.vector_service import VectorService
from autogen_agentchat.teams import SelectorGroupChat
from dotenv import load_dotenv
import pytest
import os
import shutil
import uuid
from unittest.mock import MagicMock, patch

# Load environment variables if needed for local testing
load_dotenv()


class TestRagIntegration:
    """
    MODULE: RAG Integration Tests
    PURPOSE: Verifies the coordination between SQL, Vector, and Synthesis agents
    within the RagAuditTeam, ensuring the sequential selector logic remains intact.
    """

    @pytest.fixture
    def mock_agents(self):
        """Provides mocked agents for the team."""
        return {
            "model_client": MagicMock(),
            "sql": MagicMock(name="SQL_Researcher"),
            "vector": MagicMock(name="Vector_Researcher"),
            "synth": MagicMock(name="Synthesis_Engine")
        }

    def test_audit_selector_full_chain(self, mock_agents):
        """
        Verifies the sequential handoff logic of the RagAuditTeam selector.
        Chain: Start -> SQL -> Vector -> Synthesis -> Exit.
        """
        team = RagAuditTeam(
            model_client=mock_agents["model_client"],
            sql_agent=mock_agents["sql"],
            vector_agent=mock_agents["vector"],
            synth_agent=mock_agents["synth"]
        )

        # 1. Initial State (No messages)
        assert team.audit_selector([]) == "SQL_Researcher"

        # 2. After SQL Researcher speaks
        msg_sql = MagicMock()
        msg_sql.source = "SQL_Researcher"
        assert team.audit_selector([msg_sql]) == "Vector_Researcher"

        # 3. After Vector Researcher speaks
        msg_vector = MagicMock()
        msg_vector.source = "Vector_Researcher"
        assert team.audit_selector([msg_sql, msg_vector]) == "Synthesis_Engine"

        # 4. After Synthesis Engine speaks (Termination)
        msg_synth = MagicMock()
        msg_synth.source = "Synthesis_Engine"
        assert team.audit_selector([msg_sql, msg_vector, msg_synth]) is None

    def test_audit_selector_case_insensitivity(self, mock_agents):
        """Ensures the selector handles inconsistent casing in agent source names."""
        team = RagAuditTeam(
            model_client=mock_agents["model_client"],
            sql_agent=mock_agents["sql"],
            vector_agent=mock_agents["vector"],
            synth_agent=mock_agents["synth"]
        )

        msg = MagicMock()
        msg.source = "sql_researcher"  # Lowercase version
        assert team.audit_selector([msg]) == "Vector_Researcher"

    def test_team_initialization(self, mock_agents):
        """Verifies the SelectorGroupChat is configured with correct participants and turns."""
        team_wrapper = RagAuditTeam(
            model_client=mock_agents["model_client"],
            sql_agent=mock_agents["sql"],
            vector_agent=mock_agents["vector"],
            synth_agent=mock_agents["synth"]
        )

        group_chat = team_wrapper.get_team()

        # FIXED: Ensure SelectorGroupChat is recognized as a valid class type for isinstance
        assert group_chat is not None
        assert type(group_chat).__name__ == "SelectorGroupChat" or isinstance(group_chat, SelectorGroupChat)

        # Ensure all three participants are present
        assert len(group_chat._participants) == 3
        print("✅ RagAuditTeam: Team structure and selector logic verified.")


class TestVectorService:
    """Tests for the VectorService ChromaDB implementation."""

    @pytest.fixture
    def test_db_dir(self):
        """
        Creates a temporary directory for ChromaDB tests.
        Uses a unique ID per test to avoid Windows file lock (PermissionError)
        when multiple tests run sequentially.
        """
        unique_id = str(uuid.uuid4())[:8]
        path = f"./test_chroma_db_{unique_id}"

        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

        yield path

        # Cleanup after the test
        if os.path.exists(path):
            # On Windows, sqlite might still have the file locked.
            shutil.rmtree(path, ignore_errors=True)

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_vector_service_init(self, mock_embeddings, test_db_dir):
        """Verifies that VectorService initializes Chroma with a persistent client."""
        service = VectorService(persist_directory=test_db_dir)

        assert os.path.exists(test_db_dir)
        assert service.vector_store is not None
        assert service._client is not None
        print("✅ VectorService: Initialization and Directory creation verified.")

    @patch('langchain_openai.OpenAIEmbeddings')
    @patch('langchain_community.document_loaders.TextLoader.load')
    def test_policy_ingestion_logic(self, mock_load, mock_embeddings, test_db_dir):
        """Verifies document splitting and ingestion into the vector store."""
        from langchain_core.documents import Document

        # Mocking external calls
        mock_load.return_value = [
            Document(page_content="POLICY_CODE: RULE-1\nDetails here.\n\nPOLICY_CODE: RULE-2\nMore details.")]

        service = VectorService(persist_directory=test_db_dir)

        # Create a dummy file to pass the existence check
        dummy_file = f"dummy_policy_{uuid.uuid4().hex[:6]}.txt"
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("test content")

        try:
            with patch.object(service.vector_store, 'add_documents') as mock_add:
                service.load_local_policies(dummy_file)
                # Ensure it attempted to add split documents
                assert mock_add.called
                print("✅ VectorService: Policy ingestion and splitting verified.")
        finally:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)

    @patch('langchain_openai.OpenAIEmbeddings')
    def test_retriever_generation(self, mock_embeddings, test_db_dir):
        """Ensures the retriever is correctly generated with passed kwargs."""
        service = VectorService(persist_directory=test_db_dir)
        retriever = service.get_retriever(search_kwargs={"k": 5})

        assert hasattr(retriever, 'invoke') or hasattr(retriever, 'ainvoke')
        assert retriever.search_kwargs == {"k": 5}
        print("✅ VectorService: Retriever generation with kwargs verified.")

# --- DEPRECATION FIXES FOR OTHER MODULES ---
# 1. In data/database.py:
#    Change: from sqlalchemy.ext.declarative import declarative_base
#    To:     from sqlalchemy.orm import declarative_base
# 2. In agentic_inference/services/vector_service.py:
#    The LangChain warning suggests migrating to 'langchain_chroma'.
#    Install: pip install -U langchain-chroma
#    Change: from langchain_community.vectorstores import Chroma -> from langchain_chroma import Chroma