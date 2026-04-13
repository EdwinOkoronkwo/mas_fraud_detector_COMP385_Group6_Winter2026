import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from backup.strategies.anomaly.neuro_pattern.vae_agent import VAEAgent
from core.decision_agent import DecisionAggregator
from strategies.anomaly.clustering.dbscan_agent import DBSCANAgent
from strategies.data_preprocessing.preprocess_agent import PreprocessAgent
from strategies.data_preprocessing.sql_ingestor_agent import SQLIngestorAgent
from strategies.supervised.dynamic_xgb_agent import DynamicXGBAgent


# --- Imports (Updating to match your new naming convention) ---


import pytest  # Make sure this is at the top

from strategies.supervised.supervised_critic import SupervisedCritic


class TestSpecialistAgents:

    @pytest.mark.asyncio  # <--- THIS IS THE KEY
    async def test_sql_ingestor_run_method(self):
        """
        Verify the SQL Ingestor returns a success status.
        """
        mock_config = {"KAGGLE_PATH": "fake.csv", "DB_PATH": "test.db"}
        mock_client = MagicMock()

        ingestor = SQLIngestorAgent(config=mock_config, model_client=mock_client)

        # Now pytest knows how to 'await' this call
        result = await ingestor.run()

        assert result["status"] == "success"
        assert result["agent_name"] == "SQL_Ingestor"
    #
    # 2. PREPROCESSING AGENT UNIT TEST
    @pytest.mark.asyncio
    async def test_preprocess_agent_run_method(self):
        """
        Verify the Preprocess Agent returns a success status.
        """
        mock_config = {"DB_PATH": "data/test.db"}
        mock_client = MagicMock()

        # Initialize the Preprocess Agent
        agent = PreprocessAgent(config=mock_config, model_client=mock_client)

        # Execute the async run method
        result = await agent.run()

        assert result["status"] == "success"
        assert result["agent_name"] == "Preprocess_Agent"
    #
    # 3. SUPERVISED (XGB) UNIT TEST
    @pytest.mark.asyncio
    async def test_dynamic_xgb_agent_initialization(self):
        """
        Verify the Dynamic XGB Agent initialization for AutoGen v0.4+
        """
        mock_client = MagicMock()
        temp_path = "data/processed/temp_gold.csv"

        agent_instance = DynamicXGBAgent(model_client=mock_client, temp_data_path=temp_path)

        # 1. Identity Check
        assert agent_instance.agent.name == "Dynamic_XGB_Agent"

        # 2. Tool Check (v0.4 uses _tools list)
        agent_tools = getattr(agent_instance.agent, "_tools", [])
        tool_names = [t.name for t in agent_tools]
        assert "run_dynamic_xgb" in tool_names

        # 3. System Message Check (v0.4 List Logic)
        sys_msg = ""
        # In v0.4, it's often in _system_messages as a list of dicts
        raw_messages = getattr(agent_instance.agent, "_system_messages", [])

        if isinstance(raw_messages, list) and len(raw_messages) > 0:
            # We look at the first message content
            first_msg = raw_messages[0]
            if isinstance(first_msg, dict):
                sys_msg = first_msg.get("content", "")
            else:
                # In case it's an object with a .content attribute
                sys_msg = getattr(first_msg, "content", str(first_msg))

        assert "F1=0.77" in sys_msg

        # 4. Path Check
        assert agent_instance.temp_data_path == temp_path

    # 4. ANOMALY (VAE) UNIT TEST
    @pytest.mark.asyncio
    async def test_vae_agent_initialization(self):
        """
        Final Alignment: The class takes 2 args, but Python sends 'self'
        automatically, so we only provide ONE explicit argument.
        """
        mock_client = MagicMock()

        # 🚀 THE ACTUAL FIX:
        # Python sends the instance as arg 1.
        # We send mock_client as arg 2.
        # Total = 2.
        vae_instance = VAEAgent(mock_client)

        assert vae_instance.agent.name == "VAE_Agent"

        # Tool Check
        agent_tools = getattr(vae_instance.agent, "_tools", [])
        assert any(t.name == "train_vae_sim" for t in agent_tools)

    # 5. ANOMALY (DBSCAN) UNIT TEST
    @pytest.mark.asyncio
    async def test_dbscan_agent_initialization(self):
        mock_client = MagicMock()
        mock_settings = MagicMock()
        mock_settings.DB_PATH = "data/test.db"

        dbscan_instance = DBSCANAgent(mock_client, mock_settings)

        # 1. Identity & Tool Check
        assert dbscan_instance.agent.name == "DBSCAN_Agent"
        agent_tools = getattr(dbscan_instance.agent, "_tools", [])
        assert any(t.name == "train_dbscan" for t in agent_tools)

        # 2. System Message Check (The Pydantic Fix)
        raw_msgs = getattr(dbscan_instance.agent, "_system_messages", [])
        sys_msg = ""

        if raw_msgs:
            first_msg = raw_msgs[0]
            # If it's a Pydantic object (SystemMessage), use .content
            # If it's a dict, use .get()
            if hasattr(first_msg, "content"):
                sys_msg = first_msg.content
            elif isinstance(first_msg, dict):
                sys_msg = first_msg.get("content", "")

        assert "eps" in sys_msg
        assert "INCREASE your 'eps'" in sys_msg

    @pytest.mark.asyncio
    async def test_supervised_critic_initialization(self):
        """
        Verify the Critic's audit protocol and Tiered Evaluation logic.
        """
        mock_client = MagicMock()
        test_db = "data/fraud_gold.db"

        # Initialize
        critic_instance = SupervisedCritic(model_client=mock_client, db_path=test_db)

        # 1. Identity Check
        assert critic_instance.agent.name == "Supervised_Critic"

        # 2. Audit Protocol Extraction (v0.4 Pydantic Safe)
        raw_msgs = getattr(critic_instance.agent, "_system_messages", [])
        sys_msg = ""
        if raw_msgs:
            msg_obj = raw_msgs[0]
            sys_msg = msg_obj.content if hasattr(msg_obj, "content") else str(msg_obj)

        # 3. Verify Key Governance Logic
        assert "STRICT AUDIT PROTOCOL" in sys_msg
        assert "GOLD TIER: F1 >= 0.75" in sys_msg
        assert "Champion Registry" in sys_msg
        assert test_db in sys_msg

    @pytest.mark.asyncio
    async def test_decision_aggregator_initialization(self):
        """
        Verify the Decision Aggregator with real function signatures for AutoGen v0.4+
        """
        mock_client = MagicMock()
        mock_settings = MagicMock()

        # 1. Define a concrete dummy function for the save_tool
        # AutoGen needs to see the type hints to avoid the TypeError
        def mock_save_tool(model_path: str) -> str:
            """Saves the champion model."""
            return f"Model saved to {model_path}"

        # 2. Patch the 'get_feature_list' to avoid file I/O
        with patch.object(DecisionAggregator, 'get_feature_list', return_value=['V1', 'V2', 'Amount']):
            aggregator_instance = DecisionAggregator(
                model_client=mock_client,
                settings=mock_settings,
                save_tool=mock_save_tool  # Pass the real function, not a Mock
            )

        # 3. Identity Check
        assert aggregator_instance.agent.name == "Decision_Aggregator"

        # 4. Tool Check (v0.4 uses _tools list)
        agent_tools = getattr(aggregator_instance.agent, "_tools", [])
        tool_names = [t.name for t in agent_tools]

        # Note: AutoGen usually names the tool after the function name
        assert "mock_save_tool" in tool_names
        assert "write_markdown_report" in tool_names

        # 5. System Message Check
        raw_msgs = getattr(aggregator_instance.agent, "_system_messages", [])
        sys_msg = raw_msgs[0].content if hasattr(raw_msgs[0], 'content') else str(raw_msgs[0])

        assert "FEATURES USED: ['V1', 'V2', 'Amount']" in sys_msg