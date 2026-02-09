import re

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from rag.agents.inference_specialist import InferenceAgent
from rag.agents.lead_investigator import LeadInvestigator
from rag.agents.preprocessing_specialist import PreprocessingAgent
from rag.agents.rag_critic import RAGCritic
from rag.agents.sql_researcher import SQLResearcher
from rag.agents.synthesis_engine import SynthesisEngine
from rag.agents.vector_researcher import VectorResearcher
from rag.agents.web_researcher import WebResearcher

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from typing import Sequence
from autogen_agentchat.messages import BaseChatMessage, BaseAgentEvent
from autogen_agentchat.teams import SelectorGroupChat

from autogen_agentchat.agents import UserProxyAgent

import os
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage


# Assuming these are imported correctly from your other files
# from .agents import LeadInvestigator, SQLResearcher, PreprocessingAgent, InferenceAgent, VectorResearcher, SynthesisEngine, RAGCritic

class RAGTeam:
    def __init__(self, model_client, db_path, vector_service):
        self.model_client = model_client

        # 1. Initialize Specialists
        self.sql_researcher = SQLResearcher(model_client, db_path).agent
        self.data_preprocessor = PreprocessingAgent(model_client).agent
        self.inference_specialist = InferenceAgent(model_client).agent
        self.vector_researcher = VectorResearcher(model_client, vector_service).agent
        self.synthesis_engine = SynthesisEngine(model_client).agent
        self.rag_critic = RAGCritic(model_client).agent

        # 2. Initialize Lead & Human Proxy
        self.compliance_officer = UserProxyAgent(name="Compliance_Officer")
        self.lead_investigator = LeadInvestigator(model_client).agent

    def fraud_selector(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if not messages:
            return "Lead_Investigator"

        last_speaker = messages[-1].source.lower()
        last_msg = messages[-1].content.lower()

        # STEP 1: The Data Pipeline
        if "sql_researcher" in last_speaker:
            return "Data_Preprocessor"
        if "data_preprocessor" in last_speaker:
            return "Inference_Specialist"

        # STEP 2: Branching after Inference
        if "inference_specialist" in last_speaker:
            return "Vector_Researcher"

        # STEP 3: Validation Loop
        if "vector_researcher" in last_speaker:
            return "Synthesis_Engine"
        if "synthesis_engine" in last_speaker:
            return "RAG_Critic"

        # STEP 4: Direct Handoff (HITL REMOVED)
        if "rag_critic" in last_speaker:
            if "revise" in last_msg:
                return "Synthesis_Engine"
            # Skip Compliance_Officer, go straight to Lead
            return "Lead_Investigator"

        # STEP 5: Final Handoff
        if "lead_investigator" in last_speaker:
            if "case_closed" in last_msg:
                return None
            return "SQL_Researcher"

        return "Lead_Investigator"

    def get_team(self):
        # ALL agents must be in this list or the selector will fail
        participants = [
            self.lead_investigator,
            self.sql_researcher,
            self.data_preprocessor,
            self.inference_specialist,
            self.vector_researcher,
            self.synthesis_engine,
            self.rag_critic,
            self.compliance_officer
        ]

        return SelectorGroupChat(
            participants=participants,
            model_client=self.model_client,
            selector_func=self.fraud_selector,
            termination_condition=TextMentionTermination("CASE_CLOSED"),
            max_turns=30
        )