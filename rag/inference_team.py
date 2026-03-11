import re


from inference.tools.inference_engine import InferenceEngine
from rag.agents.preprocessing_specialist import PreprocessingAgent

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

# Assuming these are your imports
from rag.agents.lead_investigator import LeadInvestigator

from rag.agents.inference_specialist import InferenceAgent
from rag.agents.inference_critic import InferenceCritic

class InferenceTeam:
    def __init__(self, model_client, infra_manager):
        self.model_client = model_client
        self.infra_manager = infra_manager

        # 1. Initialize the Engine with Dynamic Gold Tier Assets
        # This uses the 24-feature list from champion_registry.json
        self.engine = InferenceEngine(infra_manager)

        # 2. Instantiate Agents
        self.lead = LeadInvestigator(model_client).agent
        # Pass infra_manager to prepper if the tool needs it for dynamic scaling
        self.prepper = PreprocessingAgent(model_client, infra_manager).agent
        # Specialist gets the engine containing gold_xgb.pkl
        self.specialist = InferenceAgent(model_client, self.engine).agent
        self.critic = InferenceCritic(model_client).agent

    def inference_selector(self, messages) -> str | None:
        if not messages: return "Lead_Investigator"

        last_msg = messages[-1]
        last_speaker = last_msg.source.lower()
        content = last_msg.content.upper()

        # 1. Lead initiates the Case
        if "lead" in last_speaker:
            if "CASE_CLOSED" in content: return None
            return "Data_Preprocessor"

        # 2. Preprocessor prepares the 24-feature vector
        if "preprocessor" in last_speaker:
            return "Inference_Specialist"

        # 3. Specialist runs the Gold XGB Inference
        if "specialist" in last_speaker:
            return "Inference_Critic"

        # 4. Critic reviews the probability and risk alignment
        if "critic" in last_speaker:
            return "Lead_Investigator"

        return "Lead_Investigator"

    def get_team(self):
        return SelectorGroupChat(
            participants=[self.lead, self.prepper, self.specialist, self.critic],
            model_client=self.model_client,
            selector_func=self.inference_selector,
            termination_condition=TextMentionTermination("CASE_CLOSED"),
            max_turns=15
        )


# Assuming these are imported correctly from your other files
# from .agents import LeadInvestigator, SQLResearcher, PreprocessingAgent, InferenceAgent, VectorResearcher, SynthesisEngine, RAGCritic

# class RAGTeam:
#     def __init__(self, model_client, db_path, vector_service):
#         self.model_client = model_client
#
#         # 1. Initialize Specialists
#         self.sql_researcher = SQLResearcher(model_client, db_path).agent
#         self.data_preprocessor = PreprocessingAgent(model_client).agent
#         self.inference_specialist = InferenceAgent(model_client).agent
#         self.vector_researcher = VectorResearcher(model_client, vector_service).agent
#         self.synthesis_engine = SynthesisEngine(model_client).agent
#         self.rag_critic = RAGCritic(model_client).agent
#
#         # 2. Initialize Lead & Human Proxy
#         self.compliance_officer = UserProxyAgent(name="Compliance_Officer")
#         self.lead_investigator = LeadInvestigator(model_client).agent
#
#     def fraud_selector(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
#         if not messages:
#             return "Lead_Investigator"
#
#         last_speaker = messages[-1].source.lower()
#         last_msg = messages[-1].content.lower()
#
#         # STEP 1: The Data Pipeline
#         if "sql_researcher" in last_speaker:
#             return "Data_Preprocessor"
#         if "data_preprocessor" in last_speaker:
#             return "Inference_Specialist"
#
#         # STEP 2: Branching after Inference
#         if "inference_specialist" in last_speaker:
#             return "Vector_Researcher"
#
#         # STEP 3: Validation Loop
#         if "vector_researcher" in last_speaker:
#             return "Synthesis_Engine"
#         if "synthesis_engine" in last_speaker:
#             return "RAG_Critic"
#
#         # STEP 4: Direct Handoff (HITL REMOVED)
#         if "rag_critic" in last_speaker:
#             if "revise" in last_msg:
#                 return "Synthesis_Engine"
#             # Skip Compliance_Officer, go straight to Lead
#             return "Lead_Investigator"
#
#         # STEP 5: Final Handoff
#         if "lead_investigator" in last_speaker:
#             if "case_closed" in last_msg:
#                 return None
#             return "SQL_Researcher"
#
#         return "Lead_Investigator"
#
#     def get_team(self):
#         # ALL agents must be in this list or the selector will fail
#         participants = [
#             self.lead_investigator,
#             self.sql_researcher,
#             self.data_preprocessor,
#             self.inference_specialist,
#             self.vector_researcher,
#             self.synthesis_engine,
#             self.rag_critic,
#             self.compliance_officer
#         ]
#
#         return SelectorGroupChat(
#             participants=participants,
#             model_client=self.model_client,
#             selector_func=self.fraud_selector,
#             termination_condition=TextMentionTermination("CASE_CLOSED"),
#             max_turns=30
#         )