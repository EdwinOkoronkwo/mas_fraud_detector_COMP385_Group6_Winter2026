from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from agentic_inference.agents.sql_researcher import SQLResearcher
from agentic_inference.agents.synthesis_engine import SynthesisEngine
from agentic_inference.agents.vector_researcher import VectorResearcher


class RagAuditTeam:
    def __init__(self, model_client, sql_agent, vector_agent, synth_agent):
        self.model_client = model_client
        self.sql_researcher = sql_agent
        self.vector_researcher = vector_agent
        self.synthesis_engine = synth_agent

    def audit_selector(self, messages) -> str | None:
        if not messages: return "SQL_Researcher"

        last_msg = messages[-1]
        last_speaker = last_msg.source.lower()

        # Step 1: SQL has found the record -> Pass to Vector for Policy search
        if "sql_researcher" in last_speaker:
            return "Vector_Researcher"

        # Step 2: Vector has found the Policy -> Pass to Synthesis for final narrative
        if "vector_researcher" in last_speaker:
            return "Synthesis_Engine"

        # Step 3: Synthesis Engine provides the "Understanding" -> Exit
        if "synthesis_engine" in last_speaker:
            return None

        return "SQL_Researcher"

    def get_team(self):
        return SelectorGroupChat(
            participants=[
                self.sql_researcher,
                self.vector_researcher,
                self.synthesis_engine
            ],
            model_client=self.model_client,
            selector_func=self.audit_selector,
            # We increase turns to 3 to accommodate the full chain
            termination_condition=MaxMessageTermination(3),
            max_turns=6
        )