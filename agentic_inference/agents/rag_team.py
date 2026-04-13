from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat


class RagAuditTeam:
    def __init__(self, model_client, sql_agent, vector_agent, synth_agent):
        self.model_client = model_client
        self.sql_researcher = sql_agent
        self.vector_researcher = vector_agent
        self.synthesis_engine = synth_agent

    def audit_selector(self, messages) -> str | None:
        if not messages: return "SQL_Researcher"

        last_speaker = messages[-1].source.lower()

        # Step 1: SQL_Researcher has laid out the grounded facts -> Get Policy
        if "sql_researcher" in last_speaker:
            return "Vector_Researcher"

        # Step 2: Vector has the policy -> Synthesize
        if "vector_researcher" in last_speaker:
            return "Synthesis_Engine"

        return None  # End conversation

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