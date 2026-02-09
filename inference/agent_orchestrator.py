import asyncio
from autogen_agentchat.ui import Console


class AgentOrchestrator:
    """Manages the execution of Multi-Agent Systems (MAS) with error resilience.

    This class wraps the agent communication loop, providing retry logic
    to handle transient infrastructure failures like SQL timeouts or API drops.
    """

    def __init__(self, investigator_team):
        """Initializes the orchestrator with a pre-configured RAGTeam.

        Args:
            investigator_team: An instance of RAGTeam containing the agents.
        """
        self.team = investigator_team
        self.max_retries = 2

    async def execute_with_resilience(self, task):
        for attempt in range(self.max_retries + 1):
            team_instance = self.team.get_team()
            result = await Console(team_instance.run_stream(task=task))

            last_msg = result.messages[-1].content if result.messages else ""

            if "SQL_QUERY_FAILED" not in last_msg and "TIMEOUT" not in last_msg:
                return self._extract_final_text(result)

            print(f"⚠️ Retry {attempt + 1}/{self.max_retries} due to system failure...")
            await asyncio.sleep(1)
        return ""

    def _extract_final_text(self, result):
        """Safely extracts text from AutoGen message objects, handling tool results."""
        for msg in reversed(result.messages):
            content = msg.content

            # 1. Handle List Content (Multimodal or Tool Results)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    # If it's a Tool Result (FunctionExecutionResult), get its .content
                    if hasattr(item, 'content'):
                        text_parts.append(str(item.content))
                    # If it's a dict (Multimodal block)
                    elif isinstance(item, dict):
                        text_parts.append(item.get("text", ""))
                    # Fallback for unexpected types
                    else:
                        text_parts.append(str(item))
                content = " ".join(text_parts)

            # 2. Check for our scoring keywords in the flattened text
            if any(key in str(content) for key in ["FINAL_SCORES", "TOTAL RISK SCORE", "ENSEMBLE_SCORE"]):
                return str(content)

        # 3. Final fallback to the very last message
        last_msg = result.messages[-1].content if result.messages else ""
        if isinstance(last_msg, list):
            return " ".join([str(getattr(i, 'content', i)) for i in last_msg])
        return str(last_msg)