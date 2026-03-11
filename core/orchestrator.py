# mas_fraud_detector/core/orchestrator.py
from typing import Any, List

from autogen_agentchat.base import TaskResult
from autogen_core.models import UserMessage

from typing import Any, Callable  # Add this line

from autogen_core.models import UserMessage

from core.state import PipelineState
from utils.logger import setup_logger


class Orchestrator:
    def __init__(self, state: PipelineState):
        self.state = state
        self.logger = setup_logger("Orchestrator")

    async def execute_phase(self, phase_name: str, runner_factory: Callable, task: str):
        self.logger.info(f"--- [PHASE START]: {phase_name} ---")
        try:
            runner = runner_factory()
            result = None

            async for message in runner.run_stream(task=task):
                if isinstance(message, TaskResult):
                    result = message
                else:
                    source = getattr(message, 'source', 'System').upper()
                    content = getattr(message, 'content', '')

                    # 1. Log Standard Chat
                    if content:
                        print(f"\n[{source}]:\n{content}\n" + "-" * 30)

                    # 2. Log Tool Activity (This is where the scaling happens)
                    # Mistral/OpenAI tool calls live in 'call_info'
                    if hasattr(message, "call_info") and message.call_info:
                        for call in message.call_info:
                            print(f"🛠️  [TOOL CALL]: {call.function_name}")
                            print(f"📦 [ARGS]: {call.arguments}")

                    # 3. Log Tool Output (The 22-feature result)
                    if source == "TOOL":
                        print(f"✨ [RESULT]: {content[:500]}...")

            if result:
                # Store the final result in the state for the next agent
                final_output = result.messages[-1].content
                self.state.update_state(phase_name, final_output)
                return result

        except Exception as e:
            self.logger.error(f"CRITICAL FAILURE in {phase_name}: {str(e)}")
            raise e