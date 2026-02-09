# mas_fraud_detector/core/orchestrator.py
from typing import Any, List

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
            # This creates the team from scratch for every phase
            runner = runner_factory()

            result = await runner.run(task=task)

            # Save state and return
            final_output = result.messages[-1].content
            self.state.update_state(phase_name, final_output)
            return result
        except Exception as e:
            self.logger.error(f"CRITICAL FAILURE in {phase_name}: {str(e)}")
            raise e
