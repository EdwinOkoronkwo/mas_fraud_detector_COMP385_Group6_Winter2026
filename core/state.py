from pydantic import BaseModel, Field
from typing import Dict, Any, List

class PipelineState(BaseModel):
    """
    Centralized Knowledge Base for the Multi-Agent System.
    Stores execution progress and data artifacts for cross-agent access.
    """
    # Track which specialized agents have committed results
    completed_tasks: List[str] = []
    
    # Store artifacts (e.g., table schemas, row counts, file paths)
    artifacts: Dict[str, Any] = {}

    def commit_result(self, task_name: str, data: Dict[str, Any]):
        """
        Finalizes an agent's task by logging artifacts and updating the global index.
        """
        if task_name not in self.completed_tasks:
            self.completed_tasks.append(task_name)
        self.artifacts[task_name] = data
        print(f"[SYSTEM STATE] Task '{task_name}' committed to shared memory.")

    # mas_fraud_detector/core/state.py
    def update_state(self, phase_name: str, content: Any):
        """
        Captures the final_output from the Orchestrator and commits it.
        """
        # We wrap the content in a dict to match your artifacts structure
        self.commit_result(task_name=phase_name, data={"final_report": content})