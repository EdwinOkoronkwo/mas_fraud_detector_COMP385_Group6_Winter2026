from abc import ABC, abstractmethod
from typing import Dict, Any

from interfaces.i_data_agent import IDataAgent


class IPreprocessorAgent(IDataAgent):
    """Contract for Feature Scaling & Persistence."""
    @abstractmethod
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expects: 'db_path', 'target_col' in context."""
        pass