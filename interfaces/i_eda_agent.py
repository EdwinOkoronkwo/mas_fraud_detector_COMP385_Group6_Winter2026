from abc import ABC, abstractmethod
from typing import Dict, Any

from interfaces.i_data_agent import IDataAgent


class IEDAAgent(IDataAgent):
    """Contract for Statistical Analysis & Visuals."""
    @abstractmethod
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expects: 'db_path' in context."""
        pass