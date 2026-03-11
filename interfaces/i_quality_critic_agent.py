from abc import ABC, abstractmethod
from typing import Dict, Any

from interfaces.i_data_agent import IDataAgent


class IQualityCritic(IDataAgent):
    """Contract for Final Schema & Logic Verification."""
    @abstractmethod
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expects: 'db_path' to verify the 'cleaned_scaled_data' table."""
        pass