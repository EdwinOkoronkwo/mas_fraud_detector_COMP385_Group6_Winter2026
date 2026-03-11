from abc import ABC, abstractmethod
from typing import Dict, Any

from interfaces.i_data_agent import IDataAgent


class IIngestorAgent(IDataAgent):
    """Contract for Ingestion: CSV -> SQLite."""
    @abstractmethod
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Expects: 'kaggle_path', 'db_path' in context/config."""
        pass