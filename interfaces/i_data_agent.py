from abc import ABC, abstractmethod
from typing import Dict, Any

from utils.logger import setup_logger

class IDataAgent(ABC):
    """
    Interface for all Data Processing Agents.
    Ensures a consistent contract for Ingestion, EDA, and Cleaning.
    """
    
    def __init__(self, name: str, role: str, config: Dict[str, Any]):
        self.name = name
        self.role = role
        self.config = config
        self.logger = setup_logger(self.name)

    @abstractmethod
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent's primary task.
        :param context: Shared state or specific task instructions.
        :return: A dictionary containing results, logs, or status codes.
        """
        pass

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str):
        self.logger.error(message)