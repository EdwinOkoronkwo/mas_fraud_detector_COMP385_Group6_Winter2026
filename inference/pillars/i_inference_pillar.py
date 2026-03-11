from abc import ABC, abstractmethod

from inference.models.VAE_model import VAE


class InferencePillar(ABC):
    @abstractmethod
    def predict(self, scaled_data: dict) -> float:
        pass
