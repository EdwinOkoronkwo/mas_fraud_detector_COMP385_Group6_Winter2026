from abc import ABC, abstractmethod

from inference.models.VAE_model import VAE


class InferencePillar(ABC):
    @abstractmethod
    def predict(self, scaled_data: dict) -> float:
        pass


import torch


class NeuralPillar(InferencePillar):
    def __init__(self, model_path, input_dim=22, latent_dim=8):
        self.model = VAE(input_dim=input_dim, latent_dim=latent_dim)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, scaled_data: dict) -> float:
        vals = [scaled_data[k] for k in sorted(scaled_data.keys())]
        tensor = torch.tensor([vals], dtype=torch.float32)
        with torch.no_grad():
            recon, _, _ = self.model(tensor)
            mse = torch.mean((tensor - recon) ** 2).item()
        return self._calibrate(mse)

    def _calibrate(self, mse, threshold=0.08):
        if mse <= threshold: return (mse / threshold) * 0.4
        return min(0.99, 0.4 + (torch.log1p(torch.tensor(mse - threshold)) / 5.0).item())