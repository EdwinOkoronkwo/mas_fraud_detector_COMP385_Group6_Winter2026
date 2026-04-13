import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from deterministic_inference.core.pillars.neuro import VAE, NeuroPillar


class TestNeuroPillar:

    @pytest.fixture
    def mock_state_dict(self):
        """Creates a dummy state dict that matches the 'Beefed' VAE architecture."""
        model = VAE(input_dim=24, latent_dim=12)
        return model.state_dict()

    @patch("torch.load")
    def test_neuro_pillar_initialization(self, mock_torch_load, mock_state_dict):
        """
        Verifies that the NeuroPillar loads the 'Beefed' VAE weights
        and sets the model to evaluation mode.
        """
        mock_torch_load.return_value = mock_state_dict

        # Initialize pillar
        pillar = NeuroPillar("dummy_vae_weights.pt")

        assert pillar.model_input_dim == 24
        assert not pillar.model.training  # Check eval() mode
        assert pillar.model.encoder[0].out_features == 128
        print("✅ NeuroPillar initialized with Beefed VAE architecture (128-64-32-12).")

    @patch("torch.load")
    def test_predict_reconstruction_loss(self, mock_torch_load, mock_state_dict):
        """
        Tests the reconstruction loss calculation.
        Ensures a float (MSE) is returned.
        """
        mock_torch_load.return_value = mock_state_dict
        pillar = NeuroPillar("dummy.pt")

        # 1. Create a sample 24-feature input (as numpy)
        sample_input = np.random.rand(1, 24).astype(np.float32)

        # 2. Run prediction
        loss_score = pillar.predict(sample_input)

        # 3. Assertions
        assert isinstance(loss_score, float)
        assert loss_score >= 0
        print(f"✅ Reconstruction Loss Calculated: {loss_score:.6f}")

    @patch("torch.load")
    def test_input_dimension_guardrail(self, mock_torch_load, mock_state_dict):
        """
        Verifies the 'Bridge' logic:
        1. Adds batch dimension to 1D inputs.
        2. Slices inputs larger than 24 features.
        """
        mock_torch_load.return_value = mock_state_dict
        pillar = NeuroPillar("dummy.pt")

        # Case A: 1D input (24 elements)
        input_1d = np.random.rand(24)
        loss_1d = pillar.predict(input_1d)
        assert isinstance(loss_1d, float)

        # Case B: Over-sized input (30 features)
        input_wide = np.random.rand(1, 30)
        loss_wide = pillar.predict(input_wide)
        assert isinstance(loss_wide, float)
        print("✅ Guardrails successfully handled 1D and Over-sized inputs.")

    @patch("torch.load")
    def test_vae_math_consistency(self, mock_torch_load, mock_state_dict):
        """
        Ensures that the VAE returns (reconstructed, mu, logvar)
        and that mu/logvar have the correct latent dimension (12).
        """
        mock_torch_load.return_value = mock_state_dict
        pillar = NeuroPillar("dummy.pt")

        test_tensor = torch.randn(1, 24)
        reconstructed, mu, logvar = pillar.model(test_tensor)

        assert reconstructed.shape == (1, 24)
        assert mu.shape == (1, 12)
        assert logvar.shape == (1, 12)
        print("✅ VAE Math: Latent space dimensions are correctly aligned to 12.")