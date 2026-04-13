import torch
import os


def check_pytorch_dimensions(model_path):
    """
    Checks the internal layer shapes of a PyTorch .pth state_dict.
    Aligned with the 24-feature MAS architecture.
    """
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"\n🔍 Inspecting MAS PyTorch Model: {model_path}")
    try:
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # 1. Check Input Layer (Encoder)
        # Expected: 24 features -> 128 hidden
        encoder_key = 'encoder.0.weight'
        if encoder_key in state_dict:
            shape = state_dict[encoder_key].shape
            # shape[1] is the input dimension
            print(f"✅ Input Features (Encoder In): {shape[1]}")
            print(f"   Structure: {list(shape)} (Neurons x Features)")
            if shape[1] != 24:
                print(f"   ⚠️ WARNING: Expected 24 features, found {shape[1]}")
        else:
            print("⚠️ Could not find 'encoder.0.weight'. Available keys:")
            print(list(state_dict.keys())[:5], "... (truncated)")

        # 2. Check Latent Space (Bottleneck)
        # Expected: 12 latent dims (per Beefed VAE spec)
        if 'fc_mu.weight' in state_dict:
            latent_dim = state_dict['fc_mu.weight'].shape[0]
            print(f"✅ Latent Dimension (mu): {latent_dim}")

        # 3. Check Output Layer (Decoder)
        # For a 24-feature VAE, the last layer mirrors the input
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k]
        if decoder_keys:
            last_layer = decoder_keys[-1]
            output_dim = state_dict[last_layer].shape[0]
            print(f"✅ Output Features (Decoder Out): {output_dim}")

        print("\n📝 All keys successfully parsed from state_dict.")

    except Exception as e:
        print(f"❌ Error loading .pth file: {e}")


def check_model_weights(model_path):
    """Provides a brief statistical summary of the weights."""
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"\n📊 Weight Statistics for {model_path}:")
        for key in ['encoder.0.weight', 'fc_mu.weight', 'decoder.8.weight']:
            if key in state_dict:
                w = state_dict[key]
                print(f"   {key:20} | Mean: {w.mean():.4f} | Std: {w.std():.4f}")
    except Exception:
        pass


def validate_against_test_specs(model_path, expected_in=24, expected_latent=12):
    """Explicitly validates the model against the project's unit test specifications."""
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        actual_in = state_dict['encoder.0.weight'].shape[1]
        actual_latent = state_dict['fc_mu.weight'].shape[0]

        print(f"\n⚖️  Validation Against Project Specs:")
        if actual_in == expected_in and actual_latent == expected_latent:
            print(f"   ✅ PASS: Model matches 'Beefed VAE' specs ({expected_in} in, {expected_latent} latent).")
        else:
            print(f"   ❌ FAIL: Dimension mismatch.")
            print(f"      Expected: {expected_in} in / {expected_latent} latent")
            print(f"      Actual:   {actual_in} in / {actual_latent} latent")
    except Exception as e:
        print(f"   ⚠️ Could not perform validation: {e}")


if __name__ == "__main__":
    # Update this path to your actual file
    PATH = "models/champion_vae.pth"
    check_pytorch_dimensions(PATH)
    check_model_weights(PATH)
    validate_against_test_specs(PATH)