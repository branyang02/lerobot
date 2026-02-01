#!/usr/bin/env python
"""
Standalone test for Wan2.2-TI2V-5B VAM module.

Tests the Wan22VAM class can:
1. Load the Wan2.2 model from HuggingFace
2. Process images through the VAE
3. Extract per-layer hidden states from the transformer
4. Project to the expected output dimensions

Run with: python test_wan2_vam.py
"""

import torch


def test_wan2_vam():
    print("=" * 60)
    print("Testing Wan2.2-TI2V-5B VAM")
    print("=" * 60)

    # Import the VAM
    from lerobot.policies.smolvla.wan_vam import Wan22VAM

    # Configuration
    num_expert_layers = 16
    vam_hidden_dim = 512
    vam_num_tokens = 64
    batch_size = 2
    num_cameras = 3
    img_size = 512

    print(f"\nConfiguration:")
    print(f"  - num_expert_layers: {num_expert_layers}")
    print(f"  - vam_hidden_dim: {vam_hidden_dim}")
    print(f"  - vam_num_tokens: {vam_num_tokens}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - num_cameras: {num_cameras}")
    print(f"  - img_size: {img_size}")

    # Initialize VAM
    print("\n" + "=" * 60)
    print("Initializing Wan22VAM...")
    print("=" * 60)

    vam = Wan22VAM(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        num_expert_layers=num_expert_layers,
        vam_hidden_dim=vam_hidden_dim,
        vam_num_tokens=vam_num_tokens,
        freeze_backbone=False,  # Fine-tuning mode
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    vam = vam.to(device)

    # Print model info
    print(f"\n{vam}")

    # Get parameter counts
    param_counts = vam.get_num_params()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        if count >= 1e9:
            print(f"  - {name}: {count / 1e9:.2f}B")
        elif count >= 1e6:
            print(f"  - {name}: {count / 1e6:.1f}M")
        else:
            print(f"  - {name}: {count}")

    # Create fake input images
    print("\n" + "=" * 60)
    print("Running forward pass...")
    print("=" * 60)

    # Images: [B, num_cameras, C, H, W]
    images = torch.randn(batch_size, num_cameras, 3, img_size, img_size, device=device)
    print(f"\nInput images shape: {images.shape}")

    # Forward pass
    with torch.no_grad():
        layer_hidden_states = vam(images)

    # Check outputs
    print(f"\nOutput: {len(layer_hidden_states)} layer hidden states")
    for i, hidden in enumerate(layer_hidden_states):
        print(f"  Layer {i}: {hidden.shape}")

    # Verify shapes
    expected_shape = (batch_size, vam_num_tokens, vam_hidden_dim)
    for i, hidden in enumerate(layer_hidden_states):
        assert hidden.shape == expected_shape, (
            f"Layer {i} shape mismatch: {hidden.shape} vs {expected_shape}"
        )

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU memory:")
        print(f"  - Allocated: {mem_allocated:.2f} GB")
        print(f"  - Reserved: {mem_reserved:.2f} GB")


if __name__ == "__main__":
    test_wan2_vam()
