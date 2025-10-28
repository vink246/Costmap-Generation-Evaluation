"""Quick test to verify UNet implementation works with project parameters."""

import sys
from pathlib import Path

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.unet import UNet


def test_unet():
    print("Testing UNet implementation...")
    print("-" * 60)
    
    # Test with project parameters from configs/train_nyu_unet.yaml
    model = UNet(in_channels=4, out_channels=1, base_channels=32, bilinear=True)
    
    # Create dummy input (batch_size=2, channels=4, height=256, width=256)
    x = torch.randn(2, 4, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    # Print results
    print(f"✓ Model initialized successfully")
    print(f"  - in_channels: {model.in_channels}")
    print(f"  - out_channels: {model.out_channels}")
    print(f"  - base_channels: {model.base_channels}")
    print(f"  - bilinear: {model.bilinear}")
    print()
    print(f"Input shape:    {tuple(x.shape)}")
    print(f"Output shape:   {tuple(y.shape)}")
    print(f"Expected shape: (2, 1, 64, 64)")
    print()
    
    # Verify output shape
    if y.shape == (2, 1, 64, 64):
        print("✓ Output shape is correct!")
    else:
        print("✗ Output shape is WRONG!")
        return False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    # Test with different base_channels
    print("\n" + "-" * 60)
    print("Testing with different base_channels...")
    for bc in [16, 32, 64]:
        model_test = UNet(in_channels=4, out_channels=1, base_channels=bc)
        x_test = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            y_test = model_test(x_test)
        params = sum(p.numel() for p in model_test.parameters())
        status = "✓" if y_test.shape == (1, 1, 64, 64) else "✗"
        print(f"  {status} base_channels={bc:2d}: output={tuple(y_test.shape)}, params={params:,}")
    
    print("-" * 60)
    print("✓ All tests passed!")
    return True


if __name__ == '__main__':
    success = test_unet()
    sys.exit(0 if success else 1)

