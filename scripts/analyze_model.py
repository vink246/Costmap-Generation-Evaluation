import os
import sys
import time
import torch
import time
import argparse
from thop import profile, clever_format
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.vit import ViT  

def analyze_model(model, input_shape=(1, 4, 256, 256), device='cuda'):
    model = model.to(device)
    model.eval()

    # Dummy input
    x = torch.randn(input_shape).to(device)

    # --- Parameter count ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # --- FLOPs (using thop) ---
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([macs, params], "%.3f")
    print(f"FLOPs (MACs): {flops}")

    # --- Latency & FPS ---
    n_warmup = 10
    n_runs = 100

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    torch.cuda.synchronize()
    end = time.time()

    avg_latency = (end - start) / n_runs * 1000  # ms
    fps = 1000.0 / avg_latency

    print(f"Average Latency: {avg_latency:.3f} ms | FPS: {fps:.2f}")

    return {
        'total_params': total_params,
        'flops': flops,
        'avg_latency_ms': avg_latency,
        'fps': fps
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=256)
    args = parser.parse_args()

    # Build model
    model = ViT(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size
    )

    results = analyze_model(
        model,
        input_shape=(1, args.in_channels, args.input_size, args.input_size),
        device=args.device
    )
