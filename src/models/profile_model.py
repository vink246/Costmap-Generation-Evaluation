# src/tools/profile_model.py
import argparse, time, torch
from importlib import import_module

def count_flops(model, x):
    """
    Very coarse FLOPs proxy using torch.profiler; for exact FLOPs use fvcore/thop if allowed.
    Here we report total matmul MACs + conv MACs (approximate).
    """
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except Exception:
        return None, sum(p.numel() for p in model.parameters())

def measure_latency(model, x, warmup=20, runs=100, device="cuda"):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") else None

        # measure
        t0 = time.time()
        for _ in range(runs):
            _ = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t1 = time.time()
    return (t1 - t0) * 1000.0 / runs  # ms per frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_module", required=True)
    ap.add_argument("--model_class", required=True)
    ap.add_argument("--in_channels", type=int, default=4)
    ap.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    mod = import_module(args.model_module)
    cls = getattr(mod, args.model_class)
    model = cls(in_channels=args.in_channels).to(args.device)

    B, H, W = 1, args.image_size[0], args.image_size[1]
    x = torch.randn(B, args.in_channels, H, W, device=args.device)

    flops, params = count_flops(model, x)
    lat_ms = measure_latency(model, x, device=args.device)

    print(f"Params: {params/1e6:.2f} M")
    if flops is not None:
        print(f"FLOPs (approx): {flops/1e9:.2f} GFLOPs")
    else:
        print("FLOPs: (thop not installed, skipped)")
    print(f"Latency: {lat_ms:.2f} ms / frame on {args.device}")
