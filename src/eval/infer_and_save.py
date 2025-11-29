import argparse
import importlib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset_npz import CostmapPairsNPZ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default="data/processed")
    ap.add_argument("--dataset", choices=["nyu", "kitti"], required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--model_module", required=True)
    ap.add_argument("--model_class", required=True)
    ap.add_argument("--model_kwargs", type=str, default="{}")

    args = ap.parse_args()

    # Convert string kwargs → Python dict
    import json
    model_kwargs = json.loads(args.model_kwargs)

    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset
    ds = CostmapPairsNPZ(args.data_root, split=args.split, dataset=args.dataset)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Load model
    ModelClass = getattr(importlib.import_module(args.model_module), args.model_class)
    model = ModelClass(**model_kwargs)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Run inference + save .npy per sample
    with torch.no_grad():
        for idx, (img, _) in enumerate(dl):
            pred = torch.sigmoid(model(img))
            pred_np = pred.squeeze().cpu().numpy()
            out_path = os.path.join(args.out_dir, f"{idx:06d}.npy")
            np.save(out_path, pred_np)

            if idx % 50 == 0:
                print(f"Saved: {out_path}")

    print("✓ Inference complete!")

if __name__ == "__main__":
    main()
