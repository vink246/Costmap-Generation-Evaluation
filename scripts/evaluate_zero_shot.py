"""Evaluate a model trained on one dataset directly on another dataset (zero-shot).

This script loads a checkpoint trained on source dataset and evaluates it
directly on target dataset without any fine-tuning.
"""

import argparse
import importlib
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset_npz import CostmapPairsNPZ
from src.train.metrics import mae, iou_binary, precision_recall_f1
import subprocess
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(description='Zero-shot cross-domain evaluation')
    ap.add_argument('--checkpoint', required=True, help='Path to source dataset checkpoint')
    ap.add_argument('--source-dataset', choices=['nyu', 'kitti'], required=True, help='Dataset model was trained on')
    ap.add_argument('--target-dataset', choices=['nyu', 'kitti'], required=True, help='Dataset to evaluate on')
    ap.add_argument('--data-root', default='data/processed')
    ap.add_argument('--split', choices=['val', 'train'], default='val')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--model-module', default='src.models.unet')
    ap.add_argument('--model-class', default='UNet')
    ap.add_argument('--in-channels', type=int, default=4)
    ap.add_argument('--out-channels', type=int, default=1)
    ap.add_argument('--base-channels', type=int, default=32)
    ap.add_argument('--channels', choices=['rgb', 'rgbd'], default='rgbd')
    ap.add_argument('--out', type=Path, default=Path('results'))
    args = ap.parse_args()

    if args.source_dataset == args.target_dataset:
        print(f"Warning: source and target datasets are the same ({args.source_dataset})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to determine model config
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('cfg', {})
    
    # Use config values if available, otherwise use args
    in_channels = cfg.get('model_kwargs', {}).get('in_channels', args.in_channels)
    out_channels = cfg.get('model_kwargs', {}).get('out_channels', args.out_channels)
    base_channels = cfg.get('model_kwargs', {}).get('base_channels', args.base_channels)
    
    # Load target dataset
    target_ds = CostmapPairsNPZ(args.data_root, split=args.split, dataset=args.target_dataset, channels=args.channels)
    target_dl = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load model
    ModelClass = getattr(importlib.import_module(args.model_module), args.model_class)
    model = ModelClass(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    
    # Evaluate
    m_mae = m_iou = m_p = m_r = m_f1 = 0.0
    n = 0
    with torch.no_grad():
        for img, cm in target_dl:
            img = img.to(device)
            cm = cm.to(device)
            pred_full = model(img)
            if pred_full.shape[-2:] != cm.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred_full, size=cm.shape[-2:], mode='bilinear', align_corners=False)
            else:
                pred = pred_full
            pred_sig = torch.sigmoid(pred)
            m_mae += mae(pred_sig, cm)
            m_iou += iou_binary(pred_sig, cm)
            p, r, f1 = precision_recall_f1(pred_sig, cm)
            m_p += p
            m_r += r
            m_f1 += f1
            n += 1
    
    if n == 0:
        print('No samples found')
        return
    
    metrics = {
        'mae': m_mae / n,
        'iou': m_iou / n,
        'precision': m_p / n,
        'recall': m_r / n,
        'f1': m_f1 / n
    }
    
    # Save results
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        'source_dataset': args.source_dataset,
        'target_dataset': args.target_dataset,
        'split': args.split,
        'checkpoint': args.checkpoint,
        'metrics': metrics,
        'num_samples': n
    }
    
    # Add reproducibility metadata
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        result['git_commit'] = git_hash
    except Exception:
        pass
    
    result['created_at'] = datetime.utcnow().isoformat() + "Z"
    
    out_file = args.out / f"zero_shot_{args.source_dataset}_to_{args.target_dataset}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Zero-shot evaluation ({args.source_dataset} -> {args.target_dataset}):")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()

