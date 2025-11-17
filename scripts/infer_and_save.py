"""Run inference for a trained model checkpoint and save per-scene predicted costmaps.

Outputs:
  - predictions/<dataset>/<model_tag>/<split>/<stem>.npy (64x64 float32 in [0,1])
Also writes a manifest JSON with args.
"""
import argparse
import importlib
from pathlib import Path
import json
import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

def ensure_src_on_path():
	cwd = os.getcwd()
	if cwd not in sys.path:
		sys.path.append(cwd)
	# allow running from scripts/ subdirectory as well
	scripts_dir = os.path.dirname(__file__)
	parent = os.path.abspath(os.path.join(scripts_dir, '..'))
	if parent not in sys.path:
		sys.path.append(parent)

ensure_src_on_path()
from src.data.dataset_npz import CostmapPairsNPZ


def save_pred(path: Path, arr: np.ndarray):
	path.parent.mkdir(parents=True, exist_ok=True)
	np.save(path, arr.astype(np.float32))


def main():
	ap = argparse.ArgumentParser(description="Inference and save predicted costmaps")
	ap.add_argument('--checkpoint', required=True)
	ap.add_argument('--dataset', choices=['nyu','kitti'], default='nyu')
	ap.add_argument('--split', choices=['train','val'], default='val')
	ap.add_argument('--data-root', default='data/processed')
	ap.add_argument('--model-module', default='src.models.unet')
	ap.add_argument('--model-class', default='UNet')
	ap.add_argument('--in-channels', type=int, default=4)
	ap.add_argument('--out-channels', type=int, default=1)
	ap.add_argument('--base-channels', type=int, default=32)
	ap.add_argument('--batch-size', type=int, default=16)
	ap.add_argument('--num-workers', type=int, default=4)
	ap.add_argument('--out-root', default='predictions')
	ap.add_argument('--model-tag', default='unet')
	ap.add_argument('--rgb-only', action='store_true', help='Use RGB-only channels (3) for dataset')
	ap.add_argument('--limit', type=int, default=0, help='Optional cap on number of predictions to write (0=all)')
	ap.add_argument('--depth', type=int, default=None, help='Depth- for vit')

	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Ensure local src is importable when running as a script
	sys.path.append(os.getcwd())
	channels = 'rgb' if args.rgb_only or args.in_channels == 3 else 'rgbd'
	ds = CostmapPairsNPZ(args.data_root, split=args.split, dataset=args.dataset, channels=channels)
	dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	ModelClass = getattr(importlib.import_module(args.model_module), args.model_class)
	model = ModelClass(in_channels=args.in_channels, out_channels=args.out_channels, base_channels=args.base_channels, depth=args.depth)
	ckpt = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(ckpt['model'])
	model.to(device).eval()

	out_dir = Path(args.out_root) / args.dataset / args.model_tag / args.split
	preds_written = 0
	first_batch_stats = None
	with torch.no_grad():
		for idx, (img, cm) in enumerate(dl):
			img = img.to(device)
			pred = model(img)
			pred_sig = torch.sigmoid(pred)
			if pred_sig.shape[-2:] != cm.shape[-2:]:
				pred_sig = torch.nn.functional.interpolate(pred_sig, size=cm.shape[-2:], mode='bilinear', align_corners=False)
			# Log stats on first batch and guard against degenerate predictions
			if first_batch_stats is None:
				first_batch_stats = {
					'logits_mean': float(pred.mean().item()),
					'logits_std': float(pred.std().item()),
					'sigmoid_mean': float(pred_sig.mean().item()),
					'sigmoid_std': float(pred_sig.std().item()),
				}
				print('[infer_and_save] First batch stats:', first_batch_stats)
				if first_batch_stats['sigmoid_std'] <= 1e-4:
					raise RuntimeError(f"Degenerate predictions detected (std={first_batch_stats['sigmoid_std']}). Aborting.")
			for b in range(pred_sig.size(0)):
				stem = f"{idx*args.batch_size + b:06d}"  # relies on ordering of dataset
				arr = pred_sig[b,0].cpu().numpy()
				save_pred(out_dir / f"{stem}.npy", arr)
				preds_written += 1
				if args.limit and preds_written >= args.limit:
					break
			if args.limit and preds_written >= args.limit:
				break

	manifest = {
		'checkpoint': args.checkpoint,
		'dataset': args.dataset,
		'split': args.split,
		'model_module': args.model_module,
		'model_class': args.model_class,
		'in_channels': args.in_channels,
		'base_channels': args.base_channels,
		'num_preds': preds_written,
		'first_batch_stats': first_batch_stats,
	}
	with open(out_dir / 'manifest.json', 'w') as f:
		json.dump(manifest, f, indent=2)
	print('Saved predictions to', out_dir, 'count:', preds_written)


if __name__ == '__main__':
	main()

