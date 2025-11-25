"""Evaluate model robustness under various perturbations.

Evaluates a model (or baseline) on clean validation set and under perturbations
(brightness, blur, perspective shift). Computes ΔF1 and ΔIoU relative to clean baseline.
"""

import argparse
import json
import importlib
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
import subprocess
import numpy as np

from src.data.dataset_npz import CostmapPairsNPZ
from src.data.augmented_dataset import AugmentedCostmapPairs
from src.train.metrics import mae, iou_binary, precision_recall_f1


def evaluate_model(model, dataloader, device):
    """Evaluate a model on a dataloader."""
    model.eval()
    m_mae = m_iou = m_p = m_r = m_f1 = 0.0
    n = 0
    
    with torch.no_grad():
        for img, cm in dataloader:
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
        return None
    
    return {
        'mae': m_mae / n,
        'iou': m_iou / n,
        'precision': m_p / n,
        'recall': m_r / n,
        'f1': m_f1 / n
    }


def evaluate_learned_model(args):
    """Evaluate a learned model (UNet) under perturbations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('cfg', {})
    
    # Get model config
    model_module = args.model_module or cfg.get('model_module', 'src.models.unet')
    model_class = args.model_class or cfg.get('model_class', 'UNet')
    model_kwargs = args.model_kwargs or cfg.get('model_kwargs', {'in_channels': 4, 'out_channels': 1, 'base_channels': 32})
    
    # Build model
    ModelClass = getattr(importlib.import_module(model_module), model_class)
    model = ModelClass(**model_kwargs)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    
    channels = 'rgb' if cfg.get('rgb_only', False) else 'rgbd'
    
    # Evaluate clean baseline
    clean_ds = CostmapPairsNPZ(args.data_root, split=args.split, dataset=args.dataset, channels=channels)
    clean_dl = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("Evaluating clean baseline...")
    clean_metrics = evaluate_model(model, clean_dl, device)
    if clean_metrics is None:
        print("Error: No samples found")
        return None
    
    results = {
        'dataset': args.dataset,
        'split': args.split,
        'model_type': 'learned',
        'checkpoint': args.checkpoint,
        'clean': clean_metrics,
        'perturbations': {}
    }
    
    # Evaluate perturbations
    perturbations = [
        {'name': 'brightness_darker', 'brightness_factor': 0.7},
        {'name': 'brightness_brighter', 'brightness_factor': 1.3},
        {'name': 'blur_sigma1', 'blur_sigma': 1.0},
        {'name': 'blur_sigma2', 'blur_sigma': 2.0},
        {'name': 'perspective_shift', 'perspective_params': {'rotation_deg': 5, 'translation_px': 5}},
    ]
    
    for pert in perturbations:
        print(f"Evaluating perturbation: {pert['name']}...")
        # Load normalization params from config
        import yaml
        from pathlib import Path
        config_path = Path('configs/data.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            norm_mean = np.array(config['image']['normalize']['mean'])
            norm_std = np.array(config['image']['normalize']['std'])
        else:
            norm_mean = None  # Use defaults
            norm_std = None
        
        pert_ds = AugmentedCostmapPairs(
            clean_ds,
            brightness_factor=pert.get('brightness_factor'),
            blur_sigma=pert.get('blur_sigma'),
            perspective_params=pert.get('perspective_params'),
            normalization_mean=norm_mean,
            normalization_std=norm_std,
            seed=args.seed
        )
        pert_dl = DataLoader(pert_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        pert_metrics = evaluate_model(model, pert_dl, device)
        if pert_metrics:
            # Compute deltas
            pert_result = {
                **pert_metrics,
                'delta_f1': pert_metrics['f1'] - clean_metrics['f1'],
                'delta_iou': pert_metrics['iou'] - clean_metrics['iou'],
                'delta_mae': pert_metrics['mae'] - clean_metrics['mae']
            }
            results['perturbations'][pert['name']] = pert_result
    
    return results


def evaluate_baseline_with_perturbations(processed_root, dataset, split, perturbations, near_thresh, dilation_radius, depth_mean, depth_std):
    """Evaluate classical baseline with perturbations applied to input images."""
    from scripts.baseline_depth_costmaps import (
        iter_npz, depth_channel, baseline_cost_from_depth,
        mae, iou_binary, precision_recall_f1
    )
    from scipy import ndimage
    import cv2
    import yaml
    from pathlib import Path
    
    # Load normalization parameters
    config_path = Path('configs/data.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        norm_mean = np.array(config['image']['normalize']['mean'])
        norm_std = np.array(config['image']['normalize']['std'])
    else:
        norm_mean = np.array([0.485, 0.456, 0.406, 0.5])
        norm_std = np.array([0.229, 0.224, 0.225, 0.25])
    
    all_results = {}
    
    for pert_name, pert_params in perturbations.items():
        records = []
        
        for stem, rgbd, label in iter_npz(processed_root, dataset, split):
            # Apply perturbations
            rgbd_perturbed = rgbd.copy()
            label_perturbed = label.copy()
            
            # Brightness adjustment (RGB channels only - doesn't affect depth-based baseline)
            if pert_params.get('brightness_factor'):
                brightness_factor = pert_params['brightness_factor']
                # Denormalize RGB
                for c in range(3):
                    rgbd_perturbed[..., c] = rgbd_perturbed[..., c] * norm_std[c] + norm_mean[c]
                # Apply brightness
                rgbd_perturbed[..., :3] *= brightness_factor
                # Clamp and renormalize
                rgbd_perturbed[..., :3] = np.clip(rgbd_perturbed[..., :3], 0.0, 1.0)
                for c in range(3):
                    rgbd_perturbed[..., c] = (rgbd_perturbed[..., c] - norm_mean[c]) / norm_std[c]
            
            # Gaussian blur (RGB channels only - doesn't affect depth-based baseline)
            if pert_params.get('blur_sigma'):
                blur_sigma = pert_params['blur_sigma']
                # Denormalize RGB
                for c in range(3):
                    rgbd_perturbed[..., c] = rgbd_perturbed[..., c] * norm_std[c] + norm_mean[c]
                # Apply blur
                for c in range(3):
                    channel = rgbd_perturbed[..., c]
                    channel_uint8 = (channel * 255).astype(np.uint8)
                    ksize = int(6 * blur_sigma) | 1
                    if ksize < 3:
                        ksize = 3
                    blurred = cv2.GaussianBlur(channel_uint8, (ksize, ksize), blur_sigma)
                    rgbd_perturbed[..., c] = blurred.astype(np.float32) / 255.0
                # Renormalize
                for c in range(3):
                    rgbd_perturbed[..., c] = (rgbd_perturbed[..., c] - norm_mean[c]) / norm_std[c]
            
            # Perspective shift (affects all channels including depth)
            if pert_params.get('perspective_params'):
                rot_deg = pert_params['perspective_params'].get('rotation_deg', 0)
                trans_px = pert_params['perspective_params'].get('translation_px', 0)
                
                if rot_deg != 0 or trans_px != 0:
                    h, w = rgbd_perturbed.shape[:2]
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
                    M[0, 2] += trans_px
                    M[1, 2] += trans_px
                    
                    # Transform each channel
                    for c in range(rgbd_perturbed.shape[-1]):
                        channel = rgbd_perturbed[..., c]
                        # Denormalize to work in pixel space
                        channel_denorm = channel * norm_std[c] + norm_mean[c]
                        # Convert to uint8 for transformation
                        channel_uint8 = np.clip(channel_denorm * 255, 0, 255).astype(np.uint8)
                        # Apply transformation
                        transformed = cv2.warpAffine(channel_uint8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        # Convert back and renormalize
                        transformed_float = transformed.astype(np.float32) / 255.0
                        rgbd_perturbed[..., c] = (transformed_float - norm_mean[c]) / norm_std[c]
                    
                    # Also transform the label costmap to match
                    label_uint8 = (label_perturbed * 255).astype(np.uint8)
                    label_transformed = cv2.warpAffine(label_uint8, M, (label_perturbed.shape[1], label_perturbed.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                    label_perturbed = label_transformed.astype(np.float32) / 255.0
            
            # Extract depth and generate baseline costmap
            depth_norm = depth_channel(rgbd_perturbed)
            depth_m = depth_norm * depth_std + depth_mean
            pred = baseline_cost_from_depth(depth_m, near_thresh=near_thresh, dilation_radius=dilation_radius)
            
            # Ensure shape matches
            if pred.shape != label_perturbed.shape:
                pred = ndimage.zoom(pred, (label_perturbed.shape[0]/pred.shape[0], label_perturbed.shape[1]/pred.shape[1]), order=1)
            
            pred_t = np.clip(pred, 0, 1)
            pred_tensor = torch.from_numpy(pred_t).float()
            label_tensor = torch.from_numpy(label_perturbed).float()
            m_mae = mae(pred_tensor, label_tensor)
            m_iou = iou_binary(pred_tensor, label_tensor)
            p, r, f1 = precision_recall_f1(pred_tensor, label_tensor)
            
            records.append({
                'stem': stem,
                'mae': m_mae,
                'iou': m_iou,
                'precision': p,
                'recall': r,
                'f1': f1
            })
        
        if records:
            all_results[pert_name] = {
                'mae': float(np.mean([r['mae'] for r in records])),
                'iou': float(np.mean([r['iou'] for r in records])),
                'precision': float(np.mean([r['precision'] for r in records])),
                'recall': float(np.mean([r['recall'] for r in records])),
                'f1': float(np.mean([r['f1'] for r in records])),
            }
    
    return all_results


def evaluate_classical_baseline(args):
    """Evaluate classical baseline under perturbations."""
    from pathlib import Path
    import yaml
    from scripts.baseline_depth_costmaps import (
        iter_npz, depth_channel, baseline_cost_from_depth,
        mae, iou_binary, precision_recall_f1
    )
    from scipy import ndimage
    
    # Load normalization parameters
    config_path = Path('configs/data.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mean = config['image']['normalize']['mean']
    std = config['image']['normalize']['std']
    depth_mean = float(mean[3])
    depth_std = float(std[3])
    
    # Evaluate clean baseline
    print("Evaluating clean baseline...")
    clean_records = []
    for stem, rgbd, label in iter_npz(Path(args.data_root), args.dataset, args.split):
        depth_norm = depth_channel(rgbd)
        depth_m = depth_norm * depth_std + depth_mean
        pred = baseline_cost_from_depth(depth_m, near_thresh=0.5, dilation_radius=2)
        if pred.shape != label.shape:
            pred = ndimage.zoom(pred, (label.shape[0]/pred.shape[0], label.shape[1]/pred.shape[1]), order=1)
        pred_t = np.clip(pred, 0, 1)
        pred_tensor = torch.from_numpy(pred_t).float()
        label_tensor = torch.from_numpy(label).float()
        m_mae = mae(pred_tensor, label_tensor)
        m_iou = iou_binary(pred_tensor, label_tensor)
        p, r, f1 = precision_recall_f1(pred_tensor, label_tensor)
        clean_records.append({
            'mae': m_mae,
            'iou': m_iou,
            'precision': p,
            'recall': r,
            'f1': f1
        })
    
    clean_metrics = {
        'mae': float(np.mean([r['mae'] for r in clean_records])),
        'iou': float(np.mean([r['iou'] for r in clean_records])),
        'precision': float(np.mean([r['precision'] for r in clean_records])),
        'recall': float(np.mean([r['recall'] for r in clean_records])),
        'f1': float(np.mean([r['f1'] for r in clean_records])),
    }
    
    results = {
        'dataset': args.dataset,
        'split': args.split,
        'model_type': 'classical_baseline',
        'clean': clean_metrics,
        'perturbations': {}
    }
    
    # Define perturbations
    perturbations = {
        'brightness_darker': {'brightness_factor': 0.7},
        'brightness_brighter': {'brightness_factor': 1.3},
        'blur_sigma1': {'blur_sigma': 1.0},
        'blur_sigma2': {'blur_sigma': 2.0},
        'perspective_shift': {'perspective_params': {'rotation_deg': 5, 'translation_px': 5}},
    }
    
    # Evaluate with perturbations
    print("Evaluating baseline with perturbations...")
    pert_results = evaluate_baseline_with_perturbations(
        Path(args.data_root),
        args.dataset,
        args.split,
        perturbations,
        near_thresh=0.5,
        dilation_radius=2,
        depth_mean=depth_mean,
        depth_std=depth_std
    )
    
    # Compute deltas and add to results
    for pert_name, pert_metrics in pert_results.items():
        results['perturbations'][pert_name] = {
            **pert_metrics,
            'delta_f1': pert_metrics['f1'] - clean_metrics['f1'],
            'delta_iou': pert_metrics['iou'] - clean_metrics['iou'],
            'delta_mae': pert_metrics['mae'] - clean_metrics['mae']
        }
    
    return results


def main():
    ap = argparse.ArgumentParser(description='Evaluate robustness under perturbations')
    ap.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (for learned models)')
    ap.add_argument('--classical-baseline', action='store_true', help='Evaluate classical baseline instead of learned model')
    ap.add_argument('--dataset', choices=['nyu', 'kitti'], required=True)
    ap.add_argument('--split', choices=['val', 'train'], default='val')
    ap.add_argument('--data-root', type=str, default='data/processed')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--model-module', type=str, default=None)
    ap.add_argument('--model-class', type=str, default=None)
    ap.add_argument('--model-kwargs', type=dict, default=None)
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    
    if not args.classical_baseline and not args.checkpoint:
        ap.error("Either --checkpoint or --classical-baseline must be specified")
    
    # Evaluate
    if args.classical_baseline:
        results = evaluate_classical_baseline(args)
    else:
        results = evaluate_learned_model(args)
    
    if results is None:
        print("Evaluation failed")
        return
    
    # Add metadata
    results['created_at'] = datetime.utcnow().isoformat() + "Z"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        results['git_commit'] = git_hash
    except Exception:
        pass
    
    # Save results
    args.out.mkdir(parents=True, exist_ok=True)
    model_type = 'baseline' if args.classical_baseline else 'unet'
    out_file = args.out / f"robustness_{model_type}_{args.dataset}.json"
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {out_file}")
    print(f"Clean metrics - F1: {results['clean']['f1']:.4f}, IoU: {results['clean']['iou']:.4f}")
    if results.get('perturbations'):
        print("\nPerturbation results:")
        for name, pert in results['perturbations'].items():
            print(f"  {name}: ΔF1={pert['delta_f1']:.4f}, ΔIoU={pert['delta_iou']:.4f}")


if __name__ == '__main__':
    main()

