"""Wrapper script for KITTI to NYU transfer learning.

This calls the bidirectional transfer_learn.py script with appropriate arguments.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Transfer learning from KITTI to NYU')
    ap.add_argument('--kitti_checkpoint', required=True, help='Path to KITTI-trained model checkpoint')
    ap.add_argument('--nyu_config', required=True, help='Path to NYU training config')
    ap.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    ap.add_argument('--warmup_epochs', type=int, default=5, help='Epochs with frozen encoder')
    args = ap.parse_args()
    
    # Import and call transfer_learn main with renamed arguments
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.train.transfer_learn import load_config, get_device, build_dataloaders, build_losses, compute_loss, evaluate
    from src.train.transfer_learn import load_pretrained_model
    import torch
    import importlib
    
    # Load NYU config
    cfg = load_config(args.nyu_config)
    device = get_device()
    
    # Build model
    model_module = cfg.get('model_module', 'src.models.unet')
    model_class = cfg.get('model_class', 'UNet')
    model_kwargs = cfg.get('model_kwargs', {})
    
    ModelClass = getattr(importlib.import_module(model_module), model_class)
    
    # Load pre-trained KITTI model
    print(f"Loading pre-trained KITTI model from: {args.kitti_checkpoint}")
    model, kitti_checkpoint = load_pretrained_model(args.kitti_checkpoint, ModelClass, model_kwargs, device)
    
    # Build dataloaders
    data_root = cfg['data_root']
    dataset_name = cfg.get('dataset', 'nyu')
    batch_size = cfg.get('batch_size', 8)
    num_workers = cfg.get('num_workers', 4)
    channels = 'rgb' if cfg.get('rgb_only', False) else 'rgbd'
    
    train_dl, val_dl = build_dataloaders(data_root, dataset_name, batch_size, num_workers, channels)
    
    # Setup optimizer and scheduler
    lr = cfg.get('lr', 5e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    scheduler = None
    if cfg.get('lr_scheduler') == 'cosine':
        epochs = cfg.get('epochs', 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    weights, l1_fn, dice_fn, bnd_fn, bce_fn = build_losses(cfg)
    
    # Training loop
    import os
    best_f1 = -1
    epochs = cfg.get('epochs', 50)
    
    for epoch in range(1, epochs + 1):
        # Freeze encoder for warmup epochs
        if args.freeze_encoder and epoch <= args.warmup_epochs:
            for name, param in model.named_parameters():
                if 'down' in name or 'inc' in name:
                    param.requires_grad = False
            print(f"Epoch {epoch}: Encoder frozen (warmup)")
        else:
            for param in model.parameters():
                param.requires_grad = True
            if epoch == args.warmup_epochs + 1:
                print(f"Epoch {epoch}: Encoder unfrozen")
        
        model.train()
        train_losses = {'total': 0, 'l1': 0, 'dice': 0, 'boundary': 0}
        train_batches = 0
        
        for img, cm in train_dl:
            img = img.to(device)
            cm = cm.to(device)
            pred_full = model(img)
            
            if pred_full.shape[-2:] != cm.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred_full, size=cm.shape[-2:], mode='bilinear', align_corners=False)
            else:
                pred = pred_full
            
            loss, logs = compute_loss(pred, cm, weights, l1_fn, dice_fn, bnd_fn, bce_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses['total'] += logs['total']
            train_losses['l1'] += logs.get('l1', 0)
            train_losses['dice'] += logs.get('dice', 0)
            train_losses['boundary'] += logs.get('boundary', 0)
            train_batches += 1
        
        # Evaluation
        metrics = evaluate(model, val_dl, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_train_losses = {k: v/train_batches for k, v in train_losses.items()}
        print(f"Epoch {epoch}: train_losses {avg_train_losses}")
        print(f"Epoch {epoch}: val metrics {metrics}, lr: {current_lr:.6f}")
        
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            out_dir = cfg.get('out_dir', 'runs')
            os.makedirs(out_dir, exist_ok=True)
            torch.save({
                'model': model.state_dict(), 
                'cfg': cfg, 
                'epoch': epoch, 
                'metrics': metrics,
                'transfer_from': args.kitti_checkpoint
            }, os.path.join(out_dir, f'{model_class}_transfer_best.pth'))
    
    print(f"Transfer learning completed. Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()

