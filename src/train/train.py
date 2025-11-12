import os
import platform
import argparse
import yaml
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset_npz import CostmapPairsNPZ
from .losses import L1Loss, DiceLoss, BoundaryLoss, BCEWithLogits
from .metrics import mae, iou_binary, precision_recall_f1


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_dataloaders(data_root, dataset_name, batch_size, num_workers, channels: str):
    train_ds = CostmapPairsNPZ(data_root, split='train', dataset=dataset_name, channels=channels)
    val_ds = CostmapPairsNPZ(data_root, split='val', dataset=dataset_name, channels=channels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


def build_losses(cfg):
    w = cfg.get('loss_weights', {'bce': 1.0,'l1': 1.0, 'dice': 0.5, 'boundary': 0.1})
    return w, L1Loss(), DiceLoss(), BoundaryLoss(), BCEWithLogits()


def compute_loss(pred, target, weights, l1_fn, dice_fn, bnd_fn, bce_fn):
    l_total = 0.0
    logs = {}
    # Apply sigmoid to get probabilities for all losses
    pred_sig = torch.sigmoid(pred)
    
    if weights.get('l1', 0) > 0:
        l1 = l1_fn(pred_sig, target)
        l_total = l_total + weights['l1'] * l1
        logs['l1'] = l1.item()
    if weights.get('dice', 0) > 0:
        d = dice_fn(pred_sig, target)
        l_total = l_total + weights['dice'] * d
        logs['dice'] = d.item()
    if weights.get('boundary', 0) > 0:
        b = bnd_fn(pred_sig, target)
        l_total = l_total + weights['boundary'] * b
        logs['boundary'] = b.item()
    if weights.get('bce', 0) > 0:
        bce = bce_fn(pred, target)
        l_total += weights['bce'] * bce
        logs['bce'] = bce.item()
    logs['total'] = float(l_total.item()) if hasattr(l_total, 'item') else float(l_total)
    return l_total, logs


def evaluate(model, val_dl, device):
    model.eval()
    m_mae = 0.0
    m_iou = 0.0
    m_p = 0.0
    m_r = 0.0
    m_f1 = 0.0
    n = 0
    with torch.no_grad():
        for img, cm in val_dl:
            img = img.to(device)
            cm = cm.to(device)
            pred_full = model(img)
            # Downsample model output to 64x64 if needed
            if pred_full.shape[-2:] != cm.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred_full, size=cm.shape[-2:], mode='bilinear', align_corners=False)
            else:
                pred = pred_full
            pred_sig = torch.sigmoid(pred)
            m_mae += mae(pred_sig, cm)
            i = iou_binary(pred_sig, cm)
            p, r, f1 = precision_recall_f1(pred_sig, cm)
            m_iou += i; m_p += p; m_r += r; m_f1 += f1
            n += 1
    if n == 0:
        return {'mae': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    return {'mae': m_mae/n, 'iou': m_iou/n, 'precision': m_p/n, 'recall': m_r/n, 'f1': m_f1/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Training config YAML')
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get('out_dir', 'runs')
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    data_root = cfg['data_root']  # e.g., data/processed
    dataset_name = cfg.get('dataset', 'nyu')  # nyu or kitti
    batch_size = cfg.get('batch_size', 16)
    epochs = cfg.get('epochs', 5)
    num_workers = cfg.get('num_workers', 4)
    # On Windows, multi-worker DataLoader can hang in some shells; default to 0 unless explicitly set lower
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"[info] Windows detected; forcing num_workers=0 to avoid DataLoader hang (was {num_workers})", flush=True)
        num_workers = 0
    model_module = cfg.get('model_module', 'src.models.unet')
    model_class = cfg.get('model_class', 'UNet')
    model_kwargs = cfg.get('model_kwargs', {'in_channels': 4, 'out_channels': 1, 'base_channels': 32})
    lr = cfg.get('lr', 1e-3)

    # modality selection (RGB-only ablation)
    rgb_only = cfg.get('rgb_only', False)
    dataset_channels = 'rgb' if rgb_only else 'rgbd'

    # if rgb_only but model kwargs still claim 4 channels, override to 3
    if rgb_only and model_kwargs.get('in_channels', 4) != 3:
        model_kwargs['in_channels'] = 3

    train_dl, val_dl = build_dataloaders(data_root, dataset_name, batch_size, num_workers, dataset_channels)
    try:
        n_train = len(train_dl.dataset)
        n_val = len(val_dl.dataset)
        b_train = len(train_dl)
        b_val = len(val_dl)
        print(f"[data] {dataset_name} | channels={dataset_channels} | train={n_train} (batches={b_train}), val={n_val} (batches={b_val}) | device={device}", flush=True)
    except Exception as e:
        print(f"[warn] Could not summarize dataloaders: {e}", flush=True)
    # Dynamically import model class; teammates should implement the class in the module below
    ModelClass = getattr(importlib.import_module(model_module), model_class)
    model = ModelClass(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    weights, l1_fn, dice_fn, bnd_fn, bce_fn = build_losses(cfg)

    best_f1 = -1
    for epoch in range(1, epochs+1):
        model.train()
        total_batches = len(train_dl)
        print(f"[epoch {epoch}/{epochs}] starting training | total_batches={total_batches}", flush=True)
        for bi, (img, cm) in enumerate(train_dl):
            img = img.to(device)
            cm = cm.to(device)
            pred_full = model(img)
            # Resize to 64x64 target
            if pred_full.shape[-2:] != cm.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred_full, size=cm.shape[-2:], mode='bilinear', align_corners=False)
            else:
                pred = pred_full
            loss, logs = compute_loss(pred, cm, weights, l1_fn, dice_fn, bnd_fn,bce_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (bi == 0) or ((bi + 1) % 50 == 0) or ((bi + 1) == total_batches):
                msg = f"[epoch {epoch}] batch {bi+1}/{total_batches} | loss={logs.get('total', 0):.4f}"
                if 'l1' in logs: msg += f", l1={logs['l1']:.4f}"
                if 'dice' in logs: msg += f", dice={logs['dice']:.4f}"
                if 'boundary' in logs: msg += f", boundary={logs['boundary']:.4f}"
                if 'bce' in logs: msg += f", bce={logs['bce']:.4f}"
                print(msg, flush=True)

        # Eval
        metrics = evaluate(model, val_dl, device)
        print(f"[epoch {epoch}] val metrics {metrics}", flush=True)
        # Save best by F1
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch, 'metrics': metrics}, os.path.join(out_dir, f'{model_class}_best.pth'))

    # Save final
    torch.save({'model': model.state_dict(), 'cfg': cfg, 'epoch': epochs}, os.path.join(out_dir, f'{model_class}_last.pth'))


if __name__ == '__main__':
    main()
