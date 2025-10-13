import argparse
import importlib
import torch
from torch.utils.data import DataLoader
from src.data.dataset_npz import CostmapPairsNPZ
from src.train.metrics import mae, iou_binary, precision_recall_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_root', default='data/processed')
    ap.add_argument('--dataset', choices=['nyu', 'kitti'], default='nyu')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--model_module', default='src.models.unet')
    ap.add_argument('--model_class', default='UNet')
    ap.add_argument('--in_channels', type=int, default=4)
    ap.add_argument('--out_channels', type=int, default=1)
    ap.add_argument('--base_channels', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CostmapPairsNPZ(args.data_root, split='val', dataset=args.dataset)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ModelClass = getattr(importlib.import_module(args.model_module), args.model_class)
    model = ModelClass(in_channels=args.in_channels, out_channels=args.out_channels, base_channels=args.base_channels)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    m_mae = m_iou = m_p = m_r = m_f1 = 0.0
    n = 0
    with torch.no_grad():
        for img, cm in dl:
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
            m_p += p; m_r += r; m_f1 += f1
            n += 1

    if n == 0:
        print('No samples found')
        return
    print({
        'mae': m_mae/n,
        'iou': m_iou/n,
        'precision': m_p/n,
        'recall': m_r/n,
        'f1': m_f1/n,
    })


if __name__ == '__main__':
    main()
