"""Comprehensive costmap analysis utility.

Computes distribution statistics (mean/std/min/max), histogram + KL divergence
between labels and predictions, per-pixel absolute error, binary IoU at a
threshold, and generates visual diagnostics (difference maps, binary overlays).

Outputs:
  - JSON summary: results/analysis_{dataset}_{split}_{tag}.json
  - Histogram overlay: docs/figures/hist_{dataset}_{split}_{tag}.png
  - Difference panels per selected stems:
        docs/figures/diff_{dataset}_{tag}_{stem}.png
  - Binary overlay panels per selected stems:
        docs/figures/bin_{dataset}_{tag}_{stem}.png

Example:
  python scripts/analyze_costmaps.py --dataset nyu --split val \
      --pred-dir predictions/nyu/val --stems 000010 000025 000040

To compare RGB-only vs RGBD predictions, run separately and compare JSON.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def load_label(processed_root: Path, dataset: str, split: str, stem: str) -> np.ndarray:
    p = processed_root / dataset / split / f"{stem}.npz"
    d = np.load(p, allow_pickle=True)
    # prefer 'costmap' or fallback keys
    for k in ("costmap", "cost", "target", "y", "label"):
        if k in d:
            arr = d[k].astype(np.float32)
            if arr.ndim == 3:
                arr = arr.squeeze()
            return arr
    raise KeyError(f"No costmap-like key in {p}")


def load_pred(pred_dir: Path, stem: str) -> np.ndarray:
    p = pred_dir / f"{stem}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.load(p).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    return arr


def kl_divergence(p_hist, q_hist, eps=1e-8):
    # p_hist, q_hist are normalized
    return float(np.sum(p_hist * np.log((p_hist + eps) / (q_hist + eps))))


def analyze(dataset: str, split: str, processed_root: Path, pred_dir: Path, stems: List[str], out_root: Path, figs_root: Path, threshold: float):
    labels = []
    preds = []
    abs_errors = []
    bin_ious = []
    for s in stems:
        lab = load_label(processed_root, dataset, split, s)
        pred = load_pred(pred_dir, s)
        if pred.shape != lab.shape:
            from scipy.ndimage import zoom
            pred = zoom(pred, (lab.shape[0]/pred.shape[0], lab.shape[1]/pred.shape[1]), order=1)
        labels.append(lab)
        preds.append(pred)
        abs_errors.append(np.abs(lab - pred))
        # binary IoU
        lb = (lab >= threshold).astype(np.uint8)
        pb = (pred >= threshold).astype(np.uint8)
        inter = (lb & pb).sum()
        union = (lb | pb).sum() + 1e-6
        bin_ious.append(float(inter / union))
    L = np.stack(labels)
    P = np.stack(preds)
    E = np.stack(abs_errors)
    stats = {
        "dataset": dataset,
        "split": split,
        "pred_dir": str(pred_dir),
        "num_samples": len(stems),
        "threshold": threshold,
        "label_mean": float(L.mean()),
        "label_std": float(L.std()),
        "label_min": float(L.min()),
        "label_max": float(L.max()),
        "pred_mean": float(P.mean()),
        "pred_std": float(P.std()),
        "pred_min": float(P.min()),
        "pred_max": float(P.max()),
        "mae_mean": float(E.mean()),
        "mae_std": float(E.std()),
        "binary_iou_mean": float(np.mean(bin_ious)),
        "binary_iou_std": float(np.std(bin_ious)),
    }
    # histograms
    bins = np.linspace(0, 1, 51)
    l_hist, _ = np.histogram(L.flatten(), bins=bins, density=True)
    p_hist, _ = np.histogram(P.flatten(), bins=bins, density=True)
    stats["kl_label_to_pred"] = kl_divergence(l_hist / (l_hist.sum() + 1e-8), p_hist / (p_hist.sum() + 1e-8))
    stats["kl_pred_to_label"] = kl_divergence(p_hist / (p_hist.sum() + 1e-8), l_hist / (l_hist.sum() + 1e-8))

    # histogram overlay figure
    figs_root.mkdir(parents=True, exist_ok=True)
    mid = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(5,3))
    plt.plot(mid, l_hist, label='Label', linewidth=2)
    plt.plot(mid, p_hist, label='Pred', linewidth=2)
    plt.xlabel('Cost')
    plt.ylabel('Density')
    plt.title(f"Histogram {dataset}-{split}")
    plt.legend(frameon=False)
    plt.tight_layout()
    base_tag = f"{dataset}_{split}_{pred_dir.parent.name if pred_dir.parent.name!=dataset else 'pred'}"
    hist_path = figs_root / f"hist_{base_tag}.png"
    plt.savefig(hist_path, dpi=160)
    plt.close()
    stats["histogram_png"] = str(hist_path)

    # difference and binary overlay for up to first 5 stems (or provided subset)
    for s in stems[:5]:
        lab = load_label(processed_root, dataset, split, s)
        pred = load_pred(pred_dir, s)
        if pred.shape != lab.shape:
            from scipy.ndimage import zoom
            pred = zoom(pred, (lab.shape[0]/pred.shape[0], lab.shape[1]/pred.shape[1]), order=1)
        diff = lab - pred
        lb = (lab >= threshold).astype(np.uint8)
        pb = (pred >= threshold).astype(np.uint8)
        # difference map
        fig, axs = plt.subplots(1, 3, figsize=(9,3))
        axs[0].imshow(lab, vmin=0, vmax=1, cmap='viridis'); axs[0].set_title('Label'); axs[0].axis('off')
        axs[1].imshow(pred, vmin=0, vmax=1, cmap='viridis'); axs[1].set_title('Pred'); axs[1].axis('off')
        im = axs[2].imshow(diff, cmap='bwr', vmin=-0.5, vmax=0.5); axs[2].set_title('Label - Pred'); axs[2].axis('off')
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        fig.tight_layout()
        diff_path = figs_root / f"diff_{dataset}_{base_tag}_{s}.png"
        fig.savefig(diff_path, dpi=160)
        plt.close(fig)
        # binary overlay
        overlay = lb + 2*pb  # encode four cases: 0 none,1 label only,2 pred only,3 both
        cmap = plt.get_cmap('tab10')
        fig2, ax2 = plt.subplots(1,1,figsize=(3,3))
        ax2.imshow(overlay, vmin=0, vmax=3, cmap='viridis')
        ax2.set_title(f'Binary Overlay τ={threshold}')
        ax2.axis('off')
        bin_path = figs_root / f"bin_{dataset}_{base_tag}_{s}.png"
        fig2.tight_layout()
        fig2.savefig(bin_path, dpi=160)
        plt.close(fig2)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / f"analysis_{base_tag}.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote analysis JSON:", out_root / f"analysis_{base_tag}.json")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['nyu','kitti'], required=True)
    ap.add_argument('--split', choices=['val','train'], default='val')
    ap.add_argument('--processed-root', type=Path, default=Path('data/processed'))
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--stems', nargs='+', default=None, help='Subset of stems to analyze (default: all)')
    ap.add_argument('--num', type=int, default=0, help='If stems not given, use first N (0=all)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out-root', type=Path, default=Path('results'))
    ap.add_argument('--figs-root', type=Path, default=Path('docs/figures'))
    args = ap.parse_args()

    all_stems = [p.stem for p in sorted((args.processed_root / args.dataset / args.split).glob('*.npz'))]
    if args.stems:
        stems = args.stems
    else:
        stems = all_stems if args.num == 0 else all_stems[:args.num]

    analyze(args.dataset, args.split, args.processed_root, args.pred_dir, stems, args.out_root, args.figs_root, args.threshold)


if __name__ == '__main__':
    main()
