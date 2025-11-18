import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import csv

def load_pr_json(path: Path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d["data"]

def compute_auprc(entry):
    rec = np.array(entry["recall"])
    prec = np.array(entry["precision"])
    sorted_idx = np.argsort(rec)
    rec_sorted = rec[sorted_idx]
    prec_sorted = prec[sorted_idx]
    return auc(rec_sorted, prec_sorted)

def find_pr_file(pr_dir: Path, dataset: str, split: str, model_tag: str):
    # Match JSON file by naming convention: pr_<dataset>_<split>_<model_tag>.json
    candidates = list(pr_dir.glob(f"pr_{dataset}_{split}_{model_tag}_{split}.json"))
    if not candidates:
        raise FileNotFoundError(f"No PR file found for dataset={dataset}, split={split}, model={model_tag}")
    return candidates[0]

def save_csv(entry, csv_path: Path):
    """Save PR and F1 data to CSV, ensuring alignment."""
    # Determine minimum length across all arrays
    n = min(len(entry['recall']), len(entry['precision']), len(entry['f1']), len(entry['thresholds']))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'recall', 'precision', 'f1'])
        for i in range(n):
            writer.writerow([entry['thresholds'][i], entry['recall'][i], entry['precision'][i], entry['f1'][i]])


def main():
    ap = argparse.ArgumentParser(description="Plot PR curves for any model/dataset")
    ap.add_argument('--model-tag', default='unet')
    ap.add_argument('--split', choices=['val','train'], default='val')
    ap.add_argument('--pr-dir', type=Path, default=Path('results/pr_curves'))
    args = ap.parse_args()

    datasets = ['nyu','kitti']
    pr_data = {}

    for ds in datasets:
        pr_file = find_pr_file(args.pr_dir, ds, args.split, args.model_tag)
        pr_data[ds] = load_pr_json(pr_file)

    # Compute AUPRC
    auprc = {ds: compute_auprc(pr_data[ds]) for ds in datasets}

    print("AUPRCs:")
    for ds in datasets:
        print(f"  {ds.upper()}: {auprc[ds]}")

    # Create plots
    plot_dir = args.pr_dir / args.model_tag
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    for ds in datasets:
        csv_path = plot_dir / f"{ds}_{args.model_tag}_pr_points.csv"
        save_csv(pr_data[ds], csv_path)
        print(f"Saved CSV for {ds.upper()} to {csv_path}")
        
    # PR curve
    plt.figure(figsize=(7,6))
    for ds in datasets:
        rec = np.array(pr_data[ds]["recall"])
        prec = np.array(pr_data[ds]["precision"])
        sorted_idx = np.argsort(rec)
        rec_sorted = rec[sorted_idx]
        prec_sorted = prec[sorted_idx]
        plt.plot(rec_sorted, prec_sorted, label=f"{ds.upper()} (AUPRC={auprc[ds]:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curves ({args.model_tag})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"pr_curves_{args.model_tag}.png")
    plt.close()

    # F1 vs Threshold
    plt.figure(figsize=(7,6))
    for ds in datasets:
        thresholds = np.array(pr_data[ds]["thresholds"])
        f1 = np.array(pr_data[ds]["f1"])
        sorted_idx = np.argsort(thresholds)
        plt.plot(thresholds[sorted_idx], f1[sorted_idx], label=ds.upper())
    plt.xlabel("Threshold τ")
    plt.ylabel("F1 Score")
    plt.title(f"F1 vs Threshold ({args.model_tag})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"f1_threshold_{args.model_tag}.png")
    plt.close()

if __name__ == "__main__":
    main()
