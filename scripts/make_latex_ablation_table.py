"""Generate LaTeX table for UNet objective ablation results.

Reads from results/unet_objective_ablation.json and generates a comparison table
showing L1-only, L1+Dice, and L1+Dice+Boundary results for both datasets.
"""

import json
from pathlib import Path
import argparse


def make_table(data: dict, threshold: float) -> str:
    """Generate LaTeX table for objective ablation."""
    header = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{UNet objective ablation. Threshold $\\tau={threshold}$. "
        "Compares L1-only, L1+Dice, and L1+Dice+Boundary loss configurations.}}\n"
        "  \\label{tab:unet_objective_ablation}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    Dataset & Loss Config & MAE $\\downarrow$ & IoU $\\uparrow$ & Precision & Recall & F1 \\\\ \n"
        "    \\midrule\n"
    )
    
    rows = data.get("results", [])
    body_lines = []
    
    # Order by dataset, then by loss config
    order = {
        "nyu": ["l1_only", "l1_dice", "l1_dice_boundary"],
        "kitti": ["l1_only", "l1_dice", "l1_dice_boundary"]
    }
    
    for dataset in ["nyu", "kitti"]:
        for loss_config in order[dataset]:
            for r in rows:
                if r.get("dataset") == dataset and r.get("loss_config") == loss_config:
                    loss_name = {
                        "l1_only": "L1 only",
                        "l1_dice": "L1 + Dice",
                        "l1_dice_boundary": "L1 + Dice + Boundary"
                    }.get(loss_config, loss_config)
                    dataset_name = dataset.upper()
                    line = (
                        f"    {dataset_name} & {loss_name}"
                        + f" & {r['mae']:.4f} & {r['iou']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} \\\\"
                    )
                    body_lines.append(line)
                    break
    
    footer = (
        "\n    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    
    return header + "\n".join(body_lines) + footer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('results/unet_objective_ablation.json'))
    ap.add_argument('--out-dir', type=Path, default=Path('docs/tables'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    if not args.results.exists():
        print(f"Error: {args.results} does not exist. Please run objective ablation evaluations first.")
        return
    
    with open(args.results, 'r') as f:
        data = json.load(f)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "unet_objective_ablation_table.tex"
    
    table = make_table(data, args.threshold)
    out_file.write_text(table)
    print(f"Wrote: {out_file}")


if __name__ == '__main__':
    main()

