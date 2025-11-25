"""Generate LaTeX table for UNet cross-domain transfer results.

Reads from results/unet_cross_domain.json and generates a comparison table
showing NYU only, KITTI only, transfer learning, and zero-shot results.
"""

import json
from pathlib import Path
import argparse


def format_variant_name(variant: str) -> str:
    """Format variant name for LaTeX table."""
    if variant == "nyu_only":
        return "NYU only"
    elif variant == "kitti_only":
        return "KITTI only"
    elif variant == "nyu_to_kitti_tl":
        return "NYU $\\rightarrow$ KITTI (TL)"
    elif variant == "kitti_to_nyu_tl":
        return "KITTI $\\rightarrow$ NYU (TL)"
    elif variant == "nyu_to_kitti_zs":
        return "NYU $\\rightarrow$ KITTI (ZS)"
    elif variant == "kitti_to_nyu_zs":
        return "KITTI $\\rightarrow$ NYU (ZS)"
    return variant


def make_table(data: dict, dataset: str, threshold: float) -> str:
    """Generate LaTeX table for a dataset."""
    header = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{UNet cross-domain evaluation on {dataset.upper()}. "
        f"Threshold $\\tau={threshold}$. TL = transfer learning, ZS = zero-shot.}}\n"
        f"  \\label{{tab:unet_cross_domain_{dataset}}}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    Training Scenario & MAE $\\downarrow$ & IoU $\\uparrow$ & Precision & Recall & F1 \\\\ \n"
        "    \\midrule\n"
    )
    
    rows = data.get("datasets", {}).get(dataset, [])
    body_lines = []
    
    # Order: NYU only, KITTI only, transfers, zero-shots
    order = ["nyu_only", "kitti_only", "nyu_to_kitti_tl", "kitti_to_nyu_tl", "nyu_to_kitti_zs", "kitti_to_nyu_zs"]
    ordered_rows = []
    for variant in order:
        for r in rows:
            if r.get("variant") == variant:
                ordered_rows.append(r)
                break
    
    # Add any remaining rows not in order
    for r in rows:
        if r not in ordered_rows:
            ordered_rows.append(r)
    
    for r in ordered_rows:
        variant_name = format_variant_name(r.get("variant", ""))
        line = (
            f"    {variant_name}"
            + f" & {r['mae']:.4f} & {r['iou']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} \\\\"
        )
        body_lines.append(line)
    
    footer = (
        "\n    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    
    return header + "\n".join(body_lines) + footer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('results/unet_cross_domain.json'))
    ap.add_argument('--out-dir', type=Path, default=Path('docs/tables'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    if not args.results.exists():
        print(f"Error: {args.results} does not exist. Please run cross-domain evaluations first.")
        return
    
    with open(args.results, 'r') as f:
        data = json.load(f)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "unet_cross_domain_tables.tex"
    
    tables = []
    for dataset in ["nyu", "kitti"]:
        if dataset in data.get("datasets", {}):
            tables.append(make_table(data, dataset, args.threshold))
            tables.append("\n% ---\n")
    
    out_file.write_text("\n\n".join(tables))
    print(f"Wrote: {out_file}")


if __name__ == '__main__':
    main()

