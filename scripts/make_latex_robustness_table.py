"""Generate LaTeX table for robustness evaluation results.

Reads from results/robustness_*.json files and generates a comparison table
showing clean metrics and perturbation deltas (ΔF1, ΔIoU).
"""

import json
from pathlib import Path
import argparse
import glob


def make_robustness_table(results_list: list, threshold: float = 0.5) -> str:
    """Generate LaTeX table for robustness results.
    
    Args:
        results_list: List of result dicts, each containing clean metrics and perturbations
    """
    header = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{Robustness evaluation. Clean metrics and perturbation deltas (relative to clean). "
        f"Threshold $\\tau={threshold}$.}}\n"
        "  \\label{tab:robustness}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    Model & Dataset & Perturbation & Clean F1 & Clean IoU & $\\Delta$F1 & $\\Delta$IoU \\\\ \n"
        "    \\midrule\n"
    )
    
    body_lines = []
    
    for result in results_list:
        model_type = result.get('model_type', 'unknown').replace('_', ' ').title()
        dataset = result.get('dataset', 'unknown').upper()
        clean = result.get('clean', {})
        clean_f1 = clean.get('f1', 0)
        clean_iou = clean.get('iou', 0)
        
        perturbations = result.get('perturbations', {})
        
        if not perturbations:
            # Show clean only if no perturbations
            body_lines.append(
                f"    {model_type} & {dataset} & Clean & {clean_f1:.4f} & {clean_iou:.4f} & -- & -- \\\\"
            )
        else:
            # First row: clean metrics
            body_lines.append(
                f"    {model_type} & {dataset} & Clean & {clean_f1:.4f} & {clean_iou:.4f} & -- & -- \\\\"
            )
            
            # Subsequent rows: perturbations
            for pert_name, pert_metrics in perturbations.items():
                # Format perturbation name
                pert_display = pert_name.replace('_', ' ').title()
                delta_f1 = pert_metrics.get('delta_f1', 0)
                delta_iou = pert_metrics.get('delta_iou', 0)
                body_lines.append(
                    f"    {model_type} & {dataset} & {pert_display} & -- & -- & {delta_f1:+.4f} & {delta_iou:+.4f} \\\\"
                )
    
    footer = (
        "\n    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    
    return header + "\n".join(body_lines) + footer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('results'))
    ap.add_argument('--pattern', type=str, default='robustness_*.json', help='Glob pattern for result files')
    ap.add_argument('--out-dir', type=Path, default=Path('docs/tables'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    
    # Find all robustness result files
    pattern = str(args.results_dir / args.pattern)
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"Warning: No result files found matching {pattern}")
        return
    
    results_list = []
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            results_list.append(json.load(f))
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "robustness_table.tex"
    
    table = make_robustness_table(results_list, args.threshold)
    out_file.write_text(table)
    print(f"Wrote: {out_file}")
    print(f"Processed {len(results_list)} result file(s)")


if __name__ == '__main__':
    main()

