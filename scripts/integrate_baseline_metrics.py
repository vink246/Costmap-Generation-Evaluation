"""Integrate classical baseline metrics into perception_baselines.json.

Reads baseline_depth_to_cost_summary.json and adds entries to perception_baselines.json
in the format expected by make_latex_tables.py.
"""

import json
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-summary', type=Path, default=Path('results/baseline_depth_to_cost_summary.json'))
    ap.add_argument('--perception-baselines', type=Path, default=Path('results/perception_baselines.json'))
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    # Load baseline results
    with open(args.baseline_summary, 'r') as f:
        baseline_data = json.load(f)
    
    # Load perception baselines
    with open(args.perception_baselines, 'r') as f:
        perception_data = json.load(f)
    
    # Add baseline entries to each dataset
    baseline_runs = baseline_data.get('runs', [])
    
    for run in baseline_runs:
        dataset = run['dataset']
        if dataset not in perception_data['datasets']:
            perception_data['datasets'][dataset] = []
        
        # Create baseline entry
        baseline_entry = {
            'method': 'Classical Baseline',
            'variant': 'heuristic',
            'mae': run['mae'],
            'iou': run['iou'],
            'precision': run['precision'],
            'recall': run['recall'],
            'f1': run['f1'],
            'params_m': None  # Classical baseline has no model params
        }
        
        # Check if baseline already exists and remove it
        perception_data['datasets'][dataset] = [
            r for r in perception_data['datasets'][dataset]
            if not (r.get('method') == 'Classical Baseline' and r.get('variant') == 'heuristic')
        ]
        
        # Insert at the beginning (baseline should be first)
        perception_data['datasets'][dataset].insert(0, baseline_entry)
    
    # Ensure threshold is set
    perception_data['threshold'] = args.threshold
    
    # Save updated perception baselines
    with open(args.perception_baselines, 'w') as f:
        json.dump(perception_data, f, indent=2)
    
    print(f"Updated {args.perception_baselines} with classical baseline metrics")


if __name__ == '__main__':
    main()

