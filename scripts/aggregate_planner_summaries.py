import json
import csv
from pathlib import Path


def extract_tag_fields(name: str):
    """Parse filename components into structured fields.

    Expected pattern after stripping prefix/suffix:
      {dataset}_{split}_{planner}_thr{T}_infl{R}_{source}_n{N}
    Where source in {labels, pred}. Older files may omit the source token.
    """
    parts = name.replace("summary_", "").replace(".json", "").split("_")
    fields = {
        "dataset": None,
        "split": None,
        "planner": None,
        "threshold": None,
        "inflation": None,
        "source": None,
        "n": None,
    }
    if len(parts) >= 6:
        fields["dataset"], fields["split"], fields["planner"] = parts[0], parts[1], parts[2]
        # the remaining parts are unordered tokens we classify
        for p in parts[3:]:
            if p.startswith("thr"):
                try:
                    fields["threshold"] = float(p[3:])
                except ValueError:
                    pass
            elif p.startswith("infl"):
                try:
                    fields["inflation"] = int(p[4:])
                except ValueError:
                    pass
            elif p in ("labels", "pred"):
                fields["source"] = p
            elif p.startswith("n") and p[1:].isdigit():
                fields["n"] = int(p[1:])
    # default source if missing (older runs were labels-only)
    if fields["source"] is None:
        fields["source"] = "labels"
    return fields


def main():
    root = Path(__file__).resolve().parents[1]
    in_dir = root / "outputs" / "planner_eval"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "planner_sweeps.csv"

    rows = []
    for p in in_dir.glob("summary_*.json"):
        fields = extract_tag_fields(p.name)
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        agg = obj.get("aggregate", {})
        args_obj = obj.get("args", {})
        base = {
            **fields,
            "num_scenes": agg.get("num_scenes"),
            "elapsed_ms_total": agg.get("elapsed_ms_total"),
            # provenance columns
            "pred_dir": args_obj.get("pred_dir"),
            "run_tag": args_obj.get("run_tag"),
        }
        if fields.get("planner") == "both":
            base.update({
                "success_rate_astar": agg.get("success_rate_astar"),
                "success_rate_rrtstar": agg.get("success_rate_rrtstar"),
                "mean_time_ms_astar": agg.get("mean_time_ms_astar"),
                "mean_time_ms_rrtstar": agg.get("mean_time_ms_rrtstar"),
                "mean_path_len_cells_astar": agg.get("mean_path_len_cells_astar"),
                "mean_path_len_cells_rrtstar": agg.get("mean_path_len_cells_rrtstar"),
                "mean_cost_sum_astar": agg.get("mean_cost_sum_astar"),
            })
        else:
            base.update({
                "success_rate": agg.get("success_rate"),
                "mean_time_ms": agg.get("mean_time_ms"),
                "mean_path_len_cells": agg.get("mean_path_len_cells"),
                "mean_cost_sum": agg.get("mean_cost_sum"),
            })
        rows.append(base)

    # write CSV
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("Wrote:", out_csv)
    else:
        print("No summaries found in", in_dir)


if __name__ == "__main__":
    main()
