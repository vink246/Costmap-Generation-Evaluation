import json
from pathlib import Path


def load_baselines(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def table_for_dataset(name: str, rows: list, threshold: float) -> str:
    header = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{{name.upper()} validation metrics. Threshold $\\tau={{}}{threshold}$.}}\n"
        f"  \\label{{tab:{name}_results}}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    Method & MAE $\\downarrow$ & IoU $\\uparrow$ & Precision & Recall & F1 & Params (M) \\\\ \n"
        "    \\midrule\n"
    )
    body_lines = []
    for r in rows:
        params = ("--" if r.get("params_m") in (None, "",) else f"{r['params_m']}")
        line = (
            f"    {r['method']}"
            + (" (NYU $\\rightarrow$ KITTI TL)" if r.get("variant") == "nyu_to_kitti_tl" and name == "kitti" else
               " (KITTI only)" if r.get("variant") == "kitti_only" and name == "kitti" else "")
            + f" & {r['mae']:.4f} & {r['iou']:.4f} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} & {params} \\\\"
        )
        body_lines.append(line)
    footer = (
        "\n    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    return header + "\n".join(body_lines) + footer


def main():
    root = Path(__file__).resolve().parents[1]
    results_path = root / "results" / "perception_baselines.json"
    out_dir = root / "docs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "perception_tables.tex"

    data = load_baselines(results_path)
    threshold = data.get("threshold", 0.5)
    nyu_rows = data["datasets"].get("nyu", [])
    kitti_rows = data["datasets"].get("kitti", [])

    tex = []
    tex.append(table_for_dataset("nyu", nyu_rows, threshold))
    tex.append("\n% ---\n")
    tex.append(table_for_dataset("kitti", kitti_rows, threshold))

    out_file.write_text("\n\n".join(tex))
    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
