import csv
from pathlib import Path


def format_table(rows, dataset: str, source: str, tag_label: str = None):
    caption_extra = f", {tag_label}" if tag_label else ""
    label_extra = f"_{tag_label}" if tag_label else ""
    header = (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{{dataset.upper()} planner-in-the-loop results across thresholds (both planners, source={source}{caption_extra}).}}\n"
        f"  \\label{{tab:{dataset}_planner_{source}{label_extra}}}\n"
        "  \\begin{tabular}{lcccccc}\n"
        "    \\toprule\n"
        "    thr & succ(A*) & succ(RRT*) & t(A*) [ms] & t(RRT*) [ms] & len(A*) & len(RRT*) \\\\ \n"
        "    \\midrule\n"
    )
    body = []
    for r in rows:
        def fnum(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)
        line = (
            f"    {r['threshold']} & {fnum(r.get('success_rate_astar', 0.0)):.3f} & {fnum(r.get('success_rate_rrtstar', 0.0)):.3f} "
            f"& {fnum(r.get('mean_time_ms_astar', 0.0)):.3f} & {fnum(r.get('mean_time_ms_rrtstar', 0.0)):.3f} "
            f"& {fnum(r.get('mean_path_len_cells_astar', 0.0)):.3f} & {fnum(r.get('mean_path_len_cells_rrtstar', 0.0)):.3f} \\\\" )
        body.append(line)
    footer = ("\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    return header + "\n".join(body) + footer


def main():
    root = Path(__file__).resolve().parents[1]
    sweeps_csv = root / 'results' / 'planner_sweeps.csv'
    out_dir = root / 'docs' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tex = out_dir / 'planner_tables.tex'

    with open(sweeps_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # filter for val + both + labels (not predictions) for baseline tables
    nyu_rows_labels = [r for r in rows if r['dataset']=='nyu' and r['split']=='val' and r['planner']=='both' and r.get('source')=='labels']
    nyu_rows_pred_all = [r for r in rows if r['dataset']=='nyu' and r['split']=='val' and r['planner']=='both' and r.get('source')=='pred']
    kitti_rows_labels = [r for r in rows if r['dataset']=='kitti' and r['split']=='val' and r['planner']=='both' and r.get('source')=='labels']
    kitti_rows_pred_all = [r for r in rows if r['dataset']=='kitti' and r['split']=='val' and r['planner']=='both' and r.get('source')=='pred']

    tex = []
    if nyu_rows_labels:
        nyu_sorted = sorted(nyu_rows_labels, key=lambda r: float(r['threshold']))
        tex.append(format_table(nyu_sorted, 'nyu', 'labels'))
    # produce a table per run_tag for predictions
    if nyu_rows_pred_all:
        tags = sorted({(r.get('run_tag') or 'pred') for r in nyu_rows_pred_all})
        for tg in tags:
            rows_t = [r for r in nyu_rows_pred_all if (r.get('run_tag') or 'pred') == tg]
            if rows_t:
                rows_sorted = sorted(rows_t, key=lambda r: float(r['threshold']))
                tex.append("\n% ---\n")
                tex.append(format_table(rows_sorted, 'nyu', 'pred', tg))
    if kitti_rows_labels:
        kitti_sorted = sorted(kitti_rows_labels, key=lambda r: float(r['threshold']))
        tex.append("\n% ---\n")
        tex.append(format_table(kitti_sorted, 'kitti', 'labels'))
    if kitti_rows_pred_all:
        tags = sorted({(r.get('run_tag') or 'pred') for r in kitti_rows_pred_all})
        for tg in tags:
            rows_t = [r for r in kitti_rows_pred_all if (r.get('run_tag') or 'pred') == tg]
            if rows_t:
                rows_sorted = sorted(rows_t, key=lambda r: float(r['threshold']))
                tex.append("\n% ---\n")
                tex.append(format_table(rows_sorted, 'kitti', 'pred', tg))

    out_tex.write_text("\n\n".join(tex))
    print('Wrote:', out_tex)


if __name__ == '__main__':
    main()
