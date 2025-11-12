# Costmap-Generation-Evaluation

Class project for Georgia Tech's CS7643: Evaluate models to directly predict costmaps from RGB(-D) images.

---

## Image-to-Costmap Data Pipeline

This repo contains a data pipeline to convert RGB+D datasets (KITTI raw and NYU Depth v2) into supervised training pairs for predicting local egocentric costmaps.

### Data Setup Rationale

Why this structure?
- Unified supervised pairs: Both datasets are converted to a common format: normalized RGB+D image, costmap label, and metadata. This enables direct benchmarking of architectures and fair comparison across domains.
- Train/val splits only: We use 80% of data for training and 20% for validation. No test set is used, as all final evaluation is done on the validation split after tuning.
- NYU Depth v2 split: The Kaggle mirror divides data into `nyu2_train` (small) and `nyu2_test` (large). We combine both and create our own 80/20 split for train/val.
- KITTI split: Drives are split into train/val according to `configs/data.yaml`. Each drive contains synchronized RGB and LiDAR/depth data.

Output structure
- `data/processed/<dataset>/<split>/*.npz` where each file contains:
  - `image`: (256, 256, 4) float32 RGB+D normalized
  - `costmap`: (64, 64) float32 in [0, 1]
  - `meta`: dict with frame_id, calibration, roi, grid_resolution

### How the pipeline works

1) Create a Python venv and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2) Configure dataset paths in `configs/data.yaml`.

3) Prepare NYU Depth v2:

```powershell
python scripts/prepare_nyu_from_kaggle.py --root data/raw/nyu_depth_v2 --source nyu_data/data
```

4) Build pairs for KITTI:

```powershell
python scripts/build_pairs.py --dataset kitti --config configs/data.yaml --out data/processed
```

5) Build pairs for NYU Depth v2:

```powershell
python scripts/build_pairs.py --dataset nyu --config configs/data.yaml --out data/processed
```

6) Train a baseline (U-Net):

```powershell
# NYU
python -m src.train.train --config configs/train_nyu_unet.yaml
# KITTI
python -m src.train.train --config configs/train_kitti_unet.yaml
```

7) Evaluate a checkpoint on val:

```powershell
# Adjust checkpoint path if needed (e.g., runs/nyu_unet/unet_best.pth)
python -m src.eval.evaluate_model --checkpoint runs/nyu_unet/unet_best.pth --dataset nyu --data_root data/processed
```

8) Swap in your own architectures:
- Implement your model in `src/models/<your_model>.py`
- Set in the config:
  - `model_module: src.models.<your_model>`
  - `model_class: <YourClassName>`
  - `model_kwargs: { ... }`

Notes for teammates: The model files are placeholders:
- `src/models/unet.py` -> class `UNet`
- `src/models/vit.py` -> class `ViT`
- `src/models/hybrid.py` -> class `HybridCNNTransformer`

### Train other models (examples)

```powershell
# ViT variant (after implementing src/models/vit.py:ViT)
python -m src.train.train --config configs/train_nyu_vit.yaml

# Hybrid CNN+Transformer (after implementing src/models/hybrid.py:HybridCNNTransformer)
python -m src.train.train --config configs/train_nyu_hybrid.yaml
```

### Troubleshooting
- Torch or cv2 “could not be resolved” in editor: ensure your venv is active and `pip install -r requirements.txt` has run.
- GPU support: install a CUDA-enabled torch build matching your system; CPU works but is slower.
- No `.npz` found: confirm you’ve run the build scripts and that configs point to the correct roots.

### Orchestration entrypoint (automation)
Use the lightweight orchestration script to reproduce planner metrics end-to-end without remembering all flags:

```powershell
# Labels (ground-truth) or predictions via --pred-dir
python scripts/run_all.py --stage planner --dataset nyu --split val --planner both --threshold 0.5 --inflation-radius 2 --num-scenes 50 --seed 42
python scripts/run_all.py --stage planner --dataset kitti --split val --planner both --threshold 0.5 --inflation-radius 2 --num-scenes 100 --seed 42 --pred-dir predictions\kitti\unet_rgbd_retrain_full\val
```

This wraps `scripts/planner_eval.py` and writes per-scene CSVs and an aggregate JSON under `outputs/planner_eval/` with provenance metadata (UTC timestamp, git hash if available, seed, run tag/source).

### Current Planner & Results Pipeline (Implemented)
The milestone work added light-weight planners and evaluation scripts:
- `scripts/planner_a_star.py`, `scripts/planner_rrt_star.py` – grid A* and RRT* with obstacle inflation, goal bias, rewiring, path cost accumulation.
- `scripts/planner_eval.py` – runs A*, RRT*, or both across processed label costmaps or predicted costmaps (`--pred-dir`). Saves per-scene CSV and an aggregate JSON.
- `scripts/aggregate_planner_summaries.py` – consolidates all `summary_*.json` into `results/planner_sweeps.csv`.
- `scripts/make_latex_planner_tables.py` – renders LaTeX tables from the consolidated CSV.

Output examples (after running sweeps):
- `outputs/planner_eval/summary_kitti_val_both_thr0.5_infl2_labels_n433.json`
- `outputs/planner_eval/summary_nyu_val_both_thr0.6_infl2_pred_n131.json`
- Aggregated: `results/planner_sweeps.csv`
- LaTeX: `docs/tables/planner_tables.tex`

Each summary JSON now includes reproducibility metadata (`meta` block) with:
```json
{
  "meta": {
    "created_at": "2025-11-09T12:34:56.123Z",
    "git_commit": "abc1234",
    "source": "labels" | "pred",
    "seed": 42
  },
  "args": { ... },
  "aggregate": { ... },
  "per_scene": [ ... ]
}
```

To regenerate planner tables end-to-end:
```powershell
# Labels (ground-truth) sweeps for NYU & KITTI val
python scripts/planner_eval.py --dataset nyu --split val --planner both --threshold 0.4 --inflation-radius 2 --num-scenes 131
python scripts/planner_eval.py --dataset nyu --split val --planner both --threshold 0.5 --inflation-radius 2 --num-scenes 131
python scripts/planner_eval.py --dataset nyu --split val --planner both --threshold 0.6 --inflation-radius 2 --num-scenes 131
python scripts/planner_eval.py --dataset kitti --split val --planner both --threshold 0.4 --inflation-radius 2 --num-scenes 433
python scripts/planner_eval.py --dataset kitti --split val --planner both --threshold 0.5 --inflation-radius 2 --num-scenes 433
python scripts/planner_eval.py --dataset kitti --split val --planner both --threshold 0.6 --inflation-radius 2 --num-scenes 433

# Aggregate & render LaTeX
python scripts/aggregate_planner_summaries.py
python scripts/make_latex_planner_tables.py
```

For predicted costmaps, add `--pred-dir <path_to_predictions>` (same stem names as processed). Filenames automatically include `pred` vs `labels` tag to avoid collisions.

## Reproducibility Metadata & Table Regeneration

We embed minimal provenance in generated artifacts:
- Git commit: short hash at generation time (if repository available).
- UTC timestamp (ISO 8601, suffixed with `Z`).
- Seed (for planner random sampling and any stochastic components).
- Source tag (`labels` or `pred`).

Scripts instrumented with metadata:
- `scripts/planner_eval.py`
- `scripts/baseline_depth_costmaps.py`
- `scripts/pr_curves.py`

LaTeX tables can be regenerated deterministically from structured CSV/JSON:
```powershell
# Perception baselines (from milestone report metrics)
python scripts/make_latex_tables.py
# Planner tables (after sweeps)
python scripts/make_latex_planner_tables.py
```

PR curves (baseline or learned models):
```powershell
# After generating predictions into predictions/<dataset>/<model_tag>/<split>/
python scripts/pr_curves.py --dataset nyu --split val --processed-root data\processed --pred-dir predictions\nyu\val --out results\pr_curves
python scripts/pr_curves.py --dataset kitti --split val --processed-root data\processed --pred-dir predictions\kitti\val --out results\pr_curves
```

Prediction generation for learned models (given a checkpoint):
```powershell
python scripts/infer_and_save.py --checkpoint runs/nyu_unet/unet_best.pth --dataset nyu --split val --model-module src.models.unet --model-class UNet --model-tag unet
python scripts/infer_and_save.py --checkpoint runs/kitti_vit/vit_best.pth --dataset kitti --split val --model-module src.models.vit --model-class ViT --model-tag vit
```

Then run planner eval on predicted maps:
```powershell
python scripts/planner_eval.py --dataset nyu --split val --planner both --threshold 0.5 --inflation-radius 2 --num-scenes 131 --pred-dir predictions\nyu\unet\val
```

Minimal Config Fingerprint (optional extension): hash your YAML config to include in metadata:
```powershell
(Get-FileHash configs\train_nyu_unet.yaml -Algorithm SHA256).Hash
```
This can be manually appended to JSON metadata or integrated into scripts.

Recommended: keep a single regeneration script (e.g. `scripts/run_all.py`) that calls the sequence above to produce fresh artifacts from processed data and checkpoints.

---

## Pinned requirements (exact freeze)

For deterministic environments, capture an exact lockfile of your current setup and reuse it on fresh machines.

Create a lockfile from an activated, working environment:

```powershell
# From the repo root, with your venv/conda env activated
pip freeze > requirements.lock.txt
```

To recreate the exact environment later:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

Notes:
- Torch/CUDA builds vary by system. If GPU is needed, install the CUDA-matched torch wheels first, then run `pip freeze` again to update the lockfile.
- We also include a lightweight, unpinned `requirements.txt` for general development; prefer the lockfile for paper results.

---

## Environment setup (PACE-ICE)

If you're using Georgia Tech's PACE-ICE cluster, here's a quick conda flow:

1) Load Anaconda module

```bash
module load anaconda3
```

2) Create a Conda environment in scratch

```bash
conda create --prefix /scratch/<path_to_env_parent_dir>/costmap_env python=3.12 -y
```

3) Activate

```bash
conda activate /scratch/<path_to_env_parent_dir>/costmap_env
```

4) Install packages

```bash
conda install pytorch torchvision torchaudio -c pytorch -y
conda install numpy matplotlib opencv scipy scikit-learn tqdm -y
conda install jupyterlab -y
```

5) Verify

```bash
python -m pip list
python -c "import torch; print(torch.__version__)"
```

6) Export env (optional)

```bash
conda env export > /scratch/<path_to_env_parent_dir>/costmap_env.yml
conda env create --prefix /scratch/<path_to_env_parent_dir>/costmap_env --file costmap_env.yml
```

---

## Generating Ground-Truth Costmaps from NYU Depth v2

Goal: Convert RGB+D frames into 64×64 continuous costmaps in [0,1].

1) Load and align data: RGB `(H×W×3)`, depth `(H×W)`, intrinsics `(fx, fy, cx, cy)`.

2) Back-project depth to 3D:

```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

3) Project to BEV: define an egocentric grid (e.g., 6.4×6.4 m), bin points by (x,z).

4) Traversability cost: use height stats; e.g., `sigmoid(a*(max_h-h_t)) + b*std_h`, normalize to [0,1].

5) Post-process: dilate by robot footprint; optional Gaussian blur.

6) Resize to 64×64.

Output sample format:

```
{ rgb: H×W×4 (4th=depth), costmap: 64×64 }
```

---

## Sharing data with the team

We do not commit large datasets. Use helper scripts:

- Package local data into zips:

```powershell
./scripts/pack_data.ps1 -OutDir artifacts
```

- Download/extract:

```powershell
./scripts/fetch_data.ps1 -KittiZipUrl "<kitti_raw.zip>" -NyuZipUrl "<nyu_raw.zip>" -ProcessedZipUrl "<processed.zip>"
```

---

## Dataset details

- KITTI: raw drives under `data/raw/kitti/<drive>` with RGB + LiDAR; preprocessing projects to BEV costmaps.
- NYU Depth v2: Kaggle mirror provides `nyu2_train` and `nyu2_test`; prep script combines and splits into `subset_train`/`subset_val`.

## Proposal-aligned settings

- ROI: 10m × 10m
- Image: 256 × 256
- Costmap: 64 × 64
- Metrics: MAE, IoU, Precision/Recall

## Notes

- KITTI requires LiDAR-to-BEV projection; NYU uses dense depth.
- Planner evaluations will be done using generated costmaps.
