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

### Planner evaluation (next step)
Planner-based metrics (A*/RRT*) are not included yet. Suggested structure when adding:
- `src/planning/astar.py`, `src/planning/rrt_star.py`
- `src/eval/evaluate_planner.py` to run planners on predicted costmaps and report success, collisions, path length, and planning time.

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
