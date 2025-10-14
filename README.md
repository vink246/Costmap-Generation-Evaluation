
# Image-to-Costmap Data Pipeline

This repo contains a data pipeline to convert RGB+D datasets (KITTI raw and NYU Depth v2) into supervised training pairs for predicting local egocentric costmaps.

## Data Setup Rationale

### Why this structure?
- **Unified supervised pairs:** Both datasets are converted to a common format: normalized RGB+D image, costmap label, and metadata. This enables direct benchmarking of architectures and fair comparison across domains.
- **Train/val splits only:** We use 80% of data for training and 20% for validation. No test set is used, as all final evaluation is done on the validation split after tuning. This avoids overfitting to a small test set and maximizes training data.
- **NYU Depth v2 split:** The Kaggle mirror divides data into `nyu2_train` (small) and `nyu2_test` (large). We combine both and create our own 80/20 split for train/val, ensuring the largest possible training set and a representative validation set.
- **KITTI split:** Drives are split into train/val according to `configs/data.yaml`. Each drive contains synchronized RGB and LiDAR/depth data.

### Output structure
- `data/processed/<dataset>/<split>/*.npz` where each file contains:
  - `image`: (256, 256, 4) float32 RGB+D normalized
  - `costmap`: (64, 64) float32 in [0, 1]
  - `meta`: dict with frame_id, calibration, roi, grid_resolution

## How the pipeline works

1. **Create a Python venv and install dependencies:**
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
  ```

2. **Configure dataset paths in `configs/data.yaml`:**
  - Set the correct root folders for KITTI and NYU raw data.

3. **Prepare NYU Depth v2:**
  - Run the prep script to combine all samples from `nyu2_train` and `nyu2_test`, then split into `subset_train` (80%) and `subset_val` (20%):
  ```powershell
  python scripts/prepare_nyu_from_kaggle.py --root data/raw/nyu_depth_v2 --source nyu_data/data
  ```

4. **Build pairs for KITTI:**
  ```powershell
  python scripts/build_pairs.py --dataset kitti --config configs/data.yaml --out data/processed
  ```

5. **Build pairs for NYU Depth v2:**
  ```powershell
  python scripts/build_pairs.py --dataset nyu --config configs/data.yaml --out data/processed
  ```

6. **Train a baseline (U-Net):**
  ```powershell
  # NYU
  python -m src.train.train --config configs/train_nyu_unet.yaml
  # KITTI
  python -m src.train.train --config configs/train_kitti_unet.yaml
  ```

7. **Evaluate a checkpoint on val:**
  ```powershell
  # Adjust checkpoint path if needed (e.g., runs/nyu_unet/unet_best.pth)
  python -m src.eval.evaluate_model --checkpoint runs/nyu_unet/unet_best.pth --dataset nyu --data_root data/processed
  ```

8. **Swap in your own architectures:**
  - Implement your model in a module like `src/models/<your_model>.py`
  - Set in the config:
    - `model_module: src.models.<your_model>`
    - `model_class: <YourClassName>`
    - `model_kwargs: { ... }` as needed

Notes for teammates: The model files are placeholders:
  - `src/models/unet.py` -> class `UNet`
  - `src/models/vit.py` -> class `ViT`
  - `src/models/hybrid.py` -> class `HybridCNNTransformer`
Fill in the actual layers/forward method and adjust `model_kwargs` in the config accordingly.

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

## Sharing data with the team
Would rather not commit large datasets, so we will use these helper scripts:

- (Rut) Package the local data into zips for sharing:
  ```powershell
  .\scripts\pack_data.ps1 -OutDir artifacts
  # Upload artifacts/kitti_raw.zip, artifacts/nyu_raw.zip, artifacts/processed.zip to OneDrive/Drive/S3
  ```

- (Teammates) Download/extract with:
  ```powershell
  .\scripts\fetch_data.ps1 -KittiZipUrl "<link-to-kitti_raw.zip>" -NyuZipUrl "<link-to-nyu_raw.zip>" -ProcessedZipUrl "<link-to-processed.zip>"
  ```

## Dataset details

- **KITTI:**
  - Raw drives under `data/raw/kitti/<drive>`
  - Each drive contains RGB images and LiDAR/depth data
  - Calibration files used for projection
  - Preprocessing projects depth/LiDAR to BEV costmap, normalizes images, and saves pairs

- **NYU Depth v2:**
  - Kaggle mirror provides `nyu2_train` and `nyu2_test` folders
  - Prep script combines both, then splits into `subset_train` and `subset_val` for training/validation
  - Each sample folder contains `rgb/<id>.png` and `depth/<id>.png`

## Proposal-aligned settings (might change based on instructor/TAs' feedback)
- ROI: 10m x 10m local egocentric frame
- Image size: 256 x 256
- Costmap resolution: 64 x 64
- Dilation encoding robot footprint + safety margin
- Metrics produced later (training): MAE, IoU, Precision/Recall

## Notes
- KITTI requires projecting LiDAR/depth to BEV; NYU uses dense depth. Calibrations from dataset folders.
- Planner evaluations will be done using generated costmaps.
