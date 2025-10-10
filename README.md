# Costmap-Generation-Evaluation
Class project for Georgia Tech's CS7643: Evaluate models to directly predict costmaps from RGB(-D) images.

## Environment setup

### Setting up a Conda Environment in Scratch (PACE-ICE)

#### 1. Load the Anaconda Module
PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
> ```bash
> module avail anaconda
> ```

#### 2. Create a Conda Environment in Scratch

By default, `conda` environments go to your home directory (`~/.conda/envs`),  
which has limited quota. To avoid filling it up, create a custom environment directory in scratch:

```bash
# Create the environment explicitly in scratch
conda create --prefix /scratch/<path_to_env_parent_dir>/costmap_env python=3.12 -y
```

#### 3. Activate the Environment

```bash
conda activate /scratch/<path_to_env_parent_dir>/costmap_env
```

> You **must use the full path** when activating environments created outside your home directory.

#### 4. Install Required Packages

Once activated, install any packages you need, such as PyTorch, OpenCV, etc.

```bash
# Example for PyTorch + utilities
conda install pytorch torchvision torchaudio -c pytorch -y
conda install numpy matplotlib opencv scipy scikit-learn tqdm -y
conda install jupyterlab -y
```

#### 5. Verify Installation

```bash
python -m pip list
python -c "import torch; print(torch.__version__)"
```

#### 6. (Optional) Export and Reuse the Environment

You can save your environment spec for reproducibility:

```bash
conda env export > /scratch/<path_to_env_parent_dir>/costmap_env.yml
```

and recreate it later with:

```bash
conda env create --prefix /scratch/<path_to_env_parent_dir>/costmap_env --file costmap_env.yml
```

#### 7. Deactivate Environment

```bash
conda deactivate
```

## Generating Ground-Truth Costmaps from NYU Depth v2

### Goal
Convert RGB+D frames into 64×64 continuous **costmaps** where:
- `0` = free / traversable
- `1` = untraversable / obstacle

These costmaps will serve as supervised labels for training the costmap predictor models.

### Steps Overview

#### 1. Load and Align Data
Use NYU Depth v2 RGB and Depth frames (labeled or unlabeled).  
Ensure RGB and depth are spatially aligned and normalized.

**Input:**  
- RGB image `(H×W×3)`  
- Depth map `(H×W)`  
- Camera intrinsics `(fx, fy, cx, cy)`

#### 2. Back-Project Depth to 3D Points
Convert each pixel `(u, v, z)` to camera-space coordinates `(x, y, z)` using intrinsics:

```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

This produces a dense **3D point cloud** for the scene.

#### 3. Project to Bird’s-Eye View (BEV)
Define an egocentric grid (e.g., 6.4×6.4 m) centered on the camera.  
Each grid cell represents an area in world space (e.g., 10 cm × 10 cm).

Bin 3D points into grid cells based on their `(x, z)` coordinates.


#### 4. Compute Traversability Cost
For each BEV cell, compute geometric statistics:
- `mean_height`, `max_height`, `height_std`
- Penalize cells with tall or uneven surfaces.

Example cost formulation:
```
cost = sigmoid(a * (max_height - h_thresh)) + b * height_std
```
Normalize cost to `[0, 1]`.


#### 5. Smooth and Inflate
Post-process the grid:
- **Dilation:** Inflate obstacles by the robot’s footprint.
- **Gaussian blur:** Smooth transitions between free and occupied space.

This yields a **continuous traversability field** instead of binary occupancy.

#### 6. Resize to 64×64
Downsample or interpolate the BEV grid to a fixed 64×64 resolution.

#### Output
A costmap:
- Shape: `64×64`
- Values in `[0, 1]`
- Aligned with the camera’s egocentric frame


#### Usage
Each training sample is stored as:
```
{ rgb: H×W×4 (4th for depth), costmap: 64×64 }
```

This dataset can then be used for supervised training of:
```
f(RGB + Depth) → Costmap
```
