import argparse
import os
import sys

# Ensure repository root is on sys.path so 'src' can be imported when running this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import kitti as kitti_mod
from src.data import nyu as nyu_mod

"""
CLI to build supervised pairs according to our proposal. Produces .npz files with:
- image: (H,W,4) normalized RGB+D
- costmap: (64,64) float32 [0,1]
- meta: dict
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.dataset == 'kitti':
        kitti_mod.build_pairs(args.config, args.out)
    else:
        nyu_mod.build_pairs(args.config, args.out)


if __name__ == '__main__':
    main()
