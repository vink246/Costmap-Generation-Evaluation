import os
import argparse
import yaml

"""
Scan data/raw/kitti for available drives and generate train/val splits.
Split: 80/20 (train/val)
"""


def get_drives(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"KITTI root not found: {root}")
    drives = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.endswith('_sync'):
            # require image_02/data to exist
            if os.path.isdir(os.path.join(p, 'image_02', 'data')):
                drives.append(name)
    return drives


def split_list(items, train_ratio=0.8, val_ratio=0.2):
    n = len(items)
    n_train = max(1, int(n * train_ratio))
    n_val = n - n_train
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/raw/kitti', help='KITTI raw root folder')
    ap.add_argument('--out', default='configs/data.yaml', help='YAML file to write updated config/splits to')
    ap.add_argument('--base', default='configs/data.yaml', help='Existing base config to merge into (preserves roi/image/costmap/nyu)')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    args = ap.parse_args()

    drives = get_drives(args.root)
    if not drives:
        print('No KITTI drives found under', args.root)
        return
    train, val = split_list(drives, args.train_ratio, args.val_ratio)

    # Load base config (preserve other keys)
    base_cfg = {}
    if os.path.exists(args.base):
        with open(args.base, 'r') as f:
            base_cfg = yaml.safe_load(f) or {}

    # Update KITTI section
    base_cfg.setdefault('kitti', {})
    base_cfg['kitti']['root'] = args.root
    base_cfg['kitti']['splits'] = {
        'train': train,
        'val': val,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)
    print('Updated config written to', args.out)
    print('Counts:', {'train': len(train), 'val': len(val)})


if __name__ == '__main__':
    main()
