"""
ViT Cross-Domain Transfer Learning Script
Supports:
    - zero-shot evaluation
    - fine-tuning
    - freeze+warmup fine-tuning

Works with models defined in: src.models.vit.ViT
"""

import argparse
import os
import sys
import importlib
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.train.train import (
    load_config, get_device, build_dataloaders,
    build_losses, compute_loss, evaluate
)


#  Load pretrained ViT
def load_pretrained_model(checkpoint_path, ModelClass, model_kwargs, device):
    print(f"Loading pretrained model from: {checkpoint_path}")
    # Try loading normally; if strict loading fails, fall back to non-strict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ModelClass(**model_kwargs).to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as e:
        print(f"[warn] strict load failed: {e} — retrying with strict=False")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


#  Freeze ViT encoder layers for warmup fine-tuning
def freeze_vit_encoder(model):
    for name, param in model.named_parameters():
        if any(k in name for k in ["patch_embed", "pos_embed", "blocks"]):
            param.requires_grad = False


def unfreeze_all(model):
    for _, param in model.named_parameters():
        param.requires_grad = True


#  Training loop
def run_finetune(model, cfg, train_dl, val_dl, device, freeze_epochs=0):
    epochs = cfg.get("epochs", 50)
    lr = cfg.get("lr", 2e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = None
    if cfg.get("lr_scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # build_losses returns: weights, L1Loss(), DiceLoss(), BoundaryLoss(), BCEWithLogits()
    weights, l1_fn, dice_fn, bnd_fn, bce_fn = build_losses(cfg)

    best_f1 = -1
    out_dir = cfg.get("out_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):

        # Freeze encoder for warmup
        if epoch <= freeze_epochs:
            freeze_vit_encoder(model)
            print(f"[Epoch {epoch}] Encoder frozen")
        else:
            unfreeze_all(model)
            if epoch == freeze_epochs + 1:
                print(f"[Epoch {epoch}] Encoder unfrozen")

        model.train()
        train_loss = 0.0
        steps = 0

        for img, cm in train_dl:
            img, cm = img.to(device), cm.to(device)

            pred_full = model(img)

            # Resize if needed
            if pred_full.shape[-2:] != cm.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred_full,
                    size=cm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                pred = pred_full

            loss, logs = compute_loss(pred, cm, weights, l1_fn, dice_fn, bnd_fn, bce_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += logs.get("total", 0.0)
            steps += 1

        # Evaluation
        metrics = evaluate(model, val_dl, device)
        avg_train_loss = train_loss / steps if steps > 0 else 0.0
        print(f"[Epoch {epoch}] TrainLoss {avg_train_loss:.4f} | Val: {metrics}")

        if scheduler:
            scheduler.step()

        # Save best model by F1
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_path = os.path.join(out_dir, f"vit_transfer_best.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                save_path,
            )
            print(f"Saved best checkpoint → {save_path}")

    return best_f1


#  Zero-shot evaluation
def run_zero_shot(model, val_dl, device):
    metrics = evaluate(model, val_dl, device)
    print("\n===== ZERO-SHOT METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


#  Main entry
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--config", required=True,
                        help="Target dataset config (NYU or KITTI)")
    parser.add_argument("--mode", choices=["zero-shot", "finetune", "freeze-warmup"],
                        required=True)
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup epochs for freeze-warmup mode")

    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    # Read dataset / loader params from config
    dataset_name = cfg.get("dataset", "kitti")
    batch_size = cfg.get("batch_size", 16)
    num_workers = cfg.get("num_workers", 4)
    channels = cfg.get("channels", "rgbd")

    # Load model class
    model_module = importlib.import_module(cfg["model_module"])
    ModelClass = getattr(model_module, cfg["model_class"])
    model_kwargs = cfg.get("model_kwargs", {"in_channels": 4, "out_channels": 1})

    # Load pretrained weights
    model = load_pretrained_model(args.checkpoint, ModelClass, model_kwargs, device)

    # Build dataloaders (always safe to build both; zero-shot will only use val_dl)
    train_dl, val_dl = build_dataloaders(
        cfg["data_root"],
        dataset_name,
        batch_size,
        num_workers,
        channels,
    )

    # Modes
    if args.mode == "zero-shot":
        # only evaluate on val set
        run_zero_shot(model, val_dl, device)
        return

    if args.mode == "finetune":
        run_finetune(model, cfg, train_dl, val_dl, device, freeze_epochs=0)
        return

    if args.mode == "freeze-warmup":
        run_finetune(model, cfg, train_dl, val_dl, device, freeze_epochs=args.warmup)
        return


if __name__ == "__main__":
    main()
