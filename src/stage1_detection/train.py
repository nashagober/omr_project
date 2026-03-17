"""
Stage 1 Training Script — Faster R-CNN on MUSCIMA++

Usage:
    python -m src.stage1_detection.train \
        --data_dir       data/raw/muscima \
        --output_dir     outputs/stage1 \
        --local_ckpt_dir /content/local_checkpoints \
        --epochs         40 \
        --batch_size     4 \
        --lr             0.0005

--local_ckpt_dir : per-epoch checkpoints saved here (use Colab local storage)
--output_dir     : best.pt, metrics.json, and plots saved here (use Drive)
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from src.stage1_detection.dataset import MUSCIMADataset, collate_fn
from src.stage1_detection.detector import build_faster_rcnn
from src.stage1_detection.evaluate import evaluate
from src.stage1_detection.visualize import plot_all


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: FasterRCNN, optimizer, loader: DataLoader,
                    device: torch.device, epoch: int) -> Dict:
    model.train()
    total_losses = {}
    n_batches    = len(loader)

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict  = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  [Epoch {epoch}] batch {i+1}/{n_batches}  "
                  f"loss={total_loss.item():.4f}")

    return {k: v / n_batches for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Backbone freeze helper
# ---------------------------------------------------------------------------

def set_backbone_frozen(model: FasterRCNN, frozen: bool):
    for param in model.backbone.parameters():
        param.requires_grad = not frozen


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ---- Paths ----
    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results"
    plots_dir   = results_dir / "plots"

    # Per-epoch checkpoints → local_ckpt_dir if provided, else output_dir
    if args.local_ckpt_dir:
        ckpt_dir = Path(args.local_ckpt_dir) / "checkpoints"
        print(f"[Train] Per-epoch checkpoints → {ckpt_dir}  (local, not Drive)")
    else:
        ckpt_dir = output_dir / "checkpoints"
        print(f"[Train] Per-epoch checkpoints → {ckpt_dir}")

    # best.pt always goes to Drive
    best_ckpt_dir = output_dir / "checkpoints"

    for d in [ckpt_dir, best_ckpt_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Datasets ----
    train_ds = MUSCIMADataset(args.data_dir, split="train")
    val_ds   = MUSCIMADataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=args.num_workers)

    # ---- Model ----
    model = build_faster_rcnn(pretrained_backbone=True)
    model.to(device)

    # Freeze backbone for first 5 epochs
    set_backbone_frozen(model, frozen=True)
    print("[Train] Backbone frozen for first 5 epochs")

    # ---- Resume ----
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Train] Resumed from epoch {start_epoch - 1}")

    # ---- Optimizer & Scheduler ----
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)

    # Linear warmup for 5 epochs then cosine decay
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 0.5 * (1 + math.cos(
            math.pi * (epoch - 5) / max(1, args.epochs - 5)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- History ----
    history = {
        "train_loss_total":       [],
        "train_loss_classifier":  [],
        "train_loss_box_reg":     [],
        "train_loss_objectness":  [],
        "train_loss_rpn_box_reg": [],
        "val_map":                [],
        "val_map_50":             [],
        "val_precision":          [],
        "val_recall":             [],
        "lr":                     [],
        "epoch":                  [],
    }
    best_map = 0.0

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")

        # Unfreeze backbone after epoch 5
        if epoch == 6:
            set_backbone_frozen(model, frozen=False)
            # Re-create optimizer with all parameters now unfrozen
            params    = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=args.lr * 0.1,
                                        momentum=0.9, weight_decay=1e-4)
            print("  [Train] Backbone unfrozen — continuing with lower LR")

        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        print(f"  [Epoch {epoch}] Running validation...")
        val_metrics = evaluate(model, val_loader, device)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss_total"].append(sum(train_losses.values()))
        history["train_loss_classifier"].append(
            train_losses.get("loss_classifier", 0))
        history["train_loss_box_reg"].append(
            train_losses.get("loss_box_reg", 0))
        history["train_loss_objectness"].append(
            train_losses.get("loss_objectness", 0))
        history["train_loss_rpn_box_reg"].append(
            train_losses.get("loss_rpn_box_reg", 0))
        history["val_map"].append(val_metrics.get("map", 0))
        history["val_map_50"].append(val_metrics.get("map_50", 0))
        history["val_precision"].append(val_metrics.get("precision", 0))
        history["val_recall"].append(val_metrics.get("recall", 0))

        print(f"  Loss      : {history['train_loss_total'][-1]:.4f}")
        print(f"  Val mAP   : {val_metrics.get('map', 0):.4f}")
        print(f"  Precision : {val_metrics.get('precision', 0):.4f}  "
              f"Recall: {val_metrics.get('recall', 0):.4f}")
        print(f"  LR        : {current_lr:.6f}")
        print(f"  Time      : {elapsed:.1f}s")

        # Per-epoch checkpoint → local storage only
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_map":              val_metrics.get("map", 0),
        }, ckpt_dir / f"epoch_{epoch:03d}.pt")

        # Best checkpoint → always Drive
        if val_metrics.get("map", 0) >= best_map:
            best_map = val_metrics.get("map", 0)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_map":          best_map,
            }, best_ckpt_dir / "best.pt")
            print(f"  ✓ New best mAP: {best_map:.4f} — saved best.pt → Drive")

        # Save metrics JSON after every epoch so progress isn't lost on timeout
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    # ---- Final plots ----
    plot_all(history, plots_dir)
    print(f"\n[Train] Complete. Best val mAP: {best_map:.4f}")
    print(f"[Train] Results saved → {output_dir}")


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def evaluate_and_plot(checkpoint_path: str, data_dir: str,
                      output_dir: str = "outputs/stage1"):
    from src.stage1_detection.dataset import MUSCIMADataset, collate_fn
    from src.stage1_detection.detector import build_faster_rcnn
    from src.stage1_detection.evaluate import evaluate
    from src.stage1_detection.visualize import plot_per_class_map, visualize_detections

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)

    model = build_faster_rcnn(pretrained_backbone=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds     = MUSCIMADataset(data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("[Eval] Running test-set evaluation...")
    metrics = evaluate(model, test_loader, device)
    print(f"  Test mAP       : {metrics['map']:.4f}")
    print(f"  Test Precision : {metrics['precision']:.4f}")
    print(f"  Test Recall    : {metrics['recall']:.4f}")

    results_dir = output_dir / "results"
    plots_dir   = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    plot_per_class_map(metrics["per_class"], plots_dir)
    visualize_detections(model, test_ds, device, plots_dir, num_samples=6)

    print(f"\n[Eval] Done. Results saved → {results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Stage 1 Faster R-CNN on MUSCIMA++")
    parser.add_argument("--data_dir",       type=str,   default="data/raw/muscima")
    parser.add_argument("--output_dir",     type=str,   default="outputs/stage1",
                        help="Drive path — best.pt, metrics, plots saved here")
    parser.add_argument("--local_ckpt_dir", type=str,   default=None,
                        help="Local path for per-epoch checkpoints (avoids filling Drive)")
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=0.0005)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--resume",         type=str,   default=None)
    args = parser.parse_args()

    train(args)
