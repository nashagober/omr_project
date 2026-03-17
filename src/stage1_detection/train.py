"""
Stage 1 Training Script — Faster R-CNN on MUSCIMA++

Usage:
    python -m src.stage1_detection.train \
        --data_dir  data/raw/muscima \
        --output_dir outputs/stage1 \
        --epochs    20 \
        --batch_size 4 \
        --lr        0.005

Saves to outputs/stage1/:
    checkpoints/epoch_<N>.pt    ← full model state
    checkpoints/best.pt         ← best val mAP checkpoint
    results/metrics.json        ← per-epoch metrics
    results/plots/              ← all graphs
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from src.stage1_detection.dataset import MUSCIMADataset, collate_fn
from src.stage1_detection.detector import build_faster_rcnn
from src.stage1_detection.evaluate import evaluate
from src.stage1_detection.visualize import plot_all

from typing import Dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: FasterRCNN, optimizer, loader: DataLoader,
                    device: torch.device, epoch: int) -> Dict:
    """
    Run one full training epoch.
    Returns dict of averaged losses for this epoch.
    """
    model.train()
    total_losses = {}
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  [Epoch {epoch}] batch {i+1}/{n_batches}  "
                  f"loss={total_loss.item():.4f}")

    # Return averaged losses
    return {k: v / n_batches for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    from typing import Dict  # local import for type hints inside function

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ---- Paths ----
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    plots_dir  = results_dir / "plots"
    for d in [ckpt_dir, results_dir, plots_dir]:
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

    # Optionally resume from checkpoint
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Train] Resumed from epoch {start_epoch - 1}")

    # ---- Optimizer & Scheduler ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    # ---- Metrics history ----
    history = {
        "train_loss_total":        [],
        "train_loss_classifier":   [],
        "train_loss_box_reg":      [],
        "train_loss_objectness":   [],
        "train_loss_rpn_box_reg":  [],
        "val_map":                 [],
        "val_map_50":              [],
        "val_precision":           [],
        "val_recall":              [],
        "lr":                      [],
        "epoch":                   [],
    }
    best_map = 0.0

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")

        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        # Validation
        print(f"  [Epoch {epoch}] Running validation...")
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Record history ---
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss_total"].append(sum(train_losses.values()))
        history["train_loss_classifier"].append(train_losses.get("loss_classifier", 0))
        history["train_loss_box_reg"].append(train_losses.get("loss_box_reg", 0))
        history["train_loss_objectness"].append(train_losses.get("loss_objectness", 0))
        history["train_loss_rpn_box_reg"].append(train_losses.get("loss_rpn_box_reg", 0))
        history["val_map"].append(val_metrics.get("map", 0))
        history["val_map_50"].append(val_metrics.get("map_50", 0))
        history["val_precision"].append(val_metrics.get("precision", 0))
        history["val_recall"].append(val_metrics.get("recall", 0))

        print(f"  Loss:      {history['train_loss_total'][-1]:.4f}")
        print(f"  Val mAP:   {val_metrics.get('map', 0):.4f}  "
              f"mAP@50: {val_metrics.get('map_50', 0):.4f}")
        print(f"  Precision: {val_metrics.get('precision', 0):.4f}  "
              f"Recall: {val_metrics.get('recall', 0):.4f}")
        print(f"  Time:      {elapsed:.1f}s")

        # --- Save checkpoint every epoch ---
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_map":            val_metrics.get("map", 0),
        }, ckpt_path)

        # --- Save best checkpoint ---
        if val_metrics.get("map", 0) >= best_map:
            best_map = val_metrics.get("map", 0)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_map":          best_map,
            }, ckpt_dir / "best.pt")
            print(f"  ✓ New best mAP: {best_map:.4f} — saved best.pt")

    # ---- Save metrics JSON ----
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Metrics saved → {metrics_path}")

    # ---- Generate all plots ----
    plot_all(history, plots_dir)
    print(f"[Train] Plots saved  → {plots_dir}")
    print(f"\n[Train] Complete. Best val mAP: {best_map:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from typing import Dict

    parser = argparse.ArgumentParser(description="Train Stage 1 Faster R-CNN on MUSCIMA++")
    parser.add_argument("--data_dir",    type=str, default="data/raw/muscima")
    parser.add_argument("--output_dir",  type=str, default="outputs/stage1")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    import torch
    print(torch.cuda.is_available())   # must be True
    print(torch.cuda.get_device_name(0))

    


# ---------------------------------------------------------------------------
# Post-training: full evaluation + detection sample plots
# NOTE: Call this after train() if you want per-class AP and sample visuals.
# ---------------------------------------------------------------------------

def evaluate_and_plot(checkpoint_path: str, data_dir: str,
                      output_dir: str = "outputs/stage1"):
    """
    Load the best checkpoint, run full evaluation on the test set,
    and generate per-class AP bar chart + detection sample images.

    Usage:
        python -c "
        from src.stage1_detection.train import evaluate_and_plot
        evaluate_and_plot('outputs/stage1/checkpoints/best.pt', 'data/raw/muscima')
        "
    """
    from src.stage1_detection.dataset import MUSCIMADataset, collate_fn, NUM_CLASSES
    from src.stage1_detection.detector import build_faster_rcnn
    from src.stage1_detection.evaluate import evaluate
    from src.stage1_detection.visualize import plot_all, plot_per_class_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)

    # Load model
    model = build_faster_rcnn(pretrained_backbone=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Test set
    test_ds     = MUSCIMADataset(data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("[Eval] Running test-set evaluation...")
    metrics = evaluate(model, test_loader, device)
    print(f"  Test mAP:       {metrics['map']:.4f}")
    print(f"  Test Precision: {metrics['precision']:.4f}")
    print(f"  Test Recall:    {metrics['recall']:.4f}")

    # Save test metrics
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Per-class AP bar chart
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_per_class_map(metrics["per_class"], plots_dir)

    # Detection sample visualizations
    from src.stage1_detection.visualize import visualize_detections
    visualize_detections(model, test_ds, device, plots_dir, num_samples=6)

    print(f"\n[Eval] Done. Results saved to {results_dir}")


# evaluate_and_plot(
#    checkpoint_path = "outputs/stage1/checkpoints/best.pt",
#    data_dir        = "data/raw/muscima",
#    output_dir      = "outputs/stage1"
#)