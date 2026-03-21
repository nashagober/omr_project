
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import torch
from PIL import Image

COLORS = {
    "primary":   "#2563EB",   # blue
    "secondary": "#16A34A",   # green
    "accent":    "#DC2626",   # red
    "warn":      "#D97706",   # amber
    "purple":    "#7C3AED",
    "gray":      "#6B7280",
}

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F9FAFB",
        "axes.grid":         True,
        "grid.color":        "#E5E7EB",
        "grid.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "legend.framealpha": 0.9,
        "lines.linewidth":   2.2,
        "lines.markersize":  5,
    })

def plot_loss_curves(history: Dict, output_dir: Path):

    _apply_style()
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 1 — Training Loss", fontsize=15, fontweight="bold", y=1.02)

    # --- Total loss ---
    ax1.plot(epochs, history["train_loss_total"],
             color=COLORS["primary"], marker="o", label="Total Loss")
    ax1.set_title("Total Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # --- Component losses ---
    components = {
        "Classifier":    ("train_loss_classifier",  COLORS["primary"]),
        "Box Reg":       ("train_loss_box_reg",      COLORS["secondary"]),
        "Objectness":    ("train_loss_objectness",   COLORS["accent"]),
        "RPN Box Reg":   ("train_loss_rpn_box_reg",  COLORS["purple"]),
    }
    for label, (key, color) in components.items():
        if key in history and any(v > 0 for v in history[key]):
            ax2.plot(epochs, history[key], color=color, marker="o", label=label)

    ax2.set_title("Loss Components per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    out = output_dir / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {out}")

def plot_pr_map_curves(history: Dict, output_dir: Path):

    _apply_style()
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 1 — Validation Metrics", fontsize=15, fontweight="bold", y=1.02)

    # --- Precision / Recall ---
    ax1.plot(epochs, history["val_precision"],
             color=COLORS["primary"],   marker="o", label="Precision")
    ax1.plot(epochs, history["val_recall"],
             color=COLORS["secondary"], marker="s", label="Recall")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Precision & Recall")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Score")
    ax1.legend()

    # Annotate best values
    best_p_epoch = int(np.argmax(history["val_precision"]))
    best_r_epoch = int(np.argmax(history["val_recall"]))
    ax1.annotate(f"{history['val_precision'][best_p_epoch]:.3f}",
                 xy=(epochs[best_p_epoch], history["val_precision"][best_p_epoch]),
                 xytext=(8, 4), textcoords="offset points",
                 fontsize=9, color=COLORS["primary"])
    ax1.annotate(f"{history['val_recall'][best_r_epoch]:.3f}",
                 xy=(epochs[best_r_epoch], history["val_recall"][best_r_epoch]),
                 xytext=(8, 4), textcoords="offset points",
                 fontsize=9, color=COLORS["secondary"])

    # --- mAP ---
    ax2.plot(epochs, history["val_map"],
             color=COLORS["accent"],  marker="o", label="mAP (IoU=0.5)")
    ax2.plot(epochs, history["val_map_50"],
             color=COLORS["purple"], marker="s", linestyle="--", label="mAP@50")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Mean Average Precision")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.legend()

    best_map_epoch = int(np.argmax(history["val_map"]))
    ax2.annotate(f"best: {history['val_map'][best_map_epoch]:.3f}",
                 xy=(epochs[best_map_epoch], history["val_map"][best_map_epoch]),
                 xytext=(8, -14), textcoords="offset points",
                 fontsize=9, color=COLORS["accent"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2))

    plt.tight_layout()
    out = output_dir / "pr_map_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {out}")


def plot_per_class_map(per_class_metrics: Dict, output_dir: Path):

    _apply_style()

    classes = list(per_class_metrics.keys())
    aps     = [per_class_metrics[c]["ap"] for c in classes]

    # Sort descending by AP
    sorted_pairs = sorted(zip(aps, classes), reverse=True)
    aps_sorted     = [p[0] for p in sorted_pairs]
    classes_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.4)))
    fig.suptitle("Stage 1 — Per-Class Average Precision (AP@0.5)",
                 fontsize=14, fontweight="bold")

    colors = [COLORS["primary"] if ap >= 0.5 else
              COLORS["warn"]    if ap >= 0.25 else
              COLORS["accent"]  for ap in aps_sorted]

    bars = ax.barh(classes_sorted, aps_sorted, color=colors, edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AP @ IoU=0.5")
    ax.axvline(x=0.5, color=COLORS["gray"], linestyle="--", linewidth=1, label="AP=0.5")
    ax.legend(loc="lower right")

    for bar, ap in zip(bars, aps_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ap:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out = output_dir / "per_class_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {out}")


_BOX_COLORS = [
    "#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6",
    "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
]

def _color_for_class(class_idx: int) -> str:
    return _BOX_COLORS[class_idx % len(_BOX_COLORS)]


@torch.no_grad()
def visualize_detections(model, dataset, device: torch.device,
                         output_dir: Path, num_samples: int = 6,
                         confidence_threshold: float = 0.5):

    from src.stage1_detection.dataset import IDX_TO_CLASS

    _apply_style()
    model.eval()

    # Pick evenly-spaced samples from the dataset
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for fig_idx, ds_idx in enumerate(indices):
        image_tensor, target = dataset[ds_idx]
        image_np = image_tensor.permute(1, 2, 0).numpy()

        # Run inference
        output = model([image_tensor.to(device)])[0]
        pred_boxes  = output["boxes"].cpu().tolist()
        pred_scores = output["scores"].cpu().tolist()
        pred_labels = output["labels"].cpu().tolist()

        gt_boxes  = target["boxes"].tolist()
        gt_labels = target["labels"].tolist()

        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Sample {fig_idx + 1}  (dataset index {ds_idx})",
                     fontsize=13, fontweight="bold")

        # --- Ground Truth ---
        ax_gt.imshow(image_np, cmap="gray" if image_np.shape[2] == 1 else None)
        ax_gt.set_title(f"Ground Truth  ({len(gt_boxes)} symbols)", fontsize=11)
        ax_gt.axis("off")
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            color = _color_for_class(label)
            rect  = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor=color, facecolor="none")
            ax_gt.add_patch(rect)
            ax_gt.text(x1, max(0, y1 - 3), IDX_TO_CLASS.get(label, "?"),
                       fontsize=7, color=color,
                       bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

        # --- Predictions ---
        ax_pred.imshow(image_np, cmap="gray" if image_np.shape[2] == 1 else None)
        high_conf = [(b, s, l) for b, s, l in zip(pred_boxes, pred_scores, pred_labels)
                     if s >= confidence_threshold]
        ax_pred.set_title(f"Predictions  ({len(high_conf)} above {confidence_threshold:.0%})",
                          fontsize=11)
        ax_pred.axis("off")
        for box, score, label in high_conf:
            x1, y1, x2, y2 = box
            color = _color_for_class(label)
            rect  = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=1.5, edgecolor=color, facecolor="none")
            ax_pred.add_patch(rect)
            ax_pred.text(x1, max(0, y1 - 3),
                         f"{IDX_TO_CLASS.get(label, '?')} {score:.2f}",
                         fontsize=7, color=color,
                         bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

        plt.tight_layout()
        out = output_dir / f"detection_sample_{fig_idx + 1:02d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Saved → {out}")

def plot_all(history: Dict, plots_dir: Path,
             per_class_metrics: Optional[Dict] = None,
             model=None, dataset=None, device=None,
             num_detection_samples: int = 6):

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Visualize] Generating plots...")

    plot_loss_curves(history, plots_dir)
    plot_pr_map_curves(history, plots_dir)

    if per_class_metrics:
        plot_per_class_map(per_class_metrics, plots_dir)

    if model is not None and dataset is not None and device is not None:
        visualize_detections(model, dataset, device, plots_dir,
                             num_samples=num_detection_samples)

    print(f"[Visualize] All plots saved to {plots_dir}")

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Regenerate plots from metrics.json")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json")
    parser.add_argument("--output",  type=str, default="outputs/stage1/results/plots")
    args = parser.parse_args()

    with open(args.metrics) as f:
        history = json.load(f)

    plot_all(history, Path(args.output))
