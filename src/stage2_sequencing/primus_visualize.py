"""
Stage 2 PrIMuS Visualization
Plots for the decoder-only language model trained on PrIMuS.

Three plot categories:
  1. Loss and perplexity curves
  2. Token accuracy and top-5 accuracy curves
  3. Sample generated sequences vs ground truth
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "primary":   "#2563EB",
    "secondary": "#16A34A",
    "accent":    "#DC2626",
    "warn":      "#D97706",
    "purple":    "#7C3AED",
    "gray":      "#6B7280",
    "teal":      "#0D9488",
}


def _style():
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
        "lines.linewidth":   2.2,
        "lines.markersize":  5,
        "legend.framealpha": 0.9,
    })


# ---------------------------------------------------------------------------
# 1. Loss and Perplexity
# ---------------------------------------------------------------------------

def plot_loss_perplexity(history: Dict, output_dir: Path):
    _style()
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 2 (PrIMuS) — Loss & Perplexity",
                 fontsize=15, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"],
             color=COLORS["primary"], marker="o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],
             color=COLORS["accent"], marker="s",
             linestyle="--", label="Val Loss")
    best_idx = int(np.argmin(history["val_loss"]))
    ax1.annotate(f"best: {history['val_loss'][best_idx]:.4f}",
                 xy=(epochs[best_idx], history["val_loss"][best_idx]),
                 xytext=(10, 8), textcoords="offset points",
                 fontsize=9, color=COLORS["accent"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["accent"]))
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Perplexity
    ax2.plot(epochs, history["val_perplexity"],
             color=COLORS["purple"], marker="o", label="Val Perplexity")
    ax2.set_title("Validation Perplexity (lower = better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.legend()

    plt.tight_layout()
    out = output_dir / "loss_perplexity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 2. Accuracy Curves
# ---------------------------------------------------------------------------

def plot_accuracy_curves(history: Dict, output_dir: Path):
    _style()
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Stage 2 (PrIMuS) — Token Prediction Accuracy",
                 fontsize=15, fontweight="bold")

    ax.plot(epochs, history["val_token_accuracy"],
            color=COLORS["primary"], marker="o", label="Token Accuracy")
    ax.plot(epochs, history["val_top5_accuracy"],
            color=COLORS["teal"], marker="s",
            linestyle="--", label="Top-5 Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    best_idx = int(np.argmax(history["val_token_accuracy"]))
    ax.annotate(f"best: {history['val_token_accuracy'][best_idx]:.3f}",
                xy=(epochs[best_idx], history["val_token_accuracy"][best_idx]),
                xytext=(8, -16), textcoords="offset points",
                fontsize=9, color=COLORS["primary"],
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"]))

    plt.tight_layout()
    out = output_dir / "accuracy_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {out}")


# ---------------------------------------------------------------------------
# 3. Generated vs Ground Truth Sequences
# ---------------------------------------------------------------------------

def visualize_generations(model, dataset, device, output_dir: Path,
                           num_samples: int = 5):
    """
    For each sample: show ground truth tokens and model-generated tokens
    side by side. Color each generated token green (match) or red (mismatch).
    """
    import torch
    from src.stage2_sequencing.vocabulary import IDX_TO_TOKEN, SOS_IDX, PAD_IDX

    _style()
    model.eval()

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for fig_idx, ds_idx in enumerate(indices):
        sample     = dataset[ds_idx]
        input_ids  = sample["input_ids"]
        target_ids = sample["target_ids"]

        # Ground truth tokens (strip PAD/EOS)
        gt_ids    = [i for i in target_ids.tolist()
                     if i not in (PAD_IDX,)]
        gt_tokens = [IDX_TO_TOKEN.get(i, "<UNK>") for i in gt_ids][:40]

        # Generate with teacher-forced prompt (first 5 tokens as seed)
        seed    = input_ids[:6].tolist()
        gen_ids = model.generate(
            prompt_ids=torch.tensor(seed, device=device),
            max_new_tokens=len(gt_tokens) + 5,
            temperature=0.8,
            device=device,
        )
        gen_tokens = [IDX_TO_TOKEN.get(i, "<UNK>") for i in gen_ids][:40]

        # Pad shorter side
        max_len    = max(len(gt_tokens), len(gen_tokens))
        gt_pad     = gt_tokens  + [""] * (max_len - len(gt_tokens))
        gen_pad    = gen_tokens + [""] * (max_len - len(gen_tokens))

        fig, (ax_gt, ax_gen) = plt.subplots(1, 2, figsize=(16, max(4, max_len * 0.3 + 2)))
        fig.suptitle(f"Sample {fig_idx + 1}  (dataset index {ds_idx})",
                     fontsize=12, fontweight="bold")

        for ax, title in [(ax_gt, "Ground Truth"), (ax_gen, "Generated")]:
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, max_len + 0.5)
            ax.axis("off")
            ax.set_title(title, fontsize=11,
                         color=COLORS["primary"] if title == "Ground Truth"
                         else COLORS["accent"])

        for row, (gt_tok, gen_tok) in enumerate(zip(gt_pad, gen_pad)):
            y = max_len - row
            ax_gt.text(0.05, y, gt_tok, fontsize=8, color=COLORS["primary"],
                       va="center", fontfamily="monospace")
            color = (COLORS["secondary"] if gen_tok == gt_tok
                     else COLORS["accent"] if gen_tok else COLORS["gray"])
            ax_gen.text(0.05, y, gen_tok, fontsize=8, color=color,
                        va="center", fontfamily="monospace")

        plt.tight_layout()
        out = output_dir / f"generation_sample_{fig_idx + 1:02d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Saved → {out}")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def plot_all(history: Dict, plots_dir: Path,
             model=None, dataset=None, device=None,
             num_samples: int = 5):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PrIMuS Visualize] Generating plots...")
    plot_loss_perplexity(history, plots_dir)
    plot_accuracy_curves(history, plots_dir)

    if model is not None and dataset is not None and device is not None:
        visualize_generations(model, dataset, device, plots_dir, num_samples)

    print(f"[PrIMuS Visualize] Done → {plots_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output",  type=str, default="outputs/stage2_primus/results/plots")
    args = parser.parse_args()
    with open(args.metrics) as f:
        history = json.load(f)
    plot_all(history, Path(args.output))
