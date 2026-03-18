"""
Stage 2 PrIMuS Training Script
Trains a decoder-only transformer language model on PrIMuS semantic sequences.

This is independent of Stage 1 — no detected symbols needed.
The model learns the statistical structure of music notation directly
from ground-truth sequences, which Stage 3 uses for error correction.

Usage:
    python -m src.stage2_sequencing.primus_train \
        --primus_dir     data/raw/primus \
        --output_dir     outputs/stage2_primus \
        --local_ckpt_dir /content/local_checkpoints \
        --epochs         30 \
        --batch_size     32 \
        --lr             1e-4

Colab:
    !cd /content/omr_project && python -m src.stage2_sequencing.primus_train \
        --primus_dir     {PRIMUS_DIR} \
        --output_dir     {STAGE2_OUTPUT} \
        --local_ckpt_dir /content/stage2_local \
        --epochs         30 \
        --batch_size     32 \
        --num_workers    2
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.stage2_sequencing.primus_dataset import PrIMuSDataset, collate_fn
from src.stage2_sequencing.decoder_model import MusicDecoderTransformer
from src.stage2_sequencing.primus_evaluate import evaluate
from src.stage2_sequencing.primus_visualize import plot_all
from src.stage2_sequencing.vocabulary import PAD_IDX, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, criterion,
                    device, epoch) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, batch in enumerate(loader):
        input_ids    = batch["input_ids"].to(device)
        target_ids   = batch["target_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(input_ids, padding_mask)   # [B, S, V]
        B, S, V = logits.shape
        loss = criterion(logits.reshape(B * S, V),
                         target_ids.reshape(B * S))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  [Epoch {epoch}] batch {i+1}/{n_batches}  "
                  f"loss={loss.item():.4f}")

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PrIMuS Train] Device : {device}")

    # ---- Paths ----
    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results"
    plots_dir   = results_dir / "plots"

    if args.local_ckpt_dir:
        ckpt_dir = Path(args.local_ckpt_dir) / "checkpoints"
        print(f"[PrIMuS Train] Per-epoch checkpoints → {ckpt_dir}  (local)")
    else:
        ckpt_dir = output_dir / "checkpoints"

    best_ckpt_dir = output_dir / "checkpoints"

    for d in [ckpt_dir, best_ckpt_dir, results_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Datasets ----
    train_ds = PrIMuSDataset(args.primus_dir, split="train",
                              max_seq_len=args.max_seq_len)
    val_ds   = PrIMuSDataset(args.primus_dir, split="val",
                              max_seq_len=args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=args.num_workers)

    # ---- Model ----
    model = MusicDecoderTransformer(
        vocab_size  = VOCAB_SIZE,
        d_model     = args.d_model,
        nhead       = args.nhead,
        num_layers  = args.num_layers,
        ffn_dim     = args.ffn_dim,
        dropout     = args.dropout,
        max_seq_len = args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PrIMuS Train] Parameters : {n_params:,}")
    print(f"[PrIMuS Train] Train size  : {len(train_ds):,}")
    print(f"[PrIMuS Train] Val size    : {len(val_ds):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9)

    # Warmup 3 epochs then slow cosine decay to 10% of peak LR
    def lr_lambda(ep):
        if ep < 3:
            return (ep + 1) / 3
        progress = (ep - 3) / max(1, args.epochs - 3)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # ---- Resume ----
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"[PrIMuS Train] Resumed from epoch {start_epoch - 1}  "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")

    # ---- History ----
    history: Dict = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_token_accuracy": [], "val_top5_accuracy": [],
        "val_perplexity": [], "lr": [],
    }
    best_token_acc = 0.0

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss  = train_one_epoch(model, optimizer, train_loader,
                                       criterion, device, epoch)
        lr_scheduler.step()

        print(f"  [Epoch {epoch}] Validating...")
        val_metrics = evaluate(model, val_loader, device)
        elapsed     = time.time() - t0
        current_lr  = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["avg_loss"])
        history["val_token_accuracy"].append(val_metrics["token_accuracy"])
        history["val_top5_accuracy"].append(val_metrics["top5_accuracy"])
        history["val_perplexity"].append(val_metrics["perplexity"])
        history["lr"].append(current_lr)

        print(f"  Train Loss  : {train_loss:.4f}")
        print(f"  Val Loss    : {val_metrics['avg_loss']:.4f}")
        print(f"  Perplexity  : {val_metrics['perplexity']:.2f}")
        print(f"  Token Acc   : {val_metrics['token_accuracy']:.4f}")
        print(f"  Top-5 Acc   : {val_metrics['top5_accuracy']:.4f}")
        print(f"  LR          : {current_lr:.6f}")
        print(f"  Time        : {elapsed:.1f}s")

        # Per-epoch checkpoint → local only
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "val_token_accuracy":   val_metrics["token_accuracy"],
        }, ckpt_dir / f"epoch_{epoch:03d}.pt")

        # Best checkpoint → always Drive
        if val_metrics["token_accuracy"] >= best_token_acc:
            best_token_acc = val_metrics["token_accuracy"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "val_token_accuracy":   best_token_acc,
            }, best_ckpt_dir / "best.pt")
            print(f"  ✓ New best token acc: {best_token_acc:.4f} — saved best.pt → Drive")

        # Save metrics after every epoch
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

    # ---- Final plots ----
    plot_all(history, plots_dir,
             model=model, dataset=val_ds, device=device,
             num_samples=5)

    print(f"\n[PrIMuS Train] Done. Best token accuracy: {best_token_acc:.4f}")
    print(f"[PrIMuS Train] Results → {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Stage 2 decoder-only LM on PrIMuS")

    # Data
    parser.add_argument("--primus_dir",    type=str, required=True,
                        help="Path to extracted PrIMuS directory")

    # Paths
    parser.add_argument("--output_dir",    type=str, default="outputs/stage2_primus",
                        help="Drive path — best.pt, metrics, plots saved here")
    parser.add_argument("--local_ckpt_dir",type=str, default=None,
                        help="Local path for per-epoch checkpoints (keeps Drive free)")
    parser.add_argument("--resume",        type=str, default=None)

    # Training
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--max_seq_len", type=int,   default=512)

    # Model architecture
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--ffn_dim",    type=int,   default=1024)
    parser.add_argument("--dropout",    type=float, default=0.1)

    args = parser.parse_args()
    train(args)
