"""
Stage 2 PrIMuS Evaluation
Metrics for the decoder-only language model trained on PrIMuS.

Metrics:
    token_accuracy    : % of tokens predicted correctly (next-token prediction)
    top5_accuracy     : % where correct next token is in top-5 predictions
    perplexity        : exp(avg cross-entropy loss) — lower is better
    avg_loss          : average cross-entropy loss
"""

from typing import Dict
import math
import torch
import torch.nn.functional as F

from src.stage2_sequencing.vocabulary import PAD_IDX


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Dict:
    """
    Evaluate decoder-only LM on PrIMuS validation/test set.

    Returns:
        token_accuracy : float — next-token prediction accuracy
        top5_accuracy  : float — next-token in top-5
        perplexity     : float — exp(loss), lower is better
        avg_loss       : float — average cross-entropy loss
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    total_tokens   = 0
    correct_tokens = 0
    correct_top5   = 0
    total_loss     = 0.0
    n_batches      = 0

    for batch in loader:
        input_ids    = batch["input_ids"].to(device)     # [B, S]
        target_ids   = batch["target_ids"].to(device)    # [B, S]
        padding_mask = batch["padding_mask"].to(device)  # [B, S]

        logits = model(input_ids, padding_mask)           # [B, S, V]
        B, S, V = logits.shape

        # Loss
        loss = criterion(logits.reshape(B * S, V),
                         target_ids.reshape(B * S))
        total_loss += loss.item()
        n_batches  += 1

        # Token accuracy (ignore PAD)
        preds   = logits.argmax(dim=-1)         # [B, S]
        non_pad = target_ids != PAD_IDX
        correct = (preds == target_ids) & non_pad
        total_tokens   += non_pad.sum().item()
        correct_tokens += correct.sum().item()

        # Top-5 accuracy
        top5    = logits.topk(5, dim=-1).indices  # [B, S, 5]
        in_top5 = (top5 == target_ids.unsqueeze(-1)).any(-1) & non_pad
        correct_top5 += in_top5.sum().item()

    avg_loss   = total_loss / max(n_batches, 1)
    token_acc  = correct_tokens / max(total_tokens, 1)
    top5_acc   = correct_top5   / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow

    return {
        "token_accuracy": token_acc,
        "top5_accuracy":  top5_acc,
        "perplexity":     perplexity,
        "avg_loss":       avg_loss,
    }
