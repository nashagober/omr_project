"""
Stage 1 Evaluation — Precision, Recall, mAP
Computes per-class and aggregate detection metrics on a DataLoader.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from torchvision.models.detection import FasterRCNN


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])

    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-class AP (Pascal VOC 11-point interpolation)
# ---------------------------------------------------------------------------

def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """11-point interpolated AP (VOC style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        ps = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(ps) if ps else 0.0
    return ap / 11.0


def compute_class_ap(predictions: List[Dict], ground_truths: List[Dict],
                     class_idx: int, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute AP, mean precision, mean recall for a single class.

    predictions : list of {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}
    ground_truths: list of {"boxes": [[x1,y1,x2,y2],...], "labels": [...]}

    Returns (ap, mean_precision, mean_recall)
    """
    # Collect all predictions for this class, sorted by score descending
    all_preds = []  # (score, image_idx, box)
    gt_by_image = defaultdict(list)

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if label == class_idx:
                all_preds.append((score, img_idx, box))
        for box, label in zip(gt["boxes"], gt["labels"]):
            if label == class_idx:
                gt_by_image[img_idx].append({"box": box, "matched": False})

    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp_list, fp_list = [], []
    matched_gt = defaultdict(lambda: defaultdict(bool))

    for score, img_idx, pred_box in all_preds:
        gts_for_img = gt_by_image[img_idx]
        best_iou, best_j = 0.0, -1

        for j, gt_entry in enumerate(gts_for_img):
            if matched_gt[img_idx][j]:
                continue
            iou = box_iou(pred_box, gt_entry["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt[img_idx][best_j] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    recalls    = tp_cum / (total_gt + 1e-8)

    ap = compute_ap(precisions.tolist(), recalls.tolist())
    mean_p = float(precisions[-1]) if len(precisions) else 0.0
    mean_r = float(recalls[-1])    if len(recalls)    else 0.0

    return ap, mean_p, mean_r


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: FasterRCNN, loader, device: torch.device,
             iou_threshold: float = 0.5) -> Dict:
    """
    Run model on the validation/test DataLoader and compute:
        map      — mean AP across all classes (IoU=0.5)
        map_50   — same as map (alias for 0.5 threshold)
        precision — macro-averaged precision
        recall    — macro-averaged recall
        per_class — {class_name: {ap, precision, recall}}

    Returns a flat dict of scalar metrics.
    """
    from src.stage1_detection.dataset import IDX_TO_CLASS, NUM_CLASSES

    model.eval()
    all_preds, all_gts = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            all_preds.append({
                "boxes":  output["boxes"].cpu().tolist(),
                "scores": output["scores"].cpu().tolist(),
                "labels": output["labels"].cpu().tolist(),
            })
            all_gts.append({
                "boxes":  target["boxes"].cpu().tolist(),
                "labels": target["labels"].cpu().tolist(),
            })

    # Compute per-class metrics (skip background = 0)
    aps, precisions, recalls = [], [], []
    per_class = {}

    for class_idx in range(1, NUM_CLASSES):
        ap, p, r = compute_class_ap(all_preds, all_gts, class_idx, iou_threshold)
        class_name = IDX_TO_CLASS.get(class_idx, f"class_{class_idx}")
        per_class[class_name] = {"ap": ap, "precision": p, "recall": r}
        aps.append(ap)
        precisions.append(p)
        recalls.append(r)

    mean_ap  = float(np.mean(aps))        if aps else 0.0
    mean_p   = float(np.mean(precisions)) if precisions else 0.0
    mean_r   = float(np.mean(recalls))    if recalls else 0.0

    return {
        "map":       mean_ap,
        "map_50":    mean_ap,
        "precision": mean_p,
        "recall":    mean_r,
        "per_class": per_class,
    }
