"""
Stage 1: Symbol Detector — Faster R-CNN
Backbone: ResNet-50-FPN (pretrained on COCO, fine-tuned on MUSCIMA++)
Reference: Pacha et al. (2018) - A Baseline for General Music Object Detection
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms.functional as TF

from src.stage1_detection.dataset import NUM_CLASSES, IDX_TO_CLASS


@dataclass
class DetectedSymbol:
    label: str
    class_idx: int
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    staff_line: int      # assigned in post-processing


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_faster_rcnn(num_classes: int = NUM_CLASSES,
                      pretrained_backbone: bool = True) -> FasterRCNN:
    """
    Build a Faster R-CNN with ResNet-50-FPN backbone.
    The classification head is replaced to match our num_classes.

    Args:
        num_classes        : total classes including background (default 26)
        pretrained_backbone: use COCO-pretrained weights for the backbone
    """
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone)

    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class SymbolDetector:
    """
    Wraps a trained Faster R-CNN for inference on sheet music images.
    Used by main.py and the evaluation script.
    """

    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(config.get("model_path"))
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: Optional[str]) -> FasterRCNN:
        model = build_faster_rcnn(pretrained_backbone=False)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[SymbolDetector] Loaded weights from {model_path}")
        else:
            print("[SymbolDetector] No checkpoint found — using random weights.")
        return model

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Load image and convert to a float tensor in [0, 1]."""
        image = Image.open(image_path).convert("RGB")
        return TF.to_tensor(image)

    @torch.no_grad()
    def detect(self, image_path: str) -> List[DetectedSymbol]:
        """
        Run detection on a sheet music image.

        Returns DetectedSymbol list sorted by (staff_line, x1).
        """
        image_tensor = self.preprocess(image_path).to(self.device)
        outputs = self.model([image_tensor])[0]

        boxes   = outputs["boxes"].cpu()
        scores  = outputs["scores"].cpu()
        labels  = outputs["labels"].cpu()

        symbols = []
        for box, score, label in zip(boxes, scores, labels):
            if score.item() < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            symbols.append(DetectedSymbol(
                label       = IDX_TO_CLASS.get(label.item(), "other"),
                class_idx   = label.item(),
                bbox        = (x1, y1, x2, y2),
                confidence  = score.item(),
                staff_line  = self._assign_staff_line(y1, y2),
            ))

        # Sort reading order: top staff first, then left-to-right
        symbols.sort(key=lambda s: (s.staff_line, s.bbox[0]))
        return symbols

    def _assign_staff_line(self, y1: float, y2: float) -> int:
        """
        Very simple staff-line assignment: bucket by vertical centre.
        TODO: Replace with a proper staff detection algorithm once staff
              positions are extracted from the image.
        Staff height ≈ 100px is a rough default; tune per dataset.
        """
        STAFF_HEIGHT_PX = 100
        y_center = (y1 + y2) / 2
        return int(y_center // STAFF_HEIGHT_PX)
