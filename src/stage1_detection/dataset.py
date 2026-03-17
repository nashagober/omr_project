"""
Stage 1 Dataset Loader — MUSCIMA++
Parses CropObject XML annotations and returns image crops + bounding boxes
for Faster R-CNN training.

MUSCIMA++ structure expected at root_dir:
    root_dir/
        data/cropobjects_withstaff/   ← XML annotation files
        data/images/                  ← PNG score images
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Class label mapping
# Subset of the most common MUSCIMA++ symbol classes.
# Extend as needed — background is always class 0.
# ---------------------------------------------------------------------------
MUSCIMA_CLASSES: Dict[str, int] = {
    "background":        0,
    "notehead-full":     1,
    "notehead-empty":    2,
    "rest-quarter":      3,
    "rest-half":         4,
    "rest-whole":        5,
    "rest-eighth":       6,
    "clef-G":            7,
    "clef-F":            8,
    "clef-C":            9,
    "time-signature":    10,
    "key-signature":     11,
    "bar-line":          12,
    "beam":              13,
    "stem":              14,
    "ledger-line":       15,
    "slur":              16,
    "tie":               17,
    "accidental-sharp":  18,
    "accidental-flat":   19,
    "accidental-natural":20,
    "flag-8th-up":       21,
    "flag-8th-down":     22,
    "dynamic-forte":     23,
    "dynamic-piano":     24,
    "other":             25,
}

SKIP_CLASSES = {
    "staff",
    "staffline",
    "staffspace", 
    "staff-grouping",
    "measure-separator",
    "system-separator",
    "staffGrouping",
    "staffLine",
    "staffSpace",
    "measureSeparator",
    "systemSeparator",
    }

NUM_CLASSES = len(MUSCIMA_CLASSES)  # 26 (including background)
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in MUSCIMA_CLASSES.items()}


def class_name_to_idx(name: str) -> int:
    """Map a raw MUSCIMA++ class name to an integer label."""
    normalized = name.lower().replace("_", "-")
    if normalized in MUSCIMA_CLASSES:
        return MUSCIMA_CLASSES[normalized]
    for key in MUSCIMA_CLASSES:
        if key in normalized or normalized in key:
            return MUSCIMA_CLASSES[key]
    return MUSCIMA_CLASSES["other"]


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------

def parse_cropobject_xml(xml_path: str) -> List[Dict]:
    """
    Parse a single MUSCIMA++ CropObject XML file.
    Returns list of {"label": int, "bbox": [x1,y1,x2,y2], "class_name": str}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for obj in root.iter("CropObject"):
        class_name_el = obj.find("ClassName")
        top_el    = obj.find("Top")
        left_el   = obj.find("Left")
        width_el  = obj.find("Width")
        height_el = obj.find("Height")

        if any(el is None for el in [class_name_el, top_el, left_el, width_el, height_el]):
            continue

        top    = int(top_el.text)
        left   = int(left_el.text)
        width  = int(width_el.text)
        height = int(height_el.text)
        x1, y1 = left, top
        x2, y2 = left + width, top + height

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = class_name_el.text.strip()

        if class_name in SKIP_CLASSES or class_name.lower() in {s.lower() for s in SKIP_CLASSES}:
          continue

        annotations.append({
            "class_name": class_name,
            "label":      class_name_to_idx(class_name),
            "bbox":       [x1, y1, x2, y2],
        })

    return annotations


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MUSCIMADataset(Dataset):
    """
    PyTorch Dataset for MUSCIMA++ handwritten music symbol detection.

    Returns samples compatible with torchvision Faster R-CNN:
        image  : FloatTensor [3, H, W] in [0, 1]
        target : dict with "boxes" [N,4], "labels" [N], "image_id" [1]

    Args:
        root_dir  : path to the MUSCIMA++ root directory
        split     : "train" | "val" | "test"
        transform : optional callable (image_tensor, target) → (image_tensor, target)
    """

    # Fixed writer-ID splits (writers 01–50)
    TRAIN_WRITERS = list(range(1, 36))   # 70%
    VAL_WRITERS   = list(range(36, 44))  # 15%
    TEST_WRITERS  = list(range(44, 51))  # 15%

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, min_box_area: int = 4):
        self.root_dir     = Path(root_dir)
        self.split        = split
        self.transform    = transform
        self.min_box_area = min_box_area

        self.image_dir = self.root_dir / "data" / "images"
        self.xml_dir   = self.root_dir / "data" / "cropobjects_withstaff"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.xml_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.xml_dir}")

        self.samples = self._load_samples()
        print(f"[MUSCIMADataset] {split}: {len(self.samples)} images loaded.")

    def _writer_ids_for_split(self) -> List[int]:
        return {"train": self.TRAIN_WRITERS,
                "val":   self.VAL_WRITERS,
                "test":  self.TEST_WRITERS}[self.split]

    def _load_samples(self) -> List[Dict]:
        """
        Pair each XML annotation file with its corresponding image.
        MUSCIMA++ filenames: CVC-MUSCIMA_W-<writer>_N-<page>_D-ideal.xml / .png
        """
        writer_ids = self._writer_ids_for_split()
        samples = []

        for xml_file in sorted(self.xml_dir.glob("*.xml")):
            stem = xml_file.stem
            try:
                writer_part = [p for p in stem.split("_") if p.startswith("W-")][0]
                writer_id   = int(writer_part.split("-")[1])
            except (IndexError, ValueError):
                continue

            if writer_id not in writer_ids:
                continue

            # Try PNG then JPG
            image_path = self.image_dir / xml_file.with_suffix(".png").name
            if not image_path.exists():
                image_path = self.image_dir / xml_file.with_suffix(".jpg").name
            if not image_path.exists():
                continue

            annotations = parse_cropobject_xml(str(xml_file))
            if not annotations:
                continue

            samples.append({
                "image_path":  str(image_path),
                "annotations": annotations,
                "image_id":    len(samples),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = TF.to_tensor(image)

        boxes, labels = [], []
        for ann in sample["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["label"])

        if boxes:
            boxes_tensor  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "image_id": torch.tensor([sample["image_id"]], dtype=torch.int64),
        }

        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)

        return image_tensor, target


def collate_fn(batch):
    """Faster R-CNN requires list of (image, target), not a stacked tensor."""
    return [item[0] for item in batch], [item[1] for item in batch]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/muscima"
    ds = MUSCIMADataset(root, split="train")
    img, tgt = ds[0]
    print(f"Image shape : {img.shape}")
    print(f"Num boxes   : {len(tgt['boxes'])}")
    print(f"Label sample: {tgt['labels'][:5].tolist()}")
