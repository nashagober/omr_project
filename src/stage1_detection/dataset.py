import xml.etree.ElementTree as ET
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

MUSCIMA_CLASSES: Dict[str, int] = {
    "background":         0,
    "notehead-full":      1,
    "notehead-empty":     2,
    "rest-quarter":       3,
    "rest-half":          4,
    "rest-whole":         5,
    "rest-eighth":        6,
    "clef-G":             7,
    "clef-F":             8,
    "clef-C":             9,
    "time-signature":     10,
    "key-signature":      11,
    "bar-line":           12,
    "beam":               13,
    "stem":               14,
    "ledger-line":        15,
    "slur":               16,
    "tie":                17,
    "accidental-sharp":   18,
    "accidental-flat":    19,
    "accidental-natural": 20,
    "flag-8th-up":        21,
    "flag-8th-down":      22,
    "dynamic-forte":      23,
    "dynamic-piano":      24,
    "other":              25,
}

NUM_CLASSES  = len(MUSCIMA_CLASSES)
IDX_TO_CLASS = {v: k for k, v in MUSCIMA_CLASSES.items()}

# Staff-level classes to skip — full-page-width boxes that confuse the detector
SKIP_CLASSES = {
    "staff", "Staff",
    "staffLine", "staffline", "staff-line",
    "staffSpace", "staffspace", "staff-space",
    "staffGrouping", "staff-grouping",
    "measureSeparator", "measure-separator",
    "systemSeparator", "system-separator",
    "systemMeasure", "system-measure",
    "staffMeasure", "staff-measure",
    "repeatDot", "repeat-dot",
    "multiMeasureRest", "multi-measure-rest",
}

_SKIP_NORMALIZED = {s.lower().replace("_", "-") for s in SKIP_CLASSES}


def class_name_to_idx(name: str) -> int:
    normalized = name.lower().replace("_", "-")
    if normalized in MUSCIMA_CLASSES:
        return MUSCIMA_CLASSES[normalized]
    for key in MUSCIMA_CLASSES:
        if key in normalized or normalized in key:
            return MUSCIMA_CLASSES[key]
    return MUSCIMA_CLASSES["other"]

def parse_cropobject_xml(xml_path: str) -> List[Dict]:
    """
    Parse a MUSCIMA++ CropObject XML file.
    Returns list of {"label": int, "bbox": [x1,y1,x2,y2], "class_name": str}
    Skips staff-level structural elements.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []

    for obj in root.iter("CropObject"):
        class_name_el = obj.find("ClassName")
        top_el        = obj.find("Top")
        left_el       = obj.find("Left")
        width_el      = obj.find("Width")
        height_el     = obj.find("Height")

        if any(el is None for el in [class_name_el, top_el,
                                      left_el, width_el, height_el]):
            continue

        class_name = class_name_el.text.strip()
        if class_name in SKIP_CLASSES:
            continue
        if class_name.lower().replace("_", "-") in _SKIP_NORMALIZED:
            continue

        top    = int(top_el.text)
        left   = int(left_el.text)
        width  = int(width_el.text)
        height = int(height_el.text)
        x1, y1 = left, top
        x2, y2 = left + width, top + height

        if x2 <= x1 or y2 <= y1:
            continue

        annotations.append({
            "class_name": class_name,
            "label":      class_name_to_idx(class_name),
            "bbox":       [x1, y1, x2, y2],
        })

    return annotations

def resize_transform(image: torch.Tensor,
                     target: Dict) -> Tuple[torch.Tensor, Dict]:
    """Resize so longest side ≤ 800px, scale boxes proportionally."""
    h, w  = image.shape[-2], image.shape[-1]
    scale = min(800 / max(h, w), 1.0)
    if scale < 1.0:
        new_h = max(int(h * scale), 1)
        new_w = max(int(w * scale), 1)
        image = TF.resize(image, [new_h, new_w])
        if len(target["boxes"]) > 0:
            target["boxes"] = target["boxes"] * scale
    return image, target


def augment_transform(image: torch.Tensor,
                      target: Dict) -> Tuple[torch.Tensor, Dict]:

    # Horizontal flip
    if random.random() > 0.5:
        w = image.shape[-1]
        image = TF.hflip(image)
        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] = w - target["boxes"][:, 2]
            boxes[:, 2] = w - target["boxes"][:, 0]
            target["boxes"] = boxes

    # Brightness / contrast jitter
    if random.random() > 0.3:
        image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
    if random.random() > 0.3:
        image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))

    # Gaussian noise (simulates scan variation)
    if random.random() > 0.5:
        noise = torch.randn_like(image) * 0.02
        image = (image + noise).clamp(0, 1)

    # Scale jitter
    if random.random() > 0.5:
        h, w  = image.shape[-2], image.shape[-1]
        scale = random.uniform(0.75, 1.25)
        new_h = max(int(h * scale), 100)
        new_w = max(int(w * scale), 100)
        image = TF.resize(image, [new_h, new_w])
        if len(target["boxes"]) > 0:
            target["boxes"] = target["boxes"] * scale

    return image, target

def _load_sample(image_path: str, annotations: List[Dict],
                 image_id: int, apply_resize: bool,
                 transform, min_box_area: int = 16) -> Tuple:
    """
    Load an image and build a Faster R-CNN compatible target dict.
    Shared by MUSCIMADataset and CVCMUSCIMADataset.
    """
    image        = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image)
    img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]

    boxes, labels = [], []
    for ann in annotations:
        x1, y1, x2, y2 = ann["bbox"]
        w = x2 - x1
        h = y2 - y1
        if w * h < min_box_area:
            continue
        # Drop boxes covering >70% of page — likely missed staff elements
        if w > img_w * 0.70 or h > img_h * 0.70:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(ann["label"])

    if boxes:
        boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
    else:
        boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,),   dtype=torch.int64)

    target = {
        "boxes":    boxes_t,
        "labels":   labels_t,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
    }

    if apply_resize:
        image_tensor, target = resize_transform(image_tensor, target)

    if transform is not None:
        image_tensor, target = transform(image_tensor, target)

    return image_tensor, target


class MUSCIMADataset(Dataset):

    TRAIN_WRITERS = list(range(1, 36))
    VAL_WRITERS   = list(range(36, 44))
    TEST_WRITERS  = list(range(44, 51))

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, min_box_area: int = 16,
                 apply_resize: bool = True):
        self.root_dir     = Path(root_dir)
        self.split        = split
        self.transform    = transform
        self.min_box_area = min_box_area
        self.apply_resize = apply_resize
        self.image_dir    = self.root_dir / "data" / "images"
        self.xml_dir      = self.root_dir / "data" / "cropobjects_withstaff"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.xml_dir.exists():
            raise FileNotFoundError(f"XML dir not found: {self.xml_dir}")

        self.samples = self._load_samples()
        print(f"[MUSCIMADataset] {split}: {len(self.samples)} images")

    def _writer_ids(self) -> List[int]:
        return {"train": self.TRAIN_WRITERS,
                "val":   self.VAL_WRITERS,
                "test":  self.TEST_WRITERS}[self.split]

    def _load_samples(self) -> List[Dict]:
        writer_ids = self._writer_ids()
        samples    = []
        for xml_file in sorted(self.xml_dir.glob("*.xml")):
            stem = xml_file.stem
            try:
                wpart     = [p for p in stem.split("_") if p.startswith("W-")][0]
                writer_id = int(wpart.split("-")[1])
            except (IndexError, ValueError):
                continue
            if writer_id not in writer_ids:
                continue
            image_path = self.image_dir / xml_file.with_suffix(".png").name
            if not image_path.exists():
                image_path = self.image_dir / xml_file.with_suffix(".jpg").name
            if not image_path.exists():
                continue
            anns = parse_cropobject_xml(str(xml_file))
            if not anns:
                continue
            samples.append({
                "image_path":  str(image_path),
                "annotations": anns,
                "image_id":    len(samples),
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return _load_sample(s["image_path"], s["annotations"],
                            s["image_id"], self.apply_resize,
                            self.transform, self.min_box_area)

class CVCMUSCIMADataset(Dataset):

    # Same writer split as MUSCIMADataset for consistency
    TRAIN_WRITERS = list(range(1, 36))
    VAL_WRITERS   = list(range(36, 44))
    TEST_WRITERS  = list(range(44, 51))

    def __init__(self, muscima_dir: str, cvc_dir: str,
                 split: str = "train", condition: str = "binary",
                 transform=None, min_box_area: int = 16,
                 apply_resize: bool = True):
        self.muscima_dir  = Path(muscima_dir)
        self.cvc_dir      = Path(cvc_dir)
        self.split        = split
        self.condition    = condition
        self.transform    = transform
        self.min_box_area = min_box_area
        self.apply_resize = apply_resize

        self.xml_dir      = self.muscima_dir / "data" / "cropobjects_withstaff"
        self.image_root   = self._find_image_root()

        if not self.xml_dir.exists():
            raise FileNotFoundError(f"XML dir not found: {self.xml_dir}")
        if not self.image_root.exists():
            raise FileNotFoundError(
                f"CVC-MUSCIMA image root not found: {self.image_root}\n"
                f"Expected: {self.image_root}")

        self.samples = self._load_samples()
        print(f"[CVCMUSCIMADataset] {split}/{condition}: "
              f"{len(self.samples)} images")

    def _find_image_root(self) -> Path:
        # Try direct
        direct = self.cvc_dir / self.condition
        if direct.exists():
            return direct
        # Try with the zip's top-level folder included
        nested = self.cvc_dir / "CVCMUSCIMA_MultiConditionAligned" / self.condition
        if nested.exists():
            return nested
        # Search one level deep
        for child in self.cvc_dir.iterdir():
            candidate = child / self.condition
            if candidate.exists():
                return candidate
        return self.cvc_dir / "CVCMUSCIMA_MultiConditionAligned" / self.condition

    def _writer_ids(self) -> List[int]:
        return {"train": self.TRAIN_WRITERS,
                "val":   self.VAL_WRITERS,
                "test":  self.TEST_WRITERS}[self.split]

    def _parse_xml_stem(self, stem: str):
        try:
            parts     = stem.split("_")
            writer_id = int([p for p in parts if p.startswith("W-")][0].split("-")[1])
            page_num  = int([p for p in parts if p.startswith("N-")][0].split("-")[1])
            return writer_id, page_num
        except (IndexError, ValueError):
            return None, None

    def _load_samples(self) -> List[Dict]:
        writer_ids = set(self._writer_ids())
        samples    = []

        for xml_file in sorted(self.xml_dir.glob("*.xml")):
            _, page_num = self._parse_xml_stem(xml_file.stem)
            if page_num is None:
                continue

            anns = parse_cropobject_xml(str(xml_file))
            if not anns:
                continue

            # Page filename: p001.png, p002.png, ...
            page_filename = f"p{page_num:03d}.png"

            # Find all writers in this split that have this page
            for writer_id in writer_ids:
                writer_folder = self.image_root / f"w-{writer_id:02d}"
                image_path    = writer_folder / page_filename

                if not image_path.exists():
                    continue

                samples.append({
                    "image_path":  str(image_path),
                    "annotations": anns,
                    "image_id":    len(samples),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return _load_sample(s["image_path"], s["annotations"],
                            s["image_id"], self.apply_resize,
                            self.transform, self.min_box_area)


# ---------------------------------------------------------------------------
# CombinedDataset — MUSCIMA++ + CVC-MUSCIMA together
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):

    def __init__(self, muscima_dir: str, cvc_dir: str,
                 split: str = "train", condition: str = "binary",
                 transform=None, min_box_area: int = 16,
                 apply_resize: bool = True):

        self.muscima_ds = MUSCIMADataset(
            muscima_dir, split=split, transform=transform,
            min_box_area=min_box_area, apply_resize=apply_resize)

        self.cvc_ds = CVCMUSCIMADataset(
            muscima_dir, cvc_dir, split=split, condition=condition,
            transform=transform, min_box_area=min_box_area,
            apply_resize=apply_resize)

        self._len_muscima = len(self.muscima_ds)
        self._len_total   = self._len_muscima + len(self.cvc_ds)

        print(f"[CombinedDataset] {split}: "
              f"{self._len_muscima} MUSCIMA++ + "
              f"{len(self.cvc_ds)} CVC-MUSCIMA = "
              f"{self._len_total} total")

    def __len__(self):
        return self._len_total

    def __getitem__(self, idx):
        if idx < self._len_muscima:
            return self.muscima_ds[idx]
        else:
            return self.cvc_ds[idx - self._len_muscima]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    muscima_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/muscima"
    cvc_dir     = sys.argv[2] if len(sys.argv) > 2 else "data/raw/cvc_muscima"

    print("=== MUSCIMADataset ===")
    ds = MUSCIMADataset(muscima_dir, split="train")
    img, tgt = ds[0]
    print(f"  Image : {img.shape}  Boxes: {len(tgt['boxes'])}")

    print("\n=== CVCMUSCIMADataset ===")
    cvc = CVCMUSCIMADataset(muscima_dir, cvc_dir, split="train")
    img, tgt = cvc[0]
    print(f"  Image : {img.shape}  Boxes: {len(tgt['boxes'])}")

    print("\n=== CombinedDataset ===")
    combined = CombinedDataset(muscima_dir, cvc_dir, split="train",
                               transform=augment_transform)
    img, tgt = combined[0]
    print(f"  Image : {img.shape}  Boxes: {len(tgt['boxes'])}")
    print(f"  Total samples: {len(combined)}")
