
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset

from src.stage2_sequencing.vocabulary import (
    TOKEN_TO_IDX, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, VOCAB_SIZE,
)

MAX_SEQ_LEN = 512


PRIMUS_PITCH_MAP = {
    "Cb": "B",  "Db": "C#", "Eb": "D#", "Fb": "E",
    "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "C": "C",   "D": "D",   "E": "E",   "F": "F",
    "G": "G",   "A": "A",   "B": "B",
    "C#": "C#", "D#": "D#", "F#": "F#", "G#": "G#", "A#": "A#",
}

# PrIMuS duration name → our duration token
PRIMUS_DURATION_MAP = {
    "whole":           "WHOLE",
    "half":            "HALF",
    "quarter":         "QUARTER",
    "eighth":          "EIGHTH",
    "sixteenth":       "SIXTEENTH",
    "thirty_second":   "SIXTEENTH",  # map to closest
    "dotted_whole":    "WHOLE",       # simplify
    "dotted_half":     "DOTTED_HALF",
    "dotted_quarter":  "DOTTED_QUARTER",
    "dotted_eighth":   "DOTTED_EIGHTH",
    "dotted_sixteenth":"SIXTEENTH",
}

# PrIMuS clef names → our clef tokens
PRIMUS_CLEF_MAP = {
    "clef-G2": "CLEF_G", "clef-G1": "CLEF_G",
    "clef-F4": "CLEF_F", "clef-F3": "CLEF_F",
    "clef-C1": "CLEF_C", "clef-C2": "CLEF_C",
    "clef-C3": "CLEF_C", "clef-C4": "CLEF_C",
}

PRIMUS_KEY_MAP = {
    "keySignature-CM":  "KEY_0",
    "keySignature-Am":  "KEY_0",
    "keySignature-GM":  "KEY_1S",
    "keySignature-Em":  "KEY_1S",
    "keySignature-DM":  "KEY_2S",
    "keySignature-Bm":  "KEY_2S",
    "keySignature-AM":  "KEY_3S",
    "keySignature-F#m": "KEY_3S",
    "keySignature-EM":  "KEY_4S",
    "keySignature-C#m": "KEY_4S",
    "keySignature-BM":  "KEY_5S",
    "keySignature-G#m": "KEY_5S",
    "keySignature-F#M": "KEY_6S",
    "keySignature-D#m": "KEY_6S",
    "keySignature-C#M": "KEY_7S",
    "keySignature-A#m": "KEY_7S",
    "keySignature-FM":  "KEY_1F",
    "keySignature-Dm":  "KEY_1F",
    "keySignature-BbM": "KEY_2F",
    "keySignature-Gm":  "KEY_2F",
    "keySignature-EbM": "KEY_3F",
    "keySignature-Cm":  "KEY_3F",
    "keySignature-AbM": "KEY_4F",
    "keySignature-Fm":  "KEY_4F",
    "keySignature-DbM": "KEY_5F",
    "keySignature-Bbm": "KEY_5F",
    "keySignature-GbM": "KEY_6F",
    "keySignature-Ebm": "KEY_6F",
    "keySignature-CbM": "KEY_7F",
    "keySignature-Abm": "KEY_7F",
}


def primus_token_to_vocab(token: str) -> Optional[str]:

    token = token.strip()
    if not token:
        return None

    # Barline
    if token == "barline":
        return "BAR"

    # Clef
    if token in PRIMUS_CLEF_MAP:
        return PRIMUS_CLEF_MAP[token]

    # Key signature
    if token in PRIMUS_KEY_MAP:
        return PRIMUS_KEY_MAP[token]
    if token.startswith("keySignature-"):
        return "KEY_0"  # fallback for unknown keys

    # Time signature: timeSignature-4/4, timeSignature-3/4, etc.
    if token.startswith("timeSignature-"):
        ts = token.replace("timeSignature-", "")
        vocab_tok = f"TIME_{ts}"
        if vocab_tok in TOKEN_TO_IDX:
            return vocab_tok
        return None  # skip unsupported time signatures

    # Rest: rest-quarter, rest-half, etc.
    if token.startswith("rest-"):
        dur_name = token.replace("rest-", "")
        dur = PRIMUS_DURATION_MAP.get(dur_name)
        if dur:
            vocab_tok = f"REST_{dur}"
            if vocab_tok in TOKEN_TO_IDX:
                return vocab_tok
        return None

    # Note: note-E4_quarter, note-Bb4_eighth, note-F#5_half
    if token.startswith("note-"):
        body = token.replace("note-", "")
        if "_" not in body:
            return None
        pitch_part, dur_part = body.rsplit("_", 1)

        octave_start = -1
        for i, ch in enumerate(pitch_part):
            if ch.isdigit() or (ch == '-' and i > 0):
                octave_start = i
                break

        if octave_start <= 0:
            return None

        pitch_name = pitch_part[:octave_start]
        octave_str = pitch_part[octave_start:]

        try:
            octave = int(octave_str)
        except ValueError:
            return None

        # Normalize pitch name
        norm_pitch = PRIMUS_PITCH_MAP.get(pitch_name)
        if norm_pitch is None:
            return None

        # Map duration
        dur = PRIMUS_DURATION_MAP.get(dur_part)
        if dur is None:
            return None

        vocab_tok = f"NOTE_{norm_pitch}{octave}_{dur}"
        if vocab_tok in TOKEN_TO_IDX:
            return vocab_tok
        # Try with octave clipped to our range
        for oct in [4, 5, 3]:
            fallback = f"NOTE_{norm_pitch}{oct}_{dur}"
            if fallback in TOKEN_TO_IDX:
                return fallback
        return None

    # Accidental
    if token == "accidental-sharp":
        return "ACCIDENTAL_SHARP"
    if token == "accidental-flat":
        return "ACCIDENTAL_FLAT"
    if token == "accidental-natural":
        return "ACCIDENTAL_NATURAL"

    # Dynamic
    if token == "dynamic-f":
        return "DYNAMIC_FORTE"
    if token == "dynamic-p":
        return "DYNAMIC_PIANO"
    if token == "dynamic-mf":
        return "DYNAMIC_MEZZO_FORTE"
    if token == "dynamic-mp":
        return "DYNAMIC_MEZZO_PIANO"

    # Skip everything else (multirest, tie, slur, fermata, etc.)
    return None


def parse_semantic_line(line: str) -> List[str]:

    raw_tokens = line.strip().split()
    result = []
    for raw in raw_tokens:
        mapped = primus_token_to_vocab(raw)
        if mapped is not None:
            result.append(mapped)
    return result


class PrIMuSDataset(Dataset):

    def __init__(self, root_dir: str, split: str = "train",
                 max_seq_len: int = MAX_SEQ_LEN, seed: int = 42):
        self.root_dir    = Path(root_dir)
        self.split       = split
        self.max_seq_len = max_seq_len

        self.samples = self._load_samples(seed)
        print(f"[PrIMuSDataset] {split}: {len(self.samples)} samples loaded")

    def _load_samples(self, seed: int) -> List[Dict]:

        semantic_files = sorted(self.root_dir.rglob("*.semantic"))
        if not semantic_files:
            raise FileNotFoundError(
                f"No .semantic files found in {self.root_dir}\n"
                f"Make sure PrIMuS is extracted correctly.")

        # Reproducible shuffle then split
        random.seed(seed)
        indices = list(range(len(semantic_files)))
        random.shuffle(indices)

        n      = len(indices)
        n_val  = int(n * 0.075)
        n_test = int(n * 0.075)
        n_train = n - n_val - n_test

        if self.split == "train":
            split_indices = indices[:n_train]
        elif self.split == "val":
            split_indices = indices[n_train:n_train + n_val]
        else:
            split_indices = indices[n_train + n_val:]

        samples = []
        for i in split_indices:
            path = semantic_files[i]
            try:
                line = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not line:
                continue

            tokens = parse_semantic_line(line)
            # Skip sequences that are too short to be useful
            if len(tokens) < 3:
                continue

            samples.append({"tokens": tokens, "path": str(path)})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens  = self.samples[idx]["tokens"]
        tok_ids = [TOKEN_TO_IDX.get(t, UNK_IDX) for t in tokens]

        # Truncate to leave room for SOS/EOS
        tok_ids = tok_ids[:self.max_seq_len - 1]

        input_ids  = [SOS_IDX] + tok_ids
        target_ids = tok_ids + [EOS_IDX]

        def pad(seq):
            seq = seq[:self.max_seq_len]
            return seq + [PAD_IDX] * (self.max_seq_len - len(seq))

        seq_len      = len(input_ids)
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        padding_mask[:seq_len] = False

        return {
            "input_ids":    torch.tensor(pad(input_ids),  dtype=torch.int64),
            "target_ids":   torch.tensor(pad(target_ids), dtype=torch.int64),
            "padding_mask": padding_mask,
        }


def collate_fn(batch):
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/primus"
    ds = PrIMuSDataset(root, split="train")
    sample = ds[0]
    print(f"\nVocab size      : {VOCAB_SIZE}")
    print(f"Dataset size    : {len(ds)}")
    for k, v in sample.items():
        print(f"  {k:20s}: shape={v.shape}  dtype={v.dtype}")

    # Show decoded tokens for first sample
    ids = sample["input_ids"].tolist()
    from src.stage2_sequencing.vocabulary import IDX_TO_TOKEN
    tokens = [IDX_TO_TOKEN.get(i, "?") for i in ids if i != PAD_IDX]
    print(f"\nFirst sample tokens ({len(tokens)}):")
    print("  " + " ".join(tokens[:20]))
