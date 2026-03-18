"""
Stage 2 Vocabulary
Defines the full music token vocabulary used by the transformer and stage 3.

Token schema:
    NOTE_<pitch>_<duration>    e.g. NOTE_C4_QUARTER, NOTE_F#5_EIGHTH
    REST_<duration>            e.g. REST_QUARTER, REST_WHOLE
    CLEF_<type>                e.g. CLEF_G, CLEF_F, CLEF_C
    TIME_<num>/<den>           e.g. TIME_4/4, TIME_3/4, TIME_6/8
    KEY_<sharps/flats>         e.g. KEY_0, KEY_1S, KEY_2F  (S=sharp, F=flat)
    ACCIDENTAL_<type>          e.g. ACCIDENTAL_SHARP, ACCIDENTAL_FLAT
    DYNAMIC_<type>             e.g. DYNAMIC_FORTE, DYNAMIC_PIANO
    BAR                        barline
    <PAD>, <SOS>, <EOS>, <UNK> special tokens
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Token construction helpers
# ---------------------------------------------------------------------------

PITCHES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES    = [2, 3, 4, 5, 6]
DURATIONS  = ["WHOLE", "HALF", "QUARTER", "EIGHTH", "SIXTEENTH", "DOTTED_HALF",
              "DOTTED_QUARTER", "DOTTED_EIGHTH"]
CLEF_TYPES = ["G", "F", "C"]
DYNAMICS   = ["FORTE", "PIANO", "MEZZO_FORTE", "MEZZO_PIANO",
              "FORTISSIMO", "PIANISSIMO", "CRESCENDO", "DECRESCENDO",
              "SFORZANDO", "FORTE_PIANO"]
# Time signatures: (numerator, denominator)
TIME_SIGS  = ["2/4", "3/4", "4/4", "6/8", "9/8", "12/8", "2/2", "3/8"]
# Key signatures: 0 = C major, 1S..7S = sharps, 1F..7F = flats
KEY_SIGS   = (["KEY_0"] +
              [f"KEY_{i}S" for i in range(1, 8)] +
              [f"KEY_{i}F" for i in range(1, 8)])


def _build_vocab() -> List[str]:
    tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "BAR"]

    # Notes: all pitch × octave × duration combinations
    for pitch in PITCHES:
        for octave in OCTAVES:
            for dur in DURATIONS:
                tokens.append(f"NOTE_{pitch}{octave}_{dur}")

    # Rests
    for dur in DURATIONS:
        tokens.append(f"REST_{dur}")

    # Clefs
    for c in CLEF_TYPES:
        tokens.append(f"CLEF_{c}")

    # Time signatures
    for ts in TIME_SIGS:
        tokens.append(f"TIME_{ts}")

    # Key signatures
    tokens.extend(KEY_SIGS)

    # Accidentals
    for acc in ["SHARP", "FLAT", "NATURAL"]:
        tokens.append(f"ACCIDENTAL_{acc}")

    # Dynamics
    for dyn in DYNAMICS:
        tokens.append(f"DYNAMIC_{dyn}")

    return tokens


VOCAB: List[str]      = _build_vocab()
VOCAB_SIZE: int       = len(VOCAB)
TOKEN_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_TOKEN: Dict[int, str] = {i: tok for i, tok in enumerate(VOCAB)}

PAD_IDX = TOKEN_TO_IDX["<PAD>"]
SOS_IDX = TOKEN_TO_IDX["<SOS>"]
EOS_IDX = TOKEN_TO_IDX["<EOS>"]
UNK_IDX = TOKEN_TO_IDX["<UNK>"]


def token_to_idx(token: str) -> int:
    return TOKEN_TO_IDX.get(token, UNK_IDX)


def idx_to_token(idx: int) -> str:
    return IDX_TO_TOKEN.get(idx, "<UNK>")


# ---------------------------------------------------------------------------
# MUSCIMA++ class → music token mapping
# Maps stage 1 detected symbol labels to their token representation.
# Note: pitch cannot be inferred from class alone — it requires staff position.
# These are used for structural/non-pitched tokens only.
# ---------------------------------------------------------------------------

MUSCIMA_CLASS_TO_TOKEN: Dict[str, str] = {
    "bar-line":          "BAR",
    "clef-G":            "CLEF_G",
    "clef-F":            "CLEF_F",
    "clef-C":            "CLEF_C",
    "time-signature":    "TIME_4/4",   # refined by reading the actual symbol
    "key-signature":     "KEY_0",      # refined by counting sharps/flats
    "accidental-sharp":  "ACCIDENTAL_SHARP",
    "accidental-flat":   "ACCIDENTAL_FLAT",
    "accidental-natural":"ACCIDENTAL_NATURAL",
    "dynamic-forte":     "DYNAMIC_FORTE",
    "dynamic-piano":     "DYNAMIC_PIANO",
    # noteheads and rests require staff position / duration context
    # — handled separately in dataset.py
}


if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Sample tokens  : {VOCAB[:10]}")
    print(f"Note tokens    : {[t for t in VOCAB if t.startswith('NOTE_C4')][:5]}")
