"""
nefs_espeak_bootstrap.py — eSpeak MBROLA bootstrap target extractor

Extracts per-phoneme prosody targets (F0, duration, rate_factor) from
eSpeak's MBROLA output format, without requiring LJSpeech audio or
forced alignment.

Why this is useful as Phase 1 training data
--------------------------------------------
eSpeak's prosody is rule-based and robotic — that's exactly the problem
we're solving. But its outputs have two important properties:

  1. They are deterministic and consistent. The same input always
     produces the same targets, so the predictor has a stable signal
     to learn from.

  2. They define a reasonable prior distribution. F0 contours are
     in the right ballpark (80–200 Hz for a male voice), durations
     are linguistically plausible (stressed vowels longer, consonants
     shorter), and stress is marked correctly.

The neural predictor trained on bootstrap targets will produce output
that sounds like eSpeak — still robotic, but correctly aligned to the
NEFS byte sequence. Phase 2 audio-supervised fine-tuning then pulls
the outputs toward natural speech using LJSpeech reference audio.

MBROLA .pho format
------------------
Each line is one of:
  _ <duration_ms>                    — silence
  <phoneme> <duration_ms>            — unvoiced phoneme
  <phoneme> <duration_ms> [t f0 ...]  — voiced phoneme with F0 envelope
    where t = position in phoneme as percentage (0–100)
    and f0 = F0 value in Hz at that position

Example:
  h    70
  @    24    0 94  20 95  40 96  59 97  80 99  100 99
  l    65
  @U   61    0 117 80 109 100 109

MBROLA en1 → NEFS byte mapping
--------------------------------
eSpeak's mb-en1 voice uses a custom phoneme symbol set. This file
contains the complete mapping from those symbols to NEFS bytes, derived
from the NEFSConverter IPA mappings and the MBROLA en1 phoneme
documentation.

Usage
-----
  # Single transcript
  python nefs_espeak_bootstrap.py --text "hello world"

  # Full LJSpeech dataset → cache bootstrap targets
  python nefs_espeak_bootstrap.py --data-dir ./LJSpeech-1.1 --output-dir ./bootstrap_cache

  # Use as dataset in training loop
  from nefs_espeak_bootstrap import BootstrapDataset, bootstrap_collate
  ds = BootstrapDataset('./LJSpeech-1.1', output_dir='./bootstrap_cache')
"""

from __future__ import annotations

import csv
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# eSpeak binary
# ---------------------------------------------------------------------------

def _find_espeak() -> str:
    if sys.platform == "win32":
        for p in [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
        ]:
            if os.path.isfile(p):
                return p
    return "espeak-ng"

ESPEAK_BIN = _find_espeak()

def espeak_available() -> bool:
    try:
        r = subprocess.run([ESPEAK_BIN, "--version"], capture_output=True, timeout=3)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def mbrola_available() -> bool:
    """Check that mb-en1 voice is installed."""
    try:
        r = subprocess.run(
            [ESPEAK_BIN, "--pho", "-v", "mb-en1", "test"],
            capture_output=True, timeout=3
        )
        return r.returncode == 0 and len(r.stdout) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MBROLA en1 → NEFS byte mapping
# ---------------------------------------------------------------------------

# Complete mapping from MBROLA en1 phoneme symbols to NEFS bytes.
# Sources: MBROLA en1 documentation + NEFSConverter IPA mappings.
# Diphthongs map to the nucleus (most perceptually salient vowel).
MBROLA_EN1_TO_NEFS: Dict[str, int] = {
    # Vowels — spec v3: high nibble = col (0xA_-0xF_), low nibble = row
    # Near-close/central row 0x_8
    "i:": 0xA4,   # /iː/  FLEECE     → i   (0xA4)
    "I":  0xA8,   # /ɪ/   KIT        → ɪ   (0xA8)
    "e":  0xA6,   # /ɛ/   DRESS      → ɛ   (0xA6, MBROLA 'e' = /ɛ/)
    "{":  0xC8,   # /æ/   TRAP       → æ   (0xC8)
    "@":  0xF8,   # /ə/   schwa      → ə   (0xF8)
    "V":  0xE6,   # /ʌ/   STRUT      → ʌ   (0xE6)
    "Q":  0xF6,   # /ɒ/   LOT        → ɔ   (0xF6, closest match)
    "O:": 0xF6,   # /ɔː/  THOUGHT    → ɔ   (0xF6)
    "u:": 0xF4,   # /uː/  GOOSE      → u   (0xF4)
    "U":  0xE8,   # /ʊ/   FOOT       → ʊ   (0xE8)
    "3:": 0xC6,   # /ɜː/  NURSE      → ɜ   (0xC6)
    "A:": 0xE7,   # /ɑː/  START      → ɑ   (0xE7)
    "a":  0xA7,   # /a/   TRAP (alt) → a   (0xA7)

    # Diphthongs — map to nucleus vowel
    "eI": 0xA5,   # /eɪ/  FACE       → e   (0xA5)
    "@U": 0xF5,   # /əʊ/  GOAT       → o   (0xF5)
    "aI": 0xA7,   # /aɪ/  PRICE      → a   (0xA7)
    "aU": 0xA7,   # /aʊ/  MOUTH      → a   (0xA7)
    "OI": 0xF6,   # /ɔɪ/  CHOICE     → ɔ   (0xF6)
    "I@": 0xA8,   # /ɪə/  NEAR       → ɪ   (0xA8)
    "e@": 0xA6,   # /eə/  SQUARE     → ɛ   (0xA6)
    "U@": 0xE8,   # /ʊə/  CURE       → ʊ   (0xE8)

    # Voiceless stops — row 0x_3, cols: B=0x1_, A=0x4_, V=0x8_
    "p":  0x13,   # p  (0x13)
    "t":  0x43,   # t  (0x43)
    "k":  0x83,   # k  (0x83)

    # Voiced stops — row 0x_2
    "b":  0x12,   # b  (0x12)
    "d":  0x42,   # d  (0x42)
    "g":  0x82,   # g  (0x82)

    # Affricates — two-byte sequences (handled separately in encoder)
    "tS": None,   # tʃ → [0x43, 0x51]
    "dZ": None,   # dʒ → [0x42, 0x50]

    # Voiceless fricatives — row 0x_1
    "f":  0x21,   # f  labiodental  (0x21)
    "T":  0x31,   # θ  dental       (0x31)
    "s":  0x41,   # s  alveolar     (0x41)
    "S":  0x51,   # ʃ  post-alveolar(0x51)
    "h":  0xC1,   # h  glottal      (0xC1)

    # Voiced fricatives — row 0x_0
    "v":  0x20,   # v  labiodental  (0x20)
    "D":  0x30,   # ð  dental       (0x30)
    "z":  0x40,   # z  alveolar     (0x40)
    "Z":  0x50,   # ʒ  post-alveolar(0x50)

    # Nasals — row 0x_8, cols: B=0x1_, A=0x4_, V=0x8_
    "m":  0x18,   # m  bilabial     (0x18)
    "n":  0x48,   # n  alveolar     (0x48)
    "N":  0x88,   # ŋ  velar        (0x88)

    # Approximants — row 0x_9
    "r":  0x49,   # ɹ  alveolar     (0x49)
    "w":  0x19,   # w  bilabial     (0x19)
    "j":  0x79,   # j  palatal      (0x79)

    # Laterals — row 0x_A
    "l":  0x4A,   # l  alveolar     (0x4A)
    "5":  0x4A,   # ɫ  dark-l (MBROLA '5') → l (0x4A, closest)
}{
    # Vowels — monophthongs
    "i:": 0x49,   # /iː/  FLEECE
    "I":  0x89,   # /ɪ/   KIT
    "e":  0x69,   # /ɛ/   DRESS  (MBROLA uses 'e' for /ɛ/)
    "{":  0x8B,   # /æ/   TRAP   (MBROLA uses '{')
    "@":  0x8E,   # /ə/   schwa
    "V":  0x6D,   # /ʌ/   STRUT
    "Q":  0x6E,   # /ɒ/→/ɔ/ LOT/CLOTH
    "O:": 0x6E,   # /ɔː/  THOUGHT
    "u:": 0x4E,   # /uː/  GOOSE
    "U":  0x8D,   # /ʊ/   FOOT
    "3:": 0x6B,   # /ɜː/  NURSE  (maps to ɜ)
    "A:": 0x7D,   # /ɑː/  START

    # Vowels — diphthongs (map to nucleus)
    "eI": 0x59,   # /eɪ/  FACE   (nucleus = e)
    "@U": 0x5E,   # /əʊ/  GOAT   (nucleus = o)
    "aI": 0x79,   # /aɪ/  PRICE  (nucleus = a)
    "aU": 0x79,   # /aʊ/  MOUTH  (nucleus = a)
    "OI": 0x6E,   # /ɔɪ/  CHOICE (nucleus = ɔ)
    "I@": 0x89,   # /ɪə/  NEAR   (nucleus = ɪ)
    "e@": 0x69,   # /eə/  SQUARE (nucleus = ɛ)
    "U@": 0x8D,   # /ʊə/  CURE   (nucleus = ʊ)

    # Plosives — voiceless
    "p":  0x40,
    "t":  0x43,
    "k":  0x47,

    # Plosives — voiced
    "b":  0x20,
    "d":  0x23,
    "g":  0x27,

    # Affricates (two-byte sequences in NEFS)
    "tS": None,   # tʃ → handled separately as [0x43, 0x34]
    "dZ": None,   # dʒ → handled separately as [0x23, 0x14]

    # Fricatives — voiceless
    "f":  0x31,
    "T":  0x32,   # θ
    "s":  0x33,
    "S":  0x34,   # ʃ
    "h":  0x3B,

    # Fricatives — voiced
    "v":  0x11,
    "D":  0x12,   # ð
    "z":  0x13,
    "Z":  0x14,   # ʒ

    # Nasals
    "m":  0x90,
    "n":  0x93,
    "N":  0x97,   # ŋ

    # Approximants
    "l":  0xB3,
    "5":  0xB3,   # dark/syllabic l (MBROLA en1 symbol)
    "r":  0xA3,   # ɹ
    "w":  0xA0,
    "j":  0xA6,
}

# Two-byte affricates
MBROLA_AFFRICATES: Dict[str, List[int]] = {
    "tS": [0x43, 0x34],
    "dZ": [0x23, 0x14],
}


# ---------------------------------------------------------------------------
# MBROLA output parser
# ---------------------------------------------------------------------------

@dataclass
class MBROLAPhoneme:
    symbol: str
    duration_ms: float
    f0_envelope: List[Tuple[float, float]]  # (position_pct, f0_hz) pairs

    @property
    def mean_f0(self) -> float:
        """Mean F0 across envelope points. 0.0 if no F0 data (unvoiced)."""
        if not self.f0_envelope:
            return 0.0
        return float(np.mean([hz for _, hz in self.f0_envelope]))

    @property
    def is_silence(self) -> bool:
        return self.symbol == "_"

    @property
    def is_voiced(self) -> bool:
        return bool(self.f0_envelope)


def parse_mbrola_output(pho_text: str) -> List[MBROLAPhoneme]:
    """
    Parse eSpeak's MBROLA .pho format into a list of MBROLAPhoneme objects.

    Format per line:
        <symbol>  <duration_ms>  [t1 f0_1  t2 f0_2  ...]
    """
    phonemes = []
    for line in pho_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        symbol = parts[0]
        try:
            duration_ms = float(parts[1])
        except ValueError:
            continue

        # F0 envelope: remaining columns are t f0 pairs
        f0_envelope = []
        remainder = parts[2:]
        for i in range(0, len(remainder) - 1, 2):
            try:
                t   = float(remainder[i])
                f0  = float(remainder[i + 1])
                f0_envelope.append((t, f0))
            except (ValueError, IndexError):
                break

        phonemes.append(MBROLAPhoneme(
            symbol=symbol,
            duration_ms=duration_ms,
            f0_envelope=f0_envelope,
        ))

    return phonemes


# ---------------------------------------------------------------------------
# eSpeak runner
# ---------------------------------------------------------------------------

def run_espeak_mbrola(text: str, lang: str = "mb-en1") -> Optional[str]:
    """
    Run eSpeak with MBROLA output on a text string.

    Returns the raw .pho text, or None on failure.
    """
    try:
        result = subprocess.run(
            [ESPEAK_BIN, "--pho", "-v", lang, text],
            capture_output=True,
            timeout=5.0,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(f"eSpeak failed: {result.stderr}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("eSpeak timed out")
        return None
    except Exception as e:
        logger.warning(f"eSpeak error: {e}")
        return None


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

@dataclass
class BootstrapTargets:
    """Per-phoneme targets extracted from eSpeak MBROLA output."""
    nefs_bytes:  np.ndarray   # uint8,   shape (N,)
    f0_hz:       np.ndarray   # float32, shape (N,)  — 0.0 if unvoiced
    duration_s:  np.ndarray   # float32, shape (N,)
    f0_mask:     np.ndarray   # bool,    shape (N,)  — True where voiced
    rate_factor: np.ndarray   # float32, shape (N,)  — relative to class mean


def mbrola_to_targets(phonemes: List[MBROLAPhoneme]) -> Optional[BootstrapTargets]:
    """
    Convert a list of MBROLAPhoneme objects to BootstrapTargets.

    Maps MBROLA symbols to NEFS bytes, expands affricates (one MBROLA
    symbol → two NEFS bytes), and computes rate_factor relative to the
    mean duration of each phoneme class.
    """
    nefs_bytes  = []
    f0_values   = []
    durations   = []
    voiced_mask = []

    for ph in phonemes:
        if ph.is_silence:
            continue

        sym = ph.symbol

        # Handle affricates — one MBROLA phoneme → two NEFS bytes
        if sym in MBROLA_AFFRICATES:
            seq = MBROLA_AFFRICATES[sym]
            for b in seq:
                nefs_bytes.append(b)
                f0_values.append(ph.mean_f0)
                # Split duration evenly across the two bytes
                durations.append(ph.duration_ms / 2.0 / 1000.0)
                voiced_mask.append(ph.is_voiced)
            continue

        # Regular phoneme
        nefs_byte = MBROLA_EN1_TO_NEFS.get(sym)
        if nefs_byte is None:
            # Unknown symbol — skip with warning
            logger.debug(f"Unknown MBROLA symbol: '{sym}'")
            continue

        nefs_bytes.append(nefs_byte)
        f0_values.append(ph.mean_f0)
        durations.append(ph.duration_ms / 1000.0)
        voiced_mask.append(ph.is_voiced)

    if len(nefs_bytes) == 0:
        return None

    nefs_arr  = np.array(nefs_bytes,  dtype=np.uint8)
    f0_arr    = np.array(f0_values,   dtype=np.float32)
    dur_arr   = np.array(durations,   dtype=np.float32)
    mask_arr  = np.array(voiced_mask, dtype=bool)

    # Rate factor: duration relative to the global mean across this utterance.
    # Using utterance-mean rather than a fixed reference makes the predictor
    # learn relative timing (stressed vs unstressed) rather than absolute ms,
    # which generalises better across speaking rates.
    mean_dur = dur_arr.mean()
    rate_arr = np.ones_like(dur_arr)
    if mean_dur > 0:
        # rate_factor < 1 means faster than average (short phoneme)
        # rate_factor > 1 means slower than average (long/stressed phoneme)
        rate_arr = dur_arr / mean_dur
        # Clip to the range the predictor outputs [0.4, 2.0]
        rate_arr = np.clip(rate_arr, 0.4, 2.0).astype(np.float32)

    return BootstrapTargets(
        nefs_bytes=nefs_arr,
        f0_hz=f0_arr,
        duration_s=dur_arr,
        f0_mask=mask_arr,
        rate_factor=rate_arr,
    )


def extract_bootstrap_targets(
    text: str,
    lang: str = "mb-en1",
) -> Optional[BootstrapTargets]:
    """
    Full pipeline: text → eSpeak MBROLA → BootstrapTargets.

    Returns None if eSpeak is unavailable or the text produces no phonemes.
    """
    pho_text = run_espeak_mbrola(text, lang=lang)
    if pho_text is None:
        return None

    phonemes = parse_mbrola_output(pho_text)
    if not phonemes:
        return None

    return mbrola_to_targets(phonemes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BootstrapDataset(Dataset):
    """
    LJSpeech dataset with eSpeak bootstrap prosody targets.

    Faster than LJSpeechProsodyDataset (no audio loading or pyin).
    Use this for Phase 1 training when LJSpeech audio is available
    but you want to iterate quickly, or when audio is not yet available.

    Cached to disk so eSpeak only runs once per transcript.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        lang: str = "mb-en1",
        max_samples: Optional[int] = None,
    ):
        if not espeak_available():
            raise RuntimeError(
                "espeak-ng not found. Install with:\n"
                "  Linux: sudo apt install espeak-ng mbrola mbrola-en1\n"
                "  macOS: brew install espeak-ng\n"
                "  Windows: https://github.com/espeak-ng/espeak-ng/releases"
            )
        if not mbrola_available():
            raise RuntimeError(
                "MBROLA en1 voice not found. Install with:\n"
                "  sudo apt install mbrola mbrola-en1"
            )

        self.data_dir   = Path(data_dir)
        self.lang       = lang
        self.cache_dir  = Path(output_dir) if output_dir else self.data_dir / "bootstrap_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load LJSpeech metadata
        self.samples: List[Tuple[str, str]] = []
        metadata_path = self.data_dir / "metadata.csv"
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    self.samples.append((parts[0], parts[1]))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(
            f"BootstrapDataset: {len(self.samples)} samples, "
            f"cache at {self.cache_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        name, transcript = self.samples[idx]
        cache_path = self.cache_dir / f"{name}.pt"

        # Load from cache if available
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=True)
            except Exception:
                cache_path.unlink(missing_ok=True)

        # Extract from eSpeak
        targets = extract_bootstrap_targets(transcript, lang=self.lang)
        if targets is None:
            return None

        n = len(targets.nefs_bytes)
        item = {
            "nefs_bytes":  torch.tensor(targets.nefs_bytes.astype(np.int64)),
            "f0_hz":       torch.tensor(targets.f0_hz),
            "duration_s":  torch.tensor(targets.duration_s),
            "f0_mask":     torch.tensor(targets.f0_mask),
            "rate_factor": torch.tensor(targets.rate_factor),
            "nefs_mask":   torch.ones(n, dtype=torch.bool),
            "nefs_lengths":torch.tensor(n, dtype=torch.long),
        }

        torch.save(item, cache_path)
        return item


def bootstrap_collate(
    batch: List[Optional[Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """Pad and batch bootstrap items. Filters None (failed extractions)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        dummy = torch.zeros(1, 1, dtype=torch.long)
        return {
            "nefs_bytes":   dummy,
            "nefs_mask":    torch.zeros(1, 1, dtype=torch.bool),
            "nefs_lengths": torch.ones(1, dtype=torch.long),
            "f0_hz":        torch.zeros(1, 1),
            "f0_mask":      torch.zeros(1, 1, dtype=torch.bool),
            "duration_s":   torch.ones(1, 1) * 0.08,
            "rate_factor":  torch.ones(1, 1),
        }

    max_len = max(b["nefs_bytes"].size(0) for b in batch)

    def pad(t, val=0):
        return F.pad(t, (0, max_len - t.size(0)), value=val)

    return {
        "nefs_bytes":   torch.stack([pad(b["nefs_bytes"])   for b in batch]),
        "nefs_mask":    torch.stack([pad(b["nefs_mask"])    for b in batch]),
        "nefs_lengths": torch.stack([b["nefs_lengths"]      for b in batch]),
        "f0_hz":        torch.stack([pad(b["f0_hz"])        for b in batch]),
        "f0_mask":      torch.stack([pad(b["f0_mask"])      for b in batch]),
        "duration_s":   torch.stack([pad(b["duration_s"])   for b in batch]),
        "rate_factor":  torch.stack([pad(b["rate_factor"],1.0) for b in batch]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="eSpeak bootstrap target extractor")
    parser.add_argument("--text",        type=str,  default=None)
    parser.add_argument("--data-dir",    type=Path, default=None)
    parser.add_argument("--output-dir",  type=Path, default=None)
    parser.add_argument("--max-samples", type=int,  default=20)
    args = parser.parse_args()

    print(f"eSpeak available:  {espeak_available()}")
    print(f"MBROLA available:  {mbrola_available()}")
    print()

    if args.text:
        # Single transcript demo
        pho = run_espeak_mbrola(args.text)
        if pho:
            print(f"MBROLA output for '{args.text}':")
            print(pho)
            phonemes = parse_mbrola_output(pho)
            targets  = mbrola_to_targets(phonemes)
            if targets is not None:
                print(f"\nExtracted {len(targets.nefs_bytes)} NEFS bytes:")
                for i, (b, f0, dur, v, rate) in enumerate(zip(
                    targets.nefs_bytes, targets.f0_hz,
                    targets.duration_s, targets.f0_mask,
                    targets.rate_factor
                )):
                    from nefs_converter import NEFSConverter
                    ipa = NEFSConverter().nafs_to_ipa(bytes([b]))
                    print(f"  [{i}] 0x{b:02X} /{ipa}/  "
                          f"f0={f0:.0f}Hz  dur={dur*1000:.0f}ms  "
                          f"voiced={v}  rate={rate:.2f}x")

    elif args.data_dir:
        # Dataset extraction
        ds = BootstrapDataset(
            args.data_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
        f0_all, dur_all, rate_all = [], [], []
        ok = 0
        for i in range(len(ds)):
            item = ds[i]
            if item is None:
                continue
            ok += 1
            voiced = item["f0_mask"]
            f0_all.extend(item["f0_hz"][voiced].tolist())
            dur_all.extend(item["duration_s"].tolist())
            rate_all.extend(item["rate_factor"].tolist())

        print(f"Extracted {ok}/{len(ds)} samples successfully")
        if f0_all:
            print(f"F0 range:        {min(f0_all):.0f}–{max(f0_all):.0f} Hz")
            print(f"F0 median:       {float(np.median(f0_all)):.0f} Hz")
        if dur_all:
            print(f"Duration range:  {min(dur_all)*1000:.0f}–{max(dur_all)*1000:.0f} ms")
            print(f"Duration mean:   {float(np.mean(dur_all))*1000:.0f} ms")
        if rate_all:
            print(f"Rate factor:     {min(rate_all):.2f}–{max(rate_all):.2f}x")
    else:
        # Default demo
        for text in ["hello world", "the cat sat on the mat", "she sells seashells"]:
            t = extract_bootstrap_targets(text)
            if t is not None:
                voiced_f0 = t.f0_hz[t.f0_mask]
                print(f"'{text}'")
                print(f"  phonemes={len(t.nefs_bytes)}  "
                      f"voiced={t.f0_mask.sum()}/{len(t.f0_mask)}  "
                      f"f0={np.median(voiced_f0):.0f}Hz  "
                      f"dur_mean={t.duration_s.mean()*1000:.0f}ms")
