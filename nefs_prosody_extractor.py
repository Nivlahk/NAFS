"""
nefs_prosody_extractor.py — Extract prosody targets from LJSpeech audio

Produces per-phoneme (f0_hz, duration_s, f0_mask) targets for training
NEFSParamPredictor, without requiring MFA forced alignment.

Pipeline per utterance
----------------------
1. G2P: transcript → IPA via eSpeak
2. Convert: IPA → NEFS bytes via NEFSConverter
3. Extract: F0 and voicing from audio via pyin (librosa)
4. Align: audio voiced/unvoiced pattern → NEFS voicing pattern via DTW
5. Segment: derive per-phoneme boundaries from alignment
6. Aggregate: mean F0 and duration per phoneme segment

Why DTW alignment works here
-----------------------------
NEFS bit 0 encodes voicing for every phoneme. The audio F0 extractor
produces a voiced/unvoiced flag per frame. These two binary sequences
share the same underlying voicing pattern — vowels voiced, most
fricatives unvoiced, stop closures silent — so DTW can warp one onto
the other to find phoneme boundaries without any external aligner.

This is approximate. Real boundaries are fuzzy (voicing onset/offset
doesn't snap exactly to phoneme edges), and some phonemes (nasals,
voiced fricatives) have ambiguous voicing in the extractor. The result
is good enough for Phase 1 training targets. MFA alignment (if
installed) can replace the DTW step for higher accuracy.

Dependencies
------------
    pip install librosa numpy torch
    pip install crepe          # optional — better F0 than pyin on noisy audio

MFA path (optional, higher accuracy)
-------------------------------------
If Montreal Forced Aligner is installed and TextGrids have been
pre-generated, set use_mfa=True and point mfa_dir at the TextGrid
directory. The DTW step is skipped entirely.
"""

from __future__ import annotations

import logging
import multiprocessing
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed — pip install librosa")

from nefs_converter import NEFSConverter
# G2P not needed for training — eSpeak called directly


# ---------------------------------------------------------------------------
# F0 extraction
# ---------------------------------------------------------------------------

def extract_f0(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    fmin: float = 60.0,
    fmax: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-frame F0 and voiced flag from audio.

    Tries crepe first (neural, more accurate on breathy/noisy audio),
    falls back to librosa pyin (fast, no GPU needed).

    Args:
        audio:      (n_samples,) float32 mono audio
        sr:         sample rate
        hop_length: frame hop in samples (~11.6ms at 22050Hz)
        fmin/fmax:  F0 search range in Hz

    Returns:
        f0_hz:   (n_frames,) float32 — 0.0 where unvoiced
        voiced:  (n_frames,) bool
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required: pip install librosa")

    try:
        import crepe
        # crepe returns time, frequency, confidence, activation
        time, freq, conf, _ = crepe.predict(
            audio, sr, viterbi=True, verbose=0,
            step_size=int(hop_length / sr * 1000),  # ms
        )
        # Resample to our hop_length grid
        n_frames = 1 + len(audio) // hop_length
        target_times = np.arange(n_frames) * hop_length / sr
        f0_hz  = np.interp(target_times, time, freq).astype(np.float32)
        voiced = np.interp(target_times, time, conf) > 0.5

    except ImportError:
        # pyin: returns (f0, voiced_flag, voiced_probs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0_hz, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop_length,
                fill_na=0.0,
            )
        f0_hz  = np.nan_to_num(f0_hz, nan=0.0).astype(np.float32)
        voiced = voiced_flag.astype(bool)

    return f0_hz, voiced


# ---------------------------------------------------------------------------
# Voicing pattern from NEFS bytes
# ---------------------------------------------------------------------------

def nefs_voicing_pattern(nefs_bytes: bytes) -> np.ndarray:
    """
    Extract binary voicing pattern from NEFS byte sequence.

    Implements spec v3 §2.2 voicing convention:
      byte = (high_nibble << 4) | low_nibble
      high_nibble = place column, low_nibble = manner row

    Voicing by manner row:
      0x_0 = voiced fricative     → voiced
      0x_1 = voiceless fricative  → unvoiced
      0x_2/4/6 = voiced stops     → voiced
      0x_3/5/7 = voiceless stops  → unvoiced
      0x_8 = nasals               → voiced
      0x_9 = approximants         → voiced
      0x_A = laterals             → voiced
      0x_B = clicks               → unvoiced (generally)
      0x_C = implosives           → voiced
      0x_D = trills/taps          → voiced
    Vowels (high >= 0xA, 0x4 <= low <= 0xB): always voiced.
    Diacritics/tones/stress/effects: skip (no phoneme slot).

    Returns:
        (n_phonemes,) float32 array — 1.0 voiced, 0.0 unvoiced
    """
    pattern = []
    for b in nefs_bytes:
        low  = b & 0x0F
        high = b >> 4

        # Skip non-phoneme bytes (spec §3.2 order)
        if b == 0x0F:
            continue
        if low == 0x0F:          # tone row
            continue
        if low == 0x0E or high == 0x0:  # diacritic L
            continue
        if high >= 0xE and low <= 0x3:  # stress table
            continue
        if high >= 0xA and 0xC <= low <= 0xD:  # effects
            continue

        # Vowel: always voiced
        if high >= 0xA and 0x4 <= low <= 0xB:
            pattern.append(1.0)
            continue

        # Consonant: voicing by manner row
        if low == 0x0:                   # voiced fricative
            pattern.append(1.0)
        elif low == 0x1:                 # voiceless fricative
            pattern.append(0.0)
        elif low in (0x2, 0x4, 0x6):    # voiced stops (plain/asp/NR)
            pattern.append(1.0)
        elif low in (0x3, 0x5, 0x7):    # voiceless stops
            pattern.append(0.0)
        elif low in (0x8, 0x9, 0xA, 0xC, 0xD):  # nasal/approx/lateral/implosive/trill
            pattern.append(1.0)
        elif low == 0xB:                 # clicks — treat as unvoiced
            pattern.append(0.0)
        else:
            pattern.append(0.0)

    return np.array(pattern, dtype=np.float32)


# ---------------------------------------------------------------------------
# DTW alignment
# ---------------------------------------------------------------------------

def dtw_align(
    audio_voiced: np.ndarray,
    phoneme_voiced: np.ndarray,
    window: Optional[int] = None,
) -> np.ndarray:
    """
    Align audio voicing frames to phoneme voicing pattern via DTW.

    Args:
        audio_voiced:   (n_frames,) float32 — voiced/unvoiced per audio frame
        phoneme_voiced: (n_phonemes,) float32 — voicing per NEFS phoneme
        window:         Sakoe-Chiba band width (None = full DTW)

    Returns:
        boundaries: (n_phonemes + 1,) int — frame indices of phoneme boundaries
                    boundaries[i] .. boundaries[i+1] = frames for phoneme i
    """
    n = len(audio_voiced)
    m = len(phoneme_voiced)

    if m == 0:
        return np.array([0, n], dtype=np.int64)

    # Cost matrix: squared difference of voicing signals
    cost = np.zeros((n, m), dtype=np.float32)
    for j in range(m):
        cost[:, j] = (audio_voiced - phoneme_voiced[j]) ** 2

    # Accumulated cost with optional Sakoe-Chiba band
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    band = window if window is not None else max(n // 4, m)

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end   = min(m, i + band)
        for j in range(j_start, j_end + 1):
            d = cost[i - 1, j - 1]
            D[i, j] = d + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Traceback to recover path
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = [(D[i-1, j], i-1, j), (D[i, j-1], i, j-1), (D[i-1, j-1], i-1, j-1)]
        _, i, j = min(options, key=lambda x: x[0])
    path.reverse()

    # Convert path to phoneme boundaries
    # For each phoneme j, find the range of audio frames i mapped to it
    frame_to_phoneme = np.full(n, -1, dtype=np.int64)
    for frame_idx, phon_idx in path:
        frame_to_phoneme[frame_idx] = phon_idx

    boundaries = np.zeros(m + 1, dtype=np.int64)
    boundaries[0] = 0
    boundaries[m] = n
    for phon_idx in range(m - 1):
        # Boundary = first frame assigned to phoneme phon_idx + 1
        frames_next = np.where(frame_to_phoneme == phon_idx + 1)[0]
        if len(frames_next) > 0:
            boundaries[phon_idx + 1] = frames_next[0]
        else:
            # Fallback: linear interpolation
            boundaries[phon_idx + 1] = int(n * (phon_idx + 1) / m)

    # Enforce minimum segment width of 2 frames to prevent collapse,
    # then ensure strictly monotonically increasing boundaries.
    min_frames = 2
    for k in range(1, m):
        if boundaries[k] < boundaries[k - 1] + min_frames:
            boundaries[k] = boundaries[k - 1] + min_frames
    # If we overflowed n, redistribute evenly from the right
    if boundaries[m] > n:
        for k in range(m, 0, -1):
            if boundaries[k] > n:
                boundaries[k] = n
            if boundaries[k] < boundaries[k-1] + min_frames:
                boundaries[k-1] = max(0, boundaries[k] - min_frames)
    boundaries = np.clip(boundaries, 0, n)

    return boundaries


# ---------------------------------------------------------------------------
# MFA TextGrid reader (optional)
# ---------------------------------------------------------------------------

def load_mfa_boundaries(
    textgrid_path: Path,
    sr: int,
    hop_length: int,
) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Load phoneme boundaries from an MFA-generated TextGrid file.

    Returns (phoneme_list, frame_boundaries) or None if unavailable.
    Requires the 'textgrid' package: pip install textgrid
    """
    try:
        import textgrid as tg_lib
        tg = tg_lib.TextGrid.fromFile(str(textgrid_path))
        # MFA puts phones in the second tier by convention
        phone_tier = tg[1]
        phonemes = []
        boundaries = [0]
        for interval in phone_tier:
            label = interval.mark.strip()
            if not label or label in ("", "sp", "sil", "SIL"):
                continue
            phonemes.append(label)
            end_frame = int(interval.maxTime * sr / hop_length)
            boundaries.append(end_frame)
        return phonemes, np.array(boundaries, dtype=np.int64)
    except Exception as e:
        logger.debug(f"MFA TextGrid load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-utterance extraction
# ---------------------------------------------------------------------------

def extract_utterance(
    audio: np.ndarray,
    sr: int,
    transcript: str,
    hop_length: int = 256,
    mfa_textgrid: Optional[Path] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract prosody targets for one utterance.

    Args:
        audio:          (n_samples,) float32 mono audio
        sr:             sample rate
        transcript:     orthographic transcript
        hop_length:     F0 extraction hop in samples
        mfa_textgrid:   path to MFA TextGrid (None = use DTW)

    Returns:
        Dict with arrays of shape (n_phonemes,):
            nefs_bytes:  uint8
            f0_hz:       float32 (0.0 where unvoiced)
            duration_s:  float32
            f0_mask:     bool (True where voiced)
        or None if extraction failed.
    """
    if not _LIBROSA_AVAILABLE:
        return None

    converter = NEFSConverter()

    # G2P + NEFS conversion
    try:
        # Call eSpeak directly for IPA — no G2P module needed
        import subprocess
        r = subprocess.run(
            ['espeak-ng', '--ipa', '-q', '--', transcript],
            capture_output=True
        )
        # Strip whitespace, CRLF, and IPA characters the converter
        # doesn't handle (spaces, stress marks — those are separate bytes)
        import re
        ipa = r.stdout.decode('utf-8').strip()
        ipa = ipa.replace('\r', '').replace('\n', '').replace(' ', '')
        # eSpeak uses U+0261 (script g) — normalise to plain g
        ipa = ipa.replace('\u0261', 'g')
        # Strip primary/secondary stress marks from IPA string
        # (they're prosody markers, not phonemes in this context)
        ipa = ipa.replace('\u02C8', '').replace('\u02CC', '')
        nefs_bytes = converter.ipa_to_nefs(ipa)
        if len(nefs_bytes) == 0:
            return None
    except Exception as e:
        logger.warning(f"G2P/NEFS failed for '{transcript[:40]}': {e}")
        return None

    # F0 extraction from audio
    try:
        f0_frames, voiced_frames = extract_f0(audio, sr, hop_length)
    except Exception as e:
        logger.warning(f"F0 extraction failed: {e}")
        return None

    n_frames = len(f0_frames)

    # Phoneme boundaries
    if mfa_textgrid is not None:
        mfa_result = load_mfa_boundaries(mfa_textgrid, sr, hop_length)
    else:
        mfa_result = None

    if mfa_result is not None:
        _, boundaries = mfa_result
        # Trim to match NEFS length if MFA has different phoneme count
        n_phon = len(nefs_bytes)
        if len(boundaries) != n_phon + 1:
            boundaries = np.linspace(0, n_frames, n_phon + 1, dtype=np.int64)
    else:
        # DTW alignment on voicing pattern
        phoneme_voiced = nefs_voicing_pattern(nefs_bytes)
        audio_voiced   = voiced_frames.astype(np.float32)
        boundaries = dtw_align(audio_voiced, phoneme_voiced)

    # Aggregate F0 and duration per phoneme segment
    n_phon = len(nefs_bytes)
    f0_per_phoneme       = np.zeros(n_phon, dtype=np.float32)
    duration_per_phoneme = np.zeros(n_phon, dtype=np.float32)
    voiced_per_phoneme   = np.zeros(n_phon, dtype=bool)

    hop_s = hop_length / sr

    # Ensure boundaries has exactly n_phon + 1 entries and stays in range
    if len(boundaries) < n_phon + 1:
        boundaries = np.linspace(0, n_frames, n_phon + 1, dtype=np.int64)
    boundaries = np.clip(boundaries, 0, n_frames)

    for i in range(n_phon):
        start = int(boundaries[i])
        end   = int(boundaries[min(i + 1, len(boundaries) - 1)])
        end   = min(end, n_frames)

        if start >= end:
            duration_per_phoneme[i] = hop_s
            continue

        duration_per_phoneme[i] = (end - start) * hop_s

        segment_f0     = f0_frames[start:end]
        segment_voiced = voiced_frames[start:end]

        voiced_f0 = segment_f0[segment_voiced]
        if len(voiced_f0) > 0:
            # Median is more robust than mean for F0 (resistant to octave errors)
            f0_per_phoneme[i]     = float(np.median(voiced_f0))
            voiced_per_phoneme[i] = True

    return {
        "nefs_bytes":  np.frombuffer(nefs_bytes, dtype=np.uint8).copy(),
        "f0_hz":       f0_per_phoneme,
        "duration_s":  duration_per_phoneme,
        "f0_mask":     voiced_per_phoneme,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LJSpeechProsodyDataset(Dataset):
    """
    LJSpeech dataset with extracted prosody targets.

    Caches extraction results to disk so the slow DTW/pyin step only
    runs once. Cache lives at {data_dir}/nefs_prosody_cache/.

    Args:
        data_dir:    Path to LJSpeech-1.1 root
        sample_rate: Audio sample rate (LJSpeech native = 22050)
        hop_length:  F0 extraction hop in samples
        mfa_dir:     Optional path to MFA TextGrid directory
        max_samples: Truncate dataset for fast iteration (None = all)
        cache:       Whether to use/write disk cache
    """

    def __init__(
        self,
        data_dir: Path,
        sample_rate: int = 22050,
        hop_length: int = 256,
        mfa_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        cache: bool = True,
    ):
        if not _LIBROSA_AVAILABLE:
            raise RuntimeError("librosa required: pip install librosa")

        self.data_dir    = Path(data_dir)
        self.sr          = sample_rate
        self.hop_length  = hop_length
        self.mfa_dir     = Path(mfa_dir) if mfa_dir else None
        self.cache       = cache
        self.cache_dir   = self.data_dir / "nefs_prosody_cache"

        if cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        self.samples: List[Tuple[Path, str]] = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                name, transcript = parts[0], parts[1]
                wav_path = self.data_dir / "wavs" / f"{name}.wav"
                if wav_path.exists():
                    self.samples.append((wav_path, transcript))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"LJSpeechProsodyDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        wav_path, transcript = self.samples[idx]

        # Try cache first
        if self.cache:
            cached = self._load_cache(wav_path)
            if cached is not None:
                return cached

        # Load audio
        try:
            audio, sr = librosa.load(str(wav_path), sr=self.sr, mono=True)
        except Exception as e:
            logger.warning(f"Audio load failed {wav_path}: {e}")
            return None

        # MFA TextGrid path (if available)
        mfa_tg = None
        if self.mfa_dir is not None:
            candidate = self.mfa_dir / (wav_path.stem + ".TextGrid")
            if candidate.exists():
                mfa_tg = candidate

        # Extract prosody
        result = extract_utterance(
            audio, sr, transcript,
            hop_length=self.hop_length,
            mfa_textgrid=mfa_tg,
        )
        if result is None:
            return None

        # Convert to tensors
        item = {
            "nefs_bytes":  torch.tensor(result["nefs_bytes"].astype(np.int64)),
            "f0_hz":       torch.tensor(result["f0_hz"]),
            "duration_s":  torch.tensor(result["duration_s"]),
            "f0_mask":     torch.tensor(result["f0_mask"]),
        }
        n = item["nefs_bytes"].size(0)
        item["nefs_mask"]    = torch.ones(n, dtype=torch.bool)
        item["nefs_lengths"] = torch.tensor(n, dtype=torch.long)

        if self.cache:
            self._save_cache(wav_path, item)

        return item

    def _cache_path(self, wav_path: Path) -> Path:
        return self.cache_dir / (wav_path.stem + ".pt")

    def _load_cache(self, wav_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        p = self._cache_path(wav_path)
        if p.exists():
            try:
                return torch.load(p, weights_only=True)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _save_cache(self, wav_path: Path, item: Dict[str, torch.Tensor]):
        try:
            torch.save(item, self._cache_path(wav_path))
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def prosody_collate(
    batch: List[Optional[Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length prosody items into a batch.
    Filters out None items (failed extractions).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        # Return a minimal valid batch so training doesn't crash on bad data
        dummy = torch.zeros(1, 1, dtype=torch.long)
        return {
            "nefs_bytes":   dummy,
            "nefs_mask":    torch.zeros(1, 1, dtype=torch.bool),
            "nefs_lengths": torch.ones(1, dtype=torch.long),
            "f0_hz":        torch.zeros(1, 1),
            "f0_mask":      torch.zeros(1, 1, dtype=torch.bool),
            "duration_s":   torch.ones(1, 1) * 0.08,
        }

    max_len = max(b["nefs_bytes"].size(0) for b in batch)

    def pad(t, val=0):
        return F.pad(t, (0, max_len - t.size(0)), value=val)

    return {
        "nefs_bytes":   torch.stack([pad(b["nefs_bytes"])  for b in batch]),
        "nefs_mask":    torch.stack([pad(b["nefs_mask"])   for b in batch]),
        "nefs_lengths": torch.stack([b["nefs_lengths"]     for b in batch]),
        "f0_hz":        torch.stack([pad(b["f0_hz"])       for b in batch]),
        "f0_mask":      torch.stack([pad(b["f0_mask"])     for b in batch]),
        "duration_s":   torch.stack([pad(b["duration_s"])  for b in batch]),
    }


# ---------------------------------------------------------------------------
# CLI — run extraction on a dataset and report coverage stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Test prosody extractor on LJSpeech")
    parser.add_argument("--data-dir",    type=Path, required=True)
    parser.add_argument("--max-samples", type=int,  default=50)
    parser.add_argument("--mfa-dir",     type=Path, default=None)
    args = parser.parse_args()

    ds = LJSpeechProsodyDataset(
        args.data_dir,
        max_samples=args.max_samples,
        mfa_dir=args.mfa_dir,
    )

    successes = 0
    f0_values = []
    dur_values = []

    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        successes += 1
        voiced = item["f0_mask"]
        f0_values.extend(item["f0_hz"][voiced].tolist())
        dur_values.extend(item["duration_s"].tolist())

    print(f"\nExtraction results ({successes}/{len(ds)} succeeded)")
    if f0_values:
        print(f"  F0 range:      {min(f0_values):.0f}–{max(f0_values):.0f} Hz")
        print(f"  F0 median:     {float(np.median(f0_values)):.0f} Hz")
    if dur_values:
        print(f"  Duration range:{min(dur_values)*1000:.0f}–{max(dur_values)*1000:.0f} ms")
        print(f"  Duration mean: {float(np.mean(dur_values))*1000:.0f} ms")
    print(f"  Voiced ratio:  {len(f0_values)/max(len(dur_values),1):.1%}")
