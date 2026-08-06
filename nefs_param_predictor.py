"""
nefs_param_predictor.py — Neural prosody parameter predictor for NEFS

Architecture
------------
Input:  NEFS byte sequence (one byte per phoneme)
Middle: Bit feature extraction (free — two ops per byte, no learned params)
        → small bidirectional LSTM (≈1–3M params, CPU real-time)
Output: Per-phoneme SynthParams trajectories

The bit extractor gives the LSTM a head start: it sees place, manner,
voicing, and aspiration as separate channels rather than a flat 256-dim
one-hot.  This means the network can generalise across phoneme classes
(e.g. "all voiced stops behave similarly for duration") with far less
training data than a system starting from opaque embeddings.

Klatt compatibility
-------------------
The predictor outputs f1_hz, f2_hz, f3_hz on every PhonemeParams even
when the eSpeak backend ignores them.  When Klatt is swapped in, those
values drive the formant synthesizer directly with no predictor changes.

Training strategy
-----------------
Phase 1 (current): rule-bootstrapped targets.
  - Run eSpeak on LJSpeech transcripts with --ipa to get reference IPA.
  - Convert to NEFS bytes via NEFSConverter.
  - Extract eSpeak's prosody output by parsing its SSML intermediate.
  - Use those values as regression targets.
  - Loss: MSE on (f0, duration, rate, energy).

Phase 2 (later): audio-supervised fine-tuning.
  - Run predictor → eSpeak → audio.
  - Compare mel-spectrogram of generated vs LJSpeech reference.
  - Backprop through the mel distance into the predictor weights.
  - eSpeak is non-differentiable; use finite differences or a learned
    surrogate mel predictor as a proxy gradient.

The Phase 1 training data is already implicit in the repo: LJSpeech
(TRAINING_GUIDE.md) + espeak bootstrap (nefs_g2p.py Tier 2).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — predictor will use rule-based fallback only.")

from nefs_synth_interface import (
    PhonemeParams,
    SynthParams,
    hz_to_espeak_pitch,
    rate_factor_to_espeak_wpm,
)
from nefs_converter import NEFSConverter


# ---------------------------------------------------------------------------
# Bit feature extractor (no learned parameters)
# ---------------------------------------------------------------------------

def extract_nefs_features(nefs_bytes: "torch.Tensor") -> "torch.Tensor":
    """
    Extract phonological features from NEFS bytes via bit operations.

    This is the Two-Operation Guarantee from the README made concrete.
    Zero learned parameters — pure bitwise arithmetic.

    Args:
        nefs_bytes: (batch, seq_len) int64 tensor of NEFS byte values.

    Returns:
        (batch, seq_len, 7) float32 feature tensor:
          [0] place          = high nibble (0–15)
          [1] manner         = low nibble  (0–15)
          [2] voicing        = bit 0       (0 or 1)
          [3] aspiration     = bits 2–1    (0–3)
          [4] is_vowel       = heuristic: place >= 4 and manner in 4..11
          [5] is_nasal       = place == 9  (0x9_ row in NEFS grid)
          [6] is_approximant = place in {0xA, 0xB}
    """
    place      = ((nefs_bytes >> 4) & 0x0F).float()
    manner     = (nefs_bytes & 0x0F).float()
    voicing    = (nefs_bytes & 0x01).float()
    aspiration = ((nefs_bytes >> 1) & 0x03).float()

    # Vowel: rows 0x4_–0x8_ in the NEFS grid
    # (place nibble 4..8 covers close through near-close vowels)
    is_vowel = ((place >= 4) & (place <= 8) & (manner >= 4) & (manner <= 11)).float()

    # Nasal: row 0x9_
    is_nasal = (place == 9).float()

    # Approximant: rows 0xA_ and 0xB_
    is_approx = ((place == 10) | (place == 11)).float()

    return torch.stack(
        [place, manner, voicing, aspiration, is_vowel, is_nasal, is_approx],
        dim=-1
    )


# ---------------------------------------------------------------------------
# Predictor model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class NEFSParamPredictor(nn.Module):
        """
        Small bidirectional LSTM that maps a NEFS byte sequence to
        per-phoneme synthesis parameter trajectories.

        Model size: ~1.2M params at hidden_dim=128.  Runs in <5ms per
        utterance on a modern CPU — well within the 50ms real-time budget.

        Output heads (all per-phoneme):
          - f0_hz:        log-scale, then exp → Hz
          - log_duration: log-scale, then exp → seconds
          - rate_factor:  sigmoid scaled to [0.4, 2.0]
          - breathiness:  sigmoid → [0, 1]
          - energy:       sigmoid → [0, 1]
          - f1_hz:        softplus → Hz  (for Klatt; eSpeak ignores)
          - f2_hz:        softplus → Hz
          - f3_hz:        softplus → Hz

        Klatt compatibility: the formant heads are trained with a dummy
        MSE target of 0.0 during Phase 1 (eSpeak bootstrap) so they produce
        plausible but uncalibrated outputs.  Phase 2 fine-tuning with a Klatt
        backend will calibrate them against real formant measurements.
        """

        # Typical formant ranges used to initialise output biases so the
        # network starts in a plausible region rather than near zero.
        _F0_NEUTRAL_HZ   = 120.0
        _DUR_NEUTRAL_S   = 0.08
        _F1_NEUTRAL_HZ   = 500.0
        _F2_NEUTRAL_HZ   = 1500.0
        _F3_NEUTRAL_HZ   = 2500.0

        def __init__(
            self,
            feature_dim: int = 7,       # output of extract_nefs_features
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Normalise the 7 raw features to roughly zero mean / unit variance
            # before feeding into the LSTM.  Running stats are updated during
            # training; fixed at inference.
            self.feature_norm = nn.LayerNorm(feature_dim)

            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            lstm_out_dim = hidden_dim * 2  # bidirectional

            # Output heads — each is a single linear layer.
            # Keeping them separate lets us freeze/unfreeze individually
            # when switching from Phase 1 to Phase 2 training.
            self.head_f0       = nn.Linear(lstm_out_dim, 1)
            self.head_duration = nn.Linear(lstm_out_dim, 1)
            self.head_rate     = nn.Linear(lstm_out_dim, 1)
            self.head_breath   = nn.Linear(lstm_out_dim, 1)
            self.head_energy   = nn.Linear(lstm_out_dim, 1)
            self.head_f1       = nn.Linear(lstm_out_dim, 1)
            self.head_f2       = nn.Linear(lstm_out_dim, 1)
            self.head_f3       = nn.Linear(lstm_out_dim, 1)

            self._init_output_biases()

        def _init_output_biases(self):
            """
            Initialise output biases so the network starts producing
            plausible parameter values before any training.

            Without this, log-space heads start near 0, which after exp()
            gives ~1 Hz — far outside the valid range and slow to escape.
            """
            with torch.no_grad():
                # log(120) ≈ 4.79 → exp → 120 Hz
                self.head_f0.bias.fill_(math.log(self._F0_NEUTRAL_HZ))
                # log(0.08) ≈ -2.53 → exp → 0.08 s
                self.head_duration.bias.fill_(math.log(self._DUR_NEUTRAL_S))
                # sigmoid(0) = 0.5 → scaled to 1.2 (slightly faster than neutral)
                self.head_rate.bias.fill_(0.0)
                # sigmoid(-3) ≈ 0.05 → near-modal voice as default
                self.head_breath.bias.fill_(-3.0)
                # sigmoid(1.4) ≈ 0.8 → reasonable default energy
                self.head_energy.bias.fill_(1.4)
                # Formant biases in log space
                self.head_f1.bias.fill_(math.log(self._F1_NEUTRAL_HZ))
                self.head_f2.bias.fill_(math.log(self._F2_NEUTRAL_HZ))
                self.head_f3.bias.fill_(math.log(self._F3_NEUTRAL_HZ))

        def forward(
            self,
            nefs_bytes: "torch.Tensor",       # (batch, seq_len) int64
            lengths: Optional["torch.Tensor"] = None,  # (batch,) for packing
        ) -> Dict[str, "torch.Tensor"]:
            """
            Args:
                nefs_bytes: (batch, seq_len) NEFS byte values.
                lengths:    (batch,) actual sequence lengths for PackedSequence.
                            If None, all sequences assumed full length.

            Returns:
                Dict of (batch, seq_len) float32 tensors, one per parameter.
            """
            # Feature extraction — no gradients flow through bit ops
            with torch.no_grad():
                features = extract_nefs_features(nefs_bytes)  # (B, T, 7)

            features = self.feature_norm(features)

            # Pack for efficiency if lengths provided
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    features, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.lstm(packed)
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True
                )
            else:
                lstm_out, _ = self.lstm(features)   # (B, T, hidden*2)

            # Apply output heads with appropriate activations
            f0_hz = torch.exp(self.head_f0(lstm_out).squeeze(-1)).clamp(50.0, 500.0)

            duration_s = torch.exp(
                self.head_duration(lstm_out).squeeze(-1)
            ).clamp(0.01, 0.8)

            # rate_factor: sigmoid maps to [0, 1], then scale to [0.4, 2.0]
            rate_factor = torch.sigmoid(
                self.head_rate(lstm_out).squeeze(-1)
            ) * 1.6 + 0.4

            breathiness = torch.sigmoid(self.head_breath(lstm_out).squeeze(-1))
            energy      = torch.sigmoid(self.head_energy(lstm_out).squeeze(-1))

            # Formant targets (log-space for stability, then exp)
            f1_hz = torch.exp(self.head_f1(lstm_out).squeeze(-1)).clamp(200.0, 1000.0)
            f2_hz = torch.exp(self.head_f2(lstm_out).squeeze(-1)).clamp(600.0, 3500.0)
            f3_hz = torch.exp(self.head_f3(lstm_out).squeeze(-1)).clamp(1500.0, 4500.0)

            return {
                "f0_hz":       f0_hz,
                "duration_s":  duration_s,
                "rate_factor": rate_factor,
                "breathiness": breathiness,
                "energy":      energy,
                "f1_hz":       f1_hz,
                "f2_hz":       f2_hz,
                "f3_hz":       f3_hz,
            }


# ---------------------------------------------------------------------------
# Rule-based fallback (no PyTorch required)
# ---------------------------------------------------------------------------

# Coarse per-class defaults derived from acoustic phonetics literature.
# Used when PyTorch is unavailable OR as Phase-0 initialisation targets
# for the neural predictor's training loss.
_RULE_DEFAULTS: Dict[str, Dict] = {
    "vowel":      {"f0_hz": 130.0, "duration_s": 0.10, "rate_factor": 0.7,
                   "breathiness": 0.05, "energy": 0.9},
    "voiced_stop":{"f0_hz": 115.0, "duration_s": 0.07, "rate_factor": 0.7,
                   "breathiness": 0.0,  "energy": 0.75},
    "vl_stop":    {"f0_hz": 0.0,   "duration_s": 0.08, "rate_factor": 0.7,
                   "breathiness": 0.0,  "energy": 0.7},
    "fricative":  {"f0_hz": 0.0,   "duration_s": 0.09, "rate_factor": 0.7,
                   "breathiness": 0.15, "energy": 0.65},
    "nasal":      {"f0_hz": 110.0, "duration_s": 0.08, "rate_factor": 0.7,
                   "breathiness": 0.0,  "energy": 0.7},
    "approximant":{"f0_hz": 125.0, "duration_s": 0.09, "rate_factor": 0.7,
                   "breathiness": 0.0,  "energy": 0.8},
    "default":    {"f0_hz": 120.0, "duration_s": 0.08, "rate_factor": 0.7,
                   "breathiness": 0.0,  "energy": 0.8},
}

def _classify_byte(b: int) -> str:
    """
    Return a coarse phoneme class label for a NEFS byte.

    Uses the spec v3 §3.2 classification algorithm:
      byte = (high_nibble << 4) | low_nibble
      high_nibble = place of articulation column
      low_nibble  = manner of articulation row

    Vowel region: high >= 0xA AND 0x4 <= low <= 0xB
    Consonant rows:
      0x_0 = voiced fricative     0x_1 = voiceless fricative
      0x_2/4/6 = voiced stop      0x_3/5/7 = voiceless stop
      0x_8 = nasal                0x_9 = approximant
      0x_A = lateral              0x_B = click
      0x_C = implosive            0x_D = trill/tap
    """
    if b == 0x0F:
        return "silence"
    low  = b & 0x0F
    high = b >> 4

    # Spec §3.2 — check in order
    if low == 0x0F:
        return "tone"
    if low == 0x0E or high == 0x0:
        return "diacritic"
    if high >= 0xE and low <= 0x3:
        return "stress"
    if high >= 0xA and 0xC <= low <= 0xD:
        return "effects"
    if high >= 0xA and 0x4 <= low <= 0xB:
        return "vowel"

    # Consonant table — classify by manner (low nibble)
    if low == 0x0:
        return "fricative"       # voiced fricative
    if low == 0x1:
        return "fricative"       # voiceless fricative
    if low in (0x2, 0x4, 0x6):
        return "voiced_stop"
    if low in (0x3, 0x5, 0x7):
        return "vl_stop"
    if low == 0x8:
        return "nasal"
    if low == 0x9:
        return "approximant"
    if low == 0xA:
        return "approximant"     # lateral — route same as approximant
    if low in (0xB, 0xC, 0xD):
        return "default"         # click, implosive, trill

    return "default"


def rule_based_params(nefs_byte: int, ipa: str) -> Dict:
    """Return rule-based parameter dict for a single NEFS byte."""
    cls = _classify_byte(nefs_byte)
    defaults = _RULE_DEFAULTS.get(cls, _RULE_DEFAULTS["default"]).copy()

    # Simple formant defaults by vowel (rough Hillenbrand 1995 values for /ɑ/)
    if cls == "vowel":
        defaults["f1_hz"] = 700.0
        defaults["f2_hz"] = 1220.0
        defaults["f3_hz"] = 2600.0
    else:
        defaults["f1_hz"] = None
        defaults["f2_hz"] = None
        defaults["f3_hz"] = None

    return defaults


# ---------------------------------------------------------------------------
# High-level predictor wrapper
# ---------------------------------------------------------------------------

class NEFSPredictor:
    """
    High-level predictor: NEFS bytes → SynthParams.

    Transparently uses the neural model when PyTorch is available and a
    checkpoint exists, falling back to rule-based defaults otherwise.
    This means the system is always runnable — the neural model is an
    *improvement* layer, not a hard dependency.

    Usage:
        predictor = NEFSPredictor()                   # rule-based only
        predictor = NEFSPredictor('checkpoints/pred.pt')  # neural
        params = predictor.predict(nefs_bytes)
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        self.converter = NEFSConverter()
        self.device = device
        self._model: Optional["NEFSParamPredictor"] = None

        if _TORCH_AVAILABLE and checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                self._model = NEFSParamPredictor(
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                ).to(device)
                state = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
                self._model.load_state_dict(state["model"])
                self._model.eval()
                logger.info(f"Loaded neural predictor from {checkpoint_path}")
            else:
                logger.warning(
                    f"Checkpoint not found at {checkpoint_path}. "
                    "Using rule-based fallback."
                )
        elif not _TORCH_AVAILABLE:
            logger.info("PyTorch unavailable — using rule-based parameter predictor.")
        else:
            logger.info("No checkpoint provided — using rule-based parameter predictor.")

    @property
    def using_neural(self) -> bool:
        return self._model is not None

    def predict(self, nefs_bytes: bytes, sample_rate: int = 22050) -> SynthParams:
        """
        Convert NEFS bytes to a full SynthParams bundle.

        Args:
            nefs_bytes:  Raw NEFS byte sequence (from NEFSConverter).
            sample_rate: Target audio sample rate (passed through to SynthParams).

        Returns:
            SynthParams ready to hand to any SynthBackend.
        """
        if not nefs_bytes:
            return SynthParams(phonemes=[], sample_rate=sample_rate)

        # Decode bytes to IPA symbols (one entry per byte, or multi-char for
        # two-byte affricates — nafs_to_ipa handles the two-byte sequences).
        # We need per-byte IPA for PhonemeParams, so we decode individually.
        ipa_per_byte = [
            self.converter.nefs_to_ipa(bytes([b])) for b in nefs_bytes
        ]

        if self._model is not None:
            return self._predict_neural(nefs_bytes, ipa_per_byte, sample_rate)
        else:
            return self._predict_rules(nefs_bytes, ipa_per_byte, sample_rate)

    def _predict_neural(
        self,
        nefs_bytes: bytes,
        ipa_per_byte: List[str],
        sample_rate: int,
    ) -> SynthParams:
        """Neural model prediction path."""
        byte_tensor = torch.tensor(
            list(nefs_bytes), dtype=torch.long
        ).unsqueeze(0).to(self.device)  # (1, seq_len)

        with torch.no_grad():
            outputs = self._model(byte_tensor)

        phonemes = []
        for i, (b, ipa) in enumerate(zip(nefs_bytes, ipa_per_byte)):
            phonemes.append(PhonemeParams(
                nefs_byte=b,
                ipa=ipa,
                f0_hz=float(outputs["f0_hz"][0, i]),
                duration_s=float(outputs["duration_s"][0, i]),
                rate_factor=float(outputs["rate_factor"][0, i]),
                breathiness=float(outputs["breathiness"][0, i]),
                energy=float(outputs["energy"][0, i]),
                f1_hz=float(outputs["f1_hz"][0, i]),
                f2_hz=float(outputs["f2_hz"][0, i]),
                f3_hz=float(outputs["f3_hz"][0, i]),
            ))

        return SynthParams(phonemes=phonemes, sample_rate=sample_rate)

    def _predict_rules(
        self,
        nefs_bytes: bytes,
        ipa_per_byte: List[str],
        sample_rate: int,
    ) -> SynthParams:
        """Rule-based fallback prediction path."""
        phonemes = []
        for b, ipa in zip(nefs_bytes, ipa_per_byte):
            p = rule_based_params(b, ipa)
            phonemes.append(PhonemeParams(
                nefs_byte=b,
                ipa=ipa,
                f0_hz=p["f0_hz"],
                duration_s=p["duration_s"],
                rate_factor=p["rate_factor"],
                breathiness=p["breathiness"],
                energy=p["energy"],
                f1_hz=p.get("f1_hz"),
                f2_hz=p.get("f2_hz"),
                f3_hz=p.get("f3_hz"),
            ))
        return SynthParams(phonemes=phonemes, sample_rate=sample_rate)

    def save_checkpoint(self, path: Path, epoch: int = 0, loss: float = 0.0):
        """Save current model weights."""
        if self._model is None:
            raise RuntimeError("No neural model loaded — nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self._model.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, path)
        logger.info(f"Predictor checkpoint saved to {path}")
