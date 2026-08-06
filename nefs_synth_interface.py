"""
nefs_synth_interface.py — Backend-agnostic synthesis parameter contract

Defines the SynthParams dataclass and SynthBackend protocol that every
synthesis backend (eSpeak, Klatt, etc.) must implement.

The neural predictor (nefs_param_predictor.py) outputs SynthParams.
The active backend consumes SynthParams to produce audio.

Swapping backends = changing one line in NEFSRealTimeSynth.__init__.
Nothing else changes.

Parameter design notes
----------------------
All parameters are *normalised* — backend implementations are responsible
for mapping them to their internal units.  This keeps the neural predictor
backend-agnostic:

  f0_hz        : fundamental frequency in Hz (absolute, not semitones).
                 Range 50–500 Hz covers virtually all human voices.
                 eSpeak maps this to its -p flag (0–99) and SSML pitch %.
                 Klatt uses it directly as AV source frequency.

  duration_s   : target phoneme duration in seconds.
                 eSpeak approximates via <break> tags and rate adjustment.
                 Klatt uses it as frame count × hop_size.

  rate_factor  : speaking rate multiplier relative to neutral (1.0).
                 0.5 = half speed, 2.0 = double.  Per-phoneme variation
                 models reduction on unstressed syllables.
                 eSpeak: injected as SSML <prosody rate="...">.
                 Klatt: stretches/compresses parameter trajectories.

  breathiness  : 0.0 = modal voice, 1.0 = fully breathy.
                 eSpeak has no direct breathiness control; approximated by
                 reducing volume slightly and adding aspiration via IPA hʰ.
                 Klatt: controls AH (aspiration amplitude) directly.

  spectral_tilt: 0.0 = flat, 1.0 = heavy low-frequency emphasis.
                 eSpeak: not controllable; stored for Klatt compatibility.
                 Klatt: adjusts TL (spectral tilt) parameter.

  energy       : relative loudness 0.0–1.0.
                 eSpeak: SSML <prosody volume="...">.
                 Klatt: AV amplitude.

The 'metadata' dict is a free-form escape hatch for backend-specific values
that don't belong in the core contract.  eSpeak uses it for 'lang'; a future
Klatt backend might store formant targets here during the transition period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable
import numpy as np


# ---------------------------------------------------------------------------
# Per-phoneme parameter bundle
# ---------------------------------------------------------------------------

@dataclass
class PhonemeParams:
    """
    Predicted synthesis parameters for a single NEFS phoneme.

    All values are floats in the ranges documented in the module docstring.
    Defaults represent neutral / rule-system-equivalent values so that a
    predictor that hasn't learned to control a parameter yet doesn't degrade
    quality relative to the baseline eSpeak rule system.
    """
    nefs_byte: int                  # source byte (0x00–0xFF)
    ipa: str                        # IPA symbol(s) this byte decodes to

    # Core prosody
    f0_hz: float = 120.0            # fundamental frequency
    duration_s: float = 0.08        # phoneme duration
    rate_factor: float = 1.0        # local speaking rate multiplier

    # Voice quality
    breathiness: float = 0.0        # 0 = modal, 1 = breathy
    spectral_tilt: float = 0.0      # 0 = flat, 1 = tilted (Klatt only for now)
    energy: float = 0.8             # relative loudness

    # Coarticulation hints (stored now, used by Klatt later)
    f1_hz: Optional[float] = None   # first formant target (None = let backend decide)
    f2_hz: Optional[float] = None   # second formant target
    f3_hz: Optional[float] = None   # third formant target

    # Escape hatch
    metadata: Dict = field(default_factory=dict)


@dataclass
class SynthParams:
    """
    Full parameter bundle for one synthesis request (a sequence of phonemes).
    """
    phonemes: List[PhonemeParams]
    sample_rate: int = 22050
    global_rate_factor: float = 1.0     # multiplied with per-phoneme rate_factor
    global_pitch_shift_st: float = 0.0  # semitone shift applied on top of f0_hz

    @property
    def total_duration_s(self) -> float:
        return sum(p.duration_s for p in self.phonemes)

    @property
    def ipa_string(self) -> str:
        return "".join(p.ipa for p in self.phonemes)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SynthBackend(Protocol):
    """
    Protocol every synthesis backend must implement.

    Backends receive SynthParams and return a float32 numpy array of PCM
    audio at params.sample_rate.

    The Protocol is @runtime_checkable so isinstance() works in tests and
    the NEFSRealTimeSynth initialiser can verify the backend at startup
    rather than discovering a missing method at first synthesis call.
    """

    @property
    def name(self) -> str:
        """Human-readable backend identifier, e.g. 'espeak' or 'klatt'."""
        ...

    @property
    def available(self) -> bool:
        """True if the backend's dependencies are installed and reachable."""
        ...

    def synthesize(self, params: SynthParams) -> np.ndarray:
        """
        Synthesize speech from SynthParams.

        Args:
            params: Full parameter bundle for the utterance.

        Returns:
            float32 numpy array of PCM audio, shape (n_samples,),
            at params.sample_rate.  Values in [-1.0, 1.0].
        """
        ...

    def synthesize_phoneme(self, phoneme: PhonemeParams, sample_rate: int) -> np.ndarray:
        """
        Synthesize a single phoneme.  Used for real-time keystroke mode.

        Backends that can't cheaply synthesize individual phonemes should
        synthesize a one-element SynthParams and return the result.
        """
        ...


# ---------------------------------------------------------------------------
# Null backend — useful for testing the predictor without audio output
# ---------------------------------------------------------------------------

class NullBackend:
    """
    Synthesis backend that returns silence.

    Used in unit tests and CI environments where eSpeak is not installed.
    Satisfies the SynthBackend protocol without producing any audio.
    """

    @property
    def name(self) -> str:
        return "null"

    @property
    def available(self) -> bool:
        return True

    def synthesize(self, params: SynthParams) -> np.ndarray:
        n_samples = int(params.total_duration_s * params.sample_rate)
        return np.zeros(n_samples, dtype=np.float32)

    def synthesize_phoneme(self, phoneme: PhonemeParams, sample_rate: int) -> np.ndarray:
        n_samples = int(phoneme.duration_s * sample_rate)
        return np.zeros(n_samples, dtype=np.float32)


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def hz_to_espeak_pitch(f0_hz: float, baseline_hz: float = 120.0) -> int:
    """
    Convert F0 in Hz to eSpeak's -p pitch parameter (0–99, default ~50).

    eSpeak's pitch parameter is roughly linear in semitones relative to a
    speaker-dependent baseline.  We treat 120 Hz as pitch=50 (neutral) and
    scale ±1 semitone ≈ ±2 pitch units, clamped to [0, 99].

    This is an approximation — eSpeak's actual mapping is voice-dependent.
    The neural predictor should learn to compensate for this quantisation.
    """
    import math
    if f0_hz <= 0:
        return 50
    semitones = 12 * math.log2(f0_hz / baseline_hz)
    pitch = int(50 + semitones * 2)
    return max(0, min(99, pitch))


def rate_factor_to_espeak_wpm(rate: float, baseline_wpm: int = 150) -> int:
    """
    Convert a rate multiplier to eSpeak words-per-minute.

    eSpeak default is ~175 wpm.  We use 150 as a slightly slower neutral
    that improves intelligibility on embedded targets.  Clamped to [40, 450].
    """
    wpm = int(baseline_wpm * rate)
    return max(40, min(450, wpm))


def energy_to_espeak_volume(energy: float) -> int:
    """
    Convert 0.0–1.0 energy to eSpeak -a amplitude (0–200, default 100).
    """
    return max(0, min(200, int(energy * 100)))
