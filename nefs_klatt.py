"""
nefs_klatt.py — Klatt (1980) cascade/parallel formant synthesizer

Implements the core Klatt 1980 speech synthesizer as described in:
    Klatt, D.H. (1980). "Software for a cascade/parallel formant
    synthesizer." JASA 67(3), 971-995.

Architecture
------------
                    ┌─────────────────────────────────┐
  Voiced source     │  Glottal pulse (LF-approximate)  │
  (AV > 0)          │  + Aspiration noise (AH)          │
                    └──────────────┬──────────────────-─┘
                                   │
                    ┌──────────────▼──────────────────-─┐
  Cascade path      │  R1 → R2 → R3 → R4 → R5           │  vowels,
  (voiced sounds)   │  Nasal pole + Nasal zero (nasals)  │  sonorants
                    └──────────────┬──────────────────-─┘
                                   │
                    ┌──────────────▼──────────────────-─┐
  Parallel path     │  Frication noise → RF1‖RF2‖RF3‖RF4│  fricatives,
  (unvoiced)        │  (parallel bank, summed)           │  affricates
                    └──────────────┬──────────────────-─┘
                                   │
                    ┌──────────────▼──────────────────-─┐
  Output            │  Radiation filter (first diff)     │
                    │  + Low-pass anti-alias             │
                    └─────────────────────────────────-──┘

Parameter frame rate: 5 ms (200 frames/sec at 10 kHz, or 100 frames/sec
at 22050 Hz with hop=220 samples). Intermediate values are linearly
interpolated sample-by-sample between frames to avoid discontinuities.

This implementation targets CPU real-time synthesis for the NEFS
keyboard instrument. It does not require a GPU or any neural network
at runtime — the neural predictor (nefs_param_predictor.py) runs once
per utterance to produce parameter frames, then this engine renders
audio sample-by-sample.

SynthBackend compatibility
--------------------------
KlattBackend implements the SynthBackend protocol from
nefs_synth_interface.py.  Swap it in with one line in nefs_espeak_rt.py.

Usage
-----
    from nefs_klatt import KlattBackend, KlattVoicePreset
    from nefs_synth_interface import SynthParams, PhonemeParams

    # Create a voice preset for a specific NPC character
    goblin = KlattVoicePreset.goblin()

    backend = KlattBackend(preset=goblin, sample_rate=22050)

    # Synthesize from SynthParams (produced by NEFSParamPredictor)
    params = SynthParams(phonemes=[...])
    audio = backend.synthesize(params)

    # Or synthesize a single phoneme (keystroke mode)
    audio = backend.synthesize_phoneme(phoneme_params, sample_rate=22050)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from nefs_synth_interface import PhonemeParams, SynthBackend, SynthParams
from nefs_converter import classify_byte, is_voiced, phoneme_class, place, manner


# ---------------------------------------------------------------------------
# Resonator (biquad bandpass filter)
# ---------------------------------------------------------------------------

class Resonator:
    """
    Second-order recursive bandpass filter (one Klatt formant).

    Implements the difference equation:
        y[n] = A * x[n]  -  B * y[n-1]  -  C * y[n-2]

    Coefficients are recomputed whenever F or BW changes.
    State (y[n-1], y[n-2]) is preserved across samples.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr    = sample_rate
        self.A     = 0.0
        self.B     = 0.0
        self.C     = 0.0
        self._y1   = 0.0   # y[n-1]
        self._y2   = 0.0   # y[n-2]

    def set_params(self, freq: float, bandwidth: float):
        """
        Update filter coefficients for centre frequency (Hz) and
        bandwidth (Hz).  Called once per parameter frame.
        """
        if freq <= 0 or bandwidth <= 0:
            self.A = self.B = self.C = 0.0
            return
        r   = math.exp(-math.pi * bandwidth / self.sr)
        cos_w = math.cos(2.0 * math.pi * freq / self.sr)
        self.C = -(r * r)
        self.B = 2.0 * r * cos_w
        self.A = 1.0 - self.B - self.C   # normalise for unity gain at F

    def process_sample(self, x: float) -> float:
        """Run one sample through the resonator."""
        y = self.A * x + self.B * self._y1 + self.C * self._y2
        self._y2 = self._y1
        self._y1 = y
        return y

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Process a block of samples (faster than per-sample loop in Python)."""
        out = np.zeros_like(x)
        y1, y2 = self._y1, self._y2
        A, B, C = self.A, self.B, self.C
        for i, xi in enumerate(x):
            y = A * xi + B * y1 + C * y2
            y2 = y1
            y1 = y
            out[i] = y
        self._y1 = y1
        self._y2 = y2
        return out

    def reset(self):
        self._y1 = self._y2 = 0.0


class AntiResonator:
    """
    Second-order notch (anti-formant) filter.
    Used for the nasal zero in nasal consonants.

    Implements:  y[n] = A * x[n]  +  B * x[n-1]  +  C * x[n-2]
    (FIR — no feedback, so always stable.)
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr  = sample_rate
        self.A   = 1.0
        self.B   = 0.0
        self.C   = 0.0
        self._x1 = 0.0
        self._x2 = 0.0

    def set_params(self, freq: float, bandwidth: float):
        if freq <= 0 or bandwidth <= 0:
            self.A = 1.0
            self.B = self.C = 0.0
            return
        r       = math.exp(-math.pi * bandwidth / self.sr)
        cos_w   = math.cos(2.0 * math.pi * freq / self.sr)
        # Coefficients are the negative of the resonator, then divided by A
        # so the notch has unity gain away from the null.
        C = -(r * r)
        B =  2.0 * r * cos_w
        A =  1.0 - B - C
        self.A =  1.0 / A
        self.B = -B / A
        self.C = -C / A

    def process_sample(self, x: float) -> float:
        y = self.A * x + self.B * self._x1 + self.C * self._x2
        self._x2 = self._x1
        self._x1 = x
        return y

    def reset(self):
        self._x1 = self._x2 = 0.0


# ---------------------------------------------------------------------------
# Glottal source (Liljencrants-Fant approximation)
# ---------------------------------------------------------------------------

class GlottalSource:
    """
    Quasi-periodic voiced source generator.

    Approximates the Liljencrants-Fant (LF) glottal pulse using a
    polynomial rise and exponential decay within each pitch period.
    This is the classic Klatt 1980 approach — simpler than the full LF
    model but perceptually close.

    The pulse shape is controlled by:
        open_quotient   — fraction of period the glottis is open (0.5–0.8)
        skew            — asymmetry of the opening phase (0 = symmetric)

    Jitter and shimmer (micro-perturbations) are added to make the
    source sound less mechanical.  These are the key parameters for
    making Klatt output sound more human.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr           = sample_rate
        self._phase       = 0.0    # current position in pitch period [0, 1)
        self._period      = 100    # samples per pitch period
        self._amp         = 0.8    # current amplitude (shimmer target)
        self._open_q      = 0.65   # open quotient
        self._jitter_amt  = 0.003  # fraction of period for jitter
        self._shimmer_amt = 0.02   # fraction of amplitude for shimmer
        self._rng         = np.random.default_rng(42)

    def set_f0(self, f0_hz: float):
        """Update fundamental frequency."""
        if f0_hz > 0:
            self._period = self.sr / f0_hz
        else:
            self._period = self.sr / 120.0  # default if zero

    def set_open_quotient(self, oq: float):
        self._open_q = float(np.clip(oq, 0.3, 0.9))

    def set_jitter(self, amount: float):
        """Jitter: cycle-to-cycle F0 variation. 0=none, 0.01=1% (natural ~0.3%)."""
        self._jitter_amt = float(np.clip(amount, 0.0, 0.05))

    def set_shimmer(self, amount: float):
        """Shimmer: cycle-to-cycle amplitude variation. 0=none, 0.03=3% (natural ~1.5%)."""
        self._shimmer_amt = float(np.clip(amount, 0.0, 0.1))

    def generate(self, n_samples: int, amp: float = 0.8) -> np.ndarray:
        """
        Generate n_samples of glottal source signal.

        Returns float32 array in [-1, 1].
        """
        out     = np.zeros(n_samples, dtype=np.float32)
        phase   = self._phase
        period  = self._period
        open_q  = self._open_q

        # Add jitter: randomise period slightly each cycle
        cycle_period = period * (
            1.0 + self._rng.normal(0, self._jitter_amt)
        )
        # Shimmer: randomise amplitude each cycle
        cycle_amp = amp * (
            1.0 + self._rng.normal(0, self._shimmer_amt)
        )

        for i in range(n_samples):
            if phase >= 1.0:
                phase -= 1.0
                # New cycle — update jitter/shimmer
                cycle_period = period * max(
                    0.5, 1.0 + self._rng.normal(0, self._jitter_amt)
                )
                cycle_amp = amp * max(
                    0.1, 1.0 + self._rng.normal(0, self._shimmer_amt)
                )

            if phase < open_q:
                # Opening phase: polynomial rise then fall
                # Normalised within open phase: 0..1
                t = phase / open_q
                # LF-approximate: x = t^2 * (3 - 2t)  (smooth step)
                pulse = t * t * (3.0 - 2.0 * t)
                # Convert to derivative (flow velocity ≈ dU/dt)
                # Simple: centre the pulse at 0 by subtracting mean
                out[i] = float(cycle_amp * (pulse - 0.5) * 2.0)
            else:
                # Closed phase: rapid return (abrupt closure creates HF energy)
                t = (phase - open_q) / (1.0 - open_q)
                # Exponential decay during closure
                out[i] = float(cycle_amp * math.exp(-8.0 * t) * -0.5)

            phase += 1.0 / cycle_period

        self._phase = phase
        return out

    def reset(self):
        self._phase = 0.0


# ---------------------------------------------------------------------------
# Voice presets
# ---------------------------------------------------------------------------

@dataclass
class KlattVoicePreset:
    """
    Per-character voice configuration.

    These values are multiplied with / added to the predictor outputs,
    so the neural predictor handles relative variation while the preset
    defines the character's base voice identity.

    All frequencies in Hz, durations implicit via rate_multiplier.
    """
    name: str = "default"

    # Source parameters
    f0_base_hz:       float = 120.0   # baseline F0 (predictor output is added)
    f0_range_scale:   float = 1.0     # scale predictor F0 variation
    open_quotient:    float = 0.65    # glottal open quotient
    breathiness:      float = 0.05    # aspiration noise level (0–1)
    creakiness:       float = 0.02    # creaky voice (low OQ + irregular)
    jitter:           float = 0.003   # F0 micro-variation
    shimmer:          float = 0.02    # amplitude micro-variation

    # Formant offsets (added to predictor F1/F2/F3 outputs)
    f1_offset_hz:     float = 0.0
    f2_offset_hz:     float = 0.0
    f3_offset_hz:     float = 0.0

    # Formant bandwidths (Hz) — wider = more nasal/breathy, narrower = clearer
    b1_hz:            float = 80.0
    b2_hz:            float = 100.0
    b3_hz:            float = 120.0
    b4_hz:            float = 200.0
    b5_hz:            float = 300.0

    # Speaking rate multiplier (applied on top of predictor rate)
    rate_multiplier:  float = 1.0

    # Amplitude
    amplitude_db:     float = 0.0     # overall gain in dB relative to neutral

    # Nasal resonances
    fn_hz:            float = 270.0   # nasal formant frequency
    bn_hz:            float = 100.0   # nasal formant bandwidth
    fnz_hz:           float = 320.0   # nasal anti-formant
    bnz_hz:           float = 100.0

    @classmethod
    def default(cls) -> 'KlattVoicePreset':
        return cls(name="default")

    @classmethod
    def goblin(cls) -> 'KlattVoicePreset':
        return cls(
            name="goblin",
            f0_base_hz=200.0,
            f0_range_scale=1.8,     # exaggerated pitch variation
            open_quotient=0.75,
            breathiness=0.12,
            jitter=0.008,
            shimmer=0.04,
            f1_offset_hz=80.0,      # raised formants = smaller tract
            f2_offset_hz=150.0,
            f3_offset_hz=100.0,
            rate_multiplier=1.3,    # faster, more erratic
            amplitude_db=2.0,
        )

    @classmethod
    def dwarf(cls) -> 'KlattVoicePreset':
        return cls(
            name="dwarf",
            f0_base_hz=85.0,
            f0_range_scale=0.7,     # gruff, less expressive pitch
            open_quotient=0.55,
            breathiness=0.03,
            creakiness=0.06,
            jitter=0.004,
            shimmer=0.025,
            f1_offset_hz=-40.0,     # lowered formants = larger tract
            f2_offset_hz=-80.0,
            b1_hz=100.0,
            rate_multiplier=0.85,
            amplitude_db=3.0,
        )

    @classmethod
    def ethereal_elf(cls) -> 'KlattVoicePreset':
        return cls(
            name="ethereal_elf",
            f0_base_hz=180.0,
            f0_range_scale=1.3,
            open_quotient=0.80,     # very breathy
            breathiness=0.25,
            jitter=0.001,           # unnaturally smooth
            shimmer=0.008,
            f1_offset_hz=20.0,
            f2_offset_hz=50.0,
            b1_hz=60.0,             # narrow bandwidths = clear, ringing
            b2_hz=70.0,
            rate_multiplier=0.9,
            amplitude_db=-1.0,
        )

    @classmethod
    def dragon(cls) -> 'KlattVoicePreset':
        return cls(
            name="dragon",
            f0_base_hz=55.0,
            f0_range_scale=0.5,     # nearly monotone
            open_quotient=0.50,
            breathiness=0.08,
            creakiness=0.15,
            jitter=0.006,
            shimmer=0.05,
            f1_offset_hz=-80.0,
            f2_offset_hz=-180.0,
            f3_offset_hz=-120.0,
            b1_hz=130.0,            # broad bandwidths = resonant cave sound
            b2_hz=180.0,
            b3_hz=250.0,
            rate_multiplier=0.6,
            amplitude_db=6.0,
        )

    @classmethod
    def snake_cultist(cls) -> 'KlattVoicePreset':
        return cls(
            name="snake_cultist",
            f0_base_hz=110.0,
            f0_range_scale=0.6,     # mostly flat with sudden rises
            open_quotient=0.82,
            breathiness=0.20,
            jitter=0.002,
            shimmer=0.015,
            f1_offset_hz=10.0,
            f2_offset_hz=-30.0,
            rate_multiplier=0.75,   # slow and deliberate
            amplitude_db=-2.0,
        )

    @classmethod
    def wizard(cls) -> 'KlattVoicePreset':
        return cls(
            name="wizard",
            f0_base_hz=100.0,
            f0_range_scale=1.1,
            open_quotient=0.58,
            breathiness=0.08,
            creakiness=0.08,        # old, slightly creaky
            jitter=0.005,
            shimmer=0.03,
            f1_offset_hz=-20.0,
            f2_offset_hz=-40.0,
            b1_hz=90.0,
            rate_multiplier=0.95,
            amplitude_db=0.0,
        )

    def amplitude_linear(self) -> float:
        return 10.0 ** (self.amplitude_db / 20.0)


# ---------------------------------------------------------------------------
# Per-frame Klatt parameters
# ---------------------------------------------------------------------------

@dataclass
class KlattFrame:
    """
    One parameter frame for the Klatt synthesizer.
    Typically 5–10 ms of speech.
    """
    # Source
    f0_hz:        float = 120.0
    av_db:        float = 60.0     # voiced amplitude (dB)
    ah_db:        float = 0.0      # aspiration amplitude (dB)
    af_db:        float = 0.0      # frication amplitude (dB)

    # Formants (cascade path — for voiced sounds)
    f1_hz:        float = 500.0
    f2_hz:        float = 1500.0
    f3_hz:        float = 2500.0
    f4_hz:        float = 3500.0
    f5_hz:        float = 4500.0

    # Formant bandwidths
    b1_hz:        float = 80.0
    b2_hz:        float = 100.0
    b3_hz:        float = 120.0
    b4_hz:        float = 200.0
    b5_hz:        float = 300.0

    # Frication resonator (parallel path — for fricatives)
    ff_hz:        float = 4500.0   # frication resonator centre
    bf_hz:        float = 1000.0   # frication resonator bandwidth

    # Nasal coupling
    fn_hz:        float = 270.0
    bn_hz:        float = 100.0
    fnz_hz:       float = 320.0
    bnz_hz:       float = 100.0
    nasal:        bool  = False

    # Voicing
    voiced:       bool  = True

    # Duration of this frame in samples
    n_samples:    int   = 220      # ~10ms at 22050Hz


# ---------------------------------------------------------------------------
# Phoneme → KlattFrames converter
# ---------------------------------------------------------------------------

# Default formant targets per phoneme class.
# These are approximate values based on Peterson & Barney (1952) and
# Klatt (1980) reference tables.
# The neural predictor overrides these with learned values when available.
_VOWEL_FORMANTS = {
    # NEFS byte : (F1, F2, F3)
    # Byte = (high_nibble << 4) | low_nibble — spec v3, Appendix A
    # F1/F2/F3 from Peterson & Barney (1952) / Hillenbrand et al. (1995)

    # Close vowels (row 0x_4)
    0xA4: (270,  2290, 3010),  # i
    0xB4: (235,  2100, 2860),  # y
    0xC4: (300,  1550, 2300),  # ɨ
    0xD4: (310,  1380, 2250),  # ʉ
    0xE4: (300,   900, 2250),  # ɯ
    0xF4: (300,   870, 2240),  # u

    # Close-mid vowels (row 0x_5)
    0xA5: (390,  2300, 2900),  # e
    0xB5: (370,  1900, 2700),  # ø
    0xC5: (400,  1500, 2400),  # ɘ
    0xD5: (400,  1300, 2200),  # ɵ
    0xE5: (400,   900, 2300),  # ɤ
    0xF5: (500,   900, 2500),  # o

    # Open-mid vowels (row 0x_6)
    0xA6: (580,  1820, 2650),  # ɛ
    0xB6: (490,  1350, 2300),  # œ
    0xC6: (500,  1400, 1800),  # ɜ
    0xD6: (500,  1100, 1700),  # ɞ
    0xE6: (760,  1100, 2670),  # ʌ
    0xF6: (570,   840, 2410),  # ɔ

    # Open vowels (row 0x_7)
    0xA7: (800,  1200, 2500),  # a
    0xB7: (700,   900, 2400),  # ɶ
    0xC7: (800,  1300, 2500),  # ä
    0xD7: (700,   900, 2300),  # ɒ with diaeresis
    0xE7: (710,  1100, 2540),  # ɑ
    0xF7: (700,   900, 2300),  # ɒ

    # Near-close vowels (row 0x_8)
    0xA8: (390,  1990, 2550),  # ɪ
    0xB8: (320,  1700, 2500),  # ʏ
    0xC8: (860,  1720, 2410),  # æ
    0xD8: (700,  1300, 2500),  # ɐ
    0xE8: (440,  1020, 2240),  # ʊ
    0xF8: (500,  1500, 2500),  # ə (schwa)
}

def phoneme_to_frames(
    phoneme: PhonemeParams,
    preset: KlattVoicePreset,
    sample_rate: int = 22050,
    frame_hop_ms: float = 5.0,
) -> List[KlattFrame]:
    """
    Convert one PhonemeParams into a list of KlattFrames.

    The predictor outputs one set of parameters per phoneme. We expand
    this into multiple frames covering the phoneme's duration, applying
    linear interpolation for smooth parameter transitions.

    The preset offsets and scalings are applied here so the Klatt engine
    receives absolute parameter values rather than relative ones.
    """
    b = phoneme.nefs_byte
    place  = (b >> 4) & 0x0F
    manner = b & 0x0F

    # Duration in samples
    duration_s = max(0.01, phoneme.duration_s / preset.rate_multiplier)
    n_total    = max(1, int(duration_s * sample_rate))
    hop_samp   = max(1, int(frame_hop_ms / 1000.0 * sample_rate))
    n_frames   = max(1, n_total // hop_samp)

    # Determine phoneme class from NEFS byte — spec v3 §3.2 / §4
    # high nibble = place (0x1_=bilabial ... 0xC_=glottal)
    # low  nibble = manner (0x_0=vd fric, 0x_1=vl fric, 0x_2/4/6=vd stop,
    #               0x_3/5/7=vl stop, 0x_8=nasal, 0x_9=approx,
    #               0x_A=lateral, 0x_B=click, 0x_C=implosive, 0x_D=trill)
    # Vowel: high >= 0xA AND 0x4 <= low <= 0xB

    is_vowel      = (place >= 0xA) and (0x4 <= manner <= 0xB)
    is_nasal      = (not is_vowel) and (manner == 0x8)
    is_lateral    = (not is_vowel) and (manner == 0xA)
    is_approx     = (not is_vowel) and (manner == 0x9)
    is_trill      = (not is_vowel) and (manner == 0xD)
    is_vd_fric    = (not is_vowel) and (manner == 0x0)
    is_vl_fric    = (not is_vowel) and (manner == 0x1)
    is_vd_stop    = (not is_vowel) and (manner in (0x2, 0x4, 0x6))
    is_vl_stop    = (not is_vowel) and (manner in (0x3, 0x5, 0x7))
    is_sonorant   = is_vowel or is_nasal or is_approx or is_lateral or is_trill
    is_voiced_ph  = is_vowel or is_nasal or is_approx or is_lateral or is_trill                     or is_vd_fric or is_vd_stop

    # F0 with preset scaling
    f0 = preset.f0_base_hz + phoneme.f0_hz * preset.f0_range_scale
    f0 = max(50.0, min(500.0, f0))

    # Formants — use predictor output if available, else look up table
    if phoneme.f1_hz and phoneme.f1_hz > 0:
        f1 = phoneme.f1_hz + preset.f1_offset_hz
        f2 = (phoneme.f2_hz or 1500.0) + preset.f2_offset_hz
        f3 = (phoneme.f3_hz or 2500.0) + preset.f3_offset_hz
    elif b in _VOWEL_FORMANTS:
        f1r, f2r, f3r = _VOWEL_FORMANTS[b]
        f1 = f1r + preset.f1_offset_hz
        f2 = f2r + preset.f2_offset_hz
        f3 = f3r + preset.f3_offset_hz
    else:
        # Consonant defaults — formants track neutral schwa
        f1 = 500.0 + preset.f1_offset_hz
        f2 = 1500.0 + preset.f2_offset_hz
        f3 = 2500.0 + preset.f3_offset_hz

    f1 = max(200.0, f1)
    f2 = max(600.0, f2)
    f3 = max(1500.0, f3)

    # Amplitude levels
    energy = max(0.01, phoneme.energy)
    av_db  = 20.0 * math.log10(energy + 1e-8) + 60.0

    # Aspiration from preset breathiness + phoneme breathiness
    breathiness = preset.breathiness + phoneme.breathiness * 0.5
    ah_db = (av_db - 20.0) * breathiness if is_voiced_ph and breathiness > 0.01 else 0.0

    # Frication
    af_db = av_db - 10.0 if (is_vl_fric or is_vd_fric) else 0.0

    # Fricative centre frequency by NEFS byte (spec v3 bytes)
    # Voiced fricatives: row 0x_0   Voiceless: row 0x_1
    # Frequencies: approximate spectral noise centre (Hz)
    ff_hz = {
        # Bilabial (col 0x1_)
        0x10: 800.0,    # β  voiced bilabial
        0x11: 900.0,    # ɸ  voiceless bilabial
        # Labiodental (col 0x2_)
        0x20: 900.0,    # v  voiced labiodental
        0x21: 1000.0,   # f  voiceless labiodental
        # Dental (col 0x3_)
        0x30: 5000.0,   # ð  voiced dental
        0x31: 5500.0,   # θ  voiceless dental
        # Alveolar (col 0x4_)
        0x40: 5500.0,   # z  voiced alveolar
        0x41: 5500.0,   # s  voiceless alveolar
        # Post-alveolar (col 0x5_)
        0x50: 2500.0,   # ʒ  voiced post-alveolar
        0x51: 2800.0,   # ʃ  voiceless post-alveolar
        # Retroflex (col 0x6_)
        0x60: 2800.0,   # ʐ  voiced retroflex
        0x61: 3000.0,   # ʂ  voiceless retroflex
        # Palatal (col 0x7_)
        0x70: 3200.0,   # ʝ  voiced palatal
        0x71: 3500.0,   # ç  voiceless palatal
        # Velar (col 0x8_)
        0x80: 1000.0,   # ɣ  voiced velar
        0x81: 1200.0,   # x  voiceless velar
        # Uvular (col 0x9_)
        0x90: 1000.0,   # ʁ  voiced uvular
        0x91: 1000.0,   # χ  voiceless uvular
        # Pharyngeal (col 0xA_)
        0xA0: 800.0,    # ʕ  voiced pharyngeal
        0xA1: 800.0,    # ħ  voiceless pharyngeal
        # Glottal (col 0xC_)
        0xC0: 2000.0,   # ɦ  voiced glottal
        0xC1: 2000.0,   # h  voiceless glottal
    }.get(b, 4000.0)

    frames = []
    for fi in range(n_frames):
        n_samp = hop_samp if fi < n_frames - 1 else (n_total - fi * hop_samp)
        n_samp = max(1, n_samp)

        frame = KlattFrame(
            f0_hz    = f0,
            av_db    = av_db if is_voiced_ph else 0.0,
            ah_db    = ah_db,
            af_db    = af_db,
            f1_hz    = f1,
            f2_hz    = f2,
            f3_hz    = f3,
            f4_hz    = 3500.0 + preset.f3_offset_hz * 0.5,
            f5_hz    = 4500.0,
            b1_hz    = preset.b1_hz,
            b2_hz    = preset.b2_hz,
            b3_hz    = preset.b3_hz,
            b4_hz    = preset.b4_hz,
            b5_hz    = preset.b5_hz,
            ff_hz    = ff_hz,
            bf_hz    = 1000.0 if b in (0x40, 0x41, 0x50, 0x51) else 1500.0,
            fn_hz    = preset.fn_hz,
            bn_hz    = preset.bn_hz,
            fnz_hz   = preset.fnz_hz,
            bnz_hz   = preset.bnz_hz,
            nasal    = is_nasal,
            voiced   = is_voiced_ph,
            n_samples= n_samp,
        )
        frames.append(frame)

    # Stop bursts: add a brief silent closure followed by burst
    if is_vl_stop or is_vd_stop:
        _apply_stop_envelope(frames, is_voiced_ph)

    return frames


def _apply_stop_envelope(frames: List[KlattFrame], voiced: bool):
    """
    Modify stop consonant frames to simulate:
    - Voiced stops: voicing bar during closure (weak low-F0 buzz)
    - Voiceless stops: complete silence during closure
    - Brief burst at release (~20% into frames)
    """
    n = len(frames)
    if n == 0:
        return

    closure_end = max(1, int(n * 0.7))  # 70% closure, 30% burst+release

    for i, f in enumerate(frames):
        if i < closure_end:
            # Closure: silence (or weak voicing bar for voiced stops)
            f.av_db = 30.0 if voiced else 0.0
            f.af_db = 0.0
        else:
            # Release burst: brief frication noise
            t = (i - closure_end) / max(1, n - closure_end)
            f.af_db = f.av_db * (1.0 - t)  # burst decays


# ---------------------------------------------------------------------------
# Klatt synthesizer engine
# ---------------------------------------------------------------------------

class KlattSynthesizer:
    """
    Core Klatt (1980) cascade/parallel formant synthesizer.

    Generates audio sample by sample from a list of KlattFrames.
    Parameter interpolation between frames is done linearly to avoid
    discontinuities (the main cause of buzzing artifacts in naive
    implementations).
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

        # Glottal source
        self.source = GlottalSource(sample_rate)

        # Cascade resonators (voiced path)
        self.r1 = Resonator(sample_rate)
        self.r2 = Resonator(sample_rate)
        self.r3 = Resonator(sample_rate)
        self.r4 = Resonator(sample_rate)
        self.r5 = Resonator(sample_rate)

        # Nasal coupling
        self.rn  = Resonator(sample_rate)
        self.rnz = AntiResonator(sample_rate)

        # Parallel frication resonator
        self.rf = Resonator(sample_rate)

        # Low-pass for aspiration/frication noise shaping
        self.lp = Resonator(sample_rate)
        self.lp.set_params(500.0, 1000.0)  # gentle LP shape

        # Radiation filter state (first difference: y[n] = x[n] - x[n-1])
        self._rad_prev = 0.0

        # RNG for noise sources
        self._rng = np.random.default_rng(0)

    def reset(self):
        """Reset all filter states (use between utterances)."""
        for r in (self.r1, self.r2, self.r3, self.r4, self.r5,
                  self.rn, self.rf, self.lp):
            r.reset()
        self.rnz.reset()
        self.source.reset()
        self._rad_prev = 0.0

    def synthesize_frames(self, frames: List[KlattFrame]) -> np.ndarray:
        """
        Synthesize audio from a list of KlattFrames.

        Linearly interpolates parameters between consecutive frames
        to avoid discontinuities.

        Returns float32 numpy array at self.sr sample rate.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)

        segments = []

        prev_frame = frames[0]
        for i, frame in enumerate(frames):
            n = frame.n_samples
            if n <= 0:
                continue

            # Get previous frame for interpolation
            if i == 0:
                pf = frame   # no interpolation on first frame
            else:
                pf = frames[i - 1]

            # Render this frame with interpolation from pf → frame
            seg = self._render_frame(frame, pf, n)
            segments.append(seg)

        if not segments:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(segments)
        return audio.astype(np.float32)

    def _render_frame(
        self,
        frame: KlattFrame,
        prev: KlattFrame,
        n_samples: int,
    ) -> np.ndarray:
        """
        Render one frame, interpolating parameters from prev to frame.
        """
        out = np.zeros(n_samples, dtype=np.float32)

        # dB → linear amplitudes
        av  = _db_to_lin(frame.av_db)
        ah  = _db_to_lin(frame.ah_db)
        af  = _db_to_lin(frame.af_db)
        pav = _db_to_lin(prev.av_db)
        pah = _db_to_lin(prev.ah_db)
        paf = _db_to_lin(prev.af_db)

        # Update source F0 (constant across frame — changes on next frame)
        self.source.set_f0(frame.f0_hz)

        # Generate source signals for entire frame
        voiced_src = self.source.generate(n_samples, amp=1.0) if frame.voiced else \
                     np.zeros(n_samples, dtype=np.float32)

        # White noise for aspiration and frication
        noise = self._rng.standard_normal(n_samples).astype(np.float32) * 0.5

        # Low-pass shaped aspiration (breathiness)
        aspiration = self.lp.process_block(noise) * 0.3

        # Parameter interpolation coefficients
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)

        # Interpolate formant frequencies
        f1 = _lerp(prev.f1_hz, frame.f1_hz, t)
        f2 = _lerp(prev.f2_hz, frame.f2_hz, t)
        f3 = _lerp(prev.f3_hz, frame.f3_hz, t)

        # Interpolate amplitudes
        av_env  = _lerp(pav, av, t)
        ah_env  = _lerp(pah, ah, t)
        af_env  = _lerp(paf, af, t)

        # Update resonator coefficients at start of frame
        # (per-sample coefficient update would be more accurate but too slow in Python;
        #  frame-level update is standard Klatt practice)
        self.r1.set_params(frame.f1_hz, frame.b1_hz)
        self.r2.set_params(frame.f2_hz, frame.b2_hz)
        self.r3.set_params(frame.f3_hz, frame.b3_hz)
        self.r4.set_params(frame.f4_hz, frame.b4_hz)
        self.r5.set_params(frame.f5_hz, frame.b5_hz)
        self.rn.set_params(frame.fn_hz,  frame.bn_hz)
        self.rnz.set_params(frame.fnz_hz, frame.bnz_hz)
        self.rf.set_params(frame.ff_hz,  frame.bf_hz)

        # --- Voiced cascade path ---
        cascade_in = voiced_src * av_env + aspiration * ah_env

        # Cascade: R1 → R2 → R3 → R4 → R5
        x = self.r1.process_block(cascade_in)
        x = self.r2.process_block(x)
        x = self.r3.process_block(x)
        x = self.r4.process_block(x)
        x = self.r5.process_block(x)

        # Nasal coupling (adds nasal resonance for nasal phonemes)
        if frame.nasal:
            xn = self.rn.process_block(cascade_in)
            xn = self.rnz.process_sample(xn[0]) if len(xn) > 0 else xn
            # Full block anti-resonator
            xn_out = np.array([self.rnz.process_sample(s) for s in self.rn.process_block(cascade_in)])
            x = x + xn_out * 0.3

        # --- Parallel frication path ---
        fric_noise  = noise * af_env
        fric_out    = self.rf.process_block(fric_noise)

        # Mix cascade (voiced) + parallel (frication)
        if frame.voiced:
            mixed = x * 0.85 + fric_out * 0.15
        else:
            mixed = fric_out

        # Radiation filter: y[n] = x[n] - x[n-1]
        rad_out = np.zeros_like(mixed)
        prev_s  = self._rad_prev
        for i in range(len(mixed)):
            rad_out[i] = mixed[i] - prev_s
            prev_s = mixed[i]
        self._rad_prev = prev_s

        out = rad_out
        return out

    def _apply_radiation(self, x: np.ndarray) -> np.ndarray:
        """First-difference radiation filter."""
        out = np.empty_like(x)
        prev = self._rad_prev
        for i in range(len(x)):
            out[i] = x[i] - prev
            prev = x[i]
        self._rad_prev = prev
        return out


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _db_to_lin(db: float) -> float:
    """Convert dB amplitude to linear, returning 0 for db <= -90."""
    if db <= -90.0:
        return 0.0
    return 10.0 ** (db / 20.0)


def _lerp(a: float, b: float, t: np.ndarray) -> np.ndarray:
    """Linear interpolation from scalar a to scalar b over array t in [0,1]."""
    return (a + (b - a) * t).astype(np.float32)


# ---------------------------------------------------------------------------
# SynthBackend implementation
# ---------------------------------------------------------------------------

class KlattBackend:
    """
    Klatt synthesizer as a SynthBackend.

    Drop-in replacement for ESpeakBackend. Change one line in
    NEFSRealTimeSynth.__init__ to activate:

        self._backend = KlattBackend(preset=KlattVoicePreset.default())
    """

    def __init__(
        self,
        preset: Optional[KlattVoicePreset] = None,
        sample_rate: int = 22050,
        frame_hop_ms: float = 5.0,
    ):
        self.preset       = preset or KlattVoicePreset.default()
        self._sample_rate = sample_rate
        self.frame_hop_ms = frame_hop_ms
        self._synth       = KlattSynthesizer(sample_rate)

    @property
    def name(self) -> str:
        return f"klatt:{self.preset.name}"

    @property
    def available(self) -> bool:
        return True   # no external dependencies

    def set_preset(self, preset: KlattVoicePreset):
        """Hot-swap voice preset (e.g. when switching NPC)."""
        self.preset = preset
        self._synth.reset()

    def synthesize(self, params: SynthParams) -> np.ndarray:
        """Synthesize full utterance from SynthParams."""
        if not params.phonemes:
            return np.zeros(0, dtype=np.float32)

        all_frames = []
        for phoneme in params.phonemes:
            frames = phoneme_to_frames(
                phoneme,
                self.preset,
                self._sample_rate,
                self.frame_hop_ms,
            )
            all_frames.extend(frames)

        if not all_frames:
            return np.zeros(0, dtype=np.float32)

        audio = self._synth.synthesize_frames(all_frames)

        # Normalise and apply preset gain
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.85 * self.preset.amplitude_linear()

        return audio.astype(np.float32)

    def synthesize_phoneme(
        self,
        phoneme: PhonemeParams,
        sample_rate: int,
    ) -> np.ndarray:
        """Synthesize a single phoneme (keystroke mode)."""
        single = SynthParams(phonemes=[phoneme], sample_rate=sample_rate)
        return self.synthesize(single)


# ---------------------------------------------------------------------------
# Quick smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import wave
    import struct

    print("Klatt synthesizer smoke test")
    print("=" * 40)

    sr = 22050
    backend = KlattBackend(sample_rate=sr)

    from nefs_synth_interface import PhonemeParams, SynthParams
    from nefs_converter import NEFSConverter, IPA_TO_NEFS
    c = NEFSConverter()

    # "hello" — hɛloʊ — C1 A6 4A F5 E8
    phonemes = [
        PhonemeParams(nefs_byte=0xC1, ipa='h',  f0_hz=0.0,   duration_s=0.07, energy=0.6, breathiness=0.2),
        PhonemeParams(nefs_byte=0xA6, ipa='ɛ',  f0_hz=130.0, duration_s=0.10, energy=0.9, f1_hz=660.0, f2_hz=1720.0, f3_hz=2410.0),
        PhonemeParams(nefs_byte=0x4A, ipa='l',  f0_hz=125.0, duration_s=0.07, energy=0.8),
        PhonemeParams(nefs_byte=0xF5, ipa='o',  f0_hz=115.0, duration_s=0.09, energy=0.9, f1_hz=500.0, f2_hz=1000.0, f3_hz=2500.0),
        PhonemeParams(nefs_byte=0xE8, ipa='ʊ',  f0_hz=100.0, duration_s=0.06, energy=0.7, f1_hz=440.0, f2_hz=1020.0, f3_hz=2240.0),
    ]

    params = SynthParams(phonemes=phonemes, sample_rate=sr)

    presets = [
        ('default',       KlattVoicePreset.default()),
        ('goblin',        KlattVoicePreset.goblin()),
        ('dwarf',         KlattVoicePreset.dwarf()),
        ('ethereal_elf',  KlattVoicePreset.ethereal_elf()),
        ('dragon',        KlattVoicePreset.dragon()),
        ('snake_cultist', KlattVoicePreset.snake_cultist()),
        ('wizard',        KlattVoicePreset.wizard()),
    ]

    for name, preset in presets:
        backend.set_preset(preset)
        audio = backend.synthesize(params)
        duration_ms = len(audio) / sr * 1000

        # Write WAV
        fname = f"hello_{name}.wav"
        with wave.open(fname, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(pcm16.tobytes())

        print(f"  {name:<16}: {len(audio)} samples ({duration_ms:.0f}ms) → {fname}")

    print("\nDone. Listen to the .wav files to compare voice presets.")
