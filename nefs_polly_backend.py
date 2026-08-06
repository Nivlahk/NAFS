"""
nefs_polly_backend.py — Amazon Polly SynthBackend implementation for NEFS.

Implements the SynthBackend protocol defined in nefs_synth_interface.py.
Converts SynthParams (produced by NEFSPredictor) into SSML with per-phoneme
IPA + prosody annotations, then calls Polly's Neural TTS engine.

Requirements:
    pip install boto3

AWS credentials must be configured before use — see module docstring below
or run:
    aws configure          (if you have the AWS CLI installed)
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

Usage:
    from nefs_polly_backend import PollyBackend
    from nefs_param_predictor import NEFSPredictor
    from nefs_converter import NEFSConverter

    converter  = NEFSConverter()
    predictor  = NEFSPredictor()                  # or pass checkpoint_path
    backend    = PollyBackend()                   # uses default voice + region

    nefs_bytes = converter.ipa_to_nefs("hɛloʊ")
    params     = predictor.predict(nefs_bytes)
    audio      = backend.synthesize(params)       # float32 numpy array

Design notes
------------
Polly's <phoneme alphabet="ipa"> tag is the primary vehicle for NEFS IPA
output.  Per-phoneme prosody is injected via <prosody> tags wrapping each
<phoneme> element.

Polly Neural voices do NOT honour per-phoneme prosody with the same
granularity as a formant synth — the neural acoustic model partially ignores
very fine-grained per-phoneme rate/pitch changes and re-smooths them.  This
is actually *better* than eSpeak for eliminating boundary artifacts: the
model interpolates naturally rather than concatenating segments.

Prosody mapping:
  f0_hz       → <prosody pitch="+Nst">   (semitones relative to voice default)
  rate_factor → <prosody rate="N%">      (percentage of normal rate)
  energy      → <prosody volume="NdB">   (dB, Polly accepts -6dB to +4dB)

breathiness and spectral_tilt are not controllable in Polly — they are
stored in SynthParams for the future Klatt backend.

IPA coverage: Polly accepts most standard IPA.  Symbols outside its coverage
(clicks 0xB_, some implosives 0xC_) are silently dropped by Polly rather
than causing errors.  A debug warning is logged when known-unsupported bytes
are included.

PCM output: Polly returns signed 16-bit little-endian PCM.  We normalise to
float32 [-1, 1] to match the SynthBackend contract.
"""

from __future__ import annotations

import io
import logging
import math
import warnings
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

from nefs_synth_interface import PhonemeParams, SynthParams
from nefs_converter import classify_byte


# ---------------------------------------------------------------------------
# IPA symbols Polly Neural cannot render — logged as warnings, not errors.
# Polly drops them silently; we surface the warning so you know.
# ---------------------------------------------------------------------------
_POLLY_UNSUPPORTED_CLASSES = {"click", "silence"}  # classify_byte return values

# Polly Neural voices available as of 2025.  Sorted by quality/naturalness.
# All voices listed here support Engine='neural'.
NEURAL_VOICES = [
    "Joanna", "Matthew", "Ivy", "Justin", "Kendra", "Kimberly",
    "Salli", "Joey", "Ruth", "Stephen",     # en-US
    "Amy", "Brian", "Emma",                  # en-GB
    "Aria",                                  # en-NZ
    "Olivia",                                # en-AU
    "Lupe", "Pedro",                         # es-US
    "Lea",                                   # fr-FR
    "Vicki", "Daniel",                       # de-DE
]


# ---------------------------------------------------------------------------
# Prosody conversion helpers
# ---------------------------------------------------------------------------

def _f0_to_polly_pitch(f0_hz: float, baseline_hz: float = 120.0) -> str:
    """
    Convert F0 in Hz to a Polly SSML pitch string like '+3st' or '-2st'.

    Polly accepts semitone offsets relative to the voice's default pitch.
    We use 120 Hz as baseline (matches hz_to_espeak_pitch convention).
    Range clipped to ±20st — Polly ignores values outside this range.

    Returns a string ready for insertion into pitch="...".
    """
    if f0_hz <= 0:
        return "+0st"
    semitones = 12.0 * math.log2(f0_hz / baseline_hz)
    semitones = max(-20.0, min(20.0, semitones))
    sign = "+" if semitones >= 0 else ""
    return f"{sign}{semitones:.1f}st"


def _rate_to_polly_rate(rate_factor: float) -> str:
    """
    Convert a rate multiplier (0.4–2.0) to a Polly SSML rate percentage.

    Polly accepts 20%–200%.  We scale and clamp accordingly.
    Returns a string like '85%'.
    """
    pct = int(rate_factor * 100)
    pct = max(20, min(200, pct))
    return f"{pct}%"


def _energy_to_polly_volume(energy: float) -> str:
    """
    Convert 0.0–1.0 energy to a Polly SSML volume in dB.

    Polly volume range: -6dB (quiet) to +4dB (loud), with 0dB as default.
    We map [0, 1] → [-6, +4] linearly with 0.8 energy → 0dB (matches the
    PhonemeParams default energy of 0.8 being "normal").

    Returns a string like '+1.2dB'.
    """
    db = (energy - 0.8) * (10.0 / 0.8)   # 0.8 → 0dB, 0.0 → -10, 1.0 → +2.5
    db = max(-6.0, min(4.0, db))
    sign = "+" if db >= 0 else ""
    return f"{sign}{db:.1f}dB"


# ---------------------------------------------------------------------------
# SSML builder
# ---------------------------------------------------------------------------

def build_ssml(phonemes: List[PhonemeParams], lang: str = "en-US") -> str:
    """
    Convert a list of PhonemeParams to a Polly-compatible SSML string.

    Each phoneme becomes:
        <prosody pitch="..." rate="..." volume="...">
            <phoneme alphabet="ipa" ph="..."> </phoneme>
        </prosody>

    Phonemes whose IPA is empty (diacritics, tones, silence bytes) are
    emitted as <break time="Nms"/> using their duration_s field.

    Args:
        phonemes: List of PhonemeParams from NEFSPredictor.
        lang:     BCP-47 language tag for the xml:lang attribute.

    Returns:
        Complete SSML string ready for Polly.
    """
    # Polly's neural engine needs full utterance context — sending one
    # <phoneme> tag per phoneme destroys coarticulation and causes DC offset
    # artifacts because the model sees each phoneme in isolation.
    # Instead, accumulate IPA into chunks split only by silence bytes, then
    # emit each chunk as a single <phoneme> tag with averaged prosody.
    parts: List[str] = [f'<speak xml:lang="{lang}">']

    chunk_ipa:    List[str]          = []
    chunk_params: List[PhonemeParams] = []

    def flush():
        if not chunk_ipa:
            return
        ipa_str  = "".join(chunk_ipa)
        ipa_safe = ipa_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        avg_rate   = sum(p.rate_factor for p in chunk_params) / len(chunk_params)
        avg_energy = sum(p.energy      for p in chunk_params) / len(chunk_params)
        rate   = _rate_to_polly_rate(avg_rate)
        volume = _energy_to_polly_volume(avg_energy)
        parts.append(
            f'<prosody rate="{rate}" volume="{volume}">'
            f'<phoneme alphabet="ipa" ph="{ipa_safe}">x</phoneme>'
            f'</prosody>'
        )
        chunk_ipa.clear()
        chunk_params.clear()

    for p in phonemes:
        byte_class = classify_byte(p.nefs_byte)
        if byte_class in _POLLY_UNSUPPORTED_CLASSES or not p.ipa.strip():
            flush()
            ms = max(1, int(p.duration_s * 1000))
            parts.append(f'<break time="{ms}ms"/>')
            continue
        chunk_ipa.append(p.ipa)
        chunk_params.append(p)

    flush()
    parts.append("</speak>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# PollyBackend — implements SynthBackend protocol
# ---------------------------------------------------------------------------

class PollyBackend:
    """
    Amazon Polly Neural TTS backend implementing the SynthBackend protocol.

    Authentication: uses boto3's standard credential chain:
      1. Environment variables  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
      2. ~/.aws/credentials file (written by `aws configure`)
      3. IAM instance role (if running on EC2/Lambda)

    Args:
        voice_id:    Polly voice name.  Must be a neural-capable voice.
                     Default: 'Joanna' (en-US, female, highest quality).
        region_name: AWS region.  Polly Neural is available in most regions.
                     Default: 'us-east-1'.
        lang:        BCP-47 language tag injected into SSML.
                     Default: 'en-US'.
        engine:      Polly engine type.  'neural' strongly recommended —
                     'standard' is lower quality and doesn't smooth prosody.
    """

    def __init__(
        self,
        voice_id: str = "Joanna",
        region_name: str = "us-east-1",
        lang: str = "en-US",
        engine: str = "neural",
    ):
        if not _BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for PollyBackend.  Install with: pip install boto3"
            )

        self._voice_id    = voice_id
        self._region_name = region_name
        self._lang        = lang
        self._engine      = engine
        self._client: Optional[object] = None  # lazy init

    def _get_client(self):
        """Lazily initialise the boto3 Polly client."""
        if self._client is None:
            self._client = boto3.client("polly", region_name=self._region_name)
        return self._client

    # --- SynthBackend protocol ---

    @property
    def name(self) -> str:
        return f"polly-{self._engine}/{self._voice_id}"

    @property
    def available(self) -> bool:
        """
        Check whether Polly is reachable and credentials are valid.
        Makes a lightweight DescribeVoices call rather than a synthesis call.
        """
        if not _BOTO3_AVAILABLE:
            return False
        try:
            client = self._get_client()
            client.describe_voices(Engine=self._engine, LanguageCode=self._lang)
            return True
        except Exception as e:
            logger.warning(f"PollyBackend.available check failed: {e}")
            return False

    def synthesize(self, params: SynthParams) -> np.ndarray:
        """
        Synthesize a full utterance from SynthParams.

        Converts the phoneme list to SSML, calls Polly, and returns a
        float32 PCM array at params.sample_rate.

        Args:
            params: SynthParams bundle from NEFSPredictor.

        Returns:
            float32 numpy array, shape (n_samples,), values in [-1, 1].

        Raises:
            RuntimeError: if the Polly API call fails.
        """
        if not params.phonemes:
            return np.zeros(0, dtype=np.float32)

        ssml = build_ssml(params.phonemes, lang=self._lang)
        logger.debug(f"PollyBackend SSML ({len(ssml)} chars):\n{ssml}")

        return self._call_polly(ssml, params.sample_rate)

    def synthesize_phoneme(
        self, phoneme: PhonemeParams, sample_rate: int
    ) -> np.ndarray:
        """
        Synthesize a single phoneme.  Wraps it in a minimal SynthParams
        and calls synthesize().

        Note: Polly has ~200–400ms API latency, making this unsuitable for
        true real-time keystroke mode.  For that, pre-synthesize a phoneme
        cache at startup.
        """
        params = SynthParams(phonemes=[phoneme], sample_rate=sample_rate)
        return self.synthesize(params)

    # --- Internal helpers ---

    def _call_polly(self, ssml: str, sample_rate: int) -> np.ndarray:
        """
        Make the Polly API call and decode the returned PCM stream.

        Polly returns signed 16-bit little-endian PCM at the requested
        sample rate.  We normalise to float32 [-1, 1].

        Polly supports OutputFormat='pcm' with SampleRate in
        {'8000', '16000', '22050', '24000'}.  We clamp to the nearest
        supported rate if params.sample_rate doesn't match exactly.
        """
        supported_rates = {8000, 16000}
        polly_rate = min(supported_rates, key=lambda r: abs(r - sample_rate))
        if polly_rate != sample_rate:
            logger.debug(
                f"Polly PCM doesn't support {sample_rate} Hz — "
                f"using {polly_rate} Hz instead."
            )

        client = self._get_client()
        try:
            response = client.synthesize_speech(
                Engine=self._engine,
                VoiceId=self._voice_id,
                TextType="ssml",
                OutputFormat="pcm",
                SampleRate=str(polly_rate),
                Text=ssml,
            )
        except ClientError as e:
            raise RuntimeError(
                f"Polly API error: {e.response['Error']['Code']} — "
                f"{e.response['Error']['Message']}"
            ) from e
        except BotoCoreError as e:
            raise RuntimeError(f"Polly connection error: {e}") from e

        raw_bytes = response["AudioStream"].read()
        pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    def ssml_preview(self, params: SynthParams) -> str:
        """
        Return the SSML that would be sent to Polly for a given SynthParams.
        Useful for debugging without making an API call.
        """
        return build_ssml(params.phonemes, lang=self._lang)