"""
test_polly.py — End-to-end NEFS → Polly synthesis test.

Usage:
    # Set credentials first:
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1

    python test_polly.py
    python test_polly.py --text "your custom text here"
    python test_polly.py --checkpoint best.pt --text "hello world"

Output:
    test_out.wav — synthesized audio
    test_ssml.xml — the SSML sent to Polly (for debugging)
"""

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_polly")


# ---------------------------------------------------------------------------
# Imports — fail loudly with clear install instructions
# ---------------------------------------------------------------------------

try:
    from nefs_converter import NEFSConverter
except ImportError:
    sys.exit("Could not import nefs_converter.py — make sure it's in the same directory.")

try:
    from nefs_param_predictor import NEFSPredictor
except ImportError:
    sys.exit("Could not import nefs_param_predictor.py — make sure it's in the same directory.")

try:
    from nefs_synth_interface import SynthParams
except ImportError:
    sys.exit("Could not import nefs_synth_interface.py — make sure it's in the same directory.")

try:
    from nefs_polly_backend import PollyBackend
except ImportError:
    sys.exit("Could not import nefs_polly_backend.py — make sure it's in the same directory.")

try:
    from nefs_polly_postprocess import postprocess
    _POSTPROCESS_AVAILABLE = True
except ImportError:
    _POSTPROCESS_AVAILABLE = False

try:
    import boto3  # noqa: F401
except ImportError:
    sys.exit("boto3 not installed. Run: pip install boto3")


# ---------------------------------------------------------------------------
# WAV writer — no soundfile dependency needed
# ---------------------------------------------------------------------------

def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    """Save float32 audio array as a 16-bit WAV file."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    logger.info(f"Saved {path} ({len(pcm)/sample_rate:.2f}s at {sample_rate}Hz)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NEFS → Polly end-to-end test")
    parser.add_argument(
        "--text",
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to synthesize (default: pangram)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to NEFSPredictor checkpoint .pt file (default: rule-based fallback)",
    )
    parser.add_argument(
        "--voice",
        default="Joanna",
        help="Polly neural voice ID (default: Joanna)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--out",
        default="test_out.wav",
        help="Output WAV file path (default: test_out.wav)",
    )
    parser.add_argument(
        "--ssml-out",
        default="test_ssml.xml",
        help="Output SSML file path for debugging (default: test_ssml.xml)",
    )
    args = parser.parse_args()

    # --- Step 1: Check credentials ---
    logger.info("Checking AWS credentials...")
    backend = PollyBackend(voice_id=args.voice, region_name=args.region)
    if not backend.available:
        sys.exit(
            "Polly is not reachable. Check that your AWS credentials are set:\n"
            "  export AWS_ACCESS_KEY_ID=AKIA...\n"
            "  export AWS_SECRET_ACCESS_KEY=...\n"
            "  export AWS_DEFAULT_REGION=us-east-1"
        )
    logger.info(f"Polly backend ready: {backend.name}")

    # --- Step 2: G2P — text → IPA ---
    # We use eSpeak for G2P (text → IPA) since the predictor operates on
    # NEFS bytes, not raw text.  eSpeak is only used here for G2P, NOT for
    # synthesis — Polly handles synthesis.
    logger.info(f"Input text: {args.text!r}")
    logger.info("Converting text → IPA via eSpeak G2P...")
    ipa_string = _text_to_ipa(args.text)
    if not ipa_string:
        sys.exit("eSpeak G2P failed or returned empty IPA. Is espeak-ng installed?")
    logger.info(f"IPA: {ipa_string!r}")

    # --- Step 3: IPA → NEFS bytes ---
    logger.info("Converting IPA → NEFS bytes...")
    converter = NEFSConverter()
    nefs_bytes = converter.ipa_to_nefs(ipa_string)
    logger.info(
        f"NEFS: {len(nefs_bytes)} bytes — "
        + " ".join(f"0x{b:02X}" for b in nefs_bytes[:20])
        + ("..." if len(nefs_bytes) > 20 else "")
    )

    # --- Step 4: NEFS bytes → SynthParams ---
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint and not checkpoint.exists():
        logger.warning(f"Checkpoint not found at {checkpoint} — using rule-based fallback.")
        checkpoint = None

    logger.info(
        f"Running predictor ({'neural: ' + str(checkpoint) if checkpoint else 'rule-based fallback'})..."
    )
    predictor = NEFSPredictor(checkpoint_path=checkpoint)
    params = predictor.predict(nefs_bytes)
    logger.info(
        f"SynthParams: {len(params.phonemes)} phonemes, "
        f"estimated duration {params.total_duration_s:.2f}s"
    )

    # --- Step 5: Save SSML for debugging ---
    ssml = backend.ssml_preview(params)
    with open(args.ssml_out, "w", encoding="utf-8") as f:
        f.write(ssml)
    logger.info(f"SSML written to {args.ssml_out} ({len(ssml)} chars)")

    # --- Step 6: Synthesize via Polly ---
    logger.info("Sending to Polly...")
    try:
        audio = backend.synthesize(params)
    except RuntimeError as e:
        sys.exit(f"Polly synthesis failed: {e}")

    logger.info(
        f"Audio received: {len(audio)} samples, "
        f"{len(audio)/params.sample_rate:.2f}s"
    )

    # --- Step 7: Postprocess — warp F0 and duration to match predictor ---
    if _POSTPROCESS_AVAILABLE:
        logger.info("Applying predictor prosody (F0 warp + time stretch)...")
        audio = postprocess(audio, params, params.sample_rate)
        logger.info(f"After postprocess: {len(audio)/params.sample_rate:.2f}s")
    else:
        logger.warning(
            "pyworld/librosa not installed — skipping postprocessing.\n"
            "Install with: pip install pyworld librosa"
        )

    # --- Step 8: Save WAV ---
    save_wav(args.out, audio, params.sample_rate)
    print(f"\nDone. Output: {args.out}")
    print(f"SSML debug: {args.ssml_out}")


# ---------------------------------------------------------------------------
# eSpeak G2P helper (text → IPA string, no synthesis)
# ---------------------------------------------------------------------------

def _text_to_ipa(text: str) -> str:
    """
    Use eSpeak-NG in G2P-only mode to convert text to IPA.

    This is the ONLY place eSpeak is used in this script.
    It's purely for grapheme-to-phoneme conversion, not synthesis.
    eSpeak's audio output is not used.
    """
    import subprocess
    import re

    try:
        result = subprocess.run(
            ["espeak-ng", "--ipa", "-q", "--", text],
            capture_output=True,
            timeout=10,
        )
        # eSpeak always outputs UTF-8 — decode explicitly so Windows
        # PowerShell's default cp1252 encoding doesn't mangle IPA characters.
        result_stdout = result.stdout.decode("utf-8")
        result_stderr = result.stderr.decode("utf-8", errors="replace")
    except FileNotFoundError:
        logger.error(
            "espeak-ng not found. Install with:\n"
            "  Ubuntu/Debian: sudo apt install espeak-ng\n"
            "  macOS:         brew install espeak-ng\n"
            "  Windows:       https://github.com/espeak-ng/espeak-ng/releases"
        )
        return ""
    except subprocess.TimeoutExpired:
        logger.error("eSpeak timed out.")
        return ""

    if result.returncode != 0:
        logger.error(f"eSpeak error: {result_stderr.strip()}")
        return ""

    # eSpeak --ipa output includes stress markers and spaces between words.
    # Strip leading/trailing whitespace; keep stress diacritics (they are
    # valid NEFS bytes in the diacritic column).
    ipa = result_stdout.strip()

    # Remove word boundary spaces but keep IPA content
    ipa = re.sub(r'\s+', '', ipa)

    return ipa


if __name__ == "__main__":
    main()
