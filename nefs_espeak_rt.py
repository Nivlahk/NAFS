"""
nefs_espeak_rt.py — Real-time eSpeak-ng backend for NEFS keyboard instrument

Slots into the existing pipeline as a drop-in synthesis backend while the
HiFi-GAN vocoder infrastructure remains intact.  Swap it out later by
replacing NEFSRealTimeSynth with NEFSHiFiGANSynthesizer.

Pipeline (with G2P assist):
    text → nefs_g2p → NEFS bytes → eSpeak-ng → audio → playback

Pipeline (expert / direct hex):
    NEFS bytes → IPA → eSpeak-ng → audio → playback

Latency target: < 50 ms per phoneme on CPU.

Dependencies:
    pip install sounddevice numpy
    sudo apt install espeak-ng        # Linux
    brew install espeak-ng            # macOS
    # Windows: https://github.com/espeak-ng/espeak-ng/releases
"""

from __future__ import annotations

import io
import logging
import subprocess
import threading
import queue
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully so the module always loads
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not installed — audio playback disabled. "
                   "Run: pip install sounddevice")

try:
    from nefs_wrapper import NEFSConverter
    _NEFS_CONVERTER = NEFSConverter()
except Exception as e:
    _NEFS_CONVERTER = None
    logger.warning(f"NEFSConverter unavailable ({e}) — direct hex input disabled.")

try:
    from nefs_g2p import text_to_ipa
    _G2P_AVAILABLE = True
except Exception as e:
    _G2P_AVAILABLE = False
    logger.warning(f"nefs_g2p unavailable ({e}) — text input disabled.")


# ---------------------------------------------------------------------------
# eSpeak-ng probe
# ---------------------------------------------------------------------------

def _espeak_available() -> bool:
    """Return True if the espeak-ng binary is reachable on PATH."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, timeout=3
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Core synthesis function
# ---------------------------------------------------------------------------

def _synth_ipa_to_pcm(
    ipa: str,
    lang: str = "en",
    sample_rate: int = 22050,
    speed: int = 150,
    pitch: int = 50,
) -> Optional[np.ndarray]:
    """
    Synthesize IPA string to a numpy float32 PCM array via eSpeak-ng.

    eSpeak-ng is invoked as a subprocess writing a WAV to stdout.
    This avoids any file I/O and keeps latency low.

    Args:
        ipa:         IPA phoneme string (e.g. 'hɛloʊ')
        lang:        eSpeak-ng language code (default 'en')
        sample_rate: Output sample rate in Hz
        speed:       Words per minute (lower = clearer for learning)
        pitch:       Pitch 0-99

    Returns:
        (samples,) float32 numpy array, or None on failure.
    """
    # eSpeak-ng accepts IPA via [[...]] notation
    ipa_input = f"[[{ipa}]]"

    cmd = [
        "espeak-ng",
        "-v", lang,
        "-s", str(speed),
        "-p", str(pitch),
        "--ipa=0",          # don't print IPA to stdout
        "--stdout",         # write WAV to stdout
        ipa_input,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=2.0,
        )
        if result.returncode != 0 or not result.stdout:
            logger.error(f"eSpeak-ng failed: {result.stderr.decode(errors='replace')}")
            return None

        # Parse WAV from stdout bytes
        with wave.open(io.BytesIO(result.stdout)) as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            native_rate = wf.getframerate()

        # Convert to float32
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample if needed (eSpeak-ng outputs at 22050 or 44100 depending on build)
        if native_rate != sample_rate:
            try:
                import librosa
                pcm = librosa.resample(pcm, orig_sr=native_rate, target_sr=sample_rate)
            except ImportError:
                # Simple linear interpolation fallback — not hi-fi but works
                ratio = sample_rate / native_rate
                new_len = int(len(pcm) * ratio)
                pcm = np.interp(
                    np.linspace(0, len(pcm) - 1, new_len),
                    np.arange(len(pcm)),
                    pcm,
                )

        return pcm

    except subprocess.TimeoutExpired:
        logger.error("eSpeak-ng timed out")
        return None
    except Exception as e:
        logger.error(f"eSpeak-ng synthesis error: {e}")
        return None


# ---------------------------------------------------------------------------
# Real-time playback engine
# ---------------------------------------------------------------------------

@dataclass
class SynthRequest:
    """A unit of work for the playback thread."""
    ipa: str
    lang: str = "en"
    speed: int = 150
    pitch: int = 50
    timestamp: float = field(default_factory=time.perf_counter)


class NEFSRealTimeSynth:
    """
    Real-time NEFS keyboard instrument synthesis backend using eSpeak-ng.

    Accepts input in three forms:
        1. Raw text       → G2P → NEFS bytes → IPA → audio
        2. NEFS bytes     → IPA → audio
        3. IPA string     → audio  (direct, lowest latency)

    Audio is played through a background worker thread so keystroke
    handling is never blocked by synthesis.

    Usage:
        synth = NEFSRealTimeSynth()
        synth.start()

        # From text (G2P assist mode — for learners)
        synth.play_text("hello")

        # From NEFS hex string (intermediate mode)
        synth.play_nefs_hex("49 93 B3 B3 5E")   # h-ɛ-l-l-oʊ

        # From NEFS bytes directly (keyboard hardware mode)
        synth.play_nefs_bytes(bytes([0x49, 0x93, 0xB3, 0xB3, 0x5E]))

        synth.stop()

    The HiFi-GAN path is preserved: swap this class for
    NEFSHiFiGANSynthesizer once a vocoder is attached.
    """

    def __init__(
        self,
        lang: str = "en",
        sample_rate: int = 22050,
        speed: int = 150,
        pitch: int = 50,
        queue_maxsize: int = 32,
    ):
        if not _espeak_available():
            raise RuntimeError(
                "espeak-ng binary not found on PATH.\n"
                "  Linux:   sudo apt install espeak-ng\n"
                "  macOS:   brew install espeak-ng\n"
                "  Windows: https://github.com/espeak-ng/espeak-ng/releases"
            )
        if not _SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice not installed. Run: pip install sounddevice"
            )

        self.lang = lang
        self.sample_rate = sample_rate
        self.speed = speed
        self.pitch = pitch

        self._queue: queue.Queue[Optional[SynthRequest]] = queue.Queue(maxsize=queue_maxsize)
        self._worker: Optional[threading.Thread] = None
        self._running = False

        # Latency tracking
        self._latencies: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background synthesis/playback worker."""
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True, name="nefs-synth")
        self._worker.start()
        logger.info("NEFSRealTimeSynth started (eSpeak-ng backend)")

    def stop(self, timeout: float = 2.0):
        """Drain the queue and stop the worker."""
        if not self._running:
            return
        self._running = False
        self._queue.put(None)  # sentinel
        if self._worker:
            self._worker.join(timeout=timeout)
        logger.info("NEFSRealTimeSynth stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Public play methods
    # ------------------------------------------------------------------

    def play_text(self, text: str):
        """
        Play text using G2P to derive the NEFS/IPA sequence.
        This is the 'sheet music' mode for learners.
        """
        if not _G2P_AVAILABLE:
            raise RuntimeError("nefs_g2p not available — cannot convert text to IPA.")
        ipa = text_to_ipa(text, lang=self.lang.replace("-", "_") if "-" in self.lang else "en-us")
        self._enqueue(ipa)

    def play_nefs_bytes(self, nefs_bytes: bytes):
        """
        Play from raw NEFS bytes — the direct keyboard instrument path.
        """
        if _NEFS_CONVERTER is None:
            raise RuntimeError("NEFSConverter not available.")
        ipa = _NEFS_CONVERTER.nafs_to_ipa(nefs_bytes)
        self._enqueue(ipa)

    def play_nefs_hex(self, hex_string: str):
        """
        Play from a hex string like '49 93 B3 B3 5E'.
        Convenience wrapper for play_nefs_bytes.
        """
        nefs_bytes = bytes.fromhex(hex_string.replace(" ", ""))
        self.play_nefs_bytes(nefs_bytes)

    def play_ipa(self, ipa: str):
        """
        Play IPA directly — lowest latency path, no conversion needed.
        """
        self._enqueue(ipa)

    def play_single_phoneme(self, nefs_byte: int):
        """
        Play a single NEFS byte value — maps one keypress to one phoneme.
        This is the core keystroke handler for the instrument.
        """
        self.play_nefs_bytes(bytes([nefs_byte]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, ipa: str):
        req = SynthRequest(
            ipa=ipa,
            lang=self.lang,
            speed=self.speed,
            pitch=self.pitch,
        )
        try:
            self._queue.put_nowait(req)
        except queue.Full:
            logger.warning("Synthesis queue full — dropping request")

    def _run(self):
        """Background worker: synthesize and play requests from the queue."""
        while self._running:
            try:
                req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is None:  # stop sentinel
                break

            pcm = _synth_ipa_to_pcm(
                req.ipa,
                lang=req.lang,
                sample_rate=self.sample_rate,
                speed=req.speed,
                pitch=req.pitch,
            )
            t1 = time.perf_counter()

            if pcm is not None and len(pcm) > 0:
                latency_ms = (t1 - req.timestamp) * 1000
                self._latencies.append(latency_ms)
                logger.debug(f"Latency: {latency_ms:.1f}ms | IPA: {req.ipa!r}")
                try:
                    sd.play(pcm, samplerate=self.sample_rate, blocking=True)
                except Exception as e:
                    logger.error(f"Playback error: {e}")

            self._queue.task_done()

    def latency_stats(self) -> dict:
        """Return latency statistics from recent requests."""
        if not self._latencies:
            return {"samples": 0}
        arr = np.array(self._latencies[-100:])  # last 100
        return {
            "samples": len(arr),
            "mean_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p95_ms": float(np.percentile(arr, 95)),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("NEFS eSpeak-ng Real-Time Synthesis — smoke test")
    print("=" * 50)

    # Check dependencies
    print(f"  espeak-ng binary:  {'✓' if _espeak_available() else '✗ NOT FOUND'}")
    print(f"  sounddevice:       {'✓' if _SOUNDDEVICE_AVAILABLE else '✗ NOT FOUND'}")
    print(f"  NEFSConverter:     {'✓' if _NEFS_CONVERTER else '✗ NOT FOUND'}")
    print(f"  nefs_g2p:          {'✓' if _G2P_AVAILABLE else '✗ NOT FOUND'}")
    print()

    if not _espeak_available() or not _SOUNDDEVICE_AVAILABLE:
        print("Install missing dependencies above, then re-run.")
        sys.exit(1)

    tests = [
        ("IPA direct",      lambda s: s.play_ipa("hɛloʊ")),
        ("Text via G2P",    lambda s: s.play_text("hello world")),
        ("NEFS hex",        lambda s: s.play_nefs_hex("49 93 B3 B3 5E")),
        ("Single phoneme",  lambda s: s.play_single_phoneme(0x49)),  # 'i'
    ]

    with NEFSRealTimeSynth(speed=130) as synth:
        for name, fn in tests:
            if name == "Text via G2P" and not _G2P_AVAILABLE:
                print(f"  SKIP  {name} (nefs_g2p unavailable)")
                continue
            if "NEFS" in name and _NEFS_CONVERTER is None:
                print(f"  SKIP  {name} (NEFSConverter unavailable)")
                continue
            print(f"  TEST  {name} ...", end=" ", flush=True)
            try:
                fn(synth)
                time.sleep(1.5)  # let audio finish
                print("✓")
            except Exception as e:
                print(f"✗  {e}")

        print()
        print("Latency stats:", synth.latency_stats())

    print("\nAll tests complete.")
