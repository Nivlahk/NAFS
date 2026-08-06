"""
nefs_polly_postprocess.py — Warp Polly audio to match NEFSPredictor targets.


After Polly synthesizes audio from an IPA string, this module:
  1. Time-stretches each phoneme segment proportionally so the internal
     rhythm matches the predictor's per-phoneme duration_s targets while
     keeping total clip length identical.
  2. Applies per-frame pitch shifting to match the predictor's f0_hz targets
     using librosa pYIN F0 extractor + librosa phase vocoder pitch shift.


Requirements:
    pip install librosa numpy


Why librosa.pyin instead of pysptk:
    pysptk requires a C compiler on Windows (no prebuilt wheel for Python
    3.11+). librosa.pyin is a pure-Python probabilistic YIN implementation
    that ships with librosa, requires no extra dependencies, and produces
    robust F0 estimates on clean TTS output like Polly's.


Usage:
    from nefs_polly_postprocess import postprocess
    audio_warped = postprocess(audio_polly, params, sample_rate=16000)
"""


from __future__ import annotations


import logging
import warnings


import numpy as np


logger = logging.getLogger(__name__)


try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed — pip install librosa. Postprocessing disabled.")


from nefs_synth_interface import SynthParams


# ---------------------------------------------------------------------------
# F0 extraction via librosa pYIN
# ---------------------------------------------------------------------------


def _extract_f0_pyin(
    audio: np.ndarray,
    sample_rate: int,
    frame_shift: int = 80,
    min_f0: float = 60.0,
    max_f0: float = 400.0,
) -> np.ndarray:
    """
    Extract F0 contour using librosa's pYIN estimator.


    pYIN (probabilistic YIN) is reliable on clean TTS output like Polly's
    and requires no compilation. Returns 0.0 for unvoiced frames.


    Args:
        audio:       (n_samples,) float32 mono audio, values in [-1, 1].
        sample_rate: Audio sample rate in Hz.
        frame_shift: Analysis hop in samples (~5ms at 16kHz = 80 samples).
        min_f0:      Minimum expected F0 in Hz.
        max_f0:      Maximum expected F0 in Hz.


    Returns:
        (n_frames,) float32 F0 array, 0.0 = unvoiced.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=min_f0,
            fmax=max_f0,
            sr=sample_rate,
            hop_length=frame_shift,
            fill_na=0.0,
        )
        # Zero out unvoiced frames explicitly using the voiced flag
        f0 = np.where(voiced_flag, f0, 0.0)
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception as e:
        logger.warning(f"pYIN F0 extraction failed: {e} — using zero F0.")
        n_frames = 1 + len(audio) // frame_shift
        return np.zeros(n_frames, dtype=np.float32)


    return f0.astype(np.float32)


# ---------------------------------------------------------------------------
# Build target F0 contour from SynthParams
# ---------------------------------------------------------------------------


def _build_target_f0(
    params: SynthParams,
    n_frames: int,
    sample_rate: int,
    frame_shift: int = 80,
) -> np.ndarray:
    """
    Build a per-frame F0 target array from SynthParams.


    Distributes each phoneme's f0_hz across frames proportional to its
    predicted duration share. Unvoiced phonemes produce 0.0 frames.
    Applies light smoothing at boundaries to avoid clicks.


    Args:
        params:      SynthParams from NEFSPredictor.
        n_frames:    Number of analysis frames.
        sample_rate: Audio sample rate.
        frame_shift: Hop size in samples (must match extraction).


    Returns:
        (n_frames,) float32 F0 target array, 0.0 = unvoiced.
    """
    from nefs_converter import is_voiced as nefs_is_voiced


    total_pred = params.total_duration_s
    if total_pred <= 0:
        return np.zeros(n_frames, dtype=np.float32)


    f0_target = np.zeros(n_frames, dtype=np.float32)
    frame_dur_s = frame_shift / sample_rate
    audio_dur_s = n_frames * frame_dur_s
    scale = audio_dur_s / total_pred


    cursor = 0.0
    for p in params.phonemes:
        voiced = nefs_is_voiced(p.nefs_byte) and p.f0_hz > 0
        start_s = cursor
        end_s   = cursor + p.duration_s
        cursor  = end_s


        if not voiced:
            continue


        frame_start = int(start_s * scale / frame_dur_s)
        frame_end   = int(end_s   * scale / frame_dur_s)
        frame_start = max(0, min(frame_start, n_frames))
        frame_end   = max(frame_start, min(frame_end, n_frames))


        if frame_end > frame_start:
            f0_target[frame_start:frame_end] = p.f0_hz


    # Light smoothing at voiced region boundaries only
    voiced_mask = f0_target > 0
    if voiced_mask.any():
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(f0_target.astype(np.float64), sigma=2.0)
            f0_target = np.where(voiced_mask, smoothed, 0.0).astype(np.float32)
        except ImportError:
            pass


    return f0_target


# ---------------------------------------------------------------------------
# Per-run pitch shifting via librosa phase vocoder
# ---------------------------------------------------------------------------


def _shift_pitch_to_target(
    audio: np.ndarray,
    f0_actual: np.ndarray,
    f0_target: np.ndarray,
    sample_rate: int,
    frame_shift: int = 80,
    n_fft: int = 256,
) -> np.ndarray:
    """
    Shift audio pitch so f0_actual approaches f0_target.


    Groups consecutive voiced frames with similar required shift into runs,
    then applies librosa.effects.pitch_shift to each run. This keeps the
    number of pitch_shift calls small (one per tonal phrase rather than one
    per frame) while still capturing the broad F0 contour.


    n_fft is set to 256 (rather than 512) to keep the analysis window narrow
    enough to avoid straddling phoneme boundaries left by the time-stretch
    phase vocoder, reducing double-vocoder smearing artifacts.


    Only touches voiced frames where both actual and target F0 are > 0.


    Args:
        audio:       (n_samples,) float32 audio.
        f0_actual:   (n_frames,) actual F0 from pYIN, 0 = unvoiced.
        f0_target:   (n_frames,) target F0 from predictor, 0 = unvoiced.
        sample_rate: Audio sample rate.
        frame_shift: Hop size matching F0 extraction.
        n_fft:       FFT window for pitch shifter (256 for speech).


    Returns:
        (n_samples,) float32 pitch-shifted audio.
    """
    n_frames  = min(len(f0_actual), len(f0_target))
    n_samples = len(audio)


    both_voiced = (f0_actual[:n_frames] > 0) & (f0_target[:n_frames] > 0)


    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            both_voiced,
            f0_target[:n_frames] / np.maximum(f0_actual[:n_frames], 1e-6),
            1.0,
        )
        semitone_shifts = np.where(
            both_voiced,
            (12.0 * np.log2(np.maximum(ratio, 1e-6))).astype(np.float32),
            0.0,
        )


    # Clamp to ±12 semitones
    semitone_shifts = np.clip(semitone_shifts, -12.0, 12.0)


    if not both_voiced.any():
        return audio


    # Group into runs where shift is similar (within 0.5st tolerance)
    runs = []
    i = 0
    while i < n_frames:
        if not both_voiced[i]:
            i += 1
            continue
        anchor = semitone_shifts[i]
        j = i + 1
        while j < n_frames and both_voiced[j] and abs(semitone_shifts[j] - anchor) < 0.5:
            anchor = float(np.mean(semitone_shifts[i:j+1]))
            j += 1
        runs.append((i, j, float(np.mean(semitone_shifts[i:j]))))
        i = j


    result = audio.copy()


    for frame_start, frame_end, n_steps in runs:
        if abs(n_steps) < 0.25:
            continue


        sample_start = frame_start * frame_shift
        sample_end   = min(frame_end * frame_shift + frame_shift, n_samples)
        segment = audio[sample_start:sample_end].astype(np.float32)


        if len(segment) < n_fft:
            continue


        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shifted = librosa.effects.pitch_shift(
                    segment,
                    sr=sample_rate,
                    n_steps=n_steps,
                    n_fft=n_fft,
                )
            if len(shifted) >= len(segment):
                result[sample_start:sample_end] = shifted[:len(segment)]
            else:
                padded = np.pad(shifted, (0, len(segment) - len(shifted)))
                result[sample_start:sample_end] = padded
        except Exception as e:
            logger.debug(f"pitch_shift failed on run [{frame_start}:{frame_end}]: {e}")


    return result


# ---------------------------------------------------------------------------
# Proportional time stretching — preserves total length
# ---------------------------------------------------------------------------


def _time_stretch_audio(
    audio: np.ndarray,
    params: SynthParams,
    sample_rate: int,
    crossfade_ms: float = 3.0,
) -> np.ndarray:
    """
    Redistribute the internal rhythm of Polly audio to match the predictor's
    per-phoneme duration proportions, while keeping total length identical.


    Divides the audio into segments proportional to predicted durations,
    time-stretches each segment to its target share, then crossfades adjacent
    segment boundaries to reduce phase-vocoder discontinuity artifacts before
    the pitch warp stage runs its own phase vocoder.


    Args:
        audio:         (n_samples,) float32 PCM.
        params:        SynthParams from NEFSPredictor.
        sample_rate:   Audio sample rate in Hz.
        crossfade_ms:  Length of linear crossfade at segment boundaries (ms).
    """
    if not _LIBROSA_AVAILABLE:
        return audio


    n_samples  = len(audio)
    total_pred = params.total_duration_s
    if total_pred <= 0 or n_samples == 0 or len(params.phonemes) == 0:
        return audio


    pred_durs  = np.array([p.duration_s for p in params.phonemes], dtype=np.float64)
    pred_share = pred_durs / pred_durs.sum()


    boundaries = np.round(
        np.concatenate([[0], np.cumsum(pred_share) * n_samples])
    ).astype(int)
    boundaries[-1] = n_samples

    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    segments = []
    for i in range(len(params.phonemes)):
        src_start = boundaries[i]
        src_end   = boundaries[i + 1]
        tgt_len   = int(src_end - src_start)
        segment   = audio[src_start:src_end].astype(np.float32)


        src_len = len(segment)
        if src_len == 0 or tgt_len == 0:
            segments.append(np.zeros(tgt_len, dtype=np.float32))
            continue


        stretch_rate = src_len / tgt_len
        stretch_rate = max(0.25, min(4.0, stretch_rate))


        if abs(stretch_rate - 1.0) < 0.02:
            if src_len >= tgt_len:
                seg = segment[:tgt_len]
            else:
                seg = np.pad(segment, (0, tgt_len - src_len))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stretched = librosa.effects.time_stretch(segment, rate=stretch_rate)
            if len(stretched) >= tgt_len:
                seg = stretched[:tgt_len].astype(np.float32)
            else:
                seg = np.pad(stretched, (0, tgt_len - len(stretched))).astype(np.float32)

        # Crossfade boundary with previous segment to smooth phase discontinuities
        if (
            crossfade_samples > 0
            and segments
            and len(segments[-1]) >= crossfade_samples
            and len(seg) >= crossfade_samples
        ):
            fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
            fade_in  = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
            segments[-1] = segments[-1].copy()
            segments[-1][-crossfade_samples:] *= fade_out
            seg = seg.copy()
            seg[:crossfade_samples] *= fade_in

        segments.append(seg)


    result = np.concatenate(segments)
    if len(result) > n_samples:
        result = result[:n_samples]
    elif len(result) < n_samples:
        result = np.pad(result, (0, n_samples - len(result)))


    logger.debug(
        f"Rhythmic warp: {len(params.phonemes)} segments, "
        f"total {len(result)/sample_rate:.2f}s (original {n_samples/sample_rate:.2f}s)"
    )
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def postprocess(
    audio: np.ndarray,
    params: SynthParams,
    sample_rate: int,
    frame_shift: int = 80,
    apply_pitch: bool = True,
    apply_duration: bool = True,
) -> np.ndarray:
    """
    Warp Polly audio to match NEFSPredictor F0 and duration targets.


    Args:
        audio:          (n_samples,) float32 PCM from Polly.
        params:         SynthParams from NEFSPredictor (same utterance).
        sample_rate:    Audio sample rate (must match Polly output).
        frame_shift:    pYIN analysis hop in samples (80 = ~5ms at 16kHz).
        apply_pitch:    Apply F0 warping (requires librosa).
        apply_duration: Apply rhythmic time stretch (requires librosa).


    Returns:
        (n_samples,) float32 PCM with predictor prosody applied.
    """
    if not params.phonemes:
        return audio


    result = audio.astype(np.float32).copy()


    # Step 1: Rhythmic time stretch with crossfade at segment boundaries
    if apply_duration and _LIBROSA_AVAILABLE:
        result = _time_stretch_audio(result, params, sample_rate)
        logger.info(
            f"After rhythmic warp: {len(result)/sample_rate:.2f}s "
            f"(original {len(audio)/sample_rate:.2f}s)"
        )
    elif apply_duration:
        logger.warning("librosa not available — skipping time stretch.")


    # Step 2: F0 warping via pYIN + librosa pitch shift
    # Runs on post-stretch audio so F0 is measured on the redistributed signal.
    # n_fft=256 keeps the analysis window narrow to avoid straddling the
    # phase-vocoder boundaries introduced in Step 1.
    if apply_pitch and _LIBROSA_AVAILABLE:
        f0_actual = _extract_f0_pyin(result, sample_rate, frame_shift=frame_shift)
        f0_target = _build_target_f0(params, len(f0_actual), sample_rate, frame_shift)


        logger.info(
            f"F0 warp: {(f0_actual>0).sum()} voiced frames detected, "
            f"{(f0_target>0).sum()} voiced frames targeted"
        )


        if (f0_actual > 0).any() and (f0_target > 0).any():
            result = _shift_pitch_to_target(
                result, f0_actual, f0_target, sample_rate, frame_shift=frame_shift
            )
            peak = np.abs(result).max()
            if peak > 0.95:
                result = result / peak * 0.95
            logger.info("F0 warp applied.")
        else:
            logger.info("F0 warp skipped — no voiced frames in audio or targets.")


    elif apply_pitch:
        logger.warning("librosa not available — skipping F0 warping.")


    return result