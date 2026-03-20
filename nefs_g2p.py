"""
nefs_g2p.py — Tiered Grapheme-to-Phoneme (G2P) pipeline for NEFS

Provides IPA output from raw text across ~100+ languages via a three-tier
fallback chain:

  Tier 1 (best accuracy):  CharsiuG2P  — ByT5-based neural model, ~100 languages,
                                          outputs IPA natively, zero-shot on unseen
                                          languages.  Requires: transformers, torch
  Tier 2 (fast / offline): espeak-ng   — rule-based, 127 languages/accents, no GPU.
                                          Requires: phonemizer + espeak-ng binary
  Tier 3 (morphology):     epitran     — rule-based with linguistic expertise, ~92
                                          languages, best for South-Asian / African
                                          scripts.  Requires: epitran

Install dependencies (all optional — only the tiers you install will be used):

    pip install phonemizer          # espeak-ng Python wrapper (Tier 2)
    pip install epitran             # Tier 3
    pip install transformers torch  # CharsiuG2P (Tier 1)

espeak-ng binary (Tier 2) must also be on PATH:
    Windows:  https://github.com/espeak-ng/espeak-ng/releases
    Linux:    sudo apt install espeak-ng
    macOS:    brew install espeak-ng
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1 — CharsiuG2P (ByT5 neural, ~100 languages, outputs IPA directly)
# ---------------------------------------------------------------------------

_charsiu_model = None
_charsiu_tokenizer = None
_charsiu_loaded = False


def _load_charsiu():
    """Lazy-load CharsiuG2P model weights on first use.

    Model notes
    -----------
    ``charsiu/g2p_multilingual_byT5_small_100`` is the canonical public
    checkpoint (ByT5-small backbone, ~100 languages). If a newer checkpoint
    becomes available under the ``charsiu`` HuggingFace org, set the env var
    ``CHARSIU_MODEL`` to override.  The tokeniser must always be loaded from
    ``google/byt5-small`` because ByT5 uses raw UTF-8 bytes as tokens and
    ships no added vocabulary.

    ``transformers`` compatibility
    ------------------------------
    ``early_stopping=True`` without an explicit beam count was silently
    accepted before transformers 4.38 but now emits DeprecationWarning and
    will become an error.  We use ``max_new_tokens`` (preferred over
    ``max_length``) and omit ``early_stopping`` entirely — the default
    stopping behaviour is correct for greedy / beam search.
    """
    global _charsiu_model, _charsiu_tokenizer, _charsiu_loaded
    if _charsiu_loaded:
        return _charsiu_model is not None
    _charsiu_loaded = True
    try:
        import os
        from transformers import T5ForConditionalGeneration, AutoTokenizer  # type: ignore
        model_name = os.environ.get(
            "CHARSIU_MODEL", "charsiu/g2p_multilingual_byT5_small_100"
        )
        logger.info(
            f"Loading CharsiuG2P model '{model_name}' "
            "(first use — downloading if needed)..."
        )
        # Tokeniser must come from the ByT5 base, not the fine-tuned checkpoint
        _charsiu_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        _charsiu_model = T5ForConditionalGeneration.from_pretrained(model_name)
        _charsiu_model.eval()
        logger.info("CharsiuG2P loaded successfully.")
        return True
    except Exception as exc:
        logger.warning(f"CharsiuG2P unavailable: {exc}")
        return False


def _g2p_charsiu(text: str, lang: str = "en-us") -> Optional[str]:
    """
    Convert text to IPA using CharsiuG2P.

    Parameters
    ----------
    text : str
        Input text (orthographic).
    lang : str
        BCP-47-style language code, e.g. 'en-us', 'fr', 'zh', 'ar'.
        CharsiuG2P uses a '<lang>: <text>' prompt format internally.

    Returns
    -------
    str or None
        IPA string, or None if CharsiuG2P is not available.
    """
    if not _load_charsiu():
        return None
    try:
        import torch  # type: ignore

        # CharsiuG2P expects the language tag prepended
        prompt = f"{lang}: {text}"
        # Do NOT pass padding=True for a single string.  ByT5 has no dedicated
        # pad token and falls back to using eos_token as pad, which corrupts the
        # attention mask and degrades output quality.  Padding is only needed
        # when batching multiple sequences together.
        inputs = _charsiu_tokenizer(
            prompt,
            return_tensors="pt",
        )
        # Move inputs to the same device as the model to avoid device mismatch
        # errors when the model has been loaded onto CUDA.
        model_device = next(_charsiu_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            # Use max_new_tokens (preferred over max_length in transformers ≥4.20).
            # early_stopping=True without explicit beam constraints was deprecated
            # in transformers 4.38 and will become an error — omit it entirely.
            outputs = _charsiu_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=4,
            )
        ipa = _charsiu_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ipa.strip()
    except Exception as exc:
        logger.warning(f"CharsiuG2P inference failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Tier 2 — espeak-ng via phonemizer (rule-based, 127 languages, no GPU)
# ---------------------------------------------------------------------------

def _check_espeak_binary() -> bool:
    """Return True if a compatible espeak-ng binary is on PATH.

    Compatibility notes
    -------------------
    * Linux/Windows: the binary is called ``espeak-ng``.
    * macOS via Homebrew: ``brew install espeak-ng`` installs ``espeak-ng``.
      The older ``brew install espeak`` formula installs the *original* eSpeak
      (not eSpeak-NG) as ``espeak``, which phonemizer's ``"espeak"`` backend
      will locate by name but may produce subtly different IPA output and lacks
      many languages supported by eSpeak-NG.  We check for ``espeak-ng`` first
      and fall back to ``espeak`` with a warning so callers know which binary
      they got.
    """
    import shutil
    if shutil.which("espeak-ng"):
        return True
    if shutil.which("espeak"):
        logger.warning(
            "Found 'espeak' binary but not 'espeak-ng'. "
            "The original eSpeak (not eSpeak-NG) is installed. "
            "phonemizer will use it, but language coverage and IPA accuracy "
            "may differ from eSpeak-NG. "
            "To install eSpeak-NG on macOS: brew install espeak-ng. "
            "On Linux: sudo apt install espeak-ng."
        )
        return True
    return False


def _g2p_espeak(text: str, lang: str = "en-us") -> Optional[str]:
    """
    Convert text to IPA using espeak-ng through the phonemizer library.

    Parameters
    ----------
    text : str
        Input text.
    lang : str
        espeak-ng language code, e.g. 'en-us', 'fr', 'de', 'zh', 'ar'.
        Run `espeak-ng --voices` for a full list.

    Returns
    -------
    str or None
        IPA string, or None if phonemizer / espeak-ng is not installed.
    """
    if not _check_espeak_binary():
        logger.warning(
            "espeak-ng (or espeak) binary not found on PATH. "
            "Install with: sudo apt install espeak-ng  (Linux) | "
            "brew install espeak-ng  (macOS) | "
            "https://github.com/espeak-ng/espeak-ng/releases  (Windows)"
        )
        return None
    try:
        from phonemizer import phonemize  # type: ignore
        from phonemizer.separator import Separator  # type: ignore

        sep = Separator(phone=" ", word=" | ", syllable="")
        ipa = phonemize(
            text,
            backend="espeak",
            language=lang,
            separator=sep,
            with_stress=True,
            njobs=1,
        )
        # Collapse separator artefacts to a clean IPA string
        ipa = ipa.replace(" | ", " ").strip()
        return ipa
    except Exception as exc:
        logger.warning(f"espeak-ng G2P failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Tier 3 — epitran (rule-based, ~92 languages, strong on low-resource scripts)
# ---------------------------------------------------------------------------

# Map common BCP-47 codes -> epitran lang-script codes where they differ
_EPITRAN_LANG_MAP: Dict[str, str] = {
    "en": "eng-Latn",
    "en-us": "eng-Latn",
    "en-gb": "eng-Latn",
    "fr": "fra-Latn",
    "de": "deu-Latn",
    "es": "spa-Latn",
    "ar": "ara-Arab",
    "hi": "hin-Deva",
    "zh": "cmn-Hans",
    "ru": "rus-Cyrl",
    "tr": "tur-Latn",
    "sw": "swa-Latn",
    "vi": "vie-Latn",
    "ja": "jpn-Hira",
    "ko": "kor-Hang",
    "fa": "fas-Arab",
    "ur": "urd-Arab",
    "bn": "ben-Beng",
    "ta": "tam-Taml",
    "te": "tel-Telu",
}


def _g2p_epitran(text: str, lang: str = "en-us") -> Optional[str]:
    """
    Convert text to IPA using epitran.

    Parameters
    ----------
    text : str
        Input text.
    lang : str
        BCP-47 language code.  Will be mapped to epitran lang-script code
        automatically for common languages; pass the epitran code directly
        (e.g. 'eng-Latn') to bypass the map.

    Returns
    -------
    str or None
        IPA string, or None if epitran is not installed or the language
        is unsupported.
    """
    try:
        import epitran  # type: ignore

        epitran_code = _EPITRAN_LANG_MAP.get(lang.lower(), lang)
        epi = epitran.Epitran(epitran_code)
        ipa = epi.transliterate(text)
        return ipa.strip()
    except Exception as exc:
        logger.warning(f"epitran G2P failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API — tiered fallback chain
# ---------------------------------------------------------------------------

def text_to_ipa(
    text: str,
    lang: str = "en-us",
    *,
    prefer: str = "charsiu",
    fallback: bool = True,
) -> str:
    """
    Convert orthographic text to IPA using the best available G2P backend.

    Tries backends in priority order and falls back automatically if a
    backend is not installed or fails at runtime.

    Parameters
    ----------
    text : str
        Input text in the native orthography of `lang`.
    lang : str
        BCP-47 language code (e.g. 'en-us', 'fr', 'de', 'zh', 'ar').
        Passed through to whichever backend is active.
    prefer : {'charsiu', 'espeak', 'epitran'}
        Which backend to try first.  Defaults to 'charsiu' (highest accuracy).
        Set to 'espeak' for offline / no-GPU environments.
    fallback : bool
        If True (default), automatically fall back through remaining backends
        when the preferred one is unavailable.  If False, raises RuntimeError
        when the preferred backend is not available.

    Returns
    -------
    str
        IPA transcription.  Returns the original text unchanged if no backend
        is available and `fallback` is True (matches the original stub behaviour
        in nefs_wrapper.py so existing code cannot break).

    Examples
    --------
    >>> from nefs_g2p import text_to_ipa
    >>> text_to_ipa("hello world", lang="en-us")
    'həˈloʊ ˈwɜːld'
    >>> text_to_ipa("bonjour", lang="fr", prefer="espeak")
    'bɔ̃ʒuʁ'
    >>> text_to_ipa("مرحبا", lang="ar")
    'marħaban'
    """
    backends = {
        "charsiu": _g2p_charsiu,
        "espeak": _g2p_espeak,
        "epitran": _g2p_epitran,
    }

    if prefer not in backends:
        raise ValueError(f"prefer must be one of {list(backends)}, got {prefer!r}")

    # Build the ordered list: preferred first, then the other two
    order = [prefer] + [k for k in ["charsiu", "espeak", "epitran"] if k != prefer]

    for backend_name in order:
        result = backends[backend_name](text, lang)
        if result is not None:
            logger.debug(f"G2P backend used: {backend_name}")
            return result
        if not fallback:
            # Only the preferred backend was attempted when fallback=False;
            # raising on any backend in the list would give a misleading error
            # message if a non-preferred backend returned None first.
            if backend_name == prefer:
                raise RuntimeError(
                    f"G2P backend '{prefer}' is not available or failed. "
                    "Install the required package(s) listed in nefs_g2p.py."
                )
            # Non-preferred backends should not be reached when fallback=False.
            break

    # All backends failed — return text unchanged so downstream code doesn't break
    warnings.warn(
        f"No G2P backend available for lang='{lang}'. "
        "Returning raw text unchanged.  "
        "Install phonemizer, epitran, or transformers+torch to enable G2P.",
        RuntimeWarning,
        stacklevel=2,
    )
    return text


def available_backends() -> Dict[str, bool]:
    """
    Return a dict showing which G2P backends are currently importable.

    Returns
    -------
    dict
        e.g. {'charsiu': True, 'espeak': False, 'epitran': True}
    """
    result: Dict[str, bool] = {}

    # charsiu
    try:
        import transformers  # noqa: F401  # type: ignore
        import torch  # noqa: F401  # type: ignore
        result["charsiu"] = True
    except ImportError:
        result["charsiu"] = False

    # espeak — requires both the phonemizer package AND the binary on PATH
    try:
        import phonemizer  # noqa: F401  # type: ignore
        # Package present — also verify the binary is reachable, otherwise
        # phonemizer will import fine but fail at runtime.
        result["espeak"] = _check_espeak_binary()
    except ImportError:
        result["espeak"] = False

    # epitran
    try:
        import epitran  # noqa: F401  # type: ignore
        result["epitran"] = True
    except ImportError:
        result["epitran"] = False

    return result


if __name__ == "__main__":
    import sys

    print("NEFS G2P — available backends:", available_backends())

    test_cases = [
        ("hello world", "en-us"),
        ("bonjour le monde", "fr"),
        ("guten Tag", "de"),
        ("hola mundo", "es"),
    ]

    prefer_backend = sys.argv[1] if len(sys.argv) > 1 else "charsiu"
    print(f"Using prefer='{prefer_backend}'\n")

    for text, lang in test_cases:
        ipa = text_to_ipa(text, lang=lang, prefer=prefer_backend)
        print(f"  [{lang}] {text!r:30s} -> {ipa}")
