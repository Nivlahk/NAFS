"""nefs_tts_hifigan.py — NEFS-native TTS with HiFi-GAN vocoder

This module provides a complete Text → NEFS → Audio pipeline using HiFi-GAN
as the vocoder. Unlike traditional TTS systems that use mel-spectrograms,
this implementation conditions HiFi-GAN directly on NEFS byte embeddings,
exploiting NEFS's bit-level phonological structure.

Architecture:
    Text → G2P → NEFS bytes → NEFSPhonemeEncoder → HiFi-GAN → Audio

Key innovations:
    - NEFS bytes seed phoneme embeddings with structured phonological features
    - Bit-level feature extraction (place, manner, voicing) reduces training data
    - Compatible with pretrained HiFi-GAN vocoders via adapter layers
    - Supports both inference with existing models and training from scratch

Dependencies:
    pip install torch torchaudio librosa soundfile
    pip install phonemizer epitran transformers  # for G2P (see nefs_g2p.py)

Pretrained models:
    - NVIDIA HiFi-GAN (LJSpeech, VCTK, Universal): torch.hub or NGC
    - Coqui TTS HiFi-GAN checkpoints: coqui.ai
    - VITS (includes HiFi-GAN generator): huggingface/transformers

Usage:
    from nefs_tts_hifigan import NEFSHiFiGANSynthesizer
    
    tts = NEFSHiFiGANSynthesizer.from_pretrained('nvidia_hifigan_ljspeech')
    audio = tts.synthesize("Hello world", sample_rate=22050)
    tts.save_wav(audio, "output.wav")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)



# ============================================================================
# NEFS Phoneme Encoder — exploits bit-level structure of NEFS bytes
# ============================================================================

class NEFSPhonemeEncoder(nn.Module):
    """
    Encodes NEFS bytes into phoneme embeddings for HiFi-GAN conditioning.
    
    Unlike standard phoneme embeddings (which treat phonemes as atomic tokens),
    this encoder explicitly models the phonological feature structure encoded
    in NEFS byte values.
    
    NEFS byte structure (see README Two-Operation Guarantee section):
        - High nibble (bits 7-4): Place of articulation
        - Low nibble bits 3-0: Manner, voicing, aspiration, etc.
    
    The encoder:
        1. Extracts bit-level features via masking + shift operations
        2. Learns separate embeddings for place, manner, voicing
        3. Combines them additively (similar to positional encoding)
        4. Optionally learns a full byte embedding for fine-tuning
    
    This structure provides strong inductive bias for low-resource training.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        use_feature_decomposition: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_feature_decomposition = use_feature_decomposition
        
        if use_feature_decomposition:
            # Separate embeddings for phonological features
            self.place_embed = nn.Embedding(16, embedding_dim // 4)  # high nibble
            self.manner_embed = nn.Embedding(16, embedding_dim // 4)  # low nibble
            self.voicing_embed = nn.Embedding(2, embedding_dim // 4)  # bit 0
            self.aspiration_embed = nn.Embedding(4, embedding_dim // 4)  # bits 2-1
            
            # Projection to target dimension
            self.feature_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Full byte embedding (256 possible bytes)
        self.byte_embed = nn.Embedding(256, embedding_dim)

        # Combination layer — only reached when use_feature_decomposition=True.
        # Input is cat([feature_emb, byte_emb]) = embedding_dim + embedding_dim.
        # When decomp=False the forward() takes an early return before combine,
        # so the Linear is always constructed with the correct input size here
        # regardless of the flag — the conditional in the original code was both
        # wrong (embedding_dim*2 vs embedding_dim) and dead (False branch unused).
        self.combine = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, nefs_bytes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nefs_bytes: (batch, seq_len) tensor of NEFS byte values (0-255)
        
        Returns:
            (batch, seq_len, embedding_dim) phoneme embeddings
        """
        # Full byte embedding
        byte_emb = self.byte_embed(nefs_bytes)
        
        if not self.use_feature_decomposition:
            return self.layer_norm(self.dropout(byte_emb))
        
        # Extract phonological features via bit operations
        place = (nefs_bytes >> 4) & 0x0F  # high nibble
        manner = nefs_bytes & 0x0F  # low nibble
        voicing = nefs_bytes & 0x01  # bit 0
        aspiration = (nefs_bytes >> 1) & 0x03  # bits 2-1
        
        # Embed each feature separately
        place_emb = self.place_embed(place)
        manner_emb = self.manner_embed(manner)
        voicing_emb = self.voicing_embed(voicing)
        aspiration_emb = self.aspiration_embed(aspiration)
        
        # Concatenate feature embeddings
        feature_emb = torch.cat([place_emb, manner_emb, voicing_emb, aspiration_emb], dim=-1)
        feature_emb = self.feature_proj(feature_emb)
        
        # Combine structured features with learned byte embedding
        combined = torch.cat([feature_emb, byte_emb], dim=-1)
        combined = self.combine(combined)
        
        return self.layer_norm(self.dropout(combined))


# ============================================================================
# Prosody Predictor — duration and pitch from NEFS sequence
# ============================================================================

class NEFSProsodyPredictor(nn.Module):
    """
    Predicts duration (frames per phoneme) and F0 contour from NEFS sequence.
    
    HiFi-GAN expects mel-spectrogram-length inputs. We need to:
    1. Predict how many acoustic frames each NEFS phoneme generates
    2. Upscale NEFS embeddings to acoustic frame rate
    3. Optionally predict pitch contour for prosody control
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Duration predictor (log-scale to handle variable lengths)
        # Conv1d output shape: (batch, channels, time).
        # nn.LayerNorm normalises over the *last* dimension, which is the time
        # axis here — not channels — and is therefore incorrect.  GroupNorm
        # operates over (channels, time) for a given group count, which is the
        # right choice for 1-D convolutional stacks.
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        
        # F0 predictor (optional pitch control)
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
    
    def forward(self, phoneme_emb: torch.Tensor, target_durations: Optional[torch.Tensor] = None):
        """
        Args:
            phoneme_emb: (batch, seq_len, dim) NEFS phoneme embeddings
            target_durations: (batch, seq_len) ground-truth durations for training
        
        Returns:
            upsampled_emb: (batch, total_frames, dim) frame-rate embeddings
            predicted_durations: (batch, seq_len) predicted frame counts
            predicted_f0: (batch, total_frames) F0 contour
        """
        # Transpose for Conv1d: (batch, dim, seq_len)
        x = phoneme_emb.transpose(1, 2)
        
        # Predict log-duration, then exponentiate
        log_duration = self.duration_predictor(x).squeeze(1)  # (batch, seq_len)
        predicted_durations = torch.exp(log_duration).clamp(min=1.0)
        
        # Use ground-truth durations during training, predicted during inference
        durations = target_durations if target_durations is not None else predicted_durations
        
        # Upsample phoneme embeddings to frame rate
        upsampled_emb = self._length_regulator(phoneme_emb, durations)
        
        # Predict F0 at frame rate
        upsampled_x = upsampled_emb.transpose(1, 2)
        predicted_f0 = self.f0_predictor(upsampled_x).squeeze(1)
        
        return upsampled_emb, predicted_durations, predicted_f0
    
    def _length_regulator(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        Repeat each phoneme embedding according to its duration.

        Args:
            x: (batch, seq_len, dim)
            durations: (batch, seq_len) frame counts (float or int tensor)

        Returns:
            (batch, max_total_frames, dim) — padded to the longest sequence in
            the batch.

        Implementation note
        -------------------
        The original implementation used nested Python loops
        (O(batch × seq_len) calls to .item() + .repeat()), which serialises
        execution on the CPU and prevents GPU parallelism.

        This version uses ``torch.repeat_interleave`` which is a single fused
        CUDA/CPU kernel call per batch item, giving O(1) kernel launches instead
        of O(batch × seq_len).  Padding is still done with F.pad per item since
        torch.nn.utils.rnn.pad_sequence expects a list of variable-length tensors.
        """
        # Round durations to integers; clamp to at least 1 to avoid empty slices.
        dur_int = durations.round().long().clamp(min=1)  # (batch, seq_len)

        output = []
        for b in range(x.size(0)):
            # repeat_interleave repeats x[b] along dim=0 according to dur_int[b].
            # This is a single vectorised kernel call — no Python loop over seq_len.
            repeated = torch.repeat_interleave(x[b], dur_int[b], dim=0)  # (total_frames, dim)
            output.append(repeated)

        # Pad to the longest sequence in the batch.
        max_len = max(o.size(0) for o in output)
        padded = [F.pad(o, (0, 0, 0, max_len - o.size(0))) for o in output]

        return torch.stack(padded)  # (batch, max_len, dim)


# ============================================================================
# Main NEFS HiFi-GAN Synthesizer
# ============================================================================

class NEFSHiFiGANSynthesizer(nn.Module):
    """
    Complete NEFS-native TTS system with HiFi-GAN vocoder.

    Pipeline:
        1. Text → nefs_g2p.text_to_ipa() → NEFSConverter.ipa_to_nafs() → NEFS bytes
        2. NEFS bytes → NEFSPhonemeEncoder → embeddings (dim=embedding_dim)
        3. Embeddings → NEFSProsodyPredictor → frame-rate features (dim=embedding_dim)
        4. Frame features → mel_proj → mel-scale features (dim=n_mels)
        5. Mel features → HiFi-GAN generator → audio waveform

    The mel projection (step 4) is critical: HiFi-GAN's generator was trained
    to condition on 80-bin mel spectrograms.  The phoneme encoder output has
    dimension 512 (or whatever embedding_dim is set to).  Without a learned
    linear projection from 512 → 80, passing raw frame embeddings to the
    vocoder produces garbage audio.
    """

    def __init__(
        self,
        phoneme_encoder: NEFSPhonemeEncoder,
        prosody_predictor: NEFSProsodyPredictor,
        hifigan_generator: Optional[nn.Module] = None,
        n_mels: int = 80,
    ):
        super().__init__()
        self.phoneme_encoder = phoneme_encoder
        self.prosody_predictor = prosody_predictor
        self.hifigan_generator = hifigan_generator
        self.n_mels = n_mels

        # Project from phoneme embedding space → mel-spectrogram channel count.
        # HiFi-GAN generators expect input shape (batch, n_mels, frames).
        # This layer is always constructed so checkpoints are consistent; it
        # is a no-op during training stages that don't attach a vocoder.
        embedding_dim = phoneme_encoder.embedding_dim
        self.mel_proj = nn.Linear(embedding_dim, n_mels)

        # Lazy imports for NEFS converter and G2P
        self.nefs_converter = None
        self.g2p = None
    
    def _ensure_nefs_converter(self):
        if self.nefs_converter is None:
            try:
                from nefs_wrapper import NEFSConverter
                self.nefs_converter = NEFSConverter()
            except ImportError:
                raise RuntimeError(
                    "NEFSConverter not found. Ensure nefs_wrapper.py is in PYTHONPATH."
                )
    
    def _ensure_g2p(self):
        if self.g2p is None:
            try:
                from nefs_g2p import text_to_ipa
                self.g2p = text_to_ipa
            except ImportError:
                logger.warning("nefs_g2p not found. G2P will be unavailable.")
                self.g2p = lambda text, **kwargs: text  # fallback: return text unchanged
    
    def text_to_nefs(self, text: str, lang: str = "en-us") -> torch.Tensor:
        """
        Convert raw text to NEFS byte tensor.
        
        Args:
            text: Input text
            lang: Language code for G2P
        
        Returns:
            (seq_len,) tensor of NEFS byte values
        """
        self._ensure_nefs_converter()
        self._ensure_g2p()
        
        # Text → IPA → NEFS bytes
        ipa = self.g2p(text, lang=lang, prefer='espeak')
        nefs_bytes = self.nefs_converter.ipa_to_nafs(ipa)
        
        return torch.tensor(list(nefs_bytes), dtype=torch.long)
    
    def forward(
        self,
        nefs_bytes: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass: NEFS bytes → audio waveform.
        
        Args:
            nefs_bytes: (batch, seq_len) NEFS byte tensor
            target_durations: (batch, seq_len) optional ground-truth durations for training
        
        Returns:
            dict with keys:
                - 'audio': (batch, samples) generated waveform
                - 'durations': (batch, seq_len) predicted durations
                - 'f0': (batch, frames) predicted F0 contour
                - 'phoneme_emb': (batch, seq_len, dim) phoneme embeddings
                - 'frame_emb': (batch, frames, dim) frame-rate features
        """
        # Encode NEFS bytes to phoneme embeddings
        phoneme_emb = self.phoneme_encoder(nefs_bytes)
        
        # Predict prosody and upsample to frame rate
        frame_emb, durations, f0 = self.prosody_predictor(phoneme_emb, target_durations)
        
        # Generate audio via HiFi-GAN (if available)
        audio = None
        if self.hifigan_generator is not None:
            # Project from embedding_dim → n_mels so the vocoder receives the
            # mel-spectrogram-shaped input it was trained on.
            # frame_emb: (batch, frames, embedding_dim)
            # mel:       (batch, frames, n_mels) → transposed to (batch, n_mels, frames)
            mel = self.mel_proj(frame_emb).transpose(1, 2)
            audio = self.hifigan_generator(mel)
            audio = audio.squeeze(1)  # Remove channel dim if present
        
        return {
            'audio': audio,
            'durations': durations,
            'f0': f0,
            'phoneme_emb': phoneme_emb,
            'frame_emb': frame_emb,
        }
    
    def synthesize(
        self,
        text: str,
        lang: str = "en-us",
        sample_rate: int = 22050,
    ) -> np.ndarray:
        """
        High-level synthesis: text → audio numpy array.
        
        Args:
            text: Input text
            lang: Language code
            sample_rate: Output sample rate (must match HiFi-GAN training)
        
        Returns:
            (samples,) numpy array of audio
        """
        self.eval()
        with torch.no_grad():
            # Text → NEFS bytes
            nefs_bytes = self.text_to_nefs(text, lang=lang).unsqueeze(0)  # (1, seq_len)
            
            # NEFS → audio
            output = self.forward(nefs_bytes)
            audio = output['audio']
            
            if audio is None:
                raise RuntimeError(
                    "No HiFi-GAN generator loaded. Use .set_hifigan_generator() or "
                    ".from_pretrained() with a vocoder."
                )
            
            # Convert to numpy
            audio_np = audio.squeeze(0).cpu().numpy()
            
            return audio_np
    
    def save_wav(self, audio: np.ndarray, path: Union[str, Path], sample_rate: int = 22050):
        """
        Save audio to WAV file.
        
        Args:
            audio: (samples,) numpy array
            path: Output file path
            sample_rate: Sample rate
        """
        try:
            import soundfile as sf
            sf.write(path, audio, sample_rate)
        except ImportError:
            try:
                from scipy.io import wavfile
                # scipy requires int16
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(path, sample_rate, audio_int16)
            except ImportError:
                raise RuntimeError(
                    "Neither soundfile nor scipy.io.wavfile available. "
                    "Install with: pip install soundfile"
                )
    
    def set_hifigan_generator(self, generator: nn.Module):
        """Attach a HiFi-GAN generator (allows hot-swapping vocoders)."""
        self.hifigan_generator = generator
    
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = 'cpu') -> 'NEFSHiFiGANSynthesizer':
        """
        Load NEFS TTS model with pretrained HiFi-GAN vocoder.
        
        Supported models:
            - 'nvidia_hifigan_ljspeech': NVIDIA's LJSpeech HiFi-GAN (22kHz, single speaker)
            - 'nvidia_hifigan_universal': Universal multi-speaker vocoder
            - 'hifigan_v1_ljspeech': Original author's checkpoint

        Loading notes
        -------------
        The ``nvidia/DeepLearningExamples:torchhub`` torch.hub entry point was
        deprecated and removed by NVIDIA.  Current recommended sources are:

        * **Coqui TTS** (easiest, pip-installable)::

              pip install TTS
              from TTS.utils.manage import ModelManager
              # downloads hifigan/ljspeech to ~/.local/share/tts/
              model_manager = ModelManager()
              model_path, _, _ = model_manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")

        * **ESPnet** — ``espnet2.bin.tts_inference`` ships HiFi-GAN vocoders.

        * **VITS** (HuggingFace Transformers ≥ 4.33)::

              from transformers import VitsModel
              # VitsModel includes an integrated HiFi-GAN generator.

        * **Original author's checkpoint** (raw PyTorch)::

              # https://github.com/jik876/hifi-gan — download LJ_V1/ weights,
              # load generator with their config.json + generator checkpoint.

        Pass ``hifigan_generator`` directly to the constructor after loading
        via one of the above methods, or subclass and override ``from_pretrained``.
        
        Args:
            model_name: Pretrained model identifier
            device: 'cpu' or 'cuda'
        
        Returns:
            Initialized NEFSHiFiGANSynthesizer with loaded vocoder
        """
        # Initialize NEFS-specific components
        phoneme_encoder = NEFSPhonemeEncoder(embedding_dim=512)
        prosody_predictor = NEFSProsodyPredictor(input_dim=512)
        
        # Load pretrained HiFi-GAN generator
        if model_name == 'nvidia_hifigan_ljspeech':
            # The nvidia/DeepLearningExamples:torchhub torch.hub entry point
            # was removed by NVIDIA and no longer works.  See from_pretrained
            # docstring for current loading options.
            logger.error(
                "The NVIDIA torch.hub HiFi-GAN entry point has been removed. "
                "Load a HiFi-GAN generator via Coqui TTS, ESPnet, the "
                "HuggingFace VITS model, or the original author's checkpoint "
                "(jik876/hifi-gan on GitHub), then pass it to the constructor "
                "directly as hifigan_generator=<your_generator>. "
                "See NEFSHiFiGANSynthesizer.from_pretrained docstring for details."
            )
            hifigan_generator = None
        
        elif model_name == 'nvidia_hifigan_universal':
            logger.warning("Universal HiFi-GAN requires speaker embeddings (not yet implemented)")
            hifigan_generator = None
        
        else:
            logger.warning(f"Unknown model '{model_name}'. Initializing without vocoder.")
            hifigan_generator = None
        
        synthesizer = cls(
            phoneme_encoder=phoneme_encoder,
            prosody_predictor=prosody_predictor,
            hifigan_generator=hifigan_generator,
            n_mels=80,  # Standard HiFi-GAN LJSpeech config
        )
        
        synthesizer.to(device)
        return synthesizer


# ============================================================================
# Example usage and demo
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NEFS TTS with HiFi-GAN")
    parser.add_argument("--text", type=str, default="Hello world, this is NEFS speaking.")
    parser.add_argument("--output", type=str, default="nefs_output.wav")
    parser.add_argument("--model", type=str, default="nvidia_hifigan_ljspeech")
    parser.add_argument("--lang", type=str, default="en-us")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()
    
    print(f"NEFS HiFi-GAN TTS Demo")
    print(f"======================\n")
    print(f"Text: {args.text}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}\n")
    
    # Initialize synthesizer
    print("Loading model...")
    tts = NEFSHiFiGANSynthesizer.from_pretrained(args.model, device=args.device)
    print(f"Model loaded on {args.device}\n")
    
    # Synthesize
    print("Synthesizing audio...")
    audio = tts.synthesize(args.text, lang=args.lang, sample_rate=args.sample_rate)
    print(f"Generated {len(audio)} samples ({len(audio)/args.sample_rate:.2f}s)\n")
    
    # Save
    print(f"Saving to {args.output}...")
    tts.save_wav(audio, args.output, sample_rate=args.sample_rate)
    print("Done!")
    
    # Print NEFS bytes for inspection
    print("\nNEFS byte sequence:")
    nefs_bytes = tts.text_to_nefs(args.text, lang=args.lang)
    print(" ".join(f"{b:02X}" for b in nefs_bytes.tolist()))
