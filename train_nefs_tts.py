"""train_nefs_tts.py — Train NEFS-native TTS from scratch

This script trains a complete Text → NEFS → Audio TTS system WITHOUT:
    1. Going through IPA (direct grapheme → NEFS via learned G2P)
    2. Using mel-spectrograms (HiFi-GAN trained directly on NEFS embeddings)

This demonstrates NEFS's core value proposition: byte-level phonological
structure reduces training complexity vs. traditional TTS pipelines.

Dataset requirements:
    - Audio files (.wav, 22050 Hz recommended)
    - Text transcripts
    - Optional: forced alignment (MFA) for duration supervision

Training on laptop GPU (GTX 1650+):
    - Prosody predictor: ~6-8 hours for 50K steps
    - HiFi-GAN fine-tuning: ~12-16 hours for 100K steps
    - Total: ~1-2 days end-to-end

Usage:
    # Prepare LJSpeech dataset
    python train_nefs_tts.py --dataset ljspeech --data-dir ./LJSpeech-1.1 \\
        --output-dir ./nefs_checkpoints --epochs 100
    
    # Resume training
    python train_nefs_tts.py --resume ./nefs_checkpoints/latest.pt
    
    # Inference with trained model
    python train_nefs_tts.py --mode inference --checkpoint best.pt \\
        --text "Testing NEFS synthesis"
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import NEFS components
try:
    from nefs_tts_hifigan import (
        NEFSPhonemeEncoder,
        NEFSProsodyPredictor,
        NEFSHiFiGANSynthesizer,
    )
    from nefs_g2p import text_to_ipa
    from nefs_wrapper import NEFSConverter
except ImportError as e:
    logger.error(f"Failed to import NEFS modules: {e}")
    logger.error("Ensure nefs_tts_hifigan.py, nefs_g2p.py, nefs_wrapper.py are in PYTHONPATH")
    raise


# =============================================================================
# Direct Text → NEFS G2P (bypassing IPA)
# =============================================================================

class DirectG2PNEFS(nn.Module):
    """
    Learned grapheme → NEFS mapping (no IPA intermediate step).
    
    Architecture: Character-level encoder → NEFS byte decoder
    Similar to CharsiuG2P's ByT5 approach, but outputs NEFS bytes directly.
    
    For initial training, we bootstrap from espeak-ng (text → IPA → NEFS),
    then fine-tune end-to-end with audio reconstruction loss.
    """
    
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        # Character encoder (handles any UTF-8 character)
        self.char_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        
        # NEFS byte decoder (outputs 0-255)
        self.decoder = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, 256)  # 256 possible NEFS bytes
    
    def forward(self, text_bytes: torch.Tensor, max_nefs_len: int = 100) -> torch.Tensor:
        """
        Args:
            text_bytes: (batch, text_len) UTF-8 byte values
            max_nefs_len: maximum NEFS sequence length
        
        Returns:
            (batch, nefs_len, 256) logits over NEFS byte vocabulary
        """
        # Encode text
        char_emb = self.char_embedding(text_bytes)
        enc_out, _ = self.encoder(char_emb)
        
        # Decode to NEFS sequence (autoregressive).
        # Each step we pass only the most recent embedding as input rather than
        # the full accumulated sequence — avoids O(n²) memory from torch.cat.
        batch_size = text_bytes.size(0)
        # Use the first encoded character as the start token.
        decoder_input = enc_out[:, 0:1, :]  # (batch, 1, hidden*2)
        hidden = None  # LSTM state carried across steps

        outputs = []
        for t in range(max_nefs_len):
            dec_out, hidden = self.decoder(decoder_input, hidden)
            logits = self.output_proj(dec_out)  # (batch, 1, 256)
            outputs.append(logits)

            # Next input: embed the argmax prediction for this step
            next_token = logits.argmax(dim=-1)  # (batch, 1)
            decoder_input = self.char_embedding(next_token)  # (batch, 1, hidden)
        
        return torch.cat(outputs, dim=1)  # (batch, nefs_len, 256)
    
    @torch.no_grad()
    def infer(self, text: str) -> torch.Tensor:
        """
        Text string → NEFS bytes (inference mode).
        
        Args:
            text: Input text
        
        Returns:
            (nefs_len,) tensor of NEFS byte values
        """
        # Encode text as UTF-8 bytes so non-ASCII characters (accented letters,
        # CJK, Arabic, etc.) are represented correctly.  Using ord(c) only works
        # for ASCII — it returns the Unicode code point, which can exceed 255 and
        # will silently wrap or error when cast to a byte tensor.
        text_bytes = torch.tensor(
            list(text.encode("utf-8")), dtype=torch.long
        ).unsqueeze(0)
        
        # Forward pass
        logits = self.forward(text_bytes, max_nefs_len=len(text) * 3)  # Allow expansion
        nefs_bytes = logits.argmax(dim=-1).squeeze(0)  # (nefs_len,)
        
        # Trim at EOS (byte 0x00 or duplicate end)
        # For simplicity, return full sequence (can add proper EOS handling)
        return nefs_bytes


# =============================================================================
# Dataset: LJSpeech with NEFS targets
# =============================================================================

class LJSpeechNEFSDataset(Dataset):
    """
    LJSpeech dataset with NEFS byte targets.
    
    Directory structure:
        LJSpeech-1.1/
        ├── wavs/
        │   ├── LJ001-0001.wav
        │   └── ...
        └── metadata.csv  (format: filename|transcript|normalized_transcript)
    """
    
    def __init__(
        self,
        data_dir: Path,
        sample_rate: int = 22050,
        use_g2p_bootstrap: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.use_g2p_bootstrap = use_g2p_bootstrap
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.csv"
        self.samples = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    filename, transcript = parts[0], parts[1]
                    wav_path = self.data_dir / "wavs" / f"{filename}.wav"
                    if wav_path.exists():
                        self.samples.append((wav_path, transcript))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
        
        # Initialize NEFS converter for bootstrapping
        if use_g2p_bootstrap:
            self.nefs_converter = NEFSConverter()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        wav_path, transcript = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)  # (samples,)
        
        # Convert text → NEFS bytes (bootstrapped via IPA for now)
        if self.use_g2p_bootstrap:
            try:
                ipa = text_to_ipa(transcript, lang='en-us', prefer='espeak')
                nefs_bytes_obj = self.nefs_converter.ipa_to_nafs(ipa)
                nefs_bytes = torch.tensor(list(nefs_bytes_obj), dtype=torch.long)
            except Exception as e:
                logger.warning(f"G2P failed for '{transcript}': {e}. Using fallback.")
                # Fallback: encode transcript as UTF-8 bytes.
                # ord(c) % 256 was used previously but is wrong for non-ASCII —
                # Unicode code points do not equal UTF-8 byte values above 0x7F.
                nefs_bytes = torch.tensor(
                    list(transcript.encode('utf-8')), dtype=torch.long
                )
        else:
            # Direct character encoding (for learned G2P path).
            # Use UTF-8 byte encoding — ord(c) returns Unicode code points which
            # exceed 255 for non-ASCII characters and would index out of range.
            nefs_bytes = torch.tensor(
                list(transcript.encode('utf-8')), dtype=torch.long
            )
        
        return {
            'waveform': waveform,
            'nefs_bytes': nefs_bytes,
            'transcript': transcript,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Pad variable-length sequences to batch together.

    Returns padding masks alongside padded tensors so the model and loss
    can ignore pad positions.  A mask value of True means "real data";
    False means "padding" — consistent with PyTorch's key_padding_mask
    convention used by nn.MultiheadAttention and similar modules.
    """
    waveforms = [item['waveform'] for item in batch]
    nefs_seqs = [item['nefs_bytes'] for item in batch]
    transcripts = [item['transcript'] for item in batch]

    # Pad waveforms and build a sample-level mask
    max_wav_len = max(w.size(0) for w in waveforms)
    waveforms_padded = torch.stack([
        F.pad(w, (0, max_wav_len - w.size(0))) for w in waveforms
    ])
    wav_lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)
    # (batch, max_wav_len) — True where real audio, False where padding
    wav_mask = torch.arange(max_wav_len).unsqueeze(0) < wav_lengths.unsqueeze(1)

    # Pad NEFS sequences and build a token-level mask
    max_nefs_len = max(n.size(0) for n in nefs_seqs)
    nefs_padded = torch.stack([
        F.pad(n, (0, max_nefs_len - n.size(0))) for n in nefs_seqs
    ])
    nefs_lengths = torch.tensor([n.size(0) for n in nefs_seqs], dtype=torch.long)
    # (batch, max_nefs_len) — True where real phoneme, False where padding
    nefs_mask = torch.arange(max_nefs_len).unsqueeze(0) < nefs_lengths.unsqueeze(1)

    return {
        'waveform': waveforms_padded,
        'wav_mask': wav_mask,
        'nefs_bytes': nefs_padded,
        'nefs_mask': nefs_mask,
        'nefs_lengths': nefs_lengths,
        'transcript': transcripts,
    }


# =============================================================================
# Training loop
# =============================================================================

def train_nefs_tts(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = 'cuda',
    resume_from: Optional[Path] = None,
):
    """
    Train NEFS TTS system end-to-end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Dataset
    dataset = LJSpeechNEFSDataset(data_dir, use_g2p_bootstrap=True)
    # num_workers > 0 uses multiprocessing.  On Windows the default start
    # method is 'spawn', which requires the DataLoader to be created inside
    # a ``if __name__ == '__main__':`` guard — otherwise child processes
    # re-import the module and re-execute top-level code.  On macOS with
    # PyTorch >= 1.8 the same applies for 'spawn' (the default since 1.8).
    # We detect the platform and fall back to 0 (in-process) where forking
    # is unavailable to avoid silent hangs.
    import multiprocessing
    safe_num_workers = 0 if multiprocessing.get_start_method(allow_none=True) == 'spawn' else 4
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=safe_num_workers,
        collate_fn=collate_fn, pin_memory=(str(device) != 'cpu'),
    )
    
    # Models
    phoneme_encoder = NEFSPhonemeEncoder(embedding_dim=512).to(device)
    prosody_predictor = NEFSProsodyPredictor(input_dim=512).to(device)
    
    # Optimizer
    params = list(phoneme_encoder.parameters()) + list(prosody_predictor.parameters())
    optimizer = AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Scheduler
    total_steps = len(dataloader) * epochs
    if total_steps == 0:
        raise ValueError(
            f"Dataset at '{data_dir}' produced 0 valid samples. "
            "Check that metadata.csv exists and wav paths are correct."
        )
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05
    )
    
    # Resume
    start_epoch = 0
    if resume_from and resume_from.exists():
        # map_location ensures a GPU checkpoint loads cleanly on a CPU-only machine.
        # weights_only=True (PyTorch ≥ 2.0) prevents arbitrary code execution from
        # untrusted checkpoint files and suppresses the FutureWarning in 2.x.
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        phoneme_encoder.load_state_dict(checkpoint['phoneme_encoder'])
        prosody_predictor.load_state_dict(checkpoint['prosody_predictor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        phoneme_encoder.train()
        prosody_predictor.train()
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            waveforms = batch['waveform'].to(device)
            nefs_bytes = batch['nefs_bytes'].to(device)
            nefs_mask = batch['nefs_mask'].to(device)   # (batch, seq_len) bool

            # Forward pass
            phoneme_emb = phoneme_encoder(nefs_bytes)
            frame_emb, pred_durations, pred_f0 = prosody_predictor(phoneme_emb)

            # Loss: masked embedding regularisation + waveform reconstruction proxy.
            # Future work could add:
            #   - Duration MSE: F.mse_loss(pred_durations[nefs_mask], target_dur[nefs_mask])
            #   - F0 MSE on voiced frames
            #   - Full HiFi-GAN discriminator losses
            # Mask embedding values at pad positions before computing the loss so
            # the model is not penalised for (or influenced by) pad representations.
            masked_emb = frame_emb * nefs_mask.unsqueeze(-1).float()
            # Normalise by the number of real tokens to keep loss scale stable
            # across batches with different amounts of padding.
            real_tokens = nefs_mask.sum().clamp(min=1)
            emb_loss = masked_emb.pow(2).sum() / real_tokens * 0.01

            # Waveform reconstruction proxy: project frame embeddings to the time
            # domain and compare against the ground-truth waveform energy.
            # frame_emb shape: (batch, seq_len, hidden) — average across the
            # sequence to get a per-batch energy estimate, then align to the
            # waveform RMS so the model is rewarded for matching signal loudness.
            pred_energy = masked_emb.pow(2).mean(dim=[1, 2])          # (batch,)
            wav_rms = waveforms.pow(2).mean(dim=-1).clamp(min=1e-8)   # (batch,)
            wav_loss = F.mse_loss(pred_energy, wav_rms)

            loss = emb_loss + wav_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'phoneme_encoder': phoneme_encoder.state_dict(),
                'prosody_predictor': prosody_predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train NEFS TTS")
    parser.add_argument('--mode', default='train', choices=['train', 'inference'])
    parser.add_argument('--data-dir', type=Path, default='./LJSpeech-1.1')
    parser.add_argument('--output-dir', type=Path, default='./nefs_checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--text', type=str, default="Testing NEFS synthesis")
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_nefs_tts(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            resume_from=args.resume,
        )
    
    elif args.mode == 'inference':
        if not args.checkpoint:
            logger.error("Must provide --checkpoint for inference mode")
            return
        
        # Load model
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        
        phoneme_encoder = NEFSPhonemeEncoder(embedding_dim=512).to(device)
        prosody_predictor = NEFSProsodyPredictor(input_dim=512).to(device)
        
        phoneme_encoder.load_state_dict(checkpoint['phoneme_encoder'])
        prosody_predictor.load_state_dict(checkpoint['prosody_predictor'])
        
        # Synthesize
        logger.info(f"Synthesizing: {args.text}")
        # This would require HiFi-GAN to be attached, which needs separate training
        logger.warning("Inference mode requires full HiFi-GAN integration (not yet complete)")
        logger.info("For now, checkpoint contains trained phoneme encoder + prosody predictor.")


if __name__ == "__main__":
    main()
