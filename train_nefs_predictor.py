"""
train_nefs_predictor.py — Training loop for NEFSParamPredictor

Phase 1: supervised regression on prosody targets extracted from audio.
  - F0 targets from pitch extraction (pyin via librosa, or crepe)
  - Duration targets from forced alignment (MFA) or eSpeak bootstrap
  - Rate factor derived from duration relative to phoneme-class mean
  - Breathiness, energy, formants: zero-weighted until Phase 2

Phase 2 (future): audio-supervised fine-tuning.
  - Predictor → eSpeak → mel spectrogram
  - MSE against LJSpeech reference mel
  - Finite-difference gradients through eSpeak (non-differentiable)

Dataset contract
----------------
The DataLoader must yield batches with these keys:

  nefs_bytes   : (B, T) int64       — NEFS byte sequence
  nefs_mask    : (B, T) bool        — True where real phoneme, False pad
  nefs_lengths : (B,)   int64       — actual sequence lengths

  f0_hz        : (B, T) float32     — per-phoneme F0 target in Hz
                                       0.0 for unvoiced phonemes
  duration_s   : (B, T) float32     — per-phoneme duration in seconds
  f0_mask      : (B, T) bool        — True where F0 is valid (voiced)

  These are optional and zero-weighted in Phase 1:
  breathiness  : (B, T) float32     — 0.0–1.0
  energy       : (B, T) float32     — 0.0–1.0

The LJSpeechNEFSDataset in train_nefs_tts.py does not yet produce
f0_hz or duration_s — those come from nefs_prosody_extractor.py (built
next). This file defines the loop so the extractor knows exactly what
shape and dtype to produce.

Usage
-----
  python train_nefs_predictor.py \
      --data-dir ./LJSpeech-1.1 \
      --output-dir ./predictor_checkpoints \
      --epochs 100

  # Resume
  python train_nefs_predictor.py \
      --data-dir ./LJSpeech-1.1 \
      --output-dir ./predictor_checkpoints \
      --resume ./predictor_checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from nefs_param_predictor import NEFSParamPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

# Phase 1: only F0 and duration are supervised.
# Everything else is zeroed so untrained heads don't destabilise training.
# Bump breathiness/energy weights to ~0.1 once Phase 2 targets are available.
LOSS_WEIGHTS: Dict[str, float] = {
    "f0":         1.0,    # primary target — log-scale MSE
    "duration":   1.0,    # primary target — log-scale MSE
    "rate":       0.0,    # derived from duration; activate in Phase 2
    "breathiness":0.0,    # no reliable Phase 1 target
    "energy":     0.0,    # no reliable Phase 1 target
    "f1":         0.0,    # Klatt Phase 2
    "f2":         0.0,
    "f3":         0.0,
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_log_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    MSE in log space over valid (masked) positions.

    Log space is correct for F0 and duration because:
      - Both are strictly positive
      - Perceptual distance is multiplicative (semitones, not Hz)
      - Prevents large absolute values from dominating small ones

    Args:
        pred:   (B, T) predicted values in Hz or seconds (already positive
                from the predictor's exp() activation)
        target: (B, T) ground-truth values
        mask:   (B, T) bool — True where loss should be computed
        eps:    small value to avoid log(0)

    Returns:
        Scalar loss.
    """
    if mask.sum() == 0:
        return pred.sum() * 0.0  # differentiable zero

    log_pred   = torch.log(pred.clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))
    sq_err = (log_pred - log_target) ** 2
    return sq_err[mask].mean()


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard MSE over masked positions (for 0–1 range targets)."""
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


# ---------------------------------------------------------------------------
# Batch loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Dict[str, float] = LOSS_WEIGHTS,
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted multi-task loss from predictor outputs and batch targets.

    Args:
        outputs: Dict from NEFSParamPredictor.forward()
        batch:   DataLoader batch dict (see module docstring for keys)
        weights: Per-head loss weights

    Returns:
        Dict with 'total' scalar and per-head scalars for logging.
    """
    nefs_mask = batch["nefs_mask"]          # (B, T) bool
    f0_mask   = batch.get("f0_mask", nefs_mask)  # voiced positions only

    losses = {}

    # F0 loss — only on voiced phonemes
    if weights["f0"] > 0 and "f0_hz" in batch:
        losses["f0"] = masked_log_mse(
            outputs["f0_hz"], batch["f0_hz"], f0_mask
        ) * weights["f0"]

    # Duration loss — all phonemes
    if weights["duration"] > 0 and "duration_s" in batch:
        losses["duration"] = masked_log_mse(
            outputs["duration_s"], batch["duration_s"], nefs_mask
        ) * weights["duration"]

    # Rate factor — derived, activate in Phase 2
    if weights["rate"] > 0 and "rate_factor" in batch:
        losses["rate"] = masked_mse(
            outputs["rate_factor"], batch["rate_factor"], nefs_mask
        ) * weights["rate"]

    # Voice quality — activate in Phase 2
    if weights["breathiness"] > 0 and "breathiness" in batch:
        losses["breathiness"] = masked_mse(
            outputs["breathiness"], batch["breathiness"], nefs_mask
        ) * weights["breathiness"]

    if weights["energy"] > 0 and "energy" in batch:
        losses["energy"] = masked_mse(
            outputs["energy"], batch["energy"], nefs_mask
        ) * weights["energy"]

    # Formant losses — activate when Klatt backend is wired
    for key in ("f1", "f2", "f3"):
        hz_key = f"{key}_hz"
        if weights[key] > 0 and hz_key in batch:
            losses[key] = masked_log_mse(
                outputs[hz_key], batch[hz_key], nefs_mask
            ) * weights[key]

    total = sum(losses.values()) if losses else torch.tensor(0.0)
    losses["total"] = total
    return losses


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    output_dir: Path,
    epochs: int = 100,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    num_layers: int = 2,
    grad_clip: float = 1.0,
    device: str = "cpu",
    resume_from: Optional[Path] = None,
    log_every: int = 50,
    val_every: int = 5,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    model = NEFSParamPredictor(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = len(dataloader) * epochs
    if total_steps == 0:
        raise ValueError("Empty dataloader — check dataset path.")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,         # 5% warmup
        anneal_strategy="cos",
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best_val={best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses: Dict[str, float] = {}
        step = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Move all tensors to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            lengths = batch.get("nefs_lengths")

            outputs = model(batch["nefs_bytes"], lengths=lengths)
            loss_dict = compute_loss(outputs, batch)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            # Accumulate for epoch-level logging
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val

            step += 1
            if step % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "f0":   f"{loss_dict.get('f0', torch.tensor(0)).item():.4f}",
                    "dur":  f"{loss_dict.get('duration', torch.tensor(0)).item():.4f}",
                })

        # Epoch summary
        n = len(dataloader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        logger.info(
            f"Epoch {epoch+1} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
        )

        # Validation
        val_loss = avg["total"]
        if val_dataloader is not None and (epoch + 1) % val_every == 0:
            val_loss = validate(model, val_dataloader, device)
            logger.info(f"  Val loss: {val_loss:.4f}")

        # Checkpoint — always save latest, save best separately
        _save(model, optimizer, epoch, val_loss, best_val_loss,
              output_dir, scheduler)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save(model, optimizer, epoch, val_loss, best_val_loss,
                  output_dir, scheduler, name="best.pt")
            logger.info(f"  New best: {best_val_loss:.4f}")


def validate(
    model: NEFSParamPredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(batch["nefs_bytes"], batch.get("nefs_lengths"))
            loss_dict = compute_loss(outputs, batch)
            total += loss_dict["total"].item()
            steps += 1
    model.train()
    return total / max(steps, 1)


def _save(
    model, optimizer, epoch, val_loss, best_val_loss,
    output_dir, scheduler, name="latest.pt"
):
    torch.save({
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "epoch":          epoch,
        "val_loss":       val_loss,
        "best_val_loss":  best_val_loss,
        "loss_weights":   LOSS_WEIGHTS,
    }, output_dir / name)


# ---------------------------------------------------------------------------
# Stub dataset — lets the training loop run before the extractor exists
# ---------------------------------------------------------------------------

class StubNEFSDataset(torch.utils.data.Dataset):
    """
    Generates synthetic prosody targets so the training loop can be
    validated end-to-end before nefs_prosody_extractor.py is built.

    Replace with the real dataset once the extractor exists:
        from nefs_prosody_extractor import LJSpeechProsodyDataset
        dataset = LJSpeechProsodyDataset(data_dir)

    Synthetic targets are drawn from realistic distributions:
      F0:       log-normal centred on 120 Hz (±40 Hz std in log space)
      Duration: log-normal centred on 80 ms
      Voiced:   ~65% of phonemes (IPA-realistic proportion)
    """

    def __init__(self, n_samples: int = 1000, max_len: int = 40):
        self.n_samples = n_samples
        self.max_len   = max_len
        from nefs_converter import _NEFS_TO_IPA, classify_byte
        # Only sample phoneme bytes — consonants and vowels, not diacritics/tones
        self.assigned = [
            b for b, ipa in _NEFS_TO_IPA.items()
            if ipa and classify_byte(b) in ('consonant', 'vowel')
        ]
        if not self.assigned:
            self.assigned = list(range(0x10, 0x90))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        seq_len = rng.integers(8, self.max_len)

        nefs_bytes = torch.tensor(
            rng.choice(self.assigned, size=seq_len), dtype=torch.long
        )

        # Synthetic F0: log-normal, 0 for ~35% (unvoiced)
        f0_raw    = np.exp(rng.normal(np.log(120), 0.25, seq_len)).astype(np.float32)
        voiced    = rng.random(seq_len) > 0.35
        f0_hz     = torch.tensor(f0_raw * voiced, dtype=torch.float32)
        f0_mask   = torch.tensor(voiced, dtype=torch.bool)

        # Synthetic duration: log-normal centred on 80ms
        duration_s = torch.tensor(
            np.exp(rng.normal(np.log(0.08), 0.3, seq_len)).clip(0.02, 0.5),
            dtype=torch.float32,
        )

        return {
            "nefs_bytes":  nefs_bytes,
            "nefs_mask":   torch.ones(seq_len, dtype=torch.bool),
            "nefs_lengths":torch.tensor(seq_len, dtype=torch.long),
            "f0_hz":       f0_hz,
            "f0_mask":     f0_mask,
            "duration_s":  duration_s,
        }


def stub_collate(batch):
    """Pad variable-length sequences from StubNEFSDataset."""
    max_len = max(b["nefs_bytes"].size(0) for b in batch)

    def pad(t, val=0):
        return F.pad(t, (0, max_len - t.size(0)), value=val)

    return {
        "nefs_bytes":   torch.stack([pad(b["nefs_bytes"])   for b in batch]),
        "nefs_mask":    torch.stack([pad(b["nefs_mask"])    for b in batch]),
        "nefs_lengths": torch.stack([b["nefs_lengths"]      for b in batch]),
        "f0_hz":        torch.stack([pad(b["f0_hz"])        for b in batch]),
        "f0_mask":      torch.stack([pad(b["f0_mask"])      for b in batch]),
        "duration_s":   torch.stack([pad(b["duration_s"])   for b in batch]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NEFSParamPredictor")
    parser.add_argument("--data-dir",    type=Path, default=None,
                        help="LJSpeech root (omit to use stub dataset)")
    parser.add_argument("--output-dir",  type=Path, default=Path("./predictor_checkpoints"))
    parser.add_argument("--epochs",      type=int,  default=100)
    parser.add_argument("--batch-size",  type=int,  default=32)
    parser.add_argument("--lr",          type=float,default=3e-4)
    parser.add_argument("--hidden-dim",  type=int,  default=128)
    parser.add_argument("--num-layers",  type=int,  default=2)
    parser.add_argument("--device",      type=str,  default="cpu")
    parser.add_argument("--resume",      type=Path, default=None)
    parser.add_argument("--val-split",   type=float,default=0.1)
    args = parser.parse_args()

    # Windows requires num_workers=0 (spawn-based multiprocessing breaks DataLoader)
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2

    if args.data_dir is not None:
        # Real dataset — requires nefs_prosody_extractor.py
        try:
            from nefs_prosody_extractor import LJSpeechProsodyDataset, prosody_collate
            full = LJSpeechProsodyDataset(args.data_dir)
            collate_fn = prosody_collate
            logger.info(f"Loaded {len(full)} real samples from {args.data_dir}")
        except ImportError:
            logger.error(
                "nefs_prosody_extractor.py not found. "
                "Run with --data-dir omitted to use the stub dataset, "
                "or build the extractor first."
            )
            return
    else:
        logger.info("No --data-dir provided — using stub dataset for loop validation.")
        full = StubNEFSDataset(n_samples=2000)
        collate_fn = stub_collate

    # Train/val split
    val_size  = max(1, int(len(full) * args.val_split))
    train_size = len(full) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(args.device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    train(
        dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
