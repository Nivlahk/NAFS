"""nefs_hifigan_discriminator.py — HiFi-GAN discriminators for NEFS TTS

Implements the Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
from the HiFi-GAN paper, adapted for NEFS-conditioned generation.

Reference:
    Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and
    High Fidelity Speech Synthesis" NeurIPS 2020
    https://arxiv.org/abs/2010.05646

These discriminators analyze audio at multiple resolutions to ensure both
local detail (phone-level clarity) and global structure (prosody, rhythm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# =============================================================================
# Multi-Period Discriminator (MPD)
# =============================================================================

class PeriodDiscriminator(nn.Module):
    """
    Discriminator that analyzes audio at a specific period.
    
    Reshapes 1D waveform into 2D with period as one dimension,
    then applies 2D convolutions to capture periodic patterns.
    """
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        # 2D convolutions (operates on reshaped periodic audio)
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, samples) audio waveform
        
        Returns:
            output: (batch, 1, T, 1) discriminator score
            feature_maps: List of intermediate feature maps
        """
        fmap = []
        
        # Reshape to (batch, 1, T, period)
        b, c, t = x.shape
        if t % self.period != 0:
            pad_amount = self.period - (t % self.period)
            x = F.pad(x, (0, pad_amount), "reflect")
            t = x.size(-1)
        
        x = x.view(b, c, t // self.period, self.period)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Ensemble of period discriminators at different periods.
    
    HiFi-GAN uses periods [2, 3, 5, 7, 11] to capture different
    rhythmic structures in speech.
    """
    
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, 1, samples) audio
        
        Returns:
            outputs: List of discriminator scores from each period
            feature_maps: List of feature map lists from each period
        """
        outputs = []
        fmaps = []
        
        for disc in self.discriminators:
            out, fmap = disc(x)
            outputs.append(out)
            fmaps.append(fmap)
        
        return outputs, fmaps


# =============================================================================
# Multi-Scale Discriminator (MSD)
# =============================================================================

class ScaleDiscriminator(nn.Module):
    """
    Discriminator that operates on a specific downsampled scale.
    Uses 1D convolutions with increasing receptive fields.
    """
    
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, 1, samples)
        
        Returns:
            output: (batch, 1, T) discriminator score
            feature_maps: List of intermediate features
        """
        fmap = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Ensemble of scale discriminators operating at different sample rates.
    
    Uses average pooling to downsample audio to 1x, 2x, 4x scales.
    """
    
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        
        # Pooling layers for downsampling
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, 1, samples)
        
        Returns:
            outputs: List of scores from each scale
            feature_maps: List of feature map lists
        """
        outputs = []
        fmaps = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.meanpools[i-1](x)
            out, fmap = disc(x)
            outputs.append(out)
            fmaps.append(fmap)
        
        return outputs, fmaps


# =============================================================================
# Combined Discriminator
# =============================================================================

class NEFSHiFiGANDiscriminator(nn.Module):
    """
    Complete HiFi-GAN discriminator combining MPD + MSD.
    
    Used for adversarial training of NEFS-conditioned generator.
    """
    
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
    
    def forward(self, x: torch.Tensor) -> Tuple:
        """
        Args:
            x: (batch, 1, samples) or (batch, samples) audio
        
        Returns:
            mpd_outputs, mpd_fmaps, msd_outputs, msd_fmaps
        """
        # Ensure (batch, 1, samples) shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        mpd_outputs, mpd_fmaps = self.mpd(x)
        msd_outputs, msd_fmaps = self.msd(x)
        
        return mpd_outputs, mpd_fmaps, msd_outputs, msd_fmaps


# =============================================================================
# GAN Loss Functions
# =============================================================================

def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Hinge loss for discriminator (want real=1, fake=0).
    
    Args:
        disc_real_outputs: List of discriminator scores on real audio
        disc_generated_outputs: List of scores on generated audio
    
    Returns:
        total_loss: Sum of losses across all discriminators
        real_losses: Per-discriminator losses on real
        fake_losses: Per-discriminator losses on fake
    """
    real_losses = []
    fake_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((1 - dr) ** 2)
        fake_loss = torch.mean(dg ** 2)
        real_losses.append(real_loss)
        fake_losses.append(fake_loss)
    
    total_loss = sum(real_losses) + sum(fake_losses)
    return total_loss, real_losses, fake_losses


def generator_loss(
    disc_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generator adversarial loss (want discriminator to output 1).
    
    Args:
        disc_outputs: Discriminator scores on generated audio
    
    Returns:
        total_loss: Sum across all discriminators
        individual_losses: Per-discriminator losses
    """
    losses = []
    for dg in disc_outputs:
        loss = torch.mean((1 - dg) ** 2)
        losses.append(loss)
    
    return sum(losses), losses


def feature_matching_loss(
    fmap_real: List[List[torch.Tensor]],
    fmap_generated: List[List[torch.Tensor]]
) -> torch.Tensor:
    """
    Feature matching loss: L1 distance between intermediate feature maps.
    
    Encourages generator to match discriminator's internal representations.
    
    Args:
        fmap_real: Feature maps from real audio
        fmap_generated: Feature maps from generated audio
    
    Returns:
        L1 distance summed across all feature maps
    """
    loss = 0
    for dr, dg in zip(fmap_real, fmap_generated):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    
    return loss


def mel_spectrogram_loss(
    y_real: torch.Tensor,
    y_generated: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """
    L1 loss between mel-spectrograms of real and generated audio.
    
    Note: Even though NEFS bypasses mel-specs for generation,
    we still use them as an auxiliary loss for training stability.
    
    Args:
        y_real: (batch, samples) real audio
        y_generated: (batch, samples) generated audio
    
    Returns:
        L1 distance between mel-spectrograms
    """
    import torchaudio
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=80,
    ).to(y_real.device)
    
    mel_real = mel_transform(y_real)
    mel_generated = mel_transform(y_generated)
    
    return F.l1_loss(mel_real, mel_generated)
