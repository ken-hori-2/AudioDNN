"""
Audio processing utilities for music source separation.

This module provides various audio processing functions including
STFT/iSTFT, audio loading/saving, and preprocessing utilities.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 44100,
    mono: bool = False,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (resample if different)
        mono: Convert to mono if True
        duration: Duration to load in seconds (None for entire file)
        offset: Start offset in seconds
    
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Calculate frame parameters if duration/offset specified
    frame_offset = int(offset * sample_rate) if offset > 0 else 0
    num_frames = int(duration * sample_rate) if duration is not None else -1
    
    # Load audio
    try:
        waveform, orig_sr = torchaudio.load(
            str(path),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path}: {e}")
    
    # Resample if necessary
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=sample_rate
        )
        waveform = resampler(waveform)
    
    # Convert to mono if requested
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Ensure stereo if not mono
    elif not mono and waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    
    return waveform, sample_rate


def save_audio(
    waveform: torch.Tensor,
    path: Union[str, Path],
    sample_rate: int = 44100,
    bits_per_sample: int = 16,
    encoding: str = "PCM_S",
) -> None:
    """
    Save audio to file.
    
    Args:
        waveform: Audio tensor of shape (channels, time)
        path: Output file path
        sample_rate: Audio sample rate
        bits_per_sample: Bits per sample (16 or 24)
        encoding: Audio encoding format
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio is in valid range
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    # Convert to CPU if on GPU
    if waveform.is_cuda:
        waveform = waveform.cpu()
    
    try:
        torchaudio.save(
            str(path),
            waveform,
            sample_rate,
            bits_per_sample=bits_per_sample,
            encoding=encoding,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {path}: {e}")


def compute_stft(
    waveform: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann",
    center: bool = True,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        waveform: Input waveform of shape (..., time)
        n_fft: FFT size
        hop_length: Hop length
        window: Window function name
        center: Whether to center the window
        normalized: Whether to normalize the STFT
    
    Returns:
        Complex STFT tensor of shape (..., freq_bins, time_frames)
    """
    if window == "hann":
        window_tensor = torch.hann_window(n_fft, device=waveform.device)
    elif window == "hamming":
        window_tensor = torch.hamming_window(n_fft, device=waveform.device)
    else:
        raise ValueError(f"Unsupported window: {window}")
    
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_tensor,
        center=center,
        normalized=normalized,
        return_complex=True,
        pad_mode="constant",
    )


def compute_istft(
    stft_tensor: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "hann",
    center: bool = True,
    normalized: bool = False,
    length: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        stft_tensor: Complex STFT tensor of shape (..., freq_bins, time_frames)
        n_fft: FFT size
        hop_length: Hop length
        window: Window function name
        center: Whether the STFT was centered
        normalized: Whether the STFT was normalized
        length: Target length of output waveform
    
    Returns:
        Reconstructed waveform of shape (..., time)
    """
    if window == "hann":
        window_tensor = torch.hann_window(n_fft, device=stft_tensor.device)
    elif window == "hamming":
        window_tensor = torch.hamming_window(n_fft, device=stft_tensor.device)
    else:
        raise ValueError(f"Unsupported window: {window}")
    
    return torch.istft(
        stft_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_tensor,
        center=center,
        normalized=normalized,
        length=length,
    )


def apply_mask(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply mask to magnitude spectrogram and reconstruct complex STFT.
    
    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        mask: Soft mask (0-1 values)
    
    Returns:
        Masked complex STFT
    """
    masked_magnitude = magnitude * mask
    return masked_magnitude * torch.exp(1j * phase)


def normalize_audio(waveform: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
    """
    Normalize audio to target LUFS (simple RMS-based approximation).
    
    Args:
        waveform: Input waveform
        target_lufs: Target loudness in LUFS
    
    Returns:
        Normalized waveform
    """
    # Simple RMS-based normalization (not true LUFS)
    rms = torch.sqrt(torch.mean(waveform ** 2))
    
    if rms > 0:
        # Convert target LUFS to linear scale (rough approximation)
        target_rms = 10 ** (target_lufs / 20)
        gain = target_rms / rms
        return waveform * gain
    
    return waveform


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """
    Normalize audio to target peak level.
    
    Args:
        waveform: Input waveform
        target_peak: Target peak level (0-1)
    
    Returns:
        Peak-normalized waveform
    """
    peak = torch.max(torch.abs(waveform))
    
    if peak > 0:
        gain = target_peak / peak
        return waveform * gain
    
    return waveform


def mix_sources(sources: Dict[str, torch.Tensor], gains: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """
    Mix multiple source signals.
    
    Args:
        sources: Dictionary of source signals
        gains: Optional gains for each source
    
    Returns:
        Mixed signal
    """
    if gains is None:
        gains = {name: 1.0 for name in sources.keys()}
    
    mix = None
    for name, signal in sources.items():
        gain = gains.get(name, 1.0)
        weighted_signal = signal * gain
        
        if mix is None:
            mix = weighted_signal
        else:
            mix = mix + weighted_signal
    
    return mix


def compute_energy(waveform: torch.Tensor, frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Compute frame-wise energy of audio signal.
    
    Args:
        waveform: Input waveform
        frame_length: Frame length for energy computation
        hop_length: Hop length
    
    Returns:
        Frame-wise energy
    """
    # Pad waveform
    pad_length = frame_length // 2
    padded = torch.nn.functional.pad(waveform, (pad_length, pad_length))
    
    # Compute frame-wise energy
    num_frames = (padded.shape[-1] - frame_length) // hop_length + 1
    energy = torch.zeros(num_frames, device=waveform.device)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = padded[..., start:end]
        energy[i] = torch.mean(frame ** 2)
    
    return energy


def apply_window_compensation(
    waveform: torch.Tensor,
    window_type: str = "hann",
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> torch.Tensor:
    """
    Apply window compensation to account for STFT window effects.
    
    Args:
        waveform: Input waveform
        window_type: Type of window used in STFT
        n_fft: FFT size
        hop_length: Hop length
    
    Returns:
        Compensated waveform
    """
    # Spleeter uses this compensation factor
    if window_type == "hann":
        compensation_factor = 2.0 / 3.0
        return waveform * compensation_factor
    
    return waveform


def trim_silence(
    waveform: torch.Tensor,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio.
    
    Args:
        waveform: Input waveform
        threshold: Energy threshold for silence detection
        frame_length: Frame length for energy computation
        hop_length: Hop length
    
    Returns:
        Trimmed waveform
    """
    energy = compute_energy(waveform, frame_length, hop_length)
    
    # Find first and last non-silent frames
    non_silent = energy > threshold
    
    if not torch.any(non_silent):
        # All frames are silent
        return waveform[..., :frame_length]
    
    first_frame = torch.argmax(non_silent.float())
    last_frame = len(non_silent) - torch.argmax(non_silent.flip(0).float()) - 1
    
    # Convert frame indices to sample indices
    start_sample = first_frame * hop_length
    end_sample = (last_frame + 1) * hop_length + frame_length // 2
    
    return waveform[..., start_sample:end_sample]


if __name__ == "__main__":
    # Test audio utilities
    
    # Create test audio
    test_audio = torch.randn(2, 44100 * 3)  # 3 seconds stereo
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test STFT/iSTFT
    stft_result = compute_stft(test_audio)
    print(f"STFT shape: {stft_result.shape}")
    
    reconstructed = compute_istft(stft_result, length=test_audio.shape[-1])
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    reconstruction_error = torch.mean((test_audio - reconstructed) ** 2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Test masking
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)
    mask = torch.rand_like(magnitude)
    
    masked_stft = apply_mask(magnitude, phase, mask)
    print(f"Masked STFT shape: {masked_stft.shape}")
    
    # Test normalization
    normalized = peak_normalize(test_audio, target_peak=0.8)
    peak_after = torch.max(torch.abs(normalized))
    print(f"Peak after normalization: {peak_after:.3f}")
    
    # Test energy computation
    energy = compute_energy(test_audio)
    print(f"Energy shape: {energy.shape}")
    
    print("Audio utilities test completed successfully!")
