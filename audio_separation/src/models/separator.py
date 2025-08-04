"""
Music source separation system using U-Net models.

This module provides a complete separation pipeline including STFT/iSTFT,
masking, and post-processing following the Spleeter approach.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .unet import UNet, create_unet_model


class SourceSeparator(nn.Module):
    """
    Complete source separation system.
    
    This class wraps the U-Net model and provides a complete separation pipeline
    including STFT computation, masking, and iSTFT reconstruction.
    
    Args:
        model: U-Net model for mask prediction
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT
        window: Window function for STFT
        source_names: Names of the sources (e.g., ["vocals", "drums", "bass", "other"])
        sample_rate: Audio sample rate
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_fft: int = 4096,
        hop_length: int = 1024,
        window: str = "hann",
        source_names: Optional[List[str]] = None,
        sample_rate: int = 44100,
    ):
        super().__init__()
        
        self.model = model
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.sample_rate = sample_rate
        
        if source_names is None:
            source_names = ["vocals", "drums", "bass", "other"]
        self.source_names = source_names
        
        # Pre-compute window for STFT/iSTFT
        if window == "hann":
            self.window_fn = torch.hann_window(n_fft)
        else:
            raise ValueError(f"Unsupported window: {window}")
        
        # Normalization statistics (to be computed from training data)
        self.register_buffer("freq_mean", torch.zeros(n_fft // 2 + 1))
        self.register_buffer("freq_std", torch.ones(n_fft // 2 + 1))
    
    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT of waveform.
        
        Args:
            waveform: Input waveform of shape (batch, channels, time)
        
        Returns:
            Complex STFT tensor of shape (batch, channels, freq_bins, time_frames)
        """
        batch_size, n_channels, _ = waveform.shape
        
        # Reshape to (batch * channels, time) for STFT
        waveform_flat = waveform.view(-1, waveform.shape[-1])
        
        # Compute STFT
        stft_result = torch.stft(
            waveform_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window_fn.to(waveform.device),
            return_complex=True,
            pad_mode="constant",
        )
        
        # Reshape back to (batch, channels, freq_bins, time_frames)
        freq_bins, time_frames = stft_result.shape[-2:]
        stft_result = stft_result.reshape(batch_size, n_channels, freq_bins, time_frames)
        
        return stft_result
    
    def istft(self, stft_tensor: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Compute inverse STFT.
        
        Args:
            stft_tensor: Complex STFT tensor of shape (batch, channels, freq_bins, time_frames)
            length: Target length of output waveform
        
        Returns:
            Reconstructed waveform of shape (batch, channels, time)
        """
        batch_size, n_channels, _, _ = stft_tensor.shape
        
        # Reshape to (batch * channels, freq_bins, time_frames) for iSTFT
        stft_flat = stft_tensor.reshape(-1, *stft_tensor.shape[-2:])
        
        # Compute iSTFT
        waveform_flat = torch.istft(
            stft_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window_fn.to(stft_tensor.device),
            length=length,
        )
        
        # Reshape back to (batch, channels, time)
        waveform = waveform_flat.reshape(batch_size, n_channels, -1)
        
        return waveform
    
    def normalize_spectrogram(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Normalize magnitude spectrogram using frequency-wise statistics.
        
        Args:
            magnitude: Magnitude spectrogram of shape (batch, channels, freq_bins, time_frames)
        
        Returns:
            Normalized magnitude spectrogram
        """
        # Expand dimensions for broadcasting
        mean = self.freq_mean.view(1, 1, -1, 1)
        std = self.freq_std.view(1, 1, -1, 1)
        
        return (magnitude - mean) / (std + 1e-8)
    
    def denormalize_spectrogram(self, normalized_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Denormalize magnitude spectrogram.
        
        Args:
            normalized_magnitude: Normalized magnitude spectrogram
        
        Returns:
            Denormalized magnitude spectrogram
        """
        # Expand dimensions for broadcasting
        mean = self.freq_mean.view(1, 1, -1, 1)
        std = self.freq_std.view(1, 1, -1, 1)
        
        return normalized_magnitude * std + mean
    
    def compute_normalization_stats(self, dataloader) -> None:
        """
        Compute normalization statistics from training data.
        
        Args:
            dataloader: Training dataloader
        """
        print("Computing normalization statistics...")
        
        freq_sum = torch.zeros(self.n_fft // 2 + 1)
        freq_sum_sq = torch.zeros(self.n_fft // 2 + 1)
        count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    mix_waveform = batch[0]  # Assuming first element is mix
                else:
                    mix_waveform = batch
                
                # Compute STFT
                mix_stft = self.stft(mix_waveform)
                mix_magnitude = torch.abs(mix_stft)
                
                # Accumulate statistics over all dimensions except frequency
                batch_mean = mix_magnitude.mean(dim=(0, 1, 3))  # Mean over batch, channels, time
                batch_mean_sq = (mix_magnitude ** 2).mean(dim=(0, 1, 3))
                
                freq_sum += batch_mean.cpu()
                freq_sum_sq += batch_mean_sq.cpu()
                count += 1
        
        # Compute mean and std
        self.freq_mean.copy_(freq_sum / count)
        self.freq_std.copy_(torch.sqrt(freq_sum_sq / count - self.freq_mean ** 2))
        
        print(f"Computed normalization stats from {count} batches")
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Separate sources from input waveform.
        
        Args:
            waveform: Input mix waveform of shape (batch, channels, time)
        
        Returns:
            Dictionary mapping source names to separated waveforms
        """
        # Compute STFT
        mix_stft = self.stft(waveform)
        mix_magnitude = torch.abs(mix_stft)
        mix_phase = torch.angle(mix_stft)
        
        # Normalize magnitude
        mix_magnitude_norm = self.normalize_spectrogram(mix_magnitude)
        
        # Predict masks
        mask_output = self.model(mix_magnitude_norm)  # Shape: (batch, channels*sources, freq, time)
        
        # Calculate correct mask shape based on model output
        batch_size, _, freq_bins, time_frames = mix_magnitude_norm.shape
        n_channels = mix_magnitude_norm.shape[1]
        n_sources = len(self.source_names)
        
        # Model output should be (batch, channels*sources, freq, time)
        # We need to reshape it to (batch, channels, freq, time, sources)
        output_channels = mask_output.shape[1]
        
        # Simple approach: just reshape the output to add the sources dimension
        masks = mask_output.view(batch_size, n_channels, n_sources, mask_output.shape[2], mask_output.shape[3])
        masks = masks.permute(0, 1, 3, 4, 2)  # (batch, channels, freq, time, sources)
        
        # Resize masks to match input spectrogram size if needed
        if masks.shape[2:4] != (freq_bins, time_frames):
            # Use interpolation to resize masks
            masks_resized = torch.nn.functional.interpolate(
                masks.permute(0, 4, 1, 2, 3).contiguous().view(-1, n_channels, masks.shape[2], masks.shape[3]),
                size=(freq_bins, time_frames),
                mode='bilinear',
                align_corners=False
            )
            masks = masks_resized.view(batch_size, n_sources, n_channels, freq_bins, time_frames).permute(0, 2, 3, 4, 1)
        
        # Apply masks to original magnitude
        separated_magnitudes = mix_magnitude.unsqueeze(-1) * masks
        
        # Reconstruct complex spectrograms with original phase
        mix_phase_expanded = mix_phase.unsqueeze(-1).expand_as(separated_magnitudes)
        separated_stfts = separated_magnitudes * torch.exp(1j * mix_phase_expanded)
        
        # Convert back to waveforms
        separated_waveforms = {}
        original_length = waveform.shape[-1]
        
        for i, source_name in enumerate(self.source_names):
            source_stft = separated_stfts[..., i]
            source_waveform = self.istft(source_stft, length=original_length)
            separated_waveforms[source_name] = source_waveform
        
        return separated_waveforms
    
    def separate_file(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        chunk_duration: float = 10.0,
        overlap: float = 0.25,
    ) -> Dict[str, Path]:
        """
        Separate sources from an audio file.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated sources
            device: Device for computation
            chunk_duration: Duration of chunks for processing (seconds)
            overlap: Overlap between chunks (fraction)
        
        Returns:
            Dictionary mapping source names to output file paths
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            device = next(self.parameters()).device
        
        # Load audio
        waveform, original_sr = torchaudio.load(input_path)
        
        # Resample if necessary
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(device)
        
        # Process in chunks if audio is long
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * chunk_samples)
        audio_length = waveform.shape[-1]
        
        if audio_length <= chunk_samples:
            # Process entire audio at once
            with torch.no_grad():
                separated = self.forward(waveform)
        else:
            # Process in overlapping chunks
            separated = {name: torch.zeros_like(waveform) for name in self.source_names}
            
            start = 0
            while start < audio_length:
                end = min(start + chunk_samples, audio_length)
                chunk = waveform[..., start:end]
                
                with torch.no_grad():
                    chunk_separated = self.forward(chunk)
                
                # Add to output with overlap handling
                for name in self.source_names:
                    if start == 0:
                        # First chunk
                        separated[name][..., start:end] = chunk_separated[name]
                    elif end == audio_length:
                        # Last chunk
                        separated[name][..., start:end] = chunk_separated[name]
                    else:
                        # Middle chunk - apply cross-fade
                        fade_length = overlap_samples
                        chunk_len = chunk_separated[name].shape[-1]
                        
                        # Fade in
                        fade_in = torch.linspace(0, 1, fade_length, device=device)
                        chunk_separated[name][..., :fade_length] *= fade_in
                        
                        # Fade out
                        fade_out = torch.linspace(1, 0, fade_length, device=device)
                        separated[name][..., start:start+fade_length] *= fade_out
                        
                        # Add chunk
                        separated[name][..., start:end] += chunk_separated[name]
                
                start += chunk_samples - overlap_samples
        
        # Save separated sources
        output_paths = {}
        for name in self.source_names:
            output_path = output_dir / f"{input_path.stem}_{name}.wav"
            
            # Remove batch dimension and move to CPU
            source_audio = separated[name].squeeze(0).cpu()
            
            # Save audio
            torchaudio.save(
                output_path,
                source_audio,
                self.sample_rate,
            )
            
            output_paths[name] = output_path
        
        return output_paths


def create_separator(
    model_config: Dict,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> SourceSeparator:
    """
    Create a source separator with a pre-trained model.
    
    Args:
        model_config: Model configuration dictionary
        checkpoint_path: Path to model checkpoint
        device: Device for computation
    
    Returns:
        Configured source separator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_unet_model(**model_config)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create separator
    separator = SourceSeparator(
        model=model,
        n_fft=model_config.get("n_fft", 4096),
        hop_length=model_config.get("hop_length", 1024),
        source_names=model_config.get("source_names", ["vocals", "drums", "bass", "other"]),
        sample_rate=model_config.get("sample_rate", 44100),
    )
    
    separator.to(device)
    
    return separator


if __name__ == "__main__":
    # Test the separator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a test model
    model_config = {
        "model_type": "spleeter",
        "n_fft": 4096,
        "n_sources": 4,
        "n_channels": 2,
    }
    
    separator = create_separator(model_config, device=device)
    
    # Test with random audio
    test_audio = torch.randn(1, 2, 44100 * 5).to(device)  # 5 seconds of stereo audio
    
    print(f"Input audio shape: {test_audio.shape}")
    
    with torch.no_grad():
        separated = separator(test_audio)
    
    for name, audio in separated.items():
        print(f"{name}: {audio.shape}")
    
    print("Separator test completed successfully!")
