"""
Audio source separation inference script.

This script performs source separation on audio files using trained U-Net models.
"""

import os
import sys
import argparse
import yaml
import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models import create_unet_model, SourceSeparator, create_separator
from utils import load_audio, save_audio


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> SourceSeparator:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for computation
    
    Returns:
        Configured source separator
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # Create model
    model = create_unet_model(**config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create separator
    separator = SourceSeparator(
        model=model,
        n_fft=config["model"]["n_fft"],
        hop_length=config["model"]["hop_length"],
        source_names=config["model"]["source_names"],
        sample_rate=config["model"]["sample_rate"],
    )
    
    # Load normalization statistics if available
    if "normalizer_mean" in checkpoint and "normalizer_std" in checkpoint:
        separator.freq_mean.copy_(checkpoint["normalizer_mean"])
        separator.freq_std.copy_(checkpoint["normalizer_std"])
    
    separator.to(device)
    
    return separator


def separate_audio_file(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    separator: SourceSeparator,
    chunk_duration: float = 10.0,
    overlap: float = 0.25,
    device: Optional[torch.device] = None,
) -> Dict[str, Path]:
    """
    Separate sources from an audio file.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save separated sources
        separator: Source separator model
        chunk_duration: Duration of chunks for processing (seconds)
        overlap: Overlap between chunks (fraction)
        device: Device for computation
    
    Returns:
        Dictionary mapping source names to output file paths
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device is None:
        device = next(separator.parameters()).device
    
    print(f"Separating: {input_path.name}")
    
    # Load audio
    try:
        waveform, sample_rate = load_audio(
            input_path,
            sample_rate=separator.sample_rate,
            mono=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {input_path}: {e}")
    
    print(f"Audio loaded: {waveform.shape}, {sample_rate} Hz")
    
    # Add batch dimension and move to device
    waveform = waveform.unsqueeze(0).to(device)
    
    # Separate sources
    with torch.no_grad():
        if waveform.shape[-1] <= chunk_duration * sample_rate:
            # Process entire audio at once
            separated = separator(waveform)
        else:
            # Process in chunks
            separated = separate_in_chunks(
                waveform,
                separator,
                chunk_duration,
                overlap,
                device,
            )
    
    # Save separated sources
    output_paths = {}
    for source_name in separator.source_names:
        if source_name in separated:
            output_path = output_dir / f"{input_path.stem}_{source_name}.wav"
            
            # Remove batch dimension and move to CPU
            source_audio = separated[source_name].squeeze(0).cpu()
            
            # Save audio
            try:
                save_audio(source_audio, output_path, sample_rate)
                output_paths[source_name] = output_path
                print(f"Saved {source_name}: {output_path}")
            except Exception as e:
                print(f"Warning: Failed to save {source_name}: {e}")
    
    return output_paths


def separate_in_chunks(
    waveform: torch.Tensor,
    separator: SourceSeparator,
    chunk_duration: float,
    overlap: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Separate audio in overlapping chunks to handle long files.
    
    Args:
        waveform: Input waveform tensor (batch, channels, time)
        separator: Source separator model
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap fraction between chunks
        device: Device for computation
    
    Returns:
        Dictionary of separated source waveforms
    """
    sample_rate = separator.sample_rate
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * chunk_samples)
    audio_length = waveform.shape[-1]
    
    # Initialize output tensors
    separated = {name: torch.zeros_like(waveform) for name in separator.source_names}
    
    # Process chunks with overlap
    start = 0
    chunk_count = 0
    
    print(f"Processing audio in chunks of {chunk_duration}s with {overlap*100}% overlap...")
    
    while start < audio_length:
        end = min(start + chunk_samples, audio_length)
        chunk = waveform[..., start:end]
        
        # Separate chunk
        chunk_separated = separator(chunk)
        
        # Handle overlapping regions
        if start == 0:
            # First chunk
            for name in separator.source_names:
                separated[name][..., start:end] = chunk_separated[name]
        elif end == audio_length:
            # Last chunk
            for name in separator.source_names:
                separated[name][..., start:end] = chunk_separated[name]
        else:
            # Middle chunk - apply cross-fade
            fade_length = overlap_samples
            chunk_len = chunk_separated[name].shape[-1]
            
            # Create fade windows
            fade_in = torch.linspace(0, 1, fade_length, device=device)
            fade_out = torch.linspace(1, 0, fade_length, device=device)
            
            for name in separator.source_names:
                # Apply fade in to chunk
                chunk_separated[name][..., :fade_length] *= fade_in
                
                # Apply fade out to existing audio
                separated[name][..., start:start+fade_length] *= fade_out
                
                # Add chunk to output
                separated[name][..., start:end] += chunk_separated[name]
        
        chunk_count += 1
        progress = min(end / audio_length * 100, 100)
        print(f"Processed chunk {chunk_count}, progress: {progress:.1f}%")
        
        # Move to next chunk
        start += chunk_samples - overlap_samples
    
    return separated


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    separator: SourceSeparator,
    extensions: tuple = (".wav", ".mp3", ".flac", ".m4a"),
    **kwargs
):
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save separated sources
        separator: Source separator model
        extensions: Audio file extensions to process
        **kwargs: Additional arguments for separate_audio_file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            file_output_dir = output_dir / audio_file.stem
            separate_audio_file(
                audio_file,
                file_output_dir,
                separator,
                **kwargs
            )
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Separate audio sources using trained U-Net model")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for separated sources",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Duration of processing chunks in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between chunks as fraction (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation (default: auto)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".m4a"],
        help="Audio file extensions to process (default: .wav .mp3 .flac .m4a)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    try:
        separator = load_model_from_checkpoint(args.model, device)
        print("Model loaded successfully")
        print(f"Source names: {separator.source_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check input path
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Process input
    try:
        if input_path.is_file():
            # Single file
            separate_audio_file(
                input_path,
                output_path,
                separator,
                chunk_duration=args.chunk_duration,
                overlap=args.overlap,
                device=device,
            )
        elif input_path.is_dir():
            # Directory
            process_directory(
                input_path,
                output_path,
                separator,
                extensions=tuple(args.extensions),
                chunk_duration=args.chunk_duration,
                overlap=args.overlap,
                device=device,
            )
        else:
            print(f"Error: Input path is neither file nor directory: {input_path}")
    except Exception as e:
        print(f"Error during separation: {e}")
        return
    
    print(f"\nSeparation completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
