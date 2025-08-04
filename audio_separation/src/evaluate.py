"""
Evaluation script for trained U-Net source separation models.

This script evaluates models on the MUSDB18 test set and computes standard metrics.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models import create_unet_model, SourceSeparator
from utils import compute_sdr, si_sdr_loss
from separate import load_model_from_checkpoint

# Optional imports
try:
    import musdb
    import museval
    HAS_MUSDB = True
except ImportError:
    HAS_MUSDB = False
    print("Warning: musdb/museval not available. Limited evaluation functionality.")


class EvaluationResults:
    """Container for evaluation results."""
    
    def __init__(self):
        self.song_results = []
        self.aggregate_results = {}
    
    def add_song_result(self, track_name: str, results: Dict[str, Dict[str, float]]):
        """Add results for a single song."""
        song_result = {
            "track": track_name,
            "results": results
        }
        self.song_results.append(song_result)
    
    def compute_aggregate(self):
        """Compute aggregate statistics across all songs."""
        if not self.song_results:
            return
        
        # Collect all metrics
        all_metrics = {}
        
        for song_result in self.song_results:
            for source, metrics in song_result["results"].items():
                if source not in all_metrics:
                    all_metrics[source] = {}
                
                for metric_name, value in metrics.items():
                    if metric_name not in all_metrics[source]:
                        all_metrics[source][metric_name] = []
                    
                    if not np.isnan(value):
                        all_metrics[source][metric_name].append(value)
        
        # Compute statistics
        self.aggregate_results = {}
        for source, metrics in all_metrics.items():
            self.aggregate_results[source] = {}
            
            for metric_name, values in metrics.items():
                if values:
                    self.aggregate_results[source][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "count": len(values),
                    }
    
    def print_summary(self):
        """Print evaluation summary."""
        self.compute_aggregate()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if not self.aggregate_results:
            print("No results to display.")
            return
        
        # Print aggregate results
        for source in sorted(self.aggregate_results.keys()):
            print(f"\n{source.upper()}:")
            source_results = self.aggregate_results[source]
            
            for metric in ["SDR", "SAR", "SIR", "ISR", "SI-SDR"]:
                if metric in source_results:
                    stats = source_results[metric]
                    print(f"  {metric:6s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} dB "
                          f"(median: {stats['median']:6.2f}, n={stats['count']})")
        
        # Overall averages
        print(f"\nOVERALL AVERAGE:")
        overall_metrics = {}
        for source, metrics in self.aggregate_results.items():
            for metric_name, stats in metrics.items():
                if metric_name not in overall_metrics:
                    overall_metrics[metric_name] = []
                overall_metrics[metric_name].append(stats["mean"])
        
        for metric in ["SDR", "SAR", "SIR", "ISR", "SI-SDR"]:
            if metric in overall_metrics:
                values = overall_metrics[metric]
                print(f"  {metric:6s}: {np.mean(values):6.2f} ± {np.std(values):5.2f} dB")
    
    def save_to_json(self, output_path: Union[str, Path]):
        """Save results to JSON file."""
        self.compute_aggregate()
        
        output_data = {
            "song_results": self.song_results,
            "aggregate_results": self.aggregate_results,
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def evaluate_separation_musdb(
    separator: SourceSeparator,
    musdb_root: str,
    subset: str = "test",
    output_dir: Optional[str] = None,
    save_estimates: bool = False,
    device: Optional[torch.device] = None,
) -> EvaluationResults:
    """
    Evaluate separator on MUSDB18 dataset.
    
    Args:
        separator: Trained source separator
        musdb_root: Path to MUSDB18 dataset
        subset: Dataset subset ("test" or "train")
        output_dir: Directory to save separated audio (optional)
        save_estimates: Whether to save separated audio files
        device: Computation device
    
    Returns:
        Evaluation results
    """
    if not HAS_MUSDB:
        raise ImportError("musdb package required for MUSDB18 evaluation")
    
    if device is None:
        device = next(separator.parameters()).device
    
    # Load MUSDB18 dataset
    mus = musdb.DB(root=musdb_root, subsets=[subset])
    
    results = EvaluationResults()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating on MUSDB18 {subset} set ({len(mus)} tracks)")
    
    separator.eval()
    with torch.no_grad():
        for track in tqdm(mus, desc="Evaluating tracks"):
            try:
                # Get mixture and references
                mixture = torch.from_numpy(track.audio.T).float()  # (channels, time)
                
                # Resample if necessary
                if track.rate != separator.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        track.rate, 
                        separator.sample_rate
                    )
                    mixture = resampler(mixture)
                
                # Add batch dimension and move to device
                mixture = mixture.unsqueeze(0).to(device)
                
                # Separate sources
                separated = separator(mixture)
                
                # Convert to numpy for evaluation
                separated_np = {}
                for source_name, waveform in separated.items():
                    audio = waveform.squeeze(0).cpu().numpy().T  # (time, channels)
                    separated_np[source_name] = audio
                
                # Get ground truth references
                references = {}
                target_mapping = {
                    "vocals": "vocals",
                    "drums": "drums", 
                    "bass": "bass",
                    "accompaniment": "other",  # MUSDB calls it accompaniment
                }
                
                for musdb_name, our_name in target_mapping.items():
                    if our_name in separator.source_names:
                        ref_audio = getattr(track.targets[musdb_name], 'audio', None)
                        if ref_audio is not None:
                            # Resample reference if necessary
                            if track.rate != separator.sample_rate:
                                ref_tensor = torch.from_numpy(ref_audio.T).float()
                                ref_tensor = resampler(ref_tensor)
                                ref_audio = ref_tensor.numpy().T
                            
                            references[our_name] = ref_audio
                
                # Evaluate using museval
                track_results = {}
                
                for source_name in separator.source_names:
                    if source_name in separated_np and source_name in references:
                        try:
                            # Ensure same length
                            ref_len = references[source_name].shape[0]
                            est_len = separated_np[source_name].shape[0]
                            min_len = min(ref_len, est_len)
                            
                            reference = references[source_name][:min_len]
                            estimate = separated_np[source_name][:min_len]
                            
                            # Compute metrics using museval
                            sdr, isr, sir, sar, _ = museval.evaluate(
                                reference, 
                                estimate,
                                win=separator.sample_rate,
                                hop=separator.sample_rate//4,
                            )
                            
                            # Compute SI-SDR
                            ref_tensor = torch.from_numpy(reference.T)  # (channels, time)
                            est_tensor = torch.from_numpy(estimate.T)
                            si_sdr = -si_sdr_loss(est_tensor, ref_tensor).item()
                            
                            track_results[source_name] = {
                                "SDR": np.median(sdr),
                                "ISR": np.median(isr),
                                "SIR": np.median(sir),
                                "SAR": np.median(sar),
                                "SI-SDR": si_sdr,
                            }
                            
                        except Exception as e:
                            print(f"Warning: Failed to evaluate {source_name} for {track.name}: {e}")
                            track_results[source_name] = {
                                "SDR": np.nan, "ISR": np.nan, "SIR": np.nan, 
                                "SAR": np.nan, "SI-SDR": np.nan
                            }
                
                # Save separated audio if requested
                if save_estimates and output_dir:
                    track_dir = output_dir / track.name
                    track_dir.mkdir(exist_ok=True)
                    
                    for source_name, audio in separated_np.items():
                        output_path = track_dir / f"{source_name}.wav"
                        torchaudio.save(
                            output_path,
                            torch.from_numpy(audio.T),
                            separator.sample_rate,
                        )
                
                results.add_song_result(track.name, track_results)
                
            except Exception as e:
                print(f"Error processing track {track.name}: {e}")
                continue
    
    return results


def evaluate_separation_simple(
    separator: SourceSeparator,
    test_dir: Union[str, Path],
    device: Optional[torch.device] = None,
) -> EvaluationResults:
    """
    Simple evaluation on a directory of mixed/separated audio pairs.
    
    Args:
        separator: Trained source separator
        test_dir: Directory containing test audio files
        device: Computation device
    
    Returns:
        Evaluation results
    """
    test_dir = Path(test_dir)
    
    if device is None:
        device = next(separator.parameters()).device
    
    results = EvaluationResults()
    
    # Look for mixture files
    mixture_files = []
    for ext in [".wav", ".mp3", ".flac"]:
        mixture_files.extend(test_dir.glob(f"*mixture{ext}"))
        mixture_files.extend(test_dir.glob(f"*mix{ext}"))
    
    if not mixture_files:
        print(f"No mixture files found in {test_dir}")
        return results
    
    print(f"Evaluating {len(mixture_files)} test files")
    
    separator.eval()
    with torch.no_grad():
        for mixture_file in tqdm(mixture_files, desc="Evaluating"):
            try:
                base_name = mixture_file.stem.replace("_mixture", "").replace("_mix", "")
                
                # Load mixture
                mixture, sr = torchaudio.load(mixture_file)
                
                # Resample if necessary
                if sr != separator.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, separator.sample_rate)
                    mixture = resampler(mixture)
                
                # Add batch dimension and move to device
                mixture = mixture.unsqueeze(0).to(device)
                
                # Separate sources
                separated = separator(mixture)
                
                # Look for reference files
                track_results = {}
                
                for source_name in separator.source_names:
                    ref_patterns = [
                        test_dir / f"{base_name}_{source_name}.wav",
                        test_dir / f"{base_name}_{source_name}.mp3", 
                        test_dir / f"{base_name}_{source_name}.flac",
                    ]
                    
                    ref_file = None
                    for pattern in ref_patterns:
                        if pattern.exists():
                            ref_file = pattern
                            break
                    
                    if ref_file:
                        try:
                            # Load reference
                            reference, ref_sr = torchaudio.load(ref_file)
                            
                            if ref_sr != separator.sample_rate:
                                resampler = torchaudio.transforms.Resample(
                                    ref_sr, separator.sample_rate
                                )
                                reference = resampler(reference)
                            
                            # Get estimate
                            estimate = separated[source_name].squeeze(0).cpu()
                            
                            # Ensure same length
                            min_len = min(reference.shape[-1], estimate.shape[-1])
                            reference = reference[..., :min_len]
                            estimate = estimate[..., :min_len]
                            
                            # Compute metrics
                            sdr = compute_sdr(estimate, reference).item()
                            si_sdr = -si_sdr_loss(estimate, reference).item()
                            
                            track_results[source_name] = {
                                "SDR": sdr,
                                "SI-SDR": si_sdr,
                            }
                            
                        except Exception as e:
                            print(f"Warning: Failed to evaluate {source_name}: {e}")
                
                if track_results:
                    results.add_song_result(base_name, track_results)
                
            except Exception as e:
                print(f"Error processing {mixture_file.name}: {e}")
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net source separation model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--musdb-root",
        type=str,
        help="Path to MUSDB18 dataset root",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        help="Path to test directory (alternative to MUSDB18)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["test", "train"],
        help="MUSDB18 subset to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save separated audio files",
    )
    parser.add_argument(
        "--save-estimates",
        action="store_true",
        help="Save separated audio estimates",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="JSON file to save detailed results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.musdb_root and not args.test_dir:
        parser.error("Either --musdb-root or --test-dir must be specified")
    
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
    
    # Run evaluation
    try:
        if args.musdb_root:
            # MUSDB18 evaluation
            results = evaluate_separation_musdb(
                separator,
                args.musdb_root,
                subset=args.subset,
                output_dir=args.output_dir,
                save_estimates=args.save_estimates,
                device=device,
            )
        else:
            # Simple directory evaluation
            results = evaluate_separation_simple(
                separator,
                args.test_dir,
                device=device,
            )
        
        # Print results
        results.print_summary()
        
        # Save detailed results
        if args.results_file:
            results.save_to_json(args.results_file)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main()
