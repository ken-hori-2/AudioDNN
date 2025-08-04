"""
Evaluation metrics for music source separation.

This module provides various metrics for evaluating source separation quality
including SDR, SAR, SIR, and ISR metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import museval
except ImportError:
    museval = None
    print("Warning: museval package not available. Install with 'pip install museval'")


def sdr(target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        target: Ground truth signal
        estimate: Estimated signal
        eps: Small value to avoid numerical issues
    
    Returns:
        SDR value in dB
    """
    # Ensure same length
    min_len = min(target.shape[-1], estimate.shape[-1])
    target = target[..., :min_len]
    estimate = estimate[..., :min_len]
    
    # Compute SDR
    numerator = torch.sum(target ** 2, dim=-1)
    denominator = torch.sum((target - estimate) ** 2, dim=-1)
    
    sdr_value = 10 * torch.log10(numerator / (denominator + eps) + eps)
    
    return sdr_value


def si_sdr(target: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        target: Ground truth signal
        estimate: Estimated signal
        eps: Small value to avoid numerical issues
    
    Returns:
        SI-SDR value in dB
    """
    # Ensure same length
    min_len = min(target.shape[-1], estimate.shape[-1])
    target = target[..., :min_len]
    estimate = estimate[..., :min_len]
    
    # Zero mean
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    
    # Compute optimal scaling factor
    dot_product = torch.sum(target * estimate, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
    scale = dot_product / (target_energy + eps)
    
    # Scale estimate
    scaled_estimate = scale * target
    
    # Compute SI-SDR
    numerator = torch.sum(scaled_estimate ** 2, dim=-1)
    denominator = torch.sum((estimate - scaled_estimate) ** 2, dim=-1)
    
    si_sdr_value = 10 * torch.log10(numerator / (denominator + eps) + eps)
    
    return si_sdr_value


def bss_eval_sources_torch(
    reference_sources: torch.Tensor,
    estimated_sources: torch.Tensor,
    compute_permutation: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    PyTorch implementation of BSS evaluation metrics.
    
    This is a simplified version of the BSS eval metrics.
    For full compatibility with the original implementation,
    use the museval package.
    
    Args:
        reference_sources: Ground truth sources of shape (n_sources, n_samples)
        estimated_sources: Estimated sources of shape (n_sources, n_samples)
        compute_permutation: Whether to compute optimal permutation
    
    Returns:
        Dictionary containing SDR, SAR, SIR, ISR values
    """
    if reference_sources.shape != estimated_sources.shape:
        raise ValueError("Reference and estimated sources must have the same shape")
    
    n_sources = reference_sources.shape[0]
    min_len = min(reference_sources.shape[-1], estimated_sources.shape[-1])
    
    reference_sources = reference_sources[..., :min_len]
    estimated_sources = estimated_sources[..., :min_len]
    
    # Initialize metrics
    sdr_vals = torch.zeros(n_sources)
    sar_vals = torch.zeros(n_sources)
    sir_vals = torch.zeros(n_sources)
    isr_vals = torch.zeros(n_sources)
    
    for i in range(n_sources):
        ref = reference_sources[i]
        est = estimated_sources[i]
        
        # Compute SDR (simplified)
        sdr_vals[i] = sdr(ref, est)
        
        # For SAR, SIR, ISR, we use simplified approximations
        # For accurate values, use museval package
        sar_vals[i] = sdr_vals[i]  # Approximation
        sir_vals[i] = sdr_vals[i]  # Approximation
        isr_vals[i] = sdr_vals[i]  # Approximation
    
    return {
        "sdr": sdr_vals,
        "sar": sar_vals,
        "sir": sir_vals,
        "isr": isr_vals,
    }


def evaluate_separation(
    reference_sources: Dict[str, torch.Tensor],
    estimated_sources: Dict[str, torch.Tensor],
    source_names: List[str],
    sample_rate: int = 44100,
    use_museval: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate source separation quality.
    
    Args:
        reference_sources: Dictionary of ground truth sources
        estimated_sources: Dictionary of estimated sources
        source_names: List of source names to evaluate
        sample_rate: Audio sample rate
        use_museval: Whether to use museval package (if available)
    
    Returns:
        Dictionary of metrics for each source
    """
    results = {}
    
    for source_name in source_names:
        if source_name not in reference_sources or source_name not in estimated_sources:
            continue
        
        ref = reference_sources[source_name]
        est = estimated_sources[source_name]
        
        # Convert to numpy for museval if available
        if use_museval and museval is not None:
            try:
                ref_np = ref.cpu().numpy().T  # museval expects (time, channels)
                est_np = est.cpu().numpy().T
                
                # Compute metrics using museval
                scores = museval.eval_mus_track(
                    reference_track=ref_np,
                    estimated_track=est_np,
                    output_dir=None,
                )
                
                results[source_name] = {
                    "sdr": float(np.median(scores["SDR"])),
                    "sar": float(np.median(scores["SAR"])),
                    "sir": float(np.median(scores["SIR"])),
                    "isr": float(np.median(scores["ISR"])),
                }
            except Exception as e:
                print(f"Warning: museval failed for {source_name}, falling back to torch metrics: {e}")
                use_museval = False
        
        if not use_museval or museval is None:
            # Use PyTorch implementation
            sdr_val = sdr(ref, est)
            si_sdr_val = si_sdr(ref, est)
            
            # Average over channels if stereo
            if sdr_val.numel() > 1:
                sdr_val = torch.mean(sdr_val)
                si_sdr_val = torch.mean(si_sdr_val)
            
            results[source_name] = {
                "sdr": float(sdr_val),
                "si_sdr": float(si_sdr_val),
                "sar": float(sdr_val),  # Approximation
                "sir": float(sdr_val),  # Approximation
                "isr": float(sdr_val),  # Approximation
            }
    
    return results


def compute_aggregate_metrics(
    results: Dict[str, Dict[str, float]],
    source_names: List[str],
) -> Dict[str, float]:
    """
    Compute aggregate metrics across all sources.
    
    Args:
        results: Per-source evaluation results
        source_names: List of source names
    
    Returns:
        Dictionary of aggregate metrics
    """
    metrics = ["sdr", "sar", "sir", "isr", "si_sdr"]
    aggregate = {}
    
    for metric in metrics:
        values = []
        for source_name in source_names:
            if source_name in results and metric in results[source_name]:
                values.append(results[source_name][metric])
        
        if values:
            aggregate[f"mean_{metric}"] = np.mean(values)
            aggregate[f"median_{metric}"] = np.median(values)
            aggregate[f"std_{metric}"] = np.std(values)
    
    return aggregate


class SeparationMetrics:
    """
    Class for computing and tracking separation metrics.
    """
    
    def __init__(self, source_names: List[str], use_museval: bool = True):
        self.source_names = source_names
        self.use_museval = use_museval and (museval is not None)
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.results = []
    
    def update(
        self,
        reference_sources: Dict[str, torch.Tensor],
        estimated_sources: Dict[str, torch.Tensor],
    ):
        """
        Update metrics with new evaluation.
        
        Args:
            reference_sources: Ground truth sources
            estimated_sources: Estimated sources
        """
        result = evaluate_separation(
            reference_sources,
            estimated_sources,
            self.source_names,
            use_museval=self.use_museval,
        )
        self.results.append(result)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final aggregated metrics.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.results:
            return {}
        
        # Aggregate across all evaluations
        source_metrics = {name: {metric: [] for metric in ["sdr", "sar", "sir", "isr", "si_sdr"]} 
                         for name in self.source_names}
        
        for result in self.results:
            for source_name in self.source_names:
                if source_name in result:
                    for metric in source_metrics[source_name]:
                        if metric in result[source_name]:
                            source_metrics[source_name][metric].append(result[source_name][metric])
        
        # Compute statistics
        final_metrics = {}
        
        for source_name in self.source_names:
            for metric in source_metrics[source_name]:
                values = source_metrics[source_name][metric]
                if values:
                    final_metrics[f"{source_name}_{metric}_mean"] = np.mean(values)
                    final_metrics[f"{source_name}_{metric}_std"] = np.std(values)
        
        # Compute overall statistics
        all_metrics = ["sdr", "sar", "sir", "isr", "si_sdr"]
        for metric in all_metrics:
            all_values = []
            for source_name in self.source_names:
                values = source_metrics[source_name].get(metric, [])
                all_values.extend(values)
            
            if all_values:
                final_metrics[f"overall_{metric}_mean"] = np.mean(all_values)
                final_metrics[f"overall_{metric}_std"] = np.std(all_values)
        
        return final_metrics


if __name__ == "__main__":
    # Test metrics
    
    # Create test signals
    target = torch.randn(2, 44100)  # 1 second stereo
    estimate = target + 0.1 * torch.randn_like(target)  # Add some noise
    
    print("Testing metrics...")
    
    # Test SDR
    sdr_val = sdr(target, estimate)
    print(f"SDR: {sdr_val}")
    
    # Test SI-SDR
    si_sdr_val = si_sdr(target, estimate)
    print(f"SI-SDR: {si_sdr_val}")
    
    # Test with perfect reconstruction
    perfect_estimate = target.clone()
    perfect_sdr = sdr(target, perfect_estimate)
    print(f"Perfect reconstruction SDR: {perfect_sdr}")
    
    # Test evaluation function
    reference_sources = {
        "vocals": target,
        "drums": torch.randn_like(target),
    }
    
    estimated_sources = {
        "vocals": estimate,
        "drums": torch.randn_like(target),
    }
    
    results = evaluate_separation(
        reference_sources,
        estimated_sources,
        ["vocals", "drums"],
        use_museval=False,  # Use torch implementation
    )
    
    print("Evaluation results:")
    for source, metrics in results.items():
        print(f"{source}: {metrics}")
    
    # Test metrics class
    metrics_tracker = SeparationMetrics(["vocals", "drums"], use_museval=False)
    metrics_tracker.update(reference_sources, estimated_sources)
    
    final_metrics = metrics_tracker.compute()
    print("\nFinal aggregated metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.3f}")
    
    print("Metrics test completed successfully!")
