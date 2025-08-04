"""
Training script for U-Net music source separation.

This script handles the complete training pipeline including data loading,
model training, validation, and checkpointing.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models import create_unet_model, SourceSeparator
from data import create_musdb_dataloader, SpectrogramNormalization
from utils import SeparationMetrics, sdr, si_sdr


class UNetTrainer:
    """
    Trainer class for U-Net source separation models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        self._set_random_seeds(config["system"]["seed"])
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize normalization
        self.normalizer = self._create_normalizer()
        
        # Initialize logging
        self.writer = self._create_writer()
        
        # Initialize metrics
        self.metrics = SeparationMetrics(
            config["model"]["source_names"],
            use_museval=config["evaluation"]["use_museval"],
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # Mixed precision training
        self.use_amp = config["training"]["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config["system"]["deterministic"]:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self.config["system"]["cuda_benchmark"]
    
    def _create_model(self) -> nn.Module:
        """Create U-Net model."""
        model_config = self.config["model"]
        return create_unet_model(**model_config)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        training_config = self.config["training"]
        
        if training_config["optimizer"] == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
            )
        elif training_config["optimizer"] == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        training_config = self.config["training"]
        scheduler_type = training_config["scheduler"]
        
        if scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=training_config["val_patience"],
                verbose=True,
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config["num_epochs"],
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5,
            )
        else:
            return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_type = self.config["training"]["loss_function"]
        
        if loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "l2" or loss_type == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def _create_data_loaders(self) -> tuple:
        """Create training and validation data loaders."""
        data_config = self.config["data"]
        model_config = self.config["model"]
        
        # Create training loader
        train_loader = create_musdb_dataloader(
            root=data_config["musdb_path"],
            subset="train",
            batch_size=self.config["training"]["batch_size"],
            num_workers=data_config["num_workers"],
            segment_duration=data_config["segment_duration"],
            sample_rate=model_config["sample_rate"],
            n_fft=model_config["n_fft"],
            hop_length=model_config["hop_length"],
            source_names=model_config["source_names"],
        )
        
        # Create validation loader
        val_loader = create_musdb_dataloader(
            root=data_config["musdb_path"],
            subset="test",  # Use test set for validation
            batch_size=self.config["training"]["batch_size"],
            num_workers=data_config["num_workers"],
            segment_duration=data_config["segment_duration"],
            sample_rate=model_config["sample_rate"],
            n_fft=model_config["n_fft"],
            hop_length=model_config["hop_length"],
            source_names=model_config["source_names"],
        )
        
        return train_loader, val_loader
    
    def _create_normalizer(self) -> Optional[SpectrogramNormalization]:
        """Create and fit spectrogram normalizer."""
        if not self.config["audio"]["normalize_input"]:
            return None
        
        normalizer = SpectrogramNormalization()
        
        if self.config["audio"]["compute_stats"]:
            print("Computing normalization statistics...")
            normalizer.fit(self.train_loader)
        
        return normalizer
    
    def _create_writer(self) -> Optional[SummaryWriter]:
        """Create tensorboard writer."""
        if not self.config["logging"]["use_tensorboard"]:
            return None
        
        log_dir = Path(self.config["logging"]["log_dir"]) / self.config["logging"]["experiment_name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return SummaryWriter(log_dir)
    
    def _compute_loss(
        self,
        mix_magnitude: torch.Tensor,
        source_magnitudes: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            mix_magnitude: Input mix magnitude spectrogram
            source_magnitudes: Target source magnitude spectrograms
        
        Returns:
            Loss value
        """
        # Normalize input if normalizer is available
        if self.normalizer is not None:
            mix_magnitude_norm = self.normalizer(mix_magnitude)
        else:
            mix_magnitude_norm = mix_magnitude
        
        # Forward pass
        predicted_masks = self.model(mix_magnitude_norm)
        
        # Apply masks to mix magnitude
        predicted_sources = mix_magnitude.unsqueeze(-1) * predicted_masks
        
        # Compute loss for each source
        total_loss = 0.0
        for i, source_name in enumerate(self.config["model"]["source_names"]):
            if source_name in source_magnitudes:
                target = source_magnitudes[source_name]
                predicted = predicted_sources[..., i]
                total_loss += self.criterion(predicted, target)
        
        return total_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (mix_magnitude, source_magnitudes) in enumerate(pbar):
            # Move to device
            mix_magnitude = mix_magnitude.to(self.device)
            source_magnitudes = {
                name: mag.to(self.device) for name, mag in source_magnitudes.items()
            }
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss computation
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(mix_magnitude, source_magnitudes)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config["training"]["gradient_clip_val"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_val"]
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self._compute_loss(mix_magnitude, source_magnitudes)
                loss.backward()
                
                # Gradient clipping
                if self.config["training"]["gradient_clip_val"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_val"]
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            if (
                self.writer is not None
                and batch_idx % self.config["logging"]["log_interval"] == 0
            ):
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
        
        return {"train_loss": total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for mix_magnitude, source_magnitudes in pbar:
                # Move to device
                mix_magnitude = mix_magnitude.to(self.device)
                source_magnitudes = {
                    name: mag.to(self.device) for name, mag in source_magnitudes.items()
                }
                
                # Compute loss
                loss = self._compute_loss(mix_magnitude, source_magnitudes)
                total_loss += loss.item()
                num_batches += 1
                
                # For evaluation metrics, we need to convert back to waveforms
                # This is a simplified evaluation - for full evaluation,
                # we would need to reconstruct waveforms using iSTFT
                
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        val_metrics = {"val_loss": total_loss / num_batches}
        
        # Add separation metrics if computed
        separation_metrics = self.metrics.compute()
        val_metrics.update(separation_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "metrics": metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.normalizer is not None:
            checkpoint["normalizer_mean"] = self.normalizer.mean
            checkpoint["normalizer_std"] = self.normalizer.std
        
        # Save latest checkpoint
        if self.config["logging"]["save_last"]:
            latest_path = checkpoint_dir / "latest.pth"
            torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint
        epoch_path = checkpoint_dir / f"epoch_{self.epoch:03d}.pth"
        torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config["training"]["val_interval"] == 0:
                val_metrics = self.validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log metrics
                if self.writer is not None:
                    for key, value in all_metrics.items():
                        self.writer.add_scalar(f"epoch/{key}", value, epoch)
                
                # Check for improvement
                val_loss = val_metrics["val_loss"]
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(all_metrics, is_best)
                
                # Print metrics
                print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if self.patience_counter >= self.config["training"]["patience"]:
                    print(f"Early stopping after {epoch} epochs")
                    break
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
        
        print("Training completed!")
        
        if self.writer is not None:
            self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for music source separation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unet_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = UNetTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        
        if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Resumed training from epoch {trainer.epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
